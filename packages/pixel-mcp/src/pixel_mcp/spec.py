"""DesignSpec extractor — the Deep Module behind ``pixel-mcp spec``.

Public entry point: :func:`extract_spec`. Takes a Figma URL, returns a
:class:`DesignSpec`. Internally:

1. Parse URL into ``(file_id, node_id)``.
2. Check the spec-cache (1h TTL). Return cached if fresh and not refreshing.
3. Fetch the node via :class:`FigmaClient`.
4. Discriminate by ``type``:
   - ``FRAME`` — full structured extraction.
   - ``INSTANCE`` — resolve master via second API call, apply overrides.
   - ``COMPONENT`` — sealed extraction, identical shape to FRAME.
   - other (``GROUP``, ``SECTION``, ...) — raise ``UnsupportedNodeTypeError``.
5. Cache the result and return.

The DesignSpec is lean by design — only fields the next slice (DeltaDiffer)
actually consumes. Adding fields later is cheaper than carrying dead weight
through schema migrations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from pixel_mcp.figma_client import FigmaClient
from pixel_mcp.figma_url import parse_figma_url
from pixel_mcp.state import SCHEMA_VERSION, read_spec_cache, write_spec_cache

SUPPORTED_NODE_TYPES = ("FRAME", "INSTANCE", "COMPONENT")

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_NODE_TYPES",
    "DesignSpec",
    "Dimensions",
    "LayoutSpec",
    "Constraints",
    "ColorOrGradient",
    "TypographySpec",
    "UnsupportedNodeTypeError",
    "extract_spec",
]


class UnsupportedNodeTypeError(Exception):
    """The Figma node is not one of the three supported types."""


# --- Pydantic models -------------------------------------------------------


class Dimensions(BaseModel):
    width: float
    height: float


class LayoutSpec(BaseModel):
    """Auto-layout shape. Falls back to neutral defaults when absent."""

    mode: Literal["NONE", "HORIZONTAL", "VERTICAL"] = "NONE"
    padding_top: float = 0.0
    padding_right: float = 0.0
    padding_bottom: float = 0.0
    padding_left: float = 0.0
    item_spacing: float = 0.0
    primary_axis_sizing: Literal["FIXED", "AUTO"] = "FIXED"
    counter_axis_sizing: Literal["FIXED", "AUTO"] = "FIXED"
    horizontal_sizing: Literal["FILL", "HUG", "FIXED"] = "FIXED"
    vertical_sizing: Literal["FILL", "HUG", "FIXED"] = "FIXED"


class Constraints(BaseModel):
    horizontal: Literal["LEFT", "RIGHT", "LEFT_RIGHT", "CENTER", "SCALE"] = "LEFT"
    vertical: Literal["TOP", "BOTTOM", "TOP_BOTTOM", "CENTER", "SCALE"] = "TOP"


class ColorOrGradient(BaseModel):
    """Lean container — SOLID fills get the color, gradient fills get a tag."""

    type: str  # "SOLID" | "GRADIENT_LINEAR" | "GRADIENT_RADIAL" | "IMAGE" | ...
    color: dict[str, float] | None = None  # {r,g,b,a} for SOLID
    opacity: float = 1.0
    visible: bool = True


class TypographySpec(BaseModel):
    font_family: str
    font_size: float
    font_weight: int = 400
    line_height_px: float | None = None
    letter_spacing: float = 0.0


class DesignSpec(BaseModel):
    schema_version: int = SCHEMA_VERSION
    source: Literal["figma"] = "figma"
    figma_file_id: str
    figma_node_id: str
    figma_node_type: str
    name: str
    dimensions: Dimensions
    layout: LayoutSpec = Field(default_factory=LayoutSpec)
    constraints: Constraints = Field(default_factory=Constraints)
    fills: list[ColorOrGradient] = Field(default_factory=list)
    strokes: list[ColorOrGradient] = Field(default_factory=list)
    corner_radius: float | None = None
    typography: TypographySpec | None = None
    text_content: str | None = None
    children: list[DesignSpec] = Field(default_factory=list)
    extracted_at: datetime

    model_config = {"arbitrary_types_allowed": True}


DesignSpec.model_rebuild()


# --- Public API ------------------------------------------------------------


def extract_spec(figma_url: str, refresh: bool = False) -> DesignSpec:
    """Extract a DesignSpec from a Figma URL.

    Caches results to ``.pixel-mcp/spec-cache.json`` (1h TTL keyed by
    ``(file_id, node_id)``). Pass ``refresh=True`` to bypass the cache.
    """
    parsed = parse_figma_url(figma_url)

    if not refresh:
        cached = read_spec_cache(parsed.file_id, parsed.node_id)
        if cached is not None:
            return cached

    with FigmaClient() as client:
        node = client.fetch_node(parsed.file_id, parsed.node_id)
        spec = _extract_dispatch(client, parsed.file_id, node)

    write_spec_cache(spec)
    return spec


# --- Internal: dispatch & extraction --------------------------------------


def _extract_dispatch(client: FigmaClient, file_id: str, node: dict[str, Any]) -> DesignSpec:
    node_type = node.get("type")
    if node_type == "FRAME":
        return _extract_frame(file_id, node)
    if node_type == "COMPONENT":
        return _extract_component(file_id, node)
    if node_type == "INSTANCE":
        return _extract_instance(client, file_id, node)
    raise UnsupportedNodeTypeError(
        f"Figma node type {node_type!r} is not supported. "
        f"Supported types: {', '.join(SUPPORTED_NODE_TYPES)}. "
        "Group/Section/Page/Vector layers are unsupported in Figma mode — "
        "use image-only mode (coming in v0.5) instead."
    )


def _extract_frame(file_id: str, node: dict[str, Any]) -> DesignSpec:
    return _build_spec(file_id, node, declared_type="FRAME")


def _extract_component(file_id: str, node: dict[str, Any]) -> DesignSpec:
    return _build_spec(file_id, node, declared_type="COMPONENT")


def _extract_instance(
    client: FigmaClient, file_id: str, instance_node: dict[str, Any]
) -> DesignSpec:
    """Resolve an Instance → master, then apply Instance overrides.

    Figma's API exposes the master id either on the Instance node itself
    (``componentId``) or in the ``components`` resolution table of the
    enclosing ``/nodes`` response. We accept both.
    """
    master_id = instance_node.get("componentId") or _pick_master_id_from_components(
        instance_node.get("__components", {})
    )
    if not master_id:
        # No master to resolve — fall back to the instance itself. This
        # keeps behavior safe for partially-detached instances.
        return _build_spec(file_id, instance_node, declared_type="INSTANCE")

    master_node = client.fetch_component_master(file_id, master_id)
    overrides = instance_node.get("overrides") or []
    overridden = _apply_overrides(master_node, instance_node, overrides)
    return _build_spec(file_id, overridden, declared_type="INSTANCE")


def _pick_master_id_from_components(components: dict[str, Any]) -> str | None:
    if not components:
        return None
    # Components table is {componentId: {name, key, ...}}; any one entry
    # tells us which master the instance points at. For non-mixed instances
    # there is exactly one entry.
    return next(iter(components.keys()), None)


def _apply_overrides(
    master_node: dict[str, Any],
    instance_node: dict[str, Any],
    overrides: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply Figma overrides on top of the master node.

    Override entries reference *instance* node ids (a list of
    ``{id, overriddenFields: [...]}``). We walk both trees in parallel,
    pairing master nodes with their instance counterparts by index, and
    swap any overridden field values from the instance node onto the
    master copy.
    """
    overridden_ids = {o.get("id") for o in overrides if o.get("id")}
    field_map: dict[str, list[str]] = {
        o["id"]: o.get("overriddenFields", []) or [] for o in overrides if o.get("id")
    }

    def merge(master: dict[str, Any], inst: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(master)
        if inst is not None and inst.get("id") in overridden_ids:
            for field in field_map.get(inst["id"], []):
                if field in inst:
                    merged[field] = inst[field]
        master_children = master.get("children") or []
        inst_children = (inst.get("children") if inst else None) or []
        if master_children:
            new_kids: list[dict[str, Any]] = []
            for i, mc in enumerate(master_children):
                ic = inst_children[i] if i < len(inst_children) else None
                new_kids.append(merge(mc, ic))
            merged["children"] = new_kids
        return merged

    out = merge(master_node, instance_node)
    # The instance root is special-cased: even without an explicit override
    # entry, the instance owns its name and position on the canvas. Any
    # field listed in the root's override entry has already been applied
    # by merge() above.
    if "name" in instance_node:
        out["name"] = instance_node["name"]
    if "absoluteBoundingBox" in instance_node:
        out["absoluteBoundingBox"] = instance_node["absoluteBoundingBox"]
    return out


def _build_spec(file_id: str, node: dict[str, Any], declared_type: str) -> DesignSpec:
    bbox = node.get("absoluteBoundingBox") or {}
    dims = Dimensions(
        width=float(bbox.get("width", 0.0)),
        height=float(bbox.get("height", 0.0)),
    )

    children_specs: list[DesignSpec] = []
    for child in node.get("children", []) or []:
        # Only structural children get spec entries. Vector-only leaves are
        # captured as raw fills/strokes on the parent for v0.
        if child.get("type") in SUPPORTED_NODE_TYPES + ("TEXT", "RECTANGLE", "VECTOR"):
            children_specs.append(_build_spec(file_id, child, declared_type=child["type"]))

    return DesignSpec(
        figma_file_id=file_id,
        figma_node_id=node.get("id", ""),
        figma_node_type=declared_type,
        name=node.get("name", ""),
        dimensions=dims,
        layout=_layout_from(node),
        constraints=_constraints_from(node),
        fills=_paints(node.get("fills", []) or []),
        strokes=_paints(node.get("strokes", []) or []),
        corner_radius=_corner_radius(node),
        typography=_typography(node),
        text_content=node.get("characters") if node.get("type") == "TEXT" else None,
        children=children_specs,
        extracted_at=datetime.now(UTC),
    )


def _layout_from(node: dict[str, Any]) -> LayoutSpec:
    mode = node.get("layoutMode", "NONE")
    if mode not in ("HORIZONTAL", "VERTICAL"):
        mode = "NONE"
    return LayoutSpec(
        mode=mode,
        padding_top=float(node.get("paddingTop", 0) or 0),
        padding_right=float(node.get("paddingRight", 0) or 0),
        padding_bottom=float(node.get("paddingBottom", 0) or 0),
        padding_left=float(node.get("paddingLeft", 0) or 0),
        item_spacing=float(node.get("itemSpacing", 0) or 0),
        primary_axis_sizing=_sizing(node.get("primaryAxisSizingMode"), default="FIXED"),
        counter_axis_sizing=_sizing(node.get("counterAxisSizingMode"), default="FIXED"),
        horizontal_sizing=_hv_sizing(node.get("layoutSizingHorizontal")),
        vertical_sizing=_hv_sizing(node.get("layoutSizingVertical")),
    )


def _sizing(value: Any, default: str) -> Any:
    if value in ("FIXED", "AUTO"):
        return value
    return default


def _hv_sizing(value: Any) -> Any:
    if value in ("FILL", "HUG", "FIXED"):
        return value
    return "FIXED"


def _constraints_from(node: dict[str, Any]) -> Constraints:
    c = node.get("constraints") or {}
    h = c.get("horizontal", "LEFT")
    v = c.get("vertical", "TOP")
    if h not in ("LEFT", "RIGHT", "LEFT_RIGHT", "CENTER", "SCALE"):
        h = "LEFT"
    if v not in ("TOP", "BOTTOM", "TOP_BOTTOM", "CENTER", "SCALE"):
        v = "TOP"
    return Constraints(horizontal=h, vertical=v)


def _paints(paints: list[dict[str, Any]]) -> list[ColorOrGradient]:
    out: list[ColorOrGradient] = []
    for p in paints:
        ptype = p.get("type", "SOLID")
        color = None
        if ptype == "SOLID" and "color" in p:
            c = p["color"]
            color = {
                "r": float(c.get("r", 0)),
                "g": float(c.get("g", 0)),
                "b": float(c.get("b", 0)),
                "a": float(c.get("a", 1)),
            }
        out.append(
            ColorOrGradient(
                type=ptype,
                color=color,
                opacity=float(p.get("opacity", 1.0)),
                visible=bool(p.get("visible", True)),
            )
        )
    return out


def _corner_radius(node: dict[str, Any]) -> float | None:
    v = node.get("cornerRadius")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _typography(node: dict[str, Any]) -> TypographySpec | None:
    style = node.get("style")
    if not style:
        return None
    lh: float | None = None
    if "lineHeightPx" in style:
        try:
            lh = float(style["lineHeightPx"])
        except (TypeError, ValueError):
            lh = None
    return TypographySpec(
        font_family=str(style.get("fontFamily", "")),
        font_size=float(style.get("fontSize", 0.0)),
        font_weight=int(style.get("fontWeight", 400)),
        line_height_px=lh,
        letter_spacing=float(style.get("letterSpacing", 0.0) or 0.0),
    )
