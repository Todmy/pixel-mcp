"""Normalizer — the highest-logic Deep Module behind ``pixel-mcp check`` width handling.

Public entry point: :func:`normalize_spec_for_viewport`. Takes a DesignSpec
extracted from Figma at one viewport width and returns a structurally
identical DesignSpec with dimensions adjusted for the target browser
viewport, respecting Figma's per-element sizing modes.

Sizing rules (per PRD #10 / CONTEXT.md):
- ``layoutSizingHorizontal: FILL`` (fluid) → scale width by
  ``target_width / spec_root_width`` ratio.
- ``layoutSizingHorizontal: HUG`` (content-sized) → absolute px unchanged.
- ``layoutSizingHorizontal: FIXED`` → absolute px unchanged.
- ``constraints.horizontal: LEFT_RIGHT`` (pinned both edges) → behaves as
  FILL (the element stretches with parent).
- ``constraints.horizontal: CENTER`` and ``SCALE`` → conservative: scaled
  proportionally in v0 (treated like FILL). Refined when position lands in
  the DesignSpec.

Vertical sizing follows identical rules against ``target_height``.

Out of v0 scope (deferred to later slices when DesignSpec gains a position
field): offset-from-pinned-edge for ``constraints.horizontal: RIGHT`` —
without explicit positions we cannot rewrite x-coordinates.

Type scale: when ``typography`` is present, font_size is scaled by the
same ratio FILL elements use, so the type scale ratio against the parent
is preserved across viewports. ``HUG``/``FIXED`` nodes leave font_size
unchanged (a fixed-size button has a fixed-size label).

The function is pure: identical inputs always yield identical output.
"""

from __future__ import annotations

from pixel_mcp.spec import DesignSpec, Dimensions, TypographySpec

__all__ = ["normalize_spec_for_viewport"]


_FLUID_SIZING = ("FILL",)
_FLUID_CONSTRAINTS = ("LEFT_RIGHT", "CENTER", "SCALE")


def normalize_spec_for_viewport(spec: DesignSpec, target_viewport: tuple[int, int]) -> DesignSpec:
    """Adjust a DesignSpec's dimensions to a target browser viewport.

    Args:
        spec: The DesignSpec from Figma — root dimensions taken as the
            "design viewport" used for scaling decisions.
        target_viewport: ``(width, height)`` of the browser viewport the
            Render was measured at.

    Returns:
        A new DesignSpec, structurally identical, with dimensions scaled
        per the sizing rules above. The root frame's own width and height
        are NOT scaled — that mismatch is user-intentional (rendering a
        1440-wide design at a 1280 browser).
    """
    target_w, target_h = target_viewport
    spec_w = spec.dimensions.width or 1.0
    spec_h = spec.dimensions.height or 1.0

    # The root scales by the ratio of target to design width. When they
    # match (the common case), the ratio is 1.0 — identity transform.
    x_ratio = target_w / spec_w if spec_w else 1.0
    y_ratio = target_h / spec_h if spec_h else 1.0

    # The root itself is NOT scaled — its dimensions express the design
    # canvas, and the browser viewport is what it is. Children are scaled.
    return _normalize_node(spec, x_ratio=x_ratio, y_ratio=y_ratio, is_root=True)


def _normalize_node(
    node: DesignSpec, *, x_ratio: float, y_ratio: float, is_root: bool
) -> DesignSpec:
    """Recursively scale a DesignSpec subtree."""
    if is_root or _ratios_are_identity(x_ratio, y_ratio):
        # Root nodes pass through; identity ratios fast-path.
        new_dimensions = node.dimensions
        new_typography = node.typography
    else:
        new_dimensions, new_typography = _scaled_dimensions_and_typography(
            node, x_ratio=x_ratio, y_ratio=y_ratio
        )

    new_children = [
        _normalize_node(child, x_ratio=x_ratio, y_ratio=y_ratio, is_root=False)
        for child in node.children
    ]

    return node.model_copy(
        update={
            "dimensions": new_dimensions,
            "typography": new_typography,
            "children": new_children,
        }
    )


def _scaled_dimensions_and_typography(
    node: DesignSpec, *, x_ratio: float, y_ratio: float
) -> tuple[Dimensions, TypographySpec | None]:
    """Apply sizing-mode rules to one node."""
    width = node.dimensions.width
    height = node.dimensions.height

    if _is_fluid_horizontal(node):
        width = node.dimensions.width * x_ratio
    if _is_fluid_vertical(node):
        height = node.dimensions.height * y_ratio

    # Typography scales with horizontal fluidity (the most common axis
    # for responsive type systems). HUG/FIXED nodes leave font size alone.
    if node.typography is not None and _is_fluid_horizontal(node):
        new_font_size = node.typography.font_size * x_ratio
        new_line_height = (
            node.typography.line_height_px * x_ratio
            if node.typography.line_height_px is not None
            else None
        )
        new_typography: TypographySpec | None = node.typography.model_copy(
            update={"font_size": new_font_size, "line_height_px": new_line_height}
        )
    else:
        new_typography = node.typography

    return Dimensions(width=width, height=height), new_typography


def _is_fluid_horizontal(node: DesignSpec) -> bool:
    if node.layout.horizontal_sizing in _FLUID_SIZING:
        return True
    if node.constraints.horizontal in _FLUID_CONSTRAINTS:
        return True
    return False


def _is_fluid_vertical(node: DesignSpec) -> bool:
    if node.layout.vertical_sizing in _FLUID_SIZING:
        return True
    if node.constraints.vertical in _FLUID_CONSTRAINTS:
        return True
    return False


def _ratios_are_identity(x_ratio: float, y_ratio: float, eps: float = 1e-6) -> bool:
    return abs(x_ratio - 1.0) < eps and abs(y_ratio - 1.0) < eps
