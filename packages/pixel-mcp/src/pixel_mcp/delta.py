"""DeltaDiffer — the Deep Module behind ``pixel-mcp diff``.

Public entry point: :func:`diff_design_vs_render`. Takes a DesignSpec and a
MeasuredDOM, emits a list of Deltas. Pure function — same inputs always
produce identical output (deterministic ordering, no time-dependence).

Design notes:
- Pairing strategy is naive in v0: exact text-content match, then flat-order
  fallback. Slice #8 (Mapping resolver) replaces this with a layered
  Code-Connect → AI → heuristic fallback. The optional ``mappings`` parameter
  lets the caller supply a pre-computed ``figma_node_id → css_selector``
  dictionary; when present we trust it and skip the naive matcher.
- Color comparison is hex-canonicalized on both sides before string-compare
  (DOM side comes from ``render.py`` already in hex; spec side converts
  Figma's ``{r, g, b, a}`` floats here).
- Dimension/numeric severity is percentage-based (per issue spec):
  >20% → critical, 5–20% → major, 2–5% → minor, <2% → no Delta.
- String mismatches (color, font-family) → critical (no magnitude).
- Missing DOM element → critical with ``observed=None``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from pixel_mcp.render import MeasuredDOM, MeasuredElement
from pixel_mcp.spec import DesignSpec

Severity = Literal["critical", "major", "minor", "regression"]


class Delta(BaseModel):
    """A single property mismatch between DesignSpec and MeasuredDOM."""

    selector: str
    """DOM CSS selector the Delta refers to."""

    figma_node_id: str | None = None
    """Source Figma node, when the Mapping is known."""

    property: str
    """Property name (e.g. ``color``, ``font_size_px``, ``padding_left``)."""

    observed: Any = None
    """Value observed in the Render (DOM). ``None`` when the element is missing."""

    expected: Any = None
    """Value declared in the Design Source (Figma)."""

    magnitude: float | None = None
    """Absolute difference for numeric properties. ``None`` for string mismatches
    or missing elements."""

    severity: Severity
    """Drives Gate Pass logic — critical and major block Final Convergence."""

    viewport: str | None = None
    """Viewport identifier (``"<W>x<H>"``) when the Delta was produced under a
    multi-viewport check. ``None`` for single-viewport (v0/v1) callers — fully
    backward compatible. Different viewports produce different hash buckets so
    stuck/regression detection stays accurate across the responsive matrix."""


class _FlatSpecNode(BaseModel):
    """Flattened view of a node in the DesignSpec tree — used internally for pairing."""

    figma_node_id: str
    figma_node_type: str
    name: str
    text_content: str | None
    depth: int
    spec: DesignSpec

    model_config = {"arbitrary_types_allowed": True}


__all__ = ["Delta", "Severity", "diff_design_vs_render", "canonicalize_figma_color"]


# --- Public API ------------------------------------------------------------


def diff_design_vs_render(
    spec: DesignSpec,
    dom: MeasuredDOM,
    mappings: dict[str, str] | None = None,
) -> list[Delta]:
    """Compute Deltas between a DesignSpec and a MeasuredDOM.

    Args:
        spec: The Design Source structure.
        dom: The Render measurement.
        mappings: Optional ``figma_node_id → css_selector`` map. When omitted,
            the naive text-content + flat-order matcher fills in pairs.

    Returns:
        Deterministically-ordered list of Deltas. Ordering: figma_node_id
        ascending, then property name ascending. Empty when no mismatches
        (Final Convergence within Tolerance, ignoring Normalizer).
    """
    flat = _flatten_spec(spec)
    pair_map = mappings if mappings is not None else _naive_pair(flat, dom)

    dom_by_selector = {el.selector: el for el in dom.elements}
    deltas: list[Delta] = []

    for fnode in flat:
        dom_selector = pair_map.get(fnode.figma_node_id)
        if dom_selector is None:
            # Unpaired figma node — silent in v0; Slice #8 will surface this
            # as a low-confidence warning via the Mapping resolver.
            continue

        dom_el = dom_by_selector.get(dom_selector)
        if dom_el is None:
            deltas.append(
                Delta(
                    selector=dom_selector,
                    figma_node_id=fnode.figma_node_id,
                    property="element_present",
                    observed=None,
                    expected="present",
                    magnitude=None,
                    severity="critical",
                )
            )
            continue

        deltas.extend(_compare_node(fnode, dom_el, dom_selector))

    # Stable ordering — pure-function contract.
    deltas.sort(key=lambda d: (d.figma_node_id or "", d.property))
    return deltas


# --- Pairing ---------------------------------------------------------------


def _flatten_spec(spec: DesignSpec, depth: int = 0) -> list[_FlatSpecNode]:
    """Pre-order walk of the DesignSpec tree."""
    flat = [
        _FlatSpecNode(
            figma_node_id=spec.figma_node_id,
            figma_node_type=spec.figma_node_type,
            name=spec.name,
            text_content=spec.text_content,
            depth=depth,
            spec=spec,
        )
    ]
    for child in spec.children:
        flat.extend(_flatten_spec(child, depth + 1))
    return flat


def _naive_pair(flat: list[_FlatSpecNode], dom: MeasuredDOM) -> dict[str, str]:
    """Pair Figma nodes to DOM selectors.

    v0 strategy: exact text-content match (strongest signal), then flat-order
    fallback for the unmatched. Replaced by Slice #8 MappingResolver.
    """
    by_text: dict[str, list[str]] = {}
    for el in dom.elements:
        if el.text_content:
            by_text.setdefault(el.text_content.strip().lower(), []).append(el.selector)

    pairs: dict[str, str] = {}
    matched_selectors: set[str] = set()

    # Phase 1 — text matching
    for fnode in flat:
        if fnode.text_content:
            key = fnode.text_content.strip().lower()
            candidates = by_text.get(key, [])
            for cand in candidates:
                if cand not in matched_selectors:
                    pairs[fnode.figma_node_id] = cand
                    matched_selectors.add(cand)
                    break

    # Phase 2 — flat-order fallback for unmatched
    unmatched_fnodes = [f for f in flat if f.figma_node_id not in pairs]
    unmatched_dom = [el for el in dom.elements if el.selector not in matched_selectors]
    for fnode, dom_el in zip(unmatched_fnodes, unmatched_dom, strict=False):
        pairs[fnode.figma_node_id] = dom_el.selector

    return pairs


# --- Per-node comparison ---------------------------------------------------


def _compare_node(fnode: _FlatSpecNode, dom_el: MeasuredElement, selector: str) -> list[Delta]:
    out: list[Delta] = []
    spec = fnode.spec
    cs = dom_el.computed_style

    def add(
        prop: str, observed: Any, expected: Any, severity: Severity, magnitude: float | None = None
    ) -> None:
        out.append(
            Delta(
                selector=selector,
                figma_node_id=fnode.figma_node_id,
                property=prop,
                observed=observed,
                expected=expected,
                magnitude=magnitude,
                severity=severity,
            )
        )

    # --- Color comparison ---
    # Figma fills on a Frame/Component map to the DOM element's
    # background_color. For TEXT nodes the fills are the text color.
    # v0 keeps text-color matching out of scope.
    expected_fill = _first_solid(spec.fills)
    if expected_fill is not None:
        if spec.figma_node_type == "TEXT":
            if expected_fill != cs.color:
                add("color", cs.color, expected_fill, "critical")
        else:
            if expected_fill != cs.background_color:
                add("background_color", cs.background_color, expected_fill, "critical")

    # --- Dimensions ---
    if spec.dimensions.width > 0:
        sev = _numeric_severity(dom_el.bounding_box.w, spec.dimensions.width)
        if sev is not None:
            add(
                "width",
                dom_el.bounding_box.w,
                spec.dimensions.width,
                sev,
                magnitude=abs(dom_el.bounding_box.w - spec.dimensions.width),
            )
    if spec.dimensions.height > 0:
        sev = _numeric_severity(dom_el.bounding_box.h, spec.dimensions.height)
        if sev is not None:
            add(
                "height",
                dom_el.bounding_box.h,
                spec.dimensions.height,
                sev,
                magnitude=abs(dom_el.bounding_box.h - spec.dimensions.height),
            )

    # --- Typography ---
    if spec.typography is not None:
        if spec.typography.font_family and not _font_family_matches(
            cs.font_family, spec.typography.font_family
        ):
            add("font_family", cs.font_family, spec.typography.font_family, "critical")
        if spec.typography.font_size > 0:
            sev = _numeric_severity(cs.font_size_px, spec.typography.font_size)
            if sev is not None:
                add(
                    "font_size_px",
                    cs.font_size_px,
                    spec.typography.font_size,
                    sev,
                    magnitude=abs(cs.font_size_px - spec.typography.font_size),
                )
        if spec.typography.font_weight and spec.typography.font_weight != cs.font_weight:
            sev = _numeric_severity(cs.font_weight, spec.typography.font_weight)
            if sev is not None:
                add(
                    "font_weight",
                    cs.font_weight,
                    spec.typography.font_weight,
                    sev,
                    magnitude=float(abs(cs.font_weight - spec.typography.font_weight)),
                )

    # --- Padding (all four sides) ---
    for side, observed_val, expected_val in (
        ("padding_top", cs.padding_top, spec.layout.padding_top),
        ("padding_right", cs.padding_right, spec.layout.padding_right),
        ("padding_bottom", cs.padding_bottom, spec.layout.padding_bottom),
        ("padding_left", cs.padding_left, spec.layout.padding_left),
    ):
        # When both are 0, no Delta. When spec is 0 but DOM has padding,
        # treat as off-by-absolute; when spec is non-zero, percentage rule.
        if expected_val == 0 and observed_val == 0:
            continue
        if expected_val == 0:
            # spec says zero; any non-zero is critical structure mismatch
            if observed_val > 0:
                add(side, observed_val, expected_val, "critical", magnitude=float(observed_val))
            continue
        sev = _numeric_severity(observed_val, expected_val)
        if sev is not None:
            add(side, observed_val, expected_val, sev, magnitude=abs(observed_val - expected_val))

    return out


# --- Helpers ---------------------------------------------------------------


def canonicalize_figma_color(color: dict[str, float] | None) -> str | None:
    """Convert Figma's ``{r,g,b,a}`` (floats in [0,1]) to ``#rrggbb`` or ``#rrggbbaa``."""
    if color is None:
        return None
    r = max(0, min(255, round(color.get("r", 0) * 255)))
    g = max(0, min(255, round(color.get("g", 0) * 255)))
    b = max(0, min(255, round(color.get("b", 0) * 255)))
    a = color.get("a", 1.0)
    base = f"#{r:02x}{g:02x}{b:02x}"
    if a >= 1.0:
        return base
    a_int = max(0, min(255, round(a * 255)))
    return base + f"{a_int:02x}"


def _first_solid(fills: list[Any]) -> str | None:
    """Return canonical hex of the first visible SOLID fill, or None."""
    for fill in fills:
        if not getattr(fill, "visible", True):
            continue
        if getattr(fill, "type", None) != "SOLID":
            continue
        hex_value = canonicalize_figma_color(getattr(fill, "color", None))
        if hex_value is not None:
            return hex_value
    return None


def _font_family_matches(dom_value: str, spec_value: str) -> bool:
    """Loose equality: case-insensitive, ignore quotes and trailing fallback chains."""
    if not dom_value or not spec_value:
        return False
    dom_primary = _strip_font_quotes(dom_value.split(",")[0]).strip().lower()
    spec_primary = _strip_font_quotes(spec_value.split(",")[0]).strip().lower()
    return dom_primary == spec_primary


def _strip_font_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _numeric_severity(observed: float, expected: float) -> Severity | None:
    """Return severity for a numeric mismatch, or None if within 2% (no Delta).

    Rules per Slice #14 spec:
      <2%  → no Delta
      2-5% → minor
      5-20% → major
      >20% → critical
    """
    if expected == 0:
        return "critical" if observed != 0 else None
    pct = abs(observed - expected) / abs(expected) * 100.0
    if pct < 2.0:
        return None
    if pct < 5.0:
        return "minor"
    if pct < 20.0:
        return "major"
    return "critical"
