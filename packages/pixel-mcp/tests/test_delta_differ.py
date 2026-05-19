"""Unit tests for the DeltaDiffer Deep Module — pure function tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pixel_mcp.delta import canonicalize_figma_color, diff_design_vs_render
from pixel_mcp.render import (
    BoundingBox,
    ComputedStyle,
    MeasuredDOM,
    MeasuredElement,
)
from pixel_mcp.spec import (
    ColorOrGradient,
    DesignSpec,
    Dimensions,
    LayoutSpec,
    TypographySpec,
)

# --- Fixtures --------------------------------------------------------------


def _solid_red() -> ColorOrGradient:
    return ColorOrGradient(type="SOLID", color={"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0})


def _style(**overrides: object) -> ComputedStyle:
    base: dict[str, object] = {
        "color": "#ff0000",
        "background_color": "#ffffff",
        "font_family": "Inter",
        "font_size_px": 16.0,
        "font_weight": 400,
        "line_height": None,
        "letter_spacing": None,
        "padding_top": 0.0,
        "padding_right": 0.0,
        "padding_bottom": 0.0,
        "padding_left": 0.0,
        "margin_top": 0.0,
        "margin_right": 0.0,
        "margin_bottom": 0.0,
        "margin_left": 0.0,
        "border_radius": None,
        "border_top_width": 0.0,
        "border_right_width": 0.0,
        "border_bottom_width": 0.0,
        "border_left_width": 0.0,
    }
    base.update(overrides)
    return ComputedStyle.model_validate(base)


def _spec(
    node_id: str = "1:1",
    name: str = "Hero",
    width: float = 400.0,
    height: float = 100.0,
    fills: list[ColorOrGradient] | None = None,
    typography: TypographySpec | None = None,
    layout: LayoutSpec | None = None,
    children: list[DesignSpec] | None = None,
    text_content: str | None = None,
    node_type: str = "FRAME",
) -> DesignSpec:
    return DesignSpec(
        figma_file_id="abc",
        figma_node_id=node_id,
        figma_node_type=node_type,
        name=name,
        dimensions=Dimensions(width=width, height=height),
        layout=layout or LayoutSpec(),
        fills=fills if fills is not None else [_solid_red()],
        typography=typography,
        text_content=text_content,
        children=children or [],
        extracted_at=datetime.now(UTC),
    )


def _dom(
    elements: list[MeasuredElement] | None = None, route: str = "http://localhost:3000/"
) -> MeasuredDOM:
    return MeasuredDOM(
        route=route,
        viewport=(1280, 720),
        measured_at=datetime.now(UTC),
        elements=elements or [],
    )


def _element(
    selector: str = "#hero",
    text: str | None = "Hero",
    style: ComputedStyle | None = None,
    bbox: BoundingBox | None = None,
) -> MeasuredElement:
    return MeasuredElement(
        selector=selector,
        bounding_box=bbox or BoundingBox(x=0.0, y=0.0, w=400.0, h=100.0),
        computed_style=style or _style(),
        text_content=text,
        aria_role=None,
        parent_chain=["body"],
    )


# --- Color canonicalization ------------------------------------------------


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ({"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}, "#ff0000"),
        ({"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0}, "#000000"),
        ({"r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0}, "#ffffff"),
        ({"r": 1.0, "g": 0.0, "b": 0.0, "a": 0.5}, "#ff000080"),
        (None, None),
    ],
)
def test_canonicalize_figma_color(color: dict[str, float] | None, expected: str | None) -> None:
    assert canonicalize_figma_color(color) == expected


# --- Empty / happy path ----------------------------------------------------


def test_empty_dom_emits_no_deltas() -> None:
    """Spec with one node, no DOM at all — unpaired, so no Deltas emitted."""
    spec = _spec(text_content="Hero")
    deltas = diff_design_vs_render(spec, _dom())
    assert deltas == []


def test_identical_inputs_yield_empty_deltas() -> None:
    """Spec describes a red-fill 400×100 with text 'Hero'; DOM matches exactly."""
    spec = _spec(text_content="Hero")  # default fills = [red]
    dom = _dom(elements=[_element(text="Hero", style=_style(background_color="#ff0000"))])
    assert diff_design_vs_render(spec, dom) == []


def test_pure_function_deterministic_ordering() -> None:
    """Same inputs → same Delta list (including order)."""
    spec = _spec(
        text_content="Hero",
        typography=TypographySpec(font_family="Roboto", font_size=14),
    )
    dom = _dom(elements=[_element(text="Hero")])  # Inter/16/white-bg vs Roboto/14/red-bg
    d1 = diff_design_vs_render(spec, dom)
    d2 = diff_design_vs_render(spec, dom)
    assert d1 == d2
    assert len(d1) >= 2  # at least background_color + font_family
    # All Deltas sorted by (figma_node_id, property)
    keys = [(d.figma_node_id or "", d.property) for d in d1]
    assert keys == sorted(keys)


# --- Color & font mismatches ----------------------------------------------


def test_frame_fill_compared_to_dom_background() -> None:
    """For Frame/Component nodes, spec.fills maps to DOM.background_color."""
    spec = _spec(text_content="Hero", fills=[_solid_red()])
    dom = _dom(elements=[_element(text="Hero", style=_style(background_color="#00ff00"))])
    deltas = diff_design_vs_render(spec, dom)
    bg = [d for d in deltas if d.property == "background_color"]
    assert len(bg) == 1
    assert bg[0].severity == "critical"
    assert bg[0].expected == "#ff0000"
    assert bg[0].observed == "#00ff00"


def test_text_node_fill_compared_to_dom_color() -> None:
    """For TEXT nodes, spec.fills maps to DOM.color (the text color)."""
    spec = _spec(
        text_content="Hero",
        node_type="TEXT",
        fills=[_solid_red()],
    )
    dom = _dom(elements=[_element(text="Hero", style=_style(color="#00ff00"))])
    color_deltas = [d for d in diff_design_vs_render(spec, dom) if d.property == "color"]
    assert len(color_deltas) == 1
    assert color_deltas[0].severity == "critical"


def test_equivalent_color_formats_no_delta() -> None:
    """Figma {r:1,g:0,b:0,a:1} should match DOM '#ff0000' on the same role."""
    spec = _spec(text_content="Hero", fills=[_solid_red()])
    dom = _dom(elements=[_element(text="Hero", style=_style(background_color="#ff0000"))])
    bg = [d for d in diff_design_vs_render(spec, dom) if d.property == "background_color"]
    assert bg == []


def test_font_family_mismatch_is_critical() -> None:
    spec = _spec(
        text_content="Hero",
        typography=TypographySpec(font_family="Inter", font_size=16),
    )
    dom = _dom(elements=[_element(text="Hero", style=_style(font_family="Arial"))])
    ff_deltas = [d for d in diff_design_vs_render(spec, dom) if d.property == "font_family"]
    assert len(ff_deltas) == 1
    assert ff_deltas[0].severity == "critical"


def test_font_family_with_quotes_and_fallback_matches() -> None:
    """DOM ``"Inter", sans-serif`` should match spec ``Inter``."""
    spec = _spec(
        text_content="Hero",
        typography=TypographySpec(font_family="Inter", font_size=16),
    )
    dom = _dom(elements=[_element(text="Hero", style=_style(font_family='"Inter", sans-serif'))])
    ff_deltas = [d for d in diff_design_vs_render(spec, dom) if d.property == "font_family"]
    assert ff_deltas == []


# --- Dimension severity rules ---------------------------------------------


@pytest.mark.parametrize(
    ("observed", "expected_severity"),
    [
        (400.0, None),  # exact
        (404.0, None),  # 1% — under 2%
        (412.0, "minor"),  # 3%
        (440.0, "major"),  # 10%
        (500.0, "critical"),  # 25%
    ],
)
def test_dimension_severity_thresholds(observed: float, expected_severity: str | None) -> None:
    spec = _spec(text_content="Hero", width=400.0)
    bbox = BoundingBox(x=0.0, y=0.0, w=observed, h=100.0)
    dom = _dom(elements=[_element(text="Hero", bbox=bbox)])
    width_deltas = [d for d in diff_design_vs_render(spec, dom) if d.property == "width"]
    if expected_severity is None:
        assert width_deltas == []
    else:
        assert len(width_deltas) == 1
        assert width_deltas[0].severity == expected_severity
        assert width_deltas[0].magnitude == pytest.approx(abs(observed - 400.0))


# --- Missing element -------------------------------------------------------


def test_missing_dom_element_yields_critical() -> None:
    """Mapping points at a selector that doesn't exist in the DOM."""
    spec = _spec(text_content="Hero")
    dom = _dom(elements=[_element(selector="#something-else", text="Other")])
    mappings = {"1:1": "#hero"}  # selector not in DOM
    deltas = diff_design_vs_render(spec, dom, mappings=mappings)
    missing = [d for d in deltas if d.property == "element_present"]
    assert len(missing) == 1
    assert missing[0].severity == "critical"
    assert missing[0].observed is None
    assert missing[0].expected == "present"


# --- Mapping overrides naive pairing --------------------------------------


def test_explicit_mappings_used_when_provided() -> None:
    """When the caller passes mappings, naive matcher is skipped."""
    spec = _spec(text_content="Hero")
    # The DOM element has no matching text, but the mapping says use it.
    dom = _dom(elements=[_element(selector="#x", text=None)])
    deltas = diff_design_vs_render(spec, dom, mappings={"1:1": "#x"})
    # Mapping was used → at least the comparison happened (color delta expected
    # since DOM bg is white and spec fill is red, but more importantly: not the
    # silent-unpaired branch which would yield zero Deltas).
    assert any(d.figma_node_id == "1:1" for d in deltas)


# --- Padding rules ---------------------------------------------------------


def test_padding_zero_on_both_sides_no_delta() -> None:
    spec = _spec(text_content="Hero")
    dom = _dom(elements=[_element(text="Hero")])
    padding_deltas = [
        d for d in diff_design_vs_render(spec, dom) if d.property.startswith("padding_")
    ]
    assert padding_deltas == []


def test_padding_spec_zero_but_dom_nonzero_is_critical() -> None:
    spec = _spec(text_content="Hero", layout=LayoutSpec())
    dom = _dom(elements=[_element(text="Hero", style=_style(padding_left=12.0))])
    delta = next(d for d in diff_design_vs_render(spec, dom) if d.property == "padding_left")
    assert delta.severity == "critical"
    assert delta.observed == 12.0
    assert delta.expected == 0.0
