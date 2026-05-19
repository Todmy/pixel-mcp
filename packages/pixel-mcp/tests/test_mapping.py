"""Unit tests for MappingResolver — Code Connect + AI + heuristic layers."""

from __future__ import annotations

from datetime import UTC, datetime

from pixel_mcp.mapping import MappingPair, resolve_mappings
from pixel_mcp.render import (
    BoundingBox,
    ComputedStyle,
    MeasuredDOM,
    MeasuredElement,
)
from pixel_mcp.spec import DesignSpec, Dimensions, LayoutSpec


def _style() -> ComputedStyle:
    return ComputedStyle(
        color="#000000",
        background_color="#ffffff",
        font_family="Inter",
        font_size_px=16.0,
        font_weight=400,
        padding_top=0.0,
        padding_right=0.0,
        padding_bottom=0.0,
        padding_left=0.0,
        margin_top=0.0,
        margin_right=0.0,
        margin_bottom=0.0,
        margin_left=0.0,
        border_top_width=0.0,
        border_right_width=0.0,
        border_bottom_width=0.0,
        border_left_width=0.0,
    )


def _spec_with_text(
    node_id: str, text: str, children: list[DesignSpec] | None = None
) -> DesignSpec:
    return DesignSpec(
        figma_file_id="file-abc",
        figma_node_id=node_id,
        figma_node_type="FRAME",
        name=text,
        dimensions=Dimensions(width=100, height=50),
        layout=LayoutSpec(),
        text_content=text,
        children=children or [],
        extracted_at=datetime.now(UTC),
    )


def _dom_with(elements: list[tuple[str, str]]) -> MeasuredDOM:
    return MeasuredDOM(
        route="http://localhost:3000/",
        viewport=(1280, 720),
        measured_at=datetime.now(UTC),
        elements=[
            MeasuredElement(
                selector=sel,
                bounding_box=BoundingBox(x=0, y=0, w=100, h=50),
                computed_style=_style(),
                text_content=text,
            )
            for sel, text in elements
        ],
    )


def test_heuristic_text_match_high_confidence() -> None:
    child = _spec_with_text("1:23", "Submit")
    root = _spec_with_text("1:1", "Hero", children=[child])
    dom = _dom_with([("button.submit", "Submit"), ("h1.hero", "Hero")])
    mappings = resolve_mappings(root, dom)
    submit = next(p for p in mappings.pairs if p.figma_node_id == "1:23")
    assert submit.dom_selector == "button.submit"
    assert submit.source == "heuristic"
    assert submit.confidence >= 0.8


def test_code_connect_layer_is_stubbed_v0() -> None:
    """v0 stub returns no Code Connect pairs."""
    spec = _spec_with_text("1:1", "Hero")
    dom = _dom_with([("h1.hero", "Hero")])
    mappings = resolve_mappings(spec, dom, code_connect_enabled=True)
    assert all(p.source != "code_connect" for p in mappings.pairs)


def test_ai_pairing_layer_is_stubbed_v0() -> None:
    """v0 stub returns no AI pairs even when enabled."""
    spec = _spec_with_text("1:1", "Hero")
    dom = _dom_with([("h1.hero", "Hero")])
    mappings = resolve_mappings(spec, dom, ai_pairing_enabled=True)
    assert all(p.source != "ai" for p in mappings.pairs)


def test_unmatched_falls_back_to_flat_order() -> None:
    """When text doesn't match, low-confidence flat-order pair is emitted (if floor passes)."""
    spec = _spec_with_text("1:1", "Hero")
    dom = _dom_with([("div.something-different", "Other text")])
    mappings = resolve_mappings(spec, dom, heuristic_confidence_floor=0.3)
    assert len(mappings.pairs) == 1
    assert mappings.pairs[0].confidence < 0.5


def test_confidence_floor_filters_low_quality() -> None:
    spec = _spec_with_text("1:1", "Hero")
    dom = _dom_with([("div.x", "Different")])
    mappings = resolve_mappings(spec, dom, heuristic_confidence_floor=0.9)
    # Flat-order (0.4) below floor (0.9) → no pair emitted
    assert mappings.pairs == []


def test_each_dom_selector_paired_at_most_once() -> None:
    """No two figma nodes should map to the same DOM selector."""
    c1 = _spec_with_text("1:2", "Submit")
    c2 = _spec_with_text("1:3", "Submit")  # duplicate text
    root = _spec_with_text("1:1", "Root", children=[c1, c2])
    dom = _dom_with([("button.a", "Submit")])
    mappings = resolve_mappings(root, dom)
    selectors = [p.dom_selector for p in mappings.pairs]
    assert len(selectors) == len(set(selectors))


def test_mappings_pair_is_pydantic_model() -> None:
    p = MappingPair(figma_node_id="1:1", dom_selector="#a", confidence=0.9, source="heuristic")
    raw = p.model_dump_json()
    assert MappingPair.model_validate_json(raw).model_dump() == p.model_dump()


def test_mappings_carries_file_id() -> None:
    spec = _spec_with_text("1:1", "Hero")
    dom = _dom_with([("h1", "Hero")])
    mappings = resolve_mappings(spec, dom)
    assert mappings.figma_file_id == "file-abc"
