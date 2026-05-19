"""Pydantic model invariants for the RenderMeasurer."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pixel_mcp.render import (
    BoundingBox,
    ComputedStyle,
    MeasuredDOM,
    MeasuredElement,
)
from pydantic import ValidationError


def _style(**overrides: object) -> ComputedStyle:
    base: dict[str, object] = {
        "color": "#111111",
        "background_color": "#ffffff",
        "font_family": "Helvetica",
        "font_size_px": 16.0,
        "font_weight": 400,
        "line_height": "24px",
        "letter_spacing": "normal",
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
    return ComputedStyle(**base)  # type: ignore[arg-type]


def test_bounding_box_accepts_floats() -> None:
    box = BoundingBox(x=10.5, y=20.5, w=100.0, h=40.0)
    assert box.w == 100.0


def test_computed_style_round_trip() -> None:
    s = _style(color="#abcdef")
    payload = s.model_dump()
    assert payload["color"] == "#abcdef"
    assert payload["font_weight"] == 400


def test_measured_element_minimum_payload() -> None:
    el = MeasuredElement(
        selector="button.cta",
        bounding_box=BoundingBox(x=0, y=0, w=10, h=10),
        computed_style=_style(),
    )
    assert el.text_content is None
    assert el.aria_role is None
    assert el.parent_chain == []


def test_measured_dom_serializes_with_viewport_tuple() -> None:
    dom = MeasuredDOM(
        route="http://localhost:3000/foo",
        viewport=(1280, 720),
        measured_at=datetime.now(UTC),
        elements=[],
    )
    js = dom.model_dump()
    assert tuple(js["viewport"]) == (1280, 720)
    assert js["schema_version"] == 1


def test_measured_dom_rejects_missing_route() -> None:
    with pytest.raises(ValidationError):
        MeasuredDOM(  # type: ignore[call-arg]
            viewport=(1280, 720),
            measured_at=datetime.now(UTC),
            elements=[],
        )
