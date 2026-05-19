"""Unit tests for the Normalizer Deep Module — pure function tests.

Per PRD #10 Testing Decisions, the Normalizer is the primary unit-test target
for Slice 5. These tests cover every supported sizing/constraint combination
against hand-built DesignSpec fixtures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pixel_mcp.normalize import normalize_spec_for_viewport
from pixel_mcp.spec import (
    Constraints,
    DesignSpec,
    Dimensions,
    LayoutSpec,
    TypographySpec,
)


def _spec(
    width: float,
    height: float = 100.0,
    layout: LayoutSpec | None = None,
    constraints: Constraints | None = None,
    typography: TypographySpec | None = None,
    children: list[DesignSpec] | None = None,
    name: str = "n",
) -> DesignSpec:
    return DesignSpec(
        figma_file_id="abc",
        figma_node_id=name,
        figma_node_type="FRAME",
        name=name,
        dimensions=Dimensions(width=width, height=height),
        layout=layout or LayoutSpec(),
        constraints=constraints or Constraints(),
        typography=typography,
        children=children or [],
        extracted_at=datetime.now(UTC),
    )


# --- Identity behavior -----------------------------------------------------


def test_identity_when_viewports_match() -> None:
    """No transformation when target viewport equals spec root dimensions."""
    child = _spec(width=400, layout=LayoutSpec(horizontal_sizing="FILL"))
    root = _spec(width=1280, height=720, children=[child])
    result = normalize_spec_for_viewport(root, (1280, 720))
    assert result.children[0].dimensions.width == 400


def test_root_dimensions_never_scaled() -> None:
    """Root node passes through; only children are transformed."""
    child = _spec(width=720, layout=LayoutSpec(horizontal_sizing="FILL"))
    root = _spec(width=1440, height=900, children=[child])
    result = normalize_spec_for_viewport(root, (1280, 800))
    # Root dimensions express the design canvas, not the browser viewport
    assert result.dimensions.width == 1440
    assert result.dimensions.height == 900


# --- Sizing-mode rules -----------------------------------------------------


@pytest.mark.parametrize(
    ("sizing", "expected_width"),
    [
        ("FILL", 720.0 * (1280.0 / 1440.0)),  # scaled
        ("HUG", 720.0),  # unchanged
        ("FIXED", 720.0),  # unchanged
    ],
)
def test_horizontal_sizing_modes(sizing: str, expected_width: float) -> None:
    child = _spec(width=720, layout=LayoutSpec(horizontal_sizing=sizing))
    root = _spec(width=1440, height=900, children=[child])
    result = normalize_spec_for_viewport(root, (1280, 900))
    assert result.children[0].dimensions.width == pytest.approx(expected_width)


@pytest.mark.parametrize(
    ("sizing", "expected_height"),
    [
        ("FILL", 200.0 * (720.0 / 900.0)),
        ("HUG", 200.0),
        ("FIXED", 200.0),
    ],
)
def test_vertical_sizing_modes(sizing: str, expected_height: float) -> None:
    child = _spec(width=400, height=200, layout=LayoutSpec(vertical_sizing=sizing))
    root = _spec(width=1280, height=900, children=[child])
    result = normalize_spec_for_viewport(root, (1280, 720))
    assert result.children[0].dimensions.height == pytest.approx(expected_height)


# --- Constraint rules ------------------------------------------------------


def test_left_right_constraint_behaves_as_fluid() -> None:
    child = _spec(
        width=600,
        layout=LayoutSpec(horizontal_sizing="FIXED"),
        constraints=Constraints(horizontal="LEFT_RIGHT"),
    )
    root = _spec(width=1440, height=900, children=[child])
    result = normalize_spec_for_viewport(root, (1280, 900))
    assert result.children[0].dimensions.width == pytest.approx(600 * (1280 / 1440))


def test_left_constraint_keeps_width_fixed() -> None:
    """LEFT constraint with FIXED sizing → width stays absolute."""
    child = _spec(
        width=600,
        layout=LayoutSpec(horizontal_sizing="FIXED"),
        constraints=Constraints(horizontal="LEFT"),
    )
    root = _spec(width=1440, height=900, children=[child])
    result = normalize_spec_for_viewport(root, (1280, 900))
    assert result.children[0].dimensions.width == 600


def test_center_constraint_treated_as_fluid_v0() -> None:
    """v0 conservative: CENTER scales until position lands in DesignSpec."""
    child = _spec(
        width=400,
        layout=LayoutSpec(horizontal_sizing="FIXED"),
        constraints=Constraints(horizontal="CENTER"),
    )
    root = _spec(width=1440, height=900, children=[child])
    result = normalize_spec_for_viewport(root, (1280, 900))
    # Treated as fluid → scaled
    assert result.children[0].dimensions.width != 400


# --- Typography scale -----------------------------------------------------


def test_fill_typography_scales_with_viewport() -> None:
    """Fluid headings scale with viewport; type-scale ratio preserved."""
    typo = TypographySpec(font_family="Inter", font_size=48, line_height_px=56)
    child = _spec(
        width=600,
        layout=LayoutSpec(horizontal_sizing="FILL"),
        typography=typo,
    )
    root = _spec(width=1440, height=900, children=[child])
    result = normalize_spec_for_viewport(root, (1280, 900))
    ratio = 1280 / 1440
    assert result.children[0].typography is not None
    assert result.children[0].typography.font_size == pytest.approx(48 * ratio)
    assert result.children[0].typography.line_height_px == pytest.approx(56 * ratio)


def test_hug_typography_unchanged() -> None:
    """A HUG button keeps its fixed font size."""
    typo = TypographySpec(font_family="Inter", font_size=14)
    child = _spec(
        width=200,
        layout=LayoutSpec(horizontal_sizing="HUG"),
        typography=typo,
    )
    root = _spec(width=1440, height=900, children=[child])
    result = normalize_spec_for_viewport(root, (1280, 900))
    assert result.children[0].typography is not None
    assert result.children[0].typography.font_size == 14


# --- Recursive nesting -----------------------------------------------------


def test_nested_children_scale_recursively() -> None:
    grandchild = _spec(width=300, layout=LayoutSpec(horizontal_sizing="FILL"), name="grand")
    middle = _spec(
        width=600,
        layout=LayoutSpec(horizontal_sizing="FILL"),
        children=[grandchild],
        name="mid",
    )
    root = _spec(width=1440, height=900, children=[middle], name="root")
    result = normalize_spec_for_viewport(root, (1280, 900))
    ratio = 1280 / 1440
    assert result.children[0].dimensions.width == pytest.approx(600 * ratio)
    assert result.children[0].children[0].dimensions.width == pytest.approx(300 * ratio)


# --- Pure function ---------------------------------------------------------


def test_pure_function_deterministic() -> None:
    child = _spec(width=400, layout=LayoutSpec(horizontal_sizing="FILL"))
    root = _spec(width=1440, height=900, children=[child])
    r1 = normalize_spec_for_viewport(root, (1280, 900))
    r2 = normalize_spec_for_viewport(root, (1280, 900))
    assert r1.model_dump() == r2.model_dump()


def test_original_spec_not_mutated() -> None:
    child = _spec(width=400, layout=LayoutSpec(horizontal_sizing="FILL"))
    root = _spec(width=1440, height=900, children=[child])
    _ = normalize_spec_for_viewport(root, (1280, 900))
    assert root.children[0].dimensions.width == 400  # unchanged


def test_type_scale_ratio_preserved() -> None:
    """Two FILL siblings keep their relative type-scale ratio after scaling."""
    h1 = _spec(
        width=600,
        layout=LayoutSpec(horizontal_sizing="FILL"),
        typography=TypographySpec(font_family="Inter", font_size=48),
        name="h1",
    )
    body = _spec(
        width=600,
        layout=LayoutSpec(horizontal_sizing="FILL"),
        typography=TypographySpec(font_family="Inter", font_size=16),
        name="body",
    )
    root = _spec(width=1440, height=900, children=[h1, body])
    result = normalize_spec_for_viewport(root, (1280, 900))
    h1_size: Any = result.children[0].typography.font_size  # type: ignore[union-attr]
    body_size: Any = result.children[1].typography.font_size  # type: ignore[union-attr]
    assert h1_size / body_size == pytest.approx(48 / 16)
