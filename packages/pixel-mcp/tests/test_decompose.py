"""Unit tests for HierarchicalDecomposer — pure-function attribution + crop IO."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pixel_mcp.decompose import Region, decompose_hot_regions
from pixel_mcp.render import (
    BoundingBox,
    ComputedStyle,
    MeasuredDOM,
    MeasuredElement,
)


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


def _dom_with(elements: list[tuple[str, float, float, float, float]]) -> MeasuredDOM:
    return MeasuredDOM(
        route="http://localhost:3000/",
        viewport=(1280, 720),
        measured_at=datetime.now(UTC),
        elements=[
            MeasuredElement(
                selector=sel,
                bounding_box=BoundingBox(x=x, y=y, w=w, h=h),
                computed_style=_style(),
            )
            for sel, x, y, w, h in elements
        ],
    )


# --- Attribution ----------------------------------------------------------


def test_empty_hot_regions_returns_empty() -> None:
    dom = _dom_with([("body", 0, 0, 1280, 720)])
    assert decompose_hot_regions([], dom) == []


def test_finds_smallest_enclosing_element() -> None:
    """When a Hot Region sits inside a nested DOM element, pick the inner one."""
    dom = _dom_with(
        [
            ("body", 0, 0, 1280, 720),
            ("main", 0, 0, 1280, 600),
            ("section.hero", 0, 0, 1280, 400),
            ("button.cta", 100, 100, 200, 50),
        ]
    )
    # Hot Region inside the button
    region = BoundingBox(x=120, y=110, w=20, h=20)
    result = decompose_hot_regions([region], dom)
    assert len(result) == 1
    assert result[0].leaf_selector == "button.cta"


def test_unattributable_region_has_no_selector() -> None:
    """When no DOM element fully encloses the region, leaf_selector is None."""
    dom = _dom_with([("button", 0, 0, 100, 50)])
    region = BoundingBox(x=500, y=500, w=200, h=200)  # outside everything
    result = decompose_hot_regions([region], dom)
    assert result[0].leaf_selector is None


# --- Severity by area -----------------------------------------------------


def test_severity_critical_for_large_region() -> None:
    dom = _dom_with([("body", 0, 0, 1280, 720)])
    region = BoundingBox(x=0, y=0, w=400, h=200)  # 80,000 px²
    result = decompose_hot_regions([region], dom)
    assert result[0].severity == "critical"
    assert result[0].area_px2 == 80_000


def test_severity_major_for_medium_region() -> None:
    dom = _dom_with([("body", 0, 0, 1280, 720)])
    region = BoundingBox(x=0, y=0, w=100, h=50)  # 5,000 px²
    result = decompose_hot_regions([region], dom)
    assert result[0].severity == "major"


def test_severity_minor_for_small_region() -> None:
    dom = _dom_with([("body", 0, 0, 1280, 720)])
    region = BoundingBox(x=0, y=0, w=20, h=20)  # 400 px²
    result = decompose_hot_regions([region], dom)
    assert result[0].severity == "minor"


# --- Mapping integration --------------------------------------------------


def test_figma_node_id_assigned_when_mapped() -> None:
    dom = _dom_with([("button.cta", 0, 0, 200, 50)])
    region = BoundingBox(x=10, y=10, w=20, h=20)
    mappings = {"1:23": "button.cta"}
    result = decompose_hot_regions([region], dom, mappings=mappings)
    assert result[0].figma_node_id == "1:23"


def test_no_figma_node_id_when_unmapped() -> None:
    dom = _dom_with([("button.cta", 0, 0, 200, 50)])
    region = BoundingBox(x=10, y=10, w=20, h=20)
    result = decompose_hot_regions([region], dom)
    assert result[0].figma_node_id is None


# --- Crop persistence -----------------------------------------------------


def test_crops_written_when_dir_and_images_provided(tmp_path: Path) -> None:
    dom = _dom_with([("body", 0, 0, 200, 200)])
    region = BoundingBox(x=10, y=10, w=50, h=50)

    expected = np.full((200, 200, 3), 255, dtype=np.uint8)
    actual = np.full((200, 200, 3), 0, dtype=np.uint8)

    result = decompose_hot_regions(
        [region],
        dom,
        expected_image=expected,
        actual_image=actual,
        crops_dir=tmp_path,
        iteration=3,
    )
    assert result[0].expected_crop_path is not None
    assert result[0].actual_crop_path is not None
    assert Path(result[0].expected_crop_path).exists()
    assert Path(result[0].actual_crop_path).exists()
    # Verify iteration sub-folder structure
    assert (tmp_path / "iter-3").exists()


def test_no_crops_when_no_images() -> None:
    dom = _dom_with([("body", 0, 0, 200, 200)])
    region = BoundingBox(x=10, y=10, w=50, h=50)
    result = decompose_hot_regions([region], dom)
    assert result[0].expected_crop_path is None
    assert result[0].actual_crop_path is None


# --- Pure-ish function ----------------------------------------------------


def test_deterministic_when_no_io() -> None:
    """No IO → identical outputs across calls (Region model is BaseModel)."""
    dom = _dom_with([("body", 0, 0, 1280, 720), ("button.cta", 100, 100, 200, 50)])
    regions = [BoundingBox(x=120, y=110, w=20, h=20)]
    r1 = decompose_hot_regions(regions, dom)
    r2 = decompose_hot_regions(regions, dom)
    assert [r.model_dump() for r in r1] == [r.model_dump() for r in r2]


def test_region_is_pydantic_model() -> None:
    """Region must JSON-roundtrip cleanly (used in AXI envelope)."""
    r = Region(bbox=BoundingBox(x=0, y=0, w=10, h=10), area_px2=100.0, severity="minor")
    raw = r.model_dump_json()
    assert Region.model_validate_json(raw).model_dump() == r.model_dump()
