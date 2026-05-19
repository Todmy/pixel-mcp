"""Unit tests for hot_regions — pure-function image-diff signals."""

from __future__ import annotations

import numpy as np
import pytest
from pixel_mcp.hot_regions import (
    align_images,
    compute_hot_regions,
    compute_ssim,
)


def _solid(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


# --- align_images ---------------------------------------------------------


def test_align_passes_through_when_shapes_match() -> None:
    a = _solid(100, 200, (255, 0, 0))
    b = _solid(100, 200, (255, 0, 0))
    aligned_a, aligned_b = align_images(a, b)
    assert aligned_a.shape == a.shape
    assert aligned_b.shape == b.shape


def test_align_resizes_expected_to_actual() -> None:
    a = _solid(100, 200, (255, 0, 0))
    b = _solid(150, 300, (255, 0, 0))
    aligned_a, aligned_b = align_images(a, b)
    assert aligned_a.shape == b.shape
    assert aligned_b.shape == b.shape


# --- compute_ssim ---------------------------------------------------------


def test_ssim_identical_images_close_to_one() -> None:
    a = _solid(100, 200, (255, 0, 0))
    b = _solid(100, 200, (255, 0, 0))
    assert compute_ssim(a, b) == pytest.approx(1.0)


def test_ssim_completely_different_images_low() -> None:
    a = _solid(100, 200, (0, 0, 0))
    b = _solid(100, 200, (255, 255, 255))
    # Two solid-color images of opposite luminance — SSIM should be very low.
    assert compute_ssim(a, b) < 0.3


def test_ssim_pure_function_deterministic() -> None:
    a = _solid(50, 50, (128, 128, 128))
    b = _solid(50, 50, (130, 130, 130))
    assert compute_ssim(a, b) == compute_ssim(a, b)


# --- compute_hot_regions --------------------------------------------------


def test_hot_regions_empty_for_identical_images() -> None:
    a = _solid(100, 200, (255, 0, 0))
    b = _solid(100, 200, (255, 0, 0))
    assert compute_hot_regions(a, b) == []


def test_hot_regions_detects_solid_diff_block() -> None:
    a = _solid(200, 200, (255, 255, 255))
    b = _solid(200, 200, (255, 255, 255))
    # Inject a 50×50 black square in b
    b[50:100, 50:100] = (0, 0, 0)
    regions = compute_hot_regions(a, b)
    assert len(regions) == 1
    r = regions[0]
    # bbox should roughly cover the injected region (allow morphological dilation slack)
    assert r.w >= 50
    assert r.h >= 50
    assert r.x >= 45
    assert r.y >= 45


def test_hot_regions_filters_below_min_area() -> None:
    a = _solid(100, 100, (255, 255, 255))
    b = a.copy()
    # Tiny 5×5 patch — below default min_bbox_area=100
    b[10:15, 10:15] = (0, 0, 0)
    assert compute_hot_regions(a, b, min_bbox_area=100) == []
    # Lower the threshold and it appears
    assert len(compute_hot_regions(a, b, min_bbox_area=1)) >= 1


def test_hot_regions_sorted_by_area_desc() -> None:
    a = _solid(300, 300, (255, 255, 255))
    b = a.copy()
    b[10:50, 10:50] = (0, 0, 0)  # 40x40 = 1600 px²
    b[100:200, 100:200] = (0, 0, 0)  # 100x100 = 10000 px²
    regions = compute_hot_regions(a, b)
    assert len(regions) == 2
    assert regions[0].w * regions[0].h >= regions[1].w * regions[1].h


def test_hot_regions_pure_function() -> None:
    a = _solid(100, 100, (255, 255, 255))
    b = a.copy()
    b[40:60, 40:60] = (0, 0, 0)
    r1 = compute_hot_regions(a, b)
    r2 = compute_hot_regions(a, b)
    assert len(r1) == len(r2)
    for x, y in zip(r1, r2, strict=True):
        assert x.model_dump() == y.model_dump()
