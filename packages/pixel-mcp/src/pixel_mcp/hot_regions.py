"""Hot Regions + SSIM — pure image-based signals for Level 0 Gate Pass.

Public entry points:
- :func:`compute_ssim` — global SSIM Score in [0, 1].
- :func:`compute_hot_regions` — local bbox clusters from per-pixel diff.

Both pure functions: same numpy inputs → identical outputs.

SSIM uses scikit-image (``structural_similarity``). Hot Regions uses
OpenCV (``findContours`` on a thresholded absolute-difference mask).

Inputs must be aligned to identical shape. :func:`align_images` resizes
the expected image to match the actual's dimensions when they differ —
the common case is Figma PNG export vs Playwright screenshot, which
agree on aspect ratio but may differ by a handful of pixels.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from pixel_mcp.render import BoundingBox

__all__ = ["align_images", "compute_hot_regions", "compute_ssim"]


def align_images(expected: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize ``expected`` to match ``actual``'s shape if they differ.

    Returns ``(expected_aligned, actual)``. When already aligned, returns inputs.
    """
    if expected.shape[:2] == actual.shape[:2]:
        return expected, actual
    h, w = actual.shape[:2]
    resized = cv2.resize(expected, (w, h), interpolation=cv2.INTER_AREA)
    return resized, actual


def compute_ssim(expected: np.ndarray, actual: np.ndarray) -> float:
    """Global SSIM Score in ``[0, 1]``.

    Both inputs are converted to greyscale and aligned to the same shape
    before scoring. ``1.0`` means identical, ``0.0`` means structurally
    dissimilar.
    """
    expected, actual = align_images(expected, actual)
    grey_e = _to_greyscale(expected)
    grey_a = _to_greyscale(actual)
    score: Any = structural_similarity(grey_e, grey_a, data_range=255)
    return float(score)


def compute_hot_regions(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    pixel_threshold: int = 30,
    min_bbox_area: int = 100,
) -> list[BoundingBox]:
    """Detect bbox clusters where the per-pixel diff exceeds ``pixel_threshold``.

    Args:
        expected: Reference image (HxWx3 or HxW). Resized to match ``actual``.
        actual: Render screenshot.
        pixel_threshold: Per-channel RGB delta above which a pixel is "different".
            Default 30 (covers font anti-aliasing without over-flagging).
        min_bbox_area: Drop clusters smaller than this many px². Default 100.

    Returns:
        List of :class:`BoundingBox` (one per surviving cluster). Empty when
        no significant differences exist. Sorted by area descending.
    """
    expected, actual = align_images(expected, actual)
    expected_3c = _to_3channel(expected)
    actual_3c = _to_3channel(actual)

    diff = cv2.absdiff(expected_3c, actual_3c)
    grey_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) if diff.ndim == 3 else diff
    _, mask = cv2.threshold(grey_diff, pixel_threshold, 255, cv2.THRESH_BINARY)

    # Light morphological closing so adjacent dirty pixels merge into one
    # contour. Kernel is small — we want clusters, not full-page blobs.
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: list[BoundingBox] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_bbox_area:
            continue
        regions.append(BoundingBox(x=float(x), y=float(y), w=float(w), h=float(h)))

    regions.sort(key=lambda r: r.w * r.h, reverse=True)
    return regions


def _to_greyscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _to_3channel(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img
