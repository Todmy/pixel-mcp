"""HierarchicalDecomposer — attribute Hot Regions to specific DOM elements.

Public entry point: :func:`decompose_hot_regions`. Given the list of Hot
Regions from Slice #6 and the MeasuredDOM, for each Hot Region:

1. Find the smallest visible DOM element whose bbox fully encloses the
   Hot Region's bbox.
2. Capture an expected/actual Crop pair from the corresponding screenshot
   bytes (if provided) and persist under ``.pixel-mcp/crops/iter-N/``.
3. Emit a :class:`Region` carrying selector, severity (from area), Crop
   paths, and (when supplied) the Figma Node ID via Mapping.

v0 keeps this to a single attribution pass — recursive descent into
sub-children is a v0.5 enhancement. For v0, the smallest-enclosing
ancestor is the leaf attribution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel

from pixel_mcp.render import BoundingBox, MeasuredDOM

__all__ = ["Region", "decompose_hot_regions"]


Severity = Literal["critical", "major", "minor"]


class Region(BaseModel):
    """One attributed Hot Region — bbox + DOM selector + optional crops.

    ``semantic_label`` and ``semantic_confidence`` are populated by the
    v1.5-2 OmniParser augmentation in ``check_cmd``. When OmniParser is
    disabled (or no detection overlaps the region's centre), both stay
    ``None`` and downstream code preserves the v1 behaviour.
    """

    bbox: BoundingBox
    area_px2: float
    severity: Severity
    leaf_selector: str | None = None
    figma_node_id: str | None = None
    expected_crop_path: str | None = None
    actual_crop_path: str | None = None
    diff_crop_path: str | None = None
    semantic_label: str | None = None
    semantic_confidence: float | None = None


def decompose_hot_regions(
    hot_regions: list[BoundingBox],
    dom: MeasuredDOM,
    *,
    expected_image: np.ndarray | None = None,
    actual_image: np.ndarray | None = None,
    crops_dir: Path | None = None,
    iteration: int = 0,
    mappings: dict[str, str] | None = None,
) -> list[Region]:
    """Attribute each Hot Region to a DOM element and emit a Region list.

    Args:
        hot_regions: Output from :func:`pixel_mcp.hot_regions.compute_hot_regions`.
        dom: MeasuredDOM for the current Render.
        expected_image: Optional Figma-side image (numpy HxWx3). When provided
            alongside ``actual_image`` and ``crops_dir``, expected/actual
            Crops are persisted.
        actual_image: Optional Render-side image.
        crops_dir: Optional ``.pixel-mcp/crops/`` root. Iteration sub-folder
            created automatically.
        iteration: Current Iteration number for the crop sub-folder naming.
        mappings: Optional ``figma_node_id → css_selector`` map (Slice #18).
            When present, regions whose leaf_selector matches get a
            figma_node_id assigned.

    Returns:
        List of :class:`Region` — one per attributed Hot Region. Empty when
        ``hot_regions`` is empty.
    """
    if not hot_regions:
        return []

    # Invert mappings for selector→figma lookup
    inverted: dict[str, str] = {}
    if mappings:
        for fid, sel in mappings.items():
            inverted[sel] = fid

    iter_dir: Path | None = None
    if crops_dir is not None and expected_image is not None and actual_image is not None:
        iter_dir = crops_dir / f"iter-{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

    out: list[Region] = []
    for i, region in enumerate(hot_regions):
        selector = _find_enclosing_element(region, dom)
        area = region.w * region.h
        sev = _severity_for_area(area)

        exp_path: str | None = None
        act_path: str | None = None
        if iter_dir is not None and expected_image is not None and actual_image is not None:
            exp_path = str(_save_crop(expected_image, region, iter_dir / f"exp-r{i + 1}.png"))
            act_path = str(_save_crop(actual_image, region, iter_dir / f"act-r{i + 1}.png"))

        figma_node = inverted.get(selector) if selector else None

        out.append(
            Region(
                bbox=region,
                area_px2=area,
                severity=sev,
                leaf_selector=selector,
                figma_node_id=figma_node,
                expected_crop_path=exp_path,
                actual_crop_path=act_path,
            )
        )
    return out


def _find_enclosing_element(region: BoundingBox, dom: MeasuredDOM) -> str | None:
    """Return the smallest DOM element bbox that fully encloses ``region``."""
    best: tuple[float, str] | None = None  # (area, selector)
    rx2 = region.x + region.w
    ry2 = region.y + region.h
    for el in dom.elements:
        ex1 = el.bounding_box.x
        ey1 = el.bounding_box.y
        ex2 = ex1 + el.bounding_box.w
        ey2 = ey1 + el.bounding_box.h
        if ex1 <= region.x and ey1 <= region.y and ex2 >= rx2 and ey2 >= ry2:
            el_area = el.bounding_box.w * el.bounding_box.h
            if best is None or el_area < best[0]:
                best = (el_area, el.selector)
    return best[1] if best else None


def _severity_for_area(area: float) -> Severity:
    """Image-only-mode severity from Hot Region area (per CONTEXT.md)."""
    if area >= 50_000:
        return "critical"
    if area >= 1_000:
        return "major"
    return "minor"


def _save_crop(image: np.ndarray, region: BoundingBox, out_path: Path) -> Path:
    """Crop ``image`` to ``region`` and write as PNG."""
    x1 = max(0, int(region.x))
    y1 = max(0, int(region.y))
    x2 = min(image.shape[1], int(region.x + region.w))
    y2 = min(image.shape[0], int(region.y + region.h))
    if x2 <= x1 or y2 <= y1:
        # Degenerate region — write a 1x1 placeholder so the path is real.
        Image.fromarray(image[:1, :1]).save(out_path)
        return out_path
    Image.fromarray(image[y1:y2, x1:x2]).save(out_path)
    return out_path
