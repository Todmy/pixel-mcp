"""Escalation-gate runners + visual-signal computation for ``pixel-mcp check``.

Houses every Level-1+ gate (DINOv2, VLM, OmniParser, perf) plus the Level-0
visual signal computer (SSIM + Hot Regions + decomposition). Each gate
returns a small result dict the orchestrator folds into the per-pass state.

The module looks up its external dependencies (``measure_render``,
``capture_screenshot``, ``state_dir``, ``collect_perf_metrics``,
``_match_detected_to_region``) via the ``check_cmd`` module at call time so
that tests using ``patch("pixel_mcp.check_cmd.<name>")`` keep working —
``check_cmd`` is the canonical attachment point for those symbols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pixel_mcp import check_cmd as _cc
from pixel_mcp.check_envelope import MIN_BBOX_AREA
from pixel_mcp.delta import Delta
from pixel_mcp.perf_metrics import (
    PerfBudget,
    PerfMetricsError,
    judge_perf_metrics,
)
from pixel_mcp.render import (
    BoundingBox,
    MeasuredDOM,
    RenderError,
)


def _compute_visual_signals(
    *,
    figma_url: str | None,
    image_bytes: bytes | None,
    route: str,
    viewport: tuple[int, int],
    wait_for: str | None,
    dom: MeasuredDOM,
    project_root: Path | None = None,
    iteration: int = 0,
    viewport_subfolder: str | None = None,
    browser_subfolder: str | None = None,
    browser: str = "chromium",
) -> tuple[float | None, list[BoundingBox], list[Any], str | None]:
    """Best-effort visual diff + decomposition.

    The expected image comes either from Figma (when ``figma_url`` is set)
    or from a local file (when ``image_bytes`` is set). Exactly one of the
    two is provided — the caller has already validated mutual exclusion.

    Returns ``(ssim, hot_regions, regions, error_message)``. A non-None
    ``error_message`` means the visual signal could not be computed. In
    Figma mode the caller treats this as best-effort and falls back to the
    structured-Delta verdict; in image-only mode the caller treats it as a
    hard fail (no other signal exists).

    When ``iteration`` is set (and a usable ``project_root`` is resolved),
    expected/actual Crops for each decomposed Region are persisted to
    ``<project_root>/.pixel-mcp/crops/iter-<iteration>/`` so the Level 1
    DINOv2 gate can score them. Without this wiring the gate sees Regions
    with ``expected_crop_path=None`` and silently auto-passes — a known
    foot-gun documented in the Valis lesson
    ``lesson-dinov2-gate-noop-without-crop-persistence``.
    """
    try:
        # Lazy imports — keep cv2/skimage out of the critical path when
        # visual signals are not configured.
        import io  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        from pixel_mcp.decompose import decompose_hot_regions  # noqa: PLC0415
        from pixel_mcp.hot_regions import (  # noqa: PLC0415
            compute_hot_regions,
            compute_ssim,
        )

        if image_bytes is not None:
            expected_png = image_bytes
        else:
            # Figma path
            from pixel_mcp.figma_client import FigmaClient  # noqa: PLC0415
            from pixel_mcp.figma_url import parse_figma_url  # noqa: PLC0415

            parsed = parse_figma_url(figma_url)  # type: ignore[arg-type]
            with FigmaClient() as client:
                expected_png = client.fetch_node_png_bytes(parsed.file_id, parsed.node_id)

        actual_png = _cc.capture_screenshot(
            route=route,
            viewport=viewport,
            wait_for=wait_for,
            browser=browser,  # type: ignore[arg-type]
        )

        expected_img = np.array(Image.open(io.BytesIO(expected_png)).convert("RGB"))
        actual_img = np.array(Image.open(io.BytesIO(actual_png)).convert("RGB"))

        ssim = compute_ssim(expected_img, actual_img)
        bboxes = compute_hot_regions(
            expected_img,
            actual_img,
            min_bbox_area=MIN_BBOX_AREA,
        )
        crops_root = _cc.state_dir(project_root) / "crops"
        regions = decompose_hot_regions(
            bboxes,
            dom,
            expected_image=expected_img,
            actual_image=actual_img,
            crops_dir=crops_root,
            iteration=iteration,
            viewport_subfolder=viewport_subfolder,
            browser_subfolder=browser_subfolder,
        )
        # Persist the full ``actual`` screenshot per-iteration so the v1.5-2
        # OmniParser augmentation can run on the page-scale image (not just
        # per-region crops). Same iter-N folder as the crops keeps the on-disk
        # layout consistent — review/snapshot tooling already iterates it.
        # Multi-viewport (v2-1) nests under ``viewport-WxH`` so each
        # breakpoint owns an isolated screenshot.
        try:
            iter_dir = crops_root / f"iter-{iteration}"
            if browser_subfolder:
                iter_dir = iter_dir / browser_subfolder
            if viewport_subfolder:
                iter_dir = iter_dir / viewport_subfolder
            iter_dir.mkdir(parents=True, exist_ok=True)
            actual_path = iter_dir / "actual.png"
            Image.fromarray(actual_img).save(actual_path)
        except OSError:
            # Crop persistence is best-effort — OmniParser step degrades to
            # a hint instead of crashing the loop.
            pass
        return ssim, bboxes, regions, None
    except Exception as exc:  # noqa: BLE001
        return None, [], [], str(exc)


def _dinov2_severity_for_gap(gap: float) -> str:
    """Map similarity-gap to Delta severity per PRD #10 (DINOv2 escalation)."""
    if gap >= 0.15:
        return "critical"
    if gap >= 0.05:
        return "major"
    return "minor"


def _run_dinov2_gate(
    *,
    regions: list[Any],
    dinov2_threshold: float,
) -> dict[str, Any]:
    """Run Level 1 (DINOv2) on residual crops produced by HierarchicalDecomposer.

    Returns a dict with:
    - ``similarities`` — list of per-region records (always a list; empty when
      no crops were available). Each entry: ``{region_index, selector,
      similarity, gap, severity}``.
    - ``failing_deltas`` — list[Delta] of pseudo-Deltas for crops that fell
      below ``dinov2_threshold``. Empty when every crop passed.
    - ``promoted`` — True iff at least one crop was scored AND all crops passed.
    - ``hint`` — optional AXI hint string (e.g. graceful-fallback notice).

    Lazy import: ``pixel_mcp_ml`` is only resolved here, so the check pipeline
    keeps working when the ML extras are not installed.
    """
    # Collect (region_index, expected_path, actual_path, selector) tuples.
    eligible: list[tuple[int, Path, Path, str | None]] = []
    for idx, region in enumerate(regions):
        exp = getattr(region, "expected_crop_path", None)
        act = getattr(region, "actual_crop_path", None)
        if exp and act:
            eligible.append(
                (
                    idx,
                    Path(exp),
                    Path(act),
                    getattr(region, "leaf_selector", None),
                )
            )

    if not eligible:
        # Nothing to score → Level 0 verdict stands. No promotion.
        return {
            "similarities": [],
            "failing_deltas": [],
            "promoted": False,
            "hint": None,
        }

    try:
        # Lazy: keep this OUT of module top-level. The pipeline must work
        # when ``pixel-mcp-ml`` (and torch/transformers underneath) is not
        # installed.
        from pixel_mcp_ml import (  # noqa: PLC0415
            compute_dinov2_similarity_batch,
        )
    except ImportError:
        return {
            "similarities": [],
            "failing_deltas": [],
            "promoted": False,
            "hint": (
                "Level 1 enabled but `pixel-mcp-ml --extra dinov2` not installed — "
                "falling back to Level 0 verdict. "
                "Install: `uv tool install pixel-mcp-ml --extra dinov2`."
            ),
        }

    pairs = [(exp, act) for _idx, exp, act, _sel in eligible]
    try:
        scores = compute_dinov2_similarity_batch(pairs)
    except Exception as exc:  # noqa: BLE001
        return {
            "similarities": [],
            "failing_deltas": [],
            "promoted": False,
            "hint": (
                f"Level 1 (DINOv2) failed at runtime ({exc!s}). Falling back to Level 0 verdict."
            ),
        }

    similarities: list[dict[str, Any]] = []
    failing_deltas: list[Delta] = []
    for (region_idx, _exp, _act, selector), score in zip(eligible, scores, strict=True):
        gap = max(0.0, dinov2_threshold - float(score))
        severity = _dinov2_severity_for_gap(gap)
        similarities.append(
            {
                "region_index": region_idx,
                "selector": selector,
                "similarity": float(score),
                "gap": gap,
                "severity": severity,
            }
        )
        if float(score) < dinov2_threshold:
            failing_deltas.append(
                Delta(
                    selector=selector or f"dinov2_region_{region_idx}",
                    figma_node_id=None,
                    property=f"dinov2_similarity_{region_idx}",
                    observed={
                        "similarity": float(score),
                        "threshold": dinov2_threshold,
                        "gap": gap,
                    },
                    expected={"similarity_gte": dinov2_threshold},
                    magnitude=gap,
                    severity=severity,  # type: ignore[arg-type]
                )
            )

    promoted = not failing_deltas  # all crops passed
    return {
        "similarities": similarities,
        "failing_deltas": failing_deltas,
        "promoted": promoted,
        "hint": None,
    }


# --- OmniParser augmentation (v1.5-2) ----------------------------------


def _bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union for ``(x, y, w, h)`` tuples.

    Returns ``0.0`` when either bbox is degenerate or they don't overlap.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _match_detected_to_region(region: Any, detections: list[Any]) -> Any | None:
    """Pick the best :class:`DetectedElement` for ``region``, if any.

    Strategy:

    1. Filter detections whose bbox contains the Region's centre point.
    2. Within that filter, return the one with the highest IoU against
       the Region bbox; ties broken by detection confidence (descending).

    Returns ``None`` when no detection covers the centre — the explicit
    contract from the v1.5-2 PRD ("if no detection covers the region
    centre, return None").
    """
    rx = region.bbox.x + region.bbox.w / 2.0
    ry = region.bbox.y + region.bbox.h / 2.0
    region_bbox = (region.bbox.x, region.bbox.y, region.bbox.w, region.bbox.h)
    candidates: list[tuple[float, float, Any]] = []  # (iou, confidence, detection)
    for det in detections:
        dx, dy, dw, dh = det.bbox
        if not (dx <= rx <= dx + dw and dy <= ry <= dy + dh):
            continue
        iou = _bbox_iou(region_bbox, det.bbox)
        candidates.append((iou, float(det.confidence), det))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def _run_omniparser_augmentation(
    *,
    regions: list[Any],
    iteration: int,
    confidence_threshold: float,
    project_root: Path | None = None,
    viewport_subfolder: str | None = None,
    browser_subfolder: str | None = None,
) -> tuple[list[Any] | None, str | None]:
    """Run OmniParser on the persisted actual screenshot and tag Regions.

    Returns ``(detections, hint)``:

    - ``detections`` — full list of :class:`DetectedElement` (model_dump-
      friendly objects), or ``None`` when no detection ran (missing
      extras, missing screenshot, runtime failure).
    - ``hint`` — optional AXI hint surfaced when the gate degraded to
      bare bboxes (the loop continues either way).

    Mutates ``regions`` in place: each Region whose centre falls inside a
    detection's bbox gets ``semantic_label`` + ``semantic_confidence``
    attached. Regions without a matching detection keep ``None`` —
    backward compatible with the v1 Region contract.
    """
    if not regions:
        return None, None

    actual_path_dir = _cc.state_dir(project_root) / "crops" / f"iter-{iteration}"
    if browser_subfolder:
        actual_path_dir = actual_path_dir / browser_subfolder
    if viewport_subfolder:
        actual_path_dir = actual_path_dir / viewport_subfolder
    actual_path = actual_path_dir / "actual.png"
    if not actual_path.exists():
        return None, (
            "OmniParser enabled but the actual screenshot was not persisted "
            "(visual signal may have failed). Falling back to bare bboxes."
        )

    try:
        from pixel_mcp_ml import (  # noqa: PLC0415
            OmniParserNotInstalledError,
            detect_ui_elements,
        )
    except ImportError:
        return None, (
            "OmniParser enabled but extras missing — falling back to bare bboxes. "
            "Install: `uv tool install pixel-mcp-ml --extra omniparser`."
        )

    try:
        detections = list(
            detect_ui_elements(actual_path, confidence_threshold=confidence_threshold)
        )
    except OmniParserNotInstalledError:
        return None, (
            "OmniParser enabled but extras missing — falling back to bare bboxes. "
            "Install: `uv tool install pixel-mcp-ml --extra omniparser`."
        )
    except Exception as exc:  # noqa: BLE001
        return None, (f"OmniParser failed at runtime ({exc!s}). Falling back to bare bboxes.")

    for region in regions:
        match = _cc._match_detected_to_region(region, detections)
        if match is not None:
            region.semantic_label = str(match.label)
            region.semantic_confidence = float(match.confidence)
    return detections, None


# --- Level 2 (VLM) gate -------------------------------------------------


def _vlm_severity_for_gap(gap: float) -> str:
    """Map VLM confidence-gap to Delta severity.

    Per PRD #10 acceptance (Level 2): gap >= 0.3 critical, >= 0.1 major,
    else minor. ``gap`` is ``threshold - confidence`` for failures and
    ``1.0`` for an outright ``no_match`` verdict (full disagreement).
    """
    if gap >= 0.3:
        return "critical"
    if gap >= 0.1:
        return "major"
    return "minor"


def _run_vlm_gate(
    *,
    regions: list[Any],
    vlm_threshold: float,
    vlm_backend: str,
) -> dict[str, Any]:
    """Run Level 2 (VLM) on residual crops the Level 1 gate let through.

    Same shape as :func:`_run_dinov2_gate`:

    - ``judgments`` — per-region records ``{region_index, selector,
      verdict, confidence, reasoning}``.
    - ``failing_deltas`` — pseudo-Deltas for crops the VLM rejected.
    - ``promoted`` — True iff at least one crop was scored AND every
      verdict was ``match`` AND every confidence >= ``vlm_threshold``.
    - ``hint`` — optional AXI hint (graceful fallback notice).

    Lazy import: ``pixel_mcp_ml`` is only resolved here so the check
    pipeline keeps working when the VLM extras are not installed. A
    missing SDK surfaces as a graceful fallback hint, not a crash.
    """
    eligible: list[tuple[int, Path, Path, str | None, str | None]] = []
    for idx, region in enumerate(regions):
        exp = getattr(region, "expected_crop_path", None)
        act = getattr(region, "actual_crop_path", None)
        if exp and act:
            eligible.append(
                (
                    idx,
                    Path(exp),
                    Path(act),
                    getattr(region, "leaf_selector", None),
                    getattr(region, "semantic_label", None),
                )
            )

    if not eligible:
        return {
            "judgments": [],
            "failing_deltas": [],
            "promoted": False,
            "hint": None,
        }

    try:
        from pixel_mcp_ml import (  # noqa: PLC0415
            VLMNotInstalledError,
            compute_vlm_judgment_batch,
        )
    except ImportError:
        return {
            "judgments": [],
            "failing_deltas": [],
            "promoted": False,
            "hint": (
                "Level 2 enabled but `pixel-mcp-ml --extra vlm` not installed — "
                "falling back to Level 1 verdict. "
                "Install: `uv tool install pixel-mcp-ml --extra vlm`."
            ),
        }

    pairs = [(exp, act) for _idx, exp, act, _sel, _lbl in eligible]
    context_labels: list[str | None] = [lbl for _idx, _e, _a, _sel, lbl in eligible]
    has_labels = any(lbl is not None for lbl in context_labels)
    backend_literal: Literal["claude", "qwen-local"] = (
        "qwen-local" if vlm_backend == "qwen-local" else "claude"
    )
    try:
        if has_labels:
            verdicts = compute_vlm_judgment_batch(
                pairs, backend=backend_literal, context_labels=context_labels
            )
        else:
            # Backward compat: keep the v1 call shape when no semantic labels
            # are present. Tests that monkeypatch compute_vlm_judgment_batch
            # without the new kwarg keep working.
            verdicts = compute_vlm_judgment_batch(pairs, backend=backend_literal)
    except VLMNotInstalledError:
        return {
            "judgments": [],
            "failing_deltas": [],
            "promoted": False,
            "hint": (
                "Level 2 enabled but vlm extras missing — "
                "falling back to Level 1 verdict. "
                "Install: `uv tool install pixel-mcp-ml --extra vlm`."
            ),
        }
    except NotImplementedError as exc:
        return {
            "judgments": [],
            "failing_deltas": [],
            "promoted": False,
            "hint": (
                f"Level 2 backend {vlm_backend!r} not implemented yet "
                f"({exc!s}). Falling back to Level 1 verdict."
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "judgments": [],
            "failing_deltas": [],
            "promoted": False,
            "hint": (
                f"Level 2 (VLM) failed at runtime ({exc!s}). Falling back to Level 1 verdict."
            ),
        }

    judgments_out: list[dict[str, Any]] = []
    failing_deltas: list[Delta] = []
    for (region_idx, _exp, _act, selector, _lbl), judgment in zip(eligible, verdicts, strict=True):
        judgments_out.append(
            {
                "region_index": region_idx,
                "selector": selector,
                "verdict": judgment.verdict,
                "confidence": float(judgment.confidence),
                "reasoning": judgment.reasoning,
            }
        )
        is_match = judgment.verdict == "match"
        confident = float(judgment.confidence) >= vlm_threshold
        if is_match and confident:
            continue
        # Failure: synthesise a pseudo-Delta. Gap = 1.0 when the verdict
        # itself disagrees (no_match), else the confidence shortfall.
        if not is_match:
            gap = (
                1.0
                if judgment.verdict == "no_match"
                else max(0.1, vlm_threshold - float(judgment.confidence))
            )
        else:
            gap = max(0.0, vlm_threshold - float(judgment.confidence))
        severity = _vlm_severity_for_gap(gap)
        failing_deltas.append(
            Delta(
                selector=selector or f"vlm_region_{region_idx}",
                figma_node_id=None,
                property=f"vlm_verdict_{region_idx}",
                observed={
                    "verdict": judgment.verdict,
                    "confidence": float(judgment.confidence),
                    "threshold": vlm_threshold,
                    "reasoning": judgment.reasoning,
                },
                expected={"verdict": "match", "confidence_gte": vlm_threshold},
                magnitude=gap,
                severity=severity,  # type: ignore[arg-type]
            )
        )

    promoted = not failing_deltas
    return {
        "judgments": judgments_out,
        "failing_deltas": failing_deltas,
        "promoted": promoted,
        "hint": None,
    }


# --- v3-1 Performance Budgets gate -----------------------------------------


def _run_perf_gate(
    *,
    route: str,
    viewport: tuple[int, int],
    browser: str,
    wait_for: str | None,
    perf_budget: PerfBudget | None,
) -> dict[str, Any]:
    """Collect Core Web Vitals for ``route`` and judge them against ``perf_budget``.

    Returns a dict with:

    - ``metrics`` — :class:`PerfMetrics` instance, or ``None`` on collection
      failure (the loop continues gracefully via the ``hint`` channel).
    - ``failing_deltas`` — pseudo-:class:`Delta`s for budget overages.
      Empty when no budget was supplied or every measured field was within
      tolerance.
    - ``hint`` — optional AXI hint surfaced when collection failed or no
      budget was provided.

    Implementation contract: NEVER raises. Any underlying
    :class:`PerfMetricsError` / :class:`RenderError` is folded into the
    hint channel so a perf failure can't crash the visual pipeline.
    """
    try:
        metrics = _cc.collect_perf_metrics(
            route=route,
            viewport=viewport,
            browser=browser,  # type: ignore[arg-type]
            wait_for=wait_for,
        )
    except (PerfMetricsError, RenderError) as exc:
        return {
            "metrics": None,
            "failing_deltas": [],
            "hint": (
                f"Perf gate enabled but collection failed ({exc!s}). "
                "Falling back — visual convergence verdict stands."
            ),
        }

    if perf_budget is None:
        return {
            "metrics": metrics,
            "failing_deltas": [],
            "hint": (
                "Perf gate enabled but no --perf-budget provided — "
                "metrics collected for visibility only, no Deltas emitted."
            ),
        }

    failing = judge_perf_metrics(metrics, perf_budget)
    return {
        "metrics": metrics,
        "failing_deltas": failing,
        "hint": None,
    }
