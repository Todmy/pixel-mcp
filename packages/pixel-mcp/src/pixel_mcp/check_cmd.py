"""Composite ``pixel-mcp check`` — one Iteration of the Convergence Loop.

Runs ``spec`` + ``measure`` + ``diff`` + ``judge`` end-to-end (Figma mode) or
``measure`` + visual diff + Hot Region attribution (image-only mode), and
returns a single AXI envelope. This is the command Ralph Loop / any Loop
Runner invokes in its inner body.

Modes
-----

- **Figma mode** — ``--figma <url> --route <url>`` (the original v0 path).
  Extracts a DesignSpec, runs DeltaDiffer + ConvergenceJudge, then layers
  Level 0 visual signals (SSIM + Hot Regions) on top.

- **Image-only mode** (v0.5-1) — ``--image <path> --route <url>``. For users
  without Figma. Skips DesignSpec extraction entirely; convergence is
  driven purely by Level 0 visual signals (``ssim_score >= ssim_threshold``
  AND zero Hot Regions ``>= min_bbox_area``). Hot Regions still feed
  through ``decompose_hot_regions`` for DOM attribution. Pseudo-Deltas are
  synthesized from Hot Regions so loop economics (stuck / regression /
  history) continue to work unchanged.

Exit codes (per [PRD #10](https://github.com/Todmy/PBaaS/issues/10)):

- 0 — Final Convergence at the highest currently-enabled Level.
- 1 — Deltas present; loop continues.
- 2 — Ready for Level 3 (Slice #19+; reserved here).
- 3 — Regression detected.
- 10 — Max iterations exceeded.
- 11 — Stuck (same delta hash N times in a row).
- 12 — Fatal (Figma/Render/IO/CLI error).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.delta import Delta, diff_design_vs_render
from pixel_mcp.figma_client import FigmaApiError, FigmaAuthError, FigmaError, FigmaNotFoundError
from pixel_mcp.figma_url import FigmaUrlError
from pixel_mcp.human_feedback_cmd import mark_consumed, read_feedback
from pixel_mcp.judge import Tolerance, judge_deltas
from pixel_mcp.loop_state import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_STUCK_THRESHOLD,
    HASH_TRAIL_MAX,
    append_history,
    detect_regression,
    detect_stuck,
    hash_deltas_bucketed,
    now_utc,
    read_state,
    write_state,
)
from pixel_mcp.normalize import normalize_spec_for_viewport
from pixel_mcp.render import (
    BoundingBox,
    ChromiumNotInstalledError,
    MeasuredDOM,
    PlaywrightNotInstalledError,
    RenderError,
    RouteUnreachableError,
    WaitForTimeoutError,
    capture_screenshot,
    measure_render,
)
from pixel_mcp.spec import DesignSpec, UnsupportedNodeTypeError, extract_spec
from pixel_mcp.state import state_dir

EXIT_CONVERGED = 0
EXIT_DELTAS = 1
EXIT_READY_FOR_LEVEL_3 = 2  # reserved
EXIT_REGRESSION = 3
EXIT_MAX_ITERATIONS = 10
EXIT_STUCK = 11
EXIT_FATAL = 12


def run(
    figma_url: str | None = None,
    route: str = "",
    viewport: tuple[int, int] = (1280, 720),
    selectors: list[str] | None = None,
    wait_for: str | None = None,
    refresh_spec: bool = False,
    treat_minor_as_blocking: bool = False,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    stuck_threshold: int = DEFAULT_STUCK_THRESHOLD,
    image_path: str | Path | None = None,
    enable_dinov2: bool = False,
    dinov2_threshold: float = 0.95,
    enable_vlm: bool = False,
    vlm_threshold: float = 0.7,
    vlm_backend: str = "claude",
    enable_human_gate: bool = False,
) -> tuple[Envelope, int]:
    """Run one Iteration of the Convergence Loop.

    Exactly one of ``figma_url`` or ``image_path`` must be provided. Both or
    neither returns an EXIT_FATAL envelope.

    Never raises — all errors are folded into the AXI envelope. Loop
    economics (iteration counter, stuck detection, regression, max-iter)
    are enforced via ``.pixel-mcp/state.json``.
    """
    # --- 0a) Validate Design Source selection ---
    if figma_url and image_path:
        return _fatal_envelope(
            "design_source_conflict",
            "Both --figma and --image were provided. Pick exactly one Design Source.",
            [
                "Use --figma <url> for Figma mode.",
                "Use --image <path> for image-only mode.",
            ],
        ), EXIT_FATAL
    if not figma_url and not image_path:
        return _fatal_envelope(
            "design_source_missing",
            "No Design Source provided. Pass --figma <url> or --image <path>.",
            [
                "Figma mode: pass --figma figma.com/design/<id>?node-id=<node>.",
                "Image-only mode: pass --image path/to/design.png.",
            ],
        ), EXIT_FATAL

    image_only = image_path is not None
    mode = "image" if image_only else "figma"

    # --- 0b) Iteration counter + max-iter pre-check ---
    state = read_state()
    state.iteration += 1
    state.last_invocation_at = now_utc()
    if state.iteration > max_iterations:
        write_state(state)
        return _fatal_envelope(
            "max_iterations_exceeded",
            f"Iteration {state.iteration} exceeded --max-iterations={max_iterations}.",
            [
                "Run `pixel-mcp reset` to start a fresh session.",
                f"Or increase --max-iterations (current {max_iterations}).",
            ],
        ), EXIT_MAX_ITERATIONS

    # --- 0c) Image-only Design Source pre-validation ---
    image_bytes: bytes | None = None
    if image_only:
        path_obj = Path(image_path)  # type: ignore[arg-type]
        if not path_obj.exists():
            return _fatal_envelope(
                "image_not_found",
                f"--image path does not exist: {path_obj}",
                ["Pass a path to a real PNG or JPG file."],
            ), EXIT_FATAL
        try:
            image_bytes = path_obj.read_bytes()
        except OSError as exc:
            return _fatal_envelope(
                "image_read_error",
                f"Failed to read --image {path_obj}: {exc}",
                ["Check file permissions and that it's a regular file."],
            ), EXIT_FATAL

    # --- 1) Extract DesignSpec (Figma mode only) ---
    spec: DesignSpec | None = None
    if not image_only:
        try:
            spec = extract_spec(figma_url, refresh=refresh_spec)  # type: ignore[arg-type]
        except FigmaUrlError as exc:
            return _fatal_envelope(
                "figma_url_error",
                str(exc),
                ["Pass a Figma URL of the form figma.com/design/<id>?node-id=<node>."],
            ), EXIT_FATAL
        except FigmaAuthError as exc:
            return _fatal_envelope(
                "figma_auth_error",
                str(exc),
                ["Set FIGMA_TOKEN to a valid personal-access token."],
            ), EXIT_FATAL
        except FigmaNotFoundError as exc:
            return _fatal_envelope(
                "figma_not_found",
                str(exc),
                ["Check the file-id and node-id — the node may have been deleted."],
            ), EXIT_FATAL
        except UnsupportedNodeTypeError as exc:
            return _fatal_envelope(
                "unsupported_node_type",
                str(exc),
                ["Use a Figma Frame, Component Instance, or Master Component."],
            ), EXIT_FATAL
        except (FigmaApiError, FigmaError) as exc:
            return _fatal_envelope(
                "figma_error",
                str(exc),
                ["See `pixel-mcp doctor` for environment diagnostics."],
            ), EXIT_FATAL

    # --- 2) Measure Render ---
    try:
        dom, truncated = measure_render(
            route=route,
            viewport=viewport,
            selectors=selectors,
            wait_for=wait_for,
        )
    except PlaywrightNotInstalledError as exc:
        return _fatal_envelope(
            "playwright_not_installed",
            str(exc),
            ["Run `uv sync` then `uv run playwright install chromium`."],
        ), EXIT_FATAL
    except ChromiumNotInstalledError as exc:
        return _fatal_envelope(
            "chromium_not_installed",
            str(exc),
            ["Run `uv run playwright install chromium` (one-time, ~150MB)."],
        ), EXIT_FATAL
    except WaitForTimeoutError as exc:
        return _fatal_envelope(
            "wait_for_timeout",
            str(exc),
            ["Re-check the --wait-for selector — the element never appeared."],
        ), EXIT_FATAL
    except RouteUnreachableError as exc:
        return _fatal_envelope(
            "route_unreachable",
            str(exc),
            ["Verify the dev server is running and the route URL is correct."],
        ), EXIT_FATAL
    except RenderError as exc:
        return _fatal_envelope(
            "render_error",
            str(exc),
            ["See `pixel-mcp doctor` for environment diagnostics."],
        ), EXIT_FATAL

    # --- 3) Structured Deltas / Judgment (Figma mode only) ---
    if not image_only:
        assert spec is not None
        normalized_spec = normalize_spec_for_viewport(spec, dom.viewport)
        deltas = diff_design_vs_render(normalized_spec, dom)
        judgment = judge_deltas(
            deltas,
            tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
        )
    else:
        deltas = []
        judgment = judge_deltas([])  # vacuously converged; visual signals will decide

    # --- 4) Level 0 visual signals: SSIM + Hot Regions + decomposition ---
    ssim_score, hot_regions, regions, visual_error = _compute_visual_signals(
        figma_url=figma_url,
        image_bytes=image_bytes,
        route=route,
        viewport=viewport,
        wait_for=wait_for,
        dom=dom,
        iteration=state.iteration,
    )

    # --- 4b) Image-only mode: synthesize pseudo-Deltas from Hot Regions ---
    # These feed loop economics (stuck / regression / history) so the rest
    # of the pipeline works unchanged. They also surface in data.deltas so
    # the Agent sees what's wrong.
    if image_only:
        significant = [r for r in hot_regions if r.w * r.h >= MIN_BBOX_AREA]
        # Build a selector lookup keyed on (x, y, w, h) so we can attach DOM
        # attribution to each pseudo-Delta.
        selector_for: dict[tuple[float, float, float, float], str | None] = {}
        for region_obj in regions:
            bbox = region_obj.bbox
            selector_for[(bbox.x, bbox.y, bbox.w, bbox.h)] = region_obj.leaf_selector

        deltas = [
            _hot_region_to_delta(
                index=i,
                bbox=r,
                selector=selector_for.get((r.x, r.y, r.w, r.h)),
            )
            for i, r in enumerate(significant)
        ]
        judgment = judge_deltas(
            deltas,
            tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
        )

    # Combined Level 0 Gate Pass. Image-only mode ignores the structured
    # Deltas verdict (there are no real structured Deltas) and gates purely
    # on visual signals.
    if visual_error is not None and not image_only:
        # Figma mode: best-effort visual signal — fall back to structured verdict.
        visual_passes = True
    else:
        if image_only and visual_error is not None:
            # Image-only: visual signal IS the verdict. Can't pass without it.
            visual_passes = False
        else:
            ssim_ok = ssim_score is None or ssim_score >= SSIM_THRESHOLD
            regions_ok = not any(r.w * r.h >= MIN_BBOX_AREA for r in hot_regions)
            visual_passes = ssim_ok and regions_ok

    if image_only:
        # Convergence is exclusively visual in image-only mode.
        overall_converged = visual_passes
    else:
        overall_converged = judgment.converged and visual_passes

    # --- 4c) Level 1 (DINOv2) gate, opt-in ---
    # Only runs once Level 0 has Gate-Passed. ``level_reached`` reflects the
    # highest gate satisfied: 0 means Level 0 (CV) is the ceiling, 1 means
    # DINOv2 also passed. If any crop fails the threshold, we synthesize one
    # pseudo-Delta per failing crop and revert overall_converged to False.
    level_reached = 0
    dinov2_similarities: list[dict[str, Any]] | None = None
    dinov2_hint: str | None = None
    if enable_dinov2 and overall_converged:
        dinov2_result = _run_dinov2_gate(
            regions=regions,
            dinov2_threshold=dinov2_threshold,
        )
        dinov2_similarities = dinov2_result["similarities"]
        dinov2_hint = dinov2_result["hint"]
        failing = dinov2_result["failing_deltas"]
        if dinov2_result["promoted"]:
            # Achieved Level 1; next un-implemented gate is Level 2 (VLM, v1 scope).
            level_reached = 1
        if failing:
            # ANY crop fail → Level 1 not satisfied; revert convergence.
            overall_converged = False
            level_reached = 0
            # Append the pseudo-Deltas so loop-economics / hash / history see them.
            deltas = list(deltas) + failing
            judgment = judge_deltas(
                deltas,
                tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
            )

    # --- 4d) Level 2 (VLM) gate, opt-in ---
    # Promotes a Level 1 Gate Pass to Level 2 by asking a vision-language
    # model to verbally judge each residual crop pair. Only runs when:
    #   - --enable-vlm is set,
    #   - Level 1 actually passed (level_reached == 1 AND overall_converged),
    # so we never call the VLM on crops the cheaper gates already rejected.
    # On any failure (no_match / ambiguous / confidence below threshold)
    # we emit pseudo-Deltas, revert overall_converged, and keep
    # level_reached at 1 — the next promotion target stays Level 2.
    vlm_judgments: list[dict[str, Any]] | None = None
    vlm_hint: str | None = None
    if enable_vlm and overall_converged and level_reached >= 1:
        vlm_result = _run_vlm_gate(
            regions=regions,
            vlm_threshold=vlm_threshold,
            vlm_backend=vlm_backend,
        )
        vlm_judgments = vlm_result["judgments"]
        vlm_hint = vlm_result["hint"]
        failing_vlm = vlm_result["failing_deltas"]
        if vlm_result["promoted"]:
            level_reached = 2
        if failing_vlm:
            overall_converged = False
            # Stay at Level 1 — Level 1 already passed, just couldn't get to Level 2.
            level_reached = 1
            deltas = list(deltas) + failing_vlm
            judgment = judge_deltas(
                deltas,
                tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
            )

    # --- 4e) Level 3 (Human review) gate, opt-in ---
    # Final escalation tier. Runs only when (a) --enable-human-gate is set
    # AND (b) every automated gate currently enabled has Gate-Passed (i.e.
    # ``overall_converged`` is still True at this point). Three branches:
    #
    # - feedback file absent / already consumed → emit EXIT_READY_FOR_LEVEL_3
    #   with a pointer at the review + human_feedback tools. The loop pauses.
    # - feedback unconsumed AND verdict=approved → promote to Level 3, mark
    #   the file consumed, fall through to a converged exit.
    # - feedback unconsumed AND verdict=rejected → synthesize a critical
    #   pseudo-Delta with property="human_review" (notes carried as
    #   `expected`). The Judge re-runs, overall_converged flips, the loop
    #   re-opens. Different rejection notes hash to different buckets so
    #   stuck-detection stays accurate.
    human_verdict: str | None = None
    human_notes: str | None = None
    human_gate_pending = False
    if enable_human_gate and overall_converged:
        feedback = read_feedback()
        if feedback is None or feedback.get("consumed", False):
            human_gate_pending = True
            human_verdict = "pending"
        else:
            verdict = feedback.get("verdict")
            if verdict == "approved":
                human_verdict = "approved"
                level_reached = 3
                mark_consumed()
            elif verdict == "rejected":
                human_verdict = "rejected"
                human_notes = feedback.get("notes") or ""
                mark_consumed()
                deltas = list(deltas) + [
                    Delta(
                        selector="<human>",
                        figma_node_id=None,
                        property="human_review",
                        observed="rejected_by_human",
                        expected=human_notes,
                        magnitude=None,
                        severity="critical",
                    )
                ]
                judgment = judge_deltas(
                    deltas,
                    tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
                )
                overall_converged = False
                # Keep level_reached at whatever the automated gates earned;
                # the human just blocks promotion to Level 3.
            else:
                # Unknown / malformed verdict — treat as pending.
                human_gate_pending = True
                human_verdict = "pending"

    # --- 5) Loop economics: stuck detection + regression + state update ---
    current_hash = hash_deltas_bucketed(deltas)
    is_stuck = detect_stuck(state.recent_hashes, current_hash, threshold=stuck_threshold)
    current_level_passed = level_reached if overall_converged else 0
    is_regression = detect_regression(state, current_level_passed)

    state.last_delta_hash = current_hash
    state.recent_hashes = (state.recent_hashes + [current_hash])[-HASH_TRAIL_MAX:]
    if overall_converged and current_level_passed > state.highest_level_reached:
        state.highest_level_reached = current_level_passed
    write_state(state)

    append_history(
        {
            "iteration": state.iteration,
            "session_id": state.session_id,
            "mode": mode,
            "delta_count": len(deltas),
            "delta_hash": current_hash,
            "ssim_score": ssim_score,
            "hot_region_count": len(hot_regions),
            "converged": overall_converged,
            "timestamp": now_utc().isoformat(),
        }
    )

    envelope = _success_envelope(
        spec=spec,
        dom=dom,
        deltas=deltas,
        judgment_data=json.loads(judgment.model_dump_json()),
        truncated=truncated,
        ssim_score=ssim_score,
        hot_regions=hot_regions,
        regions=regions,
        visual_error=visual_error,
        overall_converged=overall_converged,
        iteration=state.iteration,
        session_id=state.session_id,
        is_stuck=is_stuck,
        is_regression=is_regression,
        max_iterations=max_iterations,
        mode=mode,
        level_reached=level_reached,
        dinov2_enabled=enable_dinov2,
        dinov2_threshold=dinov2_threshold if enable_dinov2 else None,
        dinov2_similarities=dinov2_similarities,
        dinov2_hint=dinov2_hint,
        vlm_enabled=enable_vlm,
        vlm_threshold=vlm_threshold if enable_vlm else None,
        vlm_backend=vlm_backend if enable_vlm else None,
        vlm_judgments=vlm_judgments,
        vlm_hint=vlm_hint,
        human_gate_enabled=enable_human_gate,
        human_verdict=human_verdict,
        human_notes=human_notes,
        human_gate_pending=human_gate_pending,
    )

    if is_regression:
        return envelope, EXIT_REGRESSION
    if is_stuck:
        return envelope, EXIT_STUCK
    if human_gate_pending:
        return envelope, EXIT_READY_FOR_LEVEL_3
    return envelope, (EXIT_CONVERGED if overall_converged else EXIT_DELTAS)


# --- Level 0 visual signals (SSIM + Hot Regions) -------------------------

SSIM_THRESHOLD = 0.97
"""Default SSIM Gate Pass threshold (per PRD/CONTEXT)."""

MIN_BBOX_AREA = 100
"""Default minimum Hot Region area in px² (filters anti-aliasing noise)."""


def _severity_for_area(area: float) -> str:
    """Image-only-mode severity from Hot Region area (per CONTEXT.md)."""
    if area >= 50_000:
        return "critical"
    if area >= 1_000:
        return "major"
    return "minor"


def _hot_region_to_delta(*, index: int, bbox: BoundingBox, selector: str | None) -> Delta:
    """Synthesize a pseudo-Delta from a Hot Region for image-only mode.

    The Delta is a structured handle on a region of pixel drift — it carries
    the bbox, the area-derived severity, and the DOM selector when DOM
    attribution found one. ``property`` is a stable identifier
    (``hot_region_<n>``) so the loop-economics hash stays stable across
    Iterations when the Agent doesn't change anything.
    """
    area = bbox.w * bbox.h
    return Delta(
        selector=selector or f"hot_region_{index + 1}",
        figma_node_id=None,
        property=f"hot_region_{index + 1}",
        observed={"x": bbox.x, "y": bbox.y, "w": bbox.w, "h": bbox.h, "area_px2": area},
        expected=None,
        magnitude=area,
        severity=_severity_for_area(area),  # type: ignore[arg-type]
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

        actual_png = capture_screenshot(route=route, viewport=viewport, wait_for=wait_for)

        expected_img = np.array(Image.open(io.BytesIO(expected_png)).convert("RGB"))
        actual_img = np.array(Image.open(io.BytesIO(actual_png)).convert("RGB"))

        ssim = compute_ssim(expected_img, actual_img)
        bboxes = compute_hot_regions(
            expected_img,
            actual_img,
            min_bbox_area=MIN_BBOX_AREA,
        )
        crops_root = state_dir(project_root) / "crops"
        regions = decompose_hot_regions(
            bboxes,
            dom,
            expected_image=expected_img,
            actual_image=actual_img,
            crops_dir=crops_root,
            iteration=iteration,
        )
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

    pairs = [(exp, act) for _idx, exp, act, _sel in eligible]
    try:
        verdicts = compute_vlm_judgment_batch(pairs, backend=vlm_backend)
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
                f"Level 2 (VLM) failed at runtime ({exc!s}). " "Falling back to Level 1 verdict."
            ),
        }

    judgments_out: list[dict[str, Any]] = []
    failing_deltas: list[Delta] = []
    for (region_idx, _exp, _act, selector), judgment in zip(eligible, verdicts, strict=True):
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


def _success_envelope(
    *,
    spec: DesignSpec | None,
    dom: MeasuredDOM,
    deltas: list[Delta],
    judgment_data: dict[str, Any],
    truncated: bool,
    ssim_score: float | None = None,
    hot_regions: list[BoundingBox] | None = None,
    regions: list[Any] | None = None,
    visual_error: str | None = None,
    overall_converged: bool | None = None,
    iteration: int | None = None,
    session_id: str | None = None,
    is_stuck: bool = False,
    is_regression: bool = False,
    max_iterations: int | None = None,
    mode: str = "figma",
    level_reached: int = 0,
    dinov2_enabled: bool = False,
    dinov2_threshold: float | None = None,
    dinov2_similarities: list[dict[str, Any]] | None = None,
    dinov2_hint: str | None = None,
    vlm_enabled: bool = False,
    vlm_threshold: float | None = None,
    vlm_backend: str | None = None,
    vlm_judgments: list[dict[str, Any]] | None = None,
    vlm_hint: str | None = None,
    human_gate_enabled: bool = False,
    human_verdict: str | None = None,
    human_notes: str | None = None,
    human_gate_pending: bool = False,
) -> Envelope:
    delta_dicts = [json.loads(d.model_dump_json()) for d in deltas]
    hot_regions = hot_regions or []
    regions = regions or []
    significant_regions = [r for r in hot_regions if r.w * r.h >= MIN_BBOX_AREA]
    data: dict[str, Any] = {
        "mode": mode,
        "converged": overall_converged
        if overall_converged is not None
        else judgment_data["converged"],
        "level_reached": level_reached,
        "summary": judgment_data["summary"],
        "judgment": judgment_data,
        "deltas": delta_dicts,
        "ssim_score": ssim_score,
        "ssim_threshold": SSIM_THRESHOLD,
        "hot_regions": [json.loads(r.model_dump_json()) for r in hot_regions],
        "significant_hot_region_count": len(significant_regions),
        "regions": [json.loads(r.model_dump_json()) for r in regions],
        "visual_error": visual_error,
        "spec_node_id": spec.figma_node_id if spec is not None else None,
        "dom_route": dom.route,
        "dom_element_count": len(dom.elements),
        "iteration": iteration,
        "session_id": session_id,
        "max_iterations": max_iterations,
        "is_stuck": is_stuck,
        "is_regression": is_regression,
        "dinov2_enabled": dinov2_enabled,
        "dinov2_threshold": dinov2_threshold,
        "dinov2_similarities": dinov2_similarities,
        "vlm_enabled": vlm_enabled,
        "vlm_threshold": vlm_threshold,
        "vlm_backend": vlm_backend,
        "vlm_judgments": vlm_judgments,
        "human_gate_enabled": human_gate_enabled,
        "human_verdict": human_verdict,
        "human_notes": human_notes,
    }

    hints: list[str] = _build_hints(judgment_data, deltas, truncated, mode=mode)
    if iteration is not None and max_iterations is not None:
        hints.append(f"Iteration {iteration} of {max_iterations}.")
    if is_stuck:
        hints.append(
            "STUCK: last 3 Iterations produced identical structured-delta hashes — "
            "the Agent isn't making progress. Either fix the listed Deltas or run "
            "`pixel-mcp reset` to clear state."
        )
    if is_regression:
        hints.append(
            "REGRESSION: a previously-passed Level is now failing. A recent edit "
            "broke something that used to work."
        )
    if ssim_score is not None and ssim_score < SSIM_THRESHOLD:
        hints.append(
            f"SSIM Score {ssim_score:.3f} below threshold {SSIM_THRESHOLD} — "
            "global structural drift detected (likely a layout-level mismatch)."
        )
    if significant_regions:
        hints.append(
            f"{len(significant_regions)} Hot Region(s) above {MIN_BBOX_AREA}px² — "
            "see data.hot_regions for bboxes."
        )
    if visual_error is not None:
        if mode == "image":
            hints.append(
                f"Visual signal failed in image-only mode ({visual_error[:120]}). "
                "Image-only mode has no fallback — convergence cannot be reached "
                "until the visual diff succeeds."
            )
        else:
            hints.append(
                f"Visual signal unavailable ({visual_error[:120]}). "
                "Level 0 Gate Pass fell back to structured Deltas only."
            )
    if dinov2_hint is not None:
        hints.append(dinov2_hint)
    if dinov2_enabled and dinov2_similarities:
        failing = [s for s in dinov2_similarities if s["similarity"] < (dinov2_threshold or 0)]
        if failing:
            hints.append(
                f"Level 1 (DINOv2) failed on {len(failing)} crop(s) — see data.dinov2_similarities."
            )
        elif level_reached >= 1:
            hints.append(
                f"Level 1 (DINOv2) Gate Pass: {len(dinov2_similarities)} crop(s) "
                f"above similarity threshold {dinov2_threshold}."
            )
    if vlm_hint is not None:
        hints.append(vlm_hint)
    if vlm_enabled and vlm_judgments:
        failing_vlm = [
            v
            for v in vlm_judgments
            if v["verdict"] != "match" or v["confidence"] < (vlm_threshold or 0)
        ]
        if failing_vlm:
            hints.append(
                f"Level 2 (VLM) failed on {len(failing_vlm)} crop(s) — "
                "see data.vlm_judgments for verdicts and reasoning."
            )
        elif level_reached >= 2:
            hints.append(
                f"Level 2 (VLM) Gate Pass: {len(vlm_judgments)} crop(s) "
                f"judged 'match' above confidence threshold {vlm_threshold}."
            )

    converged_now = (
        overall_converged if overall_converged is not None else judgment_data["converged"]
    )
    # Human-gate hints surface near the bottom so the reviewer notices the
    # ask after the per-level summaries (which are still useful context).
    if human_gate_enabled and human_gate_pending:
        hints.append(
            "Level 3 (human review) is pending — call `mcp__pixel_mcp__review` to see "
            "the crop pairs inline, then `mcp__pixel_mcp__human_feedback`."
        )
    elif human_gate_enabled and human_verdict == "approved":
        hints.append("Level 3 (human review) APPROVED — Final Convergence reached.")
    elif human_gate_enabled and human_verdict == "rejected":
        hints.append(
            f"Level 3 (human review) REJECTED — notes: {human_notes!s}. "
            "Loop re-opened with a synthesized human_review Delta."
        )

    if human_gate_enabled and human_gate_pending:
        next_action = (
            "Call `mcp__pixel_mcp__review` to inspect the expected/actual crops inline, "
            "then `mcp__pixel_mcp__human_feedback` to record the verdict."
        )
    elif not converged_now:
        next_action = (
            "Have the Agent fix the listed Deltas, then re-invoke `mcp__pixel_mcp__check`."
        )
    elif level_reached >= 3:
        next_action = "Final Convergence at Level 3 (human review). Nothing left to do."
    elif level_reached >= 2 and human_gate_enabled:
        next_action = (
            "Level 2 Gate Pass. Awaiting Level 3 human verdict — call " "`mcp__pixel_mcp__review`."
        )
    elif level_reached >= 2:
        next_action = (
            "Level 2 Gate Pass. Promote to Level 3 by re-running with "
            "`--enable-human-gate` and using `mcp__pixel_mcp__review`."
        )
    elif level_reached >= 1:
        next_action = (
            "Level 1 Gate Pass. Promote to Level 2 (VLM) by re-running "
            "`pixel-mcp check --enable-vlm`."
        )
    else:
        next_action = (
            "Promote to Level 1 (DINOv2 per-crop similarity) by re-running "
            "`pixel-mcp check --enable-dinov2`."
        )

    return make_envelope(
        data=data,
        hints=hints,
        diagnostics={
            "mode": mode,
            "delta_count": len(deltas),
            "critical_count": judgment_data["critical_count"],
            "major_count": judgment_data["major_count"],
            "minor_count": judgment_data["minor_count"],
            "regression_count": judgment_data["regression_count"],
            "dom_truncated": truncated,
        },
        next_suggested_action=next_action,
        affordances=[
            {
                "tool": "mcp__pixel_mcp__diff",
                "when": "to inspect Deltas in isolation",
            },
            {
                "tool": "mcp__pixel_mcp__judge",
                "when": "to apply a custom Tolerance to existing Deltas",
            },
            {
                "tool": "mcp__pixel_mcp__measure",
                "when": "to inspect specific DOM elements (--selectors)",
            },
        ],
    )


def _build_hints(
    judgment_data: dict[str, Any],
    deltas: list[Delta],
    truncated: bool,
    *,
    mode: str = "figma",
) -> list[str]:
    hints: list[str] = [judgment_data["summary"]]
    if not judgment_data["converged"]:
        # Per-property guidance — only meaningful in Figma mode where
        # `property` is a real CSS-style key. In image-only mode every
        # pseudo-Delta has a unique `hot_region_<n>` so the histogram is
        # noise; skip it.
        if mode == "figma":
            property_counts: dict[str, int] = {}
            for d in deltas:
                if d.severity in ("critical", "major"):
                    property_counts[d.property] = property_counts.get(d.property, 0) + 1
            if property_counts:
                top = sorted(property_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
                top_props = ", ".join(f"{p} (×{n})" for p, n in top)
                hints.append(f"Top properties with critical/major Deltas: {top_props}.")
            # Common-cause nudges
            if any(d.property == "color" and d.severity == "critical" for d in deltas):
                hints.append("Color mismatch is typically a missing or wrong design-token import.")
            if any(d.property == "font_family" and d.severity == "critical" for d in deltas):
                hints.append(
                    "Font-family mismatch — confirm the font is loaded (link or @font-face) "
                    "on the Render side."
                )
    if truncated:
        hints.append(
            "Auto-discover hit the 200-element cap — narrow with --selectors for sharper Deltas."
        )
    return hints


def _fatal_envelope(error_type: str, error_message: str, hints: list[str]) -> Envelope:
    return make_envelope(
        data=None,
        hints=hints,
        diagnostics={"error_type": error_type, "error_message": error_message},
        next_suggested_action="Resolve the error above, then re-run `pixel-mcp check`.",
        affordances=[
            {"tool": "mcp__pixel_mcp__doctor", "when": "to diagnose environment issues"},
        ],
    )
