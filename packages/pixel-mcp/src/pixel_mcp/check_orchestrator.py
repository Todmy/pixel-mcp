"""Multi-axis convergence orchestrator for ``pixel-mcp check``.

Owns the per-(browser, viewport) pipeline (:func:`_run_one_pass`) and the
cross-product loop / aggregation / loop-economics that drives the matrix
(:func:`_run_multi_viewport`). The single-axis ``run()`` entry point lives
in ``check_cmd.py`` and short-circuits to this module when the caller sets
``viewports`` or ``browsers``.

External dependencies (``measure_render``, ``extract_spec``, ``read_state``,
``write_state``, ``append_history``, ``_compute_visual_signals``,
``_run_dinov2_gate``, ``_run_vlm_gate``) are looked up via the ``check_cmd``
module at call time so that tests using ``patch("pixel_mcp.check_cmd.<name>")``
keep working — ``check_cmd`` is the canonical patch surface for those names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pixel_tools_shared import Envelope

from pixel_mcp import check_cmd as _cc
from pixel_mcp.check_envelope import (
    MIN_BBOX_AREA,
    SSIM_THRESHOLD,
    _fatal_envelope,
    _hot_region_to_delta,
    _stamp_browser,
    _stamp_viewport,
    _success_envelope_multi,
    _viewport_str,
)
from pixel_mcp.delta import Delta, diff_design_vs_render
from pixel_mcp.figma_client import FigmaApiError, FigmaAuthError, FigmaError, FigmaNotFoundError
from pixel_mcp.figma_url import FigmaUrlError
from pixel_mcp.human_feedback_cmd import mark_consumed, read_feedback
from pixel_mcp.judge import Tolerance, judge_deltas
from pixel_mcp.loop_state import (
    HASH_TRAIL_MAX,
    detect_regression,
    detect_stuck,
    hash_deltas_bucketed,
    now_utc,
)
from pixel_mcp.normalize import normalize_spec_for_viewport
from pixel_mcp.perf_metrics import PerfBudget, PerfMetrics
from pixel_mcp.render import (
    VALID_BROWSERS,
    BrowserNotInstalledError,
    ChromiumNotInstalledError,
    PlaywrightNotInstalledError,
    RenderError,
    RouteUnreachableError,
    WaitForTimeoutError,
)
from pixel_mcp.spec import DesignSpec, UnsupportedNodeTypeError

# Exit-code constants (mirror check_cmd to avoid an import cycle on bare names).
EXIT_CONVERGED = 0
EXIT_DELTAS = 1
EXIT_READY_FOR_LEVEL_3 = 2  # reserved
EXIT_REGRESSION = 3
EXIT_MAX_ITERATIONS = 10
EXIT_STUCK = 11
EXIT_FATAL = 12


def _run_one_pass(
    *,
    spec: DesignSpec | None,
    image_only: bool,
    image_bytes: bytes | None,
    figma_url: str | None,
    route: str,
    viewport: tuple[int, int],
    browser: str,
    selectors: list[str] | None,
    wait_for: str | None,
    iteration: int,
    treat_minor_as_blocking: bool,
    enable_dinov2: bool,
    dinov2_threshold: float,
    enable_vlm: bool,
    vlm_threshold: float,
    vlm_backend: str,
    enable_omniparser: bool,
    omniparser_confidence_threshold: float,
    namespace_browser: bool = False,
    enable_perf: bool = False,
    perf_budget: PerfBudget | None = None,
) -> tuple[dict[str, Any] | None, tuple[Envelope, int] | None]:
    """Run the per-(browser, viewport) pipeline (measure → diff → judge → visual → gates).

    Returns ``(result, None)`` on success, ``(None, (envelope, exit_code))``
    when the pass hit a fatal error so the orchestrator can short-circuit
    the whole check (matches single-pass semantics — any measure failure
    aborts the run rather than silently skipping that cell of the matrix).

    ``namespace_browser=True`` nests crops under ``browser-<name>/`` to
    isolate the cross-browser matrix on disk. Defaults to False so callers
    that only set ``--viewports`` keep the v2-1 layout
    (``.pixel-mcp/crops/iter-N/viewport-WxH/``) — backward compatibility for
    the existing v2-1 acceptance tests.
    """
    vp_str = _viewport_str(viewport)
    vp_subfolder = f"viewport-{vp_str}"
    browser_subfolder = f"browser-{browser}" if namespace_browser else None

    # --- Measure ---
    try:
        dom, truncated = _cc.measure_render(
            route=route,
            viewport=viewport,
            selectors=selectors,
            wait_for=wait_for,
            browser=browser,  # type: ignore[arg-type]
        )
    except PlaywrightNotInstalledError as exc:
        return None, (
            _fatal_envelope(
                "playwright_not_installed",
                str(exc),
                ["Run `uv sync` then `uv run playwright install chromium`."],
            ),
            EXIT_FATAL,
        )
    except ChromiumNotInstalledError as exc:
        return None, (
            _fatal_envelope(
                "chromium_not_installed",
                str(exc),
                ["Run `uv run playwright install chromium` (one-time, ~150MB)."],
            ),
            EXIT_FATAL,
        )
    except BrowserNotInstalledError as exc:
        return None, (
            _fatal_envelope(
                "browser_not_installed",
                str(exc),
                [
                    "Run `uv run playwright install firefox webkit` (one-time)."
                    f" Missing engine: {exc.browser}.",
                ],
            ),
            EXIT_FATAL,
        )
    except WaitForTimeoutError as exc:
        return None, (
            _fatal_envelope(
                "wait_for_timeout",
                str(exc),
                ["Re-check the --wait-for selector — the element never appeared."],
            ),
            EXIT_FATAL,
        )
    except RouteUnreachableError as exc:
        return None, (
            _fatal_envelope(
                "route_unreachable",
                str(exc),
                ["Verify the dev server is running and the route URL is correct."],
            ),
            EXIT_FATAL,
        )
    except RenderError as exc:
        return None, (
            _fatal_envelope(
                "render_error",
                str(exc),
                ["See `pixel-mcp doctor` for environment diagnostics."],
            ),
            EXIT_FATAL,
        )

    # --- Structured Deltas / Judgment ---
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
        judgment = judge_deltas([])

    # --- Visual signals (SSIM + Hot Regions + decomposition) ---
    ssim_score, hot_regions, regions, visual_error = _cc._compute_visual_signals(
        figma_url=figma_url,
        image_bytes=image_bytes,
        route=route,
        viewport=viewport,
        wait_for=wait_for,
        dom=dom,
        iteration=iteration,
        viewport_subfolder=vp_subfolder,
        browser_subfolder=browser_subfolder,
        browser=browser,
    )

    # --- OmniParser augmentation ---
    omniparser_detections: list[Any] | None = None
    omniparser_hint: str | None = None
    if enable_omniparser:
        omniparser_detections, omniparser_hint = _cc._run_omniparser_augmentation(
            regions=regions,
            iteration=iteration,
            confidence_threshold=omniparser_confidence_threshold,
            viewport_subfolder=vp_subfolder,
            browser_subfolder=browser_subfolder,
        )

    # --- Image-only pseudo-Delta synthesis from Hot Regions ---
    if image_only:
        significant = [r for r in hot_regions if r.w * r.h >= MIN_BBOX_AREA]
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

    # --- Level 0 gate ---
    if visual_error is not None and not image_only:
        visual_passes = True
    else:
        if image_only and visual_error is not None:
            visual_passes = False
        else:
            ssim_ok = ssim_score is None or ssim_score >= SSIM_THRESHOLD
            regions_ok = not any(r.w * r.h >= MIN_BBOX_AREA for r in hot_regions)
            visual_passes = ssim_ok and regions_ok

    if image_only:
        viewport_converged = visual_passes
    else:
        viewport_converged = judgment.converged and visual_passes

    # --- Level 1 (DINOv2) gate ---
    level_reached = 0
    dinov2_similarities: list[dict[str, Any]] | None = None
    dinov2_hint: str | None = None
    if enable_dinov2 and viewport_converged:
        dinov2_result = _cc._run_dinov2_gate(
            regions=regions,
            dinov2_threshold=dinov2_threshold,
        )
        dinov2_similarities = dinov2_result["similarities"]
        dinov2_hint = dinov2_result["hint"]
        failing = dinov2_result["failing_deltas"]
        if dinov2_result["promoted"]:
            level_reached = 1
        if failing:
            viewport_converged = False
            level_reached = 0
            deltas = list(deltas) + failing
            judgment = judge_deltas(
                deltas,
                tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
            )

    # --- Level 2 (VLM) gate ---
    vlm_judgments: list[dict[str, Any]] | None = None
    vlm_hint: str | None = None
    if enable_vlm and viewport_converged and level_reached >= 1:
        vlm_result = _cc._run_vlm_gate(
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
            viewport_converged = False
            level_reached = 1
            deltas = list(deltas) + failing_vlm
            judgment = judge_deltas(
                deltas,
                tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
            )

    # --- v3-1 Perf gate (per-cell collection) ---
    perf_metrics_result: PerfMetrics | None = None
    perf_hint: str | None = None
    if enable_perf:
        perf_result = _cc._run_perf_gate(
            route=route,
            viewport=viewport,
            browser=browser,
            wait_for=wait_for,
            perf_budget=perf_budget,
        )
        perf_metrics_result = perf_result["metrics"]
        perf_hint = perf_result["hint"]
        perf_failing = perf_result["failing_deltas"]
        if perf_failing:
            deltas = list(deltas) + perf_failing
            judgment = judge_deltas(
                deltas,
                tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
            )
            if not judgment.converged:
                viewport_converged = False

    # Stamp the viewport + browser onto every Delta + Region produced by this pass.
    deltas = _stamp_viewport(deltas, vp_str)
    deltas = _stamp_browser(deltas, browser)
    for r in regions:
        if getattr(r, "viewport", None) is None:
            try:
                r.viewport = vp_str
            except (AttributeError, TypeError):
                pass
        if getattr(r, "browser", None) is None:
            try:
                r.browser = browser
            except (AttributeError, TypeError):
                pass

    return (
        {
            "viewport": vp_str,
            "browser": browser,
            "dom": dom,
            "truncated": truncated,
            "deltas": deltas,
            "judgment": judgment,
            "ssim_score": ssim_score,
            "hot_regions": hot_regions,
            "regions": regions,
            "visual_error": visual_error,
            "viewport_converged": viewport_converged,
            "level_reached": level_reached,
            "dinov2_similarities": dinov2_similarities,
            "dinov2_hint": dinov2_hint,
            "vlm_judgments": vlm_judgments,
            "vlm_hint": vlm_hint,
            "omniparser_detections": omniparser_detections,
            "omniparser_hint": omniparser_hint,
            "perf_metrics": perf_metrics_result,
            "perf_hint": perf_hint,
        },
        None,
    )


def _run_multi_viewport(
    *,
    figma_url: str | None,
    route: str,
    viewports: list[tuple[int, int]],
    browsers: list[str],
    selectors: list[str] | None,
    wait_for: str | None,
    refresh_spec: bool,
    treat_minor_as_blocking: bool,
    max_iterations: int,
    stuck_threshold: int,
    image_path: str | Path | None,
    enable_dinov2: bool,
    dinov2_threshold: float,
    enable_vlm: bool,
    vlm_threshold: float,
    vlm_backend: str,
    enable_human_gate: bool,
    enable_omniparser: bool,
    omniparser_confidence_threshold: float,
    browsers_specified: bool = False,
    viewports_specified: bool = True,
    enable_perf: bool = False,
    perf_budget: PerfBudget | None = None,
) -> tuple[Envelope, int]:
    """Multi-axis convergence orchestrator (v2-1 viewports × v2-2 browsers).

    Runs the per-(browser, viewport) pipeline once per cell of the matrix and
    aggregates the results. Overall convergence is the AND-fold; the
    reported ``level_reached`` is the MIN across cells (worst-case wins).
    Each Delta + Region produced by a per-pass carries both the viewport and
    browser identifier so the Agent can locate which (browser × breakpoint)
    cell regressed.

    Session-level concerns (state, history, human gate) run once across the
    aggregated result — never per pass.

    Backward compatibility: ``viewports_specified`` / ``browsers_specified``
    control which axes the envelope surfaces; when only ``viewports`` was set
    by the caller, the envelope keeps the v2-1 shape (``viewport_results``
    only). When only ``browsers`` was set, ``measurement_results`` is the
    canonical aggregation key.
    """
    # --- 0a) Design Source validation ---
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
    if not viewports:
        return _fatal_envelope(
            "viewports_empty",
            "viewports list must contain at least one (W, H) entry.",
            ["Pass --viewports '1280x720,375x667' or --viewports-preset responsive."],
        ), EXIT_FATAL
    if not browsers:
        return _fatal_envelope(
            "browsers_empty",
            "browsers list must contain at least one engine.",
            ["Pass --browsers 'chromium,firefox' or --browsers-preset all."],
        ), EXIT_FATAL
    invalid_browsers = [b for b in browsers if b not in VALID_BROWSERS]
    if invalid_browsers:
        return _fatal_envelope(
            "browsers_invalid",
            f"Unsupported browser(s): {invalid_browsers}. Choose from {sorted(VALID_BROWSERS)}.",
            ["Pass --browsers chromium,firefox,webkit or --browsers-preset all."],
        ), EXIT_FATAL

    image_only = image_path is not None
    mode = "image" if image_only else "figma"

    # --- 0b) Iteration counter + max-iter ---
    state = _cc.read_state()
    state.iteration += 1
    state.last_invocation_at = now_utc()
    if state.iteration > max_iterations:
        _cc.write_state(state)
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

    # --- 1) Extract DesignSpec once (Figma mode only) ---
    spec: DesignSpec | None = None
    if not image_only:
        try:
            spec = _cc.extract_spec(figma_url, refresh=refresh_spec)  # type: ignore[arg-type]
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

    # --- 2) Iterate cross-product (browsers × viewports) ---
    pass_results: list[dict[str, Any]] = []
    for browser_name in browsers:
        for vp in viewports:
            result, fatal = _run_one_pass(
                spec=spec,
                image_only=image_only,
                image_bytes=image_bytes,
                figma_url=figma_url,
                route=route,
                viewport=vp,
                browser=browser_name,
                selectors=selectors,
                wait_for=wait_for,
                iteration=state.iteration,
                treat_minor_as_blocking=treat_minor_as_blocking,
                enable_dinov2=enable_dinov2,
                dinov2_threshold=dinov2_threshold,
                enable_vlm=enable_vlm,
                vlm_threshold=vlm_threshold,
                vlm_backend=vlm_backend,
                enable_omniparser=enable_omniparser,
                omniparser_confidence_threshold=omniparser_confidence_threshold,
                namespace_browser=browsers_specified,
                enable_perf=enable_perf,
                perf_budget=perf_budget,
            )
            if fatal is not None:
                # Bubble up the first fatal (matches single-pass semantics).
                # State was already incremented; persist so the iteration counter
                # stays honest across the failed run.
                _cc.write_state(state)
                return fatal
            assert result is not None
            pass_results.append(result)

    # --- 3) Aggregate ---
    all_converged = all(r["viewport_converged"] for r in pass_results)
    aggregated_level = min(r["level_reached"] for r in pass_results)
    combined_deltas: list[Delta] = []
    for r in pass_results:
        combined_deltas.extend(r["deltas"])
    combined_judgment = judge_deltas(
        combined_deltas,
        tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
    )

    overall_converged = all_converged
    level_reached = aggregated_level if overall_converged else 0

    # --- 4e) Level 3 human gate (session-level, runs once on aggregate) ---
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
                combined_deltas = list(combined_deltas) + [
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
                combined_judgment = judge_deltas(
                    combined_deltas,
                    tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
                )
                overall_converged = False
            else:
                human_gate_pending = True
                human_verdict = "pending"

    # --- 5) Loop economics + history ---
    current_hash = hash_deltas_bucketed(combined_deltas)
    is_stuck = detect_stuck(state.recent_hashes, current_hash, threshold=stuck_threshold)
    current_level_passed = level_reached if overall_converged else 0
    is_regression = detect_regression(state, current_level_passed)

    state.last_delta_hash = current_hash
    state.recent_hashes = (state.recent_hashes + [current_hash])[-HASH_TRAIL_MAX:]
    if overall_converged and current_level_passed > state.highest_level_reached:
        state.highest_level_reached = current_level_passed
    _cc.write_state(state)

    _cc.append_history(
        {
            "iteration": state.iteration,
            "session_id": state.session_id,
            "mode": mode,
            "delta_count": len(combined_deltas),
            "delta_hash": current_hash,
            "viewports": sorted({r["viewport"] for r in pass_results}),
            "browsers": sorted({r["browser"] for r in pass_results}),
            "converged": overall_converged,
            "timestamp": now_utc().isoformat(),
        }
    )

    # --- 6) Build envelope ---
    envelope = _success_envelope_multi(
        spec=spec,
        pass_results=pass_results,
        viewports=viewports,
        browsers=browsers,
        viewports_specified=viewports_specified,
        browsers_specified=browsers_specified,
        combined_deltas=combined_deltas,
        combined_judgment_data=json.loads(combined_judgment.model_dump_json()),
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
        vlm_enabled=enable_vlm,
        vlm_threshold=vlm_threshold if enable_vlm else None,
        vlm_backend=vlm_backend if enable_vlm else None,
        human_gate_enabled=enable_human_gate,
        human_verdict=human_verdict,
        human_notes=human_notes,
        human_gate_pending=human_gate_pending,
        omniparser_enabled=enable_omniparser,
        treat_minor_as_blocking=treat_minor_as_blocking,
        perf_enabled=enable_perf,
        perf_budget=perf_budget,
    )

    if is_regression:
        return envelope, EXIT_REGRESSION
    if is_stuck:
        return envelope, EXIT_STUCK
    if human_gate_pending:
        return envelope, EXIT_READY_FOR_LEVEL_3
    return envelope, (EXIT_CONVERGED if overall_converged else EXIT_DELTAS)


def _run_single_axis(
    *,
    figma_url: str | None,
    route: str,
    viewport: tuple[int, int],
    selectors: list[str] | None,
    wait_for: str | None,
    refresh_spec: bool,
    treat_minor_as_blocking: bool,
    max_iterations: int,
    stuck_threshold: int,
    image_path: str | Path | None,
    enable_dinov2: bool,
    dinov2_threshold: float,
    enable_vlm: bool,
    vlm_threshold: float,
    vlm_backend: str,
    enable_human_gate: bool,
    enable_omniparser: bool,
    omniparser_confidence_threshold: float,
    enable_perf: bool,
    perf_budget: PerfBudget | None,
) -> tuple[Envelope, int]:
    """Run the v0/v1 single-viewport convergence pipeline.

    This is the original ``check_cmd.run()`` body; it stays a separate code
    path from :func:`_run_multi_viewport` so the v0/v1 envelope shape, on-disk
    crop layout (``.pixel-mcp/crops/iter-N/`` with no ``viewport-WxH``
    nesting), and exit-code semantics remain bit-for-bit identical to the
    pre-refactor implementation. Tests that ``patch("pixel_mcp.check_cmd.X")``
    still bite here because every dependency is looked up via the
    ``check_cmd`` module at call time (``_cc.X``).
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
    state = _cc.read_state()
    state.iteration += 1
    state.last_invocation_at = now_utc()
    if state.iteration > max_iterations:
        _cc.write_state(state)
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
            spec = _cc.extract_spec(figma_url, refresh=refresh_spec)  # type: ignore[arg-type]
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
        dom, truncated = _cc.measure_render(
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
    ssim_score, hot_regions, regions, visual_error = _cc._compute_visual_signals(
        figma_url=figma_url,
        image_bytes=image_bytes,
        route=route,
        viewport=viewport,
        wait_for=wait_for,
        dom=dom,
        iteration=state.iteration,
    )

    # --- 4a) OmniParser augmentation (v1.5-2) ---
    # Opt-in semantic-label attribution on top of bare bboxes. Runs against
    # the full actual screenshot persisted by ``_compute_visual_signals``.
    # Mutates Regions in place — each Region gets ``semantic_label`` and
    # ``semantic_confidence`` when a detection covers its centre. The
    # detections themselves are also surfaced on the envelope so the Agent
    # can correlate failures with the semantic map.
    omniparser_detections: list[Any] | None = None
    omniparser_hint: str | None = None
    if enable_omniparser:
        omniparser_detections, omniparser_hint = _cc._run_omniparser_augmentation(
            regions=regions,
            iteration=state.iteration,
            confidence_threshold=omniparser_confidence_threshold,
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
        dinov2_result = _cc._run_dinov2_gate(
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
        vlm_result = _cc._run_vlm_gate(
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

    # --- 4f) v3-1 Performance Budgets gate, opt-in ---
    # Independent of visual convergence — collects Core Web Vitals for the
    # same (route, viewport, browser) cell. When a budget is supplied any
    # field exceeding it (>5% over) synthesises a perf Delta with
    # property=f"perf_{field}". These join the existing delta stream so loop
    # economics (hash bucket, stuck detection) and the Judge severity counts
    # both see them — a critical/major perf miss blocks convergence the same
    # way a critical visual Delta does.
    perf_metrics_result: PerfMetrics | None = None
    perf_hint: str | None = None
    if enable_perf:
        perf_result = _cc._run_perf_gate(
            route=route,
            viewport=viewport,
            browser="chromium",
            wait_for=wait_for,
            perf_budget=perf_budget,
        )
        perf_metrics_result = perf_result["metrics"]
        perf_hint = perf_result["hint"]
        perf_failing = perf_result["failing_deltas"]
        if perf_failing:
            deltas = list(deltas) + perf_failing
            judgment = judge_deltas(
                deltas,
                tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking),
            )
            # A critical/major perf miss reverts overall_converged via the Judge;
            # minor perf Deltas only block when --strict is set, same rules as
            # visual minor Deltas.
            if not judgment.converged:
                overall_converged = False

    # --- 5) Loop economics: stuck detection + regression + state update ---
    current_hash = hash_deltas_bucketed(deltas)
    is_stuck = detect_stuck(state.recent_hashes, current_hash, threshold=stuck_threshold)
    current_level_passed = level_reached if overall_converged else 0
    is_regression = detect_regression(state, current_level_passed)

    state.last_delta_hash = current_hash
    state.recent_hashes = (state.recent_hashes + [current_hash])[-HASH_TRAIL_MAX:]
    if overall_converged and current_level_passed > state.highest_level_reached:
        state.highest_level_reached = current_level_passed
    _cc.write_state(state)

    _cc.append_history(
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

    envelope = _cc._success_envelope(
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
        omniparser_enabled=enable_omniparser,
        omniparser_detections=omniparser_detections,
        omniparser_hint=omniparser_hint,
        perf_enabled=enable_perf,
        perf_budget=perf_budget,
        perf_metrics=[perf_metrics_result] if perf_metrics_result is not None else [],
        perf_hint=perf_hint,
    )

    if is_regression:
        return envelope, EXIT_REGRESSION
    if is_stuck:
        return envelope, EXIT_STUCK
    if human_gate_pending:
        return envelope, EXIT_READY_FOR_LEVEL_3
    return envelope, (EXIT_CONVERGED if overall_converged else EXIT_DELTAS)
