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

Module layout (refactored)
--------------------------

This module is now the public surface only: the ``run()`` entry point, the
``EXIT_*`` constants, plus re-exports of the helpers tests patch via
``patch("pixel_mcp.check_cmd.<name>")``. The actual implementation lives in
sibling modules:

- ``check_envelope`` — :func:`_success_envelope`, :func:`_success_envelope_multi`,
  :func:`_fatal_envelope`, hint construction, Delta-stamping helpers,
  ``SSIM_THRESHOLD`` / ``MIN_BBOX_AREA`` constants.
- ``check_gates`` — :func:`_compute_visual_signals`, :func:`_run_dinov2_gate`,
  :func:`_run_vlm_gate`, :func:`_run_omniparser_augmentation`,
  :func:`_run_perf_gate`, plus the severity helpers.
- ``check_orchestrator`` — :func:`_run_one_pass`, :func:`_run_multi_viewport`.

The external dependencies tests like to patch (``measure_render``,
``capture_screenshot``, ``extract_spec``, ``read_state``, ``write_state``,
``append_history``, ``state_dir``, ``collect_perf_metrics``) live at this
module's top level. The sibling modules look them up via
``check_cmd.<name>`` at call time so a ``patch("pixel_mcp.check_cmd.X")``
in tests propagates through the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

from pixel_tools_shared import Envelope

# Re-exports — many of these are not referenced directly by ``run()``, but
# they MUST live at this module's top level so that:
#   1. ``patch("pixel_mcp.check_cmd.<name>")`` in tests intercepts the call
#      (the sibling modules look them up via ``_cc.<name>`` at call time).
#   2. ``from pixel_mcp.check_cmd import <name>`` from any external caller
#      keeps working post-refactor.
from pixel_mcp.check_envelope import (
    _BROWSER_PRESETS,
    MIN_BBOX_AREA,
    SSIM_THRESHOLD,
    _build_hints,
    _fatal_envelope,
    _hot_region_to_delta,
    _severity_for_area,
    _stamp_browser,
    _stamp_viewport,
    _success_envelope,
    _success_envelope_multi,
    _viewport_str,
)
from pixel_mcp.loop_state import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_STUCK_THRESHOLD,
    append_history,
    read_state,
    write_state,
)
from pixel_mcp.perf_metrics import (
    PerfBudget,
    collect_perf_metrics,
)
from pixel_mcp.render import (
    capture_screenshot,
    measure_render,
)
from pixel_mcp.spec import extract_spec
from pixel_mcp.state import state_dir

EXIT_CONVERGED = 0
EXIT_DELTAS = 1
EXIT_READY_FOR_LEVEL_3 = 2  # reserved
EXIT_REGRESSION = 3
EXIT_MAX_ITERATIONS = 10
EXIT_STUCK = 11
EXIT_FATAL = 12

# Re-export gate runners so ``patch("pixel_mcp.check_cmd._run_dinov2_gate")``
# (and friends) continues to intercept the call. The sibling modules call
# back through this module's attribute table, so a patch here propagates.
from pixel_mcp.check_gates import (  # noqa: E402
    _bbox_iou,
    _compute_visual_signals,
    _dinov2_severity_for_gap,
    _match_detected_to_region,
    _run_dinov2_gate,
    _run_omniparser_augmentation,
    _run_perf_gate,
    _run_vlm_gate,
    _vlm_severity_for_gap,
)

# Imported last so ``check_cmd`` is already populated by the time the
# orchestrator pulls it in via ``from pixel_mcp import check_cmd``.
from pixel_mcp.check_orchestrator import (  # noqa: E402
    _run_multi_viewport,
    _run_one_pass,
    _run_single_axis,
)

__all__ = [
    "EXIT_CONVERGED",
    "EXIT_DELTAS",
    "EXIT_FATAL",
    "EXIT_MAX_ITERATIONS",
    "EXIT_READY_FOR_LEVEL_3",
    "EXIT_REGRESSION",
    "EXIT_STUCK",
    "MIN_BBOX_AREA",
    "SSIM_THRESHOLD",
    "_BROWSER_PRESETS",
    "_bbox_iou",
    "_build_hints",
    "_compute_visual_signals",
    "_dinov2_severity_for_gap",
    "_fatal_envelope",
    "_hot_region_to_delta",
    "_match_detected_to_region",
    "_run_dinov2_gate",
    "_run_multi_viewport",
    "_run_omniparser_augmentation",
    "_run_one_pass",
    "_run_perf_gate",
    "_run_single_axis",
    "_run_vlm_gate",
    "_severity_for_area",
    "_stamp_browser",
    "_stamp_viewport",
    "_success_envelope",
    "_success_envelope_multi",
    "_viewport_str",
    "_vlm_severity_for_gap",
    "append_history",
    "capture_screenshot",
    "collect_perf_metrics",
    "extract_spec",
    "measure_render",
    "read_state",
    "run",
    "state_dir",
    "write_state",
]


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
    enable_omniparser: bool = False,
    omniparser_confidence_threshold: float = 0.3,
    viewports: list[tuple[int, int]] | None = None,
    browsers: list[str] | None = None,
    enable_perf: bool = False,
    perf_budget: PerfBudget | None = None,
) -> tuple[Envelope, int]:
    """Run one Iteration of the Convergence Loop.

    Exactly one of ``figma_url`` or ``image_path`` must be provided. Both or
    neither returns an EXIT_FATAL envelope.

    When ``viewports`` is set (v2-1), the full convergence pipeline (extract
    spec → measure → diff → judge → visual signals → gates) runs at EACH
    viewport in the list. The overall verdict is the AND-fold across
    viewports; ``level_reached`` is the MIN (worst-case wins); each Delta
    carries a ``viewport`` field so the Agent can locate the failing
    breakpoint. The default (``viewports=None``) preserves the v0/v1 single-
    viewport behaviour exactly — the ``viewport`` parameter still applies.

    Never raises — all errors are folded into the AXI envelope. Loop
    economics (iteration counter, stuck detection, regression, max-iter)
    are enforced via ``.pixel-mcp/state.json``.
    """
    # --- v2-1/v2-2) Multi-axis short-circuit ---
    # When ``viewports`` OR ``browsers`` is set, delegate to the multi-axis
    # orchestrator. The single-axis contract below is preserved exactly when
    # both are ``None`` (the defaults) — all v0/v1 tests continue to exercise
    # the original path. Missing axes degrade to a single-element list
    # (current viewport / chromium) inside the orchestrator so the same code
    # path handles 1×N, N×1, and N×M matrices uniformly.
    if viewports is not None or browsers is not None:
        return _run_multi_viewport(
            figma_url=figma_url,
            route=route,
            viewports=viewports if viewports is not None else [viewport],
            browsers=browsers if browsers is not None else ["chromium"],
            selectors=selectors,
            wait_for=wait_for,
            refresh_spec=refresh_spec,
            treat_minor_as_blocking=treat_minor_as_blocking,
            max_iterations=max_iterations,
            stuck_threshold=stuck_threshold,
            image_path=image_path,
            enable_dinov2=enable_dinov2,
            dinov2_threshold=dinov2_threshold,
            enable_vlm=enable_vlm,
            vlm_threshold=vlm_threshold,
            vlm_backend=vlm_backend,
            enable_human_gate=enable_human_gate,
            enable_omniparser=enable_omniparser,
            omniparser_confidence_threshold=omniparser_confidence_threshold,
            browsers_specified=browsers is not None,
            viewports_specified=viewports is not None,
            enable_perf=enable_perf,
            perf_budget=perf_budget,
        )

    # --- Single-axis path: delegate to the orchestrator's v0/v1 pipeline. ---
    # Same dispatch boundary as the multi-axis branch above — keeps ``run()``
    # a thin façade and lets the orchestrator own all of the actual work.
    return _run_single_axis(
        figma_url=figma_url,
        route=route,
        viewport=viewport,
        selectors=selectors,
        wait_for=wait_for,
        refresh_spec=refresh_spec,
        treat_minor_as_blocking=treat_minor_as_blocking,
        max_iterations=max_iterations,
        stuck_threshold=stuck_threshold,
        image_path=image_path,
        enable_dinov2=enable_dinov2,
        dinov2_threshold=dinov2_threshold,
        enable_vlm=enable_vlm,
        vlm_threshold=vlm_threshold,
        vlm_backend=vlm_backend,
        enable_human_gate=enable_human_gate,
        enable_omniparser=enable_omniparser,
        omniparser_confidence_threshold=omniparser_confidence_threshold,
        enable_perf=enable_perf,
        perf_budget=perf_budget,
    )
