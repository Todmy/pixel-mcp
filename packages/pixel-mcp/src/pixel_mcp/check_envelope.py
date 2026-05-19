"""Envelope construction + lightweight helpers for ``pixel-mcp check``.

Pure envelope-shape concerns: single-pass + multi-axis success envelopes,
fatal-envelope wrapper, hint construction, and the small Delta-stamping /
viewport-string helpers used by the orchestrator. Lives in its own module so
``check_cmd.py`` can stay a thin public surface and the orchestrator can
delegate all envelope assembly here.
"""

from __future__ import annotations

import json
from typing import Any

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.delta import Delta
from pixel_mcp.perf_metrics import PerfBudget, PerfMetrics
from pixel_mcp.render import BoundingBox, MeasuredDOM
from pixel_mcp.spec import DesignSpec

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


def _viewport_str(viewport: tuple[int, int]) -> str:
    """Stable ``"<W>x<H>"`` rendering used for hash buckets + on-disk paths."""
    return f"{viewport[0]}x{viewport[1]}"


_BROWSER_PRESETS: dict[str, list[str]] = {
    "all": ["chromium", "firefox", "webkit"],
}


def _stamp_viewport(deltas: list[Delta], viewport_str: str) -> list[Delta]:
    """Return new Deltas with the ``viewport`` field set.

    Pydantic models are immutable from the outside in our usage — rebuild with
    ``model_copy(update=...)`` so callers can keep treating Deltas as plain
    value objects. Idempotent: a Delta already carrying ``viewport`` keeps it.
    """
    out: list[Delta] = []
    for d in deltas:
        if d.viewport is None:
            out.append(d.model_copy(update={"viewport": viewport_str}))
        else:
            out.append(d)
    return out


def _stamp_browser(deltas: list[Delta], browser_str: str) -> list[Delta]:
    """Return new Deltas with the ``browser`` field set (v2-2 cross-browser)."""
    out: list[Delta] = []
    for d in deltas:
        if d.browser is None:
            out.append(d.model_copy(update={"browser": browser_str}))
        else:
            out.append(d)
    return out


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
    omniparser_enabled: bool = False,
    omniparser_detections: list[Any] | None = None,
    omniparser_hint: str | None = None,
    perf_enabled: bool = False,
    perf_budget: PerfBudget | None = None,
    perf_metrics: list[PerfMetrics] | None = None,
    perf_hint: str | None = None,
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
        "omniparser_enabled": omniparser_enabled,
        "omniparser_detections": (
            [json.loads(d.model_dump_json()) for d in omniparser_detections]
            if omniparser_detections is not None
            else None
        ),
        "perf_enabled": perf_enabled,
        "perf_budget": perf_budget.model_dump() if perf_budget is not None else None,
        "perf_metrics": (
            [json.loads(m.model_dump_json()) for m in (perf_metrics or [])] if perf_enabled else []
        ),
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
    if omniparser_hint is not None:
        hints.append(omniparser_hint)
    if perf_hint is not None:
        hints.append(perf_hint)
    if perf_enabled and perf_metrics:
        failing_perf = [d for d in deltas if d.property.startswith("perf_")]
        if failing_perf:
            hints.append(
                f"Performance budget exceeded on {len(failing_perf)} metric(s) — "
                "see data.perf_metrics + Deltas with property `perf_*`."
            )
    if omniparser_enabled and omniparser_detections is not None:
        labelled = sum(1 for r in (regions or []) if getattr(r, "semantic_label", None) is not None)
        if labelled > 0:
            hints.append(
                f"OmniParser tagged {labelled} of {len(regions or [])} Region(s) "
                "with semantic labels — see data.regions[*].semantic_label."
            )
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
            "Level 2 Gate Pass. Awaiting Level 3 human verdict — call `mcp__pixel_mcp__review`."
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


def _success_envelope_multi(
    *,
    spec: DesignSpec | None,
    pass_results: list[dict[str, Any]],
    viewports: list[tuple[int, int]],
    browsers: list[str],
    viewports_specified: bool,
    browsers_specified: bool,
    combined_deltas: list[Delta],
    combined_judgment_data: dict[str, Any],
    overall_converged: bool,
    iteration: int,
    session_id: str,
    is_stuck: bool,
    is_regression: bool,
    max_iterations: int,
    mode: str,
    level_reached: int,
    dinov2_enabled: bool,
    dinov2_threshold: float | None,
    vlm_enabled: bool,
    vlm_threshold: float | None,
    vlm_backend: str | None,
    human_gate_enabled: bool,
    human_verdict: str | None,
    human_notes: str | None,
    human_gate_pending: bool,
    omniparser_enabled: bool,
    treat_minor_as_blocking: bool,
    perf_enabled: bool = False,
    perf_budget: PerfBudget | None = None,
) -> Envelope:
    """Build the AXI envelope for a multi-axis check (v2-1 + v2-2).

    Surfaces ``data.measurement_results`` as the canonical per-pass list
    (one entry per ``(browser, viewport)`` cell). For backward compatibility
    with the v2-1 envelope shape, ``data.viewport_results`` is also emitted
    whenever the caller passed ``viewports`` (independent of whether
    ``browsers`` was set). The legacy field carries one entry per viewport;
    when only one browser was used the two lists are content-equivalent.
    """
    delta_dicts = [json.loads(d.model_dump_json()) for d in combined_deltas]

    measurement_results: list[dict[str, Any]] = []
    aggregated_hot_regions: list[dict[str, Any]] = []
    aggregated_regions: list[dict[str, Any]] = []
    aggregated_dinov2: list[dict[str, Any]] = []
    aggregated_vlm: list[dict[str, Any]] = []
    aggregated_omniparser: list[dict[str, Any]] = []
    aggregated_perf: list[dict[str, Any]] = []
    hints: list[str] = [combined_judgment_data["summary"]]

    for pr in pass_results:
        vp_str = pr["viewport"]
        br_str = pr["browser"]
        cell_label = f"{br_str} @ {vp_str}"
        ssim_score = pr["ssim_score"]
        hot_regions = pr["hot_regions"] or []
        regions = pr["regions"] or []
        significant = [r for r in hot_regions if r.w * r.h >= MIN_BBOX_AREA]
        measurement_results.append(
            {
                "browser": br_str,
                "viewport": vp_str,
                "converged": pr["viewport_converged"],
                "level_reached": pr["level_reached"],
                "ssim_score": ssim_score,
                "hot_region_count": len(hot_regions),
                "significant_hot_region_count": len(significant),
                "delta_count": len(pr["deltas"]),
                "visual_error": pr["visual_error"],
                "summary": ("Gate Pass" if pr["viewport_converged"] else "Deltas present"),
            }
        )
        for hr in hot_regions:
            hr_d = json.loads(hr.model_dump_json())
            hr_d["viewport"] = vp_str
            hr_d["browser"] = br_str
            aggregated_hot_regions.append(hr_d)
        for region in regions:
            aggregated_regions.append(json.loads(region.model_dump_json()))
        if pr["dinov2_similarities"]:
            for s in pr["dinov2_similarities"]:
                aggregated_dinov2.append({**s, "viewport": vp_str, "browser": br_str})
        if pr["vlm_judgments"]:
            for j in pr["vlm_judgments"]:
                aggregated_vlm.append({**j, "viewport": vp_str, "browser": br_str})
        if pr["omniparser_detections"] is not None:
            for det in pr["omniparser_detections"]:
                det_d = json.loads(det.model_dump_json())
                det_d["viewport"] = vp_str
                det_d["browser"] = br_str
                aggregated_omniparser.append(det_d)

        if ssim_score is not None and ssim_score < SSIM_THRESHOLD:
            hints.append(
                f"[{cell_label}] SSIM Score {ssim_score:.3f} below threshold {SSIM_THRESHOLD}."
            )
        if significant:
            hints.append(
                f"[{cell_label}] {len(significant)} Hot Region(s) above {MIN_BBOX_AREA}px²."
            )
        if pr["visual_error"] is not None:
            hints.append(f"[{cell_label}] Visual signal unavailable ({pr['visual_error'][:80]}).")
        if pr["dinov2_hint"]:
            hints.append(f"[{cell_label}] {pr['dinov2_hint']}")
        if pr["vlm_hint"]:
            hints.append(f"[{cell_label}] {pr['vlm_hint']}")
        if pr["omniparser_hint"]:
            hints.append(f"[{cell_label}] {pr['omniparser_hint']}")
        perf_metric_obj = pr.get("perf_metrics")
        if perf_metric_obj is not None:
            aggregated_perf.append(json.loads(perf_metric_obj.model_dump_json()))
        if pr.get("perf_hint"):
            hints.append(f"[{cell_label}] {pr['perf_hint']}")

    data: dict[str, Any] = {
        "mode": mode,
        "converged": overall_converged,
        "level_reached": level_reached,
        "summary": combined_judgment_data["summary"],
        "judgment": combined_judgment_data,
        "deltas": delta_dicts,
        "measurement_results": measurement_results,
        "hot_regions": aggregated_hot_regions,
        "regions": aggregated_regions,
        "ssim_threshold": SSIM_THRESHOLD,
        "spec_node_id": spec.figma_node_id if spec is not None else None,
        "iteration": iteration,
        "session_id": session_id,
        "max_iterations": max_iterations,
        "is_stuck": is_stuck,
        "is_regression": is_regression,
        "dinov2_enabled": dinov2_enabled,
        "dinov2_threshold": dinov2_threshold,
        "dinov2_similarities": aggregated_dinov2 if dinov2_enabled else None,
        "vlm_enabled": vlm_enabled,
        "vlm_threshold": vlm_threshold,
        "vlm_backend": vlm_backend,
        "vlm_judgments": aggregated_vlm if vlm_enabled else None,
        "human_gate_enabled": human_gate_enabled,
        "human_verdict": human_verdict,
        "human_notes": human_notes,
        "omniparser_enabled": omniparser_enabled,
        "omniparser_detections": (aggregated_omniparser if omniparser_enabled else None),
        "perf_enabled": perf_enabled,
        "perf_budget": perf_budget.model_dump() if perf_budget is not None else None,
        "perf_metrics": aggregated_perf if perf_enabled else [],
    }
    # Surface the axes the caller actually selected. The v2-1 envelope used
    # ``data.viewports`` + ``data.viewport_results``; keep them when the
    # caller passed --viewports so v2-1 tests / consumers stay green.
    if viewports_specified:
        data["viewports"] = [_viewport_str(v) for v in viewports]
        # Collapse the cross-product back to a per-viewport summary using the
        # first browser (the v2-1 shape always had one browser anyway). When
        # browsers_specified=True too, callers should read measurement_results.
        first_browser = browsers[0]
        data["viewport_results"] = [
            {k: v for k, v in entry.items() if k != "browser"}
            for entry in measurement_results
            if entry["browser"] == first_browser
        ]
    if browsers_specified:
        data["browsers"] = list(browsers)

    axes_desc: list[str] = []
    if browsers_specified:
        axes_desc.append(f"{len(browsers)} browser(s)")
    if viewports_specified:
        axes_desc.append(f"{len(viewports)} viewport(s)")
    if axes_desc:
        hints.append(
            "Cross-product check across "
            + " × ".join(axes_desc)
            + f" = {len(pass_results)} measurement pass(es)."
        )
    hints.append(f"Iteration {iteration} of {max_iterations}.")
    if is_stuck:
        hints.append(
            "STUCK: last 3 Iterations produced identical structured-delta hashes — "
            "the Agent isn't making progress. Either fix the listed Deltas or run "
            "`pixel-mcp reset` to clear state."
        )
    if is_regression:
        hints.append(
            "REGRESSION: a previously-passed Level is now failing across the "
            "matrix. A recent edit broke a (browser, viewport) cell that used to work."
        )
    if human_gate_enabled and human_gate_pending:
        hints.append(
            "Level 3 (human review) is pending — call `mcp__pixel_mcp__review` to see "
            "the crop pairs inline, then `mcp__pixel_mcp__human_feedback`."
        )
    if perf_enabled:
        failing_perf = [d for d in combined_deltas if d.property.startswith("perf_")]
        if failing_perf:
            hints.append(
                f"Performance budget exceeded across the matrix on {len(failing_perf)} "
                "metric(s) — see data.perf_metrics + Deltas with property `perf_*`."
            )

    if not overall_converged:
        next_action = (
            "Have the Agent fix the listed Deltas (note the per-Delta "
            "`browser` + `viewport` fields), then re-invoke `mcp__pixel_mcp__check`."
        )
    elif level_reached >= 3:
        next_action = "Final Convergence at Level 3 across the full matrix."
    elif level_reached >= 2 and human_gate_enabled:
        next_action = "Level 2 Gate Pass across the matrix. Awaiting Level 3 human verdict."
    elif level_reached >= 2:
        next_action = (
            "Level 2 Gate Pass across the matrix. Promote to Level 3 by re-running "
            "with `--enable-human-gate`."
        )
    elif level_reached >= 1:
        next_action = (
            "Level 1 Gate Pass across the matrix. Promote to Level 2 by re-running "
            "`pixel-mcp check --enable-vlm`."
        )
    else:
        next_action = (
            "Level 0 Gate Pass across the matrix. Promote to Level 1 by re-running "
            "`pixel-mcp check --enable-dinov2`."
        )

    diagnostics: dict[str, Any] = {
        "mode": mode,
        "viewport_count": len(viewports),
        "browser_count": len(browsers),
        "pass_count": len(pass_results),
        "delta_count": len(combined_deltas),
        "critical_count": combined_judgment_data["critical_count"],
        "major_count": combined_judgment_data["major_count"],
        "minor_count": combined_judgment_data["minor_count"],
        "regression_count": combined_judgment_data["regression_count"],
    }

    return make_envelope(
        data=data,
        hints=hints,
        diagnostics=diagnostics,
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
