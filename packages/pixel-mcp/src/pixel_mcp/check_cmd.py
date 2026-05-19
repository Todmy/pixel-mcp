"""Composite ``pixel-mcp check`` — one Iteration of the Convergence Loop.

Runs ``spec`` + ``measure`` + ``diff`` + ``judge`` end-to-end and returns a
single AXI envelope. This is the command Ralph Loop / any Loop Runner
invokes in its inner body.

Exit codes (per [PRD #10](https://github.com/Todmy/PBaaS/issues/10)):
- 0 — Final Convergence at the highest currently-enabled Level.
- 1 — Deltas present; loop continues.
- 2 — Ready for Level 3 (Slice #19+; reserved here).
- 3 — Regression detected (Slice #19+; reserved).
- 12 — Fatal (Figma/Render/IO error).

v0 scope: Level 0 with naïve DeltaDiffer only. SSIM Score and Hot Regions
arrive in Slice #6; Normalizer in Slice #5; Hierarchical decomposition in
Slice #7. The envelope leaves placeholders (``ssim_score: null``,
``hot_regions: []``) so the schema is stable across slices.
"""

from __future__ import annotations

import json
from typing import Any

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.delta import Delta, diff_design_vs_render
from pixel_mcp.figma_client import FigmaApiError, FigmaAuthError, FigmaError, FigmaNotFoundError
from pixel_mcp.figma_url import FigmaUrlError
from pixel_mcp.judge import Tolerance, judge_deltas
from pixel_mcp.normalize import normalize_spec_for_viewport
from pixel_mcp.render import (
    ChromiumNotInstalledError,
    MeasuredDOM,
    PlaywrightNotInstalledError,
    RenderError,
    RouteUnreachableError,
    WaitForTimeoutError,
    measure_render,
)
from pixel_mcp.spec import DesignSpec, UnsupportedNodeTypeError, extract_spec

EXIT_CONVERGED = 0
EXIT_DELTAS = 1
EXIT_READY_FOR_LEVEL_3 = 2  # reserved
EXIT_REGRESSION = 3  # reserved
EXIT_FATAL = 12


def run(
    figma_url: str,
    route: str,
    viewport: tuple[int, int] = (1280, 720),
    selectors: list[str] | None = None,
    wait_for: str | None = None,
    refresh_spec: bool = False,
    treat_minor_as_blocking: bool = False,
) -> tuple[Envelope, int]:
    """Run one Iteration of the Convergence Loop.

    Never raises — all errors are folded into the AXI envelope.
    """
    # --- 1) Extract DesignSpec ---
    try:
        spec = extract_spec(figma_url, refresh=refresh_spec)
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

    # --- 3) Normalize spec for viewport, then Diff + Judge ---
    normalized_spec = normalize_spec_for_viewport(spec, dom.viewport)
    deltas = diff_design_vs_render(normalized_spec, dom)
    judgment = judge_deltas(
        deltas, tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking)
    )

    envelope = _success_envelope(
        spec=spec,
        dom=dom,
        deltas=deltas,
        judgment_data=json.loads(judgment.model_dump_json()),
        truncated=truncated,
    )
    return envelope, (EXIT_CONVERGED if judgment.converged else EXIT_DELTAS)


def _success_envelope(
    *,
    spec: DesignSpec,
    dom: MeasuredDOM,
    deltas: list[Delta],
    judgment_data: dict[str, Any],
    truncated: bool,
) -> Envelope:
    delta_dicts = [json.loads(d.model_dump_json()) for d in deltas]
    data: dict[str, Any] = {
        "converged": judgment_data["converged"],
        "level_reached": 0,
        "summary": judgment_data["summary"],
        "judgment": judgment_data,
        "deltas": delta_dicts,
        # Reserved fields for later slices — keep the schema shape stable.
        "ssim_score": None,
        "hot_regions": [],
        "spec_node_id": spec.figma_node_id,
        "dom_route": dom.route,
        "dom_element_count": len(dom.elements),
    }

    hints: list[str] = _build_hints(judgment_data, deltas, truncated)

    next_action = (
        "Promote to Level 1 (DINOv2 per-crop similarity) once that plugin is enabled."
        if judgment_data["converged"]
        else "Have the Agent fix the listed Deltas, then re-invoke `mcp__pixel_mcp__check`."
    )

    return make_envelope(
        data=data,
        hints=hints,
        diagnostics={
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


def _build_hints(judgment_data: dict[str, Any], deltas: list[Delta], truncated: bool) -> list[str]:
    hints: list[str] = [judgment_data["summary"]]
    if not judgment_data["converged"]:
        # Per-property guidance
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
                "Font-family mismatch — confirm the font is loaded (link or @font-face) on the Render side."
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
