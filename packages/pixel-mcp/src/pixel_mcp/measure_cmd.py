"""Command-layer wrapper for ``measure_render``.

Both the CLI subcommand and the MCP tool delegate here so the AXI envelope
shape stays consistent across surfaces.

Exit codes:
- 0 — MeasuredDOM captured successfully.
- 12 — Playwright missing, Chromium missing, route unreachable, wait-for
       timeout, or any other fatal capture error. (One fatal exit per v0
       convention from Slice 2.)
"""

from __future__ import annotations

import json
from typing import Any

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.render import (
    MAX_ELEMENTS,
    ChromiumNotInstalledError,
    MeasuredDOM,
    PlaywrightNotInstalledError,
    RenderError,
    RouteUnreachableError,
    WaitForTimeoutError,
    measure_render,
)

EXIT_OK = 0
EXIT_FATAL = 12

_SUCCESS_AFFORDANCES: list[dict[str, str]] = [
    {
        "tool": "mcp__pixel_mcp__diff",
        "when": "when you have both a DesignSpec (from `spec`) and this MeasuredDOM — diff them",
    },
    {
        "tool": "mcp__pixel_mcp__check",
        "when": "for the full composite pipeline (spec + measure + diff + judge in one call)",
    },
]


def run(
    route: str,
    viewport: tuple[int, int] = (1280, 720),
    selectors: list[str] | None = None,
    wait_for: str | None = None,
    wait_for_network_idle: bool = True,
) -> tuple[Envelope, int]:
    """Capture a MeasuredDOM and wrap it in an AXI envelope.

    Returns ``(envelope, exit_code)``. Never raises — capture errors fold
    into the envelope's ``diagnostics`` and ``hints``.
    """
    try:
        dom, truncated = measure_render(
            route=route,
            viewport=viewport,
            selectors=selectors,
            wait_for=wait_for,
            wait_for_network_idle=wait_for_network_idle,
        )
    except PlaywrightNotInstalledError as exc:
        return _error_envelope(
            error_type="playwright_not_installed",
            error_message=str(exc),
            hints=[
                "Install Playwright: `uv sync` (the dependency is declared in pyproject.toml).",
                "Then install the Chromium binary: `uv run playwright install chromium` (one-time, ~150MB).",
            ],
        ), EXIT_FATAL
    except ChromiumNotInstalledError as exc:
        return _error_envelope(
            error_type="chromium_not_installed",
            error_message=str(exc),
            hints=[
                "Install the Chromium browser binary: `uv run playwright install chromium`.",
                "This is a one-time ~150MB download.",
            ],
        ), EXIT_FATAL
    except WaitForTimeoutError as exc:
        return _error_envelope(
            error_type="wait_for_timeout",
            error_message=str(exc),
            hints=[
                "Check that the `--wait-for` selector exists on the rendered page.",
                "Increase the timeout if the page is slow to hydrate, or remove `--wait-for` to rely on networkidle only.",
            ],
        ), EXIT_FATAL
    except RouteUnreachableError as exc:
        return _error_envelope(
            error_type="route_unreachable",
            error_message=str(exc),
            hints=[
                "Check that the dev server is running and listening on the URL you passed.",
                "Try `curl <route>` from the same shell to verify reachability.",
            ],
        ), EXIT_FATAL
    except RenderError as exc:  # catch-all for any future RenderError subclass
        return _error_envelope(
            error_type="render_error",
            error_message=str(exc),
            hints=["See diagnostics for details; run `pixel-mcp doctor` first."],
        ), EXIT_FATAL

    return _success_envelope(dom, truncated=truncated), EXIT_OK


def _success_envelope(dom: MeasuredDOM, truncated: bool) -> Envelope:
    data: dict[str, Any] = json.loads(dom.model_dump_json())
    hints: list[str] = [
        f"MeasuredDOM captured for {dom.route!r} at viewport "
        f"{dom.viewport[0]}x{dom.viewport[1]} ({len(dom.elements)} elements).",
    ]
    if truncated:
        hints.append(
            f"Auto-discover hit the {MAX_ELEMENTS}-element cap — pass `--selectors` "
            "to narrow the measurement window if the missing elements matter."
        )
    return make_envelope(
        data=data,
        hints=hints,
        diagnostics={
            "route": dom.route,
            "viewport": list(dom.viewport),
            "element_count": len(dom.elements),
            "truncated": truncated,
            "schema_version": dom.schema_version,
        },
        next_suggested_action=(
            "Run `pixel-mcp diff` (Slice 4) to compute Deltas between this "
            "MeasuredDOM and a DesignSpec."
        ),
        affordances=_SUCCESS_AFFORDANCES,
    )


def _error_envelope(error_type: str, error_message: str, hints: list[str]) -> Envelope:
    return make_envelope(
        data=None,
        hints=hints,
        diagnostics={
            "error_type": error_type,
            "error_message": error_message,
        },
        next_suggested_action="Resolve the error above, then re-run `pixel-mcp measure --route <url>`.",
        affordances=[
            {
                "tool": "mcp__pixel_mcp__doctor",
                "when": "to diagnose environment issues (Playwright, Chromium, etc.)",
            },
        ],
    )


__all__ = ["EXIT_FATAL", "EXIT_OK", "run"]
