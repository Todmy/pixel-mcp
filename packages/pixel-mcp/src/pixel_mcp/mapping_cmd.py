"""Command-layer wrapper for ``resolve_mappings``.

Generates the Mappings file by calling ``extract_spec`` + ``measure_render`` +
``resolve_mappings``, persists to ``.pixel-mcp/mappings.json``, and returns
an AXI envelope.

Exit codes:
- 0 — Mappings file written.
- 12 — Fatal (Figma / Render / IO).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.figma_client import FigmaApiError, FigmaAuthError, FigmaError, FigmaNotFoundError
from pixel_mcp.figma_url import FigmaUrlError
from pixel_mcp.mapping import Mappings, resolve_mappings
from pixel_mcp.render import (
    ChromiumNotInstalledError,
    PlaywrightNotInstalledError,
    RenderError,
    RouteUnreachableError,
    WaitForTimeoutError,
    measure_render,
)
from pixel_mcp.spec import UnsupportedNodeTypeError, extract_spec
from pixel_mcp.state import state_dir

EXIT_OK = 0
EXIT_FATAL = 12


def run(
    figma_url: str,
    route: str,
    viewport: tuple[int, int] = (1280, 720),
    project_root: Path | None = None,
    refresh: bool = False,
) -> tuple[Envelope, int]:
    """Resolve Mappings end-to-end. Writes ``.pixel-mcp/mappings.json``."""
    project_root = project_root or Path.cwd()
    try:
        spec = extract_spec(figma_url)
    except (
        FigmaUrlError,
        FigmaAuthError,
        FigmaNotFoundError,
        UnsupportedNodeTypeError,
        FigmaApiError,
        FigmaError,
    ) as exc:
        return _error("figma_error", str(exc)), EXIT_FATAL

    try:
        dom, _truncated = measure_render(route=route, viewport=viewport)
    except (
        PlaywrightNotInstalledError,
        ChromiumNotInstalledError,
        WaitForTimeoutError,
        RouteUnreachableError,
        RenderError,
    ) as exc:
        return _error("render_error", str(exc)), EXIT_FATAL

    _ = refresh  # v0 always recomputes; refresh flag reserved for caching
    mappings = resolve_mappings(spec, dom)
    out_path = state_dir(project_root) / "mappings.json"
    out_path.write_text(mappings.model_dump_json(indent=2))

    counts = _counts_by_source(mappings)
    return _success(mappings, out_path, counts), EXIT_OK


def _counts_by_source(mappings: Mappings) -> dict[str, int]:
    out: dict[str, int] = {"code_connect": 0, "ai": 0, "heuristic": 0}
    for p in mappings.pairs:
        out[p.source] = out.get(p.source, 0) + 1
    return out


def _success(mappings: Mappings, out_path: Path, counts: dict[str, int]) -> Envelope:
    data: dict[str, Any] = json.loads(mappings.model_dump_json())
    data["written_to"] = str(out_path)
    data["counts_by_source"] = counts
    hints: list[str] = [
        f"Resolved {len(mappings.pairs)} Mapping(s) — written to {out_path}.",
    ]
    if counts["heuristic"] and not counts["code_connect"] and not counts["ai"]:
        hints.append(
            "All Mappings came from heuristic matching — set up Figma Code Connect "
            "or enable AI pairing (v0.5) for higher confidence."
        )
    return make_envelope(
        data=data,
        hints=hints,
        diagnostics={"counts_by_source": counts, "pair_count": len(mappings.pairs)},
        next_suggested_action="Run `pixel-mcp check` — mapped pairs feed into Delta attribution.",
        affordances=[
            {
                "tool": "mcp__pixel_mcp__check",
                "when": "to use the new Mappings in the Convergence Loop",
            },
        ],
    )


def _error(error_type: str, error_message: str) -> Envelope:
    return make_envelope(
        data=None,
        hints=[f"Failed to resolve Mappings: {error_message[:200]}"],
        diagnostics={"error_type": error_type, "error_message": error_message},
        next_suggested_action="Resolve the error above, then re-run `pixel-mcp mapping`.",
        affordances=[
            {"tool": "mcp__pixel_mcp__doctor", "when": "to diagnose environment issues"},
        ],
    )
