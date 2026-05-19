"""Command-layer wrapper for ``diff_design_vs_render``.

Reads a DesignSpec JSON and a MeasuredDOM JSON from disk, runs the
DeltaDiffer, and wraps the result in an AXI envelope. Both the CLI
subcommand and the MCP tool delegate here.

Exit codes:
- 0 — Zero Deltas (the inputs match within naïve v0 rules).
- 1 — One or more Deltas present.
- 12 — Fatal: malformed JSON, missing keys, or unreadable file.
"""

from __future__ import annotations

import json
from pathlib import Path

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.delta import Delta, diff_design_vs_render
from pixel_mcp.render import MeasuredDOM
from pixel_mcp.spec import DesignSpec

EXIT_OK = 0
EXIT_DELTAS = 1
EXIT_FATAL = 12

_SUCCESS_AFFORDANCES: list[dict[str, str]] = [
    {
        "tool": "mcp__pixel_mcp__judge",
        "when": "to determine whether Final Convergence holds for these Deltas",
    },
    {
        "tool": "mcp__pixel_mcp__check",
        "when": "to run the full composite (spec + measure + diff + judge) instead",
    },
]


def run(
    spec_path: Path,
    measured_path: Path,
    mappings: dict[str, str] | None = None,
) -> tuple[Envelope, int]:
    """Load both inputs, diff them, return an AXI envelope.

    Never raises — all I/O and schema errors are folded into the envelope
    diagnostics.
    """
    try:
        spec = DesignSpec.model_validate_json(spec_path.read_text())
    except FileNotFoundError:
        return _error_envelope(
            "spec_not_found",
            f"Spec file not found: {spec_path}",
            ["Pass --spec pointing at a file produced by `pixel-mcp spec --out`."],
        ), EXIT_FATAL
    except Exception as exc:  # JSON or schema error
        return _error_envelope(
            "spec_invalid",
            f"Failed to load DesignSpec from {spec_path}: {exc}",
            [
                "The file must be a JSON DesignSpec — produce one via `pixel-mcp spec --out spec.json`."
            ],
        ), EXIT_FATAL

    try:
        dom = MeasuredDOM.model_validate_json(measured_path.read_text())
    except FileNotFoundError:
        return _error_envelope(
            "measured_not_found",
            f"MeasuredDOM file not found: {measured_path}",
            ["Pass --measured pointing at a file produced by `pixel-mcp measure --out`."],
        ), EXIT_FATAL
    except Exception as exc:
        return _error_envelope(
            "measured_invalid",
            f"Failed to load MeasuredDOM from {measured_path}: {exc}",
            [
                "The file must be a JSON MeasuredDOM — produce one via `pixel-mcp measure --out measured.json`."
            ],
        ), EXIT_FATAL

    deltas = diff_design_vs_render(spec, dom, mappings=mappings)
    envelope = _success_envelope(deltas, spec=spec, dom=dom)
    return envelope, (EXIT_OK if not deltas else EXIT_DELTAS)


def _success_envelope(deltas: list[Delta], spec: DesignSpec, dom: MeasuredDOM) -> Envelope:
    data: dict[str, object] = {
        "deltas": [json.loads(d.model_dump_json()) for d in deltas],
        "spec_node_id": spec.figma_node_id,
        "spec_node_type": spec.figma_node_type,
        "dom_route": dom.route,
        "dom_element_count": len(dom.elements),
    }
    hints: list[str] = _build_hints(deltas)
    return make_envelope(
        data=data,
        hints=hints,
        diagnostics={
            "delta_count": len(deltas),
            "spec_child_count": len(spec.children),
        },
        next_suggested_action=(
            "Run `pixel-mcp judge` against these Deltas to get the Convergence verdict."
            if deltas
            else "Final Convergence within Tolerance — promote to the next Level when ready."
        ),
        affordances=_SUCCESS_AFFORDANCES,
    )


def _build_hints(deltas: list[Delta]) -> list[str]:
    if not deltas:
        return ["Zero Deltas — the Render matches the Design Source within naïve v0 rules."]
    by_sev = {"critical": 0, "major": 0, "minor": 0, "regression": 0}
    for d in deltas:
        by_sev[d.severity] = by_sev.get(d.severity, 0) + 1
    parts = [f"{n} {s}" for s, n in by_sev.items() if n]
    return [
        f"{len(deltas)} Deltas: " + ", ".join(parts) + ".",
        "Fix critical and major Deltas first — they block Final Convergence.",
    ]


def _error_envelope(error_type: str, error_message: str, hints: list[str]) -> Envelope:
    return make_envelope(
        data=None,
        hints=hints,
        diagnostics={"error_type": error_type, "error_message": error_message},
        next_suggested_action="Resolve the input error, then re-run `pixel-mcp diff`.",
        affordances=[
            {"tool": "mcp__pixel_mcp__spec", "when": "to (re-)generate the DesignSpec file"},
            {"tool": "mcp__pixel_mcp__measure", "when": "to (re-)generate the MeasuredDOM file"},
        ],
    )
