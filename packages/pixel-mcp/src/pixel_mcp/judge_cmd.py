"""Command-layer wrapper for ``judge_deltas``.

Reads a Delta[] JSON from disk (the ``data.deltas`` field produced by
``pixel-mcp diff``, or the top-level JSON if the user piped just the array),
runs the ConvergenceJudge, and wraps the result in an AXI envelope.

Exit codes:
- 0 — Converged.
- 1 — Not converged.
- 12 — Fatal: malformed JSON or unreadable file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pixel_tools_shared import Envelope, make_envelope
from pydantic import ValidationError

from pixel_mcp.delta import Delta
from pixel_mcp.judge import Judgment, Tolerance, judge_deltas

EXIT_OK = 0
EXIT_NOT_CONVERGED = 1
EXIT_FATAL = 12


def run(
    deltas_path: Path,
    treat_minor_as_blocking: bool = False,
) -> tuple[Envelope, int]:
    """Load Delta[] from a file, run the Judge, return an AXI envelope."""
    try:
        raw = json.loads(deltas_path.read_text())
    except FileNotFoundError:
        return _error_envelope(
            "deltas_not_found",
            f"Deltas file not found: {deltas_path}",
            ["Produce a Deltas file via `pixel-mcp diff --out deltas.json` first."],
        ), EXIT_FATAL
    except json.JSONDecodeError as exc:
        return _error_envelope(
            "deltas_invalid_json",
            f"Failed to parse {deltas_path}: {exc}",
            [
                "The file must be valid JSON — either a top-level Delta[] or an AXI envelope with `data.deltas`."
            ],
        ), EXIT_FATAL

    delta_dicts = _extract_delta_list(raw)
    if delta_dicts is None:
        return _error_envelope(
            "deltas_shape_unknown",
            "Could not find a Delta[] in the input — expected either a top-level array or an AXI envelope with data.deltas.",
            [
                "Re-generate with `pixel-mcp diff --out deltas.json` (the file will have the right shape)."
            ],
        ), EXIT_FATAL

    try:
        deltas = [Delta.model_validate(d) for d in delta_dicts]
    except ValidationError as exc:
        return _error_envelope(
            "deltas_schema_error",
            f"Delta entries failed schema validation: {exc}",
            ["The file does not match the current Delta schema — re-run `pixel-mcp diff`."],
        ), EXIT_FATAL

    judgment = judge_deltas(
        deltas, tolerance=Tolerance(treat_minor_as_blocking=treat_minor_as_blocking)
    )
    envelope = _judgment_envelope(judgment, len(deltas))
    return envelope, (EXIT_OK if judgment.converged else EXIT_NOT_CONVERGED)


def _extract_delta_list(raw: Any) -> list[dict[str, Any]] | None:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # AXI envelope shape: {data: {deltas: [...]}, ...}
        data = raw.get("data")
        if isinstance(data, dict):
            arr = data.get("deltas")
            if isinstance(arr, list):
                return arr
        # Or {deltas: [...]} at the top level
        arr = raw.get("deltas")
        if isinstance(arr, list):
            return arr
    return None


def _judgment_envelope(judgment: Judgment, total_input: int) -> Envelope:
    data: dict[str, Any] = json.loads(judgment.model_dump_json())
    hints: list[str] = [judgment.summary]
    if not judgment.converged:
        hints.append("Resolve critical and major Deltas before re-running the Convergence Loop.")
    return make_envelope(
        data=data,
        hints=hints,
        diagnostics={
            "input_delta_count": total_input,
            "critical_count": judgment.critical_count,
            "major_count": judgment.major_count,
            "minor_count": judgment.minor_count,
            "regression_count": judgment.regression_count,
        },
        next_suggested_action=(
            "Promote to the next enabled Level — Final Convergence at the current Level."
            if judgment.converged
            else "Have the Agent fix the listed Deltas, then re-run `pixel-mcp check`."
        ),
        affordances=[
            {
                "tool": "mcp__pixel_mcp__check",
                "when": "to re-run the full composite after the Agent fixes Deltas",
            },
        ],
    )


def _error_envelope(error_type: str, error_message: str, hints: list[str]) -> Envelope:
    return make_envelope(
        data=None,
        hints=hints,
        diagnostics={"error_type": error_type, "error_message": error_message},
        next_suggested_action="Resolve the input error, then re-run `pixel-mcp judge`.",
        affordances=[
            {"tool": "mcp__pixel_mcp__diff", "when": "to (re-)generate the Deltas file"},
        ],
    )
