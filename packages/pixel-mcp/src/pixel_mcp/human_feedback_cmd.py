"""``pixel-mcp human-feedback`` — capture the Level 3 human verdict.

Writes ``.pixel-mcp/human-feedback.json`` with a verdict (``approved`` or
``rejected``) plus optional rejection notes. The next ``pixel-mcp check
--enable-human-gate`` invocation consumes this file:

- ``approved`` → Final Convergence at Level 3.
- ``rejected`` → a pseudo-Delta (``property="human_review"``) is appended to
  the Delta list so loop-economics (stuck / regression / hash) and the
  ConvergenceJudge can react. Different rejection notes → different hash →
  not stuck. ``severity="critical"`` blocks Final Convergence.

The file carries ``consumed: false`` at write-time; ``check`` flips it to
``true`` after applying the verdict so the same feedback is never
double-counted. A second ``human-feedback`` invocation overwrites the file
(last-wins) even when the previous record is still unconsumed.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.state import state_dir

EXIT_OK = 0
EXIT_FATAL = 12

FEEDBACK_FILENAME = "human-feedback.json"
SCHEMA_VERSION = 1


def feedback_path(project_root: Path | None = None) -> Path:
    return state_dir(project_root) / FEEDBACK_FILENAME


def run(
    approve: bool = False,
    rejection_notes: str | None = None,
    project_root: Path | None = None,
) -> tuple[Envelope, int]:
    """Persist the human's Level 3 verdict.

    Exactly one of ``approve`` (set to ``True``) or ``rejection_notes`` (a
    non-empty string) must be provided. Other combinations return an
    EXIT_FATAL envelope.
    """
    has_approve = bool(approve)
    has_notes = bool(rejection_notes and rejection_notes.strip())

    if has_approve and has_notes:
        return _error(
            "feedback_args_conflict",
            "Pass either --approve OR --rejection-notes, not both.",
        ), EXIT_FATAL
    if not has_approve and not has_notes:
        return _error(
            "feedback_args_missing",
            'Pass --approve (sign off) or --rejection-notes "..." (request changes).',
        ), EXIT_FATAL

    payload = {
        "schema_version": SCHEMA_VERSION,
        "verdict": "approved" if has_approve else "rejected",
        "notes": None if has_approve else (rejection_notes or "").strip(),
        "captured_at": datetime.now(UTC).isoformat(),
        "consumed": False,
    }

    path = feedback_path(project_root)
    _atomic_write_json(path, payload)

    verdict = payload["verdict"]
    if verdict == "approved":
        next_action = (
            "Re-run `pixel-mcp check --enable-human-gate` to record Final "
            "Convergence at Level 3."
        )
        hints = [
            "Human verdict captured: APPROVED (Final Convergence will land on next check).",
        ]
    else:
        next_action = (
            "Have the Agent address the rejection notes, then re-run "
            "`pixel-mcp check --enable-human-gate`."
        )
        hints = [
            f"Human verdict captured: REJECTED — notes: {payload['notes']!s}",
            "The next `check` injects this as a pseudo-Delta and re-opens the loop.",
        ]

    envelope = make_envelope(
        data={
            "verdict": payload["verdict"],
            "notes": payload["notes"],
            "captured_at": payload["captured_at"],
            "feedback_path": str(path),
            "consumed": False,
        },
        hints=hints,
        diagnostics={"feedback_path": str(path), "verdict": payload["verdict"]},
        next_suggested_action=next_action,
        affordances=[
            {
                "tool": "mcp__pixel_mcp__check",
                "when": "to consume the verdict on the next loop turn",
            },
        ],
    )
    return envelope, EXIT_OK


def read_feedback(project_root: Path | None = None) -> dict[str, Any] | None:
    """Return the on-disk feedback record, or ``None`` if absent/corrupt."""
    path = feedback_path(project_root)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("schema_version") != SCHEMA_VERSION:
        return None
    return raw


def mark_consumed(project_root: Path | None = None) -> None:
    """Flip ``consumed`` to True on the existing feedback record."""
    path = feedback_path(project_root)
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(raw, dict):
        return
    raw["consumed"] = True
    _atomic_write_json(path, raw)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write: temp file + rename. Mirrors state.py convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".human-feedback-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _error(error_type: str, error_message: str) -> Envelope:
    return make_envelope(
        data=None,
        hints=[error_message],
        diagnostics={"error_type": error_type, "error_message": error_message},
        next_suggested_action=('Pass exactly one of --approve OR --rejection-notes "...".'),
        affordances=[
            {
                "tool": "mcp__pixel_mcp__human_feedback",
                "when": "after fixing the argument set",
            },
        ],
    )
