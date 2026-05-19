"""Loop economics — iteration counter, stuck detection, history, regression.

Public entry points:
- :func:`read_state` / :func:`write_state` — persist ``.pixel-mcp/state.json``.
- :func:`hash_deltas_bucketed` — magnitude-bucketed hash for stuck detection.
- :func:`detect_stuck` — last N hashes identical → True.
- :func:`detect_regression` — current level lower than highest-reached.
- :func:`append_history` — append-only ``.pixel-mcp/history.jsonl``.
- :func:`reset_state` — wipe state files (keeps named snapshots).

Per CONTEXT.md / PRD #10:
- Iteration counter ticks on every ``pixel-mcp check`` invocation.
- Counter resets on explicit ``pixel-mcp reset`` OR on Final Convergence.
- Stuck = last ``stuck_threshold`` hashes identical (default 3).
- Hash uses ``(selector, property, magnitude_bucket)`` tuples — bucketed so
  float jitter doesn't mask a stuck state.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from pixel_mcp.delta import Delta
from pixel_mcp.state import state_dir

__all__ = [
    "IterationState",
    "append_history",
    "bucket_for_magnitude",
    "compute_file_hashes",
    "detect_regression",
    "detect_stuck",
    "hash_deltas_bucketed",
    "read_state",
    "reset_state",
    "write_state",
]

STATE_FILENAME = "state.json"
HISTORY_FILENAME = "history.jsonl"
FILE_HASHES_FILENAME = "file-hashes.json"
DEFAULT_STUCK_THRESHOLD = 3
DEFAULT_MAX_ITERATIONS = 15
HASH_TRAIL_MAX = 10


class IterationState(BaseModel):
    """Persisted state for one Convergence Loop session."""

    schema_version: int = 1
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    iteration: int = 0
    last_delta_hash: str | None = None
    highest_level_reached: int = 0
    last_invocation_at: datetime | None = None
    recent_hashes: list[str] = Field(default_factory=list)


def read_state(project_root: Path | None = None) -> IterationState:
    """Read state.json, creating a fresh session if absent."""
    root = project_root or Path.cwd()
    path = state_dir(root) / STATE_FILENAME
    if not path.exists():
        return IterationState()
    try:
        return IterationState.model_validate_json(path.read_text())
    except Exception:
        # Corrupt state file → start a fresh session rather than crashing.
        return IterationState()


def write_state(state: IterationState, project_root: Path | None = None) -> None:
    """Atomic write of state.json."""
    root = project_root or Path.cwd()
    path = state_dir(root) / STATE_FILENAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    tmp.replace(path)


def reset_state(project_root: Path | None = None) -> None:
    """Wipe state.json, history.jsonl, file-hashes.json; preserve snapshots/."""
    root = project_root or Path.cwd()
    sd = state_dir(root)
    for fname in (STATE_FILENAME, HISTORY_FILENAME, FILE_HASHES_FILENAME):
        p = sd / fname
        if p.exists():
            p.unlink()
    # Crops live in iter-N/ sub-folders — clear them too.
    crops_root = sd / "crops"
    if crops_root.exists():
        for sub in crops_root.iterdir():
            if sub.is_dir() and sub.name.startswith("iter-"):
                for f in sub.iterdir():
                    f.unlink()
                sub.rmdir()


# --- Delta hashing & stuck detection -------------------------------------


_MAGNITUDE_BUCKETS = [
    (0.5, "<0.5"),
    (5.0, "<5"),
    (10.0, "<10"),
    (50.0, "<50"),
    (200.0, "<200"),
    (float("inf"), ">=200"),
]


def bucket_for_magnitude(magnitude: float | None) -> str:
    """Coarsen a numeric magnitude into a stable bucket label.

    Float jitter (sub-pixel rendering, FP noise) is washed out so a "no
    progress" run hashes consistently.
    """
    if magnitude is None:
        return "none"
    for ceiling, label in _MAGNITUDE_BUCKETS:
        if magnitude < ceiling:
            return label
    return ">=200"


def hash_deltas_bucketed(deltas: list[Delta]) -> str:
    """Hash a Delta[] by (selector, property, magnitude_bucket, severity, viewport, browser).

    The ``viewport`` field (v2-1) and ``browser`` field (v2-2) are folded in so
    the same property mismatch observed under different breakpoints / engines
    hashes to distinct buckets — stuck detection stays accurate when one cell
    of the (browser × viewport) matrix regresses while another converges.
    Pre-v2-1/v2-2 callers leave those fields as ``None`` (rendered as the
    empty string), which keeps cross-version hash semantics stable for the
    single-viewport / single-browser case.
    """
    rows = sorted(
        (
            (
                d.selector,
                d.property,
                bucket_for_magnitude(d.magnitude),
                d.severity,
                d.viewport or "",
                d.browser or "",
            )
            for d in deltas
        ),
        key=lambda t: (t[0], t[1], t[2], t[3], t[4], t[5]),
    )
    payload = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detect_stuck(
    recent_hashes: list[str], current_hash: str, threshold: int = DEFAULT_STUCK_THRESHOLD
) -> bool:
    """Return True when the last ``threshold`` hashes (including current) match."""
    tail = recent_hashes[-(threshold - 1) :] if threshold > 1 else []
    chain = [*tail, current_hash]
    if len(chain) < threshold:
        return False
    return all(h == current_hash for h in chain)


def detect_regression(state: IterationState, current_level_passed: int) -> bool:
    """Return True when a previously-passed higher level is now failing."""
    return current_level_passed < state.highest_level_reached


# --- History trace --------------------------------------------------------


def append_history(entry: dict[str, object], project_root: Path | None = None) -> None:
    """Append one JSON entry to history.jsonl."""
    root = project_root or Path.cwd()
    path = state_dir(root) / HISTORY_FILENAME
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# --- File-hashes for Productive Iteration ---------------------------------


def compute_file_hashes(files: list[Path]) -> dict[str, str]:
    """SHA-256 of each file. Used to detect code change between Iterations."""
    out: dict[str, str] = {}
    for p in files:
        try:
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            out[str(p)] = "<missing>"
    return out


def now_utc() -> datetime:
    return datetime.now(UTC)
