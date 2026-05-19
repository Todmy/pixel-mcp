"""``pixel-mcp reset`` — clear the State Directory (preserve snapshots)."""

from __future__ import annotations

from pathlib import Path

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.loop_state import reset_state
from pixel_mcp.state import state_dir

EXIT_OK = 0


def run(project_root: Path | None = None, all_artifacts: bool = False) -> tuple[Envelope, int]:
    """Wipe state.json / history.jsonl / file-hashes / crops.

    ``--all`` also removes named snapshots.
    """
    project_root = project_root or Path.cwd()
    sd = state_dir(project_root)
    reset_state(project_root=project_root)
    if all_artifacts:
        snapshots = sd / "snapshots"
        if snapshots.exists():
            for sub in snapshots.iterdir():
                if sub.is_dir():
                    for f in sub.iterdir():
                        if f.is_file():
                            f.unlink()
                    sub.rmdir()
            snapshots.rmdir()
    return make_envelope(
        data={
            "state_dir": str(sd),
            "cleared_snapshots": all_artifacts,
        },
        hints=[
            f"State Directory cleared at {sd}.",
            "Next `pixel-mcp check` will start a fresh session.",
        ],
        diagnostics={"state_dir": str(sd)},
        next_suggested_action="Run `pixel-mcp check` to start a new Convergence Loop session.",
        affordances=[
            {"tool": "mcp__pixel_mcp__check", "when": "to begin a new session"},
        ],
    ), EXIT_OK
