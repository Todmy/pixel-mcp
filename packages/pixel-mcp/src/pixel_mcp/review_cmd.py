"""``pixel-mcp review`` — emit expected/actual images for Level 3 human review.

v0 stub: writes ``expected.png`` and ``actual.png`` to the State Directory
(when available from the most recent ``check`` run). The Claude Code chat
surface integration (the full Level 3 flow) lands in v1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.state import state_dir

EXIT_OK = 0
EXIT_READY_FOR_LEVEL_3 = 2
EXIT_FATAL = 12


def run(project_root: Path | None = None) -> tuple[Envelope, int]:
    """Prepare a Level 3 review packet from the most recent ``check`` outputs."""
    project_root = project_root or Path.cwd()
    sd = state_dir(project_root)
    crops_root = sd / "crops"

    if not crops_root.exists():
        return _error(
            "no_check_run_yet",
            "No Crops found in the State Directory. Run `pixel-mcp check` first.",
        ), EXIT_FATAL

    # Find the most recent iter-N folder by iteration number
    iter_dirs = sorted(
        (p for p in crops_root.iterdir() if p.is_dir() and p.name.startswith("iter-")),
        key=lambda p: int(p.name.removeprefix("iter-")),
    )
    if not iter_dirs:
        return _error(
            "no_check_run_yet",
            "No iteration sub-folders found. Run `pixel-mcp check` first.",
        ), EXIT_FATAL

    latest = iter_dirs[-1]
    expected_paths = sorted(latest.glob("exp-*.png"))
    actual_paths = sorted(latest.glob("act-*.png"))

    data: dict[str, Any] = {
        "iteration_dir": str(latest),
        "expected_crops": [str(p) for p in expected_paths],
        "actual_crops": [str(p) for p in actual_paths],
        "crop_pair_count": min(len(expected_paths), len(actual_paths)),
    }
    return make_envelope(
        data=data,
        hints=[
            f"Level 3 review packet ready — {data['crop_pair_count']} crop pair(s) at {latest}.",
            "v0 stub: open the images side-by-side in your editor. Full Claude "
            "Code chat-attachment flow lands in v1.",
        ],
        diagnostics={"iteration_dir": str(latest)},
        next_suggested_action=(
            "Inspect the crop pairs. Approve (Final Convergence) or note "
            "rejections and re-run `pixel-mcp check`."
        ),
        affordances=[
            {"tool": "mcp__pixel_mcp__check", "when": "after the Agent addresses rejection notes"},
        ],
    ), EXIT_READY_FOR_LEVEL_3


def _error(error_type: str, error_message: str) -> Envelope:
    return make_envelope(
        data=None,
        hints=[f"Cannot prepare review: {error_message}"],
        diagnostics={"error_type": error_type, "error_message": error_message},
        next_suggested_action="Run `pixel-mcp check` first to generate the review artifacts.",
        affordances=[
            {"tool": "mcp__pixel_mcp__check", "when": "to produce Crops and Deltas"},
        ],
    )
