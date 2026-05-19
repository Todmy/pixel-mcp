"""``pixel-mcp snapshot`` — capture a named Render baseline.

Writes ``.pixel-mcp/snapshots/<tag>/`` containing:
- ``measured.json`` — the MeasuredDOM
- ``screenshot.png`` — full-page screenshot
- ``metadata.json`` — tag, timestamp, route, viewport

Useful as a baseline for ad-hoc visual checks outside the Convergence Loop.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.loop_state import now_utc
from pixel_mcp.render import (
    ChromiumNotInstalledError,
    PlaywrightNotInstalledError,
    RenderError,
    RouteUnreachableError,
    WaitForTimeoutError,
    capture_screenshot,
    measure_render,
)
from pixel_mcp.state import state_dir

EXIT_OK = 0
EXIT_FATAL = 12


def run(
    route: str,
    tag: str,
    viewport: tuple[int, int] = (1280, 720),
    project_root: Path | None = None,
) -> tuple[Envelope, int]:
    """Capture a tagged baseline and write to ``.pixel-mcp/snapshots/<tag>/``."""
    project_root = project_root or Path.cwd()
    snapshots_dir = state_dir(project_root) / "snapshots" / tag
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    try:
        dom, _truncated = measure_render(route=route, viewport=viewport)
        screenshot_bytes = capture_screenshot(route=route, viewport=viewport)
    except (
        PlaywrightNotInstalledError,
        ChromiumNotInstalledError,
        WaitForTimeoutError,
        RouteUnreachableError,
        RenderError,
    ) as exc:
        return _error("render_error", str(exc)), EXIT_FATAL

    (snapshots_dir / "measured.json").write_text(dom.model_dump_json(indent=2))
    (snapshots_dir / "screenshot.png").write_bytes(screenshot_bytes)
    metadata: dict[str, Any] = {
        "tag": tag,
        "route": route,
        "viewport": list(viewport),
        "captured_at": _to_isoformat(now_utc()),
        "schema_version": 1,
    }
    (snapshots_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return _success(tag, snapshots_dir, metadata, dom_element_count=len(dom.elements)), EXIT_OK


def _to_isoformat(dt: datetime) -> str:
    return dt.isoformat()


def _success(
    tag: str, snapshots_dir: Path, metadata: dict[str, Any], dom_element_count: int
) -> Envelope:
    data: dict[str, Any] = {
        "tag": tag,
        "snapshot_dir": str(snapshots_dir),
        "metadata": metadata,
        "dom_element_count": dom_element_count,
    }
    return make_envelope(
        data=data,
        hints=[f"Snapshot {tag!r} written to {snapshots_dir}."],
        diagnostics={"snapshot_dir": str(snapshots_dir)},
        next_suggested_action=(
            f"Use this Snapshot as a baseline — compare future Renders against "
            f"{snapshots_dir / 'measured.json'} via `pixel-mcp diff`."
        ),
        affordances=[
            {"tool": "mcp__pixel_mcp__check", "when": "to run a Convergence Loop iteration"},
        ],
    )


def _error(error_type: str, error_message: str) -> Envelope:
    return make_envelope(
        data=None,
        hints=[f"Failed to capture snapshot: {error_message[:200]}"],
        diagnostics={"error_type": error_type, "error_message": error_message},
        next_suggested_action="Resolve the error above, then re-run `pixel-mcp snapshot`.",
        affordances=[
            {"tool": "mcp__pixel_mcp__doctor", "when": "to diagnose environment issues"},
        ],
    )
