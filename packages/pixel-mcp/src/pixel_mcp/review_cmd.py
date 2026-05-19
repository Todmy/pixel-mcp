"""``pixel-mcp review`` — emit expected/actual images for Level 3 human review.

Promoted from the v0 stub to the real Level 3 packet emitter. Pairs every
``exp-r<n>.png`` / ``act-r<n>.png`` crop from the latest ``iter-N`` folder,
attaches DOM / Figma / severity metadata where the last ``check`` recorded
it, and (in the MCP path) returns base64 PNG image content blocks so they
render inline in the Claude Code chat surface.

Three outputs:

- ``run()`` — the CLI-friendly path. Returns ``(envelope, exit_code)`` only.
  Image paths land in ``envelope.data.crop_pairs[*].expected_path`` /
  ``actual_path``; the CLI prints the JSON.
- ``build_packet()`` — the MCP path. Returns the same envelope plus a
  sequence of ``FastMCP Image`` objects ready to be returned alongside it
  from the MCP tool. Claude Code renders them inline.

Exit codes mirror ``check``: ``EXIT_READY_FOR_LEVEL_3 = 2`` on success
(crops present, human verdict pending), ``EXIT_FATAL = 12`` when there are
no crops to review.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.state import state_dir

EXIT_OK = 0
EXIT_READY_FOR_LEVEL_3 = 2
EXIT_FATAL = 12

_REGION_INDEX_RE = re.compile(r"r(\d+)\.png$", re.IGNORECASE)


@dataclass
class ReviewPacket:
    """Structured Level 3 review payload.

    ``envelope`` is the AXI response; ``images`` are FastMCP ``Image``
    handles for the MCP path to attach inline. The CLI path ignores the
    images and only emits the envelope JSON.
    """

    envelope: Envelope
    exit_code: int
    images: list[Any]  # list[mcp.server.fastmcp.utilities.types.Image]


def run(project_root: Path | None = None) -> tuple[Envelope, int]:
    """CLI entry point — envelope + exit code only (no image attachments)."""
    packet = build_packet(project_root)
    return packet.envelope, packet.exit_code


def build_packet(project_root: Path | None = None) -> ReviewPacket:
    """Build the full Level 3 packet (envelope + FastMCP Image attachments)."""
    project_root = project_root or Path.cwd()
    sd = state_dir(project_root)
    crops_root = sd / "crops"

    if not crops_root.exists():
        return ReviewPacket(
            envelope=_error(
                "no_check_run_yet",
                "No Crops found in the State Directory. Run `pixel-mcp check` first.",
            ),
            exit_code=EXIT_FATAL,
            images=[],
        )

    iter_dirs = sorted(
        (p for p in crops_root.iterdir() if p.is_dir() and p.name.startswith("iter-")),
        key=lambda p: int(p.name.removeprefix("iter-")),
    )
    if not iter_dirs:
        return ReviewPacket(
            envelope=_error(
                "no_check_run_yet",
                "No iteration sub-folders found. Run `pixel-mcp check` first.",
            ),
            exit_code=EXIT_FATAL,
            images=[],
        )

    latest = iter_dirs[-1]
    expected_paths = sorted(latest.glob("exp-*.png"))
    actual_paths = sorted(latest.glob("act-*.png"))

    if not expected_paths or not actual_paths:
        return ReviewPacket(
            envelope=_error(
                "no_crop_pairs",
                f"No exp-*/act-* crop pairs in {latest}. Re-run `pixel-mcp check`.",
            ),
            exit_code=EXIT_FATAL,
            images=[],
        )

    # Pair by region index. exp-r3.png pairs with act-r3.png; orphans are
    # dropped (we never want to show a half-pair to the reviewer).
    by_idx_exp = {_region_index(p): p for p in expected_paths}
    by_idx_act = {_region_index(p): p for p in actual_paths}
    common_idx = sorted(i for i in by_idx_exp if i in by_idx_act and i is not None)

    # Region metadata (selector, figma_node_id, severity, area) lives in the
    # last check envelope's ``data.regions``. Optional — when absent we still
    # emit the crop pairs with bare paths.
    region_meta = _load_region_metadata(sd)

    crop_pairs: list[dict[str, Any]] = []
    images: list[Any] = []
    for idx in common_idx:
        exp = by_idx_exp[idx]
        act = by_idx_act[idx]
        # Region metadata uses 0-based indexing in `regions[*]`; crop files
        # use 1-based (`exp-r1.png`). Convert when looking up.
        meta = region_meta[idx - 1] if 0 < idx <= len(region_meta) else {}
        crop_pairs.append(
            {
                "region_index": idx,
                "expected_path": str(exp),
                "actual_path": str(act),
                "selector": meta.get("leaf_selector"),
                "figma_node_id": meta.get("figma_node_id"),
                "severity": meta.get("severity"),
                "area_px2": meta.get("area_px2"),
            }
        )
        images.extend(_make_image_pair(exp, act))

    # Full-page overview, when the iter folder happens to have one.
    overview_expected: str | None = None
    overview_actual: str | None = None
    exp_overview = latest / "expected.png"
    act_overview = latest / "actual.png"
    if exp_overview.exists() and act_overview.exists():
        overview_expected = str(exp_overview)
        overview_actual = str(act_overview)
        images = _make_image_pair(exp_overview, act_overview) + images

    review_session_id = str(uuid.uuid4())
    data: dict[str, Any] = {
        "review_session_id": review_session_id,
        "iteration_dir": str(latest),
        "crop_pair_count": len(crop_pairs),
        "crop_pairs": crop_pairs,
        "expected_overview_path": overview_expected,
        "actual_overview_path": overview_actual,
        # Legacy v0 fields retained so existing tests / consumers keep working.
        "expected_crops": [str(p) for p in expected_paths],
        "actual_crops": [str(p) for p in actual_paths],
    }
    envelope = make_envelope(
        data=data,
        hints=[
            f"Level 3 review packet ready — {len(crop_pairs)} crop pair(s) from {latest.name}.",
            "Inspect the inline images above (expected = design, actual = render).",
            "Approve via `mcp__pixel_mcp__human_feedback(approve=true)`, or reject with "
            '`mcp__pixel_mcp__human_feedback(rejection_notes="...")`.',
        ],
        diagnostics={
            "iteration_dir": str(latest),
            "crop_pair_count": len(crop_pairs),
            "overview_available": overview_expected is not None,
        },
        next_suggested_action=(
            "Inspect the crops above. Call `mcp__pixel_mcp__human_feedback` with "
            'either approve=true or rejection_notes="..."'
        ),
        affordances=[
            {
                "tool": "mcp__pixel_mcp__human_feedback",
                "when": "to record the Level 3 verdict (approve OR rejection_notes)",
            },
            {
                "tool": "mcp__pixel_mcp__check",
                "when": "to consume the verdict (re-run with --enable-human-gate)",
            },
        ],
    )
    return ReviewPacket(envelope=envelope, exit_code=EXIT_READY_FOR_LEVEL_3, images=images)


def _region_index(path: Path) -> int | None:
    """Extract the 1-based region index from a crop filename (exp-r3.png → 3)."""
    m = _REGION_INDEX_RE.search(path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _load_region_metadata(sd: Path) -> list[dict[str, Any]]:
    """Best-effort load of the last ``check`` envelope's regions[].

    The CLI writes the envelope to wherever ``--out`` points, so there's no
    guaranteed on-disk path. We probe a handful of conventional locations.
    Returns an empty list when nothing is found — the review packet still
    emits crop pairs with bare paths.
    """
    candidates = [
        sd / "last-check.json",
        sd / "check.json",
        sd.parent / "last-check.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(raw, dict):
            data = raw.get("data")
            if isinstance(data, dict):
                regs = data.get("regions")
                if isinstance(regs, list):
                    return [r for r in regs if isinstance(r, dict)]
    return []


def _make_image_pair(expected: Path, actual: Path) -> list[Any]:
    """Build FastMCP Image attachments for a crop pair.

    Lazy import on FastMCP so non-MCP callers (CLI, tests that just inspect
    the envelope) don't pay the import cost. Failures fall back to an empty
    list — the envelope's path fields still let the reviewer open the files
    in their editor.
    """
    try:
        from mcp.server.fastmcp.utilities.types import Image as FMImage  # noqa: PLC0415
    except ImportError:
        return []
    out: list[Any] = []
    for p in (expected, actual):
        try:
            out.append(FMImage(path=p))
        except Exception:  # noqa: BLE001
            # Bad path / unreadable file — skip silently. The envelope still
            # carries the path so the reviewer can open it manually.
            continue
    return out


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
