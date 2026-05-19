"""Level 3 review command — packet emission with crop pairs and overview."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pixel_mcp.review_cmd import (
    EXIT_FATAL,
    EXIT_READY_FOR_LEVEL_3,
    build_packet,
    run,
)


def _make_iter_dir(sd: Path, iter_n: int, region_count: int) -> Path:
    """Create ``crops/iter-N/`` populated with ``region_count`` paired crops."""
    iter_dir = sd / "crops" / f"iter-{iter_n}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, region_count + 1):
        (iter_dir / f"exp-r{i}.png").write_bytes(b"\x89PNG\r\n\x1a\nfake-exp")
        (iter_dir / f"act-r{i}.png").write_bytes(b"\x89PNG\r\n\x1a\nfake-act")
    return iter_dir


def test_review_emits_envelope_with_crop_pairs(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    _make_iter_dir(sd, iter_n=1, region_count=3)

    with patch("pixel_mcp.review_cmd.state_dir", return_value=sd):
        envelope, exit_code = run()

    assert exit_code == EXIT_READY_FOR_LEVEL_3
    assert envelope["data"]["crop_pair_count"] == 3
    pairs = envelope["data"]["crop_pairs"]
    assert len(pairs) == 3
    indexes = sorted(p["region_index"] for p in pairs)
    assert indexes == [1, 2, 3]
    for p in pairs:
        assert Path(p["expected_path"]).exists()
        assert Path(p["actual_path"]).exists()
    # Envelope must carry a unique session id per review request.
    assert envelope["data"]["review_session_id"]
    # Affordances point at the human_feedback tool.
    tools = [a["tool"] for a in envelope["affordances"]]
    assert "mcp__pixel_mcp__human_feedback" in tools


def test_review_no_crops_returns_fatal(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    sd.mkdir(parents=True)
    # crops dir absent → fatal
    with patch("pixel_mcp.review_cmd.state_dir", return_value=sd):
        envelope, exit_code = run()
    assert exit_code == EXIT_FATAL
    assert envelope["data"] is None
    assert envelope["diagnostics"]["error_type"] == "no_check_run_yet"


def test_review_picks_latest_iteration(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    _make_iter_dir(sd, iter_n=1, region_count=1)
    _make_iter_dir(sd, iter_n=2, region_count=2)
    _make_iter_dir(sd, iter_n=10, region_count=4)  # numeric, not lexical, ordering

    with patch("pixel_mcp.review_cmd.state_dir", return_value=sd):
        envelope, exit_code = run()

    assert exit_code == EXIT_READY_FOR_LEVEL_3
    # 10 > 2 > 1 → iter-10 picked
    assert envelope["data"]["iteration_dir"].endswith("iter-10")
    assert envelope["data"]["crop_pair_count"] == 4


def test_review_includes_overview_when_present(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    iter_dir = _make_iter_dir(sd, iter_n=1, region_count=1)
    # Drop overview pair next to the crops.
    (iter_dir / "expected.png").write_bytes(b"\x89PNG-exp-overview")
    (iter_dir / "actual.png").write_bytes(b"\x89PNG-act-overview")

    with patch("pixel_mcp.review_cmd.state_dir", return_value=sd):
        envelope, exit_code = run()

    assert exit_code == EXIT_READY_FOR_LEVEL_3
    assert envelope["data"]["expected_overview_path"] is not None
    assert envelope["data"]["actual_overview_path"] is not None
    assert envelope["data"]["expected_overview_path"].endswith("expected.png")
    assert envelope["data"]["actual_overview_path"].endswith("actual.png")
    assert envelope["diagnostics"]["overview_available"] is True


def test_review_drops_orphan_crops(tmp_path: Path) -> None:
    """exp-r5 without a matching act-r5 must NOT show up in crop_pairs."""
    sd = tmp_path / ".pixel-mcp"
    iter_dir = _make_iter_dir(sd, iter_n=1, region_count=2)
    # Orphan expected crop with no actual counterpart.
    (iter_dir / "exp-r5.png").write_bytes(b"\x89PNG-orphan")

    with patch("pixel_mcp.review_cmd.state_dir", return_value=sd):
        envelope, _exit = run()

    indexes = sorted(p["region_index"] for p in envelope["data"]["crop_pairs"])
    assert indexes == [1, 2]


def test_build_packet_returns_image_attachments(tmp_path: Path) -> None:
    """MCP path must return FastMCP Image objects ready to render inline."""
    sd = tmp_path / ".pixel-mcp"
    _make_iter_dir(sd, iter_n=1, region_count=2)

    with patch("pixel_mcp.review_cmd.state_dir", return_value=sd):
        packet = build_packet()

    assert packet.exit_code == EXIT_READY_FOR_LEVEL_3
    # 2 pairs → 4 images (expected + actual per pair). No overview here.
    assert len(packet.images) == 4
    # Each image must expose the FastMCP Image API (path or data attribute).
    for img in packet.images:
        assert hasattr(img, "path") or hasattr(img, "data")


def test_review_envelope_metadata_keys_present(tmp_path: Path) -> None:
    """Crop pair payload must expose selector/figma_node_id/severity/area keys."""
    sd = tmp_path / ".pixel-mcp"
    _make_iter_dir(sd, iter_n=1, region_count=1)
    # Optional metadata file the v0 review_cmd doesn't write; absence is OK
    # — keys must still be present in the pair dict (as None).
    with patch("pixel_mcp.review_cmd.state_dir", return_value=sd):
        envelope, _exit = run()
    pair = envelope["data"]["crop_pairs"][0]
    for key in (
        "region_index",
        "expected_path",
        "actual_path",
        "selector",
        "figma_node_id",
        "severity",
        "area_px2",
    ):
        assert key in pair, f"missing key {key}"


def test_review_loads_region_metadata_when_last_check_json_present(
    tmp_path: Path,
) -> None:
    """When .pixel-mcp/last-check.json is on disk, selector/figma_node_id flow through."""
    sd = tmp_path / ".pixel-mcp"
    _make_iter_dir(sd, iter_n=1, region_count=2)
    # Stash the regions array as if a previous `check` had written it.
    last_check = {
        "data": {
            "regions": [
                {
                    "leaf_selector": "#hero",
                    "figma_node_id": "1:42",
                    "severity": "critical",
                    "area_px2": 12345.0,
                },
                {
                    "leaf_selector": ".cta",
                    "figma_node_id": None,
                    "severity": "minor",
                    "area_px2": 250.0,
                },
            ]
        }
    }
    (sd / "last-check.json").write_text(json.dumps(last_check))

    with patch("pixel_mcp.review_cmd.state_dir", return_value=sd):
        envelope, _exit = run()

    pairs = sorted(envelope["data"]["crop_pairs"], key=lambda p: p["region_index"])
    assert pairs[0]["selector"] == "#hero"
    assert pairs[0]["figma_node_id"] == "1:42"
    assert pairs[0]["severity"] == "critical"
    assert pairs[1]["selector"] == ".cta"
    assert pairs[1]["severity"] == "minor"


@pytest.fixture
def _patch_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Convenience fixture: redirect state_dir to a tmp dir."""
    sd = tmp_path / ".pixel-mcp"
    monkeypatch.setattr("pixel_mcp.review_cmd.state_dir", lambda *_a, **_kw: sd)
    return sd
