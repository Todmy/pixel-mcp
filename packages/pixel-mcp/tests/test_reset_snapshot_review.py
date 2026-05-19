"""CLI tests for reset / snapshot / review subcommands."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from pixel_mcp.cli import app
from pixel_mcp.render import (
    BoundingBox,
    ComputedStyle,
    MeasuredDOM,
    MeasuredElement,
)
from typer.testing import CliRunner


def _fake_dom() -> MeasuredDOM:
    return MeasuredDOM(
        route="http://localhost:3000/",
        viewport=(1280, 720),
        measured_at=datetime.now(UTC),
        elements=[
            MeasuredElement(
                selector="#x",
                bounding_box=BoundingBox(x=0, y=0, w=10, h=10),
                computed_style=ComputedStyle(
                    color="#000000",
                    background_color="#ffffff",
                    font_family="Inter",
                    font_size_px=16,
                    font_weight=400,
                    padding_top=0,
                    padding_right=0,
                    padding_bottom=0,
                    padding_left=0,
                    margin_top=0,
                    margin_right=0,
                    margin_bottom=0,
                    margin_left=0,
                    border_top_width=0,
                    border_right_width=0,
                    border_bottom_width=0,
                    border_left_width=0,
                ),
            )
        ],
    )


def test_reset_subcommand_writes_envelope(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "envelope.json"
    with patch("pixel_mcp.reset_cmd.reset_state"):
        result = runner.invoke(app, ["reset", "--out", str(out)])
    assert result.exit_code == 0
    envelope = json.loads(out.read_text())
    assert envelope["data"]["cleared_snapshots"] is False


def test_snapshot_writes_artifacts(tmp_path: Path, monkeypatch: object) -> None:
    runner = CliRunner()
    with (
        patch("pixel_mcp.snapshot_cmd.measure_render", return_value=(_fake_dom(), False)),
        patch("pixel_mcp.snapshot_cmd.capture_screenshot", return_value=b"\x89PNG\r\n\x1a\nfake"),
        patch("pixel_mcp.snapshot_cmd.state_dir", return_value=tmp_path / ".pixel-mcp"),
    ):
        result = runner.invoke(
            app,
            ["snapshot", "--route", "http://localhost:3000/", "--tag", "baseline"],
        )
    assert result.exit_code == 0
    # Verify artifacts written
    snapdir = tmp_path / ".pixel-mcp" / "snapshots" / "baseline"
    assert (snapdir / "measured.json").exists()
    assert (snapdir / "screenshot.png").exists()
    assert (snapdir / "metadata.json").exists()


def test_review_errors_when_no_crops(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch("pixel_mcp.review_cmd.state_dir", return_value=tmp_path / ".pixel-mcp"):
        (tmp_path / ".pixel-mcp").mkdir()
        result = runner.invoke(app, ["review"])
    assert result.exit_code == 12  # EXIT_FATAL when no crops


def test_review_emits_envelope_with_crops(tmp_path: Path) -> None:
    runner = CliRunner()
    sd = tmp_path / ".pixel-mcp"
    crops = sd / "crops" / "iter-1"
    crops.mkdir(parents=True)
    (crops / "exp-r1.png").write_bytes(b"\x89PNG")
    (crops / "act-r1.png").write_bytes(b"\x89PNG")

    with patch("pixel_mcp.review_cmd.state_dir", return_value=sd):
        result = runner.invoke(app, ["review"])
    assert result.exit_code == 2  # EXIT_READY_FOR_LEVEL_3
    envelope = json.loads(result.output)
    assert envelope["data"]["crop_pair_count"] == 1
