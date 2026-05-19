"""CLI tests for `pixel-mcp judge`."""

from __future__ import annotations

import json
from pathlib import Path

from pixel_mcp.cli import app
from typer.testing import CliRunner


def _write_deltas(tmp: Path, raw: object) -> Path:
    p = tmp / "deltas.json"
    p.write_text(json.dumps(raw))
    return p


def test_judge_empty_array_exits_zero(tmp_path: Path) -> None:
    runner = CliRunner()
    p = _write_deltas(tmp_path, [])
    result = runner.invoke(app, ["judge", "--deltas", str(p)])
    assert result.exit_code == 0
    env = json.loads(result.output)
    assert env["data"]["converged"] is True


def test_judge_only_minor_exits_zero(tmp_path: Path) -> None:
    runner = CliRunner()
    p = _write_deltas(
        tmp_path,
        [{"selector": "#x", "property": "width", "severity": "minor"}],
    )
    result = runner.invoke(app, ["judge", "--deltas", str(p)])
    assert result.exit_code == 0
    env = json.loads(result.output)
    assert env["data"]["converged"] is True
    assert env["data"]["minor_count"] == 1


def test_judge_strict_minor_blocks(tmp_path: Path) -> None:
    runner = CliRunner()
    p = _write_deltas(
        tmp_path,
        [{"selector": "#x", "property": "width", "severity": "minor"}],
    )
    result = runner.invoke(app, ["judge", "--deltas", str(p), "--strict"])
    assert result.exit_code == 1
    env = json.loads(result.output)
    assert env["data"]["converged"] is False


def test_judge_critical_exits_one(tmp_path: Path) -> None:
    runner = CliRunner()
    p = _write_deltas(
        tmp_path,
        [{"selector": "#x", "property": "color", "severity": "critical"}],
    )
    result = runner.invoke(app, ["judge", "--deltas", str(p)])
    assert result.exit_code == 1


def test_judge_accepts_axi_envelope_shape(tmp_path: Path) -> None:
    """Pass through the full `pixel-mcp diff` envelope; data.deltas is found."""
    runner = CliRunner()
    p = _write_deltas(
        tmp_path,
        {"data": {"deltas": [{"selector": "#x", "property": "color", "severity": "critical"}]}},
    )
    result = runner.invoke(app, ["judge", "--deltas", str(p)])
    assert result.exit_code == 1


def test_judge_missing_file_exits_twelve(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["judge", "--deltas", str(tmp_path / "missing.json")])
    assert result.exit_code == 12


def test_judge_invalid_json_exits_twelve(tmp_path: Path) -> None:
    runner = CliRunner()
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    result = runner.invoke(app, ["judge", "--deltas", str(p)])
    assert result.exit_code == 12
    env = json.loads(result.output)
    assert env["diagnostics"]["error_type"] == "deltas_invalid_json"
