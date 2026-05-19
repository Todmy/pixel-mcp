"""Smoke tests for the ``omniparser-detect`` Typer CLI verb.

The heavy compute is patched out — these tests cover argument parsing,
exit codes, and output formatting (human table vs ``--json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pixel_mcp_ml import omniparser_detect
from pixel_mcp_ml.cli import app
from pixel_mcp_ml.omniparser_detect import DetectedElement
from typer.testing import CliRunner


def _patch_detect(
    monkeypatch: pytest.MonkeyPatch,
    detections: list[DetectedElement] | None = None,
) -> None:
    """Stub the compute function with deterministic output."""
    detections = (
        detections
        if detections is not None
        else [
            DetectedElement(
                bbox=(10.0, 20.0, 100.0, 40.0),
                label="button",
                confidence=0.91,
            ),
            DetectedElement(
                bbox=(50.0, 200.0, 300.0, 20.0),
                label="text",
                confidence=0.72,
            ),
        ]
    )

    def fake_detect(image: Any, confidence_threshold: float = 0.3) -> list[DetectedElement]:
        return detections

    monkeypatch.setattr("pixel_mcp_ml.cli.detect_ui_elements", fake_detect)


def test_omniparser_detect_cli_outputs_human_table(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    _patch_detect(monkeypatch)
    image = tiny_image_factory("screen.png")

    result = runner.invoke(app, ["omniparser-detect", str(image)])

    assert result.exit_code == 0, result.output
    out = result.stdout
    assert "button" in out
    assert "text" in out
    assert "0.91" in out
    assert "0.72" in out
    # bbox values present in the human table.
    assert "10" in out and "20" in out


def test_omniparser_detect_cli_json_flag(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    _patch_detect(monkeypatch)
    image = tiny_image_factory("screen.png")

    result = runner.invoke(app, ["omniparser-detect", str(image), "--json"])

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["label"] == "button"
    assert parsed[0]["confidence"] == 0.91
    assert tuple(parsed[0]["bbox"]) == (10.0, 20.0, 100.0, 40.0)


def test_omniparser_detect_cli_missing_image_exits_one(
    tmp_path: Path,
    runner: CliRunner,
) -> None:
    missing = tmp_path / "no-such-image.png"
    result = runner.invoke(app, ["omniparser-detect", str(missing)])

    assert result.exit_code == 1
    assert "not found" in (result.stderr or result.output)


def test_omniparser_detect_cli_missing_dependency_exits_twelve(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    """OmniParserNotInstalledError -> exit 12 with install hint surfaced."""

    def raising(*args: Any, **kwargs: Any) -> list[DetectedElement]:
        raise omniparser_detect.OmniParserNotInstalledError("transformers")

    monkeypatch.setattr("pixel_mcp_ml.cli.detect_ui_elements", raising)

    image = tiny_image_factory("screen.png")
    result = runner.invoke(app, ["omniparser-detect", str(image)])

    assert result.exit_code == 12
    combined = (result.stderr or "") + (result.output or "")
    assert "--extra omniparser" in combined


def test_omniparser_detect_cli_no_detections_is_friendly(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    """Empty detection list prints a friendly message, not a blank screen."""
    _patch_detect(monkeypatch, detections=[])
    image = tiny_image_factory("screen.png")

    result = runner.invoke(app, ["omniparser-detect", str(image)])

    assert result.exit_code == 0
    assert "No UI elements" in result.stdout


def test_help_lists_omniparser_detect(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "omniparser-detect" in result.stdout
