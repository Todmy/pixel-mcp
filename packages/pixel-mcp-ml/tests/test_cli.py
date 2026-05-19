"""Smoke tests for the `pixel-mcp-ml` Typer CLI.

The heavy compute is patched out via ``monkeypatch`` — these tests are
about argument parsing, exit codes, and output formatting, not real
DINOv2 inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pixel_mcp_ml import dinov2_compare
from pixel_mcp_ml.cli import app
from typer.testing import CliRunner


def _patch_compute(monkeypatch: pytest.MonkeyPatch, value: float = 0.97) -> None:
    """Replace the heavy compute function with a deterministic stub and
    seed the model cache so the JSON ``device`` field has a value."""

    def fake_compute(image_a: Any, image_b: Any, model_size: str = "small") -> float:
        return value

    monkeypatch.setattr(
        "pixel_mcp_ml.cli.compute_dinov2_similarity",
        fake_compute,
    )
    dinov2_compare._MODEL_CACHE["small"] = (None, None, "cpu")  # type: ignore[assignment]


def test_dinov2_compare_cli_outputs_similarity(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    _patch_compute(monkeypatch, value=0.9712)
    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")

    result = runner.invoke(app, ["dinov2-compare", str(a), str(b)])

    assert result.exit_code == 0, result.output
    assert "Similarity:" in result.stdout
    assert "0.9712" in result.stdout


def test_dinov2_compare_cli_json_flag(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    _patch_compute(monkeypatch, value=0.5)
    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")

    result = runner.invoke(app, ["dinov2-compare", str(a), str(b), "--json"])

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed == {"similarity": 0.5, "model_size": "small", "device": "cpu"}


def test_dinov2_compare_cli_missing_image_exits_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    _patch_compute(monkeypatch)
    real = tiny_image_factory("real.png")
    missing = tmp_path / "does-not-exist.png"

    result = runner.invoke(app, ["dinov2-compare", str(real), str(missing)])

    assert result.exit_code == 1
    assert "not found" in (result.stderr or result.output)


def test_dinov2_compare_cli_missing_dependency_exits_twelve(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    """When the underlying compute raises ``DINOv2NotInstalledError``,
    the CLI should exit with code 12 and print the install hint."""

    def raising(*args: Any, **kwargs: Any) -> float:
        raise dinov2_compare.DINOv2NotInstalledError("transformers")

    monkeypatch.setattr("pixel_mcp_ml.cli.compute_dinov2_similarity", raising)

    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")
    result = runner.invoke(app, ["dinov2-compare", str(a), str(b)])

    assert result.exit_code == 12
    combined = (result.stderr or "") + (result.output or "")
    assert "--extra dinov2" in combined


def test_dinov2_compare_cli_invalid_model_size_exits_two(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    _patch_compute(monkeypatch)
    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")

    result = runner.invoke(app, ["dinov2-compare", str(a), str(b), "--model-size", "huge"])

    assert result.exit_code == 2


def test_version_flag(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip()  # non-empty


def test_help_lists_dinov2_compare(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "dinov2-compare" in result.stdout
