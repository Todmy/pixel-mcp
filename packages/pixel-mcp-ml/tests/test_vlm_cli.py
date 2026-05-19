"""Smoke tests for ``pixel-mcp-ml vlm-verify``.

The heavy compute (the actual VLM call) is patched out — these tests
focus on argument parsing, exit codes, and output formatting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pixel_mcp_ml import vlm_verify
from pixel_mcp_ml.cli import app
from pixel_mcp_ml.vlm_verify import VLMJudgment, VLMNotInstalledError, VLMOllamaError
from typer.testing import CliRunner


def _patch_compute(
    monkeypatch: pytest.MonkeyPatch,
    verdict: str = "match",
    confidence: float = 0.92,
    reasoning: str = "Both crops show the same content.",
) -> None:
    def fake_compute(
        image_a: Any,
        image_b: Any,
        backend: str = "claude",
        model: str | None = None,
    ) -> VLMJudgment:
        return VLMJudgment(
            verdict=verdict,  # type: ignore[arg-type]
            confidence=confidence,
            reasoning=reasoning,
        )

    monkeypatch.setattr("pixel_mcp_ml.cli.compute_vlm_judgment", fake_compute)


def test_vlm_verify_cli_outputs_human_verdict(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    _patch_compute(monkeypatch, verdict="match", confidence=0.94)
    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")

    result = runner.invoke(app, ["vlm-verify", str(a), str(b)])

    assert result.exit_code == 0, result.output
    assert "Verdict:" in result.stdout
    assert "match" in result.stdout
    assert "0.94" in result.stdout


def test_vlm_verify_cli_json_flag(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    _patch_compute(monkeypatch, verdict="ambiguous", confidence=0.5, reasoning="unclear")
    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")

    result = runner.invoke(app, ["vlm-verify", str(a), str(b), "--json"])

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert parsed == {
        "verdict": "ambiguous",
        "confidence": 0.5,
        "reasoning": "unclear",
    }


def test_vlm_verify_cli_missing_image_exits_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    _patch_compute(monkeypatch)
    real = tiny_image_factory("real.png")
    missing = tmp_path / "does-not-exist.png"

    result = runner.invoke(app, ["vlm-verify", str(real), str(missing)])

    assert result.exit_code == 1
    assert "not found" in (result.stderr or result.output)


def test_vlm_verify_cli_missing_dependency_exits_twelve(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    def raising(*args: Any, **kwargs: Any) -> VLMJudgment:
        raise VLMNotInstalledError("anthropic")

    monkeypatch.setattr("pixel_mcp_ml.cli.compute_vlm_judgment", raising)

    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")
    result = runner.invoke(app, ["vlm-verify", str(a), str(b)])

    assert result.exit_code == 12
    combined = (result.stderr or "") + (result.output or "")
    assert "--extra vlm" in combined


def test_vlm_verify_cli_qwen_ollama_unreachable_exits_twelve(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    """qwen-local backend with no Ollama running → exit 12 + actionable hint."""

    def raising(*args: Any, **kwargs: Any) -> VLMJudgment:
        raise VLMOllamaError("connection refused", host="http://localhost:11434")

    monkeypatch.setattr("pixel_mcp_ml.cli.compute_vlm_judgment", raising)

    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")
    result = runner.invoke(app, ["vlm-verify", str(a), str(b), "--backend", "qwen-local"])

    assert result.exit_code == 12
    combined = (result.stderr or "") + (result.output or "")
    assert "ollama serve" in combined
    assert "qwen2.5vl" in combined


def test_vlm_verify_cli_invalid_backend_exits_two(
    monkeypatch: pytest.MonkeyPatch,
    tiny_image_factory: Any,
    runner: CliRunner,
) -> None:
    _patch_compute(monkeypatch)
    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")

    result = runner.invoke(app, ["vlm-verify", str(a), str(b), "--backend", "gpt-5"])

    assert result.exit_code == 2


def test_help_lists_vlm_verify(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "vlm-verify" in result.stdout


def test_vlm_module_constant_set() -> None:
    """Sanity — DEFAULT_CLAUDE_MODEL is a non-empty string."""
    assert isinstance(vlm_verify.DEFAULT_CLAUDE_MODEL, str)
    assert vlm_verify.DEFAULT_CLAUDE_MODEL
