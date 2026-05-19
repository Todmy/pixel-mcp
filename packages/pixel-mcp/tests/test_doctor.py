"""Smoke + envelope tests for the doctor subcommand."""

from __future__ import annotations

import json

import pytest
from pixel_mcp import doctor as doctor_mod
from pixel_mcp.cli import app
from typer.testing import CliRunner

pytestmark = pytest.mark.smoke


def test_doctor_envelope_has_required_fields() -> None:
    env = doctor_mod.build_envelope()
    assert set(env.keys()) == {
        "data",
        "hints",
        "diagnostics",
        "next_suggested_action",
        "affordances",
    }
    assert "checks" in env["data"]
    assert "summary" in env["data"]


def test_doctor_envelope_includes_python_check() -> None:
    env = doctor_mod.build_envelope()
    names = {c["name"] for c in env["data"]["checks"]}
    assert {
        "python_version",
        "playwright",
        "chromium",
        "figma_token",
        "httpx",
        "figma_api_reachable",
        "pixel_mcp_ml",
        "uv",
    } <= names


def test_doctor_python_check_green_on_supported_runtime() -> None:
    # Tests run on Python >= 3.11 (project requirement); this should be green.
    env = doctor_mod.build_envelope()
    py = next(c for c in env["data"]["checks"] if c["name"] == "python_version")
    assert py["status"] == "green"


def test_doctor_exit_code_zero_when_no_red(runner: CliRunner) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output


def test_doctor_json_output_parses(runner: CliRunner) -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.stdout)
    assert set(parsed.keys()) == {
        "data",
        "hints",
        "diagnostics",
        "next_suggested_action",
        "affordances",
    }


def test_version_flag(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip()  # non-empty


def test_help_lists_subcommands(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for verb in [
        "doctor",
        "spec",
        "measure",
        "diff",
        "judge",
        "check",
        "review",
        "mapping",
        "snapshot",
        "reset",
        "mcp",
    ]:
        assert verb in result.stdout, f"missing {verb} in help output"


def test_pixel_mcp_ml_check_green_when_full_stack_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three specs found -> green status."""
    import importlib.util

    real_find = importlib.util.find_spec

    def fake_find(name: str, *args: object, **kwargs: object) -> object:
        if name in ("pixel_mcp_ml", "transformers", "torch"):
            return object()  # truthy non-None — pretend module exists
        return real_find(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find)
    result = doctor_mod._check_pixel_mcp_ml()
    assert result["status"] == "green"


def test_pixel_mcp_ml_check_amber_when_backend_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package present, ML backend missing -> amber + name in detail."""
    import importlib.util

    real_find = importlib.util.find_spec

    def fake_find(name: str, *args: object, **kwargs: object) -> object:
        if name == "pixel_mcp_ml":
            return object()
        if name in ("transformers", "torch"):
            return None
        return real_find(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find)
    result = doctor_mod._check_pixel_mcp_ml()
    assert result["status"] == "amber"
    assert "transformers" in result["detail"] or "torch" in result["detail"]


def test_pixel_mcp_ml_check_red_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package absent -> red, no further probing happens."""
    import importlib.util

    real_find = importlib.util.find_spec

    def fake_find(name: str, *args: object, **kwargs: object) -> object:
        if name == "pixel_mcp_ml":
            return None
        return real_find(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find)
    result = doctor_mod._check_pixel_mcp_ml()
    assert result["status"] == "red"


def test_no_stub_subcommands_remain(runner: CliRunner) -> None:
    """All v0 subcommands implemented — invoking them without args should
    yield a Typer 'missing argument' error (exit 2), not the stub message."""
    result = runner.invoke(app, ["snapshot"])
    assert result.exit_code != 0
    assert "Todmy/PBaaS" not in (result.stderr or result.output)
