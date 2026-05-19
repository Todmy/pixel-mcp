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


def test_stub_subcommand_exits_nonzero(runner: CliRunner) -> None:
    # spec/measure/diff/judge/check (Slices 2-4) and mapping (Slice 8) are now
    # real. `snapshot` is the next stub (Slice 10).
    result = runner.invoke(app, ["snapshot"])
    assert result.exit_code != 0
    assert "Todmy/PBaaS#20" in (result.stderr or result.output)
