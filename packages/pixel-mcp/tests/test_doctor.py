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
        "pixel_mcp_ml_vlm",
        "ollama_qwen",
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


def test_pixel_mcp_ml_vlm_check_green_when_anthropic_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package found AND anthropic importable -> green for the VLM extra."""
    import importlib.util

    real_find = importlib.util.find_spec

    def fake_find(name: str, *args: object, **kwargs: object) -> object:
        if name in ("pixel_mcp_ml", "anthropic"):
            return object()
        return real_find(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find)
    result = doctor_mod._check_pixel_mcp_ml_vlm()
    assert result["status"] == "green"


def test_pixel_mcp_ml_vlm_check_amber_when_anthropic_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package present, anthropic missing -> amber + name in detail."""
    import importlib.util

    real_find = importlib.util.find_spec

    def fake_find(name: str, *args: object, **kwargs: object) -> object:
        if name == "pixel_mcp_ml":
            return object()
        if name == "anthropic":
            return None
        return real_find(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find)
    result = doctor_mod._check_pixel_mcp_ml_vlm()
    assert result["status"] == "amber"
    assert "anthropic" in result["detail"]


def test_pixel_mcp_ml_vlm_check_red_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Package absent -> red for the VLM extra too."""
    import importlib.util

    real_find = importlib.util.find_spec

    def fake_find(name: str, *args: object, **kwargs: object) -> object:
        if name == "pixel_mcp_ml":
            return None
        return real_find(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find)
    result = doctor_mod._check_pixel_mcp_ml_vlm()
    assert result["status"] == "red"


def test_pixel_mcp_ml_extras_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dinov2 extra and the vlm extra report independently.

    Scenario: package present, transformers/torch present (dinov2 green),
    anthropic absent (vlm amber). Both rows must reflect the truth.
    """
    import importlib.util

    real_find = importlib.util.find_spec

    def fake_find(name: str, *args: object, **kwargs: object) -> object:
        if name in ("pixel_mcp_ml", "transformers", "torch"):
            return object()
        if name == "anthropic":
            return None
        return real_find(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", fake_find)
    assert doctor_mod._check_pixel_mcp_ml()["status"] == "green"
    assert doctor_mod._check_pixel_mcp_ml_vlm()["status"] == "amber"


def test_ollama_qwen_check_green_when_model_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Daemon reachable + qwen2.5vl tag listed → green."""
    import httpx as httpx_mod

    fake_response = pytest.MonkeyPatch  # placeholder
    fake_response = type(
        "Resp",
        (),
        {
            "status_code": 200,
            "json": lambda self: {"models": [{"name": "qwen2.5vl:7b"}]},
        },
    )()

    def fake_get(url: str, timeout: float = 1.0) -> object:
        assert url.endswith("/api/tags")
        return fake_response

    monkeypatch.setattr(httpx_mod, "get", fake_get)
    result = doctor_mod._check_ollama_qwen()
    assert result["status"] == "green"
    assert "qwen2.5vl" in result["detail"]


def test_ollama_qwen_check_amber_when_model_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Daemon reachable but no qwen2.5vl tag → amber."""
    import httpx as httpx_mod

    fake_response = type(
        "Resp",
        (),
        {
            "status_code": 200,
            "json": lambda self: {"models": [{"name": "llama3:8b"}]},
        },
    )()

    monkeypatch.setattr(httpx_mod, "get", lambda url, timeout=1.0: fake_response)
    result = doctor_mod._check_ollama_qwen()
    assert result["status"] == "amber"
    assert "missing" in result["detail"]


def test_ollama_qwen_check_red_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ConnectError → red with diagnostic name."""
    import httpx as httpx_mod

    def raising(url: str, timeout: float = 1.0) -> object:
        raise httpx_mod.ConnectError("Connection refused")

    monkeypatch.setattr(httpx_mod, "get", raising)
    result = doctor_mod._check_ollama_qwen()
    assert result["status"] == "red"
    assert "unreachable" in result["detail"]


def test_ollama_qwen_red_does_not_flip_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional ollama_qwen red must not break the overall doctor exit code.

    The Qwen backend is an opt-in alternative to the Claude default —
    most operators never run Ollama, so doctor must stay exit 0 in that
    case (assuming every other check is green/amber).
    """
    import httpx as httpx_mod

    def raising(url: str, timeout: float = 1.0) -> object:
        raise httpx_mod.ConnectError("Connection refused")

    monkeypatch.setattr(httpx_mod, "get", raising)
    env = doctor_mod.build_envelope()
    # The only red in this scenario should be ollama_qwen — exit code 0.
    reds = [c["name"] for c in env["data"]["checks"] if c["status"] == "red"]
    assert "ollama_qwen" in reds
    # ``figma_api_reachable`` may also flip to amber, never red here.
    non_optional_reds = [n for n in reds if n != "ollama_qwen"]
    if not non_optional_reds:
        assert doctor_mod.exit_code_for(env) == 0


def test_ollama_qwen_respects_ollama_host_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """OLLAMA_HOST env overrides the probed endpoint."""
    import httpx as httpx_mod

    captured: dict[str, str] = {}

    def fake_get(url: str, timeout: float = 1.0) -> object:
        captured["url"] = url
        raise httpx_mod.ConnectError("nope")

    monkeypatch.setenv("OLLAMA_HOST", "http://elsewhere:7777")
    monkeypatch.setattr(httpx_mod, "get", fake_get)
    doctor_mod._check_ollama_qwen()
    assert captured["url"] == "http://elsewhere:7777/api/tags"


def test_no_stub_subcommands_remain(runner: CliRunner) -> None:
    """All v0 subcommands implemented — invoking them without args should
    yield a Typer 'missing argument' error (exit 2), not the stub message."""
    result = runner.invoke(app, ["snapshot"])
    assert result.exit_code != 0
    assert "Todmy/PBaaS" not in (result.stderr or result.output)
