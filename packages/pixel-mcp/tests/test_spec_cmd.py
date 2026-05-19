"""CLI integration tests for `pixel-mcp spec`."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pixel_mcp.cli import app
from pixel_mcp.figma_client import FigmaClient
from typer.testing import CliRunner

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "figma"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


@pytest.fixture(autouse=True)
def _cwd_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def _patch_transport(monkeypatch: pytest.MonkeyPatch, fixture: dict) -> None:
    real_init = FigmaClient.__init__

    def patched_init(self: FigmaClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = httpx.MockTransport(lambda req: httpx.Response(200, json=fixture))
        kwargs.setdefault("token", "test_token")
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(FigmaClient, "__init__", patched_init)


def test_spec_happy_path(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    _patch_transport(monkeypatch, _load("fixture_frame_response.json"))
    result = runner.invoke(
        app,
        ["spec", "--figma", "https://www.figma.com/file/AbC/p?node-id=123-456"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload.keys()) == {
        "data",
        "hints",
        "diagnostics",
        "next_suggested_action",
        "affordances",
    }
    assert payload["data"]["figma_node_id"] == "123:456"
    assert payload["data"]["figma_node_type"] == "FRAME"


def test_spec_missing_token_returns_axi_error(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner
) -> None:
    monkeypatch.delenv("FIGMA_TOKEN", raising=False)
    result = runner.invoke(
        app,
        ["spec", "--figma", "https://www.figma.com/file/AbC/p?node-id=1-2"],
    )
    assert result.exit_code == 12
    payload = json.loads(result.stdout)
    assert payload["data"] is None
    assert payload["diagnostics"]["error_type"] == "figma_auth_error"
    assert any("FIGMA_TOKEN" in h for h in payload["hints"])


def test_spec_unsupported_node_type(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    _patch_transport(monkeypatch, _load("fixture_group_response.json"))
    result = runner.invoke(
        app,
        ["spec", "--figma", "https://www.figma.com/design/GFile/g?node-id=99-1"],
    )
    assert result.exit_code == 12
    payload = json.loads(result.stdout)
    assert payload["diagnostics"]["error_type"] == "unsupported_node_type"
    assert any("Supported Figma node types" in h for h in payload["hints"])


def test_spec_malformed_url(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    monkeypatch.setenv("FIGMA_TOKEN", "t")
    result = runner.invoke(
        app,
        ["spec", "--figma", "https://www.figma.com/file/abc"],
    )
    assert result.exit_code == 12
    payload = json.loads(result.stdout)
    assert payload["diagnostics"]["error_type"] == "figma_url_error"


def test_spec_writes_to_file(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    _patch_transport(monkeypatch, _load("fixture_frame_response.json"))
    out_path = tmp_path / "spec.json"
    result = runner.invoke(
        app,
        [
            "spec",
            "--figma",
            "https://www.figma.com/file/AbC/p?node-id=123-456",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["data"]["name"] == "PrimaryButton"


def test_spec_help_lists_flags(runner: CliRunner) -> None:
    result = runner.invoke(app, ["spec", "--help"])
    assert result.exit_code == 0
    assert "--figma" in result.stdout
    assert "--out" in result.stdout
    assert "--refresh-spec" in result.stdout
