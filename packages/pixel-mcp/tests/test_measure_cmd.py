"""CLI integration tests for `pixel-mcp measure`.

These tests bypass Playwright by monkeypatching ``measure_render`` —
they exercise the envelope shape and error wiring, not the browser. The
browser path is covered by ``test_render_measurer.py``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pixel_mcp import measure_cmd as measure_cmd_mod
from pixel_mcp.cli import app
from pixel_mcp.render import (
    BoundingBox,
    ChromiumNotInstalledError,
    ComputedStyle,
    MeasuredDOM,
    MeasuredElement,
    PlaywrightNotInstalledError,
    RouteUnreachableError,
    WaitForTimeoutError,
)
from typer.testing import CliRunner


def _style() -> ComputedStyle:
    return ComputedStyle(
        color="#111111",
        background_color="#ffffff",
        font_family="Helvetica",
        font_size_px=16.0,
        font_weight=400,
        line_height="24px",
        letter_spacing="normal",
        padding_top=0,
        padding_right=0,
        padding_bottom=0,
        padding_left=0,
        margin_top=0,
        margin_right=0,
        margin_bottom=0,
        margin_left=0,
        border_radius=None,
        border_top_width=0,
        border_right_width=0,
        border_bottom_width=0,
        border_left_width=0,
    )


def _stub_dom(route: str = "http://localhost:3000/foo") -> MeasuredDOM:
    return MeasuredDOM(
        route=route,
        viewport=(1280, 720),
        measured_at=datetime.now(UTC),
        elements=[
            MeasuredElement(
                selector="button.cta",
                bounding_box=BoundingBox(x=10, y=20, w=120, h=40),
                computed_style=_style(),
                text_content="Get started",
                aria_role=None,
                parent_chain=["body"],
            )
        ],
    )


def _patch_measure_render(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: tuple[MeasuredDOM, bool] | None = None,
    raises: Exception | None = None,
) -> None:
    def fake(
        route: str,
        viewport: tuple[int, int] = (1280, 720),
        selectors: list[str] | None = None,
        wait_for: str | None = None,
        wait_for_network_idle: bool = True,
        timeout_ms: int = 15_000,
    ) -> tuple[MeasuredDOM, bool]:
        if raises is not None:
            raise raises
        if result is not None:
            return result
        return _stub_dom(route=route), False

    monkeypatch.setattr(measure_cmd_mod, "measure_render", fake)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_measure_help_lists_flags(runner: CliRunner) -> None:
    result = runner.invoke(app, ["measure", "--help"])
    assert result.exit_code == 0
    for flag in ("--route", "--selectors", "--viewport", "--wait-for", "--out"):
        assert flag in result.stdout, f"missing {flag}"


def test_measure_happy_path(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    _patch_measure_render(monkeypatch)
    result = runner.invoke(app, ["measure", "--route", "http://localhost:3000/foo"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert set(payload.keys()) == {
        "data",
        "hints",
        "diagnostics",
        "next_suggested_action",
        "affordances",
    }
    assert payload["data"]["route"] == "http://localhost:3000/foo"
    assert payload["data"]["viewport"] == [1280, 720]
    assert len(payload["data"]["elements"]) == 1
    # Affordances point at diff + check (Slice 4)
    tools = {a["tool"] for a in payload["affordances"]}
    assert "mcp__pixel_mcp__diff" in tools
    assert "mcp__pixel_mcp__check" in tools


def test_measure_route_unreachable(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    _patch_measure_render(monkeypatch, raises=RouteUnreachableError("connection refused"))
    result = runner.invoke(app, ["measure", "--route", "http://localhost:9999/foo"])
    assert result.exit_code == 12
    payload = json.loads(result.stdout)
    assert payload["data"] is None
    assert payload["diagnostics"]["error_type"] == "route_unreachable"
    assert any("dev server" in h for h in payload["hints"])


def test_measure_playwright_missing(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    _patch_measure_render(monkeypatch, raises=PlaywrightNotInstalledError("no playwright"))
    result = runner.invoke(app, ["measure", "--route", "http://localhost:3000/foo"])
    assert result.exit_code == 12
    payload = json.loads(result.stdout)
    assert payload["diagnostics"]["error_type"] == "playwright_not_installed"
    assert any("uv sync" in h for h in payload["hints"])


def test_measure_chromium_missing(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    _patch_measure_render(monkeypatch, raises=ChromiumNotInstalledError("no chromium"))
    result = runner.invoke(app, ["measure", "--route", "http://localhost:3000/foo"])
    assert result.exit_code == 12
    payload = json.loads(result.stdout)
    assert payload["diagnostics"]["error_type"] == "chromium_not_installed"
    assert any("playwright install chromium" in h for h in payload["hints"])


def test_measure_wait_for_timeout(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    _patch_measure_render(monkeypatch, raises=WaitForTimeoutError("selector never appeared"))
    result = runner.invoke(
        app,
        [
            "measure",
            "--route",
            "http://localhost:3000/foo",
            "--wait-for",
            ".never-shows",
        ],
    )
    assert result.exit_code == 12
    payload = json.loads(result.stdout)
    assert payload["diagnostics"]["error_type"] == "wait_for_timeout"


def test_measure_writes_to_file(
    monkeypatch: pytest.MonkeyPatch, runner: CliRunner, tmp_path: Path
) -> None:
    _patch_measure_render(monkeypatch)
    out_path = tmp_path / "measured.json"
    result = runner.invoke(
        app,
        [
            "measure",
            "--route",
            "http://localhost:3000/foo",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["data"]["route"] == "http://localhost:3000/foo"


def test_measure_viewport_flag_parsed(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    captured: dict[str, object] = {}

    def fake(
        route: str,
        viewport: tuple[int, int] = (1280, 720),
        selectors: list[str] | None = None,
        wait_for: str | None = None,
        wait_for_network_idle: bool = True,
        timeout_ms: int = 15_000,
    ) -> tuple[MeasuredDOM, bool]:
        captured["viewport"] = viewport
        captured["selectors"] = selectors
        return _stub_dom(route=route), False

    monkeypatch.setattr(measure_cmd_mod, "measure_render", fake)
    result = runner.invoke(
        app,
        [
            "measure",
            "--route",
            "http://localhost:3000/foo",
            "--viewport",
            "1920x1080",
            "--selectors",
            "button.cta, nav.top",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["viewport"] == (1920, 1080)
    assert captured["selectors"] == ["button.cta", "nav.top"]


def test_measure_viewport_invalid(runner: CliRunner) -> None:
    result = runner.invoke(
        app,
        [
            "measure",
            "--route",
            "http://localhost:3000/foo",
            "--viewport",
            "not-a-viewport",
        ],
    )
    assert result.exit_code != 0
    assert "viewport" in (result.output + (result.stderr or "")).lower()


def test_measure_truncated_hint(monkeypatch: pytest.MonkeyPatch, runner: CliRunner) -> None:
    _patch_measure_render(monkeypatch, result=(_stub_dom(), True))
    result = runner.invoke(app, ["measure", "--route", "http://localhost:3000/foo"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert any("200-element cap" in h or "200" in h for h in payload["hints"])
    assert payload["diagnostics"]["truncated"] is True
