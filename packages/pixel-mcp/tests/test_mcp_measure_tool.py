"""MCP integration test for the `measure` tool.

Same pattern as Slice 2's ``test_mcp_spec_tool``: call the registered
FastMCP tool function in-process with ``measure_render`` patched so we
don't actually launch a browser.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pixel_mcp import measure_cmd as measure_cmd_mod
from pixel_mcp.render import (
    BoundingBox,
    ComputedStyle,
    MeasuredDOM,
    MeasuredElement,
    RouteUnreachableError,
)

pytestmark = pytest.mark.smoke


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


def _stub_dom() -> MeasuredDOM:
    return MeasuredDOM(
        route="http://localhost:3000/foo",
        viewport=(1280, 720),
        measured_at=datetime.now(UTC),
        elements=[
            MeasuredElement(
                selector="button.cta",
                bounding_box=BoundingBox(x=0, y=0, w=120, h=40),
                computed_style=_style(),
                text_content="Get started",
            )
        ],
    )


def test_mcp_server_registers_measure_tool() -> None:
    from pixel_mcp import mcp_server

    assert callable(mcp_server.measure)


def test_mcp_measure_tool_returns_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(
        route: str,
        viewport: tuple[int, int] = (1280, 720),
        selectors: list[str] | None = None,
        wait_for: str | None = None,
        wait_for_network_idle: bool = True,
        timeout_ms: int = 15_000,
    ) -> tuple[MeasuredDOM, bool]:
        return _stub_dom(), False

    monkeypatch.setattr(measure_cmd_mod, "measure_render", fake)
    from pixel_mcp.mcp_server import measure as mcp_measure

    env = mcp_measure(route="http://localhost:3000/foo")
    assert set(env.keys()) == {
        "data",
        "hints",
        "diagnostics",
        "next_suggested_action",
        "affordances",
    }
    assert env["data"]["route"] == "http://localhost:3000/foo"
    tools = {a["tool"] for a in env["affordances"]}
    assert "mcp__pixel_mcp__diff" in tools
    assert "mcp__pixel_mcp__check" in tools


def test_mcp_measure_tool_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(
        route: str,
        viewport: tuple[int, int] = (1280, 720),
        selectors: list[str] | None = None,
        wait_for: str | None = None,
        wait_for_network_idle: bool = True,
        timeout_ms: int = 15_000,
    ) -> tuple[MeasuredDOM, bool]:
        raise RouteUnreachableError("connection refused")

    monkeypatch.setattr(measure_cmd_mod, "measure_render", fake)
    from pixel_mcp.mcp_server import measure as mcp_measure

    env = mcp_measure(route="http://localhost:9999/foo")
    assert env["data"] is None
    assert env["diagnostics"]["error_type"] == "route_unreachable"


def test_mcp_measure_tool_accepts_viewport_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
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
        return _stub_dom(), False

    monkeypatch.setattr(measure_cmd_mod, "measure_render", fake)
    from pixel_mcp.mcp_server import measure as mcp_measure

    mcp_measure(
        route="http://localhost:3000/foo",
        viewport_width=1920,
        viewport_height=1080,
        selectors=["button.cta"],
    )
    assert captured["viewport"] == (1920, 1080)
    assert captured["selectors"] == ["button.cta"]
