"""MCP integration test for the `spec` tool.

Slice 1's `test_mcp_smoke.py` validates the doctor tool by calling the
registered function directly. Slice 2 follows the same pattern — calling
the registered FastMCP tool function in-process — plus a check that the
MCP server module exposes the spec tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pixel_mcp.figma_client import FigmaClient

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "figma"

pytestmark = pytest.mark.smoke


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


def _patch_transport(monkeypatch: pytest.MonkeyPatch, fixture: dict) -> None:
    real_init = FigmaClient.__init__

    def patched_init(self: FigmaClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = httpx.MockTransport(lambda req: httpx.Response(200, json=fixture))
        kwargs.setdefault("token", "test_token")
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(FigmaClient, "__init__", patched_init)


def test_mcp_server_registers_spec_tool() -> None:
    from pixel_mcp import mcp_server

    assert callable(mcp_server.spec)


def test_mcp_spec_tool_returns_envelope(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    _patch_transport(monkeypatch, _load("fixture_frame_response.json"))
    from pixel_mcp.mcp_server import spec as mcp_spec

    env = mcp_spec(
        figma_url="https://www.figma.com/file/AbC/p?node-id=123-456",
        refresh=False,
    )
    assert set(env.keys()) == {
        "data",
        "hints",
        "diagnostics",
        "next_suggested_action",
        "affordances",
    }
    assert env["data"]["figma_node_id"] == "123:456"
    assert env["data"]["figma_node_type"] == "FRAME"
    # Affordance points at the next tool in the pipeline.
    tools = {a["tool"] for a in env["affordances"]}
    assert "mcp__pixel_mcp__measure" in tools


def test_mcp_spec_tool_error_envelope(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FIGMA_TOKEN", raising=False)
    from pixel_mcp.mcp_server import spec as mcp_spec

    env = mcp_spec(figma_url="https://www.figma.com/file/AbC/p?node-id=1-2")
    assert env["data"] is None
    assert env["diagnostics"]["error_type"] == "figma_auth_error"
