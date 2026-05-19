"""Smoke test: MCP server module imports and exposes the doctor tool."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def test_mcp_server_imports_and_registers_doctor() -> None:
    from pixel_mcp import mcp_server

    assert mcp_server.server is not None
    # FastMCP keeps registered tools accessible via list_tools(); we don't
    # call run() (it would block on stdio). Importing the module and
    # checking the singleton is enough for a Slice 1 smoke test.
    assert callable(mcp_server.run)
    assert callable(mcp_server.doctor)


def test_mcp_doctor_tool_returns_envelope() -> None:
    """The MCP doctor tool function returns the same envelope as the CLI."""
    from pixel_mcp.mcp_server import doctor as mcp_doctor

    env = mcp_doctor()
    assert set(env.keys()) == {
        "data",
        "hints",
        "diagnostics",
        "next_suggested_action",
        "affordances",
    }
    assert "checks" in env["data"]
