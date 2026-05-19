"""MCP integration tests — `diff`, `judge`, `check` tools exposed via FastMCP."""

from __future__ import annotations

import asyncio

import pytest
from pixel_mcp.mcp_server import server


@pytest.fixture
def tool_names() -> list[str]:
    """All tools registered on the MCP server."""

    async def _list() -> list[str]:
        tools = await server.list_tools()
        return [t.name for t in tools]

    return asyncio.run(_list())


def test_mcp_lists_slice4_tools(tool_names: list[str]) -> None:
    for name in ("doctor", "spec", "measure", "diff", "judge", "check"):
        assert name in tool_names, f"missing MCP tool: {name}"


def test_mcp_check_tool_has_route_and_figma_args(tool_names: list[str]) -> None:
    """Smoke: the check tool is registered and the schema matches expectations."""

    async def _schema() -> dict[str, object]:
        tools = await server.list_tools()
        check_tool = next(t for t in tools if t.name == "check")
        return check_tool.inputSchema or {}

    schema = asyncio.run(_schema())
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    assert isinstance(properties, dict)
    assert "figma_url" in properties
    assert "route" in properties
    assert "viewport_width" in properties
    assert "viewport_height" in properties


def test_mcp_diff_tool_has_path_args(tool_names: list[str]) -> None:
    async def _schema() -> dict[str, object]:
        tools = await server.list_tools()
        diff_tool = next(t for t in tools if t.name == "diff")
        return diff_tool.inputSchema or {}

    schema = asyncio.run(_schema())
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    assert isinstance(properties, dict)
    assert "spec_path" in properties
    assert "measured_path" in properties


def test_mcp_judge_tool_has_deltas_path(tool_names: list[str]) -> None:
    async def _schema() -> dict[str, object]:
        tools = await server.list_tools()
        judge_tool = next(t for t in tools if t.name == "judge")
        return judge_tool.inputSchema or {}

    schema = asyncio.run(_schema())
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    assert isinstance(properties, dict)
    assert "deltas_path" in properties
    assert "treat_minor_as_blocking" in properties
