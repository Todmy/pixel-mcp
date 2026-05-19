"""MCP server entry point.

Slice 1 exposes a single tool, ``doctor``, returning the same AXI envelope
as the CLI. Future slices append one tool per subcommand. Transport is
stdio — the standard for Claude Code MCP integration.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from pixel_mcp import doctor as doctor_mod
from pixel_mcp import spec_cmd as spec_cmd_mod

server: FastMCP = FastMCP("pixel-mcp")


@server.tool()
def doctor() -> dict[str, Any]:
    """Run the environment Check.

    Returns the AXI envelope: data (checks + summary), hints, diagnostics,
    next_suggested_action, affordances.
    """
    # FastMCP serializes the return value to JSON content. Round-trip
    # through json to guarantee the envelope is plain-JSON-safe before the
    # server framework touches it.
    envelope = doctor_mod.build_envelope()
    # Round-trip through JSON to guarantee plain-JSON-safe types before the
    # MCP framework serializes the response.
    serialized: dict[str, Any] = json.loads(json.dumps(envelope))
    return serialized


@server.tool()
def spec(figma_url: str, refresh: bool = False) -> dict[str, Any]:
    """Extract a DesignSpec from a Figma Source.

    Args:
        figma_url: A Figma URL — Frame, Component Instance, or Master Component.
        refresh: Bypass the spec-cache and re-fetch from the Figma API.

    Returns the AXI envelope wrapping the DesignSpec (or a diagnostic
    envelope with ``data: null`` if extraction failed).
    """
    envelope, _exit_code = spec_cmd_mod.run(figma_url=figma_url, refresh=refresh)
    serialized: dict[str, Any] = json.loads(json.dumps(envelope, default=str))
    return serialized


def run() -> None:
    """Start the stdio MCP server. Blocks until the client disconnects."""
    server.run()


if __name__ == "__main__":
    run()
