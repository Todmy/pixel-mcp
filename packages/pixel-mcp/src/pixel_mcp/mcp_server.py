"""MCP server entry point.

Slice 1 exposes ``doctor``; Slice 2 adds ``spec``; Slice 3 adds ``measure``.
Future slices append one tool per subcommand. Transport is stdio — the
standard for Claude Code MCP integration.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from pixel_mcp import doctor as doctor_mod
from pixel_mcp import measure_cmd as measure_cmd_mod
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


@server.tool()
def measure(
    route: str,
    selectors: list[str] | None = None,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    wait_for: str | None = None,
    wait_for_network_idle: bool = True,
) -> dict[str, Any]:
    """Capture a MeasuredDOM from a Render.

    Args:
        route: URL of the Render (e.g. http://localhost:3000/foo).
        selectors: Optional list of CSS selectors. If None, auto-discover
            visible elements.
        viewport_width: Viewport width in CSS pixels. Default 1280.
        viewport_height: Viewport height in CSS pixels. Default 720.
        wait_for: Optional CSS selector to wait for before measuring.
        wait_for_network_idle: Wait for ``networkidle`` then one rAF quiet
            before measuring (default True). Deterministic snapshot.

    Returns the AXI envelope wrapping the MeasuredDOM (or a diagnostic
    envelope with ``data: null`` if capture failed).
    """
    envelope, _exit_code = measure_cmd_mod.run(
        route=route,
        viewport=(viewport_width, viewport_height),
        selectors=selectors,
        wait_for=wait_for,
        wait_for_network_idle=wait_for_network_idle,
    )
    serialized: dict[str, Any] = json.loads(json.dumps(envelope, default=str))
    return serialized


def run() -> None:
    """Start the stdio MCP server. Blocks until the client disconnects."""
    server.run()


if __name__ == "__main__":
    run()
