"""Typer entry point — `pixel-mcp <verb>`.

Slice 1 implements `doctor` and `mcp`; Slice 2 adds `spec`; Slice 3 adds
`measure`. Every other subcommand is a stub that points at the tracking
issue and exits non-zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from pixel_tools_shared import Envelope

from pixel_mcp import doctor as doctor_mod
from pixel_mcp import measure_cmd as measure_cmd_mod
from pixel_mcp import spec_cmd as spec_cmd_mod
from pixel_mcp.version import __version__

app = typer.Typer(
    name="pixel-mcp",
    help="Figma to Browser convergence harness (CLI + MCP server).",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(  # noqa: B008
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """pixel-mcp root command."""


@app.command()
def doctor(
    json_output: bool = typer.Option(  # noqa: B008
        False, "--json", help="Emit the raw AXI envelope as JSON."
    ),
) -> None:
    """Run the environment Check and print status."""
    envelope = doctor_mod.build_envelope()
    if json_output:
        typer.echo(json.dumps(envelope, indent=2))
    else:
        _print_doctor_human(envelope)
    raise typer.Exit(code=doctor_mod.exit_code_for(envelope))


def _print_doctor_human(envelope: Envelope) -> None:
    data = envelope["data"]
    typer.echo(f"pixel-mcp doctor — {data['summary']}")
    typer.echo("")
    for c in data["checks"]:
        marker = {"green": "[OK]", "amber": "[--]", "red": "[XX]"}[c["status"]]
        typer.echo(f"  {marker} {c['name']}: {c['detail']}")
    if envelope["hints"]:
        typer.echo("")
        typer.echo("Hints:")
        for h in envelope["hints"]:
            typer.echo(f"  - {h}")
    if envelope["next_suggested_action"]:
        typer.echo("")
        typer.echo(f"Next: {envelope['next_suggested_action']}")


@app.command()
def mcp() -> None:
    """Launch the MCP server over stdio (for Claude Code)."""
    from pixel_mcp.mcp_server import run

    run()


def _stub(verb: str, issue: int) -> None:
    typer.echo(
        f"`pixel-mcp {verb}` not yet implemented — see Todmy/PBaaS#{issue}",
        err=True,
    )
    sys.exit(2)


@app.command()
def spec(
    figma: str = typer.Option(  # noqa: B008
        ...,
        "--figma",
        help="Figma URL — Frame, Component Instance, or Master Component.",
    ),
    out: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None,
        "--out",
        help="Write the AXI envelope JSON to this file. Defaults to stdout.",
    ),
    refresh_spec: bool = typer.Option(  # noqa: B008
        False,
        "--refresh-spec",
        help="Bypass the spec-cache and re-fetch from the Figma API.",
    ),
) -> None:
    """Extract a DesignSpec from a Figma Source."""
    envelope, exit_code = spec_cmd_mod.run(figma_url=figma, refresh=refresh_spec)
    payload = json.dumps(envelope, indent=2, default=str)
    if out is not None:
        out.write_text(payload)
    else:
        typer.echo(payload)
    raise typer.Exit(code=exit_code)


def _parse_viewport(raw: str) -> tuple[int, int]:
    """Parse ``WxH`` strings (e.g. ``1280x720``) into a (w, h) tuple."""
    try:
        w_str, h_str = raw.lower().split("x", 1)
        return (int(w_str), int(h_str))
    except (ValueError, AttributeError) as exc:
        raise typer.BadParameter(
            f"Viewport must be of the form 'WIDTHxHEIGHT' (e.g. 1280x720); got {raw!r}."
        ) from exc


def _parse_selectors(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    return parts or None


@app.command()
def measure(
    route: str = typer.Option(  # noqa: B008
        ...,
        "--route",
        help="URL of the Render (e.g. http://localhost:3000/foo).",
    ),
    selectors: Optional[str] = typer.Option(  # noqa: B008, UP007
        None,
        "--selectors",
        help="Comma-separated CSS selectors to measure. If omitted, auto-discover visible elements.",
    ),
    viewport: str = typer.Option(  # noqa: B008
        "1280x720",
        "--viewport",
        help="Viewport size as WIDTHxHEIGHT (default 1280x720).",
    ),
    wait_for: Optional[str] = typer.Option(  # noqa: B008, UP007
        None,
        "--wait-for",
        help="CSS selector that must appear before measurement begins.",
    ),
    out: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None,
        "--out",
        help="Write the AXI envelope JSON to this file. Defaults to stdout.",
    ),
) -> None:
    """Capture a MeasuredDOM from a Render."""
    envelope, exit_code = measure_cmd_mod.run(
        route=route,
        viewport=_parse_viewport(viewport),
        selectors=_parse_selectors(selectors),
        wait_for=wait_for,
    )
    payload = json.dumps(envelope, indent=2, default=str)
    if out is not None:
        out.write_text(payload)
    else:
        typer.echo(payload)
    raise typer.Exit(code=exit_code)


@app.command()
def diff() -> None:
    """Compute Deltas between DesignSpec and MeasuredDOM. (stub)"""
    _stub("diff", 14)


@app.command()
def judge() -> None:
    """Run the Convergence Judge for the current Iteration. (stub)"""
    _stub("judge", 14)


@app.command()
def check() -> None:
    """One Iteration of the Convergence Loop. (stub)"""
    _stub("check", 14)


@app.command()
def review() -> None:
    """Open a human review surface for Level 3. (stub)"""
    _stub("review", 20)


@app.command()
def mapping() -> None:
    """Manage the Mappings between Figma nodes and DOM selectors. (stub)"""
    _stub("mapping", 18)


@app.command()
def snapshot() -> None:
    """Persist a tagged Render baseline. (stub)"""
    _stub("snapshot", 20)


@app.command()
def reset() -> None:
    """Clear the State Directory. (stub)"""
    _stub("reset", 20)


if __name__ == "__main__":
    app()
