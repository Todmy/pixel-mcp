"""Typer entry point — `pixel-mcp <verb>`.

Slice 1 implements `doctor` and `mcp`. Every other subcommand is a stub
that points at the tracking issue and exits non-zero.
"""

from __future__ import annotations

import json
import sys

import typer
from pixel_tools_shared import Envelope

from pixel_mcp import doctor as doctor_mod
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
    typer.echo(f"Not yet implemented — see Todmy/PBaaS#{issue}", err=True)
    sys.exit(2)


@app.command()
def spec() -> None:
    """Extract a DesignSpec from a Figma Source. (stub)"""
    _stub("spec", 12)


@app.command()
def measure() -> None:
    """Capture a MeasuredDOM from a Render. (stub)"""
    _stub("measure", 13)


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
