"""Typer entry point — `pixel-mcp <verb>`.

Slice 1 implements `doctor` and `mcp`; Slice 2 adds `spec`; Slice 3 adds
`measure`; Slice 4 adds `diff`, `judge`, `check`. Remaining subcommands
are stubs pointing at their tracking issue.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from pixel_tools_shared import Envelope

from pixel_mcp import check_cmd as check_cmd_mod
from pixel_mcp import diff_cmd as diff_cmd_mod
from pixel_mcp import doctor as doctor_mod
from pixel_mcp import human_feedback_cmd as human_feedback_cmd_mod
from pixel_mcp import judge_cmd as judge_cmd_mod
from pixel_mcp import mapping_cmd as mapping_cmd_mod
from pixel_mcp import measure_cmd as measure_cmd_mod
from pixel_mcp import reset_cmd as reset_cmd_mod
from pixel_mcp import review_cmd as review_cmd_mod
from pixel_mcp import snapshot_cmd as snapshot_cmd_mod
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


_VIEWPORT_PRESETS: dict[str, str] = {
    "responsive": "1280x720,768x1024,375x667",
}


def _parse_viewports(raw: str | None, preset: str | None) -> list[tuple[int, int]] | None:
    """Resolve the multi-viewport list from ``--viewports`` / ``--viewports-preset``.

    Returns ``None`` when neither flag is set (single-viewport behaviour
    preserved). When both are set, the explicit ``--viewports`` value wins
    and a Typer warning surfaces. Each entry is parsed via ``_parse_viewport``
    so mistyped values fail loud at the CLI boundary instead of producing a
    broken multi-viewport check.
    """
    chosen: str | None = None
    if raw is not None:
        chosen = raw
        if preset is not None:
            typer.echo(
                f"--viewports overrides --viewports-preset={preset!r}.",
                err=True,
            )
    elif preset is not None:
        if preset not in _VIEWPORT_PRESETS:
            raise typer.BadParameter(
                f"Unknown viewport preset {preset!r}. "
                f"Known: {', '.join(sorted(_VIEWPORT_PRESETS))}."
            )
        chosen = _VIEWPORT_PRESETS[preset]
    if chosen is None:
        return None
    parts = [p.strip() for p in chosen.split(",") if p.strip()]
    if not parts:
        return None
    return [_parse_viewport(p) for p in parts]


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
def diff(
    spec: Path = typer.Option(  # noqa: B008
        ...,
        "--spec",
        help="Path to a DesignSpec JSON (produced by `pixel-mcp spec --out`).",
    ),
    measured: Path = typer.Option(  # noqa: B008
        ...,
        "--measured",
        help="Path to a MeasuredDOM JSON (produced by `pixel-mcp measure --out`).",
    ),
    out: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None,
        "--out",
        help="Write the AXI envelope JSON to this file. Defaults to stdout.",
    ),
) -> None:
    """Compute Deltas between DesignSpec and MeasuredDOM."""
    envelope, exit_code = diff_cmd_mod.run(spec_path=spec, measured_path=measured)
    _emit(envelope, out)
    raise typer.Exit(code=exit_code)


@app.command()
def judge(
    deltas: Path = typer.Option(  # noqa: B008
        ...,
        "--deltas",
        help="Path to a Delta[] JSON (top-level array or AXI envelope with data.deltas).",
    ),
    strict: bool = typer.Option(  # noqa: B008
        False,
        "--strict",
        help="Treat minor Deltas as blocking.",
    ),
    out: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None,
        "--out",
        help="Write the AXI envelope JSON to this file. Defaults to stdout.",
    ),
) -> None:
    """Run the Convergence Judge over a Delta[]."""
    envelope, exit_code = judge_cmd_mod.run(deltas_path=deltas, treat_minor_as_blocking=strict)
    _emit(envelope, out)
    raise typer.Exit(code=exit_code)


@app.command()
def check(
    figma: Optional[str] = typer.Option(  # noqa: B008, UP007
        None,
        "--figma",
        help="Figma URL — Frame, Component Instance, or Master Component. "
        "Mutually exclusive with --image.",
    ),
    image: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None,
        "--image",
        help="Path to a static design image (PNG/JPG) — image-only mode. "
        "Mutually exclusive with --figma.",
    ),
    route: str = typer.Option(  # noqa: B008
        ...,
        "--route",
        help="URL of the Render (e.g. http://localhost:3000/foo).",
    ),
    selectors: Optional[str] = typer.Option(  # noqa: B008, UP007
        None,
        "--selectors",
        help="Comma-separated CSS selectors to measure. Auto-discover when omitted.",
    ),
    viewport: str = typer.Option(  # noqa: B008
        "1280x720",
        "--viewport",
        help="Viewport size as WIDTHxHEIGHT (default 1280x720). Ignored when --viewports is set.",
    ),
    viewports: Optional[str] = typer.Option(  # noqa: B008, UP007
        None,
        "--viewports",
        help="Comma-separated viewport list (e.g. '1280x720,375x667,768x1024'). "
        "Runs the full convergence pipeline at each viewport (v2-1). "
        "Mutually compatible with --viewport — when both are set, --viewports wins.",
    ),
    viewports_preset: Optional[str] = typer.Option(  # noqa: B008, UP007
        None,
        "--viewports-preset",
        help="Convenience: expand a named preset to --viewports. "
        "'responsive' = 1280x720,768x1024,375x667.",
    ),
    wait_for: Optional[str] = typer.Option(  # noqa: B008, UP007
        None,
        "--wait-for",
        help="CSS selector that must appear before measurement begins.",
    ),
    refresh_spec: bool = typer.Option(  # noqa: B008
        False,
        "--refresh-spec",
        help="Bypass the spec-cache and re-fetch from the Figma API (Figma mode only).",
    ),
    strict: bool = typer.Option(  # noqa: B008
        False,
        "--strict",
        help="Treat minor Deltas as blocking.",
    ),
    enable_dinov2: bool = typer.Option(  # noqa: B008
        False,
        "--enable-dinov2/--no-enable-dinov2",
        help="Opt in to Level 1 (DINOv2 per-crop similarity) escalation gate. "
        "Requires `pixel-mcp-ml --extra dinov2`.",
    ),
    dinov2_threshold: float = typer.Option(  # noqa: B008
        0.95,
        "--dinov2-threshold",
        help="Cosine-similarity threshold for Level 1 Gate Pass (default 0.95).",
    ),
    enable_vlm: bool = typer.Option(  # noqa: B008
        False,
        "--enable-vlm/--no-enable-vlm",
        help="Opt in to Level 2 (VLM verification) escalation gate. "
        "Runs only after Level 1 passes. Requires `pixel-mcp-ml --extra vlm`.",
    ),
    vlm_threshold: float = typer.Option(  # noqa: B008
        0.7,
        "--vlm-threshold",
        help="Confidence threshold for Level 2 Gate Pass (default 0.7).",
    ),
    vlm_backend: str = typer.Option(  # noqa: B008
        "claude",
        "--vlm-backend",
        help="VLM backend: 'claude' (default) or 'qwen-local' (v1-2 STUB).",
    ),
    enable_human_gate: bool = typer.Option(  # noqa: B008
        False,
        "--enable-human-gate/--no-enable-human-gate",
        help="Opt in to Level 3 (human review) escalation gate. Runs only "
        "after the highest enabled automated level passes.",
    ),
    enable_omniparser: bool = typer.Option(  # noqa: B008
        False,
        "--enable-omniparser/--no-enable-omniparser",
        help="Augment Region attribution with OmniParser semantic labels "
        "(button/input/icon/...). Requires `pixel-mcp-ml --extra omniparser`.",
    ),
    omniparser_confidence_threshold: float = typer.Option(  # noqa: B008
        0.3,
        "--omniparser-confidence-threshold",
        help="Drop OmniParser detections below this confidence (default 0.3).",
    ),
    out: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None,
        "--out",
        help="Write the AXI envelope JSON to this file. Defaults to stdout.",
    ),
) -> None:
    """One Iteration of the Convergence Loop.

    Figma mode (``--figma``): spec + measure + diff + judge with visual signals.
    Image-only mode (``--image``): measure + visual signals (SSIM + Hot Regions).
    Exactly one of ``--figma`` or ``--image`` must be provided.
    """
    parsed_viewports = _parse_viewports(viewports, viewports_preset)
    envelope, exit_code = check_cmd_mod.run(
        figma_url=figma,
        image_path=image,
        route=route,
        viewport=_parse_viewport(viewport),
        viewports=parsed_viewports,
        selectors=_parse_selectors(selectors),
        wait_for=wait_for,
        refresh_spec=refresh_spec,
        treat_minor_as_blocking=strict,
        enable_dinov2=enable_dinov2,
        dinov2_threshold=dinov2_threshold,
        enable_vlm=enable_vlm,
        vlm_threshold=vlm_threshold,
        vlm_backend=vlm_backend,
        enable_human_gate=enable_human_gate,
        enable_omniparser=enable_omniparser,
        omniparser_confidence_threshold=omniparser_confidence_threshold,
    )
    _emit(envelope, out)
    raise typer.Exit(code=exit_code)


def _emit(envelope: Envelope, out: Path | None) -> None:
    payload = json.dumps(envelope, indent=2, default=str)
    if out is not None:
        out.write_text(payload)
    else:
        typer.echo(payload)


@app.command()
def review(
    out: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None, "--out", help="Write the AXI envelope JSON to this file."
    ),
) -> None:
    """Prepare a Level 3 human review packet from the most recent check."""
    envelope, exit_code = review_cmd_mod.run()
    _emit(envelope, out)
    raise typer.Exit(code=exit_code)


@app.command("human-feedback")
def human_feedback(
    approve: bool = typer.Option(  # noqa: B008
        False,
        "--approve",
        help="Sign off — record Final Convergence at Level 3 on the next check.",
    ),
    rejection_notes: Optional[str] = typer.Option(  # noqa: B008, UP007
        None,
        "--rejection-notes",
        help="Reject the review packet and inject the given notes as a "
        "pseudo-Delta on the next check.",
    ),
    out: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None, "--out", help="Write the AXI envelope JSON to this file."
    ),
) -> None:
    """Capture the Level 3 human verdict (approve OR rejection notes)."""
    envelope, exit_code = human_feedback_cmd_mod.run(
        approve=approve, rejection_notes=rejection_notes
    )
    _emit(envelope, out)
    raise typer.Exit(code=exit_code)


@app.command()
def mapping(
    figma: str = typer.Option(  # noqa: B008
        ...,
        "--figma",
        help="Figma URL — Frame, Component Instance, or Master Component.",
    ),
    route: str = typer.Option(  # noqa: B008
        ...,
        "--route",
        help="URL of the Render (e.g. http://localhost:3000/foo).",
    ),
    viewport: str = typer.Option(  # noqa: B008
        "1280x720",
        "--viewport",
        help="Viewport size as WIDTHxHEIGHT (default 1280x720).",
    ),
    out: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None,
        "--out",
        help="Write the AXI envelope JSON to this file. Defaults to stdout.",
    ),
) -> None:
    """Resolve Figma <-> DOM Mappings and write .pixel-mcp/mappings.json."""
    envelope, exit_code = mapping_cmd_mod.run(
        figma_url=figma,
        route=route,
        viewport=_parse_viewport(viewport),
    )
    _emit(envelope, out)
    raise typer.Exit(code=exit_code)


@app.command()
def snapshot(
    route: str = typer.Option(..., "--route", help="URL of the Render."),  # noqa: B008
    tag: str = typer.Option(..., "--tag", help="Baseline tag name."),  # noqa: B008
    viewport: str = typer.Option(  # noqa: B008
        "1280x720", "--viewport", help="Viewport size as WIDTHxHEIGHT."
    ),
    out: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None, "--out", help="Write the AXI envelope JSON to this file."
    ),
) -> None:
    """Capture and persist a named Render baseline."""
    envelope, exit_code = snapshot_cmd_mod.run(
        route=route, tag=tag, viewport=_parse_viewport(viewport)
    )
    _emit(envelope, out)
    raise typer.Exit(code=exit_code)


@app.command()
def reset(
    all_artifacts: bool = typer.Option(  # noqa: B008
        False, "--all", help="Also remove named snapshots."
    ),
    out: Optional[Path] = typer.Option(  # noqa: B008, UP007
        None, "--out", help="Write the AXI envelope JSON to this file."
    ),
) -> None:
    """Clear the State Directory (preserve snapshots unless --all)."""
    envelope, exit_code = reset_cmd_mod.run(all_artifacts=all_artifacts)
    _emit(envelope, out)
    raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    app()
