"""Typer entry point — `pixel-mcp-ml <verb>`.

Currently exposes one verb: ``dinov2-compare``. Future slices add
``omniparser-detect`` and VLM bridges.

Exit codes:

- ``0`` — Success.
- ``1`` — One or both input images do not exist on disk.
- ``12`` — DINOv2 deps not installed (use the install hint).
"""

from __future__ import annotations

import json as json_mod
from pathlib import Path
from typing import cast

import typer

from pixel_mcp_ml.dinov2_compare import (
    DINOv2NotInstalledError,
    ModelSize,
    compute_dinov2_similarity,
)
from pixel_mcp_ml.version import __version__

app = typer.Typer(
    name="pixel-mcp-ml",
    help="ML extras for pixel-mcp (DINOv2 perceptual similarity).",
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
    """pixel-mcp-ml root command."""


@app.command("dinov2-compare")
def dinov2_compare(
    image_a: Path = typer.Argument(..., help="Path to the first image."),  # noqa: B008
    image_b: Path = typer.Argument(..., help="Path to the second image."),  # noqa: B008
    model_size: str = typer.Option(  # noqa: B008
        "small",
        "--model-size",
        help="DINOv2 model size: small (~88MB, default) or base (~330MB).",
    ),
    json: bool = typer.Option(  # noqa: B008
        False,
        "--json",
        help="Emit machine-readable JSON instead of the human-readable line.",
    ),
) -> None:
    """Compute DINOv2 cosine similarity between two images."""
    if model_size not in ("small", "base"):
        typer.echo(
            f"--model-size must be 'small' or 'base'; got {model_size!r}",
            err=True,
        )
        raise typer.Exit(code=2)

    for label, path in (("image_a", image_a), ("image_b", image_b)):
        if not path.exists():
            typer.echo(f"{label} not found: {path}", err=True)
            raise typer.Exit(code=1)

    # ``compute_dinov2_similarity`` is imported at module level — it does
    # *not* import transformers/torch until the function is called.
    narrowed_size = cast(ModelSize, model_size)
    try:
        similarity = compute_dinov2_similarity(
            image_a,
            image_b,
            model_size=narrowed_size,
        )
    except DINOv2NotInstalledError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=12) from exc

    if json:
        # Re-detect the device cheaply by reading the cache — by this point
        # the model has been loaded so the cache is warm.
        from pixel_mcp_ml.dinov2_compare import _MODEL_CACHE

        cached = _MODEL_CACHE.get(narrowed_size)
        device = cached[2] if cached is not None else "unknown"
        payload = {
            "similarity": similarity,
            "model_size": model_size,
            "device": device,
        }
        typer.echo(json_mod.dumps(payload))
    else:
        typer.echo(f"Similarity: {similarity:.4f}")


if __name__ == "__main__":
    app()
