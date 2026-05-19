"""Image-only mode tests for ``pixel-mcp check`` (v0.5-1).

Image-only mode skips DesignSpec extraction entirely. Convergence is driven
purely by Level 0 visual signals:

- ``ssim_score >= ssim_threshold``
- zero Hot Regions ``>= min_bbox_area``

Hot Regions still feed ``decompose_hot_regions`` for DOM attribution, and we
synthesize pseudo-Deltas from them so loop economics work unchanged.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from pixel_mcp.check_cmd import (
    EXIT_CONVERGED,
    EXIT_DELTAS,
    EXIT_FATAL,
)
from pixel_mcp.check_cmd import (
    run as check_run,
)
from pixel_mcp.render import (
    BoundingBox,
    ComputedStyle,
    MeasuredDOM,
    MeasuredElement,
)

# --- Helpers --------------------------------------------------------------


def _dom() -> MeasuredDOM:
    style: dict[str, Any] = {
        "color": "#000000",
        "background_color": "#ff0000",
        "font_family": "Inter",
        "font_size_px": 16.0,
        "font_weight": 400,
        "padding_top": 0.0,
        "padding_right": 0.0,
        "padding_bottom": 0.0,
        "padding_left": 0.0,
        "margin_top": 0.0,
        "margin_right": 0.0,
        "margin_bottom": 0.0,
        "margin_left": 0.0,
        "border_top_width": 0.0,
        "border_right_width": 0.0,
        "border_bottom_width": 0.0,
        "border_left_width": 0.0,
    }
    return MeasuredDOM(
        route="http://localhost:3000/",
        viewport=(400, 300),
        measured_at=datetime.now(UTC),
        elements=[
            MeasuredElement(
                selector="#hero",
                bounding_box=BoundingBox(x=0, y=0, w=400, h=300),
                computed_style=ComputedStyle.model_validate(style),
                text_content="Hero",
            )
        ],
    )


def _solid_png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (400, 300)) -> bytes:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[:, :, 0] = color[0]
    arr[:, :, 1] = color[1]
    arr[:, :, 2] = color[2]
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mocked_image_pipeline():
    """Patch heavy deps + isolate loop_state so tests don't share iteration counts."""
    from pixel_mcp.loop_state import IterationState

    fresh_state = IterationState()

    def _read(*args: object, **kwargs: object) -> IterationState:
        return IterationState(
            session_id=fresh_state.session_id,
            iteration=fresh_state.iteration,
            last_delta_hash=fresh_state.last_delta_hash,
            highest_level_reached=fresh_state.highest_level_reached,
            recent_hashes=list(fresh_state.recent_hashes),
        )

    def _write(state: IterationState, *args: object, **kwargs: object) -> None:
        fresh_state.iteration = state.iteration
        fresh_state.last_delta_hash = state.last_delta_hash
        fresh_state.highest_level_reached = state.highest_level_reached
        fresh_state.recent_hashes = list(state.recent_hashes)

    with (
        patch("pixel_mcp.check_cmd.measure_render") as m_measure,
        patch("pixel_mcp.check_cmd.capture_screenshot") as m_shot,
        patch("pixel_mcp.check_cmd.extract_spec") as m_spec,
        patch("pixel_mcp.check_cmd.read_state", side_effect=_read),
        patch("pixel_mcp.check_cmd.write_state", side_effect=_write),
        patch("pixel_mcp.check_cmd.append_history"),
    ):
        yield m_measure, m_shot, m_spec


# --- Tests ----------------------------------------------------------------


def test_check_image_only_happy_path(tmp_path: Path, mocked_image_pipeline: Any) -> None:
    """Matching design image + matching render → exit 0, mode='image', no spec."""
    m_measure, m_shot, m_spec = mocked_image_pipeline

    design_png = tmp_path / "design.png"
    matching_bytes = _solid_png_bytes((255, 0, 0))
    design_png.write_bytes(matching_bytes)

    m_measure.return_value = (_dom(), False)
    m_shot.return_value = matching_bytes  # screenshot identical to design

    envelope, exit_code = check_run(
        image_path=design_png,
        route="http://localhost:3000/",
        viewport=(400, 300),
    )

    assert exit_code == EXIT_CONVERGED, envelope
    assert envelope["data"]["mode"] == "image"
    assert envelope["data"]["converged"] is True
    assert envelope["data"]["spec_node_id"] is None
    assert envelope["data"]["hot_regions"] == []
    assert envelope["data"]["deltas"] == []
    # Figma side never invoked in image-only mode.
    m_spec.assert_not_called()


def test_check_image_only_mismatch(tmp_path: Path, mocked_image_pipeline: Any) -> None:
    """Different design vs render → exit 1, hot regions present, pseudo-Deltas emitted."""
    m_measure, m_shot, m_spec = mocked_image_pipeline

    design_png = tmp_path / "design.png"
    design_png.write_bytes(_solid_png_bytes((255, 0, 0)))  # red design
    m_measure.return_value = (_dom(), False)
    m_shot.return_value = _solid_png_bytes((0, 255, 0))  # green render — fully drifted

    envelope, exit_code = check_run(
        image_path=design_png,
        route="http://localhost:3000/",
        viewport=(400, 300),
    )

    assert exit_code == EXIT_DELTAS, envelope
    assert envelope["data"]["mode"] == "image"
    assert envelope["data"]["converged"] is False
    # Whole image differs → at least one significant Hot Region.
    assert envelope["data"]["significant_hot_region_count"] >= 1
    # Pseudo-Delta synthesized for each significant Hot Region.
    assert len(envelope["data"]["deltas"]) >= 1
    delta = envelope["data"]["deltas"][0]
    assert delta["property"].startswith("hot_region_")
    assert delta["severity"] in {"critical", "major", "minor"}
    # SSIM should be well below the threshold for a full-image color flip.
    assert envelope["data"]["ssim_score"] is not None
    assert envelope["data"]["ssim_score"] < 0.97
    m_spec.assert_not_called()


def test_check_rejects_both_figma_and_image(tmp_path: Path) -> None:
    """Passing both --figma and --image must exit 12 with a clear error."""
    design_png = tmp_path / "design.png"
    design_png.write_bytes(_solid_png_bytes((255, 0, 0)))

    envelope, exit_code = check_run(
        figma_url="https://figma.com/design/abc?node-id=1-1",
        image_path=design_png,
        route="http://localhost:3000/",
    )

    assert exit_code == EXIT_FATAL
    assert envelope["data"] is None
    assert envelope["diagnostics"]["error_type"] == "design_source_conflict"
    assert "both" in envelope["diagnostics"]["error_message"].lower()


def test_check_rejects_neither_figma_nor_image() -> None:
    """Passing neither --figma nor --image must exit 12 with a clear error."""
    envelope, exit_code = check_run(route="http://localhost:3000/")

    assert exit_code == EXIT_FATAL
    assert envelope["data"] is None
    assert envelope["diagnostics"]["error_type"] == "design_source_missing"
    assert "design source" in envelope["diagnostics"]["error_message"].lower()


def test_check_image_only_image_not_found() -> None:
    """A missing --image path returns EXIT_FATAL with a helpful error."""
    envelope, exit_code = check_run(
        image_path="/does/not/exist.png",
        route="http://localhost:3000/",
    )

    assert exit_code == EXIT_FATAL
    assert envelope["data"] is None
    assert envelope["diagnostics"]["error_type"] == "image_not_found"


def test_check_image_only_via_cli(tmp_path: Path, mocked_image_pipeline: Any) -> None:
    """End-to-end through the CLI with --image."""
    import json

    from pixel_mcp.cli import app
    from typer.testing import CliRunner

    m_measure, m_shot, _m_spec = mocked_image_pipeline

    design_png = tmp_path / "design.png"
    matching_bytes = _solid_png_bytes((255, 0, 0))
    design_png.write_bytes(matching_bytes)

    m_measure.return_value = (_dom(), False)
    m_shot.return_value = matching_bytes

    out = tmp_path / "envelope.json"
    result = CliRunner().invoke(
        app,
        [
            "check",
            "--image",
            str(design_png),
            "--route",
            "http://localhost:3000/",
            "--viewport",
            "400x300",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    envelope = json.loads(out.read_text())
    assert envelope["data"]["mode"] == "image"
    assert envelope["data"]["converged"] is True


def test_check_cli_rejects_both_figma_and_image(tmp_path: Path) -> None:
    """CLI surface: passing both --figma and --image exits 12."""
    from pixel_mcp.cli import app
    from typer.testing import CliRunner

    design_png = tmp_path / "design.png"
    design_png.write_bytes(_solid_png_bytes((255, 0, 0)))

    result = CliRunner().invoke(
        app,
        [
            "check",
            "--figma",
            "https://figma.com/design/abc?node-id=1-1",
            "--image",
            str(design_png),
            "--route",
            "http://localhost:3000/",
        ],
    )
    assert result.exit_code == EXIT_FATAL


def test_check_cli_rejects_neither_figma_nor_image() -> None:
    """CLI surface: passing neither --figma nor --image exits 12."""
    from pixel_mcp.cli import app
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        app,
        ["check", "--route", "http://localhost:3000/"],
    )
    assert result.exit_code == EXIT_FATAL
