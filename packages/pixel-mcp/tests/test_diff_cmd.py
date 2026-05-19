"""CLI tests for `pixel-mcp diff`."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pixel_mcp.cli import app
from pixel_mcp.render import (
    BoundingBox,
    ComputedStyle,
    MeasuredDOM,
    MeasuredElement,
)
from pixel_mcp.spec import (
    ColorOrGradient,
    DesignSpec,
    Dimensions,
    LayoutSpec,
)
from typer.testing import CliRunner


def _fixture_spec_path(tmp: Path) -> Path:
    spec = DesignSpec(
        figma_file_id="abc",
        figma_node_id="1:1",
        figma_node_type="FRAME",
        name="Hero",
        dimensions=Dimensions(width=400, height=100),
        layout=LayoutSpec(),
        fills=[ColorOrGradient(type="SOLID", color={"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0})],
        children=[],
        extracted_at=datetime.now(UTC),
    )
    p = tmp / "spec.json"
    p.write_text(spec.model_dump_json())
    return p


def _fixture_dom_path(tmp: Path, bg: str = "#ff0000", text: str = "Hero") -> Path:
    style_kwargs: dict[str, object] = {
        "color": "#000000",
        "background_color": bg,
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
    dom = MeasuredDOM(
        route="http://localhost:3000/",
        viewport=(1280, 720),
        measured_at=datetime.now(UTC),
        elements=[
            MeasuredElement(
                selector="#hero",
                bounding_box=BoundingBox(x=0, y=0, w=400, h=100),
                computed_style=ComputedStyle.model_validate(style_kwargs),
                text_content=text,
            )
        ],
    )
    p = tmp / "measured.json"
    p.write_text(dom.model_dump_json())
    return p


def test_diff_happy_exits_zero(tmp_path: Path) -> None:
    runner = CliRunner()
    spec_p = _fixture_spec_path(tmp_path)
    dom_p = _fixture_dom_path(tmp_path, bg="#ff0000")
    result = runner.invoke(app, ["diff", "--spec", str(spec_p), "--measured", str(dom_p)])
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.output)
    assert envelope["data"]["deltas"] == []


def test_diff_mismatch_exits_one(tmp_path: Path) -> None:
    runner = CliRunner()
    spec_p = _fixture_spec_path(tmp_path)
    dom_p = _fixture_dom_path(tmp_path, bg="#00ff00")  # injected color defect
    result = runner.invoke(app, ["diff", "--spec", str(spec_p), "--measured", str(dom_p)])
    assert result.exit_code == 1, result.output
    envelope = json.loads(result.output)
    assert any(d["property"] == "background_color" for d in envelope["data"]["deltas"])


def test_diff_missing_spec_exits_twelve(tmp_path: Path) -> None:
    runner = CliRunner()
    dom_p = _fixture_dom_path(tmp_path)
    result = runner.invoke(
        app,
        ["diff", "--spec", str(tmp_path / "missing.json"), "--measured", str(dom_p)],
    )
    assert result.exit_code == 12
    envelope = json.loads(result.output)
    assert envelope["data"] is None
    assert envelope["diagnostics"]["error_type"] == "spec_not_found"


def test_diff_malformed_measured_exits_twelve(tmp_path: Path) -> None:
    runner = CliRunner()
    spec_p = _fixture_spec_path(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    result = runner.invoke(app, ["diff", "--spec", str(spec_p), "--measured", str(bad)])
    assert result.exit_code == 12
    envelope = json.loads(result.output)
    assert envelope["diagnostics"]["error_type"] in {"measured_invalid", "measured_shape_unknown"}


def test_diff_writes_out_file(tmp_path: Path) -> None:
    runner = CliRunner()
    spec_p = _fixture_spec_path(tmp_path)
    dom_p = _fixture_dom_path(tmp_path, bg="#ff0000")
    out = tmp_path / "deltas.json"
    result = runner.invoke(
        app,
        ["diff", "--spec", str(spec_p), "--measured", str(dom_p), "--out", str(out)],
    )
    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert "data" in payload
    assert payload["data"]["deltas"] == []
