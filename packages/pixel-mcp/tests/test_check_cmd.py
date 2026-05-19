"""Composite `pixel-mcp check` tests — orchestration of spec + measure + diff + judge."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pixel_mcp.check_cmd import (
    EXIT_CONVERGED,
    EXIT_DELTAS,
    EXIT_FATAL,
)
from pixel_mcp.check_cmd import (
    run as check_run,
)
from pixel_mcp.figma_client import FigmaAuthError
from pixel_mcp.render import (
    BoundingBox,
    ComputedStyle,
    MeasuredDOM,
    MeasuredElement,
    RouteUnreachableError,
)
from pixel_mcp.spec import ColorOrGradient, DesignSpec, Dimensions, LayoutSpec


def _spec() -> DesignSpec:
    return DesignSpec(
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


def _dom(bg: str = "#ff0000") -> MeasuredDOM:
    style: dict[str, Any] = {
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
    return MeasuredDOM(
        route="http://localhost:3000/",
        viewport=(1280, 720),
        measured_at=datetime.now(UTC),
        elements=[
            MeasuredElement(
                selector="#hero",
                bounding_box=BoundingBox(x=0, y=0, w=400, h=100),
                computed_style=ComputedStyle.model_validate(style),
                text_content="Hero",
            )
        ],
    )


@pytest.fixture
def mocked_pipeline():
    """Patch the heavy dependencies — Figma + Playwright — at module level."""
    with (
        patch("pixel_mcp.check_cmd.extract_spec") as m_spec,
        patch("pixel_mcp.check_cmd.measure_render") as m_measure,
    ):
        yield m_spec, m_measure


def test_check_happy_path_exits_zero(mocked_pipeline: Any) -> None:
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)

    envelope, exit_code = check_run(
        figma_url="https://figma.com/design/abc?node-id=1-1",
        route="http://localhost:3000/",
    )
    assert exit_code == EXIT_CONVERGED
    assert envelope["data"]["converged"] is True
    assert envelope["data"]["deltas"] == []
    assert envelope["data"]["ssim_score"] is None  # reserved for Slice 6
    assert envelope["data"]["hot_regions"] == []  # reserved for Slice 6


def test_check_mismatch_exits_one(mocked_pipeline: Any) -> None:
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#00ff00"), False)  # injected color defect

    envelope, exit_code = check_run(
        figma_url="https://figma.com/design/abc?node-id=1-1",
        route="http://localhost:3000/",
    )
    assert exit_code == EXIT_DELTAS
    assert envelope["data"]["converged"] is False
    assert any(d["property"] == "background_color" for d in envelope["data"]["deltas"])


def test_check_figma_auth_error_exits_twelve(mocked_pipeline: Any) -> None:
    m_spec, _m_measure = mocked_pipeline
    m_spec.side_effect = FigmaAuthError("No FIGMA_TOKEN")

    envelope, exit_code = check_run(
        figma_url="https://figma.com/design/abc?node-id=1-1",
        route="http://localhost:3000/",
    )
    assert exit_code == EXIT_FATAL
    assert envelope["data"] is None
    assert envelope["diagnostics"]["error_type"] == "figma_auth_error"


def test_check_route_unreachable_exits_twelve(mocked_pipeline: Any) -> None:
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.side_effect = RouteUnreachableError("connection refused")

    envelope, exit_code = check_run(
        figma_url="https://figma.com/design/abc?node-id=1-1",
        route="http://localhost:3000/",
    )
    assert exit_code == EXIT_FATAL
    assert envelope["diagnostics"]["error_type"] == "route_unreachable"


def test_check_envelope_includes_severity_hints(mocked_pipeline: Any) -> None:
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#00ff00"), False)

    envelope, _exit = check_run(
        figma_url="https://figma.com/design/abc?node-id=1-1",
        route="http://localhost:3000/",
    )
    hints_text = " ".join(envelope["hints"])
    # Severity summary present
    assert "critical" in hints_text.lower() or "Blocked" in hints_text
    # Diagnostics carry severity counts
    diag = envelope["diagnostics"]
    assert "critical_count" in diag
    assert diag["critical_count"] >= 1


def test_check_truncated_dom_emits_hint(mocked_pipeline: Any) -> None:
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), True)  # truncated=True

    envelope, _exit = check_run(
        figma_url="https://figma.com/design/abc?node-id=1-1",
        route="http://localhost:3000/",
    )
    assert any("200-element cap" in h or "narrow with --selectors" in h for h in envelope["hints"])


def test_check_writes_envelope_via_cli(tmp_path: Path, mocked_pipeline: Any) -> None:
    """End-to-end through the CLI surface."""
    from pixel_mcp.cli import app
    from typer.testing import CliRunner

    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)

    runner = CliRunner()
    out = tmp_path / "envelope.json"
    result = runner.invoke(
        app,
        [
            "check",
            "--figma",
            "https://figma.com/design/abc?node-id=1-1",
            "--route",
            "http://localhost:3000/",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    envelope = json.loads(out.read_text())
    assert envelope["data"]["converged"] is True
