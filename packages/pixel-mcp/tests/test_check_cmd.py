"""Composite `pixel-mcp check` tests — orchestration of spec + measure + diff + judge."""

from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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
    """Patch the heavy dependencies — Figma + Playwright — at module level.

    Also isolates loop_state to a fresh in-memory IterationState so tests
    don't carry highest_level_reached forward (which would otherwise
    trigger regression detection on the next test).
    """
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
        patch("pixel_mcp.check_cmd.extract_spec") as m_spec,
        patch("pixel_mcp.check_cmd.measure_render") as m_measure,
        patch("pixel_mcp.check_cmd.read_state", side_effect=_read),
        patch("pixel_mcp.check_cmd.write_state", side_effect=_write),
        patch("pixel_mcp.check_cmd.append_history"),
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


# ---------------------------------------------------------------------------
# Level 1 (DINOv2) gate — v0.5-3
# ---------------------------------------------------------------------------


def _regions_with_crops(tmp_path: Path, count: int = 2) -> list[Any]:
    """Build fake Region objects with expected_crop_path / actual_crop_path set.

    The DINOv2 gate iterates ``regions`` — each must expose those attributes
    plus ``leaf_selector``. We don't go through the real decomposer here
    because the gate code only reads the four attributes.
    """
    from pixel_mcp.decompose import Region
    from pixel_mcp.render import BoundingBox

    out: list[Region] = []
    for i in range(count):
        exp = tmp_path / f"exp_{i}.png"
        act = tmp_path / f"act_{i}.png"
        exp.write_bytes(b"fake-expected-png")
        act.write_bytes(b"fake-actual-png")
        out.append(
            Region(
                bbox=BoundingBox(x=0, y=0, w=10, h=10),
                area_px2=100.0,
                severity="minor",
                leaf_selector=f"#region-{i}",
                expected_crop_path=str(exp),
                actual_crop_path=str(act),
            )
        )
    return out


def _patch_visual_signals(regions: list[Any]) -> Any:
    """Patch ``_compute_visual_signals`` to return a passing Level 0 + given regions."""
    return patch(
        "pixel_mcp.check_cmd._compute_visual_signals",
        return_value=(0.99, [], regions, None),
    )


def test_dinov2_gate_skipped_when_disabled(mocked_pipeline: Any, tmp_path: Path) -> None:
    """No DINOv2 import attempted when --enable-dinov2 is off (default)."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path)

    with (
        _patch_visual_signals(regions),
        patch("pixel_mcp.check_cmd._run_dinov2_gate") as m_gate,
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
        )

    assert exit_code == EXIT_CONVERGED
    assert m_gate.call_count == 0
    assert envelope["data"]["dinov2_enabled"] is False
    assert envelope["data"]["dinov2_threshold"] is None
    assert envelope["data"]["dinov2_similarities"] is None
    assert envelope["data"]["level_reached"] == 0  # Level 0 only


def test_dinov2_gate_runs_after_level0_pass(mocked_pipeline: Any, tmp_path: Path) -> None:
    """When --enable-dinov2 and Level 0 passes, all crops scored; level_reached=1."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=2)

    fake_batch = MagicMock(return_value=[0.99, 0.98])
    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.compute_dinov2_similarity_batch = fake_batch  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
        )

    assert exit_code == EXIT_CONVERGED
    assert envelope["data"]["converged"] is True
    assert envelope["data"]["level_reached"] == 1
    assert envelope["data"]["dinov2_enabled"] is True
    assert envelope["data"]["dinov2_threshold"] == pytest.approx(0.95)
    sims = envelope["data"]["dinov2_similarities"]
    assert isinstance(sims, list) and len(sims) == 2
    assert all(s["similarity"] >= 0.95 for s in sims)


def test_dinov2_gate_fails_with_low_similarity(mocked_pipeline: Any, tmp_path: Path) -> None:
    """One low score → overall_converged=False, one pseudo-Delta emitted."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=2)

    fake_batch = MagicMock(return_value=[0.5, 0.99])  # first crop fails (gap=0.45)
    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.compute_dinov2_similarity_batch = fake_batch  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
        )

    assert exit_code == EXIT_DELTAS
    assert envelope["data"]["converged"] is False
    assert envelope["data"]["level_reached"] == 0
    # One pseudo-Delta for the failing crop, with property dinov2_similarity_*
    dinov2_deltas = [
        d for d in envelope["data"]["deltas"] if d["property"].startswith("dinov2_similarity_")
    ]
    assert len(dinov2_deltas) == 1
    assert dinov2_deltas[0]["severity"] == "critical"  # gap 0.45 >= 0.15 → critical


def test_dinov2_gate_graceful_fallback_when_ml_missing(
    mocked_pipeline: Any, tmp_path: Path
) -> None:
    """If pixel_mcp_ml can't be imported, hint surfaces and check doesn't crash."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=1)

    # Force ImportError by sabotaging sys.modules with an object that raises.
    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": None}),  # ImportError on import
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
        )

    # Graceful fallback: Level 0 verdict stands (converged from mocks above),
    # an AXI hint mentions the install command, no crash.
    assert exit_code == EXIT_CONVERGED
    assert envelope["data"]["converged"] is True
    assert envelope["data"]["level_reached"] == 0  # didn't reach Level 1
    hints_text = " ".join(envelope["hints"])
    assert "pixel-mcp-ml" in hints_text
    assert "--extra dinov2" in hints_text


def test_dinov2_gate_batched_loads_model_once(mocked_pipeline: Any, tmp_path: Path) -> None:
    """The gate calls compute_dinov2_similarity_batch ONCE for N crops, not N times."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=3)

    fake_batch = MagicMock(return_value=[0.99, 0.98, 0.97])
    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.compute_dinov2_similarity_batch = fake_batch  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, _exit = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
        )

    assert fake_batch.call_count == 1
    # And it received all 3 (expected, actual) pairs in one call.
    pairs_arg = fake_batch.call_args[0][0]
    assert len(pairs_arg) == 3
    assert envelope["data"]["level_reached"] == 1


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


# ---------------------------------------------------------------------------
# v0.5-4 — Crop persistence wiring through `_compute_visual_signals`
# ---------------------------------------------------------------------------


def _png_bytes(
    color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (200, 200)
) -> bytes:
    """Build a small in-memory PNG so tests don't depend on disk fixtures."""
    import io

    from PIL import Image

    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def real_visual_signals_with_tmp_crops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Wire `_compute_visual_signals` to run for real but with safe upstream stubs.

    Patches the heavy edges — Figma fetch, Playwright capture, and the
    compute_hot_regions/compute_ssim numerics — so the real
    `decompose_hot_regions` (and its disk persistence) executes end-to-end
    inside ``tmp_path``. The whole point of this fixture is to verify the
    ``crops_dir`` + ``iteration`` plumbing actually lands files on disk.
    """
    monkeypatch.chdir(tmp_path)

    expected_png = _png_bytes(color=(255, 0, 0))
    actual_png = _png_bytes(color=(0, 0, 255))

    # Stub Figma fetch — used only when figma_url is set.
    fake_client = MagicMock()
    fake_client.__enter__ = lambda self: fake_client
    fake_client.__exit__ = lambda self, *a: None
    fake_client.fetch_node_png_bytes = MagicMock(return_value=expected_png)

    # Stub Playwright screenshot — used in both modes.
    monkeypatch.setattr(
        "pixel_mcp.check_cmd.capture_screenshot",
        lambda **_: actual_png,
    )

    # Stub the numerics. compute_hot_regions returns one synthetic BoundingBox
    # large enough to clear MIN_BBOX_AREA. compute_ssim returns a passing score.
    fake_bbox = BoundingBox(x=10.0, y=10.0, w=50.0, h=50.0)
    monkeypatch.setattr(
        "pixel_mcp.hot_regions.compute_hot_regions",
        lambda *_a, **_kw: [fake_bbox],
    )
    monkeypatch.setattr(
        "pixel_mcp.hot_regions.compute_ssim",
        lambda *_a, **_kw: 0.99,
    )
    # Stub FigmaClient at its source module so the lazy import inside
    # _compute_visual_signals picks up the stub.
    monkeypatch.setattr(
        "pixel_mcp.figma_client.FigmaClient",
        lambda *a, **kw: fake_client,
    )

    return tmp_path


def test_check_persists_crops_to_state_dir(
    mocked_pipeline: Any,
    real_visual_signals_with_tmp_crops: Path,
) -> None:
    """Figma-mode check writes exp-r*.png + act-r*.png under .pixel-mcp/crops/iter-1/."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)

    tmp_root = real_visual_signals_with_tmp_crops

    envelope, _exit = check_run(
        figma_url="https://figma.com/design/abc?node-id=1-1",
        route="http://localhost:3000/",
    )

    iter_dir = tmp_root / ".pixel-mcp" / "crops" / "iter-1"
    assert iter_dir.is_dir(), f"expected {iter_dir} to exist"
    assert (iter_dir / "exp-r1.png").exists(), "expected expected-crop on disk"
    assert (iter_dir / "act-r1.png").exists(), "expected actual-crop on disk"

    # Envelope must reflect the Region with non-None crop paths.
    regions_payload = envelope["data"].get("regions") or []
    assert any(
        r.get("expected_crop_path") and r.get("actual_crop_path") for r in regions_payload
    ), "Region payload should carry crop paths once persistence is wired"


def test_check_image_only_persists_crops_to_state_dir(
    mocked_pipeline: Any,
    real_visual_signals_with_tmp_crops: Path,
    tmp_path: Path,
) -> None:
    """Image-only mode writes the same .pixel-mcp/crops/iter-1/ pair."""
    _m_spec, m_measure = mocked_pipeline
    m_measure.return_value = (_dom(bg="#ff0000"), False)

    # An --image path on disk so the pre-validation passes.
    img_path = tmp_path / "design.png"
    img_path.write_bytes(_png_bytes())

    tmp_root = real_visual_signals_with_tmp_crops

    check_run(
        image_path=str(img_path),
        route="http://localhost:3000/",
    )

    iter_dir = tmp_root / ".pixel-mcp" / "crops" / "iter-1"
    assert iter_dir.is_dir()
    assert (iter_dir / "exp-r1.png").exists()
    assert (iter_dir / "act-r1.png").exists()


def test_dinov2_gate_scores_persisted_crops(
    mocked_pipeline: Any,
    real_visual_signals_with_tmp_crops: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: persisted crops feed real Region objects with non-None paths into the gate."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)

    # Override the bbox to be just below MIN_BBOX_AREA so Level 0 visual gate
    # passes (no significant hot region) while decompose still emits a Region
    # with crop paths for the DINOv2 gate to score.
    small_bbox = BoundingBox(x=10.0, y=10.0, w=9.0, h=10.0)  # area = 90 < 100
    monkeypatch.setattr(
        "pixel_mcp.hot_regions.compute_hot_regions",
        lambda *_a, **_kw: [small_bbox],
    )

    # The DINOv2 batch fn captures whatever crop paths the gate hands it.
    captured_pairs: list[tuple[str, str]] = []

    def _fake_batch(pairs):
        captured_pairs.extend(pairs)
        return [0.99] * len(pairs)

    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.compute_dinov2_similarity_batch = _fake_batch  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
        )

    # Crops were on disk AND fed into the gate with real, non-None paths.
    assert captured_pairs, "DINOv2 gate should have received at least one crop pair"
    for exp_path, act_path in captured_pairs:
        assert exp_path is not None and act_path is not None
        assert Path(exp_path).exists(), f"exp crop missing on disk: {exp_path}"
        assert Path(act_path).exists(), f"act crop missing on disk: {act_path}"

    assert exit_code == EXIT_CONVERGED
    assert envelope["data"]["level_reached"] == 1


# ---------------------------------------------------------------------------
# Level 2 (VLM) gate — v1-1
# ---------------------------------------------------------------------------


def _vlm_judgment_obj(
    verdict: str = "match", confidence: float = 0.9, reasoning: str = "ok"
) -> Any:
    """Return an object compatible with VLMJudgment (verdict/confidence/reasoning)."""
    from pixel_mcp_ml.vlm_verify import VLMJudgment

    return VLMJudgment(
        verdict=verdict,  # type: ignore[arg-type]
        confidence=confidence,
        reasoning=reasoning,
    )


def test_vlm_gate_skipped_when_disabled(mocked_pipeline: Any, tmp_path: Path) -> None:
    """No VLM import attempted when --enable-vlm is off (default)."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path)

    with (
        _patch_visual_signals(regions),
        patch("pixel_mcp.check_cmd._run_vlm_gate") as m_gate,
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
        )

    assert exit_code == EXIT_CONVERGED
    assert m_gate.call_count == 0
    assert envelope["data"]["vlm_enabled"] is False
    assert envelope["data"]["vlm_threshold"] is None
    assert envelope["data"]["vlm_backend"] is None
    assert envelope["data"]["vlm_judgments"] is None


def test_vlm_gate_runs_after_level1_pass(mocked_pipeline: Any, tmp_path: Path) -> None:
    """When --enable-dinov2 + --enable-vlm and Level 1 passes, Level 2 promotes."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=2)

    fake_dinov2 = MagicMock(return_value=[0.99, 0.98])
    fake_vlm = MagicMock(
        return_value=[
            _vlm_judgment_obj(verdict="match", confidence=0.92, reasoning="ok 1"),
            _vlm_judgment_obj(verdict="match", confidence=0.85, reasoning="ok 2"),
        ]
    )
    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.compute_dinov2_similarity_batch = fake_dinov2  # type: ignore[attr-defined]
    fake_pkg.compute_vlm_judgment_batch = fake_vlm  # type: ignore[attr-defined]

    # Real VLMNotInstalledError class so the gate's `except VLMNotInstalledError`
    # statement keeps working when the fake package is in sys.modules.
    from pixel_mcp_ml.vlm_verify import VLMNotInstalledError

    fake_pkg.VLMNotInstalledError = VLMNotInstalledError  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
            enable_vlm=True,
        )

    assert exit_code == EXIT_CONVERGED
    assert envelope["data"]["converged"] is True
    assert envelope["data"]["level_reached"] == 2
    assert envelope["data"]["vlm_enabled"] is True
    assert envelope["data"]["vlm_threshold"] == pytest.approx(0.7)
    assert envelope["data"]["vlm_backend"] == "claude"
    judgments = envelope["data"]["vlm_judgments"]
    assert isinstance(judgments, list) and len(judgments) == 2
    assert all(j["verdict"] == "match" for j in judgments)


def test_vlm_gate_fails_with_no_match(mocked_pipeline: Any, tmp_path: Path) -> None:
    """Any ``no_match`` verdict → pseudo-Delta + level_reached stays at 1."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=2)

    fake_dinov2 = MagicMock(return_value=[0.99, 0.98])
    fake_vlm = MagicMock(
        return_value=[
            _vlm_judgment_obj(verdict="no_match", confidence=0.95, reasoning="different colour"),
            _vlm_judgment_obj(verdict="match", confidence=0.9, reasoning="ok"),
        ]
    )
    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.compute_dinov2_similarity_batch = fake_dinov2  # type: ignore[attr-defined]
    fake_pkg.compute_vlm_judgment_batch = fake_vlm  # type: ignore[attr-defined]
    from pixel_mcp_ml.vlm_verify import VLMNotInstalledError

    fake_pkg.VLMNotInstalledError = VLMNotInstalledError  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
            enable_vlm=True,
        )

    assert exit_code == EXIT_DELTAS
    assert envelope["data"]["converged"] is False
    assert envelope["data"]["level_reached"] == 1  # Level 1 passed, Level 2 didn't
    vlm_deltas = [d for d in envelope["data"]["deltas"] if d["property"].startswith("vlm_verdict_")]
    assert len(vlm_deltas) == 1
    assert vlm_deltas[0]["severity"] == "critical"  # gap 1.0 → critical


def test_vlm_gate_fails_when_confidence_below_threshold(
    mocked_pipeline: Any, tmp_path: Path
) -> None:
    """``match`` verdict with low confidence → pseudo-Delta."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=1)

    fake_dinov2 = MagicMock(return_value=[0.99])
    fake_vlm = MagicMock(
        return_value=[_vlm_judgment_obj(verdict="match", confidence=0.3, reasoning="unsure")]
    )
    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.compute_dinov2_similarity_batch = fake_dinov2  # type: ignore[attr-defined]
    fake_pkg.compute_vlm_judgment_batch = fake_vlm  # type: ignore[attr-defined]
    from pixel_mcp_ml.vlm_verify import VLMNotInstalledError

    fake_pkg.VLMNotInstalledError = VLMNotInstalledError  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
            enable_vlm=True,
            vlm_threshold=0.7,
        )

    assert exit_code == EXIT_DELTAS
    assert envelope["data"]["converged"] is False
    assert envelope["data"]["level_reached"] == 1
    vlm_deltas = [d for d in envelope["data"]["deltas"] if d["property"].startswith("vlm_verdict_")]
    assert len(vlm_deltas) == 1
    # gap = 0.7 - 0.3 = 0.4 → critical
    assert vlm_deltas[0]["severity"] == "critical"


def test_vlm_gate_graceful_fallback_when_extras_missing(
    mocked_pipeline: Any, tmp_path: Path
) -> None:
    """Level 2 hint surfaces, Level 1 verdict stands, no crash."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=1)

    # Provide a DINOv2 path through pixel_mcp_ml but NO VLM symbols.
    fake_dinov2 = MagicMock(return_value=[0.99])
    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.compute_dinov2_similarity_batch = fake_dinov2  # type: ignore[attr-defined]

    # Make `from pixel_mcp_ml import compute_vlm_judgment_batch, VLMNotInstalledError`
    # raise ImportError by deleting those attributes — Python's ``from`` lookup
    # on a module without the name raises ImportError, mirroring the missing-extra
    # path the gate guards against.

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
            enable_vlm=True,
        )

    assert exit_code == EXIT_CONVERGED
    assert envelope["data"]["converged"] is True
    # Level 1 stands; Level 2 did NOT promote.
    assert envelope["data"]["level_reached"] == 1
    hints_text = " ".join(envelope["hints"])
    assert "Level 2" in hints_text
    assert "vlm" in hints_text.lower()


def test_vlm_gate_not_run_when_level1_fails(mocked_pipeline: Any, tmp_path: Path) -> None:
    """If Level 1 (DINOv2) failed, Level 2 must NOT execute — saves API calls."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=1)

    fake_dinov2 = MagicMock(return_value=[0.5])  # below 0.95 → Level 1 fails
    fake_vlm = MagicMock()  # should NEVER be called
    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.compute_dinov2_similarity_batch = fake_dinov2  # type: ignore[attr-defined]
    fake_pkg.compute_vlm_judgment_batch = fake_vlm  # type: ignore[attr-defined]
    from pixel_mcp_ml.vlm_verify import VLMNotInstalledError

    fake_pkg.VLMNotInstalledError = VLMNotInstalledError  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
            enable_vlm=True,
        )

    assert exit_code == EXIT_DELTAS
    assert envelope["data"]["level_reached"] == 0
    assert fake_vlm.call_count == 0  # the whole point of the test


# ---------------------------------------------------------------------------
# Level 3 (Human review) gate — v1-3
# ---------------------------------------------------------------------------


def _write_feedback(
    sd: Path, *, verdict: str, notes: str | None = None, consumed: bool = False
) -> None:
    """Pre-stage a human-feedback record on disk."""
    sd.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "verdict": verdict,
        "notes": notes,
        "captured_at": "2026-05-19T00:00:00+00:00",
        "consumed": consumed,
    }
    (sd / "human-feedback.json").write_text(json.dumps(payload))


def test_human_gate_disabled_skips_review(mocked_pipeline: Any, tmp_path: Path) -> None:
    """enable_human_gate=False → no feedback file probe, exit 0 immediately."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path)

    with _patch_visual_signals(regions):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
        )

    assert exit_code == EXIT_CONVERGED
    assert envelope["data"]["human_gate_enabled"] is False
    assert envelope["data"]["human_verdict"] is None
    assert envelope["data"]["level_reached"] == 0


def test_human_gate_enabled_returns_pending_when_no_feedback(
    mocked_pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """enable_human_gate=True with no feedback file → EXIT_READY_FOR_LEVEL_3."""
    from pixel_mcp.check_cmd import EXIT_READY_FOR_LEVEL_3

    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path)

    # Isolate state_dir so the test never picks up a real human-feedback.json.
    sd = tmp_path / ".pixel-mcp"
    monkeypatch.setattr("pixel_mcp.human_feedback_cmd.state_dir", lambda *_a, **_k: sd)

    with _patch_visual_signals(regions):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_human_gate=True,
        )

    assert exit_code == EXIT_READY_FOR_LEVEL_3
    assert envelope["data"]["human_gate_enabled"] is True
    assert envelope["data"]["human_verdict"] == "pending"
    # next_suggested_action must point at the review tool.
    assert envelope["next_suggested_action"] is not None
    assert "review" in envelope["next_suggested_action"].lower()


def test_human_gate_approved_reaches_final_convergence(
    mocked_pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-staged approved feedback → exit 0, level_reached=3, file marked consumed."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path)

    sd = tmp_path / ".pixel-mcp"
    _write_feedback(sd, verdict="approved")
    monkeypatch.setattr("pixel_mcp.human_feedback_cmd.state_dir", lambda *_a, **_k: sd)

    with _patch_visual_signals(regions):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_human_gate=True,
        )

    assert exit_code == EXIT_CONVERGED
    assert envelope["data"]["converged"] is True
    assert envelope["data"]["level_reached"] == 3
    assert envelope["data"]["human_verdict"] == "approved"
    # File flipped to consumed=true so a second invocation treats it as pending.
    after = json.loads((sd / "human-feedback.json").read_text())
    assert after["consumed"] is True


def test_human_gate_rejected_emits_pseudo_delta(
    mocked_pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rejection → critical pseudo-Delta property=human_review, exit 1."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path)

    sd = tmp_path / ".pixel-mcp"
    notes = "Margin on hero too tight; button colour drift."
    _write_feedback(sd, verdict="rejected", notes=notes)
    monkeypatch.setattr("pixel_mcp.human_feedback_cmd.state_dir", lambda *_a, **_k: sd)

    with _patch_visual_signals(regions):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_human_gate=True,
        )

    assert exit_code == EXIT_DELTAS
    assert envelope["data"]["converged"] is False
    assert envelope["data"]["human_verdict"] == "rejected"
    assert envelope["data"]["human_notes"] == notes
    human_deltas = [d for d in envelope["data"]["deltas"] if d["property"] == "human_review"]
    assert len(human_deltas) == 1
    delta = human_deltas[0]
    assert delta["severity"] == "critical"
    assert delta["expected"] == notes
    # Consumed flipped to true to prevent re-application on the next iteration.
    after = json.loads((sd / "human-feedback.json").read_text())
    assert after["consumed"] is True


def test_human_gate_consumed_feedback_treated_as_pending(
    mocked_pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Already-consumed feedback file → loop pauses again (EXIT_READY_FOR_LEVEL_3)."""
    from pixel_mcp.check_cmd import EXIT_READY_FOR_LEVEL_3

    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path)

    sd = tmp_path / ".pixel-mcp"
    _write_feedback(sd, verdict="approved", consumed=True)
    monkeypatch.setattr("pixel_mcp.human_feedback_cmd.state_dir", lambda *_a, **_k: sd)

    with _patch_visual_signals(regions):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_human_gate=True,
        )

    assert exit_code == EXIT_READY_FOR_LEVEL_3
    assert envelope["data"]["human_verdict"] == "pending"
    # The approved record was NOT re-applied: level stays below 3.
    assert envelope["data"]["level_reached"] < 3


def test_human_gate_not_run_when_lower_level_fails(
    mocked_pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If Level 0 already fails, the human gate must NOT pause the loop."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    # Inject a colour delta so Level 0 (structured) fails immediately.
    m_measure.return_value = (_dom(bg="#00ff00"), False)

    sd = tmp_path / ".pixel-mcp"
    _write_feedback(sd, verdict="approved")
    monkeypatch.setattr("pixel_mcp.human_feedback_cmd.state_dir", lambda *_a, **_k: sd)

    envelope, exit_code = check_run(
        figma_url="https://figma.com/design/abc?node-id=1-1",
        route="http://localhost:3000/",
        enable_human_gate=True,
    )

    assert exit_code == EXIT_DELTAS
    # Gate enabled, but verdict not applied because automated gate failed first.
    assert envelope["data"]["human_gate_enabled"] is True
    assert envelope["data"]["human_verdict"] is None
    # Feedback file must NOT have been consumed.
    after = json.loads((sd / "human-feedback.json").read_text())
    assert after["consumed"] is False


# ---------------------------------------------------------------------------
# v1.5-2 — OmniParser semantic-label augmentation
# ---------------------------------------------------------------------------


def _detected_element_obj(
    bbox: tuple[float, float, float, float],
    label: str = "button",
    confidence: float = 0.85,
) -> Any:
    """Build a real :class:`DetectedElement` so `.model_dump_json()` works."""
    from pixel_mcp_ml.omniparser_detect import DetectedElement

    return DetectedElement(
        bbox=bbox,
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
    )


def _seed_actual_screenshot(tmp_path: Path, iteration: int = 1) -> Path:
    """Pre-place a one-pixel actual.png so OmniParser's existence check passes.

    The actual bytes don't matter — `detect_ui_elements` is monkeypatched.
    """
    from PIL import Image

    iter_dir = tmp_path / ".pixel-mcp" / "crops" / f"iter-{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    out = iter_dir / "actual.png"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(out)
    return out


def _patch_state_dir_to(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route check_cmd's ``state_dir`` lookups at a sandboxed tmp_path."""
    monkeypatch.setattr(
        "pixel_mcp.check_cmd.state_dir",
        lambda *_a, **_k: tmp_path / ".pixel-mcp",
    )


def test_omniparser_disabled_leaves_regions_bare(mocked_pipeline: Any, tmp_path: Path) -> None:
    """Default (enable_omniparser=False) → Region.semantic_label stays None."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=1)

    with _patch_visual_signals(regions):
        envelope, _exit = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
        )

    assert envelope["data"]["omniparser_enabled"] is False
    assert envelope["data"]["omniparser_detections"] is None
    for r in envelope["data"]["regions"]:
        assert r["semantic_label"] is None
        assert r["semantic_confidence"] is None


def test_omniparser_enriches_regions_with_labels(
    mocked_pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detection covering a Region's centre → semantic_label attached."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=1)
    # Region #0 bbox is (0, 0, 10, 10) — centre (5, 5).
    detections = [_detected_element_obj((0.0, 0.0, 20.0, 20.0), label="button", confidence=0.87)]
    _patch_state_dir_to(tmp_path, monkeypatch)
    _seed_actual_screenshot(tmp_path, iteration=1)

    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.detect_ui_elements = MagicMock(return_value=detections)  # type: ignore[attr-defined]
    from pixel_mcp_ml.omniparser_detect import OmniParserNotInstalledError

    fake_pkg.OmniParserNotInstalledError = OmniParserNotInstalledError  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, _exit = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_omniparser=True,
        )

    assert envelope["data"]["omniparser_enabled"] is True
    detections_out = envelope["data"]["omniparser_detections"]
    assert isinstance(detections_out, list) and len(detections_out) == 1
    assert detections_out[0]["label"] == "button"
    region_out = envelope["data"]["regions"][0]
    assert region_out["semantic_label"] == "button"
    assert region_out["semantic_confidence"] == pytest.approx(0.87)


def test_omniparser_no_match_leaves_region_label_none(
    mocked_pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detections that don't overlap any Region centre → semantic_label=None."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=2)
    # All Region bboxes are (0,0,10,10); detection at (500,500,20,20) misses.
    detections = [
        _detected_element_obj((500.0, 500.0, 20.0, 20.0), label="icon", confidence=0.91),
    ]
    _patch_state_dir_to(tmp_path, monkeypatch)
    _seed_actual_screenshot(tmp_path, iteration=1)

    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.detect_ui_elements = MagicMock(return_value=detections)  # type: ignore[attr-defined]
    from pixel_mcp_ml.omniparser_detect import OmniParserNotInstalledError

    fake_pkg.OmniParserNotInstalledError = OmniParserNotInstalledError  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, _exit = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_omniparser=True,
        )

    assert envelope["data"]["omniparser_enabled"] is True
    assert envelope["data"]["omniparser_detections"] is not None
    for r in envelope["data"]["regions"]:
        assert r["semantic_label"] is None
        assert r["semantic_confidence"] is None


def test_omniparser_graceful_fallback_when_extras_missing(
    mocked_pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ImportError on `pixel_mcp_ml` → AXI hint, no crash, no detections."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=1)
    _patch_state_dir_to(tmp_path, monkeypatch)
    _seed_actual_screenshot(tmp_path, iteration=1)

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": None}),  # ImportError on import
    ):
        envelope, exit_code = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_omniparser=True,
        )

    assert exit_code == EXIT_CONVERGED  # no crash
    assert envelope["data"]["omniparser_enabled"] is True
    assert envelope["data"]["omniparser_detections"] is None
    hints_text = " ".join(envelope["hints"])
    assert "OmniParser" in hints_text
    assert "--extra omniparser" in hints_text


def test_vlm_prompt_includes_semantic_label_when_present(
    mocked_pipeline: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OmniParser-tagged Region → semantic label threads into the VLM prompt.

    Patches `compute_vlm_judgment_batch` to capture the kwargs and asserts
    the ``context_labels`` keyword carries the expected label string.
    """
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=1)
    detections = [_detected_element_obj((0.0, 0.0, 20.0, 20.0), label="button", confidence=0.9)]
    _patch_state_dir_to(tmp_path, monkeypatch)
    _seed_actual_screenshot(tmp_path, iteration=1)

    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.detect_ui_elements = MagicMock(return_value=detections)  # type: ignore[attr-defined]
    fake_pkg.compute_dinov2_similarity_batch = MagicMock(return_value=[0.99])  # type: ignore[attr-defined]
    fake_vlm = MagicMock(
        return_value=[_vlm_judgment_obj(verdict="match", confidence=0.9, reasoning="ok")]
    )
    fake_pkg.compute_vlm_judgment_batch = fake_vlm  # type: ignore[attr-defined]
    from pixel_mcp_ml.omniparser_detect import OmniParserNotInstalledError
    from pixel_mcp_ml.vlm_verify import VLMNotInstalledError

    fake_pkg.OmniParserNotInstalledError = OmniParserNotInstalledError  # type: ignore[attr-defined]
    fake_pkg.VLMNotInstalledError = VLMNotInstalledError  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        envelope, _exit = check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_omniparser=True,
            enable_dinov2=True,
            enable_vlm=True,
        )

    assert fake_vlm.call_count == 1
    kwargs = fake_vlm.call_args.kwargs
    assert "context_labels" in kwargs
    assert kwargs["context_labels"] == ["button"]
    # And the region itself made the round-trip with the label attached.
    region_out = envelope["data"]["regions"][0]
    assert region_out["semantic_label"] == "button"


def test_match_detected_to_region_picks_highest_iou_inside_centre() -> None:
    """Unit test for the matcher: centre-containment + highest-IoU tie-breaker."""
    from pixel_mcp.check_cmd import _match_detected_to_region
    from pixel_mcp.decompose import Region
    from pixel_mcp.render import BoundingBox

    region = Region(
        bbox=BoundingBox(x=0, y=0, w=10, h=10),
        area_px2=100.0,
        severity="minor",
    )
    # Two detections both contain centre (5, 5); the tighter one (higher IoU) wins.
    big = _detected_element_obj((0.0, 0.0, 100.0, 100.0), label="container", confidence=0.95)
    tight = _detected_element_obj((0.0, 0.0, 10.0, 10.0), label="button", confidence=0.80)
    match = _match_detected_to_region(region, [big, tight])
    assert match is tight  # IoU=1.0 beats IoU≈0.01

    # No detection covers the centre → None.
    far = _detected_element_obj((500.0, 500.0, 10.0, 10.0), label="icon", confidence=0.99)
    assert _match_detected_to_region(region, [far]) is None


def test_vlm_prompt_unchanged_when_semantic_label_absent(
    mocked_pipeline: Any, tmp_path: Path
) -> None:
    """No OmniParser labels → VLM gate calls the v1 entry-point (no context_labels)."""
    m_spec, m_measure = mocked_pipeline
    m_spec.return_value = _spec()
    m_measure.return_value = (_dom(bg="#ff0000"), False)
    regions = _regions_with_crops(tmp_path, count=1)

    fake_dinov2 = MagicMock(return_value=[0.99])
    fake_vlm = MagicMock(
        return_value=[_vlm_judgment_obj(verdict="match", confidence=0.9, reasoning="ok")]
    )
    fake_pkg = types.ModuleType("pixel_mcp_ml")
    fake_pkg.compute_dinov2_similarity_batch = fake_dinov2  # type: ignore[attr-defined]
    fake_pkg.compute_vlm_judgment_batch = fake_vlm  # type: ignore[attr-defined]
    from pixel_mcp_ml.vlm_verify import VLMNotInstalledError

    fake_pkg.VLMNotInstalledError = VLMNotInstalledError  # type: ignore[attr-defined]

    with (
        _patch_visual_signals(regions),
        patch.dict(sys.modules, {"pixel_mcp_ml": fake_pkg}),
    ):
        check_run(
            figma_url="https://figma.com/design/abc?node-id=1-1",
            route="http://localhost:3000/",
            enable_dinov2=True,
            enable_vlm=True,
        )

    # Backward-compat path: no context_labels kwarg passed.
    assert fake_vlm.call_count == 1
    assert "context_labels" not in fake_vlm.call_args.kwargs
