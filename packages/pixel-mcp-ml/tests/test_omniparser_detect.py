"""Behavioral tests for the OmniParser detection Deep Module.

Tests do **not** download the real OmniParser weights (~720MB). They
patch ``transformers.AutoModelForObjectDetection.from_pretrained`` and
``transformers.AutoImageProcessor.from_pretrained`` to return mocks with
controllable post-processed detection output. What we verify:

- the raw model output is shaped into :class:`DetectedElement` records
  with the right semantic labels,
- low-confidence detections are filtered out,
- the module-level model cache reuses the loaded model across calls,
- the class-id-to-label mapping table is total over its declared keys,
- a clean :class:`OmniParserNotInstalledError` is raised when the
  optional ``transformers``/``torch`` dependencies are missing,
- bad image paths surface a clear error.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pixel_mcp_ml import omniparser_detect
from pixel_mcp_ml.omniparser_detect import (
    _CLASS_ID_TO_LABEL,
    DetectedElement,
    OmniParserNotInstalledError,
    UILabel,
    detect_ui_elements,
    detect_ui_elements_batch,
)

# ---------------------------------------------------------------------------
# Mock-fabric helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_omniparser_cache() -> Any:
    omniparser_detect._MODEL_CACHE.clear()
    yield
    omniparser_detect._MODEL_CACHE.clear()


def _stub_torch() -> MagicMock:
    """torch stub with CUDA + MPS unavailable, no_grad as ctx manager."""
    torch_stub = MagicMock(name="torch")
    torch_stub.cuda.is_available.return_value = False
    torch_stub.backends.mps.is_available.return_value = False
    torch_stub.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    torch_stub.no_grad.return_value.__exit__ = MagicMock(return_value=None)
    return torch_stub


def _stub_transformers(per_image_raw: list[dict[str, list[Any]]]) -> MagicMock:
    """Build a transformers stub whose processor.post_process_object_detection
    returns ``per_image_raw`` (one dict per image).

    Each dict matches what HuggingFace object-detection post-processors
    emit: ``{"boxes": [...], "scores": [...], "labels": [...]}``.
    """
    transformers_stub = MagicMock(name="transformers")

    processor = MagicMock(name="processor")
    processor.return_value = {"pixel_values": MagicMock(name="pixel_values")}
    processor.post_process_object_detection.return_value = per_image_raw
    transformers_stub.AutoImageProcessor.from_pretrained.return_value = processor

    model = MagicMock(name="model")
    model.to.return_value = model
    model.return_value = MagicMock(name="model_output")
    transformers_stub.AutoModelForObjectDetection.from_pretrained.return_value = model

    return transformers_stub


def _install_mocks(
    monkeypatch: pytest.MonkeyPatch,
    per_image_raw: list[dict[str, list[Any]]],
) -> tuple[MagicMock, MagicMock]:
    transformers_stub = _stub_transformers(per_image_raw)
    torch_stub = _stub_torch()
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "transformers":
            return transformers_stub
        if name == "torch":
            return torch_stub
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(omniparser_detect.importlib, "import_module", fake_import)
    return transformers_stub, torch_stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_detected_elements_with_labels(
    monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any
) -> None:
    """Two raw detections -> two DetectedElement records with mapped labels."""
    raw = [
        {
            "boxes": [[10.0, 20.0, 50.0, 60.0], [100.0, 100.0, 200.0, 200.0]],
            "scores": [0.9, 0.8],
            "labels": [0, 3],  # button, text
        }
    ]
    _install_mocks(monkeypatch, raw)

    image = tiny_image_factory("screen.png")
    detections = detect_ui_elements(image)

    assert len(detections) == 2
    assert all(isinstance(d, DetectedElement) for d in detections)
    labels = {d.label for d in detections}
    assert labels == {"button", "text"}
    # bbox is xywh — first detection: x=10, y=20, w=40, h=40
    first = next(d for d in detections if d.label == "button")
    assert first.bbox == (10.0, 20.0, 40.0, 40.0)


def test_confidence_threshold_filters_low_confidence(
    monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any
) -> None:
    """5 raw detections with threshold 0.5 -> only 2 survive."""
    raw = [
        {
            "boxes": [
                [0, 0, 10, 10],
                [0, 0, 20, 20],
                [0, 0, 30, 30],
                [0, 0, 40, 40],
                [0, 0, 50, 50],
            ],
            "scores": [0.9, 0.6, 0.4, 0.2, 0.05],
            "labels": [0, 1, 2, 3, 4],
        }
    ]
    _install_mocks(monkeypatch, raw)

    image = tiny_image_factory("screen.png")
    detections = detect_ui_elements(image, confidence_threshold=0.5)

    assert len(detections) == 2
    assert all(d.confidence >= 0.5 for d in detections)
    # Output is sorted by confidence desc.
    assert detections[0].confidence >= detections[1].confidence


def test_batch_reuses_model(monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any) -> None:
    """Batch of 3 images -> from_pretrained called exactly once."""
    raw = [
        {"boxes": [[0, 0, 5, 5]], "scores": [0.9], "labels": [0]},
        {"boxes": [[0, 0, 5, 5]], "scores": [0.9], "labels": [1]},
        {"boxes": [[0, 0, 5, 5]], "scores": [0.9], "labels": [2]},
    ]
    transformers_stub, _ = _install_mocks(monkeypatch, raw)

    images = [tiny_image_factory(f"s{i}.png") for i in range(3)]
    results = detect_ui_elements_batch(images)

    assert len(results) == 3
    assert all(len(r) == 1 for r in results)
    model_loader = transformers_stub.AutoModelForObjectDetection.from_pretrained
    proc_loader = transformers_stub.AutoImageProcessor.from_pretrained
    assert model_loader.call_count == 1
    assert proc_loader.call_count == 1


def test_class_id_to_label_mapping_is_total() -> None:
    """Every raw class ID in the lookup table must map to a UILabel literal."""
    # Resolve the runtime values of the UILabel Literal type at import.
    from typing import get_args

    valid_labels = set(get_args(UILabel))
    assert valid_labels, "UILabel must declare at least one literal"
    for class_id, label in _CLASS_ID_TO_LABEL.items():
        assert isinstance(class_id, int)
        assert label in valid_labels, f"{label!r} not in UILabel literal set"


def test_missing_dependency_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any
) -> None:
    """Missing transformers -> OmniParserNotInstalledError with install hint."""

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr(omniparser_detect.importlib, "import_module", fake_import)

    image = tiny_image_factory("screen.png")
    with pytest.raises(OmniParserNotInstalledError) as exc_info:
        detect_ui_elements(image)

    msg = str(exc_info.value)
    assert "transformers" in msg
    assert "--extra omniparser" in msg


def test_missing_torch_dependency_also_raises(
    monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any
) -> None:
    """Same contract when torch is the missing piece."""
    transformers_stub = _stub_transformers([{"boxes": [], "scores": [], "labels": []}])

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "transformers":
            return transformers_stub
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr(omniparser_detect.importlib, "import_module", fake_import)

    image = tiny_image_factory("screen.png")
    with pytest.raises(OmniParserNotInstalledError) as exc_info:
        detect_ui_elements(image)

    assert "torch" in str(exc_info.value)


def test_image_path_not_found_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Bad image path surfaces FileNotFoundError from PIL.Image.open."""
    _install_mocks(monkeypatch, [{"boxes": [], "scores": [], "labels": []}])

    bad = tmp_path / "does-not-exist.png"
    with pytest.raises(FileNotFoundError):
        detect_ui_elements(bad)


def test_empty_batch_returns_empty_list() -> None:
    """No images -> no model load, empty result list."""
    assert detect_ui_elements_batch([]) == []


def test_unknown_class_id_collapses_to_other(
    monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any
) -> None:
    """Class id outside the canonical table maps to ``"other"`` without crashing."""
    raw = [
        {
            "boxes": [[0, 0, 10, 10]],
            "scores": [0.9],
            "labels": [9999],
        }
    ]
    _install_mocks(monkeypatch, raw)

    image = tiny_image_factory("screen.png")
    detections = detect_ui_elements(image)
    assert len(detections) == 1
    assert detections[0].label == "other"


def test_device_detection_prefers_cuda() -> None:
    torch_stub = MagicMock(name="torch")
    torch_stub.cuda.is_available.return_value = True
    torch_stub.backends.mps.is_available.return_value = True
    assert omniparser_detect._detect_device(torch_stub) == "cuda"


def test_device_detection_falls_back_to_mps() -> None:
    torch_stub = MagicMock(name="torch")
    torch_stub.cuda.is_available.return_value = False
    torch_stub.backends.mps.is_available.return_value = True
    assert omniparser_detect._detect_device(torch_stub) == "mps"


def test_device_detection_falls_back_to_cpu() -> None:
    torch_stub = MagicMock(name="torch")
    torch_stub.cuda.is_available.return_value = False
    torch_stub.backends.mps.is_available.return_value = False
    assert omniparser_detect._detect_device(torch_stub) == "cpu"
