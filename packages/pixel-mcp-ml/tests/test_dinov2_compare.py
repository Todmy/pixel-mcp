"""Behavioral tests for the DINOv2 similarity Deep Module.

These tests intentionally avoid downloading the real DINOv2 weights
(~88MB) — instead they patch ``transformers.AutoModel.from_pretrained``
and ``transformers.AutoImageProcessor.from_pretrained`` to return mocks
with controllable embedding output. What we are verifying is:

- the cosine math is correct on known vectors,
- the module-level cache reuses a loaded model across calls,
- the default ``model_size`` value is ``"small"``,
- a clean ``DINOv2NotInstalledError`` is raised when the optional
  ``transformers``/``torch`` dependencies are missing.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pixel_mcp_ml import dinov2_compare
from pixel_mcp_ml.dinov2_compare import (
    DINOv2NotInstalledError,
    compute_dinov2_similarity,
    compute_dinov2_similarity_batch,
)

pytestmark = pytest.mark.usefixtures("clear_model_cache")


# ---------------------------------------------------------------------------
# Mock-fabric helpers
# ---------------------------------------------------------------------------


def _stub_torch() -> MagicMock:
    """Return a torch-like stub. ``no_grad`` works as a context manager,
    ``cuda.is_available`` / ``backends.mps.is_available`` both False, so
    device falls through to CPU.
    """
    torch_stub = MagicMock(name="torch")
    torch_stub.cuda.is_available.return_value = False
    torch_stub.backends.mps.is_available.return_value = False
    torch_stub.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    torch_stub.no_grad.return_value.__exit__ = MagicMock(return_value=None)
    return torch_stub


def _stub_transformers(embeddings: np.ndarray) -> MagicMock:
    """Return a transformers-like stub whose model emits the given
    embeddings as ``outputs.last_hidden_state``.

    ``embeddings`` has shape ``(N, D)`` where N matches the number of
    images that will be passed in one ``processor(...)`` call. The stub
    wraps them into a ``(N, 1, D)`` tensor so that the CLS-token slice
    ``[:, 0, :]`` recovers exactly the input array.
    """
    transformers_stub = MagicMock(name="transformers")

    # Processor: identity-ish — just records the call.
    processor = MagicMock(name="processor")
    processor.return_value = {"pixel_values": MagicMock(name="pixel_values")}
    transformers_stub.AutoImageProcessor.from_pretrained.return_value = processor

    # Model: returns an object whose last_hidden_state.[:, 0, :] yields
    # `embeddings`. We use a tiny helper class to back the indexing.
    class _LastHiddenState:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr.reshape(arr.shape[0], 1, arr.shape[1])

        def __getitem__(self, key: Any) -> Any:
            sliced = self._arr[key]
            tensor = MagicMock(name="tensor")
            tensor.detach.return_value.cpu.return_value.numpy.return_value = sliced
            return tensor

    model = MagicMock(name="model")
    # Chainable .to(device) returns self so model = model.to(...) is happy.
    model.to.return_value = model
    output_obj = MagicMock(name="model_output")
    output_obj.last_hidden_state = _LastHiddenState(embeddings)
    model.return_value = output_obj
    transformers_stub.AutoModel.from_pretrained.return_value = model

    return transformers_stub


def _install_mocks(
    monkeypatch: pytest.MonkeyPatch,
    embeddings: np.ndarray,
) -> tuple[MagicMock, MagicMock]:
    """Patch ``importlib.import_module`` so that ``transformers``/``torch``
    return our stubs while every other module loads normally.
    """
    transformers_stub = _stub_transformers(embeddings)
    torch_stub = _stub_torch()
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "transformers":
            return transformers_stub
        if name == "torch":
            return torch_stub
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(dinov2_compare.importlib, "import_module", fake_import)
    return transformers_stub, torch_stub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_identical_images_similarity_close_to_one(
    monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any
) -> None:
    """When the mocked model returns the same vector for both images,
    cosine similarity must be ~1.0."""
    same_vec = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    _install_mocks(monkeypatch, same_vec)

    a = tiny_image_factory("a.png", (255, 0, 0))
    b = tiny_image_factory("b.png", (0, 255, 0))

    sim = compute_dinov2_similarity(a, b)
    assert sim >= 0.99


def test_dissimilar_images_similarity_lower(
    monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any
) -> None:
    """Orthogonal embeddings -> cosine similarity ~0 (< 0.5)."""
    embeddings = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    _install_mocks(monkeypatch, embeddings)

    a = tiny_image_factory("a.png", (255, 0, 0))
    b = tiny_image_factory("b.png", (0, 0, 255))

    sim = compute_dinov2_similarity(a, b)
    assert sim < 0.5


def test_batch_reuses_model(monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any) -> None:
    """Three pairs (6 images) processed in one batch call. The mocked
    ``from_pretrained`` must be called exactly once for the model and
    once for the processor."""
    n_pairs = 3
    n_images = n_pairs * 2
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n_images, 16)).astype(np.float32)
    transformers_stub, _ = _install_mocks(monkeypatch, embeddings)

    pairs = [
        (tiny_image_factory(f"a{i}.png"), tiny_image_factory(f"b{i}.png")) for i in range(n_pairs)
    ]

    results = compute_dinov2_similarity_batch(pairs)

    assert len(results) == n_pairs
    assert transformers_stub.AutoModel.from_pretrained.call_count == 1
    assert transformers_stub.AutoImageProcessor.from_pretrained.call_count == 1


def test_model_size_small_default() -> None:
    """The default ``model_size`` parameter must be the string ``"small"``."""
    sig = inspect.signature(compute_dinov2_similarity)
    assert sig.parameters["model_size"].default == "small"
    sig_batch = inspect.signature(compute_dinov2_similarity_batch)
    assert sig_batch.parameters["model_size"].default == "small"


def test_missing_dependency_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any
) -> None:
    """If ``transformers`` cannot be imported, we raise
    ``DINOv2NotInstalledError`` (not a bare ImportError) with an
    actionable install hint pointing at the dinov2 extra."""

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr(dinov2_compare.importlib, "import_module", fake_import)

    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")

    with pytest.raises(DINOv2NotInstalledError) as exc_info:
        compute_dinov2_similarity(a, b)

    assert "transformers" in str(exc_info.value)
    assert "pixel-mcp-ml" in str(exc_info.value)
    assert "--extra dinov2" in str(exc_info.value)


def test_missing_torch_dependency_also_raises(
    monkeypatch: pytest.MonkeyPatch, tiny_image_factory: Any
) -> None:
    """Same contract when ``torch`` is the missing piece."""
    transformers_stub = _stub_transformers(np.zeros((2, 4), dtype=np.float32))

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "transformers":
            return transformers_stub
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return importlib.import_module(name, *args, **kwargs)

    monkeypatch.setattr(dinov2_compare.importlib, "import_module", fake_import)

    a = tiny_image_factory("a.png")
    b = tiny_image_factory("b.png")

    with pytest.raises(DINOv2NotInstalledError) as exc_info:
        compute_dinov2_similarity(a, b)

    assert "torch" in str(exc_info.value)


def test_device_detection_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Device-selection unit test — CUDA available means we report 'cuda'."""
    torch_stub = MagicMock(name="torch")
    torch_stub.cuda.is_available.return_value = True
    torch_stub.backends.mps.is_available.return_value = True
    assert dinov2_compare._detect_device(torch_stub) == "cuda"


def test_device_detection_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_stub = MagicMock(name="torch")
    torch_stub.cuda.is_available.return_value = False
    torch_stub.backends.mps.is_available.return_value = True
    assert dinov2_compare._detect_device(torch_stub) == "mps"


def test_device_detection_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    torch_stub = MagicMock(name="torch")
    torch_stub.cuda.is_available.return_value = False
    torch_stub.backends.mps.is_available.return_value = False
    assert dinov2_compare._detect_device(torch_stub) == "cpu"


def test_empty_batch_returns_empty_list() -> None:
    """No pairs means no work — no model load, empty result."""
    assert compute_dinov2_similarity_batch([]) == []
