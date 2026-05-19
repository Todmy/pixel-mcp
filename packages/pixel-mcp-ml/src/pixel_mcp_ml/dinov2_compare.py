"""DINOv2 perceptual similarity — Deep Module.

Public surface intentionally small:

- ``compute_dinov2_similarity(a, b, model_size)`` — cosine similarity of two images.
- ``compute_dinov2_similarity_batch(pairs, model_size)`` — same, batched.

Everything else (model loading, device selection, tensor wrangling) is
private. The module-level cache ``_MODEL_CACHE`` ensures the heavy
``from_pretrained`` call happens once per process per model size.

Design notes:

- ``transformers`` and ``torch`` are *optional* runtime dependencies.
  We import them lazily inside ``_load_model`` so that the package
  remains importable (and the rest of the CLI usable) on installs that
  skipped the ``dinov2`` extra. A clean ``DINOv2NotInstalledError`` is
  raised with an actionable install hint when the user actually needs
  the feature.

- Device priority: CUDA -> MPS -> CPU. MPS catches Apple Silicon dev
  machines; CUDA covers Linux/Windows GPU boxes; CPU is the universal
  fallback.

- Embedding extraction follows the standard DINOv2 recipe — take the
  CLS token (index 0 of ``last_hidden_state``). The original DINOv2
  paper and the HuggingFace model card both recommend this for image-
  level similarity.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

ModelSize = Literal["small", "base"]

_MODEL_NAME: dict[ModelSize, str] = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
}

# Module-level cache. Key: model_size -> (model, processor, device_str).
_MODEL_CACHE: dict[ModelSize, tuple[Any, Any, str]] = {}


class DINOv2NotInstalledError(RuntimeError):
    """Raised when DINOv2 backends (``transformers`` + ``torch``) are missing.

    The error message points at the canonical install command so the
    operator does not have to dig through docs.
    """

    def __init__(self, missing: str) -> None:
        super().__init__(
            f"DINOv2 dependency {missing!r} is not installed. "
            "Install the ML extras: `uv tool install pixel-mcp-ml --extra dinov2`"
        )
        self.missing = missing


def _detect_device(torch_mod: Any) -> str:
    """Return the best available torch device as a string."""
    if torch_mod.cuda.is_available():
        return "cuda"
    # MPS (Apple Silicon) — guard the attribute access; older torch lacks it.
    mps = getattr(torch_mod.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _load_model(model_size: ModelSize) -> tuple[Any, Any, str]:
    """Load (or return cached) model, processor, and chosen device.

    Lazy: ``transformers``/``torch`` are only imported the first time
    this function is called. Subsequent calls hit the cache.
    """
    cached = _MODEL_CACHE.get(model_size)
    if cached is not None:
        return cached

    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise DINOv2NotInstalledError("transformers") from exc
    try:
        torch_mod = importlib.import_module("torch")
    except ImportError as exc:
        raise DINOv2NotInstalledError("torch") from exc

    model_id = _MODEL_NAME[model_size]
    processor = transformers.AutoImageProcessor.from_pretrained(model_id)
    model = transformers.AutoModel.from_pretrained(model_id)

    device = _detect_device(torch_mod)
    model = model.to(device)
    # Switch to inference mode — disables dropout/batch-norm updates.
    model.train(False)

    _MODEL_CACHE[model_size] = (model, processor, device)
    return model, processor, device


def _embed_images(
    images: list[Image.Image],
    model_size: ModelSize,
) -> np.ndarray:
    """Return an ``(N, D)`` numpy array of CLS-token embeddings."""
    model, processor, device = _load_model(model_size)
    torch_mod = importlib.import_module("torch")

    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch_mod.no_grad():
        outputs = model(**inputs)

    # CLS token at index 0 — standard DINOv2 image embedding recipe.
    cls = outputs.last_hidden_state[:, 0, :]
    return cls.detach().cpu().numpy()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically-stable cosine similarity between two 1-D vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _open_rgb(path: str | Path) -> Image.Image:
    """Open an image as RGB (DINOv2 expects 3-channel input)."""
    return Image.open(path).convert("RGB")


def compute_dinov2_similarity(
    image_a: str | Path,
    image_b: str | Path,
    model_size: ModelSize = "small",
) -> float:
    """Cosine similarity in ``[-1, 1]`` of DINOv2 embeddings.

    Higher = more perceptually similar. For perceptually identical
    images the value approaches 1.0; for unrelated images it typically
    sits in ``[0.0, 0.5]`` depending on shared low-level structure.
    """
    img_a = _open_rgb(image_a)
    img_b = _open_rgb(image_b)
    embeddings = _embed_images([img_a, img_b], model_size=model_size)
    return _cosine_similarity(embeddings[0], embeddings[1])


def compute_dinov2_similarity_batch(
    pairs: list[tuple[Path, Path]],
    model_size: ModelSize = "small",
) -> list[float]:
    """Batched cosine similarity — loads the model once for all pairs.

    All images across all pairs go through a single ``processor`` call
    and a single ``model.forward`` pass, then similarities are computed
    pairwise in numpy.
    """
    if not pairs:
        return []
    flat: list[Image.Image] = []
    for a, b in pairs:
        flat.append(_open_rgb(a))
        flat.append(_open_rgb(b))
    embeddings = _embed_images(flat, model_size=model_size)
    return [_cosine_similarity(embeddings[2 * i], embeddings[2 * i + 1]) for i in range(len(pairs))]
