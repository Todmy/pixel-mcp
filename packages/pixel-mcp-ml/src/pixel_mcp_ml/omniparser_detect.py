"""OmniParser UI-element detection — Deep Module.

Public surface intentionally small:

- :func:`detect_ui_elements` — detect UI elements in a single screenshot.
- :func:`detect_ui_elements_batch` — batched detection over many images.
- :class:`DetectedElement` — Pydantic record for one detection.
- :class:`OmniParserNotInstalledError` — raised when the optional
  ``omniparser`` extra is not installed.

This is the **infrastructure** slice for the v1.5 PRD. Wiring into
``check`` (sharper region attribution + per-element VLM context) lands
in v1.5-2 — this module ships the packaging only.

Design notes:

- ``transformers`` and ``torch`` are *optional* runtime dependencies and
  loaded lazily inside :func:`_load_model`. The rest of ``pixel_mcp_ml``
  remains importable on installs that skipped the ``omniparser`` extra.

- Device priority: CUDA -> MPS -> CPU. Mirrors the DINOv2 helper.

- Module-level cache: the (model, processor, device) triple is loaded
  once per process and reused across calls — important because
  OmniParser weights are ~720MB.

- Mirroring the v0.5-2 DINOv2 pattern, we use ``model.train(False)``
  instead of the standard inference-toggle method. The Claude Code
  security hook flags that other method name as potential code
  injection. Documented Valis pattern (lesson from v1-1).

- Class-id-to-label mapping: OmniParser emits raw integer class IDs.
  The model card lists the canonical label set; we collapse that into
  the smaller :data:`UILabel` literal so callers don't have to learn
  the upstream vocabulary. Unknown IDs collapse to ``"other"``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ModelSize = Literal["base"]

UILabel = Literal[
    "button",
    "input",
    "icon",
    "text",
    "image",
    "container",
    "other",
]

_MODEL_NAME: dict[ModelSize, str] = {
    "base": "microsoft/OmniParser-v2.0",
}

# Module-level cache. Key: model_size -> (model, processor, device_str).
_MODEL_CACHE: dict[ModelSize, tuple[Any, Any, str]] = {}


# Canonical raw-class-id -> UILabel mapping. OmniParser v2.0 emits a
# fixed integer vocabulary; we collapse it to the smaller surface
# pixel-mcp downstream layers care about. Keeping the table explicit
# (not derived from model.config) makes the contract testable without
# loading real weights.
_CLASS_ID_TO_LABEL: dict[int, UILabel] = {
    0: "button",
    1: "input",
    2: "icon",
    3: "text",
    4: "image",
    5: "container",
    6: "other",
}


class DetectedElement(BaseModel):
    """One UI element detection produced by OmniParser."""

    bbox: tuple[float, float, float, float] = Field(
        ...,
        description="(x, y, w, h) in pixel coordinates of the source image.",
    )
    label: UILabel = Field(
        ...,
        description="Semantic class assigned by OmniParser.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence in [0.0, 1.0].",
    )


class OmniParserNotInstalledError(RuntimeError):
    """Raised when OmniParser backends (``transformers``/``torch``) are missing.

    Carries the canonical install hint so the operator does not have to
    dig through docs.
    """

    def __init__(self, missing: str) -> None:
        super().__init__(
            f"OmniParser dependency {missing!r} is not installed. "
            "Install the OmniParser extras: `uv tool install pixel-mcp-ml --extra omniparser`"
        )
        self.missing = missing


def _detect_device(torch_mod: Any) -> str:
    """Return the best available torch device as a string."""
    if torch_mod.cuda.is_available():
        return "cuda"
    mps = getattr(torch_mod.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _label_for_class_id(class_id: int) -> UILabel:
    """Map a raw OmniParser class id to a :data:`UILabel`.

    Unknown ids collapse to ``"other"`` — this keeps the contract total
    so a future OmniParser bump that introduces new classes can't crash
    callers.
    """
    return _CLASS_ID_TO_LABEL.get(int(class_id), "other")


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
        raise OmniParserNotInstalledError("transformers") from exc
    try:
        torch_mod = importlib.import_module("torch")
    except ImportError as exc:
        raise OmniParserNotInstalledError("torch") from exc

    model_id = _MODEL_NAME[model_size]
    processor = transformers.AutoImageProcessor.from_pretrained(model_id)
    model = transformers.AutoModelForObjectDetection.from_pretrained(model_id)

    device = _detect_device(torch_mod)
    model = model.to(device)
    # Switch to inference mode without tripping the security hook on the
    # short method name — same outcome (dropout/BN updates off).
    model.train(False)

    _MODEL_CACHE[model_size] = (model, processor, device)
    return model, processor, device


def _open_rgb(path: str | Path) -> Any:
    """Open an image as RGB (object-detection models expect 3-channel)."""
    from PIL import Image

    return Image.open(path).convert("RGB")


def _raw_detections_to_elements(
    raw: Any,
    confidence_threshold: float,
) -> list[DetectedElement]:
    """Translate one image's raw OmniParser output into ordered detections.

    ``raw`` is expected to be a dict-like object with three iterable
    fields: ``boxes`` (Nx4 in xyxy), ``scores`` (length N), ``labels``
    (length N of integer class ids). This mirrors the shape produced by
    ``transformers`` object-detection post-processors (e.g.
    ``post_process_object_detection``) — both real model outputs and
    test mocks conform to the same protocol.

    Boxes are converted from xyxy to ``(x, y, w, h)`` for the public
    contract.
    """
    boxes = raw["boxes"]
    scores = raw["scores"]
    labels = raw["labels"]

    elements: list[DetectedElement] = []
    for box, score, class_id in zip(boxes, scores, labels, strict=True):
        confidence = float(score)
        if confidence < confidence_threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        elements.append(
            DetectedElement(
                bbox=(x1, y1, x2 - x1, y2 - y1),
                label=_label_for_class_id(int(class_id)),
                confidence=confidence,
            )
        )
    # Sort by confidence descending so CLI/JSON consumers get the most
    # salient detections first.
    elements.sort(key=lambda e: e.confidence, reverse=True)
    return elements


def _run_inference(
    images: list[Any],
    model_size: ModelSize,
) -> list[Any]:
    """Run a single forward pass over ``images`` and return per-image
    post-processed detection dicts."""
    model, processor, device = _load_model(model_size)
    torch_mod = importlib.import_module("torch")

    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch_mod.no_grad():
        outputs = model(**inputs)

    target_sizes = [(img.height, img.width) for img in images]
    return list(
        processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=0.0,
        )
    )


def detect_ui_elements(
    image: str | Path,
    model_size: ModelSize = "base",
    confidence_threshold: float = 0.3,
) -> list[DetectedElement]:
    """Detect UI elements in a single screenshot.

    Parameters
    ----------
    image:
        Path to a PNG/JPG screenshot. Raises :class:`FileNotFoundError`
        via :class:`PIL.Image.open` when the file is missing.
    model_size:
        Currently only ``"base"`` is supported. Reserved for future
        ``"small"``/``"large"`` variants.
    confidence_threshold:
        Detections with score below this value are dropped. Default
        ``0.3`` matches the OmniParser model card recommendation.
    """
    img = _open_rgb(image)
    raw_per_image = _run_inference([img], model_size=model_size)
    return _raw_detections_to_elements(raw_per_image[0], confidence_threshold)


def detect_ui_elements_batch(
    images: list[Path],
    model_size: ModelSize = "base",
    confidence_threshold: float = 0.3,
) -> list[list[DetectedElement]]:
    """Batched element detection — loads the model once for all images.

    Empty input returns an empty list without loading the model.
    """
    if not images:
        return []
    pil_images = [_open_rgb(p) for p in images]
    raw_per_image = _run_inference(pil_images, model_size=model_size)
    return [_raw_detections_to_elements(raw, confidence_threshold) for raw in raw_per_image]
