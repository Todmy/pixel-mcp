"""VLM-based pixel verification — Deep Module.

Public surface:

- :func:`compute_vlm_judgment` — judge a single (expected, actual) crop pair.
- :func:`compute_vlm_judgment_batch` — judge many pairs, reusing the client.
- :class:`VLMJudgment` — structured verdict (Pydantic).
- :class:`VLMNotInstalledError` — raised when the VLM backend SDK is missing.

This is the **Level 2 escalation gate** of the pixel-mcp Convergence Loop
(PRD #10). Level 0 = pure CV (SSIM + Hot Regions). Level 1 = DINOv2 cosine
similarity. Level 2 = a vision-language model looks at the expected/actual
crops and renders a verbal verdict — useful for residual ambiguities the
similarity score can't resolve (font swap inside the same colour palette,
icon vs. text node, etc.).

Backend dispatch
----------------

The module exposes a tiny strategy interface through ``backend=``:

- ``"claude"`` (default) — call the Anthropic API. Multi-image messages.
- ``"qwen-local"`` — STUB. Reserved for v1-2 (Ollama + Qwen2.5-VL).

Lazy imports
------------

The Anthropic SDK is **only** imported inside :func:`_claude_judgment` so
the package keeps importing cleanly when the ``vlm`` extra wasn't picked
up at install time. A clean :class:`VLMNotInstalledError` is raised with
an actionable install hint when the user actually needs the feature.

JSON contract
-------------

The Claude backend instructs the model to emit strict JSON of the form
``{"verdict": "match|no_match|ambiguous", "confidence": 0.0-1.0,
"reasoning": "<one sentence>"}``. We tolerate the model wrapping that in
Markdown ``json`` fences and best-effort recover. Anything we can't parse
is folded into a safe-default ``ambiguous`` verdict with confidence
``0.0`` — never a crash.
"""

from __future__ import annotations

import base64
import importlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Backend = Literal["claude", "qwen-local"]

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
"""Default Anthropic model for Level 2 verification.

Per Dmytro's instructions for v1-1 — Claude Sonnet 4.6 is the current
generation. Future bump goes through here.
"""

_VLM_SYSTEM_PROMPT = (
    "You are a strict visual diff judge. You will receive two images: "
    "first the EXPECTED design crop, then the ACTUAL rendered crop. "
    "Decide whether they show the same UI region with negligible visual "
    "difference.\n\n"
    "Respond with STRICT JSON ONLY — no prose, no Markdown fences:\n"
    '{"verdict": "match|no_match|ambiguous", '
    '"confidence": <float 0.0-1.0>, '
    '"reasoning": "<one short sentence>"}\n\n'
    "- match: visually equivalent; any drift is anti-aliasing / sub-pixel.\n"
    "- no_match: clearly different colours, text, layout, or content.\n"
    "- ambiguous: real difference but small or content you can't read.\n"
    "Confidence reflects YOUR certainty in the verdict, not similarity."
)


class VLMNotInstalledError(RuntimeError):
    """Raised when the VLM backend SDK (e.g. ``anthropic``) is missing.

    The error message points at the canonical install command so the
    operator does not have to dig through docs.
    """

    def __init__(self, missing: str) -> None:
        super().__init__(
            f"VLM dependency {missing!r} is not installed. "
            "Install the VLM extras: `uv tool install pixel-mcp-ml --extra vlm`"
        )
        self.missing = missing


class VLMJudgment(BaseModel):
    """Structured Level 2 verdict for a single crop pair."""

    verdict: Literal["match", "no_match", "ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ---------------------------------------------------------------------------
# JSON parsing — tolerant of code-fence wrapping
# ---------------------------------------------------------------------------


def _safe_parse_judgment(raw: str) -> VLMJudgment:
    """Parse the VLM's text into a :class:`VLMJudgment`.

    Recovery order: raw JSON → first ``{...}`` substring → safe default.
    A safe default is ``ambiguous`` with confidence ``0.0`` — never a
    crash. The whole point of the Level 2 gate is to be best-effort: a
    parse failure shouldn't take down the loop.
    """
    candidates: list[str] = [raw.strip()]
    # Strip Markdown code fences if the model wrapped its answer.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))
    # Fallback — first JSON-object-looking blob.
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        try:
            return VLMJudgment.model_validate(obj)
        except Exception:  # noqa: BLE001 — fall through to next candidate
            continue

    return VLMJudgment(
        verdict="ambiguous",
        confidence=0.0,
        reasoning="VLM returned unparseable response — defaulted to ambiguous.",
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _image_to_base64(path: str | Path) -> tuple[str, str]:
    """Return ``(media_type, base64_payload)`` for an image on disk.

    PNG and JPEG are the practical pair for pixel-mcp crops; anything else
    is treated as PNG (Anthropic accepts ``image/png`` for opaque rasters).
    """
    p = Path(path)
    raw = p.read_bytes()
    suffix = p.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        media_type = "image/jpeg"
    elif suffix == ".webp":
        media_type = "image/webp"
    elif suffix == ".gif":
        media_type = "image/gif"
    else:
        media_type = "image/png"
    return media_type, base64.standard_b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Backend: Claude (Anthropic SDK)
# ---------------------------------------------------------------------------


def _build_claude_message_content(
    expected_image: str | Path, actual_image: str | Path
) -> list[dict[str, Any]]:
    """Two image blocks + an instruction block, in expected→actual order."""
    exp_type, exp_b64 = _image_to_base64(expected_image)
    act_type, act_b64 = _image_to_base64(actual_image)
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": exp_type, "data": exp_b64},
        },
        {
            "type": "image",
            "source": {"type": "base64", "media_type": act_type, "data": act_b64},
        },
        {
            "type": "text",
            "text": (
                "Image 1 above is the EXPECTED crop. Image 2 is the ACTUAL "
                "rendered crop. Judge per system instructions."
            ),
        },
    ]


def _claude_judgment(
    expected_image: str | Path,
    actual_image: str | Path,
    model: str | None,
    client: Any | None = None,
) -> VLMJudgment:
    """Single-pair Claude verdict. ``client`` is the reusable Anthropic client.

    A ``None`` ``client`` triggers construction here — the batch helper
    threads one client through every call to avoid repeated TLS setup.
    """
    if client is None:
        client = _make_anthropic_client()
    chosen_model = model or DEFAULT_CLAUDE_MODEL
    response = client.messages.create(
        model=chosen_model,
        max_tokens=512,
        system=_VLM_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": _build_claude_message_content(expected_image, actual_image),
            }
        ],
    )
    # Anthropic responses surface a ``content`` list of blocks. Concatenate
    # any text blocks — typically there's exactly one for a JSON reply.
    text_chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            text_chunks.append(text)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            text_chunks.append(block["text"])
    return _safe_parse_judgment("".join(text_chunks))


def _make_anthropic_client() -> Any:
    """Lazy-import + construct an Anthropic client.

    Raises :class:`VLMNotInstalledError` when the SDK isn't installed.
    Auth follows the SDK default — ``ANTHROPIC_API_KEY`` from the env.
    """
    try:
        anthropic_mod = importlib.import_module("anthropic")
    except ImportError as exc:
        raise VLMNotInstalledError("anthropic") from exc
    return anthropic_mod.Anthropic()


# ---------------------------------------------------------------------------
# Backend: Qwen local — STUB for v1-1
# ---------------------------------------------------------------------------


def _qwen_judgment(
    expected_image: str | Path,
    actual_image: str | Path,
    model: str | None,
) -> VLMJudgment:
    """Placeholder for the local Qwen2.5-VL backend (v1-2 scope).

    Always raises ``NotImplementedError`` with a pointer to the next slice.
    """
    raise NotImplementedError(
        "Qwen local backend lands in v1-2 (Ollama + Qwen2.5-VL). "
        "For v1-1 use backend='claude' with ANTHROPIC_API_KEY set."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_vlm_judgment(
    expected_image: str | Path,
    actual_image: str | Path,
    backend: Backend = "claude",
    model: str | None = None,
) -> VLMJudgment:
    """Ask a vision-language model to judge two crops.

    Returns a :class:`VLMJudgment`. Never raises on parse failure — falls
    back to ``ambiguous`` / confidence 0.0. Does raise
    :class:`VLMNotInstalledError` if the chosen backend's SDK is missing,
    and ``NotImplementedError`` for the ``qwen-local`` STUB.
    """
    if backend == "claude":
        return _claude_judgment(expected_image, actual_image, model=model)
    if backend == "qwen-local":
        return _qwen_judgment(expected_image, actual_image, model=model)
    raise ValueError(f"Unknown VLM backend: {backend!r}")


def compute_vlm_judgment_batch(
    pairs: list[tuple[Path, Path]],
    backend: Backend = "claude",
    model: str | None = None,
) -> list[VLMJudgment]:
    """Batched Level 2 — one Anthropic client construction for N pairs.

    The Anthropic API does not have a multi-pair batch endpoint with the
    response shape we need, so we loop sequentially but reuse the client
    across pairs (skips per-call TLS setup and re-reads of env vars).
    """
    if not pairs:
        return []
    if backend == "qwen-local":
        # Force the same error every other entry point would surface.
        return [_qwen_judgment(a, b, model=model) for a, b in pairs]
    if backend != "claude":
        raise ValueError(f"Unknown VLM backend: {backend!r}")

    client = _make_anthropic_client()
    out: list[VLMJudgment] = []
    for expected_image, actual_image in pairs:
        out.append(_claude_judgment(expected_image, actual_image, model=model, client=client))
    return out
