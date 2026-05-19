"""VLM-based pixel verification — Deep Module.

Public surface:

- :func:`compute_vlm_judgment` — judge a single (expected, actual) crop pair.
- :func:`compute_vlm_judgment_batch` — judge many pairs, reusing the client.
- :class:`VLMJudgment` — structured verdict (Pydantic).
- :class:`VLMNotInstalledError` — raised when the VLM backend SDK is missing.
- :class:`VLMOllamaError` — raised when the local Ollama daemon is unreachable.

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
- ``"qwen-local"`` — POST to a local Ollama daemon (``/api/chat``) running
  ``qwen2.5vl``. Offline / API-cost-free Level 2. Endpoint defaults to
  ``http://localhost:11434`` (override via ``OLLAMA_HOST`` env).

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
import os
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

DEFAULT_QWEN_MODEL = "qwen2.5vl:7b"
"""Default Ollama tag for the Qwen2.5-VL backend.

Override per-call via the ``model=`` parameter. Operators with more VRAM
budget can pull e.g. ``qwen2.5vl:32b`` and pass that explicitly.
"""

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
"""Default Ollama daemon endpoint. Override via ``OLLAMA_HOST`` env."""

_OLLAMA_TIMEOUT_SECONDS = 120.0
"""VLM inference is genuinely slow on local CPU/MPS — give it room. Doctor
checks use a different (1s) timeout; this constant only applies to the
inference path."""

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


class VLMOllamaError(RuntimeError):
    """Raised when the local Ollama daemon is unreachable or unhealthy.

    Distinct from :class:`VLMNotInstalledError` (which is an *SDK* gap):
    here the Python deps are fine, but the runtime service the qwen-local
    backend depends on isn't responding. Message includes the canonical
    setup commands so the operator can self-rescue.
    """

    def __init__(self, detail: str, *, host: str | None = None) -> None:
        host_str = host or DEFAULT_OLLAMA_HOST
        super().__init__(
            f"Ollama call to {host_str} failed: {detail}. "
            "Start it with `ollama serve` (one-time install: "
            "`brew install ollama`), then `ollama pull qwen2.5vl:7b`."
        )
        self.detail = detail
        self.host = host_str


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
    expected_image: str | Path,
    actual_image: str | Path,
    context_label: str | None = None,
) -> list[dict[str, Any]]:
    """Two image blocks + an instruction block, in expected→actual order.

    When ``context_label`` is set (v1.5-2 OmniParser wiring), a short
    semantic prelude is prepended to the user instruction so the VLM can
    bias its verdict toward the known element class (button / input /
    icon / …). Backward compatible: omit ``context_label`` and the prompt
    is byte-for-byte identical to v1.
    """
    exp_type, exp_b64 = _image_to_base64(expected_image)
    act_type, act_b64 = _image_to_base64(actual_image)
    if context_label:
        instruction = (
            f"This region appears to be a `{context_label}`. "
            "Compare the expected and actual versions: "
            "Image 1 above is the EXPECTED crop. Image 2 is the ACTUAL "
            "rendered crop. Judge per system instructions."
        )
    else:
        instruction = (
            "Image 1 above is the EXPECTED crop. Image 2 is the ACTUAL "
            "rendered crop. Judge per system instructions."
        )
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
            "text": instruction,
        },
    ]


def _claude_judgment(
    expected_image: str | Path,
    actual_image: str | Path,
    model: str | None,
    client: Any | None = None,
    context_label: str | None = None,
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
                "content": _build_claude_message_content(
                    expected_image, actual_image, context_label=context_label
                ),
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
# Backend: Qwen local (Ollama + qwen2.5vl)
# ---------------------------------------------------------------------------


def _ollama_host() -> str:
    """Resolve the Ollama endpoint, honouring the ``OLLAMA_HOST`` env var."""
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")


def _build_qwen_messages(
    expected_image: str | Path,
    actual_image: str | Path,
    context_label: str | None = None,
) -> list[dict[str, Any]]:
    """Ollama ``/api/chat`` payload — system + user (with two image b64s).

    Ollama accepts multiple base64 images per message via the ``images``
    list; we don't bother sending media types — Ollama sniffs them from
    the bytes. The user text reuses the same expected→actual framing as
    the Claude path so model prompts stay comparable. When
    ``context_label`` is set (v1.5-2 OmniParser wiring) a semantic
    prelude is prepended so the local model gets the same hint as Claude.
    """
    _, exp_b64 = _image_to_base64(expected_image)
    _, act_b64 = _image_to_base64(actual_image)
    if context_label:
        user_text = (
            f"This region appears to be a `{context_label}`. "
            "Compare the expected and actual versions: "
            "Image 1 is the EXPECTED design crop. Image 2 is the "
            "ACTUAL rendered crop. Judge per system instructions."
        )
    else:
        user_text = (
            "Image 1 is the EXPECTED design crop. Image 2 is the "
            "ACTUAL rendered crop. Judge per system instructions."
        )
    return [
        {"role": "system", "content": _VLM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_text,
            "images": [exp_b64, act_b64],
        },
    ]


def _qwen_judgment(
    expected_image: str | Path,
    actual_image: str | Path,
    model: str | None,
    client: Any | None = None,
    context_label: str | None = None,
) -> VLMJudgment:
    """Single-pair Qwen verdict via Ollama ``/api/chat``.

    ``client`` is a reusable ``httpx.Client`` threaded through by the
    batch helper. When ``None``, we construct a one-shot client here.
    Lazy ``import httpx`` — if pixel-mcp-ml is somehow installed without
    httpx, we surface the standard :class:`VLMNotInstalledError`.
    """
    try:
        httpx = importlib.import_module("httpx")
    except ImportError as exc:
        raise VLMNotInstalledError("httpx") from exc

    chosen_model = model or DEFAULT_QWEN_MODEL
    host = _ollama_host()
    url = f"{host}/api/chat"
    payload = {
        "model": chosen_model,
        "messages": _build_qwen_messages(expected_image, actual_image, context_label=context_label),
        "stream": False,
    }

    owns_client = client is None
    active_client: Any = httpx.Client(timeout=_OLLAMA_TIMEOUT_SECONDS) if owns_client else client
    try:
        try:
            response = active_client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise VLMOllamaError(f"connection refused ({exc})", host=host) from exc
        except httpx.TimeoutException as exc:
            raise VLMOllamaError(f"request timed out ({exc})", host=host) from exc
        except httpx.HTTPError as exc:
            raise VLMOllamaError(
                f"HTTP transport error ({exc.__class__.__name__}: {exc})",
                host=host,
            ) from exc

        if response.status_code != 200:
            raise VLMOllamaError(
                f"non-200 response: HTTP {response.status_code} {response.text[:200]!r}",
                host=host,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise VLMOllamaError(
                f"non-JSON response body: {response.text[:200]!r}", host=host
            ) from exc
    finally:
        if owns_client:
            active_client.close()

    # Ollama returns ``{"message": {"role": "assistant", "content": "..."}, ...}``.
    message = body.get("message") if isinstance(body, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return VLMJudgment(
            verdict="ambiguous",
            confidence=0.0,
            reasoning="Ollama response had no message.content string.",
        )
    return _safe_parse_judgment(content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_vlm_judgment(
    expected_image: str | Path,
    actual_image: str | Path,
    backend: Backend = "claude",
    model: str | None = None,
    context_label: str | None = None,
) -> VLMJudgment:
    """Ask a vision-language model to judge two crops.

    Returns a :class:`VLMJudgment`. Never raises on parse failure — falls
    back to ``ambiguous`` / confidence 0.0. Does raise
    :class:`VLMNotInstalledError` if the chosen backend's SDK is missing,
    and :class:`VLMOllamaError` if the Qwen backend can't reach the
    local Ollama daemon.

    ``context_label`` (v1.5-2): optional semantic hint produced by
    OmniParser (e.g. ``"button"``, ``"input"``). When supplied, the
    user-side prompt is prefixed with a short sentence pointing the VLM
    at the element class. Default ``None`` preserves the v1 prompt.
    """
    if backend == "claude":
        return _claude_judgment(
            expected_image, actual_image, model=model, context_label=context_label
        )
    if backend == "qwen-local":
        return _qwen_judgment(
            expected_image, actual_image, model=model, context_label=context_label
        )
    raise ValueError(f"Unknown VLM backend: {backend!r}")


def compute_vlm_judgment_batch(
    pairs: list[tuple[Path, Path]],
    backend: Backend = "claude",
    model: str | None = None,
    context_labels: list[str | None] | None = None,
) -> list[VLMJudgment]:
    """Batched Level 2 — one backend client construction for N pairs.

    Neither Anthropic nor Ollama exposes a multi-pair batch endpoint with
    the response shape we want, so we loop sequentially but reuse one
    client across pairs (skips per-call TLS setup, env re-reads, and
    HTTP connection pool warmup).

    ``context_labels`` (v1.5-2): optional per-pair OmniParser semantic
    labels, aligned by index with ``pairs``. ``None`` (or a list of
    ``None``s) preserves v1 behaviour. A length mismatch raises
    :class:`ValueError` — caller bug, not a runtime fallback.
    """
    if not pairs:
        return []
    if context_labels is not None and len(context_labels) != len(pairs):
        raise ValueError(
            f"context_labels length {len(context_labels)} != pairs length {len(pairs)}"
        )
    labels: list[str | None] = (
        list(context_labels) if context_labels is not None else [None] * len(pairs)
    )
    if backend == "claude":
        client = _make_anthropic_client()
        out: list[VLMJudgment] = []
        for (expected_image, actual_image), label in zip(pairs, labels, strict=True):
            out.append(
                _claude_judgment(
                    expected_image,
                    actual_image,
                    model=model,
                    client=client,
                    context_label=label,
                )
            )
        return out
    if backend == "qwen-local":
        try:
            httpx = importlib.import_module("httpx")
        except ImportError as exc:
            raise VLMNotInstalledError("httpx") from exc
        out_q: list[VLMJudgment] = []
        with httpx.Client(timeout=_OLLAMA_TIMEOUT_SECONDS) as client:
            for (expected_image, actual_image), label in zip(pairs, labels, strict=True):
                out_q.append(
                    _qwen_judgment(
                        expected_image,
                        actual_image,
                        model=model,
                        client=client,
                        context_label=label,
                    )
                )
        return out_q
    raise ValueError(f"Unknown VLM backend: {backend!r}")
