"""Behavioural tests for the VLM verification Deep Module.

These tests **never make real Anthropic or Ollama calls** — every test
patches either ``importlib.import_module`` (to inject a fake
``anthropic`` module), ``_make_anthropic_client`` (to inject a pre-built
mock client), or ``httpx.Client`` (for the Qwen local backend). The
real SDKs / daemons are never reached.

What we verify:

- Claude backend sends one request per pair with two image blocks.
- Valid JSON in the response becomes a typed :class:`VLMJudgment`.
- Malformed responses degrade gracefully to ``ambiguous`` /
  confidence 0.0 (no crash).
- Qwen local backend POSTs to ``/api/chat`` with both base64 images.
- Qwen connection refused → :class:`VLMOllamaError` with install hint.
- ``OLLAMA_HOST`` env overrides the endpoint.
- Missing ``anthropic`` SDK raises :class:`VLMNotInstalledError`.
- Batch reuses one client across N pairs (both backends).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pixel_mcp_ml import vlm_verify
from pixel_mcp_ml.vlm_verify import (
    VLMJudgment,
    VLMNotInstalledError,
    VLMOllamaError,
    compute_vlm_judgment,
    compute_vlm_judgment_batch,
)

# ---------------------------------------------------------------------------
# Mock fabric — fake ``anthropic`` module + client + response
# ---------------------------------------------------------------------------


def _fake_response(text: str) -> MagicMock:
    """Build an Anthropic-like response whose ``content[0].text == text``."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    response_text: str,
) -> MagicMock:
    """Patch ``importlib.import_module`` so that ``anthropic`` returns a stub
    whose ``Anthropic()`` builds a client whose ``.messages.create(...)``
    returns a controlled response. Returns the constructed client.
    """
    client = MagicMock(name="anthropic_client")
    client.messages.create.return_value = _fake_response(response_text)

    anthropic_stub = MagicMock(name="anthropic")
    anthropic_stub.Anthropic.return_value = client

    real_import = importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            return anthropic_stub
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(vlm_verify.importlib, "import_module", fake_import)
    return client


@pytest.fixture
def crop_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two tiny solid-colour PNGs on disk."""
    import numpy as np
    from PIL import Image

    a = tmp_path / "expected.png"
    b = tmp_path / "actual.png"
    Image.fromarray(np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8)).save(a)
    Image.fromarray(np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8)).save(b)
    return a, b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_claude_backend_calls_anthropic_with_two_images(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    client = _install_fake_anthropic(
        monkeypatch,
        json.dumps({"verdict": "match", "confidence": 0.9, "reasoning": "ok"}),
    )
    a, b = crop_pair
    compute_vlm_judgment(a, b)

    assert client.messages.create.call_count == 1
    kwargs = client.messages.create.call_args.kwargs
    content = kwargs["messages"][0]["content"]
    # Two image blocks (expected, actual) + one text block.
    image_blocks = [block for block in content if block.get("type") == "image"]
    assert len(image_blocks) == 2
    # Both blocks carry base64 + a media_type.
    for block in image_blocks:
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"].startswith("image/")
        assert block["source"]["data"]  # non-empty


def test_claude_backend_parses_match_verdict(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    _install_fake_anthropic(
        monkeypatch,
        json.dumps({"verdict": "match", "confidence": 0.94, "reasoning": "Both crops identical."}),
    )
    a, b = crop_pair
    judgment = compute_vlm_judgment(a, b)

    assert isinstance(judgment, VLMJudgment)
    assert judgment.verdict == "match"
    assert judgment.confidence == pytest.approx(0.94)
    assert "identical" in judgment.reasoning


def test_claude_backend_handles_invalid_json_gracefully(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    _install_fake_anthropic(monkeypatch, "this is not json at all")
    a, b = crop_pair
    judgment = compute_vlm_judgment(a, b)

    assert judgment.verdict == "ambiguous"
    assert judgment.confidence == pytest.approx(0.0)


def test_claude_backend_recovers_from_fenced_json(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    """Models sometimes wrap JSON in ```json ... ``` fences."""
    fenced = '```json\n{"verdict": "no_match", "confidence": 0.8, "reasoning": "diff"}\n```'
    _install_fake_anthropic(monkeypatch, fenced)
    a, b = crop_pair
    judgment = compute_vlm_judgment(a, b)

    assert judgment.verdict == "no_match"
    assert judgment.confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Qwen local (Ollama) backend tests
# ---------------------------------------------------------------------------


def _ollama_chat_response(content: str) -> MagicMock:
    """Mimic an ``httpx.Response`` for a successful Ollama ``/api/chat``.

    Body shape matches Ollama's documented contract: top-level
    ``{"message": {"role": "assistant", "content": "..."}, "done": true, ...}``.
    """
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {
        "model": "qwen2.5vl:7b",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    response.text = json.dumps({"message": {"role": "assistant", "content": content}})
    return response


def test_qwen_backend_calls_ollama_chat_with_two_images(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    """One POST to /api/chat carrying both base64 images in the user message."""
    valid = json.dumps({"verdict": "match", "confidence": 0.9, "reasoning": "ok"})
    fake_client = MagicMock(name="httpx_client")
    fake_client.post.return_value = _ollama_chat_response(valid)

    with patch.object(httpx, "Client", return_value=fake_client) as ctor:
        a, b = crop_pair
        compute_vlm_judgment(a, b, backend="qwen-local")

    assert ctor.call_count == 1
    assert fake_client.post.call_count == 1
    url, *_ = fake_client.post.call_args.args
    assert url.endswith("/api/chat")
    payload = fake_client.post.call_args.kwargs["json"]
    assert payload["stream"] is False
    assert payload["model"] == vlm_verify.DEFAULT_QWEN_MODEL
    messages = payload["messages"]
    # System prompt + user message with two images.
    assert messages[0]["role"] == "system"
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["images"], list)
    assert len(user_msg["images"]) == 2
    for img in user_msg["images"]:
        assert isinstance(img, str) and img  # non-empty base64
    fake_client.close.assert_called_once()


def test_qwen_backend_parses_match_verdict(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    valid = json.dumps({"verdict": "match", "confidence": 0.87, "reasoning": "Identical crops."})
    fake_client = MagicMock(name="httpx_client")
    fake_client.post.return_value = _ollama_chat_response(valid)

    with patch.object(httpx, "Client", return_value=fake_client):
        a, b = crop_pair
        judgment = compute_vlm_judgment(a, b, backend="qwen-local")

    assert isinstance(judgment, VLMJudgment)
    assert judgment.verdict == "match"
    assert judgment.confidence == pytest.approx(0.87)
    assert "Identical" in judgment.reasoning


def test_qwen_backend_handles_invalid_json_gracefully(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    """Non-JSON ``message.content`` → safe ambiguous default, no crash."""
    fake_client = MagicMock(name="httpx_client")
    fake_client.post.return_value = _ollama_chat_response("not json at all")

    with patch.object(httpx, "Client", return_value=fake_client):
        a, b = crop_pair
        judgment = compute_vlm_judgment(a, b, backend="qwen-local")

    assert judgment.verdict == "ambiguous"
    assert judgment.confidence == pytest.approx(0.0)


def test_qwen_backend_raises_on_connection_refused(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    fake_client = MagicMock(name="httpx_client")
    fake_client.post.side_effect = httpx.ConnectError("Connection refused")

    with patch.object(httpx, "Client", return_value=fake_client):
        a, b = crop_pair
        with pytest.raises(VLMOllamaError) as exc_info:
            compute_vlm_judgment(a, b, backend="qwen-local")

    msg = str(exc_info.value)
    assert "ollama serve" in msg
    assert "qwen2.5vl" in msg
    # Even on error, we must release the client we own.
    fake_client.close.assert_called_once()


def test_qwen_backend_raises_on_non_200(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    bad = MagicMock(spec=httpx.Response)
    bad.status_code = 500
    bad.text = "internal error"
    fake_client = MagicMock(name="httpx_client")
    fake_client.post.return_value = bad

    with patch.object(httpx, "Client", return_value=fake_client):
        a, b = crop_pair
        with pytest.raises(VLMOllamaError) as exc_info:
            compute_vlm_judgment(a, b, backend="qwen-local")

    assert "500" in str(exc_info.value)


def test_qwen_batch_reuses_httpx_client(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    """3 pairs → one ``httpx.Client()`` construction, 3 ``.post()`` calls."""
    valid = json.dumps({"verdict": "match", "confidence": 0.9, "reasoning": "ok"})
    fake_client = MagicMock(name="httpx_client")
    fake_client.post.return_value = _ollama_chat_response(valid)
    # Context-manager protocol — the batch path uses ``with httpx.Client() as c:``.
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)

    with patch.object(httpx, "Client", return_value=fake_client) as ctor:
        a, b = crop_pair
        results = compute_vlm_judgment_batch([(a, b), (a, b), (a, b)], backend="qwen-local")

    assert len(results) == 3
    assert ctor.call_count == 1
    assert fake_client.post.call_count == 3


def test_qwen_respects_ollama_host_env(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://other:9999")
    valid = json.dumps({"verdict": "match", "confidence": 0.9, "reasoning": "ok"})
    fake_client = MagicMock(name="httpx_client")
    fake_client.post.return_value = _ollama_chat_response(valid)

    with patch.object(httpx, "Client", return_value=fake_client):
        a, b = crop_pair
        compute_vlm_judgment(a, b, backend="qwen-local")

    url, *_ = fake_client.post.call_args.args
    assert url == "http://other:9999/api/chat"


def test_qwen_custom_model_override(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    """``model=`` kwarg overrides the default ``qwen2.5vl:7b`` tag."""
    valid = json.dumps({"verdict": "match", "confidence": 0.9, "reasoning": "ok"})
    fake_client = MagicMock(name="httpx_client")
    fake_client.post.return_value = _ollama_chat_response(valid)

    with patch.object(httpx, "Client", return_value=fake_client):
        a, b = crop_pair
        compute_vlm_judgment(a, b, backend="qwen-local", model="qwen2.5vl:32b")

    payload = fake_client.post.call_args.kwargs["json"]
    assert payload["model"] == "qwen2.5vl:32b"


def test_missing_anthropic_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(vlm_verify.importlib, "import_module", fake_import)

    a, b = crop_pair
    with pytest.raises(VLMNotInstalledError) as exc_info:
        compute_vlm_judgment(a, b)

    assert "anthropic" in str(exc_info.value)
    assert "--extra vlm" in str(exc_info.value)
    assert "pixel-mcp-ml" in str(exc_info.value)


def test_batch_reuses_client(monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]) -> None:
    """3 pairs → ``Anthropic()`` constructed once, ``messages.create`` called 3x."""
    client = _install_fake_anthropic(
        monkeypatch,
        json.dumps({"verdict": "match", "confidence": 0.9, "reasoning": "ok"}),
    )
    anthropic_stub = vlm_verify.importlib.import_module("anthropic")

    a, b = crop_pair
    pairs = [(a, b), (a, b), (a, b)]
    results = compute_vlm_judgment_batch(pairs)

    assert len(results) == 3
    assert anthropic_stub.Anthropic.call_count == 1
    assert client.messages.create.call_count == 3


def test_empty_batch_returns_empty_list() -> None:
    assert compute_vlm_judgment_batch([]) == []


def test_default_model_is_sonnet_4_6(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    """Default Claude model is the configured ``DEFAULT_CLAUDE_MODEL``."""
    client = _install_fake_anthropic(
        monkeypatch,
        json.dumps({"verdict": "match", "confidence": 0.9, "reasoning": "ok"}),
    )
    a, b = crop_pair
    compute_vlm_judgment(a, b)

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == vlm_verify.DEFAULT_CLAUDE_MODEL
    assert kwargs["model"] == "claude-sonnet-4-6"


def test_unknown_backend_raises_value_error(crop_pair: tuple[Path, Path]) -> None:
    a, b = crop_pair
    with pytest.raises(ValueError):
        compute_vlm_judgment(a, b, backend="gpt-5")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# v1.5-2 — OmniParser semantic-label context wiring
# ---------------------------------------------------------------------------


def test_context_label_prepended_to_claude_user_prompt(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    """When ``context_label`` is set, the Claude user prompt mentions the label."""
    client = _install_fake_anthropic(
        monkeypatch,
        json.dumps({"verdict": "match", "confidence": 0.9, "reasoning": "ok"}),
    )
    a, b = crop_pair
    compute_vlm_judgment(a, b, context_label="button")

    kwargs = client.messages.create.call_args.kwargs
    text_blocks = [
        block for block in kwargs["messages"][0]["content"] if block.get("type") == "text"
    ]
    assert any("button" in block["text"] for block in text_blocks)
    assert any("This region appears to be a `button`" in block["text"] for block in text_blocks)


def test_context_label_omitted_keeps_v1_prompt(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    """Default ``context_label=None`` → no semantic prelude in the prompt."""
    client = _install_fake_anthropic(
        monkeypatch,
        json.dumps({"verdict": "match", "confidence": 0.9, "reasoning": "ok"}),
    )
    a, b = crop_pair
    compute_vlm_judgment(a, b)

    kwargs = client.messages.create.call_args.kwargs
    text_blocks = [
        block for block in kwargs["messages"][0]["content"] if block.get("type") == "text"
    ]
    joined = " ".join(block["text"] for block in text_blocks)
    assert "This region appears to be" not in joined


def test_batch_context_labels_length_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch, crop_pair: tuple[Path, Path]
) -> None:
    """Mismatched ``context_labels`` length → ValueError (caller bug)."""
    from pixel_mcp_ml import compute_vlm_judgment_batch

    _install_fake_anthropic(
        monkeypatch,
        json.dumps({"verdict": "match", "confidence": 0.9, "reasoning": "ok"}),
    )
    a, b = crop_pair
    with pytest.raises(ValueError):
        compute_vlm_judgment_batch(
            [(a, b)],
            backend="claude",
            context_labels=["button", "input"],  # length 2 != 1
        )
