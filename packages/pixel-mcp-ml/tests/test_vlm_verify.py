"""Behavioural tests for the VLM verification Deep Module.

These tests **never make real Anthropic API calls** — every test patches
either ``importlib.import_module`` (to inject a fake ``anthropic``
module) or ``_make_anthropic_client`` (to inject a pre-built mock
client). The real SDK is never installed in the worktree venv.

What we verify:

- Claude backend sends one request per pair with two image blocks.
- Valid JSON in the response becomes a typed :class:`VLMJudgment`.
- Malformed responses degrade gracefully to ``ambiguous`` /
  confidence 0.0 (no crash).
- ``qwen-local`` raises :class:`NotImplementedError` with a clear hint.
- Missing ``anthropic`` SDK raises :class:`VLMNotInstalledError`.
- Batch reuses one client across N pairs.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pixel_mcp_ml import vlm_verify
from pixel_mcp_ml.vlm_verify import (
    VLMJudgment,
    VLMNotInstalledError,
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


def test_qwen_backend_raises_not_implemented(crop_pair: tuple[Path, Path]) -> None:
    a, b = crop_pair
    with pytest.raises(NotImplementedError) as exc_info:
        compute_vlm_judgment(a, b, backend="qwen-local")
    assert "v1-2" in str(exc_info.value)


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
