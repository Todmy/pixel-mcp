"""Shared pytest fixtures for pixel-mcp-ml tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def clear_model_cache() -> Iterator[None]:
    """Reset the DINOv2 module-level model cache around each test."""
    from pixel_mcp_ml import dinov2_compare

    dinov2_compare._MODEL_CACHE.clear()
    yield
    dinov2_compare._MODEL_CACHE.clear()


@pytest.fixture
def tiny_image_factory(tmp_path: Path):
    """Factory that writes a tiny solid-color PNG and returns its path."""

    def _make(name: str, color: tuple[int, int, int] = (128, 128, 128)) -> Path:
        path = tmp_path / name
        Image.fromarray(np.full((8, 8, 3), color, dtype=np.uint8)).save(path)
        return path

    return _make
