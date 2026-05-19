"""Unit tests for project_config — ``.pixel-mcp.json`` loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pixel_mcp.project_config import ProjectConfig, load_project_config
from pydantic import ValidationError


def test_load_returns_defaults_when_file_absent(tmp_path: Path) -> None:
    cfg = load_project_config(project_root=tmp_path)
    assert cfg.max_iterations == 15
    assert cfg.ssim_threshold == pytest.approx(0.97)
    assert cfg.viewport.width == 1280


def test_load_applies_overrides(tmp_path: Path) -> None:
    (tmp_path / ".pixel-mcp.json").write_text('{"max_iterations": 25, "ssim_threshold": 0.92}')
    cfg = load_project_config(project_root=tmp_path)
    assert cfg.max_iterations == 25
    assert cfg.ssim_threshold == pytest.approx(0.92)


def test_load_mask_regions(tmp_path: Path) -> None:
    (tmp_path / ".pixel-mcp.json").write_text(
        '{"mask_regions": [{"selector": ".timestamp", "reason": "dynamic"}]}'
    )
    cfg = load_project_config(project_root=tmp_path)
    assert len(cfg.mask_regions) == 1
    assert cfg.mask_regions[0].selector == ".timestamp"


def test_malformed_config_raises(tmp_path: Path) -> None:
    (tmp_path / ".pixel-mcp.json").write_text('{"max_iterations": "not-a-number"}')
    with pytest.raises(ValidationError):
        load_project_config(project_root=tmp_path)


def test_default_enabled_levels_contains_zero() -> None:
    cfg = ProjectConfig()
    assert 0 in cfg.enabled_levels
