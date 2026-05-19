"""Project Config — optional ``.pixel-mcp.json`` reader.

Public entry point: :func:`load_project_config`. Returns a
:class:`ProjectConfig` Pydantic model that callers can apply to override
defaults (tolerance, viewport, max_iterations, etc.).

CLI flags > config file > built-in defaults (precedence).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "ProjectConfig",
    "Tolerance",
    "ViewportConfig",
    "load_project_config",
]

CONFIG_FILENAME = ".pixel-mcp.json"


class ViewportConfig(BaseModel):
    width: int = 1280
    height: int = 720


class Tolerance(BaseModel):
    """Per-property Tolerance overrides."""

    color: str = "exact"
    dimensions_absolute_px: float = 2.0
    dimensions_ratio: float = 0.02
    typography_font_size_pt: float = 0.5
    position_absolute_px: float = 2.0
    position_ratio: float = 0.01


class MaskRegion(BaseModel):
    selector: str
    reason: str | None = None


class ProjectConfig(BaseModel):
    """Loaded from ``.pixel-mcp.json`` at the project root."""

    schema_version: int = 1
    max_iterations: int = 15
    iteration_timeout_seconds: int = 60
    stuck_threshold: int = 3
    enabled_levels: list[int] = Field(default_factory=lambda: [0])
    tolerance: Tolerance = Field(default_factory=Tolerance)
    ssim_threshold: float = 0.97
    min_bbox_area: int = 100
    enable_dinov2: bool = False
    dinov2_threshold: float = 0.95
    enable_vlm: bool = False
    vlm_threshold: float = 0.7
    vlm_backend: Literal["claude", "qwen-local"] = "claude"
    enable_human_gate: bool = False
    viewport: ViewportConfig = Field(default_factory=ViewportConfig)
    mask_regions: list[MaskRegion] = Field(default_factory=list)
    figma_token_env: str = "FIGMA_TOKEN"


def load_project_config(project_root: Path | None = None) -> ProjectConfig:
    """Load ``.pixel-mcp.json`` if present; otherwise return defaults.

    A malformed config file raises pydantic ``ValidationError`` (caught at
    the command layer and surfaced as a clear AXI envelope error).
    """
    root = project_root or Path.cwd()
    path = root / CONFIG_FILENAME
    if not path.exists():
        return ProjectConfig()
    return ProjectConfig.model_validate_json(path.read_text())
