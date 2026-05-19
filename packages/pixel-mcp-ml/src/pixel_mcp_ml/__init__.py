"""pixel-mcp-ml — ML extras for the pixel-mcp Convergence Loop.

Currently exposes DINOv2-based perceptual similarity (Level 1 escalation
gate). Future slices add OmniParser element detection and VLM bridges.
"""

from pixel_mcp_ml.dinov2_compare import (
    DINOv2NotInstalledError,
    compute_dinov2_similarity,
    compute_dinov2_similarity_batch,
)
from pixel_mcp_ml.version import __version__

__all__ = [
    "DINOv2NotInstalledError",
    "__version__",
    "compute_dinov2_similarity",
    "compute_dinov2_similarity_batch",
]
