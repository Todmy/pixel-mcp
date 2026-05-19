"""pixel-mcp-ml — ML extras for the pixel-mcp Convergence Loop.

Exposes:

- DINOv2-based perceptual similarity (Level 1 escalation gate).
- VLM-based verification (Level 2 escalation gate).

Future slices add OmniParser element detection and a local Qwen2.5-VL
backend for offline VLM verification.
"""

from pixel_mcp_ml.dinov2_compare import (
    DINOv2NotInstalledError,
    compute_dinov2_similarity,
    compute_dinov2_similarity_batch,
)
from pixel_mcp_ml.version import __version__
from pixel_mcp_ml.vlm_verify import (
    VLMJudgment,
    VLMNotInstalledError,
    compute_vlm_judgment,
    compute_vlm_judgment_batch,
)

__all__ = [
    "DINOv2NotInstalledError",
    "VLMJudgment",
    "VLMNotInstalledError",
    "__version__",
    "compute_dinov2_similarity",
    "compute_dinov2_similarity_batch",
    "compute_vlm_judgment",
    "compute_vlm_judgment_batch",
]
