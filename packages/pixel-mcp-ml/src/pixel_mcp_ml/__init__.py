"""pixel-mcp-ml — ML extras for the pixel-mcp Convergence Loop.

Exposes:

- DINOv2-based perceptual similarity (Level 1 escalation gate).
- VLM-based verification (Level 2 escalation gate).
- OmniParser UI-element detection (region-attribution infrastructure).

Future slices wire OmniParser detections into ``pixel-mcp check`` for
sharper region attribution + per-element VLM context.
"""

from pixel_mcp_ml.dinov2_compare import (
    DINOv2NotInstalledError,
    compute_dinov2_similarity,
    compute_dinov2_similarity_batch,
)
from pixel_mcp_ml.omniparser_detect import (
    DetectedElement,
    OmniParserNotInstalledError,
    detect_ui_elements,
    detect_ui_elements_batch,
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
    "DetectedElement",
    "OmniParserNotInstalledError",
    "VLMJudgment",
    "VLMNotInstalledError",
    "__version__",
    "compute_dinov2_similarity",
    "compute_dinov2_similarity_batch",
    "compute_vlm_judgment",
    "compute_vlm_judgment_batch",
    "detect_ui_elements",
    "detect_ui_elements_batch",
]
