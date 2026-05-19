"""Internal shared helpers for pixel-mcp.

Not published to PyPI. Provides the AXI envelope helper and any
cross-package primitives that need a single canonical definition.
"""

from pixel_tools_shared.envelope import Affordance, Envelope, make_envelope

__all__ = ["Affordance", "Envelope", "make_envelope"]
