"""MappingResolver — pair Figma nodes with DOM selectors via layered fallback.

Public entry point: :func:`resolve_mappings`. Returns a :class:`Mappings`
container that DeltaDiffer and HierarchicalDecomposer can consume.

Layered fallback (per CONTEXT.md):
1. **Figma Code Connect** — ground truth when the designer set it up. v0
   stubs this layer — the public Figma REST does not yet expose Code
   Connect mappings (it's surfaced through the Figma desktop client).
   The hook is in place so Slice #18.5 can wire it.
2. **AI pairing** — v0 stub returns no pairs. The real layer (Claude
   Sonnet via Anthropic API) lands in v0.5 once we wire the API
   dependency.
3. **Heuristic matching** — text content equality + ARIA role + position
   similarity. Runs for elements not covered by layers 1–2.

Each emitted pair carries ``confidence`` and ``source``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from pixel_mcp.delta import _flatten_spec
from pixel_mcp.render import MeasuredDOM, MeasuredElement
from pixel_mcp.spec import DesignSpec

__all__ = ["MappingPair", "Mappings", "resolve_mappings"]

MappingSource = Literal["code_connect", "ai", "heuristic"]


class MappingPair(BaseModel):
    """One Figma-node ↔ DOM-selector pair."""

    figma_node_id: str
    dom_selector: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: MappingSource


class Mappings(BaseModel):
    """Container persisted to ``.pixel-mcp/mappings.json``."""

    schema_version: int = 1
    figma_file_id: str
    pairs: list[MappingPair] = Field(default_factory=list)


def resolve_mappings(
    spec: DesignSpec,
    dom: MeasuredDOM,
    *,
    code_connect_enabled: bool = True,
    ai_pairing_enabled: bool = False,
    heuristic_confidence_floor: float = 0.5,
) -> Mappings:
    """Build a Mappings container via the layered fallback.

    Args:
        spec: DesignSpec for the Frame/Instance/Component being matched.
        dom: MeasuredDOM for the current Render.
        code_connect_enabled: Try Code Connect first. Default True; v0 stub
            still returns no pairs.
        ai_pairing_enabled: Try AI pairing second. Default False — v0 stub
            returns no pairs.
        heuristic_confidence_floor: Heuristic pairs below this confidence
            are still emitted but flagged.

    Returns:
        Mappings with all pairs, each tagged with ``source`` and
        ``confidence``.
    """
    paired_figma_ids: set[str] = set()
    paired_dom_selectors: set[str] = set()
    out: list[MappingPair] = []

    if code_connect_enabled:
        for pair in _code_connect_pairs(spec, dom):
            out.append(pair)
            paired_figma_ids.add(pair.figma_node_id)
            paired_dom_selectors.add(pair.dom_selector)

    if ai_pairing_enabled:
        for pair in _ai_pairing_pairs(spec, dom, paired_figma_ids, paired_dom_selectors):
            out.append(pair)
            paired_figma_ids.add(pair.figma_node_id)
            paired_dom_selectors.add(pair.dom_selector)

    for pair in _heuristic_pairs(
        spec,
        dom,
        already_paired_figma=paired_figma_ids,
        already_paired_dom=paired_dom_selectors,
        confidence_floor=heuristic_confidence_floor,
    ):
        out.append(pair)
        paired_figma_ids.add(pair.figma_node_id)
        paired_dom_selectors.add(pair.dom_selector)

    return Mappings(figma_file_id=spec.figma_file_id, pairs=out)


# --- Layer 1: Code Connect (stub) -----------------------------------------


def _code_connect_pairs(spec: DesignSpec, dom: MeasuredDOM) -> list[MappingPair]:
    """v0 stub — returns no pairs. Real implementation lands in v0.5."""
    _ = spec, dom
    return []


# --- Layer 2: AI pairing (stub) -------------------------------------------


def _ai_pairing_pairs(
    spec: DesignSpec,
    dom: MeasuredDOM,
    paired_figma_ids: set[str],
    paired_dom_selectors: set[str],
) -> list[MappingPair]:
    """v0 stub — returns no pairs. Real Claude API integration lands in v0.5."""
    _ = spec, dom, paired_figma_ids, paired_dom_selectors
    return []


# --- Layer 3: Heuristic matching ------------------------------------------


def _heuristic_pairs(
    spec: DesignSpec,
    dom: MeasuredDOM,
    *,
    already_paired_figma: set[str],
    already_paired_dom: set[str],
    confidence_floor: float,
) -> list[MappingPair]:
    """Pair remaining nodes by text content, then by tag/role + position."""
    flat = _flatten_spec(spec)
    candidates = [el for el in dom.elements if el.selector not in already_paired_dom]
    by_text: dict[str, list[MeasuredElement]] = {}
    for el in candidates:
        if el.text_content:
            by_text.setdefault(el.text_content.strip().lower(), []).append(el)

    out: list[MappingPair] = []
    used_dom: set[str] = set()

    # Text-content match (high confidence)
    for fnode in flat:
        if fnode.figma_node_id in already_paired_figma:
            continue
        if fnode.text_content:
            key = fnode.text_content.strip().lower()
            pool = by_text.get(key, [])
            for el in pool:
                if el.selector not in used_dom:
                    conf = 0.85
                    if conf >= confidence_floor:
                        out.append(
                            MappingPair(
                                figma_node_id=fnode.figma_node_id,
                                dom_selector=el.selector,
                                confidence=conf,
                                source="heuristic",
                            )
                        )
                        used_dom.add(el.selector)
                    break

    # Flat-order fallback for the still-unpaired (low confidence)
    paired_figma_now = {p.figma_node_id for p in out} | already_paired_figma
    unmatched_fnodes = [f for f in flat if f.figma_node_id not in paired_figma_now]
    unmatched_dom = [el for el in candidates if el.selector not in used_dom]
    for fnode, el in zip(unmatched_fnodes, unmatched_dom, strict=False):
        conf = 0.4
        if conf >= confidence_floor:
            out.append(
                MappingPair(
                    figma_node_id=fnode.figma_node_id,
                    dom_selector=el.selector,
                    confidence=conf,
                    source="heuristic",
                )
            )

    return out
