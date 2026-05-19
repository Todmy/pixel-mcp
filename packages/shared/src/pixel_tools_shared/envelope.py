"""AXI envelope helper.

Every MCP tool in pixel-mcp returns the same response shape:

    {
        "data": <tool-specific payload>,
        "hints": [<actionable string>, ...],
        "diagnostics": {<key>: <value>, ...},
        "next_suggested_action": <one-line string or None>,
        "affordances": [{"tool": ..., "when": ...}, ...],
    }

The shape stays stable even when fields are unused — empty list/dict/None
defaults keep downstream JSON parsers honest.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Affordance(TypedDict):
    """A pointer at a follow-up MCP tool plus when to use it."""

    tool: str
    when: str


class Envelope(TypedDict):
    """Canonical AXI response shape."""

    data: Any
    hints: list[str]
    diagnostics: dict[str, Any]
    next_suggested_action: str | None
    affordances: list[Affordance]


def make_envelope(
    data: Any = None,
    hints: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
    next_suggested_action: str | None = None,
    affordances: list[dict[str, str]] | None = None,
) -> Envelope:
    """Build an AXI envelope with stable field ordering.

    All five fields are always present. None inputs map to empty list / empty
    dict / explicit None so downstream consumers can rely on field presence.
    """
    return {
        "data": data,
        "hints": list(hints) if hints else [],
        "diagnostics": dict(diagnostics) if diagnostics else {},
        "next_suggested_action": next_suggested_action,
        "affordances": [_normalize_affordance(a) for a in affordances] if affordances else [],
    }


def _normalize_affordance(raw: dict[str, str]) -> Affordance:
    """Coerce a loose dict into the Affordance TypedDict.

    Required keys: ``tool``, ``when``. Extra keys are dropped silently to
    keep the surface narrow.
    """
    if "tool" not in raw or "when" not in raw:
        raise ValueError(f"Affordance must have 'tool' and 'when' keys; got {sorted(raw.keys())}")
    return {"tool": raw["tool"], "when": raw["when"]}
