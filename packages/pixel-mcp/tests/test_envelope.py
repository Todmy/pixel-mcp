"""Unit tests for the AXI envelope helper.

The envelope is the foundational module — its shape is depended on by every
MCP tool. These tests pin the shape so accidental drift breaks loudly.
"""

from __future__ import annotations

import json

import pytest
from pixel_tools_shared import make_envelope
from pixel_tools_shared.envelope import Envelope

REQUIRED_FIELDS = {"data", "hints", "diagnostics", "next_suggested_action", "affordances"}


def test_empty_envelope_has_all_fields() -> None:
    env = make_envelope()
    assert set(env.keys()) == REQUIRED_FIELDS


def test_empty_envelope_defaults() -> None:
    env = make_envelope()
    assert env["data"] is None
    assert env["hints"] == []
    assert env["diagnostics"] == {}
    assert env["next_suggested_action"] is None
    assert env["affordances"] == []


def test_envelope_with_all_fields_round_trips_json() -> None:
    env = make_envelope(
        data={"checks": [{"name": "x", "status": "green", "detail": "ok"}], "summary": "1/1"},
        hints=["do a thing"],
        diagnostics={"python": "3.11.7"},
        next_suggested_action="proceed",
        affordances=[{"tool": "mcp__pixel_mcp__spec", "when": "now"}],
    )
    encoded = json.dumps(env)
    decoded = json.loads(encoded)
    assert decoded == env


def test_envelope_preserves_field_order() -> None:
    env = make_envelope(data=1, hints=["h"])
    keys = list(env.keys())
    assert keys == ["data", "hints", "diagnostics", "next_suggested_action", "affordances"]


def test_affordance_missing_keys_raises() -> None:
    with pytest.raises(ValueError, match="tool"):
        make_envelope(affordances=[{"tool": "x"}])
    with pytest.raises(ValueError, match="when"):
        make_envelope(affordances=[{"when": "later"}])


def test_affordance_extra_keys_dropped() -> None:
    env = make_envelope(affordances=[{"tool": "t", "when": "w", "stray": "drop"}])
    assert env["affordances"] == [{"tool": "t", "when": "w"}]


def test_envelope_hints_input_is_copied() -> None:
    src = ["a"]
    env: Envelope = make_envelope(hints=src)
    src.append("b")
    assert env["hints"] == ["a"]
