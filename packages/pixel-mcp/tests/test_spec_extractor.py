"""Unit tests for the DesignSpec extractor.

Uses httpx MockTransport + fixture JSON files. Real Figma API is never
touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pixel_mcp import spec as spec_module
from pixel_mcp.figma_client import FigmaClient
from pixel_mcp.spec import (
    SUPPORTED_NODE_TYPES,
    DesignSpec,
    UnsupportedNodeTypeError,
    extract_spec,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "figma"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Make state_dir() resolve to a clean temp directory per test so cache
    # writes don't leak between tests.
    monkeypatch.chdir(tmp_path)


def _install_transport(monkeypatch: pytest.MonkeyPatch, fixture: dict) -> None:
    """Patch FigmaClient to use a MockTransport returning ``fixture`` for any GET."""
    real_init = FigmaClient.__init__

    def patched_init(self: FigmaClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = httpx.MockTransport(lambda req: httpx.Response(200, json=fixture))
        kwargs.setdefault("token", "test_token")
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(FigmaClient, "__init__", patched_init)


def test_extract_frame_returns_design_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, _load("fixture_frame_response.json"))
    spec = extract_spec("https://www.figma.com/file/AbC123/My-Project?node-id=123-456")
    assert isinstance(spec, DesignSpec)
    assert spec.figma_file_id == "AbC123"
    assert spec.figma_node_id == "123:456"
    assert spec.figma_node_type == "FRAME"
    assert spec.name == "PrimaryButton"
    assert spec.dimensions.width == 200
    assert spec.dimensions.height == 48
    assert spec.layout.mode == "HORIZONTAL"
    assert spec.layout.padding_left == 24
    assert spec.layout.item_spacing == 8
    assert spec.corner_radius == 8
    assert len(spec.fills) == 1
    assert spec.fills[0].type == "SOLID"

    # Text child captured with typography
    assert len(spec.children) == 1
    text_child = spec.children[0]
    assert text_child.text_content == "Continue"
    assert text_child.typography is not None
    assert text_child.typography.font_family == "Inter"
    assert text_child.typography.font_size == 16
    assert text_child.typography.font_weight == 600


def test_extract_component_returns_sealed_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_transport(monkeypatch, _load("fixture_component_response.json"))
    spec = extract_spec("https://www.figma.com/design/CompFile/Lib?node-id=10-20")
    assert spec.figma_node_type == "COMPONENT"
    assert spec.name == "CardMaster"
    assert spec.layout.mode == "VERTICAL"
    assert spec.constraints.horizontal == "LEFT_RIGHT"
    assert len(spec.strokes) == 1


def test_extract_instance_resolves_master_and_applies_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance_fixture = _load("fixture_instance_response.json")
    component_fixture = _load("fixture_component_response.json")

    # Re-route the second `/nodes` call (for the master) by returning the
    # component fixture when the request includes ``10:20`` (or 10-20) in
    # the ?ids= query string.
    def handler(request: httpx.Request) -> httpx.Response:
        ids = request.url.params.get("ids", "")
        if "10" in ids and "20" in ids:
            return httpx.Response(200, json=component_fixture)
        return httpx.Response(200, json=instance_fixture)

    transport = httpx.MockTransport(handler)
    real_init = FigmaClient.__init__

    def patched_init(self: FigmaClient, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = transport
        kwargs.setdefault("token", "test_token")
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(FigmaClient, "__init__", patched_init)

    spec = extract_spec("https://www.figma.com/design/InstFile/Page?node-id=55-100")
    assert spec.figma_node_type == "INSTANCE"
    # Master layout (VERTICAL, FIXED sizing) preserved from CardMaster
    assert spec.layout.mode == "VERTICAL"
    # Instance overrides "fills" — should NOT be the master's white;
    # should be the instance's light-blue tint.
    assert spec.fills[0].color is not None
    assert spec.fills[0].color["b"] == 1.0
    assert spec.fills[0].color["r"] == pytest.approx(0.95)


def test_unsupported_node_type_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_transport(monkeypatch, _load("fixture_group_response.json"))
    with pytest.raises(UnsupportedNodeTypeError, match="GROUP"):
        extract_spec("https://www.figma.com/design/GroupFile/Loose?node-id=99-1")


def test_supported_types_constant() -> None:
    # Sanity: docs + code agree on the supported set.
    assert set(SUPPORTED_NODE_TYPES) == {"FRAME", "INSTANCE", "COMPONENT"}
    assert spec_module.SCHEMA_VERSION == 1
