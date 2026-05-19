"""Unit tests for the Figma REST API client.

We use httpx MockTransport so the tests do not touch the real network.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from pixel_mcp.figma_client import (
    FigmaApiError,
    FigmaAuthError,
    FigmaClient,
    FigmaNotFoundError,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "figma"


def _frame_fixture() -> dict:
    return json.loads((FIXTURE_DIR / "fixture_frame_response.json").read_text())


def test_missing_token_raises_auth_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIGMA_TOKEN", raising=False)
    with pytest.raises(FigmaAuthError, match="FIGMA_TOKEN"):
        FigmaClient()


def test_auth_header_set_on_request() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["x-figma-token"] = request.headers.get("X-Figma-Token", "")
        return httpx.Response(200, json=_frame_fixture())

    transport = httpx.MockTransport(handler)
    client = FigmaClient(token="t_test_value", transport=transport)
    client.fetch_node("AbC123", "123:456")
    assert captured["x-figma-token"] == "t_test_value"


def test_fetch_node_returns_document() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_frame_fixture()))
    client = FigmaClient(token="t", transport=transport)
    document = client.fetch_node("AbC123", "123:456")
    assert document["id"] == "123:456"
    assert document["type"] == "FRAME"


def test_401_raises_auth_error() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(401, json={"err": "bad token"}))
    client = FigmaClient(token="t", transport=transport)
    with pytest.raises(FigmaAuthError):
        client.fetch_node("AbC123", "123:456")


def test_403_raises_auth_error() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(403, json={"err": "forbidden"}))
    client = FigmaClient(token="t", transport=transport)
    with pytest.raises(FigmaAuthError):
        client.fetch_node("AbC123", "123:456")


def test_404_raises_not_found() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(404, json={"err": "not found"}))
    client = FigmaClient(token="t", transport=transport)
    with pytest.raises(FigmaNotFoundError):
        client.fetch_node("AbC123", "123:456")


def test_500_raises_api_error() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="internal"))
    client = FigmaClient(token="t", transport=transport)
    with pytest.raises(FigmaApiError):
        client.fetch_node("AbC123", "123:456")


def test_empty_nodes_response_raises_not_found() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"nodes": {}}))
    client = FigmaClient(token="t", transport=transport)
    with pytest.raises(FigmaNotFoundError):
        client.fetch_node("AbC123", "999:999")
