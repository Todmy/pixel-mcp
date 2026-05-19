"""Unit tests for the Figma URL parser."""

from __future__ import annotations

import pytest
from pixel_mcp.figma_url import FigmaUrlError, parse_figma_url


def test_parse_legacy_file_url() -> None:
    parsed = parse_figma_url("https://www.figma.com/file/AbC123/My-Project?node-id=10-20")
    assert parsed.file_id == "AbC123"
    assert parsed.node_id == "10:20"


def test_parse_design_url() -> None:
    parsed = parse_figma_url("https://www.figma.com/design/XyZ789/Another?node-id=42-7")
    assert parsed.file_id == "XyZ789"
    assert parsed.node_id == "42:7"


def test_parse_url_with_extra_query_params_any_order() -> None:
    parsed = parse_figma_url(
        "https://www.figma.com/design/XyZ789/Another?t=abc&node-id=5-9&type=design"
    )
    assert parsed.file_id == "XyZ789"
    assert parsed.node_id == "5:9"


def test_parse_url_already_colon_form() -> None:
    parsed = parse_figma_url("https://www.figma.com/file/AbC123/Project?node-id=10:20")
    assert parsed.node_id == "10:20"


def test_missing_node_id_raises() -> None:
    with pytest.raises(FigmaUrlError, match="node-id"):
        parse_figma_url("https://www.figma.com/file/AbC123/Project")


def test_empty_url_raises() -> None:
    with pytest.raises(FigmaUrlError, match="empty"):
        parse_figma_url("")


def test_non_figma_host_raises() -> None:
    with pytest.raises(FigmaUrlError, match="figma.com"):
        parse_figma_url("https://example.com/file/abc?node-id=1-2")


def test_bad_path_raises() -> None:
    with pytest.raises(FigmaUrlError, match="/file/<id>"):
        parse_figma_url("https://www.figma.com/proto/abc?node-id=1-2")


def test_non_http_scheme_raises() -> None:
    with pytest.raises(FigmaUrlError, match="scheme"):
        parse_figma_url("ftp://www.figma.com/file/abc?node-id=1-2")
