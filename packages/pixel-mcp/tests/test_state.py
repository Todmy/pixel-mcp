"""Unit tests for the State Directory helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pixel_mcp.spec import DesignSpec, Dimensions
from pixel_mcp.state import (
    SPEC_CACHE_FILENAME,
    read_spec_cache,
    state_dir,
    write_spec_cache,
)


def _make_spec(file_id: str = "file1", node_id: str = "1:1", name: str = "Frame") -> DesignSpec:
    return DesignSpec(
        figma_file_id=file_id,
        figma_node_id=node_id,
        figma_node_type="FRAME",
        name=name,
        dimensions=Dimensions(width=100, height=50),
        extracted_at=datetime.now(UTC),
    )


def test_state_dir_creates_pixel_mcp_dir(tmp_path: Path) -> None:
    d = state_dir(tmp_path)
    assert d.exists()
    assert d == tmp_path / ".pixel-mcp"


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    spec = _make_spec()
    write_spec_cache(spec, project_root=tmp_path)
    got = read_spec_cache("file1", "1:1", project_root=tmp_path)
    assert got is not None
    assert got.name == "Frame"
    assert got.figma_node_id == "1:1"


def test_read_returns_none_when_no_cache(tmp_path: Path) -> None:
    got = read_spec_cache("nope", "nope:nope", project_root=tmp_path)
    assert got is None


def test_ttl_expiry_returns_none(tmp_path: Path) -> None:
    spec = _make_spec()
    write_spec_cache(spec, project_root=tmp_path)
    # Tamper with the cached_at timestamp to simulate expiry.
    cache_path = tmp_path / ".pixel-mcp" / SPEC_CACHE_FILENAME
    payload = json.loads(cache_path.read_text())
    key = "file1:1:1"
    payload["entries"][key]["cached_at"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    cache_path.write_text(json.dumps(payload))

    got = read_spec_cache("file1", "1:1", project_root=tmp_path, ttl_seconds=3600)
    assert got is None


def test_multiple_specs_coexist(tmp_path: Path) -> None:
    a = _make_spec(file_id="A", node_id="1:1", name="A-frame")
    b = _make_spec(file_id="B", node_id="2:2", name="B-frame")
    write_spec_cache(a, project_root=tmp_path)
    write_spec_cache(b, project_root=tmp_path)

    got_a = read_spec_cache("A", "1:1", project_root=tmp_path)
    got_b = read_spec_cache("B", "2:2", project_root=tmp_path)
    assert got_a is not None and got_a.name == "A-frame"
    assert got_b is not None and got_b.name == "B-frame"


def test_corrupt_cache_returns_none(tmp_path: Path) -> None:
    d = state_dir(tmp_path)
    (d / SPEC_CACHE_FILENAME).write_text("not json {")
    got = read_spec_cache("any", "any:any", project_root=tmp_path)
    assert got is None


def test_old_schema_version_discarded(tmp_path: Path) -> None:
    d = state_dir(tmp_path)
    (d / SPEC_CACHE_FILENAME).write_text(
        json.dumps({"schema_version": 999, "entries": {"file1:1:1": {}}})
    )
    got = read_spec_cache("file1", "1:1", project_root=tmp_path)
    assert got is None


def test_overwriting_same_key_replaces_entry(tmp_path: Path) -> None:
    write_spec_cache(_make_spec(name="v1"), project_root=tmp_path)
    write_spec_cache(_make_spec(name="v2"), project_root=tmp_path)
    got = read_spec_cache("file1", "1:1", project_root=tmp_path)
    assert got is not None
    assert got.name == "v2"


def test_state_dir_defaults_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    d = state_dir()
    assert d == tmp_path / ".pixel-mcp"
