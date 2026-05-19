"""State Directory helpers — ``.pixel-mcp/`` on disk.

Slice 2 only touches ``spec-cache.json``. Later slices add ``state.json``,
``history.jsonl``, ``mappings.json``, ``crops/``. We keep helpers narrow:
one function per persisted file, atomic writes, schema_version on each
top-level object so we can migrate without breaking existing caches.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pixel_mcp.spec import DesignSpec

SPEC_CACHE_FILENAME = "spec-cache.json"
SPEC_CACHE_TTL_SECONDS = 3600  # 1h
SCHEMA_VERSION = 1


def state_dir(project_root: Path | None = None) -> Path:
    """Return ``<project_root>/.pixel-mcp/``. Creates it if absent."""
    root = project_root if project_root is not None else Path.cwd()
    d = root / ".pixel-mcp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _spec_cache_path(project_root: Path | None = None) -> Path:
    return state_dir(project_root) / SPEC_CACHE_FILENAME


def _entry_key(file_id: str, node_id: str) -> str:
    return f"{file_id}:{node_id}"


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    if raw.get("schema_version") != SCHEMA_VERSION:
        # Future migration hook. For v0 we just discard incompatible caches.
        return {}
    entries = raw.get("entries") or {}
    return entries if isinstance(entries, dict) else {}


def _write_cache(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    payload = {"schema_version": SCHEMA_VERSION, "entries": entries}
    # Atomic write: temp file in the same directory, then rename. Prevents
    # half-written caches if the process dies mid-write.
    fd, tmp_path = tempfile.mkstemp(prefix=".spec-cache-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def read_spec_cache(
    file_id: str,
    node_id: str,
    project_root: Path | None = None,
    ttl_seconds: int = SPEC_CACHE_TTL_SECONDS,
) -> DesignSpec | None:
    """Return a cached DesignSpec if one exists and is within TTL."""
    path = _spec_cache_path(project_root)
    entries = _load_cache(path)
    entry = entries.get(_entry_key(file_id, node_id))
    if not entry:
        return None

    cached_at_raw = entry.get("cached_at")
    if not cached_at_raw:
        return None
    try:
        cached_at = datetime.fromisoformat(cached_at_raw)
    except ValueError:
        return None
    if cached_at.tzinfo is None:
        cached_at = cached_at.replace(tzinfo=UTC)

    age = (datetime.now(UTC) - cached_at).total_seconds()
    if age > ttl_seconds:
        return None

    spec_payload = entry.get("spec")
    if not isinstance(spec_payload, dict):
        return None
    from pixel_mcp.spec import DesignSpec  # lazy: spec.py depends on state.py

    try:
        return DesignSpec.model_validate(spec_payload)
    except Exception:
        return None


def write_spec_cache(spec: DesignSpec, project_root: Path | None = None) -> None:
    """Atomically write ``spec`` into the spec-cache. Preserves other entries."""
    path = _spec_cache_path(project_root)
    entries = _load_cache(path)
    entries[_entry_key(spec.figma_file_id, spec.figma_node_id)] = {
        "cached_at": datetime.now(UTC).isoformat(),
        "spec": json.loads(spec.model_dump_json()),
    }
    _write_cache(path, entries)
