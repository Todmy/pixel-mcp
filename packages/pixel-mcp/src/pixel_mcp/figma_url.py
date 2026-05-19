"""Figma URL parsing.

Accepts both ``figma.com/file/<id>`` (legacy) and ``figma.com/design/<id>``
(current) URL forms. Extracts the ``file_id`` and the ``node_id`` from the
``?node-id=`` query parameter.

The Figma REST API uses ``node-id`` values of the form ``123:456`` (colon
separator). URLs typically encode them as ``123-456`` (dash). We normalize
back to the colon form on extraction so the value can be handed straight to
the REST client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

_FILE_PATH_RE = re.compile(r"^/(?:file|design)/(?P<file_id>[A-Za-z0-9]+)(?:/.*)?$")


class FigmaUrlError(ValueError):
    """Raised when a Figma URL cannot be parsed into ``(file_id, node_id)``."""


@dataclass(frozen=True)
class ParsedFigmaUrl:
    """The two identifiers we need to call the Figma REST API."""

    file_id: str
    node_id: str


def parse_figma_url(raw: str) -> ParsedFigmaUrl:
    """Parse a Figma URL into ``(file_id, node_id)``.

    Raises ``FigmaUrlError`` for malformed URLs or when ``node-id`` is absent.
    """
    if not raw or not isinstance(raw, str):
        raise FigmaUrlError("Figma URL is empty")

    parsed = urlparse(raw.strip())
    if parsed.scheme not in ("http", "https"):
        raise FigmaUrlError(f"Figma URL must use http(s) scheme; got scheme={parsed.scheme!r}")
    if not parsed.netloc.endswith("figma.com"):
        raise FigmaUrlError(f"Figma URL host must be figma.com; got host={parsed.netloc!r}")

    match = _FILE_PATH_RE.match(parsed.path)
    if not match:
        raise FigmaUrlError(
            "Figma URL path must start with /file/<id> or /design/<id>; "
            f"got path={parsed.path!r}"
        )
    file_id = match.group("file_id")

    qs = parse_qs(parsed.query)
    raw_node = qs.get("node-id", [None])[0]
    if not raw_node:
        raise FigmaUrlError("Figma URL is missing the required ?node-id=<id> query parameter")

    node_id = raw_node.replace("-", ":", 1)
    return ParsedFigmaUrl(file_id=file_id, node_id=node_id)
