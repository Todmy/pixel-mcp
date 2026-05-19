"""Figma REST API client.

A thin sync wrapper around the two endpoints we need for DesignSpec
extraction:
- ``GET /v1/files/{file_id}/nodes?ids={node_id}`` for the primary fetch.
- ``GET /v1/files/{file_id}/components/{component_id}`` for resolving the
  master behind a Component Instance (used by ``spec.py``).

Sync httpx — Slice 2 has no concurrent IO need.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

FIGMA_API_BASE = "https://api.figma.com"
DEFAULT_TIMEOUT = 30.0


class FigmaError(Exception):
    """Base class for all Figma client errors."""


class FigmaAuthError(FigmaError):
    """No ``FIGMA_TOKEN`` configured or Figma returned 401/403."""


class FigmaApiError(FigmaError):
    """Figma API returned a non-2xx status that isn't auth-related."""


class FigmaNotFoundError(FigmaError):
    """The requested file or node was not found (404 or empty ``nodes``)."""


class FigmaClient:
    """Sync Figma REST client. One instance per extraction call is fine."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str = FIGMA_API_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        resolved = token if token is not None else os.environ.get("FIGMA_TOKEN")
        if not resolved:
            raise FigmaAuthError(
                "FIGMA_TOKEN is not set. Export FIGMA_TOKEN=<your-personal-access-token>."
            )
        self._token = resolved
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers={"X-Figma-Token": resolved},
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FigmaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch_node(self, file_id: str, node_id: str) -> dict[str, Any]:
        """Return the raw Figma node payload for a single node.

        Returns the inner ``document`` (with ``components`` resolution table
        merged into a synthetic ``__components`` key for callers that need
        master metadata for Instances).
        """
        path = f"/v1/files/{file_id}/nodes"
        params = {"ids": node_id}
        response = self._request_get(path, params=params)
        payload = response.json()

        nodes = payload.get("nodes") or {}
        # Figma returns the requested key (using the colon form) but is
        # tolerant of either. Look up both.
        entry = nodes.get(node_id) or nodes.get(node_id.replace(":", "-"))
        if not entry or not entry.get("document"):
            raise FigmaNotFoundError(f"Figma node {node_id!r} not found in file {file_id!r}")
        document = dict(entry["document"])
        document["__components"] = entry.get("components", {}) or {}
        return document

    def fetch_component_master(self, file_id: str, component_id: str) -> dict[str, Any]:
        """Resolve a Component Instance's ``componentId`` to its master node.

        Figma stores master components inside the same file under the
        ``/nodes`` endpoint (the ``componentId`` is itself a node id), so
        this re-uses ``fetch_node``.
        """
        return self.fetch_node(file_id, component_id)

    def fetch_node_png_bytes(self, file_id: str, node_id: str, *, scale: float = 1.0) -> bytes:
        """Render the node as PNG via the Figma ``/v1/images`` endpoint.

        Returns the raw PNG bytes. Two-step: first ask Figma for an S3 URL,
        then download. Raises the same error hierarchy as ``fetch_node``.
        """
        # Step 1: request render
        response = self._request_get(
            f"/v1/images/{file_id}",
            params={"ids": node_id, "format": "png", "scale": str(scale)},
        )
        body = response.json()
        images = body.get("images") or {}
        url = images.get(node_id)
        if not url:
            raise FigmaNotFoundError(
                f"Figma /images returned no URL for node {node_id!r} in file {file_id!r}."
            )

        # Step 2: download the S3 URL (uses a fresh client — different host)
        try:
            with httpx.Client(timeout=self._client.timeout) as fresh:
                dl = fresh.get(url)
        except httpx.HTTPError as exc:
            raise FigmaApiError(f"Figma image download failed: {exc}") from exc
        if dl.status_code != 200:
            raise FigmaApiError(f"Figma image S3 download returned {dl.status_code} for {url}")
        return dl.content

    def _request_get(self, path: str, params: dict[str, str]) -> httpx.Response:
        try:
            response = self._client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise FigmaApiError(f"Figma API request failed: {exc}") from exc

        if response.status_code in (401, 403):
            raise FigmaAuthError(
                f"Figma API rejected the token ({response.status_code}). "
                "Check FIGMA_TOKEN is valid and has access to the file."
            )
        if response.status_code == 404:
            raise FigmaNotFoundError(f"Figma API 404 for {path} params={params}")
        if response.status_code >= 400:
            raise FigmaApiError(
                f"Figma API {response.status_code} for {path}: " f"{response.text[:200]}"
            )
        return response
