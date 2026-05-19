"""Command-layer wrapper for ``extract_spec``.

Both the CLI subcommand and the MCP tool delegate here so the AXI envelope
shape stays consistent across surfaces. The CLI adds I/O (stdout vs --out);
the MCP tool returns the envelope dict directly.

Exit codes:
- 0 — DesignSpec extracted successfully.
- 12 — Auth, network, missing node, unsupported-type, or any other fatal
       extraction error. (Per slice spec: one fatal exit code for v0.)
"""

from __future__ import annotations

import json
from typing import Any

from pixel_tools_shared import Envelope, make_envelope

from pixel_mcp.figma_client import (
    FigmaApiError,
    FigmaAuthError,
    FigmaError,
    FigmaNotFoundError,
)
from pixel_mcp.figma_url import FigmaUrlError
from pixel_mcp.spec import (
    SUPPORTED_NODE_TYPES,
    DesignSpec,
    UnsupportedNodeTypeError,
    extract_spec,
)

EXIT_OK = 0
EXIT_FATAL = 12

_SUCCESS_AFFORDANCES: list[dict[str, str]] = [
    {
        "tool": "mcp__pixel_mcp__measure",
        "when": "to capture the MeasuredDOM from the current Render and compare it against this DesignSpec",
    },
]


def run(figma_url: str, refresh: bool = False) -> tuple[Envelope, int]:
    """Extract a DesignSpec and wrap it in an AXI envelope.

    Returns ``(envelope, exit_code)``. Never raises — all extraction errors
    are folded into the envelope's ``diagnostics`` and ``hints``.
    """
    try:
        spec = extract_spec(figma_url, refresh=refresh)
    except FigmaUrlError as exc:
        return _error_envelope(
            error_type="figma_url_error",
            error_message=str(exc),
            hints=[
                "Pass a URL of the form https://www.figma.com/design/<file-id>/...?node-id=<id>",
                "Both /file/ and /design/ prefixes are accepted; node-id query parameter is required.",
            ],
        ), EXIT_FATAL
    except FigmaAuthError as exc:
        return _error_envelope(
            error_type="figma_auth_error",
            error_message=str(exc),
            hints=[
                "Set FIGMA_TOKEN to a valid personal-access token.",
                "Check that the token has access to the requested file.",
            ],
        ), EXIT_FATAL
    except FigmaNotFoundError as exc:
        return _error_envelope(
            error_type="figma_not_found",
            error_message=str(exc),
            hints=[
                "Check the file-id and node-id in the URL — the file may have been moved or the node deleted.",
            ],
        ), EXIT_FATAL
    except UnsupportedNodeTypeError as exc:
        return _error_envelope(
            error_type="unsupported_node_type",
            error_message=str(exc),
            hints=[
                f"Supported Figma node types: {', '.join(SUPPORTED_NODE_TYPES)}.",
                "For Groups, Pages, Sections, or vector layers — switch to image-only mode (coming in v0.5).",
            ],
        ), EXIT_FATAL
    except FigmaApiError as exc:
        return _error_envelope(
            error_type="figma_api_error",
            error_message=str(exc),
            hints=[
                "Re-try after a moment — the Figma API may be transiently unavailable.",
                "If the failure persists, check `pixel-mcp doctor` for environment diagnostics.",
            ],
        ), EXIT_FATAL
    except FigmaError as exc:  # catch-all for any future FigmaError subclass
        return _error_envelope(
            error_type="figma_error",
            error_message=str(exc),
            hints=["See diagnostics for details; check `pixel-mcp doctor` first."],
        ), EXIT_FATAL

    return _success_envelope(spec, refresh=refresh), EXIT_OK


def _success_envelope(spec: DesignSpec, refresh: bool) -> Envelope:
    # Round-trip through JSON to get plain-JSON-safe types for the envelope.
    data: dict[str, Any] = json.loads(spec.model_dump_json())
    hints: list[str] = [
        f"DesignSpec extracted for Figma node {spec.figma_node_id!r} (type={spec.figma_node_type}).",
    ]
    if refresh:
        hints.append("Cache bypassed via --refresh-spec.")
    return make_envelope(
        data=data,
        hints=hints,
        diagnostics={
            "figma_file_id": spec.figma_file_id,
            "figma_node_id": spec.figma_node_id,
            "figma_node_type": spec.figma_node_type,
            "schema_version": spec.schema_version,
            "child_count": len(spec.children),
        },
        next_suggested_action=(
            "Run `pixel-mcp measure` (Slice 3) to capture the Render's MeasuredDOM, "
            "then `pixel-mcp diff` to compute Deltas against this DesignSpec."
        ),
        affordances=_SUCCESS_AFFORDANCES,
    )


def _error_envelope(error_type: str, error_message: str, hints: list[str]) -> Envelope:
    return make_envelope(
        data=None,
        hints=hints,
        diagnostics={
            "error_type": error_type,
            "error_message": error_message,
        },
        next_suggested_action="Resolve the error above, then re-run `pixel-mcp spec --figma <url>`.",
        affordances=[
            {
                "tool": "mcp__pixel_mcp__doctor",
                "when": "to diagnose environment issues (FIGMA_TOKEN, network, etc.)",
            },
        ],
    )
