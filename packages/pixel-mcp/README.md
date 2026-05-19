# pixel-mcp

Figma to Browser convergence harness with escalation gates. CLI plus MCP server.

This package is the core, published artifact. See the [repo README](../../README.md) for installation and Claude Code wiring.

## Surface (Slice 2)

Implemented:
- `pixel-mcp doctor` — environment Check, returns AXI envelope.
- `pixel-mcp spec --figma <url>` — extract a DesignSpec from a Figma Source.
- `pixel-mcp mcp` — launch the MCP server over stdio. Exposes tools: `doctor`, `spec`.

Stubbed (one issue per slice):
- `measure`, `diff`, `judge`, `check`, `review`, `mapping`, `snapshot`, `reset`.

Each stub prints the tracking issue and exits non-zero.

## `pixel-mcp spec`

Fetches a Figma Frame, Component Instance, or Master Component via the Figma
REST API and emits a normalized **DesignSpec** wrapped in an AXI envelope.

### Setup — `FIGMA_TOKEN`

```sh
export FIGMA_TOKEN=<your-personal-access-token>
```

Get a token at https://www.figma.com/developers/api#access-tokens. Make sure
it has access to the file you point `--figma` at.

### Usage

```sh
# Print the envelope to stdout
pixel-mcp spec --figma 'https://www.figma.com/design/<file-id>/Project?node-id=123-456'

# Write to a file
pixel-mcp spec --figma '<url>' --out spec.json

# Bypass the spec-cache (1h TTL) and re-fetch
pixel-mcp spec --figma '<url>' --refresh-spec
```

The MCP equivalent:

```jsonc
// Call from Claude Code
{
  "tool": "mcp__pixel_mcp__spec",
  "args": { "figma_url": "<url>", "refresh": false }
}
```

### URL formats accepted

Both legacy and current Figma URL shapes work:

- `https://www.figma.com/file/<file-id>/<slug>?node-id=<id>`
- `https://www.figma.com/design/<file-id>/<slug>?node-id=<id>`

The `node-id` query parameter is required. URLs encode node ids with a dash
(`123-456`); pixel-mcp normalizes them to the colon form (`123:456`) that
the Figma REST API expects.

### Supported Figma node types

| Type | Status | Notes |
|---|---|---|
| `FRAME` | Supported | The canonical primary input — auto-layout aware. |
| `INSTANCE` | Supported | Resolved to master + overrides at extraction time. |
| `COMPONENT` | Supported | Master component, sealed (no overrides). |

### Unsupported types

These return an AXI envelope with `error_type: "unsupported_node_type"` and
exit code 12:

- `GROUP` — no semantic constraints. Use image-only mode (coming in v0.5).
- `SECTION`, `PAGE` — too broad to match against.
- Vector layers — not coherent designs.
- Whole-file URLs without `node-id`.

For these, switch to image-only mode once it ships.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | DesignSpec extracted successfully. |
| 12 | Auth, network, missing node, unsupported-type, or any other fatal extraction error. |

### Spec-cache

Successful extractions land in `.pixel-mcp/spec-cache.json` keyed by
`(file_id, node_id)` with a 1-hour TTL. Pass `--refresh-spec` (CLI) or
`refresh=true` (MCP) to bypass. The cache survives between iterations of
the Convergence Loop.
