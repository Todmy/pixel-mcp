# pixel-mcp

Figma to Browser convergence harness with escalation gates. CLI plus MCP server.

This package is the core, published artifact. See the [repo README](../../README.md) for installation and Claude Code wiring.

## Surface (Slice 4)

Implemented:
- `pixel-mcp doctor` — environment Check, returns AXI envelope.
- `pixel-mcp spec --figma <url>` — extract a DesignSpec from a Figma Source.
- `pixel-mcp measure --route <url>` — capture a MeasuredDOM from a Render.
- `pixel-mcp diff --spec X --measured Y` — compute Deltas between a DesignSpec and a MeasuredDOM.
- `pixel-mcp judge --deltas X [--strict]` — run the ConvergenceJudge over a Delta[].
- `pixel-mcp check --figma <url> --route <url>` — one Iteration of the Convergence Loop (composite).
- `pixel-mcp mcp` — launch the MCP server over stdio. Exposes tools: `doctor`, `spec`, `measure`, `diff`, `judge`, `check`.

Stubbed (one issue per slice):
- `review`, `mapping`, `snapshot`, `reset`.

Each stub prints the tracking issue and exits non-zero.

## `pixel-mcp check`

The composite — one Iteration of the Convergence Loop. Calls `spec`, `measure`,
`diff`, and `judge` in sequence and returns a single AXI envelope.

```sh
pixel-mcp check --figma 'https://www.figma.com/design/<id>?node-id=<n>' \
                --route http://localhost:3000/component
```

Exit codes drive Loop Runners (Ralph Loop, Makefile, CI):
- `0` — Final Convergence at the highest enabled Level (v0 = Level 0).
- `1` — Deltas present; the Agent should fix and re-invoke.
- `12` — Fatal (Figma/Render/IO).

The envelope's `data` carries `{converged, deltas, judgment, ssim_score: null, hot_regions: []}` —
the latter two are reserved for Slice 6.

## Setup — Playwright + Chromium

`measure` drives a headless Chromium browser via Playwright. After
`uv sync`, install the browser binary once:

```sh
uv run playwright install chromium
```

This is a one-time download of roughly 150 MB. `pixel-mcp doctor`
reports both checks (`playwright` importable, `chromium` binary present)
and emits the install hint when either is missing.

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

## `pixel-mcp measure`

Launches headless Chromium and emits a **MeasuredDOM** — per-element
bounding boxes, computed styles, text content, ARIA role, parent chain.
Wrapped in an AXI envelope, ready for the DeltaDiffer (Slice 4) to
compare against a DesignSpec.

### Usage

```sh
# Auto-discover visible elements at the default viewport (1280x720)
pixel-mcp measure --route http://localhost:3000/foo

# Narrow to specific selectors
pixel-mcp measure --route http://localhost:3000/foo \
    --selectors "button.cta, nav.top"

# Custom viewport
pixel-mcp measure --route http://localhost:3000/foo --viewport 1920x1080

# Wait for a selector before measuring (slow-hydrating SPAs)
pixel-mcp measure --route http://localhost:3000/foo --wait-for ".content-ready"

# Write to a file instead of stdout
pixel-mcp measure --route http://localhost:3000/foo --out measured.json
```

The MCP equivalent:

```jsonc
{
  "tool": "mcp__pixel_mcp__measure",
  "args": {
    "route": "http://localhost:3000/foo",
    "selectors": ["button.cta"],
    "viewport_width": 1280,
    "viewport_height": 720,
    "wait_for": ".content-ready"
  }
}
```

### Auto-discover

When `--selectors` is omitted, `measure` walks the DOM and picks visible
elements that are either leaves or semantic containers (`<button>`,
`<input>`, `<nav>`, `<section>`, `<article>`, `<header>`, `<footer>`, or
any element with a `role` attribute). Elements smaller than 16 px² are
filtered out (anti-aliasing noise). The discovery is capped at 200
elements per route — when the cap is hit the envelope hints to pass
`--selectors` for a narrower window.

### Determinism

By default `measure` waits for `networkidle` then one
`requestAnimationFrame` quiet before snapping the page. This gives
reproducible bounding boxes on most SPAs. Pages with long-poll or SSE
connections never reach `networkidle`; `measure` continues anyway.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | MeasuredDOM captured successfully. |
| 12 | Playwright missing, Chromium missing, route unreachable, wait-for timeout, or any other fatal capture error. |
