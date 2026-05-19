# pixel-mcp

Figma to Browser convergence harness with escalation gates. CLI plus MCP server for Claude Code.

> **Status**: Slice 1 — scaffold + doctor + AXI envelope foundation. See [Todmy/PBaaS#10](https://github.com/Todmy/PBaaS/issues/10) for the full PRD and [Todmy/PBaaS#11](https://github.com/Todmy/PBaaS/issues/11) for this slice.

## What this is

You hand pixel-mcp a Figma Frame (or a screenshot) and your Render — it tells you the structured Deltas between the two, gated through escalating Levels of Validators (CV → DINOv2 → VLM → human). The Convergence Loop runs the cheapest Validator on every Iteration and only promotes to the expensive ones when the cheap ones Gate-Pass. See [CONTEXT.md](CONTEXT.md) for the full vocabulary.

Language choice: Python. Rationale in [docs/adr/0001-python-over-typescript.md](docs/adr/0001-python-over-typescript.md).

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone <this-repo>
cd pixel-mcp
uv tool install --from . pixel-mcp
```

## Verify

```sh
pixel-mcp doctor
```

You should see four Checks (`python_version`, `playwright`, `figma_token`, `uv`) and a summary line. `playwright` and `figma_token` are amber until later slices wire them in — that's expected for Slice 1.

For the raw AXI envelope:

```sh
pixel-mcp doctor --json
```

## Wire into Claude Code

```sh
claude mcp add pixel-mcp -- pixel-mcp mcp
```

This registers the MCP server over stdio. In Slice 1 the server exposes one tool: `mcp__pixel_mcp__doctor`, returning the same AXI envelope as the CLI.

## Dev quickstart

```sh
git clone <this-repo>
cd pixel-mcp
uv sync                       # link the workspace
uv run pixel-mcp doctor       # smoke
uv run pytest                 # tests
uv run pre-commit install     # wire git hooks
```

## Project config (`.pixel-mcp.json`, optional)

Drop a `.pixel-mcp.json` at the project root to override built-in defaults.
CLI flags > config file > built-in defaults.

| Key | Default | Notes |
|---|---|---|
| `max_iterations` | `15` | Loop ceiling per session |
| `stuck_threshold` | `3` | Identical delta-hash count that trips STUCK |
| `enabled_levels` | `[0]` | Escalation gates active for this project |
| `ssim_threshold` | `0.97` | Level 0 SSIM Gate Pass |
| `min_bbox_area` | `100` | Hot Region filter (px²) |
| `enable_dinov2` | `false` | Opt in to Level 1 (DINOv2 per-crop similarity) |
| `dinov2_threshold` | `0.95` | Cosine-similarity threshold for Level 1 Gate Pass |
| `viewport` | `{width: 1280, height: 720}` | Default browser viewport |
| `mask_regions` | `[]` | CSS selectors to mask from visual diff |

### Level 1 (DINOv2) escalation gate

Once Level 0 (CV) Gate-Passes, DINOv2 scores every residual Hot Region crop
(expected vs actual) with cosine similarity over CLS-token embeddings. Pass
condition: every crop's similarity ≥ `dinov2_threshold`. Any failing crop
emits a pseudo-Delta (`property=dinov2_similarity_<n>`) with severity
derived from the similarity gap:

- gap ≥ 0.15 → critical
- gap ≥ 0.05 → major
- otherwise → minor

The gate is **opt-in**: pass `--enable-dinov2` on the CLI, set
`enable_dinov2: true` in `.pixel-mcp.json`, or pass `enable_dinov2=true` to
the MCP `check` tool. Requires the ML extras:

```sh
uv tool install pixel-mcp-ml --extra dinov2
```

If the extras aren't installed but the gate is enabled, `check` falls back
to the Level 0 verdict and emits an AXI hint with the install command — it
never crashes the loop.

## Project layout

```
pixel-mcp/
├── CONTEXT.md                  # Canonical domain glossary
├── docs/adr/                   # Architecture decision records
├── pyproject.toml              # uv workspace root
└── packages/
    ├── shared/                 # Internal helpers (AXI envelope). Not published.
    ├── pixel-mcp/              # Core CLI + MCP server. Published.
    └── pixel-mcp-ml/           # ML extras (DINOv2, OmniParser, VLM). Stub until Slice 6+.
```

## License

Apache-2.0. See [LICENSE](LICENSE).
