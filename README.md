# pixel-mcp

Figma to Browser convergence harness with escalation gates. CLI plus MCP server for Claude Code.

> **Status**: v3-1 shipped (visual + perf gates complete). PRD: [Todmy/PBaaS#10](https://github.com/Todmy/PBaaS/issues/10).

## What this is

You hand pixel-mcp a Figma Frame (or a screenshot) and your Render — it tells you the structured Deltas between the two, gated through escalating Levels of Validators (CV → DINOv2 → VLM → human) plus orthogonal axes (multi-viewport, cross-browser, performance budgets). The Convergence Loop runs the cheapest Validator on every Iteration and only promotes to the expensive ones when the cheap ones Gate-Pass. See [CONTEXT.md](CONTEXT.md) for the full vocabulary.

Language: Python. Rationale in [docs/adr/0001-python-over-typescript.md](docs/adr/0001-python-over-typescript.md).

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/Todmy/pixel-mcp.git
cd pixel-mcp
uv tool install --from . pixel-mcp
uv run playwright install chromium
```

For optional ML extras (DINOv2, VLM, OmniParser) and additional browsers:

```sh
# DINOv2 per-crop perceptual similarity (Level 1 gate)
uv tool install pixel-mcp-ml --extra dinov2

# VLM judgment via Claude API (Level 2 gate)
uv tool install pixel-mcp-ml --extra vlm

# OmniParser UI element detection (semantic region labels)
uv tool install pixel-mcp-ml --extra omniparser

# Cross-browser support
uv run playwright install firefox webkit
```

## Verify

```sh
pixel-mcp doctor
```

Reports the status of every dependency: Python version, Playwright, Chromium/Firefox/WebKit binaries, Figma token, httpx, ML extras (DINOv2/VLM/OmniParser), Ollama+Qwen2.5-VL local backend. Each extra is opt-in — missing ones surface as amber/red but don't block core usage.

For the raw AXI envelope:

```sh
pixel-mcp doctor --json
```

## Wire into Claude Code

```sh
claude mcp add pixel-mcp -- pixel-mcp mcp
```

Registers the MCP server over stdio. Exposes 11 tools: `doctor`, `spec`, `measure`, `diff`, `judge`, `check`, `mapping`, `snapshot`, `reset`, `review`, `human_feedback`.

## Quick run

```sh
# Figma mode
pixel-mcp check --figma 'https://www.figma.com/design/<id>?node-id=<n>' \
                --route http://localhost:3000/component

# Image-only mode (no Figma access)
pixel-mcp check --image design.png \
                --route http://localhost:3000/component

# Multi-viewport
pixel-mcp check --figma <url> --route <url> \
                --viewports-preset responsive   # 1280x720,768x1024,375x667

# Cross-browser
pixel-mcp check --figma <url> --route <url> \
                --browsers-preset all           # chromium,firefox,webkit

# Full convergence loop with all gates
pixel-mcp check --figma <url> --route <url> \
                --enable-dinov2 \
                --enable-vlm \
                --enable-human-gate \
                --enable-omniparser \
                --enable-perf \
                --perf-budget '{"fcp_ms": 1800, "lcp_ms": 2500, "cls": 0.1}'
```

Exit codes drive Loop Runners (Ralph Loop, Makefile, CI):
- `0` — Final Convergence at the highest enabled Level
- `1` — Deltas present; the Agent should fix and re-invoke
- `2` — Ready for Level 3 human review
- `3` — Regression detected
- `10` — Max iterations exceeded
- `11` — Stuck (same delta hash N times in a row)
- `12` — Fatal (Figma/Render/IO/CLI error)

## Escalation gates — the core convergence pattern

Each Level is opt-in. Level N runs only after Level N-1 Gate-Passes.

| Level | Validator | Cost | How to enable |
|---|---|---|---|
| 0 | CV deterministic: odiff + OpenCV Hot Regions + DOM measurement + Figma token comparison | Near-zero per Iteration | Always on |
| 1 | DINOv2 per-crop cosine similarity (default threshold 0.95) | ~400MB model, local CPU/GPU | `--enable-dinov2` |
| 2 | VLM verification — Claude Sonnet (cloud) or Qwen2.5-VL via Ollama (local) | One API call / per crop | `--enable-vlm` |
| 3 | Human side-by-side review via Claude Code chat image attachments | Human time | `--enable-human-gate` |

Each level's failure resets the loop to Level 0 — the Agent fixes Level 0 Deltas before re-attempting Level 1. Pseudo-Deltas synthesized from non-structural signals (Hot Regions, DINOv2 similarities, VLM verdicts, human rejection notes, perf budget misses) flow through the same `ConvergenceJudge` and `hash_deltas_bucketed` machinery so stuck/regression detection works uniformly.

## Orthogonal axes

Beyond escalation gates, pixel-mcp runs the full pipeline across:

- **Multi-viewport** — `--viewports 1280x720,768x1024,375x667` or `--viewports-preset responsive`
- **Cross-browser** — `--browsers chromium,firefox,webkit` or `--browsers-preset all`
- **Performance budgets** — `--enable-perf --perf-budget '{"fcp_ms": 1800, ...}'` — Core Web Vitals as gate

When combined: `browsers × viewports` cross-product per Iteration. Overall `converged` is the AND-fold; aggregated Deltas carry `browser` + `viewport` fields so the Agent knows which cell failed.

## Dev quickstart

```sh
git clone https://github.com/Todmy/pixel-mcp.git
cd pixel-mcp
uv sync                       # link the workspace
uv run pixel-mcp doctor       # smoke
uv run pytest                 # 396 tests
uv run pre-commit install     # wire git hooks
```

## Project config (`.pixel-mcp.json`, optional)

Drop at project root to override built-in defaults. CLI flags > config file > built-in defaults.

Most-asked keys:

| Key | Default | Notes |
|---|---|---|
| `max_iterations` | `15` | Loop ceiling per session |
| `stuck_threshold` | `3` | Identical delta-hash count that trips STUCK |
| `ssim_threshold` | `0.97` | Level 0 SSIM Gate Pass |
| `min_bbox_area` | `100` | Hot Region filter (px²) |
| `enable_dinov2` | `false` | Opt in to Level 1 |
| `dinov2_threshold` | `0.95` | Cosine-similarity floor for Level 1 Gate Pass |
| `enable_vlm` | `false` | Opt in to Level 2 |
| `vlm_backend` | `"claude"` | `claude` (cloud) or `qwen-local` (Ollama) |
| `vlm_threshold` | `0.7` | VLM confidence floor for Level 2 Gate Pass |
| `enable_human_gate` | `false` | Opt in to Level 3 |
| `enable_omniparser` | `false` | UI element labeling on Regions |
| `enable_perf` | `false` | Performance budgets gate |
| `perf_budget` | `null` | `{"fcp_ms": 1800, "lcp_ms": 2500, "cls": 0.1, ...}` |
| `enabled_browsers` | `["chromium"]` | List of Playwright engines |
| `viewport` | `{width: 1280, height: 720}` | Default single viewport (when `--viewports` not set) |
| `mask_regions` | `[]` | CSS selectors to mask from visual diff |

## Project layout

```
pixel-mcp/
├── CONTEXT.md                  # Canonical domain glossary
├── docs/adr/                   # Architecture decision records
├── pyproject.toml              # uv workspace root
└── packages/
    ├── shared/                 # Internal AXI envelope helper. Not published.
    ├── pixel-mcp/              # Core CLI + MCP server. Published.
    │   └── src/pixel_mcp/
    │       ├── cli.py
    │       ├── check_cmd.py    # Convergence Loop orchestrator
    │       ├── spec.py         # DesignSpec extractor (Figma)
    │       ├── render.py       # Playwright wrapper + screenshot
    │       ├── delta.py        # DeltaDiffer (pure function)
    │       ├── judge.py        # ConvergenceJudge (pure function)
    │       ├── normalize.py    # Width-handling Normalizer
    │       ├── hot_regions.py  # SSIM + OpenCV bbox clustering
    │       ├── decompose.py    # Hot Region → DOM attribution + crops
    │       ├── mapping.py      # Figma ↔ DOM selector resolver
    │       ├── loop_state.py   # Iteration counter, stuck, regression
    │       ├── perf_metrics.py # Core Web Vitals collection + judge
    │       ├── review_cmd.py   # Level 3 review packet builder
    │       ├── human_feedback_cmd.py
    │       ├── project_config.py
    │       ├── doctor.py
    │       └── mcp_server.py
    └── pixel-mcp-ml/           # Optional ML extras (DINOv2, VLM, OmniParser)
        └── src/pixel_mcp_ml/
            ├── dinov2_compare.py
            ├── vlm_verify.py
            └── omniparser_detect.py
```

## License

Apache-2.0. See [LICENSE](LICENSE).
