# ADR-0001: Python over TypeScript for pixel-mcp

**Status**: Accepted
**Date**: 2026-05-19
**Decided by**: Grilling session for PRD [#10](https://github.com/Todmy/PBaaS/issues/10)

## Context

pixel-mcp ships as a CLI + MCP server with a deterministic CV pipeline (Level 0) and optional ML plugins (Levels 1–2: DINOv2 per-crop similarity, OmniParser element detection, VLM verification).

The user's existing `*-axi` tooling (`gh-axi`, `chrome-devtools-axi`) is distributed via npm as Node.js binaries — this established a TypeScript convention in their immediate environment. The natural default would be TypeScript.

However, the ML stack we depend on is Python-native:
- **DINOv2** — HuggingFace transformers, no production-grade JS binding without ONNX gymnastics
- **OmniParser** — Microsoft research code, Python-only
- **Qwen2.5-VL / Qwen3-VL** — Python via transformers OR Ollama (language-agnostic at API boundary, but tooling is Python-first)
- **OpenCV** — has WASM binding but feature-incomplete and slow
- **Pillow** — no comparable JS image library at this level

Three architectural options were considered:
1. **TypeScript core + Python sidecar** — TS does Playwright/odiff/state; Python subprocess handles ML
2. **Pure TypeScript with ONNX/WASM ML** — drag ML into JS via ONNX runtime, accept feature loss
3. **Pure Python** — Python all the way down

## Decision

**Pure Python.** Distribute via `uv tool install pixel-mcp`. The `*-axi` naming convention is treated as language-agnostic — it names the brand pattern, not the runtime.

## Consequences

**Positive:**
- Zero interop tax — Python calls Python natively
- All ML deps install in one pyproject.toml extras
- `uv` distribution in 2026 matches `npm i -g` DX and is faster
- Playwright Python is mature (parity with TS Playwright on features we use)
- MCP has official Python SDK
- PBaaS existing scripts (`news-scan.py`, `tg-post.py`, `hype_finder.py`) are already Python — consistent with author's scripting environment

**Negative:**
- Breaks visual convention with sibling tools `gh-axi` (TS) and `chrome-devtools-axi` (TS). Users may expect `npm i -g pixel-mcp`.
- Python startup time slightly higher than Node for cold invocations (~150ms vs ~50ms) — material for sub-second loops, negligible for our use case (each Iteration takes seconds anyway).
- TypeScript users in the ecosystem cannot easily fork and extend without learning Python.

**Mitigations:**
- Document language choice prominently in README
- Provide a thin Node shim (`pixel-mcp-shim`) that subprocess-calls the Python binary — for users who want npm-installable wrapper (post-v1 if requested)

## Alternatives Considered

### Option 1 — TypeScript core + Python sidecar
- **Pros**: Matches `*-axi` naming convention literally; Node startup faster
- **Cons**: Two-language codebase, two install paths, two debugging stacks. Filesystem/JSON IPC adds latency per ML call. Sidecar lifecycle management is its own complexity.
- **Why rejected**: Complexity tax not justified — the bulk of pixel-mcp's hot path is ML-adjacent. Two-language split would touch every architectural seam.

### Option 2 — Pure TypeScript with ONNX/WASM
- **Pros**: Truly single-language, single-install
- **Cons**: DINOv2 ONNX inference in Node lags reference Python by 3-5x. OmniParser has no ONNX export. WASM OpenCV missing key contour functions. Months of work to chase parity.
- **Why rejected**: Speculative engineering against an unstable target. Even if it worked, model authors release updates for Python first; we'd always lag.

### Option 3 — Pure Python (chosen)
- **Pros**: Above
- **Cons**: Above
- **Why chosen**: Smallest delta from working code. Best amortization of effort over ML-heavy roadmap.

## Revisit conditions

Reconsider if any of these become true:
- TypeScript-based ML inference reaches Python parity (likely never for our model set)
- Anthropic ships an official Node MCP server template that becomes ecosystem-standard
- DINOv2 or successor model ships with a maintained TS-native wrapper of equivalent quality
