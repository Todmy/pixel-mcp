# pixel-mcp — Domain Glossary

> Source PRD: [Todmy/PBaaS#10](https://github.com/Todmy/PBaaS/issues/10)
> ADRs: [`docs/adr/`](docs/adr/)

This glossary captures the canonical vocabulary for the pixel-mcp domain. Terms here are the names we use in user-facing surfaces (CLI, MCP tools, docs, error messages) and in internal module names. Where two concepts could blur, we pick one and stick with it.

---

## Core concepts

**Design Source** — the reference appearance we're matching against. Two kinds:
- **Figma Source** — a URL pointing at a Figma Frame, Component Instance, or Master Component. Structured tokens available.
- **Image Source** — a static screenshot (PNG/JPG); no structured spec.

**Figma Source — supported reference types** (in priority order):
- **Figma Frame** — a Frame node with auto-layout. The canonical primary input. `type: "FRAME"` from Figma API. Provides full `layoutSizingHorizontal`, padding, itemSpacing, alignment tokens.
- **Figma Component Instance** — an instance of a master component. `type: "INSTANCE"`. Resolved to master + overrides at extraction time.
- **Figma Master Component** — a master component (in a library or local). `type: "COMPONENT"`. Sealed, no overrides.

**Figma Source — explicitly unsupported types** (return clear error in `doctor`/`spec`):
- Group without auto-layout (`type: "GROUP"`) — no semantic constraints; use image-only mode instead.
- Whole-file URL without `node-id` — too broad.
- Pages, sections, vector layers — not coherent designs to match against.

URL format: both `figma.com/file/<id>` and `figma.com/design/<id>` prefixes are accepted (extract `node-id` from query string).

(Removed from glossary: generic "Figma Node" — too vague. Always say which of the three supported types.)

Synonyms to avoid: "design", "reference", "expected", "spec source" (when context could mean something else). The thing the user passes via `--figma` or `--image` is always a **Design Source**.

**Render** — the actual visual output produced by the browser given the current code. The thing we're comparing the Design Source against. Synonyms to avoid: "actual", "output", "implementation result".

**DesignSpec** — structured intermediate representation extracted from a Design Source. Contains: layout tree, computed dimensions, color tokens, type tokens, spacing tokens, constraints. JSON-serializable.

**MeasuredDOM** — structured intermediate representation extracted from a Render. Contains per-element: bounding box, computed styles (color/font/spacing/dimensions), text content, role, parent chain. JSON-serializable.

**Mapping** — a pairing between a Figma node ID and a DOM CSS selector. The data structure is `Mappings` (a list of these pairs, persisted to disk).

---

## The convergence loop

**Verification** — the broader concept of "does Render match Design Source?". The user-facing brand word. Marketing surface.

**Check** — the action a single invocation of pixel-mcp performs. CLI command, MCP tool. One Check produces a result that either passes a Level or fails with deltas.

**Iteration** — one invocation of `pixel-mcp check`. Counted regardless of result. Counter increments on every invocation (productive or debug), persists in `.pixel-mcp/state.json`. Resets on explicit `pixel-mcp reset` OR on Final Convergence (exit 0).

**Productive Iteration** — diagnostic metric: an Iteration where code changed since the previous Iteration (detected via file checksums stored in `.pixel-mcp/file-hashes.json`). Tracked in `history.jsonl`. **Does NOT affect** stuck detection or max-iter logic — pure observability.

**Convergence Loop** — the external iterative process (Ralph Loop, Makefile loop, CI loop) that calls pixel-mcp repeatedly until exit code says done. pixel-mcp does NOT drive the loop itself — it's the inner body.

**Level** — a tier of validation. Levels 0–3.
- **Level 0** — CV-based (odiff + OpenCV + DOM measurement + computed-style diff). Always on. Cheapest.
- **Level 1** — DINOv2 per-crop similarity. Optional plugin.
- **Level 2** — VLM verification (local Qwen2.5-VL OR Claude Sonnet API). Optional.
- **Level 3** — Human gate via Claude Code chat attachments. Optional, interactive.

**Validator** — the checker for a given Level. Each Level has one Validator. (Level 0 Validator = CV stack; Level 1 Validator = DINOv2; etc.)

**Gate Pass** — a single Level's Validator returning OK. NOT the same as Final Convergence.

**Promotion** — moving from Level N to Level N+1 when N's Gate Passes. Promotion always advances by one level; never skips.

**Final Convergence** — all enabled Levels have Gate-Passed within the same Iteration. Exit code 0. This is the loop terminator on the happy path.

**Regression** — a Level had previously Gate-Passed in an earlier Iteration, but in the current Iteration it's failing. Exit code 3.

**Stuck** — last N Iterations produced identical structured-delta hashes. Default N=3. Exit code 11. Hash includes `(selector, property, magnitude_bucket)` tuples, sorted. Magnitude bucketed (not raw float) so jitter doesn't mask stuck state.

---

## What pixel-mcp emits

**Delta** — a single property mismatch. Shape: `{ selector, property, observed, expected, magnitude, severity }`. Property-level. One element may produce multiple Deltas (one per mismatched property).

**Severity** — one of `critical | major | minor | regression`. Drives Gate Pass logic. Source of severity differs by mode:
- **Figma mode** — from Delta property mismatch (e.g. color wrong = critical, dimension off >20% = critical, off 5-20% = major, within 2-5% = minor).
- **Image-only mode** — from Hot Region area (`>50,000px²` = critical, `1,000-50,000` = major, `100-1,000` = minor; below `min_bbox_area` filtered out).

**SSIM Score** — single global scalar in `[0, 1]` from Structural Similarity Index computation over the (Design Source render, Render) pair. Threshold: `ssim_threshold` (default `0.97`). Falls when overall structure drifts. **Global signal — one number per image pair.**

**Hot Region** — a bbox cluster `{ x, y, w, h }` derived from per-pixel diff mask via OpenCV contour detection. Filtered by `min_bbox_area` (default `100px²`) to remove anti-aliasing noise. **Local signal — N bboxes per image pair.** Independent of SSIM Score.

**Level 0 Gate Pass** — passes when ALL three hold:
1. SSIM Score ≥ `ssim_threshold`
2. Zero Hot Regions above `min_bbox_area`
3. Zero `critical` or `major` Deltas (Figma mode only — image-only mode skips this clause)

Hot Region and Delta are NOT the same: one Hot Region may produce zero or many Deltas (in Figma mode) depending on what DOM elements sit inside it. In image-only mode, Hot Regions carry their own severity classification based on area.

**Crop** — a PNG image cut from a Hot Region. Comes in pairs: `expected_crop` (from Design Source) + `actual_crop` (from Render). Used as input to Level 1 (DINOv2) and Level 2 (VLM).

**Tolerance** — per-property acceptable difference threshold. Configurable per-project via `.pixel-mcp.json`. Example: color = exact, spacing = ±2px OR ±2% (larger), typography = ±0.5pt.

---

## Configuration & state

**State Directory** — `.pixel-mcp/` in the project root. Persists across Iterations within a single Convergence Loop run. Contains: state.json, history.jsonl, mappings.json, spec-cache.json, crops/. Survives between Loop runs (on-disk); cleared via `pixel-mcp reset`.

**Project Config** — `.pixel-mcp.json` in project root. Declarative. Holds Tolerance overrides, mask regions, enabled Levels, max-iterations override.

---

## Surfaces

**CLI Subcommand** — a `pixel-mcp <verb>` form. One per pipeline stage (spec, measure, diff, judge, check, review, mapping, snapshot, doctor, reset). Each Subcommand maps 1:1 to a Deep Module.

**MCP Tool** — a callable in Claude Code via `mcp__pixel_mcp__<name>`. Each MCP Tool wraps a CLI Subcommand and returns an AXI-style response envelope.

**AXI Envelope** — the response shape every MCP Tool returns:
```
{ data, hints, diagnostics, next_suggested_action, affordances }
```
This is the AXI pattern. The MCP brand sits atop it for discoverability.

**Affordance** — an entry in the `affordances` list of an AXI Envelope. Points the calling Agent at a follow-up MCP Tool with a one-line "when to use" hint.

**Deep Module** — an internal module with stable, narrow interface, testable in isolation against JSON fixtures. The Normalizer, DeltaDiffer, ConvergenceJudge, MappingResolver, HierarchicalDecomposer, etc.

---

## Roles (who acts on what)

**Agent** — the orchestrating Claude (or other LLM) that's reading Deltas, editing code, and re-invoking pixel-mcp. This is the user's primary AI partner. Singular: always the same entity per Convergence Loop.

**VLM** — a vision-language model that pixel-mcp itself invokes at Level 2 (Qwen2.5-VL local or Claude Sonnet API). Distinct from the Agent: the Agent calls pixel-mcp; pixel-mcp calls the VLM. They never overlap.

**Loop Runner** — the external orchestrator that wraps the iteration logic. Ralph Loop is the primary target. Could be a Makefile, a CI step, or a custom script. pixel-mcp is loop-runner-agnostic.

---

## Naming conventions

- Brand / artifact name: **`pixel-mcp`**
- Repo: **`pixel-mcp/`** (single repo; uv workspace with `packages/pixel-mcp` and `packages/pixel-mcp-ml`)
- Internal pattern: **AXI** (helpers + affordances in responses). Not user-facing.
- Video / walkthrough recording is **out of scope** — separate product, separate repo (name and ownership TBD). Cross-tool interop, if any, happens via published State Directory schema and public MCP surface, not via shared internal code.

---

## Anti-vocabulary (words we don't use)

- ❌ **"Validate"** — too generic, confuses with form validation. Use **Check** (action) or **Verification** (concept).
- ❌ **"Verify the spec"** — Spec is what we extract; we verify the Render against the Design Source. Use "**Check the Render against the Design Source**".
- ❌ **"Snapshot"** as a synonym for Render — Snapshot is a named baseline (via `pixel-mcp snapshot --tag X`). Render is the live result.
- ❌ **"Pixel-perfect"** in code/docs — it's a marketing concept, not a precise term. The precise term is **Final Convergence within Tolerance**.
- ❌ **"Element"** alone — ambiguous. Use **Figma Node** OR **DOM Element** OR **Leaf** explicitly.
