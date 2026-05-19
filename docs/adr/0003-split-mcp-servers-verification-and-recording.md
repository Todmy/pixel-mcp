# ADR-0003: Recording lives in a separate repo; pixel-mcp scope is verification only

**Status**: Accepted (revised 2026-05-19 — original assumed monorepo sibling, user clarified separate repo entirely)
**Date**: 2026-05-19
**Decided by**: Grilling session for PRD [#10](https://github.com/Todmy/PBaaS/issues/10)

## Context

During PRD design two adjacent capabilities emerged:

1. **Verification** — verify a Render matches a Design Source through escalating gates. Drives the Convergence Loop. The subject of this PRD.
2. **Recording** — produce humanized walkthrough videos of a Render (synthetic cursor with Bezier paths, typing with jitter, optional AI-generated scripts) for sharing/posting.

Both share some surface area: Playwright, dev server, can theoretically share `mappings.json`. Initial instinct was a monorepo with `pixel-mcp` (verify) and `pixel-demo-mcp` (record) as sibling packages.

The user clarified: recording is a separate MCP server, lives in an entirely separate repo. The two domains are independent products that may share Playwright as common infrastructure but otherwise have separate release cadences, separate brands, separate user audiences (verification = devs in a Convergence Loop; recording = devs/marketers producing demo content).

## Decision

**`pixel-mcp` (verification) and the future recording MCP live in separate repos.**

pixel-mcp repo (`~/github/pixel-mcp/`) contains only verification:

```
pixel-mcp/                                 # Single repo
├── pyproject.toml                         # uv workspace root
├── packages/
│   ├── pixel-mcp/                         # Core MCP server + CLI (published)
│   └── pixel-mcp-ml/                      # Optional ML extras: DINOv2, OmniParser, VLM bridges (published)
├── CONTEXT.md
├── docs/adr/
└── README.md
```

Recording artifact lives in a separate repo (name TBD by the recording project owner). Cross-domain interop, if needed in the future, happens through public contracts (filesystem state schemas, public MCP tool surfaces) — not through shared internal packages.

## Consequences

**Positive:**
- **Pure product positioning** — pixel-mcp is "verify your design implementation". One-line elevator pitch. No marketing surface bleed.
- **Independent release cycles** — bugs/features in recording never affect pixel-mcp releases.
- **No accidental coupling** — recording can't import internal verification modules. Forced contract via public surface (filesystem state, MCP tools).
- **Slim install** — verification users `uv tool install pixel-mcp` without recording-related deps (ffmpeg, ghost-cursor, video encoding).
- **Slim MCP tool surface** — pixel-mcp exposes ~6-10 verification tools. Claude Code context budget stays focused.

**Negative:**
- **Cross-domain code duplication risk** — Playwright session config, browser launch boilerplate, possibly selector-resolver logic could end up duplicated in the recording repo.
- **Cross-domain workflow has more friction** — "verify, then record" requires two installs and possibly two separate sessions; no in-process state sharing.
- **No shared CONTEXT.md** — recording domain glossary develops independently. Risk of term drift if both ever need to interoperate.

**Mitigations:**
- If duplication becomes painful in the future, publish a small `pixel-render-core` PyPI package with the shared Playwright wrapper + selector resolver. Both repos depend on it. Defer until painful (not speculatively).
- State Directory (`.pixel-mcp/`) schema is the public contract for any future tool that wants to read pixel-mcp's output. Schema versioned (`schema_version` field in `state.json`).
- Cross-domain affordances stay possible — pixel-mcp's `check` response can include an affordance pointing at the recording MCP ("Final Convergence achieved — for sharing, see the recording MCP at <url>"). Affordance text references a tool by public name, not by internal module.

## Alternatives Considered

### Option 1 — One MCP server with both verification and recording tools (monolithic)
- **Pros**: Single install, single mental model, in-process state sharing
- **Cons**: Heavy install for verification-only users; bloated tool surface in Claude context; muddled product positioning
- **Why rejected**: Trade-offs all favor split. Monolithic save is one install step; cost is paid every invocation.

### Option 2 — Two MCP servers in one monorepo (original ADR-0003 decision)
- **Pros**: Decoupled distribution, shared internals via `shared/` workspace package, single repo to clone for full ecosystem
- **Cons**: Couples release cadences subtly (one CI, one issue tracker, one git history); blends product positioning of two distinct domains
- **Why rejected (revised)**: User clarified recording is a separate product. Monorepo coupling — even with separate published artifacts — implies a shared product story that doesn't match the user's intent.

### Option 3 — Two MCP servers in separate repos (chosen)
- See Decision above
- **Pros**: Clean separation across every dimension
- **Cons**: Duplication risk; cross-workflow friction
- **Why chosen**: Matches user's stated product intent. Easier to relax later (publish shared core) than to break apart later (split monorepo).

## Revisit conditions

- If verification and recording develop strong workflow coupling (e.g. always invoked together in user practice) → consider publishing a shared `pixel-render-core` package; both repos depend on it
- If Playwright session config or selector resolution diverges across the two repos and causes real bugs → forced consolidation into `pixel-render-core`
- If `pixel-mcp-ml` grows to need its own monorepo split → revisit packaging strategy, but that's a separate decision (not this ADR)
