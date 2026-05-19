# ADR-0002: Escalation gates over parallel multi-signal validation

**Status**: Accepted
**Date**: 2026-05-19
**Decided by**: Grilling session for PRD [#10](https://github.com/Todmy/PBaaS/issues/10)

## Context

pixel-mcp judges whether a Render matches a Design Source using multiple validators of varying cost and signal quality:

- **CV stack** (odiff + OpenCV + DOM measurement + Figma token comparison) — near-zero per-invocation cost, deterministic, narrow signal (catches structural drift but misses semantic equivalence)
- **DINOv2 per-crop similarity** — ~400MB model, local inference (~50-200ms per crop pair), perceptual similarity signal
- **VLM verification** (Qwen2.5-VL local or Claude Sonnet API) — heavy: $0.003-0.015 per call, ~2-5s latency, semantic judgment
- **Human review** (side-by-side via Claude Code chat) — bottleneck of human attention, ground truth

Industry standard for visual regression tools (Percy, Applitools, Chromatic) is **single-pass with all enabled signals fired in parallel**. The intuition: more signals = richer judgment.

This approach has a documented failure mode for our use case (per [buildmvpfast.com 2026](https://www.buildmvpfast.com/blog/figma-to-code-pixel-perfect-loop-ai-agent-screenshot-iterate-2026) and [vadim.blog 2026](https://vadim.blog/pixel-perfect-playwright-figma-mcp)): when an AI judge fires on every Iteration of a Convergence Loop, the AI self-confirms wrong outputs ("looks fine" while button is broken). The expensive signal becomes the noisy signal — exact opposite of intent.

## Decision

**Escalation gates, not parallel signals.** Each Validator is a *gate* — it only runs when the previous, cheaper Validator has passed. Failures at any gate flow back to the Agent; loop restarts from Level 0 next Iteration.

Pipeline shape:

```
Level 0 (CV)      → Gate Pass → promote → Level 1
Level 1 (DINOv2)  → Gate Pass → promote → Level 2
Level 2 (VLM)     → Gate Pass → promote → Level 3
Level 3 (Human)   → Approval → Final Convergence (exit 0)

Any gate failure → deltas back to Agent, restart from Level 0 next Iteration
```

## Consequences

**Positive:**
- **Loop economics**: each Iteration costs near-zero (only Level 0 runs on most ticks). Heavy validators fire ≤1× per Convergence Loop run.
- **AI judgment isolated**: the VLM only sees Level 1-passed candidates. It judges seriously rather than rubber-stamping noisy candidates.
- **Failure attribution clarity**: when a level fails, we know exactly which level failed. Easier to diagnose than "all signals voted, one disagreed".
- **Cost predictability**: max VLM calls per Convergence Loop = 1 per Final Convergence achieved (best case) or zero (loop aborted before Level 2). Bounded.

**Negative:**
- **Slower to "perfect"**: each level adds one more Iteration to reach Final Convergence. If user converges to Level 0 in 5 Iterations, then needs Level 1 fixes (3 more), then Level 2 fixes (2 more) — 10 Iterations vs ~5 with parallel approach.
- **Coupled to Validator ordering**: if a future Validator is *cheaper* than Level 1 but *higher signal* than Level 2, ordering becomes ambiguous. Mitigated by treating Levels as named tiers, not just integers.
- **Promotion bookkeeping**: state must track "highest level reached" for regression detection (was at Level 2, now failing Level 0 → regression flag).

**Mitigations:**
- Allow user to skip Levels via config (`enabled_levels: [0, 2]` skips Level 1) — they trade off thoroughness for speed when context warrants
- Doctor command warns about non-monotonic level enablement (e.g. Level 2 enabled but Level 1 disabled — surprising cost order)

## Alternatives Considered

### Option 1 — Parallel signals (industry default)
- All enabled Validators fire each Iteration. Convergence Judge combines results.
- **Pros**: Richer signal per Iteration; less coupled to Validator ordering
- **Cons**: AI self-confirms noise; cost-per-Iteration scales with enabled Validators; documented failure mode for AI-driven Convergence Loops
- **Why rejected**: Direct contradiction with PRD problem statement (the failure mode we're solving for)

### Option 2 — Hybrid (cheap Validators every Iteration, expensive only on Final candidate)
- Levels 0 + 1 fire each Iteration; Level 2 only fires when both pass
- **Pros**: Compromise between parallel and gates
- **Cons**: Level 1 (DINOv2) is local — feasible. But still runs every Iteration even when Level 0 fails — wasted compute.
- **Why rejected**: Marginal cost savings, doesn't address the AI noise problem (since Level 2 fires only after Level 1 passes already)

### Option 3 — Escalation gates (chosen)
- See Decision above
- **Pros**: Solves documented failure mode; bounded cost; clear failure attribution
- **Cons**: More Iterations to reach Final Convergence
- **Why chosen**: The problem we're solving IS "AI judges every Iteration and gets it wrong". Gates make that impossible by construction.

## Revisit conditions

- If VLM cost drops 10x and Level 2 latency falls under 200ms → reconsider hybrid Option 2 for richer per-Iteration signal
- If user reports that Final Convergence regularly needs 20+ Iterations because each level adds delay → consider parallel firing of Levels 0+1 (keep Level 2 gated)
- If a new Validator category emerges (e.g. accessibility-AI or design-system-AI) that doesn't fit the cost ordering → revisit ordering, possibly switch to named-tiers without numeric ordering
