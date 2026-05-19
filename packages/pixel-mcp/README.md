# pixel-mcp

Figma to Browser convergence harness with escalation gates. CLI plus MCP server.

This package is the core, published artifact. See the [repo README](../../README.md) for installation and Claude Code wiring.

## Surface (Slice 1)

Implemented:
- `pixel-mcp doctor` — environment Check, returns AXI envelope.
- `pixel-mcp mcp` — launch the MCP server over stdio. Exposes one tool: `doctor`.

Stubbed (one issue per slice):
- `spec`, `measure`, `diff`, `judge`, `check`, `review`, `mapping`, `snapshot`, `reset`.

Each stub prints the tracking issue and exits non-zero.
