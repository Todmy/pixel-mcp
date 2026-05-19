"""ConvergenceJudge — the Deep Module behind ``pixel-mcp judge``.

Public entry point: :func:`judge_deltas`. Takes a list of Deltas (plus an
optional Tolerance override) and returns a Judgment with ``converged`` flag,
a human-readable summary, and per-severity counts.

Pure function — same Delta[] always yields the same Judgment.

Default Gate Pass policy (per Slice #14 spec):

    Final Convergence (within Tolerance, naïve v0) holds when
    ``critical_count == 0 AND major_count == 0``.

``minor`` Deltas are informational — they don't block convergence by
default. The Tolerance object can flip this for strict projects.
``regression`` Deltas (introduced in Slice #9) always block.
"""

from __future__ import annotations

from pydantic import BaseModel

from pixel_mcp.delta import Delta

__all__ = ["Judgment", "Tolerance", "judge_deltas"]


class Tolerance(BaseModel):
    """Convergence policy overrides.

    v0 surface is intentionally minimal — per-property numeric thresholds
    arrive in Slice #10 (``.pixel-mcp.json`` Project Config) which will
    extend this model.
    """

    treat_minor_as_blocking: bool = False
    """If True, even a single ``minor`` Delta keeps the loop running."""


class Judgment(BaseModel):
    """Result of running the ConvergenceJudge on a Delta[]."""

    converged: bool
    summary: str
    critical_count: int = 0
    major_count: int = 0
    minor_count: int = 0
    regression_count: int = 0


def judge_deltas(deltas: list[Delta], tolerance: Tolerance | None = None) -> Judgment:
    """Apply Gate Pass policy to a Delta[] and return a Judgment.

    Args:
        deltas: Deltas produced by :func:`pixel_mcp.delta.diff_design_vs_render`.
        tolerance: Optional override; uses default policy when omitted.

    Returns:
        Judgment with ``converged`` and per-severity counts.
    """
    tol = tolerance or Tolerance()

    critical = sum(1 for d in deltas if d.severity == "critical")
    major = sum(1 for d in deltas if d.severity == "major")
    minor = sum(1 for d in deltas if d.severity == "minor")
    regression = sum(1 for d in deltas if d.severity == "regression")

    blocking = critical + major + regression
    if tol.treat_minor_as_blocking:
        blocking += minor

    converged = blocking == 0

    if converged:
        if minor == 0:
            summary = "Final Convergence reached — zero Deltas at this Level."
        else:
            summary = (
                f"Final Convergence within Tolerance — {minor} minor Delta"
                f"{'s' if minor != 1 else ''} remain (informational)."
            )
    else:
        parts = []
        if critical:
            parts.append(f"{critical} critical")
        if major:
            parts.append(f"{major} major")
        if regression:
            parts.append(f"{regression} regression")
        if tol.treat_minor_as_blocking and minor:
            parts.append(f"{minor} minor")
        summary = "Blocked by " + ", ".join(parts) + " Delta" + ("s" if blocking != 1 else "") + "."

    return Judgment(
        converged=converged,
        summary=summary,
        critical_count=critical,
        major_count=major,
        minor_count=minor,
        regression_count=regression,
    )
