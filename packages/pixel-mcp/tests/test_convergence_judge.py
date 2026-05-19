"""Unit tests for the ConvergenceJudge Deep Module — pure function tests."""

from __future__ import annotations

import pytest
from pixel_mcp.delta import Delta, Severity
from pixel_mcp.judge import Tolerance, judge_deltas


def _d(severity: Severity, prop: str = "color") -> Delta:
    return Delta(
        selector="#x",
        figma_node_id="1:1",
        property=prop,
        observed="x",
        expected="y",
        severity=severity,
    )


def test_empty_deltas_yield_converged() -> None:
    j = judge_deltas([])
    assert j.converged is True
    assert j.critical_count == 0
    assert j.major_count == 0
    assert j.minor_count == 0
    assert j.regression_count == 0
    assert "zero deltas" in j.summary.lower()


def test_only_minor_yields_converged_under_default_tolerance() -> None:
    j = judge_deltas([_d("minor"), _d("minor", prop="padding_left")])
    assert j.converged is True
    assert j.minor_count == 2
    assert "within Tolerance" in j.summary


def test_one_critical_blocks_convergence() -> None:
    j = judge_deltas([_d("critical")])
    assert j.converged is False
    assert j.critical_count == 1
    assert "1 critical" in j.summary


def test_one_major_blocks_convergence() -> None:
    j = judge_deltas([_d("major")])
    assert j.converged is False
    assert j.major_count == 1


def test_regression_blocks_convergence() -> None:
    j = judge_deltas([_d("regression")])
    assert j.converged is False
    assert j.regression_count == 1


def test_strict_tolerance_treats_minor_as_blocking() -> None:
    j = judge_deltas(
        [_d("minor"), _d("minor")],
        tolerance=Tolerance(treat_minor_as_blocking=True),
    )
    assert j.converged is False
    assert j.minor_count == 2


def test_pure_function_deterministic() -> None:
    deltas = [_d("critical"), _d("major"), _d("minor")]
    j1 = judge_deltas(deltas)
    j2 = judge_deltas(deltas)
    assert j1 == j2


@pytest.mark.parametrize(
    ("severities", "expected_converged"),
    [
        ([], True),
        (["minor"], True),
        (["minor", "minor"], True),
        (["major"], False),
        (["critical"], False),
        (["regression"], False),
        (["minor", "critical"], False),
    ],
)
def test_convergence_table(severities: list[Severity], expected_converged: bool) -> None:
    j = judge_deltas([_d(s) for s in severities])
    assert j.converged is expected_converged


def test_summary_counts_match_aggregate() -> None:
    deltas = [_d("critical"), _d("critical"), _d("major"), _d("minor"), _d("regression")]
    j = judge_deltas(deltas)
    assert j.critical_count == 2
    assert j.major_count == 1
    assert j.minor_count == 1
    assert j.regression_count == 1
