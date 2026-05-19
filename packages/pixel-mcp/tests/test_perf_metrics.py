"""Tests for the v3-1 Perf Metrics module (collector + budget judge).

Mocks Playwright so no real page loads happen. Synthetic
``page.evaluate`` return values stand in for the
``PerformanceObserver`` / ``getEntriesByType`` payload the production
script produces inside the browser.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pixel_mcp.perf_metrics import (
    PerfBudget,
    PerfMetrics,
    collect_perf_metrics,
    judge_perf_metrics,
)

# ---------------------------------------------------------------------------
# collect_perf_metrics — Playwright mocks
# ---------------------------------------------------------------------------


def _fake_playwright(evaluate_payload: dict[str, Any]) -> Any:
    """Return a ``sync_playwright`` mock whose ``page.evaluate`` yields the payload."""
    page = MagicMock()
    page.goto = MagicMock()
    page.wait_for_load_state = MagicMock()
    page.wait_for_selector = MagicMock()
    page.evaluate = MagicMock(return_value=evaluate_payload)

    context = MagicMock()
    context.new_page = MagicMock(return_value=page)

    browser_obj = MagicMock()
    browser_obj.new_context = MagicMock(return_value=context)
    browser_obj.close = MagicMock()

    launcher = MagicMock()
    launcher.launch = MagicMock(return_value=browser_obj)

    p = MagicMock()
    p.chromium = launcher
    p.firefox = launcher
    p.webkit = launcher

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=p)
    cm.__exit__ = MagicMock(return_value=False)

    sync_playwright = MagicMock(return_value=cm)
    return sync_playwright, page


def test_collect_perf_metrics_extracts_fcp_lcp_cls() -> None:
    """Synthetic perf payload → PerfMetrics populated correctly."""
    payload = {
        "fcp_ms": 1200.5,
        "lcp_ms": 2100.0,
        "cls": 0.05,
        "ttfb_ms": 350.0,
        "dcl_ms": 800.0,
        "load_ms": 2500.0,
    }
    sync_pw, _page = _fake_playwright(payload)

    fake_module = MagicMock()
    fake_module.sync_playwright = sync_pw
    fake_module.Error = type("Error", (Exception,), {})
    fake_module.TimeoutError = type("TimeoutError", (Exception,), {})

    with patch.dict(
        "sys.modules",
        {"playwright": MagicMock(), "playwright.sync_api": fake_module},
    ):
        metrics = collect_perf_metrics(
            route="http://localhost:3000/",
            viewport=(1280, 720),
            browser="chromium",
        )

    assert isinstance(metrics, PerfMetrics)
    assert metrics.route == "http://localhost:3000/"
    assert metrics.viewport == (1280, 720)
    assert metrics.browser == "chromium"
    assert metrics.fcp_ms == pytest.approx(1200.5)
    assert metrics.lcp_ms == pytest.approx(2100.0)
    assert metrics.cls == pytest.approx(0.05)
    assert metrics.ttfb_ms == pytest.approx(350.0)
    assert metrics.dcl_ms == pytest.approx(800.0)
    assert metrics.load_ms == pytest.approx(2500.0)


def test_collect_handles_missing_entries_with_none() -> None:
    """Partial payload (e.g. webkit lacks LCP) → PerfMetrics has None for missing fields."""
    payload = {
        "fcp_ms": 900.0,
        "lcp_ms": None,
        "cls": None,
        "ttfb_ms": 200.0,
        "dcl_ms": None,
        "load_ms": None,
    }
    sync_pw, _page = _fake_playwright(payload)

    fake_module = MagicMock()
    fake_module.sync_playwright = sync_pw
    fake_module.Error = type("Error", (Exception,), {})
    fake_module.TimeoutError = type("TimeoutError", (Exception,), {})

    with patch.dict(
        "sys.modules",
        {"playwright": MagicMock(), "playwright.sync_api": fake_module},
    ):
        metrics = collect_perf_metrics(
            route="http://localhost:3000/",
            viewport=(375, 667),
            browser="webkit",
        )

    assert metrics.fcp_ms == pytest.approx(900.0)
    assert metrics.lcp_ms is None
    assert metrics.cls is None
    assert metrics.ttfb_ms == pytest.approx(200.0)
    assert metrics.dcl_ms is None
    assert metrics.load_ms is None
    assert metrics.browser == "webkit"


# ---------------------------------------------------------------------------
# judge_perf_metrics — severity gradient
# ---------------------------------------------------------------------------


def _metrics(**fields: float | None) -> PerfMetrics:
    defaults: dict[str, Any] = {
        "route": "http://localhost:3000/",
        "viewport": (1280, 720),
        "browser": "chromium",
        "collected_at": datetime.now(UTC),
        "fcp_ms": None,
        "lcp_ms": None,
        "cls": None,
        "ttfb_ms": None,
        "dcl_ms": None,
        "load_ms": None,
    }
    defaults.update(fields)
    return PerfMetrics.model_validate(defaults)


def test_judge_perf_within_budget_no_deltas() -> None:
    """Observed FCP 1500ms vs budget 1800ms → no Delta."""
    metrics = _metrics(fcp_ms=1500.0)
    budget = PerfBudget(fcp_ms=1800.0)
    deltas = judge_perf_metrics(metrics, budget)
    assert deltas == []


def test_judge_perf_over_budget_emits_critical() -> None:
    """Observed FCP 3000ms vs budget 1800ms (~67% over) → critical Delta."""
    metrics = _metrics(fcp_ms=3000.0)
    budget = PerfBudget(fcp_ms=1800.0)
    deltas = judge_perf_metrics(metrics, budget)
    assert len(deltas) == 1
    delta = deltas[0]
    assert delta.property == "perf_fcp_ms"
    assert delta.severity == "critical"
    assert delta.observed == pytest.approx(3000.0)
    assert delta.expected == pytest.approx(1800.0)
    assert delta.magnitude == pytest.approx(1200.0)


@pytest.mark.parametrize(
    "pct_over,expected_severity",
    [
        (5, "minor"),  # 5% over → minor (boundary)
        (15, "minor"),  # 15% over → minor
        (30, "major"),  # 30% over → major
        (60, "critical"),  # 60% over → critical
    ],
)
def test_judge_perf_severity_thresholds(pct_over: float, expected_severity: str) -> None:
    """Parametrize across the severity gradient."""
    budget_value = 1000.0
    observed = budget_value * (1.0 + pct_over / 100.0)
    metrics = _metrics(fcp_ms=observed)
    budget = PerfBudget(fcp_ms=budget_value)
    deltas = judge_perf_metrics(metrics, budget)
    assert len(deltas) == 1
    assert deltas[0].severity == expected_severity


def test_cls_judged_against_unit_value() -> None:
    """CLS 0.15 vs budget 0.1 → 50% over, major."""
    metrics = _metrics(cls=0.15)
    budget = PerfBudget(cls=0.1)
    deltas = judge_perf_metrics(metrics, budget)
    assert len(deltas) == 1
    delta = deltas[0]
    assert delta.property == "perf_cls"
    assert delta.severity == "major"
    # Magnitude is the raw difference (not ms-over).
    assert delta.magnitude == pytest.approx(0.05, abs=1e-6)


def test_judge_skips_unmeasurable_metrics_silently() -> None:
    """Metrics that are None (platform didn't expose them) emit no Delta."""
    metrics = _metrics(fcp_ms=None, lcp_ms=None, cls=None)
    budget = PerfBudget(fcp_ms=1800.0, lcp_ms=2500.0, cls=0.1)
    assert judge_perf_metrics(metrics, budget) == []


def test_judge_skips_unbudgeted_fields() -> None:
    """Budget left None per field → that field doesn't emit Deltas even when high."""
    metrics = _metrics(fcp_ms=5000.0, lcp_ms=5000.0)
    budget = PerfBudget(fcp_ms=1800.0)  # lcp_ms not budgeted
    deltas = judge_perf_metrics(metrics, budget)
    # Only the fcp_ms Delta should fire.
    assert len(deltas) == 1
    assert deltas[0].property == "perf_fcp_ms"


def test_judge_emits_deterministic_order() -> None:
    """Multiple budget violations → stable property-name order."""
    metrics = _metrics(fcp_ms=4000.0, lcp_ms=5000.0, ttfb_ms=2000.0)
    budget = PerfBudget(fcp_ms=1800.0, lcp_ms=2500.0, ttfb_ms=800.0)
    deltas = judge_perf_metrics(metrics, budget)
    properties = [d.property for d in deltas]
    # Insertion follows the declared _BUDGETED_FIELDS order.
    assert properties == ["perf_fcp_ms", "perf_lcp_ms", "perf_ttfb_ms"]
