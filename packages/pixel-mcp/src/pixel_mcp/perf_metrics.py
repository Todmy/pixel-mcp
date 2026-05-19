"""Perf Metrics — Core Web Vitals collector and budget judge (v3-1).

Public entry points:

- :func:`collect_perf_metrics` — drives Playwright against ``route`` and
  returns a :class:`PerfMetrics` snapshot (FCP / LCP / CLS / TTFB / DCL /
  load). Reuses the same lazy-import + browser-engine selection pattern as
  :mod:`pixel_mcp.render`; the same :class:`BrowserNotInstalledError` /
  :class:`RouteUnreachableError` hierarchy bubbles up so the check pipeline
  can fold perf failures into the existing fatal-envelope machinery.

- :func:`judge_perf_metrics` — pure function that compares a
  :class:`PerfMetrics` against a :class:`PerfBudget` and synthesises a list
  of pseudo-:class:`Delta` objects (one per budget-exceeded field). The
  Delta severity follows the v3-1 percentage-over-budget gradient:

      ≥ 50 %   over budget → ``critical``
      20–50 %  over budget → ``major``
       5–20 %  over budget → ``minor``
      <  5 %   over budget → no Delta

For CLS (a unitless score, no millisecond magnitude) the same percentage
rule applies relative to the budgeted value; ``magnitude`` is the
difference (not the ms-over).

Design notes
------------

- Single ``page.evaluate`` round-trip handles all measurement; we observe
  ``largest-contentful-paint`` and ``layout-shift`` via ``PerformanceObserver``
  inside the page, then ``disconnect()`` after one ``requestIdleCallback``
  tick. This is the standard idiom — see ``web.dev/lcp`` and ``web.dev/cls``.
- The JS payload returns ``null`` for any metric that is not observable on
  the platform under test (e.g. webkit lacks
  ``largest-contentful-paint`` entries). The Pydantic model carries
  ``Optional[float]`` for the same reason — judging skips ``None`` observed
  values silently.
- Budget comparison is opt-in per field: a ``PerfBudget`` with only
  ``fcp_ms`` set ignores LCP/CLS/etc. This keeps the v3-1 surface trivially
  small while leaving room for richer per-route budgets in ``.pixel-mcp.json``
  later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from pixel_mcp.delta import Delta
from pixel_mcp.render import (
    VALID_BROWSERS,
    BrowserName,
    BrowserNotInstalledError,
    ChromiumNotInstalledError,
    PlaywrightNotInstalledError,
    RenderError,
    RouteUnreachableError,
)

__all__ = [
    "PerfBudget",
    "PerfMetrics",
    "PerfMetricsError",
    "collect_perf_metrics",
    "judge_perf_metrics",
]


class PerfMetricsError(RenderError):
    """Base class for perf-metrics collection failures.

    Subclassing :class:`RenderError` keeps the existing render-error catch
    blocks in :mod:`check_cmd` working uniformly — a perf failure surfaces
    through the same fatal-envelope path as a measure failure when it
    propagates that far. The v3-1 wiring traps it earlier and degrades to a
    hint instead.
    """


# --- Pydantic models -------------------------------------------------------


class PerfMetrics(BaseModel):
    """Core Web Vitals snapshot for a single (route, viewport, browser).

    Fields are ``None`` when the metric was not measurable on the platform
    under test. The judge silently skips ``None`` observed values so a
    partial collection never produces spurious Deltas.
    """

    route: str
    viewport: tuple[int, int]
    browser: str
    collected_at: datetime
    fcp_ms: float | None = Field(default=None, description="First Contentful Paint (ms)")
    lcp_ms: float | None = Field(default=None, description="Largest Contentful Paint (ms)")
    cls: float | None = Field(default=None, description="Cumulative Layout Shift (unitless)")
    ttfb_ms: float | None = Field(default=None, description="Time to First Byte (ms)")
    dcl_ms: float | None = Field(default=None, description="DOMContentLoaded (ms)")
    load_ms: float | None = Field(default=None, description="window.load (ms)")


class PerfBudget(BaseModel):
    """Opt-in performance budget. Every field is optional.

    Fields left as ``None`` are not judged — only set fields can emit
    perf Deltas. CLS is unitless; the remaining metrics are in milliseconds.
    """

    fcp_ms: float | None = None
    lcp_ms: float | None = None
    cls: float | None = None
    ttfb_ms: float | None = None
    dcl_ms: float | None = None
    load_ms: float | None = None


# --- JS payload ------------------------------------------------------------
#
# Executed inside the page (main world) right after ``networkidle``. The
# script keeps observers alive long enough to catch the page's "final" LCP
# + cumulative layout-shift, then resolves with a single JSON payload.
# Timing is bounded by an outer ``setTimeout`` so a misbehaving page can't
# block the collection indefinitely — the Promise resolves with whatever
# data we have by then.

_PERF_JS = r"""
() => {
    return new Promise((resolve) => {
        const result = {
            fcp_ms: null,
            lcp_ms: null,
            cls: null,
            ttfb_ms: null,
            dcl_ms: null,
            load_ms: null,
        };

        // --- Navigation timing (TTFB / DCL / load) ---
        try {
            const navs = performance.getEntriesByType("navigation");
            if (navs && navs.length > 0) {
                const nav = navs[0];
                // TTFB = responseStart relative to navigation start.
                if (typeof nav.responseStart === "number") {
                    result.ttfb_ms = nav.responseStart;
                }
                if (typeof nav.domContentLoadedEventEnd === "number") {
                    result.dcl_ms = nav.domContentLoadedEventEnd;
                }
                if (typeof nav.loadEventEnd === "number" && nav.loadEventEnd > 0) {
                    result.load_ms = nav.loadEventEnd;
                }
            }
        } catch (e) { /* swallow */ }

        // --- FCP (paint timing) ---
        try {
            const paints = performance.getEntriesByType("paint") || [];
            for (const p of paints) {
                if (p.name === "first-contentful-paint") {
                    result.fcp_ms = p.startTime;
                    break;
                }
            }
        } catch (e) { /* swallow */ }

        // --- LCP via PerformanceObserver ---
        let lcpObserver = null;
        try {
            lcpObserver = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                if (entries.length > 0) {
                    // The last entry is the latest "largest" candidate.
                    const last = entries[entries.length - 1];
                    result.lcp_ms = last.renderTime || last.startTime;
                }
            });
            lcpObserver.observe({ type: "largest-contentful-paint", buffered: true });
        } catch (e) { /* observer unsupported */ }

        // --- CLS via PerformanceObserver ---
        let clsObserver = null;
        let clsValue = 0;
        try {
            clsObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (!entry.hadRecentInput) {
                        clsValue += entry.value || 0;
                    }
                }
            });
            clsObserver.observe({ type: "layout-shift", buffered: true });
        } catch (e) { /* observer unsupported */ }

        const finalize = () => {
            try {
                if (lcpObserver) lcpObserver.disconnect();
            } catch (e) { /* swallow */ }
            try {
                if (clsObserver) clsObserver.disconnect();
            } catch (e) { /* swallow */ }
            if (clsObserver) {
                // Only emit a CLS value when the observer attached — otherwise
                // leave it null so the judge skips it.
                result.cls = clsValue;
            }
            resolve(result);
        };

        // requestIdleCallback isn't available everywhere; fall back to a
        // short timeout so the collection still terminates on webkit.
        const idle = window.requestIdleCallback || ((cb) => setTimeout(cb, 200));
        idle(() => finalize());

        // Hard ceiling — never block longer than 2s past the caller's wait.
        setTimeout(finalize, 2000);
    });
}
"""


# --- Public API ------------------------------------------------------------


def collect_perf_metrics(
    route: str,
    viewport: tuple[int, int] = (1280, 720),
    browser: BrowserName = "chromium",
    wait_for: str | None = None,
    timeout_ms: int = 30_000,
) -> PerfMetrics:
    """Drive Playwright against ``route`` and return a :class:`PerfMetrics`.

    Reuses the same launch-engine / wait-for / network-idle pattern as
    :func:`pixel_mcp.render.measure_render`. The collection is best-effort
    per metric — fields the platform doesn't expose come back as ``None``
    rather than raising.

    Raises:
        PlaywrightNotInstalledError: ``playwright`` module not importable.
        ChromiumNotInstalledError: Chromium binary missing.
        BrowserNotInstalledError: firefox/webkit binary missing.
        RouteUnreachableError: navigation to ``route`` failed.
        PerfMetricsError: unexpected runtime failure during collection.
    """
    if browser not in VALID_BROWSERS:
        raise RenderError(
            f"Unsupported browser {browser!r}. Choose one of: {sorted(VALID_BROWSERS)}."
        )
    try:
        from playwright.sync_api import (  # noqa: PLC0415
            Error as PlaywrightError,
        )
        from playwright.sync_api import (
            TimeoutError as PlaywrightTimeoutError,
        )
        from playwright.sync_api import (
            sync_playwright,
        )
    except ImportError as exc:
        raise PlaywrightNotInstalledError(
            "playwright is not installed. Run `uv sync` then `uv run playwright install chromium`."
        ) from exc

    payload: dict[str, Any]
    try:
        with sync_playwright() as p:
            launcher = getattr(p, browser)
            try:
                browser_obj = launcher.launch(headless=True)
            except PlaywrightError as exc:
                msg = str(exc).lower()
                if "executable doesn" in msg or "browsertype.launch" in msg:
                    if browser == "chromium":
                        raise ChromiumNotInstalledError(
                            "Chromium browser binary not found. Run "
                            "`uv run playwright install chromium` (one-time, ~150MB)."
                        ) from exc
                    raise BrowserNotInstalledError(browser, exc) from exc
                raise RouteUnreachableError(f"Failed to launch {browser}: {exc}") from exc

            try:
                context = browser_obj.new_context(
                    viewport={"width": viewport[0], "height": viewport[1]}
                )
                page = context.new_page()

                try:
                    page.goto(route, timeout=timeout_ms)
                except PlaywrightTimeoutError as exc:
                    raise RouteUnreachableError(
                        f"Navigation to {route!r} timed out after {timeout_ms}ms."
                    ) from exc
                except PlaywrightError as exc:
                    raise RouteUnreachableError(f"Failed to navigate to {route!r}: {exc}") from exc

                if wait_for:
                    try:
                        page.wait_for_selector(wait_for, timeout=timeout_ms)
                    except PlaywrightTimeoutError as exc:
                        raise RouteUnreachableError(
                            f"Selector {wait_for!r} did not appear within {timeout_ms}ms."
                        ) from exc

                try:
                    page.wait_for_load_state("networkidle", timeout=timeout_ms)
                except PlaywrightTimeoutError:
                    # Some pages never go idle — we still try to collect what
                    # the platform has gathered up to this point.
                    pass

                payload = page.evaluate(_PERF_JS)
            finally:
                browser_obj.close()
    except RenderError:
        raise
    except Exception as exc:  # unexpected — wrap so callers get a stable type
        raise PerfMetricsError(f"Unexpected perf collection error: {exc}") from exc

    def _as_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return PerfMetrics(
        route=route,
        viewport=viewport,
        browser=browser,
        collected_at=datetime.now(UTC),
        fcp_ms=_as_float(payload.get("fcp_ms")),
        lcp_ms=_as_float(payload.get("lcp_ms")),
        cls=_as_float(payload.get("cls")),
        ttfb_ms=_as_float(payload.get("ttfb_ms")),
        dcl_ms=_as_float(payload.get("dcl_ms")),
        load_ms=_as_float(payload.get("load_ms")),
    )


# --- Budget judging --------------------------------------------------------

# Field name → human-readable Delta property name. The DOM-style suffix
# (``perf_fcp_ms``) keeps Delta.property a stable identifier consumers can
# bucket on, while staying readable in envelopes / hints.
_BUDGETED_FIELDS: tuple[str, ...] = (
    "fcp_ms",
    "lcp_ms",
    "cls",
    "ttfb_ms",
    "dcl_ms",
    "load_ms",
)


def _perf_severity(pct_over: float) -> str | None:
    """Map percent-over-budget to Delta severity (v3-1 gradient).

    Returns ``None`` for any value within 5 % of budget — no Delta emitted.
    """
    if pct_over < 5.0:
        return None
    if pct_over < 20.0:
        return "minor"
    if pct_over <= 50.0:
        return "major"
    return "critical"


def judge_perf_metrics(metrics: PerfMetrics, budget: PerfBudget) -> list[Delta]:
    """Compare ``metrics`` against ``budget`` and emit pseudo-:class:`Delta` per overage.

    Pure function — same inputs always produce the same list (no time
    dependence, deterministic ordering by field name).

    Rules:
    - A metric the platform reported as ``None`` is skipped silently (no
      Delta) — we never punish a browser for not exposing LCP.
    - A budget field left ``None`` is skipped — opt-in per metric.
    - ``observed == budget`` (or within 5 %) emits no Delta.
    - Otherwise the severity gradient documented at module level applies.
    - ``Delta.magnitude`` is the absolute over-budget amount (ms for time
      metrics, raw difference for CLS).
    """
    deltas: list[Delta] = []
    for field in _BUDGETED_FIELDS:
        observed = getattr(metrics, field)
        expected = getattr(budget, field)
        if observed is None or expected is None:
            continue
        try:
            observed_f = float(observed)
            expected_f = float(expected)
        except (TypeError, ValueError):
            continue
        if expected_f <= 0:
            # A zero/negative budget would make the percentage calc explode;
            # treat any positive observed value as critical structural drift.
            if observed_f > 0:
                deltas.append(
                    Delta(
                        selector="<perf>",
                        figma_node_id=None,
                        property=f"perf_{field}",
                        observed=observed_f,
                        expected=expected_f,
                        magnitude=observed_f,
                        severity="critical",
                    )
                )
            continue
        diff = observed_f - expected_f
        if diff <= 0:
            # Within budget — no Delta.
            continue
        pct_over = (diff / expected_f) * 100.0
        severity = _perf_severity(pct_over)
        if severity is None:
            continue
        deltas.append(
            Delta(
                selector="<perf>",
                figma_node_id=None,
                property=f"perf_{field}",
                observed=observed_f,
                expected=expected_f,
                magnitude=diff,
                severity=severity,  # type: ignore[arg-type]
            )
        )
    return deltas
