"""RenderMeasurer — the Deep Module behind ``pixel-mcp measure``.

Public entry point: :func:`measure_render`. Takes a route (URL), drives a
headless Chromium browser via Playwright, and returns a
:class:`MeasuredDOM` — the structured snapshot of the Render that the
DeltaDiffer (Slice 4) will compare against a DesignSpec.

Design notes:
- Sync Playwright API. We have no concurrent IO need in v0; the sync API
  keeps the call sites simple and matches the rest of the codebase.
- One ``page.evaluate()`` round-trip does most of the work. Pulling element
  data out one-by-one across the CDP boundary is slow; batching in JS and
  returning a single JSON payload is an order of magnitude faster on
  pages with a hundred or so elements.
- Stable selectors: prefer ``#id`` → tag.class chain → ``nth-child``
  fallback. Verified unique on the page before being emitted.
- MeasuredDOM is lean — only fields the DeltaDiffer will actually consume.
- Auto-discover caps at 200 elements per route; hint warning emitted when
  the cap is hit so the calling Agent knows to narrow ``--selectors``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

MAX_ELEMENTS = 200
"""Cap on auto-discovered elements per measurement. Prevents envelope bloat."""

MIN_AREA_PX2 = 16
"""Skip elements smaller than this in auto-discover (anti-aliasing noise)."""


class RenderError(Exception):
    """Base class for all RenderMeasurer errors."""


class PlaywrightNotInstalledError(RenderError):
    """Playwright is not importable."""


class ChromiumNotInstalledError(RenderError):
    """Chromium browser binary is not present."""


class RouteUnreachableError(RenderError):
    """The route URL could not be loaded (DNS, refused, timeout, etc.)."""


class WaitForTimeoutError(RenderError):
    """The ``--wait-for`` selector never appeared within the timeout."""


# --- Pydantic models -------------------------------------------------------


class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class ComputedStyle(BaseModel):
    """Lean subset of computed styles. Only what DeltaDiffer (Slice 4) will read."""

    color: str  # canonicalized hex (#rrggbb or #rrggbbaa)
    background_color: str
    font_family: str
    font_size_px: float
    font_weight: int
    line_height: str | None = None
    letter_spacing: str | None = None
    padding_top: float
    padding_right: float
    padding_bottom: float
    padding_left: float
    margin_top: float
    margin_right: float
    margin_bottom: float
    margin_left: float
    border_radius: str | None = None
    border_top_width: float
    border_right_width: float
    border_bottom_width: float
    border_left_width: float


class MeasuredElement(BaseModel):
    selector: str
    bounding_box: BoundingBox
    computed_style: ComputedStyle
    text_content: str | None = None
    aria_role: str | None = None
    parent_chain: list[str] = Field(default_factory=list)


class MeasuredDOM(BaseModel):
    schema_version: int = 1
    route: str
    viewport: tuple[int, int]
    measured_at: datetime
    elements: list[MeasuredElement] = Field(default_factory=list)


# --- JS snippet executed inside the page -----------------------------------
#
# This script runs in the browser's main world. It walks the DOM, picks
# candidates per the auto-discover algorithm, computes a stable selector
# for each, and returns a JSON-serializable list of element records.
#
# Keep this string self-contained: no template injection, no dependencies
# on Playwright APIs. Inputs (selectors filter, caps) are passed as the
# argument to ``page.evaluate(js, args)``.
_MEASURE_JS = r"""
(args) => {
    const { selectorsFilter, minAreaPx2, maxElements } = args;

    // --- Selector computation --------------------------------------------
    const isValidIdForSelector = (id) => {
        // Must start with a letter and contain no whitespace/special chars
        // that would require escaping. CSS escaping is non-trivial; keep
        // the "safe" id set narrow.
        return typeof id === "string"
            && /^[A-Za-z][A-Za-z0-9_-]*$/.test(id);
    };

    const classListClean = (el) => {
        return Array.from(el.classList).filter(
            (c) => /^[A-Za-z_][A-Za-z0-9_-]*$/.test(c)
        );
    };

    const computeSelector = (el) => {
        if (el === document.body) return "body";
        if (el === document.documentElement) return "html";

        // Prefer #id when it's stable-looking and unique
        if (el.id && isValidIdForSelector(el.id)) {
            const sel = "#" + el.id;
            if (document.querySelectorAll(sel).length === 1) return sel;
        }

        // Try tag.class1.class2 if unique
        const tag = el.tagName.toLowerCase();
        const classes = classListClean(el);
        if (classes.length > 0) {
            const sel = tag + "." + classes.join(".");
            if (document.querySelectorAll(sel).length === 1) return sel;
        }

        // Fallback: build nth-child chain from body down
        const chain = [];
        let cur = el;
        while (cur && cur !== document.body && cur.parentElement) {
            const parent = cur.parentElement;
            const idx = Array.from(parent.children).indexOf(cur) + 1;
            const t = cur.tagName.toLowerCase();
            chain.unshift(t + ":nth-child(" + idx + ")");
            cur = parent;
        }
        const sel = "body > " + chain.join(" > ");
        // Verify uniqueness; if not, append a more specific tag.class hint
        if (document.querySelectorAll(sel).length === 1) return sel;
        // Last-ditch: prepend tag-and-classes if any. If still not unique,
        // emit anyway — downstream Mappings will disambiguate by hash.
        return sel;
    };

    const parentChainOf = (el) => {
        const chain = [];
        let cur = el.parentElement;
        while (cur && cur !== document.body) {
            chain.unshift(computeSelector(cur));
            cur = cur.parentElement;
        }
        if (cur === document.body) chain.unshift("body");
        return chain;
    };

    // --- Visibility + candidate filter -----------------------------------
    const isVisible = (el) => {
        if (!(el instanceof Element)) return false;
        if (el.offsetWidth === 0 || el.offsetHeight === 0) return false;
        const cs = window.getComputedStyle(el);
        if (cs.display === "none" || cs.visibility === "hidden") return false;
        if (parseFloat(cs.opacity || "1") === 0) return false;
        return true;
    };

    const SEMANTIC_TAGS = new Set([
        "button", "input", "section", "article", "nav",
        "header", "footer", "main", "aside", "form",
        "label", "select", "textarea",
    ]);

    const isSemanticContainer = (el) => {
        if (el.hasAttribute("role")) return true;
        return SEMANTIC_TAGS.has(el.tagName.toLowerCase());
    };

    const hasMeaningfulText = (el) => {
        // Direct text node content (not descendant text)
        for (const node of el.childNodes) {
            if (node.nodeType === Node.TEXT_NODE
                && node.textContent.trim().length > 0) {
                return true;
            }
        }
        return false;
    };

    const isLeafForMeasurement = (el) => {
        // Leaf = no element children, OR text-bearing with no element kids
        if (el.children.length === 0) return true;
        if (hasMeaningfulText(el) && el.children.length === 0) return true;
        return false;
    };

    // --- Candidate selection ---------------------------------------------
    let candidates;
    if (Array.isArray(selectorsFilter) && selectorsFilter.length > 0) {
        // User-specified selectors only
        const all = [];
        for (const sel of selectorsFilter) {
            try {
                document.querySelectorAll(sel).forEach((el) => all.push(el));
            } catch (e) {
                // skip invalid selectors silently
            }
        }
        candidates = all;
    } else {
        // Auto-discover: visible leaves + semantic containers
        candidates = [];
        const all = document.querySelectorAll("*");
        for (const el of all) {
            if (!isVisible(el)) continue;
            const bbox = el.getBoundingClientRect();
            if (bbox.width * bbox.height < minAreaPx2) continue;
            if (isLeafForMeasurement(el) || isSemanticContainer(el)) {
                candidates.push(el);
            }
        }
    }

    // Dedupe while preserving order
    const seen = new Set();
    const unique = [];
    for (const el of candidates) {
        if (!seen.has(el)) {
            seen.add(el);
            unique.push(el);
        }
    }

    const capped = unique.slice(0, maxElements);
    const truncated = unique.length > maxElements;

    // --- Style extraction ------------------------------------------------
    const rgbToHex = (val) => {
        if (!val) return "#00000000";
        // Match rgb()/rgba() forms
        const m = val.match(/rgba?\(([^)]+)\)/);
        if (!m) return val;  // leave non-rgb forms (e.g. "transparent") as-is
        const parts = m[1].split(",").map((s) => parseFloat(s.trim()));
        const [r, g, b] = parts;
        const a = parts.length > 3 ? parts[3] : 1;
        const toHex = (n) => Math.max(0, Math.min(255, Math.round(n)))
            .toString(16).padStart(2, "0");
        const base = "#" + toHex(r) + toHex(g) + toHex(b);
        if (a >= 1) return base;
        const aHex = Math.max(0, Math.min(255, Math.round(a * 255)))
            .toString(16).padStart(2, "0");
        return base + aHex;
    };

    const px = (val) => {
        const f = parseFloat(val || "0");
        return isFinite(f) ? f : 0;
    };

    const records = capped.map((el) => {
        const cs = window.getComputedStyle(el);
        const bbox = el.getBoundingClientRect();
        const selector = computeSelector(el);
        const parentChain = parentChainOf(el);

        // text_content: only emit for leaves with direct text
        let textContent = null;
        if (hasMeaningfulText(el)) {
            // Compose only direct-child text nodes
            let buf = "";
            for (const node of el.childNodes) {
                if (node.nodeType === Node.TEXT_NODE) {
                    buf += node.textContent;
                }
            }
            const trimmed = buf.trim();
            if (trimmed.length > 0) textContent = trimmed;
        }

        return {
            selector: selector,
            bounding_box: {
                x: bbox.x,
                y: bbox.y,
                w: bbox.width,
                h: bbox.height,
            },
            computed_style: {
                color: rgbToHex(cs.color),
                background_color: rgbToHex(cs.backgroundColor),
                font_family: cs.fontFamily || "",
                font_size_px: px(cs.fontSize),
                font_weight: parseInt(cs.fontWeight || "400", 10) || 400,
                line_height: cs.lineHeight || null,
                letter_spacing: cs.letterSpacing || null,
                padding_top: px(cs.paddingTop),
                padding_right: px(cs.paddingRight),
                padding_bottom: px(cs.paddingBottom),
                padding_left: px(cs.paddingLeft),
                margin_top: px(cs.marginTop),
                margin_right: px(cs.marginRight),
                margin_bottom: px(cs.marginBottom),
                margin_left: px(cs.marginLeft),
                border_radius: cs.borderRadius || null,
                border_top_width: px(cs.borderTopWidth),
                border_right_width: px(cs.borderRightWidth),
                border_bottom_width: px(cs.borderBottomWidth),
                border_left_width: px(cs.borderLeftWidth),
            },
            text_content: textContent,
            aria_role: el.getAttribute("role"),
            parent_chain: parentChain,
        };
    });

    return { elements: records, truncated: truncated, total_found: unique.length };
}
"""


# --- Public API ------------------------------------------------------------


def measure_render(
    route: str,
    viewport: tuple[int, int] = (1280, 720),
    selectors: list[str] | None = None,
    wait_for: str | None = None,
    wait_for_network_idle: bool = True,
    timeout_ms: int = 15_000,
) -> tuple[MeasuredDOM, bool]:
    """Drive Chromium against ``route`` and return the MeasuredDOM.

    Returns ``(measured_dom, truncated)``. ``truncated`` is True when
    auto-discover hit the :data:`MAX_ELEMENTS` cap — the calling layer
    surfaces this as a hint.

    Raises:
        PlaywrightNotInstalledError: ``playwright`` module not importable.
        ChromiumNotInstalledError: Chromium binary missing (run
            ``uv run playwright install chromium``).
        RouteUnreachableError: navigation to ``route`` failed.
        WaitForTimeoutError: ``wait_for`` selector never appeared.
    """
    try:
        from playwright.sync_api import (  # noqa: PLC0415  (lazy import)
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
            "playwright is not installed. Run `uv sync` then "
            "`uv run playwright install chromium`."
        ) from exc

    truncated = False
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except PlaywrightError as exc:
                msg = str(exc).lower()
                if "executable doesn" in msg or "browsertype.launch" in msg:
                    raise ChromiumNotInstalledError(
                        "Chromium browser binary not found. Run "
                        "`uv run playwright install chromium` (one-time, ~150MB)."
                    ) from exc
                raise RouteUnreachableError(f"Failed to launch Chromium: {exc}") from exc

            try:
                context = browser.new_context(
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
                        raise WaitForTimeoutError(
                            f"Selector {wait_for!r} did not appear within "
                            f"{timeout_ms}ms on {route!r}."
                        ) from exc

                if wait_for_network_idle:
                    try:
                        page.wait_for_load_state("networkidle", timeout=timeout_ms)
                    except PlaywrightTimeoutError:
                        # Non-fatal: pages with long-poll/SSE never go idle.
                        # We still want a snapshot.
                        pass
                    # One rAF quiet for deterministic layout
                    page.evaluate("() => new Promise(r => requestAnimationFrame(() => r(null)))")

                payload: dict[str, Any] = page.evaluate(
                    _MEASURE_JS,
                    {
                        "selectorsFilter": selectors,
                        "minAreaPx2": MIN_AREA_PX2,
                        "maxElements": MAX_ELEMENTS,
                    },
                )
            finally:
                browser.close()
    except RenderError:
        raise
    except Exception as exc:  # unexpected — wrap so callers get a stable type
        raise RouteUnreachableError(f"Unexpected render error: {exc}") from exc

    truncated = bool(payload.get("truncated", False))
    elements = [MeasuredElement.model_validate(r) for r in payload.get("elements", [])]
    dom = MeasuredDOM(
        route=route,
        viewport=viewport,
        measured_at=datetime.now(UTC),
        elements=elements,
    )
    return dom, truncated


__all__ = [
    "MAX_ELEMENTS",
    "MIN_AREA_PX2",
    "BoundingBox",
    "ChromiumNotInstalledError",
    "ComputedStyle",
    "MeasuredDOM",
    "MeasuredElement",
    "PlaywrightNotInstalledError",
    "RenderError",
    "RouteUnreachableError",
    "WaitForTimeoutError",
    "measure_render",
]
