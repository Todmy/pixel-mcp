"""Integration tests for :func:`pixel_mcp.render.measure_render`.

Spins up a stdlib ``http.server`` against the static fixtures under
``tests/fixtures/render/`` and drives a real Chromium browser at them via
Playwright. Slow-ish (a couple seconds per test) but exercises the real
end-to-end path the CLI/MCP tool will run.

If chromium is not installed (CI without ``playwright install chromium``),
the fixture skips the whole module.
"""

from __future__ import annotations

import socket
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest
from pixel_mcp.render import (
    MAX_ELEMENTS,
    MeasuredDOM,
    RouteUnreachableError,
    measure_render,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "render"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _QuietHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that doesn't spam stderr during tests."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture(scope="module")
def fixture_server() -> tuple[str, HTTPServer]:  # type: ignore[type-arg]
    """Serve ``tests/fixtures/render/`` over loopback for the test module."""
    port = _pick_free_port()

    def _make(*args: object, **kwargs: object) -> _QuietHandler:
        return _QuietHandler(*args, directory=str(FIXTURE_DIR), **kwargs)  # type: ignore[arg-type]

    httpd = HTTPServer(("127.0.0.1", port), _make)  # type: ignore[arg-type]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    # Tiny wait so the bind settles. 50ms is enough on localhost.
    time.sleep(0.05)
    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url, httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_chromium() -> None:
    """Skip the whole module if Chromium isn't installed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed")
        return
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:
        pytest.skip(f"chromium not available: {exc}")


def test_measure_auto_discover(fixture_server: tuple[str, HTTPServer]) -> None:
    base_url, _ = fixture_server
    dom, truncated = measure_render(f"{base_url}/simple.html")
    assert isinstance(dom, MeasuredDOM)
    assert dom.route.endswith("/simple.html")
    assert dom.viewport == (1280, 720)
    assert not truncated
    selectors = {el.selector for el in dom.elements}
    # The button, nav, headline, lede paragraph should all appear.
    assert any(s.startswith("#primary-nav") or "top-nav" in s for s in selectors)
    assert any(s.startswith("#headline") or "h1" in s for s in selectors)
    assert any("cta" in s for s in selectors)


def test_measure_selectors_filter(fixture_server: tuple[str, HTTPServer]) -> None:
    base_url, _ = fixture_server
    dom, _ = measure_render(
        f"{base_url}/simple.html",
        selectors=["button.cta"],
    )
    assert len(dom.elements) == 1
    assert "cta" in dom.elements[0].selector
    assert dom.elements[0].text_content == "Get started"


def test_measure_unreachable_route_raises() -> None:
    # Bind to an unused port; nothing listening there.
    port = _pick_free_port()
    with pytest.raises(RouteUnreachableError):
        measure_render(
            f"http://127.0.0.1:{port}/nowhere.html",
            wait_for_network_idle=False,
            timeout_ms=3000,
        )


def test_measure_viewport_applied(fixture_server: tuple[str, HTTPServer]) -> None:
    base_url, _ = fixture_server
    dom, _ = measure_render(
        f"{base_url}/simple.html",
        viewport=(800, 600),
        selectors=["nav.top-nav"],
    )
    assert dom.viewport == (800, 600)
    assert len(dom.elements) == 1
    # The nav stretches to viewport width minus default body margins (0 here).
    box = dom.elements[0].bounding_box
    assert box.w <= 800.0
    # Sanity: width should be substantial, not 0
    assert box.w > 100


def test_measure_tiny_elements_filtered(fixture_server: tuple[str, HTTPServer]) -> None:
    base_url, _ = fixture_server
    dom, _ = measure_render(f"{base_url}/simple.html")
    # ``span.tiny-dot`` is 2x2 (4 px², below MIN_AREA_PX2). Auto-discover
    # should skip it.
    selectors = {el.selector for el in dom.elements}
    assert not any("tiny-dot" in s for s in selectors)


def test_measure_hidden_elements_filtered(fixture_server: tuple[str, HTTPServer]) -> None:
    base_url, _ = fixture_server
    dom, _ = measure_render(f"{base_url}/simple.html")
    selectors = {el.selector for el in dom.elements}
    assert not any("hidden-block" in s for s in selectors)


def test_measure_caps_at_max_elements() -> None:
    """Sanity: MAX_ELEMENTS is the published cap, used by the cmd layer."""
    assert MAX_ELEMENTS == 200


# ---------------------------------------------------------------------------
# v2-2 — cross-browser plumbing (mocked Playwright; no real engines launched)
# ---------------------------------------------------------------------------


class _FakeBrowser:
    """Stand-in for the Playwright Browser object used in measure_render."""

    def __init__(self) -> None:
        self.closed = False

    def new_context(self, **_kwargs: object) -> _FakeContext:
        return _FakeContext()

    def close(self) -> None:
        self.closed = True


class _FakeContext:
    def new_page(self) -> _FakePage:
        return _FakePage()


class _FakePage:
    def goto(self, *_args: object, **_kwargs: object) -> None:
        return None

    def wait_for_selector(self, *_args: object, **_kwargs: object) -> None:
        return None

    def wait_for_load_state(self, *_args: object, **_kwargs: object) -> None:
        return None

    def evaluate(self, _js: object, *_args: object, **_kwargs: object) -> dict:
        return {"elements": [], "truncated": False, "total_found": 0}

    def screenshot(self, **_kwargs: object) -> bytes:
        return b"fake-png"


class _FakeLauncher:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def launch(self, **_kwargs: object) -> _FakeBrowser:
        self.calls.append(self.name)
        return _FakeBrowser()


class _FakePlaywright:
    def __init__(self, calls: list[str]) -> None:
        self.chromium = _FakeLauncher("chromium", calls)
        self.firefox = _FakeLauncher("firefox", calls)
        self.webkit = _FakeLauncher("webkit", calls)


class _FakePlaywrightContextManager:
    def __init__(self, pw: _FakePlaywright) -> None:
        self._pw = pw

    def __enter__(self) -> _FakePlaywright:
        return self._pw

    def __exit__(self, *_a: object) -> None:
        return None


def test_measure_browser_param_routes_to_firefox(monkeypatch: pytest.MonkeyPatch) -> None:
    """``browser='firefox'`` reaches ``p.firefox.launch(...)``."""
    import playwright.sync_api  # noqa: PLC0415

    calls: list[str] = []
    fake = _FakePlaywright(calls)

    def _fake_sync_playwright() -> _FakePlaywrightContextManager:
        return _FakePlaywrightContextManager(fake)

    monkeypatch.setattr(playwright.sync_api, "sync_playwright", _fake_sync_playwright)
    dom, _trunc = measure_render(
        "http://example.invalid/",
        viewport=(800, 600),
        wait_for_network_idle=False,
        browser="firefox",
    )
    assert calls == ["firefox"]
    assert dom.viewport == (800, 600)


def test_measure_browser_default_is_chromium(monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting ``browser`` still launches chromium (backward compat)."""
    import playwright.sync_api  # noqa: PLC0415

    calls: list[str] = []
    fake = _FakePlaywright(calls)

    def _fake_sync_playwright() -> _FakePlaywrightContextManager:
        return _FakePlaywrightContextManager(fake)

    monkeypatch.setattr(playwright.sync_api, "sync_playwright", _fake_sync_playwright)
    measure_render(
        "http://example.invalid/",
        wait_for_network_idle=False,
    )
    assert calls == ["chromium"]


def test_measure_browser_invalid_raises() -> None:
    """An unsupported browser name fails loud before touching Playwright."""
    from pixel_mcp.render import RenderError

    with pytest.raises(RenderError):
        measure_render(
            "http://example.invalid/",
            wait_for_network_idle=False,
            browser="opera",  # type: ignore[arg-type]
        )
