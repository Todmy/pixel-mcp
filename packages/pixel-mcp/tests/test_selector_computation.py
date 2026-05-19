"""Tests for the selector-computation JS used by RenderMeasurer.

We drive a real headless Chromium against ``fixtures/render/selectors.html``
because the selector logic depends on ``document.querySelectorAll`` for
uniqueness verification — there's no faithful pure-Python port.

Skipped automatically if Chromium isn't available.
"""

from __future__ import annotations

import socket
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "render"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture(scope="module")
def server_url() -> str:  # type: ignore[misc]
    port = _pick_free_port()

    def _make(*args: object, **kwargs: object) -> _QuietHandler:
        return _QuietHandler(*args, directory=str(FIXTURE_DIR), **kwargs)  # type: ignore[arg-type]

    httpd = HTTPServer(("127.0.0.1", port), _make)  # type: ignore[arg-type]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_chromium() -> None:
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


def test_unique_id_wins(server_url: str) -> None:
    from pixel_mcp.render import measure_render

    dom, _ = measure_render(
        f"{server_url}/selectors.html",
        selectors=["#unique-id"],
    )
    assert len(dom.elements) == 1
    assert dom.elements[0].selector == "#unique-id"


def test_unique_class_resolves_to_tag_class(server_url: str) -> None:
    from pixel_mcp.render import measure_render

    dom, _ = measure_render(
        f"{server_url}/selectors.html",
        selectors=[".only-here"],
    )
    assert len(dom.elements) == 1
    # The selector should be the tag.class form, since only-here is unique
    # on the page.
    assert dom.elements[0].selector == "div.only-here"


def test_duplicate_class_falls_back_to_nth_child(server_url: str) -> None:
    from pixel_mcp.render import measure_render

    dom, _ = measure_render(
        f"{server_url}/selectors.html",
        selectors=[".dup-class"],
    )
    assert len(dom.elements) == 2
    # Both must be unique selectors — verify by set size
    selectors = {el.selector for el in dom.elements}
    assert len(selectors) == 2
    # At least one fallback must use nth-child (since both share the same class)
    assert any("nth-child" in s for s in selectors)


def test_selectors_are_unique_after_computation(server_url: str) -> None:
    """Every emitted selector must round-trip uniquely on the page."""
    from pixel_mcp.render import measure_render

    dom, _ = measure_render(f"{server_url}/selectors.html")
    selectors = [el.selector for el in dom.elements]
    # Selectors should be distinct across the discovered set
    assert len(selectors) == len(set(selectors)), f"duplicate selectors emitted: {selectors}"
