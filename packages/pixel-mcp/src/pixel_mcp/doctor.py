"""`pixel-mcp doctor` — environment Check.

Returns an AXI envelope describing the runtime environment: Python version,
optional dependencies (Playwright), Figma API token, uv binary presence.

Status convention:
- "green"  — the Check passed. No action needed.
- "amber"  — the Check is non-fatal. A later slice will need it. Hint emitted.
- "red"    — the Check is fatal. Doctor exits non-zero.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import sys
from typing import Literal, TypedDict

from pixel_tools_shared import Envelope, make_envelope

CheckStatus = Literal["green", "amber", "red"]

MIN_PYTHON = (3, 11)


class CheckResult(TypedDict):
    name: str
    status: CheckStatus
    detail: str


def _check_python_version() -> CheckResult:
    current = sys.version_info[:2]
    if current >= MIN_PYTHON:
        return {
            "name": "python_version",
            "status": "green",
            "detail": f"Python {current[0]}.{current[1]} >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
        }
    return {
        "name": "python_version",
        "status": "red",
        "detail": (
            f"Python {current[0]}.{current[1]} is below required "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
        ),
    }


def _check_playwright() -> CheckResult:
    if importlib.util.find_spec("playwright") is not None:
        return {
            "name": "playwright",
            "status": "green",
            "detail": "playwright importable",
        }
    return {
        "name": "playwright",
        "status": "amber",
        "detail": "playwright not installed (run `uv sync` — required by `pixel-mcp measure`)",
    }


def _check_chromium() -> CheckResult:
    """Chromium browser binary readiness.

    We avoid launching a browser here — it's slow and racy. Instead we
    probe the Playwright cache directory for any ``chromium-*`` entry.
    Mac, Linux, and Windows all use the same ``ms-playwright`` cache
    layout under platform-specific roots.
    """
    if importlib.util.find_spec("playwright") is None:
        return {
            "name": "chromium",
            "status": "amber",
            "detail": "Skipped — playwright not installed yet",
        }

    home = pathlib.Path.home()
    candidate_roots = [
        home / "Library" / "Caches" / "ms-playwright",  # macOS
        home / ".cache" / "ms-playwright",  # Linux
        home / "AppData" / "Local" / "ms-playwright",  # Windows
    ]
    env_override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if env_override:
        candidate_roots.insert(0, pathlib.Path(env_override))

    for root in candidate_roots:
        if not root.exists():
            continue
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            name = entry.name
            if name.startswith("chromium-") or name.startswith("chromium_headless_shell-"):
                return {
                    "name": "chromium",
                    "status": "green",
                    "detail": f"chromium binary present at {entry}",
                }

    return {
        "name": "chromium",
        "status": "amber",
        "detail": "Chromium binary missing (run `uv run playwright install chromium`)",
    }


def _check_figma_token() -> CheckResult:
    token = os.environ.get("FIGMA_TOKEN")
    if token:
        return {
            "name": "figma_token",
            "status": "green",
            "detail": "FIGMA_TOKEN present in environment",
        }
    return {
        "name": "figma_token",
        "status": "amber",
        "detail": "FIGMA_TOKEN missing (needed by Slice 2 — Figma extractor)",
    }


def _check_httpx() -> CheckResult:
    if importlib.util.find_spec("httpx") is not None:
        return {
            "name": "httpx",
            "status": "green",
            "detail": "httpx importable (Figma REST client)",
        }
    return {
        "name": "httpx",
        "status": "red",
        "detail": "httpx not installed — required by `pixel-mcp spec`",
    }


def _check_figma_api_reachable() -> CheckResult:
    """HEAD https://api.figma.com — confirms DNS + TCP + TLS reach.

    Non-fatal: a transient network blip shouldn't block the rest of doctor.
    The check uses a short timeout so it never hangs the CLI.
    """
    try:
        import httpx  # local import — doctor must still work without it
    except ImportError:
        return {
            "name": "figma_api_reachable",
            "status": "amber",
            "detail": "Skipped — httpx not importable",
        }
    try:
        response = httpx.head("https://api.figma.com", timeout=5.0)
        # Any HTTP response (even 404 or 401) means the host is reachable.
        return {
            "name": "figma_api_reachable",
            "status": "green",
            "detail": f"api.figma.com reachable (HTTP {response.status_code})",
        }
    except Exception as exc:  # network unreachable, DNS fail, TLS fail, etc.
        return {
            "name": "figma_api_reachable",
            "status": "amber",
            "detail": f"api.figma.com not reachable: {exc.__class__.__name__}",
        }


def _check_uv() -> CheckResult:
    path = shutil.which("uv")
    if path:
        return {
            "name": "uv",
            "status": "green",
            "detail": f"uv binary at {path}",
        }
    return {
        "name": "uv",
        "status": "amber",
        "detail": "uv binary not on PATH (informational — used for install/distribution)",
    }


def run_checks() -> list[CheckResult]:
    return [
        _check_python_version(),
        _check_playwright(),
        _check_chromium(),
        _check_figma_token(),
        _check_httpx(),
        _check_figma_api_reachable(),
        _check_uv(),
    ]


def _summary(checks: list[CheckResult]) -> str:
    greens = sum(1 for c in checks if c["status"] == "green")
    return f"{greens}/{len(checks)} green"


def _hints_for(checks: list[CheckResult]) -> list[str]:
    hints: list[str] = []
    for c in checks:
        if c["status"] != "amber":
            continue
        if c["name"] == "playwright":
            hints.append("Install Playwright: `uv sync` (the dependency is in pyproject.toml).")
        elif c["name"] == "chromium":
            hints.append(
                "Install the Chromium browser binary: `uv run playwright install chromium` "
                "(one-time, ~150MB)."
            )
        elif c["name"] == "figma_token":
            hints.append(
                "Set FIGMA_TOKEN before Slice 2: export FIGMA_TOKEN=<your-personal-access-token>"
            )
        elif c["name"] == "uv":
            hints.append(
                "Install uv from https://docs.astral.sh/uv/ to use the documented install path"
            )
        elif c["name"] == "figma_api_reachable":
            hints.append(
                "Check network — api.figma.com unreachable. `pixel-mcp spec` will fail until this clears."
            )
    return hints


def _next_action(checks: list[CheckResult]) -> str:
    if any(c["status"] == "red" for c in checks):
        red = [c["name"] for c in checks if c["status"] == "red"]
        return f"Resolve red Checks before proceeding: {', '.join(red)}"
    if any(c["status"] == "amber" for c in checks):
        return (
            "Amber Checks are non-fatal. `pixel-mcp spec` needs FIGMA_TOKEN. "
            "`pixel-mcp measure` needs Playwright + Chromium."
        )
    return "All green — `pixel-mcp spec` and `pixel-mcp measure` are ready."


def build_envelope() -> Envelope:
    checks = run_checks()
    affordances: list[dict[str, str]] = []
    if any(c["name"] == "figma_token" and c["status"] == "green" for c in checks):
        affordances.append(
            {
                "tool": "mcp__pixel_mcp__spec",
                "when": "FIGMA_TOKEN configured — ready to extract a DesignSpec from a Figma Source",
            }
        )
    if all(
        any(c["name"] == n and c["status"] == "green" for c in checks)
        for n in ("playwright", "chromium")
    ):
        affordances.append(
            {
                "tool": "mcp__pixel_mcp__measure",
                "when": "Playwright + Chromium ready — capture a MeasuredDOM from a Render URL",
            }
        )
    return make_envelope(
        data={
            "checks": checks,
            "summary": _summary(checks),
        },
        hints=_hints_for(checks),
        diagnostics={
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
        },
        next_suggested_action=_next_action(checks),
        affordances=affordances,
    )


def exit_code_for(envelope: Envelope) -> int:
    """Exit 0 unless any Check is red."""
    checks: list[CheckResult] = envelope["data"]["checks"]
    return 1 if any(c["status"] == "red" for c in checks) else 0
