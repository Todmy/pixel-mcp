"""Unit tests for loop_state — iteration counter, stuck detection, history."""

from __future__ import annotations

import json
from pathlib import Path

from pixel_mcp.delta import Delta
from pixel_mcp.loop_state import (
    DEFAULT_STUCK_THRESHOLD,
    IterationState,
    append_history,
    bucket_for_magnitude,
    compute_file_hashes,
    detect_regression,
    detect_stuck,
    hash_deltas_bucketed,
    read_state,
    reset_state,
    write_state,
)

# --- Magnitude bucketing -------------------------------------------------


def test_bucket_handles_none() -> None:
    assert bucket_for_magnitude(None) == "none"


def test_bucket_thresholds() -> None:
    assert bucket_for_magnitude(0.2) == "<0.5"
    assert bucket_for_magnitude(3.0) == "<5"
    assert bucket_for_magnitude(8.0) == "<10"
    assert bucket_for_magnitude(40.0) == "<50"
    assert bucket_for_magnitude(150.0) == "<200"
    assert bucket_for_magnitude(500.0) == ">=200"


# --- Hash + stuck --------------------------------------------------------


def _d(severity: str = "critical", prop: str = "color", mag: float | None = None) -> Delta:
    return Delta(
        selector="#x",
        property=prop,
        observed="x",
        expected="y",
        magnitude=mag,
        severity=severity,  # type: ignore[arg-type]
    )


def test_hash_deterministic_for_same_deltas() -> None:
    deltas = [_d(), _d(prop="font_size", mag=2.0)]
    assert hash_deltas_bucketed(deltas) == hash_deltas_bucketed(deltas)


def test_hash_invariant_under_order() -> None:
    a = [_d(prop="color"), _d(prop="width", mag=3.0)]
    b = [_d(prop="width", mag=3.0), _d(prop="color")]
    assert hash_deltas_bucketed(a) == hash_deltas_bucketed(b)


def test_hash_changes_when_severity_changes() -> None:
    a = [_d(severity="critical")]
    b = [_d(severity="major")]
    assert hash_deltas_bucketed(a) != hash_deltas_bucketed(b)


def test_hash_stable_under_magnitude_jitter_in_same_bucket() -> None:
    """0.2 and 0.4 both bucket as <0.5 → same hash."""
    a = [_d(mag=0.2)]
    b = [_d(mag=0.4)]
    assert hash_deltas_bucketed(a) == hash_deltas_bucketed(b)


def test_hash_differs_across_buckets() -> None:
    a = [_d(mag=0.4)]
    b = [_d(mag=3.0)]
    assert hash_deltas_bucketed(a) != hash_deltas_bucketed(b)


def test_detect_stuck_fires_after_threshold_identical_hashes() -> None:
    history = ["abc", "abc"]
    assert detect_stuck(history, "abc", threshold=3) is True


def test_detect_stuck_not_yet() -> None:
    history = ["abc"]
    assert detect_stuck(history, "abc", threshold=3) is False


def test_detect_stuck_resets_on_different_hash() -> None:
    history = ["abc", "abc"]
    assert detect_stuck(history, "def", threshold=3) is False


# --- Regression detection ------------------------------------------------


def test_no_regression_when_at_highest_level() -> None:
    state = IterationState(highest_level_reached=1)
    assert detect_regression(state, current_level_passed=1) is False


def test_regression_when_falling_back() -> None:
    state = IterationState(highest_level_reached=2)
    assert detect_regression(state, current_level_passed=0) is True


# --- State IO ------------------------------------------------------------


def test_read_state_returns_fresh_when_absent(tmp_path: Path) -> None:
    state = read_state(project_root=tmp_path)
    assert state.iteration == 0
    assert state.session_id  # uuid generated


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    state = IterationState(iteration=7, last_delta_hash="abc", highest_level_reached=1)
    write_state(state, project_root=tmp_path)
    loaded = read_state(project_root=tmp_path)
    assert loaded.iteration == 7
    assert loaded.last_delta_hash == "abc"
    assert loaded.highest_level_reached == 1


def test_reset_clears_state_files(tmp_path: Path) -> None:
    state = IterationState(iteration=4)
    write_state(state, project_root=tmp_path)
    append_history({"x": 1}, project_root=tmp_path)
    reset_state(project_root=tmp_path)
    new = read_state(project_root=tmp_path)
    assert new.iteration == 0


# --- History trace -------------------------------------------------------


def test_history_appends(tmp_path: Path) -> None:
    append_history({"iteration": 1}, project_root=tmp_path)
    append_history({"iteration": 2}, project_root=tmp_path)
    log = (tmp_path / ".pixel-mcp" / "history.jsonl").read_text()
    lines = [line for line in log.splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["iteration"] == 1


# --- File hashes ---------------------------------------------------------


def test_file_hashes_for_existing_files(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("print('hello')")
    hashes = compute_file_hashes([f])
    assert hashes[str(f)] != "<missing>"


def test_file_hashes_missing_path(tmp_path: Path) -> None:
    f = tmp_path / "doesnt-exist.py"
    hashes = compute_file_hashes([f])
    assert hashes[str(f)] == "<missing>"


# --- Default threshold constant ------------------------------------------


def test_default_stuck_threshold_is_three() -> None:
    assert DEFAULT_STUCK_THRESHOLD == 3
