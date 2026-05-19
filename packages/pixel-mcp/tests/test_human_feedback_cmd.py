"""Level 3 human-feedback capture — CLI + write semantics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from pixel_mcp.cli import app
from pixel_mcp.human_feedback_cmd import (
    EXIT_FATAL,
    EXIT_OK,
    feedback_path,
    mark_consumed,
    read_feedback,
    run,
)
from typer.testing import CliRunner


def test_approve_writes_file_with_verdict_approved(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        envelope, exit_code = run(approve=True)
    assert exit_code == EXIT_OK
    raw = json.loads((sd / "human-feedback.json").read_text())
    assert raw["verdict"] == "approved"
    assert raw["notes"] is None
    assert raw["consumed"] is False
    assert envelope["data"]["verdict"] == "approved"


def test_rejection_writes_file_with_notes(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    notes = "Button colour is off; missing border-radius on the card."
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        envelope, exit_code = run(rejection_notes=notes)
    assert exit_code == EXIT_OK
    raw = json.loads((sd / "human-feedback.json").read_text())
    assert raw["verdict"] == "rejected"
    assert raw["notes"] == notes
    assert raw["consumed"] is False
    assert envelope["data"]["notes"] == notes


def test_mutually_exclusive_flags_error(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        envelope, exit_code = run(approve=True, rejection_notes="something")
    assert exit_code == EXIT_FATAL
    assert envelope["diagnostics"]["error_type"] == "feedback_args_conflict"
    assert not (sd / "human-feedback.json").exists()


def test_neither_flag_error(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        envelope, exit_code = run(approve=False, rejection_notes=None)
    assert exit_code == EXIT_FATAL
    assert envelope["diagnostics"]["error_type"] == "feedback_args_missing"
    assert not (sd / "human-feedback.json").exists()


def test_empty_rejection_notes_treated_as_missing(tmp_path: Path) -> None:
    """Whitespace-only --rejection-notes must be rejected as missing input."""
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        envelope, exit_code = run(rejection_notes="   ")
    assert exit_code == EXIT_FATAL
    assert envelope["diagnostics"]["error_type"] == "feedback_args_missing"


def test_overwrites_existing_unconsumed_feedback(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        # First write — rejection.
        run(rejection_notes="first take")
        first = json.loads((sd / "human-feedback.json").read_text())
        assert first["verdict"] == "rejected"
        assert first["consumed"] is False

        # Second write — approval. Must overwrite, not append.
        run(approve=True)
        second = json.loads((sd / "human-feedback.json").read_text())
        assert second["verdict"] == "approved"
        assert second["notes"] is None
        assert second["consumed"] is False


def test_mark_consumed_flips_flag(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        run(approve=True)
        before = json.loads((sd / "human-feedback.json").read_text())
        assert before["consumed"] is False

        mark_consumed()
        after = json.loads((sd / "human-feedback.json").read_text())
        assert after["consumed"] is True


def test_read_feedback_returns_none_when_absent(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        assert read_feedback() is None


def test_feedback_path_under_state_dir(tmp_path: Path) -> None:
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        assert feedback_path() == sd / "human-feedback.json"


def test_cli_approve_invocation(tmp_path: Path) -> None:
    runner = CliRunner()
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        result = runner.invoke(app, ["human-feedback", "--approve"])
    assert result.exit_code == 0, result.output
    assert (sd / "human-feedback.json").exists()


def test_cli_rejection_invocation(tmp_path: Path) -> None:
    runner = CliRunner()
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        result = runner.invoke(app, ["human-feedback", "--rejection-notes", "Border radius wrong"])
    assert result.exit_code == 0, result.output
    raw = json.loads((sd / "human-feedback.json").read_text())
    assert raw["verdict"] == "rejected"
    assert raw["notes"] == "Border radius wrong"


def test_cli_neither_flag_exits_nonzero(tmp_path: Path) -> None:
    runner = CliRunner()
    sd = tmp_path / ".pixel-mcp"
    with patch("pixel_mcp.human_feedback_cmd.state_dir", return_value=sd):
        result = runner.invoke(app, ["human-feedback"])
    assert result.exit_code != 0
