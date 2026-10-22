"""CLI failure-path tests for the Phase 8 ``explain`` command."""

from __future__ import annotations

from churnwise.cli import main


def test_explain_invalid_sampling_config_is_structured(settings, capsys) -> None:
    code = main(["explain", "--sample-size", "0"], settings=settings)
    captured = capsys.readouterr()

    assert code == 3
    assert "CW-XAI-001" in captured.err
    assert "Traceback" not in captured.err


def test_explain_without_artifacts_reports_missing_model(settings, capsys) -> None:
    code = main(["explain"], settings=settings)
    captured = capsys.readouterr()

    assert code == 2
    assert "CW-XAI-006" in captured.err
    assert "Traceback" not in captured.err
