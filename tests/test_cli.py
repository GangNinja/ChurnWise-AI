from __future__ import annotations

import json

from churnwise.cli import main
from churnwise.contracts import SourceStatus
from tests.helpers import FakeAdapter, make_candidate, make_result


def test_successful_discovery_writes_reports_and_returns_zero(settings, tmp_path, capsys):
    adapter = FakeAdapter(make_result("openml", [make_candidate()]))
    code = main(
        ["discover", "--json", "--report-dir", str(tmp_path / "reports")],
        settings=settings,
        adapters=[adapter],
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert payload["selected"]["candidate_id"] == "openml:46915"
    assert (tmp_path / "reports" / "acquisition" / "candidates.json").is_file()
    assert (tmp_path / "reports" / "acquisition" / "rejections.json").is_file()


def test_human_summary_names_the_selection(settings, capsys):
    adapter = FakeAdapter(make_result("openml", [make_candidate()]))
    code = main(["discover"], settings=settings, adapters=[adapter])
    out = capsys.readouterr().out
    assert code == 0
    assert "Selected: openml:46915" in out
    assert "Report:" in out


def test_discovery_failure_returns_two(settings, capsys):
    adapter = FakeAdapter(make_result("openml", status=SourceStatus.EMPTY))
    code = main(["discover"], settings=settings, adapters=[adapter])
    out = capsys.readouterr().out
    assert code == 2
    assert "Discovery failed" in out
    assert "CW-DISC-002" in out


def test_invalid_configuration_returns_three(settings):
    adapter = FakeAdapter(make_result("openml", [make_candidate()]))
    code = main(["discover", "--min-rows", "1"], settings=settings, adapters=[adapter])
    assert code == 3


def test_run_requires_an_explicit_run_id_to_resume(settings, capsys):
    code = main(["run", "--resume"], settings=settings)
    assert code == 3
    assert "--resume requires --run-id" in capsys.readouterr().err


def test_run_missing_dataset_is_a_configuration_error(settings, tmp_path, capsys):
    code = main(
        [
            "run",
            "--dataset",
            str(tmp_path / "missing.csv"),
            "--run-id",
            "phase12-missing-dataset",
        ],
        settings=settings,
    )
    assert code == 3
    assert "CW-RUN-001" in capsys.readouterr().err


def test_run_offline_network_mode_is_rejected(settings, capsys):
    code = main(["run", "--offline", "--run-id", "phase12-offline"], settings=settings)
    assert code == 3
    assert "CW-RUN-001" in capsys.readouterr().err


def test_offline_mode_returns_two_with_structured_reason(settings, tmp_path, capsys):
    code = main(
        ["discover", "--offline", "--json", "--report-dir", str(tmp_path / "reports")],
        settings=settings,
        adapters=[],
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["failure"]["code"] == "CW-DISC-005"
