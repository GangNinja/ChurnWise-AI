"""CLI wiring tests for ``churnwise validate`` (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path

from churnwise.acquisition.coordinator import AcquisitionCoordinator
from churnwise.cli import main
from churnwise.config import Settings
from tests.acquisition_helpers import (
    FakeDownloadTransport,
    FakeStream,
    make_acquisition_candidate,
    make_acquisition_settings,
    make_downloader,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_churn.csv"
URL = "https://www.openml.org/data/v1/download/900002"


def write_report(report_dir: Path, candidate) -> None:
    directory = report_dir / "acquisition"
    directory.mkdir(parents=True, exist_ok=True)
    entry = candidate.to_dict()
    entry.update({"eligible": True, "duplicate_of": None, "rank": 1, "score": 0.9})
    (directory / "candidates.json").write_text(
        json.dumps({"ranking": [entry], "selected": None}), encoding="utf-8"
    )


def make_candidate():
    return make_acquisition_candidate(
        candidate_id="openml:fixture",
        source_dataset_id="900002",
        title="mini customer churn",
        resource_url=URL,
        target_column="churn",
    )


def populate_cache(settings, candidate) -> None:
    coordinator = AcquisitionCoordinator(
        settings,
        downloader_factory=lambda hosts: make_downloader(
            FakeDownloadTransport({URL: FakeStream(body=FIXTURE.read_bytes())}),
            hosts=tuple(hosts),
        ),
    )
    result = coordinator.acquire([candidate])
    assert result.succeeded, result.failure


def test_validate_cli_reports_a_warm_cached_dataset(tmp_path, capsys):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_candidate()
    populate_cache(settings, candidate)
    write_report(settings.report_dir, candidate)

    code = main(
        [
            "validate",
            "--offline",
            "--json",
            "--data-dir",
            str(settings.data_dir),
            "--report-dir",
            str(settings.report_dir),
        ],
        settings=Settings(project_root=tmp_path),
    )

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert code == 0
    assert out.isascii(), "JSON stdout must survive any console codec"
    assert payload["status"] == "pass_with_warnings"
    assert payload["can_preprocess"] is True
    assert payload["report"]["target"]["column"] == "churn"
    assert len(payload["report"]["checks"]) == 12
    assert (settings.report_dir / "validation" / "openml_fixture.json").is_file()
    assert (settings.report_dir / "validation" / "openml_fixture.md").is_file()
    assert (settings.report_dir / "validation" / "attempts.json").is_file()


def test_validate_cli_summary_is_human_readable(tmp_path, capsys):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_candidate()
    populate_cache(settings, candidate)
    write_report(settings.report_dir, candidate)

    code = main(
        [
            "validate",
            "--offline",
            "--data-dir",
            str(settings.data_dir),
            "--report-dir",
            str(settings.report_dir),
        ],
        settings=Settings(project_root=tmp_path),
    )

    out = capsys.readouterr().out
    assert code == 0
    assert out.isascii(), "human summary must survive any console codec"
    assert "Status: pass_with_warnings" in out
    assert "Target: churn" in out
    assert "Report:" in out


def test_validate_cli_cold_offline_exits_two(tmp_path, capsys):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_candidate()
    write_report(settings.report_dir, candidate)

    code = main(
        [
            "validate",
            "--offline",
            "--json",
            "--data-dir",
            str(settings.data_dir),
            "--report-dir",
            str(settings.report_dir),
        ],
        settings=Settings(project_root=tmp_path),
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "reject"
    assert payload["failure"]["code"] == "CW-VAL-002"
    assert any("CW-ACQ-005" in cause for cause in payload["failure"]["causes"])


def test_validate_cli_without_candidates_exits_two(tmp_path, capsys):
    settings = make_acquisition_settings(tmp_path)

    code = main(
        [
            "validate",
            "--offline",
            "--data-dir",
            str(settings.data_dir),
            "--report-dir",
            str(settings.report_dir),
        ],
        settings=Settings(project_root=tmp_path),
    )

    out = capsys.readouterr().out
    assert code == 2
    assert "CW-VAL-001" in out


def test_validate_cli_rejects_invalid_candidate_budget(tmp_path, capsys):
    settings = make_acquisition_settings(tmp_path)

    code = main(
        ["validate", "--max-candidates", "0"],
        settings=Settings(project_root=tmp_path),
    )

    assert code == 3
    assert "--max-candidates" in capsys.readouterr().err
