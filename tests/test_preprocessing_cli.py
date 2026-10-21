"""CLI wiring tests for ``churnwise prepare`` (Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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
URL = "https://www.openml.org/data/v1/download/910002"


def make_candidate():
    return make_acquisition_candidate(
        candidate_id="openml:fixture",
        source_dataset_id="910002",
        title="mini customer churn",
        resource_url=URL,
        target_column="churn",
    )


def write_report(report_dir: Path, candidate) -> None:
    directory = report_dir / "acquisition"
    directory.mkdir(parents=True, exist_ok=True)
    entry = candidate.to_dict()
    entry.update({"eligible": True, "duplicate_of": None, "rank": 1, "score": 0.9})
    (directory / "candidates.json").write_text(
        json.dumps({"ranking": [entry], "selected": None}), encoding="utf-8"
    )


def prepare_workspace(tmp_path: Path, content: bytes | None = None):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_candidate()
    coordinator = AcquisitionCoordinator(
        settings,
        downloader_factory=lambda hosts: make_downloader(
            FakeDownloadTransport(
                {URL: FakeStream(body=content or FIXTURE.read_bytes())}
            ),
            hosts=tuple(hosts),
        ),
    )
    result = coordinator.acquire([candidate])
    assert result.succeeded, result.failure
    write_report(settings.report_dir, candidate)
    return settings


def prepare_args(settings, *extra: str) -> list[str]:
    return [
        "prepare",
        "--offline",
        *extra,
        "--data-dir",
        str(settings.data_dir),
        "--report-dir",
        str(settings.report_dir),
    ]


def test_prepare_cli_writes_a_reloadable_run(tmp_path, capsys):
    settings = prepare_workspace(tmp_path)

    code = main(
        prepare_args(settings, "--json"),
        settings=Settings(project_root=tmp_path),
    )

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert code == 0
    assert out.isascii(), "JSON stdout must survive any console codec"
    assert payload["dataset_id"] == "openml:fixture"
    assert payload["sizes"]["train"] > payload["sizes"]["test"] > 0
    assert payload["constraints"] == {"test_touched": False, "fitted_on": "train_only"}
    assert "churn" not in payload["feature_names"]
    assert payload["feature_count"] == len(payload["feature_names"])

    run_id = payload["run_id"]
    artifacts = tmp_path / "artifacts"
    assert (artifacts / "splits" / f"{run_id}.json").is_file()
    assert (artifacts / "preprocessors" / run_id / "pipeline.joblib").is_file()
    assert (artifacts / "prepared" / run_id / "matrices.npz").is_file()
    assert (settings.report_dir / "preprocessing" / f"{run_id}.json").is_file()
    assert (settings.report_dir / "preprocessing" / f"{run_id}.md").is_file()


def test_prepare_cli_summary_is_human_readable(tmp_path, capsys):
    settings = prepare_workspace(tmp_path)

    code = main(prepare_args(settings), settings=Settings(project_root=tmp_path))

    out = capsys.readouterr().out
    assert code == 0
    assert out.isascii(), "human summary must survive any console codec"
    assert "ChurnWise AI - preprocessing" in out
    assert "fitted_on" not in out
    assert "Class weights:" in out
    assert "Report:" in out


def test_prepare_cli_rejects_invalid_fractions(tmp_path, capsys):
    settings = prepare_workspace(tmp_path)

    code = main(
        prepare_args(settings, "--train-fraction", "0.5"),
        settings=Settings(project_root=tmp_path),
    )

    assert code == 3
    assert "sum to 1.0" in capsys.readouterr().err


def test_prepare_cli_stops_when_validation_rejects(tmp_path, capsys):
    frame = pd.read_csv(FIXTURE)
    frame["churn"] = "No"
    settings = prepare_workspace(tmp_path, frame.to_csv(index=False).encode("utf-8"))

    code = main(prepare_args(settings), settings=Settings(project_root=tmp_path))

    captured = capsys.readouterr()
    assert code == 2
    assert "did not approve" in captured.err


def test_prepare_cli_disables_derived_features(tmp_path, capsys):
    settings = prepare_workspace(tmp_path)

    code = main(
        prepare_args(settings, "--no-derived", "--json"),
        settings=Settings(project_root=tmp_path),
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    schema = json.loads(
        (
            tmp_path
            / "artifacts"
            / "preprocessors"
            / payload["run_id"]
            / "schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["derived"] == []
    assert schema["derived_skipped"]
    assert all(
        "disabled" in item["reason"] for item in schema["derived_skipped"]
    )
