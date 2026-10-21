from __future__ import annotations

import json

from churnwise.acquisition import downloader as downloader_module
from churnwise.acquisition.coordinator import AcquisitionCoordinator
from churnwise.cli import main
from churnwise.config import Settings
from tests.acquisition_helpers import (
    FakeDownloadTransport,
    FakeStream,
    csv_bytes,
    make_acquisition_candidate,
    make_acquisition_settings,
    make_downloader,
)

URL = "https://www.openml.org/data/v1/download/1"


def write_report(report_dir, candidate):
    directory = report_dir / "acquisition"
    directory.mkdir(parents=True, exist_ok=True)
    entry = candidate.to_dict()
    entry.update({"eligible": True, "duplicate_of": None, "rank": 1, "score": 0.9})
    (directory / "candidates.json").write_text(
        json.dumps({"ranking": [entry], "selected": None}), encoding="utf-8"
    )


def populate_cache(settings, candidate):
    coordinator = AcquisitionCoordinator(
        settings,
        downloader_factory=lambda hosts: make_downloader(
            FakeDownloadTransport({URL: FakeStream(body=csv_bytes())}), hosts=tuple(hosts)
        ),
    )
    result = coordinator.acquire([candidate])
    assert result.succeeded, result.failure


def test_acquire_cli_reuses_warm_cache_offline(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL)
    populate_cache(settings, candidate)
    write_report(settings.report_dir, candidate)

    code = main(
        [
            "acquire",
            "--offline",
            "--data-dir",
            str(settings.data_dir),
            "--report-dir",
            str(settings.report_dir),
        ],
        settings=Settings(project_root=tmp_path),
    )

    assert code == 0
    attempts = json.loads((settings.report_dir / "acquisition" / "attempts.json").read_text())
    assert attempts["attempts"][0]["status"] == "cached"


def test_acquire_cli_cold_offline_reports_offline_code(tmp_path, capsys):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL)
    write_report(settings.report_dir, candidate)

    code = main(
        [
            "acquire",
            "--offline",
            "--data-dir",
            str(settings.data_dir),
            "--report-dir",
            str(settings.report_dir),
        ],
        settings=Settings(project_root=tmp_path),
    )

    assert code == 2
    output = capsys.readouterr().out
    assert "CW-ACQ-005" in output
    attempts = json.loads((settings.report_dir / "acquisition" / "attempts.json").read_text())
    assert attempts["attempts"][0]["error_code"] == "CW-ACQ-005"


def test_acquire_cli_json_output_is_machine_readable(tmp_path, capsys):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL)
    populate_cache(settings, candidate)
    write_report(settings.report_dir, candidate)

    code = main(
        [
            "acquire",
            "--offline",
            "--json",
            "--data-dir",
            str(settings.data_dir),
            "--report-dir",
            str(settings.report_dir),
        ],
        settings=Settings(project_root=tmp_path),
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "success"
    assert payload["artifact"]["sha256"]
    assert payload["manifest"]["integrity"]["verified"] is True


def test_acquire_cli_honours_offline_environment_for_warm_cache(tmp_path, monkeypatch):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL)
    populate_cache(settings, candidate)
    write_report(settings.report_dir, candidate)
    monkeypatch.setenv("CHURNWISE_OFFLINE", "1")
    monkeypatch.setenv("CHURNWISE_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("CHURNWISE_REPORT_DIR", str(settings.report_dir))
    monkeypatch.setattr(
        downloader_module.RequestsDownloadTransport,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network access attempted")),
    )

    code = main(["acquire"])

    assert code == 0
    attempts = json.loads((settings.report_dir / "acquisition" / "attempts.json").read_text())
    assert attempts["offline"] is True
    assert attempts["dataset_sha256"]
    assert attempts["seed"] == settings.seed
    assert attempts["code_version"]
    assert attempts["attempts"][0]["status"] == "cached"


def test_acquire_cli_honours_offline_environment_on_cold_cache(tmp_path, monkeypatch, capsys):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL)
    write_report(settings.report_dir, candidate)
    monkeypatch.setenv("CHURNWISE_OFFLINE", "1")
    monkeypatch.setenv("CHURNWISE_DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("CHURNWISE_REPORT_DIR", str(settings.report_dir))

    code = main(["acquire"])

    assert code == 2
    assert "CW-ACQ-005" in capsys.readouterr().out


def test_acquire_cli_rejects_invalid_size_configuration(tmp_path):
    code = main(
        ["acquire", "--offline", "--max-download-bytes", "1"],
        settings=Settings(project_root=tmp_path),
    )
    assert code == 3
