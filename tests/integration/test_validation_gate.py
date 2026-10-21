"""End-to-end Phase 4 gate tests on a real acquisition pipeline (T4.4–T4.6)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from churnwise.acquisition.coordinator import AcquisitionCoordinator
from churnwise.validation import ValidationCoordinator, ValidationStatus
from tests.acquisition_helpers import (
    FakeDownloadTransport,
    FakeStream,
    make_acquisition_candidate,
    make_acquisition_settings,
    make_downloader,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_churn.csv"
URL = "https://www.openml.org/data/v1/download/900001"


def acquire(tmp_path: Path, content: bytes):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(
        candidate_id="openml:fixture",
        source_dataset_id="900001",
        title="mini customer churn",
        resource_url=URL,
        target_column="churn",
    )
    transport = FakeDownloadTransport({URL: FakeStream(body=content)})
    coordinator = AcquisitionCoordinator(
        settings,
        downloader_factory=lambda hosts: make_downloader(
            transport, hosts=tuple(hosts), max_bytes=settings.max_download_bytes
        ),
    )
    result = coordinator.acquire([candidate])
    assert result.succeeded, result.failure
    return settings, result


def test_validation_writes_report_profile_and_plots(tmp_path):
    content = FIXTURE.read_bytes()
    settings, acquisition = acquire(tmp_path, content)
    result = ValidationCoordinator(settings).validate(acquisition)

    assert result.status is ValidationStatus.PASS_WITH_WARNINGS
    assert result.report is not None
    assert result.report.gate.can_preprocess is True
    assert result.report.gate.blocking == []
    assert "V2.small" in {
        issue.code
        for check in result.report.checks
        for issue in check.issues
    }
    assert [check.id for check in result.report.checks] == [
        f"V{index}" for index in range(1, 13)
    ]
    assert all(check.evidence is not None for check in result.report.checks)

    report_json = tmp_path / "reports" / "validation" / "openml_fixture.json"
    report_md = tmp_path / "reports" / "validation" / "openml_fixture.md"
    profile_json = tmp_path / "reports" / "eda" / "openml_fixture.json"
    assert report_json.is_file()
    assert report_md.is_file()
    assert profile_json.is_file()
    plots = sorted((tmp_path / "reports" / "eda").glob("openml_fixture*.png"))
    assert len(plots) >= 3

    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "pass_with_warnings"
    assert payload["artifact_sha256"] == hashlib.sha256(content).hexdigest()
    assert payload["manifest_sha256"]
    assert len(payload["checks"]) == 12
    assert payload["gate"]["can_preprocess"] is True
    assert payload["target"]["column"] == "churn"
    assert "pass_with_warnings" in report_md.read_text(encoding="utf-8")


def test_report_values_match_pandas_recomputation(tmp_path):
    content = FIXTURE.read_bytes()
    settings, acquisition = acquire(tmp_path, content)
    result = ValidationCoordinator(settings).validate(acquisition)
    report = result.report
    assert report is not None

    raw_path = tmp_path / "data" / acquisition.artifact.raw_path
    frame = pd.read_csv(raw_path)
    assert report.summary["n_rows"] == int(len(frame))
    assert report.summary["n_cols"] == int(frame.shape[1])
    assert report.summary["missing_cells"] == int(frame.isna().sum().sum())
    assert report.summary["duplicate_rows"] == int(frame.duplicated().sum())
    recomputed_rate = float((frame["churn"] == "Yes").mean())
    assert report.target["positive_rate"] == pytest.approx(recomputed_rate, abs=1e-12)
    assert report.summary["target_rate"] == pytest.approx(recomputed_rate, abs=1e-12)
    assert report.target["positive_count"] == int((frame["churn"] == "Yes").sum())
    assert report.summary["usable_numeric"] == [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_calls",
    ]
    assert report.summary["usable_categorical"] == ["contract"]


def test_identifier_is_flagged_and_excluded_from_features(tmp_path):
    settings, acquisition = acquire(tmp_path, FIXTURE.read_bytes())
    result = ValidationCoordinator(settings).validate(acquisition)

    assert result.report is not None
    assert result.report.roles["id"] == "customerID"
    assert "customerID" not in result.features
    identifier = next(
        check for check in result.report.checks if check.id == "V11"
    )
    assert identifier.passed is False
    assert "V11.identifier" in {issue.code for issue in identifier.issues}
    id_column = next(
        column
        for column in result.profile.columns
        if column.name == "customerID"
    )
    assert "id_like" in id_column.flags
    assert id_column.role == "id"


def test_normalized_frame_is_one_for_churn_and_leaves_artifact_untouched(tmp_path):
    settings, acquisition = acquire(tmp_path, FIXTURE.read_bytes())
    raw_path = tmp_path / "data" / acquisition.artifact.raw_path
    before = raw_path.read_bytes()

    result = ValidationCoordinator(settings).validate(acquisition)
    normalized = result.normalized_frame()

    assert normalized is not None
    assert set(normalized["churn"].unique()) == {0, 1}
    assert normalized["churn"].dtype == "Int64"
    assert result.frame is not None
    assert set(result.frame["churn"].unique()) == {"Yes", "No"}
    assert raw_path.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == result.report.artifact_sha256


def test_single_class_target_blocks_the_gate_end_to_end(tmp_path):
    frame = pd.read_csv(FIXTURE)
    frame["churn"] = "No"
    content = frame.to_csv(index=False).encode("utf-8")
    settings, acquisition = acquire(tmp_path, content)

    result = ValidationCoordinator(settings).validate(acquisition)

    assert result.report is not None
    assert result.status is ValidationStatus.REJECT
    assert result.report.gate.can_preprocess is False
    assert result.can_preprocess is False
    assert "V4" in result.report.gate.blocking
    assert result.normalized_frame() is None
    assert (tmp_path / "reports" / "validation" / "openml_fixture.json").is_file()


def test_tampered_artifact_is_rejected_and_manifest_quarantined(tmp_path):
    settings, acquisition = acquire(tmp_path, FIXTURE.read_bytes())
    raw_path = tmp_path / "data" / acquisition.artifact.raw_path
    raw_path.write_bytes(raw_path.read_bytes() + b"\n0,0,0,0,monthly,0,No\n")

    result = ValidationCoordinator(settings).validate(acquisition)

    assert result.report is not None
    assert result.status is ValidationStatus.REJECT
    checks = {check.id: check for check in result.report.checks}
    assert checks["V1"].passed is False
    assert checks["V1"].severity.value == "fatal"
    assert checks["V2"].severity.value == "fatal"
    assert result.frame is None
    assert list((tmp_path / "data" / "manifests" / "quarantine").glob("*.json"))


def test_rejected_candidate_falls_back_to_the_next_eligible_one(tmp_path):
    bad_frame = pd.read_csv(FIXTURE)
    bad_frame["churn"] = "No"
    bad_url = "https://www.openml.org/data/v1/download/900003"
    good_url = "https://www.openml.org/data/v1/download/900004"
    bad_candidate = make_acquisition_candidate(
        candidate_id="openml:bad",
        source_dataset_id="900003",
        title="single-class churn",
        resource_url=bad_url,
        target_column="churn",
    )
    good_candidate = make_acquisition_candidate(
        candidate_id="openml:good",
        source_dataset_id="900004",
        title="mini customer churn",
        resource_url=good_url,
        target_column="churn",
    )
    settings = make_acquisition_settings(tmp_path)
    transport = FakeDownloadTransport(
        {
            bad_url: FakeStream(body=bad_frame.to_csv(index=False).encode("utf-8")),
            good_url: FakeStream(body=FIXTURE.read_bytes()),
        }
    )
    acquisition = AcquisitionCoordinator(
        settings,
        downloader_factory=lambda hosts: make_downloader(
            transport, hosts=tuple(hosts), max_bytes=settings.max_download_bytes
        ),
    )

    result = ValidationCoordinator(settings, acquisition=acquisition).validate_candidates(
        [bad_candidate, good_candidate]
    )

    assert result.can_preprocess is True
    assert [attempt.candidate_id for attempt in result.attempts] == [
        "openml:bad",
        "openml:good",
    ]
    assert result.attempts[0].status == "reject"
    assert result.attempts[0].error_code == "V4"
    assert result.attempts[0].report_path
    assert result.attempts[1].status == "pass_with_warnings"
    payload = json.loads(
        (settings.report_dir / "validation" / "attempts.json").read_text(
            encoding="utf-8"
        )
    )
    assert [attempt["candidate_id"] for attempt in payload["attempts"]] == [
        "openml:bad",
        "openml:good",
    ]
    assert payload["status"] == "pass_with_warnings"


def test_candidate_budget_stops_the_fallback(tmp_path):
    bad_frame = pd.read_csv(FIXTURE)
    bad_frame["churn"] = "No"
    bad_url = "https://www.openml.org/data/v1/download/900005"
    good_url = "https://www.openml.org/data/v1/download/900006"
    candidates = [
        make_acquisition_candidate(
            candidate_id="openml:bad",
            source_dataset_id="900005",
            title="single-class churn",
            resource_url=bad_url,
            target_column="churn",
        ),
        make_acquisition_candidate(
            candidate_id="openml:good",
            source_dataset_id="900006",
            title="mini customer churn",
            resource_url=good_url,
            target_column="churn",
        ),
    ]
    settings = make_acquisition_settings(tmp_path)
    transport = FakeDownloadTransport(
        {
            bad_url: FakeStream(body=bad_frame.to_csv(index=False).encode("utf-8")),
            good_url: FakeStream(body=FIXTURE.read_bytes()),
        }
    )
    acquisition = AcquisitionCoordinator(
        settings,
        downloader_factory=lambda hosts: make_downloader(
            transport, hosts=tuple(hosts), max_bytes=settings.max_download_bytes
        ),
    )

    result = ValidationCoordinator(settings, acquisition=acquisition).validate_candidates(
        candidates, max_candidates=1
    )

    assert result.can_preprocess is False
    assert result.status is ValidationStatus.REJECT
    assert [attempt.candidate_id for attempt in result.attempts] == ["openml:bad"]
    assert result.failure is not None
    assert result.failure.code == "CW-VAL-002"
    assert transport.calls == [bad_url]
