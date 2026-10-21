from __future__ import annotations

import json

from churnwise.contracts import LicenseStatus, SearchQuery, SourceStatus
from churnwise.discovery.coordinator import (
    CODE_ALL_SOURCES_FAILED,
    CODE_INVALID_CONFIG,
    CODE_NO_CANDIDATES,
    CODE_NO_ELIGIBLE,
    CODE_NO_SOURCES,
    CODE_OFFLINE,
    discover,
)
from churnwise.discovery.reporting import write_reports
from tests.helpers import FakeAdapter, make_candidate, make_result

QUERY = SearchQuery()


def test_happy_path_selects_top_eligible_candidate(settings):
    openml = make_result("openml", [make_candidate()])
    uci = make_result(
        "uci",
        [
            make_candidate(
                candidate_id="uci:563",
                source_id="uci",
                source_name="UCI",
                source_dataset_id="563",
                title="Iranian Churn",
                landing_url="https://archive.ics.uci.edu/dataset/563",
                format="csv",
                license_status=LicenseStatus.UNVERIFIED,
                license_name=None,
                row_count=3150,
                feature_count=13,
            )
        ],
    )
    report = discover([FakeAdapter(openml), FakeAdapter(uci)], QUERY, settings)

    assert report.selected is not None
    assert report.selected.candidate_id == "openml:46915"
    assert report.failure is None
    assert len(report.source_results) == 2
    assert len(report.ranked()) == 2
    assert report.ranked()[0].rank == 1


def test_partial_source_failure_does_not_abort_successful_source(settings):
    failed = make_result(
        "uci",
        status=SourceStatus.ERROR,
        error_code="CW-SRC-001",
        error_message="timed out",
    )
    ok = make_result("openml", [make_candidate()])
    report = discover([FakeAdapter(failed), FakeAdapter(ok)], QUERY, settings)

    assert report.selected is not None
    assert report.source_results[0].status is SourceStatus.ERROR
    assert report.source_results[0].error_code == "CW-SRC-001"


def test_all_sources_failing_is_a_structured_total_failure(settings):
    failed = make_result(
        "openml", status=SourceStatus.UNAVAILABLE, error_code="CW-SRC-001", error_message="offline"
    )
    report = discover([FakeAdapter(failed)], QUERY, settings)

    assert report.selected is None
    assert report.failure is not None
    assert report.failure.code == CODE_ALL_SOURCES_FAILED
    assert report.failure.sources_checked == ["openml"]
    assert "offline" in " ".join(report.failure.reasons)


def test_empty_sources_produce_no_candidate_failure(settings):
    report = discover(
        [FakeAdapter(make_result("openml", status=SourceStatus.EMPTY))], QUERY, settings
    )
    assert report.failure is not None
    assert report.failure.code == CODE_NO_CANDIDATES


def test_no_eligible_candidate_reports_gate_reasons(settings):
    ineligible = make_candidate(
        license_status=LicenseStatus.UNVERIFIED, license_name=None
    )
    report = discover([FakeAdapter(make_result("openml", [ineligible]))], QUERY, settings)

    assert report.selected is None
    assert report.failure is not None
    assert report.failure.code == CODE_NO_ELIGIBLE
    assert any("G6" in reason for reason in report.failure.reasons)
    assert "allow-unverified-license" in report.failure.next_action


def test_duplicate_listings_collapse_to_one_selected_candidate(settings):
    first = make_candidate(candidate_id="openml:1", source_dataset_id="1", title="churn")
    second = make_candidate(
        candidate_id="openml:2", source_dataset_id="2", title="churn", feature_count=20
    )
    report = discover(
        [FakeAdapter(make_result("openml", [first, second]))], QUERY, settings
    )

    assert report.selected is not None
    assert report.selected.candidate_id == "openml:1"
    assert len(report.duplicate_groups) == 1
    assert len(report.duplicates()) == 1
    assert len(report.ranked()) == 1


def test_invalid_candidate_metadata_is_rejected_with_evidence(settings):
    broken = make_candidate(title="", landing_url="http://example.org/x")
    report = discover([FakeAdapter(make_result("openml", [broken]))], QUERY, settings)

    assert report.selected is None
    assert report.failure is not None
    assert report.failure.code == CODE_NO_CANDIDATES
    assert report.invalid_candidates
    assert report.invalid_candidates[0]["stage"] == "validation"
    assert report.invalid_candidates[0]["issues"]


def test_offline_mode_fails_fast(settings):
    report = discover([], QUERY, settings.with_overrides(offline=True))
    assert report.failure is not None
    assert report.failure.code == CODE_OFFLINE


def test_invalid_configuration_is_reported(settings):
    report = discover([], SearchQuery(min_rows=1), settings)
    assert report.failure is not None
    assert report.failure.code == CODE_INVALID_CONFIG


def test_no_adapters_is_reported(settings):
    report = discover([], QUERY, settings)
    assert report.failure is not None
    assert report.failure.code == CODE_NO_SOURCES


def test_reports_are_written_atomically_with_documented_shape(settings, tmp_path):
    report = discover(
        [FakeAdapter(make_result("openml", [make_candidate()]))], QUERY, settings
    )
    paths = write_reports(report, tmp_path / "reports")

    candidates = json.loads(paths["candidates"].read_text(encoding="utf-8"))
    rejections = json.loads(paths["rejections"].read_text(encoding="utf-8"))

    assert candidates["schema_version"] == "1.0"
    assert candidates["phase"] == 2
    assert candidates["summary"]["selected"] == "openml:46915"
    assert candidates["selected"]["resource_url"].startswith("https://")
    assert candidates["selected"]["snapshot"]["license"]["status"] == "verified"
    assert candidates["query"]["fingerprint"]
    assert candidates["ranking"][0]["gates"]["G3"]["status"] == "pass"
    assert rejections["summary"]["selected"] == "openml:46915"
    assert isinstance(rejections["rejections"], list)
    assert not list(paths["candidates"].parent.glob("*.tmp"))
