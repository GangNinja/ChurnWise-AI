from __future__ import annotations

import hashlib
import json
from pathlib import Path

from churnwise.acquisition.coordinator import AcquisitionCoordinator
from churnwise.acquisition.reporting import render_summary
from churnwise.acquisition.errors import (
    CODE_ACCESS,
    CODE_ALL_FAILED,
    CODE_CACHE_INVALID,
    CODE_INTEGRITY,
    CODE_NETWORK,
    CODE_NO_CANDIDATE,
    CODE_OFFLINE,
    CODE_SIZE_LIMIT,
    CODE_UNEXPECTED_CONTENT,
)
from churnwise.acquisition.store import DatasetStore
from churnwise.contracts import DiscoveryReport, SearchQuery, SelectedCandidate
from churnwise.discovery.suitability import assess_candidate
from tests.acquisition_helpers import (
    FakeDownloadTransport,
    FakeStream,
    csv_bytes,
    make_acquisition_candidate,
    make_acquisition_settings,
    make_downloader,
    zip_bytes,
)

URL1 = "https://www.openml.org/data/v1/download/1"
URL2 = "https://www.openml.org/data/v1/download/2"


def coordinator(settings, routes):
    transport = FakeDownloadTransport(routes)
    return (
        AcquisitionCoordinator(
            settings,                downloader_factory=lambda hosts: make_downloader(
                    transport,
                    hosts=tuple(hosts),
                    max_bytes=settings.max_download_bytes,
                ),
        ),
        transport,
    )


def test_successful_acquisition_writes_verified_artifact_and_manifest(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1)
    content = csv_bytes()
    coordinator_, _ = coordinator(settings, {URL1: FakeStream(body=content)})

    result = coordinator_.acquire([candidate])

    assert result.succeeded, result.failure
    assert result.artifact is not None
    assert result.manifest is not None
    assert result.manifest.validate() == []
    assert result.artifact.sha256 == hashlib.sha256(content).hexdigest()
    assert result.artifact.bytes == len(content)
    assert result.manifest.integrity.verified is True
    assert result.manifest.integrity.checksum_status == "unavailable"
    assert result.manifest.run_id == result.run_id
    assert result.manifest.dataset_sha256 == result.artifact.sha256
    assert result.manifest.seed == settings.seed
    assert result.manifest.code_version
    assert result.manifest.source.download_url == URL1
    assert result.manifest.acquisition.attempt == 1
    assert result.manifest.acquisition.fallback_of is None
    assert (tmp_path / "data" / result.artifact.cache_path).read_bytes() == content
    assert (tmp_path / "data" / result.artifact.raw_path).read_bytes() == content
    assert (tmp_path / "data" / "manifests").glob("*.json")
    assert result.attempts[0].status == "success"
    assert result.attempts[0].url == URL1


def test_phase2_selected_candidate_is_accepted(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1)
    coordinator_, _ = coordinator(settings, {URL1: FakeStream(body=csv_bytes())})

    result = coordinator_.acquire(SelectedCandidate.from_candidate(candidate))

    assert result.succeeded, result.failure
    assert result.candidate["candidate_id"] == candidate.candidate_id


def test_phase2_discovery_report_uses_only_eligible_ranked_candidates(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1)
    query = SearchQuery()
    assessment = assess_candidate(candidate, query)
    assert assessment.eligible
    report = DiscoveryReport(
        generated_at="2026-10-20T00:00:00Z",
        query=query,
        source_results=[],
        assessments=[assessment],
        selected=SelectedCandidate.from_candidate(candidate),
    )
    coordinator_, _ = coordinator(settings, {URL1: FakeStream(body=csv_bytes())})

    result = coordinator_.acquire(report)

    assert result.succeeded, result.failure
    assert result.manifest.discovery["origin"] == "phase2_ranking"
    assert result.manifest.discovery["score"] == assessment.score


def test_manifest_attempt_is_an_ordinal_not_the_candidate_rank(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1)
    query = SearchQuery()
    assessment = assess_candidate(candidate, query)
    assessment.rank = 5
    report = DiscoveryReport(
        generated_at="2026-10-20T00:00:00Z",
        query=query,
        source_results=[],
        assessments=[assessment],
        selected=SelectedCandidate.from_candidate(candidate),
    )
    coordinator_, _ = coordinator(settings, {URL1: FakeStream(body=csv_bytes())})

    result = coordinator_.acquire(report)

    assert result.succeeded, result.failure
    assert result.manifest.acquisition.attempt == 1
    assert result.manifest.acquisition.candidate_rank == 5
    assert result.manifest.discovery["rank"] == 5


def test_warm_cache_is_reused_without_network_even_offline(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1)
    first, _ = coordinator(settings, {URL1: FakeStream(body=csv_bytes())})
    assert first.acquire([candidate]).succeeded

    second, transport = coordinator(settings, {})
    result = second.acquire([candidate], offline=True)

    assert result.succeeded, result.failure
    assert result.cached is True
    assert result.attempts[0].status == "cached"
    assert transport.calls == []
    assert any("offline mode" in warning for warning in result.warnings)
    assert any("expected size" in warning for warning in result.warnings)


def test_policy_change_blocks_a_warm_cache_hit(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1)
    first, _ = coordinator(settings, {URL1: FakeStream(body=csv_bytes())})
    assert first.acquire([candidate]).succeeded

    changed = make_acquisition_candidate(
        resource_url=URL1,
        access="credential-required",
    )
    second, transport = coordinator(settings, {})
    result = second.acquire([changed], offline=True)

    assert not result.succeeded
    assert result.failure.code == CODE_ACCESS
    assert result.attempts[0].error_code == CODE_ACCESS
    assert transport.calls == []


def test_cold_cache_offline_fails_with_offline_code_and_no_network(tmp_path):
    settings = make_acquisition_settings(tmp_path, offline=True)
    candidate = make_acquisition_candidate(resource_url=URL1)
    coordinator_, transport = coordinator(settings, {})

    result = coordinator_.acquire([candidate], offline=True)

    assert not result.succeeded
    assert result.failure.code == CODE_OFFLINE
    assert result.attempts[0].error_code == CODE_OFFLINE
    assert transport.calls == []
    assert result.artifact is None


def test_fallback_moves_to_next_candidate_and_records_lineage(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    first = make_acquisition_candidate(
        candidate_id="openml:1", source_dataset_id="1", resource_url=URL1
    )
    second = make_acquisition_candidate(
        candidate_id="openml:2", source_dataset_id="2", resource_url=URL2
    )
    coordinator_, _ = coordinator(
        settings,
        {URL1: FakeStream(status=404), URL2: FakeStream(body=csv_bytes())},
    )

    result = coordinator_.acquire([first, second])

    assert result.succeeded, result.failure
    assert result.manifest.acquisition.fallback_of == "openml:1"
    assert [attempt.status for attempt in result.attempts] == ["failed", "success"]
    assert result.attempts[0].error_code == CODE_NETWORK
    assert result.artifact.dataset_id == "openml:2"


def test_total_failure_lists_every_attempted_source(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidates = [
        make_acquisition_candidate(
            candidate_id=f"openml:{index}",
            source_dataset_id=str(index),
            resource_url=f"https://www.openml.org/data/v1/download/{index}",
        )
        for index in range(1, 11)
    ]
    routes = {
        f"https://www.openml.org/data/v1/download/{index}": FakeStream(status=404)
        for index in range(1, 11)
    }
    coordinator_, _ = coordinator(settings, routes)

    result = coordinator_.acquire(candidates)
    summary = render_summary(result)

    assert not result.succeeded
    assert len(result.failure.causes) == 10
    assert len([line for line in summary.splitlines() if "attempt" in line]) == 10
    for candidate in candidates:
        assert candidate.candidate_id in summary


def test_all_candidates_failing_returns_structured_total_failure(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    first = make_acquisition_candidate(
        candidate_id="openml:1", source_dataset_id="1", resource_url=URL1
    )
    second = make_acquisition_candidate(
        candidate_id="openml:2", source_dataset_id="2", resource_url=URL2
    )
    coordinator_, _ = coordinator(
        settings, {URL1: FakeStream(status=404), URL2: FakeStream(status=404)}
    )

    result = coordinator_.acquire([first, second])

    assert not result.succeeded
    assert result.failure.code == CODE_NETWORK
    assert len(result.attempts) == 2
    assert result.artifact is None
    assert result.manifest is None

    summary = render_summary(result, Path("attempts.json"))
    assert "openml:1" in summary
    assert "openml:2" in summary
    assert summary.count(CODE_NETWORK) >= 2
    assert "next:" in summary


def test_no_candidate_is_structured_and_fabricates_nothing(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    coordinator_, transport = coordinator(settings, {})

    result = coordinator_.acquire([])

    assert not result.succeeded
    assert result.failure.code == CODE_NO_CANDIDATE
    assert result.attempts == []
    assert transport.calls == []


def test_publisher_checksum_mismatch_rejects_the_artifact(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    content = csv_bytes()
    candidate = make_acquisition_candidate(
        resource_url=URL1, evidence={"sha256": "0" * 64}
    )
    coordinator_, _ = coordinator(settings, {URL1: FakeStream(body=content)})

    result = coordinator_.acquire([candidate])

    assert not result.succeeded
    assert result.failure.code == CODE_INTEGRITY
    assert not list((tmp_path / "data" / "manifests").glob("*.json"))


def test_publisher_checksum_match_is_recorded_as_verified(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    content = csv_bytes()
    digest = hashlib.sha256(content).hexdigest()
    candidate = make_acquisition_candidate(resource_url=URL1, evidence={"sha256": digest})
    coordinator_, _ = coordinator(settings, {URL1: FakeStream(body=content)})

    result = coordinator_.acquire([candidate])

    assert result.succeeded, result.failure
    assert result.manifest.integrity.checksum_status == "verified"
    assert result.manifest.integrity.publisher_checksum.value == digest


def test_html_response_is_rejected_even_with_http_200(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1)
    coordinator_, _ = coordinator(
        settings,
        {URL1: FakeStream(body=b"<!doctype html><html>error</html>", headers={"Content-Type": "text/html"})},
    )

    result = coordinator_.acquire([candidate])

    assert not result.succeeded
    assert result.failure.code == CODE_UNEXPECTED_CONTENT


def test_archive_acquisition_extracts_one_tabular_member(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1, format="zip")
    archive = zip_bytes()
    coordinator_, _ = coordinator(settings, {URL1: FakeStream(body=archive)})

    result = coordinator_.acquire([candidate])

    assert result.succeeded, result.failure
    assert result.artifact.format == "zip"
    assert result.artifact.extracted_files == ["churn.csv"]
    extracted = tmp_path / "data" / result.artifact.extracted_path
    assert extracted.is_file()
    assert extracted.read_bytes().startswith(b"customer_id")
    assert result.manifest.file.extracted_sha256 == hashlib.sha256(
        extracted.read_bytes()
    ).hexdigest()
    assert result.manifest.integrity.archive_members
    assert result.manifest.file.sha256 == hashlib.sha256(archive).hexdigest()


def test_archive_member_with_spaces_and_unicode_is_accepted(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1, format="zip")
    member = "churn données 2024.csv"
    archive = zip_bytes(name=member)
    coordinator_, _ = coordinator(settings, {URL1: FakeStream(body=archive)})

    result = coordinator_.acquire([candidate])

    assert result.succeeded, result.failure
    assert result.manifest is not None
    assert result.manifest.validate() == []
    assert result.artifact.extracted_files == [member]
    extracted = tmp_path / "data" / result.artifact.extracted_path
    assert extracted.is_file()
    assert extracted.name == member


def test_tampered_extracted_manifest_hash_is_rejected(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1, format="zip")
    first, _ = coordinator(settings, {URL1: FakeStream(body=zip_bytes())})
    acquired = first.acquire([candidate])
    assert acquired.succeeded, acquired.failure

    store = DatasetStore(tmp_path / "data")
    manifest = store.lookup(candidate).hit.manifest
    manifest.file.extracted_sha256 = "b" * 64
    store.save_manifest(manifest)

    second, transport = coordinator(settings, {})
    result = second.acquire([candidate], offline=True)

    assert not result.succeeded
    assert result.failure is not None
    assert result.failure.code == CODE_CACHE_INVALID
    assert transport.calls == []


def test_tampered_extracted_file_is_restored_from_the_verified_archive(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1, format="zip")
    first, _ = coordinator(settings, {URL1: FakeStream(body=zip_bytes())})
    acquired = first.acquire([candidate])
    assert acquired.succeeded, acquired.failure
    assert acquired.artifact.extracted_path is not None
    extracted = tmp_path / "data" / acquired.artifact.extracted_path
    extracted.write_bytes(b"tampered")

    second, transport = coordinator(settings, {})
    result = second.acquire([candidate], offline=True)

    assert result.succeeded, result.failure
    assert result.cached is True
    assert transport.calls == []
    assert extracted.read_bytes().startswith(b"customer_id")


def test_download_cap_is_enforced_by_the_coordinator(tmp_path):
    settings = make_acquisition_settings(tmp_path, max_download_bytes=5)
    candidate = make_acquisition_candidate(resource_url=URL1)
    coordinator_, _ = coordinator(settings, {URL1: FakeStream(body=csv_bytes())})

    result = coordinator_.acquire([candidate])

    assert not result.succeeded
    assert result.failure.code == CODE_SIZE_LIMIT
    assert result.failure.category == "storage"


def test_malformed_candidate_metadata_is_rejected_without_crashing(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(
        resource_url=URL1,
        class_balance={"majority": "not-a-number", "minority": "1"},
    )
    coordinator_, transport = coordinator(settings, {URL1: FakeStream(body=csv_bytes())})

    result = coordinator_.acquire([candidate])

    assert not result.succeeded
    assert result.failure.code == CODE_NO_CANDIDATE
    assert result.attempts[0].error_code == CODE_NO_CANDIDATE
    assert transport.calls == []


def test_unsupported_format_is_rejected_before_download(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1, format="docx")
    coordinator_, transport = coordinator(settings, {URL1: FakeStream(body=csv_bytes())})

    result = coordinator_.acquire([candidate])

    assert not result.succeeded
    assert result.failure.code == CODE_UNEXPECTED_CONTENT
    assert transport.calls == []


def test_manifest_sanitizes_sensitive_url_parameters(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    url = URL1 + "?token=super-secret&foo=bar"
    candidate = make_acquisition_candidate(resource_url=url)
    coordinator_, _ = coordinator(settings, {url: FakeStream(body=csv_bytes())})

    result = coordinator_.acquire([candidate])

    assert result.succeeded, result.failure
    assert "super-secret" not in result.manifest.source.download_url
    assert "token=" not in result.manifest.source.download_url
    assert "foo=bar" in result.manifest.source.download_url
    assert "super-secret" not in json.dumps(result.to_dict())


def test_result_exposes_the_documented_phase4_contract(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1)
    coordinator_, _ = coordinator(settings, {URL1: FakeStream(body=csv_bytes())})

    result = coordinator_.acquire([candidate])
    payload = result.to_dict()

    assert payload["phase"] == 3
    assert payload["status"] == "success"
    assert payload["seed"] == settings.seed
    assert payload["code_version"]
    assert payload["dataset_sha256"] == result.artifact.sha256
    assert payload["artifact"]["sha256"] == result.artifact.sha256
    assert payload["manifest"]["integrity"]["verified"] is True
    assert payload["storage"]["data_dir"] == str(settings.data_dir)
    assert payload["candidate"]["candidate_id"] == candidate.candidate_id


def test_restricted_or_non_open_candidate_is_rejected_before_download(tmp_path):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(resource_url=URL1, access="credential-required")
    coordinator_, transport = coordinator(settings, {URL1: FakeStream(body=csv_bytes())})

    result = coordinator_.acquire([candidate])

    assert not result.succeeded
    assert result.failure.code == CODE_ACCESS
    assert transport.calls == []
