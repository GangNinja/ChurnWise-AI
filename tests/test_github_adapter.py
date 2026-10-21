from __future__ import annotations

from churnwise.contracts import LicenseStatus, SearchQuery, SourceStatus
from churnwise.discovery.adapters.github import GitHubPinnedAdapter
from churnwise.discovery.registry import get_source
from churnwise.net import HttpResponse
from tests.helpers import github_repo_payload, json_response, make_client

REPO_URL = "https://api.github.com/repos/IBM/telco-customer-churn-on-icp4d"


def build_adapter(routes):
    client, _, _ = make_client(routes)
    return GitHubPinnedAdapter(get_source("github-ibm"), client)


def test_pinned_resource_is_returned_with_repository_licence_evidence():
    adapter = build_adapter({REPO_URL: json_response(github_repo_payload())})
    result = adapter.search(SearchQuery(terms=("churn",)))

    assert result.status is SourceStatus.OK
    candidate = result.candidates[0]
    assert candidate.candidate_id == "github-ibm:ibm-telco-churn"
    assert candidate.format == "csv"
    assert candidate.row_count is None
    assert candidate.target_column is None
    assert candidate.license_status is LicenseStatus.UNVERIFIED
    assert candidate.license_name is None
    assert candidate.evidence["repository_license"] == "Apache-2.0"
    assert any("repository licence" in limitation for limitation in candidate.limitations)
    assert any("not verified until Phase 3" in limitation for limitation in candidate.limitations)


def test_repository_api_failure_degrades_gracefully():
    adapter = build_adapter({REPO_URL: HttpResponse(status=403)})
    result = adapter.search(SearchQuery(terms=("churn",)))

    assert result.status is SourceStatus.OK
    assert len(result.candidates) == 1
    assert result.notes and "unavailable" in result.notes[0]
    assert result.candidates[0].license_status is LicenseStatus.UNVERIFIED


def test_unrelated_query_returns_empty_source():
    adapter = build_adapter({})
    result = adapter.search(SearchQuery(terms=("housing",)))
    assert result.status is SourceStatus.EMPTY
    assert result.candidates == []
