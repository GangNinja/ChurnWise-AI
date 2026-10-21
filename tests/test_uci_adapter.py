from __future__ import annotations

from churnwise.contracts import LicenseStatus, SearchQuery, SourceStatus
from churnwise.discovery.adapters.uci import UCIAdapter
from churnwise.discovery.registry import get_source
from churnwise.net import HttpResponse
from tests.helpers import json_response, make_client, uci_metadata_payload, uci_search_payload

SEARCH_URL = "https://archive.ics.uci.edu/api/datasets/list?search=churn"
META_URL = "https://archive.ics.uci.edu/api/dataset?id=563"


def build_adapter(routes):
    client, _, _ = make_client(routes)
    return UCIAdapter(get_source("uci"), client)


def test_candidate_is_built_from_api_metadata():
    adapter = build_adapter(
        {
            SEARCH_URL: json_response(uci_search_payload((563, "Iranian Churn"))),
            META_URL: json_response(uci_metadata_payload()),
        }
    )
    result = adapter.search(SearchQuery(terms=("churn",)))
    assert result.status is SourceStatus.OK
    candidate = result.candidates[0]

    assert candidate.candidate_id == "uci:563"
    assert candidate.title == "Iranian Churn"
    assert candidate.row_count == 3150
    assert candidate.feature_count == 13
    assert candidate.target_column == "Churn"
    assert candidate.target_class_count == 2
    assert candidate.format == "csv"
    assert candidate.license_status is LicenseStatus.UNVERIFIED
    assert candidate.license_name is None
    assert any("licence" in limitation.lower() for limitation in candidate.limitations)
    assert candidate.resource_url == "https://archive.ics.uci.edu/static/public/563/data.csv"


def test_empty_search_is_empty_not_error():
    adapter = build_adapter({SEARCH_URL: json_response({"data": []})})
    result = adapter.search(SearchQuery(terms=("churn",)))
    assert result.status is SourceStatus.EMPTY
    assert result.error_code is None


def test_null_search_result_is_empty():
    adapter = build_adapter({SEARCH_URL: json_response({"data": None})})
    result = adapter.search(SearchQuery(terms=("churn",)))
    assert result.status is SourceStatus.EMPTY


def test_metadata_http_error_is_reported_not_swallowed():
    adapter = build_adapter(
        {
            SEARCH_URL: json_response(uci_search_payload((563, "Iranian Churn"))),
            META_URL: HttpResponse(status=500),
        }
    )
    result = adapter.search(SearchQuery(terms=("churn",)))
    assert result.status is SourceStatus.ERROR
    assert result.error_code == "CW-SRC-003"
    assert result.notes and "skipped" in result.notes[0]


def test_non_binary_target_leaves_cardinality_unknown():
    adapter = build_adapter(
        {
            SEARCH_URL: json_response(uci_search_payload((563, "Iranian Churn"))),
            META_URL: json_response(uci_metadata_payload(target_type="Integer")),
        }
    )
    candidate = adapter.search(SearchQuery(terms=("churn",))).candidates[0]
    assert candidate.target_class_count is None
    assert any("cardinality is not declared" in limitation for limitation in candidate.limitations)
