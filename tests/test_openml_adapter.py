from __future__ import annotations

from churnwise.contracts import LicenseStatus, SearchQuery, SourceStatus
from churnwise.discovery.adapters.openml import OpenMLAdapter
from churnwise.discovery.registry import get_source
from churnwise.net import HttpResponse
from tests.helpers import (
    json_response,
    make_client,
    openml_description_payload,
    openml_features_payload,
    openml_qualities_payload,
    openml_search_payload,
)

DID = 46915
SEARCH_URL = "https://www.openml.org/api/v1/json/data/list/data_name/churn/limit/10"
META_URL = f"https://www.openml.org/api/v1/json/data/{DID}"
QUALITY_URL = f"https://www.openml.org/api/v1/json/data/qualities/{DID}"
FEATURE_URL = f"https://www.openml.org/api/v1/json/data/features/{DID}"


def build_adapter(routes):
    client, transport, _ = make_client(routes)
    return OpenMLAdapter(get_source("openml"), client), transport


def standard_routes():
    return {
        SEARCH_URL: json_response(openml_search_payload(DID)),
        META_URL: json_response(openml_description_payload(DID)),
        QUALITY_URL: json_response(openml_qualities_payload()),
        FEATURE_URL: json_response(openml_features_payload()),
    }


def test_search_enriches_candidate_from_four_endpoints():
    adapter, transport = build_adapter(standard_routes())
    result = adapter.search(SearchQuery(terms=("churn",)))

    assert result.status is SourceStatus.OK
    assert len(result.candidates) == 1
    assert result.request_count == 4
    assert len(transport.calls) == 4

    candidate = result.candidates[0]
    assert candidate.candidate_id == "openml:46915"
    assert candidate.row_count == 5000
    assert candidate.feature_count == 21
    assert candidate.target_column == "class"
    assert candidate.target_class_count == 2
    assert candidate.class_balance == {"majority": 4293, "minority": 707}
    assert candidate.format == "arff"
    assert candidate.license_status is LicenseStatus.VERIFIED
    assert candidate.license_name == "Public"
    assert candidate.resource_url == "https://www.openml.org/data/v1/download/4046915"
    assert candidate.has_numeric is True
    assert candidate.has_categorical is True
    assert candidate.is_valid


def test_openml_captures_publisher_checksum_and_version_evidence():
    routes = standard_routes()
    routes[META_URL] = json_response(
        openml_description_payload(
            DID, version="4", md5_checksum="a" * 32
        )
    )
    adapter, _ = build_adapter(routes)

    candidate = adapter.search(SearchQuery(terms=("churn",))).candidates[0]

    assert candidate.evidence["md5_checksum"] == "a" * 32
    assert candidate.evidence["version"] == "4"


BROKEN_CHECKSUM_BODY = (
    b'{"data":{"dataset":[{"did":46915,"name":"churn","format":"ARFF",'
    b'"file_id":4046915,"md5_checksum":"\n'
    b'<div style="border:1px solid #990000">\n'
    b'<p>Message: Undefined property: stdClass::$md5_checksum</p>\n'
    b'</div>","quality":[]}]}}'
)


def test_search_recovers_from_php_notice_in_optional_checksum():
    routes = standard_routes()
    routes[SEARCH_URL] = HttpResponse(status=200, body=BROKEN_CHECKSUM_BODY)
    adapter, _ = build_adapter(routes)

    result = adapter.search(SearchQuery(terms=("churn",)))

    assert result.status is SourceStatus.OK
    assert [candidate.candidate_id for candidate in result.candidates] == ["openml:46915"]
    assert result.request_count == 4


def test_empty_search_is_reported_as_empty_not_error():
    adapter, _ = build_adapter(
        {SEARCH_URL: json_response({"data": {"dataset": []}})}
    )
    result = adapter.search(SearchQuery(terms=("churn",)))
    assert result.status is SourceStatus.EMPTY
    assert result.error_code is None
    assert result.candidates == []


NO_RESULTS_BODY = b'{"error":{"code":"372","message":"No results"}}'
ATTRITION_SEARCH_URL = "https://www.openml.org/api/v1/json/data/list/data_name/attrition/limit/10"


def test_no_results_412_is_an_empty_search_not_an_error():
    adapter, _ = build_adapter({SEARCH_URL: HttpResponse(status=412, body=NO_RESULTS_BODY)})
    result = adapter.search(SearchQuery(terms=("churn",)))
    assert result.status is SourceStatus.EMPTY
    assert result.error_code is None


def test_multi_term_search_skips_terms_with_no_results():
    routes = standard_routes()
    routes[ATTRITION_SEARCH_URL] = HttpResponse(status=412, body=NO_RESULTS_BODY)
    adapter, _ = build_adapter(routes)
    result = adapter.search(SearchQuery(terms=("churn", "attrition")))

    assert result.status is SourceStatus.OK
    assert len(result.candidates) == 1
    assert result.request_count == 5


def test_other_412_responses_remain_source_errors():
    adapter, _ = build_adapter(
        {SEARCH_URL: HttpResponse(status=412, body=b'{"error":{"code":"999"}}')}
    )
    result = adapter.search(SearchQuery(terms=("churn",)))
    assert result.status is SourceStatus.ERROR
    assert result.error_code == "CW-SRC-002"


def test_malformed_json_is_a_source_error():
    adapter, _ = build_adapter({SEARCH_URL: HttpResponse(status=200, body=b"<html>")})
    result = adapter.search(SearchQuery(terms=("churn",)))
    assert result.status is SourceStatus.ERROR
    assert result.error_code == "CW-SRC-003"


def test_metadata_failure_for_one_dataset_does_not_hide_others():
    second = 40701
    routes = {
        SEARCH_URL: json_response(openml_search_payload(DID, second)),
        META_URL: json_response(openml_description_payload(DID)),
        QUALITY_URL: json_response(openml_qualities_payload()),
        FEATURE_URL: json_response(openml_features_payload()),
        f"https://www.openml.org/api/v1/json/data/{second}": HttpResponse(status=404),
    }
    adapter, _ = build_adapter(routes)
    result = adapter.search(SearchQuery(terms=("churn",)))

    assert result.status is SourceStatus.OK
    assert [c.candidate_id for c in result.candidates] == ["openml:46915"]
    assert any("skipped" in note for note in result.notes)


def test_all_enrichment_failures_are_not_a_silent_empty_search():
    routes = {
        SEARCH_URL: json_response(openml_search_payload(DID)),
        META_URL: HttpResponse(status=404),
    }
    adapter, _ = build_adapter(routes)
    result = adapter.search(SearchQuery(terms=("churn",)))
    assert result.status is SourceStatus.ERROR
    assert result.error_code == "CW-SRC-003"
    assert result.notes


def test_zero_reported_classes_is_flagged_as_limitation():
    routes = {
        SEARCH_URL: json_response(openml_search_payload(DID)),
        META_URL: json_response(openml_description_payload(DID)),
        QUALITY_URL: json_response(openml_qualities_payload(NumberOfClasses="0")),
        FEATURE_URL: json_response({"data_features": {"feature": []}}),
    }
    adapter, _ = build_adapter(routes)
    result = adapter.search(SearchQuery(terms=("churn",)))
    candidate = result.candidates[0]

    assert candidate.target_class_count is None
    assert any("zero target classes" in limitation for limitation in candidate.limitations)
    assert candidate.schema_fields is None
    assert candidate.has_numeric is None


def test_duplicate_dataset_ids_across_terms_are_enriched_once():
    routes = {
        SEARCH_URL: json_response(openml_search_payload(DID)),
        "https://www.openml.org/api/v1/json/data/list/data_name/attrition/limit/10": json_response(
            openml_search_payload(DID)
        ),
        META_URL: json_response(openml_description_payload(DID)),
        QUALITY_URL: json_response(openml_qualities_payload()),
        FEATURE_URL: json_response(openml_features_payload()),
    }
    adapter, _ = build_adapter(routes)
    result = adapter.search(SearchQuery(terms=("churn", "attrition")))

    assert len(result.candidates) == 1
    assert result.request_count == 5  # 2 searches + 3 enrichments


def test_pii_column_is_detected_and_limited():
    routes = standard_routes()
    routes[FEATURE_URL] = json_response(openml_features_payload(include_pii=True))
    adapter, _ = build_adapter(routes)
    candidate = adapter.search(SearchQuery(terms=("churn",))).candidates[0]

    assert candidate.pii_columns() == ["customer_email"]
    assert any("possible PII" in limitation for limitation in candidate.limitations)
