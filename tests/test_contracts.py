from __future__ import annotations

from churnwise.contracts import DatasetCandidate, FieldInfo, LicenseStatus, SelectedCandidate
from tests.helpers import make_candidate


def test_candidate_round_trips_through_json_dict():
    candidate = make_candidate()
    payload = candidate.to_dict()
    restored = DatasetCandidate.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.license_status is LicenseStatus.VERIFIED
    assert restored.schema_fields[0].role == "target"


def test_unknown_optional_fields_are_explicit_nulls():
    candidate = make_candidate(
        publisher=None, size_bytes=None, target_column=None, license_name=None,
        license_status=LicenseStatus.UNKNOWN,
    )
    payload = candidate.to_dict()
    assert payload["publisher"] is None
    assert payload["size_bytes"] is None
    assert payload["target_column"] is None
    assert payload["license"]["name"] is None


def test_validation_rejects_empty_title_and_non_https_landing_url():
    candidate = make_candidate(title="", landing_url="http://example.org/dataset")
    codes = {issue.code for issue in candidate.validate()}
    assert {"C002", "C004"} <= codes
    assert candidate.is_valid is False


def test_validation_rejects_non_positive_counts_and_single_class_target():
    candidate = make_candidate(row_count=0, feature_count=-2, target_class_count=1)
    codes = {issue.code for issue in candidate.validate()}
    assert {"C007", "C008", "C010"} <= codes


def test_class_balance_mismatch_is_a_warning_not_an_error():
    candidate = make_candidate(row_count=5000, class_balance={"majority": 10, "minority": 5})
    issues = candidate.validate()
    codes = {issue.code for issue in issues}
    assert "C013" in codes
    assert candidate.is_valid is True


def test_unknown_format_is_warned_but_serializable():
    candidate = make_candidate(format="docx")
    issues = candidate.validate()
    assert any(issue.code == "C014" for issue in issues)
    assert candidate.is_valid is True


def test_pii_detection_uses_schema_names_only():
    candidate = make_candidate(
        schema_fields=[
            FieldInfo("CustomerChurned", "nominal", "target"),
            FieldInfo("customer_email", "string", "feature"),
            FieldInfo("tenure", "numeric", "feature"),
        ]
    )
    assert candidate.pii_columns() == ["customer_email"]


def test_churn_target_hint_matching():
    assert make_candidate(target_column="CustomerChurned").target_name_is_churn_like()
    assert not make_candidate(target_column="species").target_name_is_churn_like()


def test_fingerprint_is_stable_and_changes_with_shape():
    first = make_candidate().fingerprint()
    second = make_candidate().fingerprint()
    third = make_candidate(row_count=4000).fingerprint()
    assert first == second
    assert first != third
    assert make_candidate(row_count=None).fingerprint() is None


def test_selected_candidate_contract_carries_resource_metadata():
    selected = SelectedCandidate.from_candidate(make_candidate())
    payload = selected.to_dict()
    assert payload["resource_url"].startswith("https://www.openml.org/data/v1/download/")
    assert payload["license"]["status"] == "verified"
    assert payload["snapshot"]["schema_version"] == "1.0"
    assert payload["target_column"] == "CustomerChurned"
