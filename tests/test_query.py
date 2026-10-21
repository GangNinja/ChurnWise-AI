from __future__ import annotations

from churnwise.contracts import SearchQuery


def test_default_query_is_valid():
    assert SearchQuery().validate() == []


def test_terms_must_be_present_unique_and_bounded():
    assert SearchQuery(terms=()).validate()
    assert SearchQuery(terms=("churn", "CHURN")).validate()
    assert SearchQuery(terms=("a",)).validate()
    assert SearchQuery(terms=tuple(f"term-{i}" for i in range(7))).validate()


def test_numeric_bounds_are_enforced():
    assert SearchQuery(min_rows=10).validate()
    assert SearchQuery(max_rows=500, min_rows=1000).validate()
    assert SearchQuery(timeout_s=0.5).validate()
    assert SearchQuery(max_retries=9).validate()
    assert SearchQuery(max_requests_per_source=1).validate()
    assert SearchQuery(max_results_per_source=0).validate()


def test_unsupported_format_is_rejected():
    assert SearchQuery(formats=("docx",)).validate()


def test_predictor_and_feature_type_criteria_are_configurable():
    assert SearchQuery(min_predictors=0).validate()
    assert SearchQuery(min_predictors=101).validate()
    assert SearchQuery(min_predictors=1).validate() == []
    assert SearchQuery(require_numeric_features=True).validate() == []
    assert SearchQuery(require_categorical_features=True).validate() == []
    assert SearchQuery(require_customer_features=True).validate() == []


def test_fingerprint_is_stable_and_sensitive_to_criteria():
    first = SearchQuery().fingerprint()
    assert first == SearchQuery().fingerprint()
    assert len(first) == 16
    assert first != SearchQuery(terms=("churn",)).fingerprint()
    assert first != SearchQuery(allow_unverified_license=True).fingerprint()
    assert first != SearchQuery(min_predictors=6).fingerprint()
    assert first != SearchQuery(require_numeric_features=True).fingerprint()
    assert first != SearchQuery(require_categorical_features=True).fingerprint()
    assert first != SearchQuery(require_customer_features=True).fingerprint()
