from __future__ import annotations

from churnwise.contracts import FieldInfo, GateStatus, LicenseStatus, SearchQuery
from churnwise.discovery.suitability import assess_candidate, evaluate_gates, is_eligible
from tests.helpers import make_candidate

QUERY = SearchQuery()


def gate(gates, gate_id):
    return next(g for g in gates if g.gate_id == gate_id)


def test_eligible_candidate_passes_all_blocking_gates_but_warns_on_unknown_size():
    assessment = assess_candidate(make_candidate(), QUERY)
    statuses = {g.gate_id: g.status for g in assessment.gates}
    assert statuses["G1"] is GateStatus.PASS
    assert statuses["G8"] is GateStatus.UNKNOWN
    assert assessment.eligible is True


def test_unsupported_format_fails_g1():
    assessment = assess_candidate(make_candidate(format="docx"), QUERY)
    assert gate(assessment.gates, "G1").status is GateStatus.FAIL
    assert assessment.eligible is False


def test_row_count_unknown_or_too_small_fails_g2():
    assert gate(evaluate_gates(make_candidate(row_count=None), QUERY), "G2").status is GateStatus.FAIL
    assert gate(evaluate_gates(make_candidate(row_count=500), QUERY), "G2").status is GateStatus.FAIL
    assert gate(evaluate_gates(make_candidate(row_count=1000), QUERY), "G2").status is GateStatus.PASS


def test_target_must_be_declared_binary_and_two_class():
    assert gate(evaluate_gates(make_candidate(target_column=None), QUERY), "G3").status is GateStatus.FAIL
    assert (
        gate(evaluate_gates(make_candidate(target_class_count=None), QUERY), "G3").status
        is GateStatus.FAIL
    )
    assert (
        gate(evaluate_gates(make_candidate(target_class_count=3), QUERY), "G3").status
        is GateStatus.FAIL
    )


def test_class_balance_unknown_is_non_blocking_but_out_of_range_fails():
    unknown = evaluate_gates(make_candidate(class_balance=None), QUERY)
    assert gate(unknown, "G4").status is GateStatus.UNKNOWN
    assert is_eligible(unknown) is True

    skewed = evaluate_gates(
        make_candidate(class_balance={"majority": 9900, "minority": 100}, row_count=10000), QUERY
    )
    assert gate(skewed, "G4").status is GateStatus.FAIL

    balanced = evaluate_gates(
        make_candidate(class_balance={"majority": 3000, "minority": 2000}, row_count=5000), QUERY
    )
    assert gate(balanced, "G4").status is GateStatus.PASS


def test_predictor_count_must_be_at_least_five():
    # feature_count includes the target column, so 6 columns = 5 predictors.
    assert gate(evaluate_gates(make_candidate(feature_count=6), QUERY), "G5").status is GateStatus.PASS
    assert gate(evaluate_gates(make_candidate(feature_count=5), QUERY), "G5").status is GateStatus.FAIL
    assert gate(evaluate_gates(make_candidate(feature_count=None), QUERY), "G5").status is GateStatus.FAIL


def test_predictor_minimum_and_feature_type_requirements_are_configurable():
    stricter = SearchQuery(min_predictors=10)
    assert gate(evaluate_gates(make_candidate(feature_count=10), stricter), "G5").status is GateStatus.FAIL
    assert gate(evaluate_gates(make_candidate(feature_count=11), stricter), "G5").status is GateStatus.PASS

    numeric = SearchQuery(require_numeric_features=True)
    assert (
        gate(evaluate_gates(make_candidate(has_numeric=False), numeric), "G5").status
        is GateStatus.FAIL
    )
    assert (
        gate(evaluate_gates(make_candidate(has_numeric=None), numeric), "G5").status
        is GateStatus.FAIL
    )

    categorical = SearchQuery(require_categorical_features=True)
    assert (
        gate(evaluate_gates(make_candidate(has_categorical=None), categorical), "G5").status
        is GateStatus.FAIL
    )
    assert (
        gate(evaluate_gates(make_candidate(has_categorical=True), categorical), "G5").status
        is GateStatus.PASS
    )


def test_customer_feature_requirement_fails_without_a_domain_column():
    required = SearchQuery(require_customer_features=True)
    assert (
        gate(evaluate_gates(make_candidate(), required), "G5").status is GateStatus.PASS
    )

    unrelated = make_candidate(
        schema_fields=[
            FieldInfo("CustomerChurned", "nominal", "target"),
            *[FieldInfo(f"x{i}", "numeric", "feature") for i in range(5)],
        ]
    )
    assert (
        gate(evaluate_gates(unrelated, required), "G5").status is GateStatus.FAIL
    )
    assert gate(evaluate_gates(make_candidate(schema_fields=None), required), "G5").status is GateStatus.FAIL


def test_id_columns_are_excluded_from_the_predictor_count():
    schema = [
        FieldInfo("CustomerChurned", "nominal", "target"),
        FieldInfo("row_id", "integer", "id"),
        *[FieldInfo(f"f{i}", "numeric", "feature") for i in range(4)],
    ]
    # 6 columns - target - id = 4 predictors, so this must fail.
    assert (
        gate(evaluate_gates(make_candidate(feature_count=6, schema_fields=schema), QUERY), "G5").status
        is GateStatus.FAIL
    )
    # 7 columns - target - id = 5 predictors, so this passes.
    assert (
        gate(evaluate_gates(make_candidate(feature_count=7, schema_fields=schema), QUERY), "G5").status
        is GateStatus.PASS
    )


def test_licence_policy_requires_opt_in_for_unverified_sources():
    unverified = make_candidate(license_status=LicenseStatus.UNVERIFIED, license_name=None)
    assert gate(evaluate_gates(unverified, QUERY), "G6").status is GateStatus.FAIL
    assert (
        gate(evaluate_gates(unverified, SearchQuery(allow_unverified_license=True)), "G6").status
        is GateStatus.PASS
    )

    restricted = make_candidate(license_status=LicenseStatus.RESTRICTED, license_name="Proprietary")
    assert (
        gate(evaluate_gates(restricted, SearchQuery(allow_unverified_license=True)), "G6").status
        is GateStatus.FAIL
    )


def test_pii_columns_fail_g7_and_missing_schema_is_blocking_unknown():
    pii = make_candidate(
        schema_fields=[
            FieldInfo("CustomerChurned", "nominal", "target"),
            FieldInfo("email", "string", "feature"),
        ]
    )
    assert gate(evaluate_gates(pii, QUERY), "G7").status is GateStatus.FAIL

    unknown = evaluate_gates(make_candidate(schema_fields=None), QUERY)
    assert gate(unknown, "G7").status is GateStatus.UNKNOWN
    assert is_eligible(unknown) is False


def test_download_size_gate_has_pass_fail_and_unknown_cases():
    assert (
        gate(evaluate_gates(make_candidate(size_bytes=10 * 1024 * 1024), QUERY), "G8").status
        is GateStatus.PASS
    )
    assert (
        gate(evaluate_gates(make_candidate(size_bytes=500 * 1024 * 1024), QUERY), "G8").status
        is GateStatus.FAIL
    )
    assert (
        gate(evaluate_gates(make_candidate(size_bytes=None), QUERY), "G8").status
        is GateStatus.UNKNOWN
    )


def test_scores_are_bounded_and_weighted_components_sum_to_score():
    assessment = assess_candidate(make_candidate(), QUERY)
    assert 0.0 <= assessment.score <= 1.0
    assert sum(c.weight for c in assessment.components) == 1.0
    recomputed = sum(c.value * c.weight for c in assessment.components)
    assert abs(recomputed - assessment.score) < 1e-9
    assert all(0.0 <= c.value <= 1.0 for c in assessment.components)


def test_customer_domain_columns_are_detected_and_rewarded():
    with_domain = assess_candidate(make_candidate(), QUERY)
    without_domain = assess_candidate(
        make_candidate(
            schema_fields=[
                FieldInfo("CustomerChurned", "nominal", "target"),
                FieldInfo("x1", "numeric", "feature"),
            ]
        ),
        QUERY,
    )

    with_component = next(c for c in with_domain.components if c.name == "schema_fit")
    without_component = next(c for c in without_domain.components if c.name == "schema_fit")
    assert "customer-domain columns" in with_component.detail
    assert "no customer-domain column names" in without_component.detail
    assert with_component.value > without_component.value


def test_licence_clarity_orders_verified_above_unverified():
    verified = assess_candidate(make_candidate(), QUERY).score
    unverified = assess_candidate(
        make_candidate(license_status=LicenseStatus.UNVERIFIED, license_name=None), QUERY
    ).score
    assert verified > unverified
