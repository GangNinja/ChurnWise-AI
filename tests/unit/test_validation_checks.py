"""Check-by-check tests for Phase 4 validation (T4.1, T4.2, T4.3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churnwise.eda.profile import build_profile
from churnwise.validation.checks import CHECK_NAMES, run_checks
from churnwise.validation.gate import decide_gate
from churnwise.validation.loader import load_dataset
from churnwise.validation.targets import detect_target, normalize_target

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_churn.csv"
ALL_CHECKS = list(CHECK_NAMES)


def churn_frame(
    rows: int = 1200,
    *,
    labels: list[str] | None = None,
    with_id: bool = False,
) -> pd.DataFrame:
    """Deterministic synthetic frame used as an in-memory test fixture."""
    data: dict[str, list] = {
        "tenure_months": [index % 24 + 1 for index in range(rows)],
        "monthly_charges": [20.0 + (index * 7 % 80) for index in range(rows)],
        "balance": [round(index * 1.37, 2) for index in range(rows)],
        "contract": ["yearly" if index % 3 == 0 else "monthly" for index in range(rows)],
        "churn": labels
        if labels is not None
        else ["Yes" if (index * 37 % 101) < 15 else "No" for index in range(rows)],
    }
    if with_id:
        data = {"customerID": [f"C{index:05d}" for index in range(rows)], **data}
    return pd.DataFrame(data)


def evaluate(
    frame: pd.DataFrame,
    *,
    title: str = "customer churn",
    description: str = "synthetic customer churn dataset",
    integrity_ok: bool = True,
) -> tuple[list, dict[str, object]]:
    target = detect_target(frame, dataset_title=title, dataset_description=description)
    profile = build_profile(
        frame,
        dataset_id="fixture:mini",
        format="csv",
        source_path="tests/fixtures/mini_churn.csv",
        target=target,
    )
    checks = run_checks(profile, target, integrity_ok=integrity_ok)
    return checks, {check.id: check for check in checks}


def status_of(checks: list) -> dict[str, bool]:
    return {check.id: check.passed for check in checks}


def test_check_catalogue_is_v1_to_v12():
    assert ALL_CHECKS == [f"V{index}" for index in range(1, 13)]


def test_clean_frame_passes_every_check():
    checks, _ = evaluate(churn_frame())
    failed = [(check.id, check.detail) for check in checks if not check.passed]
    assert failed == []
    assert decide_gate(checks).status.value == "pass"


def test_v1_integrity_failure_is_fatal_and_not_loaded():
    checks, by_id = evaluate(churn_frame(), integrity_ok=False)
    assert by_id["V1"].passed is False
    assert by_id["V1"].severity.value == "fatal"
    assert by_id["V2"].passed is False
    assert by_id["V3"].severity.value == "warning"  # not evaluated, not a fatal claim
    gate = decide_gate(checks)
    assert gate.status.value == "reject"
    assert "V1" in gate.blocking and "V2" in gate.blocking


def test_v1_load_error_is_fatal():
    checks = run_checks(None, None, integrity_ok=True, load_error="CW-VAL-010: unreadable")
    by_id = {check.id: check for check in checks}
    assert by_id["V1"].passed is False
    assert by_id["V1"].severity.value == "fatal"
    assert by_id["V1"].evidence["integrity_ok"] is True
    assert decide_gate(checks).can_preprocess is False


def test_v2_too_few_rows_fails():
    checks, by_id = evaluate(churn_frame(rows=50))
    assert by_id["V2"].passed is False
    assert by_id["V2"].severity.value == "fatal"
    assert any(issue.code == "V2.too_small" for issue in by_id["V2"].issues)


def test_v3_missing_target_fails():
    frame = churn_frame()[["tenure_months", "monthly_charges", "contract"]]
    checks, by_id = evaluate(frame)
    assert by_id["V3"].passed is False
    assert by_id["V3"].severity.value == "fatal"
    assert any(issue.code == "V3.missing_target" for issue in by_id["V3"].issues)


def test_v4_multiclass_target_fails():
    labels = ["Yes", "No", "Maybe"] * 400
    checks, by_id = evaluate(churn_frame(labels=labels))
    assert by_id["V4"].passed is False
    assert by_id["V4"].severity.value == "fatal"
    assert any(issue.code == "V4.not_binary" for issue in by_id["V4"].issues)


def test_v5_unrecognised_labels_fail_interpretation():
    labels = ["alpha" if (index * 37 % 101) < 15 else "beta" for index in range(1200)]
    checks, by_id = evaluate(churn_frame(labels=labels))
    assert by_id["V3"].passed is True
    assert by_id["V4"].passed is True
    assert by_id["V5"].passed is False
    assert by_id["V5"].severity.value == "fatal"
    assert any(issue.code == "V5.low_confidence" for issue in by_id["V5"].issues)


def test_v6_rare_positive_class_fails_without_blocking():
    labels = ["Yes"] + ["No"] * 1199
    checks, by_id = evaluate(churn_frame(labels=labels))
    assert by_id["V6"].passed is False
    assert by_id["V6"].severity.value == "warning"
    gate = decide_gate(checks)
    assert gate.status.value == "pass_with_warnings"
    assert gate.can_preprocess is True


def test_v7_duplicate_column_fails():
    frame = churn_frame()
    frame["monthly_charges_copy"] = frame["monthly_charges"]
    checks, by_id = evaluate(frame)
    assert by_id["V7"].passed is False
    assert any(issue.code == "V7.duplicate_content" for issue in by_id["V7"].issues)


def test_v8_duplicate_rows_fail():
    frame = churn_frame()
    frame = pd.concat([frame, frame.iloc[:40]], ignore_index=True)
    checks, by_id = evaluate(frame)
    assert by_id["V8"].passed is False
    assert any(issue.code == "V8.duplicate_rows" for issue in by_id["V8"].issues)


def test_v9_missing_target_is_fatal():
    labels = ["Yes" if index % 7 == 0 else "No" for index in range(1200)]
    frame = churn_frame(labels=labels)
    frame.loc[frame.index[:400], "churn"] = None
    checks, by_id = evaluate(frame)
    assert by_id["V9"].passed is False
    assert by_id["V9"].severity.value == "fatal"
    assert any(
        issue.code == "V9.target_missing_fatal" for issue in by_id["V9"].issues
    )


def test_v9_predictor_missingness_warns():
    frame = churn_frame()
    frame.loc[frame.index[:60], "monthly_charges"] = None
    checks, by_id = evaluate(frame)
    assert by_id["V9"].passed is False
    assert by_id["V9"].severity.value == "warning"
    assert any(issue.code == "V9.cells_missing" for issue in by_id["V9"].issues)


def test_v9_non_finite_numeric_values_warn():
    frame = churn_frame()
    frame.loc[frame.index[0], "monthly_charges"] = float("inf")
    checks, by_id = evaluate(frame)
    assert by_id["V9"].passed is False
    assert by_id["V9"].severity.value == "warning"
    assert any(issue.code == "V9.non_finite" for issue in by_id["V9"].issues)
    assert by_id["V9"].evidence["non_finite_columns"] == ["monthly_charges"]
    assert decide_gate(checks).can_preprocess is True


def test_duplicate_column_labels_do_not_break_profiling():
    frame = churn_frame()
    frame.columns = [
        "churn",
        "tenure_months",
        "monthly_charges",
        "monthly_charges",
        "contract",
    ]
    checks, by_id = evaluate(frame)
    assert [check.id for check in checks] == ALL_CHECKS
    assert by_id["V7"].passed is False
    assert any(
        issue.code == "V7.duplicate_header" for issue in by_id["V7"].issues
    )
    assert by_id["V3"].passed is True


def test_non_string_column_labels_are_normalised_not_crashed():
    frame = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]] * 100)
    target = detect_target(frame, dataset_title="headerless export")
    assert target.column is None
    profile = build_profile(
        frame,
        dataset_id="fixture:headerless",
        format="csv",
        source_path="headerless.csv",
        target=target,
    )
    assert [column.name for column in profile.columns] == ["0", "1", "2"]
    checks = run_checks(profile, target, integrity_ok=True)
    assert [check.id for check in checks] == ALL_CHECKS
    assert decide_gate(checks).can_preprocess is False


def test_v10_constant_column_fails():
    frame = churn_frame()
    frame["loyalty_tier"] = "basic"
    checks, by_id = evaluate(frame)
    assert by_id["V10"].passed is False
    assert any(issue.code == "V10.constant" for issue in by_id["V10"].issues)
    assert "loyalty_tier" in by_id["V10"].evidence["columns"]


def test_v11_identifier_and_cardinality_warnings():
    frame = churn_frame(with_id=True)
    frame["state"] = [f"state_{index % 51}" for index in range(len(frame))]
    _, by_id = evaluate(frame)
    assert by_id["V11"].passed is False
    codes = {issue.code for issue in by_id["V11"].issues}
    assert {"V11.identifier", "V11.high_cardinality"} <= codes


def test_v11_no_usable_features_is_fatal():
    labels = ["Yes" if index % 7 == 0 else "No" for index in range(1200)]
    frame = pd.DataFrame(
        {"churn": labels, "contract": ["monthly"] * len(labels)}
    )
    checks, by_id = evaluate(frame)
    assert by_id["V11"].passed is False
    assert by_id["V11"].severity.value == "fatal"
    assert any(issue.code == "V11.no_features" for issue in by_id["V11"].issues)
    assert decide_gate(checks).can_preprocess is False


def test_v11_missing_numeric_predictor_warns():
    labels = ["Yes" if index % 7 == 0 else "No" for index in range(1200)]
    frame = pd.DataFrame(
        {
            "churn": labels,
            "contract": ["yearly" if index % 3 == 0 else "monthly" for index in range(1200)],
        }
    )
    checks, by_id = evaluate(frame)
    assert by_id["V11"].passed is False
    assert by_id["V11"].severity.value == "warning"
    assert any(issue.code == "V11.no_numeric" for issue in by_id["V11"].issues)
    assert decide_gate(checks).can_preprocess is True


def test_v11_two_numeric_features_only_is_fatal():
    labels = ["Yes" if index % 7 == 0 else "No" for index in range(1200)]
    frame = pd.DataFrame(
        {
            "churn": labels,
            "monthly_charges": [40.0 + (index % 60) for index in range(1200)],
            "tenure_months": [index % 48 for index in range(1200)],
        }
    )
    checks, by_id = evaluate(frame)
    assert by_id["V11"].severity.value == "fatal"
    assert any(
        issue.code == "V11.insufficient_features" for issue in by_id["V11"].issues
    )
    assert decide_gate(checks).can_preprocess is False


def test_v11_one_categorical_feature_satisfies_the_gate():
    labels = ["Yes" if index % 7 == 0 else "No" for index in range(1200)]
    frame = pd.DataFrame(
        {
            "churn": labels,
            "monthly_charges": [40.0 + (index % 60) for index in range(1200)],
            "contract": ["yearly" if index % 3 == 0 else "monthly" for index in range(1200)],
        }
    )
    checks, by_id = evaluate(frame)
    assert not any(
        issue.code == "V11.insufficient_features" for issue in by_id["V11"].issues
    )
    assert decide_gate(checks).can_preprocess is True


def test_v12_leakage_column_fails():
    frame = churn_frame()
    frame["churn_reason"] = ["price"] * len(frame)
    checks, by_id = evaluate(frame)
    assert by_id["V12"].passed is False
    assert any(issue.code == "V12.leakage_risk" for issue in by_id["V12"].issues)
    profile = build_profile(
        frame,
        dataset_id="fixture:mini",
        format="csv",
        source_path="mini_churn.csv",
        target=detect_target(frame, dataset_title="customer churn"),
    )
    assert "churn_reason" in [item["column"] for item in profile.roles.dropped]


def test_every_check_has_a_passing_and_failing_fixture():
    """T4.1: each check id must be exercised both ways."""
    cases = [
        evaluate(churn_frame())[0],
        evaluate(churn_frame(), integrity_ok=False)[0],
        evaluate(churn_frame(rows=50))[0],
        evaluate(churn_frame()[["tenure_months", "monthly_charges", "contract"]])[0],
        evaluate(churn_frame(labels=["Yes", "No", "Maybe"] * 400))[0],
        evaluate(
            churn_frame(
                labels=["alpha" if index % 7 else "beta" for index in range(1200)]
            )
        )[0],
        evaluate(churn_frame(labels=["Yes"] + ["No"] * 1199))[0],
        evaluate(
            pd.concat(
                [churn_frame(), churn_frame().iloc[:40]], ignore_index=True
            )
        )[0],
    ]
    missing_predictor = churn_frame()
    missing_predictor.loc[missing_predictor.index[:60], "monthly_charges"] = None
    cases.append(evaluate(missing_predictor)[0])
    missing_target = churn_frame()
    missing_target.loc[missing_target.index[:400], "churn"] = None
    cases.append(evaluate(missing_target)[0])
    duplicate = churn_frame()
    duplicate["monthly_charges_copy"] = duplicate["monthly_charges"]
    cases.append(evaluate(duplicate)[0])
    constant = churn_frame()
    constant["loyalty_tier"] = "basic"
    cases.append(evaluate(constant)[0])
    identity = churn_frame(with_id=True)
    identity["state"] = [f"state_{index % 51}" for index in range(len(identity))]
    cases.append(evaluate(identity)[0])
    leakage = churn_frame()
    leakage["churn_reason"] = "price"
    cases.append(evaluate(leakage)[0])

    seen: dict[str, set[bool]] = {check_id: set() for check_id in ALL_CHECKS}
    for checks in cases:
        for check in checks:
            seen[check.id].add(check.passed)
    missing = [check_id for check_id, values in seen.items() if values != {True, False}]
    assert missing == []


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["Yes", "No"], [1, 0]),
        (["1", "0"], [1, 0]),
        (["True", "False"], [1, 0]),
        (["yes", "no"], [1, 0]),
    ],
)
def test_target_normalisation_generic_pairs(labels, expected):
    frame = pd.DataFrame({"churn": labels * 20, "tenure": list(range(40))})
    target = detect_target(frame, dataset_title="customer churn")
    mapped = normalize_target(frame["churn"], target).tolist()
    assert mapped == expected * 20
    assert target.mapping == {labels[0]: 1, labels[1]: 0}


def test_retention_style_target_is_inverted():
    frame = pd.DataFrame({"retained": ["Yes", "No"] * 20, "tenure": list(range(40))})
    target = detect_target(frame, dataset_title="customer retention")
    assert target.column == "retained"
    assert target.mapping == {"No": 1, "Yes": 0}
    assert normalize_target(frame["retained"], target).tolist() == [0, 1] * 20


def test_missing_target_labels_are_excluded_from_normalisation():
    frame = pd.DataFrame({"churn": ["Yes", None, "No"], "tenure": [1, 2, 3]})
    target = detect_target(frame, dataset_title="customer churn")
    mapped = normalize_target(frame["churn"], target)
    assert mapped.isna().tolist() == [False, True, False]
    assert mapped.dtype == "Int64"


def test_fixture_file_loads_with_labelled_synthetic_rows():
    loaded = load_dataset(FIXTURE, "csv")
    target = detect_target(loaded.frame, dataset_title="mini customer churn")
    profile = build_profile(
        loaded.frame,
        dataset_id="fixture:mini_churn",
        format="csv",
        source_path=str(FIXTURE),
        target=target,
    )
    assert len(loaded.frame) == 160
    assert target.column == "churn"
    assert target.positive_rate == pytest.approx(0.15, abs=0.02)
    assert profile.roles.id == "customerID"
    assert "customerID" not in profile.roles.features
    assert profile.capabilities["has_tenure"] is True
    assert profile.capabilities["has_monthly_charge"] is True
    assert profile.capabilities["has_contract"] is True
