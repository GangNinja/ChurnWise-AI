"""Column planning, missing values and derived-feature tests (T5.3, T5.4, T5.5)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from churnwise.eda.profile import build_profile
from churnwise.preprocessing.contracts import PreprocessConfig
from churnwise.preprocessing.pipeline import TypePlan, drop_unusable_columns
from churnwise.preprocessing.features import apply_derived, derive_features
from churnwise.preprocessing.pipeline import (
    build_preprocessor,
    clean_frame,
    describe_outputs,
    numeric_quality,
    plan_features,
    plan_from_schema,
    plan_missing_indicators,
    transform_cleaned,
)
from churnwise.validation.targets import detect_target

CONFIG = PreprocessConfig(seed=42)


def profile_of(frame: pd.DataFrame, *, title: str = "customer churn"):
    target = detect_target(frame, dataset_title=title)
    return build_profile(
        frame,
        dataset_id="fixture:columns",
        format="csv",
        source_path="fixture.csv",
        target=target,
    )


def base_frame(rows: int = 1000, **overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "tenure_months": [index % 48 + 1 for index in range(rows)],
            "monthly_charges": [20.0 + (index * 7 % 80) for index in range(rows)],
            "total_charges": [round((index % 48 + 1) * 35.0, 2) for index in range(rows)],
            "support_calls": [index % 8 for index in range(rows)],
            "usage_minutes": [100 + (index * 37 % 900) for index in range(rows)],
            "contract": ["yearly" if index % 3 == 0 else "monthly" for index in range(rows)],
            "paperless_billing": [bool(index % 2) for index in range(rows)],
            "churn": ["Yes" if index % 7 == 0 else "No" for index in range(rows)],
        }
    )
    for name, values in overrides.items():
        frame[name] = values
    return frame


def fit(frame: pd.DataFrame):
    profile = profile_of(frame)
    derived = derive_features(frame, profile, CONFIG)
    plan = plan_features(derived.frame, profile, CONFIG, derived.applied)
    cleaned = clean_frame(derived.frame, plan)
    plan_missing_indicators(cleaned, plan, CONFIG)
    cleaned = clean_frame(derived.frame, plan)
    preprocessor = build_preprocessor(plan, CONFIG)
    preprocessor.fit(cleaned)
    return profile, derived, plan, cleaned, preprocessor


def test_phase4_rejected_columns_never_enter_the_plan():
    frame = base_frame()
    profile = profile_of(frame)
    profile.roles.dropped.append({"column": "tenure_months", "reason": "fixture rejection"})

    derived = derive_features(frame, profile, CONFIG)
    plan = plan_features(derived.frame, profile, CONFIG, derived.applied)

    assert "tenure_months" not in plan.numeric
    assert "tenure_months" not in plan.categorical
    assert any(item["column"] == "tenure_months" for item in plan.dropped)


def test_numeric_strings_are_coerced_but_nominal_codes_are_not():
    frame = base_frame(
        addon_amount=[f"{10.5 + (index % 20)}" for index in range(1000)],
        area_code=[["408.0", "415.0", "510.0"][index % 3] for index in range(1000)],
    )
    profile, _, plan, cleaned, _ = fit(frame)

    assert "addon_amount" in plan.numeric
    assert "addon_amount" in plan.numeric_coerced
    assert "area_code" in plan.categorical
    assert "area_code" not in plan.numeric
    assert cleaned["addon_amount"].dtype == "float64"
    assert set(cleaned["area_code"].dropna()) == {"408.0", "415.0", "510.0"}
    assert any("addon_amount" in warning for warning in plan.warnings)


def test_median_imputation_and_missing_indicators_use_training_rows():
    rows = 1000
    frame = base_frame(rows)
    frame["paperless_billing"] = frame["paperless_billing"].astype(object)
    frame.loc[frame.index[:120], "monthly_charges"] = np.nan  # 12%
    frame.loc[frame.index[:40], "contract"] = np.nan  # 4%
    frame.loc[frame.index[:5], "paperless_billing"] = np.nan  # 0.5%
    profile, derived, plan, cleaned, preprocessor = fit(frame)

    assert plan.missing_indicators == ["monthly_charges"]
    assert "monthly_charges__missing" in preprocessor.feature_names_in_
    imputer = preprocessor.named_transformers_["numeric"].named_steps["imputer"]
    index = list(preprocessor.feature_names_in_).index("monthly_charges")
    expected = float(cleaned["monthly_charges"].median())
    assert imputer.statistics_[index] == pytest.approx(expected, rel=1e-12)

    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    contract_index = plan.categorical.index("contract")
    assert "__missing__" in encoder.categories_[contract_index]

    matrix = transform_cleaned(preprocessor, cleaned)
    assert np.isfinite(matrix).all()


def test_missing_indicator_tracks_cleaned_missing_values():
    frame = base_frame()
    frame.loc[frame.index[:50], "monthly_charges"] = np.inf  # 5% after cleaning
    profile, derived, plan, cleaned, preprocessor = fit(frame)

    assert "monthly_charges" in plan.missing_indicators
    assert cleaned["monthly_charges"].isna().sum() == 50
    assert cleaned["monthly_charges__missing"].sum() == 50
    assert np.isfinite(transform_cleaned(preprocessor, cleaned)).all()


def test_derived_source_that_reproduces_the_target_is_rejected():
    rows = 200
    frame = pd.DataFrame(
        {
            "tenure": ["Yes" if index % 2 else "No" for index in range(rows)],
            "monthly_charges": [20.0 + index % 50 for index in range(rows)],
            "churn": ["Yes" if index % 2 else "No" for index in range(rows)],
        }
    )
    target = detect_target(frame, dataset_title="customer churn")
    profile = SimpleNamespace(
        capabilities={"has_tenure": True},
        roles=SimpleNamespace(capability_sources={"has_tenure": "tenure"}),
        target=target,
    )

    derived = derive_features(frame, profile, CONFIG)

    applied = {item["name"] for item in derived.applied}
    assert "tenure_years" not in applied
    assert "tenure_band" not in applied
    assert any("deterministically reproduces" in item["reason"] for item in derived.skipped)


def test_columns_without_usable_values_after_cleaning_are_dropped():
    raw = pd.DataFrame({"tenure": [1.0, 2.0, 3.0, 4.0], "broken": [np.inf] * 4})
    plan = TypePlan(numeric=["tenure", "broken"])
    cleaned = clean_frame(raw, plan)

    removed = drop_unusable_columns(plan, cleaned)

    assert removed == ["broken"]
    assert plan.numeric == ["tenure"]
    assert any(item["column"] == "broken" for item in plan.dropped)


def test_numeric_quality_reports_unparseable_text():
    raw = pd.DataFrame({"charge": ["1.5", "oops", "2.5", "3.5"]})
    plan = TypePlan(numeric=["charge"])
    cleaned = clean_frame(raw, plan)
    quality = numeric_quality(raw, cleaned, plan.numeric)

    assert quality["charge"]["invalid"] == 1
    assert quality["charge"]["unparseable"] == 1
    assert quality["charge"]["infinite"] == 0
    assert cleaned["charge"].isna().sum() == 1


def test_numeric_quality_reports_invalid_and_infinite_values():
    frame = base_frame()
    frame.loc[frame.index[:50], "monthly_charges"] = np.inf
    frame.loc[frame.index[50:60], "monthly_charges"] = -np.inf
    profile, derived, plan, cleaned, _ = fit(frame)
    quality = numeric_quality(derived.frame, cleaned, plan.numeric)

    assert quality["monthly_charges"]["invalid"] == 60
    assert quality["monthly_charges"]["infinite"] == 60
    assert quality["monthly_charges"]["unparseable"] == 0
    assert quality["monthly_charges"]["pct"] == pytest.approx(0.06)


def test_categorical_imputation_modes_are_honoured():
    frame = base_frame()
    frame["paperless_billing"] = frame["paperless_billing"].astype(object)
    frame.loc[frame.index[:50], "contract"] = np.nan
    frame.loc[frame.index[:50], "paperless_billing"] = np.nan
    profile = profile_of(frame)
    derived = derive_features(frame, profile, CONFIG)
    plan = plan_features(derived.frame, profile, CONFIG, derived.applied)
    cleaned = clean_frame(derived.frame, plan)

    missing_level = build_preprocessor(plan, CONFIG)
    missing_level.fit(cleaned)
    encoder = missing_level.named_transformers_["categorical"].named_steps["encoder"]
    assert "__missing__" in encoder.categories_[plan.categorical.index("contract")]

    frequent = PreprocessConfig(seed=42, categorical_imputation="most_frequent")
    frequent_pipeline = build_preprocessor(plan, frequent)
    frequent_pipeline.fit(cleaned)
    encoder = frequent_pipeline.named_transformers_["categorical"].named_steps["encoder"]
    contract = encoder.categories_[plan.categorical.index("contract")]
    assert "__missing__" not in contract
    assert {"monthly", "yearly"} <= set(contract)
    assert np.isfinite(transform_cleaned(frequent_pipeline, cleaned)).all()


def test_constant_and_near_constant_columns_stay_finite():
    frame = base_frame(
        constant_value=[7.0] * 1000,
        near_constant_plan=["basic" if index else "premium" for index in range(1000)],
    )
    profile, derived, plan, cleaned, preprocessor = fit(frame)

    assert "constant_value" not in plan.numeric
    assert "near_constant_plan" in plan.categorical
    matrix = transform_cleaned(preprocessor, cleaned)
    assert np.isfinite(matrix).all()

    # A column that is constant in the training partition must still transform.
    train = frame.iloc[:600].copy()
    train["near_constant_plan"] = "basic"
    plan2 = plan_features(train, profile, CONFIG, derived.applied)
    cleaned2 = clean_frame(train, plan2)
    plan_missing_indicators(cleaned2, plan2, CONFIG)
    cleaned2 = clean_frame(train, plan2)
    preprocessor2 = build_preprocessor(plan2, CONFIG)
    preprocessor2.fit(cleaned2)
    assert np.isfinite(transform_cleaned(preprocessor2, cleaned2)).all()


def test_unseen_categories_do_not_raise_and_are_ignored():
    frame = base_frame()
    profile, derived, plan, cleaned, preprocessor = fit(frame)
    unseen = derived.frame.iloc[:2].copy()
    unseen.loc[unseen.index[0], "contract"] = "quarterly"

    matrix = transform_cleaned(preprocessor, clean_frame(unseen, plan))
    assert np.isfinite(matrix).all()
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    contract_index = plan.categorical.index("contract")
    assert "quarterly" not in encoder.categories_[contract_index]


def test_high_cardinality_is_dropped_with_a_reason():
    frame = base_frame(state=[f"state_{index % 20}" for index in range(1000)])
    profile = profile_of(frame)
    assert "state" in profile.roles.categorical
    plan = plan_features(frame, profile, PreprocessConfig(seed=42, max_categorical_levels=10), [])
    assert "state" not in plan.categorical
    assert any(item["column"] == "state" for item in plan.dropped)


def test_feature_names_order_and_source_mapping_are_stable():
    frame = base_frame()
    profile, derived, plan, cleaned, preprocessor = fit(frame)
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    outputs = describe_outputs(names, plan)

    assert [item["name"] for item in outputs] == names
    assert len(outputs) == preprocessor.transform(cleaned).shape[1]
    assert all(item["source"] for item in outputs)
    assert "tenure_months" in {item["source"] for item in outputs}
    assert "contract" in {item["source"] for item in outputs}

    second = build_preprocessor(plan, CONFIG)
    second.fit(cleaned)
    assert [str(name) for name in second.get_feature_names_out()] == names
    assert np.allclose(preprocessor.transform(cleaned), second.transform(cleaned))


def test_schema_plan_round_trip_reproduces_the_matrix():
    frame = base_frame()
    _, derived, plan, cleaned, preprocessor = fit(frame)
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    schema = {
        "preprocessing": {
            "numeric": plan.numeric,
            "boolean": plan.boolean,
            "categorical": plan.categorical,
            "numeric_coerced": plan.numeric_coerced,
            "missing_indicators": plan.missing_indicators,
        },
        "derived": derived.applied,
        "feature_names": names,
    }
    restored = plan_from_schema(schema)
    assert np.array_equal(
        transform_cleaned(preprocessor, clean_frame(derived.frame, restored)),
        transform_cleaned(preprocessor, cleaned),
    )


def test_derived_features_are_capability_guarded_and_reproducible():
    frame = base_frame()
    profile = profile_of(frame)
    derived = derive_features(frame, profile, CONFIG)
    names = {item["name"] for item in derived.applied}
    assert {
        "tenure_years",
        "tenure_band",
        "avg_monthly_charge",
        "support_calls_per_year",
        "usage_band",
        "support_calls_per_usage_hour",
    } <= names
    assert derived.frame["tenure_years"].iloc[0] == pytest.approx(
        frame["tenure_months"].iloc[0] / 12
    )

    replayed = apply_derived(frame, derived.applied)
    for name in names:
        pd.testing.assert_series_equal(
            replayed[name], derived.frame[name], check_names=False
        )

    sparse = frame.drop(columns=["tenure_months"])
    sparse_profile = profile_of(sparse)
    sparse_derived = derive_features(sparse, sparse_profile, CONFIG)
    assert "tenure_years" not in {item["name"] for item in sparse_derived.applied}
    assert any(
        item["feature"] == "tenure_years" for item in sparse_derived.skipped
    )

    disabled = derive_features(frame, profile, PreprocessConfig(seed=42, derived_features=False))
    assert disabled.applied == []
    assert all("disabled" in item["reason"] for item in disabled.skipped)


def test_configuration_validation_catches_impossible_splits():
    problems = PreprocessConfig(train_fraction=0.9, val_fraction=0.2).validate()
    assert any("sum to 1.0" in problem for problem in problems)
    problems = PreprocessConfig(imbalance="smote").validate()
    assert any("imbalance" in problem for problem in problems)
