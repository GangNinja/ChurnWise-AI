"""Leakage checklist L1-L8 with train-only fitting assertions (T5.2, T5.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnwise.eda.profile import build_profile
from churnwise.preprocessing.contracts import PreprocessConfig
from churnwise.preprocessing.features import derive_features
from churnwise.preprocessing.pipeline import (
    build_preprocessor,
    clean_frame,
    plan_features,
    plan_missing_indicators,
)
from churnwise.preprocessing.split import make_split
from churnwise.validation.targets import detect_target

CONFIG = PreprocessConfig(seed=42)


def make_frame(rows: int = 1500) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "customerID": [f"C{index:06d}" for index in range(rows)],
            "tenure_months": [index % 48 + 1 for index in range(rows)],
            "monthly_charges": [20.0 + (index * 7 % 80) for index in range(rows)],
            "total_charges": [round((index % 48 + 1) * 35.0, 2) for index in range(rows)],
            "contract": ["yearly" if index % 3 == 0 else "monthly" for index in range(rows)],
            "support_calls": [index % 8 for index in range(rows)],
            "paperless_billing": [bool(index % 2) for index in range(rows)],
            "churn_reason": [
                "price" if index % 7 == 0 else "none" for index in range(rows)
            ],
            "churn": ["Yes" if index % 7 == 0 else "No" for index in range(rows)],
        }
    )
    return frame


def prepared_frames(*, mutate_holdout: bool = False):
    frame = make_frame()
    target = detect_target(frame, dataset_title="customer churn")
    profile = build_profile(
        frame,
        dataset_id="fixture:leakage",
        format="csv",
        source_path="fixture.csv",
        target=target,
    )
    normalised = frame.copy()
    normalised["churn"] = (frame["churn"] == "Yes").astype(int)
    derived = derive_features(normalised, profile, CONFIG)
    split = make_split(
        derived.frame,
        dataset_id="fixture:leakage",
        dataset_sha256="c" * 64,
        run_id="run-leakage",
        target_column="churn",
        id_column="customerID",
        config=CONFIG,
    )
    train = derived.frame.iloc[split.train_indices].reset_index(drop=True)
    val = derived.frame.iloc[split.val_indices].reset_index(drop=True)
    test = derived.frame.iloc[split.test_indices].reset_index(drop=True)
    if mutate_holdout:
        val.loc[val.index[:50], "tenure_months"] = 1e9
        test.loc[test.index[:50], "monthly_charges"] = -1e9
        val.loc[val.index[0], "contract"] = "unseen_option"
    plan = plan_features(train, profile, CONFIG, derived.applied)
    cleaned_train = clean_frame(train, plan)
    plan_missing_indicators(
        cleaned_train, plan, CONFIG, derived_names={spec["name"] for spec in derived.applied}
    )
    cleaned_train = clean_frame(train, plan)
    preprocessor = build_preprocessor(plan, CONFIG)
    preprocessor.fit(cleaned_train)
    return {
        "frame": frame,
        "target": target,
        "profile": profile,
        "derived": derived,
        "split": split,
        "train": train,
        "val": val,
        "test": test,
        "plan": plan,
        "cleaned_train": cleaned_train,
        "preprocessor": preprocessor,
    }


def numeric_scaler(preprocessor):
    return preprocessor.named_transformers_["numeric"].named_steps["scaler"]


def numeric_imputer(preprocessor):
    return preprocessor.named_transformers_["numeric"].named_steps["imputer"]


def column_index(preprocessor, column: str) -> int:
    return list(preprocessor.feature_names_in_).index(column)


def test_l1_identifier_is_excluded_everywhere():
    state = prepared_frames()
    outputs = state["preprocessor"].get_feature_names_out().tolist()
    assert all("customerID" not in name for name in outputs)
    features = state["profile"].roles.features
    assert "customerID" not in features


def test_l2_target_and_leakage_columns_are_excluded():
    state = prepared_frames()
    excluded = {item["column"] for item in state["profile"].roles.dropped}
    assert "churn_reason" in excluded
    assert "churn_reason" not in state["plan"].feature_columns
    assert "churn" not in state["plan"].feature_columns


def test_l3_l4_split_precedes_fitting_and_is_disjoint():
    state = prepared_frames()
    split = state["split"]
    assert split.spec.constraints["test_touched"] is False
    assert split.spec.constraints["fitted_on"] == "train_only"
    assert not set(split.train_indices) & set(split.val_indices)
    assert not set(split.train_indices) & set(split.test_indices)
    assert not set(split.val_indices) & set(split.test_indices)
    assert len(split.train_indices) + len(split.val_indices) + len(split.test_indices) == len(
        state["derived"].frame
    )


def test_l5_no_resampling_by_default():
    state = prepared_frames()
    assert state["plan"].missing_indicators == []
    assert len(state["train"]) == state["split"].spec.sizes["train"]


def test_scaler_and_imputer_are_fitted_on_train_only():
    plain = prepared_frames()
    mutated = prepared_frames(mutate_holdout=True)

    scaler = numeric_scaler(plain["preprocessor"])
    index = column_index(plain["preprocessor"], "tenure_months")
    train_mean = float(plain["cleaned_train"]["tenure_months"].mean())
    assert scaler.mean_[index] == pytest.approx(train_mean, rel=1e-12)

    mutated_scaler = numeric_scaler(mutated["preprocessor"])
    assert np.allclose(scaler.mean_, mutated_scaler.mean_, rtol=0, atol=0)

    imputer = numeric_imputer(plain["preprocessor"])
    assert imputer.statistics_[index] == pytest.approx(
        float(plain["cleaned_train"]["tenure_months"].median()), rel=1e-12
    )


def test_encoder_categories_come_from_train_only():
    plain = prepared_frames()
    mutated = prepared_frames(mutate_holdout=True)
    plain_encoder = plain["preprocessor"].named_transformers_["categorical"].named_steps[
        "encoder"
    ]
    mutated_encoder = mutated["preprocessor"].named_transformers_["categorical"].named_steps[
        "encoder"
    ]
    contract_index = plain["plan"].categorical.index("contract")
    assert list(plain_encoder.categories_[contract_index]) == ["monthly", "yearly"]
    assert list(mutated_encoder.categories_[contract_index]) == ["monthly", "yearly"]
    assert "unseen_option" not in mutated_encoder.categories_[contract_index]


def test_missing_indicators_use_training_missingness_only():
    state = prepared_frames()
    # The fixture has no missing values in training, so no indicator may exist,
    # even though the mutation injects extreme (not missing) values later.
    assert state["plan"].missing_indicators == []


def test_l6_l7_l8_no_model_selection_threshold_or_feature_selection():
    state = prepared_frames()
    preprocessor = state["preprocessor"]
    names = [name for name, _, _ in preprocessor.transformers]
    assert names == ["numeric", "boolean", "categorical"]
    assert preprocessor.remainder == "drop"
    for transformer in preprocessor.transformers:
        steps = dict(transformer[1].named_steps)
        assert "select" not in steps
    # Phase 5 produces no threshold or model artifact.
    assert not hasattr(preprocessor, "threshold_")
    assert not hasattr(preprocessor, "best_params_")
