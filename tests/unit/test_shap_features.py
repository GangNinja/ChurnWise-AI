"""Feature mapping and grouped SHAP aggregation (Phase 8)."""

from __future__ import annotations

import numpy as np
import pytest

from churnwise.explain import (
    CODE_VALIDATION_FAILED,
    ExplainError,
    build_contributions,
    build_feature_groups,
    describe_mapping,
    group_matrix,
    ranked_importance,
)
from churnwise.explain.features import mapped_columns

NAMES = [
    "numeric__tenure",
    "categorical__contract_monthly",
    "categorical__contract_yearly",
    "categorical__region_a",
    "categorical__region_b",
]
OUTPUT_FEATURES = [
    {"name": "numeric__tenure", "source": "tenure", "role": "numeric"},
    {
        "name": "categorical__contract_monthly",
        "source": "contract",
        "role": "categorical",
        "value": "monthly",
    },
    {
        "name": "categorical__contract_yearly",
        "source": "contract",
        "role": "categorical",
        "value": "yearly",
    },
    {
        "name": "categorical__region_a",
        "source": "region",
        "role": "categorical",
        "value": "a",
    },
    {
        "name": "categorical__region_b",
        "source": "region",
        "role": "categorical",
        "value": "b",
    },
]


def test_groups_map_one_hot_columns_to_sources() -> None:
    groups = build_feature_groups(NAMES, OUTPUT_FEATURES)
    assert [group.source for group in groups] == ["contract", "region", "tenure"]
    contract = next(group for group in groups if group.source == "contract")
    assert contract.kind == "categorical"
    assert contract.columns == ["categorical__contract_monthly", "categorical__contract_yearly"]
    assert contract.labels == ["contract=monthly", "contract=yearly"]
    mapping = describe_mapping(groups)
    assert mapping["transformed_columns"] == len(NAMES)
    assert mapping["aggregation"] == "sum of transformed-column SHAP values per source feature"


def test_missing_mapping_uses_documented_fallback_label() -> None:
    groups = build_feature_groups(["numeric__a", "strange_column"], [])
    sources = {group.source for group in groups}
    assert sources == {"a", "strange_column"}
    fallback = next(group for group in groups if group.source == "strange_column")
    assert fallback.kind == "unknown"


def test_duplicate_or_empty_feature_names_are_rejected() -> None:
    with pytest.raises(ExplainError) as raised:
        build_feature_groups(["numeric__a", "numeric__a"], [])
    assert raised.value.code == CODE_VALIDATION_FAILED
    with pytest.raises(ExplainError):
        build_feature_groups([], [])


def test_group_matrix_sums_columns_within_source() -> None:
    groups = build_feature_groups(NAMES, OUTPUT_FEATURES)
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [0.5, -0.5, 1.0, -1.0, 2.0]])
    grouped, sources = group_matrix(values, groups)
    assert sources == ["contract", "region", "tenure"]
    assert grouped[0, 0] == pytest.approx(5.0)
    assert grouped[0, 1] == pytest.approx(9.0)
    assert grouped[1, 0] == pytest.approx(0.5)
    assert grouped.sum(axis=1) == pytest.approx(values.sum(axis=1))


def test_mapped_columns_preserve_model_column_order() -> None:
    groups = build_feature_groups(NAMES, OUTPUT_FEATURES)
    assert mapped_columns(NAMES, groups) == [
        "tenure",
        "contract=monthly",
        "contract=yearly",
        "region=a",
        "region=b",
    ]


def test_group_matrix_rejects_wrong_width() -> None:
    groups = build_feature_groups(NAMES, OUTPUT_FEATURES)
    with pytest.raises(ExplainError) as raised:
        group_matrix(np.zeros((2, 3)), groups)
    assert raised.value.code == CODE_VALIDATION_FAILED


def test_ranked_importance_orders_by_mean_absolute_value() -> None:
    groups = build_feature_groups(
        ["numeric__a", "numeric__b"], [
            {"name": "numeric__a", "source": "a", "role": "numeric"},
            {"name": "numeric__b", "source": "b", "role": "numeric"},
        ]
    )
    values = np.array([[1.0, -0.1], [2.0, 0.2], [-1.5, 0.0]])
    rows = ranked_importance(values, groups, base_value=0.0)
    assert [row["feature"] for row in rows] == ["a", "b"]
    assert rows[0]["rank"] == 1
    assert rows[0]["mean_abs_shap"] == pytest.approx(1.5)
    assert rows[0]["share"] == pytest.approx(1.5 / 1.6)
    assert rows[0]["positive_share"] == pytest.approx(2 / 3)
    assert rows[0]["direction"] == "increases_churn"
    assert sum(row["share"] for row in rows) == pytest.approx(1.0)


def test_ranked_importance_marks_zero_contribution_neutral() -> None:
    groups = build_feature_groups(
        ["numeric__constant"],
        [{"name": "numeric__constant", "source": "constant", "role": "numeric"}],
    )
    rows = ranked_importance(np.zeros((3, 1)), groups, base_value=0.0)
    assert rows[0]["direction"] == "neutral"
    assert rows[0]["share"] == 0.0


def test_contributions_respect_tolerance_and_top_k() -> None:
    groups = build_feature_groups(NAMES, OUTPUT_FEATURES)
    raw = np.array([0.4, 0.01, -0.2, 0.05, 0.03])
    matrix_row = np.array([3.0, 1.0, 0.0, 1.0, 0.0])
    ordered, positives, negatives, detail = build_contributions(
        raw, groups, matrix_row, tolerance=1e-6, top_k=2
    )
    assert len(ordered) == 3
    assert len(positives) == 2
    assert len(negatives) == 1
    assert positives[0].contribution == pytest.approx(0.4)
    assert positives[0].feature == "tenure"
    assert negatives[0].feature == "contract"
    assert negatives[0].contribution == pytest.approx(-0.19)
    assert detail and detail[0]["rule"] == "sum"
    assert all(item.direction in {"increases_churn", "decreases_churn", "neutral"} for item in ordered)


def test_contributions_near_zero_are_neutral() -> None:
    groups = build_feature_groups(["numeric__a"], [{"name": "numeric__a", "source": "a", "role": "numeric"}])
    ordered, positives, negatives, _ = build_contributions(
        np.array([1e-9]), groups, np.array([1.0]), tolerance=1e-4, top_k=5
    )
    assert ordered[0].direction == "neutral"
    assert positives == [] and negatives == []
