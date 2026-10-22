"""Customer-level explanations and contribution semantics (Phase 8)."""

from __future__ import annotations

import numpy as np
import pytest

from churnwise.explain import (
    CODE_VALIDATION_FAILED,
    ExplainConfig,
    ExplainError,
    build_explainer,
    build_feature_groups,
    chunked_explain,
    local_explanations,
    local_summary,
    matrix_for_positions,
)

NAMES = ["numeric__a", "numeric__b", "categorical__plan_0", "categorical__plan_1"]
OUTPUT_FEATURES = [
    {"name": "numeric__a", "source": "a", "role": "numeric"},
    {"name": "numeric__b", "source": "b", "role": "numeric"},
    {"name": "categorical__plan_0", "source": "plan", "role": "categorical", "value": 0},
    {"name": "categorical__plan_1", "source": "plan", "role": "categorical", "value": 1},
]


@pytest.fixture(scope="module")
def local_setup():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(5)
    features = rng.normal(size=(120, 4))
    labels = (0.9 * features[:, 0] - 0.4 * features[:, 1] + 0.3 * features[:, 2] > 0).astype(int)
    model = LogisticRegression(max_iter=500).fit(features, labels)
    config = ExplainConfig(sample_size=10, background_size=20, top_k=2)
    explainer = build_explainer(
        "logistic_regression",
        model,
        lambda matrix: model.predict_proba(matrix)[:, 1],
        background=features[:20],
        feature_names=NAMES,
        config=config,
    )
    explanation = chunked_explain(explainer, features[40:50], config=config)
    groups = build_feature_groups(NAMES, OUTPUT_FEATURES)
    blocks = local_explanations(
        family="logistic_regression",
        explainer=explainer,
        explanation=explanation,
        matrix=features[40:50],
        groups=groups,
        threshold=0.5,
        positive_label="Yes",
        top_k=2,
        tolerance=config.tolerance,
        row_positions=list(range(10)),
    )
    return model, explanation, blocks


def test_local_probability_matches_model(local_setup) -> None:
    model, explanation, blocks = local_setup
    for item, probability in zip(blocks, explanation.predictions):
        assert item.probability == pytest.approx(float(probability))
        assert item.predicted_class == int(probability >= 0.5)
        assert item.threshold == 0.5
        assert item.output_space == "log_odds"
        assert item.additivity_error <= 1e-8


def test_local_directions_and_top_k(local_setup) -> None:
    _, explanation, blocks = local_setup
    item = blocks[0]
    assert len(item.contributions) == 3
    assert len(item.top_positive) <= 2
    assert len(item.top_negative) <= 2
    assert all(contribution.contribution > 0 for contribution in item.top_positive)
    assert all(contribution.contribution < 0 for contribution in item.top_negative)
    assert all(
        contribution.direction == "increases_churn"
        for contribution in item.top_positive
    )
    reconstructed = item.base_value + sum(
        contribution.contribution for contribution in item.contributions
    )
    assert reconstructed == pytest.approx(float(explanation.predictions_raw[0]), abs=1e-8)
    assert item.model_output == pytest.approx(float(explanation.predictions_raw[0]))


def test_summary_uses_measured_values_and_disclaims_causality(local_setup) -> None:
    _, _, blocks = local_setup
    text = blocks[0].summary
    assert "log-odds" in text
    assert "P(churn = 1) =" in text
    assert "not causal" in text
    assert "threshold 0.50" in text
    assert "SHAP attributions" in text
    if blocks[0].top_positive:
        assert blocks[0].top_positive[0].feature in text


def test_summary_handles_no_contributors() -> None:
    text = local_summary(
        probability=0.5,
        predicted_class=0,
        threshold=0.5,
        positive_label="Yes",
        output_space="probability",
        model_output=0.5,
        top_positive=[],
        top_negative=[],
    )
    assert "No feature contributed" in text
    assert "not causal" in text


def test_matrix_for_positions_validates_bounds() -> None:
    matrix = np.zeros((4, 2))
    assert matrix_for_positions(matrix, [3, 0]).shape == (2, 2)
    with pytest.raises(ExplainError) as raised:
        matrix_for_positions(matrix, [9])
    assert raised.value.code == CODE_VALIDATION_FAILED
    with pytest.raises(ExplainError):
        matrix_for_positions(matrix, [])
