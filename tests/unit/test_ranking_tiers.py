"""Phase 9 tier, ranking, filter and value-at-risk unit tests."""

from __future__ import annotations

import math

import pytest

from churnwise.intelligence import (
    RiskAssessment,
    RiskConfig,
    RiskError,
    RiskFilter,
    RiskThresholds,
    attach_value_at_risk,
    filter_records,
    rank_records,
    sanitize_csv_value,
    value_at_risk,
)


def _assessment(
    reference: str,
    probability: float,
    *,
    tier: str | None = None,
    position: int = 0,
    segment: dict | None = None,
    value: float | None = None,
) -> RiskAssessment:
    thresholds = RiskThresholds.from_value((0.30, 0.50, 0.70))
    item = RiskAssessment(
        record_reference=reference,
        input_position=position,
        churn_probability=probability,
        predicted_class=int(probability >= 0.45),
        risk_tier=tier or thresholds.tier_for(probability),
        classification_threshold=0.45,
        risk_thresholds=thresholds,
        model_family="random_forest",
        model_run_id="fixture/random_forest",
        model_sha256="a" * 64,
        preprocessing_run_id="fixture-preprocess",
        assessed_at="2026-10-22T00:00:00Z",
        positive_label="Yes",
        segment=segment or {},
        value_at_risk=value,
    )
    return item


def test_tier_boundaries_are_lower_inclusive() -> None:
    thresholds = RiskThresholds.from_value((0.30, 0.50, 0.70))
    cases = {
        0.0: "low",
        0.2999: "low",
        0.30: "medium",
        0.4999: "medium",
        0.50: "high",
        0.6999: "high",
        0.70: "critical",
        1.0: "critical",
    }
    assert {probability: thresholds.tier_for(probability) for probability in cases} == cases


def test_two_boundary_configuration_has_three_tiers() -> None:
    thresholds = RiskThresholds.from_value((0.4, 0.8))
    assert thresholds.labels == ("low", "medium", "high")
    assert thresholds.tier_for(0.4) == "medium"
    assert thresholds.tier_for(0.8) == "high"


@pytest.mark.parametrize(
    "value",
    [
        (0.5,),
        (0.5, 0.4),
        (0.5, 0.5),
        (0.0, 0.5),
        (0.5, 1.0),
        (0.3, 0.5, 0.7, 0.9),
        (math.nan, 0.5),
    ],
)
def test_invalid_tier_configurations_are_rejected(value) -> None:
    with pytest.raises(RiskError) as raised:
        RiskThresholds.from_value(value)
    assert raised.value.code == "CW-RISK-001"


def test_risk_config_keeps_classification_and_tiers_separate() -> None:
    config = RiskConfig(
        thresholds=RiskThresholds.from_value((0.25, 0.60)),
        classification_threshold=0.45,
    )
    assert config.thresholds.labels == ("low", "medium", "high")
    assert config.classification_threshold == 0.45
    config.validate_or_raise()


def test_invalid_horizon_is_a_structured_configuration_error() -> None:
    config = RiskConfig(value_horizon_months="six")
    with pytest.raises(RiskError) as raised:
        config.validate_or_raise()
    assert raised.value.code == "CW-RISK-001"
    assert "value_horizon_months" in raised.value.message


def test_tier_counts_sum_to_population() -> None:
    records = [
        _assessment("a", 0.10),
        _assessment("b", 0.35),
        _assessment("c", 0.55),
        _assessment("d", 0.90),
    ]
    table = _table(records)
    counts = table.tier_counts()
    assert sum(value["count"] for value in counts.values()) == len(records)
    assert abs(sum(value["pct"] for value in counts.values()) - 1.0) < 1e-12


def _table(records):
    from churnwise.intelligence import RiskTable

    return RiskTable(
        records=records,
        thresholds=records[0].risk_thresholds,
        classification_threshold=0.45,
        max_export_rows=100,
    )


def test_ranking_is_deterministic_and_breaks_ties_by_reference() -> None:
    records = [
        _assessment("b", 0.8, position=1),
        _assessment("a", 0.8, position=2),
        _assessment("c", 0.2, position=3),
    ]
    ranked = rank_records(records)
    assert [item.record_reference for item in ranked] == ["a", "b", "c"]
    assert [item.rank for item in ranked] == [1, 2, 3]
    assert len({item.rank for item in ranked}) == len(ranked)


def test_value_at_risk_ordering_matches_formula() -> None:
    records = [
        _assessment("a", 0.2, position=0),
        _assessment("b", 0.5, position=1),
        _assessment("c", 0.4, position=2),
    ]
    values = [100.0, 100.0, 1000.0]
    assert attach_value_at_risk(records, values, horizon_months=6.0)
    ranked = rank_records(records, sort_by="value_at_risk")
    assert [item.record_reference for item in ranked] == ["c", "b", "a"]
    for item, value in zip(records, values):
        assert item.value_at_risk == pytest.approx(item.churn_probability * value * 6.0)


def test_missing_value_proxy_is_not_fabricated() -> None:
    records = [_assessment("a", 0.8), _assessment("b", 0.2)]
    assert not attach_value_at_risk(records, None, horizon_months=6.0)
    assert all(item.value_at_risk is None for item in records)
    assert value_at_risk(0.5, 100.0, 6.0) == pytest.approx(300.0)


def test_value_at_risk_requires_a_value_for_every_row() -> None:
    with pytest.raises(RiskError):
        _ = rank_records(
            [_assessment("a", 0.8), _assessment("b", 0.2)],
            sort_by="value_at_risk",
        )


def test_filters_return_exactly_matching_rows() -> None:
    records = [
        _assessment("a", 0.10, segment={"contract": "monthly"}),
        _assessment("b", 0.35, segment={"contract": "yearly"}),
        _assessment("c", 0.55, segment={"contract": "monthly"}),
        _assessment("d", 0.90, segment={"contract": "monthly"}),
    ]
    labels = records[0].risk_thresholds.labels
    high = filter_records(
        records,
        RiskFilter(tiers=("high", "critical")),
        labels=labels,
    )
    assert [item.record_reference for item in high] == ["c", "d"]
    monthly = filter_records(
        records,
        RiskFilter(segment_equals={"contract": ("monthly",)}),
        labels=labels,
    )
    assert [item.record_reference for item in monthly] == ["a", "c", "d"]
    bounded = filter_records(
        records,
        RiskFilter(min_probability=0.30, max_probability=0.70),
        labels=labels,
    )
    assert [item.record_reference for item in bounded] == ["b", "c"]


def test_invalid_filters_are_rejected() -> None:
    with pytest.raises(RiskError):
        filter_records(
            [_assessment("a", 0.5)],
            RiskFilter(tiers=("impossible",)),
            labels=("low", "medium", "high", "critical"),
        )
    with pytest.raises(RiskError):
        filter_records(
            [_assessment("a", 0.5)],
            RiskFilter(min_probability=0.8, max_probability=0.2),
            labels=("low", "medium", "high", "critical"),
        )
    with pytest.raises(RiskError):
        filter_records(
            [_assessment("a", 0.5)],
            RiskFilter(segment_min={"tenure": "not-a-number"}),
            labels=("low", "medium", "high", "critical"),
        )
    with pytest.raises(RiskError):
        filter_records(
            [_assessment("a", 0.5)],
            RiskFilter(segment_max={"tenure": math.inf}),
            labels=("low", "medium", "high", "critical"),
        )


def test_csv_formula_injection_is_neutralised() -> None:
    for value in (
        "=1+1",
        "+SUM(A1)",
        "-2",
        "@cmd",
        "  =1+1",
        "\t=1+1",
        "\tplain text",
        "\rplain text",
    ):
        assert sanitize_csv_value(value).startswith("'")
    assert sanitize_csv_value("normal") == "normal"
    assert sanitize_csv_value("normal\tvalue") == "normal\tvalue"
    assert sanitize_csv_value(0.5) == 0.5
