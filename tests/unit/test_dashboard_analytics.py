"""Pure presentation helpers: tier boundaries, formatting and export safety."""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from churnwise.dashboard import analytics


def test_tier_for_uses_lower_inclusive_upper_exclusive_boundaries():
    boundaries = (0.3, 0.6, 0.8)
    labels = analytics.tier_labels(boundaries)
    assert labels == ("low", "medium", "high", "critical")
    assert analytics.tier_for(0.0, boundaries) == "low"
    assert analytics.tier_for(0.2999, boundaries) == "low"
    assert analytics.tier_for(0.3, boundaries) == "medium"
    assert analytics.tier_for(0.6, boundaries) == "high"
    assert analytics.tier_for(0.8, boundaries) == "critical"
    assert analytics.tier_for(1.0, boundaries) == "critical"


def test_tier_labels_support_two_boundaries_only():
    assert analytics.tier_labels((0.4, 0.7)) == ("low", "medium", "high")
    assert analytics.tier_labels((0.4,)) == ()


@pytest.mark.parametrize(
    "boundaries, expected",
    [
        ((0.2, 0.5), None),
        ((0.2, 0.5, 0.8), None),
        ((0.5,), "exactly two or three"),
        ((0.5, 0.5), "strictly increasing"),
        ((0.0, 0.5), "between 0 and 1"),
        ((0.5, 1.0), "between 0 and 1"),
        ((0.8, 0.2), "strictly increasing"),
    ],
)
def test_validate_boundaries(boundaries, expected):
    problem = analytics.validate_boundaries(boundaries)
    if expected is None:
        assert problem is None
    else:
        assert problem is not None and expected in problem


def test_apply_tiers_replaces_stored_labels():
    frame = pd.DataFrame(
        {
            "churn_probability": [0.1, 0.45, 0.65, 0.9],
            "record_reference": list("abcd"),
            "rank": [1, 2, 3, 4],
            "risk_tier": ["critical"] * 4,
        }
    )
    result = analytics.apply_tiers(frame, (0.3, 0.6, 0.8))
    assert list(result["risk_tier"]) == ["low", "medium", "high", "critical"]
    assert list(frame["risk_tier"]) == ["critical"] * 4


def test_filter_risk_frame_combines_tier_probability_and_reference():
    frame = pd.DataFrame(
        {
            "churn_probability": [0.1, 0.45, 0.65, 0.9],
            "record_reference": ["r1", "r2", "r3", "r4"],
            "rank": [4, 2, 3, 1],
            "risk_tier": ["low", "medium", "high", "critical"],
        }
    )
    assert list(analytics.filter_risk_frame(frame, tiers=("high", "critical"))["rank"]) == [1, 3]
    assert list(analytics.filter_risk_frame(frame, min_probability=0.5)["rank"]) == [1, 3]
    assert list(analytics.filter_risk_frame(frame, reference_query=" R2 ")["rank"]) == [2]


def test_formatting_handles_missing_values_and_units():
    assert analytics.format_probability(0.123456) == "0.1235"
    assert analytics.format_probability(None) == "not available"
    assert analytics.format_percent(0.5) == "50.0%"
    assert analytics.format_count(1234) == "1,234"
    assert analytics.format_money(12.5, "USD") == "12.50 USD"
    assert analytics.format_money(None) == "undefined"


def test_csv_export_neutralises_formulas_and_round_trips_json():
    frame = pd.DataFrame(
        {
            "record_reference": ["=cmd()", "+1", "-1", "@x", "safe"],
            "churn_probability": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )
    payload = analytics.frame_to_csv_bytes(frame)
    parsed = pd.read_csv(io.BytesIO(payload))
    assert list(parsed["record_reference"]) == ["'=cmd()", "'+1", "'-1", "'@x", "safe"]
    json_frame = pd.read_json(io.BytesIO(analytics.frame_to_json_bytes(frame)))
    assert list(json_frame["record_reference"]) == list(frame["record_reference"])


def test_nested_get_never_raises_on_missing_or_wrong_types():
    payload = {"a": {"b": [1, 2]}}
    assert analytics.nested_get(payload, "a", "b") == [1, 2]
    assert analytics.nested_get(payload, "a", "x", default="fallback") == "fallback"
    assert analytics.nested_get(None, "a", default=0) == 0
    assert analytics.nested_get({"a": 3}, "a", "b", default="x") == "x"


def test_dict_to_json_bytes_serialises_non_native_values():
    payload = {"threshold": 0.5, "path": __import__("pathlib").Path("artifacts/x")}
    decoded = json.loads(analytics.dict_to_json_bytes(payload))
    assert decoded["threshold"] == 0.5
    assert "artifacts" in decoded["path"]
