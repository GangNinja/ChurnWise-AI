"""Retention-simulator form defaults must preserve valid zero assumptions."""

from __future__ import annotations

from churnwise.dashboard.pages.simulator import _defaults


def test_defaults_preserve_zero_values_and_only_fill_missing_ones():
    payload = {
        "budget": 0.0,
        "assumptions": {
            "cost_per_contact": 0.0,
            "effectiveness": 0.0,
            "monthly_value": 0.0,
            "horizon_months": 1.0,
            "value_unit": "USD",
        },
        "targeting": {
            "strategy": "highest_risk",
            "min_risk_probability": 0.0,
            "risk_tiers": [],
            "max_contacts": None,
        },
    }

    defaults = _defaults(payload)

    assert defaults["budget"] == 0.0
    assert defaults["cost_per_contact"] == 0.0
    assert defaults["effectiveness"] == 0.0
    assert defaults["monthly_value"] == 0.0
    assert defaults["min_risk_probability"] == 0.0
    assert defaults["horizon_months"] == 1.0
    assert defaults["value_unit"] == "USD"


def test_defaults_fill_absent_values_without_masking_zero():
    defaults = _defaults({"assumptions": {}, "targeting": {}})

    assert defaults["budget"] == 1000.0
    assert defaults["cost_per_contact"] == 5.0
    assert defaults["effectiveness"] == 0.30
    assert defaults["monthly_value"] is None
    assert defaults["min_risk_probability"] is None
