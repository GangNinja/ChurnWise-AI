"""Phase 10 hand-computed campaign math, allocation and configuration tests.

Every expected number below is calculated by hand from the fixture
probabilities and stated assumptions; none is copied from engine output.
"""

from __future__ import annotations

import json
import math

import pytest

from churnwise.intelligence import RiskAssessment, RiskThresholds
from churnwise.simulation import (
    CampaignAssumptions,
    CampaignScenario,
    RiskContext,
    SimulationError,
    TargetingConfig,
    assume_scenario,
    compare_scenarios,
    project_outcomes,
    run_campaign,
    run_sensitivity,
)
from churnwise.simulation.campaign import allocate_budget, resolve_eligibility

THRESHOLDS = RiskThresholds.from_value((0.30, 0.50, 0.70))


def _assessment(
    reference: str,
    probability: float,
    *,
    position: int = 0,
    segment: dict | None = None,
) -> RiskAssessment:
    return RiskAssessment(
        record_reference=reference,
        input_position=position,
        churn_probability=probability,
        predicted_class=int(probability >= 0.45),
        risk_tier=THRESHOLDS.tier_for(probability),
        classification_threshold=0.45,
        risk_thresholds=THRESHOLDS,
        model_family="random_forest",
        model_run_id="fixture/random_forest",
        model_sha256="a" * 64,
        preprocessing_run_id="fixture-preprocess",
        assessed_at="2026-10-22T00:00:00Z",
        positive_label="Yes",
        segment=segment or {},
    )


def _population() -> list[RiskAssessment]:
    return [
        _assessment("c1", 0.90, position=0),
        _assessment("c2", 0.60, position=1),
        _assessment("c3", 0.50, position=2),
        _assessment("c4", 0.30, position=3),
        _assessment("c5", 0.20, position=4),
    ]


def _selected_references(result) -> list[str]:
    return [item.record_reference for item in result.selected]


# ---------------------------------------------------------------------------
# three hand-computed scenarios
# ---------------------------------------------------------------------------


def test_scenario_a_budget_caps_selection_and_projection_is_exact() -> None:
    scenario = assume_scenario(
        "A",
        budget=15.0,
        cost_per_contact=5.0,
        effectiveness=0.5,
        monthly_value=10.0,
        horizon_months=6.0,
    )
    result = run_campaign(_population(), scenario)

    assert _selected_references(result) == ["c1", "c2", "c3"]
    assert result.allocation.max_affordable == 3
    assert result.allocation.allocated_spend == pytest.approx(15.0)
    assert result.allocation.remaining_budget == pytest.approx(0.0)
    assert result.allocation.utilization == pytest.approx(1.0)
    assert result.allocation.unaffordable_count == 2
    assert result.exclusion_counts["budget_exhausted"] == 2
    assert result.exclusion_counts["contact_limit"] == 0

    projection = result.projection
    assert projection.aggregate_churn_probability == pytest.approx(2.0)
    assert projection.expected_prevented_churn == pytest.approx(1.0)
    assert projection.assumed_average_value == pytest.approx(60.0)
    assert projection.expected_saved_value == pytest.approx(60.0)
    assert projection.net_benefit == pytest.approx(45.0)
    assert projection.roi == pytest.approx(3.0)
    assert projection.cost_per_expected_prevented_churn == pytest.approx(15.0)
    assert projection.break_even_effectiveness == pytest.approx(0.125)
    # mean(0.9, 0.6, 0.5, 0.3, 0.2) = 0.5; 0.5 * 3 * 0.5 = 0.75
    assert projection.random_contact_expected_prevented_churn == pytest.approx(0.75)
    assert projection.targeting_lift == pytest.approx(4.0 / 3.0)
    assert result.counts_reconcile
    assert any(
        "left untargeted because the remaining budget" in warning
        for warning in result.warnings
    )


def test_scenario_b_full_population_and_missing_value_metrics() -> None:
    scenario = assume_scenario(
        "B",
        budget=1000.0,
        cost_per_contact=7.5,
        effectiveness=1.0,
        monthly_value=None,
    )
    result = run_campaign(_population(), scenario)

    assert _selected_references(result) == ["c1", "c2", "c3", "c4", "c5"]
    assert result.allocation.max_affordable == 133  # floor(1000 / 7.5)
    assert result.allocation.allocated_spend == pytest.approx(37.5)
    assert result.allocation.remaining_budget == pytest.approx(962.5)
    assert result.allocation.utilization == pytest.approx(0.0375)

    projection = result.projection
    assert projection.aggregate_churn_probability == pytest.approx(2.5)
    assert projection.expected_prevented_churn == pytest.approx(2.5)
    assert projection.cost_per_expected_prevented_churn == pytest.approx(15.0)
    assert projection.expected_saved_value is None
    assert projection.net_benefit is None
    assert projection.roi is None
    assert projection.break_even_effectiveness is None
    assert "roi" in projection.undefined_metrics
    assert "expected_saved_value" in projection.undefined_metrics
    assert "no per-customer value assumption" in projection.undefined_metrics["roi"]


def test_scenario_c_ties_break_on_reference_and_cost_efficiency_defined() -> None:
    rows = [
        _assessment("b", 0.4),
        _assessment("a", 0.4),
        _assessment("c", 0.1),
    ]
    scenario = assume_scenario(
        "C", budget=4.0, cost_per_contact=4.0, effectiveness=0.25
    )
    result = run_campaign(rows, scenario)

    # Equal 0.4 probabilities: "a" sorts before "b", so budget selects "a".
    assert _selected_references(result) == ["a"]
    assert result.allocation.unaffordable_count == 2
    assert result.projection.aggregate_churn_probability == pytest.approx(0.4)
    assert result.projection.expected_prevented_churn == pytest.approx(0.1)
    # 4.0 / 0.1 = 40, computed without any customer-value assumption.
    assert result.projection.cost_per_expected_prevented_churn == pytest.approx(40.0)
    assert result.projection.expected_saved_value is None


def test_selection_is_deterministic_across_runs() -> None:
    scenario = assume_scenario(
        "Stable", budget=20.0, cost_per_contact=5.0, effectiveness=0.4
    )
    first = run_campaign(_population(), scenario)
    second = run_campaign(list(reversed(_population())), scenario)
    assert _selected_references(first) == _selected_references(second)
    assert first.allocation.to_dict() == second.allocation.to_dict()
    assert first.projection.to_dict() == second.projection.to_dict()


# ---------------------------------------------------------------------------
# targeting strategies
# ---------------------------------------------------------------------------


def test_risk_threshold_boundary_is_lower_inclusive() -> None:
    rows = [
        _assessment("at", 0.30),
        _assessment("just-below", 0.2999),
        _assessment("above", 0.55),
    ]
    scenario = assume_scenario(
        "Threshold",
        budget=100.0,
        cost_per_contact=1.0,
        strategy="risk_threshold",
        min_risk_probability=0.30,
    )
    result = run_campaign(rows, scenario)
    assert _selected_references(result) == ["above", "at"]
    assert result.exclusion_counts["below_risk_threshold"] == 1
    assert [item.reason for item in result.exclusions] == ["below_risk_threshold"]


def test_risk_category_strategy_only_targets_configured_tiers() -> None:
    rows = [
        _assessment("critical", 0.95),
        _assessment("high", 0.60),
        _assessment("medium", 0.40),
        _assessment("low", 0.10),
    ]
    scenario = assume_scenario(
        "Categories",
        budget=100.0,
        cost_per_contact=5.0,
        strategy="risk_category",
        risk_tiers=("high", "critical"),
    )
    result = run_campaign(rows, scenario)
    assert _selected_references(result) == ["critical", "high"]
    assert result.exclusion_counts["tier_not_targeted"] == 2
    assert result.eligible_count == 2


def test_unknown_risk_tier_is_rejected_against_the_table_labels() -> None:
    thresholds = RiskThresholds.from_value((0.40, 0.60))
    rows = [
        RiskAssessment(
            record_reference="three-tier",
            input_position=0,
            churn_probability=0.9,
            predicted_class=1,
            risk_tier=thresholds.tier_for(0.9),
            classification_threshold=0.5,
            risk_thresholds=thresholds,
            model_family="random_forest",
            model_run_id="fixture/random_forest",
            model_sha256=None,
            preprocessing_run_id="fixture-preprocess",
            assessed_at="2026-10-22T00:00:00Z",
        )
    ]
    scenario = assume_scenario(
        "Bad tier",
        strategy="risk_category",
        risk_tiers=("critical",),
    )
    # Tier labels are derived from the assessments when no context is supplied,
    # so a three-tier run cannot silently accept a "critical" targeting rule.
    with pytest.raises(SimulationError) as caught:
        run_campaign(rows, scenario)
    assert "unknown risk tier" in caught.value.message
    with pytest.raises(SimulationError) as explicit:
        run_campaign(
            _population(),
            scenario,
            context=RiskContext(tier_labels=("low", "medium", "high")),
        )
    assert "unknown risk tier" in explicit.value.message


def test_contact_limit_is_enforced_before_budget() -> None:
    scenario = assume_scenario(
        "Limited", budget=100.0, cost_per_contact=5.0, max_contacts=2
    )
    result = run_campaign(_population(), scenario)
    assert len(result.selected) == 2
    assert result.allocation.contact_limited_count == 3
    assert result.allocation.allocated_spend == pytest.approx(10.0)
    assert result.allocation.remaining_budget == pytest.approx(90.0)


def test_budget_constrained_strategy_stops_at_first_unaffordable_customer() -> None:
    rows = [
        _assessment("expensive", 0.9),
        _assessment("cheap-a", 0.8),
        _assessment("cheap-b", 0.7),
    ]
    base = dict(
        budget=20.0,
        cost_per_contact=5.0,
        effectiveness=0.5,
        cost_overrides={"expensive": 50.0},
    )
    greedy = run_campaign(
        rows, assume_scenario("Greedy", strategy="highest_risk", **base)
    )
    assert _selected_references(greedy) == ["cheap-a", "cheap-b"]
    assert greedy.allocation.unaffordable_count == 1
    assert greedy.allocation.allocated_spend == pytest.approx(10.0)

    strict = run_campaign(
        rows, assume_scenario("Strict", strategy="budget_constrained", **base)
    )
    assert _selected_references(strict) == []
    assert strict.allocation.unaffordable_count == 3
    assert strict.allocation.max_affordable is None
    assert "vary" in strict.allocation.notes[-1]


def test_unknown_cost_override_is_ignored_with_a_warning() -> None:
    scenario = assume_scenario(
        "Overrides",
        budget=100.0,
        cost_per_contact=5.0,
        cost_overrides={"missing-reference": 9.0},
    )
    result = run_campaign(_population(), scenario)
    assert len(result.selected) == 5
    assert any("did not match" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# zero and edge cases
# ---------------------------------------------------------------------------


def test_zero_cost_interventions_ignore_budget_and_leave_roi_undefined() -> None:
    scenario = assume_scenario(
        "Free", budget=100.0, cost_per_contact=0.0, effectiveness=0.5,
        monthly_value=10.0,
    )
    result = run_campaign(_population(), scenario)
    assert len(result.selected) == 5
    assert result.allocation.zero_cost is True
    assert result.allocation.max_affordable is None
    assert result.allocation.allocated_spend == pytest.approx(0.0)
    assert result.allocation.remaining_budget == pytest.approx(100.0)
    assert result.allocation.utilization == pytest.approx(0.0)
    assert result.projection.roi is None
    assert "campaign cost is zero" in result.projection.undefined_metrics["roi"]
    assert result.projection.cost_per_expected_prevented_churn == pytest.approx(0.0)


def test_zero_budget_with_paid_interventions_selects_nobody() -> None:
    scenario = assume_scenario(
        "No budget", budget=0.0, cost_per_contact=5.0, effectiveness=0.5
    )
    result = run_campaign(_population(), scenario)
    assert result.selected == []
    assert result.allocation.max_affordable == 0
    assert result.allocation.unaffordable_count == 5
    assert result.allocation.utilization is None
    assert result.projection.expected_prevented_churn == pytest.approx(0.0)
    assert result.projection.cost_per_expected_prevented_churn is None
    assert (
        "expected prevented churn is zero"
        in result.projection.undefined_metrics[
            "cost_per_expected_prevented_churn"
        ]
    )
    assert any("budget is zero" in warning for warning in result.warnings)


def test_zero_expected_prevented_churn_does_not_divide_by_zero() -> None:
    rows = [_assessment("zero-a", 0.0), _assessment("zero-b", 0.0)]
    scenario = assume_scenario(
        "Zero risk", budget=10.0, cost_per_contact=5.0, effectiveness=0.5
    )
    result = run_campaign(rows, scenario)
    assert result.projection.expected_prevented_churn == pytest.approx(0.0)
    assert result.projection.cost_per_expected_prevented_churn is None
    assert result.projection.targeting_lift is None
    assert result.projection.random_contact_expected_prevented_churn == pytest.approx(0.0)


def test_empty_population_yields_structured_zero_result() -> None:
    scenario = assume_scenario("Empty", budget=100.0, cost_per_contact=5.0)
    result = run_campaign([], scenario)
    assert result.assessed_count == 0
    assert result.selected == []
    assert result.allocation.utilization == pytest.approx(0.0)
    assert result.projection.expected_prevented_churn == pytest.approx(0.0)
    assert any("No Phase 9 assessments" in warning for warning in result.warnings)
    assert result.counts_reconcile


def test_effectiveness_zero_and_one() -> None:
    zero = run_campaign(
        _population(),
        assume_scenario("e=0", budget=15.0, cost_per_contact=5.0, effectiveness=0.0),
    )
    assert zero.projection.expected_prevented_churn == pytest.approx(0.0)
    assert any("effectiveness is zero" in item for item in zero.warnings)

    one = run_campaign(
        _population(),
        assume_scenario("e=1", budget=15.0, cost_per_contact=5.0, effectiveness=1.0),
    )
    assert one.projection.expected_prevented_churn == pytest.approx(2.0)
    assert one.projection.expected_prevented_churn == pytest.approx(
        one.projection.aggregate_churn_probability
    )


def test_budget_is_enforced_under_every_strategy() -> None:
    rows = [_assessment("a", 0.9), _assessment("b", 0.8), _assessment("c", 0.7)]
    strategies = {
        "highest_risk": {},
        "risk_threshold": {"min_risk_probability": 0.65},
        "risk_category": {"risk_tiers": ("high", "critical")},
        "budget_constrained": {},
    }
    for strategy, options in strategies.items():
        result = run_campaign(
            rows,
            assume_scenario(
                f"Budget {strategy}",
                budget=5.0,
                cost_per_contact=5.0,
                strategy=strategy,
                **options,
            ),
        )
        assert len(result.selected) == 1, strategy
        assert result.allocation.unaffordable_count == 2, strategy
        assert result.allocation.allocated_spend == pytest.approx(5.0), strategy
        assert result.allocation.allocated_spend <= 5.0
        assert result.allocation.remaining_budget == pytest.approx(0.0)


def test_non_assessment_records_are_rejected() -> None:
    with pytest.raises(SimulationError) as caught:
        run_campaign(
            [{"record_reference": "not-an-assessment", "churn_probability": 0.9}],
            assume_scenario("Bad input", budget=10.0),
        )
    assert caught.value.code == "CW-SIM-002"
    assert caught.value.details["index"] == 0
    assert "RiskService" in caught.value.remediation

    with pytest.raises(Exception) as invalid:
        _assessment("impossible", 1.5)
    assert "between 0 and 1" in str(invalid.value)


def test_cost_overrides_respect_the_budget() -> None:
    rows = [_assessment("a", 0.9), _assessment("b", 0.8)]
    scenario = assume_scenario(
        "Custom cost",
        budget=12.0,
        cost_per_contact=5.0,
        cost_overrides={"a": 11.0},
    )
    result = run_campaign(rows, scenario)
    assert result.allocation.allocated_spend == pytest.approx(11.0)
    assert result.allocation.allocated_spend <= scenario.budget
    assert _selected_references(result) == ["a"]


# ---------------------------------------------------------------------------
# configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"budget": -1.0},
        {"budget": math.inf},
        {"cost_per_contact": -0.01},
        {"effectiveness": 1.01},
        {"effectiveness": -0.01},
        {"monthly_value": -5.0},
        {"horizon_months": 0.0},
        {"value_unit": ""},
        {"strategy": "revenue_first"},
        {"max_contacts": 0},
        {"cost_overrides": {"": 5.0}},
        {"cost_overrides": {"ref": -2.0}},
    ],
)
def test_invalid_scenarios_are_rejected_with_remediation(overrides: dict) -> None:
    with pytest.raises(SimulationError) as caught:
        assume_scenario("Invalid", **overrides).validate_or_raise()
    assert caught.value.code == "CW-SIM-001"
    assert caught.value.remediation


def test_strategy_specific_requirements_are_enforced() -> None:
    with pytest.raises(SimulationError) as threshold:
        assume_scenario("T", strategy="risk_threshold").validate_or_raise()
    assert "min_risk_probability" in threshold.value.message

    with pytest.raises(SimulationError) as category:
        assume_scenario("C", strategy="risk_category").validate_or_raise()
    assert "at least one risk tier" in category.value.message


def test_scenario_payload_round_trips_and_rejects_unknown_keys() -> None:
    original = assume_scenario(
        "Round trip",
        budget=250.0,
        cost_per_contact=2.5,
        effectiveness=0.15,
        monthly_value=12.0,
        horizon_months=3.0,
        strategy="risk_threshold",
        min_risk_probability=0.4,
        cost_overrides={"a": 3.0},
    )
    rebuilt = CampaignScenario.from_dict(original.to_dict())
    assert rebuilt.to_dict() == original.to_dict()
    assert rebuilt.fingerprint() == original.fingerprint()

    payload = original.to_dict()
    payload["unexpected"] = True
    with pytest.raises(SimulationError):
        CampaignScenario.from_dict(payload)

    assumptions = original.to_dict()
    assumptions["assumptions"]["not_a_field"] = 1
    with pytest.raises(SimulationError):
        CampaignScenario.from_dict(assumptions)


def test_scenario_slug_is_ascii_for_safe_artifact_ids() -> None:
    scenario = assume_scenario("Café — Q4 trial")
    assert scenario.slug
    assert scenario.slug.isascii()
    assert all(character.isalnum() or character == "-" for character in scenario.slug)


def test_defaults_are_explicit_and_documented() -> None:
    assumptions = CampaignAssumptions()
    assert assumptions.effectiveness == 0.30
    assert assumptions.cost_per_contact == 5.0
    assert assumptions.monthly_value is None
    assert assumptions.average_value is None
    targeting = TargetingConfig()
    assert targeting.strategy == "highest_risk"
    assert targeting.max_contacts is None

    from churnwise.simulation.assumptions import (
        budget_grid,
        cost_grid,
        effectiveness_grid,
    )

    assert effectiveness_grid() == (0.10, 0.20, 0.30, 0.40, 0.50)
    assert budget_grid(None, base_budget=100.0) == (50.0, 100.0, 150.0, 200.0)
    assert cost_grid(None, base_cost=4.0) == (2.0, 4.0, 8.0)


# ---------------------------------------------------------------------------
# assumptions embedded in results
# ---------------------------------------------------------------------------


def test_result_embeds_the_exact_assumptions_used() -> None:
    scenario = assume_scenario(
        "Embedded",
        budget=123.0,
        cost_per_contact=4.5,
        effectiveness=0.35,
        monthly_value=17.0,
        horizon_months=9.0,
        value_unit="USD",
        min_risk_probability=0.35,
    )
    result = run_campaign(_population(), scenario)
    payload = result.to_dict()
    assert payload["assumptions"]["cost_per_contact"] == pytest.approx(4.5)
    assert payload["assumptions"]["effectiveness"] == pytest.approx(0.35)
    assert payload["assumptions"]["monthly_value"] == pytest.approx(17.0)
    assert payload["assumptions"]["horizon_months"] == pytest.approx(9.0)
    assert payload["assumptions"]["value_unit"] == "USD"
    assert payload["budget"] == pytest.approx(123.0)
    assert payload["targeting"]["min_risk_probability"] == pytest.approx(0.35)
    assert payload["hypothetical"] is True
    assert payload["projection"]["projection"] is True
    assert payload["projection"]["observed_retention_result"] is False
    assert payload["disclaimers"]
    assert all(payload["disclaimers"])
    assert any("hypothetical projections" in item for item in payload["disclaimers"])


def test_result_json_does_not_leak_segment_values() -> None:
    rows = [
        _assessment("opaque-1", 0.8, segment={"email": "person@example.com"}),
        _assessment("opaque-2", 0.2, segment={"email": "other@example.com"}),
    ]
    result = run_campaign(
        rows, assume_scenario("Privacy", budget=10.0, cost_per_contact=5.0)
    )
    encoded = json.dumps(result.to_dict())
    assert "person@example.com" not in encoded
    assert "other@example.com" not in encoded
    assert "opaque-1" in encoded


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def test_comparison_discloses_differing_assumptions() -> None:
    rows = _population()
    generous = run_campaign(
        rows,
        assume_scenario(
            "Generous", budget=100.0, cost_per_contact=5.0, effectiveness=0.6,
            monthly_value=20.0,
        ),
    )
    tight = run_campaign(
        rows,
        assume_scenario(
            "Tight", budget=10.0, cost_per_contact=5.0, effectiveness=0.1,
            monthly_value=20.0,
        ),
    )
    comparison = compare_scenarios([generous, tight])
    assert [row.name for row in comparison.rows] == ["Generous", "Tight"]
    assert comparison.shared_inputs is True
    assert set(comparison.differing_assumptions) == {"budget", "effectiveness"}
    assert comparison.rows[0].selected_count == 5
    assert comparison.rows[1].selected_count == 2
    assert any("does not" in note for note in comparison.notes)
    assert comparison.limitations
    assert "fractional expectation" in comparison.limitations[0]
    assert comparison.disclaimers


def test_comparison_flags_differing_value_units() -> None:
    rows = _population()
    dollars = run_campaign(
        rows,
        assume_scenario(
            "Dollars", budget=10.0, cost_per_contact=5.0, monthly_value=10.0,
            value_unit="USD",
        ),
    )
    euros = run_campaign(
        rows,
        assume_scenario(
            "Euros", budget=10.0, cost_per_contact=5.0, monthly_value=10.0,
            value_unit="EUR",
        ),
    )
    comparison = compare_scenarios([dollars, euros])
    assert "value_unit" in comparison.differing_assumptions
    assert [row.value_unit for row in comparison.rows] == ["USD", "EUR"]
    assert any("not directly comparable" in note for note in comparison.notes)


def test_comparison_flags_mismatched_risk_inputs() -> None:
    rows = _population()
    scenario = assume_scenario("Same", budget=10.0, cost_per_contact=5.0)
    first = run_campaign(rows, scenario, context=RiskContext(run_id="run-a", model_sha256="a" * 64))
    second = run_campaign(rows, scenario, context=RiskContext(run_id="run-b", model_sha256="b" * 64))
    comparison = compare_scenarios([first, second])
    assert comparison.shared_inputs is False
    assert len(comparison.input_differences) == 2
    assert "risk run id" in comparison.input_differences[0]


# ---------------------------------------------------------------------------
# projection unit behaviour
# ---------------------------------------------------------------------------


def test_project_outcomes_is_independent_of_selection_order() -> None:
    scenario = assume_scenario(
        "Order", budget=100.0, cost_per_contact=5.0, effectiveness=0.5, monthly_value=10.0
    )
    result = run_campaign(_population(), scenario)
    reversed_projection = project_outcomes(
        list(reversed(result.selected)),
        scenario,
        population_probabilities=[0.9, 0.6, 0.5, 0.3, 0.2],
    )
    assert reversed_projection.expected_prevented_churn == pytest.approx(
        result.projection.expected_prevented_churn
    )
    assert reversed_projection.random_contact_expected_prevented_churn == pytest.approx(
        result.projection.random_contact_expected_prevented_churn
    )


def test_eligibility_does_not_apply_budget_or_limits() -> None:
    scenario = assume_scenario(
        "Eligibility", budget=0.0, cost_per_contact=5.0, max_contacts=1,
        strategy="risk_threshold", min_risk_probability=0.3,
    )
    eligible, exclusions = resolve_eligibility(_population(), scenario)
    assert len(eligible) == 4
    assert [item.reason for item in exclusions] == ["below_risk_threshold"]


def test_allocate_budget_reconciles_eligible_counts() -> None:
    scenario = assume_scenario("Allocate", budget=10.0, cost_per_contact=5.0)
    eligible, _ = resolve_eligibility(_population(), scenario)
    selected, allocation, exclusions = allocate_budget(eligible, scenario)
    assert allocation.selected_count == len(selected) == 2
    assert allocation.eligible_count == len(eligible) == 5
    assert len(selected) + allocation.unaffordable_count + allocation.contact_limited_count == 5
    assert len(exclusions) == allocation.unaffordable_count


# ---------------------------------------------------------------------------
# sensitivity
# ---------------------------------------------------------------------------


def test_sensitivity_effectiveness_series_matches_hand_computation() -> None:
    scenario = assume_scenario(
        "Sensitivity", budget=15.0, cost_per_contact=5.0, effectiveness=0.5,
        monthly_value=10.0, horizon_months=6.0,
    )
    analysis = run_sensitivity(
        _population(), scenario, dimensions=("effectiveness",),
        effectiveness_values="0,0.5,1",
    )
    series = analysis.series[0]
    assert series.dimension == "effectiveness"
    assert [point.value for point in series.points] == [0.0, 0.5, 1.0]
    assert [point.is_baseline for point in series.points] == [False, True, False]
    # Selected set unchanged: sum(p) over c1..c3 = 2.0
    assert [point.selected_count for point in series.points] == [3, 3, 3]
    assert [point.expected_prevented_churn for point in series.points] == pytest.approx(
        [0.0, 1.0, 2.0]
    )
    assert [point.allocated_spend for point in series.points] == pytest.approx(
        [15.0, 15.0, 15.0]
    )
    assert len({point.selection_fingerprint for point in series.points}) == 1
    assert all(point.selection_fingerprint for point in series.points)


def test_sensitivity_budget_series_changes_the_selected_set() -> None:
    scenario = assume_scenario(
        "Budget sensitivity", budget=15.0, cost_per_contact=5.0, effectiveness=0.5
    )
    analysis = run_sensitivity(
        _population(), scenario, dimensions=("budget",), budget_values="5,15,30"
    )
    points = analysis.series[0].points
    assert [point.value for point in points] == [5.0, 15.0, 30.0]
    assert [point.selected_count for point in points] == [1, 3, 5]
    assert [point.expected_prevented_churn for point in points] == pytest.approx(
        [0.45, 1.0, 1.25]
    )
    assert [point.allocated_spend for point in points] == pytest.approx([5.0, 15.0, 25.0])
    assert len({point.selection_fingerprint for point in points}) == 3
    assert analysis.fixed_parameters["effectiveness"] == pytest.approx(0.5)


def test_sensitivity_cost_series_can_reach_zero_cost() -> None:
    scenario = assume_scenario(
        "Cost sensitivity", budget=15.0, cost_per_contact=5.0, effectiveness=0.5
    )
    analysis = run_sensitivity(
        _population(), scenario, dimensions=("cost_per_contact",), cost_values="0,5"
    )
    points = analysis.series[0].points
    assert [point.selected_count for point in points] == [5, 3]
    assert [point.allocated_spend for point in points] == pytest.approx([0.0, 15.0])


def test_degenerate_sensitivity_grids_are_skipped_with_a_warning() -> None:
    free = run_sensitivity(
        _population(),
        assume_scenario("Free", budget=100.0, cost_per_contact=0.0, effectiveness=0.5),
        dimensions=("cost_per_contact",),
    )
    assert free.series == []
    assert any(
        "skipped" in warning and "cost_per_contact" in warning
        for warning in free.warnings
    )

    broke = run_sensitivity(
        _population(),
        assume_scenario("No budget", budget=0.0, cost_per_contact=5.0),
        dimensions=("budget",),
    )
    assert broke.series == []
    assert any(
        "skipped" in warning and "budget" in warning for warning in broke.warnings
    )
    assert any("projections" in warning for warning in broke.warnings)


def test_invalid_sensitivity_grids_are_rejected() -> None:
    scenario = assume_scenario("Grid", budget=100.0, cost_per_contact=5.0)
    with pytest.raises(SimulationError):
        run_sensitivity(_population(), scenario, dimensions=("budget",), budget_values="5")
    with pytest.raises(SimulationError):
        run_sensitivity(
            _population(), scenario, dimensions=("effectiveness",),
            effectiveness_values="-0.1,0.5",
        )
    with pytest.raises(SimulationError):
        run_sensitivity(_population(), scenario, dimensions=("revenue",))
    with pytest.raises(SimulationError):
        run_sensitivity(_population(), scenario, dimensions=())
