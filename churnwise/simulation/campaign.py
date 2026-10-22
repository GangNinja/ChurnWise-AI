"""Deterministic campaign selection, budget allocation and outcome projection.

The engine is deliberately small and explicit:

* eligibility comes from Phase 9 tiers/probabilities;
* selection is a greedy descending-probability walk with stable tie-breaking;
* spend is enforced against the budget, never estimated;
* projected saves are ``sum(p_i * e)`` with ``e`` a user assumption.

No function here fits a model, estimates uplift or rounds a fractional expected
count into a claimed number of retained customers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from churnwise import __version__
from churnwise.intelligence.contracts import RiskAssessment
from churnwise.simulation.contracts import (
    DISCLAIMERS,
    EXCLUSION_REASONS,
    LIMITATIONS,
    STRATEGY_LABELS,
    BudgetAllocation,
    CampaignResult,
    CampaignScenario,
    ComparisonRow,
    ExcludedCustomer,
    OutcomeProjection,
    RiskContext,
    ScenarioComparison,
    SelectedCustomer,
    SensitivityAnalysis,
    SensitivityPoint,
    SensitivitySeries,
    comparison_row,
)
from churnwise.simulation.assumptions import (
    DEFAULT_FIXED_PARAMETERS_NOTE,
    budget_grid,
    cost_grid,
    dimension_unit,
    effectiveness_grid,
    sensitivity_dimensions,
)
from churnwise.simulation.errors import (
    CODE_INVALID_CONFIG,
    CODE_INVALID_INPUT,
    SimulationError,
)

BUDGET_TOLERANCE = 1e-9


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _selection_fingerprint(selected: Sequence[SelectedCustomer]) -> str:
    """Short, stable digest of the selected reference set.

    Sensitivity rows record this instead of duplicating every reference, so a
    reader can see whether the selected membership changed without the
    artifacts growing with each grid point.
    """
    joined = "|".join(item.record_reference for item in selected)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _ordered(records: Iterable[RiskAssessment]) -> list[RiskAssessment]:
    """Descending churn probability with a stable reference/position tie-break."""
    return sorted(
        records,
        key=lambda item: (
            -float(item.churn_probability),
            item.record_reference,
            int(item.input_position),
        ),
    )


def _validate_records(records: Sequence[Any]) -> None:
    """Reject anything that is not a Phase 9 assessment with a usable score."""
    for index, item in enumerate(records):
        if not isinstance(item, RiskAssessment):
            raise SimulationError(
                CODE_INVALID_INPUT,
                f"record {index} is not a Phase 9 RiskAssessment",
                "Pass assessments produced by RiskService or RiskTable.records; "
                "do not construct ad-hoc customer dictionaries.",
                {"index": index, "type": type(item).__name__},
            )
        probability = float(item.churn_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise SimulationError(
                CODE_INVALID_INPUT,
                f"record {index} has an invalid churn probability",
                "Rescore the customer with Phase 9; probabilities must be in [0, 1].",
                {"index": index},
            )


def _cost_for(item: RiskAssessment, scenario: CampaignScenario) -> float:
    override = scenario.cost_overrides.get(item.record_reference)
    if override is not None:
        return float(override)
    return float(scenario.assumptions.cost_per_contact)


def _context_from_records(records: Sequence[RiskAssessment]) -> RiskContext:
    """Derive tier labels/boundaries from the assessments when none is supplied.

    Guessing the default four-tier labels for a three-tier risk run would let a
    ``critical`` targeting rule pass validation and silently select nobody.
    """
    if not records:
        return RiskContext()
    thresholds = records[0].risk_thresholds
    return RiskContext(
        tier_boundaries=tuple(thresholds.boundaries),
        tier_labels=tuple(thresholds.labels),
    )


def _validate_tiers(scenario: CampaignScenario, labels: Sequence[str]) -> None:
    unknown = sorted(set(scenario.targeting.risk_tiers) - set(labels))
    if unknown:
        raise SimulationError(
            CODE_INVALID_CONFIG,
            f"scenario targets unknown risk tier(s): {', '.join(unknown)}",
            f"The Phase 9 run defines these tiers: {list(labels)}.",
        )


def resolve_eligibility(
    records: Sequence[RiskAssessment], scenario: CampaignScenario
) -> tuple[list[RiskAssessment], list[ExcludedCustomer]]:
    """Apply the threshold/tier rules; no budget or count limit is applied here.

    Boundary semantics: a customer is eligible when
    ``churn_probability >= min_risk_probability`` (lower-inclusive), matching the
    Phase 9 tier boundary convention.
    """
    scenario.validate_or_raise()
    _validate_records(records)
    threshold = scenario.targeting.min_risk_probability
    tiers = set(scenario.targeting.risk_tiers)
    eligible: list[RiskAssessment] = []
    exclusions: list[ExcludedCustomer] = []
    for item in records:
        probability = float(item.churn_probability)
        if tiers and item.risk_tier not in tiers:
            exclusions.append(
                ExcludedCustomer(
                    record_reference=item.record_reference,
                    reason="tier_not_targeted",
                    detail=f"tier {item.risk_tier!r} is not in {sorted(tiers)}",
                )
            )
            continue
        if threshold is not None and probability < float(threshold):
            exclusions.append(
                ExcludedCustomer(
                    record_reference=item.record_reference,
                    reason="below_risk_threshold",
                    detail=(
                        f"churn probability {probability:.6f} is below the "
                        f"configured minimum {float(threshold):.6f}"
                    ),
                )
            )
            continue
        eligible.append(item)
    return eligible, exclusions


def allocate_budget(
    eligible: Sequence[RiskAssessment],
    scenario: CampaignScenario,
) -> tuple[list[SelectedCustomer], BudgetAllocation, list[ExcludedCustomer]]:
    """Select customers in risk order while enforcing count and budget limits.

    With the ``budget_constrained`` strategy the walk stops at the first
    customer whose intervention cost does not fit, preserving a strict
    risk-order prefix. Other strategies skip an unaffordable customer when
    per-customer cost overrides make later, cheaper customers affordable.

    Budget arithmetic uses exact decimal-free floats with a ``1e-9`` tolerance;
    ``allocated_spend`` is the sum of the *actual* selected intervention costs.
    """
    scenario.validate_or_raise()
    budget = float(scenario.budget)
    base_cost = float(scenario.assumptions.cost_per_contact)
    effectiveness = float(scenario.assumptions.effectiveness)
    limit = scenario.targeting.max_contacts
    tolerance = BUDGET_TOLERANCE * max(1.0, budget)

    ordered = _ordered(eligible)
    costs = [_cost_for(item, scenario) for item in ordered]
    zero_cost = (
        all(cost == 0.0 for cost in costs)
        if ordered
        else base_cost == 0.0 and all(value == 0.0 for value in scenario.cost_overrides.values())
    )
    uniform_cost = len({round(cost, 12) for cost in costs}) <= 1
    notes: list[str] = []

    if not ordered:
        max_affordable = 0
        notes.append("No eligible customer is available, so no intervention can be funded.")
    elif zero_cost:
        max_affordable = None
        notes.append(
            "Intervention cost is zero: the budget does not constrain the "
            "selection and the maximum affordable count is unbounded."
        )
    elif uniform_cost:
        max_affordable = int(math.floor(budget / costs[0] + BUDGET_TOLERANCE))
        notes.append(
            "Maximum affordable count = floor(budget / uniform intervention cost)."
        )
    else:
        max_affordable = None
        notes.append(
            "Per-customer intervention costs vary, so the maximum affordable "
            "count is determined greedily rather than by a single division."
        )

    selected: list[SelectedCustomer] = []
    exclusions: list[ExcludedCustomer] = []
    spend = 0.0
    unaffordable = 0
    contact_limited = 0

    for index, item in enumerate(ordered):
        if limit is not None and len(selected) >= int(limit):
            for remaining in ordered[index:]:
                exclusions.append(
                    ExcludedCustomer(
                        record_reference=remaining.record_reference,
                        reason="contact_limit",
                        detail=f"campaign contact limit of {int(limit)} reached",
                    )
                )
                contact_limited += 1
            break
        cost = costs[index]
        if spend + cost > budget + tolerance:
            if scenario.targeting.strategy == "budget_constrained":
                for remaining in ordered[index:]:
                    exclusions.append(
                        ExcludedCustomer(
                            record_reference=remaining.record_reference,
                            reason="budget_exhausted",
                            detail=(
                                f"remaining budget {max(0.0, budget - spend):.6f} "
                                f"cannot cover an intervention cost of "
                                f"{cost:.6f} {scenario.assumptions.value_unit}"
                            ),
                        )
                    )
                    unaffordable += 1
                break
            exclusions.append(
                ExcludedCustomer(
                    record_reference=item.record_reference,
                    reason="budget_exhausted",
                    detail=(
                        f"remaining budget {max(0.0, budget - spend):.6f} cannot "
                        f"cover an intervention cost of {cost:.6f} "
                        f"{scenario.assumptions.value_unit}"
                    ),
                )
            )
            unaffordable += 1
            continue
        selected.append(
            SelectedCustomer(
                record_reference=item.record_reference,
                input_position=int(item.input_position),
                churn_probability=float(item.churn_probability),
                risk_tier=item.risk_tier,
                rank=item.rank,
                intervention_cost=float(cost),
                expected_prevented_churn=float(item.churn_probability) * effectiveness,
            )
        )
        spend += cost

    allocated = math.fsum(item.intervention_cost for item in selected)
    remaining = budget - allocated
    if abs(remaining) <= tolerance:
        remaining = 0.0
    if remaining < 0.0:  # pragma: no cover - guards against float drift
        remaining = 0.0
    utilization = None if budget <= 0.0 else allocated / budget
    if zero_cost and budget > 0.0:
        notes.append(
            "Budget utilisation is zero because no intervention cost was incurred."
        )
    if len(selected) + unaffordable + contact_limited != len(eligible):
        raise SimulationError(
            CODE_INVALID_INPUT,
            "internal allocation error: selected, unaffordable and contact-limited "
            "counts do not reconcile with the eligible population",
            "Report this defect; do not use the simulation output.",
            {
                "eligible": len(eligible),
                "selected": len(selected),
                "unaffordable": unaffordable,
                "contact_limited": contact_limited,
            },
        )

    allocation = BudgetAllocation(
        budget=budget,
        cost_per_contact=base_cost,
        eligible_count=len(eligible),
        selected_count=len(selected),
        allocated_spend=allocated,
        remaining_budget=remaining,
        max_affordable=max_affordable,
        utilization=utilization,
        unaffordable_count=unaffordable,
        contact_limited_count=contact_limited,
        zero_cost=zero_cost,
        notes=notes,
    )
    return selected, allocation, exclusions


def project_outcomes(
    selected: Sequence[SelectedCustomer],
    scenario: CampaignScenario,
    *,
    population_probabilities: Sequence[float] | None = None,
) -> OutcomeProjection:
    """Compute assumption-driven expected outcomes and undefined-metric reasons.

    ``expected_prevented_churn = sum(p_i * e)``. It is a fractional expectation,
    not an observed count. Financial metrics exist only when the caller supplies
    a per-customer value assumption.
    """
    assumptions = scenario.assumptions
    effectiveness = float(assumptions.effectiveness)
    aggregate = math.fsum(item.churn_probability for item in selected)
    prevented = math.fsum(
        float(item.churn_probability) * effectiveness for item in selected
    )
    campaign_cost = math.fsum(item.intervention_cost for item in selected)
    average_value = assumptions.average_value
    undefined: dict[str, str] = {}

    expected_saved_value: float | None = None
    net_benefit: float | None = None
    roi: float | None = None
    break_even: float | None = None
    if average_value is None:
        undefined["expected_saved_value"] = (
            "no per-customer value assumption was supplied (monthly_value is None)"
        )
        undefined["net_benefit"] = (
            "no per-customer value assumption was supplied (monthly_value is None)"
        )
        undefined["roi"] = (
            "no per-customer value assumption was supplied (monthly_value is None)"
        )
        undefined["break_even_effectiveness"] = (
            "no per-customer value assumption was supplied (monthly_value is None)"
        )
    else:
        expected_saved_value = prevented * average_value
        net_benefit = expected_saved_value - campaign_cost
        if campaign_cost > 0.0:
            roi = net_benefit / campaign_cost
        else:
            undefined["roi"] = "campaign cost is zero, so a ratio is undefined"
        if aggregate > 0.0 and average_value > 0.0:
            break_even = campaign_cost / (aggregate * average_value)
        else:
            undefined["break_even_effectiveness"] = (
                "expected at-risk mass or assumed customer value is zero"
            )

    cost_per_prevented: float | None = None
    if prevented > 0.0:
        cost_per_prevented = campaign_cost / prevented
    else:
        undefined["cost_per_expected_prevented_churn"] = (
            "expected prevented churn is zero"
        )

    baseline: float | None = None
    lift: float | None = None
    if population_probabilities is None:
        undefined["random_contact_expected_prevented_churn"] = (
            "no eligible-population probabilities were supplied"
        )
        undefined["targeting_lift_vs_random"] = (
            "random-contact baseline is unavailable"
        )
    else:
        probabilities = [float(value) for value in population_probabilities]
        if not selected or not probabilities:
            baseline = 0.0
            undefined["targeting_lift_vs_random"] = (
                "no customer was selected, so targeting lift is undefined"
            )
        else:
            mean_probability = math.fsum(probabilities) / len(probabilities)
            baseline = mean_probability * len(selected) * effectiveness
            if baseline > 0.0:
                lift = prevented / baseline
            else:
                undefined["targeting_lift_vs_random"] = (
                    "random-contact expected prevented churn is zero"
                )

    return OutcomeProjection(
        aggregate_churn_probability=aggregate,
        expected_prevented_churn=prevented,
        effectiveness=effectiveness,
        campaign_cost=campaign_cost,
        assumed_average_value=average_value,
        expected_saved_value=expected_saved_value,
        net_benefit=net_benefit,
        roi=roi,
        cost_per_expected_prevented_churn=cost_per_prevented,
        break_even_effectiveness=break_even,
        random_contact_expected_prevented_churn=baseline,
        targeting_lift=lift,
        financial_metrics_available=average_value is not None,
        undefined_metrics=undefined,
    )


def _exclusion_counts(exclusions: Sequence[ExcludedCustomer]) -> dict[str, int]:
    counts = {reason: 0 for reason in EXCLUSION_REASONS}
    for item in exclusions:
        counts[item.reason] = counts.get(item.reason, 0) + 1
    return counts


def _build_warnings(
    scenario: CampaignScenario,
    allocation: BudgetAllocation,
    projection: OutcomeProjection,
    *,
    assessed: int,
    eligible: int,
    unknown_overrides: Sequence[str],
    context: RiskContext,
) -> list[str]:
    warnings: list[str] = [f"Upstream risk run: {item}" for item in context.warnings]
    if assessed == 0:
        warnings.append(
            "No Phase 9 assessments were supplied; every campaign metric is zero."
        )
    if assessed > 0 and eligible == 0:
        targeting = scenario.targeting
        if targeting.strategy == "risk_threshold":
            warnings.append(
                "No customer reached the configured risk threshold of "
                f"{float(targeting.min_risk_probability or 0.0):.6f}."
            )
        elif targeting.strategy == "risk_category":
            warnings.append(
                "No customer belongs to the targeted risk tiers "
                f"{list(targeting.risk_tiers)}."
            )
        else:
            warnings.append("No customer met the targeting criteria.")
    if allocation.selected_count == 0 and assessed > 0 and eligible > 0:
        warnings.append(
            "Eligible customers exist but none could be selected; inspect the "
            "budget and contact limits."
        )
    if float(scenario.budget) <= 0.0 and float(scenario.assumptions.cost_per_contact) > 0.0:
        warnings.append(
            "The campaign budget is zero, so no paid intervention can be funded."
        )
    if allocation.zero_cost:
        warnings.append(
            "Intervention cost is zero; campaign cost is zero, so ROI and "
            "cost-per-prevented-churn are undefined rather than infinite."
        )
    if not projection.financial_metrics_available:
        warnings.append(
            "No per-customer value assumption was supplied; saved-value, "
            "net-benefit, ROI and break-even projections are unavailable. No "
            "revenue or lifetime value was invented."
        )
    if float(scenario.assumptions.effectiveness) == 0.0:
        warnings.append(
            "Assumed effectiveness is zero, so every prevented-churn projection "
            "is zero by construction."
        )
    if unknown_overrides:
        warnings.append(
            f"{len(unknown_overrides)} per-customer cost override(s) did not match "
            "an assessed record and were ignored."
        )
    if allocation.unaffordable_count > 0 and allocation.contact_limited_count == 0:
        warnings.append(
            f"{allocation.unaffordable_count} eligible customer(s) were left "
            "untargeted because the remaining budget could not cover their "
            "intervention cost."
        )
    break_even = projection.break_even_effectiveness
    if break_even is not None and break_even > 1.0:
        warnings.append(
            "Break-even effectiveness exceeds 1.0: no effectiveness assumption "
            "in [0, 1] recovers the campaign cost under these assumptions."
        )
    return warnings


def run_campaign(
    records: Sequence[RiskAssessment],
    scenario: CampaignScenario,
    *,
    context: RiskContext | None = None,
    scenario_id: str = "",
    generated_at: str | None = None,
) -> CampaignResult:
    """Run one scenario against measured Phase 9 assessments."""
    scenario.validate_or_raise()
    _validate_records(records)
    resolved_context = (
        context if context is not None else _context_from_records(records)
    )
    _validate_tiers(scenario, resolved_context.tier_labels)

    eligible, eligibility_exclusions = resolve_eligibility(records, scenario)
    selected, allocation, allocation_exclusions = allocate_budget(eligible, scenario)
    population_probabilities = [float(item.churn_probability) for item in eligible]
    projection = project_outcomes(
        selected, scenario, population_probabilities=population_probabilities
    )
    exclusions = list(eligibility_exclusions) + list(allocation_exclusions)
    unknown_overrides = sorted(
        reference
        for reference in scenario.cost_overrides
        if reference not in {item.record_reference for item in records}
    )
    warnings = _build_warnings(
        scenario,
        allocation,
        projection,
        assessed=len(records),
        eligible=len(eligible),
        unknown_overrides=unknown_overrides,
        context=resolved_context,
    )
    timestamp = generated_at or _now()
    provenance = {
        "phase": 10,
        "simulation_schema_version": "1.0",
        "risk_run_id": resolved_context.run_id,
        "risk_generated_at": resolved_context.generated_at,
        "model": resolved_context.to_dict()["model"],
        "dataset_sha256": resolved_context.dataset_sha256,
        "split_fingerprint": resolved_context.split_fingerprint,
        "risk_thresholds": resolved_context.thresholds.to_dict(),
        "code_version": __version__,
        "seed": None,
        "randomness": "none; selection and arithmetic are deterministic",
        "probability_source": "Phase 9 RiskAssessment.churn_probability",
        "assumption_source": "user-supplied campaign configuration",
        "selection_order": "descending churn probability, then record reference",
        "random_baseline_population": "eligible customers under the same targeting rule",
        "scenario_fingerprint": scenario.fingerprint(),
        "generated_at": timestamp,
        "assessed_rows": len(records),
    }
    return CampaignResult(
        scenario=scenario,
        context=resolved_context,
        allocation=allocation,
        projection=projection,
        selected=selected,
        exclusions=exclusions,
        exclusion_counts=_exclusion_counts(exclusions),
        generated_at=timestamp,
        scenario_id=scenario_id,
        assessed_count=len(records),
        eligible_count=len(eligible),
        warnings=warnings,
        limitations=list(LIMITATIONS),
        disclaimers=list(DISCLAIMERS),
        provenance=provenance,
    )


def compare_scenarios(results: Sequence[CampaignResult]) -> ScenarioComparison:
    """Compare scenarios, keeping their differing assumptions visible."""
    if not results:
        raise SimulationError(
            CODE_INVALID_INPUT,
            "scenario comparison requires at least one result",
            "Run at least one campaign scenario before comparing.",
        )
    rows: list[ComparisonRow] = [comparison_row(item) for item in results]
    first = results[0]
    differences: list[str] = []
    shared = True
    for result in results[1:]:
        checks = (
            ("risk run id", first.context.run_id, result.context.run_id),
            ("model SHA-256", first.context.model_sha256, result.context.model_sha256),
            (
                "dataset SHA-256",
                first.context.dataset_sha256,
                result.context.dataset_sha256,
            ),
            (
                "assessed population size",
                first.assessed_count,
                result.assessed_count,
            ),
            (
                "risk tier boundaries",
                tuple(first.context.tier_boundaries),
                tuple(result.context.tier_boundaries),
            ),
        )
        for label, left, right in checks:
            if left != right:
                shared = False
                differences.append(
                    f"{result.scenario.name!r} differs from "
                    f"{first.scenario.name!r} in {label}: {left!r} vs {right!r}"
                )
    assumption_fields = (
        "budget",
        "cost_per_contact",
        "effectiveness",
        "monthly_value",
        "horizon_months",
        "value_unit",
        "min_risk_probability",
        "risk_tiers",
        "max_contacts",
        "strategy",
    )
    differing: list[str] = []
    for field_name in assumption_fields:
        values = [getattr(row, field_name) for row in rows]
        if any(value != values[0] for value in values[1:]):
            differing.append(field_name)
    notes = [
        "All rows use the same Phase 9 churn probabilities unless an input "
        "difference is listed above.",
        "A larger projected figure under more optimistic assumptions does not "
        "mean that scenario is a better campaign; compare the assumptions.",
        "Total campaign cost is the simulated spend actually allocated, not the "
        "configured budget.",
    ]
    units = sorted({row.value_unit for row in rows})
    if len(units) > 1:
        notes.append(
            "Scenarios use different value units ("
            + ", ".join(units)
            + "); monetary columns are not directly comparable."
        )
    else:
        notes.append(f"All monetary columns use the {units[0]!r} value unit.")
    if differing:
        notes.append(
            "Scenarios differ in: " + ", ".join(differing) + "."
        )
    return ScenarioComparison(
        rows=rows,
        risk_run_id=first.context.run_id,
        shared_inputs=shared,
        input_differences=differences,
        differing_assumptions=differing,
        notes=notes,
        limitations=list(LIMITATIONS),
        disclaimers=list(DISCLAIMERS),
    )


def run_sensitivity(
    records: Sequence[RiskAssessment],
    scenario: CampaignScenario,
    *,
    context: RiskContext | None = None,
    scenario_id: str = "",
    dimensions: Sequence[str] | None = None,
    effectiveness_values: Any | None = None,
    budget_values: Any | None = None,
    cost_values: Any | None = None,
) -> SensitivityAnalysis:
    """Vary one assumption at a time and recompute genuine campaign results."""
    scenario.validate_or_raise()
    resolved_dimensions = sensitivity_dimensions(dimensions)
    series: list[SensitivitySeries] = []
    warnings: list[str] = []
    if not records:
        warnings.append(
            "No assessments were supplied, so every sensitivity point is zero."
        )

    for dimension in resolved_dimensions:
        if dimension == "effectiveness":
            values = effectiveness_grid(effectiveness_values)
            baseline = float(scenario.assumptions.effectiveness)
        elif dimension == "budget":
            values = budget_grid(budget_values, base_budget=float(scenario.budget))
            baseline = float(scenario.budget)
        else:
            values = cost_grid(
                cost_values, base_cost=float(scenario.assumptions.cost_per_contact)
            )
            baseline = float(scenario.assumptions.cost_per_contact)

        distinct = sorted({float(value) for value in values})
        if len(distinct) < 2:
            warnings.append(
                f"Sensitivity for {dimension} was skipped: every configured grid "
                f"value resolves to {distinct[0]!r} {dimension_unit(dimension, scenario.assumptions)}, "
                "which cannot show a trend. Pass explicit distinct values."
            )
            continue

        points: list[SensitivityPoint] = []
        for value in distinct:
            if dimension == "effectiveness":
                varied = replace(
                    scenario,
                    assumptions=replace(
                        scenario.assumptions, effectiveness=float(value)
                    ),
                )
            elif dimension == "budget":
                varied = replace(scenario, budget=float(value))
            else:
                varied = replace(
                    scenario,
                    assumptions=replace(
                        scenario.assumptions, cost_per_contact=float(value)
                    ),
                )
            result = run_campaign(records, varied, context=context)
            points.append(
                SensitivityPoint(
                    value=float(value),
                    is_baseline=abs(float(value) - baseline) <= 1e-12,
                    eligible_count=result.allocation.eligible_count,
                    selected_count=result.allocation.selected_count,
                    selection_fingerprint=_selection_fingerprint(result.selected),
                    allocated_spend=result.allocation.allocated_spend,
                    remaining_budget=result.allocation.remaining_budget,
                    expected_prevented_churn=result.projection.expected_prevented_churn,
                    expected_saved_value=result.projection.expected_saved_value,
                    cost_per_expected_prevented_churn=(
                        result.projection.cost_per_expected_prevented_churn
                    ),
                    utilization=result.allocation.utilization,
                )
            )
        series.append(
            SensitivitySeries(
                dimension=dimension,
                unit=dimension_unit(dimension, scenario.assumptions),
                baseline_value=float(baseline),
                points=points,
            )
        )

    warnings.append(
        "Sensitivity points are projections under each assumption value; they "
        "are not measurements of real-world campaign effectiveness."
    )
    return SensitivityAnalysis(
        scenario_id=scenario_id,
        series=series,
        fixed_parameters={
            "strategy": scenario.targeting.strategy,
            "strategy_description": STRATEGY_LABELS.get(
                scenario.targeting.strategy, ""
            ),
            "budget": float(scenario.budget),
            "cost_per_contact": float(scenario.assumptions.cost_per_contact),
            "effectiveness": float(scenario.assumptions.effectiveness),
            "monthly_value": scenario.assumptions.monthly_value,
            "horizon_months": float(scenario.assumptions.horizon_months),
            "value_unit": str(scenario.assumptions.value_unit),
            "min_risk_probability": scenario.targeting.min_risk_probability,
            "risk_tiers": list(scenario.targeting.risk_tiers),
            "max_contacts": scenario.targeting.max_contacts,
            "note": DEFAULT_FIXED_PARAMETERS_NOTE,
        },
        warnings=warnings,
        limitations=list(LIMITATIONS),
        disclaimers=list(DISCLAIMERS),
    )


__all__ = [
    "allocate_budget",
    "compare_scenarios",
    "project_outcomes",
    "resolve_eligibility",
    "run_campaign",
    "run_sensitivity",
]
