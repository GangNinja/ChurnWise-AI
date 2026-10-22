"""Human-readable Phase 10 summaries, reports and comparison tables.

Every number rendered here is read from a computed :class:`CampaignResult`,
:class:`ScenarioComparison` or :class:`SensitivityAnalysis`. Nothing is
illustrative, sampled or hard-coded.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from churnwise.simulation.contracts import (
    CampaignResult,
    ScenarioComparison,
    SensitivityAnalysis,
    SensitivitySeries,
)


def _number(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def _percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _metric(value: float | None, unit: str, *, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{_number(value, digits=digits)} {unit}"


def _md(value: Any) -> str:
    """Keep user-supplied labels from breaking or injecting Markdown tables."""
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def render_summary(result: CampaignResult) -> str:
    scenario = result.scenario
    assumptions = scenario.assumptions
    projection = result.projection
    allocation = result.allocation
    unit = assumptions.value_unit
    lines = [
        f"Campaign scenario {result.scenario_id or '(unsaved)'} — {scenario.name}",
        f"Targeting: {scenario.targeting.strategy} "
        f"({scenario.targeting.to_dict()['strategy_description']})",
        (
            "Assumptions: budget "
            f"{_number(scenario.budget, digits=2)} {unit}; cost per contact "
            f"{_number(assumptions.cost_per_contact, digits=2)} {unit}; assumed "
            f"effectiveness {_percent(assumptions.effectiveness)}; monthly value "
            f"{'not supplied' if assumptions.monthly_value is None else _number(assumptions.monthly_value, digits=2) + ' ' + unit}; "
            f"horizon {_number(assumptions.horizon_months, digits=1)} months"
        ),
        (
            f"Risk input: run {result.context.run_id or 'unknown'} "
            f"({result.context.model_family or 'unknown model'})"
        ),
        (
            f"Population: {result.assessed_count} assessed, {result.eligible_count} "
            f"eligible, {len(result.selected)} selected, {result.excluded_count} excluded"
        ),
        (
            f"Allocation: spend {_number(allocation.allocated_spend, digits=2)} {unit} "
            f"of {_number(allocation.budget, digits=2)} {unit}; remaining "
            f"{_number(allocation.remaining_budget, digits=2)} {unit}; utilisation "
            f"{_percent(allocation.utilization)}; max affordable "
            f"{'unbounded' if allocation.max_affordable is None else allocation.max_affordable}"
        ),
        "Projection (hypothetical, assumption-based):",
        (
            f"  expected prevented churn: "
            f"{_number(projection.expected_prevented_churn)} customers "
            f"(fractional expectation)"
        ),
        (
            "  expected saved value: "
            f"{'n/a (no value assumption)' if projection.expected_saved_value is None else _number(projection.expected_saved_value, digits=2) + ' ' + unit}"
        ),
        (
            "  net benefit: "
            f"{'n/a' if projection.net_benefit is None else _number(projection.net_benefit, digits=2) + ' ' + unit}; "
            f"ROI: {'n/a' if projection.roi is None else f'{projection.roi:.2%}'}"
        ),
        (
            "  cost per expected prevented churn: "
            f"{'n/a' if projection.cost_per_expected_prevented_churn is None else _number(projection.cost_per_expected_prevented_churn, digits=2) + ' ' + unit}"
        ),
        (
            "  break-even effectiveness: "
            f"{'n/a' if projection.break_even_effectiveness is None else _percent(projection.break_even_effectiveness)}; "
            f"targeting lift vs random: "
            f"{'n/a' if projection.targeting_lift is None else f'{projection.targeting_lift:.3f}x'}"
        ),
    ]
    if result.exclusion_counts:
        lines.append("Exclusions:")
        for reason, count in result.exclusion_counts.items():
            if count:
                lines.append(f"  {reason}: {count}")
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in result.warnings)
    lines.append("Disclaimer: " + result.disclaimers[0])
    return "\n".join(lines)


def _assumption_rows(result: CampaignResult) -> list[tuple[str, str]]:
    scenario = result.scenario
    assumptions = scenario.assumptions
    unit = assumptions.value_unit
    rows = [
        ("Scenario name", scenario.name),
        ("Description", scenario.description or "(none supplied)"),
        ("Budget", f"{_number(scenario.budget, digits=2)} {unit}"),
        (
            "Intervention cost per contacted customer",
            f"{_number(assumptions.cost_per_contact, digits=2)} {unit}",
        ),
        (
            "Assumed effectiveness (prevents churn among would-be churners)",
            _percent(assumptions.effectiveness),
        ),
        (
            "Assumed monthly value per retained customer",
            "not supplied" if assumptions.monthly_value is None
            else f"{_number(assumptions.monthly_value, digits=2)} {unit}",
        ),
        ("Value horizon", f"{_number(assumptions.horizon_months, digits=1)} months"),
        ("Value unit", unit),
        ("Targeting strategy", scenario.targeting.strategy),
        (
            "Minimum risk probability",
            "not configured" if scenario.targeting.min_risk_probability is None
            else _number(scenario.targeting.min_risk_probability, digits=4),
        ),
        (
            "Targeted risk tiers",
            ", ".join(scenario.targeting.risk_tiers) or "all tiers",
        ),
        (
            "Maximum contacts",
            "not configured" if scenario.targeting.max_contacts is None
            else str(scenario.targeting.max_contacts),
        ),
        (
            "Per-customer cost overrides",
            f"{len(scenario.cost_overrides)} configured"
            if scenario.cost_overrides else "none",
        ),
    ]
    return rows


def render_report(
    result: CampaignResult,
    *,
    comparison: ScenarioComparison | None = None,
    sensitivity: SensitivityAnalysis | None = None,
    plots: Sequence[dict[str, str]] | None = None,
    manifest_path: str = "",
) -> str:
    """Markdown campaign report with assumptions, projections and provenance."""
    assumptions = result.scenario.assumptions
    unit = assumptions.value_unit
    projection = result.projection
    allocation = result.allocation
    lines: list[str] = [
        f"# Campaign simulation — {_md(result.scenario.name)}",
        "",
        f"Scenario id: `{result.scenario_id or '(unsaved)'}`  ",
        f"Generated: {result.generated_at}  ",
        f"Schema: phase 10 / v{result.provenance.get('simulation_schema_version', '1.0')}",
        "",
        "> **These are hypothetical projections.** " + result.disclaimers[0],
        "> " + result.disclaimers[1],
        "",
        "## 1. Campaign configuration and assumptions",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    for label, value in _assumption_rows(result):
        lines.append(f"| {label} | {_md(value)} |")
    lines.extend(
        [
            "",
            f"Assumed effectiveness means the assumed probability that an "
            f"intervention prevents churn for a customer who would otherwise "
            f"churn. It is an input, not a measured uplift.",
            "",
            "## 2. Targeting, eligibility and allocation",
            "",
            "| Measure | Value |",
            "| --- | --- |",
            f"| Assessed customers | {result.assessed_count} |",
            f"| Eligible customers | {result.eligible_count} |",
            f"| Selected customers | {len(result.selected)} |",
            f"| Excluded total | {result.excluded_count} |",
            f"| Excluded: below risk threshold | "
            f"{result.exclusion_counts.get('below_risk_threshold', 0)} |",
            f"| Excluded: tier not targeted | "
            f"{result.exclusion_counts.get('tier_not_targeted', 0)} |",
            f"| Untargeted because of budget | {allocation.unaffordable_count} |",
            f"| Untargeted because of contact limit | {allocation.contact_limited_count} |",
            f"| Maximum affordable interventions | "
            f"{'unbounded (zero cost)' if allocation.max_affordable is None else allocation.max_affordable} |",
            f"| Allocated spend | {_number(allocation.allocated_spend, digits=2)} {unit} |",
            f"| Remaining budget | {_number(allocation.remaining_budget, digits=2)} {unit} |",
            f"| Budget utilisation | {_percent(allocation.utilization)} |",
            f"| Counts reconcile | {'yes' if allocation.to_dict()['counts_reconcile'] else 'no'} |",
            "",
            "Selection order is descending churn probability with the record "
            "reference as a stable tie-break, so identical inputs produce "
            "identical selections.",
            "",
            "## 3. Hypothetical outcome projection",
            "",
            "| Metric | Value | Basis |",
            "| --- | --- | --- |",
            f"| Aggregate churn probability of selected set | "
            f"{_number(projection.aggregate_churn_probability)} | "
            f"sum of Phase 9 probabilities |",
            f"| Expected prevented churn | "
            f"{_number(projection.expected_prevented_churn)} | "
            f"sum(p_i x e), a fractional expectation |",
            f"| Expected saved value | "
            f"{'n/a' if projection.expected_saved_value is None else _number(projection.expected_saved_value, digits=2) + ' ' + unit} | "
            f"expanded churn x assumed value |",
            f"| Campaign cost | {_number(projection.campaign_cost, digits=2)} {unit} | "
            f"sum of allocated intervention costs |",
            f"| Net benefit | "
            f"{'n/a' if projection.net_benefit is None else _number(projection.net_benefit, digits=2) + ' ' + unit} | "
            f"saved value - cost |",
            f"| ROI | {'n/a' if projection.roi is None else _percent(projection.roi)} | "
            f"net benefit / cost |",
            f"| Cost per expected prevented churn | "
            f"{'n/a' if projection.cost_per_expected_prevented_churn is None else _number(projection.cost_per_expected_prevented_churn, digits=2) + ' ' + unit} | "
            f"cost / expected prevented churn |",
            f"| Break-even effectiveness | "
            f"{'n/a' if projection.break_even_effectiveness is None else _percent(projection.break_even_effectiveness)} | "
            f"cost / (at-risk mass x assumed value) |",
            f"| Random-contact baseline | "
            f"{'n/a' if projection.random_contact_expected_prevented_churn is None else _number(projection.random_contact_expected_prevented_churn)} | "
            f"same count contacted at random |",
            f"| Targeting lift vs random | "
            f"{'n/a' if projection.targeting_lift is None else f'{projection.targeting_lift:.3f}x'} | "
            f"targeted / random |",
            "",
        ]
    )
    if projection.undefined_metrics:
        lines.extend(["Undefined metrics and why:", ""])
        for name, reason in projection.undefined_metrics.items():
            lines.append(f"- `{name}`: {reason}")
        lines.append("")
    if comparison is not None:
        lines.extend(["## 4. Scenario comparison", ""])
        lines.append(
            "Every projection column below is hypothetical and assumption-dependent; "
            "a difference between scenarios is not measured campaign performance."
        )
        lines.append("")
        lines.append(render_comparison(comparison))
        lines.append("")
    section = 5 if comparison is not None else 4
    if sensitivity is not None:
        lines.extend([f"## {section}. Sensitivity analysis", ""])
        lines.append(
            "One assumption is varied at a time while the rest of the scenario "
            "is held fixed. Each row is a genuine recalculation, not an "
            "interpolation."
        )
        lines.append("")
        for series in sensitivity.series:
            lines.append(render_sensitivity_series(series, unit=unit))
            lines.append("")
        if sensitivity.warnings:
            lines.append("Sensitivity warnings:")
            lines.append("")
            lines.extend(f"- {_md(warning)}" for warning in sensitivity.warnings)
            lines.append("")
        section += 1
    if plots:
        lines.extend([f"## {section}. Visualizations", ""])
        for item in plots:
            label = item.get("label") or item.get("path", "")
            path = (item.get("path") or "").replace("\\", "/")
            lines.append(f"- **{label}**: `{path}`")
            if path:
                lines.append(f"  ![{label}]({path})")
        lines.append("")
        section += 1
    lines.extend(
        [
            f"## {section}. Warnings",
            "",
        ]
    )
    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("- None.")
    lines.extend(["", f"## {section + 1}. Inputs, provenance and limitations", ""])
    lines.append(f"- Risk run: `{result.context.run_id or 'unknown'}`")
    lines.append(
        f"- Model: {result.context.model_family or 'unknown'} "
        f"(`{result.context.model_run_id or 'unknown'}`)"
    )
    lines.append(
        f"- Model SHA-256: `{result.context.model_sha256 or 'not recorded'}`"
    )
    lines.append(
        f"- Preprocessing run: `{result.context.preprocessing_run_id or 'unknown'}`"
    )
    lines.append(
        f"- Dataset SHA-256: `{result.context.dataset_sha256 or 'not recorded'}`"
    )
    lines.append(
        f"- Split fingerprint: `{result.context.split_fingerprint or 'not recorded'}`"
    )
    lines.append(f"- Package version: `{result.provenance.get('code_version') or 'unknown'}`")
    lines.append(
        "- Seed: none; selection and arithmetic are deterministic and nothing is sampled"
    )
    lines.append(
        f"- Risk tier boundaries: {list(result.context.tier_boundaries)}"
    )
    lines.append(f"- Scenario fingerprint: `{result.scenario.fingerprint()}`")
    lines.append(
        "- Probabilities are Phase 9 model estimates; campaign assumptions are "
        "user-supplied; projected outcomes are arithmetic consequences."
    )
    if manifest_path:
        lines.append(f"- Run manifest: `{manifest_path}`")
    lines.extend(["", "Limitations:", ""])
    lines.extend(f"- {item}" for item in result.limitations)
    return "\n".join(lines) + "\n"


def render_comparison(comparison: ScenarioComparison) -> str:
    """Markdown table with assumptions visible next to each projection."""
    lines = [
        "| Scenario | Strategy | Budget | Cost/contact | Effectiveness | Selected | "
        "Spend | Remaining | Expected prevented churn | Cost per prevented churn | ROI |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison.rows:
        lines.append(
            f"| {_md(row.name)} | {_md(row.strategy)} | {_number(row.budget, digits=2)} | "
            f"{_number(row.cost_per_contact, digits=2)} | "
            f"{_percent(row.effectiveness)} | {row.selected_count} | "
            f"{_number(row.allocated_spend, digits=2)} | "
            f"{_number(row.remaining_budget, digits=2)} | "
            f"{_number(row.expected_prevented_churn)} | "
            f"{'n/a' if row.cost_per_expected_prevented_churn is None else _number(row.cost_per_expected_prevented_churn, digits=2)} | "
            f"{'n/a' if row.roi is None else _percent(row.roi)} |"
        )
    lines.append("")
    units = sorted({row.value_unit for row in comparison.rows})
    if len(units) == 1:
        lines.append(f"Monetary columns use the `{units[0]}` value unit.")
    else:
        lines.append(
            "**Value units differ across scenarios ("
            + ", ".join(units)
            + "); monetary columns are not directly comparable.**"
        )
    lines.extend(
        [
            "",
            "| Scenario | Min risk | Target tiers | Max contacts | "
            "Monthly value | Horizon (months) |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison.rows:
        lines.append(
            f"| {_md(row.name)} | "
            f"{'not set' if row.min_risk_probability is None else _number(row.min_risk_probability, digits=4)} | "
            f"{', '.join(row.risk_tiers) or 'all tiers'} | "
            f"{'not set' if row.max_contacts is None else row.max_contacts} | "
            f"{_metric(row.monthly_value, row.value_unit)} | "
            f"{_number(row.horizon_months, digits=1)} |"
        )
    lines.append("")
    if comparison.shared_inputs:
        lines.append(
            f"All scenarios use risk run `{comparison.risk_run_id}` and the same "
            "assessed population."
        )
    else:
        lines.append("**Inputs differ between scenarios:**")
        lines.extend(f"- {item}" for item in comparison.input_differences)
    if comparison.differing_assumptions:
        lines.append(
            "Differing assumptions: "
            + ", ".join(comparison.differing_assumptions)
            + "."
        )
    lines.extend(f"- {note}" for note in comparison.notes)
    if comparison.limitations:
        lines.extend(["", "Limitations:", ""])
        lines.extend(f"- {item}" for item in comparison.limitations)
    return "\n".join(lines)


def render_sensitivity_series(series: SensitivitySeries, *, unit: str) -> str:
    lines = [
        f"### {series.dimension} (baseline {_number(series.baseline_value)})",
        "",
        f"| {series.dimension} | Selected | Spend ({unit}) | Remaining ({unit}) | "
        f"Expected prevented churn | Cost per prevented churn ({unit}) | Utilisation |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in series.points:
        marker = " (baseline)" if point.is_baseline else ""
        lines.append(
            f"| {_number(point.value)}{marker} | {point.selected_count} | "
            f"{_number(point.allocated_spend, digits=2)} | "
            f"{_number(point.remaining_budget, digits=2)} | "
            f"{_number(point.expected_prevented_churn)} | "
            f"{'n/a' if point.cost_per_expected_prevented_churn is None else _number(point.cost_per_expected_prevented_churn, digits=2)} | "
            f"{_percent(point.utilization)} |"
        )
    return "\n".join(lines)


def render_sensitivity(analysis: SensitivityAnalysis, *, unit: str = "value_units") -> str:
    lines = [
        "Sensitivity analysis (one assumption varied at a time)",
    ]
    for series in analysis.series:
        lines.append(render_sensitivity_series(series, unit=unit))
        lines.append("")
    if analysis.warnings:
        lines.extend(f"- {warning}" for warning in analysis.warnings)
    return "\n".join(lines)


def render_comparison_rows(comparison: ScenarioComparison) -> list[dict[str, Any]]:
    """Flat rows for CSV persistence (assumptions remain attached)."""
    return [row.to_dict() for row in comparison.rows]


__all__ = [
    "render_comparison",
    "render_comparison_rows",
    "render_report",
    "render_sensitivity",
    "render_sensitivity_series",
    "render_summary",
]
