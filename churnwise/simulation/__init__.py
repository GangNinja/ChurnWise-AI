"""Phase 10 retention-campaign simulation.

Public entry points:

* :class:`SimulationService` — load Phase 9 risk tables, run scenarios,
  compare scenarios, run sensitivity grids and persist artifacts.
* :class:`CampaignScenario` / :class:`CampaignAssumptions` /
  :class:`TargetingConfig` — validated, reproducible campaign configuration.
* :func:`run_campaign`, :func:`allocate_budget`, :func:`project_outcomes` —
  the deterministic engine (selection, spend, expected prevented churn).
* :func:`compare_scenarios`, :func:`run_sensitivity` — assumption-visible
  comparison and one-at-a-time sensitivity.
* :class:`CampaignResult`, :class:`ScenarioComparison`,
  :class:`SensitivityAnalysis` — stable structured return values for Phase 11.
* :class:`SimulationError` — ``CW-SIM-*`` failures with remediation.

Every projected outcome is hypothetical and assumption-driven. Nothing here
estimates causal uplift or claims that a campaign will prevent churn.
"""

from __future__ import annotations

from churnwise.simulation.assumptions import (
    ASSUMPTION_DEFAULTS,
    DEFAULT_BUDGET_MULTIPLIERS,
    DEFAULT_COST_MULTIPLIERS,
    DEFAULT_EFFECTIVENESS_GRID,
    describe_defaults,
    effectiveness_grid,
)
from churnwise.simulation.campaign import (
    allocate_budget,
    compare_scenarios,
    project_outcomes,
    resolve_eligibility,
    run_campaign,
    run_sensitivity,
)
from churnwise.simulation.contracts import (
    DEFAULT_BUDGET,
    DEFAULT_COST_PER_CONTACT,
    DEFAULT_EFFECTIVENESS,
    DEFAULT_HORIZON_MONTHS,
    DEFAULT_VALUE_UNIT,
    DISCLAIMERS,
    EXCLUSION_REASONS,
    LIMITATIONS,
    SIMULATION_SCHEMA_VERSION,
    STRATEGY_LABELS,
    VALID_STRATEGIES,
    BudgetAllocation,
    CampaignAssumptions,
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
    SimulationRun,
    TargetingConfig,
    assume_scenario,
    comparison_row,
)
from churnwise.simulation.errors import (
    CODE_ARTIFACT_FAILED,
    CODE_EXPORT_FAILED,
    CODE_INVALID_CONFIG,
    CODE_INVALID_INPUT,
    CODE_NO_RISK_RUN,
    SimulationError,
)
from churnwise.simulation.reporting import (
    render_comparison,
    render_report,
    render_sensitivity,
    render_summary,
)
from churnwise.simulation.service import (
    SimulationService,
    validate_scenario_id,
)

__all__ = [
    "ASSUMPTION_DEFAULTS",
    "BudgetAllocation",
    "CODE_ARTIFACT_FAILED",
    "CODE_EXPORT_FAILED",
    "CODE_INVALID_CONFIG",
    "CODE_INVALID_INPUT",
    "CODE_NO_RISK_RUN",
    "CampaignAssumptions",
    "CampaignResult",
    "CampaignScenario",
    "ComparisonRow",
    "DEFAULT_BUDGET",
    "DEFAULT_BUDGET_MULTIPLIERS",
    "DEFAULT_COST_MULTIPLIERS",
    "DEFAULT_COST_PER_CONTACT",
    "DEFAULT_EFFECTIVENESS",
    "DEFAULT_EFFECTIVENESS_GRID",
    "DEFAULT_HORIZON_MONTHS",
    "DEFAULT_VALUE_UNIT",
    "DISCLAIMERS",
    "EXCLUSION_REASONS",
    "ExcludedCustomer",
    "LIMITATIONS",
    "OutcomeProjection",
    "RiskContext",
    "SIMULATION_SCHEMA_VERSION",
    "STRATEGY_LABELS",
    "ScenarioComparison",
    "SelectedCustomer",
    "SensitivityAnalysis",
    "SensitivityPoint",
    "SensitivitySeries",
    "SimulationError",
    "SimulationRun",
    "SimulationService",
    "TargetingConfig",
    "VALID_STRATEGIES",
    "allocate_budget",
    "assume_scenario",
    "compare_scenarios",
    "comparison_row",
    "describe_defaults",
    "effectiveness_grid",
    "project_outcomes",
    "render_comparison",
    "render_report",
    "render_sensitivity",
    "render_summary",
    "resolve_eligibility",
    "run_campaign",
    "run_sensitivity",
    "validate_scenario_id",
]
