"""Stable Phase 10 contracts: scenarios, selection, allocation and projections.

Three kinds of information are deliberately kept apart:

* **measured model output** — ``churn_probability``, tier, rank and model
  provenance come from the Phase 9 risk run and are never recomputed here;
* **user-supplied assumptions** — budget, intervention cost, assumed
  effectiveness, per-customer value and horizon are inputs, not measurements;
* **projections** — expected prevented churn, expected saved value, net benefit,
  ROI and break-even are arithmetic consequences of those assumptions. They are
  never observed retention results and are never described as causal uplift.

Monetary figures use a caller-supplied unit label (``value_unit``). When no real
currency is supplied the default ``value_units`` is an abstract, consistent unit
for budget, cost and customer value; the report states that explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from churnwise.intelligence.contracts import (
    DEFAULT_TIER_BOUNDARIES,
    TIER_LABELS,
    RiskTable,
    RiskThresholds,
)
from churnwise.simulation.errors import (
    CODE_INVALID_CONFIG,
    CODE_INVALID_INPUT,
    SimulationError,
)

SIMULATION_SCHEMA_VERSION = "1.0"

VALID_STRATEGIES = (
    "highest_risk",
    "risk_threshold",
    "risk_category",
    "budget_constrained",
)
STRATEGY_LABELS = {
    "highest_risk": "highest churn risk first",
    "risk_threshold": "risk threshold, highest risk first",
    "risk_category": "configured risk categories, highest risk first",
    "budget_constrained": "budget-constrained risk order (stops at the first unaffordable customer)",
}
EXCLUSION_REASONS = {
    "below_risk_threshold": "churn probability below the configured minimum",
    "tier_not_targeted": "risk tier is not in the targeted set",
    "budget_exhausted": "remaining budget could not cover the intervention cost",
    "contact_limit": "campaign contact limit reached",
}
DIMENSION_LABELS = {
    "effectiveness": "assumed intervention effectiveness",
    "budget": "campaign budget",
    "cost_per_contact": "intervention cost per contacted customer",
}

DEFAULT_BUDGET = 1_000.0
DEFAULT_COST_PER_CONTACT = 5.0
DEFAULT_EFFECTIVENESS = 0.30
DEFAULT_HORIZON_MONTHS = 6.0
DEFAULT_VALUE_UNIT = "value_units"
MAX_BUDGET = 1_000_000_000_000.0
MAX_CONTACTS = 10_000_000
MAX_VALUE = 1_000_000_000.0

DISCLAIMERS = (
    "All saved-customer, saved-value, net-benefit, ROI and break-even figures "
    "are hypothetical projections derived from user-supplied assumptions; they "
    "are not observed retention results.",
    "Assumed effectiveness is an input, not a measured uplift. This simulator "
    "does not estimate causal intervention effectiveness and does not prove "
    "that a campaign will prevent churn.",
    "Churn probabilities are Phase 9 model estimates, not guaranteed outcomes, "
    "and are not re-estimated by this simulation.",
)

LIMITATIONS = (
    "Expected prevented churn is a fractional expectation, not a count of "
    "customers who were actually retained.",
    "Effectiveness is applied uniformly to every selected customer; no "
    "segment-specific or heterogeneous uplift is modelled.",
    "The random-contact baseline assumes contact is drawn at random from the "
    "same eligible pool and ignores saturation and contact fatigue.",
    "Sensitivity analysis varies one assumption at a time; interactions "
    "between budget, cost and effectiveness are not explored.",
    "Cost overrides, when supplied, are assumptions about per-customer "
    "intervention cost and are not derived from customer records.",
)


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _plain_number(
    value: Any,
    name: str,
    problems: list[str],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    """Parse a finite number, appending an actionable problem instead of raising."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        problems.append(f"{name} must be a finite number, got {value!r}")
        return None
    if isinstance(value, bool) or not math.isfinite(parsed):
        problems.append(f"{name} must be a finite number, got {value!r}")
        return None
    if minimum is not None and parsed < minimum:
        problems.append(f"{name} must be >= {minimum:g}, got {parsed:g}")
    if maximum is not None and parsed > maximum:
        problems.append(f"{name} must be <= {maximum:g}, got {parsed:g}")
    return parsed


def _probability(
    value: Any, name: str, problems: list[str]
) -> float | None:
    return _plain_number(value, name, problems, minimum=0.0, maximum=1.0)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignAssumptions:
    """Explicit, user-supplied campaign assumptions.

    ``effectiveness`` is the assumed probability that a campaign intervention
    prevents churn for a customer who would otherwise churn. ``monthly_value``
    is the assumed value of keeping one customer for one month; it is optional
    and is never inferred from the dataset.
    """

    cost_per_contact: float = DEFAULT_COST_PER_CONTACT
    effectiveness: float = DEFAULT_EFFECTIVENESS
    monthly_value: float | None = None
    horizon_months: float = DEFAULT_HORIZON_MONTHS
    value_unit: str = DEFAULT_VALUE_UNIT

    def validate(self) -> list[str]:
        problems: list[str] = []
        _plain_number(
            self.cost_per_contact,
            "cost_per_contact",
            problems,
            minimum=0.0,
            maximum=MAX_VALUE,
        )
        _probability(self.effectiveness, "effectiveness", problems)
        if self.monthly_value is not None:
            _plain_number(
                self.monthly_value,
                "monthly_value",
                problems,
                minimum=0.0,
                maximum=MAX_VALUE,
            )
        _plain_number(
            self.horizon_months,
            "horizon_months",
            problems,
            minimum=1e-9,
            maximum=1_200.0,
        )
        if not isinstance(self.value_unit, str) or not self.value_unit.strip():
            problems.append("value_unit must be a non-empty label such as 'USD' or 'value_units'")
        elif len(self.value_unit.strip()) > 24:
            problems.append("value_unit must be at most 24 characters")
        return problems

    def validate_or_raise(self) -> None:
        problems = self.validate()
        if problems:
            raise SimulationError(
                CODE_INVALID_CONFIG,
                "invalid campaign assumptions: " + "; ".join(problems),
                "Use a finite non-negative budget/cost/value, effectiveness in "
                "[0, 1] and a positive horizon, then retry.",
            )

    @property
    def average_value(self) -> float | None:
        """Assumed value of one prevented churn over the configured horizon."""
        if self.monthly_value is None:
            return None
        return float(self.monthly_value) * float(self.horizon_months)

    @property
    def financial_metrics_available(self) -> bool:
        return self.average_value is not None

    def payload(self) -> dict[str, Any]:
        """The assumption inputs only; this is the round-trippable form."""
        return {
            "cost_per_contact": float(self.cost_per_contact),
            "effectiveness": float(self.effectiveness),
            "monthly_value": None if self.monthly_value is None else float(self.monthly_value),
            "horizon_months": float(self.horizon_months),
            "value_unit": str(self.value_unit),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_per_contact": float(self.cost_per_contact),
            "effectiveness": float(self.effectiveness),
            "monthly_value": None if self.monthly_value is None else float(self.monthly_value),
            "horizon_months": float(self.horizon_months),
            "average_value_per_prevented_churn": self.average_value,
            "value_unit": str(self.value_unit),
            "financial_metrics_available": self.financial_metrics_available,
            "effectiveness_definition": (
                "assumed probability that an intervention prevents churn for a "
                "customer who would otherwise churn"
            ),
            "semantics": "user-supplied assumptions, not measured uplift",
        }


@dataclass(frozen=True)
class TargetingConfig:
    """Who is eligible and how the selection is ordered.

    Tier names are validated against the risk table's own tier labels at run
    time; this class only rejects unknown labels and impossible combinations.
    """

    strategy: str = "highest_risk"
    min_risk_probability: float | None = None
    risk_tiers: tuple[str, ...] = ()
    max_contacts: int | None = None

    def __post_init__(self) -> None:
        tiers = self.risk_tiers
        if isinstance(tiers, str):
            tiers = tuple(part.strip() for part in tiers.replace(" ", ",").split(",") if part.strip())
        else:
            tiers = tuple(str(part).strip() for part in tiers if str(part).strip())
        object.__setattr__(self, "risk_tiers", tiers)

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.strategy not in VALID_STRATEGIES:
            problems.append(
                f"strategy must be one of {list(VALID_STRATEGIES)}, got {self.strategy!r}"
            )
        if self.min_risk_probability is not None:
            _probability(
                self.min_risk_probability, "min_risk_probability", problems
            )
        unknown = sorted(set(self.risk_tiers) - set(TIER_LABELS))
        if unknown:
            problems.append(
                f"unknown risk tiers {unknown}; expected labels from {list(TIER_LABELS)}"
            )
        if self.max_contacts is not None and (
            not isinstance(self.max_contacts, int)
            or isinstance(self.max_contacts, bool)
            or not 1 <= self.max_contacts <= MAX_CONTACTS
        ):
            problems.append(
                f"max_contacts must be an integer between 1 and {MAX_CONTACTS} or None"
            )
        if self.strategy == "risk_threshold" and self.min_risk_probability is None:
            problems.append(
                "the risk_threshold strategy requires min_risk_probability"
            )
        if self.strategy == "risk_category" and not self.risk_tiers:
            problems.append("the risk_category strategy requires at least one risk tier")
        return problems

    def validate_or_raise(self) -> None:
        problems = self.validate()
        if problems:
            raise SimulationError(
                CODE_INVALID_CONFIG,
                "invalid targeting configuration: " + "; ".join(problems),
                "Correct the targeting strategy, threshold, tiers or contact limit.",
            )

    def payload(self) -> dict[str, Any]:
        """The targeting inputs only; this is the round-trippable form."""
        return {
            "strategy": self.strategy,
            "min_risk_probability": (
                None
                if self.min_risk_probability is None
                else float(self.min_risk_probability)
            ),
            "risk_tiers": list(self.risk_tiers),
            "max_contacts": (
                None if self.max_contacts is None else int(self.max_contacts)
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "strategy_description": STRATEGY_LABELS.get(self.strategy, ""),
            "min_risk_probability": (
                None
                if self.min_risk_probability is None
                else float(self.min_risk_probability)
            ),
            "risk_tiers": list(self.risk_tiers),
            "max_contacts": (
                None if self.max_contacts is None else int(self.max_contacts)
            ),
            "ordering": "descending churn probability, then record reference",
        }


@dataclass(frozen=True)
class CampaignScenario:
    """A complete, validated, reproducible campaign configuration."""

    name: str
    budget: float = DEFAULT_BUDGET
    assumptions: CampaignAssumptions = field(default_factory=CampaignAssumptions)
    targeting: TargetingConfig = field(default_factory=TargetingConfig)
    description: str = ""
    cost_overrides: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.assumptions, CampaignAssumptions):
            object.__setattr__(
                self, "assumptions", CampaignAssumptions(**self.assumptions)  # type: ignore[arg-type]
            )
        if not isinstance(self.targeting, TargetingConfig):
            object.__setattr__(
                self, "targeting", TargetingConfig(**self.targeting)  # type: ignore[arg-type]
            )
        object.__setattr__(self, "cost_overrides", dict(self.cost_overrides or {}))

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not isinstance(self.name, str) or not self.name.strip():
            problems.append("scenario name must be a non-empty string")
        elif len(self.name.strip()) > 120:
            problems.append("scenario name must be at most 120 characters")
        if not isinstance(self.description, str):
            problems.append("scenario description must be a string")
        _plain_number(self.budget, "budget", problems, minimum=0.0, maximum=MAX_BUDGET)
        problems.extend(self.assumptions.validate())
        problems.extend(self.targeting.validate())
        for reference, value in self.cost_overrides.items():
            if not isinstance(reference, str) or not reference.strip():
                problems.append("cost_overrides keys must be non-empty record references")
                continue
            _plain_number(
                value,
                f"cost_overrides[{reference!r}]",
                problems,
                minimum=0.0,
                maximum=MAX_VALUE,
            )
        return problems

    def validate_or_raise(self) -> None:
        problems = self.validate()
        if problems:
            raise SimulationError(
                CODE_INVALID_CONFIG,
                "invalid campaign scenario: " + "; ".join(problems),
                "Correct the scenario budget, assumptions, targeting rule or cost overrides.",
            )

    @property
    def slug(self) -> str:
        cleaned = [
            character.lower()
            if character.isascii() and character.isalnum()
            else "-"
            for character in self.name.strip()
        ]
        slug = "".join(cleaned)
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug.strip("-")[:48] or "scenario"

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name.strip(),
            "description": self.description.strip(),
            "budget": float(self.budget),
            "assumptions": self.assumptions.payload(),
            "targeting": self.targeting.payload(),
            "cost_overrides": {
                str(key): float(value)
                for key, value in sorted(self.cost_overrides.items())
            },
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "fingerprint": self.fingerprint()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CampaignScenario":
        """Rebuild a scenario from its JSON payload (unknown keys are rejected)."""
        if not isinstance(payload, Mapping):
            raise SimulationError(
                CODE_INVALID_INPUT,
                "scenario payload must be a JSON object",
                "Pass the object produced by CampaignScenario.to_dict().",
            )
        allowed = {"name", "description", "budget", "assumptions", "targeting", "cost_overrides"}
        unknown = sorted(set(payload) - allowed - {"fingerprint"})
        if unknown:
            raise SimulationError(
                CODE_INVALID_INPUT,
                f"scenario payload has unsupported key(s): {', '.join(unknown)}",
                "Remove the extra keys; assumptions and targeting are nested objects.",
            )
        assumptions_block = payload.get("assumptions") or {}
        targeting_block = payload.get("targeting") or {}
        if not isinstance(assumptions_block, Mapping) or not isinstance(targeting_block, Mapping):
            raise SimulationError(
                CODE_INVALID_INPUT,
                "scenario assumptions/targeting must be JSON objects",
                "Pass the object produced by CampaignScenario.to_dict().",
            )
        assumptions_allowed = {
            "cost_per_contact",
            "effectiveness",
            "monthly_value",
            "horizon_months",
            "value_unit",
        }
        targeting_allowed = {
            "strategy",
            "min_risk_probability",
            "risk_tiers",
            "max_contacts",
        }
        for block, allowed_keys, label in (
            (assumptions_block, assumptions_allowed, "assumptions"),
            (targeting_block, targeting_allowed, "targeting"),
        ):
            extra = sorted(set(block) - allowed_keys)
            if extra:
                raise SimulationError(
                    CODE_INVALID_INPUT,
                    f"scenario {label} has unsupported key(s): {', '.join(extra)}",
                    "Use only documented scenario fields.",
                )
        try:
            scenario = cls(
                name=str(payload.get("name") or ""),
                budget=payload.get("budget", DEFAULT_BUDGET),
                description=str(payload.get("description") or ""),
                assumptions=CampaignAssumptions(**assumptions_block),
                targeting=TargetingConfig(**targeting_block),
                cost_overrides=payload.get("cost_overrides") or {},
            )
        except TypeError as exc:
            raise SimulationError(
                CODE_INVALID_INPUT,
                f"scenario payload could not be read: {exc}",
                "Use the field names in CampaignScenario.to_dict().",
            ) from exc
        scenario.validate_or_raise()
        return scenario


# ---------------------------------------------------------------------------
# selection and allocation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskContext:
    """Provenance of the Phase 9 risk assessments used by a simulation."""

    run_id: str = ""
    generated_at: str = ""
    model_family: str = ""
    model_run_id: str = ""
    model_sha256: str | None = None
    preprocessing_run_id: str = ""
    dataset_sha256: str | None = None
    split_fingerprint: str | None = None
    classification_threshold: float = 0.5
    tier_boundaries: tuple[float, ...] = DEFAULT_TIER_BOUNDARIES
    tier_labels: tuple[str, ...] = TIER_LABELS
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @classmethod
    def from_table(cls, table: RiskTable, *, run_id: str | None = None) -> "RiskContext":
        provenance = table.provenance if isinstance(table.provenance, dict) else {}
        dataset_sha256 = provenance.get("dataset_sha256")
        split_fingerprint = provenance.get("split_fingerprint")
        return cls(
            run_id=run_id or table.run_id,
            generated_at=table.generated_at,
            model_family=table.model_family,
            model_run_id=table.model_run_id,
            model_sha256=table.model_sha256,
            preprocessing_run_id=table.preprocessing_run_id,
            dataset_sha256=(str(dataset_sha256) if dataset_sha256 else None),
            split_fingerprint=(str(split_fingerprint) if split_fingerprint else None),
            classification_threshold=float(table.classification_threshold),
            tier_boundaries=tuple(table.thresholds.boundaries),
            tier_labels=tuple(table.thresholds.labels),
            warnings=tuple(table.warnings),
            limitations=tuple(table.limitations),
        )

    @property
    def thresholds(self) -> RiskThresholds:
        return RiskThresholds(boundaries=tuple(self.tier_boundaries))

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": 9,
            "risk_run_id": self.run_id,
            "risk_generated_at": self.generated_at,
            "model": {
                "family": self.model_family,
                "model_run_id": self.model_run_id,
                "model_sha256": self.model_sha256,
                "preprocessing_run_id": self.preprocessing_run_id,
            },
            "dataset_sha256": self.dataset_sha256,
            "split_fingerprint": self.split_fingerprint,
            "classification_threshold": float(self.classification_threshold),
            "risk_thresholds": self.thresholds.to_dict(),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class SelectedCustomer:
    """One selected customer, with the projection attributable to them."""

    record_reference: str
    input_position: int
    churn_probability: float
    risk_tier: str
    rank: int | None
    intervention_cost: float
    expected_prevented_churn: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_reference": self.record_reference,
            "input_position": int(self.input_position),
            "churn_probability": float(self.churn_probability),
            "risk_tier": self.risk_tier,
            "risk_rank": None if self.rank is None else int(self.rank),
            "intervention_cost": float(self.intervention_cost),
            "expected_prevented_churn": float(self.expected_prevented_churn),
        }


@dataclass(frozen=True)
class ExcludedCustomer:
    """One customer who was assessed but not selected, with the reason."""

    record_reference: str
    reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in EXCLUSION_REASONS:
            raise SimulationError(
                CODE_INVALID_INPUT,
                f"unknown exclusion reason {self.reason!r}",
                f"Use one of {list(EXCLUSION_REASONS)}.",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_reference": self.record_reference,
            "reason": self.reason,
            "detail": self.detail or EXCLUSION_REASONS[self.reason],
        }


@dataclass
class BudgetAllocation:
    """Transparent budget arithmetic for one scenario."""

    budget: float
    cost_per_contact: float
    eligible_count: int
    selected_count: int
    allocated_spend: float
    remaining_budget: float
    max_affordable: int | None
    utilization: float | None
    unaffordable_count: int
    contact_limited_count: int
    zero_cost: bool
    notes: list[str] = field(default_factory=list)

    @property
    def utilizable(self) -> bool:
        return self.budget > 0 and not self.zero_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": float(self.budget),
            "cost_per_contact": float(self.cost_per_contact),
            "eligible_count": int(self.eligible_count),
            "selected_count": int(self.selected_count),
            "allocated_spend": float(self.allocated_spend),
            "remaining_budget": float(self.remaining_budget),
            "max_affordable": (
                None if self.max_affordable is None else int(self.max_affordable)
            ),
            "budget_utilization": None if self.utilization is None else float(self.utilization),
            "budget_utilization_is_fraction": True,
            "unaffordable_count": int(self.unaffordable_count),
            "contact_limited_count": int(self.contact_limited_count),
            "zero_cost_interventions": bool(self.zero_cost),
            "counts_reconcile": (
                int(self.eligible_count)
                == int(self.selected_count)
                + int(self.unaffordable_count)
                + int(self.contact_limited_count)
            ),
            "notes": list(self.notes),
        }


@dataclass
class OutcomeProjection:
    """Assumption-driven expected outcomes; every field is a projection."""

    aggregate_churn_probability: float
    expected_prevented_churn: float
    effectiveness: float
    campaign_cost: float
    assumed_average_value: float | None
    expected_saved_value: float | None
    net_benefit: float | None
    roi: float | None
    cost_per_expected_prevented_churn: float | None
    break_even_effectiveness: float | None
    random_contact_expected_prevented_churn: float | None
    targeting_lift: float | None
    financial_metrics_available: bool
    undefined_metrics: dict[str, str] = field(default_factory=dict)

    @property
    def expected_churn_count(self) -> float:
        """Expected number of churners in the selected set (a fractional count).

        The legacy JSON field ``aggregate_churn_probability`` stores this same
        sum of individual probabilities. It is a probability mass, not a
        fraction between zero and one; this alias makes the meaning explicit.
        """
        return float(self.aggregate_churn_probability)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_churn_probability": float(self.aggregate_churn_probability),
            "aggregate_churn_probability_is_probability_mass": True,
            "expected_churn_count": self.expected_churn_count,
            "expected_prevented_churn": float(self.expected_prevented_churn),
            "expected_prevented_churn_is_fractional": True,
            "assumed_effectiveness": float(self.effectiveness),
            "campaign_cost": float(self.campaign_cost),
            "assumed_average_value_per_prevented_churn": self.assumed_average_value,
            "expected_saved_value": self.expected_saved_value,
            "net_benefit": self.net_benefit,
            "roi": self.roi,
            "roi_is_ratio": True,
            "roi_definition": "net_benefit / campaign_cost (reports render it as a percentage)",
            "cost_per_expected_prevented_churn": self.cost_per_expected_prevented_churn,
            "break_even_effectiveness": self.break_even_effectiveness,
            "random_contact_expected_prevented_churn": self.random_contact_expected_prevented_churn,
            "targeting_lift_vs_random": self.targeting_lift,
            "financial_metrics_available": bool(self.financial_metrics_available),
            "undefined_metrics": dict(self.undefined_metrics),
            "projection": True,
            "observed_retention_result": False,
        }


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class CampaignResult:
    """One simulated scenario with inputs, projections and provenance."""

    scenario: CampaignScenario
    context: RiskContext
    allocation: BudgetAllocation
    projection: OutcomeProjection
    selected: list[SelectedCustomer] = field(default_factory=list)
    exclusions: list[ExcludedCustomer] = field(default_factory=list)
    exclusion_counts: dict[str, int] = field(default_factory=dict)
    generated_at: str = ""
    scenario_id: str = ""
    assessed_count: int = 0
    eligible_count: int = 0
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def excluded_count(self) -> int:
        return self.assessed_count - len(self.selected)

    @property
    def counts_reconcile(self) -> bool:
        return (
            len(self.selected) + len(self.exclusions) == self.assessed_count
            and len(self.selected) + sum(self.exclusion_counts.values())
            == self.assessed_count
        )

    def to_dict(self, *, include_selection: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "phase": 10,
            "scenario_id": self.scenario_id,
            "generated_at": self.generated_at,
            "scenario": self.scenario.to_dict(),
            "assumptions": self.scenario.assumptions.to_dict(),
            "targeting": self.scenario.targeting.to_dict(),
            "budget": float(self.scenario.budget),
            "input": self.context.to_dict(),
            "population": {
                "assessed": int(self.assessed_count),
                "eligible": int(self.eligible_count),
                "selected": len(self.selected),
                "excluded": int(self.excluded_count),
                "exclusion_counts": dict(self.exclusion_counts),
                "counts_reconcile": bool(self.counts_reconcile),
            },
            "allocation": self.allocation.to_dict(),
            "projection": self.projection.to_dict(),
            "selected_customers": (
                [item.to_dict() for item in self.selected] if include_selection else []
            ),
            "exclusions": [item.to_dict() for item in self.exclusions],
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "disclaimers": list(self.disclaimers),
            "provenance": dict(self.provenance),
            "hypothetical": True,
        }
        return payload


@dataclass(frozen=True)
class ComparisonRow:
    """One scenario's assumption-visible metrics inside a comparison."""

    scenario_id: str
    name: str
    strategy: str
    budget: float
    cost_per_contact: float
    effectiveness: float
    monthly_value: float | None
    horizon_months: float
    value_unit: str
    min_risk_probability: float | None
    risk_tiers: tuple[str, ...]
    max_contacts: int | None
    eligible_count: int
    selected_count: int
    allocated_spend: float
    remaining_budget: float
    utilization: float | None
    expected_prevented_churn: float
    cost_per_expected_prevented_churn: float | None
    expected_saved_value: float | None
    net_benefit: float | None
    roi: float | None
    break_even_effectiveness: float | None
    targeting_lift: float | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "targeting_strategy": self.strategy,
            "budget": float(self.budget),
            "cost_per_contact": float(self.cost_per_contact),
            "assumed_effectiveness": float(self.effectiveness),
            "monthly_value": self.monthly_value,
            "horizon_months": float(self.horizon_months),
            "value_unit": self.value_unit,
            "min_risk_probability": self.min_risk_probability,
            "risk_tiers": list(self.risk_tiers),
            "max_contacts": self.max_contacts,
            "eligible_count": int(self.eligible_count),
            "selected_count": int(self.selected_count),
            "allocated_spend": float(self.allocated_spend),
            "remaining_budget": float(self.remaining_budget),
            "budget_utilization": self.utilization,
            "expected_prevented_churn": float(self.expected_prevented_churn),
            "cost_per_expected_prevented_churn": self.cost_per_expected_prevented_churn,
            "expected_saved_value": self.expected_saved_value,
            "net_benefit": self.net_benefit,
            "roi": self.roi,
            "break_even_effectiveness": self.break_even_effectiveness,
            "targeting_lift_vs_random": self.targeting_lift,
            "warnings": list(self.warnings),
        }


@dataclass
class ScenarioComparison:
    """Side-by-side scenarios sharing one risk-assessment population."""

    rows: list[ComparisonRow] = field(default_factory=list)
    risk_run_id: str = ""
    shared_inputs: bool = True
    input_differences: list[str] = field(default_factory=list)
    differing_assumptions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "phase": 10,
            "risk_run_id": self.risk_run_id,
            "shared_risk_inputs": bool(self.shared_inputs),
            "input_differences": list(self.input_differences),
            "differing_assumptions": list(self.differing_assumptions),
            "rows": [row.to_dict() for row in self.rows],
            "notes": list(self.notes),
            "limitations": list(self.limitations),
            "disclaimers": list(self.disclaimers),
            "hypothetical": True,
        }


@dataclass(frozen=True)
class SensitivityPoint:
    """One recalculated scenario inside a one-at-a-time sensitivity series."""

    value: float
    is_baseline: bool
    eligible_count: int
    selected_count: int
    selection_fingerprint: str
    allocated_spend: float
    remaining_budget: float
    expected_prevented_churn: float
    expected_saved_value: float | None
    cost_per_expected_prevented_churn: float | None
    utilization: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": float(self.value),
            "is_baseline": bool(self.is_baseline),
            "eligible_count": int(self.eligible_count),
            "selected_count": int(self.selected_count),
            "selection_fingerprint": self.selection_fingerprint,
            "allocated_spend": float(self.allocated_spend),
            "remaining_budget": float(self.remaining_budget),
            "expected_prevented_churn": float(self.expected_prevented_churn),
            "expected_saved_value": self.expected_saved_value,
            "cost_per_expected_prevented_churn": self.cost_per_expected_prevented_churn,
            "budget_utilization": self.utilization,
        }


@dataclass
class SensitivitySeries:
    """All sensitivity points for one varied assumption, others held fixed."""

    dimension: str
    unit: str
    baseline_value: float
    points: list[SensitivityPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "description": DIMENSION_LABELS.get(self.dimension, self.dimension),
            "unit": self.unit,
            "baseline_value": float(self.baseline_value),
            "points": [point.to_dict() for point in self.points],
        }

    def csv_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for point in self.points:
            rows.append(
                {
                    "dimension": self.dimension,
                    "unit": self.unit,
                    "value": repr(float(point.value)),
                    "is_baseline": str(bool(point.is_baseline)).lower(),
                    "eligible_count": str(int(point.eligible_count)),
                    "selected_count": str(int(point.selected_count)),
                    "selection_fingerprint": point.selection_fingerprint,
                    "allocated_spend": repr(float(point.allocated_spend)),
                    "remaining_budget": repr(float(point.remaining_budget)),
                    "expected_prevented_churn": repr(
                        float(point.expected_prevented_churn)
                    ),
                    "expected_saved_value": (
                        ""
                        if point.expected_saved_value is None
                        else repr(float(point.expected_saved_value))
                    ),
                    "cost_per_expected_prevented_churn": (
                        ""
                        if point.cost_per_expected_prevented_churn is None
                        else repr(float(point.cost_per_expected_prevented_churn))
                    ),
                    "budget_utilization": (
                        ""
                        if point.utilization is None
                        else repr(float(point.utilization))
                    ),
                }
            )
        return rows


@dataclass
class SensitivityAnalysis:
    """One-at-a-time sensitivity of projections to campaign assumptions."""

    scenario_id: str
    series: list[SensitivitySeries] = field(default_factory=list)
    fixed_parameters: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "phase": 10,
            "scenario_id": self.scenario_id,
            "method": "one assumption varied at a time; all other parameters held fixed",
            "series": [item.to_dict() for item in self.series],
            "fixed_parameters": dict(self.fixed_parameters),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "disclaimers": list(self.disclaimers),
            "hypothetical": True,
        }

    def csv_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for series in self.series:
            rows.extend(series.csv_rows())
        return rows


@dataclass
class SimulationRun:
    """Persisted Phase 10 output for one primary scenario."""

    scenario_id: str
    result: CampaignResult
    comparison: ScenarioComparison | None = None
    sensitivity: SensitivityAnalysis | None = None
    paths: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SIMULATION_SCHEMA_VERSION,
            "phase": 10,
            "scenario_id": self.scenario_id,
            "result": self.result.to_dict(),
            "comparison": None if self.comparison is None else self.comparison.to_dict(),
            "sensitivity": None if self.sensitivity is None else self.sensitivity.to_dict(),
            "paths": dict(self.paths),
            "duration_seconds": float(self.duration_seconds),
            "warnings": list(self.warnings),
        }


def comparison_row(result: CampaignResult) -> ComparisonRow:
    """Flatten a result into the comparable fields, keeping assumptions visible."""
    scenario = result.scenario
    projection = result.projection
    allocation = result.allocation
    return ComparisonRow(
        scenario_id=result.scenario_id,
        name=scenario.name,
        strategy=scenario.targeting.strategy,
        budget=float(scenario.budget),
        cost_per_contact=float(scenario.assumptions.cost_per_contact),
        effectiveness=float(scenario.assumptions.effectiveness),
        monthly_value=scenario.assumptions.monthly_value,
        horizon_months=float(scenario.assumptions.horizon_months),
        value_unit=str(scenario.assumptions.value_unit),
        min_risk_probability=scenario.targeting.min_risk_probability,
        risk_tiers=tuple(scenario.targeting.risk_tiers),
        max_contacts=scenario.targeting.max_contacts,
        eligible_count=int(allocation.eligible_count),
        selected_count=int(allocation.selected_count),
        allocated_spend=float(allocation.allocated_spend),
        remaining_budget=float(allocation.remaining_budget),
        utilization=allocation.utilization,
        expected_prevented_churn=float(projection.expected_prevented_churn),
        cost_per_expected_prevented_churn=projection.cost_per_expected_prevented_churn,
        expected_saved_value=projection.expected_saved_value,
        net_benefit=projection.net_benefit,
        roi=projection.roi,
        break_even_effectiveness=projection.break_even_effectiveness,
        targeting_lift=projection.targeting_lift,
        warnings=tuple(result.warnings),
    )


def assume_scenario(
    name: str,
    *,
    budget: float = DEFAULT_BUDGET,
    cost_per_contact: float = DEFAULT_COST_PER_CONTACT,
    effectiveness: float = DEFAULT_EFFECTIVENESS,
    monthly_value: float | None = None,
    horizon_months: float = DEFAULT_HORIZON_MONTHS,
    value_unit: str = DEFAULT_VALUE_UNIT,
    strategy: str = "highest_risk",
    min_risk_probability: float | None = None,
    risk_tiers: Sequence[str] = (),
    max_contacts: int | None = None,
    description: str = "",
    cost_overrides: Mapping[str, float] | None = None,
) -> CampaignScenario:
    """Convenience constructor used by the CLI, tests and the dashboard."""
    return CampaignScenario(
        name=name,
        budget=budget,
        description=description,
        assumptions=CampaignAssumptions(
            cost_per_contact=cost_per_contact,
            effectiveness=effectiveness,
            monthly_value=monthly_value,
            horizon_months=horizon_months,
            value_unit=value_unit,
        ),
        targeting=TargetingConfig(
            strategy=strategy,
            min_risk_probability=min_risk_probability,
            risk_tiers=tuple(risk_tiers),
            max_contacts=max_contacts,
        ),
        cost_overrides=dict(cost_overrides or {}),
    )


__all__ = [
    "BudgetAllocation",
    "DEFAULT_BUDGET",
    "DEFAULT_COST_PER_CONTACT",
    "DEFAULT_EFFECTIVENESS",
    "DEFAULT_HORIZON_MONTHS",
    "DEFAULT_VALUE_UNIT",
    "DISCLAIMERS",
    "DIMENSION_LABELS",
    "EXCLUSION_REASONS",
    "LIMITATIONS",
    "MAX_BUDGET",
    "SIMULATION_SCHEMA_VERSION",
    "STRATEGY_LABELS",
    "VALID_STRATEGIES",
    "CampaignAssumptions",
    "CampaignResult",
    "CampaignScenario",
    "ComparisonRow",
    "ExcludedCustomer",
    "OutcomeProjection",
    "RiskContext",
    "ScenarioComparison",
    "SelectedCustomer",
    "SensitivityAnalysis",
    "SensitivityPoint",
    "SensitivitySeries",
    "SimulationRun",
    "TargetingConfig",
    "assume_scenario",
    "comparison_row",
]
