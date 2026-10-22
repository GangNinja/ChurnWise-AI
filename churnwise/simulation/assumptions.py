"""Defaults, validation ranges and sensitivity grids for campaign simulation.

Every default here is an **assumption**, not a measurement. Defaults are exposed
in the CLI help, the report and ``result.json`` so a reader can see exactly what
was assumed. Grids are plain ascending tuples of values with a documented unit.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from churnwise.simulation.contracts import (
    DEFAULT_BUDGET,
    DEFAULT_COST_PER_CONTACT,
    DEFAULT_EFFECTIVENESS,
    DEFAULT_HORIZON_MONTHS,
    DIMENSION_LABELS,
    MAX_BUDGET,
    MAX_VALUE,
)
from churnwise.simulation.errors import CODE_INVALID_CONFIG, SimulationError

DEFAULT_EFFECTIVENESS_GRID = (0.10, 0.20, 0.30, 0.40, 0.50)
DEFAULT_BUDGET_MULTIPLIERS = (0.5, 1.0, 1.5, 2.0)
DEFAULT_COST_MULTIPLIERS = (0.5, 1.0, 2.0)

ASSUMPTION_DEFAULTS = {
    "budget": DEFAULT_BUDGET,
    "cost_per_contact": DEFAULT_COST_PER_CONTACT,
    "effectiveness": DEFAULT_EFFECTIVENESS,
    "monthly_value": None,
    "horizon_months": DEFAULT_HORIZON_MONTHS,
}

RANGES = {
    "budget": (0.0, MAX_BUDGET),
    "cost_per_contact": (0.0, MAX_VALUE),
    "effectiveness": (0.0, 1.0),
    "monthly_value": (0.0, MAX_VALUE),
    "horizon_months": (1e-9, 1_200.0),
}

DEFAULT_FIXED_PARAMETERS_NOTE = (
    "Sensitivity varies one assumption at a time; budget, cost, effectiveness, "
    "targeting and value assumptions not being varied are held at the scenario values."
)


def parse_number_list(
    value: Any, name: str, *, minimum: float, maximum: float
) -> tuple[float, ...]:
    """Parse and validate a sequence or comma-separated string of numbers."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(" ", ",").split(",") if part.strip()]
        raw: Iterable[Any] = parts
    else:
        raw = value
    try:
        values = tuple(float(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise SimulationError(
            CODE_INVALID_CONFIG,
            f"{name} must be a sequence of numbers, got {value!r}",
            "Pass comma-separated numbers, e.g. 0.1,0.2,0.3.",
        ) from exc
    if len(values) < 2:
        raise SimulationError(
            CODE_INVALID_CONFIG,
            f"{name} needs at least two values to show a trend, got {len(values)}",
            "Pass at least two distinct numbers.",
        )
    if len(values) > 50:
        raise SimulationError(
            CODE_INVALID_CONFIG,
            f"{name} has {len(values)} values; at most 50 are supported",
            "Reduce the number of grid values.",
        )
    for item in values:
        if not math.isfinite(item) or not minimum <= item <= maximum:
            raise SimulationError(
                CODE_INVALID_CONFIG,
                f"{name} value {item!r} must be finite and within "
                f"[{minimum:g}, {maximum:g}]",
                "Correct the grid values and retry.",
            )
    return values


def effectiveness_grid(values: Any | None = None) -> tuple[float, ...]:
    """Assumed effectiveness values in [0, 1]."""
    return parse_number_list(
        DEFAULT_EFFECTIVENESS_GRID if values is None else values,
        "effectiveness grid",
        minimum=0.0,
        maximum=1.0,
    )


def budget_grid(values: Any | None = None, *, base_budget: float) -> tuple[float, ...]:
    """Absolute budgets; multipliers are resolved against the scenario budget."""
    resolved = (
        tuple(float(base_budget) * multiplier for multiplier in DEFAULT_BUDGET_MULTIPLIERS)
        if values is None
        else values
    )
    return parse_number_list(
        resolved,
        "budget grid",
        minimum=0.0,
        maximum=MAX_BUDGET,
    )


def cost_grid(
    values: Any | None = None, *, base_cost: float
) -> tuple[float, ...]:
    """Absolute per-contact intervention costs."""
    resolved = (
        tuple(float(base_cost) * multiplier for multiplier in DEFAULT_COST_MULTIPLIERS)
        if values is None
        else values
    )
    return parse_number_list(
        resolved,
        "cost_per_contact grid",
        minimum=0.0,
        maximum=MAX_VALUE,
    )


def dimension_unit(dimension: str, assumptions: Any) -> str:
    """Human-readable unit for one sensitivity dimension."""
    if dimension == "effectiveness":
        return "probability 0-1"
    if dimension == "budget":
        return str(assumptions.value_unit)
    if dimension == "cost_per_contact":
        return f"{assumptions.value_unit} per contacted customer"
    return ""


def describe_defaults() -> dict[str, Any]:
    """Documented defaults used when a field is omitted, with their ranges."""
    return {
        "defaults": dict(ASSUMPTION_DEFAULTS),
        "ranges": {key: list(value) for key, value in RANGES.items()},
        "sensitivity_defaults": {
            "effectiveness": list(DEFAULT_EFFECTIVENESS_GRID),
            "budget_multipliers": list(DEFAULT_BUDGET_MULTIPLIERS),
            "cost_multipliers": list(DEFAULT_COST_MULTIPLIERS),
        },
        "dimension_descriptions": dict(DIMENSION_LABELS),
        "note": "All defaults are user-facing assumptions, not measured values.",
    }


def sensitivity_dimensions(values: Sequence[str] | None = None) -> tuple[str, ...]:
    """Validate the requested sensitivity dimensions."""
    supported = tuple(DIMENSION_LABELS)
    requested = tuple(dict.fromkeys(supported if values is None else values))
    unknown = sorted(set(requested) - set(supported))
    if unknown:
        raise SimulationError(
            CODE_INVALID_CONFIG,
            f"unsupported sensitivity dimension(s): {', '.join(unknown)}",
            f"Use one or more of {list(supported)}.",
        )
    if not requested:
        raise SimulationError(
            CODE_INVALID_CONFIG,
            "at least one sensitivity dimension is required",
            f"Use one or more of {list(supported)}.",
        )
    return requested


__all__ = [
    "ASSUMPTION_DEFAULTS",
    "DEFAULT_BUDGET_MULTIPLIERS",
    "DEFAULT_COST_MULTIPLIERS",
    "DEFAULT_EFFECTIVENESS_GRID",
    "RANGES",
    "budget_grid",
    "cost_grid",
    "describe_defaults",
    "dimension_unit",
    "effectiveness_grid",
    "parse_number_list",
    "sensitivity_dimensions",
]
