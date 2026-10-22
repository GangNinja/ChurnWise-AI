"""Deterministic Phase 9 ranking, tiering and optional value-at-risk ordering."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterable, Sequence

from churnwise.intelligence.contracts import (
    RiskAssessment,
    RiskFilter,
    RiskThresholds,
    VALID_SORT_FIELDS,
)
from churnwise.intelligence.errors import CODE_INVALID_CONFIG, RiskError


def value_at_risk(probability: float, value: float, horizon_months: float) -> float:
    """Return ``p x value x horizon`` after validating every input."""
    for name, number in (
        ("probability", probability),
        ("value", value),
        ("horizon_months", horizon_months),
    ):
        try:
            parsed = float(number)
        except (TypeError, ValueError) as exc:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"{name} must be numeric for value-at-risk",
                "Supply a finite, non-negative value and horizon.",
            ) from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"{name} must be finite and non-negative for value-at-risk",
                "Supply a finite, non-negative value and horizon.",
            )
    return float(probability) * float(value) * float(horizon_months)


def attach_value_at_risk(
    records: Sequence[RiskAssessment],
    values: Sequence[float] | None,
    *,
    horizon_months: float,
) -> bool:
    """Attach expected loss when an explicit revenue proxy is supplied.

    Returns ``True`` when values were attached, ``False`` when the proxy is
    absent. The function never invents a missing value.
    """
    if values is None:
        return False
    if len(values) != len(records):
        raise RiskError(
            CODE_INVALID_CONFIG,
            f"value proxy has {len(values)} values for {len(records)} records",
            "Pass exactly one value for each scored customer.",
        )
    for item, value in zip(records, values):
        if value is None:
            item.value_at_risk = None
            item.value_horizon_months = float(horizon_months)
            continue
        item.value_at_risk = value_at_risk(
            item.churn_probability, value, horizon_months
        )
        item.value_horizon_months = float(horizon_months)
    return True


def _sort_key(item: RiskAssessment, sort_by: str) -> Any:
    reference = item.record_reference
    if sort_by == "probability":
        return (-float(item.churn_probability), reference, int(item.input_position))
    if sort_by == "value_at_risk":
        if item.value_at_risk is None:
            return (1, 0.0, reference, int(item.input_position))
        return (-float(item.value_at_risk), -float(item.churn_probability), reference)
    if sort_by == "record_reference":
        return (reference, int(item.input_position))
    if sort_by == "risk_tier":
        # Higher tiers first while keeping probability deterministic inside a tier.
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return (-order.get(item.risk_tier, -1), -float(item.churn_probability), reference)
    raise RiskError(
        CODE_INVALID_CONFIG,
        f"unknown sort field {sort_by!r}",
        f"Use one of {list(VALID_SORT_FIELDS)}.",
    )


def rank_records(
    records: Iterable[RiskAssessment],
    *,
    sort_by: str = "probability",
    descending: bool = True,
) -> list[RiskAssessment]:
    """Return a deterministic ranking with unique ranks and documented ties.

    Ties are broken by record reference, then input position. This makes the
    order stable across runs and avoids relying on DataFrame/Python hash order.
    """
    if sort_by not in VALID_SORT_FIELDS:
        raise RiskError(
            CODE_INVALID_CONFIG,
            f"unknown sort field {sort_by!r}",
            f"Use one of {list(VALID_SORT_FIELDS)}.",
        )
    items = list(records)
    if sort_by == "value_at_risk" and any(item.value_at_risk is None for item in items):
        raise RiskError(
            CODE_INVALID_CONFIG,
            "value-at-risk ordering requires a value proxy for every scored record",
            "Supply the proxy or sort by probability instead.",
        )
    keyed = sorted(items, key=lambda item: _sort_key(item, sort_by), reverse=not descending)
    return [replace(item, rank=index) for index, item in enumerate(keyed, start=1)]


def tier_counts(
    records: Sequence[RiskAssessment], thresholds: RiskThresholds
) -> dict[str, dict[str, Any]]:
    """Build tier counts that always sum to the scored population."""
    counts: dict[str, dict[str, Any]] = {}
    total = len(records)
    for label in thresholds.labels:
        count = sum(1 for item in records if item.risk_tier == label)
        counts[label] = {
            "count": int(count),
            "pct": (float(count) / total) if total else 0.0,
        }
    return counts


def filter_records(
    records: Iterable[RiskAssessment],
    criteria: RiskFilter,
    *,
    labels: Sequence[str],
) -> list[RiskAssessment]:
    problems = criteria.validate(labels)
    if problems:
        raise RiskError(
            CODE_INVALID_CONFIG,
            "invalid risk filter: " + "; ".join(problems),
            "Correct the tier/probability/segment filters and retry.",
        )
    return [item for item in records if criteria.matches(item)]


def paginate(
    records: Sequence[RiskAssessment],
    *,
    limit: int,
    offset: int = 0,
    max_rows: int = 50_000,
) -> list[RiskAssessment]:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= max_rows:
        raise RiskError(
            CODE_INVALID_CONFIG,
            f"limit must be an integer between 1 and {max_rows}",
            "Pass a bounded page size.",
        )
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise RiskError(
            CODE_INVALID_CONFIG,
            "offset must be a non-negative integer",
            "Pass a non-negative page offset.",
        )
    return list(records[offset : offset + limit])


__all__ = [
    "attach_value_at_risk",
    "filter_records",
    "paginate",
    "rank_records",
    "tier_counts",
    "value_at_risk",
]
