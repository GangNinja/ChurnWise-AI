"""Small pure helpers used by the dashboard.

These are presentation rules, not model logic: tier labels use the same
lower-inclusive/upper-exclusive boundaries documented by Phase 9; exports apply
the project's spreadsheet formula guard. No scoring, fitting or SHAP code lives
here.
"""

from __future__ import annotations

import io
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

TIER_LABELS_BY_COUNT = {
    2: ("low", "medium", "high"),
    3: ("low", "medium", "high", "critical"),
}


def tier_labels(boundaries: Sequence[float]) -> tuple[str, ...]:
    """Return the tier labels for a 2- or 3-boundary configuration."""

    return TIER_LABELS_BY_COUNT.get(len(boundaries), ())


def validate_boundaries(boundaries: Sequence[float]) -> str | None:
    """Return a human-readable problem, or ``None`` when the boundaries are valid."""

    values = tuple(float(value) for value in boundaries)
    if len(values) not in (2, 3):
        return "Use exactly two or three tier boundaries."
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in values):
        return "Each boundary must be between 0 and 1."
    if any(right <= left for left, right in zip(values, values[1:])):
        return "Boundaries must be strictly increasing."
    return None


def validation_status_kind(status: Any) -> str:
    """Map a Phase 4 gate status to the dashboard's visual tone.

    ``pass_with_warnings`` is deliberately amber rather than green: the gate is
    open, but at least one warning-level check did not pass.
    """

    normalized = str(status or "").strip().lower()
    return {
        "pass": "ok",
        "pass_with_warnings": "warn",
        "reject": "risk",
    }.get(normalized, "info")


def validation_check_counts(checks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count passed checks and non-passing checks by severity.

    The gate is decided by severity, so the UI must not report a warning-level
    check as a fatal failure. Counts are derived from the same detailed records
    shown in the table and always reconcile with ``len(checks)``.
    """

    passed = 0
    warnings = 0
    fatal = 0
    for check in checks:
        if check.get("passed"):
            passed += 1
        elif str(check.get("severity") or "").lower() == "fatal":
            fatal += 1
        else:
            warnings += 1
    return {
        "total": len(checks),
        "passed": passed,
        "warnings": warnings,
        "fatal": fatal,
    }


def tier_for(probability: float, boundaries: Sequence[float]) -> str:
    """Lower-inclusive tier assignment matching the Phase 9 contract."""

    labels = tier_labels(boundaries)
    if not labels:
        raise ValueError("tier boundaries must contain two or three values")
    value = float(probability)
    for boundary, label in zip(boundaries, labels):
        if value < float(boundary):
            return label
    return labels[-1]


def apply_tiers(frame: pd.DataFrame, boundaries: Sequence[float]) -> pd.DataFrame:
    """Copy a risk table and re-derive ``risk_tier`` from the supplied boundaries."""

    problem = validate_boundaries(boundaries)
    if problem:
        raise ValueError(problem)
    result = frame.copy()
    result["risk_tier"] = [
        tier_for(value, boundaries) for value in result["churn_probability"].astype(float)
    ]
    return result


def filter_risk_frame(
    frame: pd.DataFrame,
    *,
    tiers: Iterable[str] = (),
    min_probability: float = 0.0,
    max_probability: float = 1.0,
    reference_query: str = "",
) -> pd.DataFrame:
    """Pure filter over a risk table. Tier labels come from the caller's boundaries."""

    result = frame
    selected = tuple(tiers)
    if selected:
        result = result[result["risk_tier"].isin(selected)]
    result = result[
        (result["churn_probability"].astype(float) >= float(min_probability))
        & (result["churn_probability"].astype(float) <= float(max_probability))
    ]
    if reference_query.strip():
        needle = reference_query.strip().lower()
        result = result[
            result["record_reference"].astype(str).str.lower().str.contains(needle, regex=False)
        ]
    return result.sort_values(["rank"], ascending=True)


def tier_counts(frame: pd.DataFrame, boundaries: Sequence[float]) -> dict[str, int]:
    labels = tier_labels(boundaries)
    counts = {label: 0 for label in labels}
    for value in frame["churn_probability"].astype(float):
        label = tier_for(value, boundaries)
        counts[label] = counts.get(label, 0) + 1
    return counts


def probability_summary(frame: pd.DataFrame) -> dict[str, float]:
    values = frame["churn_probability"].astype(float)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def format_probability(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "not available"


def format_percent(value: Any, *, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return "not available"


def format_count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "not available"


def format_money(value: Any, unit: str = "value_units") -> str:
    if value is None:
        return "undefined"
    try:
        return f"{float(value):,.2f} {unit}".strip()
    except (TypeError, ValueError):
        return "undefined"


def safe_csv_cell(value: Any) -> Any:
    """Neutralise spreadsheet formulas in string cells (Phase 9 export safety)."""

    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@", "\t", "\r"}:
        return "'" + value
    return value


def frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(safe_csv_cell)
    buffer = io.StringIO()
    safe.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def frame_to_json_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_json(orient="records", indent=2).encode("utf-8")


def dict_to_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def nested_get(payload: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = payload or {}
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def flatten_selected_customers(result: Mapping[str, Any] | None, limit: int = 100) -> pd.DataFrame:
    rows = nested_get(result, "selected_customers", default=[]) or []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.head(limit)


__all__ = [
    "TIER_LABELS_BY_COUNT",
    "apply_tiers",
    "dict_to_json_bytes",
    "filter_risk_frame",
    "flatten_selected_customers",
    "format_count",
    "format_money",
    "format_percent",
    "format_probability",
    "frame_to_csv_bytes",
    "frame_to_json_bytes",
    "nested_get",
    "probability_summary",
    "safe_csv_cell",
    "tier_counts",
    "tier_for",
    "tier_labels",
    "validate_boundaries",
    "validation_check_counts",
    "validation_status_kind",
]
