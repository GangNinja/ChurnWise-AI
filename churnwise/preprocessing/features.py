"""Capability-guarded derived features for Phase 5.

A derived feature is only built when Phase 4's capability matrix names a source
column for it. The formula id is persisted in the schema so inference can
recompute exactly the same columns without refitting anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from churnwise.eda.profile import normalize_label
from churnwise.preprocessing.contracts import PreprocessConfig

TENURE_BANDS = (-np.inf, 6, 12, 24, 48, np.inf)
TENURE_LABELS = ("0-6", "6-12", "12-24", "24-48", "48+")
USAGE_BANDS = (-np.inf, 100, 200, 300, np.inf)
USAGE_LABELS = ("0-100", "100-200", "200-300", "300+")


@dataclass
class DerivedResult:
    frame: pd.DataFrame
    applied: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _tenure_years(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    return _numeric(frame, spec["sources"][0]) / 12.0


def _tenure_band(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    tenure = _numeric(frame, spec["sources"][0])
    binned = pd.cut(tenure, bins=list(TENURE_BANDS), labels=list(TENURE_LABELS), right=True)
    return binned.astype("object")


def _avg_monthly_charge(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    total = _numeric(frame, spec["sources"][0])
    tenure = _numeric(frame, spec["sources"][1]).clip(lower=1.0)
    return total / tenure


def _support_calls_per_year(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    calls = _numeric(frame, spec["sources"][0])
    years = (_numeric(frame, spec["sources"][1]) / 12.0).clip(lower=1.0)
    return calls / years


def _usage_band(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    usage = _numeric(frame, spec["sources"][0])
    binned = pd.cut(usage, bins=list(USAGE_BANDS), labels=list(USAGE_LABELS), right=True)
    return binned.astype("object")


def _support_calls_per_usage_hour(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    calls = _numeric(frame, spec["sources"][0])
    hours = (_numeric(frame, spec["sources"][1]) / 60.0).clip(lower=1.0)
    return calls / hours


FORMULAS: dict[str, Callable[[pd.DataFrame, dict[str, Any]], pd.Series]] = {
    "tenure_years": _tenure_years,
    "tenure_band": _tenure_band,
    "avg_monthly_charge": _avg_monthly_charge,
    "support_calls_per_year": _support_calls_per_year,
    "usage_band": _usage_band,
    "support_calls_per_usage_hour": _support_calls_per_usage_hour,
}


# Static derived-feature catalogue. Every entry names the Phase 4 capability
# and the role each source column must play; the catalogue is resolved against
# the profile so unevidenced features are reported instead of silently missing.
CATALOGUE: tuple[dict[str, Any], ...] = (
    {
        "name": "tenure_years",
        "formula": "tenure_years",
        "sources": (("has_tenure", "tenure"),),
        "dtype": "numeric",
        "description": "tenure in years",
    },
    {
        "name": "tenure_band",
        "formula": "tenure_band",
        "sources": (("has_tenure", "tenure"),),
        "dtype": "categorical",
        "description": "tenure grouped into 0-6, 6-12, 12-24, 24-48 and 48+ month bands",
    },
    {
        "name": "avg_monthly_charge",
        "formula": "avg_monthly_charge",
        "sources": (("has_total_charges", "total_charges"), ("has_tenure", "tenure")),
        "dtype": "numeric",
        "description": "total charges divided by tenure in months",
    },
    {
        "name": "support_calls_per_year",
        "formula": "support_calls_per_year",
        "sources": (("has_support_calls", "support_calls"), ("has_tenure", "tenure")),
        "dtype": "numeric",
        "description": "support calls divided by tenure in years",
    },
    {
        "name": "usage_band",
        "formula": "usage_band",
        "sources": (("has_usage_minutes", "usage_minutes"),),
        "dtype": "categorical",
        "description": "usage minutes grouped into 0-100, 100-200, 200-300 and 300+ bands",
    },
    {
        "name": "support_calls_per_usage_hour",
        "formula": "support_calls_per_usage_hour",
        "sources": (("has_support_calls", "support_calls"), ("has_usage_minutes", "usage_minutes")),
        "dtype": "numeric",
        "description": "support calls divided by usage hours",
    },
)


def _resolve_spec(
    entry: dict[str, Any], capabilities: dict[str, bool], sources: dict[str, str | None]
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve a catalogue entry into a persisted spec, or explain why not."""
    unresolved: list[str] = []
    for capability, _role in entry["sources"]:
        if not capabilities.get(capability) or not sources.get(capability):
            unresolved.append(capability)
    if unresolved:
        return None, (
            "Phase 4 did not evidence the required capability: " + ", ".join(unresolved)
        )
    spec = {
        "name": entry["name"],
        "capability": entry["sources"][0][0],
        "formula": entry["formula"],
        "sources": [sources[capability] for capability, _role in entry["sources"]],
        "source_roles": [role for _capability, role in entry["sources"]],
        "dtype": entry["dtype"],
        "description": entry["description"],
    }
    return spec, None


def _candidate_specs(
    capabilities: dict[str, bool], sources: dict[str, str | None]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the resolvable derived specs plus a skip record for every other."""
    specs: list[dict[str, Any]] = []
    unevidenced: list[dict[str, Any]] = []
    for entry in CATALOGUE:
        spec, reason = _resolve_spec(entry, capabilities, sources)
        if spec is not None:
            specs.append(spec)
        else:
            unevidenced.append(
                {"feature": entry["name"], "capability": entry["sources"][0][0], "reason": reason}
            )
    return specs, unevidenced


def _target_binary(frame: pd.DataFrame, target: Any) -> pd.Series | None:
    column = getattr(target, "column", None)
    mapping = getattr(target, "mapping", None) or {}
    if column is None or not mapping or column not in frame.columns:
        return None
    return frame[column].map(normalize_label).map(mapping).astype("float64")


def _deterministically_predicts_target(
    values: pd.Series, target_binary: pd.Series | None
) -> bool:
    """True when a column is effectively a renamed copy of the target."""
    if target_binary is None:
        return False
    labelled = values.map(normalize_label)
    unique = labelled.nunique(dropna=True)
    if unique < 2 or unique > 20:
        return False
    paired = pd.DataFrame({"value": labelled, "target": target_binary}).dropna()
    if paired.empty or paired["target"].nunique() < 2:
        return False
    return bool((paired.groupby("value", observed=True)["target"].nunique() == 1).all())


def _source_verification(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    target_binary: pd.Series | None,
) -> str | None:
    """Reject a source column that is effectively a renamed target.

    Phase 4 flags and drops most leakage-risk columns, but a profile can still
    name a source that reproduces the label. Building a derived feature from
    such a column would copy the target into the feature matrix.
    """
    for source in spec["sources"]:
        if _deterministically_predicts_target(frame[source], target_binary):
            return (
                f"source column {source!r} deterministically reproduces the "
                "target and is treated as a renamed label"
            )
    return None


def derive_features(
    frame: pd.DataFrame,
    profile: Any,
    config: PreprocessConfig,
) -> DerivedResult:
    """Add the evidenced derived features; record every skip with a reason."""
    result_frame = frame.copy()
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    capabilities = dict(getattr(profile, "capabilities", {}) or {})
    sources = dict(getattr(profile.roles, "capability_sources", {}) or {})
    target_binary = _target_binary(frame, getattr(profile, "target", None))
    candidates, unevidenced = _candidate_specs(capabilities, sources)
    skipped.extend(unevidenced)

    if not config.derived_features:
        skipped = [
            {
                "feature": entry["name"],
                "capability": entry["sources"][0][0],
                "reason": "derived features are disabled in config",
            }
            for entry in CATALOGUE
        ]
        return DerivedResult(frame=result_frame, applied=applied, skipped=skipped)

    for spec in candidates:
        name = spec["name"]
        if name in result_frame.columns:
            skipped.append(
                {
                    "feature": name,
                    "capability": spec["capability"],
                    "reason": "a column with this name already exists",
                }
            )
            continue
        missing = [source for source in spec["sources"] if source not in result_frame.columns]
        if missing:
            skipped.append(
                {
                    "feature": name,
                    "capability": spec["capability"],
                    "reason": f"source column(s) missing: {', '.join(missing)}",
                }
            )
            continue
        rejection = _source_verification(result_frame, spec, target_binary)
        if rejection:
            skipped.append(
                {"feature": name, "capability": spec["capability"], "reason": rejection}
            )
            continue
        try:
            values = FORMULAS[spec["formula"]](result_frame, spec)
        except Exception as exc:  # noqa: BLE001 - reported, never silently ignored
            skipped.append(
                {
                    "feature": name,
                    "capability": spec["capability"],
                    "reason": f"formula failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        result_frame[name] = values
        applied.append(spec)

    return DerivedResult(frame=result_frame, applied=applied, skipped=skipped)


def apply_derived(frame: pd.DataFrame, specs: list[dict[str, Any]]) -> pd.DataFrame:
    """Recompute persisted derived features for inference without refitting."""
    result = frame.copy()
    for spec in specs:
        name = spec["name"]
        if name in result.columns:
            continue
        formula = FORMULAS.get(spec.get("formula", ""))
        if formula is None:
            continue
        missing = [source for source in spec.get("sources", []) if source not in result.columns]
        if missing:
            continue
        result[name] = formula(result, spec)
    return result
