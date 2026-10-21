"""Dataset profiling, canonical role mapping and capability detection.

The profile is descriptive only. It never imputes, scales, encodes, drops rows
from disk or changes the acquired artifact. Every flag carries a reason so a
human can audit why a column was assigned a role or excluded.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"

TARGET_HINTS: tuple[str, ...] = (
    "churn",
    "churned",
    "attrition",
    "cancel",
    "cancelled",
    "canceled",
    "exited",
    "retained",
    "subscription_status",
)

ID_NAME_PATTERN = re.compile(
    r"(^|_)(id|uuid|guid|index|rowid|row_id|customerid|customer_id|"
    r"account_number|phone_number|phone|msisdn|key)($|_)",
    re.IGNORECASE,
)

DATE_NAME_PATTERN = re.compile(
    r"(date|datetime|timestamp|_at$|_time$|start|end|created|updated|closed)",
    re.IGNORECASE,
)

LEAKAGE_NAME_PATTERN = re.compile(
    r"(churn[_\s-]?reason|churn[_\s-]?date|cancel|termination|terminated|"
    r"retention|offer|exit[_\s-]?date|closed|end[_\s-]?date|"
    r"status$|^status)",
    re.IGNORECASE,
)

CATEGORICAL_NAME_PATTERN = re.compile(
    r"(state|region|area[_\s-]?code|country|city|contract|plan|payment|"
    r"gender|segment|category|type|method|channel|status|grade|tier)",
    re.IGNORECASE,
)

NUMERIC_NAME_PATTERN = re.compile(
    r"(charge|amount|price|cost|total|balance|revenue|spend|salary|income|"
    r"minutes|usage|calls|tenure|age|quantity|qty|count|score|rate|value)",
    re.IGNORECASE,
)

TEXT_NAME_PATTERN = re.compile(
    r"(description|comment|note|reason|message|text|review|address|name)",
    re.IGNORECASE,
)

MONTHLY_REVENUE_PATTERN = re.compile(
    r"(monthly[_\s-]?(charge|fee|amount|revenue)|mrr|monthlyrevenue)", re.IGNORECASE
)

CAPABILITY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "has_tenure": (re.compile(r"tenure", re.IGNORECASE),),
    "has_monthly_charge": (re.compile(r"monthly.*(charge|fee|amount)", re.IGNORECASE),),
    "has_total_charges": (re.compile(r"^total[\s_-]?charges?$", re.IGNORECASE),),
    "has_contract": (re.compile(r"contract", re.IGNORECASE),),
    "has_payment_method": (re.compile(r"payment", re.IGNORECASE),),
    "has_internet_service": (re.compile(r"internet", re.IGNORECASE),),
    "has_support_calls": (
        re.compile(r"(service|support).*call", re.IGNORECASE),
        re.compile(r"call.*(service|support)", re.IGNORECASE),
    ),
    "has_usage_minutes": (
        re.compile(r"minutes", re.IGNORECASE),
        re.compile(r"usage", re.IGNORECASE),
    ),
    "has_plan": (re.compile(r"plan", re.IGNORECASE),),
    "has_senior_citizen": (re.compile(r"senior", re.IGNORECASE),),
    "has_gender": (re.compile(r"gender", re.IGNORECASE), re.compile(r"^sex$", re.IGNORECASE)),
}

CLASS_LABEL_VALUES = {
    "yes",
    "no",
    "y",
    "n",
    "true",
    "false",
    "1",
    "0",
    "churn",
    "churned",
    "not churned",
    "not_churned",
    "retained",
    "active",
    "inactive",
    "cancelled",
    "canceled",
    "exited",
    "left",
    "stayed",
    "renewed",
    "current",
}


def normalize_label(value: Any) -> str | None:
    """Canonical text form used for class counts, mappings and leakage checks."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, "g")
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "?", "na", "n/a"}:
        return None
    return text


def dedupe_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a frame with unique string labels plus the names duplicated.

    The loader relies on pandas to mangle repeated CSV headers, but the profiling
    and target-detection helpers are also called directly with in-memory frames.
    Select-by-label on such a frame returns a DataFrame, and Excel/JSON sources
    can carry non-string labels, so every public entry point normalises labels
    first instead of raising a pandas ``Grouper``/``AttributeError`` later.
    """
    names = [str(name).strip() for name in frame.columns]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if not duplicates and names == list(frame.columns):
        return frame, []
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique.append(f"{name}.{seen[name]}")
        else:
            seen[name] = 0
            unique.append(name)
    deduped = frame.copy()
    deduped.columns = unique
    return deduped, duplicates


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _json_number(value: float | int | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass
class TargetInterpretation:
    column: str | None = None
    classes: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    mapping: dict[str, int] = field(default_factory=dict)
    positive_label: str | None = None
    negative_label: str | None = None
    confidence: str = "low"
    method: str = ""
    reason: str | None = None
    ambiguous: bool = False
    candidates: list[dict[str, Any]] = field(default_factory=list)
    missing_count: int = 0
    missing_pct: float = 0.0

    @property
    def positive_count(self) -> int:
        if self.positive_label is None:
            return 0
        return int(self.counts.get(self.positive_label, 0))

    @property
    def observed_count(self) -> int:
        return int(sum(self.counts.values()))

    @property
    def positive_rate(self) -> float | None:
        if self.observed_count <= 0:
            return None
        return self.positive_count / self.observed_count

    @property
    def usable(self) -> bool:
        return (
            self.column is not None
            and not self.ambiguous
            and len(self.classes) == 2
            and len(self.mapping) == 2
            and self.positive_label is not None
            and self.confidence in {"high", "medium"}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "classes": list(self.classes),
            "counts": dict(self.counts),
            "mapping": dict(self.mapping),
            "positive_label": self.positive_label,
            "negative_label": self.negative_label,
            "positive_count": self.positive_count,
            "positive_rate": _json_number(self.positive_rate),
            "confidence": self.confidence,
            "method": self.method,
            "reason": self.reason,
            "ambiguous": self.ambiguous,
            "candidates": list(self.candidates),
            "missing_count": self.missing_count,
            "missing_pct": _json_number(self.missing_pct),
        }


@dataclass
class RoleMapping:
    target: str | None = None
    id: str | None = None
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    revenue_proxy: str | None = None
    capabilities: dict[str, bool] = field(default_factory=dict)
    capability_sources: dict[str, str | None] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "id": self.id,
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "dropped": list(self.dropped),
            "revenue_proxy": self.revenue_proxy,
            "capabilities": dict(self.capabilities),
            "capability_sources": dict(self.capability_sources),
        }


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    kind: str
    role: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_ratio: float
    constant: bool
    near_constant: bool
    numeric: dict[str, Any] | None = None
    top_values: list[dict[str, Any]] = field(default_factory=list)
    sample_values: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    duplicate_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "kind": self.kind,
            "role": self.role,
            "missing_count": self.missing_count,
            "missing_pct": _json_number(self.missing_pct),
            "unique_count": self.unique_count,
            "unique_ratio": _json_number(self.unique_ratio),
            "constant": self.constant,
            "near_constant": self.near_constant,
            "numeric": dict(self.numeric) if self.numeric else None,
            "top_values": list(self.top_values),
            "sample_values": list(self.sample_values),
            "flags": list(self.flags),
            "reasons": list(self.reasons),
            "duplicate_of": self.duplicate_of,
        }


@dataclass
class DatasetProfile:
    dataset_id: str
    format: str
    source_path: str
    n_rows: int
    n_cols: int
    memory_bytes: int
    missing_cells: int
    missing_cell_pct: float
    duplicate_rows: int
    duplicate_row_pct: float
    duplicate_columns: list[dict[str, str]]
    duplicate_headers: list[str]
    columns: list[ColumnProfile]
    target: TargetInterpretation
    roles: RoleMapping
    capabilities: dict[str, bool]
    capability_sources: dict[str, str | None]
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @property
    def n_features(self) -> int:
        return len(self.roles.features)

    def column(self, name: str | None) -> ColumnProfile | None:
        if name is None:
            return None
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "format": self.format,
            "source_path": self.source_path,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "memory_bytes": self.memory_bytes,
            "missing_cells": self.missing_cells,
            "missing_cell_pct": _json_number(self.missing_cell_pct),
            "duplicate_rows": self.duplicate_rows,
            "duplicate_row_pct": _json_number(self.duplicate_row_pct),
            "duplicate_columns": list(self.duplicate_columns),
            "duplicate_headers": list(self.duplicate_headers),
            "columns": [column.to_dict() for column in self.columns],
            "target": self.target.to_dict(),
            "roles": self.roles.to_dict(),
            "capabilities": dict(self.capabilities),
            "capability_sources": dict(self.capability_sources),
            "warnings": list(self.warnings),
        }


def _column_hash(series: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(series, index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def _numeric_statistics(series: pd.Series) -> dict[str, Any]:
    numeric = _numeric_series(series)
    finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    infinite_count = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    if finite.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "skew": None,
            "outlier_count": 0,
            "outlier_pct": 0.0,
            "infinite_count": infinite_count,
        }
    q25, q75 = finite.quantile([0.25, 0.75])
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr
    outliers = int(((finite < lower) | (finite > upper)).sum())
    return {
        "count": int(finite.size),
        "mean": _json_number(finite.mean()),
        "std": _json_number(finite.std()),
        "min": _json_number(finite.min()),
        "q25": _json_number(q25),
        "median": _json_number(finite.median()),
        "q75": _json_number(q75),
        "max": _json_number(finite.max()),
        "skew": _json_number(finite.skew()),
        "outlier_count": outliers,
        "outlier_pct": _json_number(outliers / finite.size),
        "infinite_count": infinite_count,
    }


def _top_values(series: pd.Series, limit: int = 5) -> list[dict[str, Any]]:
    labels = series.map(normalize_label).dropna()
    if labels.empty:
        return []
    counts = labels.value_counts().head(limit)
    total = int(labels.size)
    return [
        {"value": str(value), "count": int(count), "pct": _json_number(count / total)}
        for value, count in counts.items()
    ]


def _samples(series: pd.Series, limit: int = 3) -> list[str]:
    labels = series.map(normalize_label).dropna()
    return [str(value) for value in labels.head(limit).tolist()]


def _numeric_text_profile(labels: pd.Series) -> tuple[float, bool]:
    text = labels.astype("string").str.strip().str.replace(",", "", regex=False)
    text = text.str.replace(r"^[\$€£]\s*", "", regex=True)
    parsed = pd.to_numeric(text, errors="coerce")
    return float(parsed.notna().mean()), bool(text.str.contains(r"\d\.\d", regex=True).any())


def _numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.replace(",", "", regex=False)
    text = text.str.replace(r"^[\$€£]\s*", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _kind(series: pd.Series, name: str, unique_count: int, n_rows: int) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        if unique_count <= 2 or CATEGORICAL_NAME_PATTERN.search(name):
            return "categorical"
        return "numeric"
    if DATE_NAME_PATTERN.search(name):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return "datetime"
    labels = series.map(normalize_label).dropna()
    if labels.empty:
        return "unknown"
    if CATEGORICAL_NAME_PATTERN.search(name):
        return "categorical"
    if TEXT_NAME_PATTERN.search(name) and labels.nunique() > 10:
        return "text"
    if unique_count > 30:
        parsed_ratio, decimal = _numeric_text_profile(labels)
        if parsed_ratio >= 0.95 and (decimal or NUMERIC_NAME_PATTERN.search(name)):
            return "numeric"
    if unique_count <= 30:
        return "categorical"
    if unique_count / max(n_rows, 1) > 0.5:
        return "text"
    return "categorical"


def _is_id_like(
    series: pd.Series,
    name: str,
    unique_count: int,
    n_rows: int,
    *,
    kind: str = "unknown",
) -> bool:
    if n_rows == 0:
        return False
    ratio = unique_count / n_rows
    named = bool(ID_NAME_PATTERN.search(name))
    if named and (ratio >= 0.8 or name.lower() in {"id", "customerid", "customer_id", "uuid"}):
        return True
    if kind in {"numeric", "datetime"}:
        return False
    if ratio < 0.995:
        return False
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    if pd.api.types.is_integer_dtype(series):
        return True
    return False


def _target_binary(series: pd.Series, target: TargetInterpretation) -> pd.Series | None:
    if target.column is None or not target.mapping:
        return None
    labels = series.map(normalize_label)
    return labels.map(target.mapping).astype("float")


def _deterministic_target_relation(
    values: pd.Series, target_binary: pd.Series, *, unique_count: int
) -> bool:
    if target_binary.nunique(dropna=True) < 2 or unique_count > 20:
        return False
    frame = pd.DataFrame({"value": values.map(normalize_label), "target": target_binary}).dropna()
    if frame.empty or frame["value"].nunique() < 2:
        return False
    return bool(frame.groupby("value")["target"].nunique().max() == 1)


def _leakage_reasons(
    series: pd.Series,
    name: str,
    target: TargetInterpretation,
    target_binary: pd.Series | None,
    *,
    unique_count: int,
    id_like: bool,
    kind: str,
) -> list[str]:
    if target.column is not None and name == target.column:
        return []
    reasons: list[str] = []
    if LEAKAGE_NAME_PATTERN.search(name):
        reasons.append("name suggests a post-outcome, cancellation or retention field")
    if id_like:
        return reasons
    if target_binary is not None:
        labels = series.map(normalize_label)
        if labels.nunique(dropna=True) >= 2:
            try:
                same = labels.map(target.mapping) if target.mapping else None
            except Exception:
                same = None
            if same is not None and same.notna().all() and same.nunique(dropna=True) >= 2:
                if bool((same.astype("float") == target_binary.fillna(-1)).all()):
                    reasons.append("values duplicate the target")
                elif bool((same.astype("float") == (1 - target_binary.fillna(-1))).all()):
                    reasons.append("values are the inverse of the target")
        if kind in {"categorical", "boolean"} and _deterministic_target_relation(
            series, target_binary, unique_count=unique_count
        ):
            reasons.append("each value maps deterministically to one target class")
    return reasons


def _capability_scan(columns: list[str]) -> tuple[dict[str, bool], dict[str, str | None]]:
    capabilities: dict[str, bool] = {}
    sources: dict[str, str | None] = {}
    for capability, patterns in CAPABILITY_PATTERNS.items():
        match = next(
            (name for pattern in patterns for name in columns if pattern.search(name)),
            None,
        )
        capabilities[capability] = match is not None
        sources[capability] = match
    return capabilities, sources


def build_profile(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    format: str,
    source_path: str,
    target: TargetInterpretation,
    duplicate_headers: list[str] | None = None,
) -> DatasetProfile:
    """Profile a loaded frame without mutating it or the acquired file."""
    frame, inline_duplicates = dedupe_columns(frame)
    if inline_duplicates:
        duplicate_headers = sorted(
            {*(duplicate_headers or []), *inline_duplicates}
        )
    n_rows = int(len(frame))
    n_cols = int(frame.shape[1])
    missing_cells = int(frame.isna().sum().sum())
    duplicate_rows = int(frame.duplicated().sum())
    target_binary = (
        _target_binary(frame[target.column], target) if target.column in frame.columns else None
    )
    target_column = target.column

    hashes: dict[str, str] = {}
    duplicate_columns: list[dict[str, str]] = []
    column_meta: dict[str, dict[str, Any]] = {}
    for name in frame.columns:
        series = frame[name]
        labels = series.map(normalize_label)
        unique_count = int(labels.nunique(dropna=True))
        digest = _column_hash(series)
        duplicate_of = hashes.get(digest)
        if duplicate_of is None:
            hashes[digest] = name
        else:
            duplicate_columns.append({"column": name, "duplicate_of": duplicate_of})
        kind = "target" if name == target_column else _kind(series, name, unique_count, n_rows)
        id_like = name != target_column and _is_id_like(
            series, name, unique_count, n_rows, kind=kind
        )
        reasons = _leakage_reasons(
            series,
            name,
            target,
            target_binary if target_column in frame.columns else None,
            unique_count=unique_count,
            id_like=id_like,
            kind=kind,
        )
        column_meta[name] = {
            "series": series,
            "labels": labels,
            "unique_count": unique_count,
            "digest": digest,
            "duplicate_of": duplicate_of,
            "kind": kind,
            "id_like": id_like,
            "leakage_reasons": reasons,
        }

    id_candidates = [
        name for name, meta in column_meta.items() if meta["id_like"] and name != target_column
    ]
    id_candidates.sort(
        key=lambda name: (
            bool(ID_NAME_PATTERN.search(name)),
            column_meta[name]["unique_count"],
            name,
        ),
        reverse=True,
    )
    chosen_id = id_candidates[0] if id_candidates else None

    columns: list[ColumnProfile] = []
    numeric: list[str] = []
    categorical: list[str] = []
    dropped: list[dict[str, Any]] = []
    warnings: list[str] = []

    for name in frame.columns:
        meta = column_meta[name]
        series: pd.Series = meta["series"]
        labels: pd.Series = meta["labels"]
        unique_count = meta["unique_count"]
        missing_count = int(series.isna().sum())
        missing_pct = missing_count / n_rows if n_rows else 0.0
        constant = unique_count <= 1
        top = labels.value_counts(dropna=True)
        near_constant = bool(
            unique_count > 1 and not top.empty and int(top.iloc[0]) / max(int(labels.notna().sum()), 1) >= 0.995
        )
        flags: list[str] = []
        reasons: list[str] = list(meta["leakage_reasons"])
        if meta["id_like"]:
            flags.append("id_like")
            reasons.append("unique key / identifier-like column")
        if constant:
            flags.append("constant")
        if near_constant:
            flags.append("near_constant")
        if missing_pct > 0.5:
            flags.append("high_missing")
            reasons.append(f"{missing_pct:.1%} of values are missing")
        if meta["kind"] == "categorical" and unique_count > 30:
            flags.append("high_cardinality")
            reasons.append(f"{unique_count} categories exceeds the 30-level limit")
        if meta["kind"] == "text":
            flags.append("text_like")
            reasons.append("free-text / very high cardinality")
        if meta["kind"] == "datetime":
            flags.append("date_like")
            reasons.append("date/time values are not used as raw features")
        if meta["duplicate_of"] is not None:
            flags.append("duplicate_column")
            reasons.append(f"duplicates column {meta['duplicate_of']!r}")
        if meta["leakage_reasons"]:
            flags.append("leakage_risk")

        numeric_stats = (
            _numeric_statistics(series)
            if meta["kind"] == "numeric" and name != target_column
            else None
        )
        column = ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            kind=meta["kind"],
            role="target" if name == target_column else "dropped",
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            unique_ratio=(unique_count / n_rows if n_rows else 0.0),
            constant=constant,
            near_constant=near_constant,
            numeric=numeric_stats,
            top_values=_top_values(series) if meta["kind"] != "numeric" else [],
            sample_values=_samples(series),
            flags=flags,
            reasons=reasons,
            duplicate_of=meta["duplicate_of"],
        )

        if name == target_column:
            columns.append(column)
            continue
        if chosen_id is not None and name == chosen_id:
            column.role = "id"
            columns.append(column)
            continue
        if meta["id_like"]:
            column.role = "dropped"
            dropped.append({"column": name, "reason": "additional identifier-like column"})
            columns.append(column)
            continue
        if meta["duplicate_of"] is not None:
            dropped.append({"column": name, "reason": f"duplicates {meta['duplicate_of']!r}"})
            columns.append(column)
            continue
        if meta["leakage_reasons"]:
            dropped.append({"column": name, "reason": "; ".join(meta["leakage_reasons"])})
            columns.append(column)
            continue
        if meta["kind"] == "datetime":
            dropped.append({"column": name, "reason": "date/time column excluded in Phase 4"})
            columns.append(column)
            continue
        if constant:
            dropped.append({"column": name, "reason": "constant column"})
            columns.append(column)
            continue
        if missing_pct > 0.5:
            dropped.append({"column": name, "reason": "more than 50% missing"})
            columns.append(column)
            continue
        if meta["kind"] == "text":
            dropped.append({"column": name, "reason": "free-text / high-cardinality text"})
            columns.append(column)
            continue
        if meta["kind"] == "categorical" and unique_count > 30:
            dropped.append({"column": name, "reason": f"{unique_count} categories (limit 30)"})
            columns.append(column)
            continue
        if meta["kind"] == "numeric":
            column.role = "numeric"
            numeric.append(name)
        else:
            column.role = "categorical"
            categorical.append(name)
        columns.append(column)

    numeric_candidates = [name for name in numeric if name != target_column]
    revenue_proxy = next(
        (
            name
            for name in numeric_candidates
            if MONTHLY_REVENUE_PATTERN.search(name)
        ),
        None,
    )
    capabilities, capability_sources = _capability_scan(numeric + categorical)
    capabilities["has_revenue_proxy"] = revenue_proxy is not None
    capability_sources["has_revenue_proxy"] = revenue_proxy

    if duplicate_headers:
        warnings.append(
            "duplicate header names were present and were made unique by the loader: "
            + ", ".join(duplicate_headers)
        )
    if not numeric:
        warnings.append("no numeric predictor remains after role mapping")
    if not numeric and not categorical:
        warnings.append("no usable predictor remains after role mapping")
    for name in categorical:
        levels = column_meta[name]["unique_count"]
        if levels > 30:
            warnings.append(f"categorical column {name!r} has {levels} levels")

    roles = RoleMapping(
        target=target_column,
        id=chosen_id,
        numeric=numeric,
        categorical=categorical,
        dropped=dropped,
        revenue_proxy=revenue_proxy,
        capabilities=capabilities,
        capability_sources=capability_sources,
    )
    return DatasetProfile(
        dataset_id=dataset_id,
        format=format,
        source_path=source_path,
        n_rows=n_rows,
        n_cols=n_cols,
        memory_bytes=int(frame.memory_usage(deep=True).sum()),
        missing_cells=missing_cells,
        missing_cell_pct=(missing_cells / (n_rows * n_cols) if n_rows and n_cols else 0.0),
        duplicate_rows=duplicate_rows,
        duplicate_row_pct=(duplicate_rows / n_rows if n_rows else 0.0),
        duplicate_columns=duplicate_columns,
        duplicate_headers=list(duplicate_headers or []),
        columns=columns,
        target=target,
        roles=roles,
        capabilities=capabilities,
        capability_sources=capability_sources,
        warnings=warnings,
    )


def empty_profile(
    *,
    dataset_id: str,
    format: str,
    source_path: str,
    target: TargetInterpretation | None = None,
) -> DatasetProfile:
    """A structurally empty profile used when no frame can be loaded."""
    interpretation = target or TargetInterpretation(reason="dataset could not be loaded")
    return DatasetProfile(
        dataset_id=dataset_id,
        format=format,
        source_path=source_path,
        n_rows=0,
        n_cols=0,
        memory_bytes=0,
        missing_cells=0,
        missing_cell_pct=0.0,
        duplicate_rows=0,
        duplicate_row_pct=0.0,
        duplicate_columns=[],
        duplicate_headers=[],
        columns=[],
        target=interpretation,
        roles=RoleMapping(target=interpretation.column),
        capabilities={},
        capability_sources={},
        warnings=["dataset could not be loaded"],
    )
