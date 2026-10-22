"""Stable Phase 9 contracts: assessments, thresholds, batches and runs.

The contracts keep four things separate on purpose:

* the **classification threshold** selects ``churn`` versus ``retained``;
* **risk tiers** are a business-facing categorisation of the probability;
* **rank** is a deterministic ordering within a scored population;
* **value at risk** is an optional expected-loss ordering when a revenue proxy
  is explicitly supplied.

A probability is a model estimate, never a guaranteed outcome. Drivers are SHAP
attributions of model behaviour, never causal effects.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from churnwise.intelligence.errors import CODE_INVALID_CONFIG, RiskError

RISK_SCHEMA_VERSION = "1.0"
DEFAULT_TIER_BOUNDARIES = (0.30, 0.50, 0.70)
TIER_LABELS = ("low", "medium", "high", "critical")
VALID_SORT_FIELDS = ("probability", "value_at_risk", "record_reference", "risk_tier")
VALID_EXPLAIN_MODES = ("none", "top", "all", "selected")
VALID_POPULATION_SPLITS = ("train", "val", "test", "all")
EXPORT_COLUMNS = (
    "record_reference",
    "input_position",
    "churn_probability",
    "churn_probability_pct",
    "predicted_class",
    "predicted_label",
    "risk_tier",
    "rank",
    "classification_threshold",
    "model_family",
    "model_run_id",
    "model_sha256",
    "preprocessing_run_id",
    "assessed_at",
    "positive_label",
    "observed_churn",
    "value_at_risk",
    "value_horizon_months",
    "explanation_status",
    "explanation_reason",
    "explanation_summary",
    "top_positive_drivers",
    "top_negative_drivers",
    "risk_thresholds",
    "warnings",
    "segment",
)


def _finite_probability(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RiskError(
            CODE_INVALID_CONFIG,
            f"{name} must be a finite probability, got {value!r}",
            "Use a number between 0 and 1.",
        ) from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RiskError(
            CODE_INVALID_CONFIG,
            f"{name} must be between 0 and 1, got {number!r}",
            "Use a number between 0 and 1.",
        )
    return float(number)


def _parse_boundaries(value: Any) -> tuple[float, ...]:
    if value is None:
        return tuple(DEFAULT_TIER_BOUNDARIES)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(" ", ",").split(",") if part.strip()]
        try:
            values = tuple(float(part) for part in parts)
        except ValueError as exc:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"risk tier boundaries are not numeric: {value!r}",
                "Pass two or three comma-separated probabilities, e.g. 0.30,0.50,0.70.",
            ) from exc
    else:
        try:
            values = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"risk tier boundaries are not a numeric sequence: {value!r}",
                "Pass two or three probabilities between 0 and 1.",
            ) from exc
    return values


@dataclass(frozen=True)
class RiskThresholds:
    """Ordered probability boundaries for low/medium/high(/critical) tiers.

    Boundary semantics are half-open on the lower edge:

    * ``p < b0`` -> ``low``
    * ``b0 <= p < b1`` -> ``medium``
    * ``b1 <= p < b2`` -> ``high``
    * ``p >= b2`` -> ``critical`` (when three boundaries are supplied)

    Two boundaries produce low/medium/high only. Tier boundaries are deliberately
    independent of the model's classification threshold.
    """

    boundaries: tuple[float, ...] = DEFAULT_TIER_BOUNDARIES

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundaries", _parse_boundaries(self.boundaries))

    def validate(self) -> list[str]:
        problems: list[str] = []
        values = self.boundaries
        if len(values) not in (2, 3):
            problems.append("exactly two or three risk tier boundaries are required")
        for value in values:
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                problems.append(f"tier boundary {value!r} must be strictly between 0 and 1")
        if any(right <= left for left, right in zip(values, values[1:])):
            problems.append("risk tier boundaries must be strictly increasing")
        return problems

    def validate_or_raise(self) -> None:
        problems = self.validate()
        if problems:
            raise RiskError(
                CODE_INVALID_CONFIG,
                "invalid risk tier configuration: " + "; ".join(problems),
                "Use two or three strictly increasing probabilities in (0, 1).",
            )

    @property
    def labels(self) -> tuple[str, ...]:
        return TIER_LABELS[: len(self.boundaries) + 1]

    def tier_for(self, probability: float) -> str:
        value = _finite_probability(probability, "churn probability")
        index = bisect.bisect_right(self.boundaries, value)
        return self.labels[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundaries": [float(value) for value in self.boundaries],
            "labels": list(self.labels),
            "semantics": "lower-inclusive, upper-exclusive; final tier is upper-inclusive",
        }

    @classmethod
    def from_value(cls, value: Any) -> "RiskThresholds":
        thresholds = cls(boundaries=_parse_boundaries(value))
        thresholds.validate_or_raise()
        return thresholds


@dataclass(frozen=True)
class RiskConfig:
    """Configuration for scoring, ranking, driver attachment and exports."""

    thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    classification_threshold: float | None = None
    batch_size: int = 200
    top_k: int = 5
    max_driver_rows: int = 1000
    population_split: str = "val"
    population_explain: str = "all"
    value_column: str | None = None
    value_horizon_months: float = 6.0
    reference_column: str | None = None
    segment_fields: tuple[str, ...] = ()
    max_export_rows: int = 50_000
    strict_record_types: bool = True
    version: str = "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, RiskThresholds):
            object.__setattr__(self, "thresholds", RiskThresholds.from_value(self.thresholds))
        segment_fields = self.segment_fields
        if isinstance(segment_fields, str):
            object.__setattr__(
                self,
                "segment_fields",
                tuple(part.strip() for part in segment_fields.split(",") if part.strip()),
            )
        else:
            object.__setattr__(
                self, "segment_fields", tuple(str(part) for part in segment_fields)
            )

    def validate(self) -> list[str]:
        problems = list(self.thresholds.validate())
        if self.classification_threshold is not None:
            try:
                _finite_probability(self.classification_threshold, "classification_threshold")
            except RiskError as exc:
                problems.append(exc.message)
        for name, value, low, high in (
            ("batch_size", self.batch_size, 1, 10_000),
            ("top_k", self.top_k, 1, 50),
            ("max_driver_rows", self.max_driver_rows, 0, 100_000),
            ("max_export_rows", self.max_export_rows, 1, 1_000_000),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                problems.append(f"{name} must be an integer between {low} and {high}")
        if self.population_split not in VALID_POPULATION_SPLITS:
            problems.append(f"population_split must be one of {list(VALID_POPULATION_SPLITS)}")
        if self.population_explain not in VALID_EXPLAIN_MODES:
            problems.append(f"population_explain must be one of {list(VALID_EXPLAIN_MODES)}")
        try:
            horizon = float(self.value_horizon_months)
        except (TypeError, ValueError):
            horizon = math.nan
        if not math.isfinite(horizon) or horizon <= 0:
            problems.append("value_horizon_months must be a positive finite number")
        if not isinstance(self.value_column, (str, type(None))) or (
            isinstance(self.value_column, str) and not self.value_column.strip()
        ):
            problems.append("value_column must be a non-empty column name or None")
        if not isinstance(self.reference_column, (str, type(None))) or (
            isinstance(self.reference_column, str) and not self.reference_column.strip()
        ):
            problems.append("reference_column must be a non-empty column name or None")
        if not isinstance(self.strict_record_types, bool):
            problems.append("strict_record_types must be a boolean")
        return problems

    def validate_or_raise(self) -> None:
        problems = self.validate()
        if problems:
            raise RiskError(
                CODE_INVALID_CONFIG,
                "invalid risk configuration: " + "; ".join(problems),
                "Correct the score/risk options and retry.",
            )

    def payload(self) -> dict[str, Any]:
        return {
            "thresholds": list(self.thresholds.boundaries),
            "classification_threshold": self.classification_threshold,
            "batch_size": int(self.batch_size),
            "top_k": int(self.top_k),
            "max_driver_rows": int(self.max_driver_rows),
            "population_split": self.population_split,
            "population_explain": self.population_explain,
            "value_column": self.value_column,
            "value_horizon_months": float(self.value_horizon_months),
            "reference_column": self.reference_column,
            "segment_fields": list(self.segment_fields),
            "max_export_rows": int(self.max_export_rows),
            "strict_record_types": bool(self.strict_record_types),
            "version": self.version,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "fingerprint": self.fingerprint()}


@dataclass(frozen=True)
class RiskDriver:
    """One human-readable SHAP driver attached to a risk assessment."""

    feature: str
    kind: str
    contribution: float
    direction: str
    value: float | None = None
    active_level: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "contribution": float(self.contribution),
            "direction": self.direction,
            "value": None if self.value is None else float(self.value),
            "active_level": self.active_level,
        }


@dataclass
class RiskAssessment:
    """One customer's score, tier, class, drivers and provenance."""

    record_reference: str
    input_position: int
    churn_probability: float
    predicted_class: int
    risk_tier: str
    classification_threshold: float
    risk_thresholds: RiskThresholds
    model_family: str
    model_run_id: str
    model_sha256: str | None
    preprocessing_run_id: str
    assessed_at: str
    positive_label: str | None = None
    predicted_label: str = ""
    rank: int | None = None
    observed_churn: int | None = None
    value_at_risk: float | None = None
    value_horizon_months: float | None = None
    top_positive: list[RiskDriver] = field(default_factory=list)
    top_negative: list[RiskDriver] = field(default_factory=list)
    explanation_summary: str = ""
    explanation_status: str = "unavailable"
    explanation_reason: str = ""
    segment: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.churn_probability = _finite_probability(
            self.churn_probability, "churn_probability"
        )
        if self.risk_tier not in self.risk_thresholds.labels:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"risk tier {self.risk_tier!r} is not one of {list(self.risk_thresholds.labels)}",
                "Assign tiers with the configured risk thresholds.",
            )
        if self.predicted_class not in (0, 1):
            raise RiskError(
                CODE_INVALID_CONFIG,
                "predicted_class must be 0 or 1",
                "Use the configured classification threshold.",
            )
        if not self.predicted_label:
            if self.predicted_class == 1:
                self.predicted_label = self.positive_label or "churn"
            else:
                self.predicted_label = "retained"

    @property
    def churn_probability_pct(self) -> float:
        return round(float(self.churn_probability) * 100.0, 2)

    def to_dict(self, *, include_segment: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "record_reference": self.record_reference,
            "input_position": int(self.input_position),
            "churn_probability": float(self.churn_probability),
            "churn_probability_pct": self.churn_probability_pct,
            "predicted_class": int(self.predicted_class),
            "predicted_label": self.predicted_label,
            "risk_tier": self.risk_tier,
            "rank": None if self.rank is None else int(self.rank),
            "classification_threshold": float(self.classification_threshold),
            "risk_thresholds": self.risk_thresholds.to_dict(),
            "model_family": self.model_family,
            "model_run_id": self.model_run_id,
            "model_sha256": self.model_sha256,
            "preprocessing_run_id": self.preprocessing_run_id,
            "assessed_at": self.assessed_at,
            "positive_label": self.positive_label,
            "observed_churn": self.observed_churn,
            "value_at_risk": None if self.value_at_risk is None else float(self.value_at_risk),
            "value_horizon_months": self.value_horizon_months,
            "explanation_status": self.explanation_status,
            "explanation_reason": self.explanation_reason,
            "explanation_summary": self.explanation_summary,
            "top_positive_drivers": [item.to_dict() for item in self.top_positive],
            "top_negative_drivers": [item.to_dict() for item in self.top_negative],
            "warnings": list(self.warnings),
        }
        if include_segment:
            payload["segment"] = dict(self.segment)
        return payload

    def to_row(self, *, include_segment: bool = False) -> dict[str, Any]:
        """One flat export row with stable column names."""
        payload = self.to_dict(include_segment=include_segment)
        payload["top_positive_drivers"] = json.dumps(
            payload["top_positive_drivers"], ensure_ascii=False, sort_keys=True
        )
        payload["top_negative_drivers"] = json.dumps(
            payload["top_negative_drivers"], ensure_ascii=False, sort_keys=True
        )
        payload["risk_thresholds"] = json.dumps(
            payload["risk_thresholds"], ensure_ascii=False, sort_keys=True
        )
        payload["warnings"] = json.dumps(payload["warnings"], ensure_ascii=False)
        payload["segment"] = json.dumps(payload.get("segment", {}), ensure_ascii=False, sort_keys=True)
        return payload


@dataclass(frozen=True)
class RiskFilter:
    """Pure predicate over an assessed table; no model call is made."""

    tiers: tuple[str, ...] = ()
    min_probability: float | None = None
    max_probability: float | None = None
    record_references: tuple[str, ...] = ()
    segment_equals: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    segment_min: Mapping[str, float] = field(default_factory=dict)
    segment_max: Mapping[str, float] = field(default_factory=dict)

    def validate(self, labels: Sequence[str]) -> list[str]:
        problems: list[str] = []
        unknown = sorted(set(self.tiers) - set(labels))
        if unknown:
            problems.append(f"unknown risk tier filters: {', '.join(unknown)}")
        for name, value in (
            ("min_probability", self.min_probability),
            ("max_probability", self.max_probability),
        ):
            if value is not None:
                try:
                    _finite_probability(value, name)
                except RiskError as exc:
                    problems.append(exc.message)
        if (
            self.min_probability is not None
            and self.max_probability is not None
            and float(self.min_probability) > float(self.max_probability)
        ):
            problems.append("min_probability must not exceed max_probability")
        for key, value in self.segment_equals.items():
            if not isinstance(value, tuple):
                problems.append(f"segment_equals[{key!r}] must be a tuple of values")
        for name, mapping in (
            ("segment_min", self.segment_min),
            ("segment_max", self.segment_max),
        ):
            for key, value in mapping.items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    problems.append(f"{name}[{key!r}] must be numeric")
                    continue
                if not math.isfinite(parsed):
                    problems.append(f"{name}[{key!r}] must be finite")
        return problems

    def matches(self, item: RiskAssessment) -> bool:
        if self.tiers and item.risk_tier not in self.tiers:
            return False
        if self.min_probability is not None and item.churn_probability < float(self.min_probability):
            return False
        if self.max_probability is not None and item.churn_probability > float(self.max_probability):
            return False
        if self.record_references and item.record_reference not in set(self.record_references):
            return False
        for key, allowed in self.segment_equals.items():
            if key not in item.segment or str(item.segment[key]) not in allowed:
                return False
        for key, lower in self.segment_min.items():
            try:
                if float(item.segment.get(key)) < float(lower):  # type: ignore[arg-type]
                    return False
            except (TypeError, ValueError):
                return False
        for key, upper in self.segment_max.items():
            try:
                if float(item.segment.get(key)) > float(upper):  # type: ignore[arg-type]
                    return False
            except (TypeError, ValueError):
                return False
        return True


@dataclass(frozen=True)
class RejectedRecord:
    """A record-level rejection with no raw customer values."""

    input_position: int
    record_reference: str
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_position": int(self.input_position),
            "record_reference": self.record_reference,
            "code": self.code,
            "message": self.message,
        }


@dataclass
class BatchAssessment:
    """Batch scoring result with explicit received/scored/rejected counts."""

    records: list[RiskAssessment] = field(default_factory=list)
    rejections: list[RejectedRecord] = field(default_factory=list)
    total_received: int = 0
    assessed_at: str = ""
    model_family: str = ""
    model_run_id: str = ""
    model_sha256: str | None = None
    preprocessing_run_id: str = ""
    classification_threshold: float = 0.5
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    warnings: list[str] = field(default_factory=list)
    batch_error: str = ""

    @property
    def scored(self) -> int:
        return len(self.records)

    @property
    def rejected(self) -> int:
        return len(self.rejections)

    @property
    def counts_reconcile(self) -> bool:
        return self.total_received == self.scored + self.rejected

    def table(self, *, run_id: str = "") -> "RiskTable":
        return RiskTable(
            records=self.records,
            thresholds=self.risk_thresholds,
            classification_threshold=self.classification_threshold,
            run_id=run_id,
            model_family=self.model_family,
            model_run_id=self.model_run_id,
            model_sha256=self.model_sha256,
            preprocessing_run_id=self.preprocessing_run_id,
            generated_at=self.assessed_at,
            warnings=list(self.warnings),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RISK_SCHEMA_VERSION,
            "phase": 9,
            "assessed_at": self.assessed_at,
            "total_received": int(self.total_received),
            "scored": int(self.scored),
            "rejected": int(self.rejected),
            "counts_reconcile": self.counts_reconcile,
            "batch_error": self.batch_error,
            "model": {
                "family": self.model_family,
                "model_run_id": self.model_run_id,
                "model_sha256": self.model_sha256,
                "preprocessing_run_id": self.preprocessing_run_id,
            },
            "classification_threshold": float(self.classification_threshold),
            "risk_thresholds": self.risk_thresholds.to_dict(),
            "records": [item.to_dict() for item in self.records],
            "rejections": [item.to_dict() for item in self.rejections],
            "warnings": list(self.warnings),
        }


@dataclass
class RiskTable:
    """Ranked assessments plus the provenance needed by Phase 10/dashboard."""

    records: list[RiskAssessment] = field(default_factory=list)
    thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    classification_threshold: float = 0.5
    run_id: str = ""
    model_family: str = ""
    model_run_id: str = ""
    model_sha256: str | None = None
    preprocessing_run_id: str = ""
    generated_at: str = ""
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    max_export_rows: int = 50_000

    def get(self, reference: str) -> RiskAssessment | None:
        return next((item for item in self.records if item.record_reference == reference), None)

    def filtered(self, criteria: RiskFilter) -> "RiskTable":
        problems = criteria.validate(self.thresholds.labels)
        if problems:
            raise RiskError(
                CODE_INVALID_CONFIG,
                "invalid risk filter: " + "; ".join(problems),
                "Correct the tier/probability/segment filters and retry.",
            )
        return replace(self, records=[item for item in self.records if criteria.matches(item)])

    def paged(self, *, limit: int = 50, offset: int = 0) -> "RiskTable":
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= self.max_export_rows
        ):
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"limit must be an integer between 1 and {self.max_export_rows}",
                "Pass a bounded page size.",
            )
        if not isinstance(offset, int) or isinstance(offset, bool) or not offset >= 0:
            raise RiskError(
                CODE_INVALID_CONFIG,
                "offset must be a non-negative integer",
                "Pass a non-negative page offset.",
            )
        return replace(self, records=self.records[offset : offset + limit])

    def tier_counts(self) -> dict[str, dict[str, Any]]:
        total = len(self.records)
        counts: dict[str, dict[str, Any]] = {}
        for label in self.thresholds.labels:
            count = sum(1 for item in self.records if item.risk_tier == label)
            counts[label] = {
                "count": int(count),
                "pct": (float(count) / total) if total else 0.0,
            }
        return counts

    def to_frame(self, *, include_segment: bool = False) -> pd.DataFrame:
        rows = [item.to_row(include_segment=include_segment) for item in self.records]
        if not rows:
            return pd.DataFrame(columns=list(EXPORT_COLUMNS))
        frame = pd.DataFrame(rows)
        for column in (
            "input_position",
            "predicted_class",
            "rank",
            "observed_churn",
        ):
            if column in frame.columns:
                frame[column] = pd.array(frame[column], dtype="Int64")
        for column in (
            "churn_probability",
            "churn_probability_pct",
            "classification_threshold",
            "value_at_risk",
            "value_horizon_months",
        ):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.reindex(columns=list(EXPORT_COLUMNS))

    def to_dict(self, *, include_records: bool = True) -> dict[str, Any]:
        return {
            "schema_version": RISK_SCHEMA_VERSION,
            "phase": 9,
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "model": {
                "family": self.model_family,
                "model_run_id": self.model_run_id,
                "model_sha256": self.model_sha256,
                "preprocessing_run_id": self.preprocessing_run_id,
            },
            "classification_threshold": float(self.classification_threshold),
            "risk_thresholds": self.thresholds.to_dict(),
            "tier_counts": self.tier_counts(),
            "n_rows": len(self.records),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "limitations": list(self.limitations),
            "records": [item.to_dict() for item in self.records] if include_records else [],
        }


@dataclass
class RiskRun:
    """Persisted Phase 9 scoring run."""

    run_id: str
    table: RiskTable
    paths: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    rejections: list[RejectedRecord] = field(default_factory=list)
    total_received: int = 0
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def complete(self) -> bool:
        return bool(self.table.records) or self.summary.get("n_rows") == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RISK_SCHEMA_VERSION,
            "phase": 9,
            "run_id": self.run_id,
            "complete": self.complete,
            "summary": dict(self.summary),
            "total_received": int(self.total_received or len(self.table.records)),
            "scored": len(self.table.records),
            "rejected": len(self.rejections),
            "rejections": [item.to_dict() for item in self.rejections],
            "paths": dict(self.paths),
            "manifest": dict(self.manifest),
            "warnings": list(self.warnings),
            "duration_seconds": float(self.duration_seconds),
        }


@dataclass
class RiskExport:
    """Result of a CSV/JSON/Parquet export."""

    run_id: str
    paths: dict[str, str] = field(default_factory=dict)
    row_count: int = 0
    formats: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "paths": dict(self.paths),
            "row_count": int(self.row_count),
            "formats": list(self.formats),
            "warnings": list(self.warnings),
        }


__all__ = [
    "BatchAssessment",
    "DEFAULT_TIER_BOUNDARIES",
    "EXPORT_COLUMNS",
    "RISK_SCHEMA_VERSION",
    "RejectedRecord",
    "RiskAssessment",
    "RiskConfig",
    "RiskDriver",
    "RiskExport",
    "RiskFilter",
    "RiskRun",
    "RiskTable",
    "RiskThresholds",
    "TIER_LABELS",
    "VALID_EXPLAIN_MODES",
    "VALID_POPULATION_SPLITS",
    "VALID_SORT_FIELDS",
]
