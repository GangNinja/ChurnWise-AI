"""Phase 9 customer churn-risk intelligence service.

All probabilities come from the shipped Phase 7 model through the verified
Phase 8 model/preprocessing loader. There is no hard-coded score, rule-based
fallback or separate prediction path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from churnwise.config import Settings
from churnwise.contracts import utc_now_iso
from churnwise.intelligence.contracts import (
    EXPORT_COLUMNS,
    RISK_SCHEMA_VERSION,
    BatchAssessment,
    RejectedRecord,
    RiskAssessment,
    RiskConfig,
    RiskDriver,
    RiskExport,
    RiskFilter,
    RiskRun,
    RiskTable,
    RiskThresholds,
)
from churnwise.intelligence.drivers import (
    attach_index,
    driver_from_contribution,
    explain_matrix_rows,
    load_phase8_index,
)
from churnwise.intelligence.errors import (
    CODE_ARTIFACT_FAILED,
    CODE_EXPORT_FAILED,
    CODE_INVALID_CONFIG,
    CODE_INVALID_INPUT,
    CODE_NO_MODEL,
    CODE_PREDICTION_FAILED,
    RiskError,
)
from churnwise.intelligence.model import LoadedRiskModel, load_selected_model
from churnwise.intelligence.ranking import (
    attach_value_at_risk,
    filter_records,
    paginate,
    rank_records,
    tier_counts,
)

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,180}$")
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
_JSON_COLUMNS = (
    "top_positive_drivers",
    "top_negative_drivers",
    "warnings",
    "segment",
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[:110] or "dataset"


def validate_risk_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise RiskError(
            CODE_INVALID_CONFIG,
            f"invalid risk run id {run_id!r}",
            "Use a run id printed by the score command.",
        )
    return run_id


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, ValueError, TypeError) as exc:
        raise RiskError(
            CODE_EXPORT_FAILED,
            f"risk artifact could not be written: {path}: {type(exc).__name__}: {exc}",
            "Check that the artifact directory is writable and the payload is JSON-safe.",
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(path)


def _atomic_write_text(path: Path, text: str) -> str:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise RiskError(
            CODE_EXPORT_FAILED,
            f"risk export could not be written: {path}: {type(exc).__name__}: {exc}",
            "Check that the export directory is writable.",
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(path)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, set, tuple)):
        return value
    if pd.isna(value):
        return None
    return value


def _opaque_reference(value: Any, *, dataset_id: str) -> str:
    """Hash an identifier so raw customer ids never enter artifacts."""
    encoded = f"{dataset_id}|{value}".encode("utf-8")
    return "cust_" + hashlib.sha256(encoded).hexdigest()[:12]


def _missing_reference(value: Any) -> bool:
    """Treat null, NaN, NaT and blank identifiers as missing, but not containers."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, set, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _resolve_artifact_path(path: str | None, root: str | Path | None = None) -> Path | None:
    """Resolve a recorded path without assuming the caller's working directory."""
    if not path:
        return None
    target = Path(str(path))
    if not target.is_absolute() and root is not None:
        target = Path(root) / target
    return target


def _portable_artifact_path(path: str | None, root: str | Path) -> str | None:
    """Prefer a project-relative path so a run tree can be moved intact."""
    target = _resolve_artifact_path(path, root)
    if target is None:
        return None
    try:
        return target.resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, ValueError):
        return str(target.resolve())


def _hash_file_still_matches(
    path: str | None,
    expected: str | None,
    root: str | Path | None = None,
) -> tuple[bool, str]:
    if not path:
        return True, ""
    target = _resolve_artifact_path(path, root)
    assert target is not None
    if not target.is_file():
        return True, f"artifact {target.name} is unavailable for a freshness check"
    if not expected:
        return True, ""
    try:
        digest = _sha256(target)
    except OSError as exc:
        return False, (
            f"artifact {target.name} could not be read for a freshness check "
            f"({type(exc).__name__})"
        )
    if digest != str(expected):
        return False, f"artifact {target.name} changed after the risk run was created"
    return True, ""


def sanitize_csv_value(value: Any) -> Any:
    """Prevent spreadsheet formula injection without changing numeric columns.

    Both the literal first character (tab and carriage return are dangerous on
    their own) and the first character after leading whitespace are checked.
    """
    if isinstance(value, str):
        if value[:1] in _FORMULA_PREFIXES or value.lstrip()[:1] in _FORMULA_PREFIXES:
            return "'" + value
    return value


def _sanitize_frame_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == object or pd.api.types.is_string_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].map(sanitize_csv_value)
    return cleaned


class RiskService:
    """Stable Phase 9 interface for scoring, ranking, drivers and exports."""

    def __init__(
        self,
        settings: Settings,
        config: RiskConfig | None = None,
        *,
        benchmark_run_id: str | None = None,
        now: Callable[[], str] = utc_now_iso,
    ) -> None:
        self.settings = settings
        self.config = config or RiskConfig()
        self.config.validate_or_raise()
        self.benchmark_run_id = benchmark_run_id
        self.artifact_dir = Path(settings.artifact_dir)
        self.predictions_dir = self.artifact_dir / "predictions"
        self._now = now
        self._loaded: LoadedRiskModel | None = None

    # -- model / thresholds -------------------------------------------------
    @property
    def model(self) -> LoadedRiskModel:
        if self._loaded is None:
            threshold = self.config.classification_threshold
            self._loaded = load_selected_model(
                self.settings,
                benchmark_run_id=self.benchmark_run_id,
                classification_threshold=threshold,
            )
        return self._loaded

    @property
    def thresholds(self) -> RiskThresholds:
        return self.config.thresholds

    def inspect_thresholds(self) -> dict[str, Any]:
        return {
            "risk_thresholds": self.thresholds.to_dict(),
            "classification_threshold": float(self.model.threshold),
            "separate": True,
        }

    # -- individual / batch ------------------------------------------------
    def assess_customer(
        self,
        record: Mapping[str, Any] | pd.Series,
        *,
        reference: str | None = None,
        explain: bool = False,
        explain_run_id: str | None = None,
        value: float | None = None,
    ) -> RiskAssessment:
        """Assess one raw customer record with the shipped model."""
        if isinstance(record, pd.Series):
            values = dict(record)
        elif isinstance(record, Mapping):
            values = dict(record)
        else:
            raise RiskError(
                CODE_INVALID_INPUT,
                "assess_customer expects a mapping or pandas Series",
                "Pass one raw customer record with the schema.required_columns fields.",
            )
        if not values:
            raise RiskError(
                CODE_INVALID_INPUT,
                "the customer record is empty",
                "Provide every column required by the Phase 5 schema.",
            )
        frame = pd.DataFrame([values])
        batch = self.assess_batch(
            frame,
            reference_override=reference,
            explain="all" if explain else "none",
            explain_run_id=explain_run_id,
            values=[value] if value is not None else None,
        )
        if not batch.records:
            rejection = batch.rejections[0] if batch.rejections else None
            raise RiskError(
                rejection.code if rejection else CODE_INVALID_INPUT,
                rejection.message if rejection else "the record could not be assessed",
                "Correct the record and retry; no default probability was substituted.",
            )
        item = batch.records[0]
        for warning in batch.warnings:
            if warning not in item.warnings:
                item.warnings.append(warning)
        return item

    def assess_batch(
        self,
        frame: pd.DataFrame,
        *,
        batch_size: int | None = None,
        explain: str = "none",
        explain_top_n: int = 10,
        explain_positions: Sequence[int] | None = None,
        explain_run_id: str | None = None,
        reference_override: str | None = None,
        reference_column: str | None = None,
        value_column: str | None = None,
        values: Sequence[float] | None = None,
        segment_fields: Sequence[str] | None = None,
    ) -> BatchAssessment:
        """Assess raw records, preserving input/output correspondence."""
        if not isinstance(frame, pd.DataFrame):
            raise RiskError(
                CODE_INVALID_INPUT,
                "assess_batch expects a pandas DataFrame",
                "Build a DataFrame with the schema.required_columns fields.",
            )
        if explain not in {"none", "top", "all", "selected"}:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"explain must be none, top, all or selected, got {explain!r}",
                "Use explain='none', 'top', 'all' or 'selected'.",
            )
        if explain == "selected" and not explain_positions:
            raise RiskError(
                CODE_INVALID_CONFIG,
                "explain='selected' requires at least one input position",
                "Pass explain_positions=[...] or use explain='top'.",
            )
        if explain != "selected" and explain_positions:
            raise RiskError(
                CODE_INVALID_CONFIG,
                "explain_positions is only valid with explain='selected'",
                "Set explain='selected' or omit the positions.",
            )
        frame = frame.reset_index(drop=True)
        size = int(batch_size or self.config.batch_size)
        if size < 1:
            raise RiskError(
                CODE_INVALID_CONFIG,
                "batch_size must be positive",
                "Pass a positive batch size.",
            )
        model = self.model
        assessed_at = self._now()
        if values is not None and len(values) != len(frame):
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"{len(values)} value proxies were supplied for {len(frame)} records",
                "Supply exactly one value for each record.",
            )
        if len(frame) == 0:
            return BatchAssessment(
                total_received=0,
                assessed_at=assessed_at,
                model_family=model.family,
                model_run_id=model.model_run_id,
                model_sha256=model.model_sha256,
                preprocessing_run_id=model.preprocessing_run_id,
                classification_threshold=model.threshold,
                risk_thresholds=self.thresholds,
                warnings=["empty batch: no records were assessed"],
            )

        required = list((model.dataset.schema or {}).get("required_columns") or [])
        missing = [name for name in required if name not in frame.columns]
        references, reference_warnings = self._references(
            frame,
            reference_override=reference_override,
            reference_column=reference_column,
        )
        if missing:
            message = "input is missing required columns: " + ", ".join(missing)
            rejections = [
                RejectedRecord(
                    input_position=index,
                    record_reference=references[index],
                    code=CODE_INVALID_INPUT,
                    message=message,
                )
                for index in range(len(frame))
            ]
            return BatchAssessment(
                rejections=rejections,
                total_received=len(frame),
                assessed_at=assessed_at,
                model_family=model.family,
                model_run_id=model.model_run_id,
                model_sha256=model.model_sha256,
                preprocessing_run_id=model.preprocessing_run_id,
                classification_threshold=model.threshold,
                risk_thresholds=self.thresholds,
                warnings=[*reference_warnings, message],
                batch_error=message,
            )

        segment_fields = tuple(segment_fields or self.config.segment_fields)
        segments = self._segments(frame, segment_fields)
        records: list[RiskAssessment] = []
        rejections: list[RejectedRecord] = []
        matrices: dict[int, np.ndarray] = {}
        warnings = list(reference_warnings)
        for warning in getattr(model.dataset, "warnings", []) or []:
            if str(warning) not in warnings:
                warnings.append(str(warning))
        intended = set(required) | set(segment_fields)
        for name in (
            reference_column,
            value_column,
            self.config.reference_column,
            model.dataset.split_spec.id_column,
        ):
            if name:
                intended.add(str(name))
        unexpected = [str(name) for name in frame.columns if str(name) not in intended]
        if unexpected:
            warnings.append(
                "input column(s) not used by the model were ignored: "
                + ", ".join(unexpected)
            )
        if values is not None:
            parsed_values: list[float] = []
            for position, value in enumerate(values):
                try:
                    parsed = float(value)
                except (TypeError, ValueError) as exc:
                    raise RiskError(
                        CODE_INVALID_INPUT,
                        f"value proxy at input position {position} is not numeric",
                        "Supply a finite, non-negative value for every record.",
                    ) from exc
                if not math.isfinite(parsed) or parsed < 0:
                    raise RiskError(
                        CODE_INVALID_INPUT,
                        f"value proxy at input position {position} is not a finite "
                        "non-negative number",
                        "Supply a finite, non-negative value for every record.",
                    )
                parsed_values.append(parsed)
            values = parsed_values

        for start in range(0, len(frame), size):
            chunk = frame.iloc[start : start + size]
            positions = list(range(start, start + len(chunk)))
            valid_positions, type_problems = self._validate_record_types(
                chunk,
                positions,
                required=required,
                strict=self.config.strict_record_types,
            )
            for position, message in type_problems:
                rejections.append(
                    RejectedRecord(
                        input_position=position,
                        record_reference=references[position],
                        code=CODE_INVALID_INPUT,
                        message=message,
                    )
                )
            if not valid_positions:
                continue
            valid_frame = chunk.loc[valid_positions]
            try:
                matrix = model.dataset.transform_new(valid_frame.reset_index(drop=True))
                probabilities = self._probabilities(model, matrix)
            except Exception:  # noqa: BLE001 - isolate malformed records below
                for position in valid_positions:
                    row = chunk.loc[[position]]
                    try:
                        row_matrix = model.dataset.transform_new(row.reset_index(drop=True))
                        probability = self._probabilities(model, row_matrix)[0]
                    except Exception as exc:  # noqa: BLE001
                        message = getattr(exc, "message", None) or (
                            f"{type(exc).__name__}: record could not be assessed"
                        )
                        rejections.append(
                            RejectedRecord(
                                input_position=position,
                                record_reference=references[position],
                                code=CODE_PREDICTION_FAILED,
                                message=str(message),
                            )
                        )
                        continue
                    records.append(
                        self._assessment(
                            model=model,
                            record_reference=references[position],
                            input_position=position,
                            probability=float(probability),
                            assessed_at=assessed_at,
                            segment=segments[position],
                            warnings=self._row_warnings(
                                chunk.loc[position], required=required
                            ),
                        )
                    )
                    matrices[position] = row_matrix.reshape(1, -1)
                continue
            for offset, position in enumerate(valid_positions):
                records.append(
                    self._assessment(
                        model=model,
                        record_reference=references[position],
                        input_position=position,
                        probability=float(probabilities[offset]),
                        assessed_at=assessed_at,
                        segment=segments[position],
                        warnings=self._row_warnings(
                            chunk.loc[position], required=required
                        ),
                    )
                )
                matrices[position] = matrix[offset].reshape(1, -1)

        records.sort(key=lambda item: item.input_position)
        if values is not None:
            attach_value_at_risk(
                records,
                [values[item.input_position] for item in records],
                horizon_months=self.config.value_horizon_months,
            )
        self._attach_drivers(
            records,
            matrices,
            mode=explain,
            top_n=int(explain_top_n),
            selected_positions=explain_positions,
            explain_run_id=explain_run_id,
            run_id=self._ad_hoc_driver_id(assessed_at, records, explain),
            warnings=warnings,
            persist=True,
        )
        if not records:
            warnings.append("no record in this batch could be assessed")
        if rejections:
            warnings.append(
                f"{len(rejections)} record(s) were rejected individually; "
                "other records were scored normally"
            )
        return BatchAssessment(
            records=records,
            rejections=rejections,
            total_received=len(frame),
            assessed_at=assessed_at,
            model_family=model.family,
            model_run_id=model.model_run_id,
            model_sha256=model.model_sha256,
            preprocessing_run_id=model.preprocessing_run_id,
            classification_threshold=model.threshold,
            risk_thresholds=self.thresholds,
            warnings=warnings,
        )

    def score_frame(
        self,
        frame: pd.DataFrame,
        *,
        explain: str = "none",
        explain_top_n: int = 10,
        explain_positions: Sequence[int] | None = None,
        explain_run_id: str | None = None,
        reference_column: str | None = None,
        value_column: str | None = None,
        segment_fields: Sequence[str] | None = None,
        batch_size: int | None = None,
        run_id: str | None = None,
        persist: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> RiskRun:
        """Assess caller-provided raw records and persist a batch run."""
        started = time.perf_counter()
        if not isinstance(frame, pd.DataFrame):
            raise RiskError(
                CODE_INVALID_INPUT,
                "score_frame expects a pandas DataFrame",
                "Load a CSV into a DataFrame before scoring.",
            )
        value_column = (
            value_column if value_column is not None else self.config.value_column
        )
        if value_column is not None:
            if value_column not in frame.columns:
                raise RiskError(
                    CODE_INVALID_CONFIG,
                    f"value column {value_column!r} is not present in the input",
                    "Pass --value-column with a real revenue-proxy column.",
                )
            values = []
            for position, value in enumerate(frame[value_column].tolist()):
                try:
                    parsed = float(value)
                except (TypeError, ValueError) as exc:
                    raise RiskError(
                        CODE_INVALID_INPUT,
                        f"value column {value_column!r} has a non-numeric value at "
                        f"input position {position}",
                        "Correct or remove the revenue proxy before scoring.",
                    ) from exc
                if not math.isfinite(parsed) or parsed < 0:
                    raise RiskError(
                        CODE_INVALID_INPUT,
                        f"value column {value_column!r} has an invalid value at input "
                        f"position {position}; values must be finite and non-negative",
                        "Correct or remove the revenue proxy before scoring.",
                    )
                values.append(parsed)
        else:
            values = None
        batch = self.assess_batch(
            frame,
            batch_size=batch_size,
            explain=explain,
            explain_top_n=explain_top_n,
            explain_positions=explain_positions,
            explain_run_id=explain_run_id,
            reference_column=reference_column,
            value_column=value_column,
            values=values,
            segment_fields=segment_fields,
        )
        generated_at = batch.assessed_at
        run_id = run_id or self._default_run_id(
            self.model.dataset.dataset_id, "input", generated_at
        )
        validate_risk_id(run_id)
        ranked = rank_records(batch.records, sort_by="probability", descending=True)
        summary = self._summary(
            run_id=run_id,
            model=self.model,
            split="input",
            records=ranked,
            generated_at=generated_at,
            warnings=batch.warnings,
            values_present=values is not None,
            value_column=value_column,
            received=batch.total_received,
            rejected=batch.rejected,
        )
        table = RiskTable(
            records=ranked,
            thresholds=self.thresholds,
            classification_threshold=self.model.threshold,
            run_id=run_id,
            model_family=self.model.family,
            model_run_id=self.model.model_run_id,
            model_sha256=self.model.model_sha256,
            preprocessing_run_id=self.model.preprocessing_run_id,
            generated_at=generated_at,
            warnings=list(batch.warnings),
            provenance=dict(self.model.provenance),
            limitations=self._limitations(values_present=values is not None),
            max_export_rows=self.config.max_export_rows,
        )
        result = RiskRun(
            run_id=run_id,
            table=table,
            summary=summary,
            rejections=list(batch.rejections),
            total_received=int(batch.total_received),
            warnings=list(batch.warnings),
            duration_seconds=time.perf_counter() - started,
        )
        if persist:
            self._persist(result)
        return result

    # -- population ---------------------------------------------------------
    def score_population(
        self,
        *,
        split: str | None = None,
        explain: str | None = None,
        explain_top_n: int = 10,
        explain_positions: Sequence[int] | None = None,
        run_id: str | None = None,
        persist: bool = True,
        values: Sequence[float] | None = None,
        explain_run_id: str | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> RiskRun:
        """Score a persisted Phase 5 split with the shipped model."""
        split = split or self.config.population_split
        explain = explain or self.config.population_explain
        if split not in {"train", "val", "test", "all"}:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"unknown population split {split!r}",
                "Use train, val, test or all.",
            )
        if explain not in {"none", "top", "all", "selected"}:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"explain must be none, top, all or selected, got {explain!r}",
                "Use none, top, all or selected.",
            )
        if explain == "selected" and not explain_positions:
            raise RiskError(
                CODE_INVALID_CONFIG,
                "explain='selected' requires at least one input position",
                "Pass explain_positions=[...] or use explain='top'.",
            )
        if explain != "selected" and explain_positions:
            raise RiskError(
                CODE_INVALID_CONFIG,
                "explain_positions is only valid with explain='selected'",
                "Set explain='selected' or omit the positions.",
            )
        model = self.model
        dataset = model.dataset
        if split == "all":
            parts = [dataset.train, dataset.val, dataset.test]
            matrix = np.vstack([np.asarray(part.X, dtype=np.float64) for part in parts])
            labels = np.concatenate([np.asarray(part.y, dtype=int) for part in parts])
        else:
            part = dataset.split(split)
            matrix = np.asarray(part.X, dtype=np.float64)
            labels = np.asarray(part.y, dtype=int)
        if len(matrix) == 0:
            raise RiskError(
                CODE_NO_MODEL,
                f"the {split} partition has no rows",
                "Regenerate the Phase 5 split or score another partition.",
            )
        if values is not None and len(values) != len(matrix):
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"{len(values)} value proxies were supplied for {len(matrix)} rows",
                "Supply exactly one proxy value per scored row.",
            )
        generated_at = self._now()
        run_id = run_id or self._default_run_id(dataset.dataset_id, split, generated_at)
        validate_risk_id(run_id)
        started = time.perf_counter()
        warnings: list[str] = []
        for warning in getattr(dataset, "warnings", []) or []:
            if str(warning) not in warnings:
                warnings.append(str(warning))
        probabilities = self._probabilities(model, matrix)
        records: list[RiskAssessment] = []
        references = [
            f"risk_{split}_{index:05d}" for index in range(len(matrix))
        ]
        for index, probability in enumerate(probabilities):
            item = self._assessment(
                model=model,
                record_reference=references[index],
                input_position=index,
                probability=float(probability),
                assessed_at=generated_at,
                segment={},
                warnings=[],
                observed_churn=int(labels[index]) if index < len(labels) else None,
            )
            records.append(item)
        if values is not None:
            attach_value_at_risk(
                records,
                list(values),
                horizon_months=self.config.value_horizon_months,
            )
        matrices = {
            index: matrix[index].reshape(1, -1) for index in range(len(matrix))
        }
        self._attach_drivers(
            records,
            matrices,
            mode=explain,
            top_n=int(explain_top_n),
            selected_positions=explain_positions,
            explain_run_id=explain_run_id,
            run_id=f"{run_id}-drivers" if explain != "none" else None,
            warnings=warnings,
            progress=progress,
            persist=persist,
        )
        ranked = rank_records(records, sort_by="probability", descending=True)
        summary = self._summary(
            run_id=run_id,
            model=model,
            split=split,
            records=ranked,
            generated_at=generated_at,
            warnings=warnings,
            values_present=values is not None,
        )
        table = RiskTable(
            records=ranked,
            thresholds=self.thresholds,
            classification_threshold=model.threshold,
            run_id=run_id,
            model_family=model.family,
            model_run_id=model.model_run_id,
            model_sha256=model.model_sha256,
            preprocessing_run_id=model.preprocessing_run_id,
            generated_at=generated_at,
            warnings=warnings,
            provenance=dict(model.provenance),
            limitations=self._limitations(values_present=values is not None),
            max_export_rows=self.config.max_export_rows,
        )
        result = RiskRun(
            run_id=run_id,
            table=table,
            summary=summary,
            warnings=warnings,
            duration_seconds=time.perf_counter() - started,
        )
        if persist:
            self._persist(result)
        return result

    # -- retrieval / ranking ------------------------------------------------
    def latest_run_id(self) -> str | None:
        if not self.predictions_dir.is_dir():
            return None
        candidates = sorted(
            path.name
            for path in self.predictions_dir.iterdir()
            if path.is_dir() and (path / "risk_summary.json").is_file()
        )
        return candidates[-1] if candidates else None

    def load_run(self, run_id: str, *, verify_freshness: bool = True) -> RiskRun:
        validate_risk_id(run_id)
        root = self.predictions_dir / run_id
        summary_path = root / "risk_summary.json"
        manifest_path = root / "run_manifest.json"
        scores_path = root / "churn_scores.parquet"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"risk run {run_id!r} could not be read",
                "Rerun the score command; the artifact may be missing or corrupt.",
            ) from exc
        if not isinstance(summary, dict) or not isinstance(manifest, dict):
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"risk run {run_id!r} metadata is not a JSON object",
                "Rerun the score command; do not edit risk metadata by hand.",
            )
        try:
            frame = pd.read_parquet(scores_path)
        except (OSError, ValueError) as exc:
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"scored table for {run_id!r} could not be read",
                "Rerun the score command; do not edit the Parquet artifact.",
            ) from exc
        missing_columns = [name for name in EXPORT_COLUMNS if name not in frame.columns]
        if missing_columns:
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"scored table for {run_id!r} is missing required column(s): "
                + ", ".join(missing_columns),
                "Rerun the score command with this project version.",
            )
        population_block = summary.get("population")
        population = population_block if isinstance(population_block, dict) else {}
        recorded_rows = summary.get("n_rows")
        if (
            isinstance(recorded_rows, int)
            and not isinstance(recorded_rows, bool)
            and recorded_rows != len(frame)
        ):
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"risk run {run_id!r} records {recorded_rows} row(s) but its "
                f"scored table has {len(frame)}",
                "Rerun the score command; do not edit the Parquet artifact.",
            )
        recorded_scored = population.get("scored")
        if (
            isinstance(recorded_scored, int)
            and not isinstance(recorded_scored, bool)
            and recorded_scored != len(frame)
        ):
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"risk run {run_id!r} records {recorded_scored} scored row(s) but its "
                f"scored table has {len(frame)}",
                "Rerun the score command; do not edit the Parquet artifact.",
            )
        recorded_received = population.get("received")
        recorded_rejected = population.get("rejected")
        if (
            isinstance(recorded_received, int)
            and not isinstance(recorded_received, bool)
            and isinstance(recorded_rejected, int)
            and not isinstance(recorded_rejected, bool)
            and recorded_received != len(frame) + recorded_rejected
        ):
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"risk run {run_id!r} counts do not reconcile: "
                f"{recorded_received} received != {len(frame)} scored + "
                f"{recorded_rejected} rejected",
                "Rerun the score command; do not edit risk metadata by hand.",
            )
        warnings = (
            list(summary.get("warnings") or [])
            if isinstance(summary.get("warnings") or [], list)
            else []
        )
        if verify_freshness:
            for path, expected in (
                (manifest.get("model_path"), manifest.get("model_sha256")),
                (manifest.get("preprocessor_path"), manifest.get("preprocessor_hash")),
            ):
                ok, message = _hash_file_still_matches(
                    path, expected, root=self.settings.project_root
                )
                if not ok:
                    raise RiskError(
                        CODE_ARTIFACT_FAILED,
                        f"risk run {run_id!r} is stale: {message}",
                        "Rerun scoring after changing a model or preprocessor artifact.",
                    )
                if message:
                    warnings.append(message)
        threshold_block = summary.get("risk_thresholds")
        config_block = manifest.get("config")
        config_block = config_block if isinstance(config_block, dict) else {}
        thresholds = RiskThresholds.from_value(
            (threshold_block if isinstance(threshold_block, dict) else {}).get(
                "boundaries"
            )
            or config_block.get("thresholds")
        )
        try:
            records = [
                self._record_from_row(row, thresholds) for _, row in frame.iterrows()
            ]
        except RiskError as exc:
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"scored table for {run_id!r} is incompatible: {exc.message}",
                "Rerun the score command with this project version.",
            ) from exc
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"scored table for {run_id!r} could not be parsed "
                f"({type(exc).__name__})",
                "Rerun the score command; do not edit the Parquet artifact.",
            ) from exc
        paths_block = manifest.get("paths")
        paths_block = paths_block if isinstance(paths_block, dict) else {}
        rejections_path = paths_block.get("rejections")
        rejections: list[RejectedRecord] = []
        if rejections_path:
            try:
                payload = json.loads(Path(rejections_path).read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("rejection payload is not an object")
                rejections = [
                    RejectedRecord(
                        input_position=int(item["input_position"]),
                        record_reference=str(item["record_reference"]),
                        code=str(item["code"]),
                        message=str(item["message"]),
                    )
                    for item in payload.get("rejections") or []
                    if isinstance(item, dict)
                ]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warnings.append(
                    f"rejection details for run {run_id!r} could not be read: "
                    f"{type(exc).__name__}"
                )
        if (
            isinstance(recorded_rejected, int)
            and not isinstance(recorded_rejected, bool)
            and len(rejections) != recorded_rejected
        ):
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"risk run {run_id!r} records {recorded_rejected} rejected row(s) "
                f"but its rejection details contain {len(rejections)}",
                "Rerun the score command; do not edit rejection artifacts.",
            )
        recorded_threshold = summary.get("classification_threshold")
        if recorded_threshold is None and len(frame):
            recorded_threshold = frame["classification_threshold"].iloc[0]
        model_block = summary.get("model")
        model_block = model_block if isinstance(model_block, dict) else {}
        provenance_block = summary.get("provenance")
        limitations_block = summary.get("limitations")
        table = RiskTable(
            records=records,
            thresholds=thresholds,
            classification_threshold=float(
                recorded_threshold if recorded_threshold is not None else 0.5
            ),
            run_id=run_id,
            model_family=str(model_block.get("family") or ""),
            model_run_id=str(model_block.get("model_run_id") or ""),
            model_sha256=model_block.get("model_sha256"),
            preprocessing_run_id=str(
                model_block.get("preprocessing_run_id") or ""
            ),
            generated_at=str(summary.get("generated_at") or ""),
            warnings=warnings,
            provenance=provenance_block if isinstance(provenance_block, dict) else {},
            limitations=limitations_block if isinstance(limitations_block, list) else [],
            max_export_rows=int(
                config_block.get("max_export_rows") or self.config.max_export_rows
            ),
        )
        summary_paths = summary.get("paths")
        return RiskRun(
            run_id=run_id,
            table=table,
            paths=summary_paths if isinstance(summary_paths, dict) else {},
            summary=summary,
            manifest=manifest,
            rejections=rejections,
            total_received=int(population.get("received", len(records))),
            warnings=warnings,
        )

    def ranked(
        self,
        *,
        run_id: str | None = None,
        filters: RiskFilter | None = None,
        sort_by: str = "probability",
        limit: int = 50,
        offset: int = 0,
        verify_freshness: bool = True,
    ) -> RiskTable:
        """Retrieve a filtered, deterministically ranked, paginated table."""
        run = self.load_run(run_id or self.latest_run_id() or "", verify_freshness=verify_freshness)
        records = run.table.records
        criteria = filters or RiskFilter()
        records = filter_records(records, criteria, labels=run.table.thresholds.labels)
        records = rank_records(records, sort_by=sort_by, descending=True)
        records = paginate(
            records,
            limit=limit,
            offset=offset,
            max_rows=run.table.max_export_rows,
        )
        return replace(run.table, records=records)

    def get_assessment(self, run_id: str, reference: str) -> RiskAssessment | None:
        return self.load_run(run_id).table.get(reference)

    def get_summary(self, run_id: str | None = None) -> dict[str, Any]:
        return dict(self.load_run(run_id or self.latest_run_id() or "").summary)

    def get_explanation(self, run_id: str, reference: str) -> dict[str, Any]:
        item = self.get_assessment(run_id, reference)
        if item is None:
            raise RiskError(
                CODE_INVALID_INPUT,
                f"record reference {reference!r} is not in run {run_id!r}",
                "Use a reference returned by the ranked table.",
            )
        return {
            "record_reference": item.record_reference,
            "explanation_status": item.explanation_status,
            "explanation_reason": item.explanation_reason,
            "summary": item.explanation_summary,
            "top_positive": [driver.to_dict() for driver in item.top_positive],
            "top_negative": [driver.to_dict() for driver in item.top_negative],
            "limitations": [
                "SHAP values describe model attributions, not causal effects.",
                "Changing a feature is not guaranteed to retain a customer.",
            ],
        }

    # -- exports ------------------------------------------------------------
    def export(
        self,
        run_id: str | None = None,
        *,
        formats: Sequence[str] = ("csv", "json"),
        filters: RiskFilter | None = None,
        sort_by: str = "probability",
        limit: int | None = None,
        offset: int = 0,
        output_dir: str | Path | None = None,
        include_segment: bool = False,
        verify_freshness: bool = True,
    ) -> RiskExport:
        """Export filtered results under the run's artifact directory."""
        run = self.load_run(run_id or self.latest_run_id() or "", verify_freshness=verify_freshness)
        requested = tuple(str(item).lower() for item in formats)
        if not requested or any(item not in {"csv", "json", "parquet"} for item in requested):
            raise RiskError(
                CODE_INVALID_CONFIG,
                "formats must contain csv, json or parquet",
                "Use --format csv, json or parquet.",
            )
        if len(set(requested)) != len(requested):
            requested = tuple(dict.fromkeys(requested))
        criteria = filters or RiskFilter()
        records = filter_records(
            run.table.records,
            criteria,
            labels=run.table.thresholds.labels,
        )
        records = rank_records(records, sort_by=sort_by, descending=True)
        count = len(records)
        if limit is not None:
            records = paginate(
                records,
                limit=limit,
                offset=offset,
                max_rows=run.table.max_export_rows,
            )
        elif offset:
            records = paginate(
                records,
                limit=run.table.max_export_rows,
                offset=offset,
                max_rows=run.table.max_export_rows,
            )
        if len(records) > run.table.max_export_rows:
            raise RiskError(
                CODE_INVALID_CONFIG,
                f"{len(records)} rows exceed the export limit {run.table.max_export_rows}",
                "Filter or paginate before exporting.",
            )
        root = self.predictions_dir / run.run_id
        target = self._export_directory(root, output_dir)
        frame = RiskTable(
            records=records,
            thresholds=run.table.thresholds,
            classification_threshold=run.table.classification_threshold,
            max_export_rows=run.table.max_export_rows,
        ).to_frame(include_segment=include_segment)
        exported: dict[str, str] = {}
        try:
            if "csv" in requested:
                exported["csv"] = _atomic_write_text(
                    target / "churn_scores.csv",
                    _sanitize_frame_for_csv(frame).to_csv(index=False),
                )
            if "parquet" in requested:
                path = target / "churn_scores.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                exported["parquet"] = str(path)
            if "json" in requested:
                payload = {
                    "schema_version": RISK_SCHEMA_VERSION,
                    "phase": 9,
                    "run_id": run.run_id,
                    "exported_at": self._now(),
                    "row_count": len(records),
                    "filters": {
                        "tiers": list(criteria.tiers),
                        "min_probability": criteria.min_probability,
                        "max_probability": criteria.max_probability,
                        "record_references": list(criteria.record_references),
                        "segment_equals": {
                            str(key): [str(value) for value in values]
                            for key, values in criteria.segment_equals.items()
                        },
                        "segment_min": {
                            str(key): float(value)
                            for key, value in criteria.segment_min.items()
                        },
                        "segment_max": {
                            str(key): float(value)
                            for key, value in criteria.segment_max.items()
                        },
                    },
                    "sort_by": str(sort_by),
                    "limit": None if limit is None else int(limit),
                    "offset": int(offset),
                    "classification_threshold": float(run.table.classification_threshold),
                    "risk_thresholds": run.table.thresholds.to_dict(),
                    "model": {
                        "family": run.table.model_family,
                        "model_run_id": run.table.model_run_id,
                        "model_sha256": run.table.model_sha256,
                        "preprocessing_run_id": run.table.preprocessing_run_id,
                    },
                    "records": [
                        {
                            **item.to_dict(include_segment=True),
                            "segment": dict(item.segment) if include_segment else {},
                        }
                        for item in records
                    ],
                    "warnings": list(run.table.warnings),
                }
                exported["json"] = _atomic_write_json(
                    target / "churn_scores.json", payload
                )
        except RiskError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert to a structured export failure
            raise RiskError(
                CODE_EXPORT_FAILED,
                f"export failed: {type(exc).__name__}: {exc}",
                "Check the export directory and available disk space.",
            ) from exc
        return RiskExport(
            run_id=run.run_id,
            paths=exported,
            row_count=len(records),
            formats=requested,
            warnings=[] if count else ["export contains no rows after filtering"],
        )

    # -- internals ----------------------------------------------------------
    def _probabilities(self, model: LoadedRiskModel, matrix: Any) -> np.ndarray:
        try:
            values = model.predict_proba(matrix)
        except Exception as exc:  # noqa: BLE001
            message = getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}"
            raise RiskError(
                CODE_PREDICTION_FAILED,
                f"the shipped model could not score the input: {message}",
                "Verify the model/preprocessing pair and the input matrix.",
            ) from exc
        array = np.asarray(values, dtype=float).reshape(-1)
        if len(array) != len(matrix):
            raise RiskError(
                CODE_PREDICTION_FAILED,
                f"model returned {len(array)} probabilities for {len(matrix)} rows",
                "Reload the verified model artifact.",
            )
        if not np.isfinite(array).all() or array.min() < 0.0 or array.max() > 1.0:
            raise RiskError(
                CODE_PREDICTION_FAILED,
                "model produced probabilities outside [0, 1]",
                "Reload a verified model artifact or retrain it.",
            )
        return np.clip(array, 0.0, 1.0)

    def _assessment(
        self,
        *,
        model: LoadedRiskModel,
        record_reference: str,
        input_position: int,
        probability: float,
        assessed_at: str,
        segment: Mapping[str, Any],
        warnings: Sequence[str],
        observed_churn: int | None = None,
    ) -> RiskAssessment:
        return RiskAssessment(
            record_reference=record_reference,
            input_position=int(input_position),
            churn_probability=float(probability),
            predicted_class=int(float(probability) >= model.threshold),
            risk_tier=self.thresholds.tier_for(float(probability)),
            classification_threshold=float(model.threshold),
            risk_thresholds=self.thresholds,
            model_family=model.family,
            model_run_id=model.model_run_id,
            model_sha256=model.model_sha256,
            preprocessing_run_id=model.preprocessing_run_id,
            assessed_at=assessed_at,
            positive_label=model.positive_label,
            observed_churn=observed_churn,
            segment=dict(segment),
            warnings=list(warnings),
        )

    def _references(
        self,
        frame: pd.DataFrame,
        *,
        reference_override: str | None,
        reference_column: str | None = None,
    ) -> tuple[list[str], list[str]]:
        dataset_id = self.model.dataset.dataset_id
        warnings: list[str] = []
        if reference_override is not None:
            if len(frame) != 1:
                raise RiskError(
                    CODE_INVALID_CONFIG,
                    "a single reference override cannot identify a batch",
                    "Leave reference_override unset for batch scoring.",
                )
            base = _opaque_reference(reference_override, dataset_id=dataset_id)
            return [base], ["a caller-supplied reference was hashed before storage"]
        column = (
            reference_column
            or self.config.reference_column
            or self.model.dataset.split_spec.id_column
        )
        references: list[str] = []
        if column and column in frame.columns:
            missing_rows = 0
            for position, value in enumerate(frame[column].tolist()):
                if _missing_reference(value):
                    references.append(f"risk_input_{position:06d}")
                    missing_rows += 1
                else:
                    references.append(_opaque_reference(value, dataset_id=dataset_id))
            if missing_rows == len(references):
                warnings.append(
                    f"reference column {column!r} is empty; positional references "
                    "were used"
                )
            elif missing_rows:
                warnings.append(
                    f"reference column {column!r} had {missing_rows} missing value(s); "
                    "positional references were used for those rows"
                )
        else:
            references = [
                f"risk_input_{index:06d}" for index in range(len(frame))
            ]
            if column:
                warnings.append(
                    f"reference column {column!r} is not present in the input; "
                    "positional references were used"
                )
            else:
                warnings.append(
                    "no identifier column was supplied; references are positional and "
                    "stable only for the same input order"
                )
        seen: dict[str, int] = {}
        for index, reference in enumerate(list(references)):
            count = seen.get(reference, 0) + 1
            seen[reference] = count
            if count > 1:
                references[index] = f"{reference}_dup{count}"
        if any(value > 1 for value in seen.values()):
            warnings.append(
                "duplicate record references were detected; suffixes keep rows distinct"
            )
        return references, warnings

    def _segments(
        self, frame: pd.DataFrame, fields: Sequence[str]
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for index in range(len(frame)):
            item: dict[str, Any] = {}
            for field in fields:
                if field in frame.columns:
                    value = _as_python(frame.iloc[index][field])
                    if isinstance(value, (dict, list, set, tuple)):
                        value = str(value)
                    item[str(field)] = value
            segments.append(item)
        return segments

    @staticmethod
    def _validate_record_types(
        chunk: pd.DataFrame,
        positions: Sequence[int],
        *,
        required: Sequence[str],
        strict: bool,
    ) -> tuple[list[int], list[tuple[int, str]]]:
        valid: list[int] = []
        problems: list[tuple[int, str]] = []
        for offset, position in enumerate(positions):
            row = chunk.iloc[offset]
            bad = [
                column
                for column in required
                if column in row.index
                and isinstance(row[column], (dict, list, set, tuple))
            ] if strict else []
            if bad:
                problems.append(
                    (
                        int(position),
                        "unsupported value type for required column(s): "
                        + ", ".join(str(name) for name in bad),
                    )
                )
            else:
                valid.append(int(position))
        return valid, problems

    @staticmethod
    def _row_warnings(row: pd.Series, *, required: Sequence[str]) -> list[str]:
        """Report missing values only for columns the fitted pipeline consumes."""
        present = [name for name in required if name in row.index]
        missing = int(row[present].isna().sum()) if present else 0
        if missing:
            return [
                f"{missing} missing value(s) were handled by the fitted Phase 5 pipeline"
            ]
        return []

    def _attach_drivers(
        self,
        records: Sequence[RiskAssessment],
        matrices: Mapping[int, np.ndarray],
        *,
        mode: str,
        top_n: int,
        selected_positions: Sequence[int] | None,
        explain_run_id: str | None,
        run_id: str | None,
        warnings: list[str],
        progress: Callable[[str], None] | None = None,
        persist: bool = True,
    ) -> None:
        if mode == "none":
            for item in records:
                item.explanation_status = "unavailable"
                item.explanation_reason = "explanation was not requested"
            return
        if not records:
            return
        ordered = sorted(
            records, key=lambda item: (-item.churn_probability, item.record_reference)
        )
        cap = max(0, int(self.config.max_driver_rows))
        if mode == "top":
            requested = max(1, int(top_n))
            chosen = ordered[: min(requested, cap)] if cap else []
            if requested > cap:
                warnings.append(
                    f"driver generation was capped at max_driver_rows={cap}; "
                    f"{requested - cap} requested explanation(s) were skipped"
                )
        elif mode == "selected":
            requested_positions = sorted({int(position) for position in selected_positions or ()})
            unknown = [
                position for position in requested_positions if position not in matrices
            ]
            if unknown:
                raise RiskError(
                    CODE_INVALID_INPUT,
                    "selected explanation positions are not in the scored batch: "
                    + ", ".join(str(position) for position in unknown),
                    "Use input_position values returned by this batch.",
                )
            chosen = [
                item for item in ordered if item.input_position in set(requested_positions)
            ]
            if len(chosen) > cap:
                chosen = chosen[:cap]
                warnings.append(
                    f"driver generation was capped at max_driver_rows={cap}; "
                    f"{len(requested_positions) - cap} selected explanation(s) were skipped"
                )
        else:
            chosen = ordered
            if len(chosen) > cap:
                chosen = chosen[:cap]
                warnings.append(
                    f"drivers were generated for the {cap} highest-risk rows "
                    f"(max_driver_rows={cap}); {len(records) - cap} rows are marked "
                    "unavailable"
                )
        if cap == 0:
            for item in records:
                item.explanation_status = "unavailable"
                item.explanation_reason = "driver generation is disabled (max_driver_rows=0)"
            return
        positions = [item.input_position for item in chosen]
        try:
            matrix = np.vstack([matrices[position] for position in positions])
            if explain_run_id:
                index = load_phase8_index(
                    self.settings,
                    explain_run_id,
                    model_sha256=self.model.model_sha256,
                    preprocessing_run_id=self.model.preprocessing_run_id,
                )
            else:
                index = explain_matrix_rows(
                    self.settings,
                    matrix,
                    positions=positions,
                    benchmark_run_id=self.model.benchmark_run_id,
                    explain_run_id=run_id,
                    top_k=self.config.top_k,
                    seed=self.settings.seed,
                    progress=progress,
                    persist=persist,
                )
            attach_index(records, index)
        except RiskError as exc:
            message = exc.message
        except Exception as exc:  # noqa: BLE001 - explanations are auxiliary evidence
            message = (
                "an unexpected explanation failure occurred "
                f"({type(exc).__name__}); probabilities and tiers remain valid"
            )
        else:
            chosen_positions = set(positions)
            for item in records:
                if item.input_position not in chosen_positions:
                    item.explanation_status = "unavailable"
                    item.explanation_reason = (
                        "this row was outside the selected explanation subset"
                        if mode == "selected"
                        else "only the highest-risk rows were explained to control cost"
                    )
            return
        warnings.append(f"explanations unavailable: {message}")
        for item in records:
            item.explanation_status = "unavailable"
            item.explanation_reason = message

    def _ad_hoc_driver_id(
        self,
        assessed_at: str,
        records: Sequence[RiskAssessment],
        mode: str,
    ) -> str | None:
        if mode == "none" or not records:
            return None
        digest = hashlib.sha256(
            json.dumps(
                {
                    "at": assessed_at,
                    "positions": [item.input_position for item in records],
                    "p": [round(item.churn_probability, 12) for item in records],
                    "mode": mode,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:10]
        stamp = re.sub(r"[^0-9A-Za-z]+", "", assessed_at)[:15]
        return f"{stamp}-risk-batch-drivers-{digest}"

    def _summary(
        self,
        *,
        run_id: str,
        model: LoadedRiskModel,
        split: str,
        records: Sequence[RiskAssessment],
        generated_at: str,
        warnings: Sequence[str],
        values_present: bool,
        value_column: str | None = None,
        received: int | None = None,
        rejected: int = 0,
    ) -> dict[str, Any]:
        probabilities = np.asarray([item.churn_probability for item in records], dtype=float)
        explained = sum(1 for item in records if item.explanation_status == "available")
        summary: dict[str, Any] = {
            "schema_version": RISK_SCHEMA_VERSION,
            "phase": 9,
            "run_id": run_id,
            "generated_at": generated_at,
            "n_rows": len(records),
            "model": {
                "family": model.family,
                "model_run_id": model.model_run_id,
                "model_sha256": model.model_sha256,
                "preprocessing_run_id": model.preprocessing_run_id,
                "benchmark_run_id": model.benchmark_run_id,
                "positive_label": model.positive_label,
            },
            "classification_threshold": float(model.threshold),
            "risk_thresholds": self.thresholds.to_dict(),
            "population": {
                "split": split,
                "received": int(received if received is not None else len(records)),
                "scored": len(records),
                "rejected": int(rejected),
            },
            "tier_counts": tier_counts(records, self.thresholds),
            "probability": {
                "min": float(probabilities.min()) if len(probabilities) else None,
                "max": float(probabilities.max()) if len(probabilities) else None,
                "mean": float(probabilities.mean()) if len(probabilities) else None,
                "median": float(np.median(probabilities)) if len(probabilities) else None,
            },
            "predicted_class_counts": {
                "churn": int(sum(item.predicted_class for item in records)),
                "retained": int(len(records) - sum(item.predicted_class for item in records)),
            },
            "explanations": {
                "available": int(explained),
                "unavailable": int(len(records) - explained),
            },
            "value_at_risk": {
                "available": bool(values_present),
                "column": value_column if values_present else None,
                "horizon_months": float(self.config.value_horizon_months)
                if values_present
                else None,
                "rows_with_value": int(
                    sum(1 for item in records if item.value_at_risk is not None)
                ),
            },
            "provenance": dict(model.provenance),
            "warnings": list(warnings),
            "limitations": self._limitations(values_present=values_present),
            "privacy": (
                "scored rows use opaque references; raw identifiers and unconfigured "
                "customer fields are not exported"
            ),
        }
        if split == "input":
            summary["observed_outcome"] = {
                "available": False,
                "note": "caller-supplied records have no observed outcome in this artifact",
            }
        elif split in {"val", "test", "all"}:
            summary["observed_outcome"] = {
                "available": True,
                "note": "observed labels are for audit only; they are not model predictions",
                "positive": int(sum(1 for item in records if item.observed_churn == 1)),
                "negative": int(sum(1 for item in records if item.observed_churn == 0)),
            }
        else:
            summary["observed_outcome"] = {
                "available": True,
                "note": "training labels are shown for audit only",
                "positive": int(sum(1 for item in records if item.observed_churn == 1)),
                "negative": int(sum(1 for item in records if item.observed_churn == 0)),
            }
        return summary

    @staticmethod
    def _limitations(*, values_present: bool) -> list[str]:
        limitations = [
            "Probabilities are model estimates, not guaranteed outcomes.",
            "SHAP drivers describe model behaviour, not causal effects.",
            "Risk tiers are configurable business categories, not calibrated decisions.",
        ]
        if not values_present:
            limitations.append(
                "No revenue proxy was supplied; value-at-risk ordering is unavailable "
                "and no monetary value was invented."
            )
        return limitations

    def _default_run_id(self, dataset_id: str, split: str, generated_at: str) -> str:
        stamp = re.sub(r"[^0-9A-Za-z]+", "", generated_at)[:15]
        digest = hashlib.sha256(
            json.dumps(
                {
                    "dataset": dataset_id,
                    "split": split,
                    "config": self.config.fingerprint(),
                    "model": self.model.model_sha256,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:8]
        return f"{stamp}-{_safe_name(dataset_id)}-risk-{digest}"

    def _persist(self, result: RiskRun) -> None:
        root = self.predictions_dir / result.run_id
        table = result.table
        # Persist only configured segment fields so stored runs can still be
        # filtered without exporting those fields by default.
        frame = table.to_frame(include_segment=True)
        try:
            root.mkdir(parents=True, exist_ok=True)
            parquet_path = root / "churn_scores.parquet"
            frame.to_parquet(parquet_path, index=False)
        except Exception as exc:  # noqa: BLE001
            raise RiskError(
                CODE_ARTIFACT_FAILED,
                f"scored table could not be persisted: {type(exc).__name__}: {exc}",
                "Check that the artifact directory is writable and pyarrow is installed.",
            ) from exc
        tier_rows = [
            {
                "tier": label,
                "count": int(data["count"]),
                "pct": round(float(data["pct"]), 8),
            }
            for label, data in table.tier_counts().items()
        ]
        tier_rows.append(
            {
                "tier": "total",
                "count": len(table.records),
                "pct": 1.0 if table.records else 0.0,
            }
        )
        tiers_path = _atomic_write_text(
            root / "tiers.csv", pd.DataFrame(tier_rows).to_csv(index=False)
        )
        summary = dict(result.summary)
        paths = {
            "root": str(root),
            "churn_scores": str(root / "churn_scores.parquet"),
            "risk_summary": str(root / "risk_summary.json"),
            "tiers": tiers_path,
            "run_manifest": str(root / "run_manifest.json"),
        }
        drivers = {
            item.record_reference: {
                "status": item.explanation_status,
                "reason": item.explanation_reason,
                "summary": item.explanation_summary,
                "top_positive": [driver.to_dict() for driver in item.top_positive],
                "top_negative": [driver.to_dict() for driver in item.top_negative],
            }
            for item in table.records
            if item.explanation_status != "unavailable"
        }
        if drivers:
            paths["drivers"] = _atomic_write_json(root / "drivers.json", drivers)
        if result.rejections:
            paths["rejections"] = _atomic_write_json(
                root / "rejections.json",
                {
                    "run_id": result.run_id,
                    "total_received": int(result.total_received),
                    "scored": len(table.records),
                    "rejected": len(result.rejections),
                    "rejections": [item.to_dict() for item in result.rejections],
                },
            )
        manifest = {
            "schema_version": RISK_SCHEMA_VERSION,
            "phase": 9,
            "run_id": result.run_id,
            "generated_at": table.generated_at,
            "duration_seconds": float(result.duration_seconds),
            "dataset_id": table.provenance.get("dataset_id"),
            "dataset_sha256": table.provenance.get("dataset_sha256"),
            "split_fingerprint": table.provenance.get("split_fingerprint"),
            "preprocessor_hash": table.provenance.get("preprocessor_hash"),
            "benchmark_run_id": table.provenance.get("benchmark_run_id"),
            "value_proxy_column": (summary.get("value_at_risk") or {}).get("column"),
            "model_family": table.model_family,
            "model_run_id": table.model_run_id,
            "model_sha256": table.model_sha256,
            "model_path": _portable_artifact_path(
                table.provenance.get("model_path"), self.settings.project_root
            ),
            "preprocessor_path": _portable_artifact_path(
                (self.model.dataset.paths or {}).get("preprocessor")
                or (self.model.dataset.schema or {}).get("preprocessor_path"),
                self.settings.project_root,
            ),
            "preprocessing_run_id": table.preprocessing_run_id,
            "config": self.config.to_dict(),
            "counts": {
                "received": int(summary.get("population", {}).get("received", len(table.records))),
                "scored": len(table.records),
                "rejected": int(summary.get("population", {}).get("rejected", 0)),
                "tiers": table.tier_counts(),
            },
            "rejections_path": paths.get("rejections"),
            "paths": paths,
            "warnings": list(result.warnings),
            "limitations": list(table.limitations),
            "privacy": (
                "opaque record references; no raw identifiers or unconfigured "
                "customer fields are written"
            ),
        }
        _atomic_write_json(root / "run_manifest.json", manifest)
        summary["paths"] = paths
        _atomic_write_json(root / "risk_summary.json", summary)
        result.paths = paths
        result.summary = summary
        result.manifest = manifest

    def _export_directory(self, root: Path, output_dir: str | Path | None) -> Path:
        run_root = root.resolve()
        target = run_root / "exports" if output_dir is None else Path(output_dir)
        if not target.is_absolute():
            target = root / target
        resolved = target.resolve()
        try:
            resolved.relative_to(run_root)
        except ValueError as exc:
            raise RiskError(
                CODE_EXPORT_FAILED,
                "export directory must stay inside this risk run's artifact directory",
                "Use a subdirectory of artifacts/predictions/<run_id>.",
            ) from exc
        if resolved == run_root:
            resolved = resolved / "exports"
        return resolved

    @staticmethod
    def _record_from_row(row: pd.Series, thresholds: RiskThresholds) -> RiskAssessment:
        def decoded(column: str, default: Any) -> Any:
            value = row.get(column)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return default
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except ValueError:
                    return default
            return value

        def drivers(column: str) -> list[RiskDriver]:
            return [
                RiskDriver(
                    feature=str(item.get("feature") or ""),
                    kind=str(item.get("kind") or "unknown"),
                    contribution=float(item.get("contribution", 0.0)),
                    direction=str(item.get("direction") or "neutral"),
                    value=item.get("value"),
                    active_level=item.get("active_level"),
                )
                for item in decoded(column, [])
                if isinstance(item, dict)
            ]

        observed = row.get("observed_churn")
        if observed is None or pd.isna(observed):
            observed = None
        else:
            observed = int(observed)
        return RiskAssessment(
            record_reference=str(row["record_reference"]),
            input_position=int(row["input_position"]),
            churn_probability=float(row["churn_probability"]),
            predicted_class=int(row["predicted_class"]),
            risk_tier=str(row["risk_tier"]),
            classification_threshold=float(row["classification_threshold"]),
            risk_thresholds=thresholds,
            model_family=str(row["model_family"]),
            model_run_id=str(row["model_run_id"]),
            model_sha256=row.get("model_sha256") if not pd.isna(row.get("model_sha256")) else None,
            preprocessing_run_id=str(row["preprocessing_run_id"]),
            assessed_at=str(row["assessed_at"]),
            positive_label=row.get("positive_label") if not pd.isna(row.get("positive_label")) else None,
            predicted_label=str(row.get("predicted_label") or ""),
            rank=None if pd.isna(row.get("rank")) else int(row["rank"]),
            observed_churn=observed,
            value_at_risk=(
                None
                if pd.isna(row.get("value_at_risk"))
                else float(row["value_at_risk"])
            ),
            value_horizon_months=(
                None
                if pd.isna(row.get("value_horizon_months"))
                else float(row["value_horizon_months"])
            ),
            top_positive=drivers("top_positive_drivers"),
            top_negative=drivers("top_negative_drivers"),
            explanation_summary=str(row.get("explanation_summary") or ""),
            explanation_status=str(row.get("explanation_status") or "unavailable"),
            explanation_reason=str(row.get("explanation_reason") or ""),
            segment=dict(decoded("segment", {})),
            warnings=list(decoded("warnings", [])),
        )


__all__ = ["RiskService", "sanitize_csv_value", "validate_risk_id"]
