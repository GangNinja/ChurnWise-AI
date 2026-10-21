"""Phase 5 orchestration: validated frame -> leakage-safe prepared dataset."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from churnwise import __version__
from churnwise.config import Settings
from churnwise.contracts import utc_now_iso
from churnwise.preprocessing import artifacts, reporting
from churnwise.preprocessing.contracts import (
    ClassBalance,
    PIPELINE_VERSION,
    PreprocessConfig,
    SCHEMA_VERSION,
    SplitData,
    SplitSpec,
)
from churnwise.preprocessing.dataset import PreparedDataset
from churnwise.preprocessing.errors import CODE_NOT_READY, PreprocessingError
from churnwise.preprocessing.features import derive_features
from churnwise.preprocessing.pipeline import (
    build_preprocessor,
    clean_frame,
    describe_outputs,
    drop_unusable_columns,
    plan_features,
    numeric_quality,
    plan_missing_indicators,
    transform_cleaned,
)
from churnwise.preprocessing.split import make_split

RECENT_RUN_LIMIT = 50


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[:120] or "dataset"


def _balanced_weights(y: np.ndarray) -> dict[str, float]:
    counts = np.bincount(y, minlength=2).astype(float)
    total = float(counts.sum())
    return {
        "0": float(total / (2.0 * counts[0])) if counts[0] else 1.0,
        "1": float(total / (2.0 * counts[1])) if counts[1] else 1.0,
    }


def _oversample_positions(y: np.ndarray, ratio: float, seed: int) -> list[int]:
    counts = np.bincount(y, minlength=2)
    if counts[1] <= counts[0]:
        minority, majority = 1, 0
    else:
        minority, majority = 0, 1
    target_minority = int(round(ratio * counts[majority]))
    extra = target_minority - counts[minority]
    if extra <= 0:
        return []
    positions = np.flatnonzero(y == minority)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(positions, size=extra, replace=True)
    return sorted(int(index) for index in chosen)


def _class_balance(
    y_all: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    weights: dict[str, float],
    strategy: str,
    resampled_train_rows: int,
) -> ClassBalance:
    splits = {"all": y_all, "train": y_train, "val": y_val, "test": y_test}
    counts: dict[str, dict[str, int]] = {}
    rates: dict[str, float | None] = {}
    for name, values in splits.items():
        observed = np.bincount(values, minlength=2)
        counts[name] = {"0": int(observed[0]), "1": int(observed[1])}
        rates[name] = float(values.mean()) if len(values) else None
    majority = max(counts["all"].values())
    minority = min(counts["all"].values())
    ratio = float(majority / minority) if minority else None
    return ClassBalance(
        counts=counts,
        rates=rates,
        strategy=strategy,
        class_weights=dict(weights),
        ratio=ratio,
        resampled=bool(resampled_train_rows),
        resampled_train_rows=resampled_train_rows,
    )


def _encoder_categories(preprocessor: Any, categorical: list[str]) -> dict[str, list[str]]:
    """Record the fitted category vocabulary per categorical source column."""
    try:
        encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
        fitted = list(encoder.categories_)
    except (AttributeError, KeyError):
        return {}
    return {
        name: [str(value) for value in values]
        for name, values in zip(categorical, fitted)
    }


def _iqr_outliers(cleaned_train: pd.DataFrame, numeric: list[str]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name in numeric:
        values = cleaned_train[name].dropna()
        if values.empty:
            continue
        q25, q75 = values.quantile([0.25, 0.75])
        iqr = float(q75 - q25)
        lower = float(q25 - 1.5 * iqr)
        upper = float(q75 + 1.5 * iqr)
        mask = (values < lower) | (values > upper)
        report[name] = {
            "count": int(mask.sum()),
            "pct": float(mask.mean()),
            "lower": lower,
            "upper": upper,
            "treated": False,
        }
    return report


class PreprocessingCoordinator:
    def __init__(
        self,
        settings: Settings,
        *,
        now: Callable[[], str] = utc_now_iso,
        code_version: str = __version__,
    ) -> None:
        self.settings = settings
        self.artifact_dir = Path(settings.artifact_dir)
        self.report_dir = Path(settings.report_dir)
        self._now = now
        self.code_version = code_version

    # -- preparation --------------------------------------------------------
    def prepare(
        self,
        validation: Any,
        *,
        config: PreprocessConfig | None = None,
        persist: bool = True,
        run_id: str | None = None,
    ) -> PreparedDataset:
        config = config or PreprocessConfig(seed=self.settings.seed)
        problems = config.validate()
        if problems:
            raise PreprocessingError(
                CODE_NOT_READY,
                "invalid preprocessing configuration: " + "; ".join(problems),
                "Correct the prepare command options and run again.",
            )
        if not getattr(validation, "can_preprocess", False):
            gate = getattr(getattr(validation, "report", None), "gate", None)
            blocking = list(getattr(gate, "blocking", []) or [])
            reasons = list(getattr(gate, "reasons", []) or [])
            raise PreprocessingError(
                CODE_NOT_READY,
                "Phase 4 did not approve this dataset for preprocessing",
                "Blocking checks: "
                + (", ".join(blocking) if blocking else "validation was not completed")
                + (f". {reasons[0]}" if reasons else ""),
            )
        report = validation.report
        profile = validation.profile
        frame = validation.frame
        if report is None or profile is None or frame is None:
            raise PreprocessingError(
                CODE_NOT_READY,
                "validation result has no report, profile or loaded frame",
                "Run the validate command first and pass its result to prepare.",
            )
        target = profile.target.column
        if not target or target not in frame.columns:
            raise PreprocessingError(
                CODE_NOT_READY,
                "the validated target column is not present in the frame",
                "Re-run Phase 4 validation.",
            )

        generated_at = self._now()
        dataset_id = str(report.dataset_id)
        dataset_sha256 = str(report.artifact_sha256)
        run_id = run_id or self._make_run_id(dataset_id, config, generated_at)
        artifacts.validate_run_id(run_id)

        normalized = validation.normalized_frame()
        if normalized is None:
            raise PreprocessingError(
                CODE_NOT_READY,
                "the validated target could not be normalised to 1 = churned",
                "Inspect the V5 check evidence in the Phase 4 report.",
            )
        target_missing_dropped = int(len(frame) - len(normalized))

        derived = derive_features(normalized, profile, config)
        base = derived.frame
        duplicate_rows_dropped = 0
        if config.drop_duplicate_rows:
            before = len(base)
            base = base.drop_duplicates(ignore_index=True)
            duplicate_rows_dropped = int(before - len(base))

        split = make_split(
            base,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
            run_id=run_id,
            target_column=target,
            id_column=getattr(profile.roles, "id", None),
            config=config,
            duplicate_rows_dropped=duplicate_rows_dropped,
            target_missing_dropped=target_missing_dropped,
        )
        spec = split.spec
        train_frame = base.iloc[split.train_indices].reset_index(drop=True)
        val_frame = base.iloc[split.val_indices].reset_index(drop=True)
        test_frame = base.iloc[split.test_indices].reset_index(drop=True)
        y_train = train_frame[target].to_numpy(dtype=np.int8)
        y_val = val_frame[target].to_numpy(dtype=np.int8)
        y_test = test_frame[target].to_numpy(dtype=np.int8)
        y_all = base[target].to_numpy(dtype=np.int8)

        plan = plan_features(train_frame, profile, config, derived.applied)
        cleaned_train_base = clean_frame(train_frame, plan)
        drop_unusable_columns(plan, cleaned_train_base)
        plan_missing_indicators(
            cleaned_train_base,
            plan,
            config,
            derived_names={spec["name"] for spec in derived.applied},
        )
        cleaned_train = clean_frame(train_frame, plan)
        cleaned_val = clean_frame(val_frame, plan)
        cleaned_test = clean_frame(test_frame, plan)

        preprocessor = build_preprocessor(plan, config)
        preprocessor.fit(cleaned_train)
        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
        X_train = transform_cleaned(preprocessor, cleaned_train, expected_width=len(feature_names))
        X_val = transform_cleaned(preprocessor, cleaned_val, expected_width=len(feature_names))
        X_test = transform_cleaned(preprocessor, cleaned_test, expected_width=len(feature_names))

        if config.imbalance == "class_weight":
            weights = _balanced_weights(y_train)
            oversample: list[int] = []
        elif config.imbalance == "oversample":
            weights = {"0": 1.0, "1": 1.0}
            oversample = _oversample_positions(y_train, config.oversample_ratio, config.seed)
        else:
            weights = {"0": 1.0, "1": 1.0}
            oversample = []

        balance = _class_balance(
            y_all,
            y_train,
            y_val,
            y_test,
            weights,
            config.imbalance,
            len(oversample),
        )
        warnings = [*split.warnings, *plan.warnings]
        if config.imbalance == "oversample" and oversample:
            warnings.append(
                f"training partition oversampled by {len(oversample)} minority rows; validation and test are untouched"
            )
        if balance.ratio is not None and balance.ratio >= 20:
            warnings.append(
                f"class imbalance ratio is {balance.ratio:.1f}:1; prefer class weights or PR-AUC reporting"
            )

        schema = self._build_schema(
            run_id=run_id,
            dataset_id=dataset_id,
            dataset_sha256=dataset_sha256,
            generated_at=generated_at,
            config=config,
            spec=spec,
            profile=profile,
            plan=plan,
            derived=derived,
            feature_names=feature_names,
            balance=balance,
            warnings=warnings,
            outliers=_iqr_outliers(cleaned_train_base, plan.numeric),
            invalid_numeric=numeric_quality(train_frame, cleaned_train_base, plan.numeric),
            categories=_encoder_categories(preprocessor, plan.categorical),
            validation=validation,
        )
        paths: dict[str, str] = {}
        if persist:
            paths["split"] = artifacts.save_split_spec(self.artifact_dir, spec)
            pipeline_path, preprocessor_hash = artifacts.save_preprocessor(
                self.artifact_dir, run_id, preprocessor, schema
            )
            paths["preprocessor"] = pipeline_path
            schema["preprocessor_hash"] = preprocessor_hash
            schema["preprocessor_path"] = pipeline_path
            arrays = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
                "train_rows": np.asarray(spec.row_indices["train"], dtype=np.int64),
                "val_rows": np.asarray(spec.row_indices["val"], dtype=np.int64),
                "test_rows": np.asarray(spec.row_indices["test"], dtype=np.int64),
                "oversample_indices": np.asarray(oversample, dtype=np.int64),
            }
            paths["arrays"] = artifacts.save_arrays(self.artifact_dir, run_id, arrays)

        dataset = PreparedDataset(
            run_id=run_id,
            dataset_id=dataset_id,
            config=config,
            split_spec=spec,
            schema=schema,
            train=SplitData("train", X_train, y_train, list(spec.row_indices["train"])),
            val=SplitData("val", X_val, y_val, list(spec.row_indices["val"])),
            test=SplitData("test", X_test, y_test, list(spec.row_indices["test"])),
            class_balance=balance,
            preprocessor=preprocessor,
            warnings=warnings,
            provenance=schema["provenance"],
            paths=paths,
            oversample_indices=oversample,
        )
        if persist:
            paths["summary"] = artifacts.save_summary(
                self.artifact_dir, run_id, dataset.to_dict()
            )
            report_paths = reporting.write_reports(dataset, self.report_dir)
            paths.update(report_paths)
        return dataset

    def load(self, run_id: str) -> PreparedDataset:
        """Reload a persisted run and verify its digests before use."""
        spec = artifacts.load_split_spec(self.artifact_dir, run_id)
        preprocessor, schema = artifacts.load_preprocessor(self.artifact_dir, run_id)
        arrays = artifacts.load_arrays(self.artifact_dir, run_id)
        schema["preprocessor_hash"] = schema.get("preprocessor_hash")
        config_payload = dict(schema.get("config") or {})
        config_payload.pop("fingerprint", None)
        try:
            config = PreprocessConfig(**config_payload)
        except TypeError as exc:
            raise PreprocessingError(
                CODE_NOT_READY,
                f"stored preprocessing config is not readable: {exc}",
                "Regenerate the run with this project version.",
            ) from exc
        balance_payload = dict(schema.get("class_balance") or {})
        balance = ClassBalance(
            counts=balance_payload.get("counts", {}),
            rates=balance_payload.get("rates", {}),
            strategy=balance_payload.get("strategy", config.imbalance),
            class_weights=balance_payload.get("class_weights", {}),
            ratio=balance_payload.get("ratio"),
            resampled=bool(balance_payload.get("resampled", False)),
            resampled_train_rows=int(balance_payload.get("resampled_train_rows", 0)),
        )
        dataset = PreparedDataset(
            run_id=run_id,
            dataset_id=str(schema.get("dataset_id", spec.dataset_id)),
            config=config,
            split_spec=spec,
            schema=schema,
            train=SplitData("train", arrays["X_train"], arrays["y_train"], arrays["train_rows"].tolist()),
            val=SplitData("val", arrays["X_val"], arrays["y_val"], arrays["val_rows"].tolist()),
            test=SplitData("test", arrays["X_test"], arrays["y_test"], arrays["test_rows"].tolist()),
            class_balance=balance,
            preprocessor=preprocessor,
            warnings=list(schema.get("warnings") or []),
            provenance=dict(schema.get("provenance") or {}),
            paths={
                "split": str(artifacts.paths_for(self.artifact_dir, run_id)["split"]),
                "preprocessor": str(artifacts.paths_for(self.artifact_dir, run_id)["pipeline"]),
                "schema": str(artifacts.paths_for(self.artifact_dir, run_id)["schema"]),
                "arrays": str(artifacts.paths_for(self.artifact_dir, run_id)["arrays"]),
                "report_json": str(self.report_dir / "preprocessing" / f"{run_id}.json"),
                "report_md": str(self.report_dir / "preprocessing" / f"{run_id}.md"),
            },
            oversample_indices=arrays["oversample_indices"].tolist(),
        )
        return dataset

    # -- helpers ------------------------------------------------------------
    def _make_run_id(
        self, dataset_id: str, config: PreprocessConfig, generated_at: str
    ) -> str:
        stamp = generated_at.replace("-", "").replace(":", "")
        if not stamp.endswith("Z"):
            stamp += "Z"
        return f"{stamp}-{_safe_name(dataset_id)}-{config.fingerprint()[:8]}"

    def _build_schema(
        self,
        *,
        run_id: str,
        dataset_id: str,
        dataset_sha256: str,
        generated_at: str,
        config: PreprocessConfig,
        spec: SplitSpec,
        profile: Any,
        plan: Any,
        derived: Any,
        feature_names: list[str],
        balance: ClassBalance,
        warnings: list[str],
        outliers: dict[str, Any],
        invalid_numeric: dict[str, Any],
        categories: dict[str, list[str]],
        validation: Any,
    ) -> dict[str, Any]:
        target = profile.target
        derived_names = {item["name"] for item in derived.applied}
        base_features = [
            name
            for name in plan.feature_columns
            if name not in derived_names and not name.endswith("__missing")
        ]
        required_columns = sorted(
            {
                *base_features,
                *(source for item in derived.applied for source in item["sources"]),
            }
        )
        excluded = [
            {"column": str(item["column"]), "reason": str(item["reason"]), "stage": "phase4"}
            for item in profile.roles.dropped
        ]
        excluded.extend(
            {"column": item["column"], "reason": item["reason"], "stage": "phase5"}
            for item in plan.dropped
        )
        if profile.roles.id:
            excluded.append(
                {
                    "column": str(profile.roles.id),
                    "reason": "identifier retained for output only; never a feature",
                    "stage": "phase5",
                }
            )
        excluded.append(
            {
                "column": str(target.column),
                "reason": "prediction target",
                "stage": "phase4",
            }
        )
        report = validation.report
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": 5,
            "pipeline_version": PIPELINE_VERSION,
            "run_id": run_id,
            "dataset_id": dataset_id,
            "dataset_sha256": dataset_sha256,
            "manifest_sha256": str(getattr(report, "manifest_sha256", "")),
            "generated_at": generated_at,
            "code_version": self.code_version,
            "seed": spec.seed,
            "split_fingerprint": spec.fingerprint,
            "config": config.to_dict(),
            "target": {
                "column": target.column,
                "positive_label": target.positive_label,
                "negative_label": target.negative_label,
                "mapping": dict(target.mapping or {}),
                "positive_rate": target.positive_rate,
            },
            "required_columns": required_columns,
            "preprocessing": {
                "numeric": list(plan.numeric),
                "boolean": list(plan.boolean),
                "categorical": list(plan.categorical),
                "numeric_coerced": list(plan.numeric_coerced),
                "missing_indicators": list(plan.missing_indicators),
                "numeric_imputation": config.numeric_imputation,
                "categorical_imputation": config.categorical_imputation,
                "missing_category": (
                    "__missing__" if config.categorical_imputation == "missing_category" else None
                ),
                "scaling": config.scaling,
                "encoding": "one_hot",
                "handle_unknown": "ignore" if config.min_frequency <= 0 else "infrequent_if_exist",
                "min_frequency": config.min_frequency,
                "max_categorical_levels": config.max_categorical_levels,
                "categories": categories,
                "output_dtype": "float64",
            },
            "derived": list(derived.applied),
            "derived_skipped": list(derived.skipped),
            "feature_names": list(feature_names),
            "output_features": describe_outputs(feature_names, plan),
            "class_weights": dict(balance.class_weights),
            "class_balance": balance.to_dict(),
            "excluded": excluded,
            "outliers": outliers,
            "invalid_numeric": invalid_numeric,
            "warnings": list(warnings),
            "provenance": {
                "dataset_id": dataset_id,
                "dataset_sha256": dataset_sha256,
                "manifest_sha256": str(getattr(report, "manifest_sha256", "")),
                "validation_report": getattr(report, "report_json_path", None),
                "source": dict(getattr(report, "provenance", {}) or {}),
                "generated_at": generated_at,
                "code_version": self.code_version,
            },
        }


__all__ = ["PreprocessingCoordinator", "PreprocessingError"]
