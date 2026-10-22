"""Phase 8 orchestration: load verified models, explain, persist, trace.

Model artifacts are opened read-only. The coordinator verifies feature names and
split fingerprints before explaining so a SHAP value is always tied to the exact
model/preprocessing pair that produced the prediction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from churnwise.config import Settings
from churnwise.contracts import utc_now_iso
from churnwise.explain.contracts import (
    EXPLAIN_SCHEMA_VERSION,
    ExplainConfig,
    ExplainResult,
    GlobalExplanation,
    LocalExplanation,
)
from churnwise.explain.engine import (
    chunked_explain,
    deterministic_sample,
    local_explanations,
    summarise_global,
)
from churnwise.explain.errors import (
    CODE_ARTIFACT_FAILED,
    CODE_INVALID_CONFIG,
    CODE_NO_MODEL,
    CODE_UNSUPPORTED_MODEL,
    ExplainError,
)
from churnwise.explain.explainers import ModelExplainer, build_explainer
from churnwise.explain.features import (
    build_feature_groups,
    describe_mapping,
    mapped_columns,
)
from churnwise.explain.plots import (
    plot_beeswarm,
    plot_dependence,
    plot_importance,
    plot_waterfall,
)
from churnwise.models.baselines import FittedBaseline, load_baseline
from churnwise.models.registry import ModelRegistry, validate_model_run_id
from churnwise.preprocessing import PreprocessingCoordinator

EXPLAIN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,180}$")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[:110] or "dataset"


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
    except (OSError, ValueError) as exc:
        raise ExplainError(
            CODE_ARTIFACT_FAILED,
            f"explanation artifact could not be written: {path}: "
            f"{type(exc).__name__}: {exc}",
            "Check that the artifact directory is writable and the payload is "
            "JSON-safe; do not report partial values.",
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExplainError(
            CODE_ARTIFACT_FAILED,
            f"artifact could not be read: {path}: {type(exc).__name__}: {exc}",
            "Rerun the upstream phase to regenerate the artifact.",
        ) from exc
    if not isinstance(payload, dict):
        raise ExplainError(
            CODE_ARTIFACT_FAILED,
            f"artifact {path.name} is not a JSON object",
            "Rerun the upstream phase to regenerate the artifact.",
        )
    return payload


def validate_explain_id(run_id: str) -> str:
    if not EXPLAIN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise ExplainError(
            CODE_INVALID_CONFIG,
            f"invalid explanation run id {run_id!r}",
            "Use the id printed by a previous explain command.",
        )
    return run_id


@dataclass
class LoadedExplanationModel:
    """A verified model plus the preprocessing and evaluation evidence around it."""

    family: str
    model: Any
    predict_proba: Callable[[np.ndarray], np.ndarray]
    dataset: Any
    threshold: float
    positive_label: str | None
    provenance: dict[str, Any] = field(default_factory=dict)


class ExplainCoordinator:
    """Run and persist Phase 8 explanations for one model family."""

    def __init__(
        self,
        settings: Settings,
        config: ExplainConfig | None = None,
        *,
        now: Any = utc_now_iso,
    ) -> None:
        self.settings = settings
        self.config = config or ExplainConfig(seed=settings.seed)
        self.config.validate_or_raise()
        self.artifact_dir = Path(settings.artifact_dir)
        self.explanation_dir = self.artifact_dir / "explanations"
        self.evaluation_dir = self.artifact_dir / "evaluation"
        self._now = now

    # -- discovery ---------------------------------------------------------
    def latest_benchmark_run(self) -> str | None:
        if not self.evaluation_dir.is_dir():
            return None
        candidates = [
            path.name
            for path in self.evaluation_dir.iterdir()
            if path.is_dir() and (path / "metrics.json").is_file()
        ]
        return sorted(candidates)[-1] if candidates else None

    def _default_family(self, metrics: Mapping[str, Any] | None) -> str:
        selection = (metrics or {}).get("selection") or {}
        family = str(selection.get("selected_family") or "")
        if family:
            return family
        families = (metrics or {}).get("families") or {}
        for name in ("random_forest", "logistic_regression", "mlp"):
            if (families.get(name) or {}).get("status") == "completed":
                return name
        raise ExplainError(
            CODE_NO_MODEL,
            "no completed model family is available to explain",
            "Run `python -m churnwise.cli evaluate` first.",
        )

    def _load_metrics(self, benchmark_run_id: str | None) -> tuple[str, dict[str, Any]]:
        run_id = benchmark_run_id or self.latest_benchmark_run()
        if not run_id:
            raise ExplainError(
                CODE_NO_MODEL,
                "no Phase 7 benchmark artifact was found",
                "Run `python -m churnwise.cli evaluate` before explaining.",
            )
        validate_explain_id(run_id)
        path = self.evaluation_dir / run_id / "metrics.json"
        if not path.is_file():
            raise ExplainError(
                CODE_NO_MODEL,
                f"benchmark {run_id!r} has no metrics.json",
                "Pass a benchmark run id printed by the evaluate command.",
            )
        return run_id, _read_json(path)

    def load_model(
        self,
        family: str | None = None,
        *,
        benchmark_run_id: str | None = None,
        model_run_id: str | None = None,
    ) -> LoadedExplanationModel:
        """Load the documented model family and its exact preprocessing run."""
        if family is not None and family not in {
            "logistic_regression",
            "random_forest",
            "mlp",
        }:
            raise ExplainError(
                CODE_UNSUPPORTED_MODEL,
                f"unsupported model family {family!r}",
                "Use logistic_regression, random_forest or mlp.",
            )
        try:
            run_id, metrics = self._load_metrics(benchmark_run_id)
        except ExplainError as exc:
            # An MLP can be explained without a Phase 7 benchmark. Do not hide a
            # corrupt benchmark artifact or an explicitly requested missing run.
            if (
                family != "mlp"
                or benchmark_run_id is not None
                or exc.code != CODE_NO_MODEL
            ):
                raise
            run_id, metrics = "", {}
        resolved_family = family or self._default_family(metrics)
        block = ((metrics.get("families") or {}).get(resolved_family) or {})
        threshold_block = block.get("threshold")
        if not isinstance(threshold_block, Mapping):
            threshold_block = {}
        benchmark_threshold = threshold_block.get("selected")
        if benchmark_threshold is None:
            benchmark_threshold = threshold_block.get("default")
        threshold = self._threshold(benchmark_threshold, 0.5)
        preprocessing = dict((metrics.get("preprocessing") or {}))
        model_info = dict(block.get("model") or {})

        if resolved_family == "mlp":
            registry = ModelRegistry(self.settings)
            selected = (
                model_run_id
                or str(model_info.get("model_run_id") or "")
                or registry.latest_model_run_id()
            )
            if not selected:
                raise ExplainError(
                    CODE_NO_MODEL,
                    "no Phase 6 MLP artifact was found",
                    "Run `python -m churnwise.cli train` first.",
                )
            validate_model_run_id(selected)
            loaded = registry.load(selected)
            metadata = loaded.metadata
            model = loaded.model

            def predict(matrix: np.ndarray) -> np.ndarray:
                return np.asarray(model.predict(matrix, verbose=0), dtype=np.float64).reshape(-1)

            preprocess_run = str(
                (metadata.get("preprocessing") or {}).get("run_id", "")
            )
            if not preprocess_run:
                raise ExplainError(
                    CODE_NO_MODEL,
                    "MLP metadata has no preprocessing run id",
                    "Retrain the MLP with this project version.",
                )
            dataset = PreprocessingCoordinator(self.settings).load(preprocess_run)
            if benchmark_threshold is None:
                threshold = self._threshold(
                    (metadata.get("training") or {}).get("threshold"), 0.5
                )
            provenance = {
                "model_run_id": selected,
                "model_sha256": metadata.get("model_sha256"),
                "model_path": loaded.paths.get("model"),
                "framework": "tensorflow.keras",
                "preprocessing_run_id": preprocess_run,
                "benchmark_run_id": run_id or None,
            }
        else:
            if not run_id:
                raise ExplainError(
                    CODE_NO_MODEL,
                    "baseline explanations require a Phase 7 benchmark run",
                    "Run `python -m churnwise.cli evaluate` first.",
                )
            model_info = dict(block.get("model") or {})
            directory = self.evaluation_dir / run_id / "models" / resolved_family
            metadata_path = directory / "metadata.json"
            if not metadata_path.is_file():
                raise ExplainError(
                    CODE_NO_MODEL,
                    f"no persisted {resolved_family} artifact exists for benchmark {run_id}",
                    "Rerun `evaluate` so Phase 7 persists the fitted baselines.",
                )
            metadata = _read_json(metadata_path)
            fitted: FittedBaseline = load_baseline(
                directory / "model.joblib",
                expected_sha256=metadata.get("model_sha256"),
            )
            if fitted.family != resolved_family:
                raise ExplainError(
                    CODE_NO_MODEL,
                    f"artifact family {fitted.family!r} does not match {resolved_family!r}",
                    "Rerun the evaluation; do not rename artifact directories.",
                )

            def predict(matrix: np.ndarray) -> np.ndarray:
                return fitted.predict_proba(matrix)

            preprocess_run = str(
                (metadata.get("preprocessing") or {}).get("run_id")
                or preprocessing.get("run_id")
                or ""
            )
            if not preprocess_run:
                raise ExplainError(
                    CODE_NO_MODEL,
                    "baseline metadata has no preprocessing run id",
                    "Rerun the evaluation so provenance is recorded.",
                )
            dataset = PreprocessingCoordinator(self.settings).load(preprocess_run)
            provenance = {
                "model_run_id": f"{run_id}/{resolved_family}",
                "model_sha256": metadata.get("model_sha256"),
                "model_path": str(directory / "model.joblib"),
                "framework": "scikit-learn",
                "preprocessing_run_id": preprocess_run,
                "benchmark_run_id": run_id,
                "config": dict(metadata.get("config") or {}),
            }
        self._verify_preprocessing(metadata, dataset, resolved_family)
        names = [str(name) for name in getattr(dataset, "feature_names", []) or []]
        if not names:
            raise ExplainError(
                CODE_NO_MODEL,
                "the preprocessing run has no feature names",
                "Regenerate the Phase 5 run.",
            )
        if model_info.get("feature_count") and int(model_info["feature_count"]) != len(names):
            raise ExplainError(
                CODE_NO_MODEL,
                "model and preprocessing feature counts differ",
                "Reload the model built from this preprocessing run.",
            )
        target = dict((dataset.schema or {}).get("target") or {})
        return LoadedExplanationModel(
            family=resolved_family,
            model=model if resolved_family == "mlp" else fitted.estimator,
            predict_proba=predict,
            dataset=dataset,
            threshold=float(threshold),
            positive_label=target.get("positive_label"),
            provenance=provenance,
        )

    # -- runs --------------------------------------------------------------
    def run(
        self,
        family: str | None = None,
        *,
        benchmark_run_id: str | None = None,
        model_run_id: str | None = None,
        explain_run_id: str | None = None,
        split: str = "val",
        persist: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> ExplainResult:
        """Explain a validation sample and persist global + local artifacts."""
        if explain_run_id is not None:
            validate_explain_id(explain_run_id)
        loaded = self.load_model(
            family,
            benchmark_run_id=benchmark_run_id,
            model_run_id=model_run_id,
        )
        dataset = loaded.dataset
        resolved_family = loaded.family
        partitions = {
            "train": np.asarray(dataset.train.X, dtype=np.float64),
            "val": np.asarray(dataset.val.X, dtype=np.float64),
            "test": np.asarray(dataset.test.X, dtype=np.float64),
        }
        if split not in {"train", "val"}:
            raise ExplainError(
                CODE_INVALID_CONFIG,
                "explanations use train or validation rows only; the test partition "
                "is reserved for evaluation",
                "Pass split='val' (default) or split='train'.",
            )
        matrix = partitions[split]
        if matrix.ndim != 2 or len(matrix) == 0:
            raise ExplainError(
                CODE_NO_MODEL,
                f"{split} partition is empty",
                "Regenerate the Phase 5 split.",
            )
        names = list(dataset.feature_names)
        if matrix.shape[1] != len(names):
            raise ExplainError(
                CODE_NO_MODEL,
                f"{split} matrix has {matrix.shape[1]} columns but the run lists "
                f"{len(names)} features",
                "Reload the Phase 5 run.",
            )
        groups = build_feature_groups(names, dataset.schema.get("output_features") or [])
        output_features = list(dataset.schema.get("output_features") or [])
        unmapped = [
            name
            for name in names
            if not any(
                str(item.get("name")) == name and item.get("source")
                for item in output_features
            )
        ]

        background_positions = deterministic_sample(
            len(partitions["train"]),
            self.config.background_limit(resolved_family),
            self.config.seed,
        )
        background = partitions["train"][background_positions]
        sample_positions = deterministic_sample(
            len(matrix), self.config.sample_limit(resolved_family), self.config.seed
        )
        sample_matrix = matrix[sample_positions]

        started = time.perf_counter()
        if progress is not None:
            progress(f"initialising explainer for {resolved_family}")
        explainer = build_explainer(
            resolved_family,
            loaded.model,
            loaded.predict_proba,
            background=background,
            feature_names=names,
            config=self.config,
        )
        explanation = chunked_explain(
            explainer, sample_matrix, config=self.config, progress=progress
        )
        generated_at = self._now()
        run_id = explain_run_id or self._default_run_id(
            dataset.dataset_id, resolved_family, generated_at, loaded
        )
        validate_explain_id(run_id)
        root = self.explanation_dir / run_id
        sample_info = {
            "split": split,
            "sampled_rows": int(len(sample_matrix)),
            "available_rows": int(len(matrix)),
            "sampling": "seeded uniform without replacement",
            "seed": int(self.config.seed),
            "positions": [int(value) for value in sample_positions],
            "test_rows_used": 0,
        }
        provenance = {
            "dataset_id": dataset.dataset_id,
            "dataset_sha256": dataset.split_spec.dataset_sha256,
            "split_fingerprint": dataset.split_spec.fingerprint,
            "preprocessing_run_id": dataset.run_id,
            **loaded.provenance,
        }
        global_block, grouped = summarise_global(
            run_id=run_id,
            family=resolved_family,
            explainer=explainer,
            explanation=explanation,
            matrix=sample_matrix,
            groups=groups,
            sample=sample_info,
            provenance=provenance,
            config=self.config,
            duration_seconds=time.perf_counter() - started,
        )
        global_block.generated_at = generated_at
        if unmapped:
            global_block.warnings.append(
                f"{len(unmapped)} transformed columns had no Phase 5 source mapping "
                "and use documented fallback labels."
            )
        global_block.explainer["mapping"] = describe_mapping(groups)
        result = ExplainResult(
            run_id=run_id,
            family=resolved_family,
            global_explanation=global_block,
            warnings=list(global_block.warnings),
            manifest={},
        )
        if not persist:
            result.warnings.append(
                "persist=False: global values were computed but no local examples or "
                "figures were written."
            )
            return result

        if progress is not None:
            progress("writing global artifacts")
        paths: dict[str, str] = {}
        paths["global_json"] = _atomic_write_json(root / "global.json", global_block.to_dict())
        try:
            paths["importance_png"] = plot_importance(
                global_block.feature_importance,
                root / "importance.png",
                title=f"{resolved_family} — mean |SHAP| by source feature",
                output_space=global_block.output_space,
            )
            global_block.plots["importance"] = paths["importance_png"]
        except ExplainError as exc:
            result.warnings.append(exc.message)
            global_block.warnings.append(exc.message)
        try:
            paths["summary_png"] = plot_beeswarm(
                explanation.values,
                sample_matrix,
                mapped_columns(names, groups),
                root / "summary.png",
                output_space=global_block.output_space,
            )
            global_block.plots["summary"] = paths["summary_png"]
        except ExplainError as exc:
            result.warnings.append(exc.message)
            global_block.warnings.append(exc.message)

        numeric_groups = {
            group.source: group
            for group in groups
            if group.kind == "numeric" and len(group.columns) == 1
        }
        top_numeric = next(
            (
                numeric_groups[row["feature"]]
                for row in global_block.feature_importance
                if row["feature"] in numeric_groups
            ),
            None,
        )
        if top_numeric is not None:
            column = names.index(top_numeric.columns[0])
            try:
                paths["dependence_png"] = plot_dependence(
                    top_numeric.source,
                    sample_matrix[:, column],
                    explanation.values[:, column],
                    root / f"dependence_{_safe_name(top_numeric.source)}.png",
                    output_space=global_block.output_space,
                )
                global_block.plots["dependence"] = paths["dependence_png"]
            except ExplainError as exc:
                result.warnings.append(exc.message)
                global_block.warnings.append(exc.message)

        local_limit = min(
            int(self.config.local_examples),
            int(self.config.max_local_batch),
            len(sample_matrix),
        )
        local_blocks: list[LocalExplanation] = []
        if local_limit > 0:
            local_indices = self._local_indices(explanation.predictions, local_limit)
            sample_info["local_selection"] = (
                "highest and lowest predicted risk within the sample"
            )
            global_block.sample["local_selection"] = sample_info["local_selection"]
            local_blocks = local_explanations(
                family=resolved_family,
                explainer=explainer,
                explanation=self._select(explanation, local_indices),
                matrix=sample_matrix[local_indices],
                groups=groups,
                threshold=loaded.threshold,
                positive_label=loaded.positive_label,
                top_k=self.config.top_k,
                tolerance=self.config.tolerance,
                row_positions=sample_positions[local_indices],
            )
            for offset, item in enumerate(local_blocks):
                index = int(local_indices[offset])
                position = int(sample_positions[index])
                plot_path = root / "local" / f"row_{position:05d}.png"
                try:
                    paths[f"local_plot_{position}"] = plot_waterfall(
                        explanation.values[index],
                        float(explanation.base_values[index]),
                        sample_matrix[index],
                        mapped_columns(names, groups),
                        plot_path,
                        output_space=global_block.output_space,
                    )
                    item.plot_path = str(plot_path)
                except ExplainError as exc:
                    item.warnings.append(exc.message)
                    result.warnings.append(
                        f"row {position}: local waterfall plot unavailable: {exc.message}"
                    )
                json_path = root / "local" / f"row_{position:05d}.json"
                item.plot_path = item.plot_path or None
                paths[f"local_json_{position}"] = _atomic_write_json(
                    json_path, item.to_dict()
                )
            result.local_explanations = local_blocks

        duration = time.perf_counter() - started
        global_block.duration_seconds = float(duration)
        paths["global_json"] = _atomic_write_json(root / "global.json", global_block.to_dict())
        manifest = self._manifest(
            run_id=run_id,
            loaded=loaded,
            explainer=explainer,
            sample_info=sample_info,
            background_positions=background_positions,
            local_blocks=local_blocks,
            paths=paths,
            generated_at=generated_at,
            duration=duration,
            warnings=result.warnings,
        )
        result.manifest = manifest
        result.paths = paths
        paths["run_manifest"] = _atomic_write_json(root / "run_manifest.json", manifest)
        return result

    def explain_records(
        self,
        frame: Any,
        *,
        family: str | None = None,
        benchmark_run_id: str | None = None,
        model_run_id: str | None = None,
        explain_run_id: str | None = None,
    ) -> ExplainResult:
        """Explain caller-provided raw customer rows (local only, bounded)."""
        import pandas as pd

        if explain_run_id is not None:
            validate_explain_id(explain_run_id)
        if not isinstance(frame, pd.DataFrame):
            raise ExplainError(
                CODE_INVALID_CONFIG,
                "explain_records expects a pandas DataFrame",
                "Load a CSV with pandas and pass the frame.",
            )
        if len(frame) == 0:
            raise ExplainError(
                CODE_INVALID_CONFIG,
                "the input frame has no rows",
                "Provide at least one customer row.",
            )
        if len(frame) > int(self.config.max_local_batch):
            raise ExplainError(
                CODE_INVALID_CONFIG,
                f"{len(frame)} rows exceed the local batch limit "
                f"({self.config.max_local_batch})",
                "Split the input or raise --max-local-batch deliberately.",
            )
        loaded = self.load_model(
            family,
            benchmark_run_id=benchmark_run_id,
            model_run_id=model_run_id,
        )
        dataset = loaded.dataset
        names = list(dataset.feature_names)
        try:
            matrix = np.asarray(dataset.transform_new(frame), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001 - preprocessing carries the message
            message = getattr(exc, "message", str(exc))
            raise ExplainError(
                CODE_INVALID_CONFIG,
                f"raw records could not be transformed: {message}",
                "Provide every column required by the Phase 5 schema.",
            ) from exc
        groups = build_feature_groups(names, dataset.schema.get("output_features") or [])
        positions = deterministic_sample(
            len(dataset.train.X),
            self.config.background_limit(loaded.family),
            self.config.seed,
        )
        background = np.asarray(dataset.train.X, dtype=np.float64)[positions]
        explainer = build_explainer(
            loaded.family,
            loaded.model,
            loaded.predict_proba,
            background=background,
            feature_names=names,
            config=self.config,
        )
        explanation = chunked_explain(explainer, matrix, config=self.config)
        generated_at = self._now()
        run_id = explain_run_id or self._default_run_id(
            dataset.dataset_id, loaded.family, generated_at, loaded
        )
        validate_explain_id(run_id)
        root = self.explanation_dir / run_id
        blocks = local_explanations(
            family=loaded.family,
            explainer=explainer,
            explanation=explanation,
            matrix=matrix,
            groups=groups,
            threshold=loaded.threshold,
            positive_label=loaded.positive_label,
            top_k=self.config.top_k,
            tolerance=self.config.tolerance,
            row_positions=list(range(len(matrix))),
        )
        paths: dict[str, str] = {}
        for index, item in enumerate(blocks):
            paths[f"local_json_{index}"] = _atomic_write_json(
                root / "local" / f"custom_{index:04d}.json", item.to_dict()
            )
            try:
                item.plot_path = plot_waterfall(
                    explanation.values[index],
                    float(explanation.base_values[index]),
                    matrix[index],
                    mapped_columns(names, groups),
                    root / "local" / f"custom_{index:04d}.png",
                    output_space=explanation.output_space,
                )
                paths[f"local_plot_{index}"] = item.plot_path
            except ExplainError as exc:
                item.warnings.append(exc.message)
        manifest = {
            "schema_version": EXPLAIN_SCHEMA_VERSION,
            "phase": 8,
            "run_id": run_id,
            "generated_at": generated_at,
            "mode": "custom_records",
            "family": loaded.family,
            "dataset_id": dataset.dataset_id,
            "dataset_sha256": dataset.split_spec.dataset_sha256,
            "split_fingerprint": dataset.split_spec.fingerprint,
            "preprocessing_run_id": dataset.run_id,
            "threshold": float(loaded.threshold),
            "positive_label": loaded.positive_label,
            "output_space": explanation.output_space,
            "explainer": explainer.metadata(),
            "background": {
                "split": "train",
                "rows": int(len(positions)),
                "seed": int(self.config.seed),
                "positions": [int(value) for value in positions],
            },
            "model": loaded.provenance,
            "rows_explained": int(len(matrix)),
            "privacy": (
                "only transformed feature values and model contributions are stored; "
                "raw records and identifiers are not"
            ),
            "config": self.config.to_dict(),
            "environment": self._environment(),
            "paths": paths,
            "warnings": [warning for item in blocks for warning in item.warnings],
        }
        paths["run_manifest"] = _atomic_write_json(root / "run_manifest.json", manifest)
        return ExplainResult(
            run_id=run_id,
            family=loaded.family,
            local_explanations=blocks,
            paths=paths,
            manifest=manifest,
            warnings=list(manifest["warnings"]),
        )

    def explain_matrix(
        self,
        matrix: Any,
        *,
        family: str | None = None,
        benchmark_run_id: str | None = None,
        model_run_id: str | None = None,
        explain_run_id: str | None = None,
        row_positions: Sequence[int] | None = None,
        persist: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> ExplainResult:
        """Explain an already-transformed matrix through the Phase 8 engine.

        Phase 9 uses this method to attach drivers to scored rows without
        duplicating explainer construction or feature mapping. The caller is
        responsible for passing rows from a permitted split; Phase 9 passes the
        validation population only. Per-row JSON is persisted by default so a
        dashboard never needs a SHAP runtime.
        """
        loaded = self.load_model(
            family,
            benchmark_run_id=benchmark_run_id,
            model_run_id=model_run_id,
        )
        dataset = loaded.dataset
        names = list(dataset.feature_names)
        try:
            values = np.asarray(matrix, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ExplainError(
                CODE_INVALID_CONFIG,
                f"explanation matrix is not numeric: {type(exc).__name__}",
                "Pass the Phase 5 transformed matrix.",
            ) from exc
        if values.ndim != 2 or len(values) == 0:
            raise ExplainError(
                CODE_INVALID_CONFIG,
                f"explanation matrix must be non-empty and 2-D, got {values.shape}",
                "Pass at least one transformed row.",
            )
        if values.shape[1] != len(names):
            raise ExplainError(
                CODE_NO_MODEL,
                f"matrix has {values.shape[1]} columns; model expects {len(names)}",
                "Use the exact Phase 5 transform for this model.",
            )
        if not np.isfinite(values).all():
            raise ExplainError(
                CODE_INVALID_CONFIG,
                "explanation matrix contains non-finite values",
                "Clean the transformed matrix before explaining.",
            )
        positions = (
            [int(value) for value in row_positions]
            if row_positions is not None
            else list(range(len(values)))
        )
        if len(positions) != len(values):
            raise ExplainError(
                CODE_INVALID_CONFIG,
                f"{len(positions)} row positions were supplied for {len(values)} rows",
                "Supply exactly one position per row.",
            )
        groups = build_feature_groups(
            names, dataset.schema.get("output_features") or []
        )
        background_positions = deterministic_sample(
            len(dataset.train.X),
            self.config.background_limit(loaded.family),
            self.config.seed,
        )
        background = np.asarray(dataset.train.X, dtype=np.float64)[background_positions]
        explainer = build_explainer(
            loaded.family,
            loaded.model,
            loaded.predict_proba,
            background=background,
            feature_names=names,
            config=self.config,
        )
        explanation = chunked_explain(
            explainer, values, config=self.config, progress=progress
        )
        generated_at = self._now()
        if explain_run_id is None:
            digest = hashlib.sha256(
                np.ascontiguousarray(values, dtype=np.float64).tobytes()
            ).hexdigest()[:8]
            run_id = (
                self._default_run_id(
                    dataset.dataset_id, loaded.family, generated_at, loaded
                )
                + f"-{digest}"
            )
        else:
            run_id = explain_run_id
        validate_explain_id(run_id)
        root = self.explanation_dir / run_id
        blocks = local_explanations(
            family=loaded.family,
            explainer=explainer,
            explanation=explanation,
            matrix=values,
            groups=groups,
            threshold=loaded.threshold,
            positive_label=loaded.positive_label,
            top_k=self.config.top_k,
            tolerance=self.config.tolerance,
            row_positions=positions,
        )
        paths: dict[str, str] = {}
        if persist:
            for index, item in enumerate(blocks):
                paths[f"local_json_{index}"] = _atomic_write_json(
                    root / "local" / f"row_{int(positions[index]):05d}.json",
                    item.to_dict(),
                )
        manifest = {
            "schema_version": EXPLAIN_SCHEMA_VERSION,
            "phase": 8,
            "mode": "matrix",
            "run_id": run_id,
            "generated_at": generated_at,
            "family": loaded.family,
            "dataset_id": dataset.dataset_id,
            "dataset_sha256": dataset.split_spec.dataset_sha256,
            "split_fingerprint": dataset.split_spec.fingerprint,
            "preprocessing_run_id": dataset.run_id,
            "threshold": float(loaded.threshold),
            "positive_label": loaded.positive_label,
            "output_space": explanation.output_space,
            "explainer": explainer.metadata(),
            "background": {
                "split": "train",
                "rows": int(len(background_positions)),
                "seed": int(self.config.seed),
                "positions": [int(value) for value in background_positions],
            },
            "model": dict(loaded.provenance),
            "rows_explained": int(len(values)),
            "row_positions": positions,
            "privacy": (
                "only transformed feature values and model contributions are stored; "
                "raw records and identifiers are not"
            ),
            "config": self.config.to_dict(),
            "environment": self._environment(),
            "paths": paths,
            "limitations": [
                "SHAP values describe model behaviour, not causal effects.",
                "One-hot contributions are summed back to their source feature.",
            ],
            "warnings": [warning for item in blocks for warning in item.warnings],
        }
        if persist:
            paths["run_manifest"] = _atomic_write_json(
                root / "run_manifest.json", manifest
            )
        return ExplainResult(
            run_id=run_id,
            family=loaded.family,
            local_explanations=blocks,
            paths=paths,
            manifest=manifest,
            warnings=list(manifest["warnings"]),
        )

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _threshold(value: Any, default: float) -> float:
        """Return a validated probability threshold, or a documented default."""
        if value is None:
            return float(default)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ExplainError(
                CODE_NO_MODEL,
                f"the operating threshold {value!r} is not numeric",
                "Rerun evaluation/training so the threshold is persisted.",
            ) from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ExplainError(
                CODE_NO_MODEL,
                f"the operating threshold {number!r} is outside [0, 1]",
                "Rerun evaluation/training so the threshold is persisted.",
            )
        return number

    @staticmethod
    def _verify_preprocessing(
        metadata: Mapping[str, Any], dataset: Any, family: str
    ) -> None:
        """Reject a model/preprocessing pair whose recorded provenance differs."""
        block = dict(metadata.get("preprocessing") or {})
        split = dataset.split_spec
        checks = (
            ("dataset_id", block.get("dataset_id"), dataset.dataset_id),
            ("dataset_sha256", block.get("dataset_sha256"), split.dataset_sha256),
            ("split_fingerprint", block.get("split_fingerprint"), split.fingerprint),
            (
                "preprocessor_hash",
                block.get("preprocessor_hash"),
                (dataset.schema or {}).get("preprocessor_hash"),
            ),
        )
        mismatches: list[str] = []
        for label, expected, actual in checks:
            if expected not in (None, "") and actual not in (None, ""):
                if str(expected) != str(actual):
                    mismatches.append(label)
        expected_names = block.get("feature_names")
        actual_names = list(getattr(dataset, "feature_names", []) or [])
        if expected_names and actual_names and list(expected_names) != actual_names:
            mismatches.append("feature_names")
        if mismatches:
            raise ExplainError(
                CODE_NO_MODEL,
                f"{family} artifact does not match preprocessing run "
                f"{getattr(dataset, 'run_id', '<unknown>')!r}: "
                + ", ".join(sorted(set(mismatches))),
                "Regenerate the model/evaluation against this preprocessing run; "
                "do not explain a mismatched artifact pair.",
            )

    @staticmethod
    def _select(explanation: Any, indices: Sequence[int]) -> Any:
        from churnwise.explain.contracts import RawExplanation

        positions = np.asarray(list(indices), dtype=np.int64)
        return RawExplanation(
            values=explanation.values[positions],
            base_values=explanation.base_values[positions],
            predictions=explanation.predictions[positions],
            predictions_raw=(
                explanation.predictions_raw[positions]
                if explanation.predictions_raw is not None
                else None
            ),
            additivity_error=explanation.additivity_error,
            output_space=explanation.output_space,
            explainer_name=explanation.explainer_name,
            notes=list(explanation.notes),
        )

    @staticmethod
    def _local_indices(predictions: Sequence[float], limit: int) -> np.ndarray:
        """Choose the highest and lowest risk rows so both directions are shown."""
        values = np.asarray(predictions, dtype=np.float64)
        if int(limit) >= len(values):
            return np.arange(len(values), dtype=np.int64)
        order = np.argsort(values, kind="stable")
        high = (int(limit) + 1) // 2
        low = int(limit) - high
        picks = list(order[-high:]) + list(order[:low])
        return np.asarray(sorted({int(value) for value in picks}), dtype=np.int64)

    def _default_run_id(
        self,
        dataset_id: str,
        family: str,
        generated_at: str,
        loaded: LoadedExplanationModel,
    ) -> str:
        stamp = re.sub(r"[^0-9A-Za-z]+", "", generated_at)[:15]
        digest = hashlib.sha256(
            (
                json.dumps(
                    {
                        "dataset": dataset_id,
                        "family": family,
                        "config": self.config.fingerprint(),
                        "model": loaded.provenance.get("model_sha256"),
                    },
                    sort_keys=True,
                )
            ).encode("utf-8")
        ).hexdigest()[:8]
        return f"{stamp}-{_safe_name(dataset_id)}-{family}-xai-{digest}"

    def _environment(self) -> dict[str, Any]:
        versions: dict[str, Any] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        }
        try:
            import shap

            versions["shap"] = shap.__version__
        except Exception:  # pragma: no cover - environment metadata only
            versions["shap"] = None
        try:
            import sklearn

            versions["scikit_learn"] = sklearn.__version__
        except Exception:  # pragma: no cover - environment metadata only
            versions["scikit_learn"] = None
        try:
            import tensorflow

            versions["tensorflow"] = tensorflow.__version__
        except Exception:  # pragma: no cover - optional for baseline-only use
            versions["tensorflow"] = None
        return versions

    def _manifest(
        self,
        *,
        run_id: str,
        loaded: LoadedExplanationModel,
        explainer: ModelExplainer,
        sample_info: Mapping[str, Any],
        background_positions: Sequence[int],
        local_blocks: Sequence[LocalExplanation],
        paths: Mapping[str, str],
        generated_at: str,
        duration: float,
        warnings: Sequence[str],
    ) -> dict[str, Any]:
        limitations = [
            "SHAP values explain model output, not causal effects.",
            "Sampling is seeded uniform without replacement; another sample "
            "changes the estimated global ranking slightly.",
            "The test partition is not used for explanations.",
        ]
        if loaded.family == "logistic_regression":
            limitations.insert(
                2,
                "Logistic Regression values are log-odds, not probability changes.",
            )
        return {
            "schema_version": EXPLAIN_SCHEMA_VERSION,
            "phase": 8,
            "run_id": run_id,
            "generated_at": generated_at,
            "family": loaded.family,
            "dataset_id": loaded.dataset.dataset_id,
            "dataset_sha256": loaded.dataset.split_spec.dataset_sha256,
            "split_fingerprint": loaded.dataset.split_spec.fingerprint,
            "preprocessing_run_id": loaded.dataset.run_id,
            "model": dict(loaded.provenance),
            "threshold": float(loaded.threshold),
            "positive_label": loaded.positive_label,
            "output_space": explainer.output_space,
            "explainer": explainer.metadata(),
            "sampling": dict(sample_info),
            "background": {
                "split": "train",
                "rows": int(len(background_positions)),
                "seed": int(self.config.seed),
                "positions": [int(value) for value in background_positions],
            },
            "local_examples": [
                {
                    "row_position": item.row_position,
                    "probability": float(item.probability),
                    "predicted_class": int(item.predicted_class),
                }
                for item in local_blocks
            ],
            "duration_seconds": float(duration),
            "config": self.config.to_dict(),
            "environment": self._environment(),
            "paths": dict(paths),
            "warnings": list(warnings),
            "privacy": (
                "global explanations expose no customer records; local files store "
                "only model contributions and transformed feature values, no "
                "identifiers or raw source rows"
            ),
            "limitations": limitations,
        }


__all__ = ["ExplainCoordinator", "LoadedExplanationModel", "validate_explain_id"]
