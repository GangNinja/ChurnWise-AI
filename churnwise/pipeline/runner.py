"""Phase 12 end-to-end orchestration.

One command runs the supported workflow:

    discovery -> acquisition -> validation -> preprocessing -> MLP training ->
    benchmark evaluation -> SHAP explanations -> risk scoring -> simulation

The runner composes the existing phase services instead of reimplementing their
rules. Every stage, its duration and its artifacts are recorded in
``artifacts/pipelines/<run-id>/run.json`` so a failed run can be diagnosed and
resumed without pretending that skipped or incomplete work succeeded.

Local datasets (``dataset_path=...``) are copied into the managed acquisition
store, given a provenance manifest and then passed through the same validation
gate as downloaded data. The source fixture is never modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from churnwise import __version__
from churnwise.acquisition.contracts import (
    AcquisitionArtifact,
    AcquisitionAttempt,
    AcquisitionResult,
    AcquisitionStatus,
    AttemptStatus,
    DatasetManifest,
    ManifestAcquisition,
    ManifestFile,
    ManifestIntegrity,
    ManifestSource,
)
from churnwise.acquisition.integrity import sha256_file
from churnwise.acquisition.inputs import load_candidates
from churnwise.acquisition.provenance import candidate_fingerprint
from churnwise.acquisition.store import DatasetStore, safe_component
from churnwise.config import Settings
from churnwise.contracts import DatasetCandidate, LicenseStatus, SearchQuery, utc_now_iso
from churnwise.discovery import build_adapters, discover
from churnwise.discovery.reporting import write_reports as write_discovery_reports
from churnwise.evaluation.benchmark import BenchmarkCoordinator
from churnwise.evaluation.contracts import EvaluationConfig
from churnwise.explain.contracts import ExplainConfig
from churnwise.explain.coordinator import ExplainCoordinator
from churnwise.intelligence.contracts import RiskConfig, RiskThresholds
from churnwise.intelligence.service import RiskService
from churnwise.models.contracts import MLPConfig
from churnwise.models.trainer import MLPTrainer
from churnwise.pipeline.errors import (
    CODE_INCOMPATIBLE_RESUME,
    CODE_INVALID_CONFIG,
    CODE_STAGE_FAILED,
    CODE_UPSTREAM_FAILURE,
    CODE_VERIFICATION_FAILED,
    PipelineError,
)
from churnwise.preprocessing.contracts import PreprocessConfig
from churnwise.preprocessing.coordinator import PreprocessingCoordinator
from churnwise.simulation.contracts import CampaignScenario
from churnwise.simulation.service import SimulationService
from churnwise.validation import ValidationCoordinator

logger = logging.getLogger("churnwise.pipeline")

PIPELINE_SCHEMA_VERSION = "1.0"
STAGE_NAMES = (
    "source",
    "preprocess",
    "train",
    "evaluate",
    "explain",
    "score",
    "simulate",
)
LOCAL_DATASET_FORMATS = frozenset({"csv", "tsv", "arff", "parquet", "json", "xlsx"})
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,110}$")


@dataclass
class PipelineConfig:
    """Inputs and explicit overrides for one reproducible pipeline run."""

    dataset_path: Path | None = None
    query: SearchQuery | None = None
    offline: bool = False
    fast: bool = False
    refresh_discovery: bool = True
    max_candidates: int = 3
    seed: int | None = None
    run_id: str | None = None
    resume: bool = False
    preprocess: PreprocessConfig | None = None
    mlp: MLPConfig | None = None
    evaluation: EvaluationConfig | None = None
    explain: ExplainConfig | None = None
    risk: RiskConfig | None = None
    scenario: CampaignScenario | None = None


@dataclass
class PipelineResult:
    """The durable identity and stage summary of a completed pipeline run."""

    run_id: str
    status: str
    run_dir: Path
    manifest_path: Path
    config_digest: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    stages: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "run_dir": self.run_dir.as_posix(),
            "manifest_path": self.manifest_path.as_posix(),
            "config_digest": self.config_digest,
            "artifacts": dict(self.artifacts),
            "stages": [dict(stage) for stage in self.stages],
            "warnings": list(self.warnings),
        }

    @property
    def durations_ms(self) -> dict[str, int]:
        return {
            str(stage.get("name")): int(stage.get("duration_ms") or 0)
            for stage in self.stages
        }


@dataclass
class StageOutcome:
    """What one stage produced, without leaking service objects into the manifest."""

    value: Any = None
    outputs: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def build_pipeline_config(
    settings: Settings,
    *,
    dataset_path: Path | str | None = None,
    fast: bool = False,
    seed: int | None = None,
    query: SearchQuery | None = None,
    offline: bool | None = None,
    refresh_discovery: bool = True,
    max_candidates: int = 3,
    run_id: str | None = None,
    resume: bool = False,
    scenario: CampaignScenario | None = None,
) -> PipelineConfig:
    """Build the documented defaults for the fixture and reference workflows.

    ``fast`` is a demonstration profile, not a benchmark result: it reduces MLP
    epochs, cross-validation folds, explanation sample sizes and driver rows so
    the complete workflow can run on a student laptop. Its artifacts are labelled
    accordingly and must not be reported as production model performance.
    """
    resolved_seed = settings.seed if seed is None else int(seed)
    resolved_offline = settings.offline if offline is None else bool(offline)
    base_query = query or SearchQuery(
        allow_unverified_license=settings.allow_unverified_license
    )
    if settings.allow_unverified_license and not base_query.allow_unverified_license:
        base_query = replace(base_query, allow_unverified_license=True)

    if fast:
        mlp = MLPConfig(
            seed=resolved_seed,
            hidden_units=(32, 16),
            dropout=(0.20, 0.10),
            epochs=24,
            early_stopping_patience=5,
            reduce_lr_patience=3,
        )
        evaluation = EvaluationConfig(seed=resolved_seed, cv_folds=3)
        explain = ExplainConfig(
            seed=resolved_seed,
            sample_size=40,
            background_size=30,
            mlp_sample_size=20,
            mlp_background_size=20,
            local_examples=3,
            chunk_size=50,
            permutation_max_evals=100,
        )
        risk = RiskConfig(
            thresholds=RiskThresholds(tuple(settings.risk_tiers)),
            batch_size=100,
            top_k=5,
            max_driver_rows=10,
            population_explain="top",
        )
    else:
        mlp = MLPConfig(seed=resolved_seed)
        evaluation = EvaluationConfig(seed=resolved_seed)
        explain = ExplainConfig(seed=resolved_seed)
        risk = RiskConfig(
            thresholds=RiskThresholds(tuple(settings.risk_tiers)),
            population_explain="top",
            max_driver_rows=100,
        )

    return PipelineConfig(
        dataset_path=Path(dataset_path) if dataset_path is not None else None,
        query=base_query,
        offline=resolved_offline,
        fast=fast,
        refresh_discovery=refresh_discovery,
        max_candidates=max_candidates,
        seed=resolved_seed,
        run_id=run_id,
        resume=resume,
        preprocess=PreprocessConfig(seed=resolved_seed),
        mlp=mlp,
        evaluation=evaluation,
        explain=explain,
        risk=risk,
        scenario=scenario
        or CampaignScenario(
            name="Baseline retention campaign",
            description=(
                "Default Phase 12 demonstration scenario. Effectiveness, cost and "
                "value assumptions are hypothetical, not measured uplift."
            ),
        ),
    )


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be a JSON object")
    return payload


class PipelineRunner:
    """Compose every phase service into one auditable, resumable workflow."""

    def __init__(
        self,
        settings: Settings,
        config: PipelineConfig | None = None,
        *,
        adapters: Any = None,
        now: Callable[[], str] = utc_now_iso,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.config = self._normalise(config or PipelineConfig())
        # Phase services read the seed from Settings, while the pipeline config
        # carries the explicit seed. Keep them identical after normalization so a
        # caller-supplied seed cannot leak a different value into provenance.
        if self.config.seed != self.settings.seed:
            self.settings = replace(self.settings, seed=self.config.seed)
        self.adapters = adapters
        self._now = now
        self._progress = progress
        self._validate()
        self.config_digest = self._compute_config_digest()
        self.run_id = self.config.run_id or self._new_run_id()
        self.run_dir = Path(settings.artifact_dir) / "pipelines" / self.run_id
        self.manifest_path = self.run_dir / "run.json"
        self._records: dict[str, dict[str, Any]] = {}
        self._manifest: dict[str, Any] = {}
        self._warnings: list[str] = []

    # -- configuration ------------------------------------------------------
    def _normalise(self, config: PipelineConfig) -> PipelineConfig:
        seed = self.settings.seed if config.seed is None else int(config.seed)
        if config.dataset_path is not None:
            config.dataset_path = Path(config.dataset_path)
        if config.query is None:
            config.query = SearchQuery(
                allow_unverified_license=self.settings.allow_unverified_license
            )
        if config.dataset_path is not None:
            config.offline = True
        else:
            config.offline = bool(config.offline or self.settings.offline)
        if config.preprocess is None:
            config.preprocess = PreprocessConfig(seed=seed)
        if config.mlp is None:
            config.mlp = (
                build_pipeline_config(self.settings, fast=True, seed=seed).mlp
                if config.fast
                else MLPConfig(seed=seed)
            )
        if config.evaluation is None:
            config.evaluation = (
                build_pipeline_config(self.settings, fast=True, seed=seed).evaluation
                if config.fast
                else EvaluationConfig(seed=seed)
            )
        if config.explain is None:
            config.explain = (
                build_pipeline_config(self.settings, fast=True, seed=seed).explain
                if config.fast
                else ExplainConfig(seed=seed)
            )
        if config.risk is None:
            defaults = build_pipeline_config(self.settings, fast=config.fast, seed=seed)
            assert defaults.risk is not None
            config.risk = defaults.risk
        if config.scenario is None:
            config.scenario = CampaignScenario(
                name="Baseline retention campaign",
                description=(
                    "Default Phase 12 demonstration scenario. Effectiveness, cost "
                    "and value assumptions are hypothetical, not measured uplift."
                ),
            )
        return config

    def _validate(self) -> None:
        problems = list(self.settings.validate())
        config = self.config
        if config.query is None:
            problems.append("a discovery query is required")
        else:
            problems.extend(config.query.validate())
        if config.dataset_path is not None:
            path = config.dataset_path
            if not path.is_file():
                problems.append(f"dataset_path does not exist or is not a file: {path}")
            elif path.suffix.lower().lstrip(".") not in LOCAL_DATASET_FORMATS:
                problems.append(
                    "dataset_path must use one of: "
                    + ", ".join(sorted(LOCAL_DATASET_FORMATS))
                )
        elif config.offline:
            problems.append(
                "network discovery cannot run in offline mode; pass an explicit "
                "dataset_path for the offline workflow or disable offline mode"
            )
        if not 1 <= int(config.max_candidates) <= 10:
            problems.append("max_candidates must be between 1 and 10")
        if config.preprocess is not None:
            problems.extend(config.preprocess.validate())
        if config.mlp is not None:
            problems.extend(config.mlp.validate())
        if config.evaluation is not None:
            problems.extend(config.evaluation.validate())
        if config.explain is not None:
            problems.extend(config.explain.validate())
        if config.risk is not None:
            problems.extend(config.risk.validate())
        if config.scenario is not None:
            problems.extend(config.scenario.validate())
        if config.run_id is not None and not _RUN_ID.fullmatch(config.run_id):
            problems.append(
                "run_id must be lowercase letters, digits, dots, dashes or "
                "underscores (3-111 characters, starting with a letter or digit)"
            )
        if problems:
            raise PipelineError(
                CODE_INVALID_CONFIG,
                "invalid pipeline configuration: " + "; ".join(problems),
                "Correct the reported pipeline settings and retry.",
            )
        # Every derived artifact identifier must fit the strict phase validators.
        if config.run_id is not None:
            for suffix in ("prep", "mlp", "bench", "xai", "risk"):
                identifier = f"{config.run_id}-{suffix}"
                if len(identifier) > 120:
                    raise PipelineError(
                        CODE_INVALID_CONFIG,
                        f"run_id is too long once '-{suffix}' is appended",
                        "Use a run_id of at most 110 characters.",
                    )

    def _dataset_identity(self) -> dict[str, Any] | None:
        path = self.config.dataset_path
        if path is None:
            return None
        try:
            digest = sha256_file(path)
            size = path.stat().st_size
        except OSError as exc:
            raise PipelineError(
                CODE_INVALID_CONFIG,
                f"dataset_path could not be read: {exc}",
                "Check that the committed dataset is present and readable.",
            ) from exc
        return {
            "name": path.name,
            "format": path.suffix.lower().lstrip("."),
            "bytes": int(size),
            "sha256": digest,
        }

    def _config_payload(self) -> dict[str, Any]:
        config = self.config
        return {
            "seed": config.seed,
            "fast": bool(config.fast),
            "offline": bool(config.offline),
            "refresh_discovery": bool(config.refresh_discovery),
            "max_candidates": int(config.max_candidates),
            "dataset": self._dataset_identity(),
            "query": config.query.to_dict() if config.query else None,
            "preprocess": config.preprocess.to_dict() if config.preprocess else None,
            "mlp": config.mlp.to_dict() if config.mlp else None,
            "evaluation": config.evaluation.to_dict() if config.evaluation else None,
            "explain": config.explain.to_dict() if config.explain else None,
            "risk": config.risk.to_dict() if config.risk else None,
            "scenario": config.scenario.to_dict() if config.scenario else None,
        }

    def _compute_config_digest(self) -> str:
        return _sha256_text(_canonical(self._config_payload()))

    def _new_run_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dataset = self.config.dataset_path
        slug = (
            re.sub(r"[^a-z0-9]+", "-", dataset.stem.lower()).strip("-")[:40]
            if dataset is not None
            else "network"
        )
        candidate = f"{stamp}-{slug or 'pipeline'}-{self.config_digest[:8]}"
        if (Path(self.settings.artifact_dir) / "pipelines" / candidate).exists():
            candidate = f"{candidate}-{time.time_ns() % 10_000:04d}"
        return candidate

    def _stage_id(self, suffix: str) -> str:
        return f"{self.run_id}-{suffix}"

    # -- manifest lifecycle -------------------------------------------------
    def _blank_records(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "duration_ms": None,
                "outputs": {},
                "artifacts": {},
                "summary": {},
                "warnings": [],
                "error": None,
            }
            for name in STAGE_NAMES
        ]

    def _start_new(self) -> None:
        if self.run_dir.exists() or self.manifest_path.exists():
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                f"pipeline run {self.run_id!r} already exists",
                "Choose a new --run-id, or resume the existing run with --resume.",
            )
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._records = {record["name"]: record for record in self._blank_records()}
        self._manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "phase": 12,
            "run_id": self.run_id,
            "status": "running",
            "config_digest": self.config_digest,
            "code_version": __version__,
            "started_at": self._now(),
            "finished_at": None,
            "settings": {
                "seed": self.config.seed,
                "data_dir": self._display(self.settings.data_dir),
                "artifact_dir": self._display(self.settings.artifact_dir),
                "report_dir": self._display(self.settings.report_dir),
            },
            "config": self._config_payload(),
            "stages": [self._records[name] for name in STAGE_NAMES],
            "artifacts": {},
            "warnings": [],
            "error": None,
        }
        self._write_manifest()

    def _load_existing(self) -> None:
        if not self.manifest_path.is_file():
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                f"no saved run manifest exists at {self.manifest_path}",
                "Omit --resume to start a new run.",
            )
        try:
            manifest = _read_json(self.manifest_path)
        except (OSError, ValueError) as exc:
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                f"saved run manifest could not be read: {exc}",
                "Start a new run; a corrupt manifest is never silently overwritten.",
            ) from exc
        if not isinstance(manifest, dict):
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                "saved run manifest is not a JSON object",
                "Start a new run; a corrupt manifest is never silently overwritten.",
            )
        if manifest.get("schema_version") != PIPELINE_SCHEMA_VERSION:
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                "the saved run uses an incompatible pipeline schema",
                "Start a new run; a different schema is never resumed silently.",
                details={
                    "saved_schema_version": manifest.get("schema_version"),
                    "current_schema_version": PIPELINE_SCHEMA_VERSION,
                },
            )
        if manifest.get("phase") != 12 or manifest.get("run_id") != self.run_id:
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                "the saved run identity does not match the requested pipeline run",
                "Use the run id recorded inside the manifest, or start a new run.",
            )
        if manifest.get("status") not in {"running", "succeeded", "failed"}:
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                "the saved run has an unknown terminal status",
                "Start a new run; the manifest is not trusted.",
                details={"saved_status": manifest.get("status")},
            )
        recorded = manifest.get("config_digest")
        if recorded != self.config_digest:
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                "the saved run was created with a different configuration",
                "Use the original configuration to resume, or start a new run id.",
                details={
                    "saved_config_digest": recorded,
                    "current_config_digest": self.config_digest,
                },
            )
        stages = manifest.get("stages")
        if (
            not isinstance(stages, list)
            or any(not isinstance(record, dict) for record in stages)
            or [record.get("name") for record in stages] != list(STAGE_NAMES)
        ):
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                "the saved run manifest does not contain the Phase 12 stages",
                "Start a new run instead of resuming an incompatible manifest.",
            )
        required_outputs = {
            "source": {"dataset_id"},
            "preprocess": {"preprocessing_run_id"},
            "train": {"model_run_id"},
            "evaluate": {"benchmark_run_id", "selected_family"},
            "explain": {"explain_run_id"},
            "score": {"risk_run_id"},
            "simulate": {"scenario_id"},
        }
        for record in stages:
            name = str(record.get("name"))
            if record.get("status") not in {"pending", "running", "succeeded", "failed"}:
                raise PipelineError(
                    CODE_INCOMPATIBLE_RESUME,
                    f"saved stage {record.get('name')!r} has an invalid status",
                    "Start a new run; the manifest is not trusted.",
                )
            artifacts = record.get("artifacts", {})
            if not isinstance(artifacts, dict) or any(
                not isinstance(value, str) for value in artifacts.values()
            ):
                raise PipelineError(
                    CODE_INCOMPATIBLE_RESUME,
                    f"saved stage {record.get('name')!r} has an invalid artifact map",
                    "Start a new run; the manifest is not trusted.",
                )
            outputs = record.get("outputs", {})
            if not isinstance(outputs, dict):
                raise PipelineError(
                    CODE_INCOMPATIBLE_RESUME,
                    f"saved stage {record.get('name')!r} has invalid outputs",
                    "Start a new run; the manifest is not trusted.",
                )
            if record.get("status") == "succeeded":
                missing_outputs = required_outputs[name] - set(outputs)
                if missing_outputs or not artifacts:
                    raise PipelineError(
                        CODE_INCOMPATIBLE_RESUME,
                        f"saved stage {name!r} is incomplete",
                        "Start a new run; a succeeded stage must record its outputs and artifacts.",
                        details={
                            "missing_outputs": sorted(missing_outputs),
                            "artifact_count": len(artifacts),
                        },
                    )
        if not isinstance(manifest.get("artifacts", {}), dict):
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                "the saved run has an invalid artifact index",
                "Start a new run; the manifest is not trusted.",
            )
        if manifest.get("status") == "succeeded" and any(
            record.get("status") != "succeeded" for record in stages
        ):
            raise PipelineError(
                CODE_INCOMPATIBLE_RESUME,
                "the saved run claims success but has incomplete stages",
                "Start a new run; the manifest is internally inconsistent.",
            )
        self._manifest = manifest
        self._records = {str(record["name"]): record for record in stages}
        self._warnings = [str(item) for item in manifest.get("warnings") or []]

        # A stage is only reusable when every recorded artifact still exists.
        # The first broken stage invalidates itself and every later stage, which
        # prevents a resumed run from silently mixing stale and fresh outputs.
        invalidated = False
        for name in STAGE_NAMES:
            record = self._records[name]
            if record.get("status") in {"running", "failed"} and not invalidated:
                invalidated = True
            elif record.get("status") == "succeeded" and not invalidated:
                missing = [
                    path
                    for path in (record.get("artifacts") or {}).values()
                    if not self._resolve(path).exists()
                ]
                if missing:
                    invalidated = True
                    record["warnings"] = [
                        *(record.get("warnings") or []),
                        "resume invalidated this stage because artifacts are missing: "
                        + ", ".join(missing),
                    ]
            if invalidated:
                record["status"] = "pending"
                record["started_at"] = None
                record["finished_at"] = None
                record["duration_ms"] = None
                record["outputs"] = {}
                record["artifacts"] = {}
                record["summary"] = {}
                record["error"] = None
        if invalidated:
            self._manifest["status"] = "running"
            self._manifest["finished_at"] = None
            self._manifest["error"] = None
            self._write_manifest()

    def _write_manifest(self) -> None:
        self._manifest["stages"] = [self._records[name] for name in STAGE_NAMES]
        self._manifest["artifacts"] = {
            key: value
            for key, value in (self._manifest.get("artifacts") or {}).items()
        }
        self._manifest["warnings"] = list(self._warnings)
        _atomic_json(self.manifest_path, self._manifest)

    # -- general helpers ----------------------------------------------------
    def _display(self, path: Path | str | None) -> str:
        if path is None:
            return ""
        target = Path(path)
        try:
            return target.resolve().relative_to(
                Path(self.settings.project_root).resolve()
            ).as_posix()
        except (ValueError, OSError):
            return str(target)

    def _resolve(self, path: str | Path) -> Path:
        target = Path(path)
        if target.is_absolute():
            return target
        return Path(self.settings.project_root) / target

    def _artifact_map(self, paths: dict[str, Any] | None) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in (paths or {}).items():
            if value is None:
                continue
            text = str(value)
            if not text:
                continue
            result[str(key)] = self._display(text)
        return result

    def _notify(self, message: str) -> None:
        if self._progress is not None:
            self._progress(message)

    def _run_stage(self, name: str, action: Callable[[], StageOutcome]) -> StageOutcome:
        record = self._records[name]
        started = time.perf_counter()
        record["status"] = "running"
        record["started_at"] = self._now()
        record["finished_at"] = None
        record["duration_ms"] = None
        record["error"] = None
        self._manifest["status"] = "running"
        self._write_manifest()
        self._notify(f"[{name}] running")
        try:
            outcome = action()
        except PipelineError as exc:
            exc.stage = exc.stage or name
            record["status"] = "failed"
            record["finished_at"] = self._now()
            record["duration_ms"] = int((time.perf_counter() - started) * 1000)
            record["error"] = exc.to_dict()
            if exc.artifacts:
                record["artifacts"] = self._artifact_map(exc.artifacts)
            self._notify(f"[{name}] failed: {exc.code}")
            self._write_manifest()
            raise
        except Exception as exc:  # noqa: BLE001 - converted to a structured stage error
            wrapped = PipelineError(
                CODE_STAGE_FAILED,
                f"{name} stage failed: {type(exc).__name__}: {exc}",
                "Inspect the recorded exception and the upstream phase artifacts.",
                stage=name,
                details={"exception_type": type(exc).__name__},
            )
            record["status"] = "failed"
            record["finished_at"] = self._now()
            record["duration_ms"] = int((time.perf_counter() - started) * 1000)
            record["error"] = wrapped.to_dict()
            self._notify(f"[{name}] failed: {wrapped.code}")
            self._write_manifest()
            raise wrapped from exc
        record["status"] = "succeeded"
        record["finished_at"] = self._now()
        record["duration_ms"] = int((time.perf_counter() - started) * 1000)
        record["outputs"] = dict(outcome.outputs)
        record["artifacts"] = self._artifact_map(outcome.artifacts)
        record["summary"] = dict(outcome.summary)
        record["warnings"] = list(outcome.warnings)
        for warning in outcome.warnings:
            if warning not in self._warnings:
                self._warnings.append(warning)
        self._manifest["artifacts"].update(outcome.outputs)
        for key, path in record["artifacts"].items():
            self._manifest["artifacts"][f"{name}.{key}"] = path
        self._notify(
            f"[{name}] succeeded in {record['duration_ms']} ms"
        )
        self._write_manifest()
        return outcome

    def _skipped(self, name: str) -> bool:
        return (
            self._records[name].get("status") == "succeeded"
            and self._records[name].get("error") is None
        )

    def _result(self) -> PipelineResult:
        return PipelineResult(
            run_id=self.run_id,
            status=str(self._manifest.get("status") or "running"),
            run_dir=self.run_dir,
            manifest_path=self.manifest_path,
            config_digest=self.config_digest,
            artifacts=dict(self._manifest.get("artifacts") or {}),
            stages=[dict(self._records[name]) for name in STAGE_NAMES],
            warnings=list(self._warnings),
        )

    # -- stages -------------------------------------------------------------
    def _source(self) -> StageOutcome:
        if self.config.dataset_path is not None:
            return self._source_local(self.config.dataset_path)
        return self._source_network()

    def _source_local(self, path: Path) -> StageOutcome:
        assert self.config.query is not None
        started = time.perf_counter()
        fmt = path.suffix.lower().lstrip(".")
        digest = sha256_file(path)
        size = path.stat().st_size
        slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:40] or "fixture"
        candidate = DatasetCandidate(
            candidate_id=f"fixture-{slug}-{digest[:8]}",
            source_id="fixture",
            source_name="Committed fixture",
            source_dataset_id=f"fixture-{slug}-{digest[:8]}",
            title=f"Committed fixture: {path.name}",
            landing_url=f"https://fixtures.churnwise.invalid/{quote(path.name)}",
            resource_url=f"https://fixtures.churnwise.invalid/{quote(path.name)}",
            retrieved_at=self._now(),
            format=fmt,
            description=(
                "Local dataset copied into the managed acquisition store so the "
                "normal validation, preprocessing and modeling workflow runs "
                "without a network request."
            ),
            publisher="ChurnWise project",
            license_name="Synthetic project fixture (no third-party data)",
            license_status=LicenseStatus.PROJECT_OWNED,
            size_bytes=size,
            access="open",
            limitations=[
                "Synthetic or committed project fixture; its metrics are workflow "
                "evidence, not production performance."
            ],
            evidence={"origin": "committed_file", "filename": path.name, "sha256": digest},
        )
        store = DatasetStore(self.settings.data_dir)
        store.ensure()
        tmp_dir = store.run_tmp(f"{self.run_id}-source")
        safe_name = safe_component(path.name, fallback=f"dataset.{fmt}")
        temporary = tmp_dir / safe_name
        try:
            shutil.copyfile(path, temporary)
            copied_digest = sha256_file(temporary)
            if copied_digest != digest or temporary.stat().st_size != size:
                raise PipelineError(
                    CODE_STAGE_FAILED,
                    "the local dataset changed while it was being copied",
                    "Do not modify the fixture during a pipeline run; run again.",
                )
            cache_path = store.promote(temporary, digest, size)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        raw_path = store.raw_path(candidate, safe_name)
        store.publish_raw(cache_path, raw_path)
        manifest = DatasetManifest(
            dataset_id=candidate.candidate_id,
            title=candidate.title,
            source=ManifestSource(
                provider="committed_fixture",
                source_id="fixture",
                dataset_id=candidate.source_dataset_id,
                landing_url=candidate.landing_url,
                download_url=candidate.resource_url or candidate.landing_url,
                license_name=candidate.license_name,
                license_url=None,
                license_status=candidate.license_status.value,
                publisher=candidate.publisher,
            ),
            retrieved_at=candidate.retrieved_at,
            file=ManifestFile(
                filename=raw_path.name,
                bytes=size,
                sha256=digest,
                content_type="text/csv" if fmt == "csv" else None,
                format=fmt,
                cache_path=store.relative(cache_path),
                raw_path=store.relative(raw_path),
            ),
            integrity=ManifestIntegrity(
                verified=True,
                method="sha256_recomputed",
                expected_size=size,
                checksum_status="unavailable",
            ),
            acquisition=ManifestAcquisition(
                method="offline_fixture",
                attempt=1,
                duration_ms=int((time.perf_counter() - started) * 1000),
                status="success",
            ),
            discovery={
                "candidate": candidate.to_dict(),
                "candidate_fingerprint": candidate_fingerprint(candidate),
                "origin": "offline_fixture",
            },
            run_id=self.run_id,
            dataset_sha256=digest,
            seed=self.settings.seed,
            code_version=__version__,
            notes=["Source file was copied, never modified, before validation."],
        )
        manifest_path = store.save_manifest(manifest)
        artifact = AcquisitionArtifact(
            dataset_id=candidate.candidate_id,
            filename=raw_path.name,
            format=fmt,
            bytes=size,
            sha256=digest,
            content_type="text/csv" if fmt == "csv" else None,
            cache_path=manifest.file.cache_path,
            raw_path=manifest.file.raw_path,
        )
        acquisition = AcquisitionResult(
            run_id=self.run_id,
            generated_at=self._now(),
            status=AcquisitionStatus.SUCCESS.value,
            attempts=[
                AcquisitionAttempt(
                    candidate_id=candidate.candidate_id,
                    rank=1,
                    status=AttemptStatus.SUCCESS.value,
                    duration_ms=manifest.acquisition.duration_ms,
                    source_id="fixture",
                    format=fmt,
                    url=candidate.resource_url,
                )
            ],
            data_dir=str(store.data_dir),
            candidate=candidate.to_dict(),
            artifact=artifact,
            manifest=manifest,
            manifest_path=str(manifest_path),
            cached=False,
            warnings=[],
            seed=self.settings.seed,
            code_version=__version__,
            dataset_sha256=digest,
        )
        validation = ValidationCoordinator(self.settings).validate(acquisition)
        self._require_validation(validation, acquisition)
        report = validation.report
        assert report is not None
        artifacts = {
            "manifest": str(manifest_path),
            "validation_report_json": validation.report_paths.get("report_json", ""),
            "validation_report_md": validation.report_paths.get("report_md", ""),
        }
        return StageOutcome(
            value=validation,
            outputs={
                "dataset_id": candidate.candidate_id,
                "dataset_sha256": digest,
                "source_mode": "offline_fixture",
            },
            artifacts={key: value for key, value in artifacts.items() if value},
            summary={
                "dataset_id": candidate.candidate_id,
                "source": "committed fixture",
                "format": fmt,
                "bytes": size,
                "sha256": digest,
                "validation_status": report.status.value,
                "can_preprocess": validation.can_preprocess,
                "target_column": report.target.get("column"),
            },
            warnings=list(report.warnings),
        )

    def _source_network(self) -> StageOutcome:
        assert self.config.query is not None
        query = self.config.query
        report_path = Path(self.settings.report_dir) / "acquisition" / "candidates.json"
        discovery_report = None
        if self.config.refresh_discovery or not report_path.is_file():
            adapters = self.adapters
            if adapters is None:
                adapters = build_adapters(self.settings, query)
            discovery_report = discover(adapters, query, self.settings)
            write_discovery_reports(discovery_report, self.settings.report_dir)
            if discovery_report.selected is None:
                failure = discovery_report.failure
                raise PipelineError(
                    CODE_UPSTREAM_FAILURE,
                    "dataset discovery did not select an eligible candidate",
                    "Inspect discovery failures and retry; no dataset is fabricated.",
                    details={
                        "upstream_code": failure.code if failure else "CW-DISC-002",
                        "reasons": list(failure.reasons) if failure else [],
                    },
                )
        else:
            try:
                cached = _read_json(report_path)
            except (OSError, ValueError) as exc:
                raise PipelineError(
                    CODE_INVALID_CONFIG,
                    f"cached discovery report could not be read: {exc}",
                    "Re-run with --refresh-discovery.",
                ) from exc
            if cached.get("query") != query.to_dict():
                raise PipelineError(
                    CODE_INVALID_CONFIG,
                    "the cached discovery report was produced by a different query",
                    "Run with --refresh-discovery so discovery matches this pipeline.",
                )

        candidates = load_candidates(report_path)[: self.config.max_candidates]
        if not candidates:
            raise PipelineError(
                CODE_UPSTREAM_FAILURE,
                "no eligible candidate with a downloadable resource was found",
                "Broaden the query or check source availability; no dataset is fabricated.",
                details={"candidates_report": self._display(report_path)},
            )
        validation = ValidationCoordinator(self.settings).validate_candidates(
            candidates,
            offline=False,
            max_candidates=self.config.max_candidates,
        )
        self._require_validation(validation, None)
        report = validation.report
        assert report is not None
        artifacts = {
            "candidates": str(report_path),
            "rejections": str(report_path.with_name("rejections.json")),
            "validation_report_json": validation.report_paths.get("report_json", ""),
            "validation_report_md": validation.report_paths.get("report_md", ""),
        }
        if validation.acquisition is not None:
            artifacts["manifest"] = validation.acquisition.manifest_path or ""
        return StageOutcome(
            value=validation,
            outputs={
                "dataset_id": report.dataset_id,
                "dataset_sha256": report.artifact_sha256,
                "source_mode": "network",
            },
            artifacts={key: value for key, value in artifacts.items() if value},
            summary={
                "dataset_id": report.dataset_id,
                "source": "network discovery",
                "validation_status": report.status.value,
                "can_preprocess": validation.can_preprocess,
                "target_column": report.target.get("column"),
                "artifact_sha256": report.artifact_sha256,
            },
            warnings=list(report.warnings),
        )

    def _require_validation(self, validation: Any, acquisition: Any) -> None:
        if validation.can_preprocess:
            return
        report = getattr(validation, "report", None)
        gate = getattr(report, "gate", None)
        failure = getattr(validation, "failure", None)
        upstream_code = (
            failure.code if failure is not None else "CW-VAL-002"
        )
        details: dict[str, Any] = {"upstream_code": upstream_code}
        if failure is not None:
            details["failure"] = failure.to_dict()
        if gate is not None:
            details["blocking_checks"] = list(getattr(gate, "blocking", []) or [])
            details["gate"] = gate.to_dict()
        artifacts: dict[str, str] = {}
        if report is not None:
            artifacts.update(
                {
                    key: value
                    for key, value in (getattr(validation, "report_paths", {}) or {}).items()
                    if value
                }
            )
        raise PipelineError(
            CODE_UPSTREAM_FAILURE,
            "the dataset did not pass the Phase 4 validation gate",
            (
                "Inspect the validation report, correct the dataset or choose another "
                "candidate, then retry."
            ),
            details=details,
            artifacts=artifacts,
        )

    def _preprocess(self, validation: Any) -> StageOutcome:
        assert self.config.preprocess is not None
        coordinator = PreprocessingCoordinator(self.settings)
        dataset = coordinator.prepare(
            validation,
            config=self.config.preprocess,
            run_id=self._stage_id("prep"),
        )
        return StageOutcome(
            value=dataset,
            outputs={
                "preprocessing_run_id": dataset.run_id,
                "dataset_id": dataset.dataset_id,
            },
            artifacts={key: value for key, value in dataset.paths.items() if value},
            summary={
                "preprocessing_run_id": dataset.run_id,
                "dataset_id": dataset.dataset_id,
                "features": len(dataset.feature_names),
                "train_rows": len(dataset.train.y),
                "val_rows": len(dataset.val.y),
                "test_rows": len(dataset.test.y),
                "class_weights": dataset.class_weights,
                "warnings": len(dataset.warnings),
            },
            warnings=list(dataset.warnings),
        )

    def _train(self, dataset: Any) -> StageOutcome:
        assert self.config.mlp is not None
        trainer = MLPTrainer(self.settings, self.config.mlp)
        result = trainer.train(dataset, model_run_id=self._stage_id("mlp"))
        metadata = result.metadata
        diagnostics = result.diagnostics or {}
        training = metadata.get("training") or {}
        architecture = metadata.get("architecture") or {}
        raw_warnings = metadata.get("warnings")
        warnings = list(raw_warnings) if isinstance(raw_warnings, list) else []
        return StageOutcome(
            value=result,
            outputs={"model_run_id": result.run_id},
            artifacts={key: value for key, value in result.paths.items() if value},
            summary={
                "model_run_id": result.run_id,
                "family": metadata.get("model_family", "mlp"),
                "epochs_trained": int(
                    diagnostics.get("epochs_completed")
                    or training.get("epochs_completed")
                    or len(result.history.get("loss") or [])
                ),
                "best_epoch": diagnostics.get("best_epoch"),
                "stopped_early": diagnostics.get("stopped_early"),
                "threshold": training.get("threshold"),
                "hidden_units": list(architecture.get("hidden_units") or []),
            },
            warnings=warnings,
        )

    def _evaluate(self, dataset: Any, model_run_id: str) -> StageOutcome:
        assert self.config.evaluation is not None
        coordinator = BenchmarkCoordinator(self.settings, self.config.evaluation)
        result = coordinator.run(
            dataset,
            mlp_run_id=model_run_id,
            benchmark_run_id=self._stage_id("bench"),
        )
        selection = result.metrics.get("selection") or {}
        return StageOutcome(
            value=result,
            outputs={
                "benchmark_run_id": result.benchmark_run_id,
                "selected_family": result.selected_family,
            },
            artifacts={key: value for key, value in result.paths.items() if value},
            summary={
                "benchmark_run_id": result.benchmark_run_id,
                "complete": result.complete,
                "selected_family": result.selected_family,
                "selected_threshold": selection.get("threshold"),
                "selection_metric": selection.get("selection_metric"),
                "n_test": result.n_test,
                "test_metrics": (
                    result.test_metrics(result.selected_family)
                    if result.selected_family
                    else {}
                ),
            },
            warnings=list(result.warnings),
        )

    def _explain(self, benchmark_run_id: str, family: str | None) -> StageOutcome:
        assert self.config.explain is not None
        coordinator = ExplainCoordinator(self.settings, self.config.explain)
        result = coordinator.run(
            family,
            benchmark_run_id=benchmark_run_id,
            explain_run_id=self._stage_id("xai"),
            split="val",
        )
        return StageOutcome(
            value=result,
            outputs={"explain_run_id": result.run_id, "explain_family": result.family},
            artifacts={key: value for key, value in result.paths.items() if value},
            summary={
                "explain_run_id": result.run_id,
                "family": result.family,
                "complete": result.complete,
                "local_examples": len(result.local_explanations),
                "global_top_features": [
                    item.get("feature")
                    for item in (
                        result.global_explanation.top_features(5)
                        if result.global_explanation is not None
                        else []
                    )
                ],
            },
            warnings=list(result.warnings),
        )

    def _score(self, benchmark_run_id: str) -> StageOutcome:
        assert self.config.risk is not None
        service = RiskService(
            self.settings,
            self.config.risk,
            benchmark_run_id=benchmark_run_id,
        )
        result = service.score_population(
            split="val",
            explain="top",
            explain_top_n=10,
            run_id=self._stage_id("risk"),
        )
        return StageOutcome(
            value=result,
            outputs={"risk_run_id": result.run_id},
            artifacts={key: value for key, value in result.paths.items() if value},
            summary={
                "risk_run_id": result.run_id,
                "n_scored": len(result.table.records),
                "n_rejected": len(result.rejections),
                "classification_threshold": result.table.classification_threshold,
                "tier_counts": result.summary.get("tier_counts"),
                "drivers_available": sum(
                    1
                    for record in result.table.records
                    if record.top_positive or record.top_negative
                ),
            },
            warnings=list(result.warnings),
        )

    def _simulate(self, table: Any) -> StageOutcome:
        assert self.config.scenario is not None
        service = SimulationService(self.settings)
        result = service.run_scenario(
            self.config.scenario,
            table=table,
            scenario_id=self._stage_id("sim"),
        )
        projection = result.result.projection
        allocation = result.result.allocation
        return StageOutcome(
            value=result,
            outputs={"scenario_id": result.scenario_id},
            artifacts={key: value for key, value in result.paths.items() if value},
            summary={
                "scenario_id": result.scenario_id,
                "budget": float(self.config.scenario.budget),
                "assumptions": self.config.scenario.assumptions.payload(),
                "contacted": allocation.selected_count,
                "spend": allocation.allocated_spend,
                "expected_prevented_churn": projection.expected_prevented_churn,
                "hypothetical": True,
            },
            warnings=list(result.warnings),
        )

    def _verify_outputs(self) -> None:
        required: list[tuple[str, str]] = []
        for name in STAGE_NAMES:
            record = self._records[name]
            if record.get("status") != "succeeded":
                raise PipelineError(
                    CODE_VERIFICATION_FAILED,
                    f"stage {name!r} did not succeed",
                    "Inspect run.json and rerun the pipeline.",
                    stage=name,
                )
            for key, path in (record.get("artifacts") or {}).items():
                required.append((f"{name}.{key}", path))
        missing = [
            f"{label}: {path}" for label, path in required if not self._resolve(path).exists()
        ]
        if missing:
            raise PipelineError(
                CODE_VERIFICATION_FAILED,
                "completed run is missing required artifacts: " + "; ".join(missing),
                "Do not trust the models/artifacts; delete the run and rerun it.",
            )

    # -- orchestration ------------------------------------------------------
    def run(self) -> PipelineResult:
        if self.config.resume:
            self._load_existing()
            if self._manifest.get("status") == "succeeded":
                self._notify(
                    f"[run] already complete: {self.manifest_path}"
                )
                return self._result()
        else:
            self._start_new()

        try:
            # 1-2. Discovery/acquisition/validation.
            if self._skipped("preprocess"):
                validation = None
            else:
                source = self._run_stage("source", self._source)
                validation = source.value

            # 3. Preprocessing.
            if self._skipped("preprocess"):
                dataset = PreprocessingCoordinator(self.settings).load(
                    str(self._records["preprocess"]["outputs"]["preprocessing_run_id"])
                )
            else:
                assert validation is not None
                dataset = self._run_stage(
                    "preprocess", lambda: self._preprocess(validation)
                ).value

            # 4. Model training.
            if self._skipped("train"):
                model_run_id = str(self._records["train"]["outputs"]["model_run_id"])
            else:
                model_run_id = self._run_stage(
                    "train", lambda: self._train(dataset)
                ).outputs["model_run_id"]

            # 5. Benchmark evaluation.
            if self._skipped("evaluate"):
                benchmark_run_id = str(
                    self._records["evaluate"]["outputs"]["benchmark_run_id"]
                )
                selected_family = self._records["evaluate"]["outputs"].get(
                    "selected_family"
                )
            else:
                evaluate = self._run_stage(
                    "evaluate",
                    lambda: self._evaluate(dataset, str(model_run_id)),
                )
                benchmark_run_id = str(evaluate.outputs["benchmark_run_id"])
                selected_family = evaluate.outputs.get("selected_family")

            # 6. Explainability.
            if not self._skipped("explain"):
                self._run_stage(
                    "explain",
                    lambda: self._explain(str(benchmark_run_id), selected_family),
                )

            # 7. Risk intelligence.
            if self._skipped("score"):
                risk_table = RiskService(
                    self.settings,
                    self.config.risk,
                    benchmark_run_id=str(benchmark_run_id),
                ).load_run(
                    str(self._records["score"]["outputs"]["risk_run_id"])
                ).table
            else:
                score = self._run_stage(
                    "score", lambda: self._score(str(benchmark_run_id))
                )
                risk_table = score.value.table

            # 8. Retention simulation.
            if not self._skipped("simulate"):
                self._run_stage("simulate", lambda: self._simulate(risk_table))

            self._verify_outputs()
            self._manifest["status"] = "succeeded"
            self._manifest["finished_at"] = self._now()
            self._manifest["error"] = None
            self._write_manifest()
            self._notify(f"[run] succeeded: {self.manifest_path}")
            return self._result()
        except PipelineError as exc:
            self._manifest["status"] = "failed"
            self._manifest["finished_at"] = self._now()
            self._manifest["error"] = exc.to_dict()
            try:
                self._write_manifest()
            except OSError:
                logger.exception("could not persist the failed run manifest")
            raise


def run_pipeline(
    settings: Settings,
    config: PipelineConfig | None = None,
    *,
    adapters: Any = None,
    progress: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Convenience wrapper used by the CLI, tests and the dashboard."""
    return PipelineRunner(
        settings,
        config,
        adapters=adapters,
        progress=progress,
    ).run()


__all__ = [
    "LOCAL_DATASET_FORMATS",
    "PIPELINE_SCHEMA_VERSION",
    "STAGE_NAMES",
    "PipelineConfig",
    "PipelineResult",
    "PipelineRunner",
    "StageOutcome",
    "build_pipeline_config",
    "run_pipeline",
]
