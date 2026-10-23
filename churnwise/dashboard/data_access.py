"""Read-only, cache-aware access to persisted ChurnWise artifacts.

Every loader takes an explicit path, checks the file's ``(mtime_ns, size)``
token and uses ``st.cache_data`` so a rerun does not re-read a 7k-row Parquet
table or a large SHAP payload. Writes by a training/discovery action change the
token and therefore invalidate the cached value on the next run.

Missing artifacts return ``None``; corrupt artifacts raise :class:`ArtifactError`
with an actionable message. Pages never invent a value to fill a gap.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from churnwise.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ARTIFACT_ERROR = "CW-UI-001"
CODE_CACHE_ERROR = "CW-UI-002"

RISK_SUMMARY = "risk_summary.json"
RISK_MANIFEST = "run_manifest.json"
RISK_SCORES = "churn_scores.parquet"
RISK_DRIVERS = "drivers.json"
SIMULATION_RESULT = "result.json"
BENCHMARK_METRICS = "metrics.json"
EXPLANATION_GLOBAL = "global.json"
PREPARED_SUMMARY = "summary.json"
MODEL_METADATA = "metadata.json"


class ArtifactError(Exception):
    """A structured, privacy-safe artifact failure for the dashboard."""

    def __init__(self, message: str, *, path: Path | str | None = None, remediation: str = ""):
        super().__init__(message)
        self.code = CODE_ARTIFACT_ERROR
        self.message = message
        self.path = str(path) if path is not None else ""
        self.remediation = remediation

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class RunIdentity:
    """Header identity shared by every dashboard page.

    Empty fields are rendered as an explicit ``not available`` label rather
    than being hidden, so a reader can always tell which identity is missing.
    """

    run_id: str = ""
    family: str = ""
    dataset_sha256: str = ""
    generated_at: str = ""
    source: str = ""

    @property
    def dataset_prefix(self) -> str:
        return self.dataset_sha256[:12] if self.dataset_sha256 else ""

    @property
    def missing(self) -> list[str]:
        labels = {
            "run_id": "run id",
            "family": "model family",
            "dataset_prefix": "dataset hash",
            "generated_at": "generated at",
        }
        return [label for key, label in labels.items() if not getattr(self, key)]

    def as_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "family": self.family,
            "dataset_prefix": self.dataset_prefix,
            "generated_at": self.generated_at,
            "source": self.source,
        }


def current_settings() -> Settings:
    """Settings for the running project (environment overrides are honoured)."""

    return Settings.load(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Cached primitive readers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=128)
def _json_cached(path: str, mtime_ns: int, size: int) -> Any:  # noqa: ARG001 - token key
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, max_entries=32)
def _parquet_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:  # noqa: ARG001
    return pd.read_parquet(path)


def _token(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), int(stat.st_mtime_ns), int(stat.st_size)


def load_json(path: Path | str | None) -> Any | None:
    """Load JSON, or return ``None`` when the file does not exist.

    Raises :class:`ArtifactError` for unreadable or malformed files so a page
    can show a useful state instead of a Streamlit traceback.
    """

    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return _json_cached(*_token(target))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(
            f"artifact could not be read as JSON: {exc.__class__.__name__}",
            path=target,
            remediation="Regenerate this artifact with the documented Phase command.",
        ) from exc


def load_frame(path: Path | str | None) -> pd.DataFrame | None:
    """Load a Parquet table, or ``None`` when it does not exist."""

    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return _parquet_cached(*_token(target))
    except Exception as exc:  # pyarrow raises engine-specific exceptions
        raise ArtifactError(
            f"artifact could not be read as Parquet: {exc.__class__.__name__}",
            path=target,
            remediation="Regenerate the risk run and retry.",
        ) from exc


def load_bytes(path: Path | str | None) -> bytes | None:
    if path is None:
        return None
    target = Path(path)
    return target.read_bytes() if target.is_file() else None


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_token(path: Path | str | None) -> tuple[int, int] | None:
    """Public cache token used by tests and the reliability page."""

    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    stat = target.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


# ---------------------------------------------------------------------------
# Latest artifact discovery
# ---------------------------------------------------------------------------
def _latest_file(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    files = [item for item in directory.glob(pattern) if item.is_file()]
    if not files:
        return None
    return max(files, key=lambda item: (item.stat().st_mtime_ns, item.name))


def _latest_dir(root: Path, marker: str) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [item for item in root.iterdir() if item.is_dir() and (item / marker).is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.name)


def run_ids(root: Path, marker: str, *, limit: int = 200) -> list[str]:
    if not root.is_dir():
        return []
    names = [item.name for item in root.iterdir() if item.is_dir() and (item / marker).is_file()]
    return sorted(names, reverse=True)[:limit]


def latest_validation_path(settings: Settings) -> Path | None:
    """Newest validation report, ignoring Phase 3 attempt logs in the same folder."""

    directory = Path(settings.report_dir) / "validation"
    if not directory.is_dir():
        return None
    files = [
        item
        for item in directory.glob("*.json")
        if item.is_file() and item.name not in {"attempts.json", "candidates.json"}
    ]
    if not files:
        return None
    return max(files, key=lambda item: (item.stat().st_mtime_ns, item.name))


def latest_eda_path(settings: Settings, dataset_id: str | None = None) -> Path | None:
    directory = Path(settings.report_dir) / "eda"
    if dataset_id:
        direct = directory / f"{dataset_id.replace(':', '_')}.json"
        if direct.is_file():
            return direct
    return _latest_file(directory, "*.json")


def candidates_path(settings: Settings) -> Path:
    return Path(settings.report_dir) / "acquisition" / "candidates.json"


def attempts_path(settings: Settings) -> Path:
    return Path(settings.report_dir) / "acquisition" / "attempts.json"


def latest_prepared_dir(settings: Settings) -> Path | None:
    return _latest_dir(Path(settings.artifact_dir) / "prepared", PREPARED_SUMMARY)


def latest_model_dir(settings: Settings) -> Path | None:
    root = Path(settings.artifact_dir) / "models"
    if not root.is_dir():
        return None
    candidates = [
        item for item in root.iterdir() if item.is_dir() and (item / "mlp" / MODEL_METADATA).is_file()
    ]
    return max(candidates, key=lambda item: item.name) if candidates else None


def latest_benchmark_dir(settings: Settings) -> Path | None:
    return _latest_dir(Path(settings.artifact_dir) / "evaluation", BENCHMARK_METRICS)


def latest_explanation_dir(settings: Settings) -> Path | None:
    return _latest_dir(Path(settings.artifact_dir) / "explanations", EXPLANATION_GLOBAL)


def latest_risk_dir(settings: Settings) -> Path | None:
    return _latest_dir(Path(settings.artifact_dir) / "predictions", RISK_SUMMARY)


def latest_simulation_dir(settings: Settings) -> Path | None:
    return _latest_dir(Path(settings.artifact_dir) / "simulations", SIMULATION_RESULT)


def benchmark_ids(settings: Settings) -> list[str]:
    return run_ids(Path(settings.artifact_dir) / "evaluation", BENCHMARK_METRICS)


def explanation_ids(settings: Settings) -> list[str]:
    return run_ids(Path(settings.artifact_dir) / "explanations", EXPLANATION_GLOBAL)


def risk_ids(settings: Settings) -> list[str]:
    return run_ids(Path(settings.artifact_dir) / "predictions", RISK_SUMMARY)


def simulation_ids(settings: Settings) -> list[str]:
    return run_ids(Path(settings.artifact_dir) / "simulations", SIMULATION_RESULT)


def model_ids(settings: Settings) -> list[str]:
    root = Path(settings.artifact_dir) / "models"
    if not root.is_dir():
        return []
    names = [item.name for item in root.iterdir() if (item / "mlp" / MODEL_METADATA).is_file()]
    return sorted(names, reverse=True)


# ---------------------------------------------------------------------------
# Typed artifact accessors
# ---------------------------------------------------------------------------
def validation_report(settings: Settings) -> tuple[dict[str, Any] | None, Path | None]:
    path = latest_validation_path(settings)
    return load_json(path), path


def eda_profile(settings: Settings, dataset_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    path = latest_eda_path(settings, dataset_id)
    return load_json(path), path


def prepared_summary(settings: Settings) -> tuple[dict[str, Any] | None, Path | None]:
    root = latest_prepared_dir(settings)
    path = root / PREPARED_SUMMARY if root else None
    return load_json(path), path


def preprocessing_schema(settings: Settings, run_id: str | None) -> tuple[dict[str, Any] | None, Path | None]:
    """Read the Phase 5 schema JSON written beside the fitted preprocessor."""

    if not run_id:
        prepared, _ = prepared_summary(settings)
        run_id = str((prepared or {}).get("run_id") or "") or None
    if not run_id:
        return None, None
    path = Path(settings.artifact_dir) / "preprocessors" / str(run_id) / "schema.json"
    return load_json(path), path


def model_metadata(settings: Settings) -> tuple[dict[str, Any] | None, Path | None]:
    root = latest_model_dir(settings)
    path = root / "mlp" / MODEL_METADATA if root else None
    return load_json(path), path


def training_history(settings: Settings) -> tuple[dict[str, Any] | None, Path | None]:
    root = latest_model_dir(settings)
    path = root / "mlp" / "history.json" if root else None
    return load_json(path), path


def benchmark_metrics(settings: Settings, run_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(settings.artifact_dir) / "evaluation" / run_id if run_id else latest_benchmark_dir(settings)
    path = root / BENCHMARK_METRICS if root else None
    return load_json(path), path


def benchmark_curves(settings: Settings, run_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(settings.artifact_dir) / "evaluation" / run_id if run_id else latest_benchmark_dir(settings)
    path = root / "curves.json" if root else None
    return load_json(path), path


def explanation_global(settings: Settings, run_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(settings.artifact_dir) / "explanations" / run_id if run_id else latest_explanation_dir(settings)
    path = root / EXPLANATION_GLOBAL if root else None
    return load_json(path), path


def explanation_manifest(settings: Settings, run_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(settings.artifact_dir) / "explanations" / run_id if run_id else latest_explanation_dir(settings)
    path = root / RISK_MANIFEST if root else None
    return load_json(path), path


def risk_summary(settings: Settings, run_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(settings.artifact_dir) / "predictions" / run_id if run_id else latest_risk_dir(settings)
    path = root / RISK_SUMMARY if root else None
    return load_json(path), path


def risk_manifest(settings: Settings, run_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(settings.artifact_dir) / "predictions" / run_id if run_id else latest_risk_dir(settings)
    path = root / RISK_MANIFEST if root else None
    return load_json(path), path


def risk_table(settings: Settings, run_id: str | None = None) -> tuple[pd.DataFrame | None, Path | None]:
    root = Path(settings.artifact_dir) / "predictions" / run_id if run_id else latest_risk_dir(settings)
    path = root / RISK_SCORES if root else None
    return load_frame(path), path


def risk_drivers(settings: Settings, run_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(settings.artifact_dir) / "predictions" / run_id if run_id else latest_risk_dir(settings)
    path = root / RISK_DRIVERS if root else None
    return load_json(path), path


def simulation_result(settings: Settings, scenario_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(settings.artifact_dir) / "simulations" / scenario_id if scenario_id else latest_simulation_dir(settings)
    path = root / SIMULATION_RESULT if root else None
    return load_json(path), path


def simulation_manifest(settings: Settings, scenario_id: str | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(settings.artifact_dir) / "simulations" / scenario_id if scenario_id else latest_simulation_dir(settings)
    path = root / RISK_MANIFEST if root else None
    return load_json(path), path


def acquisition_candidates(settings: Settings) -> tuple[dict[str, Any] | None, Path | None]:
    path = candidates_path(settings)
    return load_json(path), path


def acquisition_attempts(settings: Settings) -> tuple[dict[str, Any] | None, Path | None]:
    path = attempts_path(settings)
    return load_json(path), path


def local_explanation(settings: Settings, row_position: int) -> tuple[dict[str, Any] | None, Path | None]:
    """Find the newest full SHAP run holding an explanation for this position."""

    root = Path(settings.artifact_dir) / "explanations"
    if not root.is_dir():
        return None, None
    for run_id in explanation_ids(settings):
        path = root / run_id / "local" / f"row_{int(row_position):05d}.json"
        if path.is_file():
            return load_json(path), path
    return None, None


def local_explanation_plot(settings: Settings, row_position: int) -> Path | None:
    root = Path(settings.artifact_dir) / "explanations"
    if not root.is_dir():
        return None
    for run_id in explanation_ids(settings):
        path = root / run_id / "local" / f"row_{int(row_position):05d}.png"
        if path.is_file():
            return path
    return None


# ---------------------------------------------------------------------------
# Identity builders
# ---------------------------------------------------------------------------
def identity_from_risk(summary: dict[str, Any] | None, manifest: dict[str, Any] | None = None) -> RunIdentity:
    summary = summary or {}
    manifest = manifest or {}
    model = summary.get("model") or {}
    provenance = summary.get("provenance") or {}
    return RunIdentity(
        run_id=str(summary.get("run_id") or manifest.get("run_id") or ""),
        family=str(model.get("family") or manifest.get("model_family") or ""),
        dataset_sha256=str(provenance.get("dataset_sha256") or manifest.get("dataset_sha256") or ""),
        generated_at=str(summary.get("generated_at") or manifest.get("generated_at") or ""),
        source="risk run",
    )


def identity_from_benchmark(metrics: dict[str, Any] | None) -> RunIdentity:
    metrics = metrics or {}
    preprocessing = metrics.get("preprocessing") or {}
    dataset = metrics.get("dataset") or {}
    selection = metrics.get("selection") or {}
    return RunIdentity(
        run_id=str(metrics.get("benchmark_run_id") or ""),
        family=str(selection.get("selected_family") or ""),
        dataset_sha256=str(preprocessing.get("dataset_sha256") or dataset.get("dataset_sha256") or ""),
        generated_at=str(metrics.get("generated_at") or ""),
        source="benchmark",
    )


def identity_from_simulation(result: dict[str, Any] | None) -> RunIdentity:
    result = result or {}
    source = result.get("input") or {}
    provenance = result.get("provenance") or {}
    model = source.get("model") or provenance.get("model") or {}
    return RunIdentity(
        run_id=str(result.get("scenario_id") or ""),
        family=str(model.get("family") or ""),
        dataset_sha256=str(source.get("dataset_sha256") or provenance.get("dataset_sha256") or ""),
        generated_at=str(result.get("generated_at") or ""),
        source="campaign simulation",
    )


def identity_from_explanation(manifest: dict[str, Any] | None) -> RunIdentity:
    manifest = manifest or {}
    model = manifest.get("model") or {}
    return RunIdentity(
        run_id=str(manifest.get("run_id") or ""),
        family=str(manifest.get("family") or model.get("family") or ""),
        dataset_sha256=str(manifest.get("dataset_sha256") or ""),
        generated_at=str(manifest.get("generated_at") or ""),
        source="explanation run",
    )


def identity_from_validation(report: dict[str, Any] | None) -> RunIdentity:
    report = report or {}
    provenance = report.get("provenance") or {}
    return RunIdentity(
        run_id=str(provenance.get("run_id") or report.get("dataset_id") or ""),
        family="not applicable",
        dataset_sha256=str(report.get("artifact_sha256") or provenance.get("dataset_sha256") or ""),
        generated_at=str(report.get("generated_at") or ""),
        source="validation report",
    )


def identity_from_training(
    prepared: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None
) -> RunIdentity:
    prepared = prepared or {}
    metadata = metadata or {}
    provenance = prepared.get("provenance") or {}
    family = str(metadata.get("model_family") or "mlp") if metadata else ""
    return RunIdentity(
        run_id=str(metadata.get("model_run_id") or prepared.get("run_id") or ""),
        family=family,
        dataset_sha256=str(prepared.get("dataset_sha256") or provenance.get("dataset_sha256") or ""),
        generated_at=str(metadata.get("created_at") or provenance.get("generated_at") or ""),
        source="training run",
    )


def identity_from_candidates(report: dict[str, Any] | None) -> RunIdentity:
    report = report or {}
    selected = report.get("selected") or {}
    return RunIdentity(
        run_id=str(selected.get("candidate_id") or "discovery"),
        family="not applicable",
        dataset_sha256="",
        generated_at=str(report.get("generated_at") or ""),
        source="dataset discovery",
    )


# ---------------------------------------------------------------------------
# Index helpers for the reliability page
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArtifactEntry:
    kind: str
    run_id: str
    path: Path
    generated_at: str = ""
    detail: str = ""

    @property
    def exists(self) -> bool:
        return self.path.exists()


def artifact_inventory(settings: Settings) -> list[ArtifactEntry]:
    entries: list[ArtifactEntry] = []
    validation, validation_path = validation_report(settings)
    if validation_path:
        entries.append(
            ArtifactEntry(
                "Validation report",
                str(validation.get("dataset_id") or validation_path.stem),
                validation_path,
                str(validation.get("generated_at") or ""),
                str(validation.get("status") or ""),
            )
        )
    eda, eda_path = eda_profile(settings, str((validation or {}).get("dataset_id") or "") or None)
    if eda_path:
        entries.append(
            ArtifactEntry(
                "EDA profile",
                str(eda.get("dataset_id") or eda_path.stem),
                eda_path,
                str(eda.get("generated_at") or ""),
                f"{eda.get('n_rows', '?')} rows x {eda.get('n_cols', '?')} columns",
            )
        )
    prepared, prepared_path = prepared_summary(settings)
    if prepared_path:
        entries.append(
            ArtifactEntry(
                "Prepared dataset",
                str(prepared.get("run_id") or prepared_path.parent.name),
                prepared_path,
                str((prepared.get("provenance") or {}).get("generated_at") or ""),
                f"{prepared.get('feature_count', '?')} features",
            )
        )
    metadata, metadata_path = model_metadata(settings)
    if metadata_path:
        entries.append(
            ArtifactEntry(
                "MLP model",
                str(metadata.get("model_run_id") or metadata_path.parents[1].name),
                metadata_path,
                str(metadata.get("created_at") or ""),
                str(metadata.get("model_family") or "mlp"),
            )
        )
    metrics, metrics_path = benchmark_metrics(settings)
    if metrics_path:
        entries.append(
            ArtifactEntry(
                "Benchmark",
                str(metrics.get("benchmark_run_id") or metrics_path.parent.name),
                metrics_path,
                str(metrics.get("generated_at") or ""),
                str((metrics.get("selection") or {}).get("selected_family") or ""),
            )
        )
    explanation, explanation_path = explanation_global(settings)
    if explanation_path:
        entries.append(
            ArtifactEntry(
                "SHAP explanation",
                str(explanation.get("run_id") or explanation_path.parent.name),
                explanation_path,
                str(explanation.get("generated_at") or ""),
                f"{len(explanation.get('feature_importance', []))} grouped features",
            )
        )
    summary, summary_path = risk_summary(settings)
    if summary_path:
        entries.append(
            ArtifactEntry(
                "Risk run",
                str(summary.get("run_id") or summary_path.parent.name),
                summary_path,
                str(summary.get("generated_at") or ""),
                f"{summary.get('n_rows', '?')} scored rows",
            )
        )
    result, result_path = simulation_result(settings)
    if result_path:
        entries.append(
            ArtifactEntry(
                "Simulation",
                str(result.get("scenario_id") or result_path.parent.name),
                result_path,
                str(result.get("generated_at") or ""),
                f"{((result.get('allocation') or {}).get('selected_count', '?'))} selected",
            )
        )
    return entries


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "ArtifactEntry",
    "ArtifactError",
    "BENCHMARK_METRICS",
    "CODE_ARTIFACT_ERROR",
    "EXPLANATION_GLOBAL",
    "MODEL_METADATA",
    "PROJECT_ROOT",
    "PREPARED_SUMMARY",
    "RISK_DRIVERS",
    "RISK_MANIFEST",
    "RISK_SCORES",
    "RISK_SUMMARY",
    "RunIdentity",
    "SIMULATION_RESULT",
    "acquisition_attempts",
    "acquisition_candidates",
    "artifact_inventory",
    "artifact_token",
    "attempts_path",
    "benchmark_curves",
    "benchmark_ids",
    "benchmark_metrics",
    "candidates_path",
    "current_settings",
    "eda_profile",
    "explanation_global",
    "explanation_ids",
    "explanation_manifest",
    "file_sha256",
    "identity_from_benchmark",
    "identity_from_candidates",
    "identity_from_explanation",
    "identity_from_risk",
    "identity_from_simulation",
    "identity_from_training",
    "identity_from_validation",
    "latest_benchmark_dir",
    "latest_eda_path",
    "latest_explanation_dir",
    "latest_model_dir",
    "latest_prepared_dir",
    "latest_risk_dir",
    "latest_simulation_dir",
    "latest_validation_path",
    "load_bytes",
    "load_frame",
    "load_json",
    "local_explanation",
    "local_explanation_plot",
    "model_ids",
    "model_metadata",
    "prepared_summary",
    "preprocessing_schema",
    "risk_drivers",
    "risk_ids",
    "risk_manifest",
    "risk_summary",
    "risk_table",
    "run_ids",
    "simulation_ids",
    "simulation_manifest",
    "simulation_result",
    "training_history",
    "utc_now",
    "validation_report",
]
