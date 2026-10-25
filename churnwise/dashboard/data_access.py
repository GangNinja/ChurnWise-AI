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
THRESHOLD_ANALYSIS = "threshold_analysis.json"
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
class CustomerPopulation:
    """Stored scores joined to original rows through a verified split manifest.

    ``frame`` is ``None`` when the provenance chain is incomplete; ``reason``
    always explains why so a page can state the limitation rather than invent
    a customer profile.
    """

    frame: pd.DataFrame | None
    raw_columns: tuple[str, ...] = ()
    split: str = ""
    split_fingerprint: str = ""
    reason: str = ""


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


@st.cache_data(show_spinner=False, max_entries=8)
def _source_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:  # noqa: ARG001
    """Read the acquired source file once; this never downloads or retrains.

    Acquisition stores the source under its content digest without an
    extension, so the format is detected from the bytes: an ARFF header begins
    with ``@relation``/``@attribute`` while a CSV begins with its header row.
    """

    head = Path(path).read_bytes()[:8192].decode("utf-8", errors="replace")
    first = next(
        (
            line.strip()
            for line in head.splitlines()
            if line.strip() and not line.lstrip().startswith("%")
        ),
        "",
    )
    if first.lower().startswith("@relation") or first.lower().startswith("@attribute"):
        from scipy.io import arff

        records, _ = arff.loadarff(path)
        frame = pd.DataFrame(records)
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(
                    lambda value: value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value
                )
        return frame
    return pd.read_csv(path)


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


def latest_population_risk_dir(settings: Settings) -> Path | None:
    """Newest risk run scored over a named split, not an ad-hoc input batch.

    Batch assessments are valid runs and remain selectable in Risk intelligence
    and Customer detail, but they must not silently replace the population run
    that the Overview summarises.
    """

    root = Path(settings.artifact_dir) / "predictions"
    if not root.is_dir():
        return None
    candidates = sorted(
        (item for item in root.iterdir() if item.is_dir() and (item / RISK_SUMMARY).is_file()),
        key=lambda item: item.name,
        reverse=True,
    )
    for directory in candidates:
        payload = load_json(directory / RISK_SUMMARY) or {}
        split = str((payload.get("population") or {}).get("split") or "").strip().lower()
        if split != "input":
            return directory
    return latest_risk_dir(settings)


def latest_simulation_dir(settings: Settings) -> Path | None:
    return _latest_dir(Path(settings.artifact_dir) / "simulations", SIMULATION_RESULT)


def benchmark_ids(settings: Settings) -> list[str]:
    return run_ids(Path(settings.artifact_dir) / "evaluation", BENCHMARK_METRICS)


def explanation_ids(settings: Settings) -> list[str]:
    return run_ids(Path(settings.artifact_dir) / "explanations", EXPLANATION_GLOBAL)


def risk_ids(settings: Settings) -> list[str]:
    """List scoring runs once per distinct scored output, newest duplicate first."""
    root = Path(settings.artifact_dir) / "predictions"
    candidates = run_ids(root, RISK_SUMMARY)
    unique: list[str] = []
    signatures: set[str] = set()
    for run_id in candidates:
        run_root = root / run_id
        summary = load_json(run_root / RISK_SUMMARY) or {}
        scores = load_frame(run_root / RISK_SCORES)
        if scores is None or scores.empty or "churn_probability" not in scores:
            unique.append(run_id)
            continue

        # Ignore timestamps and insignificant floating-point noise, while
        # retaining row order, customer references, predictions and model
        # provenance so different input batches remain selectable.
        columns = [
            name
            for name in (
                "record_reference",
                "input_position",
                "churn_probability",
                "predicted_class",
                "predicted_label",
                "risk_tier",
                "classification_threshold",
                "model_sha256",
                "preprocessing_run_id",
                "observed_churn",
                "explanation_status",
            )
            if name in scores.columns
        ]
        normalized = scores[columns].copy()
        sort_columns = [name for name in ("input_position", "record_reference") if name in normalized]
        if sort_columns:
            normalized = normalized.sort_values(sort_columns, kind="stable")
        normalized = normalized.reset_index(drop=True)
        normalized["churn_probability"] = pd.to_numeric(
            normalized["churn_probability"], errors="coerce"
        ).round(12)
        identity = {
            "dataset_sha256": summary.get("dataset_sha256"),
            "split_fingerprint": summary.get("split_fingerprint"),
            "model_sha256": summary.get("model_sha256"),
            "preprocessor_hash": summary.get("preprocessor_hash"),
            "classification_threshold": summary.get("classification_threshold"),
            "split": (summary.get("population") or {}).get("split"),
            "rows": json.loads(normalized.to_json(orient="records", double_precision=12)),
        }
        signature = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        if signature not in signatures:
            signatures.add(signature)
            unique.append(run_id)
    return unique


def risk_run_labels(settings: Settings, runs: list[str] | None = None) -> dict[str, str]:
    """Build readable, distinct labels for population and uploaded-data runs."""
    root = Path(settings.artifact_dir) / "predictions"
    run_ids = list(runs if runs is not None else risk_ids(settings))
    summaries = {
        run_id: load_json(root / run_id / RISK_SUMMARY) or {}
        for run_id in run_ids
    }

    input_runs = sorted(
        (
            run_id
            for run_id, summary in summaries.items()
            if str((summary.get("population") or {}).get("split") or "").lower() == "input"
        ),
        key=lambda run_id: (str(summaries[run_id].get("generated_at") or ""), run_id),
    )
    input_numbers = {run_id: number for number, run_id in enumerate(input_runs, start=1)}
    labels: dict[str, str] = {}
    for run_id, summary in summaries.items():
        population = summary.get("population") or {}
        split = str(population.get("split") or "unknown").strip().lower()
        received = int(population.get("received") or summary.get("n_rows") or 0)
        family = str((summary.get("model") or {}).get("family") or "model")
        generated_at = str(summary.get("generated_at") or "")
        try:
            timestamp = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            stamp = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except (TypeError, ValueError):
            stamp = run_id[:15]

        if split == "input":
            source = str(summary.get("source_label") or "Uploaded dataset").strip()
            labels[run_id] = (
                f"Upload {input_numbers[run_id]:02d} · {source} · "
                f"{received:,} rows · {stamp}"
            )
        else:
            split_label = {
                "train": "Training",
                "val": "Validation",
                "test": "Test",
                "all": "Full population",
            }.get(split, split.title())
            labels[run_id] = f"{split_label} · {received:,} rows · {family} · {stamp}"
    return labels


def simulation_ids(settings: Settings) -> list[str]:
    return run_ids(Path(settings.artifact_dir) / "simulations", SIMULATION_RESULT)


def default_risk_run_id(settings: Settings, runs: list[str] | None = None) -> str | None:
    """Default to the newest population run, while keeping batches selectable."""

    available = list(runs if runs is not None else risk_ids(settings))
    primary = latest_population_risk_dir(settings)
    if primary is not None and primary.name in available:
        return primary.name
    return available[0] if available else None


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


def risk_summary(
    settings: Settings,
    run_id: str | None = None,
    *,
    prefer_population: bool = False,
) -> tuple[dict[str, Any] | None, Path | None]:
    if run_id:
        root = Path(settings.artifact_dir) / "predictions" / run_id
    elif prefer_population:
        root = latest_population_risk_dir(settings)
    else:
        root = latest_risk_dir(settings)
    path = root / RISK_SUMMARY if root else None
    return load_json(path), path


def risk_manifest(
    settings: Settings,
    run_id: str | None = None,
    *,
    prefer_population: bool = False,
) -> tuple[dict[str, Any] | None, Path | None]:
    if run_id:
        root = Path(settings.artifact_dir) / "predictions" / run_id
    elif prefer_population:
        root = latest_population_risk_dir(settings)
    else:
        root = latest_risk_dir(settings)
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


def customer_population(
    settings: Settings,
    run_id: str | None = None,
    *,
    prefer_population: bool = True,
) -> CustomerPopulation:
    """Join stored scores to original rows through the persisted split manifest.

    The join is accepted only when the dataset digest, split fingerprint,
    preprocessing run and row count all agree. Observed labels, when present,
    must match the source target for every row. Any mismatch returns ``None``
    and a reason; a guessed row mapping is never used.
    """

    summary, summary_path = risk_summary(settings, run_id, prefer_population=prefer_population)
    manifest, _ = risk_manifest(settings, run_id, prefer_population=prefer_population)
    if not summary or summary_path is None:
        return CustomerPopulation(None, reason="No population risk summary is available.")
    split_name = str((summary.get("population") or {}).get("split") or "").strip().lower()
    if split_name not in {"train", "val", "test"}:
        return CustomerPopulation(
            None,
            reason="This run was scored from an input batch, so it has no named split to join back to.",
        )

    preprocessing_id = str(
        (summary.get("model") or {}).get("preprocessing_run_id")
        or (manifest or {}).get("preprocessing_run_id")
        or ""
    )
    dataset_sha = str(
        (summary.get("provenance") or {}).get("dataset_sha256")
        or (manifest or {}).get("dataset_sha256")
        or ""
    )
    fingerprint = str(
        (summary.get("provenance") or {}).get("split_fingerprint")
        or (manifest or {}).get("split_fingerprint")
        or ""
    )
    if not preprocessing_id or not dataset_sha or not fingerprint:
        return CustomerPopulation(None, reason="The run does not record the preprocessing run, dataset digest and split fingerprint together.")

    split_path = Path(settings.artifact_dir) / "splits" / f"{preprocessing_id}.json"
    split = load_json(split_path)
    if not split:
        return CustomerPopulation(None, reason="The split manifest for this run is not available, so original rows cannot be matched.")
    if str(split.get("fingerprint") or "") != fingerprint:
        return CustomerPopulation(None, reason="The split fingerprint does not match the risk run; the join was refused.")
    if str(split.get("dataset_sha256") or "") != dataset_sha:
        return CustomerPopulation(None, reason="The split dataset digest does not match the risk run; the join was refused.")
    indices = ((split.get("row_indices") or {}).get(split_name) or [])
    if not indices:
        return CustomerPopulation(None, reason=f"The split manifest does not contain the {split_name} row indices.")

    raw_path = Path(settings.data_dir) / "cache" / "sha256" / dataset_sha
    if not raw_path.is_file():
        return CustomerPopulation(None, reason="The acquired source file is not in the local cache, so the original attributes are unavailable.")
    try:
        raw = _source_cached(*_token(raw_path))
    except Exception as exc:  # malformed source: report the limitation, never guess rows
        return CustomerPopulation(
            None,
            reason=(
                "The acquired source file could not be read as ARFF or CSV "
                f"({exc.__class__.__name__}); original attributes are unavailable."
            ),
        )

    risk = load_frame(summary_path.parent / RISK_SCORES)
    if risk is None or risk.empty:
        return CustomerPopulation(None, reason="The stored score table is empty.")
    if len(indices) != len(risk):
        return CustomerPopulation(
            None,
            reason=(
                f"The split has {len(indices):,} rows but the score table has {len(risk):,}; "
                "the join was refused."
            ),
        )
    positions = pd.to_numeric(risk["input_position"], errors="coerce")
    if positions.isna().any() or not ((positions % 1) == 0).all():
        return CustomerPopulation(None, reason="The score table has invalid input positions; the join was refused.")
    positions = positions.astype(int).tolist()
    if len(set(positions)) != len(positions) or min(positions) < 0 or max(positions) >= len(indices):
        return CustomerPopulation(None, reason="The score positions do not fit the split range; the join was refused.")
    if max(indices) >= len(raw) or min(indices) < 0:
        return CustomerPopulation(None, reason="The split points outside the source file; the join was refused.")

    validation, _ = validation_report(settings)
    target_column = str(((validation or {}).get("target") or {}).get("column") or "")
    target_mapping = ((validation or {}).get("target") or {}).get("mapping") or {}
    usable = set(((validation or {}).get("summary") or {}).get("usable_features") or [])
    if target_column and target_column in raw.columns:
        if "observed_churn" in risk.columns and target_mapping:
            expected = []
            for value in raw.iloc[[indices[position] for position in positions]][target_column].tolist():
                mapped = target_mapping.get(value, target_mapping.get(str(value)))
                if mapped is None:
                    try:
                        mapped = int(float(value))
                    except (TypeError, ValueError):
                        mapped = None
                expected.append(mapped)
            observed = pd.to_numeric(risk["observed_churn"], errors="coerce")
            compared = observed.notna()
            if compared.any() and any(
                expected[index] is None or int(expected[index]) != int(observed.iloc[index])
                for index in range(len(risk))
                if compared.iloc[index]
            ):
                return CustomerPopulation(None, reason="Observed labels do not match the source rows; the join was refused.")

    raw_rows = raw.iloc[[indices[position] for position in positions]].reset_index(drop=True)
    keep_raw = [str(column) for column in raw_rows.columns if str(column) != target_column]
    if usable:
        keep_raw = [column for column in keep_raw if column in usable]
    raw_rows = raw_rows[keep_raw].copy()
    collisions = set(raw_rows.columns) & set(risk.columns)
    if collisions:
        raw_rows = raw_rows.rename(columns={column: f"raw_{column}" for column in collisions})
    risk_sorted = risk.sort_values("input_position").reset_index(drop=True)
    joined = pd.concat([risk_sorted, raw_rows], axis=1)
    return CustomerPopulation(
        joined,
        raw_columns=tuple(str(column) for column in raw_rows.columns),
        split=split_name,
        split_fingerprint=fingerprint,
        reason=f"Verified {len(joined):,}-row join to the {split_name} split.",
    )


def benchmark_thresholds(
    settings: Settings, run_id: str | None = None
) -> tuple[dict[str, Any] | None, Path | None]:
    root = Path(settings.artifact_dir) / "evaluation" / run_id if run_id else latest_benchmark_dir(settings)
    path = root / THRESHOLD_ANALYSIS if root else None
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


def local_explanation(
    settings: Settings,
    row_position: int,
    *,
    family: str | None = None,
) -> tuple[dict[str, Any] | None, Path | None, Path | None]:
    """Find the newest full SHAP run holding a matching local explanation.

    The payload and plot are always returned from the same run. When ``family``
    is supplied, a run explaining a different model family is skipped even if it
    happens to contain the same matrix position.
    """

    root = Path(settings.artifact_dir) / "explanations"
    if not root.is_dir():
        return None, None, None
    for run_id in explanation_ids(settings):
        path = root / run_id / "local" / f"row_{int(row_position):05d}.json"
        if not path.is_file():
            continue
        payload = load_json(path)
        if family and str(payload.get("family") or "") not in ("", str(family)):
            continue
        declared_position = payload.get("row_position")
        if declared_position is not None:
            try:
                if int(declared_position) != int(row_position):
                    continue
            except (TypeError, ValueError):
                continue
        plot = path.with_suffix(".png")
        return payload, path, plot if plot.is_file() else None
    return None, None, None


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
    summary, summary_path = risk_summary(settings, prefer_population=True)
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
    "CustomerPopulation",
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
    "THRESHOLD_ANALYSIS",
    "acquisition_attempts",
    "acquisition_candidates",
    "artifact_inventory",
    "default_risk_run_id",
    "artifact_token",
    "attempts_path",
    "benchmark_curves",
    "benchmark_ids",
    "benchmark_metrics",
    "benchmark_thresholds",
    "candidates_path",
    "customer_population",
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
    "latest_population_risk_dir",
    "latest_prepared_dir",
    "latest_risk_dir",
    "latest_simulation_dir",
    "latest_validation_path",
    "load_bytes",
    "load_frame",
    "load_json",
    "local_explanation",
    "model_ids",
    "model_metadata",
    "prepared_summary",
    "preprocessing_schema",
    "risk_drivers",
    "risk_ids",
    "risk_manifest",
    "risk_run_labels",
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
