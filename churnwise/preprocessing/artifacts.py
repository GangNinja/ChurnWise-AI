"""Trusted, atomic artifact persistence for Phase 5.

Only artifacts produced by this project are loaded. ``joblib`` is never used on
downloaded content; every path is resolved under the configured artifact
directory and the serialized pipeline's digest is verified on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from churnwise.preprocessing.contracts import SCHEMA_VERSION, SplitSpec
from churnwise.preprocessing.errors import CODE_ARTIFACT_FAILED, PreprocessingError

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,160}$")


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(run_id) or ".." in run_id:
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            f"invalid run id {run_id!r}",
            "Use the run id printed by the prepare command.",
        )
    return run_id


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root(artifact_dir: Path | str, run_id: str) -> Path:
    validate_run_id(run_id)
    base = Path(artifact_dir).resolve()
    target = (base / "preprocessors" / run_id).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            "artifact path escaped the artifact directory",
            "Use a run id printed by this project.",
        ) from exc
    return target


def paths_for(artifact_dir: Path | str, run_id: str) -> dict[str, Path]:
    validate_run_id(run_id)
    base = Path(artifact_dir)
    directory = _root(base, run_id)
    return {
        "split": base / "splits" / f"{run_id}.json",
        "pipeline": directory / "pipeline.joblib",
        "schema": directory / "schema.json",
        "arrays": base / "prepared" / run_id / "matrices.npz",
        "summary": base / "prepared" / run_id / "summary.json",
    }


def save_split_spec(artifact_dir: Path | str, spec: SplitSpec) -> str:
    path = paths_for(artifact_dir, spec.run_id)["split"]
    _atomic_write_json(path, spec.to_dict())
    return str(path)


def load_split_spec(artifact_dir: Path | str, run_id: str) -> SplitSpec:
    path = paths_for(artifact_dir, run_id)["split"]
    if not path.is_file():
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            f"split spec is missing for run {run_id!r}",
            "Run the prepare command again.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            f"split spec could not be read: {type(exc).__name__}: {exc}",
            "Regenerate the run; the artifact may be corrupt.",
        ) from exc
    if not isinstance(payload, dict):
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            "split spec is not a JSON object",
            "Regenerate the run; the artifact may be corrupt.",
        )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            f"split schema version {payload.get('schema_version')!r} is not supported",
            "Regenerate the split with this project version.",
        )
    return SplitSpec.from_dict(payload)


def save_preprocessor(
    artifact_dir: Path | str,
    run_id: str,
    preprocessor: Any,
    schema: dict[str, Any],
) -> tuple[str, str]:
    """Persist the fitted pipeline and schema; return their paths and hash."""
    targets = paths_for(artifact_dir, run_id)
    pipeline_path = targets["pipeline"]
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pipeline_path.with_name(f".{pipeline_path.name}.{os.getpid()}.tmp")
    try:
        joblib.dump(preprocessor, temporary)
        os.replace(temporary, pipeline_path)
    finally:
        temporary.unlink(missing_ok=True)
    preprocessor_hash = _sha256(pipeline_path)
    schema = {
        **schema,
        "preprocessor_hash": preprocessor_hash,
        "preprocessor_path": str(pipeline_path),
    }
    _atomic_write_json(targets["schema"], schema)
    return str(pipeline_path), preprocessor_hash


def load_preprocessor(
    artifact_dir: Path | str, run_id: str
) -> tuple[Any, dict[str, Any]]:
    targets = paths_for(artifact_dir, run_id)
    pipeline_path = targets["pipeline"]
    schema_path = targets["schema"]
    if not pipeline_path.is_file() or not schema_path.is_file():
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            f"preprocessor artifacts are missing for run {run_id!r}",
            "Run the prepare command again.",
        )
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            f"preprocessor schema could not be read: {type(exc).__name__}: {exc}",
            "Regenerate the run; the artifact may be corrupt.",
        ) from exc
    if not isinstance(schema, dict):
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            "preprocessor schema is not a JSON object",
            "Regenerate the run; the artifact may be corrupt.",
        )
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            f"preprocessor schema version {schema.get('schema_version')!r} is not supported",
            "Regenerate the preprocessor with this project version.",
        )
    expected = schema.get("preprocessor_hash")
    if not expected:
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            "preprocessor schema has no digest; the artifact cannot be verified",
            "Regenerate the run with this project version.",
        )
    if _sha256(pipeline_path) != expected:
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            "preprocessor digest does not match its schema",
            "Regenerate the preprocessor; the artifact may be corrupt or edited.",
        )
    try:
        preprocessor = joblib.load(pipeline_path)
    except Exception as exc:  # noqa: BLE001 - surfaced with a stable code
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            f"preprocessor could not be loaded: {type(exc).__name__}: {exc}",
            "Regenerate the preprocessor with the same project version.",
        ) from exc
    return preprocessor, schema


def save_arrays(
    artifact_dir: Path | str,
    run_id: str,
    arrays: dict[str, np.ndarray],
) -> str:
    path = paths_for(artifact_dir, run_id)["arrays"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def load_arrays(artifact_dir: Path | str, run_id: str) -> dict[str, np.ndarray]:
    path = paths_for(artifact_dir, run_id)["arrays"]
    if not path.is_file():
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            f"prepared matrices are missing for run {run_id!r}",
            "Run the prepare command again.",
        )
    try:
        with np.load(path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    except (OSError, ValueError) as exc:
        raise PreprocessingError(
            CODE_ARTIFACT_FAILED,
            f"prepared matrices could not be read: {type(exc).__name__}: {exc}",
            "Regenerate the run; do not load matrices from untrusted sources.",
        ) from exc


def save_summary(artifact_dir: Path | str, run_id: str, summary: dict[str, Any]) -> str:
    path = paths_for(artifact_dir, run_id)["summary"]
    _atomic_write_json(path, summary)
    return str(path)
