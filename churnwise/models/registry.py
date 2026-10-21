"""Trusted model-artifact persistence and loading for Phase 6."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from churnwise.config import Settings
from churnwise.models.contracts import MODEL_SCHEMA_VERSION
from churnwise.models.errors import CODE_MODEL_ARTIFACT, ModelingError

MODEL_RUN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,160}$")


def validate_model_run_id(model_run_id: str) -> str:
    if not MODEL_RUN_PATTERN.fullmatch(model_run_id) or ".." in model_run_id:
        raise ModelingError(
            CODE_MODEL_ARTIFACT,
            f"invalid model run id {model_run_id!r}",
            "Use the id printed by the train command.",
        )
    return model_run_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class LoadedModel:
    """A verified model artifact and its accompanying records."""

    model: Any
    metadata: dict[str, Any]
    history: dict[str, Any]
    paths: dict[str, str]

    @property
    def model_run_id(self) -> str:
        return str(self.metadata.get("model_run_id", ""))


class ModelRegistry:
    """Resolve, write and load model artifacts under ``artifact_dir/models``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.artifact_dir = Path(settings.artifact_dir)
        self.report_dir = Path(settings.report_dir)

    def paths_for(self, model_run_id: str) -> dict[str, Path]:
        validate_model_run_id(model_run_id)
        root = self.artifact_dir.resolve()
        directory = (root / "models" / model_run_id / "mlp").resolve()
        try:
            directory.relative_to(root)
        except ValueError as exc:
            raise ModelingError(
                CODE_MODEL_ARTIFACT,
                "model artifact path escaped the artifact directory",
                "Use a model run id printed by this project.",
            ) from exc
        return {
            "model": directory / "model.keras",
            "metadata": directory / "metadata.json",
            "history": directory / "history.json",
            "curves": directory / "learning_curves.png",
            "report_json": self.report_dir / "models" / f"{model_run_id}.json",
            "report_md": self.report_dir / "models" / f"{model_run_id}.md",
        }

    def save_json(self, path: Path, payload: dict[str, Any]) -> str:
        _atomic_write_json(path, payload)
        return str(path)

    def save_model(self, model_run_id: str, model: Any) -> tuple[str, str]:
        """Persist a Keras model atomically and return its path and digest."""
        path = self.paths_for(model_run_id)["model"]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.keras")
        try:
            model.save(temporary)
            os.replace(temporary, path)
        except Exception as exc:  # noqa: BLE001 - surfaced with a stable code
            raise ModelingError(
                CODE_MODEL_ARTIFACT,
                f"model could not be saved: {type(exc).__name__}: {exc}",
                "Check that the artifact directory is writable and has free space.",
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        return str(path), _sha256(path)

    def load(self, model_run_id: str) -> LoadedModel:
        paths = self.paths_for(model_run_id)
        for key in ("model", "metadata", "history"):
            if not paths[key].is_file():
                raise ModelingError(
                    CODE_MODEL_ARTIFACT,
                    f"model artifact {key!r} is missing for run {model_run_id!r}",
                    "Run the train command again.",
                )
        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            history = json.loads(paths["history"].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelingError(
                CODE_MODEL_ARTIFACT,
                f"model metadata could not be read: {type(exc).__name__}: {exc}",
                "Regenerate the model; the artifact may be corrupt.",
            ) from exc
        if not isinstance(metadata, dict) or metadata.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ModelingError(
                CODE_MODEL_ARTIFACT,
                "model metadata has an unsupported schema version",
                "Regenerate the model with this project version.",
            )
        if str(metadata.get("model_run_id", "")) != model_run_id:
            raise ModelingError(
                CODE_MODEL_ARTIFACT,
                "model metadata belongs to a different run id",
                "Reload the run id recorded in metadata; do not rename artifact directories.",
            )
        expected = metadata.get("model_sha256")
        if not expected:
            raise ModelingError(
                CODE_MODEL_ARTIFACT,
                "model metadata has no artifact digest; it cannot be verified",
                "Regenerate the model with this project version.",
            )
        if _sha256(paths["model"]) != expected:
            raise ModelingError(
                CODE_MODEL_ARTIFACT,
                "model digest does not match its metadata",
                "Regenerate the model; the artifact may be corrupt or edited.",
            )
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(
                paths["model"], compile=False, safe_mode=True
            )
        except Exception as exc:  # noqa: BLE001 - surfaced with a stable code
            raise ModelingError(
                CODE_MODEL_ARTIFACT,
                f"model could not be loaded: {type(exc).__name__}: {exc}",
                "Regenerate the model with the same TensorFlow/Keras version.",
            ) from exc
        return LoadedModel(
            model=model,
            metadata=metadata,
            history=history,
            paths={key: str(value) for key, value in paths.items()},
        )

    def latest_model_run_id(self) -> str | None:
        root = Path(self.artifact_dir) / "models"
        if not root.is_dir():
            return None
        candidates = [
            path.name
            for path in root.iterdir()
            if path.is_dir() and (path / "mlp" / "metadata.json").is_file()
        ]
        return sorted(candidates)[-1] if candidates else None
