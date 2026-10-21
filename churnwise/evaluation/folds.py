"""Fold-safe preprocessing for the Phase 7 stability estimate.

The persisted Phase 5 hand-off contains transformed matrices, not raw rows. To
refit preprocessing inside each cross-validation fold without inventing a second
preprocessing system, this module reconstructs the exact Phase 5 base frame from
the recorded raw artifact, target mapping and derived-feature specifications,
then reuses the Phase 5 pipeline builders on each fold's training rows.

Reconstruction is verified against the persisted matrices before it is used:

* labels at every split's recorded row indices must match the Phase 5 labels;
* a pipeline fitted on the full training frame must reproduce the Phase 5
  train/validation/test matrices within ``1e-9``.

If either check fails, fold-safe CV is reported as unavailable for that run —
the benchmark does not silently fall back to a frozen transform.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd

from churnwise.config import Settings
from churnwise.preprocessing.contracts import PreprocessConfig
from churnwise.preprocessing.dataset import PreparedDataset
from churnwise.preprocessing.features import apply_derived
from churnwise.preprocessing.pipeline import (
    build_preprocessor,
    clean_frame,
    drop_unusable_columns,
    plan_features,
    plan_missing_indicators,
    transform_cleaned,
)
from churnwise.validation.loader import DatasetLoadError, load_dataset
from churnwise.validation.targets import normalize_label

TOLERANCE = 1e-9


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")[:120] or "dataset"


@dataclass
class FoldContext:
    """The reconstructed Phase 5 base frame plus the inputs needed to refit."""

    frame: pd.DataFrame
    target_column: str
    config: PreprocessConfig
    roles: Any
    derived_specs: list[dict[str, Any]]
    source: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def derived_names(self) -> set[str]:
        return {str(spec["name"]) for spec in self.derived_specs}


def _resolve_raw_path(settings: Settings, source: dict[str, Any]) -> Path | None:
    raw = str(source.get("artifact_path") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = Path(settings.project_root) / path
    return path if path.is_file() else None


def _roles_from_profile(profile: dict[str, Any]) -> Any:
    roles = dict(profile.get("roles") or {})
    return SimpleNamespace(
        numeric=list(roles.get("numeric") or []),
        categorical=list(roles.get("categorical") or []),
        id=roles.get("id"),
        dropped=list(roles.get("dropped") or []),
    )


def _fit_pipeline(
    train_frame: pd.DataFrame,
    context: "FoldContext",
) -> tuple[Any, Any, list[str]]:
    """Fit the Phase 5 column plan and transformer on one training frame."""
    profile = SimpleNamespace(
        roles=context.roles,
        target=SimpleNamespace(column=context.target_column),
    )
    plan = plan_features(
        train_frame, profile, context.config, context.derived_specs
    )
    cleaned = clean_frame(train_frame, plan)
    drop_unusable_columns(plan, cleaned)
    plan_missing_indicators(
        cleaned,
        plan,
        context.config,
        derived_names=context.derived_names,
    )
    cleaned = clean_frame(train_frame, plan)
    preprocessor = build_preprocessor(plan, context.config)
    preprocessor.fit(cleaned)
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    return preprocessor, plan, names


def _transform(frame: pd.DataFrame, preprocessor: Any, plan: Any) -> np.ndarray:
    return transform_cleaned(preprocessor, clean_frame(frame, plan))


def build_fold_context(
    dataset: PreparedDataset, settings: Settings
) -> tuple[FoldContext | None, str | None]:
    """Reconstruct and verify the Phase 5 base frame for fold-safe CV.

    Returns ``(context, None)`` on success or ``(None, reason)`` when the raw
    artifact, profile or verification evidence is unavailable.
    """
    schema = dataset.schema or {}
    source = dict((dataset.provenance or {}).get("source") or {})
    raw_path = _resolve_raw_path(settings, source)
    if raw_path is None:
        return None, (
            "the Phase 5 provenance does not point to a readable raw artifact, so "
            "fold-specific preprocessing cannot be refitted"
        )

    profile_path = (
        Path(settings.data_dir)
        / "manifests"
        / f"{_safe_name(str(dataset.dataset_id))}.profile.json"
    )
    if not profile_path.is_file():
        return None, (
            f"the archived dataset profile {profile_path.name!r} is missing, so the "
            "original feature roles are unavailable for fold refitting"
        )
    try:
        profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"the archived dataset profile could not be read: {exc}"

    target = dict(schema.get("target") or {})
    target_column = str(target.get("column") or "")
    mapping = {str(key): int(value) for key, value in (target.get("mapping") or {}).items()}
    if not target_column or not mapping:
        return None, "the Phase 5 schema has no target column/mapping for reconstruction"

    config_payload = dict(schema.get("config") or {})
    config_payload.pop("fingerprint", None)
    try:
        config = PreprocessConfig(**config_payload)
    except TypeError as exc:
        return None, f"the stored preprocessing configuration is not readable: {exc}"

    try:
        loaded = load_dataset(raw_path, raw_path.suffix.lstrip(".").lower())
    except (DatasetLoadError, OSError, ValueError) as exc:
        return None, f"the raw artifact could not be reloaded: {type(exc).__name__}: {exc}"

    try:
        raw = loaded.frame
        labels = raw[target_column].map(normalize_label).map(mapping)
        frame = raw.loc[labels.notna()].copy()
        frame[target_column] = labels[labels.notna()].astype("Int64")
        frame = apply_derived(frame, list(schema.get("derived") or []))
        if config.drop_duplicate_rows:
            frame = frame.drop_duplicates(ignore_index=True)
        else:
            frame = frame.reset_index(drop=True)
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"the Phase 5 base frame could not be reconstructed: {exc}"

    spec = dataset.split_spec
    row_indices = {name: list(values) for name, values in spec.row_indices.items()}
    try:
        highest = max(
            (max(values) for values in row_indices.values() if values), default=-1
        )
    except (TypeError, ValueError) as exc:
        return None, f"the recorded split indices are not readable: {exc}"
    if highest >= len(frame):
        return None, (
            f"the reconstructed frame has {len(frame)} rows but the split references "
            f"row {highest}; reconstruction is not faithful"
        )

    expected_labels = {
        "train": dataset.train.y,
        "val": dataset.val.y,
        "test": dataset.test.y,
    }
    for name, expected in expected_labels.items():
        observed = (
            frame.iloc[row_indices[name]][target_column].to_numpy(dtype=np.int8)
        )
        if not np.array_equal(observed, np.asarray(expected).astype(np.int8)):
            return None, (
                f"reconstructed {name} labels do not match the Phase 5 split; "
                "fold-safe CV is disabled rather than using different rows"
            )

    roles = _roles_from_profile(profile_payload)
    context = FoldContext(
        frame=frame,
        target_column=target_column,
        config=config,
        roles=roles,
        derived_specs=list(schema.get("derived") or []),
        source={
            "artifact_path": str(raw_path),
            "profile_path": str(profile_path),
            "dataset_sha256": dataset.split_spec.dataset_sha256,
            "row_count": int(len(frame)),
        },
    )

    # Strong parity check: a full-train refit must reproduce all Phase 5 matrices.
    train_frame = frame.iloc[row_indices["train"]].reset_index(drop=True)
    try:
        preprocessor, plan, _ = _fit_pipeline(train_frame, context)
        for name, expected in (
            ("train", dataset.train.X),
            ("val", dataset.val.X),
            ("test", dataset.test.X),
        ):
            matrix = _transform(
                frame.iloc[row_indices[name]].reset_index(drop=True),
                preprocessor,
                plan,
            )
            reference = np.asarray(expected, dtype=np.float64)
            if matrix.shape != reference.shape or not np.allclose(
                matrix, reference, rtol=0.0, atol=TOLERANCE
            ):
                return None, (
                    f"a full-training refit does not reproduce the Phase 5 {name} "
                    "matrix; fold-safe CV is disabled to avoid reporting metrics from "
                    "different preprocessing"
                )
    except Exception as exc:  # noqa: BLE001 - CV is optional; report why it is off
        return None, f"fold preprocessing could not be verified: {type(exc).__name__}: {exc}"

    context.notes.append(
        "Reconstruction verified: labels match every split and a full-training "
        "refit reproduces the Phase 5 train/validation/test matrices."
    )
    return context, None


def prepare_fold_matrices(
    context: FoldContext,
    train_positions: Sequence[int],
    validation_positions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit preprocessing on one fold's training rows and transform its validation rows."""
    train_positions = list(int(value) for value in train_positions)
    validation_positions = list(int(value) for value in validation_positions)
    if not train_positions or not validation_positions:
        raise ValueError("a fold needs both training and validation rows")
    train_frame = context.frame.iloc[train_positions].reset_index(drop=True)
    validation_frame = context.frame.iloc[validation_positions].reset_index(drop=True)
    preprocessor, plan, names = _fit_pipeline(train_frame, context)
    X_train = _transform(train_frame, preprocessor, plan)
    X_validation = _transform(validation_frame, preprocessor, plan)
    if X_train.shape[1] != X_validation.shape[1]:
        raise ValueError(
            "fold preprocessing produced different widths for train and validation"
        )
    return X_train, X_validation, names


def fold_labels(context: FoldContext, positions: Sequence[int]) -> np.ndarray:
    """Return the 0/1 labels for base-frame positions."""
    values = context.frame.iloc[list(int(value) for value in positions)][
        context.target_column
    ]
    return values.to_numpy(dtype=np.int8)


__all__ = [
    "FoldContext",
    "build_fold_context",
    "fold_labels",
    "prepare_fold_matrices",
]
