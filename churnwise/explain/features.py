"""Map transformed one-hot columns back to their source features.

Phase 5 stores an ``output_features`` list whose entries carry the transformed
column ``name``, the original ``source`` and, for categorical levels, the level
``value``. Aggregation sums contributions within a source because SHAP values are
additive: the grouped value is the total effect of that original feature on the
documented model output.

When mapping metadata is missing, the Phase 5 prefix (``numeric__`` or
``categorical__``) is stripped and used as a documented fallback label; the
fallback is recorded so a report never presents a guessed name as a true one.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from churnwise.explain.contracts import Contribution, FeatureGroup
from churnwise.explain.errors import CODE_VALIDATION_FAILED, ExplainError


def _fallback_group(name: str, position: int) -> FeatureGroup:
    for prefix in ("numeric__", "categorical__", "boolean__"):
        if name.startswith(prefix):
            return FeatureGroup(
                source=name[len(prefix) :],
                kind=prefix.rstrip("_"),
                columns=[name],
                labels=[name[len(prefix) :]],
                positions=[position],
            )
    return FeatureGroup(
        source=name,
        kind="unknown",
        columns=[name],
        labels=[name],
        positions=[position],
    )


def build_feature_groups(
    feature_names: Sequence[str],
    output_features: Sequence[dict[str, Any]] | None = None,
) -> list[FeatureGroup]:
    """Group model-width feature names by their Phase 5 source feature."""
    names = [str(name) for name in feature_names]
    if not names:
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            "the model has no transformed feature names",
            "Load the Phase 5 schema from the same run as the model.",
        )
    if len(set(names)) != len(names):
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            "transformed feature names are not unique",
            "Regenerate the Phase 5 schema before mapping contributions.",
        )
    index_of = {name: index for index, name in enumerate(names)}
    mapped: dict[str, FeatureGroup] = {}
    known: set[str] = set()
    for item in output_features or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        source = str(item.get("source") or "").strip()
        if not name or not source or name not in names:
            continue
        role = str(item.get("role") or "unknown")
        kind = "categorical" if role == "categorical" else role
        group = mapped.setdefault(source, FeatureGroup(source=source, kind=kind))
        if name not in group.columns:
            group.columns.append(name)
            group.positions.append(index_of[name])
            label = name.split("__", 1)[-1]
            value = item.get("value")
            if kind == "categorical" and value is not None:
                label = f"{source}={value}"
            group.labels.append(label)
        known.add(name)
    for index, name in enumerate(names):
        if name in known:
            continue
        group = _fallback_group(name, index)
        existing = mapped.get(group.source)
        if existing is None:
            mapped[group.source] = group
        else:
            existing.columns.append(group.columns[0])
            existing.labels.append(group.labels[0])
            existing.positions.append(group.positions[0])
    ordered = [mapped[source] for source in sorted(mapped)]
    covered = {name for group in ordered for name in group.columns}
    if covered != set(names):
        missing = sorted(set(names) - covered)
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            "feature mapping does not cover every transformed column: "
            + ", ".join(missing[:5]),
            "Check the Phase 5 schema output_features list for consistency.",
        )
    return ordered


def group_matrix(
    values: np.ndarray,
    groups: Sequence[FeatureGroup],
) -> tuple[np.ndarray, list[str]]:
    """Sum raw feature contributions into one column per source feature."""
    matrix = np.asarray(values, dtype=np.float64)
    width = sum(len(group.columns) for group in groups)
    if matrix.ndim != 2 or matrix.shape[1] != width:
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            f"contribution matrix has shape {matrix.shape}; expected (n, {width})",
            "Regenerate the explanation with the model's documented feature names.",
        )
    grouped = np.zeros((matrix.shape[0], len(groups)), dtype=np.float64)
    for index, group in enumerate(groups):
        grouped[:, index] = matrix[:, list(group.positions)].sum(axis=1)
    return grouped, [group.source for group in groups]


def _active_level(
    group: FeatureGroup, row: np.ndarray, columns: Sequence[str]
) -> tuple[str | None, float | None]:
    """Return the active categorical level or the numeric value for one row."""
    if group.kind == "categorical":
        for column, value, label in zip(columns, row, group.labels):
            if float(value) != 0.0:
                return label, float(value)
        return None, None
    if len(columns) == 1:
        return None, float(row[0])
    return None, None


def build_contributions(
    row_contributions: Sequence[float],
    groups: Sequence[FeatureGroup],
    feature_row: Sequence[float],
    *,
    tolerance: float,
    top_k: int,
) -> tuple[list[Contribution], list[Contribution], list[Contribution], list[dict[str, Any]]]:
    """Build signed, grouped contributions with raw column detail."""
    grouped, _ = group_matrix(
        np.asarray([row_contributions], dtype=np.float64), groups
    )
    raw = np.asarray(feature_row, dtype=np.float64).reshape(-1)
    contributions_raw = np.asarray(row_contributions, dtype=np.float64).reshape(-1)
    detail: list[dict[str, Any]] = []
    contributions: list[Contribution] = []
    for index, group in enumerate(groups):
        columns = list(group.columns)
        positions = list(group.positions)
        values = raw[positions]
        parts = [float(contributions_raw[position]) for position in positions]
        level, numeric = _active_level(group, values, columns)
        total = float(grouped[0, index])
        if abs(total) < tolerance:
            direction = "neutral"
        elif total > 0:
            direction = "increases_churn"
        else:
            direction = "decreases_churn"
        contributions.append(
            Contribution(
                feature=group.source,
                kind=group.kind,
                contribution=total,
                direction=direction,
                value=numeric,
                active_level=level,
                columns=[
                    {
                        "name": column,
                        "value": float(value),
                        "contribution": float(part),
                    }
                    for column, value, part in zip(columns, values, parts)
                ],
            )
        )
        if len(group.columns) > 1:
            detail.append(
                {
                    "feature": group.source,
                    "aggregated_from": list(group.columns),
                    "rule": "sum",
                }
            )
    ordered = sorted(contributions, key=lambda item: abs(item.contribution), reverse=True)
    positives = [item for item in contributions if item.contribution > float(tolerance)]
    negatives = [item for item in contributions if item.contribution < -float(tolerance)]
    positives.sort(key=lambda item: item.contribution, reverse=True)
    negatives.sort(key=lambda item: item.contribution)
    return ordered, positives[:top_k], negatives[:top_k], detail


def ranked_importance(
    grouped_values: np.ndarray,
    groups: Sequence[FeatureGroup],
    *,
    base_value: float,
) -> list[dict[str, Any]]:
    """Rank source features by mean absolute SHAP value."""
    matrix = np.asarray(grouped_values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(groups):
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            f"grouped values have shape {matrix.shape}; expected (n, {len(groups)})",
            "Regenerate the explanation with the model's documented feature names.",
        )
    mean_abs = np.abs(matrix).mean(axis=0)
    signed = matrix.mean(axis=0)
    positive_share = (matrix > 0).mean(axis=0)
    total = float(mean_abs.sum())
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        share_of_rows = float(positive_share[index])
        if float(mean_abs[index]) <= 1e-12:
            direction = "neutral"
        elif share_of_rows > 0.5:
            direction = "increases_churn"
        elif share_of_rows < 0.5:
            direction = "decreases_churn"
        else:
            direction = "mixed"
        rows.append(
            {
                "feature": group.source,
                "kind": group.kind,
                "mean_abs_shap": float(mean_abs[index]),
                "mean_shap": float(signed[index]),
                "share": float(mean_abs[index] / total) if total > 0 else 0.0,
                "positive_share": share_of_rows,
                "n_columns": len(group.columns),
                "columns": list(group.columns),
                "direction": direction,
            }
        )
    rows.sort(key=lambda item: (-item["mean_abs_shap"], item["feature"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def grouped_feature_values(
    matrix: np.ndarray,
    groups: Sequence[FeatureGroup],
) -> tuple[np.ndarray, list[str]]:
    """Represent each source feature by one numeric value for plots."""
    data = np.asarray(matrix, dtype=np.float64)
    values = np.zeros((data.shape[0], len(groups)), dtype=np.float64)
    for index, group in enumerate(groups):
        block = data[:, list(group.positions)]
        if group.kind == "categorical":
            # Encode the active level index so the beeswarm can colour by level.
            values[:, index] = block.argmax(axis=1)
        else:
            values[:, index] = block[:, 0]
    return values, [group.source for group in groups]


def mapped_columns(
    feature_names: Sequence[str], groups: Sequence[FeatureGroup]
) -> list[str]:
    """Human-readable column labels in model order (for native SHAP plots)."""
    names = [str(name) for name in feature_names]
    labels: list[str | None] = [None] * len(names)
    for group in groups:
        for position, label in zip(group.positions, group.labels):
            if 0 <= int(position) < len(labels):
                labels[int(position)] = (
                    group.source if len(group.columns) == 1 else label
                )
    if any(label is None for label in labels):
        return names
    return [str(label) for label in labels]


def describe_mapping(groups: Iterable[FeatureGroup]) -> dict[str, Any]:
    """Small provenance block describing how aggregation was performed."""
    rows = list(groups)
    return {
        "aggregation": "sum of transformed-column SHAP values per source feature",
        "source_features": len(rows),
        "transformed_columns": int(sum(len(group.columns) for group in rows)),
        "one_hot_groups": [
            {
                "feature": group.source,
                "columns": list(group.columns),
                "labels": list(group.labels),
            }
            for group in rows
            if len(group.columns) > 1
        ],
    }


__all__ = [
    "build_contributions",
    "build_feature_groups",
    "describe_mapping",
    "group_matrix",
    "grouped_feature_values",
    "mapped_columns",
    "ranked_importance",
]
