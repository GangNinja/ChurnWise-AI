"""Compute dataset-level and customer-level explanations from SHAP values.

The engine is deliberately independent of model loading: it receives an
initialised :class:`ModelExplainer` and the exact Phase 5 matrix. That keeps the
math testable with synthetic models and makes provenance the coordinator's job.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from churnwise.explain.contracts import (
    OUTPUT_SPACE_LOG_ODDS,
    Contribution,
    ExplainConfig,
    FeatureGroup,
    GlobalExplanation,
    LocalExplanation,
    RawExplanation,
)
from churnwise.explain.errors import CODE_VALIDATION_FAILED, ExplainError
from churnwise.explain.explainers import ModelExplainer
from churnwise.explain.features import (
    build_contributions,
    group_matrix,
    ranked_importance,
)


def deterministic_sample(rows: int, size: int, seed: int) -> np.ndarray:
    """Pick sorted, reproducible row positions without replacement."""
    if rows <= 0:
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            "cannot sample from an empty partition",
            "Regenerate the Phase 5 split or explain another split.",
        )
    count = int(min(max(1, int(size)), rows))
    if count >= rows:
        return np.arange(rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    positions = rng.choice(rows, size=count, replace=False)
    return np.sort(np.asarray(positions, dtype=np.int64))


def chunked_explain(
    explainer: ModelExplainer,
    matrix: np.ndarray,
    *,
    config: ExplainConfig,
    progress: Callable[[str], None] | None = None,
) -> RawExplanation:
    """Explain a matrix in bounded chunks and merge the validation evidence."""
    features = np.asarray(matrix, dtype=np.float64)
    if features.ndim != 2 or len(features) == 0:
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            f"explanation matrix must be non-empty and 2-D, got {features.shape}",
            "Provide at least one transformed row.",
        )
    values: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    raws: list[np.ndarray] = []
    error = 0.0
    for start in range(0, len(features), max(1, int(config.chunk_size))):
        block = features[start : start + int(config.chunk_size)]
        if progress is not None:
            progress(
                f"explaining rows {start + 1}-{start + len(block)} of {len(features)}"
            )
        result = explainer.explain(block, tolerance=config.tolerance)
        values.append(result.values)
        bases.append(result.base_values)
        predictions.append(result.predictions)
        if result.predictions_raw is not None:
            raws.append(np.asarray(result.predictions_raw, dtype=np.float64))
        error = max(error, float(result.additivity_error))
    return RawExplanation(
        values=np.concatenate(values, axis=0),
        base_values=np.concatenate(bases, axis=0),
        predictions=np.concatenate(predictions, axis=0),
        predictions_raw=(
            np.concatenate(raws, axis=0) if raws else None
        ),
        additivity_error=error,
        output_space=explainer.output_space,
        explainer_name=explainer.explainer_name,
        notes=list(explainer.notes),
    )


def _summary_sentence(
    item: Contribution,
    *,
    output_space: str,
) -> str:
    value = ""
    if item.active_level:
        value = f" ({item.active_level})"
    elif item.value is not None:
        value = f" (transformed value {item.value:.4g})"
    sign = "+" if item.contribution >= 0 else ""
    return f"{item.feature}{value} {sign}{item.contribution:.3f}"


def local_summary(
    *,
    probability: float,
    predicted_class: int,
    threshold: float,
    positive_label: str | None,
    output_space: str,
    model_output: float,
    top_positive: Sequence[Contribution],
    top_negative: Sequence[Contribution],
) -> str:
    """Build a summary strictly from measured values (no invented reasons)."""
    label = positive_label or "churn"
    class_text = f"{predicted_class}"
    if predicted_class == 1:
        class_text += f" ({label})"
    else:
        class_text += " (no churn)"
    if output_space == OUTPUT_SPACE_LOG_ODDS:
        output_text = (
            f"model log-odds {model_output:.3f} (P(churn = 1) = {probability:.3f})"
        )
    else:
        output_text = f"churn probability {probability:.3f}"
    pieces = [
        f"The model predicts {output_text} at threshold {threshold:.2f}; "
        f"predicted class {class_text}."
    ]
    if top_positive:
        pieces.append(
            "Largest upward contributions: "
            + ", ".join(_summary_sentence(item, output_space=output_space) for item in top_positive[:3])
            + "."
        )
    if top_negative:
        pieces.append(
            "Largest downward contributions: "
            + ", ".join(_summary_sentence(item, output_space=output_space) for item in top_negative[:3])
            + "."
        )
    if not top_positive and not top_negative:
        pieces.append("No feature contributed beyond the reporting tolerance.")
    pieces.append(
        "These are SHAP attributions of the model output, not causal effects; "
        "changing a feature is not guaranteed to change churn by the shown amount."
    )
    return " ".join(pieces)


def summarise_global(
    *,
    run_id: str,
    family: str,
    explainer: ModelExplainer,
    explanation: RawExplanation,
    matrix: np.ndarray,
    groups: Sequence[FeatureGroup],
    sample: dict[str, Any],
    provenance: dict[str, Any],
    config: ExplainConfig,
    duration_seconds: float,
) -> tuple[GlobalExplanation, np.ndarray]:
    """Group model-width values into source features and rank them."""
    grouped, sources = group_matrix(explanation.values, groups)
    importance = ranked_importance(
        grouped, groups, base_value=float(np.mean(explanation.base_values))
    )
    base = float(np.mean(explanation.base_values))
    global_block = GlobalExplanation(
        run_id=run_id,
        family=family,
        output_space=explanation.output_space,
        feature_importance=importance,
        base_value=base,
        sample=dict(sample),
        additivity_error=float(explanation.additivity_error),
        explainer={
            **explainer.metadata(),
            "source_features": len(groups),
            "transformed_columns": int(len(explainer.feature_names)),
            "aggregation": "sum of transformed-column SHAP values per source feature",
        },
        provenance=dict(provenance),
        warnings=list(explainer.warnings),
        duration_seconds=float(duration_seconds),
        config=config.to_dict(),
    )
    if explanation.output_space == OUTPUT_SPACE_LOG_ODDS:
        global_block.warnings.append(
            "Logistic Regression importances are in log-odds; they are not "
            "probability changes."
        )
    return global_block, grouped


def local_explanations(
    *,
    family: str,
    explainer: ModelExplainer,
    explanation: RawExplanation,
    matrix: np.ndarray,
    groups: Sequence[FeatureGroup],
    threshold: float,
    positive_label: str | None,
    top_k: int,
    tolerance: float,
    row_positions: Sequence[int] | None = None,
) -> list[LocalExplanation]:
    """Build signed customer explanations for selected rows."""
    results: list[LocalExplanation] = []
    for index in range(len(matrix)):
        ordered, positives, negatives, mapping_detail = build_contributions(
            explanation.values[index],
            groups,
            matrix[index],
            tolerance=float(tolerance),
            top_k=int(top_k),
        )
        probability = float(explanation.predictions[index])
        model_output = (
            float(explanation.predictions_raw[index])
            if explanation.predictions_raw is not None
            else probability
        )
        predicted = int(probability >= float(threshold))
        summary = local_summary(
            probability=probability,
            predicted_class=predicted,
            threshold=float(threshold),
            positive_label=positive_label,
            output_space=explanation.output_space,
            model_output=model_output,
            top_positive=positives,
            top_negative=negatives,
        )
        warnings: list[str] = []
        if mapping_detail:
            warnings.append(
                "Contributions for one-hot encoded features were aggregated by "
                "summation; raw column detail is included."
            )
        results.append(
            LocalExplanation(
                family=family,
                output_space=explanation.output_space,
                probability=probability,
                predicted_class=predicted,
                threshold=float(threshold),
                positive_label=positive_label,
                base_value=float(explanation.base_values[index]),
                model_output=model_output,
                contributions=ordered,
                top_positive=positives,
                top_negative=negatives,
                summary=summary,
                additivity_error=float(explanation.additivity_error),
                row_position=(
                    int(row_positions[index])
                    if row_positions is not None and index < len(row_positions)
                    else int(index)
                ),
                warnings=warnings,
            )
        )
    return results


def matrix_for_positions(matrix: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    """Select rows while keeping the caller's order and validating bounds."""
    data = np.asarray(matrix, dtype=np.float64)
    values = [int(value) for value in positions]
    if not values:
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            "no rows were selected for explanation",
            "Pass at least one row position or increase --local-examples.",
        )
    bad = [value for value in values if value < 0 or value >= len(data)]
    if bad:
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            f"row position {bad[0]} is outside the matrix (0..{len(data) - 1})",
            "Use a position from the reported split.",
        )
    return data[np.asarray(values, dtype=np.int64)]


__all__ = [
    "chunked_explain",
    "deterministic_sample",
    "local_explanations",
    "local_summary",
    "matrix_for_positions",
    "summarise_global",
]
