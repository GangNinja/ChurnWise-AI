"""Model-specific SHAP adapters with explicit output spaces.

Selection:

============  ==========================  ==================  =====================
family        primary explainer           output space        fallback
============  ==========================  ==================  =====================
random_forest ``shap.TreeExplainer``      ``probability``     none (tree-specific)
logistic      ``shap.LinearExplainer``    ``log_odds``        none (linear-specific)
mlp           ``shap.DeepExplainer``      ``probability``     permutation explainer
============  ==========================  ==================  =====================

The MLP fallback is model-agnostic and expensive, so it is bounded by
``ExplainConfig.permutation_max_evals`` and the caller's sample/background caps.
No adapter returns placeholder values: a failure raises :class:`ExplainError`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from churnwise.explain.contracts import (
    OUTPUT_SPACE_LOG_ODDS,
    OUTPUT_SPACE_PROBABILITY,
    ExplainConfig,
    RawExplanation,
)
from churnwise.explain.errors import (
    CODE_EXPLAIN_FAILED,
    CODE_UNSUPPORTED_MODEL,
    CODE_VALIDATION_FAILED,
    ExplainError,
)


def _import_shap() -> Any:
    """Import SHAP once with an actionable error when the dependency is absent."""
    try:
        import shap
    except ImportError as exc:
        raise ExplainError(
            CODE_UNSUPPORTED_MODEL,
            "SHAP is not installed in this environment",
            "Install the declared dependencies with `pip install -r requirements.txt`.",
        ) from exc
    return shap


def _positive_class(values: Any, base: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract the churn-class slice from binary/multi-output SHAP values.

    Older SHAP releases return one array per class (and a list of expected
    values) from ``shap_values``; newer releases return a single 3-D array. Both
    shapes are normalised here before the class slice is selected.
    """
    if isinstance(values, (list, tuple)):
        if len(values) == 1:
            values = values[0]
            if isinstance(base, (list, tuple)) and len(base) == 1:
                base = base[0]
        elif len(values) == 2:
            values = values[1]
            if isinstance(base, (list, tuple)) and len(base) == 2:
                base = base[1]
    array = np.asarray(values, dtype=np.float64)
    bases = np.asarray(base, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[-1] != 2:
            raise ExplainError(
                CODE_UNSUPPORTED_MODEL,
                f"explainer returned {array.shape[-1]} output classes; expected 2",
                "Use a binary churn model with the positive class at index 1.",
            )
        array = array[..., 1]
        if bases.ndim == 2 and bases.shape[-1] == 2:
            bases = bases[:, 1]
        elif bases.ndim == 1 and bases.size == 2:
            bases = bases[1:]
    if array.ndim != 2:
        raise ExplainError(
            CODE_EXPLAIN_FAILED,
            f"explainer returned values with shape {array.shape}; expected (n, features)",
            "Check the explainer and model output shape.",
        )
    if bases.ndim == 0:
        bases = np.repeat(float(bases), array.shape[0])
    bases = bases.reshape(-1)
    if bases.size == 1:
        bases = np.repeat(bases[0], array.shape[0])
    if bases.size != array.shape[0]:
        raise ExplainError(
            CODE_EXPLAIN_FAILED,
            f"explainer returned {bases.size} base values for {array.shape[0]} rows",
            "Reload the model artifact and retry the explanation.",
        )
    return array, bases


def _additivity_error(
    values: np.ndarray, bases: np.ndarray, target: np.ndarray
) -> float:
    reconstructed = bases + values.sum(axis=1)
    return float(np.max(np.abs(reconstructed - target)))


def _require_finite(label: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            f"{label} contain non-finite values",
            "Do not report these values. Reload the verified model artifact and "
            "retry the explanation.",
        )


@dataclass
class ModelExplainer:
    """A ready-to-use explainer bound to one model and feature space."""

    family: str
    explainer_name: str
    output_space: str
    feature_names: list[str]
    background_size: int
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    selected_by_fallback: bool = False
    fallback_reason: str | None = None
    _draw: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None
    _predict_proba: Callable[[np.ndarray], np.ndarray] | None = None
    _decision_function: Callable[[np.ndarray], np.ndarray] | None = None
    _raw_values: Callable[[np.ndarray], Any] | None = None

    def explain(self, matrix: np.ndarray, *, tolerance: float) -> RawExplanation:
        """Compute SHAP values and verify additivity for the documented space."""
        if self._draw is None or self._predict_proba is None:
            raise ExplainError(
                CODE_EXPLAIN_FAILED,
                "the explainer was not initialised",
                "Build the explainer with build_explainer().",
            )
        try:
            tolerance_value = float(tolerance)
        except (TypeError, ValueError) as exc:
            raise ExplainError(
                CODE_VALIDATION_FAILED,
                f"additivity tolerance {tolerance!r} is not numeric",
                "Use a finite tolerance in (0, 1].",
            ) from exc
        if not math.isfinite(tolerance_value) or not 0.0 < tolerance_value <= 1.0:
            raise ExplainError(
                CODE_VALIDATION_FAILED,
                f"additivity tolerance {tolerance_value!r} is not in (0, 1]",
                "Use a finite tolerance in (0, 1].",
            )
        features = np.asarray(matrix, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ExplainError(
                CODE_VALIDATION_FAILED,
                f"features have shape {features.shape}; expected (n, {len(self.feature_names)})",
                "Use the Phase 5 transformed matrix from the model's run.",
            )
        if not np.isfinite(features).all():
            raise ExplainError(
                CODE_VALIDATION_FAILED,
                "features contain non-finite values",
                "Run raw records through the Phase 5 pipeline before explaining.",
            )
        started = time.perf_counter()
        try:
            values, bases = self._draw(features)
        except ExplainError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced with an actionable code
            raise ExplainError(
                CODE_EXPLAIN_FAILED,
                f"{self.explainer_name} could not explain this batch: "
                f"{type(exc).__name__}: {exc}",
                "Reduce the sample/background size or use another supported model family.",
            ) from exc
        values = np.asarray(values, dtype=np.float64)
        bases = np.asarray(bases, dtype=np.float64).reshape(-1)
        if values.ndim != 2 or values.shape != (len(features), len(self.feature_names)):
            raise ExplainError(
                CODE_VALIDATION_FAILED,
                f"SHAP values have shape {values.shape}; expected "
                f"({len(features)}, {len(self.feature_names)})",
                "Reload the verified model artifact and retry.",
            )
        if bases.size != len(features):
            raise ExplainError(
                CODE_VALIDATION_FAILED,
                f"SHAP base values have {bases.size} entries for {len(features)} rows",
                "Reload the verified model artifact and retry.",
            )
        try:
            probabilities = np.asarray(
                self._predict_proba(features), dtype=np.float64
            ).reshape(-1)
            if self.output_space == OUTPUT_SPACE_LOG_ODDS and self._decision_function:
                raw = np.asarray(
                    self._decision_function(features), dtype=np.float64
                ).reshape(-1)
            else:
                raw = probabilities
        except ExplainError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced with an actionable code
            raise ExplainError(
                CODE_EXPLAIN_FAILED,
                f"prediction failed while explaining {self.family}: "
                f"{type(exc).__name__}: {exc}",
                "Verify the model artifact and transformed feature matrix, then retry.",
            ) from exc
        if probabilities.size != len(features) or raw.size != len(features):
            raise ExplainError(
                CODE_EXPLAIN_FAILED,
                f"prediction returned {probabilities.size} probabilities for "
                f"{len(features)} rows",
                "Reload the verified model artifact and retry.",
            )
        _require_finite("SHAP values", np.asarray(values, dtype=np.float64))
        _require_finite("SHAP base values", np.asarray(bases, dtype=np.float64))
        _require_finite("predicted probabilities", probabilities)
        if (probabilities < -1e-6).any() or (probabilities > 1.0 + 1e-6).any():
            raise ExplainError(
                CODE_VALIDATION_FAILED,
                "predicted probabilities fall outside [0, 1]",
                "Reload the verified model artifact; do not report these values.",
            )
        if self.output_space == OUTPUT_SPACE_LOG_ODDS:
            _require_finite("model decision values", raw)
        if self.output_space == OUTPUT_SPACE_LOG_ODDS:
            error = _additivity_error(values, bases, raw)
            target = raw
        else:
            target = probabilities
            error = _additivity_error(values, bases, probabilities)
        if error > tolerance_value:
            raise ExplainError(
                CODE_VALIDATION_FAILED,
                f"SHAP additivity failed for {self.family}: max error {error:.3g} "
                f"exceeds tolerance {tolerance_value:.3g}",
                "Do not report these values. Check the model output space and "
                "reload the verified model artifact.",
                details={"additivity_error": error, "tolerance": tolerance_value},
            )
        notes = list(self.notes)
        notes.append(
            f"{self.explainer_name} produced values in {self.output_space} space; "
            f"max additivity error {error:.3g} over {len(features)} rows."
        )
        return RawExplanation(
            values=values,
            base_values=bases,
            predictions=probabilities,
            predictions_raw=target if self.output_space == OUTPUT_SPACE_LOG_ODDS else None,
            additivity_error=error,
            output_space=self.output_space,
            explainer_name=self.explainer_name,
            notes=notes,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.explainer_name,
            "output_space": self.output_space,
            "background_rows": int(self.background_size),
            "feature_count": len(self.feature_names),
            "selected_by_fallback": bool(self.selected_by_fallback),
            "fallback_reason": self.fallback_reason,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }


def _tree_explainer(
    estimator: Any,
    *,
    feature_names: Sequence[str],
    background_size: int,
    predict_proba: Callable[[np.ndarray], np.ndarray],
) -> ModelExplainer:
    shap = _import_shap()
    try:
        explainer = shap.TreeExplainer(
            estimator, feature_perturbation="tree_path_dependent"
        )
    except Exception as exc:  # noqa: BLE001 - surfaced with an actionable code
        raise ExplainError(
            CODE_UNSUPPORTED_MODEL,
            f"shap.TreeExplainer could not initialise: {type(exc).__name__}: {exc}",
            "Use a supported tree ensemble (e.g. RandomForestClassifier) or explain "
            "another model family.",
        ) from exc

    def draw(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            explanation = explainer(matrix, check_additivity=False)
            values = explanation.values
            bases = explanation.base_values
        except Exception:  # older/newer API surface
            values = explainer.shap_values(matrix)
            bases = explainer.expected_value
        return _positive_class(values, bases)

    return ModelExplainer(
        family="random_forest",
        explainer_name="shap.TreeExplainer",
        output_space=OUTPUT_SPACE_PROBABILITY,
        feature_names=[str(name) for name in feature_names],
        background_size=int(background_size),
        notes=[
            "Tree SHAP is exact for tree ensembles; values sum to P(churn = 1)."
        ],
        _draw=draw,
        _predict_proba=predict_proba,
    )


def _linear_explainer(
    estimator: Any,
    background: np.ndarray,
    *,
    feature_names: Sequence[str],
    predict_proba: Callable[[np.ndarray], np.ndarray],
) -> ModelExplainer:
    shap = _import_shap()
    try:
        explainer = shap.LinearExplainer(
            estimator,
            background,
            feature_names=[str(name) for name in feature_names],
        )
    except Exception as exc:  # noqa: BLE001 - surfaced with an actionable code
        raise ExplainError(
            CODE_UNSUPPORTED_MODEL,
            f"shap.LinearExplainer could not initialise: {type(exc).__name__}: {exc}",
            "Use a supported linear classifier with a transformed background matrix, "
            "or explain another model family.",
        ) from exc

    def draw(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        explanation = explainer(matrix)
        return _positive_class(explanation.values, explanation.base_values)

    def decision(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(estimator.decision_function(matrix), dtype=np.float64)

    return ModelExplainer(
        family="logistic_regression",
        explainer_name="shap.LinearExplainer",
        output_space=OUTPUT_SPACE_LOG_ODDS,
        feature_names=[str(name) for name in feature_names],
        background_size=int(len(background)),
        notes=[
            "Linear SHAP is exact for the logistic decision function; values sum to "
            "log-odds, not probability. Convert with sigmoid to compare probabilities."
        ],
        _draw=draw,
        _predict_proba=predict_proba,
        _decision_function=decision,
    )


def _mlp_explainer(
    model: Any,
    background: np.ndarray,
    *,
    feature_names: Sequence[str],
    predict_proba: Callable[[np.ndarray], np.ndarray],
    config: ExplainConfig,
) -> ModelExplainer:
    shap = _import_shap()
    try:
        explainer = shap.DeepExplainer(model, background)

        def draw_deep(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            values = explainer.shap_values(matrix)
            bases = explainer.expected_value
            if isinstance(values, np.ndarray):
                array = np.asarray(values, dtype=np.float64)
                if array.ndim == 3 and array.shape[-1] == 1:
                    values = array[..., 0]
            return _positive_class(values, bases)

        # Probe one row so a broken DeepExplainer is caught before a large batch.
        draw_deep(np.asarray(background[:1], dtype=np.float64))
        return ModelExplainer(
            family="mlp",
            explainer_name="shap.DeepExplainer",
            output_space=OUTPUT_SPACE_PROBABILITY,
            feature_names=[str(name) for name in feature_names],
            background_size=int(len(background)),
            notes=[
                "Deep SHAP on the sigmoid output; values sum to P(churn = 1)."
            ],
            _draw=draw_deep,
            _predict_proba=predict_proba,
        )
    except Exception as exc:  # noqa: BLE001 - documented fallback path
        reason = f"shap.DeepExplainer: {type(exc).__name__}: {exc}"
        notes = [
            "DeepExplainer failed for this environment/model; using the bounded "
            "model-agnostic permutation explainer instead.",
            f"DeepExplainer failure: {reason}",
        ]
        try:
            perm = shap.Explainer(
                predict_proba,
                background,
                algorithm="permutation",
                feature_names=[str(name) for name in feature_names],
            )
            probe = perm(
                np.asarray(background[:1], dtype=np.float64),
                max_evals=int(config.permutation_max_evals),
            )
            del probe
        except Exception as perm_exc:  # noqa: BLE001 - no fabrication allowed
            raise ExplainError(
                CODE_UNSUPPORTED_MODEL,
                "neither DeepExplainer nor the permutation fallback could explain "
                f"the MLP: {type(perm_exc).__name__}: {perm_exc}",
                "Check the TensorFlow/SHAP versions or explain another model family.",
                details={"deep_failure": reason},
            ) from perm_exc

        def draw_perm(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            explanation = perm(
                matrix,
                max_evals=int(config.permutation_max_evals),
            )
            return _positive_class(explanation.values, explanation.base_values)

        return ModelExplainer(
            family="mlp",
            explainer_name="shap.Explainer(permutation)",
            output_space=OUTPUT_SPACE_PROBABILITY,
            feature_names=[str(name) for name in feature_names],
            background_size=int(len(background)),
            notes=notes,
            warnings=notes[:1],
            selected_by_fallback=True,
            fallback_reason=reason,
            _draw=draw_perm,
            _predict_proba=predict_proba,
        )


def build_explainer(
    family: str,
    model: Any,
    predict_proba: Callable[[np.ndarray], np.ndarray],
    *,
    background: np.ndarray,
    feature_names: Sequence[str],
    config: ExplainConfig | None = None,
) -> ModelExplainer:
    """Initialise the appropriate explainer for a supported model family."""
    config = config or ExplainConfig()
    config.validate_or_raise()
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
            "Regenerate the Phase 5 schema; duplicate names cannot be mapped "
            "back to source features safely.",
        )
    background_matrix = np.asarray(background, dtype=np.float64)
    if background_matrix.ndim != 2 or background_matrix.shape[1] != len(names):
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            f"background has shape {background_matrix.shape}; expected (n, {len(names)})",
            "Use the Phase 5 training matrix for the background sample.",
        )
    if not np.isfinite(background_matrix).all():
        raise ExplainError(
            CODE_VALIDATION_FAILED,
            "background contains non-finite values",
            "Regenerate the Phase 5 run before explaining.",
        )
    if family == "random_forest":
        return _tree_explainer(
            model,
            feature_names=names,
            background_size=len(background_matrix),
            predict_proba=predict_proba,
        )
    if family == "logistic_regression":
        if len(background_matrix) < 2:
            raise ExplainError(
                CODE_VALIDATION_FAILED,
                "Logistic SHAP needs at least two background rows",
                "Increase --background-size.",
            )
        return _linear_explainer(
            model,
            background_matrix,
            feature_names=names,
            predict_proba=predict_proba,
        )
    if family == "mlp":
        return _mlp_explainer(
            model,
            background_matrix,
            feature_names=names,
            predict_proba=predict_proba,
            config=config,
        )
    raise ExplainError(
        CODE_UNSUPPORTED_MODEL,
        f"unsupported model family {family!r}",
        "Use logistic_regression, random_forest or mlp.",
    )


__all__ = ["ModelExplainer", "build_explainer"]
