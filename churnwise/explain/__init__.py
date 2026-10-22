"""Phase 8 explainability: SHAP-based global and customer-level explanations.

Public entry points:

* :class:`ExplainCoordinator` — load a verified Phase 6/7 model, explain a
  validation sample, persist global/local artifacts and provenance.
* :class:`ExplainConfig` — sampling, background and performance bounds.
* :func:`build_explainer` — model-specific SHAP adapter with additivity checks.
* :func:`build_feature_groups` — group one-hot columns back to source features.
* :class:`GlobalExplanation` / :class:`LocalExplanation` — stable result records.
* :class:`ExplainError` — ``CW-XAI-*`` failures with remediation.
"""

from __future__ import annotations

from churnwise.explain.contracts import (
    EXPLAIN_SCHEMA_VERSION,
    OUTPUT_SPACE_LOG_ODDS,
    OUTPUT_SPACE_PROBABILITY,
    SUPPORTED_FAMILIES,
    Contribution,
    ExplainConfig,
    ExplainResult,
    FeatureGroup,
    GlobalExplanation,
    LocalExplanation,
    RawExplanation,
)
from churnwise.explain.coordinator import (
    ExplainCoordinator,
    LoadedExplanationModel,
    validate_explain_id,
)
from churnwise.explain.engine import (
    chunked_explain,
    deterministic_sample,
    local_explanations,
    local_summary,
    matrix_for_positions,
    summarise_global,
)
from churnwise.explain.errors import (
    CODE_ARTIFACT_FAILED,
    CODE_EXPLAIN_FAILED,
    CODE_INVALID_CONFIG,
    CODE_NO_MODEL,
    CODE_UNSUPPORTED_MODEL,
    CODE_VALIDATION_FAILED,
    ExplainError,
)
from churnwise.explain.explainers import ModelExplainer, build_explainer
from churnwise.explain.features import (
    build_contributions,
    build_feature_groups,
    describe_mapping,
    group_matrix,
    ranked_importance,
)
from churnwise.explain.plots import (
    plot_beeswarm,
    plot_dependence,
    plot_importance,
    plot_waterfall,
)

__all__ = [
    "CODE_ARTIFACT_FAILED",
    "CODE_EXPLAIN_FAILED",
    "CODE_INVALID_CONFIG",
    "CODE_NO_MODEL",
    "CODE_UNSUPPORTED_MODEL",
    "CODE_VALIDATION_FAILED",
    "EXPLAIN_SCHEMA_VERSION",
    "OUTPUT_SPACE_LOG_ODDS",
    "OUTPUT_SPACE_PROBABILITY",
    "SUPPORTED_FAMILIES",
    "Contribution",
    "ExplainConfig",
    "ExplainCoordinator",
    "ExplainError",
    "ExplainResult",
    "FeatureGroup",
    "GlobalExplanation",
    "LoadedExplanationModel",
    "LocalExplanation",
    "ModelExplainer",
    "RawExplanation",
    "build_contributions",
    "build_explainer",
    "build_feature_groups",
    "chunked_explain",
    "describe_mapping",
    "deterministic_sample",
    "group_matrix",
    "local_explanations",
    "local_summary",
    "matrix_for_positions",
    "plot_beeswarm",
    "plot_dependence",
    "plot_importance",
    "plot_waterfall",
    "ranked_importance",
    "summarise_global",
    "validate_explain_id",
]
