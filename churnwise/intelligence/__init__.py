"""Phase 9 customer churn-risk intelligence.

Public entry points:

* :class:`RiskService` — individual/batch/population scoring, ranking,
  filters, driver retrieval and exports.
* :class:`RiskConfig` / :class:`RiskThresholds` — validated tier and scoring
  configuration, separate from the model's classification threshold.
* :class:`RiskAssessment` / :class:`RiskTable` / :class:`RiskRun` — stable
  records for Phase 10 and the dashboard.
* :class:`RiskFilter` — pure filtering over stored assessments.
* :class:`RiskError` — stable ``CW-RISK-*`` failures with remediation.
* :func:`sanitize_csv_value` — spreadsheet formula-injection guard.
"""

from __future__ import annotations

from churnwise.intelligence.contracts import (
    BatchAssessment,
    DEFAULT_TIER_BOUNDARIES,
    RISK_SCHEMA_VERSION,
    RejectedRecord,
    RiskAssessment,
    RiskConfig,
    RiskDriver,
    RiskExport,
    RiskFilter,
    RiskRun,
    RiskTable,
    RiskThresholds,
    TIER_LABELS,
    VALID_EXPLAIN_MODES,
    VALID_POPULATION_SPLITS,
    VALID_SORT_FIELDS,
)
from churnwise.intelligence.errors import (
    CODE_ARTIFACT_FAILED,
    CODE_EXPLANATION_FAILED,
    CODE_EXPORT_FAILED,
    CODE_INVALID_CONFIG,
    CODE_INVALID_INPUT,
    CODE_NO_MODEL,
    CODE_PREDICTION_FAILED,
    RiskError,
)
from churnwise.intelligence.model import LoadedRiskModel, load_selected_model
from churnwise.intelligence.ranking import (
    attach_value_at_risk,
    filter_records,
    paginate,
    rank_records,
    tier_counts,
    value_at_risk,
)
from churnwise.intelligence.reporting import (
    render_batch_summary,
    render_summary,
    render_table,
)
from churnwise.intelligence.service import RiskService, sanitize_csv_value, validate_risk_id

__all__ = [
    "BatchAssessment",
    "CODE_ARTIFACT_FAILED",
    "CODE_EXPLANATION_FAILED",
    "CODE_EXPORT_FAILED",
    "CODE_INVALID_CONFIG",
    "CODE_INVALID_INPUT",
    "CODE_NO_MODEL",
    "CODE_PREDICTION_FAILED",
    "DEFAULT_TIER_BOUNDARIES",
    "LoadedRiskModel",
    "RISK_SCHEMA_VERSION",
    "RejectedRecord",
    "RiskAssessment",
    "RiskConfig",
    "RiskDriver",
    "RiskError",
    "RiskExport",
    "RiskFilter",
    "RiskRun",
    "RiskService",
    "RiskTable",
    "RiskThresholds",
    "TIER_LABELS",
    "VALID_EXPLAIN_MODES",
    "VALID_POPULATION_SPLITS",
    "VALID_SORT_FIELDS",
    "attach_value_at_risk",
    "filter_records",
    "load_selected_model",
    "paginate",
    "rank_records",
    "render_batch_summary",
    "render_summary",
    "render_table",
    "sanitize_csv_value",
    "tier_counts",
    "validate_risk_id",
    "value_at_risk",
]
