"""Table formatting and download widgets for dashboard artifacts."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from churnwise.dashboard.analytics import (
    dict_to_json_bytes,
    frame_to_csv_bytes,
    frame_to_json_bytes,
)

RISK_COLUMNS = [
    "record_reference",
    "churn_probability",
    "risk_tier",
    "rank",
    "predicted_label",
    "value_at_risk",
    "explanation_status",
]


def risk_view(frame: pd.DataFrame, *, limit: int = 200) -> pd.DataFrame:
    columns = [column for column in RISK_COLUMNS if column in frame.columns]
    view = frame[columns].head(limit).copy()
    if "churn_probability" in view:
        view["churn_probability"] = view["churn_probability"].astype(float).round(4)
    if "value_at_risk" in view:
        view["value_at_risk"] = pd.to_numeric(view["value_at_risk"], errors="coerce").round(2)
    if "rank" in view:
        view["rank"] = pd.to_numeric(view["rank"], errors="coerce").astype("Int64")
    return view


def download_pair(
    frame: pd.DataFrame,
    *,
    prefix: str,
    key: str,
    json_payload: Mapping[str, Any] | None = None,
) -> None:
    """Two download buttons with formula-safe CSV and JSON bytes."""

    cols = st.columns(2)
    with cols[0]:
        st.download_button(
            "Download CSV",
            data=frame_to_csv_bytes(frame),
            file_name=f"{prefix}.csv",
            mime="text/csv",
            key=f"{key}_csv",
        )
    with cols[1]:
        if json_payload is None:
            data = frame_to_json_bytes(frame)
        elif isinstance(json_payload, (list, tuple)):
            data = json.dumps(json_payload, indent=2, default=str).encode("utf-8")
        else:
            data = dict_to_json_bytes(json_payload)
        st.download_button(
            "Download JSON",
            data=data,
            file_name=f"{prefix}.json",
            mime="application/json",
            key=f"{key}_json",
        )


def checks_frame(checks: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "ID": check.get("id"),
            "Check": str(check.get("name") or "").replace("_", " "),
            "Severity": check.get("severity"),
            "Passed": "yes" if check.get("passed") else "no",
            "Detail": check.get("detail"),
        }
        for check in checks
    ]
    return pd.DataFrame(rows)


def metrics_frame(comparison: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in comparison:
        rows.append(
            {
                "Family": item.get("label") or item.get("family"),
                "Threshold": item.get("threshold"),
                "Accuracy": item.get("accuracy"),
                "Precision": item.get("precision"),
                "Recall": item.get("recall"),
                "F1": item.get("f1"),
                "ROC-AUC": item.get("roc_auc"),
                "PR-AUC": item.get("pr_auc"),
                "Brier": item.get("brier"),
            }
        )
    frame = pd.DataFrame(rows)
    for column in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier", "Threshold"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").round(4)
    return frame


def sensitivity_frame(series: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in series:
        for point in item.get("points") or []:
            rows.append(
                {
                    "Assumption": item.get("dimension"),
                    "Value": point.get("value"),
                    "Selected": point.get("selected_count"),
                    "Spend": point.get("allocated_spend"),
                    "Expected prevented churn": point.get("expected_prevented_churn"),
                    "Cost per prevented": point.get("cost_per_expected_prevented_churn"),
                    "Baseline": point.get("is_baseline"),
                }
            )
    return pd.DataFrame(rows)


def local_contributions_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = payload.get("contributions") or []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    keep = [column for column in ("feature", "active_level", "kind", "contribution", "direction", "value") if column in frame]
    frame = frame[keep].copy()
    if "contribution" in frame:
        frame["contribution"] = pd.to_numeric(frame["contribution"], errors="coerce").round(4)
    if "value" in frame:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce").round(4)
    return frame.sort_values("contribution", key=lambda values: values.abs(), ascending=False)


def selected_customers_frame(rows: Sequence[Mapping[str, Any]], limit: int = 100) -> pd.DataFrame:
    frame = pd.DataFrame(rows or []).head(limit)
    if frame.empty:
        return frame
    keep = [column for column in ("record_reference", "churn_probability", "risk_tier", "intervention_cost", "expected_prevented_churn") if column in frame]
    frame = frame[keep].copy()
    if "churn_probability" in frame:
        frame["churn_probability"] = pd.to_numeric(frame["churn_probability"], errors="coerce").round(4)
    return frame


__all__ = [
    "RISK_COLUMNS",
    "checks_frame",
    "download_pair",
    "local_contributions_frame",
    "metrics_frame",
    "risk_view",
    "selected_customers_frame",
    "sensitivity_frame",
]
