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

CUSTOMER_EXTRA_COLUMNS = {
    "account_length": "Account length",
    "area_code": "Area code",
    "number_customer_service_calls": "Service calls",
    "total_day_minutes": "Day minutes",
    "total_day_charge": "Day charge",
    "total_eve_minutes": "Evening minutes",
    "total_night_minutes": "Night minutes",
    "total_intl_minutes": "International minutes",
    "international_plan": "International plan",
    "voice_mail_plan": "Voice mail plan",
}


def risk_view(frame: pd.DataFrame, *, limit: int = 200) -> pd.DataFrame:
    columns = [column for column in RISK_COLUMNS if column in frame.columns]
    view = frame[columns].head(limit).copy()
    if "churn_probability" in view:
        view["churn_probability"] = view["churn_probability"].astype(float).round(4)
    if "value_at_risk" in view:
        view["value_at_risk"] = pd.to_numeric(view["value_at_risk"], errors="coerce").round(2)
    if "rank" in view:
        view["rank"] = pd.to_numeric(view["rank"], errors="coerce").astype("Int64")
    view = view.rename(
        columns={
            "record_reference": "Customer",
            "churn_probability": "Churn probability",
            "risk_tier": "Risk tier",
            "rank": "Rank",
            "predicted_label": "Predicted churn",
            "value_at_risk": "Value at risk",
            "explanation_status": "Explanation",
        }
    )
    for column in ("Value at risk", "Explanation"):
        if column in view:
            view[column] = view[column].where(view[column].notna(), "—")
    return view


def customer_view(
    frame: pd.DataFrame,
    *,
    limit: int = 25,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Compact customer table for exploration, not the full export schema."""

    base = ["record_reference", "churn_probability", "risk_tier", "predicted_label"]
    columns = base + [column for column in extra_columns if column in frame.columns]
    view = frame[[column for column in columns if column in frame.columns]].head(limit).copy()
    if "churn_probability" in view:
        view["churn_probability"] = pd.to_numeric(view["churn_probability"], errors="coerce").round(4)
    for column in ("account_length", "area_code", "number_customer_service_calls"):
        if column in view:
            view[column] = pd.to_numeric(view[column], errors="coerce")
    for column in ("international_plan", "voice_mail_plan"):
        if column in view:
            view[column] = view[column].map(lambda value: "—" if pd.isna(value) else ("Yes" if float(value) == 1 else "No"))
    if "risk_tier" in view:
        view["risk_tier"] = view["risk_tier"].astype(str).str.title()
    if "predicted_label" in view:
        view["predicted_label"] = view["predicted_label"].map(
            lambda value: "Churn" if str(value).lower() in {"yes", "1", "true", "churn"} else "Retained"
        )
    labels = {
        "record_reference": "Customer",
        "churn_probability": "Churn probability",
        "risk_tier": "Risk tier",
        "predicted_label": "Predicted class",
        **CUSTOMER_EXTRA_COLUMNS,
    }
    return view.rename(columns={key: value for key, value in labels.items() if key in view.columns})


def risk_column_config() -> dict[str, Any]:
    """Column presentation shared by customer tables."""

    return {
        "Churn probability": st.column_config.ProgressColumn(
            "Churn probability",
            help="Stored model estimate; click or sort like any table column.",
            format="%.2f",
            min_value=0.0,
            max_value=1.0,
        ),
        "Customer": st.column_config.TextColumn("Customer", width="medium"),
        "Risk tier": st.column_config.TextColumn("Risk tier", width="small"),
        "Predicted class": st.column_config.TextColumn("Predicted class", width="medium"),
    }


def profile_items(frame: pd.DataFrame, raw_columns: Sequence[str]) -> list[tuple[str, str]]:
    """Pick business-readable source attributes for the customer profile."""

    if frame is None or frame.empty:
        return []
    row = frame.iloc[0]
    priority = [
        ("account_length", "Account length"),
        ("area_code", "Area code"),
        ("international_plan", "International plan"),
        ("voice_mail_plan", "Voice mail plan"),
        ("number_customer_service_calls", "Service calls"),
        ("total_day_minutes", "Day minutes"),
        ("total_day_charge", "Day charge"),
        ("total_eve_minutes", "Evening minutes"),
        ("total_night_minutes", "Night minutes"),
        ("total_intl_minutes", "International minutes"),
        ("total_intl_calls", "International calls"),
    ]
    present = set(raw_columns)
    items: list[tuple[str, str]] = []
    for column, label in priority:
        if column not in present:
            continue
        value = row.get(column)
        if pd.isna(value):
            text = "—"
        elif column in {"international_plan", "voice_mail_plan"}:
            try:
                text = "Yes" if float(value) == 1 else "No"
            except (TypeError, ValueError):
                text = str(value)
        elif column == "area_code":
            try:
                text = f"{int(float(value))}"
            except (TypeError, ValueError):
                text = str(value)
        elif isinstance(value, (int, float)):
            text = f"{float(value):,.1f}" if abs(float(value)) < 10000 else f"{float(value):,.2f}"
            if float(value).is_integer() and column in {"account_length", "number_customer_service_calls", "total_intl_calls"}:
                text = f"{int(float(value)):,}"
        else:
            text = str(value)
        items.append((label, text))
    return items[:8]


def full_record_frame(frame: pd.DataFrame, raw_columns: Sequence[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Field", "Value"])
    row = frame.iloc[0]
    labels = {**CUSTOMER_EXTRA_COLUMNS}
    rows = []
    for column in raw_columns:
        value = row.get(column)
        if pd.isna(value):
            display = "—"
        elif column in {"international_plan", "voice_mail_plan"}:
            try:
                display = "Yes" if float(value) == 1 else "No"
            except (TypeError, ValueError):
                display = str(value)
        else:
            display = str(value)
        rows.append({"Field": labels.get(column, column.replace("_", " ").title()), "Value": display})
    return pd.DataFrame(rows)


def download_pair(
    frame: pd.DataFrame,
    *,
    prefix: str,
    key: str,
    json_payload: Mapping[str, Any] | None = None,
    csv_label: str = "Download CSV",
    json_label: str = "Download JSON",
) -> None:
    """Two download buttons with formula-safe CSV and JSON bytes."""

    cols = st.columns(2)
    with cols[0]:
        st.download_button(
            csv_label,
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
            json_label,
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
    "CUSTOMER_EXTRA_COLUMNS",
    "checks_frame",
    "customer_view",
    "download_pair",
    "full_record_frame",
    "local_contributions_frame",
    "metrics_frame",
    "profile_items",
    "risk_column_config",
    "risk_view",
    "selected_customers_frame",
    "sensitivity_frame",
]
