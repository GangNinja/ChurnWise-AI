"""Data quality: the Phase 4 gate, profile and leakage review."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import format_count, format_percent
from churnwise.dashboard.components import charts, layout, tables


def _column_frame(columns: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for column in columns:
        rows.append(
            {
                "Column": column.get("name"),
                "Kind": column.get("kind"),
                "Role": column.get("role"),
                "Missing %": column.get("missing_pct"),
                "Unique": column.get("unique_count"),
                "Flags": ", ".join(column.get("flags") or []) or "none",
            }
        )
    frame = pd.DataFrame(rows)
    if "Missing %" in frame:
        frame = frame.sort_values("Missing %", ascending=False)
        frame["Missing %"] = pd.to_numeric(frame["Missing %"], errors="coerce").round(2)
    return frame


def render(settings: Settings) -> None:
    report, report_path = access.validation_report(settings)
    eda, eda_path = access.eda_profile(settings, str((report or {}).get("dataset_id") or "") or None)
    identity = access.identity_from_validation(report)
    layout.page_header(
        "Data quality",
        "Validation checks, the preprocessing gate, class balance, missingness and leakage review for the "
        "acquired dataset. This page is read-only; the dataset is never modified here.",
        identity,
    )
    if not report:
        layout.empty_state(
            "No validation report exists",
            "Acquire a dataset and run the Phase 4 validation gate. No quality score is shown without a report.",
            "python -m churnwise.cli acquire && python -m churnwise.cli validate",
        )
        return

    summary = report.get("summary") or {}
    target = report.get("target") or {}
    gate = report.get("gate") or {}
    layout.status_line(
        "Validation gate",
        str(report.get("status") or "unknown").replace("_", " "),
        "ok" if gate.get("passed") else "warn",
    )
    if gate.get("blocking"):
        st.error("Blocking checks: " + ", ".join(str(item) for item in gate["blocking"]))
    elif gate.get("reasons"):
        st.warning("Gate reasons: " + "; ".join(str(item) for item in gate["reasons"]))
    if not gate.get("can_preprocess", False):
        st.error("Phase 4 did not approve this dataset for preprocessing.")

    layout.metric_row(
        [
            ("Rows", format_count(summary.get("n_rows")), "Validated artifact"),
            ("Columns", format_count(summary.get("n_cols")), "Raw columns"),
            ("Missing cells", format_percent(summary.get("missing_cell_pct")), "Share of all cells"),
            ("Duplicate rows", format_percent(summary.get("duplicate_row_pct")), "Exact duplicates"),
        ]
    )

    layout.section("Target interpretation", "The positive class is the churn label used by every later phase.")
    counts = target.get("counts") or {}
    method_detail = str(target.get("method") or "not recorded").replace("_", " ")
    method_label = method_detail.split(":", 1)[0].capitalize()
    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.metric("Target column", target.get("column") or "not available")
    with cols[1]:
        st.metric("Positive rate", format_percent(target.get("positive_rate")))
    with cols[2]:
        st.metric("Method", method_label, help=method_detail)
    if ":" in method_detail:
        st.caption(method_detail[0].upper() + method_detail[1:] + ".")
    if counts:
        frame = pd.DataFrame({"Class": list(counts), "Customers": list(counts.values())})
        st.plotly_chart(
            charts.class_balance_chart(frame),
            width="stretch",
        )
    if target.get("ambiguous"):
        st.warning("Target detection was ambiguous. Review the candidate list before trusting later phases.")

    layout.section("Validation checks", f"{len(report.get('checks') or [])} checks recorded.")
    checks = report.get("checks") or []
    if checks:
        st.dataframe(tables.checks_frame(checks), width="stretch", hide_index=True)
        failed = [check for check in checks if not check.get("passed")]
        if failed:
            st.warning(f"{len(failed)} check(s) did not pass; see the table and gate reasons.")
    st.caption(f"Validation report: {report_path}")

    layout.section("Feature roles and leakage", "Dropped identifiers and leakage risks are never silently used.")
    roles = report.get("roles") or {}
    leak = summary.get("leakage_risks") or []
    role_cols = st.columns(4)
    with role_cols[0]:
        st.metric("Numeric features", format_count(len(roles.get("numeric") or [])))
    with role_cols[1]:
        st.metric("Categorical features", format_count(len(roles.get("categorical") or [])))
    with role_cols[2]:
        st.metric("Dropped columns", format_count(len(roles.get("dropped") or [])))
    with role_cols[3]:
        st.metric("Leakage risks", format_count(len(leak)))
    if leak:
        for item in leak:
            st.warning(str(item))
    else:
        st.caption("No leakage risks were recorded by the validator.")
    if roles.get("dropped"):
        st.caption("Dropped: " + ", ".join(str(item) for item in roles["dropped"]))

    layout.section("Column profile", "Missingness and cardinality flags from the Phase 4 EDA profile.")
    if eda and eda.get("columns"):
        st.dataframe(_column_frame(eda["columns"]), width="stretch", hide_index=True)
        st.caption(f"EDA profile: {eda_path}")
    else:
        layout.empty_state(
            "No EDA profile was found",
            "The validation report exists but its companion profile is missing.",
            "python -m churnwise.cli validate",
        )

    capabilities = report.get("capabilities") or {}
    if capabilities:
        with st.expander("Dataset capability matrix"):
            st.dataframe(
                pd.DataFrame(
                    [{"Capability": key.replace("has_", "").replace("_", " "), "Detected": value} for key, value in capabilities.items()]
                ),
                width="stretch",
                hide_index=True,
            )
    warnings = report.get("warnings") or []
    if warnings:
        with st.expander(f"Warnings ({len(warnings)})"):
            for warning in warnings:
                st.write(f"- {warning}")
