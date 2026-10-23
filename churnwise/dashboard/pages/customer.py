"""Customer detail: one stored assessment, its drivers and a verified local explanation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import dict_to_json_bytes, format_probability
from churnwise.dashboard.components import layout, tables


def _parse_driver_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _driver_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    keep = [column for column in ("feature", "active_level", "direction", "contribution", "value") if column in frame]
    frame = frame[keep]
    if "contribution" in frame:
        frame["contribution"] = pd.to_numeric(frame["contribution"], errors="coerce").round(4)
    return frame


def _priority(tier: str) -> tuple[str, str]:
    return {
        "critical": ("Immediate review", "Highest-probability customers; verify contactability and owner."),
        "high": ("Prioritise outreach", "High-probability customers worth a retention intervention."),
        "medium": ("Monitor / test", "Ambiguous risk; use a measured pilot before broad spend."),
        "low": ("Standard service", "No special retention action is indicated by the model alone."),
    }.get(str(tier), ("Unclassified", "No tier was available for this assessment."))


def render(settings: Settings) -> None:
    runs = access.risk_ids(settings)
    run_id = st.selectbox("Risk run", runs, index=0, key="cw_customer_run") if runs else None
    summary, _ = access.risk_summary(settings, run_id)
    manifest, _ = access.risk_manifest(settings, run_id)
    frame, _ = access.risk_table(settings, run_id)
    drivers_map, _ = access.risk_drivers(settings, run_id)
    identity = access.identity_from_risk(summary, manifest)
    layout.page_header(
        "Customer detail",
        "One stored assessment with its tier, ranked position, attached Phase 8 drivers and a probability-checked "
        "local explanation. Raw customer attributes are not stored in the risk artifacts.",
        identity,
    )
    if frame is None or frame.empty:
        layout.empty_state(
            "No scored customers are available",
            "Create a Phase 9 risk run before inspecting individual assessments.",
            "python -m churnwise.cli score --population val --explain all",
        )
        return

    options = {
        str(row.get("record_reference")): int(row.get("input_position", -1))
        for row in frame[["record_reference", "input_position"]].to_dict(orient="records")
    }
    reference = st.selectbox("Customer reference", list(options), key="cw_customer_reference")
    position = options[reference]
    row = frame[frame["record_reference"].astype(str) == str(reference)].iloc[0]
    probability = float(row.get("churn_probability"))
    tier = str(row.get("risk_tier") or "unclassified")
    priority, priority_detail = _priority(tier)

    layout.metric_row(
        [
            ("Churn probability", format_probability(probability), "Phase 9 model estimate"),
            ("Risk tier", tier.title(), "Display tier stored with the run"),
            ("Rank", row.get("rank", "not available"), "Descending probability order"),
            ("Predicted class", row.get("predicted_label", "not available"), "At the frozen classification threshold"),
        ]
    )
    st.caption(
        f"Reference `{reference}` · input position {position} · assessed {row.get('assessed_at')} · "
        f"classification threshold {format_probability(row.get('classification_threshold'))}."
    )
    layout.status_line(priority, tier, "risk" if tier in {"critical", "high"} else "info", priority_detail)
    layout.note(
        "The priority label is a presentation heuristic derived from the tier, not a causal recommendation "
        "or a measured treatment effect."
    )

    layout.section("Attached drivers", "Top upward and downward SHAP contributions stored with this assessment.")
    driver_entry = (drivers_map or {}).get(str(reference)) or {}
    if not driver_entry:
        status = str(row.get("explanation_status") or "unavailable")
        reason = str(row.get("explanation_reason") or "No driver payload was stored for this customer.")
        if status == "available":
            reason = "The run summary says drivers are available, but no per-customer payload was found."
        st.warning(f"Drivers unavailable ({status}): {reason}")
    else:
        for title, key in (
            ("Increases churn", "top_positive"),
            ("Decreases churn", "top_negative"),
        ):
            st.markdown(f"**{title}**")
            drivers = _driver_frame(_parse_driver_rows(driver_entry.get(key)))
            if drivers.empty:
                st.caption("None stored.")
            else:
                st.dataframe(drivers, width="stretch", hide_index=True)
        st.write(driver_entry.get("summary") or "No summary sentence stored.")
        st.caption(
            "`contribution` is in probability space; `value` is the transformed model input. "
            "These are model attributions, not causal effects."
        )

    layout.section("Local explanation", "Shown only when a persisted local artifact matches this row's probability.")
    local_payload, local_path = access.local_explanation(settings, position)
    matched = False
    if local_payload:
        local_probability = local_payload.get("probability")
        try:
            matched = local_probability is not None and abs(float(local_probability) - probability) < 1e-9
        except (TypeError, ValueError):
            matched = False
    if local_payload and matched:
        cols = st.columns(3)
        with cols[0]:
            st.metric("Base value", format_probability(local_payload.get("base_value")))
        with cols[1]:
            st.metric("Model output", format_probability(local_payload.get("model_output")))
        with cols[2]:
            st.metric("Output space", local_payload.get("output_space", "not available"))
        st.write(local_payload.get("summary") or "No summary sentence stored.")
        contributions = tables.local_contributions_frame(local_payload)
        if not contributions.empty:
            st.dataframe(contributions, width="stretch", hide_index=True)
        plot = access.local_explanation_plot(settings, position)
        if plot:
            st.image(str(plot), caption=f"Local SHAP waterfall · row {position}")
        st.caption(f"Local artifact: {local_path}")
    elif local_payload:
        st.warning(
            "A local explanation exists at this matrix position but its probability does not match this "
            "assessment, so it is not shown."
        )
    else:
        st.info(
            "No local explanation was persisted at this matrix position. Local SHAP is computed only for the "
            "explicit examples selected during Phase 8; this page does not run SHAP."
        )

    predicted_value = row.get("predicted_class")
    predicted_class = int(predicted_value) if pd.notna(predicted_value) else None
    rank_value = row.get("rank")
    payload = {
        "risk_run_id": summary.get("run_id") if summary else None,
        "record_reference": reference,
        "input_position": position,
        "churn_probability": probability,
        "risk_tier": tier,
        "predicted_class": predicted_class,
        "rank": None if pd.isna(rank_value) else int(rank_value),
        "drivers": driver_entry or None,
        "local_explanation": local_payload if matched else None,
    }
    st.download_button(
        "Download this assessment (JSON)",
        data=dict_to_json_bytes(payload),
        file_name=f"assessment_{reference}.json",
        mime="application/json",
        key="cw_customer_download",
    )
    limitations = (summary or {}).get("limitations") or []
    if limitations:
        with st.expander("Assessment limitations"):
            for item in limitations:
                st.write(f"- {item}")
