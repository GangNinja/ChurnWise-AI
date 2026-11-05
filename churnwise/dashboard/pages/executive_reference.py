"""Executive dashboard using the user's Churn Analytics reference layout."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import data_access as access
from churnwise.dashboard import actions
from churnwise.dashboard.analytics import (
    filter_risk_frame,
    format_count,
    format_probability,
    nested_get,
)
from churnwise.dashboard.components import charts, layout, reference_dashboard as reference, tables
from churnwise.dashboard.pages import explorer as explorer_page

_PROBABILITY_RANGES: dict[str, tuple[float, float] | None] = {
    "Any probability": None,
    "0.00–0.20": (0.0, 0.20),
    "0.20–0.40": (0.20, 0.40),
    "0.40–0.60": (0.40, 0.60),
    "0.60–0.80": (0.60, 0.80),
    "0.80–1.00": (0.80, 1.0),
}


def _run_state(settings: Settings) -> tuple[str | None, list[str]]:
    runs = access.risk_ids(settings)
    labels = access.risk_run_labels(settings, runs)
    if not runs:
        return None, []
    pending = st.session_state.pop("cw_exec_run_pending", None)
    chosen = pending if pending in runs else st.session_state.get("cw_exec_run")
    preferred = access.default_risk_run_id(settings, runs)
    prior_default = st.session_state.get("cw_exec_default_run")
    if pending in runs:
        chosen = pending
    elif chosen not in runs or (prior_default and chosen == prior_default):
        chosen = preferred
    elif not prior_default:
        selected_summary, _ = access.risk_summary(settings, chosen)
        split = str(nested_get(selected_summary, "population", "split", default="") or "").lower()
        if split != "input":
            chosen = preferred
    st.session_state["cw_exec_run"] = chosen
    st.session_state["cw_exec_default_run"] = preferred
    st.session_state["cw_exec_run_labels"] = labels
    return chosen, runs


def _global_explanation(
    settings: Settings,
    family: str | None,
    model_run_id: str | None,
) -> Mapping[str, Any] | None:
    if not family:
        return None
    for explanation_id in access.explanation_ids(settings):
        payload, _ = access.explanation_global(settings, explanation_id)
        if str((payload or {}).get("family") or "").lower() != family.lower():
            continue
        provenance = (payload or {}).get("provenance") or {}
        if model_run_id and str(provenance.get("model_run_id") or "") != model_run_id:
            continue
        return payload
    return None


def _model_summary(metrics: Mapping[str, Any] | None, threshold: Any) -> None:
    metrics = metrics or {}
    selection = metrics.get("selection") or {}
    family = str(selection.get("selected_family") or "mlp")
    selected = (metrics.get("families") or {}).get(family) or {}
    test = selected.get("test_metrics") or {}
    with st.container(border=True):
        columns = st.columns([2.15, 1, 1, 1])
        with columns[0]:
            st.markdown("### MLP Neural Network")
            st.caption("Selected on validation F1; held-out test rows were scored once and are not part of this customer population.")
        with columns[1]:
            st.metric("Test F1", format_probability(test.get("f1")))
        with columns[2]:
            st.metric("Test ROC-AUC", format_probability(test.get("roc_auc")))
        with columns[3]:
            st.metric("Threshold", format_probability(selection.get("threshold", threshold)))
        if st.button("Open Model Laboratory", key="cw_exec_open_lab"):
            layout.navigate_to("Model Laboratory")


def _attention_table(frame: pd.DataFrame, run_id: str, run_label: str) -> None:
    layout.section(
        "Customers needing attention",
        "Review the highest-scored customers in the selected run. Select a row to open its stored customer assessment.",
    )
    token = int(st.session_state.get("cw_exec_attention_token", 0))
    controls = st.columns([1.25, 1.45, 1.2, 1.15])
    with controls[0]:
        search = st.text_input("Search customer reference", key=f"cw_att_search_{token}", placeholder="Customer reference")
    with controls[1]:
        tiers = ["All", "Low", "Medium", "High", "Critical"]
        tier = st.pills("Risk tier", tiers, default="All", key=f"cw_att_tier_{token}")
    with controls[2]:
        probability = st.selectbox("Probability range", list(_PROBABILITY_RANGES), key=f"cw_att_probability_{token}")
    with controls[3]:
        st.write("")
        if st.button("Reset filters", key=f"cw_att_reset_{token}", width="stretch"):
            st.session_state["cw_exec_attention_token"] = token + 1
            st.rerun()
    bounds = _PROBABILITY_RANGES.get(str(probability))
    filtered = filter_risk_frame(
        frame,
        tiers=() if tier in (None, "All") else (str(tier).lower(),),
        min_probability=bounds[0] if bounds else 0.0,
        max_probability=bounds[1] if bounds else 1.0,
        reference_query=search,
    )
    high_only = st.toggle("Only high and critical customers", key=f"cw_att_high_{token}")
    if high_only and "risk_tier" in filtered:
        filtered = filtered[filtered["risk_tier"].astype(str).str.lower().isin({"high", "critical"})]
    if filtered.empty:
        st.info("No customers match these filters.")
        return
    filtered = filtered.sort_values("churn_probability", ascending=False)
    view = tables.customer_view(filtered, limit=15).reset_index(drop=True)
    st.caption(f"Showing {min(15, len(filtered)):,} highest-risk of {len(filtered):,} matching ({len(frame):,} scored).")
    result = st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=f"cw_exec_attention_table_{run_id}_{run_label}",
        column_config=tables.risk_column_config(),
    )
    rows = []
    try:
        rows = list(result.selection.rows or [])
    except (AttributeError, KeyError, TypeError):
        pass
    if rows:
        position = int(rows[0])
        if 0 <= position < len(view):
            reference_id = str(view.iloc[position].get("Customer", ""))
            if st.button("Open customer assessment", type="primary", key="cw_exec_attention_open"):
                st.session_state["cw_customer_run"] = run_id
                st.session_state["cw_customer_reference"] = reference_id
                layout.navigate_to("Customer Risk Explorer")


def render(settings: Settings) -> None:
    reference.inject_styles()
    run_id, runs = _run_state(settings)
    summary, risk_path = access.risk_summary(settings, run_id)
    manifest, _ = access.risk_manifest(settings, run_id)
    validation, _ = access.validation_report(settings)
    if not summary:
        layout.empty_state(
            "No scoring runs are available",
            "Score a population in Model Laboratory or upload a dataset after a compatible model is available.",
            "python -m churnwise.cli score --population val --explain all",
        )
        return

    population = access.customer_population(settings, run_id)
    if population.frame is not None and not population.frame.empty:
        original = population.frame.copy()
        raw_columns = population.raw_columns
    else:
        original, _ = access.risk_table(settings, run_id)
        original = original.copy() if original is not None else pd.DataFrame()
        raw_columns = ()
    if original.empty:
        layout.empty_state(
            "This scoring run has no score rows",
            "Choose a different saved run or score a dataset.",
            "python -m churnwise.cli score --population val --explain all",
        )
        return

    original["churn_probability"] = pd.to_numeric(original["churn_probability"], errors="coerce")
    original = original.dropna(subset=["churn_probability"]).reset_index(drop=True)
    label_map = st.session_state.get("cw_exec_run_labels", {})
    run_label = label_map.get(run_id, run_id or "Scoring run")
    source_label = str(summary.get("source_label") or "").strip()
    dataset_label = source_label.split("·", 1)[-1].strip() if source_label else "Telco Customer Churn (Kaggle)"
    hero_slot = st.empty()

    selected_run = st.selectbox(
        "Scoring run",
        runs,
        key="cw_exec_run",
        format_func=lambda value: label_map.get(value, value),
        help="Each option is a persisted score run. Selecting one updates this dashboard's metrics and charts.",
    )

    with st.container(border=True):
        upload_copy, upload_action = st.columns([3.4, 1.2], vertical_alignment="bottom")
        with upload_copy:
            st.markdown("#### Analyze another customer dataset")
            st.caption(
                "Upload a compatible Telco CSV to score customers with the trained MLP, "
                "then review churn patterns, predictions, and available SHAP explanations here."
            )
        with upload_action:
            if st.button("Upload & predict", key="cw_exec_open_upload", type="primary", width="stretch"):
                st.session_state["cw_exec_show_upload"] = not st.session_state.get("cw_exec_show_upload", False)
                st.rerun()

    with st.expander("Scoring run details", expanded=False):
        threshold = summary.get("classification_threshold")
        st.caption(
            f"{len(original):,} customers scored · "
            f"{str(nested_get(summary, 'model', 'family', default='mlp')).upper()} model · "
            f"classification threshold {format_probability(threshold)}"
        )
        identity = access.identity_from_risk(summary, manifest)
        cards = st.columns(4)
        for col, label, value in zip(
            cards,
            ("Run ID", "Model family", "Dataset hash", "Generated at"),
            (identity.run_id, identity.family, identity.dataset_sha256, identity.generated_at),
        ):
            with col:
                st.caption(label)
                st.code(value or "not available")

    uploaded_runs: list[str] = []
    for candidate in runs:
        candidate_summary, _ = access.risk_summary(settings, candidate)
        population_info = (candidate_summary or {}).get("population") or {}
        source_info = str((candidate_summary or {}).get("source_label") or "").strip().lower()
        if (
            str(population_info.get("split") or "").lower() == "input"
            and not source_info.startswith(("preloaded ·", "primary dataset ·"))
        ):
            uploaded_runs.append(candidate)
    if uploaded_runs:
        with st.expander("Manage uploaded datasets", expanded=False):
            st.caption("Remove a saved uploaded scoring run from the selector. The primary Telco dataset and bundled datasets are protected.")
            remove_id = st.selectbox(
                "Uploaded scoring run",
                uploaded_runs,
                key="cw_exec_remove_upload_run",
                format_func=lambda value: label_map.get(value, value),
            )
            confirm_remove = st.checkbox(
                "I understand this removes the saved scores and dashboard attributes for this upload.",
                key="cw_exec_confirm_remove_upload",
            )
            if st.button("Remove uploaded dataset", disabled=not confirm_remove, key="cw_exec_remove_upload", type="secondary"):
                try:
                    actions.remove_uploaded_scoring_run(settings, remove_id)
                except actions.DashboardActionError as exc:
                    st.error(f"{exc.code}: {exc.message}")
                    if exc.remediation:
                        st.info(exc.remediation)
                else:
                    remaining_runs = access.risk_ids(settings)
                    st.session_state["cw_exec_run_pending"] = access.default_risk_run_id(settings, remaining_runs)
                    st.session_state.pop("cw_exec_confirm_remove_upload", None)
                    st.success("Uploaded scoring run removed.")
                    st.rerun()

    if st.session_state.get("cw_exec_show_upload", False):
        with st.container(border=True):
            st.caption("Upload a CSV or choose a bundled testing dataset. The fitted MLP scores it and saves a new run for this selector.")
            schema, _ = explorer_page._schema(settings)
            if not schema:
                st.info("No fitted model schema is available. Prepare and train the Telco MLP in Model Laboratory first.")
            else:
                explorer_page._batch_section(settings, schema)

    frame, active_filters, has_observed = reference.dashboard_filters(original, run_id=run_id)
    with hero_slot:
        reference.hero(
            "Customer Churn Analytics Dashboard",
            "Telco Customer Retention Intelligence — ChurnWise AI",
            len(frame),
            len(original),
            dataset_label,
            has_filters=bool(active_filters),
        )

    reference.metric_cards(frame, has_observed=has_observed)
    if not has_observed:
        st.caption("This run has no observed outcomes. Churn rates and customer segments below use model-predicted classes.")
    elif active_filters:
        st.caption("Filtered churn rates use observed outcomes from the verified source rows.")
    reference.charts(frame, has_observed=has_observed)

    _attention_table(frame, run_id or "", run_label)

    linked_benchmark = str(nested_get(summary, "model", "benchmark_run_id", default="") or "")
    metrics, _ = access.benchmark_metrics(settings, linked_benchmark or None)
    if metrics is None:
        metrics, _ = access.benchmark_metrics(settings)
    _model_summary(metrics, threshold)
    family = str(nested_get(summary, "model", "family", default="") or "")
    model_run = str(nested_get(summary, "model", "model_run_id", default="") or "")
    explanation = _global_explanation(settings, family, model_run or None)
    with st.expander("Global model drivers · SHAP", expanded=True):
        if explanation and explanation.get("feature_importance"):
            sample = explanation.get("sample") or {}
            st.caption(
                f"Stored global SHAP explanation for {family.replace('_', ' ').upper()} · "
                f"{format_count(sample.get('sampled_rows'))} sampled rows. These are model associations, not causal effects."
            )
            st.plotly_chart(charts.importance_chart(explanation["feature_importance"], top=8), width="stretch", config={"displayModeBar": False})
        else:
            st.info("A global SHAP artifact is not available for this model run.")


__all__ = ["render"]
