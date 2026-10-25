"""Executive Dashboard: an interactive workspace over the scored population.

Every number is read from a persisted artifact. Predicted churn is always
labelled as a model output, separate from the labelled historical churn rate.
Explicit filters change the view only; no rescoring happens on render.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import (
    filter_risk_frame,
    format_count,
    format_percent,
    format_probability,
    nested_get,
    tier_counts,
)
from churnwise.dashboard.components import charts, layout, tables

_SEARCH_PLACEHOLDER = "Search a customer reference"
_PROBABILITY_RANGES: dict[str, tuple[float, float] | None] = {
    "Any probability": None,
    "0.00–0.20": (0.0, 0.20),
    "0.20–0.40": (0.20, 0.40),
    "0.40–0.60": (0.40, 0.60),
    "0.60–0.80": (0.60, 0.80),
    "0.80–1.00": (0.80, 1.0),
}


def _selected_points(state: object) -> list[Mapping[str, Any]]:
    """Normalise the Plotly selection payload across Streamlit versions."""

    selection = getattr(state, "selection", None)
    if selection is None and isinstance(state, Mapping):
        selection = state.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, Mapping):
        points = selection.get("points")
    return [point for point in (points or []) if isinstance(point, Mapping)]


def _tier_from_points(points: list[Mapping[str, Any]]) -> str | None:
    tiers = {"low", "medium", "high", "critical"}
    for point in points:
        data = point.get("customdata")
        if isinstance(data, (list, tuple)) and data:
            value = str(data[0]).strip().lower()
            if value in tiers:
                return value
        value = str(point.get("x") or "").strip().lower()
        if value in tiers:
            return value
    return None


def _bin_from_points(points: list[Mapping[str, Any]]) -> tuple[float, float] | None:
    for point in points:
        data = point.get("customdata")
        if isinstance(data, (list, tuple)) and len(data) >= 5:
            try:
                low, high = float(data[3]), float(data[4])
            except (TypeError, ValueError):
                continue
            if 0.0 <= low < high <= 1.0:
                return (low, high)
    return None


def _run_state(settings: Settings) -> tuple[str | None, list[str]]:
    runs = access.risk_ids(settings)
    run_labels = access.risk_run_labels(settings, runs)
    if not runs:
        return None, []
    chosen = st.session_state.get("cw_exec_run")
    if chosen not in runs:
        st.session_state.pop("cw_exec_run", None)
        chosen = access.default_risk_run_id(settings, runs)
    st.session_state["cw_exec_run_labels"] = run_labels
    return chosen, runs


def _insights(
    frame: pd.DataFrame,
    counts: Mapping[str, int],
) -> None:
    labels = [label for label in ("low", "medium", "high", "critical") if label in counts]
    if not counts or not labels:
        return
    largest = max(labels, key=lambda label: counts[label])
    total = int(len(frame))
    probabilities = pd.to_numeric(frame["churn_probability"], errors="coerce").dropna()
    median = float(probabilities.median()) if len(probabilities) else 0.0
    very_high = int((probabilities >= 0.80).sum()) if len(probabilities) else 0
    layout.insight_cards(
        [
            (
                "Largest risk tier",
                f"{largest.title()} · {format_count(counts[largest])}",
                f"{format_percent(counts[largest] / total if total else 0)} of scored customers",
            ),
            (
                "Probability at or above 0.80",
                format_count(very_high),
                f"{format_percent(very_high / total if total else 0)} of scored customers",
            ),
            ("Median probability", format_probability(median), "Middle stored churn probability"),
        ]
    )


def _model_summary(metrics: Mapping[str, Any] | None) -> None:
    metrics = metrics or {}
    selection = metrics.get("selection") or {}
    families = metrics.get("families") or {}
    family = selection.get("selected_family")
    selected = families.get(family) or {}
    test = selected.get("test_metrics") or {}
    with st.container(border=True):
        cols = st.columns([2.2, 1, 1, 1])
        with cols[0]:
            st.markdown(f"**{selected.get('label') or family or 'No selected model'}**")
            st.caption(
                "Selected on validation F1; test rows were scored once. "
                "Held-out evidence, not the scored population."
            )
        with cols[1]:
            st.metric("Test F1", format_probability(test.get("f1")))
        with cols[2]:
            st.metric("Test ROC-AUC", format_probability(test.get("roc_auc")))
        with cols[3]:
            st.metric("Threshold", format_probability(selection.get("threshold")))
        if st.button("Open Model Laboratory", key="cw_exec_open_lab", type="secondary"):
            layout.navigate_to("Model Laboratory")


def _global_explanation(settings: Settings, family: str | None) -> Mapping[str, Any] | None:
    """Return a stored global explanation only when it matches this risk model."""

    if not family:
        return None
    for explanation_id in access.explanation_ids(settings):
        payload, _ = access.explanation_global(settings, explanation_id)
        if str((payload or {}).get("family") or "").lower() == family.lower():
            return payload
    return None


def render(settings: Settings) -> None:
    run_id, runs = _run_state(settings)
    summary, risk_path = access.risk_summary(settings, run_id)
    manifest, _ = access.risk_manifest(settings, run_id)
    validation, _ = access.validation_report(settings)
    metrics, _ = access.benchmark_metrics(settings)

    identity = (
        access.identity_from_risk(summary, manifest)
        if summary
        else access.identity_from_benchmark(metrics)
        if metrics
        else access.identity_from_validation(validation)
    )
    if run_id:
        identity = replace(identity, run_id=st.session_state.get("cw_exec_run_labels", {}).get(run_id, identity.run_id))
    layout.page_header(
        "Executive Dashboard",
        "Review scored churn risk, inspect customers, and route the retention queue. Predictions are model "
        "outputs, not observed outcomes.",
        identity,
    )

    if not summary:
        layout.empty_state(
            "No scored population is available yet",
            "Run the pipeline, then score a population. The dashboard never substitutes sample numbers.",
            "python -m churnwise.cli discover && python -m churnwise.cli acquire && "
            "python -m churnwise.cli validate && python -m churnwise.cli prepare && "
            "python -m churnwise.cli train && python -m churnwise.cli evaluate && "
            "python -m churnwise.cli score --population val --explain all",
        )
        return

    population = access.customer_population(settings, run_id)
    if population.frame is not None and not population.frame.empty:
        frame = population.frame.copy()
        raw_columns = population.raw_columns
    else:
        frame, _ = access.risk_table(settings, run_id)
        raw_columns = ()
    if frame is None or frame.empty:
        layout.empty_state(
            "The selected run has no score rows",
            "Choose another scoring run or create a population run from the Phase 9 command.",
            "python -m churnwise.cli score --population val --explain all",
        )
        return

    frame["churn_probability"] = pd.to_numeric(frame["churn_probability"], errors="coerce")
    frame = frame.dropna(subset=["churn_probability"]).reset_index(drop=True)
    boundaries = ((summary or {}).get("risk_thresholds") or {}).get("boundaries")
    boundaries = tuple(float(value) for value in boundaries) if boundaries else tuple(float(value) for value in settings.risk_tiers)
    counts = tier_counts(frame, boundaries)
    total = int(len(frame))
    likely_churn = int((frame["predicted_class"].astype(float) == 1).sum()) if "predicted_class" in frame else int(
        (frame.get("predicted_label", pd.Series(dtype=str)).astype(str).str.lower() == "yes").sum()
    )
    priority = int(counts.get("high", 0) + counts.get("critical", 0))
    medium_predicted = int(
        (
            (frame["predicted_class"].astype(float) == 1)
            & (frame.get("risk_tier", pd.Series(index=frame.index, dtype=str)).astype(str) == "medium")
        ).sum()
    ) if "predicted_class" in frame else 0
    mean_probability = float(frame["churn_probability"].mean())
    threshold = (summary or {}).get("classification_threshold")
    split = str(nested_get(summary, "population", "split", default="not recorded") or "not recorded")
    population_label = "Uploaded dataset" if split.strip().lower() == "input" else f"Population {split} split"

    # Compact run picker. The header above already reflects this value because
    # widget state is resolved before the page is drawn.
    control_cols = st.columns([1.7, 1.5])
    with control_cols[0]:
        chosen = st.selectbox(
            "Scoring run",
            runs,
            index=runs.index(run_id) if run_id in runs else 0,
            key="cw_exec_run",
            format_func=lambda value: st.session_state.get("cw_exec_run_labels", {}).get(value, value),
            help="Only persisted scoring runs are listed; changing the run never retrains the model.",
        )
    with control_cols[1]:
        st.caption(
            f"{population_label} · {format_count(total)} customers · "
            f"classification threshold {format_probability(threshold)}"
        )
    layout.metric_row(
        [
            ("Customers Scored", format_count(total), "Rows with a stored model probability in this run"),
            (
                "Predicted Churn",
                f"{format_count(likely_churn)} ({format_percent(likely_churn / total if total else 0)})",
                f"Customers with predicted probability ≥ {format_probability(threshold)}; includes medium-tier customers that cross this cutoff",
            ),
            (
                "High-Risk Customers",
                f"{format_count(priority)} ({format_percent(priority / total if total else 0)})",
                "Customers in the high or critical tier at the stored boundaries",
            ),
            (
                "Mean Churn Probability",
                format_probability(mean_probability),
                "Average stored probability; not a threshold and not an observed rate",
            ),
        ]
    )
    high_tier_boundary = boundaries[1] if len(boundaries) >= 2 else None
    st.caption(
        f"Predicted churn uses the {format_probability(threshold)} classification cutoff. "
        f"High-risk counts only high + critical tiers (starting at {format_probability(high_tier_boundary)}). "
        f"{format_count(medium_predicted)} medium-tier customers meet the churn cutoff in this run."
    )

    token = int(st.session_state.get("cw_exec_filter_token", 0))
    chart_token = int(st.session_state.get("cw_exec_chart_token", 0))
    # Streamlit restores widget values before the page reruns, so these keys
    # can coordinate the charts even though the compact controls render below.
    tier_choice = st.session_state.get(f"cw_exec_tier_{token}", "All")
    range_choice = st.session_state.get(f"cw_exec_probability_{token}", "Any probability")
    selected_tier = None if tier_choice in (None, "All") else str(tier_choice).lower()
    selected_bin = _PROBABILITY_RANGES.get(str(range_choice))
    chart_tier = st.session_state.get("cw_exec_chart_tier")
    chart_bin = st.session_state.get("cw_exec_chart_bin")
    effective_tier = selected_tier or chart_tier
    effective_bin = selected_bin or chart_bin

    layout.section("Risk distribution", "Select a bar or probability band to filter the preview. The stored model stays fixed.")
    chart_cols = st.columns([1.12, 1], gap="large")
    with chart_cols[0]:
        st.markdown("##### Risk-tier distribution")
        tier_state = st.plotly_chart(
            charts.tier_distribution_chart(counts, mode="counts", selected=effective_tier),
            width="stretch",
            config={"displayModeBar": False},
            on_select="rerun",
            selection_mode=("points", "box"),
            key=f"cw_exec_tier_chart_{chart_token}",
        )
        st.caption("Stored risk boundaries · box-select a bar or choose a tier below.")
    with chart_cols[1]:
        st.markdown("##### Churn-probability distribution")
        bins = charts.probability_bins(frame, bins=20)
        probability_state = st.plotly_chart(
            charts.probability_distribution_chart(
                bins,
                threshold=float(threshold) if threshold is not None else None,
                selected=effective_bin,
            ),
            width="stretch",
            config={"displayModeBar": False},
            on_select="rerun",
            selection_mode=("points", "box"),
            key=f"cw_exec_probability_chart_{chart_token}",
        )
        st.caption("Stored probabilities in 0.05 intervals · box-select a band or choose a range below.")

    clicked_tier = _tier_from_points(_selected_points(tier_state))
    clicked_bin = _bin_from_points(_selected_points(probability_state))
    changed = False
    if clicked_tier and st.session_state.get("cw_exec_chart_tier") != clicked_tier:
        st.session_state["cw_exec_chart_tier"] = clicked_tier
        changed = True
    if clicked_bin and st.session_state.get("cw_exec_chart_bin") != clicked_bin:
        st.session_state["cw_exec_chart_bin"] = clicked_bin
        changed = True
    if changed:
        # Re-render once so the selected bar is highlighted in the same view.
        st.rerun()
    if chart_tier:
        effective_tier = selected_tier or chart_tier
    if chart_bin:
        effective_bin = selected_bin or chart_bin
    if chart_tier or chart_bin:
        if st.button("Clear chart selection", key=f"cw_exec_clear_charts_{token}"):
            st.session_state.pop("cw_exec_chart_tier", None)
            st.session_state.pop("cw_exec_chart_bin", None)
            # Changing only the chart keys clears Streamlit's persisted Plotly
            # payload while preserving search and direct filter controls.
            st.session_state["cw_exec_chart_token"] = chart_token + 1
            st.rerun()

    layout.section("Population insights")
    _insights(frame, counts)
    with st.expander("Historical validation cohort"):
        target = (validation or {}).get("target") or {}
        historical = (validation or {}).get("summary") or {}
        st.caption(
            f"{format_count(historical.get('n_rows'))} labelled records · "
            f"{format_count(target.get('positive_count'))} observed churn · "
            f"{format_percent(target.get('positive_rate'))} historical rate. "
            "This is a different population from the scored run."
        )

    if raw_columns:
        segment_options = [column for column in raw_columns if column in frame.columns]
        if segment_options:
            with st.expander("Explore risk by source segment"):
                segment = st.selectbox(
                    "Customer attribute",
                    segment_options,
                    format_func=lambda value: value.replace("_", " ").title(),
                    key=f"cw_exec_segment_{run_id or 'latest'}",
                    help="Compares stored mean probabilities by verified original customer attributes.",
                )
                st.plotly_chart(
                    charts.segment_risk_chart(frame, segment),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                st.caption(
                    "Segment averages describe model scores in this population; they do not show causal effects."
                )

    filter_cols = st.columns([1.4, 1.7, 1.2])
    with filter_cols[0]:
        query = st.text_input(
            "Search customer reference",
            placeholder=_SEARCH_PLACEHOLDER,
            key=f"cw_exec_search_{token}",
        )
    with filter_cols[1]:
        tier_choice = st.pills(
            "Risk tier",
            ["All", "Low", "Medium", "High", "Critical"],
            default="All",
            key=f"cw_exec_tier_{token}",
            help="Uses the stored risk-tier definitions; not a new boundary.",
        )
    with filter_cols[2]:
        range_choice = st.selectbox(
            "Probability range",
            list(_PROBABILITY_RANGES),
            index=0,
            key=f"cw_exec_probability_{token}",
        )
    selected_tier = None if tier_choice in (None, "All") else str(tier_choice).lower()
    selected_bin = _PROBABILITY_RANGES.get(str(range_choice))
    effective_tier = selected_tier or chart_tier
    effective_bin = selected_bin or chart_bin

    control_row = st.columns([1.2, 1, 3])
    with control_row[0]:
        priority_only = st.toggle(
            "Only high & critical",
            key=f"cw_exec_priority_{token}",
            help="Filters the list to the two highest stored risk tiers.",
        )
    with control_row[1]:
        if st.button("Reset filters", key=f"cw_exec_reset_{token}", width="stretch"):
            st.session_state.pop("cw_exec_chart_tier", None)
            st.session_state.pop("cw_exec_chart_bin", None)
            st.session_state["cw_exec_filter_token"] = token + 1
            st.session_state["cw_exec_chart_token"] = chart_token + 1
            st.rerun()
    active = []
    if effective_tier:
        active.append(f"tier: {effective_tier.title()}")
    if effective_bin:
        active.append(f"probability: {effective_bin[0]:.2f}–{effective_bin[1]:.2f}")
    if priority_only:
        active.append("high & critical")
    if query.strip():
        active.append(f"search: {query.strip()}")
    with control_row[2]:
        if active:
            st.markdown(
                '<div class="cw-filter-line" style="padding-top:0.35rem">Active filters: '
                + " ".join(f'<span class="cw-chip">{item}</span>' for item in active)
                + "</div>",
                unsafe_allow_html=True,
            )

    filtered = filter_risk_frame(
        frame,
        tiers=(effective_tier,) if effective_tier else (),
        min_probability=effective_bin[0] if effective_bin else 0.0,
        max_probability=effective_bin[1] if effective_bin else 1.0,
        reference_query=query,
    )
    if priority_only:
        filtered = filtered[filtered["risk_tier"].isin(["high", "critical"])]

    shown = min(len(filtered), 15)
    layout.section(
        "Customers needing attention",
        f"Showing {format_count(shown)} highest-risk of {format_count(len(filtered))} matching "
        f"({format_count(total)} scored). Select a row to open its assessment.",
    )
    if filtered.empty:
        st.info("No customers match these filters. Click Reset selection to restore the full population.")
        _model_summary(metrics)
        return

    extras = [column for column in ("account_length", "international_plan", "number_customer_service_calls") if column in raw_columns]
    view = tables.customer_view(
        filtered.sort_values("churn_probability", ascending=False),
        limit=15,
        extra_columns=extras,
    ).reset_index(drop=True)
    result = st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="cw_exec_customer_table",
        column_config=tables.risk_column_config(),
    )
    selected_rows: list[int] = []
    try:
        selected_rows = list(result.selection.rows or [])
    except (AttributeError, KeyError, TypeError):
        selected_rows = []
    if selected_rows:
        position = int(selected_rows[0])
        if 0 <= position < len(view):
            reference = str(view.iloc[position].get("Customer"))
            action_cols = st.columns([1.1, 1.6, 2.2])
            with action_cols[0]:
                if st.button("Open assessment", type="primary", key="cw_exec_open_customer"):
                    st.session_state["cw_customer_run"] = run_id or ""
                    st.session_state["cw_customer_reference"] = reference
                    layout.navigate_to("Customer Risk Explorer")
            with action_cols[1]:
                if st.button("Open Retention Workbench", key="cw_exec_open_workbench"):
                    layout.navigate_to("Retention Workbench")
            with action_cols[2]:
                st.caption(f"Selected `{reference}`. Raw attributes are joined only after split verification.")
    if population.frame is None:
        st.caption(f"Original customer attributes are not shown: {population.reason}")
    elif raw_columns:
        st.caption("Original account attributes are verified against the persisted split and row positions.")

    _model_summary(metrics)

    selected_family = str(nested_get(summary, "model", "family", default="") or "")
    global_explanation = _global_explanation(settings, selected_family)
    with st.expander("Global model drivers · SHAP", expanded=False):
        if global_explanation and global_explanation.get("feature_importance"):
            sample = global_explanation.get("sample") or {}
            st.caption(
                f"Stored global explanation for {selected_family.replace('_', ' ')} · "
                f"{format_count(sample.get('sampled_rows'))} sampled rows. "
                "These are model associations, not causal effects or customer-specific contributions."
            )
            st.plotly_chart(
                charts.importance_chart(global_explanation["feature_importance"], top=6),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.caption(
                "No persisted global explanation matches the selected scoring model. "
                "The dashboard does not create feature importance at render time."
            )

    warnings: list[str] = []
    for source in (validation, metrics, summary):
        warnings.extend((source or {}).get("warnings") or [])
    warnings = list(dict.fromkeys(str(item) for item in warnings))
    if warnings:
        with st.expander(f"Recorded warnings ({len(warnings)})"):
            for warning in warnings:
                st.write(f"- {warning}")
