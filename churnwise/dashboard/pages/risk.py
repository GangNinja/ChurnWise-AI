"""Churn risk intelligence: ranked scores, filters, tier configuration and exports."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import (
    apply_tiers,
    filter_risk_frame,
    format_count,
    format_percent,
    format_probability,
    probability_summary,
    tier_counts,
    tier_labels,
    validate_boundaries,
)
from churnwise.dashboard.components import charts, layout, tables


def _boundaries_from_summary(summary: dict[str, Any], settings: Settings) -> tuple[float, ...]:
    thresholds = summary.get("risk_thresholds") or {}
    values = thresholds.get("boundaries")
    if values and len(values) in (2, 3):
        return tuple(float(item) for item in values)
    return tuple(settings.risk_tiers)


def render(settings: Settings) -> None:
    runs = access.risk_ids(settings)
    run_id = st.selectbox("Risk run", runs, index=0, key="cw_risk_run") if runs else None
    summary, summary_path = access.risk_summary(settings, run_id)
    manifest, _ = access.risk_manifest(settings, run_id)
    frame, frame_path = access.risk_table(settings, run_id)
    identity = access.identity_from_risk(summary, manifest)
    layout.page_header(
        "Risk intelligence",
        "Ranked Phase 9 churn probabilities with configurable display tiers, filters and safe exports. "
        "Probabilities are model estimates, not guaranteed outcomes.",
        identity,
    )
    if not summary or frame is None or frame.empty:
        layout.empty_state(
            "No risk run is available",
            "Score a population with Phase 9 to populate the explorer. The dashboard does not score records on render.",
            "python -m churnwise.cli score --population val --explain all",
        )
        return

    default_boundaries = _boundaries_from_summary(summary, settings)
    labels = tier_labels(default_boundaries)
    with st.expander("Display tier boundaries", expanded=False):
        st.caption(
            "Re-deriving tiers changes only this view. It does not rescore records or rewrite the persisted run. "
            "Lower-inclusive semantics match Phase 9."
        )
        cols = st.columns(len(default_boundaries))
        boundaries = []
        for index, (col, value) in enumerate(zip(cols, default_boundaries)):
            with col:
                boundaries.append(
                    st.number_input(
                        f"Boundary {index + 1}",
                        min_value=0.01,
                        max_value=0.99,
                        value=float(value),
                        step=0.01,
                        format="%.2f",
                        key=f"cw_risk_boundary_{index}",
                    )
                )
        problem = validate_boundaries(boundaries)
        if problem:
            st.error(problem)
        else:
            labels = tier_labels(boundaries)
    if problem:
        boundaries = default_boundaries
        labels = tier_labels(boundaries)

    working = apply_tiers(frame, boundaries)
    layout.section("Filter", "Filters are pure operations over the stored table; no rescoring happens.")
    filter_cols = st.columns([1.3, 1.3, 1.0])
    with filter_cols[0]:
        selected_tiers = st.multiselect(
            "Risk tier",
            options=list(labels),
            default=[],
            key="cw_risk_tiers",
            help="Empty means every tier.",
        )
    with filter_cols[1]:
        probability_range = st.slider(
            "Probability range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
            key="cw_risk_probability",
        )
    with filter_cols[2]:
        reference_query = st.text_input("Reference contains", value="", key="cw_risk_reference")
    sort_by = st.selectbox(
        "Sort by",
        options=["rank", "churn_probability", "value_at_risk"],
        format_func=lambda value: {
            "rank": "Risk rank",
            "churn_probability": "Churn probability",
            "value_at_risk": "Value at risk",
        }[value],
        key="cw_risk_sort",
    )
    filtered = filter_risk_frame(
        working,
        tiers=selected_tiers,
        min_probability=probability_range[0],
        max_probability=probability_range[1],
        reference_query=reference_query,
    )
    if sort_by == "churn_probability":
        filtered = filtered.sort_values("churn_probability", ascending=False)
    elif sort_by == "value_at_risk" and "value_at_risk" in filtered:
        filtered = filtered.sort_values("value_at_risk", ascending=False, na_position="last")
    else:
        filtered = filtered.sort_values("rank", ascending=True)

    counts = tier_counts(frame, boundaries)
    stats = probability_summary(frame)
    high_critical = counts.get("high", 0) + counts.get("critical", 0)
    layout.metric_row(
        [
            ("Scored", format_count(len(frame)), "Stored risk rows"),
            ("Shown", format_count(len(filtered)), "After filters"),
            ("High + critical", format_count(high_critical), "Configured display tiers"),
            ("Mean probability", format_probability(stats["mean"]), "All scored rows"),
        ]
    )
    st.caption(
        f"Run `{summary.get('run_id')}` generated {summary.get('generated_at')}; "
        f"model `{(summary.get('model') or {}).get('model_run_id')}`; "
        f"classification threshold {format_probability(summary.get('classification_threshold'))}."
    )

    chart_cols = st.columns([1, 1])
    with chart_cols[0]:
        st.plotly_chart(charts.tier_chart(counts), width="stretch")
    with chart_cols[1]:
        st.plotly_chart(charts.probability_chart(frame), width="stretch")

    layout.section("Ranked customers", f"{len(filtered):,} rows match the current filters.")
    page_sizes = [25, 50, 100, 250, 500, 1000]
    page_size = st.selectbox("Rows shown", page_sizes, index=2, key="cw_risk_page_size")
    view = tables.risk_view(filtered, limit=int(page_size))
    if view.empty:
        st.info("No customers match the current filters. Widen the tier or probability range.")
    else:
        st.dataframe(view, width="stretch", hide_index=True)
        st.caption(
            "`value_at_risk` is defined only when a real revenue proxy was supplied to Phase 9; it is blank otherwise."
        )
    tables.download_pair(
        filtered,
        prefix=f"churn_risk_{run_id or 'latest'}",
        key="cw_risk_export",
        json_payload=filtered.to_dict(orient="records"),
    )
    st.caption(
        "Exports contain the filtered stored rows only. Spreadsheet-formula neutralisation is applied to string cells."
    )

    warnings = summary.get("warnings") or []
    if warnings:
        with st.expander(f"Run warnings ({len(warnings)})"):
            for warning in warnings:
                st.write(f"- {warning}")
    st.caption(f"Risk table: {frame_path} · summary: {summary_path}")
