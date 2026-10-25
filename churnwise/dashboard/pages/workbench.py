"""Retention Workbench: filter, prioritise, select and export real risk rows.

Selection is a deterministic greedy pass over risk rank under a uniform contact
cost. It is a planning aid, not a projection: no effectiveness is assumed and
no campaign is executed.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

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


def _plan(frame: pd.DataFrame, budget: float, cost: float, max_contacts: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Greedy rank-ordered selection under a uniform contact cost."""

    ordered = frame.sort_values("rank", ascending=True).copy()
    chosen: list[bool] = []
    spend = 0.0
    limited = 0
    for _, row in ordered.iterrows():
        within_budget = spend + cost <= budget + 1e-9
        within_cap = max_contacts <= 0 or sum(chosen) < max_contacts
        if within_budget and within_cap:
            chosen.append(True)
            spend += cost
        else:
            chosen.append(False)
            if not within_budget:
                limited += 1
    ordered["Selected"] = chosen
    selected = ordered[ordered["Selected"]]
    summary = {
        "selected": int(len(selected)),
        "spend": round(spend, 2),
        "remaining": round(max(budget - spend, 0.0), 2),
        "unaffordable": limited,
        "budget": budget,
        "cost": cost,
        "max_contacts": max_contacts,
    }
    return ordered, summary


def _amount(value: float) -> str:
    """Drop the cents only when the amount is whole, so cards stay compact."""

    text = f"{float(value):,.2f}"
    return text[:-3] if text.endswith(".00") else text


def _view(frame: pd.DataFrame, limit: int, *, extras: tuple[str, ...] = ()) -> pd.DataFrame:
    # Only the working columns: rank and explanation metadata stay in the export.
    columns = [
        "Selected",
        "record_reference",
        "churn_probability",
        "risk_tier",
        "predicted_label",
        *extras,
        "value_at_risk",
    ]
    keep = [column for column in columns if column in frame.columns]
    view = frame[keep].head(limit).copy()
    if "churn_probability" in view:
        view["churn_probability"] = pd.to_numeric(view["churn_probability"], errors="coerce").round(4)
    if "value_at_risk" in view:
        view["value_at_risk"] = pd.to_numeric(view["value_at_risk"], errors="coerce").round(2)
    if "risk_tier" in view:
        view["risk_tier"] = view["risk_tier"].astype(str).str.title()
    if "Selected" in view:
        view["Selected"] = view["Selected"].map({True: "Yes", False: "No"})
    if "predicted_label" in view:
        view["predicted_label"] = view["predicted_label"].map(
            lambda value: "Churn" if str(value).lower() in {"yes", "1", "true", "churn"} else "Retained"
        )
    if "value_at_risk" in view:
        if view["value_at_risk"].isna().all():
            view = view.drop(columns=["value_at_risk"])
        else:
            view["value_at_risk"] = view["value_at_risk"].where(view["value_at_risk"].notna(), "—")
    return view.rename(
        columns={
            "record_reference": "Customer",
            "churn_probability": "Churn probability",
            "risk_tier": "Risk tier",
            "predicted_label": "Predicted class",
            "value_at_risk": "Value at risk",
            **tables.CUSTOMER_EXTRA_COLUMNS,
        }
    )


def render(settings: Settings) -> None:
    runs = access.risk_ids(settings)
    run_labels = access.risk_run_labels(settings, runs)
    default_run = access.default_risk_run_id(settings, runs)
    default_index = runs.index(default_run) if default_run in runs else 0
    chosen = st.session_state.get("cw_workbench_run")
    if chosen not in runs:
        st.session_state.pop("cw_workbench_run", None)
        chosen = default_run
    run_id = chosen if runs else None
    summary, _ = access.risk_summary(settings, run_id)
    manifest, _ = access.risk_manifest(settings, run_id)
    population = access.customer_population(settings, run_id)
    raw_columns: tuple[str, ...] = ()
    if population.frame is not None and not population.frame.empty:
        frame = population.frame.copy()
        raw_columns = population.raw_columns
    else:
        frame, _ = access.risk_table(settings, run_id)
    identity = access.identity_from_risk(summary, manifest)
    if run_id:
        identity = replace(identity, run_id=run_labels.get(run_id, identity.run_id))
    layout.page_header(
        "Retention Workbench",
        "Turn scored churn risk into an actionable contact list. Filter, rank, fit customers to a budget, "
        "and export exactly what will be worked. No intervention effect is assumed here.",
        identity,
    )

    if not summary or frame is None or frame.empty:
        layout.empty_state(
            "No risk run is available",
            "Score a population with Phase 9 to populate the workbench. The dashboard does not score records on render.",
            "python -m churnwise.cli score --population val --explain all",
        )
        return

    selector_cols = st.columns([1.8, 2])
    with selector_cols[0]:
        selected_run = st.selectbox(
            "Scoring run",
            runs,
            index=default_index,
            key="cw_workbench_run",
            format_func=lambda value: run_labels.get(value, value),
        )
    with selector_cols[1]:
        st.caption(
            f"Population {nested_get(summary, 'population', 'split', default='not recorded')} split · "
            f"{format_count(len(frame))} stored rows · threshold "
            f"{format_probability(summary.get('classification_threshold'))}"
        )
    if selected_run != run_id:
        st.rerun()

    layout.section("Filter", "Filters are pure operations over the stored table; no rescoring happens.")
    filter_cols = st.columns([1.2, 1.2, 1.0])
    with filter_cols[0]:
        selected_tiers = st.multiselect(
            "Risk tier",
            options=["low", "medium", "high", "critical"],
            default=[],
            key="cw_workbench_tiers",
            help="Empty means every tier.",
        )
    with filter_cols[1]:
        probability_range = st.slider(
            "Probability range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
            key="cw_workbench_probability",
        )
    with filter_cols[2]:
        reference_query = st.text_input("Customer contains", value="", key="cw_workbench_reference")

    sort_options = ["rank", "churn_probability"]
    if "value_at_risk" in frame:
        sort_options.append("value_at_risk")
    sort_by = st.selectbox(
        "Sort by",
        options=sort_options,
        format_func=lambda value: {
            "rank": "Risk rank (highest risk first)",
            "churn_probability": "Churn probability",
            "value_at_risk": "Value at risk",
        }[value],
        key="cw_workbench_sort",
    )
    filtered = filter_risk_frame(
        frame,
        tiers=selected_tiers,
        min_probability=probability_range[0],
        max_probability=probability_range[1],
        reference_query=reference_query,
    )
    if filtered.empty:
        st.info("No customers match the current filters. Widen the tier or probability range.")
        st.stop()

    layout.section(
        "Fit the list to a budget",
        "Customers are taken in risk-rank order at a uniform contact cost. This is capacity planning, not a "
        "causal retention estimate.",
    )
    plan_cols = st.columns(3)
    with plan_cols[0]:
        budget = st.number_input(
            "Campaign budget (value units)",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            key="cw_workbench_budget",
        )
    with plan_cols[1]:
        cost = st.number_input(
            "Cost per contacted customer (value units)",
            min_value=0.0,
            value=5.0,
            step=1.0,
            key="cw_workbench_cost",
        )
    with plan_cols[2]:
        max_contacts = st.number_input(
            "Maximum contacts (0 = no cap)",
            min_value=0,
            max_value=10_000_000,
            value=0,
            step=10,
            key="cw_workbench_cap",
        )

    ordered, plan = _plan(filtered, float(budget), float(cost), int(max_contacts))
    layout.metric_row(
        [
            ("Matching customers", format_count(len(ordered)), "Rows after filters"),
            ("Selected", format_count(plan["selected"]), "Fits budget and contact cap"),
            ("Planned spend", _amount(plan["spend"]), "Value units: selected count × contact cost"),
            ("Remaining budget", _amount(plan["remaining"]), "Value units: budget minus planned spend"),
        ]
    )
    if plan["unaffordable"]:
        st.caption(
            f"{format_count(plan['unaffordable'])} matching customers do not fit the remaining budget or cap."
        )
    st.caption(
        "Risk tiers are the stored boundaries; `predicted_label` is the separate classification-threshold decision."
    )

    view_sorted = ordered.sort_values(
        {"rank": "rank", "churn_probability": "churn_probability", "value_at_risk": "value_at_risk"}[sort_by],
        ascending={"rank": True, "churn_probability": False, "value_at_risk": False}[sort_by],
    )
    selected = view_sorted[view_sorted["Selected"]]

    layout.section(
        "Prioritised customers",
        f"{format_count(plan['selected'])} selected of {format_count(len(ordered))} matching. "
        "Sort the table, then select a row to open that customer's stored assessment.",
    )
    page_cols = st.columns([1, 3])
    with page_cols[0]:
        page_sizes = [25, 50, 100, 250, 500]
        page_size = st.selectbox("Rows shown", page_sizes, index=2, key="cw_workbench_page_size")
    extras = tuple(
        column for column in ("number_customer_service_calls",) if column in raw_columns
    )
    view = _view(view_sorted, int(page_size), extras=extras).reset_index(drop=True)
    event = st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="cw_workbench_table",
        column_config=tables.risk_column_config(),
    )
    selected_rows: list[int] = []
    try:
        selected_rows = list(event.selection.rows or [])
    except (AttributeError, KeyError, TypeError):
        selected_rows = []
    if selected_rows and 0 <= int(selected_rows[0]) < len(view):
        reference = str(view.iloc[int(selected_rows[0])].get("Customer"))
        selected_record = view_sorted[
            view_sorted["record_reference"].astype(str) == reference
        ].head(1)
        if not selected_record.empty:
            selected_row = selected_record.iloc[0]
            probability = pd.to_numeric(selected_row.get("churn_probability"), errors="coerce")
            predicted = str(selected_row.get("predicted_label") or "not recorded")
            predicted = "Churn" if predicted.lower() in {"yes", "1", "true", "churn"} else "Retained"
            with st.container(border=True):
                layout.section("Selected customer", reference)
                detail_cols = st.columns(4)
                detail_cols[0].metric(
                    "Churn probability",
                    format_probability(probability) if pd.notna(probability) else "not available",
                )
                detail_cols[1].metric("Risk tier", str(selected_row.get("risk_tier") or "not recorded").title())
                detail_cols[2].metric("Predicted class", predicted)
                detail_cols[3].metric(
                    "Contact list",
                    "Selected" if bool(selected_row.get("Selected")) else "Not selected",
                )
                selected_profile = tables.profile_items(selected_record, raw_columns)
                if selected_profile:
                    layout.profile_grid(selected_profile[:4])
                else:
                    st.caption("Verified source attributes are unavailable for this scoring run.")
                if st.button("Open assessment", type="primary", key="cw_workbench_open_customer"):
                    st.session_state["cw_customer_run"] = run_id
                    st.session_state["cw_customer_reference"] = reference
                    layout.navigate_to("Customer Risk Explorer")
    if "Value at risk" in view:
        st.caption("`Value at risk` is blank unless a real revenue proxy was supplied to Phase 9.")
    if population.frame is None:
        st.caption(f"Original account attributes are not shown: {population.reason}")

    with st.expander("Run distribution and export notes"):
        counts = tier_counts(frame, tuple(settings.risk_tiers))
        chart_cols = st.columns([1.4, 1])
        with chart_cols[0]:
            st.plotly_chart(charts.tier_chart(counts), width="stretch")
            st.caption("Full run distribution, independent of these filters.")
        with chart_cols[1]:
            share = plan["selected"] / len(ordered) if len(ordered) else 0.0
            st.metric("Share of matching list selected", format_percent(share))
            st.caption(
                f"Run `{summary.get('run_id')}` · population "
                f"{(summary.get('population') or {}).get('split') or 'not recorded'} split · threshold "
                f"{format_probability(summary.get('classification_threshold'))}."
            )

    layout.section("Export", "Exports contain stored rows only; string cells are neutralised for spreadsheets.")
    if selected.empty:
        st.warning("No customers are selected under the current budget and cap, so there is nothing to export.")
    else:
        tables.download_pair(
            selected.drop(columns=["Selected"]),
            prefix=f"retention_selected_{run_id or 'latest'}",
            key="cw_workbench_selected_export",
            json_payload=selected.drop(columns=["Selected"]).to_dict(orient="records"),
        )
        with st.expander("Export the full matching list instead"):
            tables.download_pair(
                ordered.drop(columns=["Selected"]),
                prefix=f"retention_matching_{run_id or 'latest'}",
                key="cw_workbench_matching_export",
                json_payload=None,
                csv_label="Download full list (CSV)",
                json_label="Download full list (JSON)",
            )

    warnings = summary.get("warnings") or []
    if warnings:
        with st.expander(f"Run warnings ({len(warnings)})"):
            for warning in warnings:
                st.write(f"- {warning}")
