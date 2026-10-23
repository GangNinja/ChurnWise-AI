"""Retention simulator: Phase 10 projections under explicit assumptions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import actions
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import (
    dict_to_json_bytes,
    format_count,
    format_money,
    format_percent,
    format_probability,
    nested_get,
)
from churnwise.dashboard.components import charts, layout, tables

STRATEGIES = {
    "highest_risk": "Highest risk",
    "risk_threshold": "Risk threshold",
    "risk_category": "Risk category",
    "budget_constrained": "Budget constrained",
}


def _normalize(payload: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    payload = payload or {}
    if "result" in payload:
        return payload.get("result") or {}, payload.get("comparison"), payload.get("sensitivity")
    return dict(payload), None, None


def _persisted_siblings(path: Path | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    root = path.parent
    return access.load_json(root / "comparison.json"), access.load_json(root / "sensitivity.json")


def _defaults(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    result = payload or {}
    assumptions = result.get("assumptions") or {}
    targeting = result.get("targeting") or {}
    return {
        "budget": float(result.get("budget") or 1000.0),
        "cost_per_contact": float(assumptions.get("cost_per_contact") or 5.0),
        "effectiveness": float(assumptions.get("effectiveness") or 0.30),
        "monthly_value": assumptions.get("monthly_value"),
        "horizon_months": float(assumptions.get("horizon_months") or 6.0),
        "value_unit": str(assumptions.get("value_unit") or "value_units"),
        "strategy": str(targeting.get("strategy") or "highest_risk"),
        "min_risk_probability": targeting.get("min_risk_probability"),
        "risk_tiers": tuple(targeting.get("risk_tiers") or ()),
        "max_contacts": targeting.get("max_contacts"),
    }


def _metric_block(result: Mapping[str, Any]) -> None:
    allocation = result.get("allocation") or {}
    projection = result.get("projection") or {}
    assumptions = result.get("assumptions") or {}
    unit = str(assumptions.get("value_unit") or "value_units")
    layout.metric_row(
        [
            ("Selected customers", format_count(allocation.get("selected_count")), "Contacts under the scenario"),
            ("Allocated spend", format_money(allocation.get("allocated_spend"), unit), "Actual selected costs"),
            ("Remaining budget", format_money(allocation.get("remaining_budget"), unit), "Budget minus spend"),
            (
                "Projected prevented churn",
                format_probability(projection.get("expected_prevented_churn")),
                "Σ(p × assumed effectiveness); a fractional expectation",
            ),
        ]
    )
    layout.metric_row(
        [
            (
                "Expected saved value",
                format_money(projection.get("expected_saved_value"), unit)
                if projection.get("financial_metrics_available")
                else "undefined",
                "Requires a supplied monthly value",
            ),
            (
                "Net benefit",
                format_money(projection.get("net_benefit"), unit)
                if projection.get("financial_metrics_available")
                else "undefined",
                "Saved value minus spend",
            ),
            (
                "ROI",
                format_percent(projection.get("roi"))
                if projection.get("financial_metrics_available") and projection.get("roi") is not None
                else "undefined",
                "Net benefit / spend",
            ),
            (
                "Break-even effectiveness",
                format_percent(projection.get("break_even_effectiveness"))
                if projection.get("break_even_effectiveness") is not None
                else "undefined",
                "Assumed effectiveness needed to cover spend",
            ),
        ]
    )
    undefined = projection.get("undefined_metrics") or {}
    if undefined:
        with st.expander(f"Undefined metrics ({len(undefined)})"):
            for key, reason in undefined.items():
                st.write(f"- **{key}**: {reason}")


def render(settings: Settings) -> None:
    runs = access.risk_ids(settings)
    if not runs:
        layout.page_header(
            "Retention simulator",
            "Campaign projections require a Phase 9 risk run. No risk run is available.",
            access.RunIdentity(source="campaign simulation"),
        )
        layout.empty_state(
            "No risk run to simulate against",
            "Score a population first; the simulator only combines stored probabilities with your assumptions.",
            "python -m churnwise.cli score --population val --explain none",
        )
        return

    persisted, persisted_path = access.simulation_result(settings)
    persisted_result, persisted_comparison, persisted_sensitivity = _normalize(persisted)
    sibling_comparison, sibling_sensitivity = _persisted_siblings(persisted_path)
    persisted_comparison = persisted_comparison or sibling_comparison
    persisted_sensitivity = persisted_sensitivity or sibling_sensitivity
    computed = st.session_state.get("cw_simulation_run")
    computed_result, computed_comparison, computed_sensitivity = _normalize(computed)
    active = computed_result or persisted_result
    active_comparison = computed_comparison or persisted_comparison
    active_sensitivity = computed_sensitivity or persisted_sensitivity
    defaults = _defaults(active)
    source_risk = str(nested_get(active, "input", "risk_run_id", default="") or "")
    default_run_index = runs.index(source_risk) if source_risk in runs else 0
    identity = access.identity_from_simulation(computed_result if computed else persisted_result)
    if not identity.run_id and active:
        identity = access.identity_from_simulation(active)

    layout.page_header(
        "Retention simulator",
        "Combine stored churn probabilities with explicit economic assumptions to project selection, spend and "
        "expected prevented churn. Every financial and retention value is hypothetical.",
        identity,
    )
    if st.session_state.get("cw_simulation_error"):
        st.error(st.session_state.pop("cw_simulation_error"))
    if computed:
        st.info("Showing a projection computed in this session. It is not persisted until you run the CLI with persistence.")
    elif persisted:
        st.caption(
            f"Showing the latest persisted scenario `{persisted_result.get('scenario_id')}`. "
            "Change assumptions and run a new projection to replace this view."
        )

    layout.disclaimer(
        "Hypothetical projection, not observed retention. Assumed effectiveness is an input, not measured uplift; "
        "no causal model is fitted and no campaign is executed."
    )

    with st.expander("Assumptions and targeting", expanded=not bool(active)):
        with st.form("cw_simulation_form"):
            cols = st.columns(3)
            with cols[0]:
                risk_run = st.selectbox(
                    "Risk run", runs, index=default_run_index, key="cw_sim_risk_run"
                )
                budget = st.number_input("Budget", min_value=0.0, value=float(defaults["budget"]), step=100.0)
                cost = st.number_input(
                    "Cost per contacted customer", min_value=0.0, value=float(defaults["cost_per_contact"]), step=1.0
                )
                effectiveness = st.slider(
                    "Assumed effectiveness", min_value=0.0, max_value=1.0, value=float(defaults["effectiveness"]), step=0.01
                )
                horizon = st.number_input(
                    "Value horizon (months)", min_value=1.0, value=float(defaults["horizon_months"]), step=1.0
                )
            with cols[1]:
                strategy = st.selectbox(
                    "Targeting strategy",
                    list(STRATEGIES),
                    index=list(STRATEGIES).index(defaults["strategy"]) if defaults["strategy"] in STRATEGIES else 0,
                    format_func=lambda value: STRATEGIES[value],
                )
                min_risk = st.number_input(
                    "Minimum risk probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(defaults["min_risk_probability"] or 0.30),
                    step=0.01,
                )
                tiers = st.multiselect(
                    "Targeted risk tiers",
                    options=["low", "medium", "high", "critical"],
                    default=list(defaults["risk_tiers"]),
                    help="Leave empty to target every tier (subject to the probability floor).",
                )
                max_contacts = st.number_input(
                    "Maximum contacts (0 = no cap)",
                    min_value=0,
                    max_value=10_000_000,
                    value=int(defaults["max_contacts"] or 0),
                    step=10,
                )
            with cols[2]:
                use_value = st.checkbox(
                    "Supply a monthly value",
                    value=defaults["monthly_value"] is not None,
                    help="Without a value, saved value, net benefit, ROI and break-even stay undefined.",
                )
                monthly_value = st.number_input(
                    "Assumed monthly value per retained customer",
                    min_value=0.0,
                    value=float(defaults["monthly_value"] or 50.0),
                    step=5.0,
                    disabled=not use_value,
                )
                value_unit = st.text_input("Value unit", value=defaults["value_unit"], max_chars=24)
                compare = st.checkbox("Compare with a higher-budget alternative", value=False)
                alt_budget = st.number_input(
                    "Alternative budget", min_value=0.0, value=float(defaults["budget"]) * 1.5, step=100.0, disabled=not compare
                )
                alt_effectiveness = st.slider(
                    "Alternative effectiveness",
                    min_value=0.0,
                    max_value=1.0,
                    value=min(float(defaults["effectiveness"]) + 0.05, 1.0),
                    step=0.01,
                    disabled=not compare,
                )
            submitted = st.form_submit_button("Run projection", type="primary")
        if submitted:
            try:
                scenario = actions.run_campaign_projection(
                    settings,
                    risk_run_id=risk_run,
                    budget=budget,
                    cost_per_contact=cost,
                    effectiveness=effectiveness,
                    monthly_value=monthly_value if use_value else None,
                    horizon_months=horizon,
                    value_unit=value_unit.strip() or "value_units",
                    strategy=strategy,
                    min_risk_probability=min_risk,
                    risk_tiers=tuple(tiers),
                    max_contacts=int(max_contacts) or None,
                    comparison=(
                        {
                            "budget": alt_budget,
                            "effectiveness": alt_effectiveness,
                            "cost_per_contact": cost,
                            "monthly_value": monthly_value if use_value else None,
                            "horizon_months": horizon,
                            "value_unit": value_unit.strip() or "value_units",
                            "strategy": strategy,
                            "min_risk_probability": min_risk,
                            "risk_tiers": tuple(tiers),
                            "max_contacts": int(max_contacts) or None,
                        }
                        if compare
                        else None
                    ),
                )
            except actions.DashboardActionError as exc:
                st.session_state["cw_simulation_error"] = (
                    f"{exc.code}: {exc.message}" + (f" — {exc.remediation}" if exc.remediation else "")
                )
                st.rerun()
            else:
                st.session_state["cw_simulation_run"] = scenario
                st.rerun()

    if not active:
        layout.empty_state(
            "No simulation has been computed or persisted",
            "Set the assumptions above and run a projection, or use the Phase 10 CLI.",
            "python -m churnwise.cli simulate --budget 1000 --cost-per-contact 5 --effectiveness 0.30 --monthly-value 50",
        )
        return

    scenario_name = nested_get(active, "scenario", "name", default="unnamed scenario")
    layout.section(
        "Projected outcome",
        f"Scenario `{scenario_name}` · strategy {nested_get(active, 'targeting', 'strategy', default='not recorded')}.",
    )
    _metric_block(active)
    warning_cols = st.columns([1, 1])
    with warning_cols[0]:
        st.plotly_chart(
            charts.allocation_chart(active.get("allocation") or {}, nested_get(active, "assumptions", "value_unit", default="value_units")),
            width="stretch",
        )
    with warning_cols[1]:
        population = active.get("population") or {}
        rows = [
            {"Measure": "Assessed", "Value": population.get("assessed")},
            {"Measure": "Eligible", "Value": population.get("eligible")},
            {"Measure": "Selected", "Value": population.get("selected")},
            {"Measure": "Excluded", "Value": population.get("excluded")},
            {"Measure": "Untargeted (budget)", "Value": (active.get("allocation") or {}).get("unaffordable_count")},
            {"Measure": "Untargeted (contact cap)", "Value": (active.get("allocation") or {}).get("contact_limited_count")},
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Counts reconcile" if population.get("counts_reconcile") else "Counts did not reconcile")

    layout.section("Assumptions applied")
    assumptions = active.get("assumptions") or {}
    targeting = active.get("targeting") or {}
    assumption_rows = [
        ("Budget", format_money(active.get("budget"), assumptions.get("value_unit", "value_units"))),
        ("Cost per contact", format_money(assumptions.get("cost_per_contact"), assumptions.get("value_unit", "value_units"))),
        ("Assumed effectiveness", format_percent(assumptions.get("effectiveness"))),
        (
            "Monthly value",
            format_money(assumptions.get("monthly_value"), assumptions.get("value_unit", "value_units"))
            if assumptions.get("monthly_value") is not None
            else "not supplied",
        ),
        ("Horizon", f"{assumptions.get('horizon_months', '?')} months"),
        ("Minimum risk", format_probability(targeting.get("min_risk_probability"))),
        ("Risk tiers", ", ".join(targeting.get("risk_tiers") or []) or "all tiers"),
        ("Contact cap", targeting.get("max_contacts") or "none"),
    ]
    st.dataframe(pd.DataFrame(assumption_rows, columns=["Assumption", "Value"]), width="stretch", hide_index=True)

    layout.section("Sensitivity", "Each point is a genuine recomputation with one assumption changed.")
    series = (active_sensitivity or {}).get("series") or []
    if series:
        tabs = st.tabs(["Chart", "Table"])
        with tabs[0]:
            st.plotly_chart(charts.sensitivity_chart(series), width="stretch")
        with tabs[1]:
            frame = tables.sensitivity_frame(series)
            st.dataframe(frame, width="stretch", hide_index=True)
        warnings = (active_sensitivity or {}).get("warnings") or []
        if warnings:
            st.caption(" ".join(str(item) for item in warnings))
    else:
        st.info("No sensitivity series is attached to this result.")

    if active_comparison and active_comparison.get("rows"):
        layout.section("Scenario comparison", "Assumption differences stay visible; neither row is evidence of real performance.")
        frame = pd.DataFrame(active_comparison["rows"])
        keep = [
            column
            for column in (
                "name",
                "strategy",
                "budget",
                "cost_per_contact",
                "effectiveness",
                "selected_count",
                "allocated_spend",
                "expected_prevented_churn",
                "cost_per_expected_prevented_churn",
                "roi",
            )
            if column in frame
        ]
        st.dataframe(frame[keep], width="stretch", hide_index=True)
        st.plotly_chart(charts.comparison_chart(active_comparison["rows"]), width="stretch")
        if active_comparison.get("shared_inputs") is False:
            st.warning("The scenarios do not share all risk inputs; see the comparison notes.")
        for text in active_comparison.get("notes") or []:
            st.caption(str(text))

    selected = active.get("selected_customers") or []
    if selected:
        layout.section("Selected customers", f"First {min(len(selected), 100)} of {len(selected)} selected records.")
        st.dataframe(tables.selected_customers_frame(selected), width="stretch", hide_index=True)

    warnings = active.get("warnings") or []
    if warnings:
        with st.expander(f"Projection warnings ({len(warnings)})"):
            for warning in warnings:
                st.write(f"- {warning}")
    limitations = active.get("limitations") or []
    if limitations:
        with st.expander("Projection limitations"):
            for item in limitations:
                st.write(f"- {item}")

    download_payload = computed if computed else persisted
    if download_payload:
        st.download_button(
            "Download result JSON",
            data=dict_to_json_bytes(download_payload),
            file_name=f"simulation_{identity.run_id or 'result'}.json",
            mime="application/json",
            key="cw_simulation_download",
        )
    if persisted_path:
        st.caption(f"Persisted artifact: {persisted_path}")
