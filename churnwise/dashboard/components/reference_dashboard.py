"""Presentation helpers for the screenshot-matched executive dashboard."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churnwise.dashboard.analytics import format_count, format_percent
from churnwise.dashboard.components import tables

GREEN = "#477d61"
RED = "#ad4936"
AMBER = "#c58e31"
INK = "#1b2922"
MUTED = "#718079"
LINE = "#e0e9e4"

FILTER_FIELDS = (
    ("Contract", "Contract Type", "All Contracts"),
    ("InternetService", "Internet Service", "All Services"),
    ("PaymentMethod", "Payment Method", "All Methods"),
    ("churn_status", "Churn Status", "All Customers"),
    ("SeniorCitizen", "Senior Citizen", "All"),
)

SERVICE_FIELDS = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f5f7f5; color: #1b2922; }
        .block-container { max-width: 1680px; padding: .8rem 1.25rem 2rem; }
        [data-testid="stSidebar"] { background: #1d3b30; }
        [data-testid="stSidebar"] * { color: #dce8e0; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #dce8e0; }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #afc4b8; font-size: .76rem; }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
          background: #29483c; border-color: #456557; border-radius: 8px;
        }
        [data-testid="stSidebar"] hr { border-color: #426253; }
        [data-testid="stSidebar"] [data-testid="stExpander"] details { border-color: #426253; }
        .cw-brand { padding: .6rem 0 .75rem; }
        .cw-brand .cw-brand-mark {
          width: 2.45rem; height: 2.45rem; margin: 0 .65rem .6rem 0;
          border-radius: 11px; background: #61ab79; vertical-align: middle;
          position: relative;
        }
        .cw-brand .cw-brand-mark:after {
          content: "▂▅▇"; position: absolute; left: .48rem; top: .48rem;
          color: white; font-size: .95rem; letter-spacing: -.12rem;
        }
        .cw-brand .cw-brand-name { font-size: 1.05rem; }
        .cw-brand .cw-brand-sub { font-size: .74rem; color: #aabeb3; }
        h1, h2, h3 { color: #1b2922; }
        [data-testid="stMetric"] { background: white; border: 1px solid #e1e9e4; border-radius: 10px; padding: .85rem 1rem; box-shadow: 0 2px 5px rgba(28,56,43,.07); }
        .cw-ref-hero { background: linear-gradient(110deg,#163a2d,#2f5b45); border-left: 6px solid #73bb82; border-radius: 13px; padding: 1.25rem 1.55rem 1.05rem; color: #f5faf6; margin: 0 0 1rem; box-shadow: 0 3px 10px rgba(24,52,40,.12); }
        .cw-ref-hero-title { font-family: Georgia, 'Times New Roman', serif; font-size: clamp(1.55rem,2.1vw,2.05rem); font-weight: 700; line-height: 1.2; }
        .cw-ref-hero-subtitle { margin-top: .32rem; color: #c5d6cb; font-size: .95rem; }
        .cw-ref-hero-meta { margin-top: .48rem; color: #85d09a; font-size: .8rem; font-weight: 650; }
        .cw-ref-filter-heading { color: #92b59d; font-size: .7rem; font-weight: 750; letter-spacing: .1em; margin: .2rem 0 .7rem; }
        .cw-ref-sidebar-foot { color: #97b09f; border-top: 1px solid #426253; margin-top: .7rem; padding-top: .7rem; font-size: .73rem; line-height: 1.55; }
        .cw-ref-kpi-grid { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 1rem; margin: .85rem 0 1.45rem; }
        .cw-ref-kpi { background: #fff; border: 1px solid #e1e9e4; border-top: 4px solid var(--cw-accent); border-radius: 10px; padding: 1rem 1.1rem .9rem; box-shadow: 0 2px 5px rgba(28,56,43,.07); min-height: 9.6rem; }
        .cw-ref-kpi-label { color: #586c61; text-transform: uppercase; letter-spacing: .09em; font-size: .68rem; font-weight: 750; }
        .cw-ref-kpi-value { font-family: Georgia, 'Times New Roman', serif; color: #1b2922; font-size: clamp(1.55rem,2vw,2.15rem); font-weight: 700; line-height: 1.22; margin: .55rem 0 .18rem; }
        .cw-ref-kpi-note { color: #718079; font-size: .78rem; min-height: 1.15rem; }
        .cw-ref-badge { display: inline-block; margin-top: .5rem; background: #f0f5f1; border-radius: 4px; padding: .2rem .48rem; color: var(--cw-accent); font-size: .65rem; font-weight: 700; }
        .cw-ref-section-title { border-bottom: 1px solid #dfe8e2; color: #1d2923; font-size: 1.04rem; font-weight: 730; margin: 1.05rem 0 .7rem; padding-bottom: .38rem; }
        .cw-ref-chart-title { color: #202b25; font-size: .93rem; font-weight: 720; margin-bottom: .6rem; }
        .cw-ref-insight { background: #f0f6f2; border-left: 3px solid #518469; border-radius: 0 5px 5px 0; padding: .55rem .68rem; color: #708078; font-size: .76rem; line-height: 1.45; margin-bottom: .5rem; }
        [data-testid="stVerticalBlockBorderWrapper"] { background: white; border-color: #e1e9e4 !important; border-radius: 11px; box-shadow: 0 2px 6px rgba(28,56,43,.065); }
        [data-testid="stBaseButton-primary"] { background: #477d61; border-color: #477d61; }
        [data-testid="stBaseButton-primary"]:hover { background: #35684e; border-color: #35684e; }
        [data-testid="stDataFrame"] { border: 1px solid #e1e9e4; border-radius: 9px; }
        @media (max-width: 950px) { .cw-ref-kpi-grid { grid-template-columns: repeat(2,minmax(0,1fr)); } }
        @media (max-width: 600px) { .cw-ref-kpi-grid { grid-template-columns: 1fr; } .block-container { padding-left:.7rem; padding-right:.7rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _numeric_outcome(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    result = frame.copy()
    observed = pd.to_numeric(result.get("observed_churn"), errors="coerce") if "observed_churn" in result else None
    has_observed = observed is not None and bool(observed.notna().any())
    if has_observed:
        result["_churn_outcome"] = observed
    elif "predicted_class" in result:
        result["_churn_outcome"] = pd.to_numeric(result["predicted_class"], errors="coerce")
    else:
        labels = result.get("predicted_label", pd.Series(index=result.index, dtype=object)).astype(str).str.lower()
        result["_churn_outcome"] = labels.isin({"yes", "churn", "1", "true"}).astype(int)
    result["_churn_outcome"] = pd.to_numeric(result["_churn_outcome"], errors="coerce")
    result = result[result["_churn_outcome"].isin([0, 1])].copy()
    result["_outcome_label"] = result["_churn_outcome"].map({0: "Retained", 1: "Churned"})
    return result, has_observed


def _filter_options(frame: pd.DataFrame, field: str) -> list[str]:
    if field == "churn_status":
        return ["All Customers", "Churned", "Retained"]
    if field not in frame:
        return []
    return [str(value) for value in frame[field].dropna().astype(str).sort_values().unique()]


def dashboard_filters(frame: pd.DataFrame, *, run_id: str | None) -> tuple[pd.DataFrame, list[str], bool]:
    """Render and apply the Executive Dashboard's population filters in-page."""

    data, has_observed = _numeric_outcome(frame)
    token = int(st.session_state.get("cw_ref_filter_token", 0))
    active: list[str] = []
    st.markdown('<div class="cw-ref-filter-heading">FILTER CUSTOMERS</div>', unsafe_allow_html=True)
    with st.container(border=True):
        rows = (st.columns(3), st.columns([1, 1, 0.8]))
        for index, (field, label, all_label) in enumerate(FILTER_FIELDS):
            if field != "churn_status" and field not in data:
                continue
            options = _filter_options(data, field)
            if not options:
                continue
            all_option = all_label
            options = [all_option, *[option for option in options if option != all_option]]
            row_index, column_index = (0, index) if index < 3 else (1, index - 3)
            with rows[row_index][column_index]:
                choice = st.selectbox(label, options, key=f"cw_ref_{run_id}_{field}_{token}")
            if choice == all_option:
                continue
            if field == "churn_status":
                target = 1 if choice == "Churned" else 0
                data = data[data["_churn_outcome"] == target]
            elif field == "SeniorCitizen":
                try:
                    target = int(choice)
                except ValueError:
                    target = choice
                data = data[data[field].astype(str) == str(target)]
            else:
                data = data[data[field].astype(str) == choice]
            active.append(f"{label}: {choice}")
        with rows[1][2]:
            if st.button("Reset filters", key="cw_ref_reset", width="stretch"):
                st.session_state["cw_ref_filter_token"] = token + 1
                st.rerun()
        if active:
            st.caption(f"{len(data):,} customers match · " + " · ".join(active))
    return data, active, has_observed


def hero(title: str, subtitle: str, count: int, source_count: int, dataset_label: str, *, has_filters: bool) -> None:
    population_line = (
        f"Showing {format_count(count)} of {format_count(source_count)} customers"
        if has_filters
        else f"Showing all {format_count(count)} customers"
    )
    st.markdown(
        f"""
        <div class="cw-ref-hero">
          <div class="cw-ref-hero-title">{html.escape(title)}</div>
          <div class="cw-ref-hero-subtitle">{html.escape(subtitle)}</div>
          <div class="cw-ref-hero-meta">{population_line} · Dataset: {html.escape(dataset_label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(frame: pd.DataFrame, *, has_observed: bool) -> None:
    if frame.empty:
        values = ("—", "—", "—", "—")
        notes = ("No matching customers", "No matching customers", "No matching customers", "No matching customers")
        rate = 0.0
        retained = 0
    else:
        churned = frame["_churn_outcome"] == 1
        retained_mask = ~churned
        rate = float(churned.mean())
        retained = int(retained_mask.sum())
        charges = pd.to_numeric(frame.get("MonthlyCharges"), errors="coerce") if "MonthlyCharges" in frame else pd.Series(index=frame.index, dtype=float)
        risk_revenue = float(charges[churned].sum()) if charges[churned].notna().any() else None
        avg_churned = float(charges[churned].mean()) if charges[churned].notna().any() else None
        avg_retained = float(charges[retained_mask].mean()) if charges[retained_mask].notna().any() else None
        values = (
            format_percent(rate),
            f"${risk_revenue:,.0f}" if risk_revenue is not None else "Unavailable",
            f"${avg_churned:,.2f}" if avg_churned is not None else "Unavailable",
            format_count(retained),
        )
        notes = (
            f"{format_count(int(churned.sum()))} of {format_count(len(frame))} {'observed' if has_observed else 'predicted'} churned",
            "Monthly charges on churned customers" if risk_revenue is not None else "Monthly charge data unavailable",
            f"vs ${avg_retained:,.2f} for retained" if avg_retained is not None else "Retained comparison unavailable",
            f"{format_percent(retained / len(frame))} retention rate",
        )
    labels = ("Churn rate", "Revenue at risk", "Avg charge — churned", "Retained customers")
    badges = ("Observed" if has_observed else "Model estimate", "Per month", "Compared with retained", "Retention rate")
    colors = (RED, AMBER, "#4f89ad", GREEN)
    cards = []
    for label, value, note, badge, color in zip(labels, values, notes, badges, colors):
        cards.append(
            '<div class="cw-ref-kpi" style="--cw-accent:' + color + '">'
            f'<div class="cw-ref-kpi-label">{html.escape(label)}</div>'
            f'<div class="cw-ref-kpi-value">{html.escape(value)}</div>'
            f'<div class="cw-ref-kpi-note">{html.escape(note)}</div>'
            f'<span class="cw-ref-badge">{html.escape(badge)}</span></div>'
        )
    st.markdown('<div class="cw-ref-kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def _rates(frame: pd.DataFrame, column: str, *, min_count: int = 1) -> pd.DataFrame:
    if column not in frame:
        return pd.DataFrame(columns=[column, "Churn rate", "Customers"])
    grouped = frame.groupby(column, dropna=False).agg(
        churn=("_churn_outcome", "mean"), customers=("_churn_outcome", "size")
    ).reset_index()
    grouped = grouped[grouped["customers"] >= min_count]
    return grouped.rename(columns={column: "Segment", "churn": "Churn rate", "customers": "Customers"})


def _insight(data: pd.DataFrame, label: str, *, high_value: bool = True) -> str:
    if data.empty:
        return "Not enough matching records to compare this segment."
    ordered = data.sort_values("Churn rate", ascending=not high_value)
    first = ordered.iloc[0]
    noun = str(first["Segment"])
    rate = float(first["Churn rate"])
    return (
        f"{noun} has the {'highest' if high_value else 'lowest'} {label} in this selection "
        f"({rate:.1%}, {int(first['Customers']):,} customers). Use it to guide a focused retention review; "
        "this pattern is descriptive, not proof of cause."
    )


def _style_chart(fig: Any, *, height: int = 330) -> Any:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=14, t=20, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color=MUTED, size=12),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#edf1ee", zeroline=False)
    return fig


def _card(title: str, insight: str, chart: Any | None, *, message: str | None = None) -> None:
    with st.container(border=True):
        st.markdown(f'<div class="cw-ref-chart-title">{html.escape(title)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cw-ref-insight">{html.escape(insight)}</div>', unsafe_allow_html=True)
        if chart is not None:
            st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
        else:
            st.info(message or "This scoring run does not include the attributes needed for this chart.")


def _rate_chart(frame: pd.DataFrame, field: str, title: str, *, horizontal: bool = False) -> tuple[Any | None, pd.DataFrame]:
    rates = _rates(frame, field)
    if rates.empty:
        return None, rates
    rates = rates.sort_values("Churn rate", ascending=False)
    rates["Churn %"] = (rates["Churn rate"] * 100).round(1)
    colors = [RED if value >= frame["_churn_outcome"].mean() else GREEN for value in rates["Churn rate"]]
    if horizontal:
        fig = px.bar(rates, x="Churn %", y="Segment", orientation="h", text="Churn %", color_discrete_sequence=[AMBER])
        fig.update_traces(marker_color=colors, texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    else:
        fig = px.bar(rates, x="Segment", y="Churn %", text="Churn %", color_discrete_sequence=[RED])
        fig.update_traces(marker_color=colors, texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Churn rate (%)")
    return _style_chart(fig), rates


def _tenure_group(frame: pd.DataFrame) -> pd.DataFrame:
    if "tenure" not in frame:
        return pd.DataFrame()
    tenure = pd.to_numeric(frame["tenure"], errors="coerce")
    work = frame.assign(_tenure=tenure).dropna(subset=["_tenure"]).copy()
    work["Tenure group"] = pd.cut(
        work["_tenure"], bins=[-1, 12, 24, 48, 72], labels=["0–12m", "12–24m", "24–48m", "48–72m"],
    )
    grouped = work.groupby("Tenure group", observed=False).agg(
        churn=("_churn_outcome", "mean"), customers=("_churn_outcome", "size")
    ).reset_index().rename(columns={"churn": "Churn rate", "customers": "Customers"})
    grouped["Churn %"] = (grouped["Churn rate"] * 100).round(1)
    return grouped


def _support_chart(frame: pd.DataFrame) -> Any | None:
    fields = [field for field in ("InternetService", "TechSupport") if field in frame]
    if len(fields) != 2:
        return None
    work = frame.copy()
    work["Tech support"] = work["TechSupport"].map({"Yes": "Tech Support", "No": "No Tech Support"}).fillna(work["TechSupport"].astype(str))
    grouped = work.groupby(["InternetService", "Tech support"], dropna=False).agg(
        churn=("_churn_outcome", "mean"), customers=("_churn_outcome", "size")
    ).reset_index()
    grouped["Churn %"] = (grouped["churn"] * 100).round(1)
    fig = px.bar(
        grouped, x="InternetService", y="Churn %", color="Tech support", barmode="group",
        color_discrete_map={"No Tech Support": RED, "Tech Support": GREEN},
        labels={"InternetService": "Internet service", "Churn %": "Churn rate (%)"},
    )
    return _style_chart(fig)


def _revenue_chart(frame: pd.DataFrame) -> Any | None:
    if "MonthlyCharges" not in frame:
        return None
    work = frame.copy()
    work["_charges"] = pd.to_numeric(work["MonthlyCharges"], errors="coerce")
    grouped = work.dropna(subset=["_charges"]).groupby("_outcome_label")["_charges"].sum().reindex(["Retained", "Churned"]).dropna()
    if grouped.empty:
        return None
    fig = go.Figure(go.Pie(
        labels=grouped.index.tolist(), values=grouped.values.tolist(), hole=0.58,
        marker=dict(colors=[GREEN, RED], line=dict(color="white", width=3)),
        texttemplate="%{percent}", textposition="outside", hovertemplate="%{label}<br>$%{value:,.0f}/month<extra></extra>",
    ))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.08, x=0.05))
    return _style_chart(fig, height=350)


def _charge_histogram(frame: pd.DataFrame) -> Any | None:
    if "MonthlyCharges" not in frame:
        return None
    work = frame.copy()
    work["_charges"] = pd.to_numeric(work["MonthlyCharges"], errors="coerce")
    work = work.dropna(subset=["_charges"])
    if work.empty:
        return None
    fig = px.histogram(
        work, x="_charges", color="_outcome_label", barmode="group", nbins=12,
        color_discrete_map={"Retained": GREEN, "Churned": RED},
        labels={"_charges": "Monthly charge (USD)", "_outcome_label": "Outcome"},
    )
    return _style_chart(fig)


def _service_chart(frame: pd.DataFrame) -> Any | None:
    present = [field for field in SERVICE_FIELDS if field in frame]
    if not present:
        return None
    work = frame.copy()
    work["Number of services"] = work[present].apply(
        lambda row: sum(str(value).strip().lower() in {"yes", "1", "true"} for value in row), axis=1
    )
    grouped = work.groupby("Number of services").agg(
        churn=("_churn_outcome", "mean"), customers=("_churn_outcome", "size")
    ).reset_index()
    fig = go.Figure()
    fig.add_bar(x=grouped["Number of services"], y=grouped["churn"] * 100, name="Churn rate (%)", marker_color=AMBER)
    fig.add_scatter(x=grouped["Number of services"], y=grouped["customers"], name="Customers", mode="lines+markers", yaxis="y2", line=dict(color=INK, width=3))
    fig.update_layout(xaxis_title="Number of services", yaxis_title="Churn rate (%)", yaxis2=dict(title="Customers", overlaying="y", side="right", showgrid=False), legend=dict(orientation="h", y=1.12))
    return _style_chart(fig, height=360)


def _priority_list(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"Contract", "PaymentMethod"}.issubset(frame.columns):
        return pd.DataFrame()
    result = frame.groupby(["Contract", "PaymentMethod"], dropna=False).agg(
        Customers=("_churn_outcome", "size"), churn_rate=("_churn_outcome", "mean")
    ).reset_index().sort_values(["churn_rate", "Customers"], ascending=[False, False]).head(8)
    result["Churn rate"] = (result.pop("churn_rate") * 100).round(1)
    return result.rename(columns={"Contract": "Contract", "PaymentMethod": "Payment"})


def charts(frame: pd.DataFrame, *, has_observed: bool) -> None:
    basis = "observed" if has_observed else "predicted"
    if frame.empty:
        st.markdown('<div class="cw-ref-section-title">Churn by Customer Segment</div>', unsafe_allow_html=True)
        st.info("No customers match the current filters. Reset the sidebar filters to restore the selected scoring run.")
        return
    st.markdown('<div class="cw-ref-section-title">Churn by Customer Segment</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        chart, rates = _rate_chart(frame, "Contract", "Churn Rate by Contract Type")
        _card("Churn Rate by Contract Type", _insight(rates, f"{basis} churn rate"), chart)
    with col2:
        rates = _tenure_group(frame)
        chart = None
        if not rates.empty:
            chart = px.bar(rates, x="Tenure group", y="Churn %", text="Churn %", color_discrete_sequence=[RED])
            chart.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            chart = _style_chart(chart)
        _card("Churn Rate by Tenure Group", _insight(rates.rename(columns={"Tenure group": "Segment"}), f"{basis} churn rate"), chart)

    st.markdown('<div class="cw-ref-section-title">Churn by Service and Payment</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        chart, rates = _rate_chart(frame, "PaymentMethod", "Churn Rate by Payment Method", horizontal=True)
        _card("Churn Rate by Payment Method", _insight(rates, f"{basis} churn rate"), chart)
    with col2:
        _card(
            "Churn Rate: Internet Service + Tech Support",
            "Compare support groups within each internet service type. This is a descriptive comparison, not a measured intervention effect.",
            _support_chart(frame),
        )

    st.markdown('<div class="cw-ref-section-title">Revenue Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        _card("Monthly Revenue Split", "Monthly charges are summed by observed or predicted outcome for this selected run.", _revenue_chart(frame))
    with col2:
        _card("Monthly Charges Distribution — Churned vs Retained", "Compare the monthly charge profile of the two outcome groups.", _charge_histogram(frame))

    st.markdown('<div class="cw-ref-section-title">High-Risk Segment Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.05, 1], gap="large")
    with col1:
        service = _service_chart(frame)
        service_insight = "Service count uses the available yes/no service features in this run. Higher service counts are not evidence that adding a service will prevent churn."
        _card("Churn Rate by Number of Services", service_insight, service)
    with col2:
        priority = _priority_list(frame)
        with st.container(border=True):
            st.markdown('<div class="cw-ref-chart-title">Retention Priority List</div>', unsafe_allow_html=True)
            st.markdown('<div class="cw-ref-insight">Highest churn-rate contract and payment segments in the selected population. Use as a review queue, not a causal recommendation.</div>', unsafe_allow_html=True)
            if priority.empty:
                st.info("Contract and payment fields are unavailable for this scoring run.")
            else:
                st.dataframe(
                    priority,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Customers": st.column_config.NumberColumn("Customers", format="%,d"),
                        "Churn rate": st.column_config.ProgressColumn("Churn rate", min_value=0, max_value=100, format="%.1f%%"),
                    },
                )


__all__ = ["dashboard_filters", "hero", "metric_cards", "charts", "SERVICE_FIELDS"]
