"""Plotly figure builders over stored metrics, explanation and risk artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

INK = "#12263a"
MUTED = "#5b6b7f"
LINE = "#dfe7ef"
TEAL = "#0b7a70"
BLUE = "#2563eb"
AMBER = "#b45b0b"
RED = "#b42318"
SLATE = "#475569"
TIER_COLORS = {"low": "#94a3b8", "medium": BLUE, "high": AMBER, "critical": RED}
TIER_ORDER = ("low", "medium", "high", "critical")
FAMILY_COLORS = {
    "logistic_regression": BLUE,
    "random_forest": TEAL,
    "mlp": AMBER,
}
FAMILY_SHORT = {"logistic_regression": "LR", "random_forest": "RF", "mlp": "MLP"}


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=13))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def _layout(fig: go.Figure, *, height: int = 330, x_title: str = "", y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=38, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, Segoe UI, sans-serif", color=INK, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        hoverlabel=dict(font_size=12),
        xaxis=dict(title=x_title, gridcolor=LINE, zeroline=False),
        yaxis=dict(title=y_title, gridcolor=LINE, zeroline=False),
    )
    return fig


def importance_chart(rows: Sequence[Mapping[str, Any]], *, top: int = 15) -> go.Figure:
    if not rows:
        return _empty("No feature-importance rows in this explanation run.")
    frame = pd.DataFrame(rows)[:top].iloc[::-1]
    colors = [TEAL if str(item) == "increases_churn" else BLUE for item in frame.get("direction", [])]
    fig = go.Figure(
        go.Bar(
            x=frame["mean_abs_shap"].astype(float),
            y=frame["feature"].astype(str),
            orientation="h",
            marker_color=colors,
            customdata=frame.get("share", pd.Series([None] * len(frame))),
            hovertemplate="%{y}<br>mean |SHAP| %{x:.4f}<extra></extra>",
        )
    )
    return _layout(fig, x_title="Mean |SHAP| (probability space)", height=max(300, 28 * len(frame) + 80))


def roc_chart(curves: Mapping[str, Any], *, selected: str | None = None) -> go.Figure:
    if not curves:
        return _empty("No ROC curve payload for this benchmark.")
    fig = go.Figure()
    for family, payload in curves.items():
        roc = (payload or {}).get("roc") or {}
        fpr, tpr = roc.get("fpr"), roc.get("tpr")
        if not fpr or not tpr:
            continue
        label = FAMILY_SHORT.get(str(family), str(family).replace("_", " ").title())
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{label} ({float(roc.get('auc', 0)):.3f})",
                line=dict(color=FAMILY_COLORS.get(family, SLATE), width=3 if family == selected else 1.6),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#b8c4d0", dash="dot", width=1),
            name="Chance",
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(range=[0, 1], title="False positive rate")
    fig.update_yaxes(range=[0, 1], title="True positive rate")
    fig = _layout(fig, height=360)
    fig.update_layout(
        legend=dict(
            orientation="v",
            x=0.99,
            y=0.03,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.78)",
        )
    )
    return fig


def pr_chart(curves: Mapping[str, Any], *, selected: str | None = None) -> go.Figure:
    if not curves:
        return _empty("No precision-recall payload for this benchmark.")
    fig = go.Figure()
    for family, payload in curves.items():
        pr = (payload or {}).get("pr") or {}
        recall, precision = pr.get("recall"), pr.get("precision")
        if not recall or not precision:
            continue
        label = FAMILY_SHORT.get(str(family), str(family).replace("_", " ").title())
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{label} ({float(pr.get('pr_auc', 0)):.3f})",
                line=dict(color=FAMILY_COLORS.get(family, SLATE), width=3 if family == selected else 1.6),
            )
        )
    fig.update_xaxes(range=[0, 1], title="Recall (positive class)")
    fig.update_yaxes(range=[0, 1], title="Precision (positive class)")
    fig = _layout(fig, height=360)
    fig.update_layout(
        legend=dict(
            orientation="v",
            x=0.99,
            y=0.03,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.78)",
        )
    )
    return fig


def confusion_chart(metrics: Mapping[str, Any], *, title: str = "Confusion matrix") -> go.Figure:
    matrix = ((metrics or {}).get("confusion_matrix") or {}).get("matrix")
    if not matrix or len(matrix) != 2:
        return _empty("No confusion matrix for this family.")
    labels = ["Retained (0)", "Churn (1)"]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=[[0, "#f7fafc"], [1, TEAL]],
            text=[[str(value) for value in row] for row in matrix],
            texttemplate="%{text}",
            hovertemplate="Predicted %{x}<br>Actual %{y}: %{z}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed", title="Actual")
    fig.update_xaxes(title="Predicted")
    fig.update_layout(title=dict(text=title, font=dict(size=13, color=MUTED)))
    return _layout(fig, height=310)


def class_balance_chart(frame: pd.DataFrame) -> go.Figure:
    if frame is None or frame.empty:
        return _empty("No target class counts are available.")
    fig = go.Figure(
        go.Bar(
            x=frame["Class"].astype(str),
            y=frame["Customers"].astype(int),
            marker_color=[BLUE, TEAL, AMBER, RED][: len(frame)],
            text=frame["Customers"].astype(int),
            textposition="outside",
            hovertemplate="%{x}: %{y} rows<extra></extra>",
        )
    )
    fig.update_yaxes(title="Rows")
    return _layout(fig, height=280)


def tier_chart(counts: Mapping[str, Any]) -> go.Figure:
    """Static tier chart retained for simple summaries."""

    labels = [label for label in TIER_ORDER if label in counts]
    if not labels:
        return _empty("No tier counts are available.")
    values = [
        int(counts[label].get("count", 0)) if isinstance(counts[label], Mapping) else int(counts[label])
        for label in labels
    ]
    fig = go.Figure(
        go.Bar(
            x=[label.title() for label in labels],
            y=values,
            marker_color=[TIER_COLORS[label] for label in labels],
            text=values,
            textposition="outside",
            hovertemplate="%{x}: %{y} customers<extra></extra>",
        )
    )
    return _layout(fig, height=300, y_title="Customers")


def tier_distribution_chart(
    counts: Mapping[str, Any],
    *,
    mode: str = "counts",
    selected: str | None = None,
) -> go.Figure:
    """Clickable tier distribution with count/share modes and a selected state."""

    labels = [label for label in TIER_ORDER if label in counts]
    if not labels:
        return _empty("No tier counts are available.")
    count_values = [
        int(counts[label].get("count", 0)) if isinstance(counts[label], Mapping) else int(counts[label])
        for label in labels
    ]
    total = sum(count_values) or 1
    shares = [value / total for value in count_values]
    share_mode = str(mode).lower().startswith("share")
    values = [share * 100 for share in shares] if share_mode else count_values
    colors = [
        (TIER_COLORS[label] if selected in (None, label) else "#cbd5e1")
        for label in labels
    ]
    text = [f"{share:.1%}" if share_mode else f"{value:,}" for value, share in zip(count_values, shares)]
    fig = go.Figure(
        go.Bar(
            x=[label.title() for label in labels],
            y=values,
            customdata=[[label, value, share] for label, value, share in zip(labels, count_values, shares)],
            marker_color=colors,
            marker_line_width=0,
            text=text,
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]:,} customers"
                "<br>%{customdata[2]:.1%} of scored<extra></extra>"
            ),
        )
    )
    top = max(values) if values else 0
    fig.update_yaxes(range=[0, top * 1.18 if top else 1], title="Share of scored customers (%)" if share_mode else "Customers")
    fig.update_layout(
        bargap=0.38, showlegend=False, clickmode="event+select", dragmode="select"
    )
    return _layout(fig, height=300)


def probability_bins(frame: pd.DataFrame, *, bins: int = 20) -> pd.DataFrame:
    """Precomputed probability intervals so a bar click has a stable meaning."""

    if frame is None or frame.empty or "churn_probability" not in frame:
        return pd.DataFrame(columns=["low", "high", "label", "count", "share"])
    values = pd.to_numeric(frame["churn_probability"], errors="coerce").dropna()
    values = values[(values >= 0.0) & (values <= 1.0)]
    if values.empty or bins < 1:
        return pd.DataFrame(columns=["low", "high", "label", "count", "share"])
    edges = [index / bins for index in range(bins + 1)]
    # Keep the displayed interval starts at exactly 0.00: pd.cut's
    # include_lowest widens the first edge to a small negative number.
    binned = values.mask(values == 0.0, 1e-12)
    cuts = pd.cut(binned, bins=edges, include_lowest=False, right=True)
    total = len(values)
    rows = []
    for interval in cuts.cat.categories:
        count = int((cuts == interval).sum())
        low = float(interval.left)
        high = float(interval.right)
        rows.append(
            {
                "low": low,
                "high": high,
                "label": f"{low:.2f}–{high:.2f}",
                "count": count,
                "share": count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def probability_distribution_chart(
    bins: pd.DataFrame,
    *,
    threshold: float | None = None,
    selected: tuple[float, float] | None = None,
) -> go.Figure:
    """Clickable probability histogram, built from precomputed bins."""

    if bins is None or bins.empty:
        return _empty("No scored probabilities are available.")
    colors = []
    for _, row in bins.iterrows():
        overlaps = (
            selected is not None
            and float(row["high"]) > float(selected[0])
            and float(row["low"]) < float(selected[1])
        )
        colors.append(TEAL if overlaps else "rgba(11,122,112,0.38)")
    centers = (bins["low"] + bins["high"]) / 2
    widths = bins["high"] - bins["low"]
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=bins["count"],
            width=widths * 0.9,
            marker_color=colors,
            marker_line_width=0,
            customdata=[[row["label"], row["count"], row["share"], row["low"], row["high"]] for _, row in bins.iterrows()],
            hovertemplate=(
                "Probability %{customdata[0]}<br>%{customdata[1]:,} customers"
                "<br>%{customdata[2]:.1%} of scored<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(range=[0, 1], title="Churn probability")
    fig.update_yaxes(title="Customers")
    if threshold is not None:
        fig.add_vline(
            x=float(threshold),
            line_width=1,
            line_dash="dot",
            line_color=AMBER,
            annotation_text=f"threshold {float(threshold):.2f}",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color=AMBER,
        )
    fig.update_layout(
        bargap=0.12, showlegend=False, clickmode="event+select", dragmode="select"
    )
    return _layout(fig, height=300)


def contribution_chart(rows: Sequence[Mapping[str, Any]]) -> go.Figure:
    """Compact SHAP view for one customer: red raises risk, blue lowers it."""

    if not rows:
        return _empty("No stored feature contributions are available.")
    frame = pd.DataFrame(rows).copy()
    if "contribution" not in frame or "feature" not in frame:
        return _empty("No stored feature contributions are available.")
    frame["contribution"] = pd.to_numeric(frame["contribution"], errors="coerce")
    frame = frame.dropna(subset=["contribution"]).sort_values("contribution")
    labels = []
    for _, row in frame.iterrows():
        label = str(row.get("feature") or "feature")
        active = row.get("active_level")
        if active not in (None, "", "None"):
            label = f"{label} · {active}"
        labels.append(label)
    colors = [RED if value > 0 else BLUE for value in frame["contribution"]]
    fig = go.Figure(
        go.Bar(
            x=frame["contribution"],
            y=labels,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{value:+.3f}" for value in frame["contribution"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>contribution %{x:+.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#cbd5e1")
    span = max(float(frame["contribution"].abs().max()), 0.001)
    fig.update_xaxes(range=[-span * 1.25, span * 1.25], title="SHAP contribution (probability space)")
    fig.update_yaxes(title="")
    fig.update_layout(showlegend=False, bargap=0.3)
    return _layout(fig, height=max(230, 34 * len(frame) + 70))


def threshold_chart(analysis: Mapping[str, Any], family: str | None) -> go.Figure:
    """Validation threshold sweep for the selected family."""

    families = (analysis or {}).get("families") or {}
    payload = families.get(str(family)) if family else None
    if not payload:
        return _empty("No threshold sweep is stored for this family.")
    sweep = payload.get("sweep") or []
    if not sweep:
        return _empty("The threshold sweep is empty.")
    frame = pd.DataFrame(sweep)
    fig = go.Figure()
    for metric, color in (("f1", TEAL), ("precision", BLUE), ("recall", AMBER)):
        if metric not in frame:
            continue
        fig.add_trace(
            go.Scatter(
                x=frame["threshold"],
                y=frame[metric],
                mode="lines",
                name=metric.upper() if metric == "f1" else metric.title(),
                line=dict(color=color, width=2),
                hovertemplate=f"threshold %{{x:.2f}}<br>{metric} %{{y:.3f}}<extra></extra>",
            )
        )
    selected = payload.get("selected_threshold")
    if selected is not None:
        fig.add_vline(
            x=float(selected),
            line_width=1,
            line_dash="dot",
            line_color=SLATE,
            annotation_text=f"selected {float(selected):.2f}",
            annotation_position="top right",
            annotation_font_size=10,
        )
    fig.update_xaxes(range=[0, 1], title="Classification threshold")
    fig.update_yaxes(range=[0, 1.02], title="Validation metric")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return _layout(fig, height=320)


def probability_chart(frame: pd.DataFrame) -> go.Figure:
    if frame is None or frame.empty or "churn_probability" not in frame:
        return _empty("No scored probabilities are available.")
    fig = go.Figure(
        go.Histogram(
            x=frame["churn_probability"].astype(float),
            nbinsx=30,
            marker_color=TEAL,
            opacity=0.85,
            hovertemplate="Probability %{x:.2f}<br>%{y} customers<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 1], title="Churn probability")
    fig.update_yaxes(title="Customers")
    return _layout(fig, height=300)


def segment_risk_chart(frame: pd.DataFrame, feature: str, *, max_segments: int = 8) -> go.Figure:
    """Compare stored mean probabilities across verified source-data segments."""

    if frame is None or frame.empty or feature not in frame or "churn_probability" not in frame:
        return _empty("Verified source attributes are unavailable for segment analysis.")
    values = frame[feature]
    probabilities = pd.to_numeric(frame["churn_probability"], errors="coerce")
    valid = values.notna() & probabilities.notna()
    if not valid.any():
        return _empty("This attribute has no values that can be compared with stored probabilities.")

    segments = values.loc[valid].copy()
    numeric = pd.to_numeric(segments, errors="coerce")
    if numeric.notna().all() and numeric.nunique() > max_segments:
        try:
            buckets = pd.qcut(numeric, q=min(4, numeric.nunique()), duplicates="drop")
            segments = buckets.map(
                lambda item: f"{item.left:g}–{item.right:g}" if pd.notna(item) else "Missing"
            )
        except (ValueError, TypeError):
            segments = segments.astype(str)
    else:
        segments = segments.astype(str)

    data = pd.DataFrame(
        {"segment": segments, "probability": probabilities.loc[valid].to_numpy()}
    )
    grouped = (
        data.groupby("segment", dropna=False, observed=True)["probability"]
        .agg(mean_probability="mean", customers="size")
        .sort_values("mean_probability", ascending=False)
    )
    if len(grouped) > max_segments:
        head = grouped.iloc[: max_segments - 1]
        tail = grouped.iloc[max_segments - 1 :]
        other_count = int(tail["customers"].sum())
        other_mean = float(
            (tail["mean_probability"] * tail["customers"]).sum() / other_count
        )
        grouped = pd.concat(
            [
                head,
                pd.DataFrame(
                    {"mean_probability": [other_mean], "customers": [other_count]},
                    index=["Other segments"],
                ),
            ]
        )
    grouped = grouped.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=grouped["mean_probability"],
            y=grouped.index.astype(str),
            orientation="h",
            marker_color=TEAL,
            customdata=grouped["customers"],
            hovertemplate=(
                "%{y}<br>Mean stored probability %{x:.3f}"
                "<br>%{customdata:,} scored customers<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(range=[0, 1], title="Mean stored probability")
    fig.update_yaxes(title="")
    return _layout(fig, height=max(230, 34 * len(grouped) + 90))


def training_history_chart(history: Mapping[str, Sequence[float]], metric: str = "loss") -> go.Figure:
    series = (history or {}).get(metric) or []
    validation = (history or {}).get(f"val_{metric}") or []
    if not series:
        return _empty(f"No training {metric} history is stored.")
    epochs = list(range(1, len(series) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=series, mode="lines+markers", name=f"Train {metric}", line=dict(color=TEAL, width=2)))
    if validation:
        fig.add_trace(
            go.Scatter(x=epochs[: len(validation)], y=validation, mode="lines+markers", name=f"Validation {metric}", line=dict(color=AMBER, width=2))
        )
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title=metric.title())
    return _layout(fig, height=310)


def cv_chart(cv_by_family: Mapping[str, Any]) -> go.Figure:
    families = list(cv_by_family or {})
    if not families:
        return _empty("No cross-validation summary is stored.")
    fig = go.Figure()
    for family in families:
        cv = (cv_by_family.get(family) or {}).get("metrics") or {}
        f1 = cv.get("f1") or {}
        if not f1:
            continue
        label = FAMILY_SHORT.get(str(family), str(family).replace("_", " ").title())
        fig.add_trace(
            go.Bar(
                x=[label],
                y=[f1.get("mean", 0)],
                error_y=dict(type="data", array=[f1.get("std", 0)], color=SLATE),
                marker_color=FAMILY_COLORS.get(family, SLATE),
                text=[f"{float(f1.get('mean', 0)):.3f}"],
                textposition="outside",
                name=label,
                hovertemplate="%{x}<br>Mean F1 %{y:.3f}<extra></extra>",
            )
        )
    fig.update_yaxes(range=[0, 1.18], title="Fold mean F1")
    fig.update_layout(showlegend=False)
    return _layout(fig, height=320)


def sensitivity_chart(series: Sequence[Mapping[str, Any]]) -> go.Figure:
    if not series:
        return _empty("No sensitivity series was produced for these settings.")
    fig = go.Figure()
    palette = [TEAL, BLUE, AMBER, RED]
    for index, item in enumerate(series):
        points = item.get("points") or []
        if not points:
            continue
        dimension = str(item.get("dimension") or "assumption").replace("_", " ")
        fig.add_trace(
            go.Scatter(
                x=[float(point.get("value", 0)) for point in points],
                y=[float(point.get("expected_prevented_churn", 0)) for point in points],
                mode="lines+markers",
                name=dimension,
                line=dict(color=palette[index % len(palette)], width=2),
                hovertemplate="%{x}<br>projected prevented churn %{y:.4f}<extra></extra>",
            )
        )
    fig.update_xaxes(title="Assumption value")
    fig.update_yaxes(title="Projected prevented churn")
    return _layout(fig, height=340)


def allocation_chart(allocation: Mapping[str, Any], unit: str = "value_units") -> go.Figure:
    if not allocation:
        return _empty("No allocation figures are available.")
    spend = float(allocation.get("allocated_spend", 0) or 0)
    remaining = float(allocation.get("remaining_budget", 0) or 0)
    fig = go.Figure(
        go.Bar(
            x=["Allocated spend", "Remaining"],
            y=[spend, remaining],
            marker_color=[TEAL, "#cbd5e1"],
            text=[f"{spend:,.0f}", f"{remaining:,.0f}"],
            textposition="outside",
            hovertemplate=f"%{{x}}: %{{y:,.2f}} {unit}<extra></extra>",
        )
    )
    return _layout(fig, height=270, y_title=unit)


def financial_bridge_chart(
    projection: Mapping[str, Any], allocation: Mapping[str, Any], unit: str = "value_units"
) -> go.Figure:
    """Show projected saved value less campaign spend as a simple value bridge."""
    saved = projection.get("expected_saved_value")
    spend = allocation.get("allocated_spend")
    net = projection.get("net_benefit")
    if saved is None or spend is None:
        return _empty("Add a monthly customer value to see the financial projection.")
    saved_value = float(saved)
    spend_value = float(spend)
    net_value = float(net) if net is not None else saved_value - spend_value
    fig = go.Figure(
        go.Waterfall(
            x=["Expected saved value", "Allocated campaign spend", "Projected net benefit"],
            measure=["relative", "relative", "total"],
            y=[saved_value, -spend_value, 0],
            text=[
                f"+{saved_value:,.2f}",
                f"−{spend_value:,.2f}",
                f"{net_value:,.2f}",
            ],
            textposition="outside",
            increasing={"marker": {"color": TEAL}},
            decreasing={"marker": {"color": AMBER}},
            totals={"marker": {"color": BLUE}},
            connector={"line": {"color": LINE, "width": 1}},
            customdata=[[unit], [unit], [unit]],
            hovertemplate="%{x}<br>%{y:,.2f} %{customdata[0]}<extra></extra>",
        )
    )
    fig.update_yaxes(title=unit, zeroline=True, zerolinecolor=LINE)
    fig.update_layout(showlegend=False)
    return _layout(fig, height=320)


def comparison_chart(rows: Sequence[Mapping[str, Any]], unit: str = "value_units") -> go.Figure:
    if not rows:
        return _empty("No comparison rows are available.")
    names = [str(row.get("name") or row.get("scenario_id") or "scenario") for row in rows]
    values = [float(row.get("expected_prevented_churn", 0) or 0) for row in rows]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=[TEAL, BLUE, AMBER, RED][: len(names)],
            text=[f"{value:.2f}" for value in values],
            textposition="outside",
            hovertemplate="%{x}<br>projected prevented churn %{y:.4f}<extra></extra>",
        )
    )
    return _layout(fig, height=300, y_title="Projected prevented churn")


def feature_dependence_image_paths(manifest: Mapping[str, Any]) -> list[str]:
    paths = ((manifest or {}).get("paths") or {})
    return [str(value) for key, value in paths.items() if str(key).startswith("dependence_png")]


__all__ = [
    "allocation_chart",
    "class_balance_chart",
    "comparison_chart",
    "confusion_chart",
    "contribution_chart",
    "cv_chart",
    "feature_dependence_image_paths",
    "financial_bridge_chart",
    "importance_chart",
    "pr_chart",
    "probability_bins",
    "probability_chart",
    "probability_distribution_chart",
    "roc_chart",
    "sensitivity_chart",
    "segment_risk_chart",
    "threshold_chart",
    "tier_chart",
    "tier_distribution_chart",
    "training_history_chart",
]
