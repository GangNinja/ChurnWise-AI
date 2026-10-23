"""Plotly figure builders over stored metrics, explanation and risk artifacts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

INK = "#12263a"
MUTED = "#5b6b7f"
LINE = "#dfe7ef"
TEAL = "#0f766e"
BLUE = "#1d4ed8"
AMBER = "#b45309"
RED = "#b42318"
SLATE = "#475569"
FAMILY_COLORS = {
    "logistic_regression": BLUE,
    "random_forest": TEAL,
    "mlp": AMBER,
}


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
        label = str(family).replace("_", " ").title()
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{label} (AUC {float(roc.get('auc', 0)):.3f})",
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
    return _layout(fig, height=360)


def pr_chart(curves: Mapping[str, Any], *, selected: str | None = None) -> go.Figure:
    if not curves:
        return _empty("No precision-recall payload for this benchmark.")
    fig = go.Figure()
    for family, payload in curves.items():
        pr = (payload or {}).get("pr") or {}
        recall, precision = pr.get("recall"), pr.get("precision")
        if not recall or not precision:
            continue
        label = str(family).replace("_", " ").title()
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{label} (AP {float(pr.get('pr_auc', 0)):.3f})",
                line=dict(color=FAMILY_COLORS.get(family, SLATE), width=3 if family == selected else 1.6),
            )
        )
    fig.update_xaxes(range=[0, 1], title="Recall (positive class)")
    fig.update_yaxes(range=[0, 1], title="Precision (positive class)")
    return _layout(fig, height=360)


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


def tier_chart(counts: Mapping[str, int]) -> go.Figure:
    labels = [label for label in ("low", "medium", "high", "critical") if label in counts]
    if not labels:
        return _empty("No tier counts are available.")
    values = [
        int(counts[label].get("count", 0)) if isinstance(counts[label], Mapping) else int(counts[label])
        for label in labels
    ]
    colors = {"low": "#94a3b8", "medium": BLUE, "high": AMBER, "critical": RED}
    fig = go.Figure(
        go.Bar(
            x=[label.title() for label in labels],
            y=values,
            marker_color=[colors[label] for label in labels],
            text=values,
            textposition="outside",
            hovertemplate="%{x}: %{y} customers<extra></extra>",
        )
    )
    return _layout(fig, height=300, y_title="Customers")


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
        label = family.replace("_", " ").title()
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
    fig.update_yaxes(range=[0, 1.05], title="Fold mean F1")
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
    "cv_chart",
    "feature_dependence_image_paths",
    "importance_chart",
    "pr_chart",
    "probability_chart",
    "roc_chart",
    "sensitivity_chart",
    "tier_chart",
    "training_history_chart",
]
