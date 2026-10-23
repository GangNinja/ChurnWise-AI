"""Model evaluation: Phase 7 benchmark evidence, test metrics and stability."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import format_count, format_percent
from churnwise.dashboard.components import charts, layout, tables


def _metric_pair(families: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for family, payload in families.items():
        val = payload.get("validation_metrics") or {}
        test = payload.get("test_metrics") or {}
        rows.append(
            {
                "Family": payload.get("label") or family,
                "Threshold": (payload.get("threshold") or {}).get("selected"),
                "Validation F1": val.get("f1"),
                "Validation ROC-AUC": val.get("roc_auc"),
                "Test F1": test.get("f1"),
                "Test ROC-AUC": test.get("roc_auc"),
                "Test precision": test.get("precision"),
                "Test recall": test.get("recall"),
                "Test Brier": test.get("brier"),
            }
        )
    frame = pd.DataFrame(rows)
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").round(4)
    return frame


def render(settings: Settings) -> None:
    runs = access.benchmark_ids(settings)
    run_id = st.selectbox("Benchmark run", runs, index=0, key="cw_eval_run") if runs else None
    metrics, metrics_path = access.benchmark_metrics(settings, run_id)
    curves, curves_path = access.benchmark_curves(settings, run_id)
    identity = access.identity_from_benchmark(metrics)
    layout.page_header(
        "Model evaluation",
        "Held-out test metrics for Logistic Regression, Random Forest and the Keras MLP. "
        "Model and threshold selection used validation rows only; the test split was scored once.",
        identity,
    )
    if not metrics:
        layout.empty_state(
            "No benchmark artifact exists",
            "Run the Phase 7 benchmark to compare families on the frozen test split.",
            "python -m churnwise.cli evaluate --cv-folds 5",
        )
        return

    selection = metrics.get("selection") or {}
    families = metrics.get("families") or {}
    selected_family = selection.get("selected_family")
    selected = families.get(selected_family) or {}
    selected_test = selected.get("test_metrics") or {}
    layout.status_line(
        "Model selection",
        str(selection.get("selected_label") or selected_family or "not recorded"),
        "ok",
        f"selected on validation {selection.get('selection_metric', 'metric')} "
        f"{selection.get('selected_validation_value')}",
    )
    layout.metric_row(
        [
            ("Selected family", selected.get("label") or selected_family or "not available", "Phase 7 selection"),
            ("Test F1", f"{selected_test.get('f1', float('nan')):.4f}", "Positive churn class"),
            (
                "Test ROC-AUC",
                f"{selected_test.get('roc_auc', float('nan')):.4f}",
                "Threshold-independent ranking quality",
            ),
            (
                "Test Brier",
                f"{selected_test.get('brier', float('nan')):.4f}",
                "Probability calibration summary (lower is better)",
            ),
        ]
    )
    st.caption(
        "Test metrics are computed at the family's validation-selected threshold. "
        f"Benchmark `{metrics.get('benchmark_run_id')}` generated {metrics.get('generated_at')}."
    )

    layout.section("Validation vs test", "The two columns are deliberately separate; test values are not selection evidence.")
    st.dataframe(_metric_pair(families), width="stretch", hide_index=True)
    comparison = metrics.get("comparison_table") or []
    if comparison:
        with st.expander("Full test metric table"):
            st.dataframe(tables.metrics_frame(comparison), width="stretch", hide_index=True)

    layout.section("Curves", "Points are the stored Phase 7 curve payload, not redrawn from test rows.")
    if curves:
        tab_roc, tab_pr = st.tabs(["ROC", "Precision–recall"])
        with tab_roc:
            st.plotly_chart(charts.roc_chart(curves, selected=selected_family), width="stretch")
        with tab_pr:
            st.plotly_chart(charts.pr_chart(curves, selected=selected_family), width="stretch")
        st.caption(f"Curves payload: {curves_path}")
    else:
        st.info("The benchmark has no stored curve payload.")

    layout.section("Confusion matrices", "Test split at each family's selected threshold.")
    if families:
        cols = st.columns(min(3, len(families)))
        for col, (family, payload) in zip(cols, families.items()):
            with col:
                st.plotly_chart(
                    charts.confusion_chart(
                        payload.get("test_metrics") or {},
                        title=payload.get("label") or family,
                    ),
                    width="stretch",
                )
    if selected_test:
        matrix = (selected_test.get("confusion_matrix") or {})
        st.caption(
            f"Selected family counts: TN {matrix.get('tn')}, FP {matrix.get('fp')}, "
            f"FN {matrix.get('fn')}, TP {matrix.get('tp')}."
        )

    layout.section("Cross-validation stability", "Five-fold estimates on train+validation rows; test rows are excluded.")
    cv_by_family = {family: payload.get("cv") or {} for family, payload in families.items()}
    if any((item or {}).get("status") == "completed" for item in cv_by_family.values()):
        st.plotly_chart(charts.cv_chart(cv_by_family), width="stretch")
        rows = []
        for family, payload in families.items():
            cv = payload.get("cv") or {}
            f1 = (cv.get("metrics") or {}).get("f1") or {}
            rows.append(
                {
                    "Family": payload.get("label") or family,
                    "Folds": f"{cv.get('folds_completed', '?')}/{cv.get('folds_requested', '?')}",
                    "F1 mean": f1.get("mean"),
                    "F1 std": f1.get("std"),
                    "Fold-safe preprocessing": (cv.get("preprocessing") or {}).get("fold_safe", "not recorded"),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No completed cross-validation summary is stored for this benchmark.")

    limitations = metrics.get("limitations") or []
    if limitations:
        with st.expander("Benchmark limitations"):
            for item in limitations:
                st.write(f"- {item}")
    st.caption(f"Metrics artifact: {metrics_path}")
