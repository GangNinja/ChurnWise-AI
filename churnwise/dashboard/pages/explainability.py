"""Explainability: global and local SHAP evidence from Phase 8 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import format_probability
from churnwise.dashboard.components import charts, layout, tables


def _local_payload(settings: Settings, run_id: str, row_position: int) -> tuple[dict[str, Any] | None, Path | None]:
    path = Path(settings.artifact_dir) / "explanations" / run_id / "local" / f"row_{int(row_position):05d}.json"
    return access.load_json(path), path


def _local_plot(settings: Settings, run_id: str, row_position: int) -> Path | None:
    path = Path(settings.artifact_dir) / "explanations" / run_id / "local" / f"row_{int(row_position):05d}.png"
    return path if path.is_file() else None


def _png_path(settings: Settings, run_id: str, name: str) -> Path | None:
    path = Path(settings.artifact_dir) / "explanations" / run_id / name
    return path if path.is_file() else None


def render(settings: Settings) -> None:
    runs = access.explanation_ids(settings)
    run_id = st.selectbox("Explanation run", runs, index=0, key="cw_xai_run") if runs else None
    global_payload, global_path = access.explanation_global(settings, run_id)
    manifest, manifest_path = access.explanation_manifest(settings, run_id)
    identity = access.identity_from_explanation(manifest)
    layout.page_header(
        "Explainability",
        "Global SHAP importance and stored customer-level explanations. SHAP describes model output, "
        "not causal effects; changing a feature is not guaranteed to change churn by the shown amount.",
        identity,
    )
    if not global_payload:
        layout.empty_state(
            "No SHAP explanation artifact exists",
            "Run the Phase 8 explain command over the selected model. The dashboard never computes SHAP itself.",
            "python -m churnwise.cli explain",
        )
        return

    sample = global_payload.get("sample") or {}
    explainer = global_payload.get("explainer") or {}
    layout.status_line(
        "Explainer",
        str(explainer.get("name") or "not recorded"),
        "ok" if not explainer.get("selected_by_fallback") else "warn",
        str(explainer.get("output_space") or ""),
    )
    layout.metric_row(
        [
            ("Family", global_payload.get("family") or "not available", "Explained model"),
            ("Sampled rows", sample.get("sampled_rows", "not available"), "Seeded sample used for global values"),
            ("Source features", explainer.get("source_features", "not available"), "After one-hot grouping"),
            (
                "Additivity error",
                f"{float(global_payload.get('additivity_error', 0)):.2e}"
                if global_payload.get("additivity_error") is not None
                else "not recorded",
                "Sum of SHAP values vs model output",
            ),
        ]
    )
    st.caption(
        f"Sample split: {sample.get('split', 'not recorded')}; sampling: {sample.get('sampling', 'not recorded')}; "
        f"test rows used: {sample.get('test_rows_used', 'not recorded')}."
    )
    if explainer.get("warnings"):
        for warning in explainer["warnings"]:
            st.warning(str(warning))

    layout.section("Global drivers", "Grouped mean absolute SHAP values computed from the stored explanation payload.")
    importance = global_payload.get("feature_importance") or []
    if importance:
        st.plotly_chart(charts.importance_chart(importance), width="stretch")
        with st.expander("Feature-importance table"):
            frame = pd.DataFrame(importance)
            keep = [column for column in ("rank", "feature", "kind", "mean_abs_shap", "mean_shap", "share", "direction") if column in frame]
            st.dataframe(frame[keep], width="stretch", hide_index=True)
    else:
        st.info("This run stored no grouped feature importance.")

    image_cols = st.columns(2)
    importance_png = _png_path(settings, str(run_id), "importance.png")
    summary_png = _png_path(settings, str(run_id), "summary.png")
    with image_cols[0]:
        if importance_png:
            st.image(str(importance_png), caption="Global mean |SHAP| (Phase 8 figure)")
        else:
            st.caption("No importance figure stored.")
    with image_cols[1]:
        if summary_png:
            st.image(str(summary_png), caption="SHAP beeswarm (Phase 8 figure)")
        else:
            st.caption("No beeswarm figure stored.")

    dependence = []
    if isinstance(global_payload.get("plots"), dict):
        dependence = [
            (name, Path(path))
            for name, path in global_payload["plots"].items()
            if "dependence" in str(name) and Path(path).is_file()
        ]
    if dependence:
        layout.section("Dependence", "Stored dependence panels for the leading numeric features.")
        for name, path in dependence[:3]:
            st.image(str(path), caption=str(name).replace("_", " "))

    layout.section("Customer-level examples", "These are the exact local artifacts produced by Phase 8, not recomputed values.")
    examples = (manifest or {}).get("local_examples") or []
    if not examples:
        st.info("No local explanation examples were stored for this run.")
    else:
        labels = {
            f"Row {item.get('row_position')} · probability {float(item.get('probability', 0)):.4f} · "
            f"class {item.get('predicted_class')}": int(item.get("row_position"))
            for item in examples
        }
        selected_label = st.selectbox("Local example", list(labels), key="cw_xai_local")
        row_position = labels[selected_label]
        payload, payload_path = _local_payload(settings, str(run_id), row_position)
        if not payload:
            st.warning("The selected local explanation file is missing.")
        else:
            cols = st.columns(4)
            with cols[0]:
                st.metric("Row position", row_position)
            with cols[1]:
                st.metric("Probability", format_probability(payload.get("probability")))
            with cols[2]:
                st.metric("Predicted class", payload.get("predicted_class", "not available"))
            with cols[3]:
                st.metric("Output space", payload.get("output_space", "not available"))
            st.write(payload.get("summary") or "No summary sentence stored.")
            plot_path = _local_plot(settings, str(run_id), row_position)
            if plot_path:
                st.image(str(plot_path), caption=f"Local SHAP waterfall · row {row_position}")
            contributions = tables.local_contributions_frame(payload)
            if not contributions.empty:
                st.dataframe(contributions, width="stretch", hide_index=True)
                st.caption(
                    "`value` is the transformed model input stored in the explanation artifact; raw customer "
                    "attributes are intentionally absent from dashboard artifacts. Negative contributions reduce "
                    "the predicted churn probability."
                )
            st.caption(f"Local artifact: {payload_path}")

    limitations = (manifest or {}).get("limitations") or []
    if limitations:
        with st.expander("Explanation limitations"):
            for item in limitations:
                st.write(f"- {item}")
    st.caption(f"Global artifact: {global_path} · manifest: {manifest_path}")
