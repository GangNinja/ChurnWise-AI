"""Model Laboratory: technical credibility, data quality and explicit operations.

Four tabs keep the evidence available without spreading it across the product:
Performance (benchmark, diagnostics, CV), Explainability (global SHAP),
Data & validation (gate, roles, provenance, integrity) and Operations
(discovery, acquisition and MLP training on explicit actions only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import actions
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import (
    format_count,
    format_probability,
    validation_check_counts,
    validation_status_kind,
)
from churnwise.dashboard.components import charts, layout, tables


def _yes_no(value: Any) -> str:
    if value is None:
        return "not recorded"
    return "yes" if bool(value) else "no"


def _comparison_frame(families: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for family, payload in families.items():
        val = payload.get("validation_metrics") or {}
        test = payload.get("test_metrics") or {}
        rows.append(
            {
                "Family": payload.get("label") or family,
                "Validation F1": val.get("f1"),
                "Test F1": test.get("f1"),
                "Precision": test.get("precision"),
                "Recall": test.get("recall"),
                "Test ROC-AUC": test.get("roc_auc"),
                "Test PR-AUC": test.get("pr_auc"),
                "Threshold": (payload.get("threshold") or {}).get("selected"),
            }
        )
    frame = pd.DataFrame(rows)
    for column in frame.columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").round(4)
    return frame


def _performance_tab(settings: Settings, metrics: dict[str, Any], run_id: str | None) -> None:
    families = metrics.get("families") or {}
    selection = metrics.get("selection") or {}
    production_family = selection.get("selected_family")
    production = families.get(production_family) or {}
    curves, _ = access.benchmark_curves(settings, run_id)

    st.caption(
        "Selection used validation F1 only; the test split was scored once at each family's selected threshold. "
        "These are held-out results, not the scored risk population."
    )
    st.dataframe(_comparison_frame(families), width="stretch", hide_index=True)
    comparison = metrics.get("comparison_table") or []
    if comparison:
        with st.expander("Full held-out metric table"):
            st.dataframe(tables.metrics_frame(comparison), width="stretch", hide_index=True)

    diagnostic_families = [family for family in families if (families.get(family) or {}).get("test_metrics")]
    if not diagnostic_families:
        diagnostic_families = list(families)
    if not diagnostic_families:
        st.info("No model families are available in this benchmark.")
        return
    saved_inspection = st.session_state.get("cw_lab_inspect_family")
    if saved_inspection not in diagnostic_families:
        saved_inspection = production_family if production_family in diagnostic_families else diagnostic_families[0]
    inspection_family = st.selectbox(
        "Inspect diagnostics for",
        diagnostic_families,
        index=diagnostic_families.index(saved_inspection),
        format_func=lambda family: (families.get(family) or {}).get("label") or family.replace("_", " ").title(),
        key="cw_lab_inspect_family",
        help="This changes the diagnostic view only; it does not change the production model selection.",
    )
    inspected = families.get(inspection_family) or {}
    inspected_test = inspected.get("test_metrics") or {}
    production_label = (production or {}).get("label") or production_family or "not recorded"
    inspection_label = inspected.get("label") or inspection_family
    st.caption(
        f"Diagnostics shown for {inspection_label}. Production selection remains {production_label}; "
        "all values below come from the stored held-out evaluation."
    )

    chart_cols = st.columns([1, 1.7])
    with chart_cols[0]:
        st.plotly_chart(
            charts.confusion_chart(inspected_test, title=f"{inspection_label} · confusion matrix"),
            width="stretch",
        )
        matrix = inspected_test.get("confusion_matrix") or {}
        st.caption(
            f"TN {matrix.get('tn')} · FP {matrix.get('fp')} · FN {matrix.get('fn')} · TP {matrix.get('tp')}"
        )
    with chart_cols[1]:
        st.plotly_chart(charts.roc_chart(curves, selected=inspection_family), width="stretch")
        with st.expander("Precision–recall curve"):
            st.plotly_chart(charts.pr_chart(curves, selected=inspection_family), width="stretch")

    threshold_analysis, _ = access.benchmark_thresholds(settings, run_id)
    if threshold_analysis:
        with st.expander("Threshold analysis (validation only)"):
            st.caption(
                "Thresholds were swept on validation rows; the test split was not used to choose the "
                "operating threshold."
            )
            st.plotly_chart(
                charts.threshold_chart(threshold_analysis, inspection_family),
                width="stretch",
            )

    layout.section("Cross-validation stability", "Five-fold estimates on train+validation rows; test rows are excluded.")
    cv_by_family = {family: payload.get("cv") or {} for family, payload in families.items()}
    global_cv = metrics.get("cross_validation") or {}
    global_preprocessing = global_cv.get("preprocessing") or {}
    if any((item or {}).get("status") == "completed" for item in cv_by_family.values()):
        st.plotly_chart(charts.cv_chart(cv_by_family), width="stretch")
        rows = []
        for family, payload in families.items():
            cv = payload.get("cv") or {}
            f1 = (cv.get("metrics") or {}).get("f1") or {}
            preprocessing = cv.get("preprocessing") or {}
            fold_safe = preprocessing.get("refitted_per_fold")
            if fold_safe is None:
                fold_safe = preprocessing.get("fold_safe", global_preprocessing.get("fold_safe"))
            verified = preprocessing.get(
                "verified_against_phase5", global_preprocessing.get("verified_against_phase5")
            )
            rows.append(
                {
                    "Family": payload.get("label") or family,
                    "Folds": f"{cv.get('folds_completed', '?')}/{cv.get('folds_requested', '?')}",
                    "F1 mean": f1.get("mean"),
                    "F1 std": f1.get("std"),
                    "Refit per fold": _yes_no(fold_safe),
                    "Verified against Phase 5": _yes_no(verified),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "CV metrics use the 0.5 reference threshold, not each family's operating threshold. They are a "
            "stability estimate; the held-out test result above is the primary evidence."
        )
    else:
        st.info("No completed cross-validation summary is stored for this benchmark.")

    prepared, _ = access.prepared_summary(settings)
    metadata, _ = access.model_metadata(settings)
    history, _ = access.training_history(settings)
    if prepared or metadata:
        with st.expander("Prepared data and MLP training history"):
            if prepared:
                sizes = prepared.get("sizes") or {}
                st.caption(
                    f"Prepared run `{prepared.get('run_id')}` · train {format_count(sizes.get('train'))} · "
                    f"validation {format_count(sizes.get('val'))} · test {format_count(sizes.get('test'))} · "
                    f"{format_count(prepared.get('feature_count'))} transformed features."
                )
            if history:
                tab_loss, tab_accuracy = st.tabs(["Loss", "Accuracy"])
                with tab_loss:
                    st.plotly_chart(charts.training_history_chart(history, "loss"), width="stretch")
                with tab_accuracy:
                    st.plotly_chart(charts.training_history_chart(history, "accuracy"), width="stretch")
            else:
                st.caption("No MLP epoch history is stored.")

    st.caption(
        "Probability calibration is a separate research artifact and is not integrated into this model or the "
        "operating threshold; no calibrated curve is displayed here."
    )
    limitations = metrics.get("limitations") or []
    if limitations:
        with st.expander("Benchmark limitations"):
            for item in limitations:
                st.write(f"- {item}")


def _explainability_tab(settings: Settings) -> None:
    runs = access.explanation_ids(settings)
    run_id = st.selectbox("Explanation run", runs, index=0, key="cw_lab_xai_run") if runs else None
    global_payload, _ = access.explanation_global(settings, run_id)
    manifest, _ = access.explanation_manifest(settings, run_id)
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
        "warn" if explainer.get("selected_by_fallback") else "ok",
        str(explainer.get("output_space") or ""),
    )
    st.caption(
        f"Family {global_payload.get('family') or 'not available'} · sampled rows "
        f"{sample.get('sampled_rows', 'not available')} · sample split {sample.get('split', 'not recorded')} · "
        f"{explainer.get('transformed_columns', 'not recorded')} transformed columns aggregated into "
        f"{explainer.get('source_features', 'not recorded')} feature groups."
    )
    st.plotly_chart(charts.importance_chart(global_payload.get("feature_importance") or []), width="stretch")
    st.caption(
        "SHAP values describe the model's output, not causal effects. Grouped features include raw source "
        "columns and persisted engineered features."
    )
    importance = global_payload.get("feature_importance") or []
    if importance:
        with st.expander("Feature-importance table"):
            frame = pd.DataFrame(importance)
            keep = [
                column
                for column in ("rank", "feature", "kind", "mean_abs_shap", "mean_shap", "share", "direction")
                if column in frame
            ]
            st.dataframe(frame[keep], width="stretch", hide_index=True)
    limitations = (manifest or {}).get("limitations") or []
    if limitations:
        with st.expander("Explanation limitations"):
            for item in limitations:
                st.write(f"- {item}")


def _data_tab(settings: Settings) -> None:
    report, _ = access.validation_report(settings)
    eda, _ = access.eda_profile(settings, str((report or {}).get("dataset_id") or "") or None)
    candidates, _ = access.acquisition_candidates(settings)
    attempts, _ = access.acquisition_attempts(settings)
    if not report:
        layout.empty_state(
            "No validation report exists",
            "Acquire a dataset and run the Phase 4 validation gate before trusting any model result.",
            "python -m churnwise.cli acquire && python -m churnwise.cli validate",
        )
        return

    summary = report.get("summary") or {}
    target = report.get("target") or {}
    gate = report.get("gate") or {}
    gate_status = str(report.get("status") or gate.get("status") or "unknown")
    layout.status_line("Validation gate", gate_status.replace("_", " "), validation_status_kind(gate_status))
    if gate.get("reasons"):
        st.caption("Gate reasons: " + "; ".join(str(item) for item in gate["reasons"]))
    if gate.get("blocking"):
        st.error("Blocking checks: " + ", ".join(str(item) for item in gate["blocking"]))
    if not gate.get("can_preprocess", False):
        st.error("Phase 4 did not approve this dataset for preprocessing.")

    layout.metric_row(
        [
            ("Labelled records", format_count(summary.get("n_rows")), "Validated artifact rows"),
            ("Usable predictors", format_count(len(summary.get("usable_features") or [])), "Retained by validation roles"),
            ("Labelled churn rate", f"{float(target.get('positive_rate', 0)):.1%}", "Observed target rate"),
            (
                "Checks",
                f"{len(report.get('checks') or [])}",
                "Validation checks recorded for this dataset",
            ),
        ]
    )
    st.caption(
        f"Target `{target.get('column') or 'not recorded'}` · method "
        f"{str(target.get('method') or 'not recorded').replace('_', ' ')}."
    )

    checks = report.get("checks") or []
    if checks:
        counts = validation_check_counts(checks)
        st.caption(
            f"{counts['passed']} passed, {counts['warnings']} warning-level not passed, "
            f"{counts['fatal']} fatal not passed (total {counts['total']})."
        )
        if counts["fatal"]:
            st.error(
                f"{counts['fatal']} fatal check(s) did not pass; the gate blocks preprocessing."
                + (
                    f" {counts['warnings']} warning-level check(s) also did not pass."
                    if counts["warnings"]
                    else ""
                )
            )
        elif counts["warnings"]:
            st.warning(
                f"{counts['warnings']} warning-level check(s) did not pass. The gate is open because no "
                "fatal check failed; review the evidence below."
            )
        failed = [check for check in checks if not check.get("passed")]
        if failed:
            st.dataframe(tables.checks_frame(failed), width="stretch", hide_index=True)
        else:
            st.success("All recorded validation checks passed.")
        with st.expander("All validation checks"):
            st.dataframe(tables.checks_frame(checks), width="stretch", hide_index=True)

    roles = report.get("roles") or {}
    dropped = roles.get("dropped") or []
    role_bits = [
        f"{format_count(len(roles.get('numeric') or []))} numeric",
        f"{format_count(len(roles.get('categorical') or []))} categorical",
        f"{format_count(len(dropped))} dropped",
    ]
    st.caption("Feature roles: " + " · ".join(role_bits))
    if dropped:
        st.caption("Dropped: " + ", ".join(str(item) for item in dropped))

    if eda and eda.get("columns"):
        with st.expander("Column profile (missingness and cardinality)"):
            frame = pd.DataFrame(
                [
                    {
                        "Column": column.get("name"),
                        "Kind": column.get("kind"),
                        "Role": column.get("role"),
                        "Missing %": column.get("missing_pct"),
                        "Unique": column.get("unique_count"),
                    }
                    for column in eda["columns"]
                ]
            )
            st.dataframe(frame, width="stretch", hide_index=True)

    with st.expander("Discovery and acquisition provenance"):
        if candidates:
            found = (candidates.get("summary") or {}).get("found", "not recorded")
            eligible = (candidates.get("summary") or {}).get("eligible", "not recorded")
            selected = (candidates.get("summary") or {}).get("selected") or "none"
            st.caption(f"Discovery found {found}, {eligible} eligible; selected {selected}.")
            st.caption(
                "`Eligible` means suitable under the discovery gates and licence policy, not permission to "
                "redistribute the data."
            )
        else:
            st.caption("No discovery report is stored.")
        if attempts and attempts.get("attempts"):
            st.dataframe(pd.DataFrame(attempts["attempts"]), width="stretch", hide_index=True)
        else:
            st.caption("No acquisition attempts are recorded.")


def _inventory(settings: Settings) -> None:
    inventory = access.artifact_inventory(settings)
    if not inventory:
        st.caption("No artifacts are available to inspect.")
        return
    present = sum(1 for entry in inventory if entry.exists)
    st.caption(f"{present} of {len(inventory)} artifact families present on disk.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Stage": entry.kind,
                    "Run / id": entry.run_id,
                    "Generated": entry.generated_at or "not recorded",
                    "Status": "present" if entry.exists else "missing",
                }
                for entry in inventory
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    manifest, _ = access.risk_manifest(settings, prefer_population=True)
    if not manifest:
        st.caption("No risk manifest is available, so model digests cannot be verified here.")
        return

    def _resolve(value: Any) -> Path:
        path = Path(str(value or ""))
        return path if path.is_absolute() else Path(settings.project_root) / path

    if st.button("Verify model and preprocessor digests", key="cw_lab_verify_digests"):
        rows = []
        for label, key, hash_key in (
            ("Selected model", "model_path", "model_sha256"),
            ("Fitted preprocessor", "preprocessor_path", "preprocessor_hash"),
        ):
            path = _resolve(manifest.get(key))
            declared = manifest.get(hash_key)
            if path.is_file() and declared:
                actual = access.file_sha256(path)
                rows.append(
                    {
                        "Artifact": label,
                        "Declared": str(declared)[:16],
                        "Actual": actual[:16],
                        "Match": actual == str(declared),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if all(row["Match"] for row in rows):
                st.success("Declared digests match the files on disk.")
            else:
                st.error("At least one digest does not match the declared value; reload the run before trusting it.")
        else:
            st.warning("The manifest does not expose readable model and preprocessor paths, so digests were not checked.")


def _operations_tab(settings: Settings) -> None:
    if st.session_state.get("cw_lab_notice"):
        st.success(st.session_state.pop("cw_lab_notice"))
    layout.note(
        "These actions run the real pipeline synchronously and write artifacts. Page rendering never triggers "
        "them; each control must be pressed explicitly."
    )
    with st.expander("Run dataset discovery", expanded=False):
        with st.form("cw_lab_discovery_form"):
            terms = st.text_input(
                "Search terms",
                value="",
                placeholder="churn, customer churn, telecom churn",
                help="Comma-separated terms. Leave blank to use project defaults.",
            )
            cols = st.columns(2)
            with cols[0]:
                min_rows = st.number_input("Minimum rows", min_value=100, max_value=1_000_000, value=1000, step=100)
            with cols[1]:
                max_results = st.number_input("Max results per source", min_value=1, max_value=50, value=10, step=1)
            submitted = st.form_submit_button("Start discovery", type="primary")
        if submitted:
            try:
                with st.spinner("Querying public sources and evaluating gates…"):
                    actions.discover_datasets(
                        settings, terms=terms, min_rows=int(min_rows), max_results=int(max_results)
                    )
            except actions.DashboardActionError as exc:
                st.error(f"{exc.code}: {exc.message}" + (f" — {exc.remediation}" if exc.remediation else ""))
            else:
                st.success("Discovery completed. Reload the Data & validation tab to see the new ranking.")
                st.cache_data.clear()

    with st.expander("Acquire the selected candidate", expanded=False):
        report, _ = access.acquisition_candidates(settings)
        ranking = (report or {}).get("ranking") or []
        options = {
            f"{item.get('rank')}. {item.get('candidate_id')} — {item.get('title') or ''}": item.get("candidate_id")
            for item in ranking
            if item.get("eligible")
        }
        if not options:
            st.caption("No eligible candidate is recorded. Run discovery first.")
        else:
            label = st.selectbox("Candidate", list(options), key="cw_lab_candidate")
            allow_fallback = st.checkbox(
                "Try lower-ranked eligible candidates if this one fails",
                value=True,
                help="Phase 3 records every attempt and stops at the first verified acquisition.",
            )
            if st.button("Acquire selected candidate", key="cw_lab_acquire"):
                try:
                    with st.spinner("Acquiring and verifying the candidate…"):
                        result = actions.acquire_candidate(
                            settings, str(options[label]), allow_fallback=allow_fallback
                        )
                except actions.DashboardActionError as exc:
                    st.error(f"{exc.code}: {exc.message}" + (f" — {exc.remediation}" if exc.remediation else ""))
                else:
                    st.session_state["cw_lab_notice"] = (
                        f"Acquisition finished: status={result.get('status', 'success')}, "
                        f"cached={result.get('cached', 'unknown')}. The attempt log below is refreshed."
                    )
                    st.cache_data.clear()
                    st.rerun()

    with st.expander("Train a new MLP", expanded=False):
        prepared, _ = access.prepared_summary(settings)
        if not prepared:
            st.caption("No prepared Phase 5 dataset exists; run `python -m churnwise.cli prepare` first.")
        else:
            with st.form("cw_lab_training_form"):
                cols = st.columns(2)
                with cols[0]:
                    hidden = st.text_input("Hidden units", value="64,32", help="Comma-separated positive integers")
                    dropout = st.text_input("Dropout per layer", value="0.3,0.2", help="One value per hidden layer")
                    l2 = st.number_input("L2 penalty", min_value=0.0, value=0.0001, step=0.0001, format="%.4f")
                    learning_rate = st.number_input(
                        "Learning rate", min_value=0.000001, max_value=1.0, value=0.001, step=0.0001, format="%.6f"
                    )
                    batch_size = st.number_input("Batch size", min_value=1, max_value=4096, value=64, step=1)
                with cols[1]:
                    epochs = st.number_input("Maximum epochs", min_value=1, max_value=1000, value=150, step=1)
                    patience = st.number_input("Early-stopping patience", min_value=1, max_value=100, value=12, step=1)
                    threshold = st.number_input(
                        "Classification threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f"
                    )
                    seed = st.number_input(
                        "Seed", min_value=0, max_value=2_000_000_000, value=int(settings.seed), step=1
                    )
                    deterministic = st.checkbox("Deterministic TensorFlow ops", value=False)
                submitted = st.form_submit_button("Train MLP", type="primary")
            if submitted:
                try:
                    with st.spinner("Training the MLP and persisting artifacts. This page will wait…"):
                        outcome = actions.train_mlp(
                            settings,
                            {
                                "hidden_units": hidden,
                                "dropout": dropout,
                                "l2": l2,
                                "learning_rate": learning_rate,
                                "batch_size": batch_size,
                                "epochs": epochs,
                                "early_stopping_patience": patience,
                                "threshold": threshold,
                                "seed": seed,
                                "deterministic": deterministic,
                            },
                        )
                except actions.DashboardActionError as exc:
                    st.error(f"{exc.code}: {exc.message}" + (f" — {exc.remediation}" if exc.remediation else ""))
                except Exception as exc:  # keep the UI honest without leaking a traceback
                    st.error(
                        f"CW-UI-009: training failed ({exc.__class__.__name__}). Inspect the server log for diagnostics."
                    )
                else:
                    st.success(
                        f"Training completed: model run `{outcome.get('run_id', 'unknown')}`. "
                        "Metrics remain unverified until Phase 7 evaluates this run."
                    )
                    st.cache_data.clear()

    with st.expander("Artifact inventory and integrity", expanded=False):
        _inventory(settings)


def render(settings: Settings) -> None:
    runs = access.benchmark_ids(settings)
    chosen = st.session_state.get("cw_lab_benchmark")
    if chosen not in runs:
        chosen = runs[0] if runs else None
    metrics, _ = access.benchmark_metrics(settings, chosen)
    identity = access.identity_from_benchmark(metrics)
    layout.page_header(
        "Model Laboratory",
        "Held-out performance, stability, explanations and data quality for the deployed model. Model and "
        "threshold selection used validation rows only; the test split was scored once.",
        identity,
    )
    if len(runs) > 1:
        selected = st.selectbox(
            "Benchmark run",
            runs,
            index=runs.index(chosen),
            key="cw_lab_benchmark",
        )
        if selected != chosen:
            st.rerun()

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
        "Selected model",
        str(selection.get("selected_label") or selected_family or "not recorded"),
        "ok",
        f"validation {str(selection.get('selection_metric', 'metric')).upper()} "
        f"{float(selection.get('selected_validation_value', 0)):.4f}",
    )
    st.caption(
        f"Test F1 {format_probability(selected_test.get('f1'))} · "
        f"Test ROC-AUC {format_probability(selected_test.get('roc_auc'))} · "
        f"operating threshold {format_probability(selection.get('threshold'))}. "
        f"Benchmark `{metrics.get('benchmark_run_id')}` generated {metrics.get('generated_at')}."
    )

    tab_performance, tab_explain, tab_data, tab_operations = st.tabs(
        ["Performance", "Explainability", "Data & validation", "Operations"]
    )
    with tab_performance:
        _performance_tab(settings, metrics, chosen)
    with tab_explain:
        _explainability_tab(settings)
    with tab_data:
        _data_tab(settings)
    with tab_operations:
        _operations_tab(settings)
