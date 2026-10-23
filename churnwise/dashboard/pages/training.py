"""Model training: inspect the prepared dataset and explicitly train the Phase 6 MLP."""

from __future__ import annotations

import json
from typing import Any, Mapping

import streamlit as st

from churnwise.config import Settings
from churnwise.dashboard import actions
from churnwise.dashboard import data_access as access
from churnwise.dashboard.analytics import format_count
from churnwise.dashboard.components import charts, layout


def _history_from(result: Mapping[str, Any] | None, disk: Mapping[str, Any] | None) -> dict[str, list[float]]:
    if result and result.get("history"):
        return dict(result["history"])
    return dict(disk or {})


def render(settings: Settings) -> None:
    prepared, prepared_path = access.prepared_summary(settings)
    metadata, metadata_path = access.model_metadata(settings)
    history, _ = access.training_history(settings)
    identity = access.identity_from_training(prepared, metadata)
    layout.page_header(
        "Model training",
        "Inspect the leakage-safe prepared datasets and train the Keras MLP on an explicit action. "
        "Training runs synchronously in this Streamlit process; it is not a background job.",
        identity,
    )
    result = st.session_state.get("cw_training_result")
    if st.session_state.get("cw_training_error"):
        st.error(st.session_state.pop("cw_training_error"))
    if st.session_state.get("cw_training_notice"):
        st.success(st.session_state.pop("cw_training_notice"))

    if not prepared:
        layout.empty_state(
            "No prepared Phase 5 dataset",
            "Training needs a persisted split, fitted preprocessor and feature matrices. Run preparation first.",
            "python -m churnwise.cli prepare",
        )
        return

    sizes = prepared.get("sizes") or {}
    balance = prepared.get("class_balance") or {}
    layout.section("Prepared dataset", f"Run `{prepared.get('run_id')}`")
    layout.metric_row(
        [
            ("Train rows", format_count(sizes.get("train")), "Used for fitting"),
            ("Validation rows", format_count(sizes.get("val")), "Used for early stopping and threshold selection"),
            ("Test rows", format_count(sizes.get("test")), "Never used for training"),
            ("Features", format_count(prepared.get("feature_count")), "Transformed feature count"),
        ]
    )
    st.caption(f"Prepared summary: {prepared_path}")

    active_metadata = (result or {}).get("metadata") if result else metadata
    active_history = _history_from(result, history)
    if active_metadata:
        layout.section("Latest MLP run", "Architecture and epoch history loaded from the persisted model run.")
        architecture = active_metadata.get("architecture") or {}
        model_run_id = active_metadata.get("model_run_id") or (result or {}).get("run_id", "not available")
        epoch_count = len(active_history.get("loss") or []) if active_history else None
        st.caption(f"Model run id: `{model_run_id}`")
        layout.metric_row(
            [
                ("Hidden layers", architecture.get("hidden_units") or "not available", "Keras dense layers"),
                (
                    "Parameters",
                    format_count(architecture.get("parameter_count")),
                    "Trainable model parameters",
                ),
                (
                    "Epochs recorded",
                    format_count(epoch_count) if epoch_count is not None else "not available",
                    "Epochs in the saved training history",
                ),
                ("Seed", active_metadata.get("seed", "not available"), "Training seed"),
            ]
        )
        if active_history:
            tab_loss, tab_accuracy = st.tabs(["Loss", "Accuracy"])
            with tab_loss:
                st.plotly_chart(charts.training_history_chart(active_history, "loss"), width="stretch")
            with tab_accuracy:
                st.plotly_chart(
                    charts.training_history_chart(active_history, "accuracy"), width="stretch"
                )
            st.caption(
                f"{len(active_history.get('loss') or [])} epochs recorded. "
                "A validation curve that turns upward while training loss falls is an overfitting signal."
            )
        else:
            st.caption("No epoch history was recorded for this run.")
        st.caption(f"Metadata: {metadata_path}")
    else:
        st.info("No MLP artifact exists yet. The form below trains one if a prepared dataset is available.")

    layout.section("Train a new MLP", "Configuration is validated by the Phase 6 service before any fitting starts.")
    layout.note(
        "Training is synchronous: the page waits until fitting and artifact persistence finish. "
        "Expect roughly tens of seconds on the reference dataset; do not navigate away mid-run."
    )
    with st.form("cw_training_form"):
        cols = st.columns(2)
        with cols[0]:
            hidden = st.text_input("Hidden units", value="64,32", help="Comma-separated positive integers, e.g. 64,32")
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
            seed = st.number_input("Seed", min_value=0, max_value=2_000_000_000, value=int(settings.seed), step=1)
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
            st.session_state["cw_training_error"] = (
                f"{exc.code}: {exc.message}" + (f" — {exc.remediation}" if exc.remediation else "")
            )
            st.rerun()
        except Exception as exc:  # keep the UI honest without leaking a traceback
            st.session_state["cw_training_error"] = (
                f"CW-UI-009: training failed ({exc.__class__.__name__}). Inspect the server log for diagnostics."
            )
            st.rerun()
        else:
            st.session_state["cw_training_result"] = outcome
            st.session_state["cw_training_notice"] = (
                f"Training completed: model run `{outcome.get('run_id', 'unknown')}`. "
                "The metrics remain unverified until Phase 7 evaluates this run."
            )
            st.cache_data.clear()
            st.rerun()

    if result:
        with st.expander("Latest training result payload"):
            st.json(result)
    if metadata:
        with st.expander("Persisted model metadata"):
            st.json(metadata)
