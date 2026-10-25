"""Synthetic artifacts for Phase 11 dashboard tests.

Nothing here touches the network or the production artifact tree. The values
are internally consistent so page renderers can be exercised without pretending
to reproduce a real model result.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pandas as pd

DASHBOARD_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = DASHBOARD_ROOT / "churnwise" / "dashboard" / "app.py"

DATASET_ID = "fixture:churn"
DATASET_SHA = "a" * 64
RISK_RUN_ID = "20260101T020000Z-fixture-risk"
BENCHMARK_ID = "20260101T000000Z-fixture-bench"
EXPLAIN_ID = "20260101T010000Z-fixture-xai"
SIMULATION_ID = "20260101T030000Z-fixture-sim"
MODEL_RUN_ID = "20260101T000000Z-fixture-mlp"

ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nXQAAAAASUVORK5CYII="
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ONE_PIXEL_PNG)


def _risk_frame(rows: int = 6) -> pd.DataFrame:
    probabilities = [0.9, 0.8, 0.6, 0.4, 0.2, 0.1][:rows]
    tiers = ["critical", "critical", "high", "medium", "low", "low"][:rows]
    records = []
    for index, (probability, tier) in enumerate(zip(probabilities, tiers)):
        records.append(
            {
                "record_reference": f"fixture_ref_{index:03d}",
                "input_position": index + 1,
                "churn_probability": probability,
                "churn_probability_pct": probability * 100,
                "predicted_class": int(probability >= 0.45),
                "predicted_label": "Yes" if probability >= 0.45 else "No",
                "risk_tier": tier,
                "rank": index + 1,
                "classification_threshold": 0.45,
                "model_family": "random_forest",
                "model_run_id": f"{BENCHMARK_ID}/random_forest",
                "model_sha256": "b" * 64,
                "preprocessing_run_id": "fixture-prep",
                "assessed_at": "2026-01-01T02:00:00Z",
                "positive_label": "Yes",
                "observed_churn": None,
                "value_at_risk": None,
                "value_horizon_months": None,
                "explanation_status": "available",
                "explanation_reason": "",
                "explanation_summary": "Fixture explanation summary.",
                "top_positive_drivers": json.dumps(
                    [
                        {
                            "feature": "support_calls",
                            "contribution": 0.2,
                            "direction": "increases_churn",
                            "kind": "numeric",
                            "value": 2.0,
                        }
                    ]
                ),
                "top_negative_drivers": json.dumps(
                    [
                        {
                            "feature": "tenure_months",
                            "contribution": -0.1,
                            "direction": "decreases_churn",
                            "kind": "numeric",
                            "value": -1.0,
                        }
                    ]
                ),
                "risk_thresholds": json.dumps(
                    {"boundaries": [0.3, 0.5, 0.7], "labels": ["low", "medium", "high", "critical"]}
                ),
                "warnings": "[]",
                "segment": "{}",
            }
        )
    return pd.DataFrame(records)


def install_dashboard_project(root: Path) -> dict[str, Path]:
    """Create a temporary project tree with every artifact family."""

    report_dir = root / "reports"
    artifact_dir = root / "artifacts"

    validation = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "artifact_sha256": DATASET_SHA,
        "generated_at": "2026-01-01T00:05:00Z",
        "status": "pass_with_warnings",
        "summary": {
            "n_rows": 6,
            "n_cols": 5,
            "missing_cells": 2,
            "missing_cell_pct": 1.5,
            "duplicate_rows": 0,
            "duplicate_row_pct": 0.0,
            "usable_features": ["tenure_months", "monthly_charges", "support_calls", "contract"],
            "leakage_risks": ["total_charges duplicates monthly_charges"],
        },
        "target": {
            "column": "churn",
            "classes": ["No", "Yes"],
            "counts": {"No": 3, "Yes": 3},
            "positive_label": "Yes",
            "negative_label": "No",
            "positive_rate": 0.5,
            "method": "name_match",
            "ambiguous": False,
        },
        "roles": {
            "target": "churn",
            "numeric": ["tenure_months", "monthly_charges", "support_calls"],
            "categorical": ["contract"],
            "dropped": ["customer_id"],
            "revenue_proxy": None,
        },
        "capabilities": {"has_tenure": True, "has_monthly_charge": True},
        "checks": [
            {"id": "V1", "name": "file_integrity", "severity": "fatal", "passed": True, "detail": "ok"},
            {
                "id": "V11",
                "name": "identifier_and_cardinality",
                "severity": "warning",
                "passed": False,
                "detail": "one high-cardinality column",
            },
        ],
        "gate": {
            "status": "pass_with_warnings",
            "passed": True,
            "can_preprocess": True,
            "blocking": [],
            "warnings": ["V11"],
            "reasons": ["one identifier/cardinality warning"],
        },
        "provenance": {
            "dataset_id": DATASET_ID,
            "title": "Fixture churn dataset",
            "source_id": "fixture",
            "provider": "Fixture",
            "download_url": "https://example.invalid/fixture.csv",
            "landing_url": "https://example.invalid/fixture",
            "license": "MIT License",
            "retrieved_at": "2026-01-01T00:00:00Z",
            "run_id": "20260101T000000Z-fixture-acquire",
            "dataset_sha256": DATASET_SHA,
        },
        "warnings": ["fixture warning"],
        "notes": [],
    }
    _write_json(report_dir / "validation" / "openml_fixture.json", validation)

    eda = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "n_rows": 6,
        "n_cols": 5,
        "missing_cells": 2,
        "missing_cell_pct": 1.5,
        "duplicate_rows": 0,
        "columns": [
            {
                "name": "customer_id",
                "kind": "categorical",
                "role": "dropped",
                "missing_pct": 0.0,
                "unique_count": 6,
                "flags": ["high_cardinality"],
            },
            {
                "name": "tenure_months",
                "kind": "numeric",
                "role": "feature",
                "missing_pct": 0.0,
                "unique_count": 6,
                "flags": [],
            },
            {
                "name": "monthly_charges",
                "kind": "numeric",
                "role": "feature",
                "missing_pct": 16.67,
                "unique_count": 5,
                "flags": [],
            },
            {
                "name": "support_calls",
                "kind": "numeric",
                "role": "feature",
                "missing_pct": 0.0,
                "unique_count": 3,
                "flags": [],
            },
            {
                "name": "contract",
                "kind": "categorical",
                "role": "feature",
                "missing_pct": 0.0,
                "unique_count": 2,
                "flags": [],
            },
        ],
        "target": validation["target"],
        "roles": validation["roles"],
    }
    _write_json(report_dir / "eda" / "openml_fixture.json", eda)

    candidates = {
        "schema_version": "1.0",
        "phase": 2,
        "generated_at": "2026-01-01T00:01:00Z",
        "summary": {"found": 1, "valid": 1, "duplicates": 0, "unique": 1, "eligible": 1, "rejected": 0, "selected": DATASET_ID},
        "ranking": [
            {
                "rank": 1,
                "candidate_id": DATASET_ID,
                "source_id": "fixture",
                "source_name": "Fixture source",
                "title": "Fixture churn dataset",
                "row_count": 6,
                "feature_count": 4,
                "license": {"name": "MIT License", "status": "verified"},
                "eligible": True,
                "gates": {
                    "G1": {"gate_id": "G1", "status": "pass", "reason": "csv supported"},
                    "G2": {"gate_id": "G2", "status": "pass", "reason": "enough rows"},
                },
            }
        ],
        "selected": {"candidate_id": DATASET_ID, "title": "Fixture churn dataset"},
        "failure": None,
    }
    _write_json(report_dir / "acquisition" / "candidates.json", candidates)
    _write_json(
        report_dir / "acquisition" / "attempts.json",
        {
            "schema_version": "1.0",
            "phase": 3,
            "run_id": "20260101T000000Z-fixture-acquire",
            "generated_at": "2026-01-01T00:02:00Z",
            "attempts": [
                {
                    "candidate_id": DATASET_ID,
                    "rank": 1,
                    "status": "cached",
                    "cached": True,
                    "duration_ms": 2,
                    "error_code": None,
                    "error_message": None,
                }
            ],
            "result": {"status": "success"},
        },
    )

    prepared = {
        "run_id": "20260101T000000Z-fixture-prep",
        "dataset_id": DATASET_ID,
        "dataset_sha256": DATASET_SHA,
        "split_fingerprint": "c" * 64,
        "seed": 42,
        "sizes": {"train": 3, "val": 2, "test": 1},
        "class_balance": {"counts": {"0": 3, "1": 3}, "rates": {"0": 0.5, "1": 0.5}, "strategy": "class_weight"},
        "feature_count": 4,
        "feature_names": ["numeric__tenure_months", "numeric__monthly_charges", "numeric__support_calls", "categorical__contract_monthly"],
        "warnings": [],
        "paths": {},
        "provenance": {"generated_at": "2026-01-01T00:10:00Z", "dataset_sha256": DATASET_SHA},
        "constraints": {"test_touched": False, "fitted_on": "train"},
    }
    _write_json(artifact_dir / "prepared" / "20260101T000000Z-fixture-prep" / "summary.json", prepared)
    _write_json(
        artifact_dir / "preprocessors" / "20260101T000000Z-fixture-prep" / "schema.json",
        {
            "schema_version": "1.0",
            "phase": 5,
            "run_id": prepared["run_id"],
            "dataset_id": DATASET_ID,
            "dataset_sha256": DATASET_SHA,
            "target": {"column": "churn", "positive_label": "Yes", "negative_label": "No"},
            "required_columns": ["tenure_months", "monthly_charges", "support_calls", "contract"],
            "preprocessing": {
                "numeric": ["tenure_months", "monthly_charges", "support_calls"],
                "categorical": ["contract"],
                "boolean": [],
                "categories": {"contract": ["monthly", "yearly"]},
            },
        },
    )

    model_dir = artifact_dir / "models" / MODEL_RUN_ID / "mlp"
    _write_json(
        model_dir / "metadata.json",
        {
            "schema_version": "1.0",
            "phase": 6,
            "model_family": "mlp",
            "model_run_id": MODEL_RUN_ID,
            "created_at": "2026-01-01T00:20:00Z",
            "seed": 42,
            "architecture": {
                "input_dim": 4,
                "hidden_units": [8],
                "dropout": [0.2],
                "parameter_count": 65,
            },
            "threshold": 0.5,
        },
    )
    _write_json(
        model_dir / "history.json",
        {
            "loss": [0.7, 0.6, 0.5],
            "val_loss": [0.72, 0.61, 0.55],
            "accuracy": [0.5, 0.6, 0.7],
            "val_accuracy": [0.5, 0.55, 0.65],
        },
    )

    def _family(family: str, label: str, f1: float, roc: float, brier: float) -> dict[str, Any]:
        metrics = {
            "threshold": 0.45,
            "n": 6,
            "n_positive": 3,
            "n_negative": 3,
            "predicted_positive": 3,
            "confusion_matrix": {"labels": [0, 1], "matrix": [[2, 1], [1, 2]], "tn": 2, "fp": 1, "fn": 1, "tp": 2},
            "accuracy": 0.67,
            "precision": 0.67,
            "recall": 0.67,
            "f1": f1,
            "roc_auc": roc,
            "pr_auc": 0.7,
            "brier": brier,
        }
        return {
            "family": family,
            "label": label,
            "status": "completed",
            "threshold": {"selected": 0.45, "strategy": "max_f1", "selected_on": "validation"},
            "validation_metrics": dict(metrics, f1=f1 - 0.05, roc_auc=roc - 0.02),
            "test_metrics": metrics,
            "cv": {
                "status": "completed",
                "folds_requested": 5,
                "folds_completed": 5,
                "preprocessing": {"refitted_per_fold": True, "verified_against_phase5": True},
                "metrics": {"f1": {"mean": f1, "std": 0.02, "n": 5, "values": [f1] * 5}},
            },
        }

    families = {
        "logistic_regression": _family("logistic_regression", "Logistic Regression", 0.60, 0.80, 0.12),
        "random_forest": _family("random_forest", "Random Forest", 0.88, 0.94, 0.04),
        "mlp": _family("mlp", "MLP Neural Network", 0.72, 0.91, 0.09),
    }
    comparison_rows = [
        {
            "family": family,
            "label": payload["label"],
            "threshold": 0.45,
            "accuracy": 0.67,
            "precision": 0.67,
            "recall": 0.67,
            "f1": payload["test_metrics"]["f1"],
            "roc_auc": payload["test_metrics"]["roc_auc"],
            "pr_auc": 0.7,
            "brier": payload["test_metrics"]["brier"],
            "tn": 2,
            "fp": 1,
            "fn": 1,
            "tp": 2,
            "n": 6,
        }
        for family, payload in families.items()
    ]
    benchmark = {
        "schema_version": "1.0",
        "phase": 7,
        "benchmark_run_id": BENCHMARK_ID,
        "generated_at": "2026-01-01T00:30:00Z",
        "code_version": "0.11.0",
        "dataset": {"dataset_sha256": DATASET_SHA, "test_rows_used": 6},
        "preprocessing": {"run_id": prepared["run_id"], "dataset_sha256": DATASET_SHA, "feature_names": prepared["feature_names"]},
        "families": families,
        "comparison_table": comparison_rows,
        "selection": {
            "selected_family": "random_forest",
            "selected_label": "Random Forest",
            "selection_metric": "f1",
            "selected_validation_value": 0.83,
            "threshold": 0.45,
            "test_used_for_selection": False,
            "rationale": "Highest validation F1; the test split was not used.",
        },
        "cross_validation": {
            "enabled": True,
            "folds": 5,
            "status": "completed",
            "preprocessing": {
                "fold_safe": True,
                "verified_against_phase5": True,
                "strategy": "refit per fold from the Phase 5 base frame",
            },
        },
        "plots": {},
        "warnings": [],
        "limitations": ["Fixture metrics do not describe a real dataset."],
    }
    _write_json(artifact_dir / "evaluation" / BENCHMARK_ID / "metrics.json", benchmark)
    curves = {}
    for family, payload in families.items():
        roc_auc = payload["test_metrics"]["roc_auc"]
        curves[family] = {
            "roc": {"fpr": [0.0, 0.2, 1.0], "tpr": [0.0, 0.8, 1.0], "threshold": [None, 0.5, 0.0], "auc": roc_auc},
            "pr": {
                "precision": [1.0, 0.9, 0.7],
                "recall": [0.0, 0.6, 1.0],
                "threshold": [None, 0.5, 0.0],
                "pr_auc": 0.7,
            },
        }
    _write_json(artifact_dir / "evaluation" / BENCHMARK_ID / "curves.json", curves)

    xai_dir = artifact_dir / "explanations" / EXPLAIN_ID
    _png(xai_dir / "importance.png")
    _png(xai_dir / "summary.png")
    _png(xai_dir / "dependence_tenure_months.png")
    _png(xai_dir / "local" / "row_00001.png")
    global_payload = {
        "schema_version": "1.0",
        "phase": 8,
        "run_id": EXPLAIN_ID,
        "family": "random_forest",
        "output_space": "probability",
        "generated_at": "2026-01-01T01:00:00Z",
        "base_value": 0.5,
        "feature_importance": [
            {
                "rank": 1,
                "feature": "support_calls",
                "kind": "numeric",
                "mean_abs_shap": 0.2,
                "mean_shap": 0.1,
                "share": 0.6,
                "direction": "increases_churn",
            },
            {
                "rank": 2,
                "feature": "tenure_months",
                "kind": "numeric",
                "mean_abs_shap": 0.1,
                "mean_shap": -0.05,
                "share": 0.4,
                "direction": "decreases_churn",
            },
        ],
        "sample": {"split": "val", "sampled_rows": 2, "available_rows": 2, "sampling": "seeded", "test_rows_used": 0},
        "additivity_error": 1e-12,
        "explainer": {"name": "shap.TreeExplainer", "output_space": "probability", "source_features": 4, "selected_by_fallback": False},
        "plots": {
            "importance": str(xai_dir / "importance.png"),
            "summary": str(xai_dir / "summary.png"),
            "dependence_tenure_months": str(xai_dir / "dependence_tenure_months.png"),
        },
    }
    _write_json(xai_dir / "global.json", global_payload)
    _write_json(
        xai_dir / "run_manifest.json",
        {
            "schema_version": "1.0",
            "phase": 8,
            "run_id": EXPLAIN_ID,
            "generated_at": "2026-01-01T01:00:00Z",
            "family": "random_forest",
            "dataset_sha256": DATASET_SHA,
            "local_examples": [
                {"row_position": 1, "probability": 0.9, "predicted_class": 1},
                {"row_position": 2, "probability": 0.2, "predicted_class": 0},
            ],
            "limitations": ["SHAP values explain model output, not causal effects."],
        },
    )
    _write_json(
        xai_dir / "local" / "row_00001.json",
        {
            "family": "random_forest",
            "output_space": "probability",
            "probability": 0.9,
            "predicted_class": 1,
            "base_value": 0.5,
            "model_output": 0.9,
            "contributions": [
                {
                    "feature": "support_calls",
                    "kind": "numeric",
                    "contribution": 0.2,
                    "direction": "increases_churn",
                    "value": 2.0,
                    "active_level": None,
                },
                {
                    "feature": "tenure_months",
                    "kind": "numeric",
                    "contribution": -0.1,
                    "direction": "decreases_churn",
                    "value": -1.0,
                    "active_level": None,
                },
            ],
            "summary": "Fixture local explanation summary.",
        },
    )

    risk_dir = artifact_dir / "predictions" / RISK_RUN_ID
    risk_frame = _risk_frame()
    risk_dir.mkdir(parents=True, exist_ok=True)
    risk_frame.to_parquet(risk_dir / "churn_scores.parquet", index=False)
    risk_summary = {
        "schema_version": "1.0",
        "phase": 9,
        "run_id": RISK_RUN_ID,
        "generated_at": "2026-01-01T02:00:00Z",
        "n_rows": 6,
        "model": {
            "family": "random_forest",
            "model_run_id": f"{BENCHMARK_ID}/random_forest",
            "model_sha256": "b" * 64,
            "preprocessing_run_id": prepared["run_id"],
            "benchmark_run_id": BENCHMARK_ID,
        },
        "classification_threshold": 0.45,
        "risk_thresholds": {
            "boundaries": [0.3, 0.5, 0.7],
            "labels": ["low", "medium", "high", "critical"],
            "semantics": "lower-inclusive, upper-exclusive",
        },
        "population": {"split": "val", "received": 6, "scored": 6, "rejected": 0},
        "tier_counts": {
            "low": {"count": 2, "pct": 33.33},
            "medium": {"count": 1, "pct": 16.67},
            "high": {"count": 1, "pct": 16.67},
            "critical": {"count": 2, "pct": 33.33},
        },
        "probability": {"min": 0.1, "max": 0.9, "mean": 0.5, "median": 0.5},
        "predicted_class_counts": {"churn": 3, "retained": 3},
        "explanations": {"available": 6, "unavailable": 0},
        "value_at_risk": {"available": False, "horizon_months": None, "rows_with_value": 0},
        "provenance": {"dataset_sha256": DATASET_SHA, "preprocessor_hash": "d" * 64},
        "warnings": [],
        "limitations": ["Fixture probabilities are not real customer predictions."],
        "privacy": "opaque references only",
    }
    _write_json(risk_dir / "risk_summary.json", risk_summary)
    _write_json(
        risk_dir / "run_manifest.json",
        {
            "schema_version": "1.0",
            "phase": 9,
            "run_id": RISK_RUN_ID,
            "generated_at": "2026-01-01T02:00:00Z",
            "dataset_sha256": DATASET_SHA,
            "model_family": "random_forest",
            "model_run_id": f"{BENCHMARK_ID}/random_forest",
            "model_sha256": "b" * 64,
            "model_path": str(root / "missing_model.joblib"),
            "preprocessor_path": str(root / "missing_preprocessor.joblib"),
            "preprocessor_hash": "d" * 64,
            "benchmark_run_id": BENCHMARK_ID,
            "preprocessing_run_id": prepared["run_id"],
            "counts": {"received": 6, "scored": 6, "rejected": 0},
            "config": {"thresholds": [0.3, 0.5, 0.7]},
        },
    )
    _write_json(
        risk_dir / "drivers.json",
        {
            "fixture_ref_000": {
                "status": "available",
                "reason": "",
                "summary": "Fixture driver summary.",
                "top_positive": [
                    {
                        "feature": "support_calls",
                        "contribution": 0.2,
                        "direction": "increases_churn",
                        "kind": "numeric",
                        "value": 2.0,
                    }
                ],
                "top_negative": [
                    {
                        "feature": "tenure_months",
                        "contribution": -0.1,
                        "direction": "decreases_churn",
                        "kind": "numeric",
                        "value": -1.0,
                    }
                ],
            }
        },
    )

    sim_dir = artifact_dir / "simulations" / SIMULATION_ID
    simulation = {
        "schema_version": "1.0",
        "phase": 10,
        "scenario_id": SIMULATION_ID,
        "generated_at": "2026-01-01T03:00:00Z",
        "scenario": {"name": "Fixture campaign", "description": "", "budget": 100.0},
        "assumptions": {
            "cost_per_contact": 5.0,
            "effectiveness": 0.3,
            "monthly_value": 50.0,
            "horizon_months": 6.0,
            "value_unit": "value_units",
            "financial_metrics_available": True,
        },
        "targeting": {
            "strategy": "highest_risk",
            "min_risk_probability": 0.3,
            "risk_tiers": [],
            "max_contacts": None,
        },
        "budget": 100.0,
        "input": {
            "risk_run_id": RISK_RUN_ID,
            "dataset_sha256": DATASET_SHA,
            "model": {"family": "random_forest"},
        },
        "population": {
            "assessed": 6,
            "eligible": 4,
            "selected": 4,
            "excluded": 2,
            "exclusion_counts": {"below_risk_threshold": 2},
            "counts_reconcile": True,
        },
        "allocation": {
            "selected_count": 4,
            "allocated_spend": 20.0,
            "remaining_budget": 80.0,
            "unaffordable_count": 0,
            "contact_limited_count": 0,
        },
        "projection": {
            "expected_prevented_churn": 0.81,
            "expected_saved_value": 243.0,
            "net_benefit": 223.0,
            "roi": 11.15,
            "break_even_effectiveness": 0.082,
            "financial_metrics_available": True,
            "undefined_metrics": {},
        },
        "selected_customers": [
            {
                "record_reference": "fixture_ref_000",
                "churn_probability": 0.9,
                "risk_tier": "critical",
                "intervention_cost": 5.0,
                "expected_prevented_churn": 0.27,
            }
        ],
        "warnings": [],
        "limitations": ["Hypothetical projection; assumed effectiveness is not measured uplift."],
        "disclaimers": ["Hypothetical projection."],
        "hypothetical": True,
        "provenance": {"dataset_sha256": DATASET_SHA, "model": {"family": "random_forest"}},
    }
    _write_json(sim_dir / "result.json", simulation)
    _write_json(
        sim_dir / "sensitivity.json",
        {
            "series": [
                {
                    "dimension": "effectiveness",
                    "unit": "probability 0-1",
                    "baseline_value": 0.3,
                    "points": [
                        {"value": 0.2, "selected_count": 4, "allocated_spend": 20.0, "expected_prevented_churn": 0.54},
                        {"value": 0.3, "selected_count": 4, "allocated_spend": 20.0, "expected_prevented_churn": 0.81, "is_baseline": True},
                    ],
                }
            ],
            "warnings": [],
        },
    )
    _write_json(
        sim_dir / "comparison.json",
        {
            "rows": [
                {
                    "name": "Fixture campaign",
                    "strategy": "highest_risk",
                    "budget": 100.0,
                    "cost_per_contact": 5.0,
                    "effectiveness": 0.3,
                    "selected_count": 4,
                    "allocated_spend": 20.0,
                    "expected_prevented_churn": 0.81,
                    "cost_per_expected_prevented_churn": 24.69,
                    "roi": 11.15,
                },
                {
                    "name": "Comparison",
                    "strategy": "highest_risk",
                    "budget": 150.0,
                    "cost_per_contact": 5.0,
                    "effectiveness": 0.35,
                    "selected_count": 4,
                    "allocated_spend": 20.0,
                    "expected_prevented_churn": 0.95,
                    "cost_per_expected_prevented_churn": 21.05,
                    "roi": 13.25,
                },
            ],
            "shared_inputs": True,
            "notes": ["Fixture comparison."],
        },
    )

    return {
        "root": root,
        "report_dir": report_dir,
        "artifact_dir": artifact_dir,
        "validation": report_dir / "validation" / "openml_fixture.json",
        "risk_frame": risk_dir / "churn_scores.parquet",
        "risk_summary": risk_dir / "risk_summary.json",
        "risk_run_id": RISK_RUN_ID,
    }


def apply_environment(monkeypatch, root: Path) -> None:
    monkeypatch.setenv("CHURNWISE_DATA_DIR", str(root / "data"))
    monkeypatch.setenv("CHURNWISE_ARTIFACT_DIR", str(root / "artifacts"))
    monkeypatch.setenv("CHURNWISE_REPORT_DIR", str(root / "reports"))
    monkeypatch.setenv("CHURNWISE_OFFLINE", "true")


__all__ = [
    "APP_PATH",
    "BENCHMARK_ID",
    "DATASET_ID",
    "EXPLAIN_ID",
    "MODEL_RUN_ID",
    "RISK_RUN_ID",
    "SIMULATION_ID",
    "apply_environment",
    "install_dashboard_project",
]
