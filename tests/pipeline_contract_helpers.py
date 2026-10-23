"""Contract assertions for JSON artifacts emitted by the Phase 12 workflow.

The fixture pipeline is intentionally small, but it writes the same artifact
families as the live workflow. This module gives each family a named contract so
a new, unclassified JSON file fails the pipeline test instead of slipping
through untested.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _dict(payload: object, label: str) -> dict:
    assert isinstance(payload, dict), f"{label} must be a JSON object"
    return payload


def _keys(payload: dict, label: str, *keys: str) -> None:
    missing = [key for key in keys if key not in payload]
    assert not missing, f"{label} is missing keys {missing}"


def _list(payload: dict, key: str, label: str, *, nonempty: bool = False) -> list:
    value = payload.get(key)
    assert isinstance(value, list), f"{label}.{key} must be a list"
    if nonempty:
        assert value, f"{label}.{key} must not be empty"
    return value


def _number(payload: dict, key: str, label: str) -> float:
    value = payload.get(key)
    assert isinstance(value, (int, float)) and not isinstance(value, bool), (
        f"{label}.{key} must be numeric"
    )
    return float(value)


def validate_json_artifact(relative: str, payload: object) -> None:
    """Validate one artifact using its path and family contract."""
    path = Path(relative)
    parts = path.parts
    name = path.name
    label = path.as_posix()

    if parts[:2] == ("artifacts", "pipelines"):
        data = _dict(payload, label)
        _keys(
            data,
            label,
            "schema_version",
            "phase",
            "run_id",
            "status",
            "config_digest",
            "stages",
            "artifacts",
            "warnings",
        )
        assert data["phase"] == 12
        assert data["status"] in {"running", "succeeded", "failed"}
        stages = _list(data, "stages", label, nonempty=True)
        assert [stage.get("name") for stage in stages] == [
            "source",
            "preprocess",
            "train",
            "evaluate",
            "explain",
            "score",
            "simulate",
        ]
        for stage in stages:
            assert stage.get("status") in {"pending", "running", "succeeded", "failed"}
            assert isinstance(stage.get("artifacts"), dict)
            assert isinstance(stage.get("outputs"), dict)
        return

    if parts[:2] == ("artifacts", "evaluation"):
        data = _dict(payload, label)
        if name == "metrics.json":
            _keys(
                data,
                label,
                "phase",
                "schema_version",
                "benchmark_run_id",
                "dataset",
                "preprocessing",
                "families",
                "selection",
                "complete",
            )
            families = _dict(data["families"], f"{label}.families")
            assert {"mlp", "logistic_regression", "random_forest"} <= set(families)
            for family, record in families.items():
                record = _dict(record, f"{label}.families.{family}")
                assert record.get("status") == "completed"
                _dict(record.get("test_metrics"), f"{label}.families.{family}.test_metrics")
            selection = _dict(data["selection"], f"{label}.selection")
            _keys(selection, f"{label}.selection", "selected_family", "threshold", "test_used_for_selection")
            assert selection["test_used_for_selection"] is False
            return
        if name == "run_manifest.json":
            _keys(
                data,
                label,
                "schema_version",
                "phase",
                "benchmark_run_id",
                "dataset_sha256",
                "preprocessing_run_id",
                "split_fingerprint",
                "selected_model_family",
                "test_evaluation_calls",
                "families",
            )
            assert data["test_evaluation_calls"] == 1
            return
        if name == "curves.json":
            assert {"mlp", "logistic_regression", "random_forest"} <= set(data)
            for family in ("mlp", "logistic_regression", "random_forest"):
                _keys(_dict(data[family], f"{label}.{family}"), f"{label}.{family}", "roc", "pr")
            return
        if name == "threshold_analysis.json":
            _keys(data, label, "strategy", "selected_on", "default_threshold", "families")
            return
        if name == "metadata.json" and "models" in parts:
            _keys(
                data,
                label,
                "phase",
                "schema_version",
                "model_family",
                "model_sha256",
                "preprocessing",
                "feature_names",
                "training",
            )
            return
        pytest.fail(f"no contract for evaluation artifact {relative}")

    if parts[:2] == ("artifacts", "explanations"):
        data = _dict(payload, label)
        if name == "global.json":
            _keys(
                data,
                label,
                "schema_version",
                "phase",
                "run_id",
                "family",
                "output_space",
                "feature_importance",
                "provenance",
            )
            _list(data, "feature_importance", label, nonempty=True)
            return
        if name == "run_manifest.json":
            _keys(
                data,
                label,
                "schema_version",
                "phase",
                "run_id",
                "family",
                "dataset_sha256",
                "preprocessing_run_id",
                "model",
                "paths",
            )
            _dict(data["model"], f"{label}.model")
            return
        if "local" in parts:
            _keys(
                data,
                label,
                "row_position",
                "family",
                "probability",
                "base_value",
                "contributions",
                "additivity_error",
            )
            _list(data, "contributions", label, nonempty=True)
            return
        pytest.fail(f"no contract for explanation artifact {relative}")

    if parts[:2] == ("artifacts", "models"):
        data = _dict(payload, label)
        if name == "history.json":
            assert data, f"{label} must not be empty"
            return
        if name == "metadata.json":
            _keys(
                data,
                label,
                "schema_version",
                "phase",
                "model_run_id",
                "model_family",
                "model_sha256",
                "preprocessing",
                "architecture",
                "training",
                "diagnostics",
            )
            return
        pytest.fail(f"no contract for model artifact {relative}")

    if parts[:2] == ("artifacts", "prepared") and name == "summary.json":
        data = _dict(payload, label)
        _keys(
            data,
            label,
            "run_id",
            "dataset_id",
            "dataset_sha256",
            "split_fingerprint",
            "feature_names",
            "sizes",
            "paths",
            "provenance",
        )
        assert data["sizes"]["train"] > 0
        return

    if parts[:2] == ("artifacts", "preprocessors") and name == "schema.json":
        data = _dict(payload, label)
        _keys(
            data,
            label,
            "schema_version",
            "phase",
            "run_id",
            "dataset_sha256",
            "feature_names",
            "required_columns",
            "pipeline_version",
            "config",
        )
        return

    if parts[:2] == ("artifacts", "splits"):
        data = _dict(payload, label)
        _keys(
            data,
            label,
            "schema_version",
            "run_id",
            "dataset_sha256",
            "fingerprint",
            "seed",
            "sizes",
            "row_indices",
            "constraints",
        )
        assert set(data["sizes"]) == {"train", "val", "test"}
        return

    if parts[:2] == ("artifacts", "predictions"):
        data = _dict(payload, label)
        if name == "risk_summary.json":
            _keys(
                data,
                label,
                "phase",
                "schema_version",
                "run_id",
                "n_rows",
                "classification_threshold",
                "risk_thresholds",
                "tier_counts",
                "model",
                "provenance",
            )
            assert data["n_rows"] > 0
            assert sum(item["count"] for item in data["tier_counts"].values()) == data["n_rows"]
            return
        if name == "run_manifest.json":
            _keys(
                data,
                label,
                "phase",
                "schema_version",
                "run_id",
                "model_sha256",
                "benchmark_run_id",
                "preprocessing_run_id",
                "counts",
                "paths",
            )
            assert data["counts"]["scored"] > 0
            assert data["counts"]["received"] == (
                data["counts"]["scored"] + data["counts"]["rejected"]
            )
            return
        if name == "drivers.json":
            assert data, f"{label} must not be empty"
            for key, value in data.items():
                assert isinstance(value, dict), f"{label}.{key} must be an object"
                assert "summary" in value
            return
        pytest.fail(f"no contract for prediction artifact {relative}")

    if parts[:2] == ("artifacts", "simulations"):
        data = _dict(payload, label)
        if name == "result.json":
            _keys(
                data,
                label,
                "phase",
                "schema_version",
                "scenario_id",
                "hypothetical",
                "input",
                "population",
                "allocation",
                "projection",
                "selected_customers",
                "provenance",
            )
            assert data["hypothetical"] is True
            assert data["population"]["counts_reconcile"] is True
            assert data["allocation"]["counts_reconcile"] is True
            assert data["projection"]["observed_retention_result"] is False
            _number(data["projection"], "expected_churn_count", f"{label}.projection")
            return
        if name == "sensitivity.json":
            _keys(
                data,
                label,
                "phase",
                "schema_version",
                "scenario_id",
                "method",
                "fixed_parameters",
                "series",
            )
            _list(data, "series", label, nonempty=True)
            return
        if name == "run_manifest.json":
            _keys(
                data,
                label,
                "phase",
                "schema_version",
                "scenario_id",
                "scenario_fingerprint",
                "risk",
                "artifacts",
                "disclaimers",
            )
            return
        pytest.fail(f"no contract for simulation artifact {relative}")

    if parts[:2] == ("data", "manifests"):
        data = _dict(payload, label)
        if name.endswith(".profile.json"):
            _keys(
                data,
                label,
                "schema_version",
                "dataset_id",
                "n_rows",
                "n_cols",
                "columns",
                "target",
                "capabilities",
            )
            assert data["n_rows"] > 0
            return
        _keys(
            data,
            label,
            "schema_version",
            "dataset_id",
            "dataset_sha256",
            "source",
            "file",
            "integrity",
            "acquisition",
            "discovery",
        )
        assert data["integrity"]["verified"] is True
        assert data["file"]["sha256"] == data["dataset_sha256"]
        return

    if parts[:2] == ("reports", "acquisition"):
        data = _dict(payload, label)
        if name == "candidates.json":
            _keys(
                data,
                label,
                "schema_version",
                "phase",
                "generated_at",
                "query",
                "config",
                "sources",
                "summary",
                "ranking",
                "selected",
                "failure",
            )
            return
        if name == "rejections.json":
            _keys(
                data,
                label,
                "schema_version",
                "phase",
                "generated_at",
                "query_fingerprint",
                "summary",
                "rejections",
                "failure",
            )
            return
        if name == "attempts.json":
            _keys(
                data,
                label,
                "schema_version",
                "phase",
                "run_id",
                "generated_at",
                "offline",
                "attempts",
                "result",
            )
            return
        pytest.fail(f"no contract for acquisition artifact {relative}")

    if parts[:2] == ("reports", "eda"):
        data = _dict(payload, label)
        _keys(data, label, "schema_version", "dataset_id", "n_rows", "n_cols", "columns", "target")
        return

    if parts[:2] == ("reports", "models"):
        data = _dict(payload, label)
        _keys(data, label, "metadata", "history", "paths", "diagnostics")
        return

    if parts[:2] == ("reports", "preprocessing"):
        data = _dict(payload, label)
        _keys(
            data,
            label,
            "schema_version",
            "phase",
            "run_id",
            "dataset_id",
            "split",
            "preprocessing",
            "target",
            "paths",
            "warnings",
        )
        return

    if parts[:2] == ("reports", "validation"):
        data = _dict(payload, label)
        _keys(
            data,
            label,
            "schema_version",
            "dataset_id",
            "artifact_sha256",
            "status",
            "gate",
            "checks",
            "roles",
            "summary",
            "provenance",
        )
        assert data["status"] in {"pass", "pass_with_warnings", "reject"}
        return

    pytest.fail(f"no contract registered for JSON artifact {relative}")
