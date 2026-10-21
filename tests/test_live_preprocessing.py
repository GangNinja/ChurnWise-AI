"""Live Phase 5 verification on a real acquired and validated dataset.

Skipped unless ``--run-live`` is passed. Discovery, acquisition, validation and
preprocessing all run against public sources; the persisted run is then reloaded
and used to transform a real customer record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from churnwise.config import Settings
from churnwise.contracts import SearchQuery
from churnwise.discovery import build_adapters, discover
from churnwise.preprocessing import PreprocessingCoordinator
from churnwise.validation import ValidationCoordinator, ValidationStatus

pytestmark = pytest.mark.live


def test_live_dataset_is_preprocessed_and_reloadable(tmp_path):
    settings = Settings(project_root=tmp_path)
    query = SearchQuery(
        terms=("churn",),
        max_results_per_source=5,
        enrich_limit=6,
        timeout_s=25,
        max_retries=1,
    )
    discovery = discover(build_adapters(settings, query), query, settings)
    assert discovery.selected is not None, (
        discovery.failure.to_dict() if discovery.failure else ""
    )
    validation = ValidationCoordinator(settings).validate_candidates(discovery)
    assert validation.status in {
        ValidationStatus.PASS,
        ValidationStatus.PASS_WITH_WARNINGS,
    }
    assert validation.can_preprocess is True

    coordinator = PreprocessingCoordinator(settings)
    dataset = coordinator.prepare(validation)

    spec = dataset.split_spec
    assert spec.constraints == {"test_touched": False, "fitted_on": "train_only"}
    assert spec.sizes["train"] + spec.sizes["val"] + spec.sizes["test"] <= len(
        validation.frame
    )
    assert spec.seed == settings.seed
    assert spec.fingerprint
    target = dataset.schema["target"]["column"]
    assert not any(
        name.endswith(f"__{target}") for name in dataset.feature_names
    )

    for name in ("train", "val", "test"):
        split = dataset.split(name)
        assert split.n_rows == spec.sizes[name]
        assert split.X.shape == (split.n_rows, len(dataset.feature_names))
        assert np.isfinite(split.X).all()
        assert set(np.unique(split.y)) <= {0, 1}

    assert dataset.train.positive_rate == pytest.approx(
        spec.class_rates["train"], abs=1e-12
    )
    excluded = {item["column"] for item in dataset.schema["excluded"]}
    assert dataset.schema["target"]["column"] in excluded

    schema_path = Path(dataset.paths["preprocessor"]).with_name("schema.json")
    assert schema_path.is_file()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["feature_names"] == dataset.feature_names
    assert schema["dataset_sha256"] == validation.report.artifact_sha256

    reloaded = coordinator.load(dataset.run_id)
    np.testing.assert_allclose(reloaded.test.X, dataset.test.X)

    normalized = validation.normalized_frame()
    required = list(schema["required_columns"])
    record = normalized[required].iloc[[0]]
    matrix = reloaded.transform_new(record)
    assert matrix.shape == (1, len(reloaded.feature_names))
    assert np.isfinite(matrix).all()

    assert (
        settings.report_dir / "preprocessing" / f"{dataset.run_id}.json"
    ).is_file()
    assert (
        settings.report_dir / "preprocessing" / f"{dataset.run_id}.md"
    ).is_file()
