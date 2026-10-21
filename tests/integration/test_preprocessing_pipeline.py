"""End-to-end Phase 5 tests on the real Phase 4 hand-off (T5.1-T5.6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnwise.acquisition.coordinator import AcquisitionCoordinator
from churnwise.preprocessing import (
    PreprocessConfig,
    PreprocessingCoordinator,
    PreprocessingError,
)
from churnwise.validation import ValidationCoordinator
from tests.acquisition_helpers import (
    FakeDownloadTransport,
    FakeStream,
    make_acquisition_candidate,
    make_acquisition_settings,
    make_downloader,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_churn.csv"
URL = "https://www.openml.org/data/v1/download/910001"


def validated(tmp_path: Path, content: bytes | None = None, *, accepted: bool = True):
    settings = make_acquisition_settings(tmp_path)
    candidate = make_acquisition_candidate(
        candidate_id="openml:preprocess",
        source_dataset_id="910001",
        title="mini customer churn",
        resource_url=URL,
        target_column="churn",
    )
    transport = FakeDownloadTransport(
        {URL: FakeStream(body=content or FIXTURE.read_bytes())}
    )
    acquisition = AcquisitionCoordinator(
        settings,
        downloader_factory=lambda hosts: make_downloader(
            transport, hosts=tuple(hosts), max_bytes=settings.max_download_bytes
        ),
    )
    result = acquisition.acquire([candidate])
    assert result.succeeded, result.failure
    validation = ValidationCoordinator(settings).validate(result)
    assert validation.can_preprocess is accepted
    return settings, validation


def test_prepare_persists_a_reproducible_leakage_free_run(tmp_path):
    settings, validation = validated(tmp_path)
    dataset = PreprocessingCoordinator(settings).prepare(validation)

    spec = dataset.split_spec
    assert spec.sizes["train"] + spec.sizes["val"] + spec.sizes["test"] == sum(
        len(spec.row_indices[name]) for name in ("train", "val", "test")
    )
    partitions = [
        set(spec.row_indices[name]) for name in ("train", "val", "test")
    ]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert spec.constraints == {"test_touched": False, "fitted_on": "train_only"}
    assert spec.dataset_sha256 == validation.report.artifact_sha256

    for name in ("train", "val", "test"):
        split = dataset.split(name)
        assert split.X.dtype == np.float64
        assert split.X.shape == (split.n_rows, len(dataset.feature_names))
        assert np.isfinite(split.X).all()
        assert set(np.unique(split.y)) <= {0, 1}

    assert "churn" not in dataset.feature_names
    assert "customerID" not in dataset.feature_names
    excluded = {item["column"]: item["reason"] for item in dataset.schema["excluded"]}
    assert excluded["churn"] == "prediction target"
    assert "identifier" in excluded["customerID"]

    paths = {key: Path(value) for key, value in dataset.paths.items()}
    assert paths["preprocessor"].is_file()
    assert paths["arrays"].is_file()
    assert paths["split"].is_file()
    assert paths["report_json"].is_file()
    assert paths["report_md"].is_file()


def test_reload_reproduces_matrices_and_transforms_new_records(tmp_path):
    settings, validation = validated(tmp_path)
    dataset = PreprocessingCoordinator(settings).prepare(validation)
    reloaded = PreprocessingCoordinator(settings).load(dataset.run_id)

    assert np.array_equal(reloaded.train.X, dataset.train.X)
    assert np.array_equal(reloaded.test.y, dataset.test.y)
    assert reloaded.feature_names == dataset.feature_names
    assert reloaded.schema["preprocessor_hash"] == dataset.schema["preprocessor_hash"]
    assert Path(reloaded.paths["report_json"]).is_file()
    assert Path(reloaded.paths["report_md"]).is_file()

    record = pd.DataFrame(
        [
            {
                "tenure_months": 7,
                "monthly_charges": 71.25,
                "total_charges": 498.75,
                "contract": "monthly",
                "support_calls": 3,
            }
        ]
    )
    matrix = reloaded.transform_new(record)
    assert matrix.shape == (1, len(reloaded.feature_names))
    assert np.isfinite(matrix).all()
    assert np.array_equal(matrix, dataset.transform_new(record))

    unknown = record.assign(contract="weekly", total_charges="not a number")
    fallback = reloaded.transform_new(unknown)
    assert fallback.shape == matrix.shape
    assert np.isfinite(fallback).all()


def test_transform_new_reports_missing_columns(tmp_path):
    settings, validation = validated(tmp_path)
    dataset = PreprocessingCoordinator(settings).prepare(validation)

    with pytest.raises(PreprocessingError) as caught:
        dataset.transform_new(pd.DataFrame([{"tenure_months": 3}]))

    assert caught.value.code == "CW-PRE-004"
    assert "monthly_charges" in caught.value.message


def test_run_is_byte_identical_for_the_same_config_and_input(tmp_path):
    settings, validation = validated(tmp_path)
    first = PreprocessingCoordinator(settings).prepare(validation, run_id="run-one")
    second = PreprocessingCoordinator(settings).prepare(validation, run_id="run-two")

    assert first.split_spec.fingerprint == second.split_spec.fingerprint
    for name in ("train", "val", "test"):
        np.testing.assert_allclose(first.split(name).X, second.split(name).X)
        np.testing.assert_array_equal(first.split(name).y, second.split(name).y)


def test_validation_rejection_blocks_preprocessing(tmp_path):
    frame = pd.read_csv(FIXTURE)
    frame["churn"] = "No"
    settings, validation = validated(
        tmp_path, frame.to_csv(index=False).encode("utf-8"), accepted=False
    )

    with pytest.raises(PreprocessingError) as caught:
        PreprocessingCoordinator(settings).prepare(validation)

    assert caught.value.code == "CW-PRE-001"


def test_reports_capture_provenance_and_derived_skips(tmp_path):
    settings, validation = validated(tmp_path)
    dataset = PreprocessingCoordinator(settings).prepare(validation)

    payload = json.loads(Path(dataset.paths["report_json"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["provenance"]["dataset_sha256"] == dataset.provenance["dataset_sha256"]
    assert payload["provenance"]["dataset_id"] == dataset.dataset_id
    assert payload["target"]["column"] == "churn"
    assert payload["split"]["fingerprint"] == dataset.split_spec.fingerprint
    assert payload["class_balance"]["counts"]["train"]["1"] == int(
        dataset.train.y.sum()
    )
    categories = payload["preprocessing"]["categories"]
    assert {"monthly", "yearly"} <= set(categories["contract"])
    assert payload["preprocessing"]["output_dtype"] == "float64"
    applied = {item["name"] for item in payload["derived"]}
    assert {"tenure_years", "avg_monthly_charge"} <= applied
    skipped = {item["feature"] for item in payload["derived_skipped"]}
    assert "usage_band" in skipped

    markdown = Path(dataset.paths["report_md"]).read_text(encoding="utf-8")
    assert "fitted_on: train_only" in markdown
    assert "Excluded columns" in markdown


def test_oversampling_only_duplicates_training_rows(tmp_path):
    settings, validation = validated(tmp_path)
    config = PreprocessConfig(imbalance="oversample", oversample_ratio=0.8)
    dataset = PreprocessingCoordinator(settings).prepare(validation, config=config)

    assert dataset.oversample_indices
    assert dataset.resampled_train().n_rows > dataset.train.n_rows
    assert set(dataset.oversample_indices) <= set(range(dataset.train.n_rows))
    assert dataset.val.n_rows == dataset.split_spec.sizes["val"]
    assert dataset.test.n_rows == dataset.split_spec.sizes["test"]


def test_exact_duplicate_rows_are_dropped_before_splitting(tmp_path):
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    content = ("\n".join([*lines, *lines[1:11]]) + "\n").encode("utf-8")
    settings, validation = validated(tmp_path, content=content)
    dataset = PreprocessingCoordinator(settings).prepare(validation)

    assert dataset.split_spec.duplicate_rows_dropped == 10
    assert sum(dataset.split_spec.sizes.values()) == len(lines) - 1
    assert sum(len(values) for values in dataset.split_spec.row_indices.values()) == 160


def test_preprocessor_without_a_digest_is_rejected(tmp_path):
    settings, validation = validated(tmp_path)
    dataset = PreprocessingCoordinator(settings).prepare(validation)
    schema_path = Path(dataset.paths["preprocessor"]).parent / "schema.json"
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    payload.pop("preprocessor_hash", None)
    schema_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PreprocessingError) as caught:
        PreprocessingCoordinator(settings).load(dataset.run_id)

    assert caught.value.code == "CW-PRE-003"
    assert "digest" in caught.value.message


def test_corrupt_schema_is_rejected(tmp_path):
    settings, validation = validated(tmp_path)
    dataset = PreprocessingCoordinator(settings).prepare(validation)
    schema_path = Path(dataset.paths["preprocessor"]).parent / "schema.json"
    schema_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(PreprocessingError) as caught:
        PreprocessingCoordinator(settings).load(dataset.run_id)

    assert caught.value.code == "CW-PRE-003"
    assert "could not be read" in caught.value.message


def test_unsafe_run_ids_are_rejected():
    from churnwise.preprocessing.artifacts import validate_run_id

    for run_id in ("../escape", "a/b", "a\\b", "a:b"):
        with pytest.raises(PreprocessingError) as caught:
            validate_run_id(run_id)
        assert caught.value.code == "CW-PRE-003"


def test_unknown_dataset_artifact_names_are_rejected(tmp_path):
    settings, validation = validated(tmp_path)
    coordinator = PreprocessingCoordinator(settings)
    coordinator.prepare(validation)

    with pytest.raises(PreprocessingError) as caught:
        coordinator.load("../../etc/passwd")

    assert caught.value.code == "CW-PRE-003"
