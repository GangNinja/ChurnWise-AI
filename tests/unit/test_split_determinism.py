"""Split determinism, stratification and grouping tests (T5.1, T5.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnwise.preprocessing.contracts import PreprocessConfig
from churnwise.preprocessing.errors import CODE_SPLIT_IMPOSSIBLE, PreprocessingError
from churnwise.preprocessing.split import make_split

DATASET_SHA = "a" * 64


def churn_frame(rows: int = 1200, *, positive_every: int = 7) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"C{index:06d}" for index in range(rows)],
            "tenure": [index % 36 for index in range(rows)],
            "charge": [20.0 + (index * 7 % 80) for index in range(rows)],
            "churn": [
                "Yes" if index % positive_every == 0 else "No" for index in range(rows)
            ],
        }
    )


def normalised(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["churn"] = (frame["churn"] == "Yes").astype(int)
    return frame


def split(frame: pd.DataFrame, *, seed: int = 42, id_column: str | None = None):
    return make_split(
        normalised(frame),
        dataset_id="fixture:split",
        dataset_sha256=DATASET_SHA,
        run_id=f"run-{seed}",
        target_column="churn",
        id_column=id_column,
        config=PreprocessConfig(seed=seed),
    )


def test_same_seed_produces_identical_fingerprint_and_rows():
    frame = churn_frame()
    first = split(frame)
    second = split(frame)

    assert first.spec.fingerprint == second.spec.fingerprint
    assert first.train_indices == second.train_indices
    assert first.val_indices == second.val_indices
    assert first.test_indices == second.test_indices
    assert first.spec.seed == 42


def test_different_seed_changes_the_assignment():
    frame = churn_frame()
    first = split(frame, seed=42)
    second = split(frame, seed=43)

    assert first.spec.fingerprint != second.spec.fingerprint
    assert first.train_indices != second.train_indices


def test_split_ratios_and_stratification():
    frame = churn_frame()
    result = split(frame)
    spec = result.spec

    total = sum(spec.sizes.values())
    assert total == len(frame)
    assert spec.sizes["train"] == pytest.approx(total * 0.60, abs=2)
    assert spec.sizes["val"] == pytest.approx(total * 0.20, abs=2)
    assert spec.sizes["test"] == pytest.approx(total * 0.20, abs=2)

    overall = float((normalised(frame)["churn"] == 1).mean())
    for name in ("train", "val", "test"):
        assert spec.class_rates[name] == pytest.approx(overall, abs=0.01)
        assert set(spec.class_counts[name]) == {"0", "1"}

    indices = [*result.train_indices, *result.val_indices, *result.test_indices]
    assert sorted(indices) == list(range(len(frame)))
    assert not set(result.train_indices) & set(result.val_indices)
    assert not set(result.train_indices) & set(result.test_indices)
    assert not set(result.val_indices) & set(result.test_indices)
    assert spec.constraints == {"test_touched": False, "fitted_on": "train_only"}


def test_repeated_customer_ids_never_cross_partitions():
    rows = 600
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index % 60:03d}" for index in range(rows)],
            "tenure": [index % 24 for index in range(rows)],
            "churn": ["Yes" if (index * 7 % 100) < 20 else "No" for index in range(rows)],
        }
    )
    result = split(frame, id_column="customer_id")

    assert result.spec.strategy == "stratified_group"
    partitions = {
        "train": set(frame.iloc[result.train_indices]["customer_id"]),
        "val": set(frame.iloc[result.val_indices]["customer_id"]),
        "test": set(frame.iloc[result.test_indices]["customer_id"]),
    }
    assert not partitions["train"] & partitions["val"]
    assert not partitions["train"] & partitions["test"]
    assert not partitions["val"] & partitions["test"]
    assert sum(len(value) for value in partitions.values()) == 60


def test_grouped_split_honours_custom_fractions():
    rows = 600
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index % 60:03d}" for index in range(rows)],
            "tenure": [index % 24 for index in range(rows)],
            "churn": ["Yes" if (index * 7 % 100) < 20 else "No" for index in range(rows)],
        }
    )
    config = PreprocessConfig(
        seed=42, train_fraction=0.70, val_fraction=0.15, test_fraction=0.15
    )
    result = make_split(
        normalised(frame),
        dataset_id="fixture:split",
        dataset_sha256=DATASET_SHA,
        run_id="run-grouped-fractions",
        target_column="churn",
        id_column="customer_id",
        config=config,
    )

    assert result.spec.strategy == "stratified_group"
    assert result.spec.ratios == {"train": 0.70, "val": 0.15, "test": 0.15}
    assert result.spec.sizes == {"train": 420, "val": 90, "test": 90}
    partitions = {
        "train": set(frame.iloc[result.train_indices]["customer_id"]),
        "val": set(frame.iloc[result.val_indices]["customer_id"]),
        "test": set(frame.iloc[result.test_indices]["customer_id"]),
    }
    assert not partitions["train"] & partitions["val"]
    assert not partitions["train"] & partitions["test"]
    assert not partitions["val"] & partitions["test"]
    assert sum(len(value) for value in partitions.values()) == 60


def test_too_small_dataset_is_an_actionable_error():
    frame = churn_frame(rows=60)
    with pytest.raises(PreprocessingError) as caught:
        split(frame)
    assert caught.value.code == CODE_SPLIT_IMPOSSIBLE
    assert "at least 100" in caught.value.message


def test_severe_imbalance_warns_but_still_splits():
    rows = 300
    frame = pd.DataFrame(
        {
            "tenure": [index % 24 for index in range(rows)],
            "churn": ["Yes" if index < 9 else "No" for index in range(rows)],
        }
    )
    result = split(frame)
    assert result.spec.sizes["train"] > 0
    assert any("minority rows" in warning for warning in result.warnings)


def test_fingerprint_covers_the_dataset_digest():
    frame = churn_frame()
    first = make_split(
        normalised(frame),
        dataset_id="fixture:split",
        dataset_sha256=DATASET_SHA,
        run_id="run-a",
        target_column="churn",
        id_column=None,
        config=PreprocessConfig(seed=42),
    )
    second = make_split(
        normalised(frame),
        dataset_id="fixture:split",
        dataset_sha256="b" * 64,
        run_id="run-a",
        target_column="churn",
        id_column=None,
        config=PreprocessConfig(seed=42),
    )
    assert first.spec.fingerprint != second.spec.fingerprint


def test_split_spec_round_trips():
    result = split(churn_frame())
    payload = result.spec.to_dict()
    assert payload["constraints"]["test_touched"] is False
    assert np.asarray(payload["row_indices"]["train"]).ndim == 1
    assert payload["sizes"]["train"] == len(result.train_indices)
