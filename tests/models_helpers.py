"""Synthetic but genuine Phase 5 hand-off for network-free Phase 6 tests.

The fixture fits the real Phase 5 ``ColumnTransformer`` on a small generated
frame, so inference tests exercise the actual preprocessing interface rather
than a stub.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnwise.config import Settings
from churnwise.preprocessing import artifacts
from churnwise.preprocessing.contracts import (
    SCHEMA_VERSION,
    ClassBalance,
    PreprocessConfig,
    SplitData,
    SplitSpec,
)
from churnwise.preprocessing.dataset import PreparedDataset
from churnwise.preprocessing.pipeline import (
    TypePlan,
    build_preprocessor,
    clean_frame,
    describe_outputs,
)

FEATURES = ["tenure_months", "monthly_charges", "support_calls", "contract"]
POSITIVE_LABEL = "Yes"


def synthetic_frame(rows: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 60, size=rows)
    charges = np.round(rng.uniform(20.0, 110.0, size=rows), 2)
    calls = rng.integers(0, 8, size=rows)
    contract = np.where(rng.random(rows) < 0.55, "monthly", "yearly")
    logit = (
        0.07 * (tenure - 30)
        + 0.02 * (charges - 65)
        + 0.35 * (calls - 3)
        - 1.2 * (contract == "yearly")
    )
    churn = (logit + rng.normal(0.0, 0.4, size=rows) > 0).astype(int)
    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": charges,
            "support_calls": calls,
            "contract": contract,
            "churn": np.where(churn == 1, POSITIVE_LABEL, "No"),
        }
    )


def _plan() -> TypePlan:
    return TypePlan(
        numeric=["tenure_months", "monthly_charges", "support_calls"],
        categorical=["contract"],
    )


def make_prepared_dataset(
    *,
    rows: int = 240,
    seed: int = 7,
    config: PreprocessConfig | None = None,
) -> PreparedDataset:
    """Return a PreparedDataset whose fitted preprocessor is real Phase 5 code."""
    config = config or PreprocessConfig(seed=seed)
    frame = synthetic_frame(rows=rows, seed=seed)
    plan = _plan()
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(len(frame))
    n_train = int(round(rows * config.train_fraction))
    n_val = int(round(rows * config.val_fraction))
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    raw_train = frame.iloc[train_idx].reset_index(drop=True)
    raw_val = frame.iloc[val_idx].reset_index(drop=True)
    raw_test = frame.iloc[test_idx].reset_index(drop=True)

    preprocessor = build_preprocessor(plan, config)
    preprocessor.fit(clean_frame(raw_train, plan))
    names = [str(name) for name in preprocessor.get_feature_names_out()]

    def matrices(raw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(
            preprocessor.transform(clean_frame(raw, plan)), dtype=np.float64
        )
        y = (raw["churn"] == POSITIVE_LABEL).to_numpy(dtype=np.int8)
        return X, y

    X_train, y_train = matrices(raw_train)
    X_val, y_val = matrices(raw_val)
    X_test, y_test = matrices(raw_test)

    def counts(labels: np.ndarray) -> dict[str, int]:
        return {"0": int((labels == 0).sum()), "1": int((labels == 1).sum())}

    sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    spec = SplitSpec(
        run_id="run-model-fixture",
        dataset_id="fixture:churn",
        dataset_sha256="d" * 64,
        strategy="stratified_random",
        seed=seed,
        ratios={"train": config.train_fraction, "val": config.val_fraction, "test": config.test_fraction},
        sizes=sizes,
        class_counts={
            "train": counts(y_train),
            "val": counts(y_val),
            "test": counts(y_test),
        },
        class_rates={
            "train": float(y_train.mean()),
            "val": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        fingerprint="e" * 64,
        row_indices={
            "train": [int(i) for i in train_idx],
            "val": [int(i) for i in val_idx],
            "test": [int(i) for i in test_idx],
        },
    )
    balance = ClassBalance(
        counts={"all": counts(np.concatenate([y_train, y_val, y_test]))},
        rates={"all": float(np.concatenate([y_train, y_val, y_test]).mean())},
        strategy="class_weight",
        class_weights={"0": 1.0, "1": 2.0},
        ratio=2.0,
    )
    schema = {
        "schema_version": SCHEMA_VERSION,
        "phase": 5,
        "run_id": spec.run_id,
        "dataset_id": spec.dataset_id,
        "dataset_sha256": spec.dataset_sha256,
        "split_fingerprint": spec.fingerprint,
        "config": config.to_dict(),
        "target": {
            "column": "churn",
            "positive_label": POSITIVE_LABEL,
            "negative_label": "No",
            "mapping": {"No": 0, "Yes": 1},
        },
        "required_columns": sorted(FEATURES),
        "preprocessing": {
            "numeric": list(plan.numeric),
            "boolean": [],
            "categorical": list(plan.categorical),
            "numeric_coerced": [],
            "missing_indicators": [],
        },
        "derived": [],
        "feature_names": names,
        "output_features": describe_outputs(names, plan),
        "class_weights": {"0": 1.0, "1": 2.0},
        "class_balance": balance.to_dict(),
    }
    return PreparedDataset(
        run_id=spec.run_id,
        dataset_id=spec.dataset_id,
        config=config,
        split_spec=spec,
        schema=schema,
        train=SplitData("train", X_train, y_train, [int(i) for i in train_idx]),
        val=SplitData("val", X_val, y_val, [int(i) for i in val_idx]),
        test=SplitData("test", X_test, y_test, [int(i) for i in test_idx]),
        class_balance=balance,
        preprocessor=preprocessor,
        provenance={"dataset_id": spec.dataset_id, "dataset_sha256": spec.dataset_sha256},
    )


def persist_phase5(dataset: PreparedDataset, settings: Settings) -> PreparedDataset:
    """Write the synthetic hand-off through the real Phase 5 artifact writers."""
    root = settings.artifact_dir
    artifacts.save_split_spec(root, dataset.split_spec)
    artifacts.save_preprocessor(root, dataset.run_id, dataset.preprocessor, dataset.schema)
    artifacts.save_arrays(
        root,
        dataset.run_id,
        {
            "X_train": dataset.train.X,
            "y_train": dataset.train.y,
            "train_rows": np.asarray(dataset.train.row_indices, dtype=np.int64),
            "X_val": dataset.val.X,
            "y_val": dataset.val.y,
            "val_rows": np.asarray(dataset.val.row_indices, dtype=np.int64),
            "X_test": dataset.test.X,
            "y_test": dataset.test.y,
            "test_rows": np.asarray(dataset.test.row_indices, dtype=np.int64),
            "oversample_indices": np.asarray(dataset.oversample_indices, dtype=np.int64),
        },
    )
    return dataset
