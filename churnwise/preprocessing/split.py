"""Reproducible stratified splitting for Phase 5.

The split is created before any imputer, encoder or scaler is fitted. Row
deduplication happens before this module runs; when a pseudonymous id column is
present and repeated, rows are split by id so one customer cannot appear in two
partitions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churnwise.preprocessing.contracts import PreprocessConfig, SplitSpec
from churnwise.preprocessing.errors import CODE_SPLIT_IMPOSSIBLE, PreprocessingError

RATE_TOLERANCE = 0.01
GROUPED_ATTEMPTS = 10


@dataclass
class SplitResult:
    spec: SplitSpec
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    warnings: list[str]


def _class_counts(y: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _positive_rate(y: np.ndarray) -> float | None:
    if len(y) == 0:
        return None
    return float(np.mean(y))


def _split_fingerprint(
    *,
    seed: int,
    dataset_sha256: str,
    train: list[int],
    val: list[int],
    test: list[int],
) -> str:
    payload = {
        "seed": seed,
        "dataset_sha256": dataset_sha256,
        "train": train,
        "val": val,
        "test": test,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sizes(y: np.ndarray, config: PreprocessConfig) -> None:
    if len(y) < config.min_rows:
        raise PreprocessingError(
            CODE_SPLIT_IMPOSSIBLE,
            f"dataset has {len(y)} usable rows; at least {config.min_rows} are required to split",
            "Acquire a larger dataset or lower Phase 4's minimum row requirement.",
        )
    counts = _class_counts(y)
    if len(counts) != 2:
        raise PreprocessingError(
            CODE_SPLIT_IMPOSSIBLE,
            f"target has {len(counts)} usable classes; exactly two are required",
            "Re-run Phase 4 on a binary churn target.",
        )
    smallest = min(counts.values())
    if smallest < config.min_class_count:
        raise PreprocessingError(
            CODE_SPLIT_IMPOSSIBLE,
            f"minority class has {smallest} rows; at least {config.min_class_count} are required",
            "Acquire a larger dataset or lower min_class_count deliberately.",
        )


def _row_split(
    y: np.ndarray, config: PreprocessConfig
) -> tuple[list[int], list[int], list[int], int]:
    indices = np.arange(len(y))
    try:
        train_idx, rest = train_test_split(
            indices,
            test_size=config.val_fraction + config.test_fraction,
            stratify=y,
            random_state=config.seed,
        )
        val_idx, test_idx = train_test_split(
            rest,
            test_size=config.test_fraction / (config.val_fraction + config.test_fraction),
            stratify=y[rest],
            random_state=config.seed,
        )
    except ValueError as exc:
        raise PreprocessingError(
            CODE_SPLIT_IMPOSSIBLE,
            f"stratified split failed: {exc}",
            "Acquire a larger dataset or reduce the number of required partitions.",
        ) from exc
    return (
        sorted(int(i) for i in train_idx),
        sorted(int(i) for i in val_idx),
        sorted(int(i) for i in test_idx),
        config.seed,
    )


def _grouped_split(
    groups: np.ndarray,
    y: np.ndarray,
    config: PreprocessConfig,
) -> tuple[list[int], list[int], list[int], int, float | None]:
    """Split whole customer groups, honouring the configured fractions.

    ``StratifiedGroupKFold`` only produces equal-sized folds (60/20/20), which
    would silently ignore custom split fractions. Splitting the unique groups
    directly keeps every customer in one partition and lets the requested
    proportions drive the partition sizes.
    """
    codes, uniques = pd.factorize(pd.Series(groups), sort=True)
    n_groups = len(uniques)
    if n_groups < 5:
        raise PreprocessingError(
            CODE_SPLIT_IMPOSSIBLE,
            f"only {n_groups} unique customer ids; at least 5 are required for a grouped split",
            "Acquire a larger dataset or drop the repeated-id rows deliberately.",
        )

    order = np.argsort(codes, kind="stable")
    group_counts = np.bincount(codes, minlength=n_groups)
    members = np.split(order, np.cumsum(group_counts)[:-1])
    label_counts = np.zeros((n_groups, 2), dtype=int)
    np.add.at(label_counts, (codes, np.asarray(y, dtype=int)), 1)
    # Ties go to the churn class so the representative label is deterministic.
    group_labels = (label_counts[:, 1] >= label_counts[:, 0]).astype(int)

    overall = _positive_rate(y)
    rest_fraction = config.val_fraction + config.test_fraction
    all_groups = np.arange(n_groups)
    best: tuple[float, int, list[int], list[int], list[int]] | None = None
    for attempt in range(GROUPED_ATTEMPTS):
        candidate_seed = config.seed + attempt
        try:
            train_groups, rest_groups = train_test_split(
                all_groups,
                test_size=rest_fraction,
                stratify=group_labels,
                random_state=candidate_seed,
            )
            val_groups, test_groups = train_test_split(
                rest_groups,
                test_size=config.test_fraction / rest_fraction,
                stratify=group_labels[rest_groups],
                random_state=candidate_seed,
            )
        except ValueError:
            # Too few groups per class for group-level stratification: keep the
            # groups intact and fall back to a deterministic unstratified split.
            train_groups, rest_groups = train_test_split(
                all_groups,
                test_size=rest_fraction,
                random_state=candidate_seed,
            )
            val_groups, test_groups = train_test_split(
                rest_groups,
                test_size=config.test_fraction / rest_fraction,
                random_state=candidate_seed,
            )

        def rows(selected: np.ndarray) -> list[int]:
            return sorted(int(i) for group in selected for i in members[group])

        train_idx, val_idx, test_idx = rows(train_groups), rows(val_groups), rows(test_groups)
        if not train_idx or not val_idx or not test_idx:
            continue
        if any(
            len(_class_counts(y[indices])) != 2
            for indices in (train_idx, val_idx, test_idx)
        ):
            continue
        deviation = 0.0
        if overall is not None:
            for indices in (train_idx, val_idx, test_idx):
                split_rate = _positive_rate(y[indices])
                if split_rate is not None:
                    deviation = max(deviation, abs(split_rate - overall))
        if best is None or deviation < best[0]:
            best = (deviation, candidate_seed, train_idx, val_idx, test_idx)
        if deviation <= RATE_TOLERANCE:
            break

    if best is None:
        raise PreprocessingError(
            CODE_SPLIT_IMPOSSIBLE,
            "no grouped split placed both classes in every partition",
            "Acquire more customers per class or relax the split fractions.",
        )
    deviation, seed_used, train_idx, val_idx, test_idx = best
    return train_idx, val_idx, test_idx, seed_used, deviation


def make_split(
    frame: pd.DataFrame,
    *,
    dataset_id: str,
    dataset_sha256: str,
    run_id: str,
    target_column: str,
    id_column: str | None,
    config: PreprocessConfig,
    duplicate_rows_dropped: int = 0,
    target_missing_dropped: int = 0,
) -> SplitResult:
    """Create the train/validation/test split and its fingerprinted identity."""
    y = pd.to_numeric(frame[target_column], errors="coerce").to_numpy()
    if np.isnan(y).any():
        raise PreprocessingError(
            CODE_SPLIT_IMPOSSIBLE,
            "target still contains missing values after normalisation",
            "Exclude missing-target rows before splitting.",
        )
    y = y.astype(np.int8)
    _validate_sizes(y, config)

    warnings: list[str] = []
    grouped = False
    deviation: float | None = None
    seed_used = config.seed
    if id_column and id_column in frame.columns:
        ids = frame[id_column]
        if ids.duplicated().any():
            grouped = True
            groups = ids.astype("string")
            missing_ids = groups.isna()
            if bool(missing_ids.any()):
                placeholders = pd.Series(
                    [f"__missing_id_{position}" for position in range(len(groups))],
                    index=groups.index,
                    dtype="string",
                )
                groups = groups.where(~missing_ids, placeholders)
            train_idx, val_idx, test_idx, seed_used, deviation = _grouped_split(
                groups.to_numpy(), y, config
            )
            if deviation is not None and deviation > RATE_TOLERANCE:
                warnings.append(
                    f"grouped split class-rate deviation {deviation:.2%} exceeds "
                    f"the {RATE_TOLERANCE:.0%} tolerance; ids must not cross partitions"
                )
        else:
            train_idx, val_idx, test_idx, seed_used = _row_split(y, config)
    else:
        train_idx, val_idx, test_idx, seed_used = _row_split(y, config)

    indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    for name, values in indices.items():
        if not values:
            raise PreprocessingError(
                CODE_SPLIT_IMPOSSIBLE,
                f"split {name!r} is empty",
                "Adjust the split fractions or acquire a larger dataset.",
            )
        split_counts = _class_counts(y[values])
        if len(split_counts) != 2:
            raise PreprocessingError(
                CODE_SPLIT_IMPOSSIBLE,
                f"split {name!r} contains {len(split_counts)} class(es); both are required",
                "Acquire a larger dataset so every partition can hold both classes.",
            )
        if min(split_counts.values()) < config.min_class_count:
            raise PreprocessingError(
                CODE_SPLIT_IMPOSSIBLE,
                f"split {name!r} has only {min(split_counts.values())} rows of one class",
                "Acquire a larger dataset or lower min_class_count deliberately.",
            )
        if min(split_counts.values()) < 10:
            warnings.append(
                f"split {name!r} has only {min(split_counts.values())} minority rows; "
                "its estimates will be noisy"
            )

    total = len(indices["train"]) + len(indices["val"]) + len(indices["test"])
    assert total == len(y)
    assert not set(indices["train"]) & set(indices["val"])
    assert not set(indices["train"]) & set(indices["test"])
    assert not set(indices["val"]) & set(indices["test"])

    if grouped and id_column:
        seen: dict[str, str] = {}
        for name, values in indices.items():
            for value in frame.iloc[values][id_column].astype(str):
                previous = seen.get(value)
                if previous is not None and previous != name:
                    raise PreprocessingError(
                        CODE_SPLIT_IMPOSSIBLE,
                        f"customer id {value!r} appears in both {previous!r} and {name!r}",
                        "Use a grouped split or remove duplicate ids before splitting.",
                    )
                seen[value] = name

    class_counts = {name: _class_counts(y[values]) for name, values in indices.items()}
    class_rates = {name: _positive_rate(y[values]) for name, values in indices.items()}
    fingerprint = _split_fingerprint(
        seed=seed_used,
        dataset_sha256=dataset_sha256,
        train=train_idx,
        val=val_idx,
        test=test_idx,
    )
    spec = SplitSpec(
        run_id=run_id,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        strategy="stratified_group" if grouped else "stratified",
        seed=seed_used,
        ratios={
            "train": config.train_fraction,
            "val": config.val_fraction,
            "test": config.test_fraction,
        },
        sizes={name: len(values) for name, values in indices.items()},
        class_counts=class_counts,
        class_rates=class_rates,
        fingerprint=fingerprint,
        row_indices=indices,
        duplicate_rows_dropped=duplicate_rows_dropped,
        target_missing_dropped=target_missing_dropped,
        id_column=id_column,
    )
    return SplitResult(
        spec=spec,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        warnings=warnings,
    )
