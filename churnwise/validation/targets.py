"""Autonomous target-column detection and churn normalisation.

The detector uses names, observed values and dataset context. It never claims a
positive class from a bare ``0/1`` column whose name says nothing about churn:
such a column is reported as low-confidence and the gate rejects it rather than
silently guessing. Retention-style targets are inverted explicitly so every
downstream phase receives ``1 = churned``.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from churnwise.eda.profile import (
    CLASS_LABEL_VALUES,
    TARGET_HINTS,
    TargetInterpretation,
    dedupe_columns,
    normalize_label,
)

CHURN_NAME_HINTS = (
    "churn",
    "churned",
    "attrition",
    "cancel",
    "cancelled",
    "canceled",
    "exited",
    "exit",
    "left",
    "lost",
    "terminated",
)
RETENTION_NAME_HINTS = (
    "retain",
    "retained",
    "active",
    "stayed",
    "renewed",
    "loyal",
)
NEUTRAL_TARGET_NAMES = {"class", "target", "label", "outcome", "y"}
NEGATIVE_TARGET_NAME = re.compile(
    r"(reason|date|time|amount|charge|score|probability|id$|_id$|name$|email$|phone$)",
    re.IGNORECASE,
)
POSITIVE_LABELS = {
    "yes",
    "y",
    "true",
    "1",
    "churn",
    "churned",
    "cancelled",
    "canceled",
    "exited",
    "attrited",
    "left",
    "lost",
    "inactive",
    "terminated",
}
NEGATIVE_LABELS = {
    "no",
    "n",
    "false",
    "0",
    "not churned",
    "not_churned",
    "not churn",
    "not_churn",
    "retained",
    "stayed",
    "active",
    "current",
    "renewed",
}
GENERIC_PAIRS = (
    ({"yes"}, {"no"}),
    ({"true"}, {"false"}),
    ({"1"}, {"0"}),
    ({"y"}, {"n"}),
)


def _name_score(name: str) -> float:
    lowered = name.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "", lowered)
    if NEGATIVE_TARGET_NAME.search(lowered):
        return -1.0
    if any(hint in normalized for hint in CHURN_NAME_HINTS):
        return 0.8
    if any(hint in normalized for hint in RETENTION_NAME_HINTS):
        return 0.7
    if normalized in NEUTRAL_TARGET_NAMES:
        return 0.5
    if "status" in normalized:
        return 0.55
    return 0.0


def _target_context(title: str, description: str, hint_target: str | None) -> str:
    context = " ".join(part for part in (title, description, hint_target or "") if part).lower()
    return context


def _generic_pair(labels: set[str]) -> tuple[str, str] | None:
    for positive, negative in GENERIC_PAIRS:
        if labels == positive | negative:
            return next(iter(positive)), next(iter(negative))
    return None


def _semantic_pair(labels: set[str]) -> tuple[str, str] | None:
    positives = labels & POSITIVE_LABELS
    negatives = labels & NEGATIVE_LABELS
    if len(positives) == 1 and len(negatives) == 1:
        return next(iter(positives)), next(iter(negatives))
    return None


def _interpret(column: str, labels: list[str], *, context: str) -> dict[str, Any]:
    """Return mapping, positive/negative labels and confidence for a column."""
    lowered = column.strip().lower()
    normalized_name = re.sub(r"[^a-z0-9]+", "", lowered)
    churn_name = any(hint in normalized_name for hint in CHURN_NAME_HINTS)
    retention_name = any(hint in normalized_name for hint in RETENTION_NAME_HINTS)
    neutral_name = normalized_name in NEUTRAL_TARGET_NAMES or normalized_name == ""

    if len(labels) != 2:
        return {
            "mapping": {},
            "positive_label": None,
            "negative_label": None,
            "confidence": "low",
            "method": "target does not have exactly two classes",
            "reason": "positive-class interpretation requires exactly two classes",
        }

    label_set = {label.lower() for label in labels}
    generic = _generic_pair(label_set)
    semantic = _semantic_pair(label_set)
    if generic is not None:
        positive, negative = generic
        if retention_name:
            positive, negative = negative, positive
            method = "retention-style target name: inverted so 1 = churned"
        elif churn_name:
            method = "churn-style target name: conventional positive label is churn"
        else:
            method = "neutral target name: conventional positive label, unverified semantics"
            mapping = {label: (1 if label.lower() == positive else 0) for label in labels}
            return {
                "mapping": mapping,
                "positive_label": next(label for label in labels if label.lower() == positive),
                "negative_label": next(label for label in labels if label.lower() == negative),
                "confidence": "low",
                "method": method,
                "reason": (
                    "target name is neutral and positive-class semantics cannot be "
                    "confirmed; refusing to assume that 1/Yes means churn"
                ),
            }
        mapping = {
            label: (1 if label.lower() == positive else 0) for label in labels
        }
        confidence = "high" if (churn_name or retention_name) else "medium"
        if neutral_name and not churn_name and not retention_name:
            confidence = "medium"
        return {
            "mapping": mapping,
            "positive_label": next(label for label in labels if label.lower() == positive),
            "negative_label": next(label for label in labels if label.lower() == negative),
            "confidence": confidence,
            "method": method,
            "reason": None,
        }

    if semantic is not None:
        positive, negative = semantic
        if retention_name:
            positive, negative = negative, positive
            method = "retention-style target name: inverted so 1 = churned"
        else:
            method = "observed labels explicitly identify churn and non-churn"
        mapping = {label: (1 if label.lower() == positive else 0) for label in labels}
        confidence = "high" if (churn_name or retention_name or len(semantic) == 2) else "medium"
        return {
            "mapping": mapping,
            "positive_label": next(label for label in labels if label.lower() == positive),
            "negative_label": next(label for label in labels if label.lower() == negative),
            "confidence": confidence,
            "method": method,
            "reason": None,
        }

    # A churn-named 0/1 column is still interpretable from its name: 1 means
    # churn, which is an explicit inference rather than a blind assumption.
    if churn_name and label_set <= {"0", "1"}:
        mapping = {"1": 1, "0": 0}
        return {
            "mapping": mapping,
            "positive_label": "1",
            "negative_label": "0",
            "confidence": "high",
            "method": "churn-style target name with numeric 0/1 values",
            "reason": None,
        }
    if retention_name and label_set <= {"0", "1"}:
        mapping = {"1": 0, "0": 1}
        return {
            "mapping": mapping,
            "positive_label": "0",
            "negative_label": "1",
            "confidence": "high",
            "method": "retention-style target name: inverted so 1 = churned",
            "reason": None,
        }

    return {
        "mapping": {},
        "positive_label": None,
        "negative_label": None,
        "confidence": "low",
        "method": "class labels could not be mapped to churn",
        "reason": f"unrecognised binary labels {sorted(labels)!r}",
    }


def detect_target(
    frame: pd.DataFrame,
    *,
    dataset_title: str = "",
    dataset_description: str = "",
    hint_target: str | None = None,
) -> TargetInterpretation:
    """Score candidate target columns and interpret the best binary candidate."""
    frame, _ = dedupe_columns(frame)
    context = _target_context(dataset_title, dataset_description, hint_target)
    candidates: list[dict[str, Any]] = []
    for column in frame.columns:
        if column.lower() in {"unnamed: 0"}:
            continue
        series = frame[column]
        labels_series = series.map(normalize_label).dropna()
        counts = labels_series.value_counts()
        labels = [str(value) for value in counts.index.tolist()]
        unique_count = int(labels_series.nunique())
        missing_count = int(series.isna().sum())
        missing_pct = missing_count / len(series) if len(series) else 0.0
        name_score = _name_score(column)
        label_set = {label.lower() for label in labels}
        has_known_labels = bool(label_set & (POSITIVE_LABELS | NEGATIVE_LABELS)) or bool(
            _generic_pair(label_set)
        )
        if name_score <= 0 and not (unique_count == 2 and has_known_labels):
            continue
        score = name_score
        if unique_count == 2:
            score += 0.30
        elif unique_count > 2:
            score += 0.10
        if missing_pct <= 0.05:
            score += 0.10
        elif missing_pct > 0.20:
            score -= 0.40
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            score += 0.05
        if column == hint_target:
            score += 0.20
        if any(hint in context for hint in CHURN_NAME_HINTS) and name_score <= 0:
            score += 0.05
        if unique_count == 2 and has_known_labels:
            score += 0.15
        if score <= 0:
            continue
        candidates.append(
            {
                "column": column,
                "score": round(float(score), 4),
                "name_score": name_score,
                "unique_count": unique_count,
                "missing_pct": missing_pct,
                "labels": labels,
                "has_known_labels": has_known_labels,
            }
        )

    candidates.sort(key=lambda item: (-item["score"], item["column"]))
    if not candidates:
        return TargetInterpretation(
            reason="no plausible churn target column was found",
            candidates=[],
        )

    best = candidates[0]
    ambiguous = False
    ambiguity_reason: str | None = None
    if len(candidates) > 1:
        second = candidates[1]
        strong = [
            item
            for item in candidates
            if item["name_score"] >= 0.6 and item["unique_count"] == 2
        ]
        if (
            len(strong) > 1
            and second["score"] >= best["score"] - 0.15
            and best["name_score"] >= 0.6
            and second["name_score"] >= 0.6
        ):
            ambiguous = True
            ambiguity_reason = (
                f"multiple plausible binary targets: {best['column']!r} and "
                f"{second['column']!r}; no target was selected"
            )

    if best["score"] < 0.75:
        return TargetInterpretation(
            reason="no candidate target was confident enough to select",
            candidates=candidates[:5],
        )

    labels = best["labels"]
    interpretation = _interpret(best["column"], labels, context=context)
    counts_series = frame[best["column"]].map(normalize_label).dropna().value_counts()
    counts = {str(key): int(value) for key, value in counts_series.items()}
    missing_count = int(frame[best["column"]].isna().sum())
    target = TargetInterpretation(
        column=best["column"],
        classes=labels,
        counts=counts,
        mapping=interpretation["mapping"],
        positive_label=interpretation["positive_label"],
        negative_label=interpretation["negative_label"],
        confidence=interpretation["confidence"],
        method=interpretation["method"],
        reason=ambiguity_reason or interpretation["reason"],
        ambiguous=ambiguous,
        candidates=candidates[:5],
        missing_count=missing_count,
        missing_pct=(missing_count / len(frame) if len(frame) else 0.0),
    )
    return target


def normalize_target(series: pd.Series, target: TargetInterpretation) -> pd.Series:
    """Map original target labels to ``1 = churned`` / ``0 = not churned``."""
    if target.column is None or not target.mapping:
        raise ValueError("target has no class mapping")
    labels = series.map(normalize_label)
    mapped = labels.map(target.mapping)
    return mapped.astype("Int64")
