"""Attach Phase 8 SHAP drivers to Phase 9 assessments.

Drivers are copied from :class:`churnwise.explain.contracts.LocalExplanation`
objects; Phase 9 never computes a second attribution method. Missing or failed
explanations are reported explicitly and are never replaced by guessed reasons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from churnwise.config import Settings
from churnwise.explain import ExplainCoordinator, ExplainConfig, ExplainError, validate_explain_id
from churnwise.intelligence.contracts import RiskAssessment, RiskDriver
from churnwise.intelligence.errors import CODE_EXPLANATION_FAILED, RiskError


def driver_from_contribution(item: Any) -> RiskDriver:
    return RiskDriver(
        feature=str(getattr(item, "feature", "")),
        kind=str(getattr(item, "kind", "unknown")),
        contribution=float(getattr(item, "contribution", 0.0)),
        direction=str(getattr(item, "direction", "neutral")),
        value=getattr(item, "value", None),
        active_level=getattr(item, "active_level", None),
    )


def explanation_index(items: Sequence[Any]) -> dict[int, Any]:
    """Map Phase 8 row positions to their local explanation objects."""
    result: dict[int, Any] = {}
    for item in items:
        position = getattr(item, "row_position", None)
        if position is not None:
            result[int(position)] = item
    return result


def attach_explanation(item: RiskAssessment, local: Any | None) -> None:
    if local is None:
        item.explanation_status = "unavailable"
        item.explanation_reason = "no precomputed SHAP explanation exists for this row"
        return
    item.top_positive = [
        driver_from_contribution(contribution)
        for contribution in getattr(local, "top_positive", []) or []
    ]
    item.top_negative = [
        driver_from_contribution(contribution)
        for contribution in getattr(local, "top_negative", []) or []
    ]
    item.explanation_summary = str(getattr(local, "summary", "") or "")
    item.explanation_status = "available"
    item.explanation_reason = ""
    item.warnings.extend(
        warning
        for warning in list(getattr(local, "warnings", []) or [])
        if warning not in item.warnings
    )


def attach_index(
    records: Sequence[RiskAssessment], index: Mapping[int, Any]
) -> None:
    """Attach all available local explanations; mark the rest unavailable."""
    for item in records:
        attach_explanation(item, index.get(int(item.input_position)))


def load_phase8_index(
    settings: Settings,
    explain_run_id: str,
    *,
    model_sha256: str | None,
    preprocessing_run_id: str,
) -> dict[int, Any]:
    """Load a persisted Phase 8 matrix run and verify its provenance."""
    try:
        validate_explain_id(explain_run_id)
    except ExplainError as exc:
        raise RiskError(
            CODE_EXPLANATION_FAILED,
            exc.message,
            exc.remediation or "Use the id printed by a previous explain command.",
            dict(exc.details),
        ) from exc
    root = Path(settings.artifact_dir) / "explanations" / explain_run_id
    manifest_path = root / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RiskError(
            CODE_EXPLANATION_FAILED,
            f"Phase 8 explanation run {explain_run_id!r} could not be read",
            "Run `python -m churnwise.cli explain` for this population first.",
        ) from exc
    if not isinstance(manifest, dict):
        raise RiskError(
            CODE_EXPLANATION_FAILED,
            f"Phase 8 explanation run {explain_run_id!r} has unreadable metadata",
            "Regenerate the Phase 8 explanation run.",
        )
    if str(manifest.get("mode") or "") != "matrix":
        raise RiskError(
            CODE_EXPLANATION_FAILED,
            "the supplied explanation run is not a population/matrix run",
            "Use an explanation run produced by the Phase 9 scoring population.",
        )
    if str(manifest.get("preprocessing_run_id") or "") != str(preprocessing_run_id):
        raise RiskError(
            CODE_EXPLANATION_FAILED,
            "the explanation run uses a different preprocessing run",
            "Regenerate explanations for the shipped model/preprocessor pair.",
        )
    model_block = manifest.get("model")
    recorded_model = (
        model_block.get("model_sha256") if isinstance(model_block, dict) else None
    )
    if model_sha256 and recorded_model and str(recorded_model) != str(model_sha256):
        raise RiskError(
            CODE_EXPLANATION_FAILED,
            "the explanation run uses a different model digest",
            "Regenerate explanations after changing the shipped model.",
        )
    index: dict[int, Any] = {}
    from churnwise.explain.contracts import LocalExplanation

    for path in sorted((root / "local").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RiskError(
                CODE_EXPLANATION_FAILED,
                f"explanation file {path.name} could not be read",
                "Regenerate the Phase 8 explanation run.",
            ) from exc
        if not isinstance(payload, dict):
            raise RiskError(
                CODE_EXPLANATION_FAILED,
                f"explanation file {path.name} is not a JSON object",
                "Regenerate the Phase 8 explanation run.",
            )
        position = payload.get("row_position")
        if position is None:
            continue
        try:
            index[int(position)] = _local_from_payload(payload, LocalExplanation)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RiskError(
                CODE_EXPLANATION_FAILED,
                f"explanation file {path.name} is malformed "
                f"({type(exc).__name__})",
                "Regenerate the Phase 8 explanation run.",
            ) from exc
    return index


def _local_from_payload(payload: dict[str, Any], _: Any) -> Any:
    """Rebuild the subset of Phase 8 local fields Phase 9 needs."""
    from churnwise.explain.contracts import Contribution, LocalExplanation

    def contributions(items: Any) -> list[Contribution]:
        result: list[Contribution] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            result.append(
                Contribution(
                    feature=str(item.get("feature") or ""),
                    kind=str(item.get("kind") or "unknown"),
                    contribution=float(item.get("contribution", 0.0)),
                    direction=str(item.get("direction") or "neutral"),
                    value=item.get("value"),
                    active_level=item.get("active_level"),
                )
            )
        return result

    return LocalExplanation(
        family=str(payload.get("family") or ""),
        output_space=str(payload.get("output_space") or "probability"),
        probability=float(payload.get("probability", 0.0)),
        predicted_class=int(payload.get("predicted_class", 0)),
        threshold=float(payload.get("threshold", 0.5)),
        positive_label=payload.get("positive_label"),
        base_value=float(payload.get("base_value", 0.0)),
        model_output=float(payload.get("model_output", 0.0)),
        top_positive=contributions(payload.get("top_positive")),
        top_negative=contributions(payload.get("top_negative")),
        summary=str(payload.get("summary") or ""),
        row_position=int(payload["row_position"]),
        warnings=list(payload.get("warnings") or []),
    )


def explain_matrix_rows(
    settings: Settings,
    matrix: Any,
    *,
    positions: Sequence[int],
    benchmark_run_id: str | None,
    explain_run_id: str | None,
    top_k: int,
    seed: int,
    progress: Any = None,
    persist: bool = True,
) -> dict[int, Any]:
    """Compute Phase 8 drivers for selected transformed rows."""
    config = ExplainConfig(top_k=int(top_k), seed=int(seed), local_examples=0)
    coordinator = ExplainCoordinator(settings, config)
    try:
        result = coordinator.explain_matrix(
            matrix,
            benchmark_run_id=benchmark_run_id,
            explain_run_id=explain_run_id,
            row_positions=list(positions),
            persist=persist,
            progress=progress,
        )
    except ExplainError as exc:
        raise RiskError(
            CODE_EXPLANATION_FAILED,
            exc.message,
            exc.remediation or "Run Phase 8 explanations for this population.",
            dict(exc.details),
        ) from exc
    return explanation_index(result.local_explanations)


__all__ = [
    "attach_explanation",
    "attach_index",
    "driver_from_contribution",
    "explain_matrix_rows",
    "explanation_index",
    "load_phase8_index",
]
