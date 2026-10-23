"""Discovery gates (G1–G9) and the documented ranking formula.

Gates answer "may this dataset be recommended at all?"; scoring answers "which
eligible dataset is the better first choice?". A high score never overrides a
failed gate, and a high rank is not a claim about data quality — Phase 4
validates the data.
"""

from __future__ import annotations

from churnwise.contracts import (
    CandidateAssessment,
    DatasetCandidate,
    GateResult,
    GateStatus,
    LicenseStatus,
    ScoreComponent,
    SearchQuery,
)
from churnwise.discovery.registry import source_trust

MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200 MB, enforced again in Phase 3
MIN_POSITIVE_RATE = 0.02
MAX_POSITIVE_RATE = 0.50

#: Column-name signals that a table describes customer accounts rather than an
#: unrelated prediction problem. Deliberately broad and surname-free.
DOMAIN_HINTS: tuple[str, ...] = (
    "tenure",
    "contract",
    "subscription",
    "monthly",
    "charge",
    "usage",
    "service",
    "account",
    "plan",
    "payment",
    "balance",
    "customer",
)

#: Unknown is acceptable only for gates whose evidence Phase 3/4 can supply.
BLOCKING_UNKNOWN = frozenset({"G1", "G2", "G3", "G5", "G6", "G7", "G9"})

GATE_NAMES = {
    "G1": "supported_format",
    "G2": "minimum_rows",
    "G3": "binary_target_identified",
    "G4": "class_balance_in_range",
    "G5": "sufficient_predictors",
    "G6": "licence_policy",
    "G7": "no_direct_pii",
    "G8": "download_size_within_cap",
    "G9": "source_metadata_available",
}


def customer_domain_hits(candidate: DatasetCandidate) -> list[str]:
    """Return the customer-domain hints present in non-target column names."""
    names = [
        (field.name or "").lower()
        for field in (candidate.schema_fields or [])
        if field.role != "target"
    ]
    return sorted({hint for hint in DOMAIN_HINTS if any(hint in name for name in names)})


def _gate(gate_id: str, status: GateStatus, reason: str, **evidence: object) -> GateResult:
    return GateResult(
        gate_id=gate_id,
        name=GATE_NAMES[gate_id],
        status=status,
        reason=reason,
        evidence=dict(evidence),
    )


def evaluate_gates(candidate: DatasetCandidate, query: SearchQuery) -> list[GateResult]:
    gates: list[GateResult] = []

    # G1 — format
    if candidate.format in query.formats:
        gates.append(_gate("G1", GateStatus.PASS, f"format {candidate.format!r} is supported"))
    else:
        gates.append(
            _gate(
                "G1",
                GateStatus.FAIL,
                f"format {candidate.format!r} is not in the allowed set {list(query.formats)}",
                format=candidate.format,
            )
        )

    # G2 — rows
    if candidate.row_count is None:
        gates.append(_gate("G2", GateStatus.FAIL, "row count not reported; minimum size cannot be confirmed"))
    elif candidate.row_count < query.min_rows:
        gates.append(
            _gate(
                "G2",
                GateStatus.FAIL,
                f"{candidate.row_count} rows is below the minimum of {query.min_rows}",
                row_count=candidate.row_count,
            )
        )
    elif query.max_rows is not None and candidate.row_count > query.max_rows:
        gates.append(
            _gate(
                "G2",
                GateStatus.FAIL,
                f"{candidate.row_count} rows exceeds the maximum of {query.max_rows}",
            )
        )
    else:
        gates.append(_gate("G2", GateStatus.PASS, f"{candidate.row_count} rows", row_count=candidate.row_count))

    # G3 — binary target
    if candidate.target_column is None:
        gates.append(_gate("G3", GateStatus.FAIL, "no target column identified"))
    elif candidate.target_class_count is None:
        gates.append(
            _gate(
                "G3",
                GateStatus.FAIL,
                f"target {candidate.target_column!r} found but binary cardinality is unconfirmed",
                target_column=candidate.target_column,
            )
        )
    elif candidate.target_class_count != 2:
        gates.append(
            _gate(
                "G3",
                GateStatus.FAIL,
                f"target {candidate.target_column!r} has {candidate.target_class_count} classes, not 2",
            )
        )
    else:
        gates.append(
            _gate(
                "G3",
                GateStatus.PASS,
                f"binary target {candidate.target_column!r} with 2 classes",
                target_column=candidate.target_column,
                target_class_count=2,
            )
        )

    # G4 — class balance (non-blocking when unknown)
    if candidate.class_balance is None:
        gates.append(
            _gate(
                "G4",
                GateStatus.UNKNOWN,
                "class balance not reported; Phase 4 must verify it",
            )
        )
    else:
        total = sum(candidate.class_balance.values())
        rate = candidate.class_balance["minority"] / total if total else 0.0
        if MIN_POSITIVE_RATE <= rate <= MAX_POSITIVE_RATE:
            gates.append(
                _gate(
                    "G4",
                    GateStatus.PASS,
                    f"minority class share {rate:.2%} is within [2%, 50%]",
                    positive_rate=round(rate, 6),
                )
            )
        else:
            gates.append(
                _gate(
                    "G4",
                    GateStatus.FAIL,
                    f"minority class share {rate:.2%} is outside [2%, 50%]",
                    positive_rate=round(rate, 6),
                )
            )

    # G5 — predictors (configurable minimum and optional type requirements)
    if candidate.feature_count is None:
        gates.append(_gate("G5", GateStatus.FAIL, "feature count not reported"))
    else:
        id_columns = sum(1 for f in (candidate.schema_fields or []) if f.role == "id")
        predictors = max(candidate.feature_count - 1 - id_columns, 0)
        if predictors < query.min_predictors:
            gates.append(
                _gate(
                    "G5",
                    GateStatus.FAIL,
                    f"only {predictors} predictor columns; at least {query.min_predictors} required",
                )
            )
        elif query.require_numeric_features and candidate.has_numeric is not True:
            gates.append(
                _gate("G5", GateStatus.FAIL, "a numeric feature is required but not confirmed")
            )
        elif query.require_categorical_features and candidate.has_categorical is not True:
            gates.append(
                _gate("G5", GateStatus.FAIL, "a categorical feature is required but not confirmed")
            )
        elif query.require_customer_features and not customer_domain_hits(candidate):
            gates.append(
                _gate("G5", GateStatus.FAIL, "no customer-related feature column detected")
            )
        else:
            gates.append(
                _gate("G5", GateStatus.PASS, f"{predictors} predictor columns", predictors=predictors)
            )

    # G6 — licence
    status = candidate.license_status
    if status in {LicenseStatus.VERIFIED, LicenseStatus.PROJECT_OWNED}:
        gates.append(
            _gate("G6", GateStatus.PASS, f"licence recorded as {candidate.license_name!r}")
        )
    elif status is LicenseStatus.RESTRICTED:
        gates.append(_gate("G6", GateStatus.FAIL, "source declares a restricted licence"))
    elif query.allow_unverified_license:
        gates.append(
            _gate(
                "G6",
                GateStatus.PASS,
                f"licence status {status.value} accepted by explicit opt-in",
            )
        )
    else:
        gates.append(
            _gate(
                "G6",
                GateStatus.FAIL,
                f"licence status {status.value}; rerun with --allow-unverified-license to accept it",
            )
        )

    # G7 — PII
    if candidate.schema_fields is None:
        gates.append(
            _gate("G7", GateStatus.UNKNOWN, "column metadata unavailable; PII cannot be ruled out")
        )
    else:
        pii = candidate.pii_columns()
        if pii:
            gates.append(_gate("G7", GateStatus.FAIL, f"possible PII columns: {', '.join(pii)}"))
        else:
            gates.append(_gate("G7", GateStatus.PASS, "no direct-PII column names detected"))

    # G8 — size (non-blocking when unknown)
    if candidate.size_bytes is None:
        gates.append(
            _gate("G8", GateStatus.UNKNOWN, "file size unknown; Phase 3 enforces the download cap")
        )
    elif candidate.size_bytes <= MAX_DOWNLOAD_BYTES:
        gates.append(
            _gate("G8", GateStatus.PASS, f"{candidate.size_bytes} bytes is within the 200 MB cap")
        )
    else:
        gates.append(
            _gate("G8", GateStatus.FAIL, f"{candidate.size_bytes} bytes exceeds the 200 MB cap")
        )

    # G9 — metadata access
    gates.append(
        _gate("G9", GateStatus.PASS, f"metadata retrieved from {candidate.source_id}")
    )

    return gates


def is_eligible(gates: list[GateResult]) -> bool:
    if any(g.status is GateStatus.FAIL for g in gates):
        return False
    blocking = [g for g in gates if g.gate_id in BLOCKING_UNKNOWN and g.status is GateStatus.UNKNOWN]
    return not blocking


def _target_confidence(candidate: DatasetCandidate) -> tuple[float, str]:
    value = 0.0
    reasons: list[str] = []
    if candidate.target_column:
        value += 0.40
        reasons.append("target column declared")
    if candidate.target_class_count == 2:
        value += 0.35
        reasons.append("binary cardinality known")
    if candidate.target_name_is_churn_like():
        value += 0.15
        reasons.append("target name is churn-like")
    if candidate.class_balance is not None:
        value += 0.10
        reasons.append("class balance reported")
    return min(value, 1.0), ", ".join(reasons) or "no target evidence"


def _domain_signal(candidate: DatasetCandidate) -> tuple[float, str]:
    hits = customer_domain_hits(candidate)
    if hits:
        return 1.0, f"customer-domain columns: {', '.join(hits)}"
    return 0.0, "no customer-domain column names detected"


def _schema_fit(candidate: DatasetCandidate) -> tuple[float, str]:
    value = 0.0
    details: list[str] = []
    if candidate.feature_count:
        ratio = min(max((candidate.feature_count - 1) / 30.0, 0.0), 1.0)
        value += 0.40 * ratio
        details.append(f"{candidate.feature_count - 1} predictors")
    if candidate.has_numeric:
        value += 0.25
        details.append("numeric features")
    if candidate.has_categorical:
        value += 0.15
        details.append("categorical features")
    if candidate.schema_fields:
        value += 0.10
        details.append("column metadata available")
    domain_value, domain_detail = _domain_signal(candidate)
    value += 0.10 * domain_value
    details.append(domain_detail)
    return min(value, 1.0), ", ".join(details)


def _licence_clarity(candidate: DatasetCandidate) -> tuple[float, str]:
    mapping = {
        LicenseStatus.VERIFIED: (1.0, "licence explicitly recorded"),
        LicenseStatus.PROJECT_OWNED: (1.0, "project-owned data"),
        LicenseStatus.UNVERIFIED: (0.25, "licence unverified"),
        LicenseStatus.UNKNOWN: (0.10, "licence unknown"),
        LicenseStatus.RESTRICTED: (0.0, "licence restricted"),
    }
    return mapping[candidate.license_status]


def _size_adequacy(candidate: DatasetCandidate, min_rows: int) -> tuple[float, str]:
    if candidate.row_count is None:
        return 0.30, "row count unknown"
    if candidate.row_count < min_rows:
        return 0.0, "below minimum size"
    if candidate.row_count <= 500_000:
        return 1.0, "size suitable for local processing"
    return 0.70, "large dataset; local processing will be slower"


def score_candidate(
    candidate: DatasetCandidate, query: SearchQuery
) -> tuple[float, list[ScoreComponent]]:
    components: list[ScoreComponent] = []
    target_value, target_detail = _target_confidence(candidate)
    schema_value, schema_detail = _schema_fit(candidate)
    licence_value, licence_detail = _licence_clarity(candidate)
    trust_value = source_trust(candidate.source_id)
    size_value, size_detail = _size_adequacy(candidate, query.min_rows)

    for name, value, weight, detail in (
        ("target_confidence", target_value, 0.40, target_detail),
        ("schema_fit", schema_value, 0.20, schema_detail),
        ("licence_clarity", licence_value, 0.15, licence_detail),
        ("source_trust", trust_value, 0.15, f"registry trust for {candidate.source_id}"),
        ("size_adequacy", size_value, 0.10, size_detail),
    ):
        components.append(ScoreComponent(name=name, value=value, weight=weight, detail=detail))

    # Keep full precision internally; reports round at serialization time.
    total = sum(c.value * c.weight for c in components)
    return total, components


def assess_candidate(candidate: DatasetCandidate, query: SearchQuery) -> CandidateAssessment:
    gates = evaluate_gates(candidate, query)
    score, components = score_candidate(candidate, query)
    return CandidateAssessment(
        candidate=candidate,
        gates=gates,
        components=components,
        score=score,
        eligible=is_eligible(gates),
    )
