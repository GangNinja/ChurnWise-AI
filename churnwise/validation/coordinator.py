"""Phase 4 orchestration: verify, load, profile, gate and persist.

The coordinator consumes only the Phase 3 result contract. It never downloads
or discovers by itself; candidate fallback calls the Phase 3 coordinator one
candidate at a time so a validation rejection can move to the next eligible
candidate without reimplementing acquisition.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from churnwise import __version__
from churnwise.acquisition.contracts import AcquisitionResult
from churnwise.acquisition.coordinator import AcquisitionCoordinator
from churnwise.acquisition.integrity import sha256_file
from churnwise.acquisition.store import DatasetStore
from churnwise.config import Settings
from churnwise.contracts import (
    DatasetCandidate,
    DiscoveryReport,
    SelectedCandidate,
    utc_now_iso,
)
from churnwise.eda.profile import (
    DatasetProfile,
    TargetInterpretation,
    build_profile,
    empty_profile,
)
from churnwise.validation.checks import run_checks
from churnwise.validation.contracts import (
    ValidationAttempt,
    ValidationFailure,
    ValidationReport,
    ValidationResult,
    ValidationStatus,
)
from churnwise.validation.gate import decide_gate
from churnwise.validation.loader import DatasetLoadError, load_dataset
from churnwise.validation.report import build_report, write_outputs, _atomic_write_json
from churnwise.validation.targets import detect_target

CODE_NO_CANDIDATES = "CW-VAL-001"
CODE_ALL_REJECTED = "CW-VAL-002"
CODE_ACQUISITION_FAILED = "CW-VAL-003"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _manifest_digest(manifest: Any, manifest_path: Path | None) -> str:
    if manifest_path is not None and manifest_path.is_file():
        try:
            return sha256_file(manifest_path)
        except OSError:
            pass
    payload = json.dumps(manifest.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_list(source: Any) -> list[DatasetCandidate]:
    if isinstance(source, DiscoveryReport):
        return [
            assessment.candidate
            for assessment in source.ranked()
            if assessment.eligible and assessment.duplicate_of is None
        ]
    if isinstance(source, SelectedCandidate):
        return [DatasetCandidate.from_dict(source.snapshot)]
    if isinstance(source, DatasetCandidate):
        return [source]
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        candidates: list[DatasetCandidate] = []
        seen: set[str] = set()
        for item in source:
            candidate = item.candidate if hasattr(item, "candidate") else item
            if not isinstance(candidate, DatasetCandidate):
                raise TypeError("validation candidates must be DatasetCandidate objects")
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)
            candidates.append(candidate)
        return candidates
    raise TypeError("validation expects DatasetCandidate(s), SelectedCandidate or DiscoveryReport")


class ValidationCoordinator:
    def __init__(
        self,
        settings: Settings,
        *,
        acquisition: AcquisitionCoordinator | None = None,
        now: Callable[[], str] = utc_now_iso,
        validator_version: str = __version__,
    ) -> None:
        self.settings = settings
        self.data_dir = Path(settings.data_dir)
        self.report_dir = Path(settings.report_dir)
        self.acquisition = acquisition or AcquisitionCoordinator(settings)
        self._now = now
        self.validator_version = validator_version
        self.store = DatasetStore(self.data_dir)

    # -- single artifact ----------------------------------------------------
    def validate(
        self,
        result: AcquisitionResult,
        *,
        persist: bool = True,
        attempts: list[ValidationAttempt] | None = None,
    ) -> ValidationResult:
        generated_at = self._now()
        if not result.succeeded or result.manifest is None or result.artifact is None:
            failure = ValidationFailure(
                code=CODE_ACQUISITION_FAILED,
                message="validation requires a successful Phase 3 acquisition result",
                remediation="Run acquisition again and inspect its structured failure.",
                causes=[result.failure.message] if result.failure else [],
            )
            return ValidationResult(
                status=ValidationStatus.REJECT,
                acquisition=result,
                attempts=list(attempts or []),
                failure=failure,
            )

        manifest = result.manifest
        artifact = result.artifact
        data_dir = Path(result.data_dir)
        manifest_path = (
            Path(result.manifest_path)
            if result.manifest_path
            else data_dir / "manifests" / f"{manifest.dataset_id}.json"
        )
        manifest_sha256 = _manifest_digest(manifest, manifest_path)

        if artifact.extracted_path:
            primary = data_dir / artifact.extracted_path
            expected_sha = manifest.file.extracted_sha256 or ""
            primary_format = primary.suffix.lstrip(".").lower()
        else:
            primary = data_dir / artifact.raw_path
            expected_sha = manifest.file.sha256
            primary_format = manifest.file.format

        integrity_ok, integrity_detail, artifact_sha256 = self._verify_artifact(
            data_dir,
            manifest,
            primary,
            expected_sha,
        )
        provenance = self._provenance(
            result,
            manifest_path,
            primary,
            artifact_sha256 or expected_sha,
        )

        if not integrity_ok:
            self.store.quarantine_manifest(manifest.dataset_id)
            target = TargetInterpretation(reason=integrity_detail)
            profile = empty_profile(
                dataset_id=manifest.dataset_id,
                format=primary_format,
                source_path=_display_path(primary),
                target=target,
            )
            checks = run_checks(
                None,
                None,
                integrity_ok=False,
                integrity_detail=integrity_detail,
            )
            gate = decide_gate(checks)
            report = build_report(
                dataset_id=manifest.dataset_id,
                manifest_sha256=manifest_sha256,
                artifact_sha256=artifact_sha256 or expected_sha,
                artifact_path=_display_path(primary),
                generated_at=generated_at,
                validator_version=self.validator_version,
                checks=checks,
                gate=gate,
                profile=profile,
                provenance=provenance,
                warnings=["artifact failed integrity verification and was not loaded"],
                notes=["The manifest was quarantined; acquisition must run again before this dataset is used."],
            )
            paths = (
                write_outputs(
                    report,
                    profile,
                    None,
                    report_dir=self.report_dir,
                    data_dir=self.data_dir,
                    generate_eda=False,
                )
                if persist
                else {}
            )
            return ValidationResult(
                status=gate.status,
                report=report,
                profile=profile,
                frame=None,
                acquisition=result,
                attempts=list(attempts or []),
                report_paths=paths,
            )

        try:
            loaded = load_dataset(primary, primary_format)
        except DatasetLoadError as exc:
            target = TargetInterpretation(reason=exc.message)
            profile = empty_profile(
                dataset_id=manifest.dataset_id,
                format=primary_format,
                source_path=_display_path(primary),
                target=target,
            )
            checks = run_checks(
                profile,
                target,
                integrity_ok=True,
                load_error=str(exc),
            )
            gate = decide_gate(checks)
            report = build_report(
                dataset_id=manifest.dataset_id,
                manifest_sha256=manifest_sha256,
                artifact_sha256=artifact_sha256,
                artifact_path=_display_path(primary),
                generated_at=generated_at,
                validator_version=self.validator_version,
                checks=checks,
                gate=gate,
                profile=profile,
                provenance=provenance,
                warnings=[str(exc)],
                notes=["No profile or EDA plots were produced because parsing failed."],
            )
            paths = (
                write_outputs(
                    report,
                    profile,
                    None,
                    report_dir=self.report_dir,
                    data_dir=self.data_dir,
                    generate_eda=False,
                )
                if persist
                else {}
            )
            return ValidationResult(
                status=gate.status,
                report=report,
                profile=profile,
                frame=None,
                acquisition=result,
                attempts=list(attempts or []),
                report_paths=paths,
            )

        frame = loaded.frame
        candidate = manifest.discovery.get("candidate", {}) if manifest.discovery else {}
        target = detect_target(
            frame,
            dataset_title=manifest.title,
            dataset_description=str(candidate.get("description") or ""),
            hint_target=candidate.get("target_column") if isinstance(candidate, dict) else None,
        )
        profile = build_profile(
            frame,
            dataset_id=manifest.dataset_id,
            format=loaded.format,
            source_path=_display_path(primary),
            target=target,
            duplicate_headers=loaded.duplicate_headers,
        )
        checks = run_checks(profile, target, integrity_ok=True)
        gate = decide_gate(checks)
        report = build_report(
            dataset_id=manifest.dataset_id,
            manifest_sha256=manifest_sha256,
            artifact_sha256=artifact_sha256,
            artifact_path=_display_path(primary),
            generated_at=generated_at,
            validator_version=self.validator_version,
            checks=checks,
            gate=gate,
            profile=profile,
            provenance=provenance,
            warnings=loaded.warnings,
            notes=[
                "Validation is descriptive only; no imputation, scaling, encoding, split or model fitting was performed.",
                "Missing target rows are excluded only in the in-memory normalized frame; the acquired artifact is untouched.",
                "Identifier columns are retained for output and excluded from predictive features.",
            ],
        )
        paths = (
            write_outputs(
                report,
                profile,
                frame,
                report_dir=self.report_dir,
                data_dir=self.data_dir,
            )
            if persist
            else {}
        )
        return ValidationResult(
            status=gate.status,
            report=report,
            profile=profile,
            frame=frame,
            acquisition=result,
            attempts=list(attempts or []),
            report_paths=paths,
        )

    # -- candidate fallback -------------------------------------------------
    def validate_candidates(
        self,
        source: Any,
        *,
        offline: bool | None = None,
        persist: bool = True,
        max_candidates: int | None = None,
    ) -> ValidationResult:
        candidates = _candidate_list(source)
        if max_candidates is not None:
            candidates = candidates[: max(0, max_candidates)]
        if not candidates:
            failure = ValidationFailure(
                code=CODE_NO_CANDIDATES,
                message="no eligible candidate was available for validation",
                remediation=(
                    "Run discovery again, or run without --offline so acquisition can populate the cache."
                ),
            )
            return ValidationResult(
                status=ValidationStatus.REJECT,
                failure=failure,
                attempts=[],
            )

        attempts: list[ValidationAttempt] = []
        for candidate in candidates:
            started = time.perf_counter()
            acquisition = self.acquisition.acquire(
                [candidate], offline=offline, origin="validation_fallback"
            )
            duration = int((time.perf_counter() - started) * 1000)
            if not acquisition.succeeded:
                failure = acquisition.failure
                attempts.append(
                    ValidationAttempt(
                        candidate_id=candidate.candidate_id,
                        stage="acquisition",
                        status="failed",
                        duration_ms=duration,
                        error_code=failure.code if failure else None,
                        message=failure.message if failure else "acquisition failed",
                        detail=failure.remediation if failure else None,
                    )
                )
                continue
            validation = self.validate(acquisition, persist=persist)
            attempts.append(
                ValidationAttempt(
                    candidate_id=candidate.candidate_id,
                    stage="validation",
                    status=validation.status.value,
                    duration_ms=duration,
                    error_code=(
                        validation.report.gate.blocking[0]
                        if validation.report and validation.report.gate.blocking
                        else None
                    ),
                    message=(
                        "; ".join(validation.report.gate.reasons)
                        if validation.report and validation.report.gate.reasons
                        else None
                    ),
                    report_path=validation.report.report_json_path if validation.report else None,
                )
            )
            if validation.can_preprocess:
                validation.attempts = attempts
                if persist:
                    self._write_attempts(validation, attempts)
                return validation

        failure = ValidationFailure(
            code=CODE_ALL_REJECTED,
            message="no candidate passed validation",
            remediation=(
                "Inspect reports/validation for each rejection, then run discovery again "
                "or acquire a different eligible candidate."
            ),
            causes=[
                f"{attempt.candidate_id}: {attempt.status} "
                f"{attempt.error_code or ''} {attempt.message or ''}".strip()
                for attempt in attempts
            ],
        )
        result = ValidationResult(
            status=ValidationStatus.REJECT,
            failure=failure,
            attempts=attempts,
        )
        if persist:
            self._write_attempts(result, attempts)
        return result

    # -- helpers ------------------------------------------------------------
    def _write_attempts(self, result: ValidationResult, attempts: list[ValidationAttempt]) -> None:
        path = self.report_dir / "validation" / "attempts.json"
        _atomic_write_json(
            path,
            {
                "schema_version": "1.0",
                "phase": 4,
                "generated_at": self._now(),
                "validator_version": self.validator_version,
                "status": result.status.value,
                "attempts": [attempt.to_dict() for attempt in attempts],
                "failure": result.failure.to_dict() if result.failure else None,
            },
        )

    def _verify_artifact(
        self,
        data_dir: Path,
        manifest: Any,
        primary: Path,
        expected_sha: str,
    ) -> tuple[bool, str, str]:
        problems: list[str] = []
        artifact_sha = ""
        raw = data_dir / manifest.file.raw_path
        if not raw.is_file():
            problems.append(f"raw copy is missing: {raw}")
        else:
            try:
                raw_sha = sha256_file(raw)
            except OSError as exc:
                problems.append(f"raw copy could not be read: {exc}")
            else:
                if raw_sha != manifest.file.sha256:
                    problems.append(
                        "raw copy SHA-256 does not match the manifest"
                    )
        if not primary.is_file():
            problems.append(f"validation artifact is missing: {primary}")
        else:
            try:
                artifact_sha = sha256_file(primary)
            except OSError as exc:
                problems.append(f"validation artifact could not be read: {exc}")
            else:
                if expected_sha and artifact_sha != expected_sha:
                    problems.append(
                        "validation artifact SHA-256 does not match the recorded digest"
                    )
                elif not expected_sha:
                    problems.append("manifest does not record a digest for the validation artifact")
        if problems:
            return False, "; ".join(problems), artifact_sha
        return True, "", artifact_sha

    def _provenance(
        self,
        result: AcquisitionResult,
        manifest_path: Path,
        primary: Path,
        artifact_sha256: str,
    ) -> dict[str, Any]:
        manifest = result.manifest
        assert manifest is not None
        license_name = manifest.source.license_name or "unknown"
        provenance: dict[str, Any] = {
            "dataset_id": manifest.dataset_id,
            "title": manifest.title,
            "source_id": manifest.source.source_id,
            "provider": manifest.source.provider,
            "source_dataset_id": manifest.source.dataset_id,
            "download_url": manifest.source.download_url,
            "landing_url": manifest.source.landing_url,
            "license": f"{license_name} ({manifest.source.license_status})",
            "retrieved_at": manifest.retrieved_at,
            "run_id": manifest.run_id,
            "manifest_path": _display_path(manifest_path),
            "acquisition_attempt": manifest.acquisition.attempt,
            "candidate_rank": manifest.acquisition.candidate_rank,
            "fallback_of": manifest.acquisition.fallback_of,
            "cached": result.cached,
            "artifact_path": _display_path(primary),
            "artifact_sha256": artifact_sha256,
            "dataset_sha256": manifest.dataset_sha256,
        }
        return provenance
