"""Acquisition coordinator.

Consumes Phase 2 candidates in rank order, reuses verified cache entries, and
falls back to the next eligible candidate only after recording the failure of
the previous one. It never fabricates a success or silently substitutes data.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from churnwise.acquisition.contracts import (
    ACQUISITION_FORMATS,
    AcquisitionArtifact,
    AcquisitionAttempt,
    AcquisitionFailure,
    AcquisitionResult,
    AcquisitionStatus,
    AttemptStatus,
    DatasetManifest,
    ManifestAcquisition,
    ManifestFile,
    ManifestIntegrity,
    ManifestSource,
    utc_now_iso,
)
from churnwise.acquisition.downloader import (
    DownloadReceipt,
    RequestsDownloadTransport,
    StreamingDownloader,
)
from churnwise.acquisition.errors import (
    CATEGORY_ARCHIVE,
    CATEGORY_CACHE,
    CATEGORY_CANDIDATE,
    CATEGORY_FORMAT,
    CATEGORY_INTEGRITY,
    CATEGORY_INTERRUPTED,
    CATEGORY_NETWORK,
    CATEGORY_OFFLINE,
    CATEGORY_PERMISSION,
    CATEGORY_STORAGE,
    CODE_ACCESS,
    CODE_ALL_FAILED,
    CODE_CACHE_INVALID,
    CODE_CORRUPT_ARCHIVE,
    CODE_INTEGRITY,
    CODE_INTERRUPTED,
    CODE_NO_CANDIDATE,
    CODE_OFFLINE,
    CODE_SIZE_LIMIT,
    CODE_STORAGE,
    CODE_UNEXPECTED_CONTENT,
    CODE_UNSAFE_ARCHIVE,
    AcquisitionError,
    failure,
)
from churnwise.acquisition.integrity import (
    ArchiveInfo,
    inspect_download,
    sha256_file,
    extract_archive,
)
from churnwise.acquisition.provenance import (
    candidate_fingerprint,
    publisher_checksum,
    sanitize_url,
    sanitized_candidate_dict,
)
from churnwise.acquisition.store import CacheHit, DatasetStore, safe_component
from churnwise import __version__
from churnwise.config import Settings
from churnwise.contracts import (
    DatasetCandidate,
    DiscoveryReport,
    LicenseStatus,
    SelectedCandidate,
)
from churnwise.discovery.registry import get_source

logger = logging.getLogger("churnwise.acquisition")

DownloaderFactory = Callable[[frozenset[str]], StreamingDownloader]


@dataclass
class RankedCandidate:
    candidate: DatasetCandidate
    rank: int
    score: float | None
    origin: str


def _default_downloader_factory(settings: Settings) -> DownloaderFactory:
    def build(hosts: frozenset[str]) -> StreamingDownloader:
        return StreamingDownloader(
            hosts,
            timeout_s=settings.download_timeout_s,
            max_retries=settings.max_retries,
            max_bytes=settings.max_download_bytes,
            transport=RequestsDownloadTransport(settings.user_agent),
        )

    return build


def _normalise(source: Any, origin: str | None = None) -> list[RankedCandidate]:
    if isinstance(source, DiscoveryReport):
        ranked = []
        for index, assessment in enumerate(source.ranked(), 1):
            if assessment.eligible:
                ranked.append(
                    RankedCandidate(
                        candidate=assessment.candidate,
                        rank=assessment.rank or index,
                        score=assessment.score,
                        origin="phase2_ranking",
                    )
                )
        return ranked
    if isinstance(source, SelectedCandidate):
        return [
            RankedCandidate(
                candidate=DatasetCandidate.from_dict(source.snapshot),
                rank=1,
                score=None,
                origin="phase2_selected",
            )
        ]
    if isinstance(source, DatasetCandidate):
        return [RankedCandidate(source, 1, None, origin or "caller_supplied")]
    if isinstance(source, Sequence):
        unique: list[RankedCandidate] = []
        seen: set[str] = set()
        for index, candidate in enumerate(source, 1):
            if not isinstance(candidate, DatasetCandidate):
                raise TypeError("candidate sequences must contain DatasetCandidate objects")
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)
            unique.append(RankedCandidate(candidate, index, None, origin or "caller_supplied"))
        return unique
    raise TypeError("acquisition expects a candidate, a candidate sequence, SelectedCandidate or DiscoveryReport")


class AcquisitionCoordinator:
    def __init__(
        self,
        settings: Settings,
        *,
        downloader_factory: DownloaderFactory | None = None,
        now: Callable[[], str] = utc_now_iso,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.store = DatasetStore(settings.data_dir)
        self._downloader_factory = downloader_factory or _default_downloader_factory(settings)
        self._now = now
        self._sleep = sleeper

    def acquire(
        self,
        source: Any,
        *,
        offline: bool | None = None,
        origin: str | None = None,
    ) -> AcquisitionResult:
        generated_at = self._now()
        run_id = f"{generated_at.replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:8]}"
        effective_offline = self.settings.offline if offline is None else offline
        candidates = _normalise(source, origin)
        attempts: list[AcquisitionAttempt] = []
        warnings: list[str] = []

        if not candidates:
            error = failure(
                CODE_NO_CANDIDATE,
                "no eligible candidate with a downloadable resource was supplied",
                category=CATEGORY_CANDIDATE,
            )
            return self._failure_result(run_id, generated_at, attempts, warnings, error)

        try:
            self.store.ensure()
        except OSError as exc:
            error = failure(
                CODE_STORAGE,
                f"could not prepare the managed data directory: {exc}",
                category=CATEGORY_STORAGE,
            )
            return self._failure_result(run_id, generated_at, attempts, warnings, error)
        previous_failure: str | None = None
        for ranked in candidates:
            started = time.perf_counter()
            candidate = ranked.candidate

            def elapsed_ms() -> int:
                return int((time.perf_counter() - started) * 1000)

            # Policy checks apply to cache hits too. A cached artifact must not
            # outlive a licence/access change or a now-invalid candidate.
            try:
                self._precheck(candidate)
            except AcquisitionError as exc:
                attempts.append(self._attempt(ranked, AttemptStatus.FAILED, elapsed_ms(), exc))
                previous_failure = candidate.candidate_id
                continue

            try:
                lookup = self.store.lookup(candidate)
            except AcquisitionError as exc:
                attempts.append(self._attempt(ranked, AttemptStatus.FAILED, elapsed_ms(), exc))
                warnings.append(f"{candidate.candidate_id}: {exc.message}")
                previous_failure = candidate.candidate_id
                continue
            except OSError as exc:
                cache_error = failure(
                    CODE_STORAGE,
                    f"local storage failure while checking the cache for {candidate.candidate_id}: {exc}",
                    category=CATEGORY_STORAGE,
                )
                attempts.append(
                    self._attempt(ranked, AttemptStatus.FAILED, elapsed_ms(), cache_error)
                )
                warnings.append(f"{candidate.candidate_id}: {cache_error.message}")
                previous_failure = candidate.candidate_id
                continue
            if lookup.warning:
                warnings.append(lookup.warning)
            if lookup.hit is not None:
                try:
                    result = self._from_cache(
                        lookup.hit, ranked, run_id, generated_at, effective_offline, lookup.warning
                    )
                except AcquisitionError as exc:
                    attempts.append(self._attempt(ranked, AttemptStatus.FAILED, elapsed_ms(), exc))
                    warnings.append(f"{candidate.candidate_id}: {exc.message}")
                    previous_failure = candidate.candidate_id
                    continue
                except OSError as exc:
                    storage_error = failure(
                        CODE_STORAGE,
                        f"local storage failure while reusing cached {candidate.candidate_id}: {exc}",
                        category=CATEGORY_STORAGE,
                    )
                    attempts.append(
                        self._attempt(ranked, AttemptStatus.FAILED, elapsed_ms(), storage_error)
                    )
                    warnings.append(f"{candidate.candidate_id}: {storage_error.message}")
                    previous_failure = candidate.candidate_id
                    continue
                attempts.append(
                    self._attempt(
                        ranked,
                        AttemptStatus.CACHED,
                        elapsed_ms(),
                        None,
                        cached=True,
                    )
                )
                result.attempts = attempts
                result.warnings = _dedupe(warnings + result.warnings)
                return result

            if effective_offline:
                exc = failure(
                    CODE_OFFLINE,
                    "offline mode is enabled and no verified cached artifact exists",
                    category=CATEGORY_OFFLINE,
                )
                attempts.append(self._attempt(ranked, AttemptStatus.FAILED, elapsed_ms(), exc))
                previous_failure = candidate.candidate_id
                continue

            try:
                result = self._acquire_one(
                    ranked,
                    run_id,
                    generated_at,
                    elapsed_ms,
                    fallback_of=previous_failure,
                    attempt_number=len(attempts) + 1,
                )
            except AcquisitionError as exc:
                attempts.append(self._attempt(ranked, AttemptStatus.FAILED, elapsed_ms(), exc))
                logger.warning(
                    "acquisition candidate failed",
                    extra={
                        "candidate_id": candidate.candidate_id,
                        "source_id": candidate.source_id,
                        "status": "failed",
                        "duration_ms": elapsed_ms(),
                    },
                )
                previous_failure = candidate.candidate_id
                continue
            except OSError as exc:
                storage_error = failure(
                    CODE_STORAGE,
                    f"local storage failure while acquiring {candidate.candidate_id}: {exc}",
                    category=CATEGORY_STORAGE,
                )
                attempts.append(
                    self._attempt(ranked, AttemptStatus.FAILED, elapsed_ms(), storage_error)
                )
                previous_failure = candidate.candidate_id
                continue
            attempts.append(self._attempt(ranked, AttemptStatus.SUCCESS, elapsed_ms(), None))
            result.attempts = attempts
            result.warnings = _dedupe(warnings + result.warnings)
            return result

        codes = {attempt.error_code for attempt in attempts if attempt.error_code}
        code = next(iter(codes)) if len(codes) == 1 else CODE_ALL_FAILED
        causes = [
            f"{attempt.candidate_id}: {attempt.error_code} {attempt.error_message}"
            for attempt in attempts
            if attempt.error_code
        ]
        error = failure(
            code,
            "no eligible candidate could be acquired",
            category=_category_for(code),
            causes=causes,
        )
        return self._failure_result(run_id, generated_at, attempts, warnings, error)

    # -- one candidate ------------------------------------------------------
    def _acquire_one(
        self,
        ranked: RankedCandidate,
        run_id: str,
        generated_at: str,
        elapsed_ms: Callable[[], int],
        *,
        fallback_of: str | None,
        attempt_number: int,
    ) -> AcquisitionResult:
        candidate = ranked.candidate
        assert candidate.resource_url is not None
        spec = get_source(candidate.source_id)
        downloader = self._downloader_factory(frozenset(spec.hosts))
        temporary = self.store.run_tmp(run_id) / f"{safe_component(candidate.candidate_id)}.part"

        receipt = downloader.download(
            candidate.resource_url,
            temporary,
            expected_size=candidate.size_bytes,
        )
        extract_root: Path | None = None
        try:
            checksum = publisher_checksum(candidate)
            checksum_status = self._verify_checksum(temporary, checksum, receipt)
            inspection = inspect_download(
                temporary,
                declared_format=candidate.format,
                content_type=receipt.content_type,
            )
            if inspection.is_archive:
                extract_root = self.store.run_tmp(
                    run_id
                ) / f"{safe_component(candidate.candidate_id)}-extract"
                extracted = extract_archive(
                    temporary,
                    extract_root,
                    max_members=self.settings.max_archive_members,
                    max_extracted_bytes=self.settings.max_extracted_bytes,
                )
                assert extracted.extracted_path is not None
                member_format = Path(extracted.selected_member or "").suffix.lower().lstrip(".")
                inspect_download(
                    extracted.extracted_path,
                    declared_format=member_format,
                    content_type=None,
                )
            else:
                extracted = None

            cache_path = self.store.promote(temporary, receipt.sha256, receipt.bytes)
            raw_path = self.store.publish_raw(
                cache_path, self.store.raw_path(candidate, receipt.filename)
            )
            extracted_path: Path | None = None
            extracted_relative: str | None = None
            extracted_sha256: str | None = None
            extracted_files: list[str] = []
            if extracted is not None:
                assert extracted.extracted_path is not None and extract_root is not None
                member_relative = extracted.selected_member or extracted.extracted_path.name
                final_dir = self.store.publish_extracted(
                    extract_root,
                    candidate.candidate_id,
                    receipt.sha256,
                    replace=True,
                )
                extracted_path = final_dir.joinpath(*Path(member_relative).parts)
                extracted_relative = self.store.relative(extracted_path)
                extracted_sha256 = sha256_file(extracted_path)
                extracted_files = [member_relative]
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            if extract_root is not None:
                _remove_tree_if_present(extract_root)
            raise

        manifest = self._manifest(
            candidate,
            ranked,
            run_id,
            receipt,
            cache_path,
            raw_path,
            extracted_relative,
            extracted_sha256,
            extracted_files,
            extracted,
            checksum_status,
            checksum,
            fallback_of,
            elapsed_ms(),
            generated_at,
            attempt_number,
        )
        manifest_path = self.store.save_manifest(manifest)
        artifact = self._artifact(manifest)
        return AcquisitionResult(
            run_id=run_id,
            generated_at=generated_at,
            status=AcquisitionStatus.SUCCESS.value,
            attempts=[],
            data_dir=str(self.store.data_dir),
            candidate=sanitized_candidate_dict(candidate),
            artifact=artifact,
            manifest=manifest,
            manifest_path=str(manifest_path),
            cached=False,
            warnings=list(manifest.acquisition.warnings),
            seed=self.settings.seed,
            code_version=__version__,
            dataset_sha256=manifest.file.sha256,
        )

    def _precheck(self, candidate: DatasetCandidate) -> None:
        try:
            issues = candidate.validate()
        except (AttributeError, TypeError, ValueError) as exc:
            raise failure(
                CODE_NO_CANDIDATE,
                f"candidate metadata could not be validated: {exc}",
                category=CATEGORY_CANDIDATE,
            ) from exc
        errors = [issue for issue in issues if issue.severity.value == "error"]
        if errors:
            raise failure(
                CODE_NO_CANDIDATE,
                f"candidate metadata is invalid: {errors[0].message}",
                category=CATEGORY_CANDIDATE,
            )
        if not isinstance(candidate.evidence, dict):
            raise failure(
                CODE_NO_CANDIDATE,
                "candidate evidence must be a JSON object",
                category=CATEGORY_CANDIDATE,
            )
        if not isinstance(candidate.source_dataset_id, str) or not candidate.source_dataset_id.strip():
            raise failure(
                CODE_NO_CANDIDATE,
                "candidate source dataset id is empty",
                category=CATEGORY_CANDIDATE,
            )
        if candidate.access != "open":
            raise failure(
                "CW-ACQ-006",
                f"candidate access is {candidate.access!r}, not open",
                category=CATEGORY_PERMISSION,
            )
        if candidate.license_status is LicenseStatus.RESTRICTED:
            raise failure(
                "CW-ACQ-006",
                "candidate has a restricted licence",
                category=CATEGORY_PERMISSION,
            )
        if (
            candidate.license_status is not LicenseStatus.VERIFIED
            and not self.settings.allow_unverified_license
        ):
            raise failure(
                "CW-ACQ-006",
                "candidate licence is not verified; discovery should have blocked it",
                category=CATEGORY_PERMISSION,
            )
        if candidate.format not in ACQUISITION_FORMATS:
            raise failure(
                CODE_UNEXPECTED_CONTENT,
                f"format {candidate.format!r} is not supported for acquisition",
                category=CATEGORY_FORMAT,
            )
        if candidate.resource_url is None:
            raise failure(
                CODE_NO_CANDIDATE,
                "candidate has no direct resource URL",
                category=CATEGORY_CANDIDATE,
            )
        try:
            get_source(candidate.source_id)
        except KeyError as exc:
            raise failure(
                "CW-ACQ-006",
                f"source {candidate.source_id!r} is not in the reviewed registry",
                category=CATEGORY_PERMISSION,
            ) from exc

    # -- cache --------------------------------------------------------------
    def _from_cache(
        self,
        hit: CacheHit,
        ranked: RankedCandidate,
        run_id: str,
        generated_at: str,
        offline: bool,
        warning: str | None,
    ) -> AcquisitionResult:
        manifest = hit.manifest
        if manifest.file.extracted_path:
            extracted_file = self.store.resolve(manifest.file.extracted_path)
            if not hit.extracted_valid or not extracted_file.is_file():
                extract_root = self.store.run_tmp(
                    run_id
                ) / f"{safe_component(ranked.candidate.candidate_id)}-reextract"
                extracted = extract_archive(
                    hit.cache_path,
                    extract_root,
                    max_members=self.settings.max_archive_members,
                    max_extracted_bytes=self.settings.max_extracted_bytes,
                )
                assert extracted.extracted_path is not None
                member_relative = extracted.selected_member or extracted.extracted_path.name
                final_dir = self.store.publish_extracted(
                    extract_root,
                    ranked.candidate.candidate_id,
                    manifest.file.sha256,
                    replace=True,
                )
                extracted_file = final_dir.joinpath(*Path(member_relative).parts)
                expected = manifest.file.extracted_sha256
                if expected is not None:
                    try:
                        actual = sha256_file(extracted_file)
                    except OSError as exc:
                        raise failure(
                            CODE_CACHE_INVALID,
                            f"cached extracted member could not be re-read: {exc}",
                            category=CATEGORY_CACHE,
                        ) from exc
                    if actual != expected:
                        raise failure(
                            CODE_CACHE_INVALID,
                            "re-extracted archive member does not match the recorded SHA-256",
                            category=CATEGORY_CACHE,
                        )
        artifact = self._artifact(manifest)
        warnings = [warning] if warning else []
        warnings.extend(manifest.acquisition.warnings)
        if offline:
            warnings.append("offline mode: verified cached artifact reused without network access")
        return AcquisitionResult(
            run_id=run_id,
            generated_at=generated_at,
            status=AcquisitionStatus.SUCCESS.value,
            attempts=[],
            data_dir=str(self.store.data_dir),
            candidate=sanitized_candidate_dict(ranked.candidate),
            artifact=artifact,
            manifest=manifest,
            manifest_path=str(self.store.manifest_path(manifest.dataset_id)),
            cached=True,
            warnings=warnings,
            seed=self.settings.seed,
            code_version=__version__,
            dataset_sha256=manifest.file.sha256,
        )

    # -- manifests ----------------------------------------------------------
    def _manifest(
        self,
        candidate: DatasetCandidate,
        ranked: RankedCandidate,
        run_id: str,
        receipt: DownloadReceipt,
        cache_path: Path,
        raw_path: Path,
        extracted_relative: str | None,
        extracted_sha256: str | None,
        extracted_files: list[str],
        extracted: ArchiveInfo | None,
        checksum_status: str,
        checksum,
        fallback_of: str | None,
        duration_ms: int,
        generated_at: str,
        attempt_number: int,
    ) -> DatasetManifest:
        download_url = sanitize_url(receipt.final_url) or sanitize_url(candidate.resource_url)
        landing_url = sanitize_url(candidate.landing_url)
        if download_url is None:
            raise failure(
                "CW-ACQ-006",
                "download URL could not be retained as a safe provenance reference",
                category=CATEGORY_PERMISSION,
            )
        if landing_url is None:
            raise failure(
                "CW-ACQ-006",
                "landing URL could not be retained as a safe provenance reference",
                category=CATEGORY_PERMISSION,
            )
        warnings = []
        if checksum_status == "unavailable":
            warnings.append("publisher checksum unavailable; local SHA-256 recorded")
        if candidate.size_bytes is None:
            warnings.append("source did not report an expected size; actual bytes recorded")
        if extracted is not None:
            warnings.append("archive members were inspected and only one tabular file was extracted")
        return DatasetManifest(
            dataset_id=candidate.candidate_id,
            title=candidate.title,
            run_id=run_id,
            dataset_sha256=receipt.sha256,
            seed=self.settings.seed,
            code_version=__version__,
            source=ManifestSource(
                provider=candidate.source_name or candidate.source_id,
                source_id=candidate.source_id,
                dataset_id=candidate.source_dataset_id,
                landing_url=landing_url,
                download_url=download_url,
                license_name=candidate.license_name,
                license_url=sanitize_url(candidate.license_url),
                license_status=candidate.license_status.value,
                publisher=candidate.publisher,
                reported_version=(
                    str(candidate.evidence.get("version"))
                    if candidate.evidence.get("version")
                    else (candidate.updated_at or candidate.published_at)
                ),
            ),
            retrieved_at=generated_at,
            file=ManifestFile(
                filename=raw_path.name,
                bytes=receipt.bytes,
                sha256=receipt.sha256,
                content_type=receipt.content_type,
                format=candidate.format,
                cache_path=self.store.relative(cache_path),
                raw_path=self.store.relative(raw_path),
                extracted_path=extracted_relative,
                extracted_sha256=extracted_sha256,
                extracted_files=extracted_files,
            ),
            integrity=ManifestIntegrity(
                verified=True,
                method="sha256_recomputed",
                expected_size=candidate.size_bytes,
                publisher_checksum=checksum,
                checksum_status=checksum_status,
                archive_members=extracted.members if extracted else [],
            ),
            acquisition=ManifestAcquisition(
                method="https_download",
                attempt=attempt_number,
                duration_ms=duration_ms,
                status="success",
                fallback_of=fallback_of,
                candidate_rank=ranked.rank,
                warnings=warnings,
            ),
            discovery={
                "candidate": sanitized_candidate_dict(candidate),
                "candidate_fingerprint": candidate_fingerprint(candidate),
                "rank": ranked.rank,
                "score": ranked.score,
                "origin": ranked.origin,
                "selection": (
                    "highest-ranked eligible Phase 2 candidate"
                    if ranked.origin.startswith("phase2")
                    else "caller-supplied candidate order"
                ),
                "limitations": list(candidate.limitations),
            },
            notes=list(candidate.limitations),
        )

    def _artifact(self, manifest: DatasetManifest) -> AcquisitionArtifact:
        return AcquisitionArtifact(
            dataset_id=manifest.dataset_id,
            filename=manifest.file.filename,
            format=manifest.file.format,
            bytes=manifest.file.bytes,
            sha256=manifest.file.sha256,
            content_type=manifest.file.content_type,
            cache_path=manifest.file.cache_path,
            raw_path=manifest.file.raw_path,
            extracted_path=manifest.file.extracted_path,
            extracted_files=list(manifest.file.extracted_files),
        )

    # -- integrity ----------------------------------------------------------
    def _verify_checksum(self, path: Path, checksum, receipt: DownloadReceipt) -> str:
        if checksum is None:
            return "unavailable"
        if checksum.algorithm == "sha256":
            actual = receipt.sha256
        elif checksum.algorithm == "md5":
            digest = hashlib.md5()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
        else:
            return "unavailable"
        if actual.lower() != checksum.value.lower():
            raise failure(
                CODE_INTEGRITY,
                f"publisher {checksum.algorithm} checksum does not match downloaded bytes",
                category=CATEGORY_INTEGRITY,
            )
        return "verified"

    # -- attempts -----------------------------------------------------------
    def _attempt(
        self,
        ranked: RankedCandidate,
        status: AttemptStatus,
        duration_ms: int,
        error: AcquisitionError | None,
        *,
        cached: bool = False,
    ) -> AcquisitionAttempt:
        return AcquisitionAttempt(
            candidate_id=ranked.candidate.candidate_id,
            rank=ranked.rank,
            source_id=ranked.candidate.source_id,
            status=status.value,
            format=ranked.candidate.format,
            url=sanitize_url(ranked.candidate.resource_url),
            duration_ms=duration_ms,
            cached=cached,
            error_code=error.code if error else None,
            error_message=error.message if error else None,
            error_category=error.category if error else None,
            retryable=error.retryable if error else False,
            remediation=error.remediation if error else None,
        )

    def _failure_result(
        self,
        run_id: str,
        generated_at: str,
        attempts: list[AcquisitionAttempt],
        warnings: list[str],
        error: AcquisitionError,
    ) -> AcquisitionResult:
        return AcquisitionResult(
            run_id=run_id,
            generated_at=generated_at,
            status=AcquisitionStatus.FAILURE.value,
            attempts=attempts,
            data_dir=str(self.store.data_dir),
            warnings=_dedupe(warnings),
            seed=self.settings.seed,
            code_version=__version__,
            dataset_sha256=None,
            failure=AcquisitionFailure(
                code=error.code,
                message=error.message,
                category=error.category,
                retryable=error.retryable,
                remediation=error.remediation,
                causes=list(error.causes),
            ),
        )


def _category_for(code: str) -> str:
    mapping = {
        CODE_INTEGRITY: CATEGORY_INTEGRITY,
        CODE_UNEXPECTED_CONTENT: CATEGORY_FORMAT,
        CODE_OFFLINE: CATEGORY_OFFLINE,
        CODE_STORAGE: CATEGORY_STORAGE,
        CODE_CACHE_INVALID: CATEGORY_CACHE,
        CODE_INTERRUPTED: CATEGORY_INTERRUPTED,
        CODE_ALL_FAILED: CATEGORY_CANDIDATE,
        CODE_SIZE_LIMIT: CATEGORY_STORAGE,
        CODE_CORRUPT_ARCHIVE: CATEGORY_ARCHIVE,
        CODE_UNSAFE_ARCHIVE: CATEGORY_ARCHIVE,
        CODE_ACCESS: CATEGORY_PERMISSION,
        "CW-ACQ-004": CATEGORY_NETWORK,
    }
    return mapping.get(code, CATEGORY_CANDIDATE)


def _remove_tree_if_present(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        except OSError:
            pass
    try:
        path.rmdir()
    except OSError:
        pass


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
