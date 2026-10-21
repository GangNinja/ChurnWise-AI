"""Managed dataset storage and content-addressed cache.

Layout (relative to the configured data directory)::

    cache/sha256/<sha256>                  verified bytes, addressed by hash
    raw/<source>/<dataset>/<version>/<file> human-browsable copy
    extracted/<dataset>/<sha-prefix>/...   extracted tabular file
    manifests/<dataset>.json               provenance record
    tmp/<run-id>/...                       confined temporary work
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from churnwise.acquisition.contracts import DatasetManifest
from churnwise.acquisition.errors import (
    CATEGORY_CACHE,
    CATEGORY_STORAGE,
    CODE_CACHE_INVALID,
    CODE_STORAGE,
    AcquisitionError,
    failure,
)
from churnwise.acquisition.integrity import sha256_file
from churnwise.acquisition.provenance import (
    candidate_fingerprint,
    read_manifest,
    write_manifest,
)
from churnwise.contracts import DatasetCandidate

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class CacheHit:
    manifest: DatasetManifest
    cache_path: Path
    raw_path: Path
    extracted_path: Path | None
    extracted_valid: bool = True


@dataclass
class CacheLookup:
    hit: CacheHit | None
    warning: str | None = None


def safe_component(value: str, *, fallback: str = "dataset") -> str:
    cleaned = _SAFE.sub("_", value.strip()).strip("._-")
    return cleaned[:120] or fallback


class DatasetStore:
    def __init__(self, data_dir: Path | str) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache" / "sha256"
        self.raw_dir = self.data_dir / "raw"
        self.extracted_dir = self.data_dir / "extracted"
        self.manifest_dir = self.data_dir / "manifests"
        self.quarantine_dir = self.manifest_dir / "quarantine"
        self.tmp_dir = self.data_dir / "tmp"

    def ensure(self) -> None:
        for directory in (
            self.cache_dir,
            self.raw_dir,
            self.extracted_dir,
            self.manifest_dir,
            self.quarantine_dir,
            self.tmp_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.data_dir.resolve()).as_posix()

    def resolve(self, relative: str) -> Path:
        return self.data_dir / relative

    def cache_path(self, sha256: str) -> Path:
        return self.cache_dir / sha256

    def manifest_path(self, dataset_id: str) -> Path:
        return self.manifest_dir / f"{safe_component(dataset_id)}.json"

    def raw_path(self, candidate: DatasetCandidate, filename: str) -> Path:
        stem = Path(filename).name or f"dataset.{candidate.format}"
        if Path(stem).suffix == "":
            stem = f"{stem}.{candidate.format}"
        stem = safe_component(stem, fallback=f"dataset.{candidate.format}")
        version = _version(candidate)
        return (
            self.raw_dir
            / safe_component(candidate.source_id)
            / safe_component(candidate.source_dataset_id or candidate.candidate_id)
            / version
            / stem
        )

    def extracted_path(self, dataset_id: str, sha256: str) -> Path:
        return self.extracted_dir / safe_component(dataset_id) / sha256[:12]

    def run_tmp(self, run_id: str) -> Path:
        path = self.tmp_dir / safe_component(run_id, fallback="run")
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- cache --------------------------------------------------------------
    def lookup(self, candidate: DatasetCandidate) -> CacheLookup:
        path = self.manifest_path(candidate.candidate_id)
        if not path.is_file():
            return CacheLookup(None)
        try:
            manifest = read_manifest(path)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self._quarantine_manifest(path)
            return CacheLookup(
                None,
                f"cached manifest for {candidate.candidate_id} was invalid and was quarantined: {exc}",
            )
        if manifest.dataset_id != candidate.candidate_id:
            return CacheLookup(None, "cached manifest does not match the candidate")
        expected = candidate_fingerprint(candidate)
        recorded = (manifest.discovery or {}).get("candidate_fingerprint")
        if recorded != expected:
            self._quarantine_manifest(path)
            return CacheLookup(
                None,
                f"cached manifest for {candidate.candidate_id} is stale (candidate metadata changed)",
            )

        cache_path = self.resolve(manifest.file.cache_path)
        if not cache_path.is_file():
            self._quarantine_manifest(path)
            return CacheLookup(
                None, f"cached blob for {candidate.candidate_id} is missing"
            )
        try:
            actual = sha256_file(cache_path)
            cache_size = cache_path.stat().st_size
        except OSError as exc:
            return CacheLookup(None, f"could not read cached blob: {exc}")
        if actual != manifest.file.sha256 or cache_size != manifest.file.bytes:
            self._quarantine_manifest(path)
            return CacheLookup(
                None,
                f"cached blob for {candidate.candidate_id} failed SHA-256 verification",
            )

        raw_path = self.resolve(manifest.file.raw_path)
        try:
            if not raw_path.is_file():
                self.publish_raw(cache_path, raw_path)
            elif sha256_file(raw_path) != manifest.file.sha256:
                self.publish_raw(cache_path, raw_path)
        except AcquisitionError as exc:
            return CacheLookup(None, f"could not restore the raw copy from cache: {exc.message}")
        except OSError as exc:
            return CacheLookup(None, f"could not verify or restore the raw copy: {exc}")

        extracted = (
            self.resolve(manifest.file.extracted_path)
            if manifest.file.extracted_path
            else None
        )
        extracted_valid = True
        if extracted is not None:
            expected = manifest.file.extracted_sha256
            if expected is not None:
                try:
                    extracted_valid = (
                        extracted.is_file() and sha256_file(extracted) == expected
                    )
                except OSError:
                    extracted_valid = False
            else:
                extracted_valid = extracted.is_file()

        return CacheLookup(
            CacheHit(
                manifest=manifest,
                cache_path=cache_path,
                raw_path=raw_path,
                extracted_path=extracted,
                extracted_valid=extracted_valid,
            )
        )

    def promote(self, temporary: Path, sha256: str, expected_size: int) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_path(sha256)
        if target.exists():
            if target.stat().st_size == expected_size and sha256_file(target) == sha256:
                temporary.unlink(missing_ok=True)
                return target
            quarantine = self.cache_dir.parent / "quarantine"
            quarantine.mkdir(parents=True, exist_ok=True)
            os.replace(target, quarantine / f"{sha256}.{uuid.uuid4().hex[:8]}.invalid")
        try:
            os.replace(temporary, target)
        except OSError as exc:
            raise failure(
                CODE_STORAGE,
                f"could not promote verified artifact into the cache: {exc}",
                category=CATEGORY_STORAGE,
            ) from exc
        return target

    def publish_raw(self, cache_path: Path, raw_path: Path) -> Path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = raw_path.with_name(f".{raw_path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            # Copy rather than hardlink: the raw tree is human-browsable and may
            # be edited, and a hardlink would let that corrupt the verified blob.
            shutil.copyfile(cache_path, temporary)
            os.replace(temporary, raw_path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise failure(
                CODE_STORAGE,
                f"could not write the raw copy: {exc}",
                category=CATEGORY_STORAGE,
            ) from exc
        return raw_path

    def publish_extracted(
        self,
        temporary_dir: Path,
        dataset_id: str,
        sha256: str,
        *,
        replace: bool = False,
    ) -> Path:
        final = self.extracted_path(dataset_id, sha256)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            if not replace:
                _remove_tree(temporary_dir)
                return final
            if final.is_file():
                final.unlink()
            else:
                _remove_tree(final)
        try:
            os.replace(temporary_dir, final)
        except OSError as exc:
            _remove_tree(temporary_dir)
            raise failure(
                CODE_STORAGE,
                f"could not promote extracted files: {exc}",
                category=CATEGORY_STORAGE,
            ) from exc
        return final

    # -- manifests ----------------------------------------------------------
    def save_manifest(self, manifest: DatasetManifest) -> Path:
        path = self.manifest_path(manifest.dataset_id)
        try:
            return write_manifest(manifest, path)
        except (OSError, ValueError) as exc:
            raise failure(
                CODE_STORAGE,
                f"could not write provenance manifest: {exc}",
                category=CATEGORY_STORAGE,
            ) from exc

    def invalidate(self, dataset_id: str) -> dict[str, Any]:
        """Remove managed pointers for a dataset; cached bytes are retained."""
        manifest_path = self.manifest_path(dataset_id)
        manifest: DatasetManifest | None = None
        if manifest_path.is_file():
            try:
                manifest = read_manifest(manifest_path)
            except (OSError, ValueError):
                manifest = None
        removed: list[str] = []
        if manifest_path.exists():
            manifest_path.unlink()
            removed.append(self.relative(manifest_path))
        if manifest is not None:
            raw_path = self.resolve(manifest.file.raw_path)
            if raw_path.exists():
                raw_path.unlink()
                removed.append(manifest.file.raw_path)
                _prune_empty_parents(raw_path.parent, self.raw_dir)
            if manifest.file.extracted_path:
                extracted_root = self.resolve(manifest.file.extracted_path)
                if extracted_root.is_file():
                    extracted_root.unlink()
                    removed.append(manifest.file.extracted_path)
                    _prune_empty_parents(extracted_root.parent, self.extracted_dir)
                elif extracted_root.exists():
                    _remove_tree(extracted_root)
                    removed.append(manifest.file.extracted_path)
        return {
            "dataset_id": dataset_id,
            "removed": removed,
            "cache_blob_retained": manifest.file.sha256 if manifest else None,
        }

    def quarantine_manifest(self, dataset_id: str) -> Path | None:
        """Move a stale/corrupt manifest out of the live manifest path.

        Returns the quarantine path when the move succeeded. A missing
        manifest is a no-op so callers can use this defensively.
        """
        path = self.manifest_path(dataset_id)
        if not path.is_file():
            return None
        return self._quarantine_manifest(path)

    def _quarantine_manifest(self, path: Path) -> Path | None:
        destination = self.quarantine_dir / f"{path.stem}.{uuid.uuid4().hex[:8]}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(path, destination)
        except OSError:
            return None
        return destination


def _version(candidate: DatasetCandidate) -> str:
    for value in (candidate.published_at, candidate.updated_at, candidate.retrieved_at):
        if value and len(value) >= 10 and value[4] == "-" and value[7] == "-":
            return safe_component(value[:10])
    return "unversioned"


def _prune_empty_parents(path: Path, stop: Path) -> None:
    current = path
    stop = stop.resolve()
    while current.resolve() != stop and stop in current.resolve().parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def _remove_tree(path: Path) -> None:
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
