from __future__ import annotations

import hashlib
import json

from churnwise.acquisition.store import DatasetStore
from tests.acquisition_helpers import csv_bytes, make_acquisition_candidate, make_manifest


def populate(store: DatasetStore, candidate=None, content: bytes | None = None):
    candidate = candidate or make_acquisition_candidate()
    content = content if content is not None else csv_bytes()
    store.ensure()
    temporary = store.run_tmp("test") / "download.part"
    temporary.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    cache_path = store.promote(temporary, sha, len(content))
    raw_path = store.publish_raw(cache_path, store.raw_path(candidate, "artifact.csv"))
    manifest = make_manifest(
        candidate,
        sha256=sha,
        bytes_=len(content),
        cache_path=store.relative(cache_path),
        raw_path=store.relative(raw_path),
    )
    manifest_path = store.save_manifest(manifest)
    return candidate, content, sha, cache_path, raw_path, manifest, manifest_path


def test_cache_hit_reuses_verified_blob(tmp_path):
    store = DatasetStore(tmp_path / "data")
    candidate, content, sha, cache_path, raw_path, manifest, _ = populate(store)

    lookup = store.lookup(candidate)

    assert lookup.hit is not None
    assert lookup.warning is None
    assert lookup.hit.manifest.file.sha256 == sha
    assert lookup.hit.cache_path == cache_path
    assert lookup.hit.raw_path.read_bytes() == content


def test_missing_raw_copy_is_restored_from_cache(tmp_path):
    store = DatasetStore(tmp_path / "data")
    candidate, content, _, _, raw_path, _, _ = populate(store)
    raw_path.unlink()

    lookup = store.lookup(candidate)

    assert lookup.hit is not None
    assert raw_path.read_bytes() == content


def test_corrupt_raw_copy_is_restored_from_the_verified_blob(tmp_path):
    store = DatasetStore(tmp_path / "data")
    candidate, content, _, _, raw_path, _, _ = populate(store)
    raw_path.write_bytes(b"not the dataset")

    lookup = store.lookup(candidate)

    assert lookup.hit is not None
    assert raw_path.read_bytes() == content


def test_unwritable_raw_restore_returns_a_warning_instead_of_raising(tmp_path):
    store = DatasetStore(tmp_path / "data")
    candidate, _, _, _, raw_path, _, _ = populate(store)
    raw_path.unlink()
    raw_path.mkdir()

    lookup = store.lookup(candidate)

    assert lookup.hit is None
    assert "raw copy" in (lookup.warning or "")


def test_corrupt_cache_blob_is_quarantined_and_not_reused(tmp_path):
    store = DatasetStore(tmp_path / "data")
    candidate, _, _, cache_path, _, _, manifest_path = populate(store)
    cache_path.write_bytes(b"tampered")

    lookup = store.lookup(candidate)

    assert lookup.hit is None
    assert "SHA-256" in (lookup.warning or "")
    assert not manifest_path.exists()
    assert list((store.quarantine_dir).glob("*.json"))


def test_stale_candidate_metadata_invalidates_cache(tmp_path):
    store = DatasetStore(tmp_path / "data")
    candidate, _, _, _, _, _, manifest_path = populate(store)
    changed = make_acquisition_candidate(resource_url="https://www.openml.org/data/v1/download/999")

    lookup = store.lookup(changed)

    assert lookup.hit is None
    assert "stale" in (lookup.warning or "")
    assert not manifest_path.exists()


def test_changed_publisher_checksum_invalidates_cache(tmp_path):
    store = DatasetStore(tmp_path / "data")
    candidate, _, _, _, _, _, manifest_path = populate(store)
    changed = make_acquisition_candidate(evidence={"sha256": "a" * 64})

    lookup = store.lookup(changed)

    assert lookup.hit is None
    assert "stale" in (lookup.warning or "")
    assert not manifest_path.exists()


def test_malformed_manifest_shape_is_quarantined_not_crashed(tmp_path):
    store = DatasetStore(tmp_path / "data")
    candidate, content, sha, _, _, _, _ = populate(store)
    store.manifest_path(candidate.candidate_id).write_text(
        json.dumps({"dataset_id": candidate.candidate_id, "file": []}),
        encoding="utf-8",
    )

    lookup = store.lookup(candidate)

    assert lookup.hit is None
    assert "invalid" in (lookup.warning or "")
    assert list(store.quarantine_dir.glob("*.json"))


def test_invalidation_removes_manifest_and_raw_but_retains_blob(tmp_path):
    store = DatasetStore(tmp_path / "data")
    candidate, _, _, cache_path, raw_path, _, manifest_path = populate(store)

    report = store.invalidate(candidate.candidate_id)

    assert not manifest_path.exists()
    assert not raw_path.exists()
    assert cache_path.exists()
    assert report["cache_blob_retained"]


def test_invalidation_removes_extracted_file(tmp_path):
    store = DatasetStore(tmp_path / "data")
    candidate, content, sha, cache_path, raw_path, manifest, _ = populate(store)
    extracted = store.extracted_path(candidate.candidate_id, sha) / "churn.csv"
    extracted.parent.mkdir(parents=True)
    extracted.write_bytes(content)
    manifest.file.extracted_path = store.relative(extracted)
    manifest.file.extracted_sha256 = hashlib.sha256(content).hexdigest()
    store.save_manifest(manifest)

    store.invalidate(candidate.candidate_id)

    assert not extracted.exists()


def test_promotion_rejects_reusing_a_blob_under_the_wrong_hash(tmp_path):
    store = DatasetStore(tmp_path / "data")
    store.ensure()
    content = csv_bytes()
    sha = hashlib.sha256(content).hexdigest()
    target = store.cache_path(sha)
    target.write_bytes(b"corrupt")
    temporary = store.run_tmp("test") / "download.part"
    temporary.write_bytes(content)

    promoted = store.promote(temporary, sha, len(content))

    assert promoted.read_bytes() == content
    assert hashlib.sha256(promoted.read_bytes()).hexdigest() == sha
