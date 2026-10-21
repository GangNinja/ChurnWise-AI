# ChurnWise AI — Phase 3 Report

**Scope:** autonomous acquisition, integrity verification, provenance, managed
storage/cache and fallback. No schema validation, cleaning, feature engineering,
model training, explanations, scoring, simulation or dashboard work was done.

---

## 1. Executive summary

Phase 3 consumes the real Phase 2 hand-off (`SelectedCandidate` and the ranked
`candidates.json`) and turns the selected candidate into a verified local
artifact with a provenance manifest.

A first-time acquisition requires no dataset, URL or credential:

```bash
python -m churnwise.cli discover   # Phase 2
python -m churnwise.cli acquire    # Phase 3
```

Or simply `python -m churnwise.cli acquire`; it runs Phase 2 discovery when no
usable candidates report exists.

**Live artifact acquired and verified**

- Dataset: `openml:46915` — “churn”, MIT Licence
- File: `22125226.arff`, **475,249 bytes**
- SHA-256: **`87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6`**
- Recomputed from the stored raw file: **matches**
- Publisher MD5 from OpenML: **`86dd09210baaa4dfaa502a23703a9436`**, **verified**
- ARFF header and `CustomerChurned` target: **present**
- Cache blob and raw copy: **both present**
- Manifest: `data/manifests/openml_46915.json`, schema validation **passes**
- Offline re-run: **exit 0, under 1 s, cached**, no network
- Offline re-run after the verification fixes: **exit 0, under 1 s, cached**

Automated verification on **2026-10-21**: **188 mocked tests pass, 5 live tests
skipped; the full suite is 193 passed in 50.77 s** with `--run-live`. The live
acquisition test alone is 1 passed in 13.77 s. The earlier 182/187 figures in
this report were the first-run measurements; the counts increased because the
verification rerun added regression tests (see §9).

No model was trained. No dataset content was parsed for schema or quality; that
is Phase 4.

---

## 2. Implemented functionality

| Area | Implementation |
| --- | --- |
| Input compatibility | Phase 2 `SelectedCandidate`, `DatasetCandidate`, ranked sequence and `DiscoveryReport`; `acquire` CLI reads `candidates.json` |
| Streaming downloader | HTTPS allowlist, manual redirects, retries/`Retry-After`, byte caps, streaming SHA-256, expected-size checks |
| Integrity | Local SHA-256, publisher checksum when actually supplied, temporary-file promotion |
| Content validation | HTML/executable rejection, CSV/TSV/ARFF/Parquet/JSON signatures, declared-vs-actual mismatch |
| Archive safety | Path/symlink/encryption/executable rejection, member and expansion caps, one-tabular-member policy |
| Storage | Content-addressed cache, raw copy, extracted tree, confined temp workspace |
| Provenance | Validated JSON manifest, sanitized URLs, candidate snapshot, fallback lineage, attempt log |
| Cache | Warm reuse without network, stale/corrupt quarantine, raw restoration, explicit invalidation |
| Fallback | Rank-order attempts, prior failure recorded, structured total failure |
| CLI | `acquire` with offline, refresh-discovery, JSON, data/report paths and size/timeout overrides |

Full design: [ACQUISITION.md](ACQUISITION.md).

---

## 3. Supported sources and formats

| Source | Access method | Formats | Live acquisition |
| --- | --- | --- | --- |
| OpenML | Direct `resource_url` from documented API | ARFF (selected), CSV/Parquet possible | **Yes — ARFF** |
| UCI ML Repository | Direct `data_url` from documented API | CSV | Not acquired live; UCI is ineligible by default without `--allow-unverified-license` |
| IBM pinned GitHub | Pinned raw URL | CSV | Not acquired live; fails Phase 2 G2/G3 before download |

Formats accepted for acquisition: **csv, tsv, arff, parquet, json, xlsx
(opaque), zip (archive)**. Tests use a real Parquet file written by PyArrow, not
a magic-number stub. A ZIP must contain exactly one supported tabular member;
zero or multiple members are rejected.

---

## 4. Storage and provenance

```text
data/cache/sha256/<sha256>                    verified bytes
data/raw/<source>/<dataset>/<version>/<file>  human-browsable copy
data/extracted/<dataset>/<sha-prefix>/...     extracted tabular file
data/manifests/<dataset>.json                 provenance
data/manifests/quarantine/*.json              stale/invalid manifests
data/cache/quarantine/*.invalid               replaced bad blobs
data/tmp/<run-id>/...                         confined temporary work
reports/acquisition/attempts.json             every attempt
```

The raw copy is a real copy, not a hardlink, so editing the browsable file cannot
corrupt the verified cache blob. Manifest paths are relative and portable; the
result exposes `data_dir` plus absolute `manifest_path` for the current machine.

Manifest fields include run id, dataset SHA-256, seed and code version,
dataset id/title, provider/source/dataset ids, landing and sanitized download
URLs, publisher, licence name/URL/status, reported version, UTC retrieval time,
filename, byte size, SHA-256, content type, format, cache/raw/extracted paths,
extracted SHA-256 (when applicable), expected size, publisher checksum and
status, archive members, acquisition method/status/attempt/duration/fallback,
sanitized candidate snapshot, rank/score/selection rationale, warnings and
limitations.

Unknowns stay `null`; the live OpenML manifest records a verified publisher MD5,
no expected size and no Content-Type because the source supplied none.

### Live manifest excerpt

```json
{
  "run_id": "20261020T184301Z-97689514",
  "dataset_id": "openml:46915",
  "dataset_sha256": "87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6",
  "seed": 42,
  "code_version": "0.3.0",
  "source": {
    "provider": "OpenML",
    "download_url": "https://www.openml.org/data/v1/download/22125226",
    "license": {"name": "MIT License", "url": null, "status": "verified"},
    "reported_version": "4"
  },
  "file": {
    "filename": "22125226.arff",
    "bytes": 475249,
    "sha256": "87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6",
    "format": "arff",
    "raw_path": "raw/openml/46915/2025-04-30/22125226.arff",
    "extracted_path": null,
    "extracted_sha256": null
  },
  "integrity": {
    "verified": true,
    "method": "sha256_recomputed",
    "publisher_checksum": {
      "algorithm": "md5",
      "value": "86dd09210baaa4dfaa502a23703a9436"
    },
    "checksum_status": "verified"
  },
  "acquisition": {
    "method": "https_download",
    "attempt": 1,
    "status": "success",
    "fallback_of": null,
    "duration_ms": 1644
  }
}
```

Sensitive URL parameters, fragments and credentials are stripped recursively
from the manifest and result, including the candidate snapshot.

---

## 5. Security and failure behavior

- HTTPS only, source-registry host allowlist, port 443, no embedded credentials.
- Redirects revalidated at every hop; maximum three.
- 200 MB download cap, 300 MB extracted cap, 200 archive members by default.
- HTML disguised as a dataset, executable magic, empty files and format
  mismatches are rejected.
- ZIP traversal, symlinks, encryption, control-character names, executable
  members and ambiguous multi-file archives are rejected before extraction.
- Extracted members carry their own SHA-256 and are re-extracted if tampered or
  missing; cache hits still run licence/access/format prechecks.
- Downloads write to `data/tmp`, are hashed, then promoted atomically; failures
  leave no partial artifact.
- No downloaded file is executed, imported or evaluated.
- Errors are `CW-ACQ-001…013` with category, retryability and remediation.

Fallback tries the next eligible Phase 2 candidate only after recording the
failure; a later success records `fallback_of`. Total failure lists every cause. Every generated manifest and attempt log
carries `run_id`, `dataset_sha256`, `seed` and `code_version`.

---

## 6. Files

### Created

| Path | Purpose |
| --- | --- |
| `churnwise/acquisition/__init__.py` | Public Phase 3 API |
| `churnwise/acquisition/errors.py` | Stable error taxonomy |
| `churnwise/acquisition/contracts.py` | Result, attempt and manifest contracts |
| `churnwise/acquisition/downloader.py` | Streamed allowlisted HTTPS downloader |
| `churnwise/acquisition/integrity.py` | SHA-256, signatures, safe archive handling |
| `churnwise/acquisition/store.py` | Managed storage, cache lookup/promotion/invalidation |
| `churnwise/acquisition/provenance.py` | URL sanitization, fingerprints, checksums, manifest IO |
| `churnwise/acquisition/coordinator.py` | Acquisition, cache reuse and fallback |
| `churnwise/acquisition/inputs.py` | Load Phase 2 ranked candidates |
| `churnwise/acquisition/reporting.py` | Attempt log and human summary |
| `tests/acquisition_helpers.py` | Fake streams, archive/Parquet/CSV fixtures |
| `tests/test_acquisition_downloader.py` | Download, retry, cap and URL tests |
| `tests/test_acquisition_integrity.py` | Signature, HTML and archive safety tests |
| `tests/test_acquisition_inputs.py` | Rank-order loading and malformed-report tests |
| `tests/test_acquisition_store.py` | Cache, stale/corrupt recovery and invalidation |
| `tests/test_acquisition_coordinator.py` | Success, fallback, offline and contract tests |
| `tests/test_acquisition_provenance.py` | Sanitization, fingerprint and checksum tests |
| `tests/test_acquisition_cli.py` | Exit codes and offline cache CLI tests |
| `tests/test_live_acquisition.py` | Opt-in real OpenML acquisition/cache test |
| `docs/ACQUISITION.md` | Phase 3 design and contracts |
| `docs/PHASE3_REPORT.md` | This report |

### Modified

| Path | Change |
| --- | --- |
| `churnwise/__init__.py` | Phase 3 package version/docstring |
| `churnwise/config.py` | Download/extraction/archive limits, seed validation and Phase 3 user agent |
| `churnwise/net.py` | Phase 3 default user agent |
| `churnwise/cli.py` | `acquire` subcommand and settings overrides |
| `docs/ROADMAP.md` | Phase 3 artifacts and verified exit gate |
| `churnwise/discovery/adapters/openml.py` | Captures OpenML `md5_checksum` and version evidence, and tolerates OpenML's embedded PHP notice in that optional field |
| `tests/test_openml_adapter.py` | Checksum/version evidence and malformed-checksum recovery tests |
| `docs/DISCOVERY.md` | Phase 3 hand-off and cache limitation updated to implemented state |
| `docs/ACCEPTANCE_CRITERIA.md` | Phase 3 artifact/test paths aligned with implementation |

Phase 1 and Phase 2 documents were otherwise preserved. The Phase 1 checker
still passes.

---

## 7. Tests and actual results

```text
$ .venv/Scripts/python -m pytest -q
188 passed, 5 skipped in 1.94s
        (5 skipped = 1 acquisition + 4 discovery live tests)

$ .venv/Scripts/python -m pytest -q --run-live
193 passed in 50.77s (0:00:50)

$ .venv/Scripts/python -m pytest -q --run-live tests/test_live_acquisition.py
1 passed in 13.77s
```

| Suite | Transport | Proves |
| --- | --- | --- |
| `test_acquisition_downloader.py` | fake streams | Direct download, redirects, retries, `Retry-After`, caps, expected size, incomplete/interrupted transfers, unsafe URLs |
| `test_acquisition_integrity.py` | local bytes | SHA-256, CSV/TSV/ARFF/Parquet/JSON signatures, HTML/executable rejection, ZIP traversal/symlink/executable/ambiguity/corruption/limits, extraction |
| `test_acquisition_store.py` | local files | Promotion, cache hit, raw restoration, corrupt blob quarantine, stale fingerprint, invalidation, wrong-hash replacement |
| `test_acquisition_coordinator.py` | fake streams + local files | Success, Phase 2 `SelectedCandidate`/`DiscoveryReport`, checksum match/mismatch, HTML, ZIP, cap, offline warm/cold, fallback, total failure, sanitization, Phase 4 contract |
| `test_acquisition_provenance.py` | none | URL sanitization, timestamp-independent fingerprints, checksum parsing, manifest schema rejection |
| `test_acquisition_cli.py` | fake streams + local cache | Exit 0/2/3, warm offline cache, cold offline `CW-ACQ-005`, JSON output |
| `test_live_acquisition.py` | **real network** | Real discovery → OpenML download → recomputed SHA-256 → manifest → offline cache reuse |

The live suite is the only evidence that the real provider and artifact behave
as claimed. Mocked tests prove parsing, policy and failure handling, not the
provider. The test environment also has PyArrow 24.0.0 solely to write a real
Parquet fixture; acquisition itself uses the standard library and `requests`.
The live artifact is real downloaded ARFF, not a fixture.

---

## 8. Live verification

```text
$ .venv/Scripts/python -m churnwise.cli acquire
Selected: openml:46915 (acquired)
  file: 22125226.arff · 475249 bytes · arff
  sha256: 87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6
  raw: raw/openml/46915/2025-04-30/22125226.arff
  cache: cache/sha256/87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6
  manifest: .../data/manifests/openml_46915.json
  warning: source did not report an expected size; actual bytes recorded

$ .venv/Scripts/python -m churnwise.cli acquire --offline
Selected: openml:46915 (cached)
  ... same sha256 ...
  warning: source did not report an expected size; actual bytes recorded
  warning: offline mode: verified cached artifact reused without network access
```

Exit code 0 for both. The offline run completed in under one second and made no
network request. The artifact was independently rehashed from the raw file and
the cache blob; both match the manifest (`475249` bytes each). `read_manifest()`
validates the stored manifest, including `acquisition.status = "success"` and
`integrity.verified = true`; the publisher MD5 is `verified` and revalidation
returns no problems.

`reports/acquisition/attempts.json` records the successful attempt with the
sanitized URL, format, duration and status. It is overwritten on a later cached
run with status `cached`; both are valid outcomes.

---

## 9. Defects found and fixed during verification

| # | Defect | Impact | Fix |
| --- | --- | --- | --- |
| 1 | Raw copy was hardlinked to the cache blob | Editing the human-browsable file corrupted the authoritative cache | Raw copy is now a real copy; a corrupt raw file is restored from the verified blob |
| 2 | Sensitive URL query parameters leaked through the candidate snapshot | A `token=` value could be stored in the manifest/result despite sanitized `download_url` | Recursive sanitization of the candidate snapshot and all URL-like strings |
| 3 | Nested ZIP extraction promoted the member's parent, not the extraction root | A member in a subdirectory could be published incompletely | Promote the extraction root, then resolve the selected member |
| 4 | Manifest had no explicit acquisition status | Phase 4 could not distinguish a successful acquisition record from an attempt | Added `acquisition.status`, validated as `success` |
| 5 | Header lookup was case-sensitive in the downloader helper | Lowercased provider headers worked, but mixed-case fixture headers silently lost `Content-Length`/`Content-Type`/`Location` | Case-insensitive header lookup with tests |
| 6 | Phase 2 dropped OpenML's published `md5_checksum` | Acquisition reported `unavailable` even though the publisher supplied a checksum | Captured checksum + version in OpenML evidence; live manifest now reports `verified` |
| 7 | Candidate-report parsing trusted array order and uncaught `KeyError`s | A reordered report could acquire the wrong rank; a malformed report could crash the CLI | Loading now validates shape, raises `ValueError` for bad entries, and sorts by `rank` |
| 8 | Manifest parsing did not wrap malformed nested values | A corrupt manifest shape could crash cache lookup instead of being quarantined | Shape/validation errors are wrapped and the manifest is quarantined |
| 9 | Cache fingerprint ignored publisher checksum, version and access posture | A changed source could reuse stale bytes under the same URL/row counts | Fingerprint includes checksum, version and access; a missing fingerprint is stale |
| 10 | Cache hits bypassed licence/access/format prechecks | A policy change could still serve a previously cached artifact | Prechecks run before cache lookup |
| 11 | Extracted archive members were not hash-verified | A tampered extracted CSV could be handed to Phase 4 despite a verified archive | Manifest records `extracted_sha256`; cache hits re-extract on mismatch |
| 12 | Archive control characters and unsupported compression could escape the taxonomy | A hostile ZIP could fail with a raw exception or ambiguous path | Member names reject control characters; unsupported/unsafe ZIP features map to `CW-ACQ-010` |
| 13 | Phase 3 artifacts lacked global run identity fields | Manifests/attempt logs could not satisfy the global artifact contract | Added `run_id`, `dataset_sha256`, `seed` and `code_version` to manifest/result/attempt records |
| 14 | ZIP backslash names were normalized before validation | A backslash member could pass inspection but fail or extract ambiguously | Backslash member names are rejected before normalization |
| 15 | Malformed URL ports/IPv6 escaped as raw `ValueError` | A bad candidate URL could crash instead of returning `CW-ACQ-006` | URL parsing errors map to the access error with a test |
| 16 | Malformed candidate types could raise inside precheck | A corrupt report field could crash the CLI | Precheck converts validation/type errors to `CW-ACQ-001` |
| 17 | Manifest path validation allowed only `[A-Za-z0-9._-]` components, but archive inspection accepted spaces/Unicode | A valid ZIP member such as `churn données 2024.csv` extracted successfully and then failed manifest validation as `CW-ACQ-008`, leaving verified bytes without provenance | Manifest validation now accepts canonical space/Unicode names while still rejecting traversal, absolute paths, colons, control characters, reserved device names and trailing dots/spaces; regression test added |
| 18 | `DatasetStore.lookup()` let an `AcquisitionError` from raw-copy restoration escape | A storage problem during a cache hit crashed the coordinator/CLI instead of returning a structured failure | Raw-copy restoration errors are converted to a `CacheLookup` warning; the coordinator also defends against unexpected cache exceptions |
| 19 | A re-extracted archive member was not rehashed against `extracted_sha256` | A corrupt manifest could cause a different extracted file to be accepted after tamper recovery | Re-extraction now hashes the result and fails with `CW-ACQ-013` on a persistent mismatch; regression test asserts the failure |
| 20 | `ManifestIntegrity.from_dict()` used `bool(...)`, so `"verified": "false"` became `True` | A malformed manifest claiming failure could be treated as a verified artifact | `integrity.verified` must now be a JSON boolean; non-boolean values fail manifest parsing and the manifest is quarantined |
| 21 | `acquisition.attempt` was populated with the Phase 2 candidate rank | A fallback or sparse rank (e.g. rank 5 on the first attempt) recorded misleading attempt history | `attempt` is now the 1-based run attempt ordinal; `candidate_rank` remains the independent Phase 2 rank |
| 22 | Cached results dropped acquisition warnings, and the human failure summary truncated causes to eight | Offline/cache output could hide a missing-size/checksum caveat, and a long fallback chain could look incomplete | Cached results now carry the manifest's warnings; the failure summary prints every cause |
| 23 | Total-failure category mapping omitted size-limit and archive codes | All-candidate failure for `CW-ACQ-007/009/010` could be mislabelled `candidate` | Category mapping now covers size, corrupt-archive and unsafe-archive failures |
| 24 | HTTPS URL validation checked username but not password | A URL with only an embedded password could pass metadata validation | Both Phase 2 candidate validation and Phase 3 manifest validation reject any embedded credentials |

The data-safety/security defects were the fingerprint gap, the cache-policy
bypass, the untrusted extracted member and the archive/URL path handling; the
parser and ordering defects were caught by new regression tests. Defects 17–24
were found by the 2026-10-21 verification rerun against the completion criteria.

---

## 10. Acceptance criteria status

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Consumes actual Phase 2 candidate metadata | **Met** | `SelectedCandidate`, `DiscoveryReport` and `candidates.json` tests; live report acquisition |
| 2 | Suitable data acquired without manual provisioning | **Met** | Live CLI, no user URL/file/credential |
| 3 | Supported formats and archives handled safely | **Met** | CSV/TSV/ARFF/Parquet/JSON/ZIP tests; live ARFF |
| 4 | Integrity prevents incomplete/corrupt artifacts | **Met** | SHA-256, expected-size and publisher-MD5 tests; interrupted/partial tests; live publisher MD5 + rehash |
| 5 | Provenance records source and integrity accurately | **Met** | Validated live manifest; unknown values are null |
| 6 | Verified artifacts cached and reused | **Met** | Warm offline CLI under 1 s, no network; cache tests |
| 7 | Fallback works for suitable alternatives | **Met** | Mocked fallback test with `fallback_of` and attempt history |
| 8 | Network/integrity/storage/security failures are structured | **Met** | `CW-ACQ-001…012` tests and result/remediation fields |
| 9 | Phase 4 has a documented, stable result | **Met** | `AcquisitionResult.to_dict()` test and ACQUISITION §12 |
| 10 | Automated tests cover critical paths | **Met** | 188 mocked tests, 5 live tests (193 with `--run-live`) |
| 11 | Real-source acquisition verified where possible | **Met** | OpenML artifact acquired, hashed and cached; limits reported |
| 12 | Documentation reflects implementation | **Met** | ACQUISITION.md + this report |
| 13 | Existing functionality intact | **Met** | Full 193-test suite; Phase 1 checker 114/114 |
| 14 | All 13 roadmap phases preserved | **Met** | Roadmap/acceptance unchanged in phase count and scope |

Nothing is marked met without the command output or artifact above. The UCI and
GitHub paths were not live-acquired because Phase 2 rejects them by default;
that is a documented eligibility limitation, not a claimed success.

---

### 10.1 Verification rerun (2026-10-21)

The phase was re-audited against T3.1–T3.9 and the completion criteria after the
original report. The real artifact was rehashed from both the cache blob and raw
copy, the manifest was revalidated, `acquire --offline` returned exit 0 in under
a second, a cold-cache offline run returned exit 2 with `CW-ACQ-005` and no
files, and a ten-candidate exhausted-fallback run listed all ten causes. The
archive/member path, cache-lookup, re-extraction, manifest-boolean, attempt and
total-failure issues above were found and fixed; the suite now has 188 mocked
tests (193 with live) and the Phase 1 checker still passes 114/114.

---

## 11. Known limitations and unresolved decisions

**Limitations**

- OpenML supplied a publisher MD5 (verified) but no expected size; the manifest
  records both the MD5 and the local SHA-256. UCI/GitHub metadata exposes no
  dataset checksum, so those paths would record `unavailable`.
- The selected live artifact is ARFF, not CSV/Parquet; those formats are covered
  by deterministic tests, including a real Parquet fixture.
- `xlsx` is stored opaquely; its parser belongs to Phase 4.
- A ZIP must contain exactly one tabular member; multi-file archives are
  rejected instead of guessed.
- Interrupted downloads restart from zero; no HTTP `Range` resume.
- Cache invalidation removes pointers/raw/extracted files but retains the
  content-addressed blob; there is no garbage collector yet.
- Fallback does not re-run discovery; it only uses eligible candidates already
  in the Phase 2 report.
- Live acquisition covers only the eligible OpenML candidate. UCI requires the
  unverified-licence opt-in and GitHub fails Phase 2 gates before download.
- Network time dominates: the 475 KB OpenML artifact took 2–6 s end-to-end
  across observed runs, versus a 60 s design budget.

**Unresolved decisions**

1. Should OpenML’s MD5 be supplemented by a source-published SHA-256 if one
   becomes available (the local SHA-256 is always recorded)?
2. Should multi-file ZIPs select the largest tabular member or require an
   explicit member hint?
3. Should cache blobs be garbage-collected when no manifest references them?
4. Should interrupted transfers support HTTP `Range` resume?
5. Phase 4 must confirm the ARFF target cardinality and class balance from data;
   Phase 3 only verified the file signature and local hash.

---

## 12. Explicit stop

Phase 3 ends here. No schema validation, EDA, cleaning, feature engineering,
model training, evaluation, SHAP, scoring, simulation or dashboard work has
begun. The next phase consumes the verified manifest and artifact; a successful
download is not a claim that the dataset is suitable for modelling.
