# ChurnWise AI — Phase 4 Report

**Scope:** dataset validation and data intelligence on the verified Phase 3
artifact. No cleaning, imputation, scaling, encoding, splitting, feature
selection, model training, explanation or dashboard work was done.

---

## 1. Executive summary

Phase 4 consumes the real Phase 3 hand-off and answers one question: *is this
dataset structurally usable for churn modelling, and what exactly does it
contain?* It proves integrity again, parses the artifact, detects and interprets
the target, maps roles, runs twelve checks, builds an EDA profile with plots,
and persists an auditable JSON + Markdown report behind an explicit gate.

A first-time run needs no user input:

```bash
python -m churnwise.cli discover   # Phase 2
python -m churnwise.cli acquire    # Phase 3
python -m churnwise.cli validate   # Phase 4
```

`validate` discovers and acquires when no candidate report or cache exists, and
falls back to the next eligible candidate when a dataset is rejected.

**Live dataset validated**

- Dataset: `openml:46915` — “churn”, MIT Licence
- Artifact: `data/raw/openml/46915/2025-04-30/22125226.arff`, **475,249 bytes**
- SHA-256: **`87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6`**,
  recomputed from the stored raw copy: **matches**; manifest digest also rechecked
- Shape: **5,000 rows × 20 columns**, 0 missing cells, 0 duplicate rows
- Target: `CustomerChurned`, classes `['No', 'Yes']`, mapping `{'No': 0, 'Yes': 1}`
- Positive class: **707 / 5,000 = 14.14%**, confidence **high** (churn-style name)
- Roles: **15 numeric + 3 categorical = 18 usable features**, no ID column
- Dropped: `state` — 51 categories (limit 30)
- Gate: **`PASS_WITH_WARNINGS`**, `can_preprocess = true`; the only failing check
  is V11 (high-cardinality `state` excluded)
- Reports: `reports/validation/openml_46915.json` and `.md`
- EDA: `reports/eda/openml_46915.json` plus **5 PNG figures**

The warning is not a defect: the completed dataset is usable, and the excluded
column is documented with its reason and evidence.

No data was generated, imputed or altered. `normalized_frame()` returns an
in-memory copy with `1 = churned`; the artifact on disk is byte-identical.

---

## 2. Implemented functionality

| Area | Implementation |
| --- | --- |
| Integrity re-verification | raw copy and validation artifact digests recomputed before parsing; failure quarantines the manifest and rejects |
| Loading | CSV/TSV delimiter sniffing, ARFF attribute/type decoding, Parquet, JSON, XLSX; non-string labels normalised and duplicate labels made unique and recorded |
| Target detection | name/value/context scoring, ambiguity rejection, explicit positive-class interpretation, retention inversion |
| Normalisation | nullable `Int64` mapping to `1 = churned`; missing target rows excluded only in memory |
| Profiling | dtypes, missingness, unique counts, numeric statistics, top values, samples, duplicate content hashing |
| Roles | exactly one role per column; dropped columns always carry a reason |
| Flags | identifier, constant, near-constant, high missing, high cardinality, text, date, duplicate, leakage |
| Capabilities | 12 derived-feature capabilities with the satisfying source column |
| Checks | V1–V12 with stable ids, severity, detail, issues and evidence, including non-finite numeric detection |
| Gate | fatal/warning rollup, blocking list, reasons; explicit `can_preprocess` |
| Reports | atomic JSON + Markdown, archived profile under `data/manifests/` |
| EDA | profile JSON and five figures: overview, numeric distributions, target-conditional, correlation, categorical |
| Fallback | per-candidate acquisition + validation attempts with codes, durations and report paths |
| CLI | `validate` with offline mode, bounded fallback, codec-safe JSON output and documented exit codes |

---

## 3. Check results on the real dataset

| ID | Check | Result | Severity | Detail |
| --- | --- | --- | --- | --- |
| V1 | file integrity and readability | PASS | fatal | artifact exists, is readable and matches its recorded digest |
| V2 | structural integrity | PASS | fatal | 5,000 rows × 20 columns |
| V3 | target identified | PASS | fatal | selected target `CustomerChurned` |
| V4 | target binary | PASS | fatal | two classes: `['No', 'Yes']` |
| V5 | target interpretation | PASS | fatal | `1 = churn` maps from `Yes`; confidence high |
| V6 | target balance | PASS | warning | positive rate 14.140% |
| V7 | duplicate columns | PASS | warning | no duplicate headers or columns |
| V8 | duplicate rows | PASS | warning | no duplicate rows |
| V9 | missingness | PASS | warning | no missing or non-finite values |
| V10 | low-information features | PASS | warning | no constant or near-constant columns |
| V11 | identifier and cardinality | **FAIL** | warning | `state` has 51 levels (limit 30); 15 numeric + 3 categorical remain |
| V12 | target leakage | PASS | warning | no suspected target leakage |

The failing warning is actionable and bounded: `state` is excluded from features
and recorded under `roles.dropped`, while the mapping and source column remain
auditable.

V11 also enforces the EDA stage gate: a dataset with fewer than three numeric
predictors and no categorical feature is rejected as fatal
(`V11.insufficient_features`), while one categorical feature is sufficient. The
real dataset clears it with 15 numeric features, and both sides of the rule are
covered in `tests/unit/test_validation_checks.py`.

---

## 4. EDA and capability intelligence

The profile records per-column statistics and canonical roles. The figures are
generated from the real frame, not from a fixture:

- `reports/eda/openml_46915.png` — overview: target balance and missingness
- `reports/eda/openml_46915_numeric.png` — numeric distributions by target
- `reports/eda/openml_46915_conditional.png` — target-conditional comparisons
- `reports/eda/openml_46915_correlation.png` — numeric correlation heatmap
- `reports/eda/openml_46915_categorical.png` — categorical composition by target

Capabilities available for Phase 5 on this dataset:

| Capability | Available | Source column |
| --- | --- | --- |
| has_tenure | no | — |
| has_monthly_charge | no | — |
| has_total_charges | no | — |
| has_contract | no | — |
| has_payment_method | no | — |
| has_internet_service | no | — |
| has_support_calls | **yes** | `number_customer_service_calls` |
| has_usage_minutes | **yes** | `total_day_minutes` |
| has_plan | **yes** | `international_plan` |
| has_senior_citizen | no | — |
| has_gender | no | — |
| has_revenue_proxy | no | — |

Phase 5 must therefore build only the three evidenced features on this dataset
and record why the other nine are skipped. No revenue proxy exists, so
value-at-risk scoring cannot claim a monetary basis from this data.

---

## 5. Artifacts

| Artifact | Contents |
| --- | --- |
| `churnwise/validation/loader.py` | format detection, safe parsing, duplicate-header handling |
| `churnwise/validation/targets.py` | target scoring, interpretation, normalisation |
| `churnwise/eda/profile.py` | column profiling, roles, flags, leakage heuristics, capabilities |
| `churnwise/validation/checks.py` | V1–V12 with thresholds and evidence |
| `churnwise/validation/gate.py` | severity rollup and blocking decision |
| `churnwise/validation/report.py` | JSON/Markdown writers and EDA orchestration |
| `churnwise/validation/coordinator.py` | verify → load → profile → check → persist, plus fallback |
| `churnwise/validation/contracts.py` | `ValidationReport`, `ValidationResult`, attempts and failure types |
| `churnwise/eda/plots.py` | five deterministic EDA figures |
| `churnwise/cli.py` | `validate` command and exit codes |
| `reports/validation/openml_46915.{json,md}` | real report and gate |
| `reports/eda/openml_46915.{json,png}` + four figures | real profile and plots |
| `tests/fixtures/mini_churn.csv` | 160-row synthetic, test-only fixture |
| `tests/unit/test_validation_checks.py` | check-by-check pass/fail coverage |
| `tests/integration/test_validation_gate.py` | end-to-end gate, provenance and pandas cross-check |
| `tests/test_validation_cli.py` | CLI wiring, offline behaviour and exit codes |
| `tests/test_live_validation.py` | live discovery → acquisition → validation |

---

## 6. Verification

**Mocked suite:** `232 passed, 6 skipped` (`pytest -q`).

**Live suite:** `pytest -q --run-live` — `238 passed` in 186 s. It runs the
five existing Phase 2/3 live tests plus the new Phase 4 live validation test.
The Phase 4 live test repeats
discovery, acquisition and validation against OpenML, then rechecks the on-disk
digest, the positive rate with pandas, the report/profile files and the plot
files.

**Phase 1 documentation checker:** `scripts/verify_phase1_docs.py` passes
**114/114** after the documentation updates.

**Real artifact rebuilt:** the live run rewrote `reports/validation/
openml_46915.{json,md}`, `reports/eda/openml_46915.{json,png}` plus four figures,
and `data/manifests/openml_46915.profile.json` from the stored raw ARFF.

**Acceptance mapping**

| ID | Evidence |
| --- | --- |
| T4.1 | `test_every_check_has_a_passing_and_failing_fixture` asserts both outcomes for all twelve ids |
| T4.2 | single-class and missing-target fixtures reject the gate (`tests/unit`, `tests/integration`) |
| T4.3 | parametrised `Yes/No`, `0/1`, `True/False`, lowercase and retention inversion tests |
| T4.4 | integration test asserts `customerID` is flagged, assigned `id`, and excluded from features |
| T4.5 | end-to-end single-class acquisition rejects and `normalized_frame()` returns `None` |
| T4.6 | integration and live tests compare report values with independent pandas recomputation |

---

## 7. Limits and honest gaps

- **No training block exists yet.** `can_preprocess = false` is the contractual
  gate; Phase 5 must consume it. There is no Phase 5 code to block today.
- **ARFF nominal decoding** is handled by the loader; XLSX remains opaque until
  Phase 4's loader is exercised on a real XLSX artifact. None was acquired.
- **Target detection can reject.** A neutral `0/1` column is deliberately
  low-confidence. This avoids silently inventing churn semantics.
- **`state` exclusion** loses geography. Phase 5 may add explicit frequency or
  target-safe encoding, using the profile evidence.
- **Capability gaps** on this dataset (no tenure, contract, charge or revenue
  fields) are real data limitations, not implementation gaps.
- **EDA is descriptive.** Correlation figures are not significance tests and do
  not imply causal retention drivers.

---

## 8. Hand-off to Phase 5

Phase 5 receives:

- the untouched validated frame,
- a `DatasetProfile` with target, roles, flags and capabilities,
- the gate decision (`can_preprocess = true` on the real dataset),
- the report digests needed to prove the same artifact was used.

It must split reproducibly, fit preprocessing on train only, build derived
features **only** where the capability matrix says the source exists, and keep
the test set untouched.
