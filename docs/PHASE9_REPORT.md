# ChurnWise AI — Phase 9 Report: Customer Churn Risk Intelligence

| Field | Value |
| --- | --- |
| Status | **Complete and verified** for the implemented scope |
| Phase | 9 — Customer Churn Risk Intelligence |
| Date | 2026-10-22 |
| Dataset | OpenML `46915` (`openml:46915`), 5,000 rows, 27 transformed features, binary target `CustomerChurned` (`Yes` = 1) |
| Preprocessing run | `20261021T123405Z-openml_46915-989d8802` |
| Phase 7 benchmark | `20261021T145336Z-openml_46915-bench-04595d7b` |
| Shipped family | Random Forest — validation F1 `0.7985`, threshold `0.45` |
| Real risk run | `20261022T155236Z-openml_46915-risk-final` |
| Interface documentation | [RISK_INTELLIGENCE.md](RISK_INTELLIGENCE.md) |
| Focused tests | **58 passed** |
| Full suite | **490 passed, 7 skipped** (the skips are opt-in live-network tests) |
| Real-model evidence | **Yes** — population, raw batch, individual assessment and filtered export (below) |

This report separates three kinds of evidence:

1. **Real-model evidence** — scoring and SHAP against the persisted OpenML
   `46915` model and preprocessing artifacts.
2. **Synthetic test evidence** — deterministic fixtures that exercise failure
   paths, thresholds, exports and rejection handling. These are not claims about
   real churn customers.
3. **Unverified scope** — business threshold optimality, probability calibration,
   revenue proxies and all later phases.

No score, probability, driver, count or test result in this report was invented,
hand-assigned, or copied from a template. No OpenML model was retrained and no
source dataset or trained production artifact was modified; the automated test
fixtures fit their own synthetic model on synthetic rows.

---

## 1. Upstream inspection and selected model

Phase 9 was built on the actual Phase 5–8 interfaces rather than a parallel
prediction stack.

| Upstream artifact | Verified content | Phase 9 use |
| --- | --- | --- |
| `artifacts/preprocessors/20261021T123405Z-openml_46915-989d8802/pipeline.joblib` + `schema.json` | 27 feature names, 18 required raw columns, `Yes`/`No` → `1`/`0`, split fingerprint, preprocessor hash | Exact raw-record transformation through `dataset.transform_new` |
| `artifacts/evaluation/20261021T145336Z-openml_46915-bench-04595d7b/metrics.json` | Three completed families; validation-only selection; Random Forest threshold `0.45` | Selected family, frozen classification threshold, provenance |
| `.../models/random_forest/model.joblib` + `metadata.json` | SHA-256 `c074db6628fc21afcc7ff71c4bb8f066379cb21c46cb03b7fab0db5cf8e949e4` | Actual `predict_proba` source |
| Phase 8 `ExplainCoordinator` | Verified model/preprocessor loading, local SHAP explanations, provenance checks | Drivers and model loading; no separate explainer |
| `artifacts/splits/...` and Phase 5 arrays | Train 3,000 / validation 1,000 / test 1,000; validation 142 positive (14.2%) | Population scoring |

Phase 7 selected Random Forest on **validation F1 0.7985** (MLP 0.6937,
Logistic Regression 0.5905); the test partition was not used for selection.
Phase 9 therefore scores the selected Random Forest and records family, model run
id, model digest, preprocessing run, split fingerprint and dataset hash in every
run.

The positive churn class is `Yes` (`1`). A probability is therefore
`P(CustomerChurned = Yes)`. The classification threshold is `0.45`; risk-tier
boundaries are independently configurable.

---

## 2. Implementation

| Module | Responsibility |
| --- | --- |
| `churnwise/intelligence/contracts.py` | `RiskThresholds`, `RiskConfig`, `RiskAssessment`, `RiskDriver`, `RiskFilter`, `RiskTable`, `BatchAssessment`, `RejectedRecord`, `RiskRun`, `RiskExport`, stable export schema |
| `churnwise/intelligence/model.py` | Read-only selected-model loading through the Phase 8 verifier |
| `churnwise/intelligence/drivers.py` | Phase 8 local-explanation loading/attachment and provenance checks |
| `churnwise/intelligence/service.py` | Individual/batch/population scoring, references, drivers, persistence, retrieval, exports, freshness checks |
| `churnwise/intelligence/ranking.py` | Tiers, deterministic ranking, filters, pagination, value-at-risk |
| `churnwise/intelligence/reporting.py` | Evidence-only console summaries |
| `churnwise/intelligence/errors.py` | Stable `CW-RISK-*` errors with remediation |
| `churnwise/config.py` | `CHURNWISE_RISK_TIERS` configuration and validation |
| `churnwise/cli.py` | `score` and `risk-export` commands |

The Phase 8 `ExplainCoordinator.load_model()` is the only model-loading path.
It verifies the benchmark selection, model digest, preprocessing run, feature
order and split provenance before returning a prediction callable. Phase 9 does
not reload a different model, refit preprocessing or call `predict` directly.

New dependencies: **none**. Phase 9 uses the already-declared pandas, numpy,
pyarrow, SHAP and scikit-learn stack.

---

## 3. Individual assessment

### 3.1 Interface

```python
assessment = service.assess_customer(
    record,
    reference=None,
    explain=True,
    explain_run_id=None,
    value=None,
)
```

`assess_customer` wraps the same `assess_batch` path used for many records.
It returns one `RiskAssessment`, never a fabricated fallback. If the single
record cannot be transformed or scored, the call raises `RiskError` with the
record-level rejection code and message.

The object contains:

- opaque `record_reference` and `input_position`;
- `churn_probability`, `churn_probability_pct`, `predicted_class`,
  `predicted_label`, `risk_tier`;
- `classification_threshold` and the complete tier configuration;
- model family/run id/digest and preprocessing run;
- `assessed_at`;
- `top_positive_drivers`, `top_negative_drivers`, `explanation_summary`,
  `explanation_status` and `explanation_reason`;
- warnings, configured segment values and optional value at risk.

An identifier passed as `reference` is hashed to `cust_<12 hex>` before storage.
Without an identifier, the reference is positional. Raw customer identifiers are
never written to a risk artifact or export.

### 3.2 Real individual result

The third raw row from the real six-row batch (source dataset row 37) was
assessed with the API at `2026-10-22T15:53:37Z`:

| Field | Result |
| --- | --- |
| Probability | `0.3891554257158917` (38.92%) |
| Predicted class at 0.45 | `0` (retained) |
| Risk tier | `medium` (0.30 ≤ p < 0.50) |
| Model | `random_forest`, `20261021T145336Z-openml_46915-bench-04595d7b/random_forest` |
| Model SHA-256 | `c074db6628fc21afcc7ff71c4bb8f066379cb21c46cb03b7fab0db5cf8e949e4` |
| Preprocessing run | `20261021T123405Z-openml_46915-989d8802` |
| Explanation | available, 3 positive / 5 negative source-feature drivers |
| Observed outcome | `null` — no outcome is known for a raw record |

Strongest upward SHAP contributors (probability space): `number_customer_service_calls`
(+0.172), `support_calls_per_usage_hour` (+0.144), `voice_mail_plan` (+0.002).
Strongest downward: `total_day_charge` (−0.109), `total_day_minutes` (−0.107),
`total_eve_minutes` (−0.033). The full summary explicitly states that these are
model attributions, not causal effects.

This probability matches the batch and population scores for the same record
(section 4 below).

---

## 4. Batch assessment

### 4.1 Input and reconciliation

`assess_batch(frame, batch_size=...)` accepts raw records with every Phase 5
`required_columns` field. The fitted Phase 5 pipeline performs all cleaning,
derived features, imputation, one-hot encoding and scaling.

- `batch_size` controls rows transformed per chunk (default 200).
- Rows retain their original `input_position`; output records are sorted by it.
- `total_received == scored + rejected` is asserted by `BatchAssessment`.
- Missing required columns reject every row with `CW-RISK-004` and a batch error.
- Dict/list/set/tuple values in required fields reject that row; other rows are
  scored.
- If a chunk transform fails, the service retries the affected rows one by one
  so one malformed record cannot discard its neighbours.
- Blank/missing identifier values fall back to positional references for those
  rows (with a warning); an absent or empty identifier column never crashes.
- Duplicate references receive `_dup2`, `_dup3`, etc.
- Unused extra columns are reported in a warning and not passed to the model.
- Missing values use the fitted Phase 5 imputer and produce a row-level warning.
- All-rejected batches persist an empty scored table, `rejections.json` and the
  received/scored/rejected counts; the CLI returns 2.

The CLI reads CSV input as text deliberately. Without this, numeric-looking
categorical levels (`0`/`1` plan flags, `408` area codes) would be read as
integers and silently encoded as unknown levels instead of matching the fitted
string vocabulary.

### 4.2 Test evidence (genuine fitted model, synthetic data)

`tests/integration/test_scoring_counts.py` fits the Random Forest on the real
Phase 5 preprocessing of a deterministic synthetic frame. It does not mock
`predict_proba`. It verifies:

- population scoring gives every row one probability, tier and unique rank;
- tier counts reconcile to the population;
- individual and batch probabilities agree exactly;
- batch size 3 and batch size 100 produce identical row probabilities;
- a malformed row rejects only itself;
- empty and missing-schema batches are structured;
- missing values are handled by Phase 5 and reported;
- blank, partially blank and absent identifier columns fall back safely;
- duplicate references stay distinct;
- explanation modes and unavailable reasons, including malformed Phase 8 files
  and invalid explanation ids;
- CSV/JSON/Parquet round-trip, identifier exclusion, path escape, stale model
  and preprocessor rejection, malformed-table and count-mismatch rejection;
- JSON keeps the documented `segment` key and records the applied filters;
- value-at-risk ordering, missing-proxy behaviour and value-proxy provenance.

### 4.3 Real raw batch

Six raw records were taken from the actual OpenML ARFF at source positions
`[10, 12, 37, 38, 42, 43]` — the first six validation rows. They were written to
CSV without the target column and scored by the real CLI with all 1,000-row
drivers enabled for the batch subset.

Run `20261022T155314Z-openml_46915-actual-batch`:

| Input position | Probability | Class at 0.45 | Tier |
| --- | --- | --- | --- |
| 0 | 0.042744 | 0 | low |
| 1 | 0.036803 | 0 | low |
| 2 | 0.389155 | 0 | medium |
| 3 | 0.028884 | 0 | low |
| 4 | 0.046594 | 0 | low |
| 5 | 0.063958 | 0 | low |

Counts: 6 received, 6 scored, 0 rejected; tiers low 5, medium 1; explanations
available 6/6. Duration 0.71 s. Warnings correctly noted that no identifier
column was supplied and that the source `state` column is not a model feature.

The maximum absolute difference between these six batch probabilities and the
same-position rows in the real validation population run was **1.39e-17**.
This is the direct evidence that the individual, batch and population paths use
the same preprocessing and prediction code.

---

## 5. Thresholds and risk categories

### 5.1 Defaults

| Range | Tier |
| --- | --- |
| `p < 0.30` | low |
| `0.30 ≤ p < 0.50` | medium |
| `0.50 ≤ p < 0.70` | high |
| `p ≥ 0.70` | critical |

Two boundaries produce low/medium/high; three produce
low/medium/high/critical. The implemented defaults are deliberately neutral, not
claimed to be business-optimal.

### 5.2 Separation from classification

The Phase 7 classification threshold decides `predicted_class` and is recorded
as `0.45` in the real run. Tier boundaries decide only `risk_tier`. The real
individual record is a concrete example: 0.389 is `medium` risk but predicted
`retained` at 0.45.

### 5.3 Validation

`RiskThresholds` and `RiskConfig` reject:

- fewer than two or more than three boundaries;
- non-finite, zero, one or out-of-range boundaries;
- duplicate or non-increasing boundaries;
- invalid batch size, top-k, driver cap, export cap, split, explanation mode,
  horizon, column name or boolean strictness;
- non-numeric value proxies.

The validation is exercised directly and through the CLI
(`CW-RISK-001`, exit 3, no traceback).

---

## 6. Ranking, filtering and prioritization

### 6.1 Deterministic ranking

Default ordering is descending churn probability. Ties break by opaque
`record_reference`, then `input_position`. Every row receives a unique rank from
1 to N. The real population has 1,000 unique ranks, and the top run is:

| Rank | Reference | Probability | Tier |
| --- | --- | --- | --- |
| 1 | `risk_val_00746` | 0.97653 | critical |
| 2 | `risk_val_00149` | 0.976172 | critical |
| 3 | `risk_val_00480` | 0.974614 | critical |

Other supported sort keys are `value_at_risk`, `risk_tier` and
`record_reference`. The documented tie-break chain is implemented in
`ranking.py` and tested.

### 6.2 Filters and pagination

`RiskFilter` supports tier sets, inclusive probability ranges, explicit
references, segment equality and segment numeric bounds. Filters operate on
stored assessments and make no model calls. Limits and offsets are validated;
pagination never returns more than `max_export_rows`.

### 6.3 Value at risk

`value_at_risk = p × proxy_value × horizon_months` is computed only when the
caller supplies a real proxy. When no proxy exists, the field is null, the
summary says `"available": false`, and sorting by value at risk raises
`CW-RISK-001` rather than inventing money. The reference OpenML dataset has no
reliable revenue field, so the real run deliberately has no value-at-risk
figures. Formula and ordering are tested with synthetic proxies only.

Risk ranking is not a revenue ranking. No lifetime-value ordering is offered.

---

## 7. Explainability integration

Phase 9 calls Phase 8, never a second SHAP implementation.

| Mode | Behaviour |
| --- | --- |
| `none` | No SHAP; every row is explicitly unavailable with reason `"explanation was not requested"`. |
| `top` | Explains the `explain_top_n` highest-probability records; others are marked unavailable with a cost-control reason. |
| `all` | Explains every scored row up to `max_driver_rows` (default 1,000). |

`explain_run_id` reuses a persisted Phase 8 matrix run after checking that its
mode, model digest and preprocessing run match. Missing/failed explanations
produce an explicit `explanation_reason` and a batch warning; probabilities,
classes and tiers remain available.

Real evidence:

- The validation population run generated 1,000 local explanations in
  `artifacts/explanations/20261022T155236Z-openml_46915-risk-final-drivers/`
  (1,000 local JSON files + manifest).
- The manifest records `mode: matrix`, family `random_forest`, explainer
  `shap.TreeExplainer`, output space `probability`, background 100 rows,
  `rows_explained: 1000`, threshold 0.45 and seed 42.
- The real individual assessment has 3 positive and 5 negative drivers with
  signed contributions and human-readable feature/level values.
- Explanations are copied into `churn_scores.parquet`, `drivers.json` and
  exports only when available.

Limitations:

- SHAP describes model behaviour, not causality. Neither the summary nor the
  interface claims that changing a feature will retain a customer.
- Logistic Regression values are log-odds while Random Forest and MLP values are
  probability-space; cross-family magnitudes are not comparable. The selected
  model for this run is the Random Forest.
- SHAP is computed for a bounded population. Rows beyond `max_driver_rows` are
  explicitly unavailable rather than silently omitted.
- Phase 8 additivity guarantees apply to the explanation values it produced;
  Phase 9 does not alter them.

---

## 8. Exports and privacy

### 8.1 Formats and schema

`RiskService.export` writes:

- **CSV** — one flat row per assessment;
- **JSON** — typed records, model/threshold/filter metadata, warnings, row count;
- **Parquet** — the same stable flat schema for downstream tools.

The 26 stable columns are listed in
[RISK_INTELLIGENCE.md](RISK_INTELLIGENCE.md#71-stable-column-schema). They
include the opaque reference, probability, class, tier, rank, thresholds,
model/preprocessor identity, timestamp, observed label (audit only), optional
value at risk, explanation fields, warnings and segment.

The JSON export always carries a `segment` object (empty unless segment export
was requested) and records the complete applied filter, sort field, limit and
offset alongside the rows, so a consumer can reproduce the subset without
re-deriving it.

Real export evidence:

| Export | Rows | Result |
| --- | --- | --- |
| Full run export | 1,000 | 26 columns; tier counts 858 low / 31 medium / 40 high / 71 critical |
| Filtered `high` + `critical` export | 111 | 40 high + 71 critical; probability range 0.5055–0.9765 |
| Top-10 filtered export | 10 | all critical, as expected from descending ranking |

All three were read back with pandas. Row counts and tier membership matched the
stored table.

### 8.2 Privacy safeguards

- Raw identifiers are hashed or replaced by positional references before use.
- Raw identifiers and unrelated raw columns never appear in the scored table or
  export.
- The `segment` field is empty unless fields were explicitly configured and
  `include_segment=True` was requested.
- Unused input columns are dropped after a warning.
- Error messages use positions and opaque references; tests assert that an
  invalid value is not echoed.
- Driver artifacts are keyed by opaque references.

### 8.3 CSV formula injection

Every string cell whose literal first character, or first character after
leading whitespace, is `=`, `+`, `-`, `@`, tab or carriage return is prefixed
with an apostrophe. This is tested for `=1+1`, `+SUM(A1)`, `-2`, `@cmd`,
`  =1+1`, tab-led formulas and tab/CR-led text; normal text and numeric
probabilities are unchanged. JSON cells inside CSV begin with `{` or `[`, so
they are not turned into formulas.

### 8.4 Path safety and failures

Exports are confined to `artifacts/predictions/<run_id>/`. An absolute directory
outside that tree, or `..` traversal, raises `CW-RISK-007`. CSV/JSON writes are
atomic; failures return an actionable export error. Re-exporting replaces only
that run's own `exports/` files. Empty filtered exports preserve the schema and
report row count 0.

---

## 9. Provenance and artifact compatibility

`run_manifest.json` records:

- risk `run_id`, schema version, generated timestamp, duration;
- dataset id and SHA-256;
- split fingerprint, preprocessing run id, preprocessor hash/path;
- Phase 7 benchmark run id;
- model family, run id, SHA-256 and path;
- full risk configuration and fingerprint;
- received/scored/rejected and tier counts;
- warning/limitation lists and artifact paths.

Before loading a persisted run, `load_run` re-hashes the recorded model and
preprocessor files. A changed file is `CW-RISK-003` (“stale”), not silently
reused. A missing file is reported as unavailable with a warning. Stored tables
are also schema-checked: a missing documented column, a row-count mismatch, or
`received != scored + rejected` is rejected as `CW-RISK-003` instead of being
loaded partially. The manifest records the effective revenue-proxy column when
one was used. Tests also confirm that a modified split fingerprint in the model
metadata is rejected before scoring.

The source dataset, Phase 5 arrays, model artifacts and benchmark metrics were
opened read-only. Phase 9 created new prediction/explanations directories and
did not delete or rewrite prior artifacts.

---

## 10. Tests and exact results

### 10.1 Focused Phase 9 suite

```text
PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest \
  tests/unit/test_ranking_tiers.py \
  tests/integration/test_scoring_counts.py \
  tests/test_risk_cli.py -q
```

**Result: 58 passed in 30.44 s.**

Coverage includes:

| Acceptance check | Test evidence |
| --- | --- |
| T9.1 probability/tier/unique rank | `test_population_scores_every_row_with_unique_ranks_and_tier_counts` |
| T9.2 tier counts sum to population | same test + `test_tier_counts_sum_to_population` |
| T9.3 boundary mapping | `test_tier_boundaries_are_lower_inclusive`, `test_two_boundary_configuration_has_three_tiers`, `test_classification_threshold_override_does_not_move_tiers` |
| T9.4 value-at-risk formula | `test_value_at_risk_ordering_matches_formula` |
| T9.5 filters exact | `test_filters_return_exactly_matching_rows`, `test_invalid_filters_are_rejected` |
| T9.6 export round-trip | `test_export_round_trip_and_privacy`, `test_json_export_keeps_stable_schema_and_records_applied_filters` |
| T9.7 missing proxy not fabricated | `test_missing_value_proxy_is_not_fabricated` |
| T9.8 real batch/population parity | six-row real run below; synthetic `test_individual_and_batch_use_the_same_probability_path` |
| T9.9 CSV injection/privacy | `test_csv_formula_injection_is_neutralised`, `test_export_round_trip_and_privacy` |
| T9.10 stale artifacts/path escape | `test_stale_model_artifact_is_rejected_on_reload`, `test_changed_preprocessor_artifact_is_rejected_on_reload`, `test_malformed_scored_table_is_a_structured_artifact_error`, `test_stored_row_count_mismatch_is_rejected`, `test_export_cannot_escape_the_predictions_directory` |
| T9.11 partial/all-rejected batches | `test_partial_batch_failure_isolates_the_bad_record`, `test_all_rejected_batch_persists_counts_and_rejection_details` |

Additional focused tests cover empty batches, missing schema, blank/partial and
absent identifier columns, duplicate references, unexpected columns,
missing-value warnings, all-rejected runs, invalid value proxies, config
errors, CLI exit codes, explanation modes (including population `top` using a
row count rather than the feature top-k), unavailable and malformed
explanations, driver caps, classification-threshold separation, JSON filter
provenance, missing model artifacts and model/preprocessor mismatch.

### 10.2 Phase 5–8 integration checks

```text
PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest \
  tests/integration/test_scoring_counts.py \
  tests/integration/test_shap_mlp.py \
  tests/unit/test_shap_coordinator.py -q
```

**Result: 43 passed in 34.87 s.**

The Phase 8 MLP integration test loads a real Keras fixture; the coordinator
tests exercise persisted SHAP artifacts. Phase 9 tests use the real Phase 5
preprocessor and a genuinely fitted baselines model; the core probability
calculation is never mocked.

### 10.3 Full suite

```text
PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest -q
```

**Result: 490 passed, 7 skipped in 98.35 s** (the skipped tests are the seven
opt-in live-network tests). The full run includes Phases 1–8 and the complete
Phase 9 suite. Warnings are dependency deprecations (Keras input-structure and
SHAP/NumPy plotting warnings), not failures.

No test requires network access or the real customer dataset; the real-model
runs below are separate manual evidence.

### 10.4 Post-implementation verification and fixes

A separate verification pass re-read the Phase 9 code and exercised edge cases
not covered by the original suite. Seven defects were found, fixed and covered
by regression tests:

1. **Blank identifier column crash.** A reference column whose values were all
   null/blank raised `UnboundLocalError`. It now falls back to positional
   references with a warning. Partially missing identifiers fall back per row,
   and a requested-but-absent column is reported explicitly rather than
   described as “no identifier supplied”.
2. **Incompatible scored table crash.** An edited Parquet table (missing a
   documented column) raised `KeyError` from `load_run` and a CLI traceback, and
   missing rejection details could silently contradict the recorded counts.
   The loader now validates the stable schema, row/scored/received counts and
   rejection-detail counts, and raises `CW-RISK-003` with remediation.
3. **Malformed Phase 8 artifact crash.** A malformed local explanation file or
   an invalid explanation run id escaped as `ValueError`/`ExplainError`. Both
   are now `CW-RISK-006` warnings; probabilities, tiers and ranks are retained
   and the rows are marked unavailable. Unexpected auxiliary explanation
   failures are also contained rather than breaking scoring.
4. **JSON export schema/filter gap.** JSON records omitted `segment` when segment
   export was off, and only tier/probability filters were recorded. JSON now
   always includes `segment` (empty unless requested) and the complete applied
   filter, sort, limit and offset metadata.
5. **Tab/CR CSV cells were not neutralised.** The guard stripped whitespace
   before checking, so a cell beginning with a literal tab or carriage return
   was not prefixed. Both the literal and whitespace-stripped first characters
   are now checked and tested.
6. **Misleading missing-value warnings.** Row warnings counted missing values in
   ignored or segment-only columns as “handled by the Phase 5 pipeline”. They
   now count only required model columns.
7. **Unrecorded value-proxy column.** The effective revenue-proxy column was not
   in provenance. `risk_summary.json` now records
   `value_at_risk.column` and `run_manifest.json` records
   `value_proxy_column`.

After the fixes, the focused suite reported **58 passed**, the Phase 5–8
integration command **43 passed**, the full suite **490 passed / 7 skipped**,
and the documentation checker **114 checks passed / 0 failed**.

---

## 11. Real-model evidence vs synthetic tests

### 11.1 Real OpenML validation population

Command:

```bash
python -m churnwise.cli score --population val --explain all \
  --run-id 20261022T155236Z-openml_46915-risk-final \
  --benchmark-run-id 20261021T145336Z-openml_46915-bench-04595d7b \
  --export csv --export json
```

Result (`risk_summary.json`, `run_manifest.json`):

| Measure | Value |
| --- | --- |
| Received / scored / rejected | 1,000 / 1,000 / 0 |
| Duration | 16.96 s |
| Low / medium / high / critical | 858 / 31 / 40 / 71 |
| Predicted churn / retained | 121 / 879 |
| Probability min / median / mean / max | 0.0 / 0.05413 / 0.15231 / 0.97653 |
| Explanations available | 1,000 / 1,000 |
| Tier percentages | 85.8% / 3.1% / 4.0% / 7.1% |

The predicted churn count (121) is below the 142 observed validation positives
but is a model decision at the frozen 0.45 threshold, not a reconstruction of
labels. Observed labels are retained only for audit.

### 11.2 Real raw batch

`20261022T155314Z-openml_46915-actual-batch`: 6 actual OpenML rows, six scored,
zero rejected, one medium and five low; probabilities and parity shown in
section 4.3. This is the genuine `raw CSV → transform_new → predict_proba`
path, not a synthetic fixture.

### 11.3 Real individual assessment

The API assessment at 15:53:37 on actual row 37 returned 0.3891554 (medium,
retained at 0.45) with available Phase 8 drivers (section 3.2). It matches the
batch result at input position 2 exactly.

### 11.4 Real filtered export

`risk-export --tier high --tier critical --limit 200` on the real run returned
111 rows: 40 high and 71 critical, with probabilities 0.5055–0.9765. A full
export returned 1,000 rows and all 26 documented columns. These are actual
OpenML-derived scores, not synthetic rows.

Synthetic tests are still used for value-at-risk because the source dataset has
no revenue proxy, and for all failure/edge paths that should not be induced in
real artifacts.

---

## 12. Acceptance criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every input row receives exactly one score; counts reconcile | Verified | Real population 1,000 = 1,000 scored + 0 rejected; `counts_reconcile` tests |
| Tiers derived from config thresholds, counts/percentages reported | Verified | Real tier counts and `tiers.csv`; boundary tests |
| Ranking deterministic and tie-break documented | Verified | Unique ranks; `test_ranking_is_deterministic_and_breaks_ties_by_reference` |
| Top drivers attached from precomputed SHAP | Verified | 1,000 real explanations attached; synthetic availability/unavailability tests |
| Value-at-risk present when proxy exists, absent/explained otherwise | Verified (synthetic proxy); real absence correctly reported | `test_value_at_risk_ordering_matches_formula`; real summary `"available": false` |
| Exports usable without recomputation | Verified | 1,000-row CSV/JSON/Parquet read-back; 111-row filtered export |
| CSV formula injection prevented | Verified | `test_csv_formula_injection_is_neutralised` (including tab/CR cells) and export round-trip |
| Sensitive identifiers excluded | Verified | Opaque and positional references; export tests assert raw identifier absent; blank/partial identifier tests |
| Stale/incompatible artifacts rejected | Verified | Model and preprocessor digest tamper, malformed table and count-mismatch tests return `CW-RISK-003` |
| No retraining or model selection during scoring | Verified by code path and artifacts | Only `ExplainCoordinator.load_model`; no training call in Phase 9 |

The original T9.1–T9.7 checks in
[ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md) remain satisfied. T9.8–T9.11
were added to reflect the Phase 9 completion criteria.

---

## 13. Known limitations and unverified scope

1. **Threshold optimality is not established.** Default boundaries are neutral
   implementation defaults. No intervention costs, contact capacity, save rates
   or outcome study were available.
2. **Probabilities are uncalibrated.** Phase 7 reports a low Brier score and
   ROC-AUC, but no isotonic/Platt calibration map was fitted. A 70% score is a
   model estimate, not a guarantee.
3. **No real revenue proxy.** OpenML `46915` has no reliable monthly value or
   lifetime-value field. Value-at-risk is tested only with synthetic values and
   is absent from the real run by design.
4. **No customer identifier in the source data.** Real population references are
   positional (`risk_val_<position>`); raw batches use positional references and
   warn about order dependence. This is privacy-safe but means references are not
   stable across reordered input.
5. **Segment filtering is caller-configured.** The real dataset has no contract,
   tenure-band or segment fields beyond its model features; segment tests use
   synthetic metadata.
6. **Large-batch stress is bounded.** Tests cover batch sizes up to 240 synthetic
   rows and 1,000 real population rows, not multi-million-row streaming.
7. **SHAP cost is bounded.** `max_driver_rows` defaults to 1,000. Rows beyond the
   cap are explicitly unavailable. The real 1,000-row Random Forest run took
   16.96 s end to end on this machine; no other hardware was measured.
8. **No database or external store.** Runs are filesystem artifacts under
   `artifacts/predictions/`, consistent with Phases 1–8.
9. **Later phases were not implemented at the time of this report.** Phases 10–13
   have since landed: campaign simulation, the dashboard, end-to-end
   orchestration and final verification are documented in
   [PHASE10_REPORT.md](PHASE10_REPORT.md), [PHASE11_REPORT.md](PHASE11_REPORT.md),
   [PHASE12_REPORT.md](PHASE12_REPORT.md) and
   [FINAL_VERIFICATION_REPORT.md](FINAL_VERIFICATION_REPORT.md). All risk evidence
   in this report still stops at Phase 9.
10. **Live-network checks remain opt-in.** Seven live tests were skipped in the
    default suite; they are unrelated to Phase 9 and were not used as evidence.

---

## 14. Conclusion

Phase 9 is complete for its defined scope. The real selected Random Forest
produced 1,000 genuine validation scores with reconciled tier counts and 1,000
Phase 8 explanations; a six-row raw batch and a single-record API assessment
reproduced the population probabilities to floating-point precision. Thresholds,
ranking, filters, rejection handling, CSV/JSON/Parquet exports, formula-injection
guards, privacy and artifact freshness are covered by automated tests.

The remaining risk is business interpretation, not missing prediction plumbing:
tier boundaries must be validated against real intervention outcomes, and
value-at-risk requires a trustworthy revenue proxy. Those are Phase 10+ concerns
and are not claimed here.
