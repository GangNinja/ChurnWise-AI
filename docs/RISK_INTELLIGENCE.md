# ChurnWise AI — Customer Risk Intelligence and Exports

| Field | Value |
| --- | --- |
| Status | Phase 9 implemented and tested; real-model evidence in [PHASE9_REPORT.md](PHASE9_REPORT.md) |
| Schema version | `1.0` |
| Module | `churnwise/intelligence/` |
| Model source | Phase 7 selected family, loaded through the verified Phase 8 coordinator |
| Positive class | Dataset-defined, `Yes` = churn for OpenML `46915` |
| Default classification threshold | Phase 7 validation-selected threshold (0.45 for the reference Random Forest) |
| Default tier boundaries | `0.30, 0.50, 0.70` |
| Related documents | [ARCHITECTURE.md](ARCHITECTURE.md) · [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md) · [PHASE7_REPORT.md](PHASE7_REPORT.md) · [PHASE8_REPORT.md](PHASE8_REPORT.md) |

---

## 1. What Phase 9 adds

Phase 9 turns the selected Phase 7 model's probabilities into a customer-level
risk view without creating another prediction path:

```text
raw record
  -> Phase 5 fitted preprocessor (transform_new)
  -> shipped Phase 7 model (predict_proba)
  -> probability + classification threshold + risk tier
  -> optional Phase 8 SHAP drivers
  -> deterministic ranking / filters / pagination
  -> persisted Parquet + JSON + CSV
```

`RiskService` reuses `ExplainCoordinator.load_model()` from Phase 8 for every
model load. That loader verifies the model digest, benchmark selection,
preprocessing run, split fingerprint and feature order before returning a
callable. Phase 9 does not load a second model, refit preprocessing, retrain,
or recompute SHAP with a different method.

This subsystem answers *"what does the shipped model estimate for this
customer?"* It does not answer *"what will necessarily happen?"*.

---

## 2. Public interface

### 2.1 Constructing the service

```python
from churnwise.config import Settings
from churnwise.intelligence import RiskConfig, RiskService, RiskThresholds

service = RiskService(
    Settings.load(),
    RiskConfig(
        thresholds=RiskThresholds.from_value((0.30, 0.50, 0.70)),
        classification_threshold=None,   # use the model's frozen threshold
        batch_size=200,
        top_k=5,
        max_driver_rows=1000,
        population_split="val",
        population_explain="all",
    ),
)
```

`benchmark_run_id` selects a specific Phase 7 benchmark. If omitted, the Phase 8
loader resolves the shipped benchmark recorded in the model artifact. A
`classification_threshold` override changes only the churn/retained decision;
it never changes the risk-tier boundaries. `top_k` is the number of signed
feature drivers kept per explained row; `explain_top_n` is the number of rows
explained in `top` mode.

### 2.2 Assessment and retrieval methods

| Method | Purpose |
| --- | --- |
| `assess_customer(record, *, reference=None, explain=False, explain_run_id=None, value=None)` | Assess one raw record. Returns one `RiskAssessment`. |
| `assess_batch(frame, *, batch_size=None, explain="none", explain_top_n=10, explain_positions=None, explain_run_id=None, reference_override=None, reference_column=None, value_column=None, values=None, segment_fields=None)` | Assess raw records and return a `BatchAssessment` with scored records and record-level rejections. |
| `score_frame(frame, *, explain="none", explain_top_n=10, explain_positions=None, explain_run_id=None, reference_column=None, value_column=None, segment_fields=None, batch_size=None, run_id=None, persist=True, progress=None)` | Score caller-supplied records, rank them and (by default) persist a risk run. |
| `score_population(*, split=None, explain=None, explain_top_n=10, explain_positions=None, run_id=None, persist=True, values=None, explain_run_id=None, progress=None)` | Score a persisted Phase 5 split (`train`, `val`, `test`, `all`) and persist a run; `top` mode explains `explain_top_n` highest-risk rows. |
| `ranked(*, run_id=None, filters=None, sort_by="probability", limit=50, offset=0, verify_freshness=True)` | Load a stored run, filter, rank and paginate without calling the model. |
| `get_assessment(run_id, reference)` | Look up one stored assessment by opaque reference. |
| `get_explanation(run_id, reference)` | Return stored drivers and explanation status for one assessment. |
| `get_summary(run_id=None)` | Return the persisted `risk_summary.json` payload. |
| `inspect_thresholds()` | Return tier boundaries, labels and the effective classification threshold. |
| `export(...)` | Write filtered CSV, JSON and/or Parquet under the run's artifact directory. |
| `load_run(run_id, *, verify_freshness=True)` | Read a persisted run and reject it if the model or preprocessor digest changed. |

`assess_customer` and `assess_batch` call the same internal
`transform_new -> predict_proba` path. An individual assessment is a
one-record batch; there is no separate single-customer model or rule engine.

### 2.3 Assessment fields

A `RiskAssessment` contains:

- `record_reference` — an opaque `cust_<12 hex>` hash when an identifier was
  supplied, or a positional `risk_input_<6 digits>` reference. Blank, null or
  absent identifiers use positional references (with a warning), so a bad
  identifier column cannot stop scoring;
- `input_position` — zero-based position in the supplied batch;
- `churn_probability` and `churn_probability_pct`;
- `predicted_class` / `predicted_label` — the churn/retained decision at the
  classification threshold;
- `risk_tier` — low/medium/high (or low/medium/high/critical with three
  boundaries);
- `rank` — assigned only after ranking; null for an unranked individual answer;
- `classification_threshold` and the complete `risk_thresholds` object;
- `model_family`, `model_run_id`, `model_sha256`, `preprocessing_run_id`;
- `assessed_at`;
- `observed_churn` — audit label when scoring a persisted split; null for raw
  input because no outcome is known;
- `value_at_risk` and `value_horizon_months` when an explicit revenue proxy was
  supplied, otherwise null;
- `explanation_status`, `explanation_reason`, `explanation_summary`,
  `top_positive_drivers`, `top_negative_drivers`;
- `segment` (only configured fields) and `warnings`.

`predicted_class` and `observed_churn` are deliberately separate. One is the
model decision; the other is a known outcome used only for audit/evaluation.

---

## 3. Thresholds

### 3.1 Tier boundaries

Boundaries are lower-inclusive and upper-exclusive:

| Range | Tier with defaults |
| --- | --- |
| `p < 0.30` | low |
| `0.30 <= p < 0.50` | medium |
| `0.50 <= p < 0.70` | high |
| `p >= 0.70` | critical |

Two boundaries give low/medium/high. Three give low/medium/high/critical.
Configuration is accepted through:

```bash
# environment
CHURNWISE_RISK_TIERS=0.25,0.55,0.75

# CLI (per run)
python -m churnwise.cli score --risk-tiers 0.30,0.50,0.70
```

Invalid values fail with `CW-RISK-001`: fewer than two or more than three
boundaries, non-finite values, values outside `(0, 1)`, duplicates, or
non-increasing boundaries.

Changing tier boundaries never retrains the model and never changes stored
probabilities. A stored run records the boundaries used in
`risk_summary.json`, `run_manifest.json` and every exported row.

### 3.2 Classification threshold is separate

The classification threshold decides `predicted_class`. It comes from Phase 7
and is frozen at the validation-selected value unless explicitly overridden.
Tier boundaries decide only `risk_tier`. A customer can be `medium` risk while
still predicted retained (for example `p = 0.39` with a 0.45 classification
threshold).

The defaults are implementation defaults, **not** business-optimal thresholds.
Choosing thresholds for interventions requires outcome and cost evidence that
this project does not have.

---

## 4. Batch rules and reconciliation

### 4.1 Input contract

`assess_batch` requires a `pandas.DataFrame` containing every column in the
Phase 5 schema's `required_columns`. Extra columns are ignored by the model and
can be retained as segments by name. `score --input` reads CSV values as text so
numeric-looking categorical levels (`0`/`1` plan flags, `408` area codes) keep
the exact string vocabulary fitted in Phase 5.

Numeric conversion, missing-value handling, derived features, one-hot encoding
and scaling are all performed by the fitted Phase 5 pipeline. Phase 9 does not
reimplement them.

### 4.2 Record-level policy

| Condition | Result |
| --- | --- |
| Missing required column(s) | Every row is rejected with `CW-RISK-004`; `batch_error` explains the missing columns. |
| Empty frame | Zero received/scored/rejected, with an explicit empty-batch warning. |
| Unsupported scalar type in a required field | That row is rejected; other rows in the chunk are still scored. |
| Per-row transform/prediction failure | The chunk is retried row-by-row; failed rows are rejected with `CW-RISK-005`; successful rows are kept. |
| Duplicate references | Later references receive `_dup2`, `_dup3`, etc.; a warning is recorded. |
| Blank or missing identifier values | Those rows use positional references; a warning records the count. An all-blank column does not raise. |
| Named reference column is absent | Positional references are used and the missing column is named in a warning. |
| Batch values length mismatch | `CW-RISK-001`; no records are returned. |
| All rows rejected | `score_frame` still persists an empty scored table plus `rejections.json`, reports received/scored/rejected, and the CLI exits 2. |

Rejection messages contain positions, references, error codes and safe
descriptions. They do not include raw customer identifiers or raw values.

`BatchAssessment` guarantees:

```text
total_received == scored + rejected
```

Rows are returned in ascending `input_position`, preserving input/output
correspondence. `batch_size` controls the maximum rows transformed together;
it does not change probabilities by row.

The configured `strict_record_types` flag controls the pre-transform check for
dict/list/set/tuple values. Even with it disabled, the Phase 5 pipeline still
reports failures; they are just classified through the row-retry path.

---

## 5. Ranking, filtering and pagination

`rank_records` assigns unique ranks and uses a documented tie-break chain:

- `probability`: descending probability, then `record_reference`, then
  `input_position`;
- `value_at_risk`: descending expected loss, then descending probability, then
  reference; every row must have a value proxy;
- `risk_tier`: highest tier first, then descending probability, then reference;
- `record_reference`: lexical reference, then input position.

Filtering and sorting a retrieved table or export recompute ranks within that
subset; the run's persisted Parquet keeps the original population ranks. This
makes the first row of a `high`/`critical` export rank 1 while the same
customer may have a larger rank in the full run.

`RiskFilter` supports:

- one or more risk tiers;
- inclusive minimum and maximum probability;
- explicit record references;
- segment equality (`segment_equals`);
- segment numeric minimum/maximum (`segment_min`, `segment_max`).

Filters operate on stored assessments and never call the model. Pagination is
bounded by `max_export_rows` (default 50,000) and rejects negative offsets and
non-integer or oversized limits.

### 5.1 Value at risk is optional and explicit

When a real revenue proxy is supplied:

```text
value_at_risk = churn_probability x proxy_value x value_horizon_months
```

When it is not supplied, every `value_at_risk` is null, the summary records
`"available": false`, and value-at-risk sorting fails rather than inventing a
value. Value-at-risk ordering is an expected-loss heuristic, not a prediction
of a particular customer's revenue loss. Risk ranking is not a revenue ranking.

### 5.2 Probability semantics

The selected Random Forest's probabilities are model scores. Phase 7 measured a
low test Brier score and ROC-AUC, but no calibration map was fitted. A 70%
score is therefore a model estimate, not evidence that 70 of 100 similar
customers will churn, and not a guarantee for the individual customer.

---

## 6. Explainability integration

Phase 9 attaches drivers from Phase 8 local explanations:

- `explain="none"` — no SHAP computation; each row gets
  `explanation_status = "unavailable"` and a reason;
- `explain="top"` — explain only the highest-probability `explain_top_n` rows;
- `explain="all"` — explain every scored row up to `max_driver_rows`
  (default 1,000), then mark excess rows unavailable with a warning;
- `explain_run_id` — reuse an existing Phase 8 matrix run after verifying its
  model digest and preprocessing run.

The top positive and negative source-feature contributors are copied from
`LocalExplanation.top_positive` / `top_negative`. The human-readable summary is
Phase 8's data-grounded summary. Drivers include the transformed feature value
and, for one-hot groups, the active level. Raw one-hot detail remains in Phase 8
artifacts.

If loading or computing explanations fails, the failure is recorded in
`explanation_reason` and a batch warning; probabilities and tiers remain valid.
No explanation is filled with a guessed or default feature.

### 6.1 What explanations mean

A positive SHAP value means the feature pushed the model output toward churn in
that run's documented output space. For the Random Forest and MLP that space is
probability; for Logistic Regression it is log-odds. Phase 9's selected model
is the Random Forest for the reference artifacts, but the contract remains
family-agnostic.

SHAP values are **not causal effects**. The system does not claim that a
feature causes churn or that changing it will retain a customer. Cross-family
SHAP magnitudes are not directly comparable because their output spaces differ.

---

## 7. Persisted artifacts and exports

A scored run is written under:

```text
artifacts/predictions/<run_id>/
├── churn_scores.parquet       # ranked assessments, stable revenue-free fields
├── risk_summary.json          # counts, tiers, model/config provenance
├── tiers.csv                  # tier counts and percentages + total
├── run_manifest.json          # model/preprocessor paths, hashes, config, counts
├── drivers.json               # only available drivers, keyed by opaque reference
├── rejections.json            # only when records were rejected
└── exports/
    ├── churn_scores.csv
    └── churn_scores.json
```

`score_frame` and `score_population` persist by default; `assess_batch` does not
create a run (it may persist its own Phase 8 driver batch). Configured segment
fields are retained in `churn_scores.parquet` so a stored run can still be
filtered by them, but they remain absent from exports unless
`include_segment=True` is explicitly requested. `export` re-reads the stored run
and never rescales, re-tiers or rescores records.

### 7.1 Stable column schema

Both CSV and JSON expose these fields:

```text
record_reference, input_position, churn_probability,
churn_probability_pct, predicted_class, predicted_label, risk_tier, rank,
classification_threshold, model_family, model_run_id, model_sha256,
preprocessing_run_id, assessed_at, positive_label, observed_churn,
value_at_risk, value_horizon_months, explanation_status, explanation_reason,
explanation_summary, top_positive_drivers, top_negative_drivers,
risk_thresholds, warnings, segment
```

Driver fields and `risk_thresholds` are JSON-encoded inside CSV cells; their
structure is documented by the dataclasses in
`churnwise/intelligence/contracts.py`. The JSON export contains typed records
with the same 26 fields (including `segment`, which is an empty object unless
segment export was requested), the complete applied filter
(`tiers`, probability bounds, references, `segment_equals`, `segment_min`,
`segment_max`), the sort field, limit, offset, thresholds, model identity,
warnings and row count. An empty filtered export keeps the same columns and
row count `0` rather than omitting the schema.

### 7.2 Privacy and CSV safety

- Raw identifiers are never a column in exports. A caller-supplied identifier
  is hashed with the dataset id before use.
- `segment` is empty in an export unless segment fields were explicitly
  configured and the caller requests `include_segment=True`; it can contain
  business fields, so export is opt-in. Configured fields are stored in the
  internal scored table only so later filtering works.
- No arbitrary customer columns are exported.
- Every string cell beginning with `=`, `+`, `-`, `@`, tab or carriage return
  literally or after leading whitespace is prefixed with an apostrophe before
  CSV writing. This includes tab-led text and carriage-return-led text, not
  only tab-led formulas. Numeric probability columns are not converted to text.
- Exports are restricted to the resolved predictions artifact directory.
  Absolute paths outside it and `..` traversal are rejected with `CW-RISK-007`.
- Writes are atomic for JSON/CSV; failures produce an actionable
  `CW-RISK-007` rather than a partial file.

Exporting the same run again replaces that run's own `exports/` files. It does
not write into source data, model artifacts or unrelated run directories.

### 7.3 CLI examples

```bash
# Score the validation split and export CSV
python -m churnwise.cli score --population val --explain all --export csv

# Score raw records, explain only the 20 highest-risk rows
python -m churnwise.cli score --input customers.csv --explain top \
  --explain-top-n 20 --export csv --export json

# Filter and export stored scores
python -m churnwise.cli risk-export --run-id <run-id> \
  --tier high --tier critical --min-probability 0.55 \
  --limit 100 --format csv --format json
```

The CLI prints a summary, not raw customer records. `--json` returns structured
counts, paths, warnings and model/config metadata for automation.

---

## 8. Provenance, compatibility and freshness

Every run manifest records at least:

- risk `run_id`, schema version, generated timestamp and duration;
- dataset id and dataset SHA-256;
- split fingerprint and preprocessing run id;
- preprocessor hash and path;
- Phase 7 benchmark run id;
- selected family, model run id, model SHA-256 and model path;
- full risk configuration and its fingerprint;
- the effective value-proxy column when one was used;
- received/scored/rejected counts and tier counts;
- warning and limitation lists;
- artifact paths.

Before scoring, Phase 8's loader verifies the model/preprocessing pair, feature
count and split provenance. Before reading an existing run, `load_run` checks
that the recorded model and preprocessor files still hash to the recorded
values. A changed model or preprocessor is `CW-RISK-003`; the stale run is not
silently reused. Missing files are reported as unavailable with a warning, not
as a fresh score. Edited or incompatible scored tables are also rejected as
`CW-RISK-003`: every documented column must be present and the recorded
`n_rows`, `scored`, `received` and `rejected` counts must reconcile with the
Parquet rows.

`observed_outcome` in the summary distinguishes audit labels from predictions:
raw input has none; persisted splits carry their labels only for audit.

---

## 9. Error codes

| Code | Meaning |
| --- | --- |
| `CW-RISK-001` | Invalid configuration or filter (thresholds, batch size, horizon, sort field, value proxy length). |
| `CW-RISK-002` | No usable selected model or benchmark provenance. |
| `CW-RISK-003` | Risk artifact is unreadable, stale, incompatible with its model/preprocessor, missing a documented column, or has counts that do not reconcile. |
| `CW-RISK-004` | Invalid input (missing schema, non-numeric value proxy, empty/invalid record, unknown reference). |
| `CW-RISK-005` | Model prediction failed for the supplied matrix. |
| `CW-RISK-006` | Phase 8 explanation loading/computation failed. |
| `CW-RISK-007` | Export path or write failure. |

Each error carries a remediation string. Batch-level failures are returned as
structured rejections or summaries; they do not become unhandled tracebacks in
the CLI.

---

## 10. Interface for Phase 10 and the dashboard

Phase 10 and the Phase 11 dashboard should read persisted runs rather than call
SHAP:

```python
from churnwise.intelligence import RiskFilter, RiskService

service = RiskService(Settings.load())

# 1. Individual assessment from a raw record
customer = service.assess_customer(record, explain=False)

# 2. Batch assessment
batch = service.assess_batch(frame, explain="none")
assert batch.counts_reconcile

# 3. Ranked high-risk customers
top = service.ranked(
    run_id=run_id,
    filters=RiskFilter(tiers=("high", "critical")),
    limit=50,
    offset=0,
)

# 4. Filter/sort stored results
page = service.ranked(
    run_id=run_id,
    filters=RiskFilter(min_probability=0.40, segment_equals={"contract": ("monthly",)}),
    sort_by="probability",
    limit=25,
)

# 5. Explanations
explanation = service.get_explanation(run_id, reference)

# 6. Thresholds
service.inspect_thresholds()

# 7. Export a filtered page
service.export(run_id, formats=("csv",), filters=RiskFilter(tiers=("critical",)))

# 8. Summary and warnings
summary = service.get_summary(run_id)

# 9. Provenance
run = service.load_run(run_id)
run.manifest["model_sha256"], run.manifest["preprocessing_run_id"]
```

The dashboard must display the model family/run id and thresholds with the
table, and must not present expected loss as observed loss or SHAP as causality.

---

## 11. Limitations and unverified behaviour

- Default tier boundaries are not proven optimal; business intervention data is
  absent.
- Probabilities are uncalibrated model scores. Brier score is reported in Phase
  7 but no isotonic/Platt calibration was fitted.
- Value-at-risk is exercised only with synthetic proxies; OpenML `46915` has no
  reliable revenue column. No monetary value is invented for it.
- Segment filters require callers to configure and retain the segment fields;
  the reference dataset has no customer identifier or contract/tenure fields.
- Drivers are bounded by `max_driver_rows`; rows beyond the cap are explicitly
  unavailable.
- SHAP runtime and memory scale with row count and explainer family; the real
  Random Forest reference run completed 1,000 rows, but larger populations may
  need a higher cap or an external batch schedule.
- Phase 9 does not implement campaign simulation (Phase 10), the Streamlit
  dashboard (Phase 11), or end-to-end orchestration (Phase 12).
