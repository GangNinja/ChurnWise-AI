# ChurnWise AI — Phase 5 Report

**Scope:** leakage-safe preprocessing and feature engineering on the dataset
approved by Phase 4. No model training, benchmarking, SHAP, scoring, dashboard or
other later-phase work was started.

---

## 1. Executive summary

Phase 5 consumes the real Phase 4 hand-off (`ValidationResult`), splits it
reproducibly, fits every learned transformation on the training partition only,
transforms validation and test with the frozen pipeline, and persists an
auditable run that Phase 6 can load without refitting.

A first-time run needs no user input:

```bash
python -m churnwise.cli discover     # Phase 2
python -m churnwise.cli acquire      # Phase 3
python -m churnwise.cli validate     # Phase 4
python -m churnwise.cli prepare      # Phase 5
```

`prepare --offline` reuses the verified cached artifact. Rejected datasets stop
before any split is created, with the Phase 4 blocking checks in the message.

**Real run on the Phase 4-approved dataset** (`openml:46915`, 5,000 × 20 raw,
SHA-256 `87bc2dd7…85fb6`, 707 positives = 14.14%; every recorded run, including
one regenerated with the final code, shares the same split fingerprint):

| Property | Result |
| --- | --- |
| Run id | `20261021T123405Z-openml_46915-989d8802` |
| Split | stratified, seed 42, **3000 / 1000 / 1000** rows |
| Split fingerprint | `4b0c45db677fa9619de99b6a8b786a3da2c4da93824c46db65412a85829594dd` |
| Class counts (0/1) | train 2576/424, val 858/142, test 859/141 |
| Positive rate | 14.13% / 14.20% / 14.10% |
| Constraint | `test_touched: false`, `fitted_on: train_only` |
| Transformed features | **27** from 16 numeric + 4 categorical + 0 Boolean |
| Derived features | `usage_band`, `support_calls_per_usage_hour` |
| Derived skips | 4, each with a missing-capability reason |
| Excluded columns | `state` (51 categories > 30), `CustomerChurned` (target) |
| Imbalance | 6.07:1, class weights 0.582 / 3.538 |
| Finite transformed values | yes, all three partitions |
| Artifacts | split spec, pipeline, schema, matrices, summary, JSON + MD report |

The source dataset file was not modified, and no customer identifiers are
stored in the transformed matrices.

---

## 2. Implemented pipeline and interfaces

| Module | Responsibility |
| --- | --- |
| `churnwise/preprocessing/contracts.py` | `PreprocessConfig` (fingerprinted), `SplitSpec`, `SplitData`, `ClassBalance` |
| `churnwise/preprocessing/split.py` | Stratified splitting, duplicate/row-identity handling, warnings, fingerprint |
| `churnwise/preprocessing/features.py` | Capability-guarded derived feature catalogue and inference replay |
| `churnwise/preprocessing/pipeline.py` | Training-only column plan, cleaning, `ColumnTransformer`, output mapping |
| `churnwise/preprocessing/coordinator.py` | Orchestration, imbalance, reports, persistence, verified reload |
| `churnwise/preprocessing/dataset.py` | `PreparedDataset`: partitions, names, weights, `transform_new` |
| `churnwise/preprocessing/artifacts.py` | Atomic, digest-verified persistence under `artifact_dir` |
| `churnwise/preprocessing/reporting.py` | JSON + Markdown run reports |
| `churnwise/preprocessing/errors.py` | Stable `CW-PRE-*` codes with remediation |
| `churnwise/cli.py` | `prepare` command, options, exit codes |

Downstream interface (`PreparedDataset`): `train/val/test`, `feature_names`,
`feature_sources`, `class_weights`, `split_spec`, `provenance`, `paths`,
`as_frame`, `resampled_train`, `transform_new`.

`PreprocessingCoordinator.load(run_id)`
([coordinator.py](../churnwise/preprocessing/coordinator.py)) verifies the schema
version and pipeline digest before deserialising, and rejects unsafe run ids or
paths outside `artifact_dir`.

---

## 3. Missing values, encoding, scaling, outliers

| Concern | Implemented rule |
| --- | --- |
| Numeric missing | Median imputation fitted on train; `<column>__missing` indicator when training missingness > 1% |
| Boolean | Most-frequent imputation, then `StandardScaler` |
| Categorical missing | Explicit `"__missing__"` level by default; `most_frequent` is a config alternative and both modes are tested |
| Numeric text | Parsed only when ≥ 98% of the training split parses **and** the name is numeric-like; nominal-looking codes/ZIP/area/state/ID names are never coerced |
| Infinity / non-finite | Converted to missing before imputation; `schema.invalid_numeric` records each affected training column with counts, share, unparseable/infinite split and treatment; transform refuses non-finite output |
| Categorical encoding | One-hot; `handle_unknown="ignore"` by default, `infrequent_if_exist` + `min_frequency` when configured; > 30 levels dropped with a reason |
| Scaling | `StandardScaler`, fitted on train only |
| Constant / near-constant | Phase 4 drops globally constant columns with a reason; a column that is constant only within the training split is retained and scaled to finite zeros (no invalid values) |
| Outliers | IQR diagnosis on the training split is reported per numeric column with `treated: false`; no rows are deleted. `outlier_policy` is `report_only` only |

Derived features are only created when the capability and its source column are
evidenced. A candidate whose source deterministically reproduces the label is
rejected as a renamed target. Every skip is persisted with its reason.

---

## 4. Split configuration and leakage safeguards

- Stratified 60/20/20 by default, `seed=42`, configurable fractions that must sum
  to 1; proportions are stratified per class.
- `SplitSpec` records seed, ratios, sizes, class counts/rates, row indices,
  dataset digests, duplicate/target-missing drops and the fingerprint.
- Exact duplicate rows are dropped before splitting by default (count recorded).
  When the Phase 4 id column repeats, whole customer groups are split and the
  configured fractions still drive partition sizes; the run fails rather than
  placing one customer in two partitions.
- Too few rows (< 100) or classes (< 2 per class) raise `CW-PRE-002`; severe
  imbalance still splits but warns.
- The column plan, imputers, encoder, scaler, missing-indicator thresholds and
  derived specs are all produced **after** the split from training rows only.
- Test rows are never used for fitting, thresholding, feature selection or model
  selection; Phase 5 creates no model.

Leakage checklist evidence (`tests/unit/test_preprocessing_leakage.py`):

| ID | Assertion |
| --- | --- |
| L1 | Identifier absent from planned features and output names |
| L2 | Target and Phase 4 leakage-flagged columns excluded |
| L3/L4 | Split exists before fitting; partitions disjoint and complete |
| L5 | No resampling by default; training row count unchanged |
| L6/L7/L8 | No threshold, `best_params_`, or feature-selection step exists |
| T5.2 | Injecting 1e9 / −1e9 / unseen categories into val/test leaves scaler means and encoder categories byte-identical |
| Train-only | Scaler mean and imputer median equal the training statistics exactly; indicators depend only on training missingness |

---

## 5. Class imbalance

- Counts, rates and imbalance ratio are computed for the full data and each
  split (`ClassBalance`), and persisted in the schema, summary and reports.
- Default strategy is class weighting with `balanced` weights (0 = 0.582,
  1 = 3.538 on the real run). Weights are stored for Phase 6.
- `oversample` duplicates minority **training** rows after splitting, using the
  seeded RNG; indices are persisted and `resampled_train()` exposes the view.
  Validation and test row counts never change (asserted end-to-end).
- `none` records the untouched distribution.
- SMOTE is **not** implemented. The Phase 1 architecture text and the Phase 5
  roadmap bullet still plan a fold-internal SMOTE option; the delivered code
  offers seeded random duplication of training rows instead. This is a
  deliberate, documented gap (T5.6 is asserted against `oversample`), not a
  claim that SMOTE exists.

---

## 6. Artifact formats and serialization

| Artifact | Format | Notes |
| --- | --- | --- |
| `artifacts/splits/<run_id>.json` | JSON, atomic write | `SplitSpec` including row indices and fingerprint |
| `artifacts/preprocessors/<run_id>/pipeline.joblib` | joblib | Written only by this project; digest recorded in the schema |
| `artifacts/preprocessors/<run_id>/schema.json` | JSON | Config, roles, target, categories, mappings, `invalid_numeric`, outliers, provenance, warnings |
| `artifacts/prepared/<run_id>/matrices.npz` | `numpy.savez_compressed`, `allow_pickle=False` on load | float64 features, int8 labels, row and oversample indices |
| `artifacts/prepared/<run_id>/summary.json` | JSON | `PreparedDataset.to_dict()` |
| `reports/preprocessing/<run_id>.{json,md}` | JSON + Markdown | Splits, exclusions, derived features, invalid numeric values, outliers, warnings, provenance |

Safety: only pipeline-written artifacts are loaded; run ids must match a strict
pattern without `..`; resolved paths must stay under `artifact_dir`; the schema
version must match `1.0`; the pipeline SHA-256 must equal
`schema.preprocessor_hash`; matrices are loaded without pickle. No identifiers
beyond the optional id column already present in the source are persisted, and
customers are never written to logs.

---

## 7. Integration instructions for Phase 6

```python
from churnwise.config import Settings
from churnwise.preprocessing import PreprocessingCoordinator

dataset = PreprocessingCoordinator(Settings()).load(run_id)

X_train, y_train = dataset.train.X, dataset.train.y
X_val, y_val = dataset.val.X, dataset.val.y
weights = dataset.class_weights            # {0: 0.582..., 1: 3.537...}
assert len(dataset.feature_names) == X_train.shape[1]

new_matrix = dataset.transform_new(customer_frame)   # raw-valued records
```

Phase 6 must assert `dataset.schema["preprocessor_hash"]` equals the hash it
records in model metadata, train only on `dataset.train` (or
`resampled_train()` when `imbalance="oversample"`), use `dataset.val` for early
stopping, and touch `dataset.test` only in Phase 7.

---

## 8. Tests run (actual results)

Executed from the repository root with the project virtualenv:

| Command | Result |
| --- | --- |
| `python -m pytest tests/unit/test_split_determinism.py -q` | 9 passed |
| `python -m pytest tests/unit/test_preprocessing_columns.py -q` | 16 passed |
| `python -m pytest tests/unit/test_preprocessing_leakage.py -q` | 8 passed |
| `python -m pytest tests/integration/test_preprocessing_pipeline.py -q` | 12 passed |
| `python -m pytest tests/test_preprocessing_cli.py -q` | 5 passed |
| `python -m pytest -q` | **282 passed, 7 skipped** in 35.5 s |
| `python -m churnwise.cli prepare --offline` | final run `20261021T123405Z-…`, same fingerprint, element-identical matrices to the first run, `invalid_numeric: {}` |
| `python scripts/verify_phase1_docs.py` | 114 checks passed, 0 failed |
| `python -m pytest tests/test_live_preprocessing.py -q --run-live` | **failed before Phase 5 code ran** |
| Manual load of the final real run (`PreprocessingCoordinator.load`) | returns (3000, 27) / (1000, 27) / (1000, 27), 27 names, weights {0: 0.582, 1: 3.538}, one raw record → (1, 27) finite |

The live failure is in Phase 2 discovery: no candidate satisfied the blocking
gates (`CW-DISC-003`, metadata gaps `G4`/`G5`). The pre-existing
`tests/test_live_validation.py` fails identically on the same run, so this is the
current state of the upstream public-source metadata, not a Phase 5 regression.
Phase 5 live coverage therefore remains **unverified**; the real-data evidence
above was produced offline from the previously verified Phase 4 artifact with
`prepare --offline`.

Phase 5 test coverage includes numeric/categorical/Boolean/mixed inputs, missing
and non-finite values, unseen inference categories, constant and near-constant
columns, high cardinality, target/id exclusion, stratified reproducible splits,
small and severely imbalanced datasets, train-only fitting, exact duplicate rows,
grouped splits that honour custom fractions, invalid/infinite numeric reporting,
imbalance reporting, oversampling scope, feature-name mapping, serialization
round-trip, missing-digest rejection, `transform_new`, provenance and error
handling, and the actual Phase 4 output contract.

---

## 9. Acceptance and remaining gaps

| ID | Evidence |
| --- | --- |
| T5.1 | Two `prepare` runs on the same input produce the same fingerprint and element-identical matrices (`test_run_is_byte_identical…`, real runs above) |
| T5.2 | Extreme values injected into val/test leave scaler means and imputer medians unchanged |
| T5.3 | Unseen category at transform time is ignored; output finite |
| T5.4 | Median/most-frequent/`__missing__`/indicator rules tested |
| T5.5 | `feature_names` and `schema.json` stable across repeated fits/runs |
| T5.6 | Training-only random oversampling adds only training rows and leaves val/test sizes unchanged; the SMOTE variant named in the acceptance table is **not implemented** (documented gap) |
| T5.7 | L1–L8 each have an executable assertion |

Completion criteria: splits are 3000/1000/1000 with class rates 14.13/14.20/14.10
(max deviation 0.07 pp), `test_touched: false`, the reloaded pipeline reproduces
identical training/held-out matrices and `transform_new` output, `schema.json` records input
columns, output names, `output_dtype` and per-column `categories`, derived
features appear only with evidenced sources (4 skips recorded), imbalance is
recorded in config and schema, and no fitted statistic uses val/test.

Known limitations, stated rather than hidden:

- **Live end-to-end verification is blocked upstream** (see §8). A live run should
  be repeated when discovery metadata reports class balance and feature counts.
- **SMOTE is not implemented.** Oversampling is random duplication of training
  rows, not synthetic minority generation, and is not applied inside CV folds;
  Phase 7 may add fold-internal resampling as a separate documented run.
- **Outlier treatment is diagnostic only.** No capping or winsorising is offered;
  `outlier_policy` accepts `report_only` alone.
- **Derived features are catalogue-bound.** On this dataset there is no tenure or
  total-charges column, so four catalogue features are skipped; richer datasets
  will build more.
- **Renamed-target detection is heuristic.** A derived source is rejected when
  its observed values map deterministically to the label and it has at most 20
  distinct values; continuous or subtle leaks may pass. Phase 4's own leakage
  flags remain the primary defence, because capability sources are drawn only
  from its non-dropped roles.
- **`most_frequent` categorical imputation** has no explicit missing level by
  design (the most frequent category absorbs missing values).
- **Excel artifacts** remain untested end-to-end, inherited from Phase 4.
- **No encoding of the Phase 4-dropped `state` column**; geography is lost until
  a documented frequency/target-safe encoder is added in a later phase.
