# Preprocessing and feature engineering (Phase 5)

Phase 5 turns the dataset approved by Phase 4 into model-ready matrices. It is
autonomous: `prepare` validates/acquirer when needed, so a first run needs only

```bash
python -m churnwise.cli prepare
```

`prepare --offline` reuses the verified cached artifact and never touches the
network. The command exits `0` on success, `2` when Phase 4 rejected the dataset
or a preprocessing error carries a stable code, and `3` for invalid options.

---

## 1. Pipeline

| Step | Behaviour |
| --- | --- |
| Target normalisation | Uses Phase 4's `TargetInterpretation`; `1 = churned`, missing target rows are excluded in memory and counted |
| Derived features | Capability-guarded catalogue, recomputed at inference from persisted specs |
| Duplicate rows | Exact duplicates dropped before splitting (configurable) and counted |
| Split | Stratified 60/20/20 by default, seeded, fingerprinted; when a customer id repeats, whole groups are split and the configured fractions still drive the sizes |
| Column plan | Built from the **training partition only**: numeric, Boolean, categorical, numeric-text coercions, missing indicators, high-cardinality drops |
| Cleaning | Infinities → NaN, numeric text parsed, strings trimmed, missing indicator columns added |
| Fitting | Median imputer + `StandardScaler`, most-frequent Boolean imputer + scaler, constant `"__missing__"` imputer + one-hot encoder — all fitted on train only |
| Transform | Validation and test are transformed with the frozen fitted pipeline |
| Imbalance | Class weights by default; optional training-only random oversampling |
| Persistence | Split spec, fitted pipeline, schema, matrices, summary, JSON + Markdown reports |

Data flow:

```text
ValidationResult (Phase 4)
  └─ normalized_frame() ──> derived features ──> drop duplicates
        └─ make_split(train/val/test) ──> plan_features(train) ──> clean ──> fit(train)
              ├─ X_train / y_train
              ├─ X_val   / y_val
              └─ X_test  / y_test   (never used for fitting or selection)
```

---

## 2. Configuration (`PreprocessConfig`)

| Field | Default | Meaning |
| --- | --- | --- |
| `seed` | 42 | Split seed and oversampling seed |
| `train_fraction`, `val_fraction`, `test_fraction` | 0.60 / 0.20 / 0.20 | Must sum to 1 |
| `imbalance` | `class_weight` | `class_weight`, `oversample` or `none` |
| `oversample_ratio` | 1.0 | Target minority/majority ratio for `oversample` |
| `numeric_imputation` | `median` | Only median is implemented |
| `categorical_imputation` | `missing_category` | `most_frequent` fills with the training mode instead; the `__missing__` level exists only in `missing_category` mode |
| `scaling` | `standard` | `StandardScaler` |
| `missing_indicator_threshold` | 0.01 | Add `<column>__missing` above this training missing share |
| `max_categorical_levels` | 30 | Above this, the column is dropped with a reason |
| `min_frequency` | 0.0 | Optional rare-category grouping (`infrequent_if_exist`) |
| `derived_features` | true | Capability-guarded derived features |
| `drop_duplicate_rows` | true | Drop exact duplicates before splitting |
| `min_rows` | 100 | Minimum row count for a split |
| `min_class_count` | 2 | Minimum rows per class |
| `numeric_coerce_threshold` | 0.98 | Share of parseable values required to treat numeric text as numeric |
| `outlier_policy` | `report_only` | Outliers are reported, never deleted |

Every field participates in `PreprocessConfig.fingerprint()`, which is embedded
in the run id.

---

## 3. Derived features

The catalogue is static; each entry needs an evidenced Phase 4 capability and a
source column.

| Feature | Required capabilities | Formula |
| --- | --- | --- |
| `tenure_years` | `has_tenure` | tenure months / 12 |
| `tenure_band` | `has_tenure` | 0-6, 6-12, 12-24, 24-48, 48+ months |
| `avg_monthly_charge` | `has_total_charges`, `has_tenure` | total charges / tenure months (≥ 1) |
| `support_calls_per_year` | `has_support_calls`, `has_tenure` | support calls / tenure years (≥ 1) |
| `usage_band` | `has_usage_minutes` | 0-100, 100-200, 200-300, 300+ minutes |
| `support_calls_per_usage_hour` | `has_support_calls`, `has_usage_minutes` | support calls / usage hours |

Skips are recorded in `schema.derived_skipped` with a reason such as
`Phase 4 did not evidence the required capability: has_tenure`. A candidate is
also rejected when its source column deterministically reproduces the target
(up to 20 distinct values), which catches a renamed label column.

`apply_derived` replays the persisted formulas at inference, so no refitting is
needed and the transformed columns keep their training-time meaning.

---

## 4. Artifacts

| Path | Contents |
| --- | --- |
| `artifacts/splits/<run_id>.json` | `SplitSpec`: sizes, class counts/rates, row indices, fingerprint, constraints, seed, dataset digests |
| `artifacts/preprocessors/<run_id>/pipeline.joblib` | Fitted `ColumnTransformer` (trusted, digest-verified on load) |
| `artifacts/preprocessors/<run_id>/schema.json` | Config, roles, categories, mappings, target interpretation, provenance |
| `artifacts/prepared/<run_id>/matrices.npz` | `X_train, y_train, X_val, y_val, X_test, y_test`, row indices, oversample indices (no object arrays) |
| `artifacts/prepared/<run_id>/summary.json` | `PreparedDataset.to_dict()` |
| `reports/preprocessing/<run_id>.json` | Machine-readable report |
| `reports/preprocessing/<run_id>.md` | Human-readable report |

Key `schema.json` fields:

| Key | Meaning |
| --- | --- |
| `schema_version`, `pipeline_version`, `phase` | Contract versions |
| `run_id`, `dataset_id`, `dataset_sha256`, `manifest_sha256` | Run identity and provenance |
| `config`, `seed`, `split_fingerprint` | Reproducibility inputs |
| `target` | Column, labels, mapping, positive rate |
| `required_columns` | Every raw column inference must supply |
| `preprocessing` | Numeric/Boolean/categorical plan, coercions, indicators, imputation/scaling/encoding mode, `categories`, `output_dtype` |
| `derived`, `derived_skipped` | Applied derived features and every recorded skip |
| `feature_names`, `output_features` | Transformed column order and source/role/value mapping |
| `excluded` | Dropped columns with stage and reason |
| `invalid_numeric` | Training-split count/share of unparseable or non-finite numeric values and their treatment |
| `outliers` | Training-split IQR diagnostics (`treated: false`) |
| `class_balance`, `class_weights` | Distribution and downstream weights |
| `warnings` | Actionable warnings |
| `provenance` | Dataset digests, validation report path, source metadata, code version |

---

## 5. Phase 6 interface

```python
from churnwise.preprocessing import PreprocessingCoordinator, prepare_dataset

dataset = PreprocessingCoordinator(settings).prepare(validation_result)  # fit + persist
dataset = PreprocessingCoordinator(settings).load(run_id)                # verify + reload

dataset.train.X, dataset.train.y          # float64 matrix, int8 labels
dataset.val, dataset.test                 # same shape contract
dataset.feature_names                     # stable order
dataset.feature_sources                   # transformed -> source column
dataset.class_weights                     # {0: ..., 1: ...}
dataset.split_spec                        # fingerprint, sizes, constraints
dataset.provenance                        # dataset digests + Phase 4 report path
dataset.transform_new(customer_frame)     # frozen pipeline for new records
dataset.resampled_train()                 # training-only oversampling view
```

`transform_new` requires every column in `schema.required_columns`. It applies
the persisted derived formulas, cleaning, and the fitted preprocessor. Unseen
categories are ignored by the encoder; numeric text that fails to parse becomes
missing and is imputed with training statistics.

Loads are restricted to artifacts written by this project: run ids are
validated against a safe pattern, paths must stay under `artifact_dir`, the schema
version must match, and the pipeline digest must match `schema.preprocessor_hash`.
`joblib` is never used on downloaded content.

---

## 6. Error codes

| Code | Raised when |
| --- | --- |
| `CW-PRE-001` | Phase 4 did not approve the dataset, or its result is incomplete |
| `CW-PRE-002` | The requested split cannot be produced (too few rows/classes) |
| `CW-PRE-003` | Artifact missing, unsafe run id, version mismatch or digest mismatch |
| `CW-PRE-004` | Transform failed: missing columns, wrong width, non-finite output |

---

## 7. Test coverage

| Command | Scope |
| --- | --- |
| `pytest tests/unit/test_split_determinism.py` | seed determinism, stratification, small/imbalanced data, fingerprint identity, no row overlap |
| `pytest tests/unit/test_preprocessing_columns.py` | coercion, imputation, indicators, unseen categories, constants, high cardinality, mapping, derived features |
| `pytest tests/unit/test_preprocessing_leakage.py` | leakage checklist L1–L8 and train-only fitting |
| `pytest tests/integration/test_preprocessing_pipeline.py` | real Phase 4 hand-off → persist → reload → `transform_new`, oversampling scope, provenance |
| `pytest tests/test_preprocessing_cli.py` | CLI wiring, exit codes, JSON/summary output |
| `pytest tests/test_live_preprocessing.py --run-live` | live discovery → acquisition → validation → preprocessing |
