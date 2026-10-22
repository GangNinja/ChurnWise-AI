# ChurnWise AI — Phase 8 Report: Explainable AI and SHAP

| Field | Value |
| --- | --- |
| Status | **Complete and verified** for the implemented scope |
| Phase | 8 — Explainable AI and SHAP Integration |
| Date | 2026-10-22 |
| Dataset | OpenML `46915` (`openml:46915`), 5,000 rows, 27 transformed features, binary target `CustomerChurned` (`Yes` = 1) |
| Preprocessing run | `20261021T123405Z-openml_46915-989d8802` |
| Phase 7 benchmark | `20261021T145336Z-openml_46915-bench-04595d7b` |
| Interface documentation | [EXPLAINABILITY.md](EXPLAINABILITY.md) |
| Tests | **42 focused passed**; full suite **432 passed, 7 skipped** |
| Real-model explanations | **Yes** — three genuine CLI runs on the saved Phase 6/7 artifacts (below) |

This report separates three kinds of evidence:

1. **Real-model evidence** — SHAP runs against the persisted OpenML model and
   preprocessing artifacts.
2. **Synthetic test evidence** — deterministic fixtures that exercise the
   interface, not claims about churn data.
3. **Unverified scope** — items deliberately left to later phases or not
   exercised in this environment.

---

## 1. Summary

Phase 8 adds a modular SHAP subsystem under `churnwise/explain/` with:

- family-specific explainers for Logistic Regression, Random Forest and the Keras
  MLP, including a bounded permutation fallback for the MLP;
- explicit `probability` vs `log_odds` output spaces, with additivity validation
  before any value is reported;
- one-hot grouping back to Phase 5 source features, preserving raw column detail;
- global rankings, SHAP beeswarm, importance and dependence figures;
- single-customer and batch raw-record explanations with signed top-k drivers and
  data-grounded summaries;
- provenance tied to the dataset hash, split fingerprint, preprocessing run,
  benchmark run and model digest;
- persistence under `artifacts/explanations/<run_id>/` and a CLI `explain`
  command.

Three real explanation runs completed successfully: Random Forest (12.93 s),
Logistic Regression (2.52 s), and MLP (7.19 s). All used validation rows only;
no test row was sampled or explained (`split="test"` is rejected). A separate real raw-record run
(Random Forest, two rows from the source dataset) exercised the
`transform_new → predict_proba → SHAP` path and completed in the normal CLI flow.

No model was retrained. No SHAP value, chart, metric or test result in this
report was fabricated or copied from a template.

---

## 2. Upstream artifacts inspected

| Artifact | Verified content | Used for |
| --- | --- | --- |
| `artifacts/preprocessors/20261021T123405Z-openml_46915-989d8802/pipeline.joblib` | Fitted Phase 5 `ColumnTransformer` | Exact transform for raw records and matrix consistency |
| `.../schema.json` | 27 `feature_names`, 20 source features, target mapping, `output_features` with `name/source/role/value`, split fingerprint | Feature mapping, thresholds, provenance |
| `artifacts/evaluation/20261021T145336Z-openml_46915-bench-04595d7b/metrics.json` | Three completed families, thresholds, feature counts, selected family | Default family, thresholds, benchmark provenance |
| `artifacts/evaluation/.../models/random_forest/model.joblib` + `metadata.json` | SHA-256 verified `FittedBaseline` | Tree SHAP |
| `artifacts/evaluation/.../models/logistic_regression/model.joblib` + `metadata.json` | SHA-256 verified `FittedBaseline` | Linear SHAP |
| `artifacts/models/20261021T132609Z-openml_46915-mlp-443bb6a1/mlp/model.keras` | Digest-verified Phase 6 Keras model | Deep SHAP |
| `artifacts/splits/...` / Phase 5 arrays | Train/validation/test matrices; row indices | Validation sample and training-only background |

Every available family was explainable with its native explainer; no family was
skipped for lack of a compatible artifact. The MLP module's `feature_count`
matched the preprocessing schema, and the coordinator rejected no interface as
incompatible.

---

## 3. Implementation

| Module | Responsibility |
| --- | --- |
| `churnwise/explain/contracts.py` | `ExplainConfig`, `FeatureGroup`, `RawExplanation`, `Contribution`, `GlobalExplanation`, `LocalExplanation`, `ExplainResult`, schema version, output-space constants |
| `churnwise/explain/explainers.py` | `build_explainer`, `ModelExplainer`, positive-class extraction, additivity validation, Deep→permutation fallback |
| `churnwise/explain/features.py` | Source grouping, column-order labels, grouped sums, ranked importance, signed contributions |
| `churnwise/explain/engine.py` | Deterministic sampling, chunked explanation, global summary, local summaries, position selection |
| `churnwise/explain/coordinator.py` | Model/preprocessing loading, run orchestration, artifact persistence, manifests, raw-record batches |
| `churnwise/explain/plots.py` | Bar chart, SHAP beeswarm, dependence scatter, SHAP waterfall (Agg backend) |
| `churnwise/explain/reporting.py` | Evidence-only CLI summary |
| `churnwise/explain/errors.py` | Stable `CW-XAI-*` errors with remediation |
| `churnwise/cli.py` | `explain` command with family/model/run selection, sampling bounds, raw `--input`, JSON output |

The already-declared `shap>=0.46,<0.55` requirement is used; the installed
version is 0.54.1. No new dependency was required. The model and preprocessing
artifacts are opened read-only.

---

## 4. Explainer selection and output spaces

| Family | Explainer | Output space | Background | Real-run additivity error |
| --- | --- | --- | --- | --- |
| Random Forest | `shap.TreeExplainer` (`tree_path_dependent`) | `probability` | 100 train rows recorded | `8.77e-15` |
| Logistic Regression | `shap.LinearExplainer` | `log_odds` | 100 train rows | `1.78e-15` |
| MLP | `shap.DeepExplainer` | `probability` | 50 train rows | `1.05e-07` |

- Tree SHAP and Linear SHAP are exact for their model classes.
- Deep SHAP is approximate; the measured additivity error is far below the
  default `1e-4` tolerance.
- If DeepExplainer fails to initialise or fails a one-row probe, the MLP uses
  `shap.Explainer(..., algorithm="permutation")` with a hard
  `permutation_max_evals` budget, and the fallback reason is written to the
  manifest. The fallback is tested with a forced failure but was **not** needed
  for the real MLP.
- No unsupported family or failed explainer returns placeholder values. Missing
  SHAP, unsupported families, shape errors, non-finite SHAP/base/prediction
  values, probabilities outside `[0, 1]`, prediction exceptions and additivity
  failures all raise `CW-XAI-*`.
- The coordinator verifies the model metadata's dataset id/hash, split
  fingerprint, preprocessor hash and feature-name order against the loaded Phase
  5 run before explaining. A mismatch is `CW-XAI-006`, not a silent explanation.
- Thresholds are validated as finite values in `[0, 1]`; a zero threshold is
  preserved rather than replaced by a truthiness fallback. Invalid thresholds
  fail with an actionable provenance error.

**Output-space convention.** Probability-space values sum to `P(churn = 1)`.
Logistic Regression values sum to the decision function in log-odds; a positive
value pushes toward churn in log-odds, not by that many probability points.
`LocalExplanation.probability` is always a probability, and `model_output` /
`base_value` are in the documented SHAP space. Cross-family mean absolute values
are not comparable.

---

## 5. Global explanations (real models)

Validation partition: 1,000 rows. Sampling: seeded uniform without replacement,
seed 42. Test rows used: **0** in all runs.

| Family | Run ID | Sample | Background | Base value | Duration |
| --- | --- | --- | --- | --- | --- |
| Random Forest | `20261022T131420-openml_46915-random_forest-xai-84eecf5d` | 200 / 1,000 | 100 | 0.5005 (probability) | 12.93 s |
| Logistic Regression | `20261022T131432-openml_46915-logistic_regression-xai-c160a8be` | 200 / 1,000 | 100 | −0.6633 (log-odds) | 2.52 s |
| MLP | `20261022T131446-openml_46915-mlp-xai-4c391837` | 60 / 1,000 | 50 | 0.3007 (probability) | 7.19 s |

The base values are explainer outputs in their documented spaces, not the 14.14%
churn prevalence. Class weighting and the explainer's expected-value convention
shift the baseline; this does not affect additivity or the signed local
contributions, but it matters when interpreting a raw base value.

### 5.1 Top source features by mean |SHAP|

**Random Forest (probability space)**

| Rank | Feature | mean \|SHAP\| | Share | Majority sign |
| --- | --- | --- | --- | --- |
| 1 | `international_plan` | 0.0806 | 14.0% | decreases_churn |
| 2 | `total_day_charge` | 0.0693 | 12.0% | decreases_churn |
| 3 | `total_day_minutes` | 0.0656 | 11.4% | decreases_churn |
| 4 | `number_customer_service_calls` | 0.0562 | 9.7% | decreases_churn |
| 5 | `support_calls_per_usage_hour` | 0.0446 | 7.7% | decreases_churn |

**Logistic Regression (log-odds space)**

| Rank | Feature | mean \|SHAP\| | Share | Majority sign |
| --- | --- | --- | --- | --- |
| 1 | `support_calls_per_usage_hour` | 1.7314 | 25.3% | decreases_churn |
| 2 | `number_customer_service_calls` | 0.9723 | 14.2% | increases_churn |
| 3 | `total_day_minutes` | 0.6556 | 9.6% | increases_churn |
| 4 | `total_day_charge` | 0.6547 | 9.6% | increases_churn |
| 5 | `voice_mail_plan` | 0.6472 | 9.5% | increases_churn |

**MLP (probability space)**

| Rank | Feature | mean \|SHAP\| | Share | Majority sign |
| --- | --- | --- | --- | --- |
| 1 | `support_calls_per_usage_hour` | 0.0670 | 14.8% | decreases_churn |
| 2 | `total_day_minutes` | 0.0654 | 14.4% | decreases_churn |
| 3 | `international_plan` | 0.0483 | 10.6% | decreases_churn |
| 4 | `total_day_charge` | 0.0397 | 8.7% | decreases_churn |
| 5 | `number_customer_service_calls` | 0.0281 | 6.2% | decreases_churn |

These rankings are computed from the actual SHAP matrices by
`mean(abs(grouped SHAP))`; `share` is the feature's fraction of the summed mean
absolute values. They are not feature-importance scores copied from a model
`feature_importances_` attribute.

### 5.2 Interpreting the majority sign correctly

`direction` in `global.json` is the sign followed by more than half of the
sampled rows (`positive_share > 0.5` / `< 0.5`), **not** a monotone effect.
Because the Random Forest's base value is 0.5005 and most rows sit below that
base, almost every feature's majority sign is `decreases_churn`. This does not
mean those features reduce risk for all customers. The local results show the
opposite for high-risk rows: for Random Forest row 149, `total_day_charge`
contributes **+0.231** and `total_day_minutes` **+0.194**, while for row 87 the
same features contribute **−0.058** and **−0.056**. Use local signed
contributions and the dependence plot for direction by value. The same caution
applies to the global field in all families.

### 5.3 Strongest observed contributors

- **Random Forest:** `international_plan`, `total_day_charge` and
  `total_day_minutes` dominate. The day-charge and day-minute features are
  strongly correlated (charge is derived from minutes), so their importance is
  split between two columns and neither should be read as an independent causal
  driver.
- **Logistic Regression:** `support_calls_per_usage_hour` is the largest
  contributor by a wide margin (25.3% of total mean absolute log-odds), followed
  by raw service-call count. This derived rate feature dominates this linear
  model much more than it does the tree/MLP models.
- **MLP:** the ranking is flatter. `support_calls_per_usage_hour` and
  `total_day_minutes` are close, with `international_plan` third. Probability-space
  values are smaller than the log-odds values and are not comparable to them.

---

## 6. Individual explanations (real models)

The persisted local examples are selected as the highest and lowest predicted
risk rows **within the explained sample**. `row_position` is a split position,
not a customer identifier. The original raw records are not stored in the
artifacts.

| Family | Row | P(churn) | Class @ threshold | Largest upward contributions | Largest downward contributions |
| --- | --- | --- | --- | --- | --- |
| RF | 149 | 0.976 | 1 (`Yes`) @ 0.45 | `total_day_charge` +0.231; `total_day_minutes` +0.194; `usage_band=300+` +0.108 | `international_plan=0` −0.022; `total_intl_calls` −0.018 |
| RF | 87 | 0.000 | 0 (`No`) @ 0.45 | `voice_mail_plan=0` +0.004 | `number_customer_service_calls` −0.065; `total_day_charge` −0.058 |
| LR | 189 | 0.986 | 1 (`Yes`) @ 0.64 | `support_calls_per_usage_hour` +13.708; `voice_mail_plan=0` +0.463; `total_intl_calls` +0.233 (log-odds) | `total_day_minutes` −2.889; `total_day_charge` −2.885; `number_customer_service_calls` −1.818 |
| LR | 117 | 0.001 | 0 (`No`) @ 0.64 | `number_customer_service_calls` +1.533; `number_vmail_messages` +0.681 (log-odds) | `support_calls_per_usage_hour` −2.698; `total_day_minutes` −1.924 |
| MLP | 122 | 0.992 | 1 (`Yes`) @ 0.73 | `international_plan=1` +0.264; `total_day_minutes` +0.140; `total_eve_minutes` +0.098 | `total_intl_minutes` −0.044; `number_customer_service_calls` −0.024 |
| MLP | 430 | 0.042 | 0 (`No`) @ 0.73 | `account_length` +0.017; `total_eve_charge` +0.007 | `support_calls_per_usage_hour` −0.095; `number_customer_service_calls` −0.039 |

For LR row 189 the saved summary reads, in part:

> The model predicts model log-odds 4.283 (P(churn = 1) = 0.986) at threshold
> 0.64; predicted class 1 (Yes). Largest upward contributions:
> support_calls_per_usage_hour (transformed value 6.159) +13.708,
> voice_mail_plan (voice_mail_plan=0) +0.463, total_intl_calls (transformed value
> −1.409) +0.233. […] These are SHAP attributions of the model output, not
> causal effects; changing a feature is not guaranteed to change churn by the
> shown amount.

Summaries are generated from measured values only. They state the model output,
threshold, class, top signed contributions and the non-causal disclaimer. No
customer identifier is included.

### 6.1 Real raw-record interface

A two-row CSV containing only the required Phase 5 input columns (no customer ID,
no target) was explained with the real Random Forest artifact:

```bash
python -m churnwise.cli explain --input <temporary-two-row.csv> \
  --family random_forest \
  --explain-run-id 20261022T132000-openml_46915-random_forest-xai-custom-demo
```

Observed results: row 0 P(churn) = 0.221, class 0; row 1 P(churn) = 0.546,
class 1 at the 0.45 threshold; local JSON and waterfall files were written. The
temporary input CSV was removed after the run, and the artifacts contain only
transformed contributions, not the raw input records. This confirms that the
real fitted Phase 5 pipeline, including derived features and categorical
encoding, drives the custom-record path.

---

## 7. Feature mapping and interpretation

- Phase 5's `output_features` maps 27 transformed columns to 20 source features.
- One-hot groups in the reference run: `area_code` (3 columns),
  `international_plan` (2), `usage_band` (4), `voice_mail_plan` (2).
- Contributions within a source are summed; local JSON retains the per-column
  name, transformed value and contribution under `columns`. The sum is signed,
  so opposite-sign one-hot levels can partly cancel in the grouped ranking; raw
  column detail is retained rather than discarded.
- A fallback strips `numeric__` / `categorical__` / `boolean__` only when the
  mapping is absent; an unmapped-column warning is recorded. On the reference run
  every column had a Phase 5 mapping, so no fallback label was used.
- Native plot labels are assembled in model column order. A regression test
  (`test_mapped_columns_preserve_model_column_order`) covers the earlier bug where
  alphabetical group order could mislabel beeswarm/waterfall axes.
- `Contribution.value` is the transformed numeric value; `active_level` is the
  original one-hot category label. Original numeric units are not reconstructed.
  This limitation is explicit in the interface documentation.

---

## 8. Sampling, performance and safeguards

| Safeguard | Default | Evidence |
| --- | --- | --- |
| Global sample | 200 LR/RF rows; 60 MLP rows | Real runs used 200/200/60 |
| Background | 100 LR/RF rows; 50 MLP rows | Drawn only from training rows |
| Reproducibility | seed 42, sorted positions | Manifest stores exact positions |
| Chunking | 100 rows per SHAP call | Values identical to a single call (test) |
| Local examples | 5 | Selected from high and low predicted risk |
| Raw batch guard | 500 rows | Oversized frame rejected before loading a model |
| Permutation fallback | 500 evaluations | Synthetic forced-failure test |
| Additivity tolerance | 1e-4 absolute | All real errors ≤ 1.05e-7 |
| Test isolation | `split="val"` or `"train"` only | `test_rows_used = 0`; coordinator rejects `test` |
| Explainer reuse | one per run | No cross-process cache; documented |

Runtimes above include model loading, SHAP computation, all figures and JSON
writes on the reference Windows CPU environment (Python 3.13.0, SHAP 0.54.1,
NumPy 2.5.4, scikit-learn 1.7.2, TensorFlow 2.21.0). They are comfortably within
the 5-minute acceptance budget; they are measurements, not guarantees.

---

## 9. Visualization artifacts

For each real global run:

```text
artifacts/explanations/<run_id>/
├── global.json
├── importance.png
├── summary.png
├── dependence_total_day_charge.png       # RF
├── dependence_support_calls_per_usage_hour.png  # LR, MLP
├── local/row_<position>.json
├── local/row_<position>.png
└── run_manifest.json
```

- `importance.png` is drawn from the computed source-feature ranking.
- `summary.png` is `shap.summary_plot` on the actual model-width SHAP values with
  labels in model column order.
- The dependence plot uses the **top-ranked numeric source feature** from the
  global ranking, not an arbitrary column. A coordinator test asserts that both a
  dependence figure and per-row waterfall figures exist.
- `local/row_*.png` are native SHAP waterfall plots for the saved examples.
- The raw-input run writes `local/custom_0000.json` / `.png` and a manifest with
  `mode: custom_records`; it has no global ranking.

All files were created and non-empty. If a figure fails, the JSON remains valid
and the failure is recorded as a warning; no decorative or invented chart is
generated. The images are Git-ignored and are not committed.

---

## 10. Tests and exact results

### 10.1 Commands actually run

```bash
# Focused Phase 8 suite
set -o pipefail
PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest \
  tests/unit/test_shap_features.py \
  tests/unit/test_shap_explainers.py \
  tests/unit/test_shap_local.py \
  tests/unit/test_shap_coordinator.py \
  tests/integration/test_shap_mlp.py \
  tests/test_explain_cli.py -q

# Full suite
set -o pipefail
PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest -q
```

### 10.2 Results

- Focused: **42 passed** in 11.82 s.
- Full: **432 passed, 7 skipped** in 71.23 s, exit code 0.
- The 7 skips are the opt-in live-network tests
  (`tests/test_live_{discovery,acquisition,validation,preprocessing}.py`), not
  Phase 8 failures. Live-network tests were not run.
- No pre-existing test was weakened, skipped or suppressed. The new tests use
  deterministic synthetic fixtures and do not require internet access.

### 10.3 Coverage map

| Requirement | Test evidence |
| --- | --- |
| Synthetic Logistic Regression SHAP | `test_logistic_explainer_reports_log_odds_and_additivity` |
| Synthetic Random Forest SHAP | `test_forest_explainer_extracts_positive_class_and_probability_space` |
| MLP SHAP (real Keras environment) | `tests/integration/test_shap_mlp.py::test_mlp_explanations_are_additive_and_probability_space` |
| MLP fallback | `test_mlp_explainer_falls_back_honestly_when_deep_fails`; `test_mlp_without_working_explainers_fails_honestly` |
| Global ranking from genuine values | `test_ranked_importance_orders_by_mean_absolute_value`, `test_ranked_importance_marks_zero_contribution_neutral` |
| Local signed contributions / top-k | `test_contributions_respect_tolerance_and_top_k`, `test_local_directions_and_top_k` |
| Baseline / output-space metadata | `test_local_probability_matches_model`, `test_summary_uses_measured_values_and_disclaims_causality`, coordinator manifest assertions |
| Feature mapping after preprocessing | `test_groups_map_one_hot_columns_to_sources`, `test_mapped_columns_preserve_model_column_order`, `test_duplicate_or_empty_feature_names_are_rejected` |
| Prediction/explanation consistency | `test_local_probability_matches_model`, `test_non_finite_shap_values_are_rejected`, `test_out_of_range_probabilities_are_rejected`, additivity assertions |
| Single-row and batch behavior | `test_single_customer_record_can_be_explained`, `test_custom_records_are_local_only_and_privacy_safe` |
| Missing model artifact / provenance mismatch | `test_coordinator_reports_missing_baseline_artifact`, `test_coordinator_reports_missing_benchmark`, `test_coordinator_rejects_mismatched_preprocessing_provenance` |
| Explainer initialization / prediction failure | `test_tree_explainer_initialization_failure_is_actionable`, `test_linear_explainer_initialization_failure_is_actionable`, `test_additivity_failure_raises_instead_of_returning_values`, `test_prediction_failure_is_actionable`, missing-SHAP test |
| Sampling limits/configuration | `test_config_rejects_unbounded_sampling`, oversized-batch test |
| Visualization artifact generation | coordinator assertions for `importance.png`, `summary.png`, `dependence_*.png`, local PNGs |
| No fabricated fallback | additivity failure test and both-explainers-fail MLP test |
| Privacy-safe reporting/provenance | coordinator manifest assertions (`no identifiers`, test rows 0, dataset/model hashes) |
| CLI failure paths and exit codes | `tests/test_explain_cli.py` (invalid config → 3, missing artifacts → 2, no traceback) |
| Integration with Phase 5–7 interfaces | coordinator fixture persists a real Phase 5 hand-off and Phase 7-style baseline artifact; real CLI runs use the actual artifacts |

The tests do not mock the SHAP calculation. Synthetic fixtures are clearly named
and live under `tests/`; they never enter `data/` or project artifacts.

---

## 11. Privacy and reliability review

- The coordinator never samples or explains test rows; `split` only accepts
  `train` or `val`, and `test_rows_used` is written as `0`. The Phase 5 load
  reads the persisted matrix file, but test rows are not used in any explanation.
- Background rows come from the training partition. The validation sample is
  seeded and recorded by position.
- No customer identifiers are present in global, local or manifest artifacts. A
  real custom-record run was executed from a temporary CSV with the ID and target
  removed; the CSV was deleted.
- Local artifacts include transformed feature values and model contributions, not
  raw source rows. They are still treated as internal because transformed values
  can be sensitive.
- Natural-language summaries are assembled only from measured prediction and SHAP
  values. They explicitly disclaim causality and do not assert a feature "caused"
  churn.
- Model loading is digest-verified; family metadata is checked; feature counts
  must match the preprocessing schema.
- Artifact writes are atomic and path ids are validated. Core failures surface as
  `CW-XAI-*` with remediation; optional figure failures degrade to warnings
  without invalidating JSON.
- The original dataset, Phase 5 preprocessor, and Phase 6/7 model files are
  unchanged. No retraining occurs.

### 11.1 Observed non-fatal warnings

- Logistic Regression: one warning states that importances are in log-odds, not
  probability changes. This is intentional.
- The real custom-record run recorded the one-hot aggregation warning twice (once
  per explained row). The warning is per-row metadata; it is not an error.
- No mapping fallback warning occurred on the reference run.

---

## 12. Known limitations and unverified acceptance criteria

These are stated rather than hidden:

1. **Full-population explanations are not persisted.** Phase 8 explains a bounded
   validation sample and a handful of local examples. The acceptance criterion
   "explanation covers every customer present in the demo/prediction set" is a
   Phase 9 concern because that scoring population is not defined yet. The
   batch interface can explain up to 500 raw rows per call and is tested; a
   production export would need deliberate batching.
2. **No full signed SHAP matrix artifact.** `global.json` stores the ranking and
   summaries; local JSON files store full grouped contributions for explained
   rows. A later phase needing per-row vectors for all scored customers must
   extend the artifact contract or call the batch API, not assume a
   `shap_values.parquet` exists.
3. **Global rankings are sample estimates.** They use one seed and one sample;
   no confidence intervals or seed-stability study are included. Ordering near
   ties may change with another sample.
4. **MLP permutation fallback is not exercised on a real model.** Deep SHAP
   succeeded in this environment; the fallback is covered by synthetic forced
   failure. Its real-model runtime and additivity are unverified.
5. **No real second dataset.** All real explanations use OpenML `46915`. The
   interface is capability-driven, but cross-dataset behavior (different
   categorical levels, missingness, feature counts) is not verified here.
6. **Numeric original units are not reconstructed.** `Contribution.value` is in
   transformed model space; categorical active levels keep original labels. This
   is documented and is a presentation limitation, not a calculation error.
7. **Correlated and derived features share importance.** `total_day_charge` is
   derived from `total_day_minutes`, and `support_calls_per_usage_hour` is a
   derived rate. Rankings split attribution among correlated columns and do not
   establish independent effects.
8. **Cross-family numerical comparison is invalid.** Logistic Regression values
   are log-odds; tree/MLP values are probabilities. The report labels every
   number with its space.
9. **No causal claims.** SHAP explains model behavior on the observed
   distribution. It is not evidence that an intervention would change churn.
10. **Visual inspection is limited to file creation and test assertions.** The
    plots are generated through the SHAP/Matplotlib APIs and inspected for
    existence/non-emptiness; they were not manually reviewed for every aesthetic
    detail.

---

## 13. Files and contracts introduced

New source:

- `churnwise/explain/__init__.py`
- `churnwise/explain/contracts.py`
- `churnwise/explain/coordinator.py`
- `churnwise/explain/engine.py`
- `churnwise/explain/errors.py`
- `churnwise/explain/explainers.py`
- `churnwise/explain/features.py`
- `churnwise/explain/plots.py`
- `churnwise/explain/reporting.py`

Changed:

- `churnwise/cli.py` — added `explain` command.

New tests:

- `tests/unit/test_shap_features.py`
- `tests/unit/test_shap_explainers.py`
- `tests/unit/test_shap_local.py`
- `tests/unit/test_shap_coordinator.py`
- `tests/integration/test_shap_mlp.py`
- `tests/test_explain_cli.py`

New documentation:

- `docs/EXPLAINABILITY.md`
- `docs/PHASE8_REPORT.md` (this file)

Updated documentation:

- `docs/ARCHITECTURE.md` — explain module map, SHAP design, artifact layout and
  resolved open decision.
- `docs/ROADMAP.md` — Phase 8 verified status.
- `README.md` — Phase 8 status, command, artifacts, guarantees and limitations.
- `docs/ACCEPTANCE_CRITERIA.md` — Phase 8 artifact names and verified status.

No model, preprocessor, split or source dataset was modified.
