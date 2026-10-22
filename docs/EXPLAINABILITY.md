# ChurnWise AI — Explainability and SHAP Interface

| Field | Value |
| --- | --- |
| Status | Implemented in Phase 8; verified against the real Phase 5–7 artifacts |
| Version | 1.0 |
| Date | 2026-10-22 |
| Scope | SHAP explanation interface, output-space semantics, artifacts, privacy, failure behaviour and Phase 9 integration |
| Related | [PHASE8_REPORT.md](PHASE8_REPORT.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [MODELING.md](MODELING.md) · [EVALUATION.md](EVALUATION.md) · [ROADMAP.md](ROADMAP.md) |

> SHAP values attribute a model's output to its input features. They are **not**
> causal effects. A positive value means the feature pushed this model output
> toward the churn-positive class; it does not mean changing the feature would
> change churn by that amount.

---

## 1. What the subsystem does

`churnwise.explain` loads a digest-verified model from Phase 6/7 together with
its exact Phase 5 preprocessing run, computes SHAP values on a bounded sample of
validation rows (or caller-supplied raw records), groups one-hot columns back to
source features, and persists structured results and figures with provenance.

It does **not** train models, rerun preprocessing, use the test partition, rank
customers, or make causal claims. Phase 9 is expected to consume the structured
outputs described here.

---

## 2. Supported models and explainer selection

| Family | Primary explainer | Background | Output space | Exactness / fallback |
| --- | --- | --- | --- | --- |
| `logistic_regression` | `shap.LinearExplainer` | Seeded sample of training rows (default 100) | `log_odds` | Exact for the linear decision function. No fallback. |
| `random_forest` | `shap.TreeExplainer` (`tree_path_dependent`) | Training sample used for the run's background metadata; Tree SHAP itself is tree-dependent | `probability` | Exact for tree ensembles. No fallback. |
| `mlp` | `shap.DeepExplainer` on the Keras model | Seeded sample of training rows (default 50) | `probability` | Deep SHAP on the sigmoid output. If DeepExplainer cannot initialise or probe one row, a bounded model-agnostic `shap.Explainer(..., algorithm="permutation")` is selected explicitly and recorded. |

Selection is implemented in `churnwise/explain/explainers.py::build_explainer`.
Unsupported families raise `CW-XAI-002`; a missing SHAP installation raises the
same code with an install remediation. A failed explainer never returns
placeholder values: the run fails or, for the MLP only, records the documented
permutation fallback in the manifest and warnings.

### MLP fallback trade-off

The permutation explainer estimates Shapley values by repeated masking, so it is
much slower and approximate. It is bounded by `permutation_max_evals` (default
500) and by the MLP sample/background limits. On the reference environment
`shap.DeepExplainer` succeeded, so the real MLP results in
[PHASE8_REPORT.md](PHASE8_REPORT.md) use Deep SHAP; the fallback is covered by a
forced-failure synthetic integration test.

---

## 3. Output spaces and additivity

Two spaces are explicit in every result and artifact:

| Space | Models | Meaning of a SHAP value | Reconstruction |
| --- | --- | --- | --- |
| `probability` | Random Forest, MLP | Change in P(churn = 1) attributable to the feature | `base_value + Σ contributions ≈ probability` |
| `log_odds` | Logistic Regression | Change in the logistic decision function (`log(p / (1-p))`) | `base_value + Σ contributions ≈ decision_function` |

Rules:

- `LocalExplanation.probability` is always a probability in `[0, 1]`.
- `LocalExplanation.model_output` and `base_value` are in the same space as the
  SHAP values (`probability` or `log_odds`).
- Log-odds contributions must not be described as probability changes. To obtain
  the predicted probability, apply the sigmoid to `base_value + Σ contributions`.
- Mean absolute SHAP values are only comparable **within** one output space. The
  Logistic Regression ranking is in log-odds units and is numerically much larger
  than the Random Forest/MLP rankings in probability units; do not compare the
  numbers across families.
- Every batch performs an additivity check before returning values. The default
  absolute tolerance is `1e-4`; failure raises `CW-XAI-004` and the partial
  explanation is not reported. Observed errors are in
  [PHASE8_REPORT.md](PHASE8_REPORT.md).
- The base value is the explainer's model baseline, not the dataset's churn
  prevalence. Class-weighted models can have a baseline well above the observed
  positive rate (reference run: RF 0.5005, MLP 0.3007, LR −0.6633 log-odds
  against a 14.14% churn rate). Always report it with its output-space label.

---

## 4. Feature mapping and contribution interpretation

### 4.1 One-hot grouping

Phase 5 stores each transformed column in `schema.output_features` as
`{name, source, role, value}`. `build_feature_groups()` maps every model-width
column to its `source` feature. For example:

```text
categorical__international_plan_0 ┐
categorical__international_plan_1 ┴─► international_plan
```

Contributions for columns of the same source are **summed**. This is valid
because SHAP values are additive within the model output space. The signed sum
is computed before taking an absolute value, so opposite-sign one-hot levels can
partly cancel in the grouped ranking; the per-column detail remains available in
`Contribution.columns` and in `global.json`'s recorded source columns. Local
results retain that raw detail, and the grouping rule is written to
`global.json.explainer.mapping`.

When a transformed column has no Phase 5 mapping, the Phase 5 prefix
(`numeric__`, `categorical__`, `boolean__`) is stripped to produce a documented
fallback label, and the run records a warning. A guessed source name is never
presented as verified metadata.

Native SHAP plots receive labels in **model column order**, not alphabetical
group order, so a plotted label always belongs to the column whose values were
passed to SHAP.

### 4.2 Local directions

- `contribution > tolerance` → `increases_churn` (pushes toward the documented
  positive class, label `Yes` for the reference dataset).
- `contribution < -tolerance` → `decreases_churn` (pushes away from it).
- `|contribution| ≤ tolerance` → `neutral`.
- `top_positive` is sorted by descending contribution; `top_negative` by
  ascending contribution (most negative first).

`Contribution.value` is the **transformed model-input value** for a
single-column numeric feature. `Contribution.active_level` is the original
one-hot level label when a categorical group is active. Raw source rows and
customer identifiers are not joined to the explanation, so numeric values are in
standardised/model space, not original units. This is a deliberate
privacy/simplicity trade-off and is a documented limitation for presentation.

### 4.3 Global `direction` is not a monotone effect

`global.json.feature_importance[].direction` is derived from
`positive_share`, the fraction of sampled rows whose grouped contribution is
positive:

- mean absolute contribution ≈ 0 → `neutral`
- otherwise `> 0.5` → `increases_churn`
- otherwise `< 0.5` → `decreases_churn`
- otherwise `= 0.5` → `mixed`

It describes the **majority sign across rows**, not a claim that larger feature
values increase churn. With a Random Forest base value near 0.5, most individual
rows sit below the base, so a feature can be labelled `decreases_churn` even
while it strongly increases risk for high-usage customers. Use the local signed
contributions and the dependence plot for direction-by-value analysis.

---

## 5. Global explanation interface

```python
from churnwise.config import Settings
from churnwise.explain import ExplainConfig, ExplainCoordinator

coordinator = ExplainCoordinator(
    Settings.load(),
    ExplainConfig(sample_size=200, background_size=100, local_examples=5, seed=42),
)
result = coordinator.run("random_forest")  # latest benchmark; validation rows
print(result.run_id)
for row in result.global_explanation.feature_importance[:10]:
    print(row["rank"], row["feature"], row["mean_abs_shap"], row["share"])
```

`run()` defaults:

| Argument | Default | Meaning |
| --- | --- | --- |
| `family` | `None` | `None` uses the Phase 7 `selection.selected_family` |
| `benchmark_run_id` | latest with `metrics.json` | Exact Phase 7 benchmark artifact |
| `model_run_id` | metadata / latest MLP | MLP artifact selection |
| `explain_run_id` | generated | Optional explicit, validated id |
| `split` | `"val"` | `train` or `val` only; test is rejected |
| `persist` | `True` | Write JSON/figures/local examples |
| `progress` | `None` | Optional callback receiving short status strings |

The sample is a **seeded uniform sample without replacement** from the chosen
partition. The exact positions and sample sizes are stored in `global.json.sample`
and `run_manifest.json.sampling`. Background rows always come from the training
partition. `test_rows_used` is recorded as `0`.

`GlobalExplanation` fields include:

| Field | Meaning |
| --- | --- |
| `run_id`, `family`, `output_space` | Identity and interpretation |
| `feature_importance[]` | Ranked source features with `mean_abs_shap`, `mean_shap`, `share`, `positive_share`, `direction`, `n_columns`, `columns` |
| `base_value` | Mean explainer base value across the sample, in `output_space` |
| `sample` | Split, available/sampled rows, seed, positions, `test_rows_used` |
| `additivity_error` | Maximum absolute additivity error over explained rows |
| `explainer` | Explainer name, background size, fallback metadata, mapping summary |
| `provenance` | Dataset hash, split fingerprint, preprocessing run, model run/hash, benchmark run |
| `plots` | Paths to generated figures |
| `warnings` | Output-space and mapping warnings |
| `duration_seconds`, `config` | Measured runtime and frozen configuration |

Ranking is computed from genuine SHAP values as
`mean(abs(grouped SHAP value))` per source feature. `share` is that value divided
by the sum of all mean absolute values; it is a relative concentration measure,
not an R² or a causal share.

---

## 6. Individual customer interface

### 6.1 From raw records

```python
import pandas as pd
from churnwise.config import Settings
from churnwise.explain import ExplainCoordinator

frame = pd.read_csv("customers.csv")   # columns required by the Phase 5 schema
result = ExplainCoordinator(Settings.load()).explain_records(
    frame,
    family="random_forest",
)
for item in result.local_explanations:
    print(item.probability, item.predicted_class, item.summary)
    for contribution in item.top_positive:
        print("+", contribution.feature, contribution.contribution)
```

The frame is transformed through the exact fitted Phase 5 pipeline
(`dataset.transform_new`), including imputation, scaling, unknown-category
handling and derived features. Missing values are therefore handled by the fitted
pipeline, not ad hoc. The input must be a non-empty `pandas.DataFrame` with at
most `max_local_batch` rows (default 500).

`LocalExplanation` returns:

| Field | Meaning |
| --- | --- |
| `probability` | Model P(churn = 1) |
| `predicted_class` | `1` when `probability >= threshold`, else `0` |
| `threshold` | Operating threshold from the benchmark/metadata |
| `positive_label` | Original positive label, e.g. `Yes` |
| `base_value`, `model_output` | Baseline and prediction in `output_space` |
| `contributions` | All grouped source features, signed and sorted by absolute value |
| `top_positive`, `top_negative` | Top-k signed drivers (`top_k`, default 5) |
| `summary` | Data-grounded sentence built only from measured values |
| `additivity_error` | Maximum additivity error for the batch |
| `row_position` | Position in the supplied frame or split, **not** a customer identifier |
| `plot_path`, `warnings` | Waterfall figure and per-row caveats |

The generated summary states the prediction, threshold, class, largest measured
upward and downward contributions, and the non-causal disclaimer. It never
invents a reason that is absent from the SHAP values.

### 6.2 From a saved validation sample

`run()` explains `local_examples` rows (default 5), selected as the highest and
lowest predicted-risk rows within the explained sample so both directions are
represented. These are positional rows in the validation split, not customer
IDs, and are persisted as `local/row_<position>.json` + `.png`.

### 6.3 Batch limits

`chunked_explain()` splits the matrix into `chunk_size` blocks (default 100),
validates additivity per block, and merges the results. This bounds peak memory
without changing values. `explain_records()` rejects oversized frames before any
model work. `max_local_batch` is a hard guard against accidentally explaining a
production-sized export in one call.

---

## 7. Configuration

| Option | Default | Valid range | Purpose |
| --- | --- | --- | --- |
| `sample_size` | 200 | 1–5,000 | Validation rows explained for LR/RF |
| `background_size` | 100 | 1–1,000 | Training background rows for LR/RF |
| `mlp_sample_size` | 60 | 1–1,000 | Validation rows explained for MLP |
| `mlp_background_size` | 50 | 1–500 | Training background rows for MLP |
| `local_examples` | 5 | 0–100 | Persisted individual examples from a run |
| `top_k` | 5 | 1–50 | Positive/negative drivers returned per row |
| `seed` | 42 | ≥ 0 | Deterministic sampling |
| `tolerance` | 1e-4 | (0, 1] | Absolute additivity tolerance |
| `chunk_size` | 100 | 1–1,000 | Rows per SHAP call |
| `max_local_batch` | 500 | 1–5,000 | Raw-record batch guard |
| `permutation_max_evals` | 500 | 2–5,000 | MLP fallback budget |

Counts above the available rows are silently capped by
`deterministic_sample`; the actual values are recorded. Invalid values raise
`CW-XAI-001` before model loading.

---

## 8. Artifacts

```text
artifacts/explanations/<explain_run_id>/
├── global.json                 # ranking, sample, output space, provenance, warnings
├── importance.png              # mean |SHAP| horizontal bar chart
├── summary.png                 # SHAP beeswarm, labels in model column order
├── dependence_<feature>.png    # top-ranked numeric source feature
├── local/
│   ├── row_00087.json          # full grouped contributions + raw column detail
│   └── row_00087.png           # SHAP waterfall
└── run_manifest.json           # exact model/preprocessing/dataset provenance + config
```

For `explain --input`, the run contains `local/custom_0000.json` / `.png` and a
`run_manifest.json` with `mode: custom_records` and no global ranking. Raw input
rows are not copied into artifacts.

`global.json` is schema version `1.0`, phase `8`, and contains no customer rows:
only aggregate values, sample positions and provenance. Local JSON files contain
model contributions and transformed feature values; they do not contain customer
IDs or the raw source record. Treat them as internal/confidential anyway because
transformed values can still be sensitive.

Artifacts are Git-ignored and reproducible from the persisted Phase 5–7 model
artifacts. Each run is immutable by convention; the coordinator uses atomic JSON
writes and refuses malformed ids / path traversal.

The plotting helpers are exported from `churnwise.explain`
(`plot_importance`, `plot_beeswarm`, `plot_dependence`, `plot_waterfall`) for
callers that already hold computed values. Prefer `ExplainCoordinator.run()` so
the figures are tied to a recorded run and provenance manifest.

---

## 9. Failure taxonomy

| Code | Meaning | Behaviour |
| --- | --- | --- |
| `CW-XAI-001` | Invalid configuration or input frame | Fail before model loading; actionable message |
| `CW-XAI-002` | Unsupported model family, missing SHAP, or no usable explainer | Fail; no fabricated values |
| `CW-XAI-003` | Explainer/prediction failure | Fail with exception context; no partial summary |
| `CW-XAI-004` | Feature/background validation or additivity failure | Fail; values are not reported |
| `CW-XAI-005` | Artifact read/write or figure failure | Fail for core JSON; optional figure failure is a warning |
| `CW-XAI-006` | Missing model, benchmark, preprocessing or feature metadata, or a model/preprocessing provenance mismatch | Fail with the exact missing/mismatched artifact and remediation |

Optional plots do not invalidate the structured explanation: if Matplotlib or a
native SHAP plot fails, the JSON remains authoritative and the failure is listed
in warnings. Core values and local JSON are never silently replaced.

---

## 10. Provenance and privacy

Every manifest records:

- `dataset_id`, `dataset_sha256`, `split_fingerprint`;
- `preprocessing_run_id`;
- Phase 7 `benchmark_run_id` and the exact model `model_run_id`/`model_sha256`;
- operating threshold and positive label;
- explainer name, output space, background rows and fallback metadata;
- frozen `ExplainConfig` and environment versions (Python, NumPy, SHAP,
  scikit-learn, TensorFlow when installed);
- sample/background positions, `test_rows_used: 0`, warnings and limitations.

The model artifact is opened read-only. The coordinator refuses a model whose
feature count does not match the preprocessing schema and rejects the test split
for explanation. It does not log raw records; local summaries are assembled in
memory and only transformed contributions are written.

---

## 11. Integration guidance for Phase 9 and the dashboard

Recommended path:

1. **Batch scoring** (Phase 9) should load the same model/preprocessing pair via
   `ExplainCoordinator.load_model()` or the existing model registry, score with
   the same `predict_proba`, and attach drivers from
   `local_explanations()` or the persisted local JSON.
2. **Do not import SHAP inside the Streamlit process.** Read precomputed JSON;
   the dashboard's explainability page should surface `global.json` and local
   files, not recompute values.
3. **Do not mix output spaces.** If displaying Logistic Regression drivers next
   to probabilities, label them as log-odds or convert the full additive
   reconstruction through the sigmoid; never add log-odds values to a
   probability.
4. **Preserve identity.** Use `row_position` only within its split. If customer
   IDs are needed, Phase 9 must join them from the scored frame and record that
   join explicitly; Phase 8 deliberately does not persist them.
5. **Reuse provenance.** Attach the Phase 8 `run_id`, model hash, preprocessing
   run and dataset hash to any downstream risk record so an explanation can be
   traced back to its exact prediction.

---

## 12. Performance and limitations

- Default settings are sized for a student CPU laptop: 200 LR/RF rows with a
  100-row background, 60 MLP rows with a 50-row background, 5 local examples.
- Reference runtimes: Random Forest 11.27 s, Logistic Regression 5.28 s, MLP
  5.21 s end-to-end (including model load, plots and JSON) on the reference
  Windows CPU environment. These are measurements, not a guarantee.
- Tree SHAP and Linear SHAP are exact; Deep SHAP is an approximation with a
  measured additivity check; permutation fallback is approximate and slower.
- Global rankings are sample estimates. A different seed/sample changes the
  ordering slightly; no confidence interval is computed.
- Derived features (`total_day_charge`, `support_calls_per_usage_hour`,
  `usage_band`) can be strongly correlated with their sources, so importance is
  split across correlated columns. Do not interpret a low rank as evidence that
  the underlying concept is unimportant.
- Numeric `Contribution.value` is in transformed model space. Original numeric
  units are not reconstructed; categorical active levels retain their original
  labels.
- There is no cross-process explainer cache; each CLI invocation reinitialises
  the explainer. Within a run, one explainer and one background sample are
  reused for all explained rows.
- SHAP values explain the model, not the customer, and are not causal. Retention
  decisions require experiments or domain evidence beyond these attributions.
