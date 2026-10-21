# ChurnWise AI — MLP Modeling and Inference

**Status:** Phase 6 implementation. This document describes the model contract used
by Phase 7 and later phases. It contains no performance claims: the MLP has been
trained on validation-monitored data, but the test set has not been evaluated.
Comparative metrics, threshold tuning and final model selection belong to Phase 7.

See [PHASE6_REPORT.md](PHASE6_REPORT.md) for the verified run, test results and
known limitations.

---

## 1. Scope and entry points

Phase 6 adds a TensorFlow/Keras multilayer perceptron for binary customer churn
prediction. It consumes the `PreparedDataset` hand-off produced by Phase 5 and
writes a self-describing model artifact that can be reloaded without refitting.

| Entry point | Purpose |
| --- | --- |
| `churnwise.models.MLPConfig` | Frozen architecture and training configuration |
| `churnwise.models.build_mlp(input_dim, config)` | Construct and compile the Keras model |
| `churnwise.models.MLPTrainer(settings, config).train(dataset, ...)` | Fit on train, early-stop on validation, persist artifacts |
| `churnwise.models.train_model(dataset, settings=..., config=...)` | One-call training wrapper |
| `churnwise.models.ModelRegistry(settings)` | Resolve, save, verify and load model artifacts |
| `churnwise.models.ChurnPredictor` | Probability/class inference for matrices or raw customer rows |
| `churnwise.models.ModelingError` | Stable `CW-MOD-*` failures with a remediation |
| `python -m churnwise.cli train` | CLI training command |
| `python -m churnwise.cli predict` | CLI raw-record inference command |

No model-family selection, benchmarking, SHAP, risk ranking, campaign simulation
or dashboard code is part of this phase.

---

## 2. Input contract and leakage boundary

`MLPTrainer.train` accepts a `churnwise.preprocessing.PreparedDataset`:

| Field | Use |
| --- | --- |
| `train.X`, `train.y` | Only arrays passed to `model.fit` for weight updates |
| `val.X`, `val.y` | `validation_data` for early stopping, LR reduction and checkpointing |
| `test.X`, `test.y` | Never read by the trainer; `diagnostics["test_rows_used"]` is always `0` |
| `feature_names` | Declares the transformed width and the model's input contract |
| `class_weights` | Passed to `fit(class_weight=...)` (the positive class is weighted higher) |
| `split_spec` | Split fingerprint, dataset SHA-256, sizes and seed copied into metadata |
| `schema` | Target mapping, feature names, preprocessor hash, provenance |
| `run_id` | Preprocessing run referenced by every model artifact |
| `resampled_train()` | Used only when Phase 5 explicitly configured training-only oversampling |

The positive class is `1` and means **churned**. Phase 5 stores the source labels and
positive label in `schema["target"]`; the model metadata copies that mapping. The
trainer rejects labels outside `{0, 1}`, empty partitions, feature/label length
mismatches, width mismatches against `feature_names`, and non-finite features.

Hard rules enforced by tests and review:

- Fitting sees training rows only.
- Early stopping, LR reduction and checkpoint selection use `val_loss` only.
- No test row or test statistic participates in fitting or model selection.
- The trainer verifies that train, validation and test source-row indices are
  disjoint before fitting, in case an upstream split is malformed.
- The model never receives the raw target column; only the transformed matrix and
  the 0/1 labels supplied by Phase 5.

---

## 3. Architecture

Default network for the current 27-feature churn dataset:

```text
input (d transformed features)
  Dense(64, ReLU, He normal, L2 1e-4)
  Dropout(0.30)
  Dense(32, ReLU, He normal, L2 1e-4)
  Dropout(0.20)
  Dense(1, sigmoid)  ->  P(churn = 1)
```

| Element | Default | Configurable via |
| --- | --- | --- |
| Input width | derived from `PreparedDataset.feature_names` | data (no manual width) |
| Hidden widths | `(64, 32)` | `MLPConfig.hidden_units`, `--hidden` (repeatable) |
| Hidden activation | ReLU | fixed by design |
| Weight initialisation | He normal | fixed by design |
| Dropout | `(0.30, 0.20)` | `MLPConfig.dropout`, `--dropout` (one rate per layer or one for all) |
| L2 penalty | `1e-4` | `MLPConfig.l2`, `--l2` |
| Output | one unit, sigmoid | fixed by design |
| Loss | binary cross-entropy | fixed by design |
| Optimiser | Adam | fixed by design |
| Learning rate | `1e-3` | `MLPConfig.learning_rate`, `--learning-rate` |
| Batch size | `64` | `MLPConfig.batch_size`, `--batch-size` |
| Epoch limit | `150` | `MLPConfig.epochs`, `--epochs` |
| Early-stopping patience | `12` | `--early-stopping-patience` |
| LR-reduction patience/factor | `5` / `0.5` | `--reduce-lr-patience`, `--reduce-lr-factor` |
| Minimum learning rate | `1e-5` | `--min-learning-rate` |
| Inference threshold | `0.5` | `MLPConfig.threshold`, `--threshold` |
| Seed | `42` (from `Settings.seed`) | `MLPConfig.seed`, `--seed` |
| Op determinism | off | `--deterministic` |

`MLPConfig.validate()` rejects empty/oversized layer lists, non-positive widths,
dropout outside `[0, 1)`, negative L2, non-positive learning rate or batch size,
invalid epochs/patience, thresholds outside `[0, 1]`, and `min_learning_rate >
learning_rate`. `build_mlp` also rejects a non-positive or non-integer input width.

### Why this shape

The dataset is small, tabular and imbalanced (14.1% positive). Two narrow hidden
layers can represent non-linear interactions without the parameter count of a wide
network. He initialisation suits ReLU; dropout and L2 reduce overfitting; sigmoid
plus binary cross-entropy is the standard probabilistic objective for binary
labels. Early stopping removes the need to guess a fixed epoch budget.

A single hidden layer can be selected with `--hidden 32 --dropout 0.2`; deeper
networks are allowed but are not justified by the current data size.

---

## 4. Forward pass, loss and backpropagation

For a transformed input row `x`:

1. Each hidden layer computes `h = ReLU(W h_prev + b)`. ReLU outputs
   `max(0, z)`, so negative pre-activations contribute zero gradient.
2. Dropout randomly zeroes a configured fraction of hidden activations during
   training only; Keras disables it for validation and inference.
3. The output unit computes `p = sigmoid(w·h + b) = 1 / (1 + exp(-z))`, giving a
   churn probability in `[0, 1]`.
4. Binary cross-entropy compares `p` with the 0/1 label:
   `L = -(y log(p) + (1-y) log(1-p))`, averaged over the mini-batch, plus the L2
   penalty on kernel weights.
5. Keras/TensorFlow automatic differentiation computes gradients of `L` with
   respect to every weight and bias. Adam maintains per-parameter first and second
   moment estimates and applies an adaptive update with the configured learning
   rate.
6. Mini-batches of 64 rows are processed per step; one epoch is one pass over the
   training partition.

The loss is not reimplemented by hand and predictions are not hardcoded. Class
weights from Phase 5 multiply each sample's loss contribution, so the minority
churn class is not overwhelmed by the majority class.

---

## 5. Training workflow

`MLPTrainer.train(dataset, model_run_id=None, persist=True)` performs:

1. Validate configuration, labels, feature widths and finiteness.
2. Resolve a safe model run id (default:
   `<UTC timestamp>Z-<dataset_id>-mlp-<config fingerprint[:8]>`).
3. Seed Python, NumPy and TensorFlow, and set `PYTHONHASHSEED`. If
   `--deterministic` is set, enable `tf.config.experimental.enable_op_determinism()`.
4. Build the model for the dataset's transformed width.
5. Fit with these callbacks:
   - `EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)`
   - `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)`
   - `ModelCheckpoint(..., monitor="val_loss", save_best_only=True)` to
     `best.keras`
   - `TerminateOnNaN()`
6. Reject missing or ragged history, non-finite losses/metrics, or a missing best
   checkpoint with an actionable `CW-MOD-002` failure.
7. Reload the best validation checkpoint and use those weights as the returned
   model, so the artifact is the selected model rather than the last epoch.
8. Record diagnostics, architecture, environment, seeds, history and paths.
9. Persist `model.keras`, `best.keras`, `history.json`, `learning_curves.png`,
   `metadata.json`, and JSON/Markdown reports.

The actual epoch count, best epoch, best validation loss and stopping status are
derived from the Keras history of that run. There is no fabricated history.

**Failure handling.** Invalid inputs raise `CW-MOD-001`; fit failures, interrupted
training, non-finite history and missing checkpoints raise `CW-MOD-002`; artifact
read/write/digest failures raise `CW-MOD-003`; prediction failures raise
`CW-MOD-004`. Each carries a remediation string. `KeyboardInterrupt` is converted
to a training failure without publishing a model artifact. The epoch limit plus
early stopping bounds CPU time.

**Defaults are deliberately small:** 3,905 parameters for the current 27-feature
input, CPU-only training, and a batch of 64. On the audited Windows laptop the real
reference run completed in about 12 seconds.

---

## 6. Training history and learning curves

`history.json` is the per-epoch Keras history converted to plain JSON floats. Keys
depended on the run, but normally include:

| Key | Meaning |
| --- | --- |
| `loss` | Training binary cross-entropy per epoch |
| `val_loss` | Validation binary cross-entropy per epoch |
| `accuracy` | Training binary accuracy |
| `val_accuracy` | Validation binary accuracy |
| `learning_rate` | Optimiser learning rate recorded by Keras when available |

All values are finite, and every series has the same length. The number of entries
equals the actual epochs completed, not the requested limit.

`learning_curves.png` is generated from that history only:

- training vs validation loss (binary cross-entropy, lower is better);
- training vs validation binary accuracy (0–1), when available;
- Adam learning rate per epoch, when available;
- labels, legends, titles and epoch axes; a single-epoch run still renders.

The plot is written under the model artifact directory and referenced from
metadata and reports. If plotting fails, training is not invalidated: the curve
path is removed, and a warning is recorded. There are no illustrative or synthetic
curve points.

---

## 7. Artifacts and metadata

Layout for a successful persisted run:

```text
artifacts/models/<model_run_id>/mlp/
├── model.keras              # best-validation Keras model, safe to reload
├── best.keras               # ModelCheckpoint copy selected on val_loss
├── history.json             # per-epoch history (actual run)
├── learning_curves.png      # loss / accuracy / learning-rate curves
└── metadata.json            # architecture, training, provenance, diagnostics

reports/models/<model_run_id>.json
reports/models/<model_run_id>.md
```

`metadata.json` is the authoritative record. It includes:

| Block | Contents |
| --- | --- |
| Identity | `schema_version`, `phase`, `model_family`, `model_run_id`, `created_at`, `code_version` |
| Architecture | input width, hidden widths, dropout, activations, initialiser, L2, parameter count, Keras summary |
| Training | full `MLPConfig` payload, optimizer, loss, class weights, epochs requested/completed, early stopping |
| Preprocessing | Phase 5 run id, dataset id and SHA-256, split fingerprint, preprocessor hash and path, feature names/count/sources, target mapping, split sizes, provenance |
| Diagnostics | actual fitted train/validation rows and positive counts/rates (training-only oversampling reflected), `test_rows_used: 0`, best epoch/loss, stopping flags, duration, history metric names |
| Reproducibility | Python/NumPy/TensorFlow seeds, deterministic-op flag, explicit nondeterminism note |
| Environment | Python, platform, TensorFlow, Keras, NumPy versions |
| Artifacts | absolute paths to model, checkpoint, history, curves, metadata and reports |
| Warnings | underfitting/overfitting/data-size/epoch-limit signals and determinism note |

### Safe loading

`ModelRegistry` only resolves model run ids matching
`^[A-Za-z0-9][A-Za-z0-9._-]{2,160}$` and refuses paths that escape
`artifact_dir/models`. Before deserialising it requires:

1. `model.keras`, `metadata.json` and `history.json` to exist;
2. metadata to be an object with the expected schema version and matching run id;
3. a recorded SHA-256 digest that matches the model file;
4. `tf.keras.models.load_model(..., compile=False, safe_mode=True)`.

`ChurnPredictor` also checks that the model's input width matches the metadata
feature-name count before accepting an artifact for inference.

A missing digest, digest mismatch, unsupported schema version or load failure
raises `CW-MOD-003` with remediation. Phase 6 only loads artifacts this project
created; it does not accept arbitrary model files or URLs. Inference additionally
reloads the referenced Phase 5 preprocessor through
`PreprocessingCoordinator.load`, which verifies its own schema version and digest.

---

## 8. Inference interface

### Matrix predictions

```python
from churnwise.models import ChurnPredictor

predictor = ChurnPredictor(model, metadata, threshold=0.4)
probabilities = predictor.predict_proba(X)       # float ndarray, shape (n,)
batch = predictor.predict(X)                     # PredictionBatch
```

`predict_proba` requires a 2-D finite matrix with exactly the feature count recorded
in metadata. It validates shape and finiteness, rejects NaN/inf, and clips the
sigmoid output defensively. Row order is preserved.

### Raw customer records

```python
batch = predictor.predict_frame(frame)          # pandas DataFrame
one = predictor.predict_one(record)             # mapping or Series -> Prediction
```

Raw rows are transformed with the **exact fitted Phase 5 pipeline** stored with the
Phase 5 run referenced by model metadata. Required columns are
`schema["required_columns"]`; derived features are replayed from the recorded
schema. Missing columns raise a `CW-MOD-004` error naming the missing fields.
Unexpected extra columns are ignored by the fitted `ColumnTransformer`. Infinite
numeric values are treated as missing and imputed; if transformation still produces
non-finite values, prediction fails rather than returning a meaningless score.

### Classes and threshold

`class = 1 if probability >= threshold else 0`, with `1 = churned` and `0 = not
churned` (the names come from Phase 5's target mapping). The threshold is
configurable per predictor or with `--threshold`; the default is the threshold
recorded during training (`0.5` unless configured otherwise). `0.5` is a neutral
default, not a validated business cut-off; Phase 7 owns threshold selection and
comparative evaluation.

### Return structures

| Type | Fields |
| --- | --- |
| `PredictionBatch` | `probabilities`, `classes`, `threshold`, `positive_label`, `row_positions`, `n_rows`, `to_dict()` |
| `Prediction` | `probability`, `predicted_class`, `threshold`, `positive_label`, `to_dict()` |

`row_positions` are `0..n-1` for the supplied batch and preserve input order. They
are positional correspondence only; no customer identifiers are added to outputs
or logs. Prediction CSVs contain `row_index`, `churn_probability` and
`churn_prediction`.

### Loading a saved predictor

```python
from churnwise.models import ChurnPredictor

predictor = ChurnPredictor.load(
    settings,
    model_run_id,
    threshold=None,             # use metadata default
    with_preprocessing=True,    # required for raw DataFrames
)
```

`with_preprocessing=False` is sufficient for transformed matrices and raises a
clear error if raw records are supplied.

---

## 9. Integration contract for Phase 7 and later

| Need | Interface |
| --- | --- |
| Construct the MLP | `build_mlp(input_dim, config)` or `MLPConfig` |
| Train on Phase 5 partitions | `MLPTrainer(settings, config).train(dataset)` |
| Fitted model and history | `TrainingResult.model`, `.history`, `.diagnostics`, `.paths` |
| Reload a saved model | `ModelRegistry(settings).load(run_id)` or `ChurnPredictor.load(...)` |
| Probabilities / classes | `predict_proba`, `predict`, `predict_frame`, `predict_one` |
| Diagnostics and metadata | `TrainingResult.metadata`, `.diagnostics`; `LoadedModel.metadata` |
| Learning curves | `result.paths["curves"]`, metadata `artifacts.curves` |
| Trace to preprocessing/provenance | metadata `preprocessing` block (`run_id`, `dataset_sha256`, `split_fingerprint`, `preprocessor_hash`) |
| Latest run | `ModelRegistry(settings).latest_model_run_id()` |

Phase 7 must use the model's frozen preprocessing contract and the **same test rows**
as every other family. Test-set evaluation is intentionally absent from Phase 6: no
test metric was computed, so there is no leakage into model selection and no
performance claim to carry forward.

---

## 10. Reproducibility and nondeterminism

`--seed` (default 42) sets Python's `random`, NumPy and `tf.keras.utils.set_random_seed`,
and `PYTHONHASHSEED`. The model configuration fingerprint covers all architecture and
training options and is embedded in the default run id.

Seeds alone do not guarantee bitwise reproducibility. OneDNN CPU kernels, parallel
floating-point reductions and threading can change low-order bits between runs or
machines. `--deterministic` enables TensorFlow op determinism at a performance cost;
even then, different TensorFlow/Keras versions may not reproduce identical weights.
Every non-deterministic run records that fact in metadata and warnings.

Within one environment and seed, the reference run was repeated and produced the
same recorded epoch count and best validation loss; this is evidence for this host,
not a cross-platform guarantee. Artifact compatibility is documented through the
recorded library versions, and a model should be reloaded with a compatible
TensorFlow/Keras installation. Malformed class weights or overlapping split indices
are rejected before fitting rather than producing a misleading run.

---

## 11. Commands and verification

```bash
# Train from the latest prepared Phase 5 run, using defaults
python -m churnwise.cli train

# Explicit Phase 5 run and a smaller CPU-friendly configuration
python -m churnwise.cli train \
  --preprocessing-run-id 20261021T123405Z-openml_46915-989d8802 \
  --epochs 60 --batch-size 64 --hidden 64 --hidden 32 \
  --dropout 0.3 --dropout 0.2 --l2 1e-4

# Predict raw customer records; threshold is optional
python -m churnwise.cli predict \
  --input customers.csv --output predictions.csv --threshold 0.4

# Predict with a specific model instead of the latest
python -m churnwise.cli predict --model-run-id <id> --input customers.csv
```

`train` exits `0` on success, `2` for a missing/invalid Phase 5 hand-off or a
modeling failure, and `3` for invalid configuration or artifact-write failures.
`predict` exits `0` on success, `2` for missing model/input or prediction failures,
and `3` for invalid threshold or output-write failures. `--json` prints the full
structured result; summaries do not contain raw customer records.

Focused tests:

```bash
pytest tests/unit/test_mlp_architecture.py \
       tests/unit/test_mlp_training.py \
       tests/unit/test_mlp_inference.py \
       tests/integration/test_mlp_trains.py -q
```

These tests use synthetic fixtures (no live network) for architecture, compile
configuration, invalid inputs, actual training, early stopping, history, checkpoint
restoration, raw-record preprocessing, thresholding, save/load consistency, artifact
tampering and real Phase 5 artifact integration when one exists in the checkout.

---

## 12. Deliberate limitations

- No Logistic Regression/Random Forest benchmark, ROC/PR curves, confusion matrices
  or held-out metrics. Those are Phase 7.
- No threshold optimisation or business-cost analysis. The threshold is configurable
  only.
- No SHAP, risk tiers, retention simulation or dashboard.
- Training is CPU-only and intentionally small; no GPU or distributed path.
- The test partition is untouched but therefore also **unevaluated**. No accuracy,
  AUC or business-lift claim is made in Phase 6.
- Model artifacts are trusted only when produced locally and verified by digest.
  Loading third-party Keras files is out of scope.
- `keras` emits a NumPy deprecation warning in this environment; it does not affect
  tests or training.
