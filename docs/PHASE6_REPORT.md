# ChurnWise AI — Phase 6 Report

**Scope:** implement and train the TensorFlow/Keras churn MLP on the Phase 5
preprocessed dataset. No baseline benchmarking, SHAP, threshold optimisation, risk
scoring, campaign simulation or dashboard work was started.

**Status:** implementation and a genuine training run are complete. The model has
**not** been evaluated on the held-out test set; no accuracy or business-performance
claim is made in this phase.

---

## 1. Executive summary

Phase 6 adds a small, fully implemented Keras MLP to ChurnWise:

- input layer sized from the Phase 5 transformed feature count (27 here);
- two configurable ReLU hidden layers (default 64/32), He initialisation, optional
  dropout (default 0.30/0.20) and optional L2 (default 1e-4);
- one sigmoid output unit producing `P(churn = 1)`;
- binary cross-entropy loss, Adam optimiser, configurable learning rate and
  mini-batch training through TensorFlow automatic differentiation;
- validation-loss early stopping with best-weight restoration, LR reduction on
  plateau, `TerminateOnNaN`, and a best-only checkpoint;
- per-epoch history, real loss/accuracy/learning-rate curves, structured
  diagnostics and a verified, reloadable `.keras` artifact;
- reusable inference for transformed matrices and raw customer records through the
  **fitted Phase 5 preprocessor**, with configurable probability thresholds;
- deterministic synthetic tests plus an integration test against the real Phase 5
  artifacts and real CLI train/predict paths.

The reference training run completed **43 of 150 epochs** (early stopped), with best
validation loss **0.365427 at epoch 31**, in **11.75 s** on the audited CPU-only
Windows laptop. Test rows were not used for fitting, early stopping, checkpoint
selection, threshold selection or any metric (`test_rows_used = 0`).

| Verification | Result |
| --- | --- |
| Focused Phase 6 tests | **51 passed** |
| Full project suite | **333 passed, 7 skipped** (live-network tests opt-in) |
| Real training run | **completed**, 43 epochs, early stopped |
| Save/reload prediction consistency | **max absolute difference 0.0** on 32 rows |
| Raw-record CLI inference | **completed**, probabilities in `[0, 1]` |
| Held-out performance | **not measured** — reserved for Phase 7 |

---

## 2. Upstream inspection and hand-off

The implementation was written after reading the Phase 1–5 contracts, docs and the
actual persisted artifacts. It consumes `PreparedDataset`, not a reconstructed or
parallel interface.

| Upstream fact | Value used |
| --- | --- |
| Phase 5 run | `20261021T123405Z-openml_46915-989d8802` |
| Dataset | `openml:46915`, 5,000 raw rows, MIT licence verified |
| Dataset SHA-256 | `87bc2dd76e08b5bffa37fdd97d3d418b41036fec3d4cbf7e31377542b8e85fb6` |
| Split | stratified 60/20/20, seed 42, **3000 / 1000 / 1000** |
| Split fingerprint | `4b0c45db677fa9619de99b6a8b786a3da2c4da93824c46db65412a85829594dd` |
| Preprocessor hash | `3c31f13c266c21084050bb021e60a41d4145a2055b983eb03e2267fab9f52174` |
| Transformed features | **27** |
| Positive class | `1 = churned`, source label `Yes`, mapping `No: 0, Yes: 1` |
| Class counts | train 2576/424, val 858/142, test 859/141 |
| Class weights | `{0: 0.5823, 1: 3.5377}` |
| Test-set status | untouched by the trainer; no test metric computed |

The Phase 5 test partition is retained for Phase 7 evaluation only. `PreparedDataset`
exposes `train`, `val` and `test` separately; the trainer reads only `train` and
`val`. Source customer identifiers are absent from the transformed matrices, and the
model metadata does not add them.

There were no upstream gaps that required inventing an interface. Phase 5 already
provided the fitted `ColumnTransformer`, feature names, target mapping, class
weights, split identity and `transform_new` for inference replay.

---

## 3. Model architecture and justification

| Element | Implementation |
| --- | --- |
| Input | `Input(shape=(d,))`, where `d = len(dataset.feature_names)` |
| Hidden 1 | `Dense(64, ReLU, he_normal, L2=1e-4)` |
| Dropout 1 | `0.30` |
| Hidden 2 | `Dense(32, ReLU, he_normal, L2=1e-4)` |
| Dropout 2 | `0.20` |
| Output | `Dense(1, sigmoid)` — churn probability |
| Parameters | **3,905** for the 27-feature reference input |

Defaults live in `MLPConfig`; widths, dropout per layer, L2, learning rate, batch
size, epoch limit, patiences and threshold are configurable through Python or CLI
flags without editing source. `build_mlp` validates the configuration and input
width, and is independently testable. The architecture is intentionally small for a
5,000-row tabular dataset: two narrow layers can capture non-linear interactions
without a large parameter budget, while dropout and L2 control overfitting. ReLU is
used in hidden layers and sigmoid in the output; the latter matches binary
cross-entropy and yields a probability.

Reference metadata:
[metadata.json](../artifacts/models/20261021T132609Z-openml_46915-mlp-443bb6a1/mlp/metadata.json).

---

## 4. Loss, optimisation and backpropagation

- **Loss:** binary cross-entropy,
  `-(y log p + (1-y) log(1-p))`, averaged over each mini-batch; L2 adds
  `lambda * sum(w^2)`.
- **Optimiser:** Adam with learning rate `1e-3`, using its first/second moment
  estimates and adaptive per-parameter updates.
- **Batch size:** 64 (configurable).
- **Epoch limit:** 150 (configurable).
- **Forward pass:** dense pre-activations -> ReLU -> optional dropout -> sigmoid
  output.
- **Backward pass:** Keras/TensorFlow autodifferentiation computes gradients of the
  total loss with respect to every weight and bias; Adam applies the update. No
  hand-written or simulated training loop is used.
- **Imbalance:** Phase 5 class weights are passed to `fit`, giving the churn class a
  larger contribution. No test or validation rows are resampled.

The full derivation and data flow are documented in [MODELING.md](MODELING.md#4-forward-pass-loss-and-backpropagation).

---

## 5. Regularisation, early stopping and checkpoints

| Mechanism | Setting | Behaviour |
| --- | --- | --- |
| Dropout | 0.30 / 0.20 | Training only; disabled by Keras at validation/inference |
| L2 | 1e-4 | Applied to hidden-layer kernels |
| Early stopping | `monitor=val_loss`, patience 12, `restore_best_weights=True` | Stops when validation loss stops improving |
| LR reduction | `monitor=val_loss`, factor 0.5, patience 5, min 1e-5 | Lowers learning rate on a plateau |
| TerminateOnNaN | enabled | Stops a diverging run |
| Model checkpoint | `best.keras`, `save_best_only=True` | Best validation checkpoint retained |
| Post-fit check | non-finite/ragged history and missing checkpoint | Raised as `CW-MOD-002` |

The reloaded best checkpoint, not the final epoch, becomes the returned model and
`model.keras`. The reference run stopped at epoch 43 after its best validation loss
at epoch 31; the learning rate had fallen from `1e-3` to `1.25e-4` by the last
epoch. No test metric influenced any of these decisions.

---

## 6. Training history and learning curves

Artifacts for the reference run:

- [history.json](../artifacts/models/20261021T132609Z-openml_46915-mlp-443bb6a1/mlp/history.json)
- [learning_curves.png](../artifacts/models/20261021T132609Z-openml_46915-mlp-443bb6a1/mlp/learning_curves.png)
- [training report (Markdown)](../reports/models/20261021T132609Z-openml_46915-mlp-443bb6a1.md)

History contains 43 entries for `loss`, `val_loss`, `accuracy`, `val_accuracy`
and `learning_rate`, all finite and equal length. Selected epochs:

| Epoch | Train loss | Val loss | Train accuracy | Val accuracy | Learning rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.731589 | 0.707348 | 0.6120 | 0.5480 | 0.001000 |
| 31 (best) | 0.359256 | **0.365427** | 0.8910 | 0.8720 | 0.000500 |
| 43 (last) | 0.348019 | 0.368502 | 0.8863 | 0.8740 | 0.000125 |

The PNG (2160 × 540, 3 panels) plots training vs validation loss, training vs
validation binary accuracy, and the Adam learning rate, with titles, axis labels,
units and legends. It is generated only from the run history; no illustrative data
is used. Early stopping and a short/incomplete history are handled without
fabricating points.

These are **training diagnostics**, not held-out performance. The validation
accuracy is not a substitute for Phase 7 test evaluation.

---

## 7. Model artifacts, serialization and inference

### Artifacts

```text
artifacts/models/20261021T132609Z-openml_46915-mlp-443bb6a1/mlp/
├── model.keras              # best-validation model
├── best.keras               # best-only checkpoint
├── history.json
├── learning_curves.png
└── metadata.json
reports/models/20261021T132609Z-openml_46915-mlp-443bb6a1.{json,md}
```

`metadata.json` records the artifact schema version `1.0`, architecture and
parameter count, all training options, seeds, class weights, history metric names,
environment versions (Python 3.13.0,
TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.4), diagnostics, warnings, artifact paths
and the Phase 5 provenance block (run id, dataset SHA-256, split fingerprint,
preprocessor hash/path, feature names, target mapping, provenance).

### Safe persistence and loading

- Run ids are validated against a restricted pattern and resolved under
  `artifact_dir/models`; path traversal is rejected.
- The model SHA-256 recorded in metadata must match the file before loading.
- Metadata schema version and run id are checked for consistency.
- `ChurnPredictor` verifies that the model input width matches the metadata
  feature-name count before accepting the artifact.
- `tf.keras.models.load_model(..., compile=False, safe_mode=True)` is used.
- No arbitrary paths, URLs or third-party models are accepted. The Phase 5
  preprocessor is loaded through its own digest-verifying loader.

### Inference

`ChurnPredictor` supports:

- transformed matrices through `predict_proba(X)` / `predict(X)`;
- raw DataFrames through `predict_frame(frame)` using the frozen Phase 5 pipeline;
- one raw record through `predict_one(record)` returning a `Prediction`;
- batch row correspondence through `PredictionBatch.row_positions`;
- configurable threshold (`>=` comparison), defaulting to the metadata threshold
  (0.5 in this run), with `1 = churned` and `0 = not churned`.

Invalid shapes, non-finite matrix values, missing required raw columns, invalid
thresholds and unavailable preprocessing raise actionable `CW-MOD-004` errors.
Inputs are never logged with customer identifiers.

The trainer also rejects overlapping train/validation/test row indices and malformed
class weights before fitting, so an invalid Phase 5 hand-off cannot silently leak
evaluation rows or produce a misleading run.

The real artifact was reloaded and compared with the in-memory model on 32 rows:
**max absolute probability difference 0.0**, all probabilities finite and within
`[0, 1]`. The CLI predicted three hand-written raw customer rows, producing
`0.1742`, `0.3811` and `0.99998` at threshold 0.5 (one predicted churn). This is an
inference smoke test only, not model-quality evidence.

See [MODELING.md](MODELING.md) for the interface reference.

---

## 8. Integration contract for Phase 7

Phase 7 can:

1. construct the MLP from `MLPConfig` / `build_mlp`;
2. train on the Phase 5 `train` and `val` partitions with `MLPTrainer`;
3. retrieve `TrainingResult.model`, `.history`, `.diagnostics` and `.paths`;
4. reload with `ModelRegistry.load(run_id)` or `ChurnPredictor.load(...)`;
5. predict probabilities/classes for matrices or raw records;
6. read diagnostics and metadata from `result.metadata` / `LoadedModel.metadata`;
7. read learning-curve paths and per-epoch history;
8. trace model -> preprocessing run -> split fingerprint -> dataset SHA-256 ->
   preprocessor hash.

Stable exceptions use `CW-MOD-001` (invalid input/configuration), `CW-MOD-002`
(training failure), `CW-MOD-003` (artifact failure) and `CW-MOD-004` (prediction
failure). These are documented in [MODELING.md](MODELING.md#9-integration-contract-for-phase-7-and-later).

Phase 7 must evaluate **once** on the untouched test rows and compare the MLP with
the other model families on the identical matrix and split. Phase 6 deliberately
does not choose a business threshold or report comparative metrics.

---

## 9. Tests run and actual results

### Focused Phase 6 tests

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m pytest \
  tests/unit/test_mlp_architecture.py \
  tests/unit/test_mlp_training.py \
  tests/unit/test_mlp_inference.py \
  tests/integration/test_mlp_trains.py -q
```

Result: **51 passed, 209 warnings in 28.58 s**. The warnings are Keras's NumPy
`__array__` deprecation warning in this environment; no test failed or was skipped.

Coverage includes:

- architecture shape, ReLU hidden layers, sigmoid output, He initialisation,
  dropout/L2 presence and absence;
- binary cross-entropy, Adam and configured learning rate;
- invalid architecture, malformed layer/dropout containers, input width and threshold validation;
- malformed class weights and overlapping train/validation/test row indices rejected before fitting;
- model metadata run-id and input-width consistency checks;
- actual training on a deterministic synthetic dataset (no mocked training);
- finite, equal-length history and actual epoch counts;
- early stopping below the epoch limit with best-checkpoint restoration;
- persisted model, checkpoint, history, curves, metadata and reports;
- class weights reaching `fit`, training-only oversampling reflected in actual training-row diagnostics, and no test rows in fitting or selection;
- non-finite features, invalid labels and feature/label length mismatches;
- probability bounds and row order, threshold-dependent classes;
- individual and batch raw-record inference through a fitted Phase 5 pipeline;
- missing raw columns and infinite-value imputation;
- save/load prediction consistency within `atol=1e-6`;
- tampered artifact digest rejection;
- CLI train -> predict round trip and missing-model/bad-threshold exit codes;
- integration with the actual persisted Phase 5 run when present in the checkout.

### Full project suite

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m pytest -q
```

Result: **333 passed, 7 skipped in 75.00 s**. The seven skips are the live-network
tests (`--run-live` opt-in) in acquisition, discovery, preprocessing and validation.
They are pre-existing by design, not Phase 6 failures.

---

## 10. Real training evidence

Command:

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m churnwise.cli train \
  --preprocessing-run-id 20261021T123405Z-openml_46915-989d8802
```

Recorded outcome:

| Field | Value |
| --- | --- |
| Model run | `20261021T132609Z-openml_46915-mlp-443bb6a1` |
| Code version | `0.5.0` |
| Preprocessing run | `20261021T123405Z-openml_46915-989d8802` |
| Dataset | `openml:46915` |
| Input features | 27 |
| Parameters | 3,905 |
| Train / validation rows | 3,000 / 1,000 |
| Test rows used | **0** |
| Epochs requested / completed | 150 / **43** |
| Best epoch / best validation loss | 31 / **0.3654266894** |
| Early stopped | **yes** (patience 12) |
| Max epochs reached | no |
| Training duration | **11.75 s** |
| Warnings | TensorFlow op determinism disabled (seeds fixed) |

The model, best checkpoint, history, curves and metadata all exist at the paths
above. The training report was generated automatically from that run:
[report](../reports/models/20261021T132609Z-openml_46915-mlp-443bb6a1.md).

**Repeatability observation.** The same configuration and seed were trained three
times on this host. Each run recorded 43 epochs and best validation loss 0.365427 at
epoch 31. That is evidence of stable behaviour here, not a bitwise reproducibility
guarantee; op determinism remains off by default and is documented as a warning in
metadata. See [MODELING.md](MODELING.md#10-reproducibility-and-nondeterminism).

**No test metrics were computed.** The trainer's diagnostics record
`test_rows_used: 0`; no accuracy, ROC-AUC, PR-AUC, confusion matrix or threshold
sweep on test data exists in this phase.

---

## 11. Known issues and unverified acceptance criteria

| Item | Status / limitation |
| --- | --- |
| Held-out test performance | **Unverified by design** — Phase 7 evaluates once on the untouched test rows |
| Threshold tuning / business cut-off | **Not implemented** — configurable threshold only; Phase 7 owns selection |
| Comparative model quality | **Not measured** — no LR/RF baselines, CV or benchmark in Phase 6 |
| Probability calibration | **Not assessed** — reserved for Phase 7 |
| Bitwise reproducibility across machines | **Not guaranteed** — oneDNN/threading/version differences; `--deterministic` is opt-in and still not a cross-version guarantee |
| GPU training | **Not exercised** — CPU-only TensorFlow on this host; GPU is unavailable in this environment |
| Keras NumPy deprecation warning | **Cosmetic** — emitted by Keras 3.15.1 / NumPy; no failure |
| Live-network tests | **Skipped** in the default suite by design; Phase 6 itself uses only local Phase 5 artifacts |
| Artifact portability | Metadata stores absolute local paths; downstream should use run ids and resolve paths through Settings rather than copying paths across machines |
| `best.keras` retention | Kept for audit; `model.keras` already contains the best-validation weights |
| Test-set exploration | No test labels or metrics were used. Inference smoke checks scored a few unlabeled feature rows only to verify bounds and reload consistency; this was not model selection or evaluation. |

All Phase 6 completion criteria are met for the implemented interface. The only
intentionally unverified item is predictive quality, because this phase must not use
the test set.

---

## 12. Files and artifacts delivered

| Area | Files |
| --- | --- |
| Model code | `churnwise/models/{contracts,errors,mlp,trainer,registry,inference,reporting,__init__}.py` |
| CLI | `churnwise/cli.py` (`train`, `predict`) |
| Version | `churnwise/__init__.py` (`0.5.0`) |
| Tests | `tests/unit/test_mlp_architecture.py`, `test_mlp_training.py`, `test_mlp_inference.py`, `tests/integration/test_mlp_trains.py`, `tests/models_helpers.py` |
| Documentation | `docs/MODELING.md`, this report, `docs/ARCHITECTURE.md` §8.2/§12, `docs/ROADMAP.md` Phase 6 status |
| Reference artifacts | `artifacts/models/20261021T132609Z-openml_46915-mlp-443bb6a1/mlp/*`, `reports/models/20261021T132609Z-openml_46915-mlp-443bb6a1.{json,md}` |

No source data was overwritten, no destructive Git operation was performed, and no
Phase 7+ implementation was started. `requirements.txt` already declared
`tensorflow>=2.20,<2.22` and `matplotlib>=3.9,<4.0`; Phase 6 added no new package
dependency.

---

## 13. Completion procedure check

| Procedure step | Outcome |
| --- | --- |
| Implement Phase 6 only | Done; baseline/SHAP/ranking/dashboard code not added |
| Run focused model tests and Phase 5 integration | 51 passed |
| Execute a genuine training run | Done, 43 epochs, real artifacts |
| Fix failures and rerun affected tests | Fixed compile-metric assertion, early-stopping fixture, plot panel newline; reran focused and full suites |
| Separate pre-existing failures from regressions | No new failures; 7 live-network skips are by design |
| Inspect saving, loading, inference, leakage | Digest-verified reload, threshold/raw-record tests, `test_rows_used=0` |
| Update the Phase 6 report with verified evidence | This document |
| Stop after Phase 6 | Yes |
