# ChurnWise AI — Phase 12 Report

**Phase:** 12 — Integration, Reliability and End-to-End Testing  
**Date:** 2026-10-23  
**Status:** Implemented and verified on the audited machine; limitations listed
below are not hidden.

## 1. What Phase 12 added

Before Phase 12, each stage existed and was individually tested, but no command
proved that a real dataset could travel from discovery to a retention projection
with consistent identities. Phase 12 adds:

| Area | Implementation |
| --- | --- |
| Single command | `python -m churnwise.cli run` composes source, preprocessing, MLP training, benchmark evaluation, SHAP, risk scoring and simulation |
| Offline provenance | `--dataset PATH` copies a committed fixture into the managed content-addressed store and writes an `offline_fixture` manifest; the source file is never modified |
| Durable run manifest | `artifacts/pipelines/<run-id>/run.json` records stage status, timings, outputs, artifact paths, summaries, warnings and structured errors |
| Safe resume | `--resume` requires the same configuration digest and reuses a stage only when all recorded artifacts exist; missing artifacts invalidate that stage and all later stages |
| Cross-phase verification | The runner refuses success if a required train/evaluate/explain/score/simulate artifact is missing |
| Cleanup tool | `python -m churnwise.pipeline.cleanup --run-id <id>` is a dry run; `--delete` removes only run-owned paths inside the artifact/report roots and never shared acquisition data |
| Reliability tests | End-to-end, resume, determinism, rejection, dashboard rendering, import-boundary and cleanup tests |
| Reproducibility | `requirements.lock.txt` pins the audited environment; a fresh install reproduces the full suite |
| Documentation | This report, [RUNBOOK.md](RUNBOOK.md), [PERFORMANCE.md](PERFORMANCE.md), [DEPENDENCIES.md](DEPENDENCIES.md), [INTEGRATION_AUDIT.md](INTEGRATION_AUDIT.md), [INTEGRATION_DEFECTS.md](INTEGRATION_DEFECTS.md) |

The package is [churnwise/pipeline](../churnwise/pipeline/), not a single
`pipeline.py`. The CLI keeps discovery constants separate from pipeline
constants so the documented discovery exit codes remain intact.

## 2. End-to-end evidence

### 2.1 Offline fixture workflow

The committed fixture is `tests/fixtures/mini_churn.csv` (160 rows). The
integration fixture runs the full fast profile with seed 13 and checks the
manifest, SHA-256 provenance, byte identity of the managed copy, target mapping,
split isolation, model/explain/risk hash agreement, benchmark selection
metadata, simulation arithmetic, dashboard rendering, resume and rejection.

Focused command:

```bash
python -m pytest -q tests/integration/test_pipeline_workflow.py
```

### 2.2 Live network workflow

The final live run used:

```bash
python -m churnwise.cli run --fast --seed 42 \
  --run-id phase12-live-final \
  --data-dir .phase12_final_live/data \
  --artifact-dir .phase12_final_live/artifacts \
  --report-dir .phase12_final_live/reports
```

It started from empty project-local data/artifact/report directories, discovered
and acquired OpenML dataset 46915 (5,000 rows), and completed all seven stages in
**58.0 s**. The manifest status is `succeeded`; all stages are `succeeded`.

| Stage | Duration | Key result |
| --- | ---: | --- |
| source | 19.800 s | `openml:46915`; validation `pass_with_warnings`; SHA-256 `87bc2dd7…` |
| preprocess | 0.260 s | 3,000 train / 1,000 validation / 1,000 test; 27 features |
| train | 8.023 s | MLP; 24 epochs; best epoch 24; hidden units `[32, 16]` |
| evaluate | 24.156 s | Random Forest selected on validation F1 at threshold 0.45 |
| explain | 3.952 s | Random Forest; 3 local examples; global top driver `international_plan` |
| score | 0.732 s | 1,000 scored, 10 with drivers, no rejections |
| simulate | 1.216 s | 200 contacted, spend 1,000, expected prevented churn 32.8334 |

The previously broken manifest summaries are correct in this run:
`best_epoch=24`, `hidden_units=[32,16]`, `selected_threshold=0.45`,
`selection_metric=f1`. The old live manifest showing nulls is superseded and
was removed as generated data; the defect and its regression test are recorded
as D-06.

The `.phase12_final_live` tree was a temporary measurement workspace and was
removed after the values above were captured, so it is not in the checkout.
The run id, timings and metrics are recorded here; the same command recreates
the artifact tree under a new directory. No generated artifact is committed.

### 2.3 Live benchmark results

Selection used validation predictions only; the test split was scored once.

| Family | Validation F1 | Test F1 | Test ROC-AUC | Test Brier | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.5905 | 0.6350 | 0.8741 | 0.1460 | 0.64 |
| Random Forest **(selected)** | 0.7985 | 0.8718 | 0.9383 | 0.0390 | 0.45 |
| MLP (Keras) | 0.6614 | 0.7068 | 0.9221 | 0.1091 | 0.80 |

These reproduce the README table to its displayed precision. They are
fast-profile results: 24 epochs, 3-fold benchmark CV and a 40-row SHAP sample.
They are workflow evidence, not a production hyperparameter search.

### 2.4 Risk and simulation

- Risk run `phase12-live-final-risk`: 1,000 validation customers, classification
  threshold 0.45, no rejected rows, tiers low 858 / medium 31 / high 40 /
  critical 71.
- Simulation `phase12-live-final-sim`: 1,000 eligible, 200 selected by budget
  (cost 5 per contact, budget 1,000), 800 excluded for budget exhaustion.
- Expected churn count is **109.4446** (probability mass), expected prevented
  churn is **32.8334** at assumed effectiveness 0.30, targeting lift versus
  random is **3.5929**.
- No monthly value was supplied, so saved value, net benefit, ROI and break-even
  are `null` with reasons. The result is labelled hypothetical and
  `observed_retention_result=false`.

## 3. Tests and checks

| Check | Result |
| --- | --- |
| Full suite, default (live marked tests skipped) | **623 passed, 7 skipped** |
| Clean environment from `requirements.lock.txt` | **623 passed, 7 skipped** in 295.46 s |
| Opt-in live suite (`pytest -q -m live --run-live`) | **7 passed** in 102.44 s |
| Pipeline integration suite | 14 tests, including fixture workflow, provenance, cross-artifact identity, resume, rejection, determinism and dashboard |
| Pipeline contract/boundary suite | 6 tests; phase modules do not import the orchestrator and the orchestrator does not import Streamlit/dashboard |
| Cleanup suite | 4 tests; dry run, confinement, deletion and missing/mismatched manifests |
| `pip check` | no broken requirements |
| Dashboard | AppTest rendered Pipeline artifacts; standalone headless server answered HTTP and pages were browsed in Phase 11/12 |

The full suite includes 1,484 third-party deprecation/runtime warnings (Keras,
gast, NumPy/SHAP). They do not fail tests. The same 623-pass/7-skip result was
reproduced in a new `.venv_clean` installed only from `requirements.lock.txt`
(295.46 s), so the suite no longer depends on the primary environment's
pre-existing packages.

## 4. Reliability and defects

The defect register is [INTEGRATION_DEFECTS.md](INTEGRATION_DEFECTS.md). The
integration-relevant items:

- **D-01/D-02:** no single command and no fixture path with provenance — fixed by
  the orchestrator and managed-fixture acquisition.
- **D-03:** a pipeline constant shadowed the discovery invalid-config constant —
  fixed by aliased imports; the existing CLI regression test passes.
- **D-04:** failed runs could not be resumed — fixed by the manifest, config
  digest and artifact-existence checks.
- **D-05:** simulation probability mass was labelled a probability — fixed with
  an `expected_churn_count` alias, a probability-mass marker, and corrected
  report wording; the legacy key remains for compatibility.
- **D-06:** manifest summaries read wrong nested metadata keys — fixed and
  asserted against the final live run.
- **D-07:** a Phase 9 risk manifest could mix relative and absolute model paths;
  fixed in Phase 13 by portable manifest paths and project-root-relative
  resolution in the loader and dashboard.
- **D-08:** MLP floats are not bitwise deterministic while TensorFlow op
  determinism is disabled; splits, Logistic Regression, scoring, ranking and
  simulation are deterministic. No bitwise MLP claim is made.
- **D-12 (found in Phase 13):** Random Forest prediction with `n_jobs=-1` could
  drift by one ULP in `log_loss`; the default is now serial (`n_jobs=1`) to make
  tree predictions bitwise stable. This was not known during the Phase 12
  measurements, so archived RF artifacts from that phase retain `n_jobs=-1`.
- **D-09:** artifact validation is phase-specific plus principal cross-phase
  assertions; there is no single JSON-Schema registry. Marked mitigated, not
  universally solved.
- **D-10/D-11:** EDA profile timestamp and project-owned fixture licence
  vocabulary; fixed in Phase 13 and covered by regression assertions.

Failure behavior verified:

- A single-class dataset is rejected at V4 in the source stage with
  `CW-RUN-002`; no model directory is created and later stages remain pending.
- Offline mode without a dataset fails before work starts with `CW-RUN-001`.
- A missing fixture and a malformed run id fail before work starts with
  `CW-RUN-001`.
- Resuming with a changed seed fails with `CW-RUN-004` and does not overwrite
  the successful manifest.
- Resuming after deleting a downstream artifact re-runs that stage and later
  stages; the source stage keeps its original timestamp.
- A corrupt manifest is refused rather than overwritten.
- Cleanup refuses paths outside artifact/report roots, paths that do not name
  the run, shared data, and mismatched manifests.

## 5. Performance

Measured on the audited CPU-only Windows machine; full details and budgets are
in [PERFORMANCE.md](PERFORMANCE.md).

- Fixture fast profile: **16.668 s**, **506.6 MB** peak, **2.55 MB** workspace.
- Live fast profile: **58.0 s**, **13 MB** workspace; discovery 11.1 s.
- Dashboard: HTTP ready under 2 s; first AppTest render 1.48 s; server working
  set about 155 MB.

No measured live stage exceeded its design budget. The full production profile
(150 epochs, 5-fold CV, larger SHAP samples) was **not** run end-to-end in this
phase, so no production timing claim is made.

## 6. Dependencies

`requirements.txt` now declares 14 direct packages; all 14 are installed and
importable. The unused `seaborn` and `imbalanced-learn` declarations were
removed. `requirements.lock.txt` pins the 82 distributions of the audited
Windows/CPython 3.13 environment, and a fresh install from that lock reproduced
the full suite. `pip check` is clean. The lock is platform-specific and has no
hash/vulnerability attestation. Details: [DEPENDENCIES.md](DEPENDENCIES.md).

## 7. Acceptance criteria status

| ID | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| T12.1 | Full pipeline offline on fixture | **Pass** | `tests/integration/test_pipeline_workflow.py`, live/fixture artifacts |
| T12.2 | Every artifact validates against a schema | **Pass (scoped method)** | Every fixture JSON artifact is path-classified by `tests/pipeline_contract_helpers.py`; phase contract suites plus principal cross-artifact assertions pass. A universal JSON-Schema registry is not used (D-09) |
| T12.3 | Two identical runs produce identical reusable outputs | **Pass (scoped method)** | Split/model/risk/simulation determinism tests pass; a deterministic MLP run reproduces metrics exactly. Default MLP floats are explicitly excluded (D-08) |
| T12.4 | Failure paths emit documented codes | **Pass for covered paths** | Single-class rejection, invalid inputs, missing run, changed resume, missing artifacts; bad-hash/cold-cache paths are also covered by acquisition suites |
| T12.5 | Import boundaries hold | **Pass** | `tests/unit/test_pipeline_contracts.py` |
| T12.6 | Timings recorded against budget | **Pass** | [PERFORMANCE.md](PERFORMANCE.md) |
| T12.7 | `pytest -q` green from clean `.venv` | **Pass** | Fresh `.venv_clean` from `requirements.lock.txt`: 623 passed / 7 skipped in 295.46 s |

## 8. Limitations carried forward

1. The full production MLP/benchmark/SHAP profile was not run on live data.
2. MLP training is not bitwise reproducible; downstream scoring and simulation
   are deterministic for fixed artifacts. Random Forest's serial default fixed
   the Phase 13 one-ULP prediction drift (D-12).
3. There is no universal JSON-Schema registry for every artifact.
4. The lock is platform-specific (Windows/CPython 3.13) and unpinned by hash;
   other platforms must use the range manifest or regenerate the lock.
5. Live tests and live runs depend on external network availability and are
   therefore evidence for this date, not a permanent guarantee.
6. Cleanup has no cross-process lock; it assumes a single-user, single-machine
   workflow and refuses to delete shared acquisition data by design.

Phase 13 subsequently verified every acceptance criterion with artifact
references, refreshed the README and architecture docs, and produced the final
demo; see [FINAL_VERIFICATION_REPORT.md](FINAL_VERIFICATION_REPORT.md) and
[FINAL_ACCEPTANCE_MATRIX.md](FINAL_ACCEPTANCE_MATRIX.md). The fast-profile metrics
in this report remain workflow evidence, not a production claim.
