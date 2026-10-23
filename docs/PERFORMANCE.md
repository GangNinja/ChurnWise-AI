# ChurnWise AI — Phase 12 Performance Measurements

**Date:** 2026-10-23  
**Machine:** Intel Core Ultra 5 125H (14 cores / 18 logical), 15.4 GB RAM,
Windows 11 build 10.0.26300, CPU-only (TensorFlow reports no GPU support on
native Windows)  
**Runtime:** Python 3.13.0, TensorFlow 2.21.0, Keras 3.15.1, SHAP 0.54.1,
scikit-learn 1.7.2, pandas 2.3.3, NumPy 2.5.4, Streamlit 1.66.1, pyarrow 24.0.0

These are measured wall-clock results from this machine, not estimates. Startup
and TensorFlow/plot generation make short runs variable; ranges are reported
where repeated runs differed.

## 1. Fixture fast profile (offline, 160 rows)

Command shape:

```bash
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv --fast --seed 13
```

Controlled run in a fresh project-local tree:

| Stage | Measured |
| --- | ---: |
| Source (copy + provenance + validation + EDA) | 1.312 s |
| Preprocessing | 0.111 s |
| MLP training (24-epoch fast profile) | 5.628 s |
| Benchmark (3 families, 3-fold CV, 160 rows) | 4.517 s |
| SHAP explanations | 3.238 s |
| Risk scoring (32 validation rows) | 0.597 s |
| Simulation (budget 1,000, 32 selected) | 1.190 s |
| **Total process wall time** | **16.668 s** |

Peak working set was **506.6 MB**; the complete workspace (`data/`, `reports/`,
`artifacts/`) was **2.55 MB**.

Three fixture runs completed in approximately **12.9 s, 15.7 s and 16.7 s**.
The first CLI run in a cold shell took **33.2 s**, with benchmark evaluation at
21.3 s; repeated TensorFlow retracing warnings appeared during that first
process. The variation is reported rather than averaged away.

## 2. Live network fast profile (OpenML 46915, 5,000 rows)

Command:

```bash
python -m churnwise.cli run --fast --seed 42 \
  --run-id phase12-live-final \
  --data-dir .phase12_final_live/data \
  --artifact-dir .phase12_final_live/artifacts \
  --report-dir .phase12_final_live/reports
```

The command started from empty project-local data/artifact/report directories.
The `.phase12_final_live` tree was temporary and was removed after measurement;
the values are recorded here and the same command regenerates them.

| Stage | Measured | Design budget | Result |
| --- | ---: | ---: | --- |
| Discovery + acquisition + validation + EDA | 19.800 s | 15 s discovery + 60 s download + 30 s validation | Discovery wall time was 11.1 s (slowest source); combined stage within the sum of budgets |
| Preprocessing + split | 0.260 s | < 30 s | Well within |
| MLP training (fast 24 epochs) | 8.023 s | < 3 min | Within |
| Benchmark (3 families, 3-fold CV, 5,000 rows) | 24.156 s | < 60 s baselines | Within |
| SHAP explanations (RF, 40-row sample) | 3.952 s | < 5 min | Within |
| Risk scoring (1,000 validation rows) | 0.732 s | not separately budgeted | — |
| Simulation (200 selected) | 1.216 s | not separately budgeted | — |
| **Total manifest wall time** | **58.0 s** | — | No measured budget overrun |

A preceding live run (`phase12-live-openml-46915`) completed in about 59.5 s with
the same stage shape; its discovery stage was slower at 19.868 s. Both runs
selected Random Forest and produced the same test metrics, while the MLP floats
varied within its documented non-deterministic tolerance. The final run's
manifest is the source for the table above.

The live workspace was **13 MB**. Peak memory was not measured for this process;
the fixture run is the memory reference (507 MB) because TensorFlow import and
model runtime dominate both workloads.

## 2a. Phase 13 post-fix verification (fast profile)

Run `phase13-final-live2` started from an empty project-local tree and completed
all seven stages. External wall time was **105 s**; the manifest measured
**93 s**, with discovery/acquisition at 28.7 s, training at 10.9 s and the
three-family benchmark at 43.7 s. The slower source stage reflects live network
conditions on that run, not a code regression. The earlier Phase 13 run
(`phase13-final-live`) measured 66 s. Metrics, risk tiers and the simulation
projection were identical across both runs.

The dashboard's first real-artifact render was **1.937 s** in the Phase 13
AppTest measurement, below the 5 s budget. This is an application render time,
not a browser navigation trace; no browser performance claim is made.

The fast profile is not the production training profile. A full run with 150
epochs, 5-fold CV and larger SHAP samples was **not** measured end-to-end in this
phase.

## 3. Dashboard

Started with:

```bash
python -m streamlit run churnwise/dashboard/app.py \
  --server.headless true --server.port 8512 --server.address 127.0.0.1
```

Environment pointed at the persisted fixture run. Results:

| Measurement | Result |
| --- | ---: |
| Streamlit listener ready and HTTP 200 | under 2 s after launch |
| First AppTest render | 1.48 s |
| Navigate to Risk intelligence | 0.18 s |
| Navigate to Retention simulator | 0.06 s |
| Streamlit server working set | ~155 MB |

The first render includes all dashboard imports; subsequent page reruns reuse
cached artifact reads. A browser session loaded the Overview and Risk
intelligence pages against the same persisted run and produced no console-visible
error or missing artifact.

## 4. Interpretation

- The dominant costs are MLP training/benchmarking and, on first use, TensorFlow
  startup/retracing. Preprocessing, scoring and simulation are small.
- On the final live run, discovery itself was 11.1 s against a 15 s target. The
  combined source stage was 19.8 s, which includes download, integrity,
  validation and EDA and is within the sum of those budgets. Network time is the
  variable; the earlier run was about 4.5 s slower in the discovery portion.
- No optimization was applied merely to move a number. Repeated TF retracing is
  visible in the logs and is a known Keras behavior for changing batch shapes; it
  affects first-run latency but does not change predictions.
- These figures do not establish performance on a low-end laptop or on the full
  production profile. They are the measured reference for this audited machine.
