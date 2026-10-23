# ChurnWise AI — Demo Script

**Purpose:** a short, reproducible demonstration of the complete system.  
**Audience:** technical and non-technical reviewers.  
**Rule:** do not present fixture metrics or projections as production results.

---

## 0. Pre-flight (about 5 minutes)

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pytest -q
```

Expected: **624 passed, 7 skipped** (the seven skips are the opt-in live-network
tests). If installation is not possible, use the existing audited `.venv` and
state that the lock is Windows/CPython 3.13-specific.

Optional live-network check:

```bash
python -m pytest -q -m live --run-live
```

Expected: **7 passed**.

---

## 1. Fast offline end-to-end demo (about 1 minute)

Use the committed synthetic fixture. No network or third-party data is involved.

```bash
python -m churnwise.cli run \
  --dataset tests/fixtures/mini_churn.csv \
  --fast --seed 13 --run-id demo-fixture \
  --data-dir demo/data \
  --artifact-dir demo/artifacts \
  --report-dir demo/reports
```

Show:

1. All seven stages succeed: source, preprocess, train, evaluate, explain, score,
   simulate.
2. The final line prints selected model, scored rows/tiers, and expected
   prevented churn.
3. `demo/artifacts/pipelines/demo-fixture/run.json` is the durable manifest.
4. `demo/reports/validation/fixture-mini-churn-2067f95f.json` records the gate
   decision (`n_rows: 160`, target `churn`, positive label `Yes`).
5. `demo/artifacts/simulations/demo-fixture-sim/result.json` contains
   `projection.projection: true` and `projection.observed_retention_result: false`.

**Expected caveat to say out loud:** this fixture is a workflow test with 160
rows; it is not model-performance evidence. The verified Phase 13 timing for
this command is **53 s**, even with the fast profile, because it still fits an
MLP and runs SHAP.

---

## 2. Dashboard over the offline run (about 3 minutes)

Start the app against the demo tree:

```bash
# Windows PowerShell
$env:CHURNWISE_DATA_DIR="demo/data"
$env:CHURNWISE_ARTIFACT_DIR="demo/artifacts"
$env:CHURNWISE_REPORT_DIR="demo/reports"
python -m streamlit run churnwise/dashboard/app.py

# POSIX
CHURNWISE_DATA_DIR=demo/data \
CHURNWISE_ARTIFACT_DIR=demo/artifacts \
CHURNWISE_REPORT_DIR=demo/reports \
python -m streamlit run churnwise/dashboard/app.py
```

Open the printed URL (default `http://localhost:8501`).

Walk through:

| Page | Demonstrate |
| --- | --- |
| Overview | Run identity, dataset shape, selected model, risk tiers, latest projection |
| Data quality | Validation checks, target mapping, EDA profile and plots |
| Model training | MLP architecture, training history and persisted model |
| Model evaluation | Persisted test table and curves; note RF was selected on validation |
| Explainability | Global ranking and stored local contributions |
| Risk intelligence | Filters, tier counts, rankings and export controls |
| Assess records | Schema-driven form; explain that a submitted record is scored by Phase 9 |
| Customer detail | Probability matching the risk table and stored drivers |
| Retention simulator | Projection disclaimer, budget/effectiveness controls and undefined financial fields |
| Reliability & limits | Artifact inventory, digest checks, warnings and limitations |

Key talking point: **ordinary page renders do not train, download, score or
compute SHAP.** Only labelled buttons invoke services.

---

## 3. Live OpenML demonstration (about 2 minutes, network required)

Run from an empty project-local tree:

```bash
python -m churnwise.cli run --fast --seed 42 \
  --run-id demo-live \
  --data-dir demo-live/data \
  --artifact-dir demo-live/artifacts \
  --report-dir demo-live/reports
```

Verified Phase 13 references: **66 s** pre-fix and **105 s** post-fix wall time,
OpenML `46915`, 5,000 rows, SHA-256 `87bc2dd7…`, Random Forest selected. Both
runs produced identical metrics, tiers and projections.

Show the same dashboard pages with:

```bash
CHURNWISE_DATA_DIR=demo-live/data \
CHURNWISE_ARTIFACT_DIR=demo-live/artifacts \
CHURNWISE_REPORT_DIR=demo-live/reports \
python -m streamlit run churnwise/dashboard/app.py
```

Expected live population:

- 1,000 validation customers scored;
- tiers low 858 / medium 31 / high 40 / critical 71;
- 121 predicted churn at threshold 0.45;
- 200 selected by a 1,000 budget at cost 5, expected prevented churn 32.8334 at
  assumed effectiveness 0.30.

Do **not** present the fast-profile test metrics as the production model result.
The reference benchmark table is in [PROJECT_PRESENTATION.md](PROJECT_PRESENTATION.md#slide-6--models-and-measured-results).

---

## 4. Explainability proof (optional, 1 minute)

1. Open **Explainability**; show the global ranking is loaded from
   `artifacts/explanations/<run>/global.json`.
2. Open **Risk intelligence → Customer detail**; show a customer probability and
   its signed drivers.
3. State that the values are SHAP attributions, not causal reasons; additivity is
   checked before reporting.
4. For the full-population proof, show:
   `artifacts/explanations/20261022T155236Z-openml_46915-risk-final-drivers/local/`
   contains 1,000 JSON files for the 1,000-row risk run.

---

## 5. Simulation proof (1 minute)

Open **Retention simulator**:

- expand **Assumptions and targeting**;
- point out budget, cost per contact, assumed effectiveness and optional monthly
  value;
- show that saved value, ROI and break-even are `undefined` when no value
  assumption is supplied;
- run a new scenario only if time permits, and read the projection banner aloud.

Independent arithmetic check from the saved live result:

```text
sum(selected probabilities) × assumed effectiveness
= 109.444617 × 0.30
= 32.833385 expected prevented churn
```

This is a hypothetical projection, not measured uplift.

---

## 6. Failure and recovery demo (optional, 1 minute)

```bash
python -m churnwise.cli run --run-id demo-fixture --resume \
  --dataset tests/fixtures/mini_churn.csv --fast --seed 13 \
  --data-dir demo/data --artifact-dir demo/artifacts --report-dir demo/reports
```

Show that verified stages are reused. Then change `--seed` and show the
structured `CW-RUN-004` refusal instead of a silent overwrite.

Cleanup is a dry run by default. Point the same `CHURNWISE_*` directory
environment variables at the demo tree, then run:

```bash
python -m churnwise.pipeline.cleanup --run-id demo-fixture
```

---

## 7. Closing statement

“The system is verified end to end on a real public dataset and a committed
fixture. It is honest about the boundary: the live run used the fast profile,
the MLP is not bitwise reproducible by default, financial outcomes are
assumptions, and the dashboard is a local single-process application. The
reference results, run IDs and artifacts are in the repository.”

**Evidence to leave open:** [FINAL_VERIFICATION_REPORT.md](FINAL_VERIFICATION_REPORT.md),
[FINAL_ACCEPTANCE_MATRIX.md](FINAL_ACCEPTANCE_MATRIX.md), and
[PROJECT_PRESENTATION.md](PROJECT_PRESENTATION.md).
