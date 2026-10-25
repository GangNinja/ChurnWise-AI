# ChurnWise AI — Demo Script

**Purpose:** a short, reproducible demonstration of the complete system.  
**Audience:** technical and non-technical reviewers.  
**Rule:** do not present fixture metrics or projections as production results.

---

## Suggested 5–8 minute structure

| Time | Section |
| --- | --- |
| 0:00–0:45 | Problem and architecture: why provenance, leakage control and selection discipline matter |
| 0:45–2:00 | Offline pipeline: show seven stages and the manifest |
| 2:00–5:00 | Dashboard: Executive → Customer Explorer → Model Laboratory → Workbench → Campaign Simulator |
| 5:00–6:00 | Explainability and one honest limitation |
| 6:00–7:00 | Validation evidence: full-profile run, calibration boundary, second dataset, browser E2E |
| 7:00–8:00 | What you would do next and questions |

For a shorter review, skip sections 3, 4 and 6 and keep the offline pipeline,
dashboard and validation talking points.

---

## 0. Pre-flight (about 5–10 minutes)

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pytest -q
```

Expected: **634 passed, 17 skipped**. The 17 skips are 7 opt-in live-network
tests and 10 opt-in browser tests. Allow several minutes: the post-audit rerun
took 293.81 s; earlier Phase 13 runs took 291.80 s and 527.67 s under different
machine load. If installation is not possible, use the existing audited `.venv`
and state that the lock is Windows/CPython 3.13-specific.

Optional live-network check:

```bash
python -m pytest -q -m live --run-live
```

Expected: **7 passed** (network required).

Optional browser check:

```bash
CHURNWISE_BROWSER_E2E=1 python -m pytest tests/e2e -m browser -q
```

Expected: **10 passed** in about 81 s. On Windows PowerShell, set
`$env:CHURNWISE_BROWSER_E2E="1"` first. The suite starts its own Streamlit
servers and does not need the demo server to be running.

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
| Executive Dashboard | Run identity, predicted churn vs labelled churn rate, risk distribution, dataset/model facts, top ten customers |
| Customer Risk Explorer | A stored probability with its signed drivers and matched explanation; the upload tab can score the bundled telecom sample when matching model artifacts are available |
| Model Laboratory | Held-out table, confusion matrix and ROC; **Data & validation** shows the gate and failed warning; **Operations** shows that training/discovery are explicit and collapsed |
| Retention Workbench | Tier filter, budget fit and contact-list export |
| Campaign Simulator | Projection disclaimer, varied Autofill, financial value bridge and scenario comparison |

Key talking point: **ordinary page renders do not train, download, score or
compute SHAP.** Only labelled controls under Model Laboratory → Operations and
the scoring expanders invoke services.

The offline `mini_churn.csv` fixture has a different schema from the telecom
upload examples, so skip the bundled CSV upload in this section. The live
OpenML-based run in section 3 builds the matching model pipeline.

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
runs produced identical metrics, tiers and projections. The later full-profile
run `post13-full-20261024c` completed all seven stages in 328 s and reproduced
the committed test metrics, but use `--fast` for a live demo unless you have the
full five minutes.

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

For a batch-upload demonstration, open **Customer Risk Explorer → Upload &
predict dataset**. Select **Proper · testing4.csv** (1,800 synthetic rows) under
**Or try a bundled dataset**, then click **Score this dataset**. The saved upload
run becomes selectable in Executive Dashboard; use **Open Executive Dashboard
insights** to view its population. Next, choose **Improper · testing4.csv** to
show the pre-score schema error for missing `voice_mail_plan`. The optional
`CustomerChurned` values in these generated files are ignored and are not
observed outcome evidence. See [testing_datasets/README.md](../testing_datasets/README.md)
for every sample size and missing column.

On **Campaign Simulator**, show the **Projection charts** beneath the outcome
KPIs: the financial value bridge and projected prevented-churn comparison. Use
**Autofill example assumptions** to vary the draft, then **Run projection** to
apply it. State that the charts are projections from the selected scoring run
and assumptions, not observed campaign results. Sample scenarios may use
generic `value_units`; do not describe them as rupees.

Do **not** present the fast-profile test metrics as the production model result.
The reference benchmark table is in [PROJECT_PRESENTATION.md](PROJECT_PRESENTATION.md#slide-6--models-and-measured-results).

### If live acquisition fails

Do not improvise numbers and do not claim a live run succeeded. Fall back to the
committed fixture pipeline in section 1 or open the tracked evidence in
[docs/evidence/phase13-live2](evidence/phase13-live2/README.md). Say clearly:
“The live source is unavailable right now; this is the previously validated run
and these are its stored artifacts.” A live failure is a source/network
limitation, not evidence that the model failed.

---

## 3b. Validation talking points (about 1 minute)

Keep these four distinctions explicit:

1. **Full profile completed:** seven stages, 328.259 s, 613.9 MiB peak, and the
   selected Random Forest reproduced the committed test metrics.
2. **Calibration is not integrated:** sigmoid improved Brier from 0.0390 to
   0.0306 and ECE from 0.0686 to 0.0301, but the production candidate remains
   the uncalibrated Random Forest at 0.45.
3. **The second dataset is not transfer evidence:** the pipeline was retrained on
   OpenML `42178`; Logistic Regression reached test ROC-AUC 0.8380 on its own
   split. This does not show that the primary model generalises to a new
   population.
4. **Browser E2E is real but opt-in:** 10 tests across all five pages, one
   browser engine and one desktop viewport (plus a 1024 px layout check). It is
   not a visual-accessibility audit and does not cover live training or network
   actions.

Full details: [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md).

---

## 4. Explainability proof (optional, 1 minute)

1. Open **Model Laboratory → Explainability**; show the global ranking is loaded
   from `artifacts/explanations/<run>/global.json`.
2. Open **Customer Risk Explorer**; show a customer probability and its signed
   drivers.
3. State that the values are SHAP attributions, not causal reasons; additivity is
   checked before reporting.
4. For the full-population proof, show:
   `artifacts/explanations/20261022T155236Z-openml_46915-risk-final-drivers/local/`
   contains 1,000 JSON files for the 1,000-row risk run.

---

## 5. Simulation proof (1 minute)

Open **Campaign Simulator**:

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
fixture, including a full-profile run and a real-browser dashboard suite. It is
honest about the boundary: calibration is evaluated but not integrated, the
second dataset is not transfer evidence, the MLP is not bitwise reproducible by
default, financial outcomes are assumptions, and the dashboard is a local
single-user application. The reference results, run IDs and artifacts are in the
repository.”

**Evidence to leave open:** [VALIDATION_SUMMARY.md](VALIDATION_SUMMARY.md),
[PROJECT_CASE_STUDY.md](PROJECT_CASE_STUDY.md) and
[README.md](../README.md). The underlying reports and curated extracts are linked
there; the full run trees remain Git-ignored.
