# ChurnWise AI — How to Run

This guide takes a fresh checkout to a working dashboard, a reproducible offline
pipeline run and the test suites. Commands are derived from the repository's
actual CLI, configuration and test setup.

If you only want the quickest working demo, follow
[§C](#c-create-and-activate-the-environment), [§D](#d-install-dependencies),
[§F](#f-run-the-application) and the first command in
[§G](#g-acquire-data-and-run-the-ml-pipeline).

## A. Prerequisites

| Requirement | Status and guidance |
| --- | --- |
| Operating system | **Verified:** Windows 11 with CPython 3.13.0. The Python code and `requirements.txt` are platform-neutral, but this repository has **not** been executed end to end on macOS or Linux. Treat non-Windows commands as untested. |
| Python | **CPython 3.13** in an isolated environment. The audited lock was generated with CPython 3.13.0; `requirements.txt` also declares the package ranges. |
| Git | Needed only to clone the repository. |
| Network | Required for live dataset discovery and acquisition. **Not** required for the committed offline fixture or for opening a dashboard over existing artifacts. |
| Browser | Any modern browser for the dashboard. Optional Playwright E2E needs a Chromium-based browser (bundled Chromium, Edge or Chrome). |
| Disk and memory | The reference full-profile artifacts occupy about **35 MB**. The Python environment, TensorFlow and browser binaries need more; this audit did not measure the installed environment size. The observed full-profile run peaked at **613.9 MiB** across its process tree, so allow at least 2 GB of free RAM in practice. |
| Services | None. No database, cloud account, API key, Docker or Node.js is required. |

Run every command from the repository root unless a section says otherwise.
Relative paths such as `tests/fixtures/mini_churn.csv` and
`churnwise/dashboard/app.py` depend on that working directory.

## B. Clone the Repository

```bash
git clone https://github.com/GangNinja/ChurnWise-AI.git
cd ChurnWise-AI
```

`README.md`, `docs/` and `tests/` should be visible in the current directory.

## C. Create and Activate the Environment

### Windows PowerShell (verified)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

`python --version` must report Python 3.13.x. If PowerShell blocks activation,
you can either allow activation for this shell only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

or call the environment's interpreter directly without activating:

```powershell
.\.venv\Scripts\python.exe --version
```

### macOS / Linux (not verified in this repository)

Only use this if your system provides CPython 3.13:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python --version
```

### Which dependency file is authoritative?

- **`requirements.lock.txt`** pins exact versions for the audited **Windows /
  CPython 3.13** environment. Use it when you want the closest reproduction of
  the tested environment on Windows.
- **`requirements.txt`** contains compatible version ranges and is the
  cross-platform manifest. Use it on macOS/Linux or when the exact Windows
  versions are unavailable.
- Neither file is a `pyproject.toml` project. Do not run `pip install -e .`; this
  repository is run from the checkout.

The lock is platform-specific and has no hash pins. See
[DEPENDENCIES.md](DEPENDENCIES.md) for the dependency policy.

## D. Install Dependencies

With the environment activated on Windows:

```powershell
python -m pip install -r requirements.lock.txt
```

Cross-platform alternative:

```powershell
python -m pip install -r requirements.txt
```

The same command works in an activated macOS/Linux environment. This installs
NumPy, pandas, scikit-learn, TensorFlow/Keras, SHAP, Streamlit, Plotly,
Matplotlib, joblib, PyArrow, pytest and the Playwright Python package.

No separate TensorFlow or Streamlit installation step is needed. TensorFlow
prints oneDNN and CPU-feature messages at import time; those informational
warnings do not by themselves mean installation failed.

### Optional browser binary for E2E

The Playwright Python package is installed with the requirements, but its
browser binary is not part of the Python lock. The E2E suite first tries bundled
Chromium, then the installed Microsoft Edge, then Chrome. If none is available:

```powershell
python -m playwright install chromium
```

You can force an installed channel instead:

```powershell
$env:CHURNWISE_BROWSER_CHANNEL="msedge"   # or "chrome"
```

## E. Configure the Application

**Default workflow: no environment variables, no config file edits and no
credentials are required.** `churnwise.config.Settings.load()` reads optional
`CHURNWISE_*` variables and otherwise roots `data/`, `artifacts/` and `reports/`
at the current working directory.

Supported overrides include:

| Variable | Purpose | Default |
| --- | --- | --- |
| `CHURNWISE_DATA_DIR` | Raw data, cache and manifests | `./data` |
| `CHURNWISE_ARTIFACT_DIR` | Prepared data, models, evaluations, risk and simulations | `./artifacts` |
| `CHURNWISE_REPORT_DIR` | Discovery, validation and EDA reports | `./reports` |
| `CHURNWISE_OFFLINE` | Disable network access where supported | `false` |
| `CHURNWISE_SEED` | Default random seed | `42` |
| `CHURNWISE_RISK_TIERS` | Comma-separated risk-tier boundaries | `0.30,0.50,0.70` |
| `CHURNWISE_HTTP_TIMEOUT_S`, `CHURNWISE_DOWNLOAD_TIMEOUT_S` | Network timeouts | `12`, `60` |
| `CHURNWISE_MAX_RETRIES` | Network retry count | `2` |
| `CHURNWISE_ALLOW_UNVERIFIED_LICENSE` | Allow candidates without verified licence metadata | `false` |

The full settings, including size/archive limits and user agent, are in
[`churnwise/config.py`](../churnwise/config.py) and
[DISCOVERY.md](DISCOVERY.md).

`.streamlit/config.toml` only sets the light dashboard theme and disables
Streamlit's automatic page navigation. It is not an application config.

### Important: keep pipeline and dashboard directories aligned

The CLI accepts `--data-dir`, `--artifact-dir` and `--report-dir`. The dashboard
reads the directories resolved from `Settings.load()`, not the CLI flags of an
earlier process. If you use custom directories for a pipeline run, set the same
`CHURNWISE_*` variables before launching Streamlit.

For example, with the CLI run in §G using the default directories, launch the
dashboard with no overrides. If you instead ran into `demo/`, launch it with:

```powershell
$env:CHURNWISE_DATA_DIR="demo/data"
$env:CHURNWISE_ARTIFACT_DIR="demo/artifacts"
$env:CHURNWISE_REPORT_DIR="demo/reports"
python -m streamlit run churnwise/dashboard/app.py
```

### Dataset and model expectations

No real source dataset or trained model binary is committed. The repository
contains a small synthetic workflow fixture,
`tests/fixtures/mini_churn.csv`, and generated upload examples under
`testing_datasets/`. The upload examples use the telecom schema and are not
compatible with a model trained on the small fixture. The real reference
dataset is acquired from OpenML on a live run.

The dashboard can start without artifacts; it shows actionable empty states. To
populate it, run at least the offline fixture pipeline once (§G). Do not copy a
model from another run or another machine and assume it is compatible: the
dashboard verifies model, preprocessing and explanation provenance.

## F. Run the Application

First create artifacts with the offline fixture command in §G. Then, from the
repository root:

```powershell
python -m streamlit run churnwise/dashboard/app.py
```

Streamlit normally prints:

```text
Local URL: http://localhost:8501
Network URL: http://<local-ip>:8501
```

Open `http://localhost:8501`. A healthy server answers
`http://localhost:8501/_stcore/health` with `200 OK`. Stop it with `Ctrl+C` in
the terminal.

To choose another port:

```powershell
python -m streamlit run churnwise/dashboard/app.py --server.port 8502
```

The default bind address is local; the dashboard is a local single-user
application with no authentication. Do not expose it to an untrusted network.

The dashboard reads persisted artifacts. Ordinary page renders do not discover,
download, train, score or compute SHAP; those operations require explicit
buttons. The first render over a real artifact tree was measured at 1.937 s in
the Phase 13 environment.

### Try the bundled customer CSVs

The examples in `testing_datasets/` use the project's telecom feature schema.
They require a fitted model/preprocessing pipeline built from the matching
telecom dataset (OpenML `46915`). The 160-row `tests/fixtures/mini_churn.csv`
workflow is useful for an offline app smoke test, but it has a different schema
and is not compatible with these telecom CSVs. If you do not already have the
matching artifacts, run the live workflow first (network required):

```powershell
python -m churnwise.cli run --seed 42 --run-id upload-demo
```

Then open **Customer Risk Explorer → Upload & predict dataset**. To use the
valid 1,800-row sample, either choose **Proper · testing4.csv** under **Or try a
bundled dataset**, or upload
`testing_datasets/proper_datasets/testing4.csv` from the file picker. Click
**Score this dataset**. The resulting saved run is selected for Executive
Dashboard insights; click **Open Executive Dashboard insights** or navigate
there and select the newly labeled uploaded run. Upload runs are also available from other
scoring-run selectors.

For a schema-rejection demo, choose **Improper · testing4.csv** or upload
`testing_datasets/improper_datasets/testing4.csv`. It is missing the required
`voice_mail_plan` column, so the page reports the mismatch and scores no rows.
The other valid and invalid files range from 1,000 to 3,000 rows. All records
are generated examples; their optional `CustomerChurned` column is ignored by
prediction and is not evidence of observed outcomes. See
[`testing_datasets/README.md`](../testing_datasets/README.md) for the complete
list and each invalid file's missing column.

### Read the Campaign Simulator charts

In **Campaign Simulator**, the scoring-run selector chooses the population for
the projection. **Autofill example assumptions** fills a varied draft but does
not compute anything; click **Run projection** to apply it. The charts directly
below the outcome KPIs show the financial value bridge and either the attached
scenario comparison or the population's risk-tier distribution. Enabling
**Compare with a higher-budget alternative** unlocks the alternative budget and
effectiveness inputs. Choose the value unit that matches your assumptions:
sample runs may use generic `value_units`, which should not be described as a
real currency. Projection values are hypothetical, not measured campaign
results.

## G. Acquire Data and Run the ML Pipeline

### Offline fixture: fastest way to populate the dashboard

This command needs no network and completes in about **53 seconds** on the
audited laptop with `--fast`:

```powershell
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv --fast --seed 13 --run-id getting-started
```

It copies the fixture into the managed store, then runs:

```text
source → preprocess → train → evaluate → explain → score → simulate
```

A successful run ends with a JSON summary and writes
`artifacts/pipelines/getting-started/run.json` with `"status": "succeeded"` and
seven succeeded stages.

Run IDs must be 3–110 characters of lowercase letters, digits, dots or dashes,
and must start with a letter or digit. Reuse a run ID only when you intend to
resume it; for a fresh experiment use a new ID. The `--data-dir`, `--artifact-dir` and `--report-dir`
options select an isolated tree, but the dashboard must be pointed at the same
tree through the matching `CHURNWISE_*` variables (§E).

The fixture has only 160 rows. Its metrics are **workflow evidence, not model
performance evidence**.

### Live OpenML run: actual reference dataset

Requires network access:

```powershell
python -m churnwise.cli run --seed 42 --run-id reference-full
```

This is the full profile: 150 configured MLP epochs with early stopping, 5-fold
benchmark cross-validation and larger explanation samples. The observed
full-profile run took **328.259 s** with a **613.9 MiB** peak and completed all
seven stages. OpenML availability and network speed can change that.

For a quicker live demonstration, use `--fast`:

```powershell
python -m churnwise.cli run --fast --seed 42 --run-id demo-live
```

Do not present fast-profile metrics as production or benchmark evidence. The
repository's reference metrics come from the committed benchmark profile.

### Resume and stage-level commands

Resume only with the original run ID and identical configuration:

```powershell
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv --fast --seed 13 --run-id getting-started --resume
```

A changed dataset, seed, profile or scenario is refused with `CW-RUN-004`
instead of silently mixing work.

The individual phases remain available for diagnosis and focused work:

```powershell
python -m churnwise.cli discover
python -m churnwise.cli acquire
python -m churnwise.cli validate
python -m churnwise.cli prepare
python -m churnwise.cli train
python -m churnwise.cli evaluate --cv-folds 5
python -m churnwise.cli explain
python -m churnwise.cli score --population val --explain all
python -m churnwise.cli simulate --budget 1000 --cost-per-contact 5 --effectiveness 0.30
```

These stages depend on the outputs of earlier stages and are intended for
advanced use. See the [RUNBOOK.md](RUNBOOK.md) for resume, failure codes,
artifact verification and cleanup, and [README.md](../README.md) for the output
of each command.

### Generated artifacts

| Location | Contents |
| --- | --- |
| `data/raw/` | Verified downloaded or copied datasets |
| `data/cache/` | Content-addressed acquisition cache |
| `data/manifests/` | Dataset provenance and acquisition manifests |
| `reports/acquisition/` | Candidate ranking and acquisition attempts |
| `reports/validation/` | Validation gate JSON and Markdown reports |
| `reports/eda/` | Dataset profiles and plots |
| `artifacts/prepared/<run-id>/` | Split matrices, schema, fingerprints and fitted preprocessing pipeline |
| `artifacts/models/<run-id>/mlp/` | Keras model, metadata, history and checkpoints |
| `artifacts/evaluation/<run-id>/` | Benchmark metrics, predictions, curves, plots and manifest |
| `artifacts/explanations/<run-id>/` | Global and local SHAP artifacts |
| `artifacts/predictions/<run-id>/` | Risk scores, tiers, drivers and exports |
| `artifacts/simulations/<scenario-id>/` | Scenario result, selection, sensitivity and report |
| `artifacts/pipelines/<run-id>/run.json` | Durable seven-stage pipeline manifest |

All of these directories are Git-ignored and reproducible. Only source, tests
and documentation are tracked.

## H. Use Dashboard Features

The app has five product pages. The sidebar is the only navigation; Streamlit's
automatic page list is disabled by `.streamlit/config.toml`.

| Page | What to use it for | Prerequisite |
| --- | --- | --- |
| **Executive Dashboard** | Scored customers, predicted churn, risk distribution, dataset/model facts and the ten highest-risk customers | A population risk run |
| **Customer Risk Explorer** | One stored assessment with drivers and a probability-matched explanation; explicit individual and batch scoring | A risk run; model and preprocessing artifacts for new scoring |
| **Model Laboratory** | Held-out metrics, confusion matrix, ROC/PR, CV stability, global SHAP, data validation, provenance, artifact integrity and pipeline operations | Benchmark artifacts; validation and explanation artifacts for their tabs |
| **Retention Workbench** | Filter, sort, fit a contact list to a budget and export selected or matching rows | A risk run from `score` |
| **Campaign Simulator** | Compare hypothetical campaigns under explicit assumptions | A risk run; assumptions are user-supplied |

Useful workflow:

1. Start on **Executive Dashboard** and confirm the run ID, predicted churn and
   high/critical counts.
2. Open **Model Laboratory → Performance** and read the held-out test evidence;
   use **Data & validation** to show the gate and failed checks.
3. Open **Retention Workbench**, filter to high/critical customers, set a budget
   and download the selected contact list.
4. Open **Customer Risk Explorer** for one of those customers and review the
   stored drivers and explanation. Use the scoring expanders only when you want
   to score new records.
5. Open **Campaign Simulator**, leave monthly value blank to see undefined
   financial fields, then run a budget-constrained projection.

Pipeline actions (discovery, acquisition, MLP training) are available only under
**Model Laboratory → Operations**, inside collapsed expanders, and run only when
explicitly pressed.

The simulator's output is a projection. It does not measure causal uplift,
observe retention or produce revenue without a user-supplied value assumption.

## I. Run the Tests

### Default suite (offline)

```powershell
python -m pytest -q
```

Verified in the post-audit rerun: **634 passed, 17 skipped**. The 17 skips are
7 live-network tests plus the 10 opt-in browser tests. Timing varies with machine
load: the post-audit rerun took 293.81 s; the earlier Phase 13 runs took 291.80 s
and 527.67 s. The suite trains small models and exercises the pipeline, so allow
several minutes either way.

Focused examples:

```powershell
python -m pytest -q tests/integration/test_pipeline_workflow.py
python -m pytest -q tests/unit/test_pipeline_contracts.py tests/unit/test_pipeline_cleanup.py
```

### Live-network tests (opt-in)

```powershell
python -m pytest -q -m live --run-live
```

Verified result: **7 passed** with network access. Without `--run-live`, these
tests are skipped.

### Browser E2E (opt-in)

Windows PowerShell:

```powershell
$env:CHURNWISE_BROWSER_E2E="1"
python -m pytest tests/e2e -m browser -q
```

macOS/Linux:

```bash
CHURNWISE_BROWSER_E2E=1 python -m pytest tests/e2e -m browser -q
```

Verified result: **10 passed in 49.89 s** on the five-page redesign rerun
(78.33 s on the earlier workstream run). The suite starts its own Streamlit
servers on free ports, uses temporary artifact trees and forces offline mode.
It does not need a pre-existing artifact run or an already-running dashboard.
If no browser is found, install bundled Chromium or set
`CHURNWISE_BROWSER_CHANNEL`.

Browser logs and failure screenshots, when produced, go under
`reports/post_phase13/workstream4/browser-logs/` and
`reports/post_phase13/workstream4/browser-failures/`. They are Git-ignored.

### Documentation checks

```powershell
python scripts/verify_phase1_docs.py
```

This checks the Phase 1 documentation structure and links, not every document in
`docs/`.

## J. Understand the Outputs

The most useful files after a run are:

- `artifacts/pipelines/<run-id>/run.json` — stage status, timings, warnings,
  configuration digest, output identifiers and artifact paths. Start here when a
  run fails.
- `artifacts/evaluation/<run-id>/metrics.json` — machine-readable test metrics,
  thresholds, confusion matrix and selection metadata.
- `artifacts/evaluation/<run-id>/benchmark.md` — human-readable benchmark.
- `artifacts/explanations/<run-id>/global.json` — global SHAP ranking.
- `artifacts/predictions/<run-id>/risk_summary.json` — tier counts and scored
  population summary.
- `artifacts/simulations/<scenario-id>/result.json` — assumptions and projection,
  including whether the result is hypothetical.
- `reports/validation/<dataset>.json` — validation gate decision and checks.

Tracked source and documentation use normal Git rules. Generated datasets,
models, metrics, predictions, plots and pipeline reports under `data/`,
`artifacts/` and `reports/` are ignored by `.gitignore`. The small curated
evidence under `docs/evidence/` is tracked intentionally.

Cleanup is a dry run by default:

```powershell
python -m churnwise.pipeline.cleanup --run-id getting-started
python -m churnwise.pipeline.cleanup --run-id getting-started --delete
```

It removes only run-owned artifacts and reports; shared acquisition data and
dataset-level reports are retained. See [RUNBOOK.md](RUNBOOK.md) §4–5 before
using `--delete`.

## K. Troubleshooting

### `python` is not Python 3.13

**Symptom:** `python --version` reports 3.12 or older, or dependency resolution
fails for TensorFlow.

**Action:** recreate the environment with `py -3.13 -m venv .venv` on Windows
and verify `python --version` after activation. The lock was generated with
CPython 3.13.0.

### PowerShell will not run `Activate.ps1`

**Symptom:** "running scripts is disabled on this system".

**Action:** allow scripts for this shell only with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or bypass
activation and call `.\.venv\Scripts\python.exe` directly. This is a local
PowerShell policy; the repository does not require a permanent machine change.

### Dependency installation fails

- On Windows/CPython 3.13, use `requirements.lock.txt`.
- On another platform, use `requirements.txt`; the lock is Windows-specific.
- TensorFlow wheels are the most likely platform constraint. Check that the
  Python version and architecture have a compatible wheel.
- Do not install standalone `keras`; TensorFlow bundles a compatible Keras 3.

### The dashboard starts but pages say artifacts are missing

This is an empty state, not a crash. Run the offline fixture pipeline in §G, then
press **Refresh artifact cache** in the sidebar or reload the page.

If you ran the pipeline with custom `--data-dir`, `--artifact-dir` or
`--report-dir`, the dashboard is probably looking at the default `data/`,
`artifacts/` and `reports/` directories. Set the matching `CHURNWISE_*`
variables before starting Streamlit.

### Dashboard import or startup failure

Run Streamlit from the repository root so that `churnwise` is importable:

```powershell
python -m streamlit run churnwise/dashboard/app.py
```

If port 8501 is already in use, pass `--server.port 8502`. Use `Ctrl+C` to stop
the existing process rather than starting a second server on the same port.

### Assessment reports a missing model or preprocessing artifact

The assessment page needs a compatible persisted model and pipeline. Run the
pipeline through `evaluate` (or the full `run`) and refresh. A model copied from
an unrelated run is rejected by design; it is not a substitute for compatible
artifacts.

### Live discovery or download fails

Live runs depend on OpenML/UCI/GitHub availability, rate limits and your network.
Use the committed fixture for a deterministic offline workflow:

```powershell
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv --fast --seed 13 --run-id offline-check
```

`--offline` restricts network access where supported, but the guaranteed offline
path is the local `--dataset` workflow. Do not treat a skipped live test as a
pass; set `--run-live` when you intend to test the network path.

### Invalid batch CSV or schema mismatch

The batch uploader validates columns against the schema recorded by the trained
preprocessing pipeline. Use the raw feature columns the pipeline was trained on,
keep the header row, and do not rename columns to match a different run. The
dashboard reports the validation problem instead of silently substituting zeros.
Try a small file first.

### Browser tests are skipped

**Symptom:** `10 skipped` with a message about `CHURNWISE_BROWSER_E2E`.

**Action:** set the variable to `"1"` in the same shell before pytest. The suite
is intentionally opt-in, so the default test run does not require a browser.

### Browser test cannot find a browser

**Symptom:** "No Playwright browser is available".

**Action:** run `python -m playwright install chromium`, or set
`CHURNWISE_BROWSER_CHANNEL=msedge` / `chrome` to use an installed browser.

### ARFF dataset with `STRING` attributes fails

The native reader handles the primary dataset's ARFF types but not OpenML
`STRING` attributes (encountered with dataset `42178`). There is no general fix
in the production reader. Use a CSV/TSV/Parquet input, or the isolated
experiment adapter documented in the second-dataset report. This limitation does
not affect the default primary dataset workflow.

### Tests are slow or the machine runs out of memory

The default suite trains small models and the full pipeline fits an MLP, a
benchmark and SHAP explanations. The observed full-profile peak was 613.9 MiB,
but leave headroom for Python, TensorFlow and the browser. Close other heavy
processes and avoid running multiple training pipelines simultaneously.

## L. Clean Reproduction

From a fresh checkout on the verified Windows/CPython 3.13 setup:

```powershell
git clone https://github.com/GangNinja/ChurnWise-AI.git
cd ChurnWise-AI
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt

# 1. Populate artifacts without network (about 1 minute)
python -m churnwise.cli run --dataset tests/fixtures/mini_churn.csv --fast --seed 13 --run-id getting-started

# 2. Start the dashboard
python -m streamlit run churnwise/dashboard/app.py

# 3. In another terminal, run the standard suite
python -m pytest -q

# 4. Optional browser E2E
$env:CHURNWISE_BROWSER_E2E="1"
python -m pytest tests/e2e -m browser -q
```

For the real reference metrics, replace step 1 with the live full-profile
command:

```powershell
python -m churnwise.cli run --seed 42 --run-id reference-full
```

Do not use the fixture's metrics as model evidence. The committed benchmark and
the full-profile validation used the live OpenML dataset.

## Verification status of this guide

- CLI commands and options were checked against `python -m churnwise.cli --help`,
  `run --help` and the source code on 2026-10-24.
- The dashboard start command is the same command used by the browser E2E suite
  and was exercised against the repository's persisted artifacts during this
  documentation pass.
- The default suite was rerun during the post-audit pass: **634 passed, 17
  skipped** in 293.81 s. The browser suite was rerun at **10 passed** in 78.33 s.
  The earlier Phase 13 evidence remains 624 passed / 17 skipped; the full and
  second-dataset experiments were **not** rerun for this guide.
- A fresh virtual environment was not recreated during this documentation pass.
  The Phase 13 clean-environment install from `requirements.lock.txt` remains the
  recorded clean-install evidence.
- macOS/Linux execution is not verified. The POSIX commands shown are the
  standard equivalents, not a tested support claim.
