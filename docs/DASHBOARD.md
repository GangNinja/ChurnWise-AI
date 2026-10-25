# Interactive Dashboard — Product Structure and Usage

Status: **five-page product shell implemented over the existing Phase 1–13
pipeline**. The redesign changes presentation and navigation only: the model,
threshold, preprocessing, artifacts and scoring services are unchanged, and the
same persisted evidence backs every page. The five-page shell's first `AppTest`
render measured **0.917 s** against the project artifacts on 2026-10-24; no
browser performance trace is claimed.

> The dashboard reads persisted artifacts. It does not invent metrics, sample
> data, training curves or campaign outcomes. Where an artifact is missing, the
> page states what is unavailable and names the command that produces it.

## 1. Stack and startup

| Component | Verified version |
| --- | --- |
| Python | 3.13 |
| Streamlit | 1.66.1 |
| Plotly | 6.9.0 |
| pandas | 2.3.3 |
| NumPy | 2.5.4 |

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# POSIX:   source .venv/bin/activate
python -m pip install -r requirements.txt

python -m streamlit run churnwise/dashboard/app.py
```

Headless check:

```bash
python -m streamlit run churnwise/dashboard/app.py \
  --server.headless true --server.address 127.0.0.1 --server.port 8507
curl -I http://127.0.0.1:8507/
```

Configuration uses the same `Settings` object as the CLI (`CHURNWISE_DATA_DIR`,
`CHURNWISE_ARTIFACT_DIR`, `CHURNWISE_REPORT_DIR`, `CHURNWISE_OFFLINE`,
`CHURNWISE_RISK_TIERS`, …). No secrets are required.

## 2. Five pages, one workflow

| # | Navigation label | Module | Purpose | Primary artifacts |
| --- | --- | --- | --- | --- |
| 1 | Executive Dashboard | `pages/executive.py` | Situation summary: scored customers, predicted churn, risk distribution, dataset/model facts, priority customers | validation, benchmark, population risk summary/table |
| 2 | Customer Risk Explorer | `pages/explorer.py` | One stored assessment, its drivers and explanation; explicit individual and batch scoring | population risk run, `drivers.json`, local SHAP, fitted schema |
| 3 | Model Laboratory | `pages/laboratory.py` | Held-out performance, CV stability, global explanations, data quality, provenance, pipeline operations | benchmark metrics/curves, explanations, validation/EDA, discovery/attempts, prepared/model metadata |
| 4 | Retention Workbench | `pages/workbench.py` | Filter, rank, fit customers to a budget, select and export a contact list | population risk table + summary |
| 5 | Campaign Simulator | `pages/simulator.py` | Compare hypothetical scenarios from stored probabilities and explicit assumptions | `artifacts/simulations/<run>/`; live Phase 10 engine for new projections |

Every page header renders four identity fields: run id, model family, dataset
hash prefix and generated-at timestamp. Missing fields show `not available`.

### Executive Dashboard

The page is an interactive workspace over one persisted population run, in this
order:

- Compact header, run details expander and a scoring-run selector.
- Four KPIs: customers scored, predicted churn at the classification threshold,
  high + critical tier membership and mean stored probability.
- Side-by-side historical and predicted scope panels. The labelled dataset
  (5,000 records, 707 observed churn, 14.1%) is kept separate from the scored
  population; neither count is presented as the same population.
- Compact filters: reference search, tier pills, probability range, a
  high/critical toggle and reset.
- Two interactive Plotly charts. Bars report stored tiers and 0.05 probability
  bins; dragging across a bar filters the table, the selected bar is highlighted
  and a clear control releases the selection. Filters never retrain or rescore.
- Three data-derived insights: largest tier, high/critical share and probability
  concentration below 0.20 with the median.
- A 15-row highest-risk table with verified original attributes (account length,
  international plan, service calls) when the split join is valid. Selecting a
  row exposes **Open assessment** and **Open Retention Workbench** actions.
- A compact selected-model strip with test F1, ROC-AUC and threshold, plus a
  navigation action to Model Laboratory.

### Customer Risk Explorer

Three tabs share one scoring-run selector:

1. **Stored customers** — customer selector, profile of original account
   attributes, prominent probability/risk card (tier text, predicted class,
   threshold and reference, so colour is never the only signal), and a stored
   SHAP contribution chart. Original attributes are shown only after the
   dataset digest, split fingerprint, preprocessing run and row count all agree;
   otherwise the page states that the join is unavailable. Attribution tables,
   the probability-matched local explanation and artifact metadata stay in a
   collapsed expander.
2. **Score a customer** — schema-driven form using the fitted Phase 5 pipeline.
   Blank values use the pipeline's recorded imputation; local SHAP is optional
   and off by default. The result is session state, not a stored artifact.
3. **Upload & predict dataset** — CSV upload or bundled synthetic dataset,
  schema preview, identifier hashing, explanations for a selected subset,
  rejection reasons and export. Scoring happens only on submit.

In **Upload & predict dataset**, use **Or try a bundled dataset** to select one
of the six valid or six schema-invalid examples in `testing_datasets/`. The app
previews row/column counts and checks required columns before scoring. A
successful score creates a uniquely identified input-population risk run; that
run is immediately selected for Executive Dashboard insights and appears in the
scoring-run selectors. The fitted model and preprocessing pipeline are reused;
this action does not retrain the model. An optional `CustomerChurned` column is
ignored as an input and does not turn predictions into observed outcomes.

### Model Laboratory

Four tabs keep technical evidence off the business pages:

1. **Performance** — compact family comparison, selected-model status line,
   confusion matrix, ROC (PR collapsed), cross-validation chart and fold-safety
   table, prepared-data/MLP-history expander, calibration boundary and benchmark
   limitations.
2. **Explainability** — global SHAP importance for the selected explanation run,
   feature-group caption, importance table and explanation limitations.
3. **Data & validation** — gate status, four compact dataset facts, failed-check
   table with reconciled counts, all-checks expander, feature roles, column
   profile, discovery/acquisition provenance.
4. **Operations** — collapsed discovery, acquisition and MLP-training actions,
   plus the artifact inventory and digest verification. These are the only
   pipeline actions in the product and each runs only on an explicit press.

### Retention Workbench

- Filters: stored tier, probability range, customer-reference search.
- Sort: risk rank, probability or value at risk.
- Capacity planning: budget, uniform contact cost and optional contact cap;
  customers are taken in risk-rank order with a deterministic greedy pass.
  Metrics show matching customers, selected count, planned spend and remaining
  budget; the page states that this is not a causal retention estimate.
- Compact prioritised table: selected flag, customer, probability, tier,
  predicted class and service calls. Rank and explanation metadata stay out of
  the default view but remain in the exports. Selecting a row opens that
  customer's stored assessment.
- Export: selected rows (CSV/JSON) or the full matching list (CSV/JSON), with the
  Phase 9 spreadsheet-formula guard.

### Campaign Simulator

- Assumptions: scoring run, budget, cost per contact, assumed
  effectiveness (labelled as an input, not measured uplift), horizon, targeting
  strategy, probability floor, tiers, contact cap, optional monthly value and
  value unit, plus an optional higher-budget comparison scenario.
- **Autofill example assumptions** varies the draft inputs on each click. It
  does not run a projection; **Run projection** applies the draft and computes
  the results. Toggling the higher-budget comparison enables its budget and
  effectiveness controls.
- Outputs: selected customers, allocated spend, remaining budget and projected
  prevented churn. Saved value, net benefit, ROI and break-even appear only when
  a monthly value is supplied; otherwise the UI says the financial projection
  is unavailable.
- Charts directly below the outcome KPIs: a financial value bridge (expected
  saved value − allocated spend = net benefit), a prevented-churn scenario
  comparison when comparison rows exist, or the selected population's risk-tier
  distribution when they do not. These visuals use the active scoring run and
  applied projection; stale persisted comparisons are not reused after a new
  projection.
- Budget allocation chart, population reconciliation table, assumptions-applied
  table, sensitivity (collapsed), scenario comparison table, selected customers
  (collapsed), warnings/limitations and JSON export.
- A persistent disclaimer marks every retention and financial value as
  hypothetical.

Currency is not inferred. A value unit such as `value_units` is a label for the
user's supplied assumptions; the current dashboard does not convert amounts to
Indian rupees, lakhs or crores.

## 3. Service integration boundary

The dashboard is a presentation layer. Model loading, preprocessing, scoring,
tier assignment, SHAP, ranking, campaign allocation and projection math are
called from the existing phase services; they are never reimplemented in a page.

| Area | Existing interface used |
| --- | --- |
| Discovery | `churnwise.discovery.discover` + adapters + `reporting.write_reports` |
| Acquisition | `AcquisitionCoordinator.acquire` with ranked candidates and fallback |
| Validation | `reports/validation/*.json`, EDA JSON |
| Training | `PreprocessingCoordinator.load`, `MLPTrainer.train` (`persist=True`) |
| Evaluation | metrics/curves JSON (read-only) |
| Explanations | `global.json`, `run_manifest.json`, stored local JSON/PNG |
| Record scoring | `RiskService.assess_customer`, `RiskService.assess_batch` |
| Population explorer | stored Phase 9 Parquet + summary |
| Simulation | `SimulationService.run_scenario(..., persist=False)` for in-session projections; persisted results are read-only |

`churnwise/dashboard/actions.py` contains the only side-effecting calls. Each is
behind an explicitly labelled control in Model Laboratory. Importing a page or
rerunning it does not start discovery, download, training, scoring or a
simulation.

### Population and batch risk runs

Executive Dashboard, Customer Risk Explorer, Retention Workbench default and the
artifact inventory use the newest risk run scored over a named split (`train`,
`val` or `test`), not an ad-hoc `input` batch. Batch runs remain selectable in
every Risk run selector. This prevents a six-row assessment from silently
replacing the 1,000-row population summary.

### Status and metric semantics

- Validation renders `pass` green, `pass_with_warnings` amber and `reject` red.
  The check table's `Passed = no` for V11 is intentional: V11 is a
  warning-level check, not a fatal gate failure. Passed, warning-level and fatal
  records are counted separately.
- Benchmark metrics are held-out test results. Executive Dashboard states that
  they do not describe the scored population; risk captions state the population
  split, classification threshold and high-tier boundary.
- `Predicted churn` (binary label at the classification threshold) and `High +
  critical` (tier membership at the tier boundaries) are distinct and are not
  expected to be equal.
- The Model Laboratory CV table reads the stored `refitted_per_fold` and
  `verified_against_phase5` flags. CV metrics use the 0.5 reference threshold;
  held-out test metrics use each family's validation-selected threshold.
- Explainability reports *feature groups*, not raw columns. The reference run
  aggregates 27 transformed columns into 20 groups: 18 usable raw predictors plus
  two persisted engineered features (`usage_band` and
  `support_calls_per_usage_hour`).
- Probability calibration is a separate research artifact, not integrated into
  the model or threshold; no calibrated curve is displayed.

## 4. State, caching and artifact freshness

`data_access.py` uses:

- `st.cache_data` keyed by `(path, st_mtime_ns, st_size)` for JSON and Parquet.
  A rewritten artifact changes the token and invalidates its cached value.
- `st.session_state` only for UI-owned results: the last training outcome, last
  individual/batch assessment, last in-session simulation, notices and errors.
  These are not written back as project artifacts.
- An explicit **Refresh artifact cache** sidebar button for external pipeline
  runs.
- Success paths call `st.cache_data.clear()` before rerunning so the page sees
  the artifact just written.

## 5. Error handling

User-facing failures use stable codes:

| Code range | Source |
| --- | --- |
| `CW-UI-001` | unreadable/corrupt artifact |
| `CW-UI-003…006` | discovery, candidate, acquisition or attempt-write failure |
| `CW-UI-007` | training configuration invalid |
| `CW-UI-008` | prepared dataset missing |
| `CW-UI-009` | training execution failure |
| `CW-UI-010/011` | individual/batch assessment wrapper failure |
| `CW-RISK-*` | Phase 9 input, model, configuration or prediction failure |
| `CW-SIM-*` | Phase 10 scenario/assumption failure |

Messages contain schema names, counts and error classes, not raw customer rows
or tracebacks. Missing artifacts render an amber empty state with the exact CLI
command; they never render a zero, placeholder chart or synthetic metric.

## 6. Exports and privacy

- Risk, workbench and assessment CSV exports apply the Phase 9
  spreadsheet-formula guard (`=`, `+`, `-`, `@`, tab and carriage return are
  prefixed with `'`).
- JSON exports contain only the stored/returned fields.
- Risk rows use opaque references and transformed values; raw customer
  attributes are not stored in the risk artifacts.
- Batch rejection tables contain position, hashed reference, code and message.
- Business pages show artifact *run ids*, not local filesystem paths.

## 7. Testing

Focused dashboard suite (57 tests):

```bash
python -m pytest -q \
  tests/unit/test_dashboard_analytics.py \
  tests/unit/test_dashboard_assessment.py \
  tests/unit/test_dashboard_data_access.py \
  tests/unit/test_dashboard_simulator.py \
  tests/integration/test_dashboard_apptest.py
```

Coverage highlights:

- `AppTest` renders all five pages against synthetic fixture artifacts and
  against the real project artifact tree; navigation carries exactly the five
  product labels.
- Missing artifacts produce actionable commands on every page.
- Tier filtering changes stored-table rows; cache invalidates after Parquet is
  rewritten; probabilities displayed for a customer match the stored value. The
  customer population join is refused unless the split and labels agree.
- CSV and ARFF source files are both read for original-attribute joins; a
  malformed source returns a stated limitation instead of a guessed row.
- Training action runs once per submit, not on rerun; failures and discovery
  failures are structured and contain no traceback.
- Assessment actions are compared directly with `RiskService` for one record and
  a batch; missing-column batches return rejection rows.
- Upload size/parse/empty/row-limit validation and schema typing.
- Simulation action output matches a direct Phase 10 engine run.
- Analytics helpers: tier boundaries, formatting, export formula guard,
  nested access, validation status/count semantics.
- Validation report discovery ignores a newer Phase 3 `attempts.json` in the
  same directory; population risk runs are preferred over newer input batches.
- Local explanations are scoped to the model family and return a plot from the
  same explanation run; simulator form defaults preserve valid zero values.

### Browser end-to-end suite (opt-in)

`tests/e2e/test_dashboard_browser.py` (10 tests) starts real Streamlit servers
against temporary fixtures and drives them with Playwright. It covers all five
pages, navigation, risk filtering and CSV download, stored customer detail with
its local explanation, a simulator projection, individual assessment, batch CSV
upload/scoring/export, missing-model handling and empty-artifact states. Chart
drag-selection and table row selection were additionally exercised manually
against the real artifacts; they are not part of the automated suite.

```bash
python -m pip install -r requirements.lock.txt
python -m playwright install chromium        # or use system Edge/Chrome
export CHURNWISE_BROWSER_E2E=1               # Windows cmd: set CHURNWISE_BROWSER_E2E=1
python -m pytest tests/e2e -m browser -q
```

The suite is skipped by default and makes no external network call. Use
`CHURNWISE_BROWSER_CHANNEL=msedge` or `chrome` when bundled Chromium is not
installed. Live discovery/acquisition/training buttons and the optional local
SHAP checkbox are not exercised by this browser suite; they remain covered at
the AppTest/unit/integration boundary.

Browser evidence for the redesign is stored in [docs/screenshots](screenshots/):
the five page captures at 1440 px, scrolled detail captures (including a
tier-filtered Executive view and a selected table row), and 1024 px narrow
captures of all five pages. Historical Phase 11–13 screenshots remain in place
for the audit trail.

## 8. Known limitations

- Training is synchronous inside the Streamlit process. The page states this;
  there is no background job or streaming epoch log.
- Discovery and acquisition perform live HTTP only when explicitly started in
  Model Laboratory → Operations.
- Batch assessment does not compute SHAP; explanations rely on stored artifacts.
- The campaign simulator computes in memory and does not persist; it labels every
  output as hypothetical and does not claim causal uplift.
- The browser suite is opt-in and runs one Chromium engine at one desktop
  viewport plus a narrow layout check; it does not cover live network/training
  actions, optional local SHAP, accessibility or visual regression.
