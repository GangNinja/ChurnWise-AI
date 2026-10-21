# ChurnWise AI — Phase 1 Report

| Field | Value |
| --- | --- |
| Phase | 1 of 13 — Repository, Architecture and Requirements |
| Status | Complete (pending review) |
| Date | 2026-10-20 |
| Workspace | `Ai Ml Project` (Windows 11, OneDrive-synced desktop folder) |
| Code written | None. Phase 1 is a requirements and architecture phase. |
| Models trained | None. No performance is claimed anywhere. |
| Companion documents | [ARCHITECTURE.md](ARCHITECTURE.md) · [ROADMAP.md](ROADMAP.md) · [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md) |

---

## 1. Executive summary

The workspace was empty apart from Freebuff metadata: no Git repository, no source files, no tests, no documentation, no dependency manifest. Nothing needed to be preserved or migrated.

The environment is usable but not ready. Python 3.13, pandas, scikit-learn, Streamlit, Plotly and pytest are present. **TensorFlow and SHAP are absent**, and the standalone Keras 3 install is unimportable because it has no backend. Both missing packages are installable for cp313 on Windows (`tensorflow` 2.20.0/2.21.0, `shap` 0.54.1 were confirmed available). The right move is an isolated `.venv` built from the new `requirements.txt` at the start of Phase 2 — the global interpreter must not be modified further.

Autonomous data acquisition is feasible without credentials. Live probes on 2026-10-20 returned HTTP 200 from the OpenML REST API (three "churn" datasets with explicit licences), the UCI API and archive (Iranian Churn, 3,150 × 13), and IBM's public Telco CSV (7,043 × 21, 26.54% churn). A widely-cited third-party mirror returned 404 — direct evidence that acquisition must verify every source rather than assume URLs are stable.

Phase 1 delivers four documents: a module-level architecture with file-based data contracts, a 13-phase roadmap mapped to a five-day schedule, per-phase acceptance criteria with 87 verification tests, and this report. All 13 phases are preserved; autonomous discovery and acquisition are recorded as mandatory; no performance target is stated. An automated checker now validates the documentation set (114 checks, all passing), including the schedule's arithmetic.

**Recommendation:** proceed to Phase 2 after two decisions are recorded — the licence policy for `unverified` sources, and whether to initialise a Git repository. Neither blocks discovery work against OpenML/UCI.

---

## 2. Environment findings

Audited with `python --version`, `pip list`, `importlib.util.find_spec`, direct imports, and `pip index versions`.

| Item | Result |
| --- | --- |
| OS | Windows 11 (`Windows-11-10.0.26300-SP0`) |
| Interpreter | Python 3.13.0 at `C:\Program Files\Python313\python.exe` |
| Package manager | pip 26.1 (user site-packages at `%APPDATA%\Roaming\Python\Python313\site-packages`) |
| Virtual environment | **None present** |
| Git repository | **None** (`fatal: not a git repository`) |
| Free disk | 158 GB of 464 GB |
| Working directory | `C:\Users\NARASIMHA\OneDrive\Desktop\Ai Ml Project` (OneDrive-synced) |

### 2.1 Package inventory

| Package | Status | Version | Needed for |
| --- | --- | --- | --- |
| numpy | present | 2.1.2 | all phases |
| pandas | present | 2.2.3 | all phases |
| scikit-learn | present | 1.5.2 | preprocessing, baselines, metrics |
| scipy | present | 1.14.1 | sklearn dependency |
| streamlit | present | 1.56.0 | dashboard (Phase 11) |
| matplotlib | present | 3.9.2 | EDA and evaluation figures |
| seaborn | present | 0.13.2 | EDA figures |
| plotly | present | 6.5.0 | dashboard charts |
| joblib | present | 1.4.2 | artifact persistence |
| pyarrow | present | 24.0.0 | Parquet IO |
| openpyxl | present | 3.1.5 | Excel IO |
| requests | present | 2.33.1 | acquisition |
| pytest | present | 9.1.1 | verification |
| **tensorflow** | **missing** | — | MLP (Phase 6) |
| **shap** | **missing** | — | explanations (Phase 8) |
| imbalanced-learn | missing | — | optional SMOTE path only |
| keras | present but **broken** | ≥3.x | `import keras` raises `ModuleNotFoundError: No module named 'tensorflow'` |

### 2.2 Installability evidence

```text
$ python -m pip index versions tensorflow
tensorflow (2.21.0)
Available versions: 2.21.0, 2.20.0

$ python -m pip index versions shap
shap (0.54.1)
```

cp313 Windows wheels exist for both, so the mission-mandated TensorFlow/Keras MLP is viable. Installation was deliberately **not** performed in Phase 1 (see §9, R1).

`pip` also reports `WARNING: Ignoring invalid distribution ~ip` — evidence that the global site-packages has a partially removed distribution. This reinforces the isolated-environment decision rather than being a blocker.

---

## 3. Repository audit

```text
$ pwd
/c/Users/NARASIMHA/OneDrive/Desktop/Ai Ml Project

$ ls -la
drwxr-xr-x  .freebuff/
drwxr-xr-x  .freebuff/project-id

$ git status
fatal: not a git repository (or any of the parent directories): .git
```

**Findings**

- No reusable implementations, tests, documentation, notebooks or dependency manifests existed. There was nothing to preserve, refactor or avoid overwriting.
- The only pre-existing content is `.freebuff/project-id`, workspace metadata that is left untouched and is not part of the project.
- Because there is no Git history, artifact metadata uses `code_version = "unversioned"` until a repository exists.
- `git init` was **not** run: the mission forbids unrequested Git operations, and repository initialisation is the user's call. A `.gitignore` is prepared so initialisation is a one-step operation.

---

## 4. Public dataset source audit

Probes were unauthenticated `GET` requests; no credentials were used or exposed. They establish **reachability and metadata only** — parsing, schema and licence capture belong to Phases 2–4.

| Source | Probe | Result |
| --- | --- | --- |
| OpenML dataset list (`data_name=churn`) | `GET /api/v1/json/data/list/...` | 200 — 3 hits: `40701`, `45076`, `46915` |
| OpenML `40701` v1 | `GET /api/v1/json/data/40701` + qualities | `licence: public`, target `class`, 5,000 × 21, class sizes 4,293 / 707, uploaded 2017, PMLB/BigML provenance |
| OpenML `45076` v3 | `GET /api/v1/json/data/45076` + qualities | `licence: Public`, target `class`, 5,000 × 21, telephony account features; metadata reports `NumberOfClasses = 0` |
| OpenML `46915` v4 | `GET /api/v1/json/data/46915` + qualities | `licence: MIT License`, target `CustomerChurned`, 5,000 × 20, class sizes 4,293 / 707, TabArena-curated |
| UCI API search `churn` | `GET /api/datasets/list?search=churn` | 200 — one hit: `563`, "Iranian Churn" |
| UCI `563` metadata | `GET /api/dataset?id=563` | 3,150 instances, 13 features, classification, area "Business" |
| UCI `563` licence | dataset page | CC BY 4.0, DOI `10.24432/C5JW3Z` |
| UCI archive ZIP | `GET /static/public/563/iranian+churn+dataset.zip` | 200 — 41,943 bytes, sha256 `696c3a18…13ead66` |
| IBM Telco CSV (IBM org repo) | raw GitHub `Telco-Customer-Churn.csv` | 200 — 970,457 bytes decoded, sha256 `16320c9c…3055e91`, shape 7,043 × 21, target `Churn` = No 5,174 / Yes 1,869 (26.54%) |
| Third-party Telco mirror | raw GitHub `dataprofessor/data` | **404** |

**Correction found during verification.** The three OpenML hits are versions or re-uploads of **one** 5,000-row telephony corpus — 5,000 rows each (21/21/20 columns), and the two entries that report class counts share an identical 4,293 / 707 split. They are version fallbacks, not independent datasets. The architecture's seed table was corrected; genuine dataset diversity comes only from UCI 563 (3,150 × 13) and IBM Telco (7,043 × 21). OpenML metadata also proved incomplete (`45076` reports zero classes), which confirms that gate G3 must inspect parsed data rather than trust source metadata.

**Implications recorded in the architecture**

1. Licence metadata is available per dataset on OpenML, but not uniformly on UCI or GitHub mirrors; the acquisition manifest therefore records `verified:<id>` / `unverified` explicitly.
2. Name-based search is unreliable: OpenML returned 0 hits for `telco` and `customer churn`, but 3 for `churn`. Discovery must combine several query terms with known dataset IDs.
3. The 404 proves that hardcoded third-party mirrors rot; the fallback chain and manifest hash-pinning are justified, not decorative.
4. The IBM CSV's licence is not clearly stated in the repository — treated as `unverified` and requiring explicit opt-in, never as a default.
5. The hashes above are **audit observations**, not pinned constants: the acquisition layer records the hash of whatever it stores and must not hardcode these values.

---

## 5. Architecture established

Full detail in [ARCHITECTURE.md](ARCHITECTURE.md). The decisions that matter most for delivery:

| Area | Decision |
| --- | --- |
| Runtime | Single local process: CLI pipeline + Streamlit. No database, backend service, auth or cloud. |
| Interchange | Versioned JSON/Parquet/Joblib artifacts on disk; every artifact carries `run_id`, dataset hash, seed, code version. |
| Modules | 11 stage packages with a forward-only dependency rule; the dashboard may not import training or acquisition code. |
| Autonomy | Static reviewed registry (OpenML, UCI, IBM GitHub, optional Kaggle) + 9 eligibility gates + deterministic ranking + fallback chain + content-addressed cache + provenance manifests. |
| Honesty | No synthetic fallback; exhausted acquisition stops the run with error codes and remediation. This is enforced by a test in Phase 3. |
| Preprocessing | One shared `ColumnTransformer` for all three models, fitted on train only, with a split fingerprint and an 8-point leakage checklist. |
| MLP | Dense 64/32 ReLU, dropout 0.3/0.2, L2 1e-4, sigmoid output, BCE, Adam 1e-3, early stopping (patience 12) and LR reduction. |
| Evaluation | Identical rows/features/weights for MLP, LR and RF; single held-out test; 5-fold CV stability; metrics + confusion matrices; no predetermined targets. |
| Explainability | Linear/Tree/Deep explainers with Kernel fallback, additivity verification, one-hot grouping to source columns, precomputed JSON. |
| Simulation | Projection arithmetic over measured risk with explicit assumptions, sensitivity grid, break-even and a projection disclaimer. |
| Dashboard | Seven read-only pages over cached artifacts; missing artifacts produce actionable empty states, never sample numbers. |

---

## 6. Requirements coverage

### 6.1 Phase coverage

All thirteen mandated phases are present in both the roadmap and the acceptance criteria, with seven documented dimensions each (objectives/scope, inputs/outputs, artifacts, dependencies, testing, completion criteria, risks).

| # | Phase | Architecture reference | Roadmap | Acceptance | Tests |
| --- | --- | --- | --- | --- | --- |
| 1 | Repository, Architecture, Requirements | whole document | §3 | §2 | T1.1–T1.7 |
| 2 | Autonomous Dataset Discovery | §7 | §3 | §3 | T2.1–T2.6 |
| 3 | Autonomous Acquisition + Provenance | §7.5–7.6 | §3 | §4 | T3.1–T3.8 |
| 4 | Validation + Data Intelligence | §5.1, §6 | §3 | §5 | T4.1–T4.6 |
| 5 | Preprocessing + Feature Engineering | §8.1 | §3 | §6 | T5.1–T5.7 |
| 6 | MLP Development + Training | §8.2 | §3 | §7 | T6.1–T6.7 |
| 7 | Comparative Benchmarking | §8.3–8.4 | §3 | §8 | T7.1–T7.7 |
| 8 | Explainable AI + SHAP | §8.5 | §3 | §9 | T8.1–T8.6 |
| 9 | Churn Risk Intelligence | §9.1 | §3 | §10 | T9.1–T9.7 |
| 10 | Campaign Simulation | §9.2 | §3 | §11 | T10.1–T10.7 |
| 11 | Interactive Dashboard | §10 | §3 | §12 | T11.1–T11.7 |
| 12 | Integration + Reliability | §5, §11 | §3 | §13 | T12.1–T12.7 |
| 13 | Final Verification + Presentation | §14 | §3 | §14 | T13.1–T13.5 |

### 6.2 Autonomy requirement

- [x] Declared **mandatory** in architecture (§7.1) and roadmap (§5.1).
- [x] First-run requires no dataset, no URL and no credentials.
- [x] Discovery source registry with per-source auth and licence posture.
- [x] Suitability gates G1–G9 and deterministic ranking.
- [x] Download verification: HTTPS allowlist, size caps, redirect re-check, content sniffing, parser probe, sha256, archive safety.
- [x] Provenance manifest: URL, UTC retrieval time, reported version, byte size, hash, licence status, fallback lineage.
- [x] Content-addressed cache with offline mode.
- [x] Fallback chain and actionable failure taxonomy (`CW-ACQ-001`…`-006`).
- [x] Explicit prohibition of synthetic data or silent substitution, with a planned test.
- [x] Implementation deferred to Phases 2–3 by design — interfaces and acceptance criteria are fixed here.

### 6.3 ML and evaluation requirements

| Requirement | Where specified | Status |
| --- | --- | --- |
| ReLU hidden layers | Architecture §8.2 | specified |
| Sigmoid output | §8.2 | specified |
| Binary cross-entropy | §8.2 | specified |
| Adam optimiser + learning rate | §8.2 | specified |
| Backpropagation | §8.2 (Keras autodiff) | specified |
| Regularisation (dropout + L2) | §8.2 | specified |
| Early stopping | §8.2 | specified |
| Logistic Regression baseline | §8.3 | specified |
| Random Forest baseline | §8.3 | specified |
| Leakage-safe fair comparison | §8.1, §8.3, checklist L1–L8 | specified |
| Accuracy, precision, recall, F1, ROC-AUC, confusion matrices | §8.4, acceptance T7.x | specified |
| SHAP global + local | §8.5 | specified |
| No predetermined performance targets | §8.4, acceptance P7 criteria | enforced |

No metric values are asserted anywhere. The only quantitative validity rule is ROC-AUC > 0.5 on test, which is a defect detector, not a performance goal.

---

## 7. Deliverables

### Created

| File | Purpose |
| --- | --- |
| [docs/ARCHITECTURE.md](ARCHITECTURE.md) | Module map, contracts, data flow, autonomy design, ML/XAI/simulation/dashboard design, decision log |
| [docs/ROADMAP.md](ROADMAP.md) | 13 phases, dependency graph, five-day schedule, effort allocation, schedule risks, scope preservation |
| [docs/ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md) | Global Definition of Done, per-phase acceptance criteria, 87 tests, 91 completion checkboxes, traceability matrix, sign-off gate |
| [docs/PHASE1_REPORT.md](PHASE1_REPORT.md) | This report |
| [requirements.txt](../requirements.txt) | Bounded dependency manifest; nothing installed in Phase 1 |
| [.gitignore](../.gitignore) | Excludes data, artifacts, reports, environments, secrets and caches |
| [scripts/verify_phase1_docs.py](../scripts/verify_phase1_docs.py) | Automated documentation consistency checker |

### Inspected

| Path | Finding |
| --- | --- |
| `.freebuff/project-id` | Workspace metadata only; untouched |
| repository root | No source, tests, docs, lockfiles or CI configuration |
| global Python environment | Inventory and import failures recorded in §2 |

No existing file was modified, moved or deleted.

---

## 8. Verification performed

| Command | Purpose | Result |
| --- | --- | --- |
| `python -m py_compile scripts/verify_phase1_docs.py` | Checker syntax | exit 0 |
| `python scripts/verify_phase1_docs.py` | Structural consistency and schedule arithmetic of all four documents | **114 checks passed, 0 failed** (exit 0) |
| Live HTTP probes (OpenML, UCI, IBM; §4) | Source reachability, metadata and licence fields | 13 successes, 1 expected 404 |
| `python -m pip index versions tensorflow` | Confirm cp313 availability | 2.20.0, 2.21.0 |
| `python -m pip index versions shap` | Confirm cp313 availability | 0.54.1 |
| Mermaid 11 parse in headless Chromium | Validate every diagram block | 4 of 4 parsed, 0 errors |

The checker validates, among other things:

- exactly 13 roadmap sections titled as mandated, and exactly 13 acceptance sections numbered 1–13;
- all seven required subsections in every acceptance phase, with ≥3 completion criteria and at least one named test per phase;
- every roadmap phase declares dependencies and all dependencies point backwards only;
- Mermaid diagrams exist in the architecture and roadmap;
- autonomy language is present in all three normative documents, including the synthetic-data prohibition;
- the MLP design terms (ReLU, sigmoid, BCE, Adam, backpropagation, early stopping, dropout, L2) and all six mandatory metrics are specified;
- no placeholder tokens and no invented performance target;
- per-phase scheduled hours reconcile with the effort allocation, and daily hours sum to the declared total;
- every relative link and anchor resolves;
- the traceability matrix covers phases 1–13.

Limitations of this verification: it is a **documentation** check. It does not prove that any future phase works, and it deliberately makes no claim about model quality or runtime.

---

## 9. Risks and unresolved decisions

| ID | Risk / decision | Likelihood | Impact | Mitigation / owner |
| --- | --- | --- | --- | --- |
| R1 | TensorFlow install friction on Python 3.13 / Windows | Medium | High — blocks Phase 6 | Wheels confirmed; install into `.venv` as Phase 2 step 1; if it fails, fall back to a Python 3.12 venv and record it. Never swap frameworks silently. **Owner: implementer.** |
| R2 | Thin contingency in a 41 h plan (Days 1–4 at 8.5–9 h) | High | High | Protect Days 1–2; Day 5 absorbs up to ~2.5 h of overrun; cut polish (extra charts, cosmetic UI), never scope; report unfinished criteria as `deferred` with risk. **Owner: implementer.** |
| R3 | SHAP runtime on CPU, especially `DeepExplainer`/`KernelExplainer` for the MLP | Medium | Medium | Background ≤200 rows, explained ≤300, precomputed JSON, hard time budget, fallback explainer. **Owner: implementer (Phase 8).** |
| R4 | Licence policy for `unverified` sources (IBM mirror) | High | Medium | Default to explicit opt-in (`CHURNWISE_ALLOW_UNVERIFIED_LICENSE=0`); prefer MIT/Public OpenML candidates. **Needs user confirmation.** |
| R5 | Primary dataset selection | Medium | Medium | Decide during Phases 2–3 from gate evidence; record the ranked candidate list and the chosen ID. **Owner: implementer, user may override.** |
| R6 | No Git repository exists | Certain | Medium — traceability | Recommend `git init` plus an initial commit before Phase 2. **Not performed; needs user decision.** |
| R7 | Global site-packages is already polluted (broken Keras, `~ip` warning) | Certain | Medium | Never install globally; project-local `.venv`; freeze resolved versions. **Owner: implementer.** |
| R8 | OpenML name search is narrow | Medium | Medium | Multiple query terms, known dataset IDs, metadata resolution, wide rejection log. **Owner: implementer (Phase 2).** |
| R9 | Project lives inside a OneDrive-synced folder | Medium | Medium | OneDrive can lock files mid-write and slow `.venv`/artifact IO; if training or caching shows latency, exclude `data/`, `artifacts/` and `.venv` from sync or relocate the project. **Owner: user.** |
| R10 | Simulation has no revenue proxy for some datasets | Medium | Low | Require an explicit user-entered per-customer value; never invent one; label as projection. **Owner: implementer (Phase 10).** |
| R11 | Metric presentation could drift from artifacts | Low | High (credibility) | Phase 13 test T13.3 compares every documented number to `metrics.json`. |
| R12 | Documentation drift as implementation evolves | Medium | Medium | Update docs in the phase that changes behaviour; re-run the checker in Phases 12–13. |

### Decisions still open

1. **Licence default (R4):** may `unverified` datasets be used without a flag? Current design: no.
2. **Repository initialisation (R6):** approve `git init` before Phase 2?
3. **Primary dataset (R5):** prefer OpenML `45076`/`46915` (clean licence) or IBM Telco (richest features, ambiguous licence)?
4. **MLP explainer (Phase 8):** `DeepExplainer` vs `KernelExplainer` — resolve after measuring runtime and additivity.
5. **Operating threshold (Phase 7):** fixed 0.5 vs validation-tuned; resolve once real metrics exist.

---

## 10. Phase 2 readiness

| Prerequisite | Status |
| --- | --- |
| Architecture and contracts fixed | Ready |
| Acceptance criteria for discovery | Ready (T2.1–T2.6) |
| Source candidates identified with reachability evidence | Ready |
| Dependency manifest declared | Ready — installation pending |
| Isolated `.venv` created and dependencies installed | **Pending — first Phase 2 action** |
| Licence policy confirmed | **Pending user decision** |
| Git repository initialised | **Pending user decision** (recommended) |

Phase 2 can begin on discovery implementation immediately, since it uses only OpenML/UCI metadata and the standard library plus `requests`. The two pending decisions affect Phase 3, not the start of Phase 2.

---

## 11. Explicit stop

Phase 1 ends with this report. **No Phase 2 work has begun:** no application package, no discovery code, no downloads written to the project, no dependencies installed, no models trained, and no performance figures produced or implied. The next action is review of this report and the three companion documents.
