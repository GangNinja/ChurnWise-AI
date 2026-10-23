# ChurnWise AI — Final Acceptance Matrix

**Verification date:** 2026-10-23  
**Evidence baseline:** repository state after the Phase 13 determinism fix,
`pytest -q --disable-warnings` → **624 passed / 7 skipped**, live run
`phase13-final-live2` (all seven stages).  
**Status vocabulary:** `VERIFIED` · `PARTIALLY VERIFIED` · `BLOCKED` · `NOT VERIFIED`

A status is assigned to the phase's stated completion criteria, not to every
conceivable production property. Known limits are listed beside the status and
carried into [FINAL_VERIFICATION_REPORT.md](FINAL_VERIFICATION_REPORT.md).

---

## 1. Phase summary

| Phase | Name | Status | Decisive evidence | Remaining limitation |
| --- | --- | --- | --- | --- |
| 1 | Foundation, architecture and plan | **VERIFIED** | 114/114 documentation checks; architecture/roadmap/acceptance documents present; Phase 1 report | None affecting scope |
| 2 | Autonomous dataset discovery | **VERIFIED** | Candidate ranking, nine gates, duplicate handling; live OpenML discovery tests passed | Live source availability is date-dependent |
| 3 | Acquisition and provenance | **VERIFIED** | 475,249-byte OpenML ARFF, SHA-256 verified, manifest and cache; format/archive integrity suites | External URLs can change; cache mitigates |
| 4 | Validation and data intelligence | **VERIFIED** | 12 quality checks, target mapping, EDA profile/plots, gate decision; single-class rejection test | One primary dataset exercised |
| 5 | Leakage-safe preprocessing | **VERIFIED** | Split isolation, train-only fitting, persisted preprocessor/schema, leakage regression tests | No cross-dataset stress test |
| 6 | MLP training | **VERIFIED** | Keras ReLU/sigmoid/BCE/Adam model, early stopping, checkpoint, history, deterministic-mode test | Default MLP floats are not bitwise reproducible (D-08) |
| 7 | Benchmark and evaluation | **VERIFIED** | Three models on identical splits; validation-only selection; test metrics, curves, CV, error analysis | One dataset; no confidence intervals |
| 8 | Explainability | **VERIFIED** | TreeSHAP/LinearSHAP/DeepSHAP, grouped features, additivity checks, global and local artifacts | Global ranking is sample-based; one dataset |
| 9 | Risk intelligence | **VERIFIED** | 1,000 scored, tiers, ranking/filtering, batch validation, safe exports, provenance | Final fast pipeline stores only 10 live drivers; 1,000-row matrix is a separate run |
| 10 | Campaign simulation | **VERIFIED** | Budget/cost/effectiveness constraints, sensitivity, comparison, explicit projections, arithmetic tests | No measured uplift or causal effect |
| 11 | Dashboard | **VERIFIED** | 11 pages, 42 focused tests, all-page AppTest, live HTTP/screenshot evidence, 1.937 s first render | No committed browser E2E suite; some actions tested via service boundary |
| 12 | Integration and reliability | **VERIFIED** | Seven-stage resumable pipeline, fixture and live runs, contract/failure tests, 24 pipeline tests | Exact lock is platform-specific; no universal JSON-Schema registry |
| 13 | Final verification and presentation | **VERIFIED** | This matrix, final report, README, presentation, demo, resume, screenshots, two live runs | Fast-profile live evidence only; residual limits documented |

**Summary:** 13 `VERIFIED`, 0 `PARTIALLY VERIFIED`, 0 `BLOCKED`, 0 `NOT VERIFIED`.
This means the documented Phase 1–13 acceptance criteria are met with evidence;
it does not mean the system is production-deployed or clinically/financially
validated.

### Criterion-ID coverage

The phase reports contain the per-test result tables. This index maps every
criterion range to its authoritative record.

| Phase | Criterion IDs | Status | Per-ID evidence |
| --- | --- | --- | --- |
| 1 | T1.1–T1.7 | VERIFIED | [PHASE1_REPORT.md](PHASE1_REPORT.md); 114/114 doc checks |
| 2 | T2.1–T2.6 | VERIFIED | [PHASE2_REPORT.md](PHASE2_REPORT.md); discovery suite |
| 3 | T3.1–T3.8 | VERIFIED | [PHASE3_REPORT.md](PHASE3_REPORT.md); acquisition/integrity suites |
| 4 | T4.1–T4.6 | VERIFIED | [PHASE4_REPORT.md](PHASE4_REPORT.md); validation suite |
| 5 | T5.1–T5.7 | VERIFIED | [PHASE5_REPORT.md](PHASE5_REPORT.md); leakage/split suites |
| 6 | T6.1–T6.7 | VERIFIED | [PHASE6_REPORT.md](PHASE6_REPORT.md); MLP suites |
| 7 | T7.1–T7.7 | VERIFIED | [PHASE7_REPORT.md](PHASE7_REPORT.md); benchmark suite |
| 8 | T8.1–T8.6 | VERIFIED | [PHASE8_REPORT.md](PHASE8_REPORT.md); SHAP suites |
| 9 | T9.1–T9.7 | VERIFIED | [PHASE9_REPORT.md](PHASE9_REPORT.md); risk suites |
| 10 | T10.1–T10.7 | VERIFIED | [PHASE10_REPORT.md](PHASE10_REPORT.md); simulation suites |
| 11 | T11.1–T11.12 | VERIFIED | [PHASE11_REPORT.md](PHASE11_REPORT.md); dashboard suite |
| 12 | T12.1–T12.7 | VERIFIED | [PHASE12_REPORT.md](PHASE12_REPORT.md); pipeline suite |
| 13 | T13.1–T13.5 | VERIFIED | Section 2 below |

---

## 2. Phase 13 completion criteria

| ID | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| T13.1 | Every criterion has a status and evidence reference | **VERIFIED** | This matrix; phase reports contain criterion checklists; 114/114 docs checks |
| T13.2 | No placeholder or unverified claim remains | **VERIFIED** | `grep` for TODO/TBD/placeholder returns no live documentation claims; claims reviewed against artifacts |
| T13.3 | Reported metrics equal artifact values | **VERIFIED** | Scripted comparison of README/presentation/showcase/resume numbers against `docs/evidence/phase13-live2/benchmark_metrics.json`, `risk_summary.json`, `simulation_summary.json` |
| T13.4 | Clean-checkout reproduction follows README | **VERIFIED** | `.venv_clean` from `requirements.lock.txt`; **624 passed / 7 skipped** in 326.77 s |
| T13.5 | Demo script runs start-to-finish | **VERIFIED** | Fixture command completed 7/7 stages in 53 s; live command completed 7/7 in 66 s and 105 s |

---

## 3. Cross-cutting acceptance gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Target detection and positive mapping | **VERIFIED** | `CustomerChurned` No/Yes → 0/1, positive rate 14.14% in live validation report |
| Train/validation/test separation | **VERIFIED** | 3,000/1,000/1,000; disjoint indices and split fingerprint in pipeline test |
| Train-only transformation fitting | **VERIFIED** | Preprocessing tests and persisted pipeline; no test-row fitting |
| MLP architecture and optimisation | **VERIFIED** | ReLU hidden layers, sigmoid output, BCE, Adam, backpropagation via Keras autodiff; trained live |
| Early stopping and checkpointing | **VERIFIED** | Best-checkpoint restore, history and diagnostics persisted |
| Baseline benchmarking | **VERIFIED** | LR, RF, MLP on shared rows; identical test partition |
| Metric correctness | **VERIFIED** | Recomputable from saved predictions; `evaluate --regenerate` parity tests |
| Test set not used for selection | **VERIFIED** | Selection records `selected_on: validation`, `test_used_for_selection: false` |
| SHAP model/preprocessor identity | **VERIFIED** | Model hash, split fingerprint and preprocessor hash checked before explanation |
| Risk probabilities and tiers | **VERIFIED** | Threshold 0.45 distinct from tier boundaries 0.3/0.5/0.7; 1,000-row reconciliation |
| Simulation arithmetic | **VERIFIED** | 109.444617 × 0.30 = 32.833385 expected prevented churn |
| Undefined financial metrics | **VERIFIED** | `null` plus reasons when no monthly value is supplied |
| Dashboard no implicit side effects | **VERIFIED** | AppTest page renders do not train/download/score; explicit actions only |
| Secrets and privacy | **VERIFIED** | No secrets in tracked files; exports omit identifiers; formula-injection guard tested |
| Reproducibility | **VERIFIED** | RF default serial, deterministic splits/models; MLP float limit documented |
| Full supported workflow | **VERIFIED** | Two live seven-stage runs; fixture run; dashboard over persisted artifacts |

---

## 4. Test evidence

| Command | Result | Notes |
| --- | --- | --- |
| `python -m pytest -q --disable-warnings` | **624 passed, 7 skipped** (268.70 s) | Seven skips are opt-in live tests |
| `python -m pytest -q` in `.venv_clean` | **624 passed, 7 skipped** (326.77 s) | Installed only from `requirements.lock.txt` |
| `python -m pytest -q -m live --run-live` | **7 passed** (165 s) | Run separately; requires network |
| `python scripts/verify_phase13_claims.py` | **PASS**, 21 claims | Documentation numbers vs evidence |
| lint / static types | **not configured** | No ruff/mypy/flake8 configuration exists in the repository |
| Dashboard-focused suite | **42 passed** (24.61 s) | AppTest all pages + analytics/assessment/data access |
| Pipeline-focused suite | **24 passed** | End-to-end, contracts, cleanup |
| `python scripts/verify_phase1_docs.py` | **114 passed, 0 failed** | Documentation structure and scope checks |
| Fixture demo command | **exit 0**, 7/7 stages, 53 s | Offline, 160 rows |
| Live run `phase13-final-live` | **exit 0**, 7/7 stages, 66 s | Pre-fix evidence |
| Live run `phase13-final-live2` | **exit 0**, 7/7 stages, 105 s | Post-fix evidence; identical metrics |

No test was deleted, skipped, or weakened to obtain this result. The one failure
found during final verification (D-12) was fixed in implementation and covered by
the existing determinism test plus a new focused regression test.

---

## 5. Defect disposition

| ID | Status | Summary |
| --- | --- | --- |
| D-01 | Fixed | Missing one-command pipeline orchestration |
| D-02 | Fixed | No managed path for a committed local fixture |
| D-03 | Fixed | CLI exit-code shadowing |
| D-04 | Fixed | Failed runs could not resume safely |
| D-05 | Fixed | Probability mass mislabelled as probability |
| D-06 | Fixed | Pipeline summaries read wrong metadata keys |
| D-07 | Fixed | Mixed relative/absolute risk artifact paths |
| D-08 | **Open (documented)** | MLP floats not bitwise reproducible by default |
| D-09 | **Mitigated** | No universal JSON-Schema registry |
| D-10 | Fixed | EDA profile missing `generated_at` |
| D-11 | Fixed | Fixture licence vocabulary |
| D-12 | Fixed | Random Forest one-ULP prediction drift |

D-08 and D-09 are non-critical, explicitly surfaced in the UI/docs, and do not
invalidate the supported workflow. No critical defect remains open.

---

## 6. Final sign-off

The Phase 13 completion procedure is satisfied:

1. Repository and evidence audit completed.
2. In-scope defects fixed with regression coverage.
3. Focused and broad suites run with preserved exit status.
4. Fixture, live pipeline and dashboard workflows verified.
5. Phase-by-phase matrix completed.
6. Documentation and portfolio materials finalized.
7. Every reported number traced to an artifact or test output.
8. Roadmap updated to the verified final state.
9. Work stops at Phase 13; no Phase 14 is introduced.
