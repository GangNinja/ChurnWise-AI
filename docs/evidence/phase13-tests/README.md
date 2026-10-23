# Phase 13 test evidence

These are results from the final audit of the post-fix code. The full offline
suite log is preserved verbatim in [`full_suite.log`](full_suite.log); its exit
status is in [`full_suite.exit`](full_suite.exit). Other results were captured
from the commands shown below and are summarized here; no result is asserted
without having been observed.

**Environment:** Windows 11, CPython 3.13.0, project `.venv`, CPU-only
TensorFlow. **Date:** 2026-10-23.

| Check | Command | Result |
| --- | --- | --- |
| Full offline suite | `python -m pytest -q --disable-warnings` | **624 passed, 7 skipped**, 268.70 s, exit 0 |
| Clean environment | `.venv_clean/Scripts/python -m pytest -q` | **624 passed, 7 skipped**, 326.77 s, exit 0 |
| Live network tests | `python -m pytest -q -m live --run-live` | **7 passed, 624 deselected**, 165.37 s, exit 0 |
| Dashboard focused | `pytest -q tests/integration/test_dashboard_apptest.py tests/unit/test_dashboard_*.py` | **42 passed**, 24.61 s, exit 0 |
| Documentation structure | `python scripts/verify_phase1_docs.py` | **114 checks passed, 0 failed**, exit 0 |
| Claim consistency | `python scripts/verify_phase13_claims.py` | **PASS**, 5 documents / 21 claims, exit 0 |
| Fixture workflow | `churnwise.cli run --dataset tests/fixtures/mini_churn.csv --fast --seed 13 ...` | 7/7 stages, 160 rows, **53 s**, exit 0 |
| Live workflow (pre-fix) | `churnwise.cli run --fast --seed 42 --run-id phase13-final-live ...` | 7/7 stages, **66 s**, exit 0 |
| Live workflow (post-fix) | `churnwise.cli run --fast --seed 42 --run-id phase13-final-live2 ...` | 7/7 stages, **105 s wall / 93 s manifest**, exit 0 |

The seven skips in the offline suite are the opt-in live-network tests; they are
not failures. The clean environment was installed only from
`requirements.lock.txt`; it was removed after the audit, and its observed result
is recorded above. The full-suite log has 1,484 warnings, mostly TensorFlow,
gast and NumPy deprecation warnings that do not fail tests.

**Failure found and fixed:** the first post-fix audit attempt failed one
determinism test because Random Forest `log_loss` differed by one ULP
(`0.7423580474286281` vs `...82`). The fix (serial `n_jobs=1` default) and its
regression tests are documented as D-12 in
[INTEGRATION_DEFECTS.md](../../INTEGRATION_DEFECTS.md). The full suite was rerun
after the fix; the log above is that run.

No test was removed, skipped, or weakened to obtain these outcomes.
