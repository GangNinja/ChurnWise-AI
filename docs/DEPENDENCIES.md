# ChurnWise AI — Phase 12 Dependency Audit

**Date:** 2026-10-23  
**Environment:** Windows 11, CPython 3.13.0, CPU-only. The primary `.venv` was
created from `requirements.txt`; the clean-room check used a separate
`.venv_clean` created only from `requirements.lock.txt`.  
**Method:** read every declared requirement; import each direct package in the
clean environment; search the source for each package; run `pip check`; inspect
the resolved versions; install the lock into a new virtual environment; run the
full test suite there. No package index vulnerability database was contacted.

## 1. Direct declarations

`requirements.txt` declares 14 direct dependencies. All 14 are installed and
importable in the locked environment.

| Dependency | Declared range | Installed | Imported by source | Assessment |
| --- | --- | --- | --- | --- |
| numpy | `>=2.1,<3.0` | 2.5.4 | yes | in range |
| pandas | `>=2.2,<3.0` | 2.3.3 | yes | in range |
| scikit-learn | `>=1.5,<1.8` | 1.7.2 | yes | in range |
| scipy | `>=1.14,<2.0` | 1.18.1 | yes | in range |
| tensorflow | `>=2.20,<2.22` | 2.21.0 | yes | in range; Keras 3.15.1 bundled |
| shap | `>=0.46,<0.55` | 0.54.1 | yes | in range, close to upper bound |
| streamlit | `>=1.40,<2.0` | 1.66.1 | yes | in range |
| plotly | `>=5.24,<7.0` | 6.9.0 | yes | in range |
| matplotlib | `>=3.9,<4.0` | 3.11.2 | yes | in range |
| joblib | `>=1.4,<2.0` | 1.6.0 | yes | in range |
| pyarrow | `>=17,<25` | 24.0.0 | yes | in range |
| openpyxl | `>=3.1,<4.0` | 3.1.5 | yes | in range |
| requests | `>=2.32,<3.0` | 2.34.2 | yes | in range |
| pytest | `>=8.3,<10.0` | 9.1.1 | yes (tests) | in range |

`pip check` reports **no broken requirements** in both environments. The lock
contains 82 pinned distributions, including transitive dependencies.

`seaborn` and `imbalanced-learn` were declared in the Phase 1 manifest but were
never installed or imported. `seaborn` was not needed because the EDA and
dashboard figures use matplotlib/plotly directly; `imbalanced-learn` was marked
as reserved for a future SMOTE path that was never implemented. Both
declarations are removed so the range manifest and the locked environment agree.

## 2. Exact reproduction

`requirements.lock.txt` was generated with `pip freeze` from the audited
`.venv` after the full suite passed. It pins every installed distribution
exactly and is the supported way to reproduce the measured environment:

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.lock.txt
```

The lock is **platform-specific**: its pins include Windows `cp313` wheels for
TensorFlow and the scientific stack. On another OS or Python minor version, use
`requirements.txt` or regenerate the lock on that platform. The lock has no
hash pins and has not been audited by a vulnerability scanner; it guarantees
version identity, not supply-chain provenance.

**Clean-environment evidence (2026-10-23):**

| Check | Result |
| --- | --- |
| New `.venv_clean` created from CPython 3.13 | success |
| `pip install -r requirements.lock.txt` | exit 0; 82 distributions installed |
| `pip check` | no broken requirements |
| Direct imports (14 packages) | all importable at the versions above |
| `python -m pytest -q` (Phase 12 baseline) | **623 passed, 7 live-network skips, 1484 warnings** in 295.46 s |
| `python -m pytest -q` (post-Phase 13 fix) | **624 passed, 7 live-network skips** in 326.77 s |

The clean environment was an untracked `.venv_clean` and was removed after the
post-fix audit; the result above is retained in the final verification report.
The 7 skips are the intentional live-network tests, which require `--run-live`.

## 3. Findings

| ID | Severity | Finding | Status | Resolution |
| --- | --- | --- | --- | --- |
| DEP-01 | Medium | No exact lock existed even though the manifest promised frozen versions. | **Fixed** | Added `requirements.lock.txt` and reproduced the full suite in a fresh environment. |
| DEP-02 | Low | `seaborn` was declared but neither installed nor imported. | **Fixed** | Removed from `requirements.txt`; figures use matplotlib/plotly. |
| DEP-03 | Low | `imbalanced-learn` was declared but not installed or imported. | **Fixed** | Removed from `requirements.txt`; no SMOTE path exists. |
| DEP-04 | Low | Range bounds allow material version drift. | **Mitigated** | The lock pins the audited versions; ranges remain the cross-platform manifest. |
| DEP-05 | Informational | TensorFlow's native-Windows build is CPU-only; no CUDA runtime is installed. | Open (environment) | Timings reflect CPU-only execution; no GPU acceleration is promised. |
| DEP-06 | Informational | No vulnerability scanner (`pip-audit`, `safety`) was run. | Open (process) | No security claim is made; add a scheduled audit in a later phase. |

No open dependency defect affects the verified workflows.

## 4. Import boundaries and declared stack

The source imports only declared direct packages for its runtime paths:
`numpy`, `pandas`, `scikit-learn`, `scipy`, `tensorflow`, `shap`, `streamlit`,
`plotly`, `matplotlib`, `joblib`, `pyarrow`, `openpyxl` and `requests`.
`pytest` is test-only. There are no undeclared runtime imports and no local
backends, database drivers, cloud SDKs or authentication packages.

The full install and test workflow was exercised from a clean environment
recreated solely from the lock file. That closes the Phase 12 reproducibility
gap: the green suite is no longer dependent on this machine's pre-existing
`.venv`.
