"""Interactive Streamlit dashboard over the persisted ChurnWise artifacts (Phase 11).

The dashboard is deliberately read-mostly: every page loads the JSON, Parquet and
PNG artifacts produced by Phases 4–10. Two explicit actions (dataset discovery
and MLP training) delegate to the existing services instead of reimplementing
their logic. No model is trained, no SHAP value is computed and no dataset is
downloaded on a page render.

Run with::

    python -m streamlit run churnwise/dashboard/app.py

See ``docs/DASHBOARD.md`` for the page map and operational notes.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.11.0"
