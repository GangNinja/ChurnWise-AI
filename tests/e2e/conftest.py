"""Browser-suite hooks.

The page fixture in ``test_dashboard_browser.py`` captures a screenshot when the
test's call phase failed. Hooks must live in a conftest/plugin, not a test module.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)
