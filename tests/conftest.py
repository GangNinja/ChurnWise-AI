from __future__ import annotations

import pytest

from churnwise.config import Settings


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run tests that contact real public dataset sources",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="live network test; run with --run-live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(project_root=tmp_path)
