"""Executive Dashboard entry point with a reloadable reference layout."""

from __future__ import annotations

import importlib

from churnwise.config import Settings


def render(settings: Settings) -> None:
    page = importlib.import_module("churnwise.dashboard.pages.executive_reference")
    page = importlib.reload(page)
    explorer = importlib.import_module("churnwise.dashboard.pages.explorer")
    page.explorer_page = importlib.reload(explorer)
    page.render(settings)


__all__ = ["render"]
