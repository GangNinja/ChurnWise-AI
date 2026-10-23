"""Dashboard pages.

Each module exposes ``render(settings)`` and reads persisted artifacts through
``churnwise.dashboard.data_access``. Page modules must not import TensorFlow,
SHAP or training services at module import time.
"""
