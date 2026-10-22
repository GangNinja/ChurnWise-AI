"""ChurnWise AI — autonomous-data customer retention intelligence platform.

Phase 8 adds SHAP-based explanation of the shipped model: sampled backgrounds,
global importance and dependence figures, per-customer local drivers with
additivity checks, and artifacts tied to dataset, split, preprocessing and model
digests. The test partition is never used for explanation.
"""

__version__ = "0.8.0"
__all__ = ["__version__"]
