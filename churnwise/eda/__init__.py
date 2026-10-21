"""Dataset profiling and diagnostic EDA (Phase 4)."""

from churnwise.eda.profile import (
    ColumnProfile,
    DatasetProfile,
    RoleMapping,
    TargetInterpretation,
    build_profile,
)
from churnwise.eda.plots import generate_plots

__all__ = [
    "ColumnProfile",
    "DatasetProfile",
    "RoleMapping",
    "TargetInterpretation",
    "build_profile",
    "generate_plots",
]
