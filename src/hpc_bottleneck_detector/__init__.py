"""
HPC Bottleneck Detector

A tool for detecting performance bottlenecks in HPC applications using
machine learning and heuristic approaches on time series metrics.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("hpc-bottleneck-detector")
except PackageNotFoundError:  # package not installed (e.g. run from a source checkout)
    __version__ = "1.0.0"

from .orchestrator import AnalysisOrchestrator
from .output.models import (
    BottleneckType,
    MacroCategoryType,
    Diagnosis,
    WindowDiagnosis,
)
from .utils.labeling import label_job, BOTTLENECK_COLUMNS

__all__ = [
    "AnalysisOrchestrator",
    "BottleneckType",
    "MacroCategoryType",
    "Diagnosis",
    "WindowDiagnosis",
    "label_job",
    "BOTTLENECK_COLUMNS",
]
