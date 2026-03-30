"""
HPC Bottleneck Detector

A tool for detecting performance bottlenecks in HPC applications using
machine learning and heuristic approaches on time series metrics.
"""

__version__ = "0.1.0"

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
