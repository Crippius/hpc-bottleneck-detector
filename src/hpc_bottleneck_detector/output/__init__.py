"""
Output Module

Domain result types and rendering helpers.
"""

from .models import BottleneckType, MacroCategoryType, Diagnosis, WindowDiagnosis
from .formatter import format_results

__all__ = [
    "BottleneckType",
    "MacroCategoryType",
    "Diagnosis",
    "WindowDiagnosis",
    "format_results",
]
