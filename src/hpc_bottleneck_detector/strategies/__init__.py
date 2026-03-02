"""
Strategies Module

Bottleneck-detection strategy implementations.
"""

from .interface import IAnalysisStrategy
from .heuristic import HeuristicStrategy

__all__ = [
    "IAnalysisStrategy",
    "HeuristicStrategy",
]
