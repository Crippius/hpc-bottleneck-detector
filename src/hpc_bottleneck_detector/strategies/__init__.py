"""
Strategies Module

Bottleneck-detection strategy implementations.
"""

from .interface import IAnalysisStrategy
from .property_node import PropertyNode
from .strategy_tree import StrategyTree
from .heuristic import HeuristicStrategy

# SupervisedMLStrategy is imported lazily to avoid requiring sklearn/tsfresh
# when the heuristic strategy is used.  Import it directly when needed:
#   from hpc_bottleneck_detector.strategies.supervised_ml import SupervisedMLStrategy

__all__ = [
    "IAnalysisStrategy",
    "PropertyNode",
    "StrategyTree",
    "HeuristicStrategy",
]
