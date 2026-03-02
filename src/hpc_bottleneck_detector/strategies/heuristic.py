"""
Heuristic Strategy — Stub

This is a placeholder implementation of :class:`IAnalysisStrategy` that
always returns a healthy (``NONE``) diagnosis.  The full implementation
(decision-tree traversal loaded from YAML files) will be added later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .interface import IAnalysisStrategy
from ..data.manager import DataManager
from ..output.models import Diagnosis, BottleneckType

logger = logging.getLogger(__name__)


class HeuristicStrategy(IAnalysisStrategy):
    """
    Rule-based bottleneck detection strategy.

    Each ``*.yaml`` file found in *strategy_folder* will eventually be
    loaded as an independent decision tree (``StrategyTree``).  For now
    this class is a **stub**: it logs that it was invoked and returns a
    healthy diagnosis for every window.

    Args:
        strategy_folder: Path to the directory containing strategy YAML files.
    """

    def __init__(self, strategy_folder: Optional[str] = None) -> None:
        self.strategy_folder = Path(strategy_folder) if strategy_folder else None
        self._strategy_trees: list = []  # will hold StrategyTree instances

        if self.strategy_folder:
            yaml_files = sorted(self.strategy_folder.glob("*.yaml"))
            logger.info(
                "HeuristicStrategy: found %d YAML strategy file(s) in '%s'. "
                "(loading is not yet implemented — stub mode)",
                len(yaml_files),
                self.strategy_folder,
            )
        else:
            logger.info("HeuristicStrategy: no strategy folder provided (stub mode).")

    # ------------------------------------------------------------------
    # IAnalysisStrategy interface
    # ------------------------------------------------------------------

    def diagnose(self, data_mgr: DataManager) -> List[Diagnosis]:
        """
        Stub implementation — always returns a single healthy diagnosis.

        TODO: implement decision-tree traversal once StrategyTree is built.
        """
        logger.debug(
            "HeuristicStrategy.diagnose called for job '%s' "
            "(%d interval(s)) — returning NONE (stub).",
            data_mgr.job_id,
            data_mgr.get_time_series_length(),
        )
        return [
            Diagnosis(
                bottleneck_type=BottleneckType.NONE,
                severity_score=0.0,
                confidence=1.0,
                recommendation=None,
                source="heuristic_stub",
            )
        ]

    def get_required_metrics(self) -> List[str]:
        """
        Return an empty list until the real trees are loaded.

        TODO: aggregate required_metrics from each loaded StrategyTree.
        """
        return []
