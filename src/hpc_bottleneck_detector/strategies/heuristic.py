"""
Heuristic Strategy

Rule-based bottleneck detection strategy.  Every ``*.yaml`` file found in
*strategy_folder* is loaded as an independent
:class:`~hpc_bottleneck_detector.strategies.strategy_tree.StrategyTree`.

During :meth:`diagnose` all trees are executed against the supplied window
:class:`~hpc_bottleneck_detector.data.manager.DataManager`.  Trees whose
required metrics are missing are silently skipped (they return a
zero-confidence NONE diagnosis).  Only non-healthy diagnoses are returned;
if all trees pass clean a single NONE diagnosis is returned.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from .interface import IAnalysisStrategy
from .strategy_tree import StrategyTree
from ..data.manager import DataManager
from ..output.models import BottleneckType, Diagnosis

logger = logging.getLogger(__name__)


class HeuristicStrategy(IAnalysisStrategy):
    """
    Rule-based bottleneck detection strategy.

    Each ``*.yaml`` file in *strategy_folder* is loaded as a separate
    :class:`~hpc_bottleneck_detector.strategies.strategy_tree.StrategyTree`.
    All trees are evaluated for every analysis window; results are merged so
    that only meaningful findings are returned.

    Args:
        strategy_folder: Path to the directory containing strategy YAML files.
                         If *None*, no trees are loaded and every window will
                         be reported as healthy.
    """

    def __init__(self, strategy_folder: Optional[str] = None) -> None:
        self.strategy_folder = Path(strategy_folder) if strategy_folder else None
        self._strategy_trees: List[StrategyTree] = []

        if self.strategy_folder:
            self._load_trees(self.strategy_folder)
        else:
            logger.info(
                "HeuristicStrategy: no strategy_folder provided; "
                "no trees loaded - every window will be diagnosed as healthy."
            )

    # ------------------------------------------------------------------
    # IAnalysisStrategy interface
    # ------------------------------------------------------------------

    def diagnose(self, data_mgr: DataManager) -> List[Diagnosis]:
        """
        Run all loaded strategy trees against *data_mgr* and return the
        aggregated bottleneck findings.

        Trees whose required metrics are not present in *data_mgr* are
        skipped gracefully (zero-confidence NONE result).

        Returns:
            A list of :class:`~hpc_bottleneck_detector.output.models.Diagnosis`
            objects.  Contains at least one entry (``NONE`` if nothing is
            detected or no trees are loaded).
        """
        if not self._strategy_trees:
            return [
                Diagnosis(
                    bottleneck_type=BottleneckType.NONE,
                    severity_score=0.0,
                    confidence=1.0,
                    source="heuristic (no trees loaded)",
                )
            ]

        findings: List[Diagnosis] = []

        for tree in self._strategy_trees:
            diag = tree.traverse(data_mgr)
            findings.append(diag)

        # Partition results into three buckets
        bottleneck_findings = [d for d in findings if not d.is_healthy and not d.is_unknown]
        unknown_findings    = [d for d in findings if d.is_unknown]

        # Return real bottlenecks first; if none, surface any UNKNOWNs so the
        # user knows some trees could not be evaluated; otherwise healthy.
        if bottleneck_findings:
            return bottleneck_findings + unknown_findings
        if unknown_findings:
            return unknown_findings
        return [
            Diagnosis(
                bottleneck_type=BottleneckType.NONE,
                severity_score=0.0,
                confidence=1.0,
                source="heuristic",
            )
        ]

    def get_required_metrics(self) -> List[str]:
        """Return the union of all metric labels required by all loaded trees."""
        seen   : set       = set()
        labels : List[str] = []
        for tree in self._strategy_trees:
            for spec in tree.get_required_metrics():
                group  = spec.get("group", "")
                metric = spec.get("metric", "")
                trace  = spec.get("trace", "")
                label  = f"{group}/{metric}/{trace}" if trace else f"{group}/{metric}"
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
        return labels

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_trees(self, folder: Path) -> None:
        """Load every ``*.yaml`` file in *folder* as a :class:`StrategyTree`."""
        yaml_files = sorted(folder.glob("*.yaml"))

        if not yaml_files:
            logger.warning(
                "HeuristicStrategy: no *.yaml files found in '%s'.", folder
            )
            return

        for yaml_path in yaml_files:
            try:
                tree = StrategyTree.load_from_yaml(str(yaml_path))
                self._strategy_trees.append(tree)
                logger.info(
                    "HeuristicStrategy: loaded tree '%s' from '%s'.",
                    tree.tree_name, yaml_path.name,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "HeuristicStrategy: failed to load '%s': %s",
                    yaml_path.name, exc,
                )

        logger.info(
            "HeuristicStrategy: %d tree(s) loaded from '%s'.",
            len(self._strategy_trees), folder,
        )
