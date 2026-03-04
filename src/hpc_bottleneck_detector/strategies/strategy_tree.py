"""
Strategy Tree

A single heuristic decision tree loaded from a YAML file.

Each YAML file in the specified folder maps to one
:class:`StrategyTree` instance.  The tree is traversed depth-first by
:meth:`traverse`, which walks from the root :class:`PropertyNode` down to
the first matching leaf and returns a fully evaluated
:class:`~hpc_bottleneck_detector.output.models.Diagnosis`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, TYPE_CHECKING

import yaml

from .property_node import PropertyNode, _metric_label
from ..output.models import BottleneckType, Diagnosis

if TYPE_CHECKING:
    from ..data.manager import DataManager

logger = logging.getLogger(__name__)


class StrategyTree:
    """
    A decision tree loaded from a single YAML strategy file.

    Attributes:
        tree_name:    Human-readable name of this tree (from YAML).
        description:  Short description of what the tree detects.
        config_path:  Absolute path to the source YAML file.
        root_node:    Root :class:`PropertyNode` of the tree.
    """

    def __init__(
        self,
        tree_name: str,
        description: str,
        root_node: PropertyNode,
        required_metrics: List[dict],
        config_path: str = "",
    ) -> None:
        self.tree_name        = tree_name
        self.description      = description
        self.root_node        = root_node
        self._required_metrics = required_metrics
        self.config_path      = config_path

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load_from_yaml(cls, path: str) -> "StrategyTree":
        """
        Build a :class:`StrategyTree` from a YAML file.

        Args:
            path: Path to the YAML strategy file.

        Returns:
            Loaded and validated :class:`StrategyTree`.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If a required field is missing from the YAML.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Strategy YAML not found: {path}")

        with p.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        tree_name   = config.get("tree_name", p.stem)
        description = config.get("description", "").strip()
        req_metrics = config.get("required_metrics", [])
        root_node   = PropertyNode(config["root"])

        logger.debug("Loaded strategy tree '%s' from '%s'.", tree_name, path)

        return cls(
            tree_name=tree_name,
            description=description,
            root_node=root_node,
            required_metrics=req_metrics,
            config_path=str(p.resolve()),
        )

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def traverse(self, data_mgr: "DataManager") -> Diagnosis:
        """
        Walk the decision tree and return the resulting :class:`Diagnosis`.

        The traversal follows ``if_true`` / ``if_false`` branches based on the
        evaluated metric condition at each decision node.  When a leaf is
        reached, :meth:`~PropertyNode.get_diagnosis` is called to produce the
        final diagnosis with the computed severity score.

        If any required metric is **missing** from *data_mgr* the tree cannot
        be evaluated and a ``NONE`` diagnosis is returned (the missing metric
        is noted in ``triggered_metrics``).

        Args:
            data_mgr: :class:`~hpc_bottleneck_detector.data.manager.DataManager`
                      scoped to the current analysis window.

        Returns:
            A single :class:`~hpc_bottleneck_detector.output.models.Diagnosis`.
        """
        # ── Quick check: are all required metrics present? ─────────────
        missing = self._missing_metrics(data_mgr)
        if missing:
            logger.debug(
                "Tree '%s': cannot evaluate — missing metrics: %s.",
                self.tree_name, missing,
            )
            return Diagnosis(
                bottleneck_type=BottleneckType.UNKNOWN,
                severity_score=0.0,
                confidence=1.0,
                recommendation=(
                    f"Tree '{self.tree_name}' could not be evaluated because "
                    f"the following metrics are absent from the data: "
                    + ", ".join(missing)
                ),
                source=self.tree_name,
                triggered_metrics=[f"MISSING:{m}" for m in missing],
            )

        # ── Walk the tree ──────────────────────────────────────────────
        node           = self.root_node
        triggered      : List[str] = []
        last_value     : float     = 0.0
        last_threshold : float     = 0.0

        while not node.is_leaf():
            try:
                branch, value, threshold = node.evaluate(data_mgr)
            except ValueError as exc:
                logger.warning(
                    "Tree '%s', node '%s': metric evaluation failed (%s). "
                    "Returning UNKNOWN.",
                    self.tree_name, node.node_id, exc,
                )
                return Diagnosis(
                    bottleneck_type=BottleneckType.UNKNOWN,
                    severity_score=0.0,
                    confidence=1.0,
                    recommendation=(
                        f"Tree '{self.tree_name}' encountered an evaluation "
                        f"error at node '{node.node_id}': {exc}"
                    ),
                    source=self.tree_name,
                    triggered_metrics=triggered,
                )

            # Record the metric that was tested at this node
            triggered.append(_metric_label(node._metric_cfg))
            last_value     = value
            last_threshold = threshold
            node = node.get_child(branch)

        # ── node is a leaf - build the diagnosis ──────────────────────
        return node.get_diagnosis(
            source=self.tree_name,
            triggered_metrics=triggered,
            metric_value=last_value,
        )

    # ------------------------------------------------------------------
    # Required metrics
    # ------------------------------------------------------------------

    def get_required_metrics(self) -> List[dict]:
        """
        Return the list of metric spec dicts declared in ``required_metrics``.

        Each element is a dict with at least ``group`` and ``metric`` keys,
        and optionally ``trace``.
        """
        return list(self._required_metrics)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _missing_metrics(self, data_mgr: "DataManager") -> List[str]:
        """Return labels of required metrics not found in *data_mgr*."""
        missing = []
        for spec in self._required_metrics:
            group  = spec.get("group", "")
            metric = spec.get("metric", "")
            trace  = spec.get("trace")
            if not data_mgr.has_metric(group, metric, trace):
                label = f"{group}/{metric}/{trace}" if trace else f"{group}/{metric}"
                missing.append(label)
        return missing

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"StrategyTree(name={self.tree_name!r}, path={self.config_path!r})"
