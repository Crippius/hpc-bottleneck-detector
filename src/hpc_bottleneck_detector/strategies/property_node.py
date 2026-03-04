"""
Property Node

A single node in a heuristic decision tree.  Nodes are either:

- **Decision nodes** — compare an aggregated metric against a threshold
  and branch to ``if_true`` or ``if_false`` child nodes.
- **Leaf nodes** — carry a :class:`~hpc_bottleneck_detector.output.models.Diagnosis`
  configuration and terminate the traversal.

The recursive tree is built directly from the YAML representation by
:class:`~hpc_bottleneck_detector.strategies.strategy_tree.StrategyTree`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..data.manager import DataManager

from ..output.models import BottleneckType, Diagnosis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _get_series(metric_cfg: dict, data_mgr: "DataManager") -> pd.Series:
    """
    Resolve a metric configuration to a pandas Series of interval values.

    Handles three cases:

    * **simple** ``{group, metric, trace}`` — direct DataManager lookup.
    * **sum** ``{type: sum, operands: [...]}`` — element-wise sum of operands.
    * **ratio** ``{type: ratio, numerator: ..., denominator: ...}`` —
      element-wise division; zeros in denominator become NaN.

    Raises:
        ValueError: If a required simple metric is absent from *data_mgr*.
    """
    kind = metric_cfg.get("type")

    if kind == "sum":
        series_list = [_get_series(op, data_mgr) for op in metric_cfg["operands"]]
        result = series_list[0].copy().astype(float)
        for s in series_list[1:]:
            result = result.add(s.astype(float), fill_value=0.0)
        return result

    if kind == "ratio":
        num = _get_series(metric_cfg["numerator"], data_mgr).astype(float)
        den = _get_series(metric_cfg["denominator"], data_mgr).astype(float)
        den_safe = den.replace(0.0, float("nan"))
        return num / den_safe

    # Simple metric
    return data_mgr.get_metric(
        group=metric_cfg["group"],
        metric=metric_cfg["metric"],
        trace=metric_cfg.get("trace"),
    ).astype(float)


def _aggregate(series: pd.Series, aggregation: str) -> float:
    """Reduce a Series to a scalar using the named aggregation."""
    agg = aggregation.lower()
    if agg == "mean":
        return float(series.mean())
    if agg == "min":
        return float(series.min())
    if agg == "max":
        return float(series.max())
    if agg in ("sum", "total"):
        return float(series.sum())
    if agg == "median":
        return float(series.median())
    raise ValueError(f"Unknown aggregation: '{aggregation}'")


def _resolve_threshold(threshold_cfg: Any, data_mgr: "DataManager") -> float:
    """
    Resolve a threshold specification to a concrete float.

    Simple thresholds are plain numbers.  Benchmark-derived thresholds have
    the form::

        benchmark: bandwidth_mem
        fraction:  0.85
        aggregate: mean          # optional, default 'mean'
        fallback:  40.0          # used when JobContext is unavailable

    Returns:
        Concrete threshold value.
    """
    if isinstance(threshold_cfg, (int, float)):
        return float(threshold_cfg)

    if isinstance(threshold_cfg, dict) and "benchmark" in threshold_cfg:
        key       = threshold_cfg["benchmark"]
        fraction  = float(threshold_cfg.get("fraction", 1.0))
        agg       = threshold_cfg.get("aggregate", "mean")
        fallback  = float(threshold_cfg.get("fallback", 0.0))

        ctx = data_mgr.job_context
        if ctx is not None:
            peak = ctx.get_benchmark(key, aggregate=agg)
            if peak is not None:
                return peak * fraction

        logger.debug(
            "Benchmark '%s' not available in JobContext; using fallback %.3g.",
            key, fallback,
        )
        return fallback

    # Shouldn't happen with valid YAML, but be safe
    return float(threshold_cfg)


def _compare(value: float, operator: str, threshold: float) -> bool:
    """Apply a comparison operator."""
    ops = {
        ">":  value >  threshold,
        "<":  value <  threshold,
        ">=": value >= threshold,
        "<=": value <= threshold,
        "==": value == threshold,
        "!=": value != threshold,
    }
    if operator not in ops:
        raise ValueError(f"Unknown operator: '{operator}'")
    return ops[operator]


def _compute_severity(
    formula: str,
    value: float,
    threshold: float,
) -> float:
    """
    Evaluate a severity formula.

    Supported formulas:

    * ``FORMULA1`` — ``min(1, max(0, value / threshold - 1))``
      Higher metric value → higher severity.
    * ``FORMULA2`` — ``min(1, max(0, 1 - value / threshold))``
      Lower metric value → higher severity.
    * Any other string is interpreted as a Python float literal (e.g. ``"0.0"``).
    """
    if formula == "FORMULA1":
        if threshold == 0:
            return 0.0
        return float(min(1.0, max(0.0, value / threshold - 1.0)))
    if formula == "FORMULA2":
        if threshold == 0:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - value / threshold)))
    try:
        return float(formula)
    except (ValueError, TypeError):
        logger.warning("Unknown severity formula '%s'; defaulting to 0.5.", formula)
        return 0.5


def _metric_label(metric_cfg: dict) -> str:
    """Return a short human-readable label for a metric config."""
    kind = metric_cfg.get("type")
    if kind == "sum":
        parts = [_metric_label(op) for op in metric_cfg["operands"]]
        return "(" + " + ".join(parts) + ")"
    if kind == "ratio":
        return (
            _metric_label(metric_cfg["numerator"])
            + " / "
            + _metric_label(metric_cfg["denominator"])
        )
    group = metric_cfg.get("group", "")
    metric = metric_cfg.get("metric", "")
    trace  = metric_cfg.get("trace", "")
    return f"{group}/{metric}/{trace}" if trace else f"{group}/{metric}"


# ---------------------------------------------------------------------------
# PropertyNode
# ---------------------------------------------------------------------------

class PropertyNode:
    """
    A single node in a heuristic decision tree.

    Attributes:
        node_id:     Unique identifier for this node (from YAML).
        description: Human-readable description of the check.
    """

    def __init__(self, config: dict) -> None:
        self.node_id: str     = config["node_id"]
        self.description: str = config.get("description", "")

        if "diagnosis" in config:
            # ── Leaf node ──────────────────────────────────────────────
            self._is_leaf = True
            self._diag_cfg = config["diagnosis"]
            self._metric_cfg  = None
            self._aggregation = None
            self._operator    = None
            self._threshold_cfg = None
            self._if_true  = None
            self._if_false = None
        else:
            # ── Decision node ──────────────────────────────────────────
            self._is_leaf = False
            self._diag_cfg = None
            self._metric_cfg    = config["metric"]
            self._aggregation   = config.get("aggregation", "mean")
            self._operator      = config["operator"]
            self._threshold_cfg = config["threshold"]
            self._if_true  = PropertyNode(config["if_true"])
            self._if_false = PropertyNode(config["if_false"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_leaf(self) -> bool:
        """True if this node terminates the traversal."""
        return self._is_leaf

    def evaluate(self, data_mgr: "DataManager") -> tuple[bool, float, float]:
        """
        Evaluate the decision condition at this node.

        Returns:
            Tuple ``(branch, metric_value, resolved_threshold)`` where
            *branch* is True if the ``if_true`` child should be followed.

        Raises:
            ValueError: If a required metric is missing.
        """
        series    = _get_series(self._metric_cfg, data_mgr)
        value     = _aggregate(series, self._aggregation)
        threshold = _resolve_threshold(self._threshold_cfg, data_mgr)
        branch    = _compare(value, self._operator, threshold)

        logger.debug(
            "Node '%s': metric_value=%.6g  %s  threshold=%.6g  → %s",
            self.node_id, value, self._operator, threshold,
            "if_true" if branch else "if_false",
        )
        return branch, value, threshold

    def get_child(self, branch: bool) -> "PropertyNode":
        """Return the ``if_true`` or ``if_false`` child."""
        return self._if_true if branch else self._if_false

    def get_diagnosis(
        self,
        source: str,
        triggered_metrics: List[str],
        metric_value: Optional[float] = None,
    ) -> Diagnosis:
        """
        Build a :class:`~hpc_bottleneck_detector.output.models.Diagnosis`
        from this leaf node's configuration.

        Args:
            source:            Name of the enclosing :class:`StrategyTree`.
            triggered_metrics: Metric labels that fired along the path.
            metric_value:      The last evaluated metric value; used to
                               compute severity when a formula is given.
        """
        cfg = self._diag_cfg

        # ── Bottleneck type ────────────────────────────────────────────
        bt_name = cfg.get("bottleneck_type", "NONE")
        try:
            bt = BottleneckType[bt_name]
        except KeyError:
            logger.warning("Unknown bottleneck_type '%s'; defaulting to NONE.", bt_name)
            bt = BottleneckType.NONE

        # ── Severity ───────────────────────────────────────────────────
        formula   = str(cfg.get("severity_formula", "0.0"))
        sev_threshold = float(cfg.get("threshold", 1.0))

        if metric_value is not None and formula in ("FORMULA1", "FORMULA2"):
            severity = _compute_severity(formula, metric_value, sev_threshold)
        else:
            severity = _compute_severity(formula, 0.0, sev_threshold)

        # ── Other fields ───────────────────────────────────────────────
        confidence     = float(cfg.get("confidence", 1.0))
        recommendation = cfg.get("recommendation")
        if isinstance(recommendation, str):
            recommendation = recommendation.strip() or None

        return Diagnosis(
            bottleneck_type=bt,
            severity_score=severity,
            confidence=confidence,
            recommendation=recommendation,
            source=source,
            triggered_metrics=triggered_metrics,
        )

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        kind = "leaf" if self._is_leaf else "decision"
        return f"PropertyNode(id={self.node_id!r}, kind={kind})"
