"""
Labeling Module

Runs a :class:`~hpc_bottleneck_detector.strategies.HeuristicStrategy` over
sliding windows of a job's time series and produces a flat, labelled
DataFrame that is ready for downstream ML training.

Output shape
------------
One row per time interval (same as
:meth:`~hpc_bottleneck_detector.data.DataManager.get_flat_dataframe`) plus
one additional column per real :class:`BottleneckType`:

- ``0.0``      - no bottleneck of this type detected for this interval.
- ``NaN``      - assessment unknown (at least one strategy tree could not run
                 due to missing metrics, and no real bottleneck was found).
- ``(0, 1]``   - bottleneck detected; value is the max severity score across
                 all windows that cover this interval.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set

import pandas as pd

from ..data.manager import DataManager
from ..output.models import BottleneckType, Diagnosis
from ..strategies.heuristic import HeuristicStrategy
from ..strategies.property_node import PropertyNode


# Ordered list of real bottleneck types that become output columns.
# NONE and UNKNOWN are sentinel values and are excluded.
BOTTLENECK_COLUMNS: List[BottleneckType] = [
    BottleneckType.PIPELINE_STALL,
    BottleneckType.COMPUTE_UNDERUTILIZATION,
    BottleneckType.PRECISION_WASTE,
    BottleneckType.BRANCH_MISPREDICTION,
    BottleneckType.CACHE_PRESSURE,
    BottleneckType.MEMORY_BANDWIDTH,
    BottleneckType.INTRA_NODE_LOAD_IMBALANCE,
    BottleneckType.INTER_NODE_LOAD_IMBALANCE,
]

_REAL_TYPES: Set[BottleneckType] = set(BOTTLENECK_COLUMNS)


def _get_leaf_types(node: PropertyNode) -> Set[BottleneckType]:
    """Recursively collect all real BottleneckTypes reachable from *node*."""
    if node.is_leaf():
        bt_name = node._diag_cfg.get("bottleneck_type", "NONE")
        try:
            bt = BottleneckType[bt_name]
            return {bt} if bt in _REAL_TYPES else set()
        except KeyError:
            return set()
    return _get_leaf_types(node._if_true) | _get_leaf_types(node._if_false)


def _build_tree_type_map(strategy: HeuristicStrategy) -> Dict[str, Set[BottleneckType]]:
    """Return a mapping ``tree_name → set of BottleneckTypes it can produce``."""
    return {
        tree.tree_name: _get_leaf_types(tree.root_node)
        for tree in strategy._strategy_trees
    }


def _window_severity(
    diagnoses: List[Diagnosis],
    tree_type_map: Dict[str, Set[BottleneckType]],
) -> Dict[str, float]:
    """
    Collapse a window's diagnoses into one severity value per bottleneck column.

    Rules:

    - Real bottleneck detected → max severity score across all matching diagnoses.
    - Not detected AND a tree that *can* produce this type was UNKNOWN → ``NaN``
      (the relevant tree couldn't run; absence cannot be confirmed).
    - Not detected AND no responsible tree was UNKNOWN → ``0.0``
      (all relevant trees ran cleanly and found nothing).
    """
    # Collect bottleneck types covered by trees that returned UNKNOWN.
    unknown_covered: Set[BottleneckType] = set()
    for d in diagnoses:
        if d.is_unknown:
            unknown_covered |= tree_type_map.get(d.source, set())

    result: Dict[str, float] = {}
    for bt in BOTTLENECK_COLUMNS:
        col = bt.value
        matching = [d.severity_score for d in diagnoses if d.bottleneck_type is bt]
        if matching:
            result[col] = max(matching)
        elif bt in unknown_covered:
            result[col] = float("nan")
        else:
            result[col] = 0.0
    return result


def label_job(
    data_mgr: DataManager,
    strategy: HeuristicStrategy,
    window_size: int,
    step_size: int,
    interval_seconds: int = 5,
) -> pd.DataFrame:
    """
    Label every time interval in *data_mgr* with bottleneck severity scores.

    The function slides a window of *window_size* intervals over the full
    time series (advancing *step_size* intervals at a time), runs
    ``strategy.diagnose()`` on each window, and maps the resulting
    :class:`~hpc_bottleneck_detector.output.models.Diagnosis` objects back to
    individual intervals.
    """
    tree_type_map = _build_tree_type_map(strategy)
    flat = data_mgr.get_flat_dataframe(interval_seconds)
    n_intervals = len(flat)

    # Per-interval accumulator: col_name → list of severity values from
    # each window that covers that interval (NaN for UNKNOWN windows).
    accum: List[Dict[str, List[float]]] = [
        {bt.value: [] for bt in BOTTLENECK_COLUMNS}
        for _ in range(n_intervals)
    ]

    for start, end_inclusive, window_dm in data_mgr.iterate_windows(window_size, step_size):
        window_sev = _window_severity(strategy.diagnose(window_dm), tree_type_map)
        for interval_idx in range(start, min(end_inclusive + 1, n_intervals)):
            for col_name, val in window_sev.items():
                accum[interval_idx][col_name].append(val)

    # Collapse each interval's accumulator into a single value per column.
    for bt in BOTTLENECK_COLUMNS:
        col_name = bt.value
        series: List[float] = []
        for idx in range(n_intervals):
            vals = accum[idx][col_name]
            if not vals:
                # Interval not covered by any window (e.g. trailing stub).
                series.append(float("nan"))
                continue
            real_vals = [v for v in vals if not math.isnan(v)]
            if real_vals:
                series.append(max(real_vals))
            else:
                # All covering windows were UNKNOWN for this bottleneck type.
                series.append(float("nan"))
        flat[col_name] = series

    return flat
