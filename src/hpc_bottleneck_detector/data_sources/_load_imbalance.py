"""
Load Imbalance Helpers

Shared by :class:`~hpc_bottleneck_detector.data_sources.csv_source.CSVDataSource`
and :class:`~hpc_bottleneck_detector.data_sources.xbat_source.XBATDataSource` to
compute the Load Imbalance Factor from per-core or per-node FLOPS/s totals.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _load_imbalance_factor(matrix: np.ndarray) -> np.ndarray:
    """
    Compute the Load Imbalance Factor per interval from a 2-D matrix of shape
    ``(n_entities, n_intervals)`` (total FLOPS/s per core/node per interval)::

        LIF[t] = (T_max[t] - T_avg[t]) / T_max[t]

    Returns a 1-D array in ``[0, 1]``; intervals where ``T_max == 0`` get ``LIF = 0``.
    """
    t_max = matrix.max(axis=0)
    t_avg = matrix.mean(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lif = np.where(t_max > 0, (t_max - t_avg) / t_max, 0.0)
    return lif


def _aligned_values(row: pd.Series, interval_cols: List[str]) -> np.ndarray:
    """
    Extract numeric interval values from row, aligned to interval_cols.
    Missing columns are filled with 0.
    """
    values = np.zeros(len(interval_cols))
    for i, col in enumerate(interval_cols):
        if col in row.index:
            v = row[col]
            values[i] = float(v) if pd.notna(v) else 0.0
    return values


def _build_imbalance_row(
    job_id: str,
    trace: str,
    interval_cols: List[str],
    imbalance: np.ndarray,
) -> dict:
    """Construct a DataFrame-compatible row dict for a load-imbalance metric."""
    row: dict = {
        "jobId": job_id,
        "group": "load_imbalance",
        "metric": "FLOPS",
        "trace": trace,
    }
    for col, val in zip(interval_cols, imbalance):
        row[col] = float(val)
    return row


def intra_node_imbalance_row(
    job_id: str,
    df: pd.DataFrame,
    interval_cols: Optional[List[str]] = None,
) -> Optional[dict]:
    """
    From core-level FLOPS traces (pattern ``<type> c<N>``, e.g. ``SP c0``,
    ``AVX512 DP c3``), compute the Load Imbalance Factor across cores for
    each interval and return a ``load_imbalance``/``FLOPS``/``intra_node``
    row, or ``None`` when fewer than two distinct cores are present.
    interval_cols defaults to df's own ``interval N`` columns when omitted.
    """
    if interval_cols is None:
        interval_cols = [c for c in df.columns if c.startswith("interval ")]

    flops_df = df[(df["group"] == "cpu") & (df["metric"] == "FLOPS")]
    if flops_df.empty:
        return None

    core_totals: Dict[str, np.ndarray] = {}
    for _, row in flops_df.iterrows():
        m = re.search(r"\bc(\d+)$", str(row["trace"]))
        if not m:
            continue
        core_id = m.group(0)
        values = _aligned_values(row, interval_cols)
        if core_id not in core_totals:
            core_totals[core_id] = values.copy()
        else:
            core_totals[core_id] += values

    if len(core_totals) < 2:
        return None

    matrix = np.stack(list(core_totals.values()))  # (n_cores, n_intervals)
    lif = _load_imbalance_factor(matrix)
    return _build_imbalance_row(job_id, "intra_node", interval_cols, lif)
