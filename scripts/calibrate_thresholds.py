#!/usr/bin/env python3
"""
Threshold Calibration Script — Pareto Rule
==========================================

Calculates statistically-derived thresholds for the persyst strategy trees
using the 80/20 Pareto rule across a diverse set of HPC mini-application jobs.


Usage
-----
    python scripts/calibrate_thresholds.py JOB_ID [JOB_ID ...]

    # or hard-code IDs in DEFAULT_JOB_IDS and run without arguments:
    python scripts/calibrate_thresholds.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# -- path setup ----------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpc_bottleneck_detector.data.manager import DataManager
from hpc_bottleneck_detector.data_sources import XBATDataSource
from hpc_bottleneck_detector.strategies.property_node import _aggregate, _get_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_JOB_IDS: List[str] = []

WINDOW_SIZE: int = 10  # intervals per window (matches orchestrator default)

STRATEGY_DIR: Path = (
    Path(__file__).parent.parent / "configs" / "strategies" / "persyst_strategy"
)

OUTPUT_CSV: Path = Path("calibration_results.csv")


# =============================================================================
# YAML parsing
# =============================================================================


def _dedup_key(metric_cfg: dict, operator: str) -> str:
    """Stable hashable key to detect duplicate decision nodes across trees."""
    return json.dumps(metric_cfg, sort_keys=True) + "|" + operator


def _walk_node(
    node: Optional[dict],
    tree_name: str,
    collected: Dict[str, dict],
) -> None:
    """
    Recursively walk a strategy tree node.

    For every *decision* node, extract the metric definition and add it to
    ``collected`` (keyed by dedup key).  Leaf nodes are skipped.
    """
    if node is None or "diagnosis" in node:
        return  # leaf node - nothing to calibrate

    metric_cfg    = node["metric"]
    aggregation   = node.get("aggregation", "mean")
    operator      = node["operator"]
    threshold_cfg = node["threshold"]

    # Direction is determined by the comparison operator
    direction = "INCREASING" if operator in (">", ">=") else "DECREASING"

    # Current threshold value and optional benchmark key
    if isinstance(threshold_cfg, dict) and "benchmark" in threshold_cfg:
        benchmark_key     = threshold_cfg["benchmark"]
        current_threshold = float(threshold_cfg.get("fraction", 1.0))
    else:
        benchmark_key     = None
        current_threshold = float(threshold_cfg)

    key = _dedup_key(metric_cfg, operator)

    if key in collected:
        # Merge tree name into existing entry
        existing = collected[key]["tree"]
        if tree_name not in existing.split(" / "):
            collected[key]["tree"] = existing + " / " + tree_name
    else:
        # Ensure the name is unique within collected (fallback for collisions)
        name = node.get("node_id")
        existing_names = {v["name"] for v in collected.values()}
        counter = 2
        while name in existing_names:
            name = f"{name}_{counter}"
            counter += 1

        collected[key] = {
            "name":              name,
            "description":       node.get("description", name),
            "tree":              tree_name,
            "metric_cfg":        metric_cfg,
            "aggregation":       aggregation,
            "direction":         direction,
            "benchmark_key":     benchmark_key,
            "current_threshold": current_threshold,
        }

    _walk_node(node.get("if_true"),  tree_name, collected)
    _walk_node(node.get("if_false"), tree_name, collected)


def load_metric_definitions(strategy_dir: Path) -> List[dict]:
    """
    Parse all ``*.yaml`` files in *strategy_dir* and return a deduplicated
    list of metric definitions, one per unique (metric_cfg, operator) pair.
    """
    collected: Dict[str, dict] = {}

    yaml_files = sorted(strategy_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML strategy files found in {strategy_dir}")

    for yaml_file in yaml_files:
        with yaml_file.open() as fh:
            tree = yaml.safe_load(fh)
        tree_name = tree.get("tree_name", yaml_file.stem)
        log.info("  Parsed  %s  (%s)", tree_name, yaml_file.name)
        _walk_node(tree.get("root"), tree_name, collected)

    metric_defs = list(collected.values())
    log.info("Found %d unique decision-node metrics across all trees.", len(metric_defs))
    return metric_defs


# =============================================================================
# Core computation helpers
# =============================================================================

def _compute_window_values(
    data_mgr: DataManager,
    metric_def: dict,
    window_size: int,
) -> List[float]:
    """
    Compute the metric scalar for every non-overlapping window of a job.

    Returns a list of finite values; NaN / Inf windows are silently skipped.
    """
    metric_cfg    = metric_def["metric_cfg"]
    aggregation   = metric_def["aggregation"]
    benchmark_key = metric_def.get("benchmark_key")

    # Pre-compute normalisation factor from the hardware peak (once per job).
    norm_factor: float = 1.0
    if benchmark_key and data_mgr.job_context is not None:
        peak = data_mgr.job_context.get_benchmark(benchmark_key, aggregate="mean")
        if peak and peak > 0.0:
            norm_factor = peak

    values: List[float] = []
    for _, _, win_dm in data_mgr.iterate_windows(
        window_size=window_size,
        step_size=window_size,
    ):
        try:
            series = _get_series(metric_cfg, win_dm)
            value  = _aggregate(series, aggregation)
        except (ValueError, ZeroDivisionError, KeyError):
            continue

        if not np.isfinite(value):
            continue

        if benchmark_key and norm_factor != 1.0:
            value = value / norm_factor  # raw value -> fraction of peak

        values.append(value)

    return values


def _collect_per_job_means(
    job_ids: List[str],
    data_source: XBATDataSource,
    metric_def: dict,
    window_size: int,
) -> Tuple[List[float], Dict[str, dict]]:
    """
    For each job compute per-window values, reduce to a single mean, and
    return the list of per-job means together with per-job statistics.
    """
    per_job_means: List[float] = []
    job_stats: Dict[str, dict] = {}

    for job_id in job_ids:
        log.info("  Job %s ...", job_id)
        try:
            data_mgr = data_source.fetch_job_data(job_id)
        except Exception as exc:
            log.warning("  Skipping job %s: %s", job_id, exc)
            continue

        n_intervals = data_mgr.get_time_series_length()
        window_vals = _compute_window_values(data_mgr, metric_def, window_size)

        if not window_vals:
            log.debug(
                "  No valid windows for job %s / metric '%s'.",
                job_id, metric_def["name"],
            )
            continue

        job_mean = float(np.mean(window_vals))
        per_job_means.append(job_mean)
        job_stats[job_id] = {
            "n_intervals": n_intervals,
            "n_windows":   len(window_vals),
            "mean":        job_mean,
            "std":         float(np.std(window_vals)),
            "min":         float(np.min(window_vals)),
            "max":         float(np.max(window_vals)),
        }
        log.info(
            "    mean=%.4g  std=%.4g  n_windows=%d",
            job_mean, job_stats[job_id]["std"], len(window_vals),
        )

    return per_job_means, job_stats


def _pareto_threshold(values: List[float], direction: str) -> float:
    """
    Apply the Pareto rule:
    - INCREASING -> 80th percentile (flag top 20 % as bottleneck).
    - DECREASING -> 20th percentile (flag bottom 20 % as bottleneck).
    """
    if not values:
        return float("nan")
    pct = 80 if direction == "INCREASING" else 20
    return float(np.percentile(values, pct))


# =============================================================================
# Output helpers
# =============================================================================

def _print_summary_table(results: List[dict]) -> None:
    W = 42
    print()
    print("=" * 100)
    print("  THRESHOLD CALIBRATION RESULTS  --  Pareto Rule (80/20)")
    print("=" * 100)
    print(
        f"{'Metric':<{W}} {'Dir':>4}  "
        f"{'Current':>10}  {'Pareto':>10}  {'N jobs':>6}  {'Change':>9}  Percentile"
    )
    print("-" * 100)

    for r in results:
        pct_label = "80th" if r["direction"] == "INCREASING" else "20th"
        if np.isnan(r["pareto_threshold"]):
            pareto_str = "N/A"
            change_str = "N/A"
        else:
            pareto_str = f"{r['pareto_threshold']:.4g}"
            cur = r["current_threshold"]
            change_str = f"{(r['pareto_threshold'] - cur) / cur * 100:+.1f}%" if cur else "N/A"

        print(
            f"{r['name']:<{W}} {r['direction'][:4]:>4}  "
            f"{r['current_threshold']:>10.4g}  "
            f"{pareto_str:>10}  "
            f"{r['n_jobs']:>6}  "
            f"{change_str:>9}  {pct_label}"
        )

    print("=" * 100)
    print()
    print("Notes:")
    print("  INCR = INCREASING severity -- higher value is worse -- 80th percentile")
    print("  DECR = DECREASING severity -- lower  value is worse -- 20th percentile")
    print(
        "  Benchmark-relative metrics are expressed as fractions of the hardware peak\n"
        "  (ready to paste into the `fraction:` field in the YAML tree)."
    )
    print()


def _print_yaml_suggestions(results: List[dict]) -> None:
    print()
    print("=" * 100)
    print("  SUGGESTED YAML THRESHOLD VALUES")
    print("=" * 100)

    by_tree: Dict[str, List[dict]] = {}
    for r in results:
        by_tree.setdefault(r["tree"], []).append(r)

    for tree, metrics in by_tree.items():
        print(f"\n# -- {tree} --")
        for r in metrics:
            if np.isnan(r["pareto_threshold"]):
                print(
                    f"  # {r['name']}: insufficient data"
                    f" -- keep current value {r['current_threshold']}"
                )
                continue

            val = r["pareto_threshold"]
            bk  = r.get("benchmark_key")
            if bk:
                print(f"  # {r['name']}  (fraction of benchmark '{bk}'):")
                print(f"  #   fraction: {val:.4f}  # was {r['current_threshold']:.4f}")
            else:
                print(
                    f"  # {r['name']}:"
                    f"  threshold: {val:.4g}"
                    f"  # was {r['current_threshold']:.4g}"
                )
    print()


# =============================================================================
# Entry point
# =============================================================================

# python scripts/calibrate_thresholds.py 43325 43319 43298 43290 43272 43260 43236 43195 43141 43129 43118

def main(job_ids: List[str]) -> None:
    if not job_ids:
        print(__doc__)
        print("ERROR: No job IDs provided.")
        print("  python scripts/calibrate_thresholds.py 43081 43082 ...")
        sys.exit(1)

    # -- Load metric definitions from the YAML trees --------------------------
    log.info("Loading metric definitions from %s ...", STRATEGY_DIR)
    metric_defs = load_metric_definitions(STRATEGY_DIR)

    # -- Connect to XBAT -------------------------------------------------------
    log.info("Connecting to XBAT (reading .env) ...")
    data_source = XBATDataSource.from_env(env_file=".env")

    # -- Calibrate each metric -------------------------------------------------
    all_results: List[dict] = []
    all_rows:    List[dict] = []

    for metric_def in metric_defs:
        log.info("")
        log.info("-- Calibrating '%s'  [%s] --", metric_def["name"], metric_def["direction"])

        per_job_means, job_stats = _collect_per_job_means(
            job_ids, data_source, metric_def, WINDOW_SIZE
        )
        pareto_thr = _pareto_threshold(per_job_means, metric_def["direction"])

        all_results.append({
            **metric_def,
            "pareto_threshold": pareto_thr,
            "n_jobs":           len(per_job_means),
            "per_job_means":    per_job_means,
            "job_stats":        job_stats,
        })

        if np.isfinite(pareto_thr):
            log.info(
                "  -> Pareto threshold: %.4g  (current: %.4g)",
                pareto_thr, metric_def["current_threshold"],
            )
        else:
            log.info("  -> No valid data -- cannot calibrate")

        for job_id, stats in job_stats.items():
            all_rows.append({"metric": metric_def["name"], "job_id": job_id, **stats})

    # -- Print results ---------------------------------------------------------
    _print_summary_table(all_results)
    _print_yaml_suggestions(all_results)

    # -- Save per-job CSV ------------------------------------------------------
    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)
        log.info("Per-job statistics saved to %s", OUTPUT_CSV)


if __name__ == "__main__":
    ids = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_JOB_IDS
    main(ids)
