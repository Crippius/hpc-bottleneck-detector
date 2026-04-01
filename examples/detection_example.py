"""
Detection Algorithm Example

Demonstrates the pipeline from data acquisition to bottleneck detection:

Two toy detectors are shown:

  1. MemoryBandwidthBottleneckDetector
       Normalises the measured memory-bandwidth trace against the node's
       theoretical peak and flags intervals
       where utilisation exceeds a configurable threshold.

  2. BranchMispredictionDetector
       Flags intervals where the branch-misprediction rate exceeds a fixed
       threshold, independent of hardware context.

Usage:
    python detection_example.py
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ── path setup for development (no install required) ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpc_bottleneck_detector.data import DataManager
from hpc_bottleneck_detector.data.job_context import JobContext
from hpc_bottleneck_detector.data_sources.xbat_source import XBATDataSource
from hpc_bottleneck_detector.strategies import HeuristicStrategy
from hpc_bottleneck_detector.output.models import BottleneckType


# =============================================================================
# Shared result type
# =============================================================================

@dataclass
class DetectionResult:
    """Result produced by a single detector run."""

    detector_name: str
    job_id: str
    bottleneck_detected: bool
    flagged_intervals: List[int] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"[{self.detector_name}]  job={self.job_id}",
            f"  bottleneck detected : {self.bottleneck_detected}",
        ]
        if self.flagged_intervals:
            lines.append(
                f"  flagged intervals   : {len(self.flagged_intervals)} "
                f"(first 5: {self.flagged_intervals[:5]})"
            )
        for key, value in self.details.items():
            lines.append(f"  {key:<20}: {value}")
        return "\n".join(lines)



# =============================================================================
# Main demo
# =============================================================================

def _lerp_color(rgb_low: tuple, rgb_high: tuple, t: float) -> tuple:
    """Linearly interpolate between two RGB tuples (values 0-1)."""
    return tuple(rgb_low[i] + (rgb_high[i] - rgb_low[i]) * t for i in range(3))


def _ansi_bg(rgb: tuple) -> str:
    """Return ANSI truecolor background escape for an RGB tuple (values 0–1)."""
    r, g, b = (int(v * 255) for v in rgb)
    return f"\033[48;2;{r};{g};{b}m"


_RESET = "\033[0m"


def print_bottleneck_timeline(
    window_records: list,
    color_low: str = "green",
    color_high: str = "red",
) -> None:
    """Print a Persyst-style timeline heatmap to the terminal.

    Rows = bottleneck types, columns = time windows.
    Cell color interpolates from *color_low* (low severity) to *color_high*
    (high severity).  Windows with no detection are left blank.

    Parameters
    ----------
    window_records:
        List of ``(start_interval, end_interval, diagnoses)`` tuples.
    color_low, color_high:
        Any matplotlib-compatible color name or hex string.
    """
    from hpc_bottleneck_detector.output.models import BottleneckType

    rgb_low  = mcolors.to_rgb(color_low)
    rgb_high = mcolors.to_rgb(color_high)

    excluded = {BottleneckType.NONE, BottleneckType.UNKNOWN}
    all_types = [bt for bt in BottleneckType if bt not in excluded]

    n_windows = len(window_records)
    severity_grid = np.full((len(all_types), n_windows), np.nan)

    for col, (_, __, diagnoses) in enumerate(window_records):
        for d in diagnoses:
            if d.bottleneck_type in excluded:
                continue
            row = all_types.index(d.bottleneck_type)
            current = severity_grid[row, col]
            severity_grid[row, col] = (
                d.severity_score if np.isnan(current) else max(current, d.severity_score)
            )

    # ── layout ────────────────────────────────────────────────────────────────
    CELL    = "  "          # two chars per window column
    label_w = max(len(bt.value) for bt in all_types)
    n_cols  = severity_grid.shape[1]

    # ── rows ──────────────────────────────────────────────────────────────────
    for row_idx, bt in enumerate(all_types):
        label = bt.value.ljust(label_w)
        cells = ""
        for col in range(n_cols):
            sev = severity_grid[row_idx, col]
            if np.isnan(sev):
                cells += " " * len(CELL)
            else:
                rgb = _lerp_color(rgb_low, rgb_high, sev)
                cells += _ansi_bg(rgb) + CELL + _RESET
        print(f"  {label}  {cells}")

    # ── legend ────────────────────────────────────────────────────────────────
    print()
    steps = 20
    gradient = "".join(
        _ansi_bg(_lerp_color(rgb_low, rgb_high, i / (steps - 1))) + " " + _RESET
        for i in range(steps)
    )
    print(f"  {'':>{label_w}}  {color_low} {gradient} {color_high}  (severity 0 -> 1)")


def print_section(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> None:

    # JOB_ID = "248750"
    # ENV_FILE = ".env.example"
    # 43325 43319 43298 43290 43272 43260 43236 43195 43141 43129 43118

    
    JOB_ID = 43298   
    # JOB_ID = "43081"
    ENV_FILE = ".env"


    # ── 1. Create the data source from .env credentials ─────────────────────
    # Copy .env.example -> .env and fill in your real values, then run.
    print_section("1. Creating XBATDataSource")
    source = XBATDataSource.from_env(env_file=ENV_FILE)
    print(f"  Connected to : {source.api_base}")
    print(f"  Level        : {source.level}")

    # ── 2. Fetch a DataManager for a specific job ─────────────────────────────
    print_section(f"2. Fetching DataManager for job {JOB_ID}")
    dm: DataManager = source.fetch_job_data(JOB_ID)

    print(f"  job_id               : {dm.job_id}")
    print(f"  time-series length   : {dm.get_time_series_length()} intervals")
    print(f"  total metric rows    : {len(dm.job_data)}")
    print(f"  job_context present  : {dm.job_context is not None}")

    # ── 3. Inspect the JobContext ─────────────────────────────────────────────
    print_section("3. Inspecting JobContext (hardware metadata)")
    ctx: Optional[JobContext] = dm.job_context
    if ctx is None:
        print("  JobContext is None - hardware metadata unavailable.")
    else:
        print(f"  {ctx}")
        print()
        print(f"  Job state      : {ctx.get_metadata('jobState')}")
        print(f"  Runtime        : {ctx.get_metadata('runtime')}")
        print(f"  Variant        : {ctx.get_metadata('variantName')}")
        print(f"  Node hashes    : {ctx.get_node_hashes()}")
        print()
        print("  CPU (first node)")
        print(f"    Model        : {ctx.get_cpu_info('Model name')}")
        print(f"    Sockets      : {ctx.get_cpu_info('Socket(s)')}")
        print(f"    Cores/socket : {ctx.get_cpu_info('Core(s) per socket')}")
        print(f"    Threads/core : {ctx.get_cpu_info('Thread(s) per core')}")
        print(f"    NUMA nodes   : {ctx.get_cpu_info('NUMA node(s)')}")
        print(f"    L3 cache     : {ctx.get_cpu_info('L3 cache')}")
        print()
        print("  Memory (first node)")
        print(f"    Type         : {ctx.get_memory_info('Type')}")
        print(f"    Size / dimm  : {ctx.get_memory_info('Size')}")
        print(f"    Speed        : {ctx.get_memory_info('Speed')}")
        print()
        print("  Benchmarks (mean across nodes)")
        bmarks = [
            ("bandwidth_mem",         "Memory BW"),
            ("bandwidth_l3",          "L3 BW"),
            ("bandwidth_l2",          "L2 BW"),
            ("bandwidth_l1",          "L1 BW"),
            ("peakflops_avx512_fma",  "Peak flops (AVX-512 FMA)"),
        ]
        for key, label in bmarks:
            val = ctx.get_benchmark(key)
            if val is not None:
                print(f"    {label:<26}: {val / 1e9:>10.1f} GB/s or GFlops/s")

    # ── 4. Browse available metrics ───────────────────────────────────────────
    print_section("4. Available metrics (first 10)")
    available = dm.list_available_metrics()
    print(f"  Total: {len(available)}")
    # available = available[available["group"] == "memory"]
    for _, row in available.head(10).iterrows():
        print(f"    {row['group']:<12} {row['metric']:<30} {row['trace']}")

    # ── 5. Fetch a single metric time series ──────────────────────────────────
    print_section("5. Fetching a single metric time series")
    try:
        ts = dm.get_metric("cpu", "Branching", "branch rate")
        print(f"  cpu/Branching/branch rate  ({len(ts)} intervals)")
        print(f"    mean  : {ts.mean():.4f}")
        print(f"    std   : {ts.std():.4f}")
        print(f"    min   : {ts.min():.4f}")
        print(f"    max   : {ts.max():.4f}")
    except ValueError as exc:
        print(f"  {exc}")

 
    # ── 6. Run HeuristicStrategy (decision-tree engine) ───────────────────────
    print_section("6. HeuristicStrategy — decision-tree analysis")

    strategy_folder = (
        Path(__file__).parent.parent / "configs" / "strategies" / "persyst_strategy"
    )
    strategy = HeuristicStrategy(str(strategy_folder))

    print(f"\n  Trees loaded   : {len(strategy._strategy_trees)}")
    print(f"  Tree names     : {[t.tree_name for t in strategy._strategy_trees]}")

    bottleneck_counter: Counter = Counter()
    bottleneck_windows = 0
    window_records = []  # (start, end, diagnoses) for every window

    # Slide a 10-interval window over the full job (single pass)
    window_size = 10
    for start, end, win_dm in dm.iterate_windows(window_size=window_size, step_size=window_size):
        diagnoses = strategy.diagnose(win_dm)
        window_records.append((start, end, diagnoses))
        findings = [d for d in diagnoses if not d.is_healthy]
        if findings:
            bottleneck_windows += 1
        for d in diagnoses:
            bottleneck_counter[d.bottleneck_type.value] += 1

    total_windows = len(window_records)
    print(f"\n  Windows with bottleneck : {bottleneck_windows} / {total_windows}")
    print("  Bottleneck distribution :")
    for bt, cnt in bottleneck_counter.most_common():
        bar = "█" * cnt
        print(f"    {bt:<42} {cnt:>3}  {bar}")

    # ── Detailed view of the first flagged window ─────────────────────────────
    print()
    for start, end, diagnoses in window_records:
        findings = [d for d in diagnoses if not d.is_healthy]
        if findings:
            print(f"  First bottleneck window: intervals {start}–{end}")
            for d in findings:
                cat = d.bottleneck_type.get_macro_category().value
                print(f"    [{d.source}]")
                print(f"      bottleneck   : {d.bottleneck_type.value}  ({cat})")
                print(f"      severity     : {d.severity_score:.3f}")
                print(f"      confidence   : {d.confidence:.2f}")
                print(f"      metrics used : {', '.join(d.triggered_metrics)}")
                if d.recommendation:
                    first_line = d.recommendation.strip().splitlines()[0]
                    print(f"      rec (1st ln) : {first_line}")
            break

    # ── 7. Bottleneck timeline heatmap ────────────────────────────────────────
    print_section("7. Bottleneck timeline heatmap")
    print_bottleneck_timeline(
        window_records,
        color_low="green",
        color_high="red",
    )

    print()
    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
