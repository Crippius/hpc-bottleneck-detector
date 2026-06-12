"""
HPC Bottleneck Detector demo

Runs the full pipeline for a single HPC job:
  1. Load config from configs/demo.yaml
  2. Connect to XBAT and fetch job data
  3. Slide an analysis window over the time series
  4. Run the configured strategy
  5. Print a severity heatmap and per-window diagnosis summary

Usage:
    python examples/demo.py --job-id <JOB_ID>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpc_bottleneck_detector import AnalysisOrchestrator, BottleneckType, WindowDiagnosis

REPO_ROOT  = Path(__file__).parent.parent
CONFIG     = REPO_ROOT / "configs" / "demo.yaml"

_RESET = "\033[0m"


def _ansi_bg(rgb: tuple) -> str:
    r, g, b = (int(v * 255) for v in rgb)
    return f"\033[48;2;{r};{g};{b}m"


def print_heatmap(window_diagnoses: list[WindowDiagnosis]) -> None:
    """Terminal heatmap: rows = bottleneck types, columns = time windows."""
    excluded  = {BottleneckType.NONE, BottleneckType.UNKNOWN}
    all_types = [bt for bt in BottleneckType if bt not in excluded]
    rgb_low   = mcolors.to_rgb("green")
    rgb_high  = mcolors.to_rgb("red")

    n = len(window_diagnoses)
    grid = np.full((len(all_types), n), np.nan)

    for col, wd in enumerate(window_diagnoses):
        for d in wd.diagnoses:
            if d.bottleneck_type in excluded:
                continue
            row = all_types.index(d.bottleneck_type)
            cur = grid[row, col]
            grid[row, col] = d.severity_score if np.isnan(cur) else max(cur, d.severity_score)

    label_w = max(len(bt.value) for bt in all_types)
    CELL    = "  "

    print()
    for row_idx, bt in enumerate(all_types):
        cells = ""
        for col in range(n):
            sev = grid[row_idx, col]
            if np.isnan(sev):
                cells += " " * len(CELL)
            else:
                t   = sev
                rgb = tuple(rgb_low[i] + (rgb_high[i] - rgb_low[i]) * t for i in range(3))
                cells += _ansi_bg(rgb) + CELL + _RESET
        print(f"  {bt.value.ljust(label_w)}  {cells}")

    steps    = 20
    gradient = "".join(
        _ansi_bg(tuple(rgb_low[i] + (rgb_high[i] - rgb_low[i]) * k / (steps - 1) for i in range(3))) + " " + _RESET
        for k in range(steps)
    )
    print(f"\n  {''.ljust(label_w)}  green {gradient} red  (severity 0 -> 1)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="HPC Bottleneck Detector demo")
    parser.add_argument("--job-id", required=True, help="Job ID to analyse")
    parser.add_argument("--config", default=str(CONFIG), help="Path to config YAML (default: configs/demo.yaml)")
    args = parser.parse_args()

    print(f"Config   : {args.config}")
    print(f"Job ID   : {args.job_id}")
    print()

    orchestrator = AnalysisOrchestrator.from_config(args.config)
    print(f"Strategy : {type(orchestrator.strategy).__name__}")
    print(f"Window   : {orchestrator.window_size} intervals  (step={orchestrator.step_size})")
    print()

    results = orchestrator.run_pipeline(args.job_id)

    bottleneck_windows = [wd for wd in results if wd.has_bottlenecks()]
    print(f"\nWindows analysed  : {len(results)}")
    print(f"With bottlenecks  : {len(bottleneck_windows)}")

    if not results:
        print("No windows returned - check your config and job ID.")
        return

    print("\n--- Severity heatmap ---------------------------------------------------------------")
    print_heatmap(results)

    if bottleneck_windows:
        print("--- First flagged window ---------------------------------------------------------")
        wd = bottleneck_windows[0]
        print(f"  Window {wd.window_index}  (intervals {wd.start_interval}-{wd.end_interval})")
        for d in wd.diagnoses:
            if d.bottleneck_type is BottleneckType.NONE:
                continue
            print(f"  - {d.bottleneck_type.value:<40} sev={d.severity_score:.2f}  conf={d.confidence:.2f}")
            if d.recommendation:
                print(f"    -> {d.recommendation.strip().splitlines()[0]}")


if __name__ == "__main__":
    main()
