"""
ML Inference Example

Demonstrates how to use a trained DefaultBackend to detect bottlenecks in
a new HPC job.  Two patterns are shown:

  1. Direct API  — load the backend, wrap it in SupervisedMLStrategy, and
                   call diagnose() window-by-window.
  2. Orchestrator — use AnalysisOrchestrator with a YAML config that sets
                    ``strategy.type: supervised_ml``.

Prerequisites
-------------
- A trained model must exist.  Run ``examples/ml_training_example.py`` first.
- ``tsfresh`` and ``scikit-learn`` must be installed.
- A ``.env`` file with XBAT credentials (copy ``.env.example`` → ``.env``).

Usage:
    python examples/ml_inference_example.py [--job-id JOB_ID]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpc_bottleneck_detector import AnalysisOrchestrator, BottleneckType, WindowDiagnosis
from hpc_bottleneck_detector.data_sources.xbat_source import XBATDataSource
from hpc_bottleneck_detector.ml.backends.default_backend import DefaultBackend
from hpc_bottleneck_detector.strategies.supervised_ml import SupervisedMLStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

REPO_ROOT  = Path(__file__).parent.parent
MODEL_PATH = REPO_ROOT / "models" / "default.pkl"

WINDOW_SIZE = 10
STEP_SIZE   = 10
THRESHOLD   = 0.3   # probability → Diagnosis


# =============================================================================
# Terminal helpers
# =============================================================================

def _lerp_color(rgb_low: tuple, rgb_high: tuple, t: float) -> tuple:
    return tuple(rgb_low[i] + (rgb_high[i] - rgb_low[i]) * t for i in range(3))


def _ansi_bg(rgb: tuple) -> str:
    r, g, b = (int(v * 255) for v in rgb)
    return f"\033[48;2;{r};{g};{b}m"


_RESET = "\033[0m"


def print_section(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_bottleneck_timeline(
    window_records: list,
    color_low: str = "green",
    color_high: str = "red",
) -> None:
    from hpc_bottleneck_detector.output.models import BottleneckType

    rgb_low  = mcolors.to_rgb(color_low)
    rgb_high = mcolors.to_rgb(color_high)

    excluded = {BottleneckType.NONE, BottleneckType.UNKNOWN}
    all_types = [bt for bt in BottleneckType if bt not in excluded]

    n_windows = len(window_records)
    severity_grid = np.full((len(all_types), n_windows), np.nan)

    for col, (*_, diagnoses) in enumerate(window_records):
        for d in diagnoses:
            if d.bottleneck_type in excluded:
                continue
            row = all_types.index(d.bottleneck_type)
            current = severity_grid[row, col]
            severity_grid[row, col] = (
                d.severity_score if np.isnan(current) else max(current, d.severity_score)
            )

    CELL    = "  "
    label_w = max(len(bt.value) for bt in all_types)
    n_cols  = severity_grid.shape[1]

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

    print()
    steps = 20
    gradient = "".join(
        _ansi_bg(_lerp_color(rgb_low, rgb_high, i / (steps - 1))) + " " + _RESET
        for i in range(steps)
    )
    print(f"  {'':>{label_w}}  {color_low} {gradient} {color_high}  (severity 0 -> 1)")


# =============================================================================
# Shared printer
# =============================================================================

def _print_results(results: list[WindowDiagnosis]) -> None:
    excluded = {BottleneckType.NONE, BottleneckType.UNKNOWN}
    bottleneck_windows = [wd for wd in results if wd.has_bottlenecks()]

    print(f"\n  Total windows      : {len(results)}")
    print(f"  With bottleneck(s) : {len(bottleneck_windows)}")

    for wd in bottleneck_windows:
        print(
            f"\n  Window {wd.window_index}  "
            f"(intervals {wd.start_interval}–{wd.end_interval})  "
            f"worst_severity={wd.worst_severity():.2f}"
        )
        for d in wd.diagnoses:
            if d.bottleneck_type in excluded:
                continue
            print(
                f"    • {d.bottleneck_type.value:<42}"
                f"  prob={d.severity_score:.2f}"
                f"  conf={d.confidence:.2f}"
                f"  source={d.source}"
            )


# =============================================================================
# Example 1 – direct API
# =============================================================================

def example_direct(job_id: int, env_file: str = ".env") -> list[WindowDiagnosis]:
    """
    Load a trained backend, run SupervisedMLStrategy on every window, and
    return the WindowDiagnosis list.

    Steps:
      1. Load the saved DefaultBackend from disk.
      2. Wrap it in SupervisedMLStrategy.
      3. Fetch job data from XBAT.
      4. Slide windows, call diagnose() for each, collect WindowDiagnosis.
    """
    print("\n" + "=" * 60)
    print("  Example 1 — direct API")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run examples/ml_training_example.py first."
        )

    # ── 1. Load backend ───────────────────────────────────────────────────────
    backend = DefaultBackend.load(str(MODEL_PATH))
    print(f"[INFO] Loaded backend — {len(backend._models)} trained classifiers:")
    for bt_name in backend._models:
        n_feat = len(backend._feature_cols[bt_name])
        print(f"  {bt_name:<42}  {n_feat} features")

    # ── 2. Wrap in strategy ───────────────────────────────────────────────────
    strategy = SupervisedMLStrategy(
        backend=backend,
        significance_threshold=THRESHOLD,
    )

    # ── 3. Fetch job data ─────────────────────────────────────────────────────
    print(f"\n[INFO] Fetching job {job_id} from XBAT…")
    source = XBATDataSource.from_env(env_file=env_file)
    dm = source.fetch_job_data(job_id)
    print(f"  intervals : {dm.get_time_series_length()}")
    print(f"  metrics   : {len(dm.job_data)}")

    # ── 4. Slide windows, diagnose ────────────────────────────────────────────
    print(f"\n[INFO] Running inference (window={WINDOW_SIZE}, step={STEP_SIZE})…")
    from hpc_bottleneck_detector.output.models import WindowDiagnosis as WD

    results: list[WindowDiagnosis] = []
    for win_idx, (start, end, win_dm) in enumerate(
        dm.iterate_windows(WINDOW_SIZE, STEP_SIZE)
    ):
        diagnoses = strategy.diagnose(win_dm)
        results.append(WD(
            window_index=win_idx,
            start_interval=start,
            end_interval=end,
            diagnoses=diagnoses,
        ))

    _print_results(results)

    window_records = [(wd.start_interval, wd.end_interval, wd.diagnoses) for wd in results]
    print_section("Bottleneck timeline heatmap")
    print_bottleneck_timeline(window_records, color_low="green", color_high="red")

    print()
    print("=" * 70)
    print("  Done.")
    print("=" * 70)

    return results


# =============================================================================
# Example 2 – via AnalysisOrchestrator
# =============================================================================

def example_via_orchestrator(job_id: int, env_file: str = ".env") -> list[WindowDiagnosis]:
    """
    Use AnalysisOrchestrator with a supervised_ml strategy block.

    The strategy is built from the ``strategy`` config dict exactly as if
    the YAML had::

        strategy:
          type: supervised_ml
          backend: tsfresh_sklearn
          model_path: models/default.pkl
          significance_threshold: 0.3
    """
    print("\n" + "=" * 60)
    print("  Example 2 — via AnalysisOrchestrator")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run examples/ml_training_example.py first."
        )

    orchestrator = AnalysisOrchestrator(
        data_source=XBATDataSource.from_env(env_file=env_file),
        strategy=AnalysisOrchestrator._build_strategy({
            "type": "supervised_ml",
            "backend": "tsfresh_sklearn",
            "model_path": str(MODEL_PATH),
            "significance_threshold": THRESHOLD,
        }),
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        output_cfg={
            "format": "print",
            "show_healthy_windows": False,
            "min_severity": THRESHOLD,
        },
    )

    print(f"[INFO] Strategy : {type(orchestrator.strategy).__name__}")
    print(f"[INFO] Running pipeline for job {job_id}…")
    results = orchestrator.run_pipeline(str(job_id))
    _print_results(results)

    window_records = [(wd.start_interval, wd.end_interval, wd.diagnoses) for wd in results]
    print_section("Bottleneck timeline heatmap")
    print_bottleneck_timeline(window_records, color_low="green", color_high="red")

    print()
    print("=" * 70)
    print("  Done.")
    print("=" * 70)

    return results


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML inference example")
    parser.add_argument("--job-id", type=int, default=45719, help="Job ID to run inference on")
    parser.add_argument("--env-file", default=".env", help="Path to .env credentials file (default: .env)")
    _args = parser.parse_args()

    # Example 1 — direct API (most transparent, best for debugging)
    results = example_direct(_args.job_id, env_file=_args.env_file)

    # Example 2 — orchestrator (matches production usage)
    # results = example_via_orchestrator(_args.job_id, env_file=_args.env_file)
