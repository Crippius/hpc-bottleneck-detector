"""
AnalysisOrchestrator — usage examples

Three patterns are shown:

  1. from_config()  - load everything from orchestrator.yaml (recommended)
  2. Manual CSV     - wire up components in code, read from a local CSV file
  3. Manual XBAT    - wire up components in code, pull live data from XBAT API

Run any section individually by adjusting the __main__ block at the bottom.

Usage:
    python examples/orchestrator_example.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── path setup so the example runs without installing the package ─────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpc_bottleneck_detector import (
    AnalysisOrchestrator,
    BottleneckType,
    Diagnosis,
    WindowDiagnosis,
)
from hpc_bottleneck_detector.data_sources import CSVDataSource, XBATDataSource
from hpc_bottleneck_detector.strategies import HeuristicStrategy

# ── logging (optional, makes strategy internals visible) ─────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s — %(message)s",
)


# =============================================================================
# Example 1 – load everything from orchestrator.yaml
# =============================================================================

def example_from_config(job_id: str = "249755") -> list[WindowDiagnosis]:
    """
    Simplest usage: point at the config file and call run_pipeline().

    All settings (data source, strategy, window size, output format) come
    from ``configs/orchestrator.yaml``.  Edit that file to switch between
    CSV / XBAT data sources or heuristic / ML strategies without touching
    code.
    """
    print("\n" + "=" * 60)
    print("  Example 1 — from_config()")
    print("=" * 60)

    # Resolve config path relative to the repo root
    config_path = Path(__file__).parent.parent / "configs" / "orchestrator.yaml"

    orchestrator = AnalysisOrchestrator.from_config(str(config_path))

    print(f"  window_size : {orchestrator.window_size}")
    print(f"  step_size   : {orchestrator.step_size}")
    print(f"  data_source : {type(orchestrator.data_source).__name__}")
    print(f"  strategy    : {type(orchestrator.strategy).__name__}")
    print()

    results = orchestrator.run_pipeline(job_id)
    return results


# =============================================================================
# Example 2 – manual construction with a local CSV file
# =============================================================================

def example_csv(
    csv_path: str = "../data/234650_all_job.csv",
    job_id: str = "234650",
) -> list[WindowDiagnosis]:
    """
    Build the orchestrator programmatically using a CSV data source.

    Useful for offline analysis of archived job data exported from XBAT.
    """
    print("\n" + "=" * 60)
    print("  Example 2 — manual CSV setup")
    print("=" * 60)

    orchestrator = AnalysisOrchestrator(
        data_source=CSVDataSource(csv_path),
        strategy=HeuristicStrategy(
            strategy_folder="configs/strategies/persyst_strategy"
        ),
        window_size=10,
        step_size=10,   # tumbling windows (non-overlapping)
        output_cfg={
            "format": "print",
            "show_healthy_windows": True,
            "min_severity": 0.0,
            "min_confidence": 0.0,
        },
    )

    results = orchestrator.run_pipeline(job_id)
    return results


# =============================================================================
# Example 3 – manual construction against the XBAT REST API
# =============================================================================

def example_xbat(job_id: str = "249755") -> list[WindowDiagnosis]:
    """
    Build the orchestrator programmatically using the XBAT data source.

    Credentials are loaded from a ``.env`` file via
    :meth:`~hpc_bottleneck_detector.data_sources.XBATDataSource.from_env`.
    Copy ``.env.example`` → ``.env`` and fill in your real values before
    running this example.
    """
    print("\n" + "=" * 60)
    print("  Example 3 — manual XBAT setup")
    print("=" * 60)

    orchestrator = AnalysisOrchestrator(
        data_source=XBATDataSource.from_env(
            group="",   # empty → all groups, job-level aggregation
            metric="",
            level="job",
        ),
        strategy=HeuristicStrategy(
            strategy_folder="configs/strategies/persyst_strategy"
        ),
        window_size=10,
        step_size=5,    # sliding windows (50 % overlap)
        output_cfg={
            "format": "json",
            "show_healthy_windows": False,  # only show windows with bottlenecks
            "min_severity": 0.3,
            "min_confidence": 0.5,
            # "save_path": "results/diagnoses.json",  # uncomment to persist
        },
    )

    results = orchestrator.run_pipeline(job_id)
    return results


# =============================================================================
# Inspecting results programmatically
# =============================================================================

def inspect_results(results: list[WindowDiagnosis]) -> None:
    """
    Show how to iterate over WindowDiagnosis objects in your own code.
    """
    print("\n" + "─" * 60)
    print("  Programmatic inspection of results")
    print("─" * 60)

    bottleneck_windows = [wd for wd in results if wd.has_bottlenecks()]
    healthy_windows    = [wd for wd in results if not wd.has_bottlenecks()]

    print(f"  Total windows    : {len(results)}")
    print(f"  With bottleneck  : {len(bottleneck_windows)}")
    print(f"  Healthy          : {len(healthy_windows)}")

    for wd in bottleneck_windows:
        print(
            f"\n  Window {wd.window_index} "
            f"(intervals {wd.start_interval}–{wd.end_interval}) "
            f"worst_severity={wd.worst_severity():.2f}"
        )
        for diag in wd.diagnoses:
            if diag.bottleneck_type is not BottleneckType.NONE:
                print(
                    f"    • {diag.bottleneck_type.value:<35} "
                    f"sev={diag.severity_score:.2f}  "
                    f"conf={diag.confidence:.2f}  "
                    f"source={diag.source}"
                )
                cat = diag.bottleneck_type.get_macro_category().value
                print(f"      macro-category: {cat}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # ── Example 2 works offline (uses the bundled CSV) ────────────────────
    results = example_csv()
    inspect_results(results)

    # ── Uncomment to try the config-file approach ─────────────────────────
    results = example_from_config(job_id="249755")
    inspect_results(results)

    # ── Uncomment to try the live XBAT API (requires network) ────────────
    results = example_xbat(job_id="249755")
    inspect_results(results)
