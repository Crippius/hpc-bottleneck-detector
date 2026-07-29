"""
HPC Bottleneck Detector CLI

    bottleneck-detect --job-id <JOB_ID> [--config PATH] [--format json|print|csv]
                       [--output PATH] [--quiet] [-v]

Must be run with the repository root as the current working directory (or
with a --config whose data_source/strategy/hardware paths are absolute),
since XBATDataSource.from_env(), HeuristicStrategy and HardwareProfileLoader
all resolve their configured paths (env_file, token_file, strategy_folder,
profiles_dir) relative to the process's current working directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from .orchestrator import AnalysisOrchestrator
from .output.formatter import format_results
from .output.models import BottleneckType, WindowDiagnosis

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "xbat_cli.yaml"

_RESET = "\033[0m"
_RGB_LOW = (0.0, 0.5, 0.0)   # green
_RGB_HIGH = (1.0, 0.0, 0.0)  # red


def _ansi_bg(rgb: tuple) -> str:
    r, g, b = (int(v * 255) for v in rgb)
    return f"\033[48;2;{r};{g};{b}m"


def print_heatmap(window_diagnoses: list[WindowDiagnosis]) -> None:
    """Terminal heatmap: rows = bottleneck types, columns = time windows."""
    excluded = {BottleneckType.NONE, BottleneckType.UNKNOWN}
    all_types = [bt for bt in BottleneckType if bt not in excluded]

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
    CELL = "  "

    print()
    for row_idx, bt in enumerate(all_types):
        cells = ""
        for col in range(n):
            sev = grid[row_idx, col]
            if np.isnan(sev):
                cells += " " * len(CELL)
            else:
                t = sev
                rgb = tuple(_RGB_LOW[i] + (_RGB_HIGH[i] - _RGB_LOW[i]) * t for i in range(3))
                cells += _ansi_bg(rgb) + CELL + _RESET
        print(f"  {bt.value.ljust(label_w)}  {cells}")

    steps = 20
    gradient = "".join(
        _ansi_bg(tuple(_RGB_LOW[i] + (_RGB_HIGH[i] - _RGB_LOW[i]) * k / (steps - 1) for i in range(3))) + " " + _RESET
        for k in range(steps)
    )
    print(f"\n  {''.ljust(label_w)}  green {gradient} red  (severity 0 -> 1)\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bottleneck-detect",
        description="Run the HPC bottleneck detection pipeline for a single job.",
    )
    p.add_argument("--job-id", required=True, help="Job ID to analyse")
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to config YAML (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--format",
        choices=["json", "print", "csv"],
        default=None,
        help="Override the config's output.format",
    )
    p.add_argument(
        "--strategy",
        choices=["heuristic", "supervised_ml"],
        default=None,
        help="Override the config's strategy.type",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Override the config's strategy.model_path (implies --strategy supervised_ml unless given)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Write results to this path (overrides output.save_path)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Do not also print results to stdout when writing to --output",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable INFO-level logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )

    strategy_overrides: dict = {}
    if args.model_path:
        strategy_overrides["type"] = "supervised_ml"
        strategy_overrides["model_path"] = args.model_path
    if args.strategy:
        strategy_overrides["type"] = args.strategy

    try:
        orchestrator = AnalysisOrchestrator.from_config(args.config, strategy_overrides or None)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # CLI default format is "json" (machine/shell-friendly), overriding the
    # library's own from_config default of "print" -- config-file values
    # still win unless explicitly overridden on the command line.
    orchestrator.output_cfg.setdefault("format", "json")
    if args.format:
        orchestrator.output_cfg["format"] = args.format
    if args.output:
        orchestrator.output_cfg["save_path"] = args.output

    try:
        results = orchestrator.run_pipeline(args.job_id)
    except (ValueError, IOError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("warning: no windows returned - check config and job ID.", file=sys.stderr)

    # run_pipeline() already wrote save_path (if set) and already printed to
    # stdout itself when format == "print". For "json"/"csv" it does neither
    # unless save_path is set, so print the rendering here explicitly.
    fmt = orchestrator.output_cfg.get("format", "json")
    if fmt == "print":
        print_heatmap(results)
    elif not args.quiet:
        print(format_results(results, fmt=fmt, save_path=None))

    return 0


if __name__ == "__main__":
    sys.exit(main())
