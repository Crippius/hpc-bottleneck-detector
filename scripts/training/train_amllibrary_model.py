"""
Training Script - AMLLibrary Backend

Trains an :class:`AMLLibraryBackend` on labelled CSVs produced by label_jobs.py.
aMLLibrary runs an automated model selection campaign per BottleneckType,
comparing LRRidge, RandomForest and XGBoost with HoldOut validation.

Usage
-----
    conda run -n thesisEnv python scripts/training/train_amllibrary_model.py
    conda run -n thesisEnv python scripts/training/train_amllibrary_model.py \\
        --data-dir data/labelled_data/ \\
        --window-size 10 \\
        --step-size 10 \\
        --output models/amllibrary.pkl

Output
------
- A .pkl file loadable with AMLLibraryBackend.load(path).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hpc_bottleneck_detector.ml.backends.amllibrary_backend import AMLLibraryBackend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an AMLLibraryBackend on labelled HPC job CSVs."
    )
    parser.add_argument("--data-dir", default="data/labelled_data/",
        help="Directory containing labelled CSV files.")
    parser.add_argument("--window-size", type=int, default=10,
        help="Number of intervals per analysis window.")
    parser.add_argument("--step-size", type=int, default=10,
        help="Interval advance between successive windows.")
    parser.add_argument("--severity-threshold", type=float, default=0.0,
        help="Severity > this value → positive label.")
    parser.add_argument("-o", "--output", default="models/amllibrary.pkl",
        help="Output path for the saved backend (.pkl).")
    return parser.parse_args()


def _collect_csv_paths(data_dir: str) -> list[str]:
    paths = sorted(Path(data_dir).rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Run scripts/training/label_jobs.py first."
        )
    return [str(p) for p in paths]


def main() -> None:
    args = _parse_args()

    logger.info("=" * 60)
    logger.info("Training AMLLibraryBackend")
    logger.info("  data_dir      : %s", args.data_dir)
    logger.info("  window_size   : %d", args.window_size)
    logger.info("  step_size     : %d", args.step_size)
    logger.info("  sev_threshold : %.3f", args.severity_threshold)
    logger.info("  output        : %s", args.output)
    logger.info("=" * 60)

    csv_paths = _collect_csv_paths(args.data_dir)
    logger.info("Found %d labelled CSV(s).", len(csv_paths))

    backend = AMLLibraryBackend()
    backend.train(
        labelled_csv_paths=csv_paths,
        window_size=args.window_size,
        step_size=args.step_size,
        severity_threshold=args.severity_threshold,
    )

    backend.save(args.output)
    logger.info("Done. Model saved to: %s", args.output)


if __name__ == "__main__":
    main()
