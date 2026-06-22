"""
Standalone hyperparameter tuning script.

Runs DefaultTrainer.tune() on the full training pool and writes a JSON report
with default params, best params, and calibrated thresholds for the chosen
classifier.

Usage
-----
    python scripts/training/tune_hyperparams.py --classifier rf
    python scripts/training/tune_hyperparams.py --classifier xgboost --n-iter 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import yaml
from sklearn.exceptions import UndefinedMetricWarning

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hpc_bottleneck_detector.ml.backends.default_trainer import DefaultTrainer
from hpc_bottleneck_detector.ml.feature_extraction import (
    extract_features_for_app,
    find_labelled_csvs,
)

from hpc_bottleneck_detector.ml.backends.config import build_classifier as _build_classifier

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
PARAM_GRIDS_PATH = REPO_ROOT / "configs" / "param_grids.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "labelled_data" / "training_set"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter tuning for RF / XGBoost")
    p.add_argument("--classifier", choices=["rf", "xgboost"], required=True)
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-size", type=int, default=12)
    p.add_argument("--step-size", type=int, default=12)
    p.add_argument("--severity-threshold", type=float, default=0.2)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.output_json is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_json = REPO_ROOT / "results" / "tuning" / ts / "tune_results.json"

    with open(PARAM_GRIDS_PATH) as f:
        param_grids: dict = yaml.safe_load(f)

    # --- Feature extraction ---
    csv_paths = find_labelled_csvs(args.data_dir)
    logger.info("Found %d labelled CSVs in %s", len(csv_paths), args.data_dir)

    app_features = []
    for i, csv_path in enumerate(csv_paths, 1):
        logger.info("[%d/%d] Extracting features: %s", i, len(csv_paths), csv_path.name)
        app_features.append(
            extract_features_for_app(
                csv_path,
                args.window_size,
                args.step_size,
                args.severity_threshold,
            )
        )

    # --- Tuning ---
    clf = _build_classifier(args.classifier)
    clf_class = type(clf).__name__
    grid_keys = list(param_grids.get(clf_class, {}).keys())

    default_params = {k: clf.get_params()[k] for k in grid_keys}

    logger.info(
        "Running tune() for %s  (n_iter=%d, n_splits=%d, seed=%d)",
        clf_class, args.n_iter, args.n_splits, args.seed,
    )
    tuned_clf, thresholds = DefaultTrainer.tune(
        app_features,
        clf,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    best_params = {k: tuned_clf.get_params()[k] for k in grid_keys}

    # --- Report ---
    col_w = max(len(k) for k in grid_keys) if grid_keys else 20
    print(f"\n{'='*60}")
    print(f"  {clf_class}  —  Tuning results")
    print(f"{'='*60}")
    print(f"  {'Parameter':<{col_w}}  {'Default':>12}  {'Best':>12}")
    print(f"  {'-'*col_w}  {'-'*12}  {'-'*12}")
    for k in grid_keys:
        print(f"  {k:<{col_w}}  {str(default_params[k]):>12}  {str(best_params[k]):>12}")
    print()
    print("  Calibrated thresholds:")
    for bt, thr in sorted(thresholds.items()):
        print(f"    {bt:<35}  {thr:.4f}")
    print()

    # --- Save JSON ---
    result = {
        "classifier": args.classifier,
        "default_params": default_params,
        "best_params": best_params,
        "best_thresholds": thresholds,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved to %s", args.output_json)


if __name__ == "__main__":
    main()
