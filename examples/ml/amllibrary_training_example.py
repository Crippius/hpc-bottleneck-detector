"""
AMLLibrary Training Example

Demonstrates how to train the AMLLibraryBackend on labelled CSVs produced by
``label_job()``.  For each BottleneckType, aMLLibrary runs a regression campaign
that compares LRRidge, RandomForest, and XGBoost with HoldOut validation and
returns the best model as an MTSRegressor.

The key difference from the DefaultBackend (tsfresh + fixed RandomForest):
  - Model selection is automated — aMLLibrary picks the best technique per type.
  - Feature extraction uses WindowFeatureExtraction (built-in, no tsfresh needed).
  - The trained MTSRegressor re-applies windowing internally at inference time.

Prerequisites
-------------
- At least one labelled CSV in ``data/labelled_data/``.
  Run ``examples/labeling_example.py`` first if the folder is empty.
- thesisEnv conda environment (xgboost, mlxtend, future must be installed).

Usage:
    conda run -n thesisEnv python examples/ml/amllibrary_training_example.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from hpc_bottleneck_detector.ml.backends.amllibrary_backend import AMLLibraryBackend
from hpc_bottleneck_detector.utils.labeling import BOTTLENECK_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

REPO_ROOT    = Path(__file__).parents[2]
DATA_DIR     = REPO_ROOT / "data" / "labelled_data"
MODEL_OUTPUT = REPO_ROOT / "models" / "amllibrary.pkl"

WINDOW_SIZE        = 10
STEP_SIZE          = 10
SEVERITY_THRESHOLD = 0.0


# =============================================================================
# Helpers
# =============================================================================

def _find_labelled_csvs(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No labelled CSVs found in '{data_dir}'.\n"
            "Run examples/labeling_example.py first to generate training data."
        )
    return paths


def _print_label_summary(csv_paths: list[Path]) -> None:
    print("\n[INFO] Label summary across all CSVs:")
    print(f"  {'Type':<42}  {'total':>7}  {'positive':>8}  {'unknown':>7}  {'pos%':>6}")
    print("  " + "-" * 75)

    combined = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

    for bt in BOTTLENECK_COLUMNS:
        col = bt.value
        if col not in combined.columns:
            continue
        total = len(combined)
        n_nan = combined[col].isna().sum()
        n_pos = (combined[col] > SEVERITY_THRESHOLD).sum()
        pct   = 100.0 * n_pos / (total - n_nan) if (total - n_nan) > 0 else 0.0
        print(f"  {col:<42}  {total:>7}  {n_pos:>8}  {n_nan:>7}  {pct:>5.1f}%")


# =============================================================================
# Training
# =============================================================================

def train(csv_paths: list[Path]) -> AMLLibraryBackend:
    """
    Train one MTSRegressor per BottleneckType via aMLLibrary and return the backend.

    For each bottleneck type the pipeline is:
      DataLoading (sorts + drops time) → TemporalWindowing (groups by job id)
      → WindowFeatureExtraction (mean, std, slope, …)
      → HoldOut model selection (LRRidge vs RandomForest vs XGBoost)
      → MTSRegressor (applies windowing internally at inference)
    """
    print("\n" + "=" * 60)
    print("  Training AMLLibraryBackend")
    print("=" * 60)
    print(f"  window_size : {WINDOW_SIZE}")
    print(f"  step_size   : {STEP_SIZE}")
    print(f"  threshold   : {SEVERITY_THRESHOLD}")
    print(f"  techniques  : LRRidge, RandomForest, XGBoost")
    print()

    backend = AMLLibraryBackend()
    backend.train(
        labelled_csv_paths=[str(p) for p in csv_paths],
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        severity_threshold=SEVERITY_THRESHOLD,
    )

    print(f"\n[INFO] Trained regressors: {list(backend._regressors.keys())}")
    return backend


# =============================================================================
# Save / reload round-trip
# =============================================================================

def save_and_reload(backend: AMLLibraryBackend, output: Path) -> AMLLibraryBackend:
    print("\n" + "=" * 60)
    print("  Save → reload round-trip")
    print("=" * 60)

    backend.save(str(output))
    print(f"[INFO] Saved to: {output}")

    reloaded = AMLLibraryBackend.load(str(output))
    print(f"[INFO] Reloaded — {len(reloaded._regressors)} regressors restored.")

    assert set(backend._regressors) == set(reloaded._regressors), \
        "Regressor key mismatch after reload!"
    print("[INFO] Round-trip check passed.")

    return reloaded


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    csv_paths = _find_labelled_csvs(DATA_DIR)
    print(f"[INFO] Found {len(csv_paths)} labelled CSV(s):")
    for p in csv_paths:
        print(f"  {p}")

    _print_label_summary(csv_paths)

    backend = train(csv_paths)
    save_and_reload(backend, MODEL_OUTPUT)

    print("\n" + "=" * 60)
    print(f"  Done.  Model saved to: {MODEL_OUTPUT}")
    print("=" * 60)