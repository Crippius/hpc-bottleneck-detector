"""
ML Training Example

Demonstrates how to train the DefaultBackend on labelled CSVs produced by
``label_job()`` and save the resulting model to disk.

Two patterns are shown:

  1. Direct API  — build and train the backend in code, inspect per-type
                   label statistics before fitting.
  2. Script      — equivalent one-liner using ``scripts/train_ml_model.py``.

Prerequisites
-------------
- At least one labelled CSV must exist under ``data/labelled_data/``.
  Run ``examples/labeling_example.py`` first if the folder is empty.
- ``tsfresh`` and ``scikit-learn`` must be installed:
      pip install tsfresh scikit-learn joblib

Usage:
    python examples/ml_training_example.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import (
    DefaultBackend,
    _LABEL_COLS,
    _NON_METRIC_COLS,
)
from hpc_bottleneck_detector.utils.labeling import BOTTLENECK_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

REPO_ROOT    = Path(__file__).parent.parent
DATA_DIR     = REPO_ROOT / "data" / "labelled_data"
MODEL_OUTPUT = REPO_ROOT / "models" / "default.pkl"

WINDOW_SIZE        = 10
STEP_SIZE          = 10
SEVERITY_THRESHOLD = 0.0   # severity > 0 → positive label


# =============================================================================
# Helpers
# =============================================================================

def _find_labelled_csvs(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No labelled CSVs found in '{data_dir}'.\n"
            "Run examples/labeling_example.py first to generate training data."
        )
    return paths


def _print_label_summary(csv_paths: list[Path]) -> None:
    """Print per-type positive / unknown counts across all labelled CSVs."""
    print("\n[INFO] Label summary across all CSVs:")
    print(f"  {'Type':<42}  {'total':>7}  {'positive':>8}  {'unknown':>7}  {'pos%':>6}")
    print("  " + "-" * 75)

    combined = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

    for bt in BOTTLENECK_COLUMNS:
        col = bt.value
        if col not in combined.columns:
            continue
        total    = len(combined)
        n_nan    = combined[col].isna().sum()
        n_pos    = (combined[col] > SEVERITY_THRESHOLD).sum()
        pct      = 100.0 * n_pos / (total - n_nan) if (total - n_nan) > 0 else 0.0
        print(
            f"  {col:<42}  {total:>7}  {n_pos:>8}  {n_nan:>7}  {pct:>5.1f}%"
        )


# =============================================================================
# Example 1 – direct API
# =============================================================================

def example_train_direct(csv_paths: list[Path]) -> DefaultBackend:
    """
    Train a DefaultBackend in code and return it.

    The backend:
      1. Slides windows of ``WINDOW_SIZE`` intervals (step ``STEP_SIZE``) over
         every job in the labelled CSVs.
      2. Extracts tsfresh features (BASIC_FC_PARAMETERS by default) per window.
      3. Selects the statistically significant features per BottleneckType.
      4. Fits one RandomForestClassifier per BottleneckType.
    """
    print("\n" + "=" * 60)
    print("  Example 1 — train via direct API")
    print("=" * 60)

    backend = DefaultBackend()  # default: RandomForest
    # backend = DefaultBackend(classifier=GradientBoostingClassifier(n_estimators=100))
    
    backend.train(
        labelled_csv_paths=[str(p) for p in csv_paths],
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        severity_threshold=SEVERITY_THRESHOLD,
    )

    print(f"\n[INFO] Trained classifiers: {list(backend._models.keys())}")
    for bt_name, clf in backend._models.items():
        n_features = len(backend._feature_cols[bt_name])
        print(f"  {bt_name:<42}  {n_features:>4} features selected")

    return backend


# =============================================================================
# Save and reload
# =============================================================================

def example_save_and_reload(backend: DefaultBackend, output: Path) -> DefaultBackend:
    """
    Save the trained backend to disk and reload it to verify the round-trip.
    """
    print("\n" + "=" * 60)
    print("  Save → reload round-trip")
    print("=" * 60)

    backend.save(str(output))
    print(f"[INFO] Saved to: {output}")

    reloaded = DefaultBackend.load(str(output))
    print(f"[INFO] Reloaded — {len(reloaded._models)} classifiers restored.")

    # Sanity check: feature column lists must match.
    for bt_name in backend._models:
        orig_cols    = backend._feature_cols.get(bt_name, [])
        reloaded_cols = reloaded._feature_cols.get(bt_name, [])
        assert orig_cols == reloaded_cols, f"Feature column mismatch for {bt_name}!"
    print("[INFO] Feature column round-trip check passed.")

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

    backend = example_train_direct(csv_paths)
    example_save_and_reload(backend, MODEL_OUTPUT)

    print("\n" + "=" * 60)
    print(f"  Done.  Model saved to: {MODEL_OUTPUT}")
    print("=" * 60)
    print()
    print("  Next step → run ml_inference_example.py to use the model.")
