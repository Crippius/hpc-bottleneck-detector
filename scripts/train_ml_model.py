"""
Training Script — Supervised ML Model

Trains a :class:`DefaultBackend` on labelled CSVs produced by
``label_job()`` and saves the result to disk.

Usage
-----
    python scripts/train_ml_model.py [options]

    # Use all defaults from configs/ml_training.yaml:
    python scripts/train_ml_model.py

    # Override specific options:
    python scripts/train_ml_model.py \\
        --data-dir data/labelled_data/ \\
        --window-size 10 \\
        --step-size 10 \\
        --output models/tsfresh_sklearn.pkl

Output
------
- A ``.pkl`` file loadable with ``DefaultBackend.load(path)``.
- A per-type classification report printed to stdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Allow running from repo root without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import (
    DefaultBackend,
    _build_window_dataframe,
    _window_labels,
    _LABEL_COLS,
    _NON_METRIC_COLS,
)
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "ml_training.yaml"


def _load_config(config_path: Path) -> dict:
    if config_path.exists():
        with config_path.open() as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _parse_args(config: dict) -> argparse.Namespace:
    training_cfg = config.get("training", {})
    backend_cfg = config.get("backend", {})
    output_cfg = config.get("output", {})

    parser = argparse.ArgumentParser(
        description="Train a DefaultBackend on labelled HPC job CSVs."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to ml_training.yaml (default: configs/ml_training.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        default=training_cfg.get("data_dir", "data/labelled_data/"),
        help="Directory containing labelled CSV files.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(training_cfg.get("window_size", 10)),
        help="Number of intervals per analysis window.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=int(training_cfg.get("step_size", 10)),
        help="Interval advance between successive windows.",
    )
    parser.add_argument(
        "--severity-threshold",
        type=float,
        default=float(training_cfg.get("severity_threshold", 0.0)),
        help="Severity > this value → positive label (default: 0.0).",
    )
    parser.add_argument(
        "--output",
        default=output_cfg.get("model_path", "models/tsfresh_sklearn.pkl"),
        help="Output path for the saved backend (.pkl).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=float(training_cfg.get("test_size", 0.2)),
        help="Fraction of windows held out for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip the held-out evaluation and train on all data.",
    )
    return parser.parse_args()


def _collect_csv_paths(data_dir: str) -> list[str]:
    paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Run label_job() or labeling_example.py to generate labelled data."
        )
    return [str(p) for p in paths]


def _build_windows_from_csvs(
    csv_paths: list[str],
    window_size: int,
    step_size: int,
    severity_threshold: float,
    fc_params: EfficientFCParameters,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """
    Load all CSVs, build tsfresh long-format DataFrame + label Series per type.

    Returns:
        X_full:  tsfresh feature matrix (rows = windows, indexed by window id).
        y_all:   ``{bt_name: pd.Series(label, index=window_id)}``
    """
    all_fragments: list[pd.DataFrame] = []
    all_window_ids: list[str] = []
    raw_labels: dict[str, list] = {col: [] for col in _LABEL_COLS}

    for csv_path in csv_paths:
        logger.info("  Loading %s", csv_path)
        df = pd.read_csv(csv_path)
        metric_cols = [c for c in df.columns if c not in _NON_METRIC_COLS]

        for job_id, job_df in df.groupby("id"):
            job_df = job_df.sort_values("time").reset_index(drop=True)

            long_df, window_ids = _build_window_dataframe(
                job_df, metric_cols, str(job_id), window_size, step_size
            )
            labels = _window_labels(
                job_df, window_size, step_size, severity_threshold
            )

            all_fragments.append(long_df)
            all_window_ids.extend(window_ids)
            for col in _LABEL_COLS:
                raw_labels[col].extend(labels[col])

    logger.info("Extracting tsfresh features for %d windows…", len(all_window_ids))
    tsfresh_df = pd.concat(all_fragments, ignore_index=True)
    X_full = extract_features(
        tsfresh_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        impute_function=impute,
        disable_progressbar=False,
    )
    X_full = X_full.reindex(all_window_ids)

    y_all = {
        col: pd.Series(raw_labels[col], index=all_window_ids, dtype=float)
        for col in _LABEL_COLS
    }
    return X_full, y_all


def main() -> None:
    # ── Config + args ─────────────────────────────────────────────────────────
    config = _load_config(DEFAULT_CONFIG)
    args = _parse_args(config)

    logger.info("=" * 60)
    logger.info("Training DefaultBackend")
    logger.info("  data_dir      : %s", args.data_dir)
    logger.info("  window_size   : %d", args.window_size)
    logger.info("  step_size     : %d", args.step_size)
    logger.info("  sev_threshold : %.3f", args.severity_threshold)
    logger.info("  output        : %s", args.output)
    logger.info("=" * 60)

    # ── Collect CSV paths ──────────────────────────────────────────────────────
    csv_paths = _collect_csv_paths(args.data_dir)
    logger.info("Found %d labelled CSV(s): %s", len(csv_paths), csv_paths)

    fc_params = EfficientFCParameters()

    if args.no_eval:
        # Train on everything, no held-out set
        backend = DefaultBackend()
        backend.train(
            labelled_csv_paths=csv_paths,
            window_size=args.window_size,
            step_size=args.step_size,
            severity_threshold=args.severity_threshold,
        )
    else:
        # ── Build windows + extract features ──────────────────────────────────
        logger.info("Building windows and extracting features…")
        X_full, y_all = _build_windows_from_csvs(
            csv_paths, args.window_size, args.step_size, args.severity_threshold, fc_params
        )

        # ── Train with evaluation ──────────────────────────────────────────────
        logger.info("\nTraining classifiers (test_size=%.0f%%)…", args.test_size * 100)

        from sklearn.ensemble import RandomForestClassifier
        from tsfresh import select_features as ts_select

        models: dict = {}
        feature_cols: dict = {}

        print("\n" + "=" * 60)
        print("Per-type classification report (test split)")
        print("=" * 60)

        for col in _LABEL_COLS:
            y = y_all[col]
            valid_mask = ~y.isna()
            y_clean = y[valid_mask].astype(int)
            X_clean = X_full.loc[y_clean.index]

            n_pos = int(y_clean.sum())
            n_neg = int((y_clean == 0).sum())
            logger.info("\n%s — %d windows (%d pos, %d neg)", col, len(y_clean), n_pos, n_neg)

            if y_clean.nunique() < 2:
                logger.warning("  Skipping — only one class present.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean,
                test_size=args.test_size,
                random_state=42,
                stratify=y_clean,
            )

            # Feature selection on train split only
            try:
                X_train_sel = ts_select(X_train, y_train)
            except Exception as exc:
                logger.warning("  Feature selection failed (%s); using all features.", exc)
                X_train_sel = X_train

            if X_train_sel.shape[1] == 0:
                X_train_sel = X_train

            selected_cols = X_train_sel.columns.tolist()
            X_test_sel = X_test.reindex(columns=selected_cols, fill_value=0.0)

            clf = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_train_sel, y_train)
            y_pred = clf.predict(X_test_sel)

            print(f"\n{col} ({X_train_sel.shape[1]} features selected)")
            print(classification_report(y_test, y_pred, zero_division=0))

            models[col] = clf
            feature_cols[col] = selected_cols

        # ── Assemble backend ───────────────────────────────────────────────────
        backend = DefaultBackend()
        backend._models = models
        backend._feature_cols = feature_cols
        backend._fc_params = fc_params

    # ── Save ──────────────────────────────────────────────────────────────────
    backend.save(args.output)
    logger.info("Done. Model saved to: %s", args.output)


if __name__ == "__main__":
    main()
