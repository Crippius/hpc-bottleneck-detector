"""
Training Script - Supervised ML Model

Trains a :class:`DefaultBackend` on labelled CSVs produced by label_jobs.py
and saves the result to disk.

Usage
-----
    python scripts/training/train_ml_model.py
    python scripts/training/train_ml_model.py \\
        --data-dir data/labelled_data/ \\
        --window-size 12 \\
        --step-size 12 \\
        --output models/default.pkl

Output
------
- A .pkl file loadable with DefaultBackend.load(path).
- A per-type classification report printed to stdout.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import (
    _build_window_dataframe,
    _window_labels,
    _merge_app_y,
    _LABEL_COLS,
    _NON_METRIC_COLS,
    EXCLUDE_METRIC_PREFIXES,
    EXCLUDE_METRIC_COLS,
    BASIC_FC_PARAMETERS,
)
from hpc_bottleneck_detector.ml.backends.default_trainer import DefaultTrainer
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DefaultBackend on labelled HPC job CSVs."
    )
    parser.add_argument("--data-dir", default="data/labelled_data/training_set/",
        help="Directory containing labelled CSV files.")
    parser.add_argument("--window-size", type=int, default=12,
        help="Number of intervals per analysis window.")
    parser.add_argument("--step-size", type=int, default=12,
        help="Interval advance between successive windows.")
    parser.add_argument("--severity-threshold", type=float, default=0.0,
        help="Severity > this value → positive label.")
    parser.add_argument("-o", "--output", default="models/default.pkl",
        help="Output path for the saved backend (.pkl).")
    parser.add_argument("--test-size", type=float, default=0.2,
        help="Fraction of windows held out for evaluation.")
    parser.add_argument("--no-eval", action="store_true",
        help="Skip the held-out evaluation and train on all data.")
    parser.add_argument("--calibrate", action="store_true",
        help="Run GroupKFold CV over training apps to calibrate per-class probability thresholds.")
    parser.add_argument("--n-splits", type=int, default=5,
        help="GroupKFold splits for threshold calibration (default: 5).")
    return parser.parse_args()


def _collect_csv_paths(data_dir: str) -> list[str]:
    paths = sorted(Path(data_dir).rglob("*.csv"))
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
    fc_params: dict,
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
        metric_cols = [
            c for c in df.columns
            if c not in _NON_METRIC_COLS
            and not any(c.startswith(p) for p in EXCLUDE_METRIC_PREFIXES)
            and c not in EXCLUDE_METRIC_COLS
        ]

        for job_id, job_df in df.groupby("id"):
            job_df = job_df.sort_values("time").dropna(subset=metric_cols).reset_index(drop=True)

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
    args = _parse_args()

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

    # ── Split: test_size=0 or --no-eval → no held-out test set ────────────────
    if args.no_eval or args.test_size == 0.0:
        train_paths, test_paths = csv_paths, []
    else:
        train_paths, test_paths = train_test_split(
            csv_paths, test_size=args.test_size, random_state=42
        )
    logger.info("Train: %d CSVs, Test: %d CSVs", len(train_paths), len(test_paths))

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = DefaultTrainer()

    if args.calibrate:
        # ── Per-app feature extraction ─────────────────────────────────────────
        logger.info("Extracting features per app for CV threshold calibration…")
        app_features: list[tuple[pd.DataFrame, dict[str, pd.Series]]] = []
        for path in train_paths:
            logger.info("  %s", path)
            X_i, y_dict_i = _build_windows_from_csvs(
                [path], args.window_size, args.step_size,
                args.severity_threshold, BASIC_FC_PARAMETERS,
            )
            y_dict_clean = {
                col: y[~y.isna()].astype(int)
                for col, y in y_dict_i.items()
                if (~y.isna()).sum() > 0
            }
            app_features.append((X_i, y_dict_clean))

        # ── GroupKFold CV to calibrate per-class thresholds ───────────────────
        n_folds = min(args.n_splits, len(app_features))
        logger.info("Running GroupKFold(n_splits=%d) threshold calibration…", n_folds)
        gkf = GroupKFold(n_splits=n_folds)
        indices = np.arange(len(app_features))
        thr_lists: dict[str, list[float]] = {col: [] for col in _LABEL_COLS}

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(indices, groups=indices)):
            X_tr = pd.concat([app_features[i][0] for i in tr_idx]).fillna(0.0)
            y_tr = _merge_app_y([app_features[i][1] for i in tr_idx])
            X_va = pd.concat([app_features[i][0] for i in va_idx]).fillna(0.0)
            y_va = _merge_app_y([app_features[i][1] for i in va_idx])

            fold_backend = trainer.from_preextracted_features(X_tr, y_tr)
            fold_backend.calibrate_thresholds(X_va, y_va)
            logger.info("  Fold %d thresholds: %s", fold, fold_backend._thresholds)
            for col in _LABEL_COLS:
                thr = fold_backend._thresholds.get(col)
                if thr is not None:
                    thr_lists[col].append(thr)

        calibrated = {
            col: float(np.nanmean(v)) if v else 0.5
            for col, v in thr_lists.items()
        }
        logger.info("Calibrated thresholds (mean over folds): %s", calibrated)

        # ── Final model: train on all training apps ────────────────────────────
        X_train_all = pd.concat([X for X, _ in app_features]).fillna(0.0)
        y_train_all = _merge_app_y([y for _, y in app_features])
        backend = trainer.from_preextracted_features(X_train_all, y_train_all)
        backend._thresholds = calibrated
    else:
        backend = trainer.train(
            labelled_csv_paths=train_paths,
            window_size=args.window_size,
            step_size=args.step_size,
            severity_threshold=args.severity_threshold,
        )

    # ── Evaluate (skipped when test_paths is empty) ────────────────────────────
    if test_paths:
        logger.info("Extracting test features from %d CSVs…", len(test_paths))
        X_test, y_test = _build_windows_from_csvs(
            test_paths, args.window_size, args.step_size, args.severity_threshold,
            backend._fc_params,
        )

        print("\n" + "=" * 60)
        print("Per-type classification report (held-out CSVs)")
        print("=" * 60)

        for col in _LABEL_COLS:
            y = y_test[col]
            valid_mask = ~y.isna()
            y_clean = y[valid_mask].astype(int)

            if col not in backend._models or y_clean.nunique() < 2:
                logger.warning("  Skipping %s - model missing or only one class in test.", col)
                continue

            n_pos = int(y_clean.sum())
            n_neg = int((y_clean == 0).sum())
            X_sel = X_test.loc[y_clean.index].reindex(
                columns=backend._feature_cols[col], fill_value=0.0
            )
            thr = backend._thresholds.get(col, 0.5)
            probs = backend._models[col].predict_proba(X_sel)[:, 1]
            y_pred = (probs >= thr).astype(int)

            print(f"\n{col} (thr={thr:.2f}, {len(backend._feature_cols[col])} features, {n_pos} pos / {n_neg} neg)")
            print(classification_report(y_clean, y_pred, zero_division=0))

    # ── Save ──────────────────────────────────────────────────────────────────
    backend.save(args.output)
    logger.info("Done. Model saved to: %s", args.output)


if __name__ == "__main__":
    main()
