"""
Leave-One-Out (LOO) Cross-Validation for DefaultBackend

Evaluates how well the Random Forest model generalises to unseen applications

Usage:
    python examples/loo_cross_validation.py [--window-size 12] [--step-size 12]
                                            [--threshold 0.5] [--prob-threshold 0.5]
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import (
    _LABEL_COLS,
    _NON_METRIC_COLS,
    _build_window_dataframe,
    _fill_metric_nans,
    _merge_app_y,
    _window_labels,
)
from hpc_bottleneck_detector.ml.backends.default_trainer import DefaultTrainer


def _build_classifier(name: str):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, scale_pos_weight=10, random_state=42, n_jobs=-1, eval_metric="logloss")

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR  = REPO_ROOT / "data" / "labelled_data" / "training_set"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_labelled_csvs(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No labelled CSVs found in '{data_dir}'.\n"
            "Run examples/labeling_example.py first."
        )
    return paths


def _extract_test_features_and_labels(
    test_csv: Path,
    window_size: int,
    step_size: int,
    severity_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """
    Build tsfresh feature matrix and ground-truth label series for test_csv.

    Returns
    -------
    X_full : pd.DataFrame
        One row per window, indexed by window id.
    y_dict : dict[str, pd.Series]
        ``{col: Series[int]}`` with binary labels (0/1) for valid windows only.
    """
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from hpc_bottleneck_detector.ml.backends.default_backend import BASIC_FC_PARAMETERS

    df = pd.read_csv(test_csv)
    metric_cols = [c for c in df.columns if c not in _NON_METRIC_COLS]

    all_fragments: list[pd.DataFrame] = []
    all_window_ids: list[str] = []
    all_labels: dict[str, list] = {col: [] for col in _LABEL_COLS}

    for job_id, job_df in df.groupby("id"):
        job_df = job_df.sort_values("time").reset_index(drop=True)

        long_df, window_ids = _build_window_dataframe(
            job_df, metric_cols, str(job_id), window_size, step_size
        )
        labels = _window_labels(job_df, window_size, step_size, severity_threshold)

        all_fragments.append(long_df)
        all_window_ids.extend(window_ids)
        for col in _LABEL_COLS:
            all_labels[col].extend(labels[col])

    tsfresh_df = _fill_metric_nans(pd.concat(all_fragments, ignore_index=True))

    X_full = extract_features(
        tsfresh_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=BASIC_FC_PARAMETERS,
        impute_function=impute,
        disable_progressbar=True,
    )
    X_full = X_full.reindex(all_window_ids)
    X_full = X_full.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_dict: dict[str, pd.Series] = {}
    for col in _LABEL_COLS:
        y = pd.Series(all_labels[col], index=all_window_ids, dtype=float)
        valid = ~y.isna()
        if valid.sum() > 0:
            y_clean = y[valid].astype(int)
            if y_clean.nunique() >= 1:
                y_dict[col] = y_clean

    return X_full, y_dict


def _metrics_from_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """
    Compute F1, False Alarm Rate, and Anomaly Miss Rate from binary arrays.

    Uses safe division - returns NaN when the denominator is zero.
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision  = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall     = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1         = (2 * precision * recall / (precision + recall)
                  if (not np.isnan(precision) and not np.isnan(recall)
                      and (precision + recall) > 0)
                  else float("nan"))
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    miss_rate        = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
    }


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LOO driver
# ---------------------------------------------------------------------------

def run_loo(
    csv_paths: list[Path],
    window_size: int,
    step_size: int,
    severity_threshold: float,
    prob_threshold: float,
    calibrate: bool = False,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Execute the full Leave-One-Out rotation and return a DataFrame of per-fold
    per-BottleneckType metrics.
    """
    trainer = DefaultTrainer(classifier=_build_classifier(args.classifier))
    records: list[dict] = []
    n = len(csv_paths)

    # Pre-extract features once per app
    print("[INFO] Pre-extracting features for all apps...")
    all_app_features: list[tuple[pd.DataFrame, dict[str, pd.Series]]] = []
    for csv_path in csv_paths:
        print(f"  Extracting features: {csv_path.name}")
        X, y_dict = _extract_test_features_and_labels(csv_path, window_size, step_size, severity_threshold)
        all_app_features.append((X, y_dict))

    for fold_idx, test_csv in enumerate(csv_paths):
        app_name = test_csv.stem.replace("_labelled", "")
        train_indices = [i for i in range(n) if i != fold_idx]
        train_app_features = [all_app_features[i] for i in train_indices]

        print(f"\n{'='*70}")
        print(f"  Fold {fold_idx + 1}/{n}  |  held-out application: {app_name}")
        print(f"{'='*70}")
        print(f"  Training on: {[csv_paths[i].stem for i in train_indices]}")

        # ----- Train -----
        X_tr = pd.concat([f[0] for f in train_app_features]).fillna(0.0)
        y_tr = _merge_app_y([f[1] for f in train_app_features])
        backend = trainer.from_preextracted_features(X_tr, y_tr)

        if not backend._models:
            logger.warning("  No classifiers trained for fold %d - skipping.", fold_idx + 1)
            continue

        # ----- Calibrate thresholds (optional) -----
        if calibrate:
            thresholds = trainer.calibrate_thresholds_cv(train_app_features, n_splits, default_threshold=prob_threshold)
        else:
            thresholds = {col: prob_threshold for col in _LABEL_COLS}

        # ----- Predict & evaluate per BottleneckType -----
        X_test, y_dict = all_app_features[fold_idx]
        print(f"\n  Evaluating on {test_csv.name} ...")

        for col, clf in backend._models.items():
            if col not in y_dict:
                logger.debug("  %s: no valid labels in test set - skipped.", col)
                continue

            feature_cols = backend._feature_cols[col]
            X_aligned = X_test.reindex(columns=feature_cols, fill_value=0.0)

            valid_idx = y_dict[col].index
            X_aligned = X_aligned.reindex(valid_idx, fill_value=0.0)

            thr        = thresholds.get(col, prob_threshold)
            probs      = clf.predict_proba(X_aligned)[:, 1]
            y_pred     = (probs >= thr).astype(int)
            y_true     = y_dict[col].values
            n_features = len(feature_cols)

            m = _metrics_from_arrays(y_true, y_pred)
            n_pos  = int(y_true.sum())
            n_neg  = int((y_true == 0).sum())

            print(
                f"    {col:<42}  "
                f"pos={n_pos:>4}  neg={n_neg:>4}  "
                f"F1={m['f1']:.3f}  FAR={m['false_alarm_rate']:.3f}  "
                f"MR={m['miss_rate']:.3f}  feat={n_features}"
            )

            records.append({
                "fold":            fold_idx + 1,
                "app":             app_name,
                "bottleneck_type": col,
                "n_windows":       len(y_true),
                "n_positive":      n_pos,
                "n_negative":      n_neg,
                "n_features":      n_features,
                **m,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: pd.DataFrame) -> None:
    """Print per-BottleneckType averages and a grand average across all types."""
    print(f"\n{'='*70}")
    print("  LEAVE-ONE-OUT CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    thr_str = f"calibrated (GroupKFold n={args.n_splits})" if args.calibrate else str(args.prob_threshold)
    print(f"\n  Classification threshold    : {thr_str}")
    print(f"  Window size / step size     : {args.window_size} / {args.step_size}")
    print(f"  Severity threshold (labels) : {args.severity_threshold}")
    print(f"  Folds completed             : {results['fold'].nunique()}")
    print(f"  Total foldxtype evaluations : {len(results)}")

    agg = (
        results
        .groupby("bottleneck_type")[["f1", "false_alarm_rate", "miss_rate", "n_features"]]
        .agg(lambda s: s.dropna().mean() if s.dropna().size > 0 else float("nan"))
    )

    print(f"\n  {'Bottleneck Type':<42}  {'Avg F1':>7}  {'Avg FAR':>8}  {'Avg Miss':>9}  {'Avg Feat':>9}")
    print("  " + "-" * 83)
    for bt_name, row in agg.iterrows():
        print(
            f"  {bt_name:<42}  "
            f"{row['f1']:>7.4f}  "
            f"{row['false_alarm_rate']:>8.4f}  "
            f"{row['miss_rate']:>9.4f}  "
            f"{row['n_features']:>9.1f}"
        )

    # Grand (macro) average across all bottleneck types
    grand_f1   = agg["f1"].dropna().mean()
    grand_far  = agg["false_alarm_rate"].dropna().mean()
    grand_mr   = agg["miss_rate"].dropna().mean()
    grand_feat = agg["n_features"].dropna().mean()
    print("  " + "-" * 83)
    print(
        f"  {'MACRO AVERAGE':<42}  "
        f"{grand_f1:>7.4f}  "
        f"{grand_far:>8.4f}  "
        f"{grand_mr:>9.4f}  "
        f"{grand_feat:>9.1f}"
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Leave-One-Out cross-validation for the ML bottleneck detector."
    )
    p.add_argument("--window-size",      type=int,   default=12,  dest="window_size")
    p.add_argument("--step-size",        type=int,   default=12,  dest="step_size")
    p.add_argument("--severity-threshold", type=float, default=0.0, dest="severity_threshold",
                   help="Severity > this value -> positive label (default: 0.0)")
    p.add_argument("--prob-threshold",   type=float, default=0.5, dest="prob_threshold",
                   help="Probability ≥ this value -> predicted bottleneck (default: 0.5)")
    p.add_argument("--output-csv",       type=str,   default=None, dest="output_csv",
                   help="Optional path to save per-fold results as CSV.")
    p.add_argument("--classifier", choices=["xgboost", "rf"], default="xgboost",
                   help="Classifier to use: 'xgboost' (default) or 'rf'.")
    p.add_argument("--calibrate", action="store_true",
                   help="Run GroupKFold CV within each fold to calibrate per-class thresholds.")
    p.add_argument("--n-splits", type=int, default=5, dest="n_splits",
                   help="GroupKFold splits for threshold calibration (default: 5).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    csv_paths = _find_labelled_csvs(DATA_DIR)
    print(f"[INFO] Found {len(csv_paths)} labelled CSV(s) - running {len(csv_paths)}-fold LOO CV")

    results = run_loo(
        csv_paths          = csv_paths,
        window_size        = args.window_size,
        step_size          = args.step_size,
        severity_threshold = args.severity_threshold,
        prob_threshold     = args.prob_threshold,
        calibrate          = args.calibrate,
        n_splits           = args.n_splits,
    )

    if results.empty:
        print("[ERROR] No results collected - check that CSVs contain valid labels.")
        sys.exit(1)

    _print_summary(results)

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
        print(f"[INFO] Per-fold results saved to: {out}")
