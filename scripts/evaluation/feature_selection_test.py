"""
Nested CV: Feature Selection Variant Comparison

Compares 4 feature-selection strategies (fdr+imp, fdr_only, imp_only, none)
using an outer K-fold (default: 4) with threshold calibration via an inner
K-fold (default: 5) on the training apps of each outer fold.

Classifier hyperparameters are fixed; only the probability threshold is tuned
in the inner loop via calibrate_thresholds_cv().

Usage:
    python scripts/evaluation/feature_selection_test.py --classifier xgboost
    python scripts/evaluation/feature_selection_test.py --classifier rf \\
        --output-csv results/fs_rf.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hpc_bottleneck_detector.ml.backends.config import build_classifier
from hpc_bottleneck_detector.ml.backends.default_backend import (
    _LABEL_COLS,
    _NON_METRIC_COLS,
    _build_window_dataframe,
    _fill_metric_nans,
    _merge_app_y,
    _window_labels,
)
from hpc_bottleneck_detector.ml.backends.default_trainer import DefaultTrainer

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "labelled_data" / "training_set"

VARIANTS: list[tuple[str, bool, bool]] = [
    ("fdr+imp",  True,  True),
    ("fdr_only", True,  False),
    ("imp_only", False, True),
    ("none",     False, False),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_labelled_csvs(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No labelled CSVs found in '{data_dir}'.")
    return paths


def _extract_app_features(
    csv_path: Path,
    window_size: int,
    step_size: int,
    severity_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Build tsfresh feature matrix and binary label series for one app CSV."""
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from hpc_bottleneck_detector.ml.backends.default_backend import BASIC_FC_PARAMETERS

    df = pd.read_csv(csv_path)
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
    X_full = X_full.reindex(all_window_ids).apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_dict: dict[str, pd.Series] = {}
    for col in _LABEL_COLS:
        y = pd.Series(all_labels[col], index=all_window_ids, dtype=float)
        valid = ~y.isna()
        if valid.sum() > 0:
            y_clean = y[valid].astype(int)
            if y_clean.nunique() >= 1:
                y_dict[col] = y_clean

    return X_full, y_dict


def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0
        else float("nan")
    )
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    miss_rate        = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "miss_rate": miss_rate,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run(
    csv_paths: list[Path],
    classifier_name: str,
    window_size: int,
    step_size: int,
    severity_threshold: float,
    n_outer_splits: int,
    n_inner_splits: int,
    classifier_config: str | None = None,
) -> pd.DataFrame:
    print(f"[INFO] Pre-extracting features for {len(csv_paths)} apps...")
    all_app_features: list[tuple[pd.DataFrame, dict[str, pd.Series]]] = []
    for csv_path in csv_paths:
        print(f"  {csv_path.name}")
        X, y_dict = _extract_app_features(csv_path, window_size, step_size, severity_threshold)
        all_app_features.append((X, y_dict))

    n_apps = len(all_app_features)
    n_total_features = all_app_features[0][0].shape[1]
    base_clf = build_classifier(classifier_name, classifier_config)

    outer_kfold = KFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
    records: list[dict] = []

    for fs_name, use_fdr, use_imp in VARIANTS:
        print(f"\n{'='*70}")
        print(f"  Variant: {fs_name}  (use_fdr={use_fdr}, use_importance_pruning={use_imp})")
        print(f"{'='*70}")

        for outer_fold, (train_idx, test_idx) in enumerate(
            outer_kfold.split(range(n_apps)), 1
        ):
            train_features = [all_app_features[i] for i in train_idx]
            test_features  = [all_app_features[i] for i in test_idx]
            test_names     = [csv_paths[i].stem.replace("_labelled", "") for i in test_idx]

            print(f"\n  Outer fold {outer_fold}/{n_outer_splits}  |  test: {test_names}")

            trainer = DefaultTrainer(
                classifier=clone(base_clf),
                use_fdr=use_fdr,
                use_importance_pruning=use_imp,
            )

            t0 = time.time()
            # Inner CV: calibrate per-class probability thresholds on training apps
            thresholds = trainer.calibrate_thresholds_cv(
                train_features, n_splits=n_inner_splits
            )
            # Train final model on all training apps, apply thresholds from inner CV
            X_tr = pd.concat([f[0] for f in train_features]).fillna(0.0)
            y_tr = _merge_app_y([f[1] for f in train_features])
            backend = trainer.from_preextracted_features(X_tr, y_tr)
            backend._thresholds = thresholds
            train_time = time.time() - t0

            if not backend._models:
                logger.warning("  No classifiers trained for fold %d - skipping.", outer_fold)
                continue

            # Avg features selected in the final model
            n_features_per_type = {
                col: len(cols) for col, cols in backend._feature_cols.items()
            }
            avg_features = (
                sum(n_features_per_type.values()) / len(n_features_per_type)
                if n_features_per_type else 0.0
            )

            # Evaluate on test apps
            X_test = pd.concat([f[0] for f in test_features]).fillna(0.0)
            y_test = _merge_app_y([f[1] for f in test_features])

            # Collect predictions for all types (measure total inference time)
            per_type_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            t_inf = time.time()
            for col, clf in backend._models.items():
                if col not in y_test:
                    continue
                feature_cols = backend._feature_cols[col]
                X_aligned = X_test.reindex(
                    index=y_test[col].index, columns=feature_cols, fill_value=0.0
                )
                probs = clf.predict_proba(X_aligned)[:, 1]
                thr   = thresholds.get(col, 0.5)
                y_pred = (probs >= thr).astype(int)
                per_type_results[col] = (y_test[col].values, y_pred)

            total_windows = sum(len(v[0]) for v in per_type_results.values())
            inference_ms = (
                (time.time() - t_inf) / total_windows * 1000
                if total_windows > 0 else float("nan")
            )

            # Build records
            for col, (y_true, y_pred) in per_type_results.items():
                m = _metrics_from_arrays(y_true, y_pred)
                n_feat = n_features_per_type.get(col, 0)
                n_pos = int(y_true.sum())
                n_neg = int((y_true == 0).sum())

                print(
                    f"    {col:<42}  pos={n_pos:>4}  neg={n_neg:>4}  "
                    f"F1={m['f1']:.3f}  FAR={m['false_alarm_rate']:.3f}  "
                    f"MR={m['miss_rate']:.3f}  feat={n_feat}"
                )

                records.append({
                    "fs_variant":             fs_name,
                    "outer_fold":             outer_fold,
                    "bottleneck_type":        col,
                    "n_windows":              len(y_true),
                    "n_positive":             n_pos,
                    "n_negative":             n_neg,
                    "n_features":             n_feat,
                    "feature_fraction":       n_feat / n_total_features if n_total_features > 0 else float("nan"),
                    "avg_features_all_types": avg_features,
                    "train_time_s":           train_time,
                    "inference_ms_per_window": inference_ms,
                    **m,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(
    results: pd.DataFrame,
    classifier_name: str,
    n_outer: int,
    n_inner: int,
) -> None:
    print(f"\n{'='*84}")
    print("  NESTED CV FEATURE SELECTION SUMMARY")
    print(f"  Classifier: {classifier_name}  |  Outer folds: {n_outer}  |  Inner folds: {n_inner}")
    print(f"{'='*84}")

    agg = (
        results
        .groupby("fs_variant")[
            ["f1", "false_alarm_rate", "miss_rate",
             "avg_features_all_types", "feature_fraction",
             "train_time_s", "inference_ms_per_window"]
        ]
        .agg(lambda s: s.dropna().mean() if s.dropna().size > 0 else float("nan"))
    )

    col_w = 12
    header = (
        f"  {'Variant':<{col_w}}  {'Macro F1':>8}  {'FAR':>8}  {'Miss':>8}  "
        f"{'Avg Feat':>9}  {'Feat %':>7}  {'Train(s)':>9}  {'Inf(ms/w)':>10}"
    )
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for variant, row in agg.iterrows():
        print(
            f"  {variant:<{col_w}}  {row['f1']:>8.4f}  "
            f"{row['false_alarm_rate']:>8.4f}  {row['miss_rate']:>8.4f}  "
            f"{row['avg_features_all_types']:>9.1f}  "
            f"{row['feature_fraction']*100:>6.1f}%  "
            f"{row['train_time_s']:>9.1f}  "
            f"{row['inference_ms_per_window']:>10.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nested CV comparison of feature-selection variants."
    )
    p.add_argument(
        "--classifier", choices=["xgboost", "rf"], required=True,
        help="Classifier to evaluate (run independently per classifier).",
    )
    p.add_argument("--n-outer-splits", type=int, default=4,  dest="n_outer_splits")
    p.add_argument("--n-inner-splits", type=int, default=5,  dest="n_inner_splits")
    p.add_argument("--window-size",    type=int, default=12, dest="window_size")
    p.add_argument("--step-size",      type=int, default=12, dest="step_size")
    p.add_argument(
        "--severity-threshold", type=float, default=0.0, dest="severity_threshold",
        help="Severity > this value -> positive label (default: 0.0).",
    )
    p.add_argument(
        "--output-csv", type=str, default=None, dest="output_csv",
        help="Optional path to save per-fold/per-type results as CSV.",
    )
    p.add_argument(
        "--classifier-config", type=str, default=None, dest="classifier_config",
        help="Path to YAML file with classifier hyperparameters to override defaults.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    csv_paths = _find_labelled_csvs(DATA_DIR)
    print(
        f"[INFO] Found {len(csv_paths)} labelled CSV(s) - "
        f"running {args.n_outer_splits}-fold outer / {args.n_inner_splits}-fold inner CV"
    )

    results = run(
        csv_paths=csv_paths,
        classifier_name=args.classifier,
        window_size=args.window_size,
        step_size=args.step_size,
        severity_threshold=args.severity_threshold,
        n_outer_splits=args.n_outer_splits,
        n_inner_splits=args.n_inner_splits,
        classifier_config=args.classifier_config,
    )

    if results.empty:
        print("[ERROR] No results collected - check that CSVs contain valid labels.")
        sys.exit(1)

    _print_summary(results, args.classifier, args.n_outer_splits, args.n_inner_splits)

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
        print(f"[INFO] Results saved to: {out}")
