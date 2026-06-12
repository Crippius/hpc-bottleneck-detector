"""
LOO cross-validation for all 4 feature-selection variants.

Runs the full Leave-One-Out rotation for each combination of
use_fdr x use_importance_pruning and prints a side-by-side comparison.

Usage:
    python scripts/loo_variants.py
    python scripts/loo_variants.py --data-dir data/labelled_data/training_set --output-csv results/loo_variants.csv
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
    _window_labels,
    BASIC_FC_PARAMETERS,
)
from hpc_bottleneck_detector.ml.backends.default_trainer import DefaultTrainer
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logging.basicConfig(
    level=logging.WARNING,  # quiet - only print our own progress lines
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

VARIANTS: list[tuple[str, bool, bool]] = [
    ("fdr+imp",  True,  True),
    ("fdr_only", True,  False),
    ("imp_only", False, True),
    ("none",     False, False),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/labelled_data/training_set")
    p.add_argument("--window-size",        type=int,   default=12)
    p.add_argument("--step-size",          type=int,   default=12)
    p.add_argument("--severity-threshold", type=float, default=0.0)
    p.add_argument("--prob-threshold",     type=float, default=0.5)
    p.add_argument("--output-csv",         default="results/loo_variants.csv")
    return p.parse_args()


def _extract_test_features_and_labels(
    test_csv: Path,
    window_size: int,
    step_size: int,
    severity_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    df = pd.read_csv(test_csv)
    metric_cols = [c for c in df.columns if c not in _NON_METRIC_COLS]

    all_fragments, all_window_ids = [], []
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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0
          else float("nan"))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "f1": f1,
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
        "miss_rate":        fn / (fn + tp) if (fn + tp) > 0 else float("nan"),
    }


def run_loo_variant(
    variant_name: str,
    use_fdr: bool,
    use_importance_pruning: bool,
    csv_paths: list[Path],
    window_size: int,
    step_size: int,
    severity_threshold: float,
    prob_threshold: float,
) -> pd.DataFrame:
    records: list[dict] = []
    n = len(csv_paths)

    for fold_idx, test_csv in enumerate(csv_paths):
        app_name = test_csv.stem.replace("_labelled", "")
        train_paths = [p for p in csv_paths if p != test_csv]

        print(f"  [{variant_name}] fold {fold_idx+1}/{n} - held-out: {app_name}")

        backend = DefaultTrainer(use_fdr=use_fdr, use_importance_pruning=use_importance_pruning).train(
            labelled_csv_paths=[str(p) for p in train_paths],
            window_size=window_size,
            step_size=step_size,
            severity_threshold=severity_threshold,
        )

        if not backend._models:
            continue

        X_test, y_dict = _extract_test_features_and_labels(
            test_csv, window_size, step_size, severity_threshold
        )

        for col, clf in backend._models.items():
            if col not in y_dict:
                continue
            feature_cols = backend._feature_cols[col]
            X_aligned = X_test.reindex(columns=feature_cols, fill_value=0.0)
            X_aligned = X_aligned.reindex(y_dict[col].index, fill_value=0.0)
            probs  = clf.predict_proba(X_aligned)[:, 1]
            y_pred = (probs >= prob_threshold).astype(int)
            y_true = y_dict[col].values
            m = _metrics(y_true, y_pred)
            records.append({
                "variant": variant_name,
                "fold": fold_idx + 1,
                "app": app_name,
                "bottleneck_type": col,
                "n_features": len(feature_cols),
                **m,
            })

    return pd.DataFrame(records)


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-variant x per-bottleneck macro averages."""
    return (
        df.groupby(["variant", "bottleneck_type"])[["f1", "false_alarm_rate", "miss_rate", "n_features"]]
        .agg(lambda s: s.dropna().mean() if s.dropna().size > 0 else float("nan"))
        .reset_index()
    )


def _print_comparison(summary: pd.DataFrame) -> None:
    bt_types = sorted(summary["bottleneck_type"].unique())
    variant_names = [v[0] for v in VARIANTS]

    for metric, label in [("f1", "F1"), ("false_alarm_rate", "FAR"), ("miss_rate", "Miss Rate")]:
        print(f"\n{'─'*90}")
        print(f"  {label}")
        print(f"{'─'*90}")
        print(f"  {'Bottleneck':<45}", end="")
        for v in variant_names:
            print(f"  {v:>10}", end="")
        print()
        print("  " + "-" * 87)

        col_vals = {v: [] for v in variant_names}
        for bt in bt_types:
            print(f"  {bt:<45}", end="")
            for v in variant_names:
                row = summary[(summary["variant"] == v) & (summary["bottleneck_type"] == bt)]
                val = row[metric].values[0] if not row.empty else float("nan")
                col_vals[v].append(val)
                print(f"  {val:>10.4f}" if not np.isnan(val) else f"  {'N/A':>10}", end="")
            print()

        print("  " + "-" * 87)
        print(f"  {'MACRO AVG':<45}", end="")
        for v in variant_names:
            vals = [x for x in col_vals[v] if not np.isnan(x)]
            avg = sum(vals) / len(vals) if vals else float("nan")
            print(f"  {avg:>10.4f}" if not np.isnan(avg) else f"  {'N/A':>10}", end="")
        print()


def main() -> None:
    args = _parse_args()
    csv_paths = sorted(Path(args.data_dir).rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found in '{args.data_dir}'.")
    print(f"Found {len(csv_paths)} CSV(s) - running {len(csv_paths)}-fold LOO for {len(VARIANTS)} variants\n")

    all_results: list[pd.DataFrame] = []

    for name, use_fdr, use_imp in VARIANTS:
        print(f"\n{'='*70}")
        print(f"  Variant: {name}  (use_fdr={use_fdr}, use_importance_pruning={use_imp})")
        print(f"{'='*70}")
        df = run_loo_variant(
            variant_name=name,
            use_fdr=use_fdr,
            use_importance_pruning=use_imp,
            csv_paths=csv_paths,
            window_size=args.window_size,
            step_size=args.step_size,
            severity_threshold=args.severity_threshold,
            prob_threshold=args.prob_threshold,
        )
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    summary = _summarise(combined)

    print(f"\n\n{'='*90}")
    print("  LOO RESULTS - VARIANT COMPARISON")
    print(f"{'='*90}")
    _print_comparison(summary)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\nFull results saved to: {out}")


if __name__ == "__main__":
    main()
