"""
Holdout Evaluation - Test a pretrained model against a labelled test set.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import DefaultBackend, _LABEL_COLS
from loo_cross_validation import _extract_test_features_and_labels, _metrics_from_arrays, _score_metrics

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run(
    backend: DefaultBackend,
    csv_paths: list[Path],
    window_size: int,
    step_size: int,
    severity_threshold: float,
) -> pd.DataFrame:
    records: list[dict] = []

    for csv_path in csv_paths:
        app_name = csv_path.stem
        logger.info("Extracting features: %s", csv_path.name)
        X, y_dict = _extract_test_features_and_labels(csv_path, window_size, step_size, severity_threshold)

        for col, clf in backend._models.items():
            if col not in y_dict:
                continue

            feature_cols = backend._feature_cols[col]
            X_aligned = X.reindex(columns=feature_cols, fill_value=backend._missing_fill_value)
            X_aligned = X_aligned.reindex(y_dict[col].index, fill_value=backend._missing_fill_value)

            thr = backend._thresholds.get(col, 0.5)
            probs = clf.predict_proba(X_aligned)[:, 1]
            y_pred = (probs >= thr).astype(int)
            y_true = y_dict[col].values

            m = _metrics_from_arrays(y_true, y_pred)
            sm = _score_metrics(y_true, probs)
            n_pos = int(y_true.sum())
            n_neg = int((y_true == 0).sum())

            print(
                f"  {app_name:<12} {col:<28} pos={n_pos:>4} neg={n_neg:>4}  "
                f"F1={m['f1']:.3f}  FAR={m['false_alarm_rate']:.3f}  MR={m['miss_rate']:.3f}"
            )

            records.append({
                "app": app_name,
                "bottleneck_type": col,
                "n_windows": len(y_true),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_features": len(feature_cols),
                "threshold": thr,
                **m,
                **sm,
            })

    return pd.DataFrame(records)


def _print_summary(results: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("  HOLDOUT EVALUATION SUMMARY (per-app average)")
    print(f"{'='*70}")

    per_app = results.groupby(["bottleneck_type", "app"])[["f1", "false_alarm_rate", "miss_rate"]].mean()
    agg = per_app.groupby("bottleneck_type").mean()

    print(f"\n  {'Bottleneck Type':<28}  {'Avg F1':>7}  {'Avg FAR':>8}  {'Avg Miss':>9}")
    print("  " + "-" * 58)
    for bt_name, row in agg.iterrows():
        print(f"  {bt_name:<28}  {row['f1']:>7.4f}  {row['false_alarm_rate']:>8.4f}  {row['miss_rate']:>9.4f}")
    print("  " + "-" * 58)
    print(f"  {'MACRO AVERAGE':<28}  {agg['f1'].mean():>7.4f}  {agg['false_alarm_rate'].mean():>8.4f}  {agg['miss_rate'].mean():>9.4f}")
    print()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a pretrained model on a labelled test set.")
    p.add_argument("--model", required=True, help="Path to trained .pkl (relative to repo root).")
    p.add_argument("--data-dir", required=True, help="Directory of labelled CSVs to test on.")
    p.add_argument("--window-size", type=int, default=None, help="Fallback if not stored in model.")
    p.add_argument("--step-size", type=int, default=None, help="Fallback if not stored in model.")
    p.add_argument("--severity-threshold", type=float, default=0.0, dest="severity_threshold")
    p.add_argument("--output-csv", type=str, default=None, dest="output_csv")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    backend = DefaultBackend.load(str(ROOT / args.model))
    window_size = args.window_size or backend._window_size
    step_size = args.step_size or window_size

    csv_paths = sorted(Path(args.data_dir).rglob("*.csv"))
    if not csv_paths:
        print(f"[ERROR] No CSVs found in {args.data_dir}")
        sys.exit(1)
    logger.info("Found %d CSV(s) in %s", len(csv_paths), args.data_dir)

    results = run(backend, csv_paths, window_size, step_size, args.severity_threshold)

    if results.empty:
        print("[ERROR] No results collected - check that CSVs contain valid labels for this model's classes.")
        sys.exit(1)

    _print_summary(results)

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
        print(f"[INFO] Per-app results saved to: {out}")
