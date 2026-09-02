"""
Heuristic vs. ML Disagreement Analysis

Compares each classifier's LOO predictions against the heuristic label
per window, and finds which features distinguish misses/false-alarms from
agreement cases.

Usage
-----
    python scripts/analysis/heuristic_ml_disagreement.py \\
        --windows results/disagreement/xgboost_windows.parquet results/disagreement/rf_windows.parquet \\
        --labels XGBoost RF \\
        --baseline-csv results/loo/loo_xgboost.csv results/loo/loo_rf_new.csv \\
        --output-dir results/disagreement
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# --- Loading -----------------------------------------------------------------

def _features_path(windows_path: Path) -> Path:
    return windows_path.with_name(f"{windows_path.stem}_features{windows_path.suffix}")


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


# --- Four-case summary + verification against the published LOO artifacts ----

def case_summary(windows: pd.DataFrame) -> pd.DataFrame:
    """Per-class TP/TN/FN/FP counts and rates from the window-detail table."""
    rows = []
    for bt, grp in windows.groupby("bottleneck_type"):
        counts = grp["case"].value_counts()
        tp, tn, fn, fp = (int(counts.get(c, 0)) for c in ("TP", "TN", "FN", "FP"))
        n = tp + tn + fn + fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
            else float("nan")
        )
        far  = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        miss = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

        rows.append({
            "bottleneck_type": bt, "n_windows": n,
            "tp": tp, "tn": tn, "fn": fn, "fp": fp,
            "f1": f1, "false_alarm_rate": far, "miss_rate": miss,
        })
    return pd.DataFrame(rows)


def verify_against_baseline(summary: pd.DataFrame, baseline_csv: Path, tol: float = 0.02) -> None:
    """
    Cross-check the rerun's aggregate F1/FAR/miss per class against the
    published LOO artifact (results/loo/loo_xgboost.csv or loo_rf_new.csv),
    which is pooled the same way (raw tp/fp/fn/tn summed across all folds).
    A mismatch beyond `tol` signals the reconstructed protocol (severity
    threshold, window/step size, calibration) doesn't match the original run.
    """
    baseline = pd.read_csv(baseline_csv)
    pooled = baseline.groupby("bottleneck_type")[["tp", "fp", "fn", "tn"]].sum()
    pooled["f1"] = 2 * pooled["tp"] / (2 * pooled["tp"] + pooled["fp"] + pooled["fn"])
    pooled["false_alarm_rate"] = pooled["fp"] / (pooled["fp"] + pooled["tn"])
    pooled["miss_rate"] = pooled["fn"] / (pooled["fn"] + pooled["tp"])

    print(f"\n  Verification against {baseline_csv.name} (pooled):")
    print(f"  {'Bottleneck Type':<28}  {'F1 (rerun)':>10}  {'F1 (baseline)':>13}  {'diff':>6}  {'OK?':>4}")
    ok_all = True
    for _, row in summary.iterrows():
        bt = row["bottleneck_type"]
        if bt not in pooled.index:
            continue
        base_f1 = pooled.loc[bt, "f1"]
        diff = abs(row["f1"] - base_f1) if not np.isnan(row["f1"]) and not np.isnan(base_f1) else float("nan")
        ok = (not np.isnan(diff)) and diff <= tol
        ok_all &= ok
        print(f"  {bt:<28}  {row['f1']:>10.4f}  {base_f1:>13.4f}  {diff:>6.3f}  {'yes' if ok else 'NO':>4}")

    if not ok_all:
        print(
            "\n  [WARN] One or more classes drifted beyond the tolerance "
            f"({tol}) from the baseline. The reconstructed LOO protocol "
            "(severity threshold / window / step size / calibration) likely "
            "doesn't match the run that produced the baseline - do not trust "
            "the FN/FP feature summaries below until this is resolved."
        )
    else:
        print("\n  All classes within tolerance - rerun matches the published LOO protocol.")


# --- FN/FP feature divergence ------------------------------------------------

def feature_divergence(
    features: pd.DataFrame,
    bottleneck_type: str,
    disagreement_case: str,
    agreement_case: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Mann-Whitney rank-sum test per feature, comparing `disagreement_case`
    windows (e.g. FN) against the `agreement_case` baseline sample (e.g. TP)
    for one bottleneck class. Returns the top_n features by p-value.
    """
    grp = features[features["bottleneck_type"] == bottleneck_type]
    dis = grp[grp["case"] == disagreement_case]
    agr = grp[grp["case"] == agreement_case]

    rows = []
    for feat_name in sorted(set(dis["feature_name"])):
        dis_vals = dis.loc[dis["feature_name"] == feat_name, "feature_value"].values
        agr_vals = agr.loc[agr["feature_name"] == feat_name, "feature_value"].values
        if len(dis_vals) < 2 or len(agr_vals) < 2:
            continue
        try:
            stat, p = mannwhitneyu(dis_vals, agr_vals, alternative="two-sided")
        except ValueError:
            continue
        rows.append({
            "feature_name": feat_name,
            "n_disagreement": len(dis_vals), "n_agreement": len(agr_vals),
            f"median_{disagreement_case.lower()}": float(np.median(dis_vals)),
            f"median_{agreement_case.lower()}": float(np.median(agr_vals)),
            "p_value": float(p),
        })

    if not rows:
        return pd.DataFrame(
            columns=["feature_name", "n_disagreement", "n_agreement", "p_value"]
        )
    return pd.DataFrame(rows).sort_values("p_value").head(top_n).reset_index(drop=True)


def severity_score_profile(windows: pd.DataFrame, bottleneck_type: str) -> dict:
    """Distribution of heuristic severity on FN windows and ML score on FP windows."""
    grp = windows[windows["bottleneck_type"] == bottleneck_type]
    fn_sev = grp.loc[grp["case"] == "FN", "severity"]
    fp_score = grp.loc[grp["case"] == "FP", "score"]
    return {
        "bottleneck_type": bottleneck_type,
        "fn_severity_median": float(fn_sev.median()) if len(fn_sev) else float("nan"),
        "fn_severity_p90": float(fn_sev.quantile(0.9)) if len(fn_sev) else float("nan"),
        "n_fn": int(len(fn_sev)),
        "fp_score_median": float(fp_score.median()) if len(fp_score) else float("nan"),
        "fp_score_p90": float(fp_score.quantile(0.9)) if len(fp_score) else float("nan"),
        "n_fp": int(len(fp_score)),
    }


# --- CLI ---------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--windows", nargs="+", required=True,
                   help="One or more window-detail files from --save-window-detail.")
    p.add_argument("--labels", nargs="+", required=True,
                   help="Display label per --windows file, e.g. XGBoost RF.")
    p.add_argument("--baseline-csv", nargs="*", default=None,
                   help="Optional per-classifier results/loo/*.csv to verify the rerun against "
                        "(same order as --windows).")
    p.add_argument("--output-dir", default="results/disagreement", dest="output_dir")
    p.add_argument("--top-n", type=int, default=15, dest="top_n",
                   help="Number of most-divergent features to report per class/case (default: 15).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if len(args.windows) != len(args.labels):
        raise SystemExit("--windows and --labels must have the same length.")
    if args.baseline_csv and len(args.baseline_csv) != len(args.windows):
        raise SystemExit("--baseline-csv must match --windows in length if given.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (windows_arg, label) in enumerate(zip(args.windows, args.labels)):
        windows_path = Path(windows_arg)
        print(f"\n{'='*70}\n  {label}\n{'='*70}")

        windows = _load(windows_path)
        summary = case_summary(windows)
        summary.insert(0, "classifier", label)
        summary.to_csv(out_dir / f"{label.lower()}_case_summary.csv", index=False)
        print(summary.to_string(index=False))

        if args.baseline_csv:
            verify_against_baseline(summary, Path(args.baseline_csv[i]))

        feat_path = _features_path(windows_path)
        if not feat_path.exists():
            print(f"  [WARN] No feature dump found at {feat_path} - skipping FN/FP feature summary.")
            continue
        features = _load(feat_path)

        for bt in sorted(windows["bottleneck_type"].unique()):
            sev_profile = severity_score_profile(windows, bt)
            fn_div = feature_divergence(features, bt, "FN", "TP", args.top_n)
            fp_div = feature_divergence(features, bt, "FP", "TN", args.top_n)

            fn_div.to_csv(out_dir / f"{label.lower()}_{bt}_fn_vs_tp_features.csv", index=False)
            fp_div.to_csv(out_dir / f"{label.lower()}_{bt}_fp_vs_tn_features.csv", index=False)

            print(
                f"\n  {bt}: n_fn={sev_profile['n_fn']} "
                f"(median heuristic severity={sev_profile['fn_severity_median']:.3f}), "
                f"n_fp={sev_profile['n_fp']} "
                f"(median ML score={sev_profile['fp_score_median']:.3f})"
            )

    print(f"\n[INFO] All summaries saved under: {out_dir}")
