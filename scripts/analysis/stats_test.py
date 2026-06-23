"""
Statistical significance testing for LOO CV result comparisons.

Compares two backends' per-app LOO F1 scores using a paired Wilcoxon
signed-rank test, Cohen's d effect size, and a bootstrap 95% CI on the
mean difference.

Usage
-----
    python scripts/analysis/stats_test.py \\
        --a results/loo_xgboost.csv \\
        --b results/loo_amllibrary.csv \\
        --metric f1

    python scripts/analysis/stats_test.py \\
        --a results/loo_xgboost.csv \\
        --b results/loo_amllibrary.csv \\
        --metric f1 --per-class
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """
    Paired Wilcoxon signed-rank test on two matched score vectors.
    Returns p-value and W statistic.
    """
    diff = a - b
    if np.all(diff == 0):
        return {"w_stat": 0.0, "p_value": 1.0}
    try:
        result = wilcoxon(a, b, alternative="two-sided")
        return {"w_stat": float(result.statistic), "p_value": float(result.pvalue)}
    except Exception:
        return {"w_stat": float("nan"), "p_value": float("nan")}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired differences (mean diff / std diff)."""
    diff = a - b
    sd = float(np.std(diff, ddof=1))
    if sd == 0:
        return float("nan")
    return float(np.mean(diff) / sd)


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI on the mean difference (a - b)."""
    rng = np.random.default_rng(seed)
    diffs = a - b
    means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _macro_per_app(df: pd.DataFrame, metric: str) -> pd.Series:
    """Mean of *metric* per app, macro-averaged across bottleneck types."""
    return df.groupby("app")[metric].mean()


def compare(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """
    Full statistical comparison of two LOO result DataFrames on *metric*.
    Returns a dict with mean, std, Wilcoxon, Cohen's d, bootstrap CI.
    """
    scores_a = _macro_per_app(df_a, metric).dropna()
    scores_b = _macro_per_app(df_b, metric).dropna()

    # Align on common apps
    common = scores_a.index.intersection(scores_b.index)
    if len(common) == 0:
        raise ValueError("No common apps between the two result files.")
    a = scores_a.loc[common].values
    b = scores_b.loc[common].values

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    std_a  = float(np.std(a, ddof=1))
    std_b  = float(np.std(b, ddof=1))
    mean_diff = mean_a - mean_b

    wtest = paired_wilcoxon(a, b)
    d = cohens_d(a, b)
    ci_lo, ci_hi = bootstrap_ci(a, b)

    return {
        "metric": metric,
        "n_apps": len(common),
        f"mean_{label_a}": mean_a,
        f"std_{label_a}": std_a,
        f"mean_{label_b}": mean_b,
        f"std_{label_b}": std_b,
        "mean_diff": mean_diff,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "w_stat": wtest["w_stat"],
        "p_value": wtest["p_value"],
        "cohens_d": d,
    }


def compare_per_class(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str,
    label_a: str = "A",
    label_b: str = "B",
) -> pd.DataFrame:
    """Per-bottleneck-type statistical comparison."""
    bt_types = df_a["bottleneck_type"].unique()
    rows = []
    for bt in sorted(bt_types):
        sub_a = df_a[df_a["bottleneck_type"] == bt]
        sub_b = df_b[df_b["bottleneck_type"] == bt]
        try:
            r = compare(sub_a, sub_b, metric, label_a, label_b)
            r["bottleneck_type"] = bt
            rows.append(r)
        except ValueError:
            pass
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_result(r: dict, label_a: str, label_b: str) -> None:
    p = r["p_value"]
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"\n  Metric          : {r['metric']}")
    print(f"  N apps          : {r['n_apps']}")
    print(f"  {label_a} mean ± std  : {r[f'mean_{label_a}']:.4f} ± {r[f'std_{label_a}']:.4f}")
    print(f"  {label_b} mean ± std  : {r[f'mean_{label_b}']:.4f} ± {r[f'std_{label_b}']:.4f}")
    print(f"  Mean diff (A-B) : {r['mean_diff']:+.4f}  95% CI [{r['ci_95_lo']:+.4f}, {r['ci_95_hi']:+.4f}]")
    print(f"  Wilcoxon W      : {r['w_stat']:.1f}  p={p:.4f}  {sig}")
    print(f"  Cohen's d       : {r['cohens_d']:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paired statistical significance test between two LOO result CSVs."
    )
    p.add_argument("--a", required=True, help="Path to first LOO results CSV (baseline).")
    p.add_argument("--b", required=True, help="Path to second LOO results CSV (comparison).")
    p.add_argument("--label-a", default="A", dest="label_a")
    p.add_argument("--label-b", default="B", dest="label_b")
    p.add_argument(
        "--metric", default="f1",
        help="Column name to compare (default: f1). Options: f1, false_alarm_rate, miss_rate, roc_auc, pr_auc, mcc.",
    )
    p.add_argument(
        "--per-class", action="store_true", dest="per_class",
        help="Also run per-bottleneck-type comparisons.",
    )
    p.add_argument("--output-csv", default=None, dest="output_csv",
                   help="Optional path to save the per-class results as CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    df_a = pd.read_csv(args.a)
    df_b = pd.read_csv(args.b)

    print(f"\n{'='*60}")
    print(f"  STATISTICAL COMPARISON: {args.label_a} vs {args.label_b}")
    print(f"  A: {args.a}")
    print(f"  B: {args.b}")
    print(f"{'='*60}")

    result = compare(df_a, df_b, args.metric, args.label_a, args.label_b)
    _print_result(result, args.label_a, args.label_b)

    if args.per_class:
        print(f"\n  PER-CLASS BREAKDOWN  (metric: {args.metric})")
        print(f"  {'Bottleneck Type':<42}  {'Diff':>7}  {'p':>8}  {'d':>7}  {'Sig'}")
        print("  " + "-" * 72)
        pc = compare_per_class(df_a, df_b, args.metric, args.label_a, args.label_b)
        for _, row in pc.iterrows():
            p = row["p_value"]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            print(
                f"  {row['bottleneck_type']:<42}  "
                f"{row['mean_diff']:>+7.4f}  "
                f"{p:>8.4f}  "
                f"{row['cohens_d']:>7.3f}  "
                f"{sig}"
            )

        if args.output_csv:
            out = Path(args.output_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            pc.to_csv(out, index=False)
            print(f"\n  Per-class results saved to: {out}")

    print()
