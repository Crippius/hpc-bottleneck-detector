"""
Calibration Quality Analysis

For each backend, collects predicted probability
scores across all LOO folds and plots reliability
diagrams per bottleneck type, plus computes per-class ECE.

Usage
-----
    # Collect predictions using a saved LOO scores file:
    python scripts/analysis/calibration.py \\
        --scores results/loo_scores_xgboost.parquet \\
        --label "XGBoost (Default)" \\
        --output results/calibration_xgboost.png

    # Compare two backends side by side:
    python scripts/analysis/calibration.py \\
        --scores results/loo_scores_xgboost.parquet \\
        --scores-b results/loo_scores_amllibrary.parquet \\
        --label "XGBoost" --label-b "AMLLibrary" \\
        --output results/calibration_compare.png

Input format
------------
The --scores file must be a CSV or Parquet with columns:
    bottleneck_type, y_true (0/1), score (predicted probability/severity)
One row per window-fold-class observation.

To generate this file from LOO CV, run loo_cross_validation.py with
--save-scores <path>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) — weighted mean absolute calibration gap."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(scores[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_data(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability diagram data.

    Returns (bin_centers, mean_predicted, fraction_positive) per non-empty bin.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, pred_means, frac_pos = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        pred_means.append(float(scores[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
    return np.array(centers), np.array(pred_means), np.array(frac_pos)


# ---------------------------------------------------------------------------
# Per-class calibration summary
# ---------------------------------------------------------------------------

def calibration_summary(
    df: pd.DataFrame,
    label: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute ECE and sample counts per bottleneck type."""
    rows = []
    for bt, grp in df.groupby("bottleneck_type"):
        y = grp["y_true"].values.astype(int)
        s = grp["score"].values.astype(float)
        if len(y) < 2 or len(set(y)) < 2:
            continue
        ece = expected_calibration_error(y, s, n_bins)
        rows.append({
            "backend": label,
            "bottleneck_type": bt,
            "n_windows": len(y),
            "n_positive": int(y.sum()),
            "ece": ece,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_reliability(
    ax,
    y_true: np.ndarray,
    scores: np.ndarray,
    label: str,
    color: str,
    n_bins: int = 10,
) -> None:
    centers, pred_means, frac_pos = reliability_data(y_true, scores, n_bins)
    ece = expected_calibration_error(y_true, scores, n_bins)
    ax.plot(pred_means, frac_pos, "o-", color=color, linewidth=1.5,
            markersize=5, label=f"{label} (ECE={ece:.3f})")
    ax.fill_between(pred_means, frac_pos, pred_means, alpha=0.15, color=color)


def plot_reliability_diagrams(
    df_a: pd.DataFrame,
    label_a: str,
    df_b: pd.DataFrame | None,
    label_b: str | None,
    output_path: str | None,
    n_bins: int = 10,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available - skipping plot.")
        return

    bt_types = sorted(df_a["bottleneck_type"].unique())
    n_types = len(bt_types)
    ncols = min(3, n_types)
    nrows = (n_types + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("Reliability Diagrams (Calibration)", fontsize=13, fontweight="bold")

    for idx, bt in enumerate(bt_types):
        ax = axes[idx // ncols][idx % ncols]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")

        grp_a = df_a[df_a["bottleneck_type"] == bt]
        if len(grp_a) >= 2 and grp_a["y_true"].nunique() >= 2:
            _plot_reliability(ax, grp_a["y_true"].values, grp_a["score"].values,
                              label_a, "steelblue", n_bins)

        if df_b is not None and label_b is not None:
            grp_b = df_b[df_b["bottleneck_type"] == bt]
            if len(grp_b) >= 2 and grp_b["y_true"].nunique() >= 2:
                _plot_reliability(ax, grp_b["y_true"].values, grp_b["score"].values,
                                  label_b, "darkorange", n_bins)

        ax.set_title(bt.replace("_", " "), fontsize=9)
        ax.set_xlabel("Mean predicted score")
        ax.set_ylabel("Fraction positive")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_types, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[INFO] Reliability diagram saved to: {out}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_scores(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot reliability diagrams and compute ECE for one or two backends."
    )
    p.add_argument("--scores", required=True,
                   help="CSV/Parquet with columns: bottleneck_type, y_true, score.")
    p.add_argument("--scores-b", default=None, dest="scores_b",
                   help="Optional second scores file for comparison.")
    p.add_argument("--label", default="Backend A", help="Label for first backend.")
    p.add_argument("--label-b", default="Backend B", dest="label_b",
                   help="Label for second backend.")
    p.add_argument("--output", default=None,
                   help="Output path for the reliability diagram image.")
    p.add_argument("--n-bins", type=int, default=10, dest="n_bins",
                   help="Number of calibration bins (default: 10).")
    p.add_argument("--output-summary-csv", default=None, dest="output_summary_csv",
                   help="Optional path to save per-class ECE summary as CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    df_a = _load_scores(args.scores)
    df_b = _load_scores(args.scores_b) if args.scores_b else None

    # Print ECE summary
    print(f"\n{'='*60}")
    print(f"  CALIBRATION SUMMARY — ECE per class")
    print(f"{'='*60}")

    summary_a = calibration_summary(df_a, args.label, args.n_bins)
    print(f"\n  {args.label}")
    print(f"  {'Bottleneck Type':<42}  {'ECE':>7}  {'N':>6}  {'Pos':>6}")
    print("  " + "-" * 62)
    for _, r in summary_a.iterrows():
        print(f"  {r['bottleneck_type']:<42}  {r['ece']:>7.4f}  {r['n_windows']:>6}  {r['n_positive']:>6}")
    print(f"  {'MACRO AVERAGE':<42}  {summary_a['ece'].mean():>7.4f}")

    all_summaries = [summary_a]

    if df_b is not None:
        summary_b = calibration_summary(df_b, args.label_b, args.n_bins)
        print(f"\n  {args.label_b}")
        print(f"  {'Bottleneck Type':<42}  {'ECE':>7}  {'N':>6}  {'Pos':>6}")
        print("  " + "-" * 62)
        for _, r in summary_b.iterrows():
            print(f"  {r['bottleneck_type']:<42}  {r['ece']:>7.4f}  {r['n_windows']:>6}  {r['n_positive']:>6}")
        print(f"  {'MACRO AVERAGE':<42}  {summary_b['ece'].mean():>7.4f}")
        all_summaries.append(summary_b)

    if args.output_summary_csv:
        out = Path(args.output_summary_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_summaries).to_csv(out, index=False)
        print(f"\n[INFO] ECE summary saved to: {out}")

    plot_reliability_diagrams(
        df_a=df_a, label_a=args.label,
        df_b=df_b, label_b=args.label_b if df_b is not None else None,
        output_path=args.output,
        n_bins=args.n_bins,
    )
