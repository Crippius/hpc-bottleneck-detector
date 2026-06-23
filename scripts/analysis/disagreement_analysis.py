"""
Disagreement Analysis — Per-Window Agreement Between Two Backends

Compares per-window binary predictions from DefaultBackend and AMLLibraryBackend
to determine:
  - Overall agreement fraction (both correct, both wrong)
  - Error overlap: errors unique to Default, unique to AML, shared
  - Complementarity: fraction of total windows where an oracle ensemble
    (correct if at least one is correct) would be right
  - Cohen's kappa inter-rater agreement

Alignment: windows are matched by (fold, app, bottleneck_type) group, position-
within-group. Both backends must use the same windowing parameters so windows
within a group are in the same order. Groups with mismatched row counts are
skipped with a warning.

Input format (--scores-a, --scores-b)
--------------------------------------
CSV or Parquet with columns: fold, app, bottleneck_type, y_true, score

This is the format produced by loo_cross_validation.py --save-scores.

Usage
-----
    python scripts/analysis/disagreement_analysis.py \\
        --scores-a results/loo_scores_xgboost.parquet \\
        --scores-b results/loo_scores_amllibrary.parquet \\
        --label-a "XGBoost (Default)" \\
        --label-b "AMLLibrary" \\
        --output results/disagreement.png \\
        --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Alignment + binarisation
# ---------------------------------------------------------------------------

def _load(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def _binarise(
    df: pd.DataFrame,
    threshold: float,
    per_class_thresholds: dict[str, float] | None,
) -> pd.DataFrame:
    df = df.copy()
    thresholds = per_class_thresholds or {}
    thr = df["bottleneck_type"].map(lambda bt: thresholds.get(bt, threshold))
    df["y_pred"] = (df["score"] >= thr).astype(int)
    return df


def align(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge A and B on (fold, app, bottleneck_type), aligning windows by position.
    Returns a DataFrame with y_true, y_pred_a, y_pred_b, score_a, score_b.
    Groups with mismatched lengths are skipped with a warning.
    """
    key_cols = ["fold", "app", "bottleneck_type"]
    merged_rows = []
    skipped = 0
    total_groups = 0

    for keys, grp_a in df_a.groupby(key_cols, sort=False):
        total_groups += 1
        grp_b = df_b
        for col, val in zip(key_cols, keys):
            grp_b = grp_b[grp_b[col] == val]

        if len(grp_a) != len(grp_b):
            skipped += 1
            continue

        y_true  = grp_a["y_true"].values.astype(int)
        pred_a  = grp_a["y_pred"].values.astype(int)
        pred_b  = grp_b["y_pred"].values.astype(int)
        score_a = grp_a["score"].values.astype(float)
        score_b = grp_b["score"].values.astype(float)

        for i in range(len(y_true)):
            row = dict(zip(key_cols, keys))
            row.update({
                "y_true":  y_true[i],
                "pred_a":  pred_a[i],
                "pred_b":  pred_b[i],
                "score_a": score_a[i],
                "score_b": score_b[i],
            })
            merged_rows.append(row)

    if skipped:
        print(f"[WARN] Skipped {skipped}/{total_groups} groups due to row count mismatch.")

    return pd.DataFrame(merged_rows)


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for binary inter-rater agreement."""
    n = len(a)
    if n == 0:
        return float("nan")
    p_obs = float(np.mean(a == b))
    p_a = float(np.mean(a))
    p_b = float(np.mean(b))
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    if p_e >= 1.0:
        return 1.0
    return (p_obs - p_e) / (1.0 - p_e)


def _quad_counts(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, int]:
    """
    Four outcome categories for each window:
      - both_correct:    both right
      - both_wrong:      both wrong (shared errors)
      - only_a_wrong:    A wrong, B right
      - only_b_wrong:    A right, B wrong
    """
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    return {
        "both_correct":  int((correct_a & correct_b).sum()),
        "both_wrong":    int((~correct_a & ~correct_b).sum()),
        "only_a_wrong":  int((~correct_a & correct_b).sum()),
        "only_b_wrong":  int((correct_a & ~correct_b).sum()),
    }


def _oracle_accuracy(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Fraction correct if an oracle ensemble picks the right one (union of correct predictions)."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    return float((correct_a | correct_b).sum() / n)


def disagreement_summary(
    df: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    """Per-class breakdown of agreement, error overlap, and complementarity."""
    rows = []
    for bt, grp in df.groupby("bottleneck_type"):
        y_true  = grp["y_true"].values
        pred_a  = grp["pred_a"].values
        pred_b  = grp["pred_b"].values
        n = len(y_true)

        quad = _quad_counts(y_true, pred_a, pred_b)
        agree = (quad["both_correct"] + quad["both_wrong"]) / n
        oracle = _oracle_accuracy(y_true, pred_a, pred_b)
        kappa = _cohens_kappa(pred_a, pred_b)
        shared_err_frac = (
            quad["both_wrong"] / (quad["both_wrong"] + quad["only_a_wrong"] + quad["only_b_wrong"])
            if (quad["both_wrong"] + quad["only_a_wrong"] + quad["only_b_wrong"]) > 0
            else float("nan")
        )

        rows.append({
            "bottleneck_type": bt,
            "n_windows": n,
            "agreement": agree,
            "kappa": kappa,
            "oracle_accuracy": oracle,
            "both_correct": quad["both_correct"],
            "both_wrong": quad["both_wrong"],
            f"only_{label_a.split()[0].lower()}_wrong": quad["only_a_wrong"],
            f"only_{label_b.split()[0].lower()}_wrong": quad["only_b_wrong"],
            "shared_error_frac": shared_err_frac,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_report(
    summary: pd.DataFrame,
    label_a: str,
    label_b: str,
    df_aligned: pd.DataFrame,
) -> None:
    n_total = len(df_aligned)
    y_true  = df_aligned["y_true"].values
    pred_a  = df_aligned["pred_a"].values
    pred_b  = df_aligned["pred_b"].values

    global_quad = _quad_counts(y_true, pred_a, pred_b)
    global_agree = (global_quad["both_correct"] + global_quad["both_wrong"]) / n_total
    global_oracle = _oracle_accuracy(y_true, pred_a, pred_b)
    global_kappa = _cohens_kappa(pred_a, pred_b)

    a_lbl = label_a.split()[0].lower()
    b_lbl = label_b.split()[0].lower()

    print(f"\n{'='*70}")
    print(f"  DISAGREEMENT ANALYSIS:  {label_a}  vs.  {label_b}")
    print(f"{'='*70}")
    print(f"\n  Total windows analysed  : {n_total}")
    print(f"  Overall agreement       : {global_agree:.4f}")
    print(f"  Cohen's kappa           : {global_kappa:.4f}")
    print(f"  Oracle ensemble acc.    : {global_oracle:.4f}")
    print(f"\n  Error breakdown  (across all windows & classes)")
    print(f"  Both correct            : {global_quad['both_correct']:>6}  ({global_quad['both_correct']/n_total:.1%})")
    print(f"  Both wrong (shared)     : {global_quad['both_wrong']:>6}  ({global_quad['both_wrong']/n_total:.1%})")
    print(f"  Only {label_a:<20} wrong: {global_quad['only_a_wrong']:>6}  ({global_quad['only_a_wrong']/n_total:.1%})")
    print(f"  Only {label_b:<20} wrong: {global_quad['only_b_wrong']:>6}  ({global_quad['only_b_wrong']/n_total:.1%})")

    print(f"\n{'='*70}")
    print(f"  PER-CLASS BREAKDOWN")
    print(f"{'='*70}\n")
    print(f"  {'Bottleneck Type':<42}  {'Agree':>7}  {'Kappa':>7}  {'Oracle':>7}  {'Shared Err%':>11}")
    print("  " + "-" * 82)
    for _, r in summary.iterrows():
        se = r["shared_error_frac"]
        se_str = f"{se:.1%}" if not np.isnan(se) else "   N/A"
        print(
            f"  {r['bottleneck_type']:<42}  "
            f"{r['agreement']:>7.4f}  "
            f"{r['kappa']:>7.4f}  "
            f"{r['oracle_accuracy']:>7.4f}  "
            f"{se_str:>11}"
        )

    macro_agree = summary["agreement"].mean()
    macro_kappa = summary["kappa"].mean()
    macro_oracle = summary["oracle_accuracy"].mean()
    macro_se = summary["shared_error_frac"].dropna().mean()
    print("  " + "-" * 82)
    print(
        f"  {'MACRO AVERAGE':<42}  "
        f"{macro_agree:>7.4f}  "
        f"{macro_kappa:>7.4f}  "
        f"{macro_oracle:>7.4f}  "
        f"{macro_se:>10.1%}"
    )

    # Interpretation
    print(f"\n  Interpretation:")
    if macro_se < 0.33:
        print("  Shared errors are rare — backends are largely COMPLEMENTARY.")
        print("  Ensemble potential: high.")
    elif macro_se < 0.60:
        print("  Shared errors are moderate — partial complementarity.")
    else:
        print("  Shared errors dominate — backends have similar failure modes.")
        print("  Ensemble potential: low.")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_disagreement(
    summary: pd.DataFrame,
    label_a: str,
    label_b: str,
    output_path: str | None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available - skipping plot.")
        return

    bt_types = summary["bottleneck_type"].tolist()
    n = len(bt_types)
    a_col = f"only_{label_a.split()[0].lower()}_wrong"
    b_col = f"only_{label_b.split()[0].lower()}_wrong"

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n * 0.55 + 1)))

    # Left: stacked bar chart (error breakdown)
    ax = axes[0]
    total = summary["n_windows"].values
    both_c = summary["both_correct"].values / total
    both_w = summary["both_wrong"].values / total
    only_a = summary[a_col].values / total if a_col in summary.columns else np.zeros(n)
    only_b = summary[b_col].values / total if b_col in summary.columns else np.zeros(n)

    y_pos = np.arange(n)
    ax.barh(y_pos, both_c, color="#4CAF50", label="Both correct", height=0.6)
    ax.barh(y_pos, only_b, left=both_c, color="#2196F3", label=f"Only {label_b} wrong", height=0.6)
    ax.barh(y_pos, only_a, left=both_c + only_b, color="#FF9800", label=f"Only {label_a} wrong", height=0.6)
    ax.barh(y_pos, both_w, left=both_c + only_b + only_a, color="#F44336", label="Both wrong (shared)", height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([bt.replace("_", " ") for bt in bt_types], fontsize=8)
    ax.set_xlabel("Fraction of windows")
    ax.set_title("Error Breakdown per Class", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)

    # Right: agreement + oracle accuracy
    ax2 = axes[1]
    agree = summary["agreement"].values
    oracle = summary["oracle_accuracy"].values
    kappa = summary["kappa"].values

    ax2.barh(y_pos - 0.2, agree,  height=0.35, color="steelblue", label="Agreement")
    ax2.barh(y_pos + 0.2, oracle, height=0.35, color="darkorange", label="Oracle accuracy")
    for i, k in enumerate(kappa):
        if not np.isnan(k):
            ax2.text(0.02, i, f"κ={k:.2f}", va="center", fontsize=7, color="dimgray")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([bt.replace("_", " ") for bt in bt_types], fontsize=8)
    ax2.set_xlabel("Fraction of windows")
    ax2.set_title("Agreement & Oracle Accuracy per Class", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 1)
    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Disagreement Analysis: {label_a} vs. {label_b}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[INFO] Disagreement plot saved to: {out}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-window disagreement analysis between two backends' LOO predictions."
    )
    p.add_argument("--scores-a", required=True, dest="scores_a",
                   help="Scores file for backend A (fold,app,bottleneck_type,y_true,score).")
    p.add_argument("--scores-b", required=True, dest="scores_b",
                   help="Scores file for backend B.")
    p.add_argument("--label-a", default="BackendA", dest="label_a")
    p.add_argument("--label-b", default="BackendB", dest="label_b")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Score threshold for binarising predictions (default: 0.5).")
    p.add_argument(
        "--per-class-thresholds", default=None, dest="per_class_thresholds",
        help="JSON file or JSON string mapping bottleneck_type → threshold."
    )
    p.add_argument("--output", default=None,
                   help="Output path for disagreement plot image.")
    p.add_argument("--output-csv", default=None, dest="output_csv",
                   help="Optional path to save per-class summary as CSV.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Load per-class thresholds
    per_class: dict[str, float] | None = None
    if args.per_class_thresholds:
        raw = args.per_class_thresholds
        try:
            per_class = json.loads(raw)
        except json.JSONDecodeError:
            with open(raw) as f:
                per_class = json.load(f)

    df_a = _binarise(_load(args.scores_a), args.threshold, per_class)
    df_b = _binarise(_load(args.scores_b), args.threshold, per_class)

    df_aligned = align(df_a, df_b)

    if df_aligned.empty:
        print("[ERROR] No aligned windows found — check that both files share the same "
              "(fold, app, bottleneck_type) groups and window counts.")
        sys.exit(1)

    summary = disagreement_summary(df_aligned, args.label_a, args.label_b)
    print_report(summary, args.label_a, args.label_b, df_aligned)

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        print(f"[INFO] Per-class summary saved to: {out}")

    plot_disagreement(summary, args.label_a, args.label_b, args.output)
