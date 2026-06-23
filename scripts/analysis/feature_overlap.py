"""
Feature Overlap Analysis — Hardware Counter Importance Comparison

Compares which hardware counters (metric columns) each backend relies on.
Since DefaultBackend uses tsfresh names ({metric}__stat) and AMLLibrary uses
its own names ({metric}_stat), direct feature name comparison is fragile.
Instead, both backends' feature importances are aggregated at the metric column
level (summing importances across all stats of a given metric). The top-N
metric columns per class are then compared via Jaccard similarity.

Usage
-----
    python scripts/analysis/feature_overlap.py \\
        --default-model models/xgboost.pkl \\
        --aml-model models/amllibrary.pkl \\
        --top-n 10 \\
        --output results/feature_overlap.png

Output
------
    Per-class Jaccard similarity table + optional heatmap.
    CSV with per-class, per-metric aggregate importances for both backends.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import DefaultBackend
from hpc_bottleneck_detector.ml.backends.amllibrary_backend import (
    AMLLibraryBackend,
    _ensure_aml_on_path,
)


# ---------------------------------------------------------------------------
# Feature extraction from DefaultBackend
# ---------------------------------------------------------------------------

def _default_metric_importances(backend: DefaultBackend) -> dict[str, pd.Series]:
    """
    Per bottleneck type: aggregate feature importances by metric column.
    tsfresh feature names are '{metric}__{stat}'; split on '__' to get the metric.
    """
    result = {}
    for bt, clf in backend._models.items():
        if not hasattr(clf, "feature_importances_"):
            continue
        feat_cols = backend._feature_cols[bt]
        importances = clf.feature_importances_

        metric_imp: dict[str, float] = {}
        for feat, imp in zip(feat_cols, importances):
            metric = feat.split("__")[0]
            metric_imp[metric] = metric_imp.get(metric, 0.0) + imp

        total = sum(metric_imp.values())
        if total > 0:
            metric_imp = {k: v / total for k, v in metric_imp.items()}

        result[bt] = pd.Series(metric_imp, name=bt).sort_values(ascending=False)

    return result


# ---------------------------------------------------------------------------
# Feature extraction from AMLLibraryBackend
# ---------------------------------------------------------------------------

def _aml_metric_importances(backend: AMLLibraryBackend) -> dict[str, pd.Series]:
    """
    Per bottleneck type: aggregate feature importances by metric column.
    aMLLibrary feature names are '{metric}_{stat}'; extract metric by stripping
    known stat suffixes.
    """
    _ensure_aml_on_path()

    _KNOWN_STATS = [
        "_minimum", "_maximum", "_mean", "_standard_deviation",
        "_skewness", "_kurtosis", "_absolute_sum_of_changes",
        "_quantile_", "_autocorrelation_", "_agg_linear_trend_",
        "_linear_trend_",
    ]

    def _extract_metric(feat_name: str) -> str:
        for stat in sorted(_KNOWN_STATS, key=len, reverse=True):
            idx = feat_name.find(stat)
            if idx > 0:
                return feat_name[:idx]
        return feat_name

    result = {}
    for bt, reg in backend._regressors.items():
        # Access the underlying sklearn estimator inside the aMLLibrary wrapper
        estimator = getattr(reg, "_best_regressor", None) or getattr(reg, "regressor", None)
        if estimator is None:
            # Try traversing the regressor chain
            for attr in ["_regressor", "best_estimator_", "_model"]:
                estimator = getattr(reg, attr, None)
                if estimator is not None:
                    break

        if estimator is None or not hasattr(estimator, "feature_importances_"):
            continue

        feat_names = getattr(reg, "feature_names_in_", None) or getattr(reg, "_feature_names", None)
        if feat_names is None:
            continue

        importances = estimator.feature_importances_
        if len(feat_names) != len(importances):
            continue

        metric_imp: dict[str, float] = {}
        for feat, imp in zip(feat_names, importances):
            metric = _extract_metric(feat)
            metric_imp[metric] = metric_imp.get(metric, 0.0) + imp

        total = sum(metric_imp.values())
        if total > 0:
            metric_imp = {k: v / total for k, v in metric_imp.items()}

        result[bt] = pd.Series(metric_imp, name=bt).sort_values(ascending=False)

    return result


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    inter = set_a & set_b
    return len(inter) / len(union)


def overlap_table(
    imp_default: dict[str, pd.Series],
    imp_aml: dict[str, pd.Series],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Per-class Jaccard similarity on top-N metric columns.
    Also reports the top-N sets for each backend.
    """
    bt_types = sorted(set(imp_default) | set(imp_aml))
    rows = []
    for bt in bt_types:
        s_def = set(imp_default[bt].head(top_n).index) if bt in imp_default else set()
        s_aml = set(imp_aml[bt].head(top_n).index) if bt in imp_aml else set()
        j = jaccard(s_def, s_aml)
        rows.append({
            "bottleneck_type": bt,
            "jaccard_top{top_n}".replace("{top_n}", str(top_n)): j,
            "n_default": len(s_def),
            "n_aml": len(s_aml),
            "intersection": ", ".join(sorted(s_def & s_aml)),
            "only_default": ", ".join(sorted(s_def - s_aml)),
            "only_aml": ", ".join(sorted(s_aml - s_def)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overlap_heatmap(
    imp_default: dict[str, pd.Series],
    imp_aml: dict[str, pd.Series],
    top_n: int,
    output_path: str | None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available - skipping plot.")
        return

    bt_types = sorted(set(imp_default) | set(imp_aml))
    all_metrics = sorted(
        set().union(*[set(s.head(top_n).index) for s in imp_default.values()])
        | set().union(*[set(s.head(top_n).index) for s in imp_aml.values()])
    )

    n_bt = len(bt_types)
    n_met = len(all_metrics)
    fig, axes = plt.subplots(1, 2, figsize=(max(12, n_met * 0.4), max(6, n_bt * 0.6) + 1))

    for ax, (imp_dict, label) in zip(axes, [(imp_default, "DefaultBackend"), (imp_aml, "AMLLibrary")]):
        mat = np.zeros((n_bt, n_met))
        for i, bt in enumerate(bt_types):
            if bt not in imp_dict:
                continue
            for j, met in enumerate(all_metrics):
                mat[i, j] = imp_dict[bt].get(met, 0.0)

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(n_met))
        ax.set_xticklabels(all_metrics, rotation=90, fontsize=7)
        ax.set_yticks(range(n_bt))
        ax.set_yticklabels([bt.replace("_", " ") for bt in bt_types], fontsize=8)
        ax.set_title(label, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Aggregated importance")

    fig.suptitle(f"Hardware Counter Importance (top {top_n} per class)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[INFO] Feature overlap heatmap saved to: {out}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare hardware counter importances between DefaultBackend and AMLLibrary."
    )
    p.add_argument("--default-model", required=True, dest="default_model",
                   help="Path to trained DefaultBackend .pkl.")
    p.add_argument("--aml-model", required=True, dest="aml_model",
                   help="Path to trained AMLLibraryBackend .pkl.")
    p.add_argument("--top-n", type=int, default=10, dest="top_n",
                   help="Top-N metric columns to include in Jaccard similarity (default: 10).")
    p.add_argument("--output", default=None,
                   help="Output path for heatmap image.")
    p.add_argument("--output-csv", default=None, dest="output_csv",
                   help="Optional path to save Jaccard overlap table as CSV.")
    p.add_argument("--output-importances-csv", default=None, dest="output_importances_csv",
                   help="Optional path to save per-metric aggregate importances.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    default_backend = DefaultBackend.load(args.default_model)
    aml_backend = AMLLibraryBackend.load(args.aml_model)

    imp_default = _default_metric_importances(default_backend)
    imp_aml = _aml_metric_importances(aml_backend)

    if not imp_default:
        print("[WARN] No feature importances found in DefaultBackend (model may not have feature_importances_).")
    if not imp_aml:
        print("[WARN] No feature importances found in AMLLibrary backend.")

    # Print Jaccard table
    table = overlap_table(imp_default, imp_aml, args.top_n)
    jcol = [c for c in table.columns if c.startswith("jaccard")][0]

    print(f"\n{'='*70}")
    print(f"  HARDWARE COUNTER OVERLAP (top {args.top_n} per class, Jaccard similarity)")
    print(f"{'='*70}\n")
    print(f"  {'Bottleneck Type':<42}  {'Jaccard':>8}  {'Intersection'}")
    print("  " + "-" * 70)
    for _, r in table.iterrows():
        print(
            f"  {r['bottleneck_type']:<42}  "
            f"{r[jcol]:>8.3f}  "
            f"{r['intersection']}"
        )

    macro_j = table[jcol].mean()
    print("  " + "-" * 70)
    print(f"  {'MACRO AVERAGE':<42}  {macro_j:>8.3f}")
    print()

    # Per-class top features
    bt_types = sorted(set(imp_default) | set(imp_aml))
    for bt in bt_types:
        print(f"  {bt}")
        def_top = list(imp_default[bt].head(args.top_n).index) if bt in imp_default else []
        aml_top = list(imp_aml[bt].head(args.top_n).index) if bt in imp_aml else []
        max_len = max(len(def_top), len(aml_top), 1)
        print(f"    {'DefaultBackend':<36}  {'AMLLibrary'}")
        for i in range(min(max_len, args.top_n)):
            d = def_top[i] if i < len(def_top) else ""
            a = aml_top[i] if i < len(aml_top) else ""
            marker = " <--" if d and d in set(aml_top) else ""
            print(f"    {d:<36}  {a}{marker}")
        print()

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        print(f"[INFO] Jaccard table saved to: {out}")

    if args.output_importances_csv:
        rows = []
        for bt in bt_types:
            if bt in imp_default:
                for metric, imp in imp_default[bt].items():
                    rows.append({"backend": "default", "bottleneck_type": bt, "metric": metric, "importance": imp})
            if bt in imp_aml:
                for metric, imp in imp_aml[bt].items():
                    rows.append({"backend": "amllibrary", "bottleneck_type": bt, "metric": metric, "importance": imp})
        out = Path(args.output_importances_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[INFO] Importances saved to: {out}")

    if imp_default or imp_aml:
        plot_overlap_heatmap(imp_default, imp_aml, args.top_n, args.output)
