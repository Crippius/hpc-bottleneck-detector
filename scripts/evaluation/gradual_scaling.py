"""
Gradual Scaling Hold-Out

Proves how much application diversity the model needs before it generalises
to entirely unseen code.

Usage
-----
    python examples/ml/gradual_scaling.py
    python examples/ml/gradual_scaling.py --steps 2 4 6 8 10 --test-size 3
    python examples/ml/gradual_scaling.py --output-fig results/learning_curve.png \\
                                          --output-csv results/lc_results.csv
    python examples/ml/gradual_scaling.py --no-plot
"""

from __future__ import annotations

import argparse
import itertools
import logging
import random
import sys
import warnings
from pathlib import Path

import math

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hpc_bottleneck_detector.ml.backends.config import build_classifier
from hpc_bottleneck_detector.ml.backends.default_backend import (
    _LABEL_COLS,
    _NON_METRIC_COLS,
    _build_window_dataframe,
    _fill_metric_nans,
    _window_labels,
    BASIC_FC_PARAMETERS,
)
from hpc_bottleneck_detector.ml.backends.default_trainer import DefaultTrainer
from hpc_bottleneck_detector.ml.feature_extraction import (
    find_labelled_csvs as _find_labelled_csvs,
    extract_features_for_app as _extract_features_for_app,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT  = Path(__file__).parent.parent.parent
TRAIN_DIR  = REPO_ROOT / "data" / "labelled_data" / "training_set"

_BT_SHORT: dict[str, str] = {
    "PIPELINE_STALL":            "Pipeline Stall",
    "COMPUTE_UNDERUTILIZATION":  "Compute Underutil.",
    "PRECISION_WASTE":           "Precision Waste",
    "BRANCH_MISPREDICTION":      "Branch Mispredict.",
    "CACHE_PRESSURE":            "Cache Pressure",
    "INTRA_NODE_LOAD_IMBALANCE": "Intra-Node Imbal.",
    "INTER_NODE_LOAD_IMBALANCE": "Inter-Node Imbal.",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_windows(csv_path: Path, window_size: int, step_size: int) -> int:
    df = pd.read_csv(csv_path, usecols=["id", "time"])
    total = 0
    for _, job_df in df.groupby("id"):
        n = len(job_df)
        start = 0
        while start < n:
            total += 1
            end = min(start + window_size, n)
            if end == n:
                break
            start += step_size
    return total


def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (not np.isnan(precision) and not np.isnan(recall)
            and (precision + recall) > 0)
        else float("nan")
    )
    far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    mr  = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    return {"f1": f1, "false_alarm_rate": far, "miss_rate": mr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------------------------------------------------------------------------
# Per-combo CV threshold calibration
# ---------------------------------------------------------------------------

def _merge_y_dicts(
    y_dicts: list[dict[str, pd.Series]],
) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for col in _LABEL_COLS:
        parts = [yd[col] for yd in y_dicts if col in yd]
        if parts:
            result[col] = pd.concat(parts)
    return result


def cv_calibrate_thresholds(
    combo: tuple[int, ...],
    app_features: list[tuple[pd.DataFrame, dict[str, pd.Series]]],
    classifier,
    n_splits: int = 5,
    default_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Calibrate per-class probability thresholds for a combo of *k* apps using
    GroupKFold(min(n_splits, k)) CV.

    For k >= n_splits this is standard n_splits-fold CV; for k < n_splits it
    degrades gracefully to leave-one-app-out (k folds).
    """
    combo_apps = [app_features[i] for i in combo]
    k = len(combo)
    k_folds = min(n_splits, k)
    app_idx = np.arange(k)

    gkf = GroupKFold(n_splits=k_folds)
    per_class: dict[str, list[float]] = {col: [] for col in _LABEL_COLS}

    for train_idx, val_idx in gkf.split(app_idx, groups=app_idx):
        X_train = pd.concat([combo_apps[i][0] for i in train_idx]).fillna(0.0)
        y_dict_train = _merge_y_dicts([combo_apps[i][1] for i in train_idx])
        X_val = pd.concat([combo_apps[i][0] for i in val_idx]).fillna(0.0)
        y_dict_val = _merge_y_dicts([combo_apps[i][1] for i in val_idx])

        backend_fold = DefaultTrainer(classifier=classifier).from_preextracted_features(
            X_train, y_dict_train
        )
        if not backend_fold._models:
            continue
        backend_fold.calibrate_thresholds(X_val, y_dict_val)

        for col in _LABEL_COLS:
            thr = backend_fold._thresholds.get(col)
            if thr is not None:
                per_class[col].append(thr)

    result: dict[str, float] = {}
    for col in _LABEL_COLS:
        vals = per_class[col]
        if vals:
            avg = float(np.nanmean(vals))
            result[col] = avg if not math.isnan(avg) else default_threshold
        else:
            result[col] = default_threshold
    return result


# ---------------------------------------------------------------------------
# Exhaustive gradual scaling driver
# ---------------------------------------------------------------------------

def run_gradual_scaling(
    app_features: list[tuple[pd.DataFrame, dict[str, pd.Series]]],
    all_paths: list[Path],
    steps: list[int],
    prob_threshold: float,
    classifier,
    max_combos: int = 50,
    seed: int = 42,
    use_cv: bool = True,
    cv_min_k: int = 3,
) -> pd.DataFrame:
    """
    For each step k, iterate over subsets of k apps, train on each, evaluate on
    the remaining (n-k) apps as a dynamic test set, and record every result.
    When C(n, k) <= max_combos the enumeration is exhaustive; otherwise
    max_combos random subsets are drawn.
    """
    rng = random.Random(seed)
    records: list[dict] = []
    n_pool = len(app_features)

    for k in steps:
        if k >= n_pool:
            logger.warning("Step k=%d leaves no test apps (pool=%d) - skipping.", k, n_pool)
            continue

        all_combos = list(itertools.combinations(range(n_pool), k))
        exhaustive = len(all_combos) <= max_combos
        if exhaustive:
            combos = all_combos
            mode_str = "exhaustive"
        else:
            combos = rng.sample(all_combos, max_combos)
            mode_str = f"sampled {max_combos} of {len(all_combos)}"

        print(f"\n{'─'*65}")
        print(f"  Step k={k:>2}  |  {len(combos)} combinations  ({mode_str})")
        print(f"{'─'*65}")

        f1_by_col: dict[str, list[float]] = {}

        for combo_idx, combo in enumerate(combos):
            combo_set = set(combo)
            app_names = [all_paths[i].stem.replace("_labelled", "") for i in combo]

            X_train = pd.concat([app_features[i][0] for i in combo]).fillna(0.0)
            y_dict_train: dict[str, pd.Series] = {}
            for col in _LABEL_COLS:
                parts = [app_features[i][1][col] for i in combo
                         if col in app_features[i][1]]
                if parts:
                    y_dict_train[col] = pd.concat(parts)

            # Dynamic test set: all apps not in this combo
            test_indices = [i for i in range(n_pool) if i not in combo_set]
            X_test = pd.concat([app_features[i][0] for i in test_indices]).fillna(0.0)
            y_dict_test: dict[str, pd.Series] = {}
            for col in _LABEL_COLS:
                parts = [app_features[i][1][col] for i in test_indices
                         if col in app_features[i][1]]
                if parts:
                    y_dict_test[col] = pd.concat(parts)

            # Threshold calibration
            if use_cv and k >= cv_min_k:
                cv_thresholds = cv_calibrate_thresholds(
                    combo, app_features, classifier,
                    default_threshold=prob_threshold,
                )
                cv_enabled = True
            else:
                if use_cv and k < cv_min_k:
                    logger.warning(
                        "k=%d < cv_min_k=%d - skipping CV, using prob_threshold=%.2f",
                        k, cv_min_k, prob_threshold,
                    )
                cv_thresholds = {col: prob_threshold for col in _LABEL_COLS}
                cv_enabled = False

            backend = DefaultTrainer(classifier=classifier).from_preextracted_features(
                X_train, y_dict_train
            )
            if not backend._models:
                continue
            backend._thresholds = cv_thresholds

            for col, clf in backend._models.items():
                if col not in y_dict_test:
                    continue
                fc        = backend._feature_cols[col]
                valid_idx = y_dict_test[col].index
                X_aligned = X_test.reindex(index=valid_idx, columns=fc, fill_value=0.0)
                probs     = clf.predict_proba(X_aligned)[:, 1]
                thr       = backend._thresholds.get(col, prob_threshold)
                y_pred    = (probs >= thr).astype(int)
                m         = _metrics_from_arrays(y_dict_test[col].values, y_pred)

                f1_by_col.setdefault(col, []).append(m["f1"])
                records.append({
                    "n_train_apps":    k,
                    "combo_idx":       combo_idx,
                    "train_apps":      str(app_names),
                    "bottleneck_type": col,
                    "threshold_used":  thr,
                    "cv_enabled":      cv_enabled,
                    **m,
                })

        # Per-step summary
        all_avg_f1: list[float] = []
        for col in sorted(f1_by_col):
            vals      = [v for v in f1_by_col[col] if not np.isnan(v)]
            avg       = float(np.mean(vals)) if vals else float("nan")
            std       = float(np.std(vals))  if vals else float("nan")
            all_avg_f1.append(avg)
            print(
                f"    {col:<42}  "
                f"avg F1={avg:.3f}  std={std:.3f}  "
                f"(n={len(vals)} combos)"
            )
        macro = float(np.nanmean(all_avg_f1)) if all_avg_f1 else float("nan")
        print(f"    {'[MACRO AVG F1]':<42}  {macro:.3f}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    results: pd.DataFrame,
    output_path: str | None,
    show: bool,
) -> None:
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[WARN] matplotlib not available - skipping plot.")
        return

    # Average over all combinations at each (n_train_apps, bottleneck_type)
    summary = (
        results.groupby(["n_train_apps", "bottleneck_type"])["f1"]
        .mean()
        .reset_index()
    )

    bt_types  = sorted(summary["bottleneck_type"].unique())
    x_vals    = sorted(summary["n_train_apps"].unique())
    bt_colors = cm.tab10(np.linspace(0, 0.9, len(bt_types)))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Gradual Scaling Hold-Out  (exhaustive combinations, averaged)",
        fontsize=13, fontweight="bold",
    )

    for bt, color in zip(bt_types, bt_colors):
        bt_df = summary[summary["bottleneck_type"] == bt].set_index("n_train_apps")
        y = [bt_df.loc[x, "f1"] if x in bt_df.index else float("nan") for x in x_vals]
        ax.plot(
            x_vals, y,
            color=color, linewidth=1.4, linestyle="--", alpha=0.7, marker="s",
            markersize=4, label=_BT_SHORT.get(bt, bt),
        )

    # Per-combo macro F1 = mean F1 across bt types for that combo
    combo_macro = (
        results.groupby(["n_train_apps", "combo_idx"])["f1"]
        .mean()
        .reset_index()
        .rename(columns={"f1": "macro_f1"})
    )
    macro_stats = (
        combo_macro.groupby("n_train_apps")["macro_f1"]
        .agg(["mean", "std"])
        .reindex(x_vals)
    )
    macro_f1  = macro_stats["mean"].tolist()
    macro_std = macro_stats["std"].fillna(0).tolist()

    macro_lo = [max(0.0, m - s) for m, s in zip(macro_f1, macro_std)]
    macro_hi = [min(1.0, m + s) for m, s in zip(macro_f1, macro_std)]

    ax.fill_between(
        x_vals, macro_lo, macro_hi,
        color="black", alpha=0.12, zorder=5, label="Macro ± 1 std",
    )
    ax.plot(
        x_vals, macro_f1,
        color="black", linewidth=3.0, linestyle="-", marker="o", markersize=7,
        label="Macro average", zorder=6,
    )

    for x, y in zip(x_vals, macro_f1):
        if not np.isnan(y):
            ax.annotate(
                f"{y:.2f}",
                xy=(x, y), xytext=(0, 10),
                textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold", color="black",
            )

    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{k} apps" for k in x_vals], fontsize=9)
    ax.set_xlabel("Number of Training Applications", fontsize=11)
    ax.set_ylabel("F1-Score (avg over all subsets)", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.88, ncol=2)
    ax.set_title("F1-Score per Bottleneck Type", fontsize=11, pad=6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n[INFO] Figure saved to: {out}")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gradual Scaling Hold-Out: F1 vs. number of training apps "
                    "(exhaustive subset enumeration)."
    )
    p.add_argument(
        "--steps", nargs="+", type=int, default=[2, 4, 6, 8, 10],
        help="Training set sizes to evaluate (default: 2 4 6 8 10).",
    )
    p.add_argument("--window-size",        type=int,   default=12,  dest="window_size")
    p.add_argument("--step-size",          type=int,   default=12,  dest="step_size")
    p.add_argument("--severity-threshold", type=float, default=0.0, dest="severity_threshold")
    p.add_argument("--prob-threshold",     type=float, default=0.5, dest="prob_threshold")
    p.add_argument("--output-csv",         type=str,   default=None, dest="output_csv")
    p.add_argument("--output-pkl",         type=str,   default=None, dest="output_pkl")
    p.add_argument("--output-fig",         type=str,   default=None, dest="output_fig")
    p.add_argument("--no-plot",            action="store_true",     dest="no_plot")
    p.add_argument(
        "--max-combos", type=int, default=50, dest="max_combos",
        help="Max random subsets per step when C(n,k) exceeds this (default: 50).",
    )
    p.add_argument(
        "--no-cv", action="store_true", dest="no_cv",
        help="Disable per-combo CV threshold calibration; use --prob-threshold for all classes.",
    )
    p.add_argument(
        "--cv-min-k", type=int, default=3, dest="cv_min_k",
        help="Min combo size k for CV calibration; below this uses prob_threshold (default: 3).",
    )
    p.add_argument(
        "--tune-hyperparams", action="store_true", dest="tune_hyperparams",
        help="Run joint 5-fold CV on all training apps to tune hyperparams + thresholds (expensive).",
    )
    p.add_argument(
        "--n-iter", type=int, default=20, dest="n_iter",
        help="Hyperparam search iterations passed to DefaultTrainer.tune() (default: 20).",
    )
    p.add_argument(
        "--classifier", choices=["xgboost", "rf"], default="rf",
        help="Classifier to use: 'rf' (default) or 'xgboost'.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for holdout selection and combo sampling (default: 42).",
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

    all_paths = _find_labelled_csvs(TRAIN_DIR)
    n_total   = len(all_paths)

    steps = sorted(set(args.steps))
    if max(steps) >= n_total:
        print(
            f"[WARN] Largest step ({max(steps)}) leaves no test apps "
            f"(pool={n_total}) - it will be skipped."
        )

    from math import comb
    print(f"\n[INFO] App pool: {n_total} apps (test set is dynamic per combo)")
    for i, p in enumerate(all_paths):
        print(f"    [{i+1:>2}] {p.stem.replace('_labelled', '')}")

    total_combos = sum(
        comb(n_total, k) for k in steps if k < n_total
    )
    print(f"\n[INFO] Steps: {steps}")
    print(f"[INFO] Total combinations to train: {total_combos}")

    # --- Pre-extract features once per app ---
    print(f"\n[INFO] Pre-extracting features for {n_total} apps ...")
    app_features: list[tuple[pd.DataFrame, dict[str, pd.Series]]] = []
    for i, p in enumerate(all_paths):
        app = p.stem.replace("_labelled", "")
        print(f"  [{i+1}/{n_total}] {app} ...", end=" ", flush=True)
        X_app, y_app = _extract_features_for_app(
            p, args.window_size, args.step_size, args.severity_threshold
        )
        app_features.append((X_app, y_app))
        print(f"{X_app.shape[0]} windows")

    # --- Optional joint CV hyperparam + threshold tuning ---
    classifier = build_classifier(args.classifier, args.classifier_config)
    if args.tune_hyperparams:
        print(f"\n[INFO] Running joint CV tuning (n_iter={args.n_iter}) ...")
        classifier, _ = DefaultTrainer.tune(
            app_features, classifier, n_iter=args.n_iter, seed=args.seed
        )

    # --- Run exhaustive scaling ---
    results = run_gradual_scaling(
        app_features   = app_features,
        all_paths      = all_paths,
        steps          = steps,
        prob_threshold = args.prob_threshold,
        classifier     = classifier,
        max_combos     = args.max_combos,
        seed           = args.seed,
        use_cv         = not args.no_cv,
        cv_min_k       = args.cv_min_k,
    )

    if results.empty:
        print("[ERROR] No results collected.")
        sys.exit(1)

    # --- Summary ---
    print(f"\n{'='*65}")
    print("  GRADUAL SCALING SUMMARY - Macro-Average F1 per Step")
    print(f"{'='*65}")
    macro = (
        results.groupby(["n_train_apps", "bottleneck_type"])["f1"]
        .mean()
        .groupby("n_train_apps")
        .mean()
    )
    for k, avg_f1 in macro.items():
        n_combos = results[results["n_train_apps"] == k]["combo_idx"].nunique()
        print(
            f"  k={k:>2} apps  ->  "
            f"macro F1 = {avg_f1:.4f}  |  "
            f"over {n_combos} combinations"
        )
    print()

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
        print(f"[INFO] Results saved to: {out}")

    if args.output_pkl:
        import pickle
        out = Path(args.output_pkl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(results, f)
        print(f"[INFO] Results pickled to: {out}")

    if not args.no_plot:
        plot_results(
            results     = results,
            output_path = args.output_fig,
            show        = args.output_fig is None,
        )
