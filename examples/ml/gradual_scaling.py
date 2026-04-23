"""
Gradual Scaling Hold-Out

Proves how much application diversity the model needs before it generalises
to entirely unseen code.

Protocol
--------
1. Lock a static test set (last *test_size* apps) — never seen during training.
2. The remaining apps form the training pool.
3. For each step k in --steps, train on ALL C(pool, k) subsets of k apps.
4. Average metrics across all subsets → low-variance estimate of "what k apps buys".
5. Plot per-BottleneckType F1 + macro-average F1 vs number of training apps.

Speed note
----------
tsfresh feature extraction is done ONCE per app (pre-cached).  Each combination
only re-runs the fast sklearn feature-selection + classifier fit.

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
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_classif
from tsfresh import select_features as tsfresh_select_features

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import (
    DefaultBackend,
    _FALLBACK_K_FEATURES,
    _LABEL_COLS,
    _NON_METRIC_COLS,
    _build_window_dataframe,
    _fill_metric_nans,
    _window_labels,
    BASIC_FC_PARAMETERS,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT  = Path(__file__).parent.parent.parent
TRAIN_DIR  = REPO_ROOT / "data" / "labelled_data" / "miniapps"
TEST_DIR   = REPO_ROOT / "data" / "labelled_data" / "demo"

_DEFAULT_CLASSIFIER = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

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

def _find_labelled_csvs(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No labelled CSVs found in '{data_dir}'.")
    return paths


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


def _extract_features_for_app(
    csv_path: Path,
    window_size: int,
    step_size: int,
    severity_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Extract tsfresh features and labels for a single labelled CSV."""
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute

    df = pd.read_csv(csv_path)
    metric_cols = [c for c in df.columns if c not in _NON_METRIC_COLS]

    all_fragments:  list[pd.DataFrame] = []
    all_window_ids: list[str] = []
    all_labels: dict[str, list] = {col: [] for col in _LABEL_COLS}

    for job_id, job_df in df.groupby("id"):
        job_df    = job_df.sort_values("time").reset_index(drop=True)
        unique_id = f"{csv_path.stem}__{job_id}"

        long_df, window_ids = _build_window_dataframe(
            job_df, metric_cols, unique_id, window_size, step_size
        )
        labels = _window_labels(job_df, window_size, step_size, severity_threshold)

        all_fragments.append(long_df)
        all_window_ids.extend(window_ids)
        for col in _LABEL_COLS:
            all_labels[col].extend(labels[col])

    tsfresh_df = _fill_metric_nans(pd.concat(all_fragments, ignore_index=True))

    X = extract_features(
        tsfresh_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=BASIC_FC_PARAMETERS,
        impute_function=impute,
        disable_progressbar=True,
    )
    X = X.reindex(all_window_ids)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_dict: dict[str, pd.Series] = {}
    for col in _LABEL_COLS:
        y = pd.Series(all_labels[col], index=all_window_ids, dtype=float)
        valid = ~y.isna()
        if valid.sum() > 0:
            y_dict[col] = y[valid].astype(int)

    return X, y_dict


def _train_classifiers(
    X_train: pd.DataFrame,
    y_dict_train: dict[str, pd.Series],
    classifier,
) -> tuple[dict, dict]:
    """Feature selection + fit. Returns (models, feature_cols)."""
    models: dict = {}
    feature_cols: dict = {}

    for col in _LABEL_COLS:
        if col not in y_dict_train:
            continue
        y = y_dict_train[col]
        X_clean = X_train.reindex(index=y.index, fill_value=0.0).fillna(0.0)

        if y.nunique() < 2:
            continue

        try:
            X_selected = tsfresh_select_features(X_clean, y)
        except Exception:
            X_selected = pd.DataFrame()

        if X_selected.shape[1] == 0:
            k = min(_FALLBACK_K_FEATURES, X_clean.shape[1])
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X_clean, y)
            X_selected = X_clean.iloc[:, selector.get_support(indices=True)]

        feature_cols[col] = X_selected.columns.tolist()
        clf = clone(classifier)
        clf.fit(X_selected, y)
        models[col] = clf

    return models, feature_cols


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
# Exhaustive gradual scaling driver
# ---------------------------------------------------------------------------

def run_gradual_scaling(
    app_features: list[tuple[pd.DataFrame, dict[str, pd.Series]]],
    train_pool: list[Path],
    steps: list[int],
    X_test: pd.DataFrame,
    y_dict_test: dict[str, pd.Series],
    prob_threshold: float,
    classifier,
) -> pd.DataFrame:
    """
    For each step k, iterate over ALL C(pool, k) subsets, train on each,
    evaluate on the fixed test set, and record every result.
    Averaging over all subsets at step k gives a low-variance estimate.
    """
    records: list[dict] = []
    n_pool = len(app_features)

    for k in steps:
        if k > n_pool:
            logger.warning("Step k=%d exceeds pool size %d — skipping.", k, n_pool)
            continue

        combos = list(itertools.combinations(range(n_pool), k))
        print(f"\n{'─'*65}")
        print(f"  Step k={k:>2}  |  {len(combos)} combinations")
        print(f"{'─'*65}")

        f1_by_col: dict[str, list[float]] = {}

        for combo_idx, combo in enumerate(combos):
            app_names = [train_pool[i].stem.replace("_labelled", "") for i in combo]

            X_train = pd.concat([app_features[i][0] for i in combo]).fillna(0.0)
            y_dict_train: dict[str, pd.Series] = {}
            for col in _LABEL_COLS:
                parts = [app_features[i][1][col] for i in combo
                         if col in app_features[i][1]]
                if parts:
                    y_dict_train[col] = pd.concat(parts)

            models, feat_cols = _train_classifiers(X_train, y_dict_train, classifier)
            if not models:
                continue

            for col, clf in models.items():
                if col not in y_dict_test:
                    continue
                fc        = feat_cols[col]
                valid_idx = y_dict_test[col].index
                X_aligned = X_test.reindex(index=valid_idx, columns=fc, fill_value=0.0)
                probs     = clf.predict_proba(X_aligned)[:, 1]
                y_pred    = (probs >= prob_threshold).astype(int)
                m         = _metrics_from_arrays(y_dict_test[col].values, y_pred)

                f1_by_col.setdefault(col, []).append(m["f1"])
                records.append({
                    "n_train_apps":    k,
                    "combo_idx":       combo_idx,
                    "train_apps":      str(app_names),
                    "bottleneck_type": col,
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
    test_app_names: list[str],
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
        print("[WARN] matplotlib not available — skipping plot.")
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
    p.add_argument("--window-size",        type=int,   default=10,  dest="window_size")
    p.add_argument("--step-size",          type=int,   default=10,  dest="step_size")
    p.add_argument("--severity-threshold", type=float, default=0.0, dest="severity_threshold")
    p.add_argument("--prob-threshold",     type=float, default=0.5, dest="prob_threshold")
    p.add_argument("--output-csv",         type=str,   default=None, dest="output_csv")
    p.add_argument("--output-fig",         type=str,   default=None, dest="output_fig")
    p.add_argument("--no-plot",            action="store_true",     dest="no_plot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()

    train_pool = sorted(
        _find_labelled_csvs(TRAIN_DIR),
        key=lambda p: _count_windows(p, args.window_size, args.step_size),
    )
    test_set       = _find_labelled_csvs(TEST_DIR)
    test_app_names = [p.stem.replace("_labelled", "") for p in test_set]

    steps = sorted(set(args.steps))
    if max(steps) > len(train_pool):
        print(
            f"[WARN] Largest step ({max(steps)}) exceeds training pool "
            f"({len(train_pool)}) — it will be skipped."
        )

    # Combination counts
    from math import comb
    print(f"\n[INFO] Training pool (miniapps): {len(train_pool)} apps")
    print(f"[INFO] Test set      (demo):     {len(test_set)} apps → {test_app_names}")
    print(f"  Training pool — sorted by #windows:")
    cumulative = 0
    for i, p in enumerate(train_pool):
        wc  = _count_windows(p, args.window_size, args.step_size)
        cumulative += wc
        app = p.stem.replace("_labelled", "")
        print(f"    [{i+1:>2}] {app:<20}  {wc:>4} windows")
    print(f"         {'TOTAL':<20}  {cumulative:>4} windows")

    total_combos = sum(
        comb(len(train_pool), k) for k in steps if k <= len(train_pool)
    )
    print(f"\n[INFO] Steps: {steps}")
    print(f"[INFO] Total combinations to train: {total_combos}")

    # --- Pre-extract features once per training app ---
    print(f"\n[INFO] Pre-extracting features for {len(train_pool)} training apps …")
    app_features: list[tuple[pd.DataFrame, dict[str, pd.Series]]] = []
    for i, p in enumerate(train_pool):
        app = p.stem.replace("_labelled", "")
        print(f"  [{i+1}/{len(train_pool)}] {app} …", end=" ", flush=True)
        X_app, y_app = _extract_features_for_app(
            p, args.window_size, args.step_size, args.severity_threshold
        )
        app_features.append((X_app, y_app))
        print(f"{X_app.shape[0]} windows")

    # --- Extract test features (once) ---
    print(f"\n[INFO] Extracting test-set features (once, reused at every step) …")
    test_X_parts: list[pd.DataFrame] = []
    test_y_parts: dict[str, list[pd.Series]] = {col: [] for col in _LABEL_COLS}
    for p in test_set:
        X_t, y_t = _extract_features_for_app(
            p, args.window_size, args.step_size, args.severity_threshold
        )
        test_X_parts.append(X_t)
        for col in _LABEL_COLS:
            if col in y_t:
                test_y_parts[col].append(y_t[col])

    X_test = pd.concat(test_X_parts)
    y_dict_test: dict[str, pd.Series] = {}
    for col in _LABEL_COLS:
        if test_y_parts[col]:
            y_dict_test[col] = pd.concat(test_y_parts[col])

    print(
        f"[INFO] Test set: {X_test.shape[0]} windows  |  "
        f"label types: {list(y_dict_test.keys())}"
    )

    # --- Run exhaustive scaling ---
    results = run_gradual_scaling(
        app_features  = app_features,
        train_pool    = train_pool,
        steps         = steps,
        X_test        = X_test,
        y_dict_test   = y_dict_test,
        prob_threshold = args.prob_threshold,
        classifier    = _DEFAULT_CLASSIFIER,
    )

    if results.empty:
        print("[ERROR] No results collected.")
        sys.exit(1)

    # --- Summary ---
    print(f"\n{'='*65}")
    print("  GRADUAL SCALING SUMMARY — Macro-Average F1 per Step")
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
            f"  k={k:>2} apps  →  "
            f"macro F1 = {avg_f1:.4f}  |  "
            f"over {n_combos} combinations"
        )
    print()

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
        print(f"[INFO] Results saved to: {out}")

    if not args.no_plot:
        plot_results(
            results        = results,
            test_app_names = test_app_names,
            output_path    = args.output_fig,
            show           = args.output_fig is None,
        )
