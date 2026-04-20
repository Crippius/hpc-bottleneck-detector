"""
Gradual Scaling Hold-Out

Proves how much application diversity the model needs before it generalises
to entirely unseen code.

Protocol
--------
1. Lock a static test set (first *test_size* apps) — never seen during training.
2. The remaining apps form the training pool.
3. Train at each step k in --steps (default 2 4 6 8 10) apps.
4. At every step evaluate on the same fixed test set.
5. Plot per-BottleneckType F1 + macro-average F1 vs number of training apps.

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
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import (
    DefaultBackend,
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

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR  = REPO_ROOT / "data" / "labelled_data" / "miniapps"

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


def _extract_features_and_labels(
    csv_paths: list[Path],
    window_size: int,
    step_size: int,
    severity_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute

    all_fragments:  list[pd.DataFrame] = []
    all_window_ids: list[str] = []
    all_labels: dict[str, list] = {col: [] for col in _LABEL_COLS}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        metric_cols = [c for c in df.columns if c not in _NON_METRIC_COLS]

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

    X_full = extract_features(
        tsfresh_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=BASIC_FC_PARAMETERS,
        impute_function=impute,
        disable_progressbar=True,
    )
    X_full = X_full.reindex(all_window_ids)
    X_full = X_full.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_dict: dict[str, pd.Series] = {}
    for col in _LABEL_COLS:
        y = pd.Series(all_labels[col], index=all_window_ids, dtype=float)
        valid = ~y.isna()
        if valid.sum() > 0:
            y_dict[col] = y[valid].astype(int)

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
        if (not np.isnan(precision) and not np.isnan(recall)
            and (precision + recall) > 0)
        else float("nan")
    )
    far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    mr  = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    return {"f1": f1, "false_alarm_rate": far, "miss_rate": mr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _evaluate_backend(
    backend: DefaultBackend,
    X_test: pd.DataFrame,
    y_dict: dict[str, pd.Series],
    prob_threshold: float,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for col, clf in backend._models.items():
        if col not in y_dict:
            continue
        feature_cols = backend._feature_cols[col]
        valid_idx    = y_dict[col].index
        X_aligned    = X_test.reindex(index=valid_idx, columns=feature_cols, fill_value=0.0)
        probs        = clf.predict_proba(X_aligned)[:, 1]
        y_pred       = (probs >= prob_threshold).astype(int)
        results[col] = _metrics_from_arrays(y_dict[col].values, y_pred)
    return results


# ---------------------------------------------------------------------------
# Gradual scaling driver
# ---------------------------------------------------------------------------

def run_gradual_scaling(
    train_pool: list[Path],
    steps: list[int],
    X_test: pd.DataFrame,
    y_dict_test: dict[str, pd.Series],
    window_size: int,
    step_size: int,
    severity_threshold: float,
    prob_threshold: float,
) -> pd.DataFrame:
    records: list[dict] = []

    for k in steps:
        if k > len(train_pool):
            logger.warning("Step k=%d exceeds training pool size %d — skipping.", k, len(train_pool))
            continue

        subset    = train_pool[:k]
        app_names = [p.stem.replace("_labelled", "") for p in subset]
        print(f"\n{'─'*65}")
        print(f"  Step k={k:>2}  |  Training on: {app_names}")
        print(f"{'─'*65}")

        backend = DefaultBackend()
        try:
            backend.train(
                labelled_csv_paths=[str(p) for p in subset],
                window_size=window_size,
                step_size=step_size,
                severity_threshold=severity_threshold,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("  k=%d — training failed (%s); skipping.", k, exc)
            continue

        if not backend._models:
            logger.warning("  k=%d — no classifiers trained; skipping.", k)
            continue

        eval_results = _evaluate_backend(backend, X_test, y_dict_test, prob_threshold)

        f1_vals: list[float] = []
        for col, m in eval_results.items():
            n_features = len(backend._feature_cols[col])
            f1_vals.append(m["f1"])
            print(
                f"    {col:<42}  "
                f"F1={m['f1']:.3f}  "
                f"FAR={m['false_alarm_rate']:.3f}  "
                f"MR={m['miss_rate']:.3f}  "
                f"feat={n_features}"
            )
            records.append({
                "n_train_apps":    k,
                "train_apps":      str(app_names),
                "bottleneck_type": col,
                "n_features":      n_features,
                **m,
            })

        macro_f1 = float(np.nanmean(f1_vals)) if f1_vals else float("nan")
        print(f"    {'[MACRO F1]':<42}  {macro_f1:.3f}")

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

    bt_types  = sorted(results["bottleneck_type"].unique())
    x_vals    = sorted(results["n_train_apps"].unique())
    bt_colors = cm.tab10(np.linspace(0, 0.9, len(bt_types)))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Gradual Scaling Hold-Out",
        fontsize=13, fontweight="bold",
    )

    for bt, color in zip(bt_types, bt_colors):
        bt_df = results[results["bottleneck_type"] == bt].set_index("n_train_apps")
        y = [bt_df.loc[x, "f1"] if x in bt_df.index else float("nan") for x in x_vals]
        ax.plot(
            x_vals, y,
            color=color, linewidth=1.4, linestyle="--", alpha=0.7, marker="s",
            markersize=4, label=_BT_SHORT.get(bt, bt),
        )

    macro_f1: list[float] = []
    for x in x_vals:
        vals = results.loc[results["n_train_apps"] == x, "f1"].dropna()
        macro_f1.append(float(vals.mean()) if len(vals) > 0 else float("nan"))

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
    ax.set_ylabel("F1-Score", fontsize=11)
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
        description="Gradual Scaling Hold-Out: F1 vs. number of training apps."
    )
    p.add_argument(
        "--steps", nargs="+", type=int, default=[2, 4, 6, 8, 10],
        help="Training set sizes to evaluate (default: 2 4 6 8 10).",
    )
    p.add_argument("--test-size",          type=int,   default=3,   dest="test_size",
                   help="Apps reserved as fixed test set (default: 3).")
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

    csv_paths = _find_labelled_csvs(DATA_DIR)
    n         = len(csv_paths)

    if args.test_size >= n:
        print(f"[ERROR] --test-size ({args.test_size}) must be < total apps ({n}).")
        sys.exit(1)

    # First test_size apps = fixed test set, rest = training pool
    test_set       = csv_paths[n-args.test_size:]
    train_pool     = csv_paths[:n-args.test_size]
    test_app_names = [p.stem.replace("_labelled", "") for p in test_set]

    steps = sorted(set(args.steps))
    if max(steps) > len(train_pool):
        print(
            f"[WARN] Largest step ({max(steps)}) exceeds training pool "
            f"({len(train_pool)}) — it will be clamped."
        )

    print(f"\n[INFO] {n} total apps")
    print(f"  Fixed test set  ({len(test_set)}): {test_app_names}")
    print(f"  Training pool  ({len(train_pool)}):")
    cumulative = 0
    for i, p in enumerate(train_pool):
        wc  = _count_windows(p, args.window_size, args.step_size)
        cumulative += wc
        app = p.stem.replace("_labelled", "")
        print(f"    [{i+1:>2}] {app:<20}  {wc:>4} windows")
    print(f"         {'TOTAL':<20}  {cumulative:>4} windows")

    print(f"\n[INFO] Extracting test-set features (once, reused at every step) …")
    X_test, y_dict_test = _extract_features_and_labels(
        test_set, args.window_size, args.step_size, args.severity_threshold
    )
    print(
        f"[INFO] Test set: {X_test.shape[0]} windows  |  "
        f"label types: {list(y_dict_test.keys())}"
    )

    print(f"\n[INFO] Running steps: {steps}\n")
    results = run_gradual_scaling(
        train_pool         = train_pool,
        steps              = steps,
        X_test             = X_test,
        y_dict_test        = y_dict_test,
        window_size        = args.window_size,
        step_size          = args.step_size,
        severity_threshold = args.severity_threshold,
        prob_threshold     = args.prob_threshold,
    )

    if results.empty:
        print("[ERROR] No results collected.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print("  GRADUAL SCALING SUMMARY — Macro-Average F1 per Step")
    print(f"{'='*65}")
    macro = results.groupby("n_train_apps").agg(
        macro_f1=("f1",         lambda s: s.dropna().mean()),
        avg_features=("n_features", lambda s: s.dropna().mean()),
    )
    for k, row in macro.iterrows():
        print(
            f"  k={k:>2} apps  →  "
            f"macro F1 = {row['macro_f1']:.4f}  |  "
            f"avg features = {row['avg_features']:.1f}"
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
