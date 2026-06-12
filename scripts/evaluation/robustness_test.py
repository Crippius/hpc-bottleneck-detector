"""
Robustness Test — Prediction Stability Under Missing Metrics

Loads a trained model and the apps it was trained on, then checks whether
predictions stay the same when metric groups are removed at inference time.

This is NOT an accuracy test — it measures prediction stability:
  full-metric prediction  vs  dropped-metric prediction

Two test types:
1. Fixed scenarios: drop semantically meaningful metric groups.
2. Random sweep: drop a random fraction of metrics (10%–90%), 30 trials
   each → stability curve.

Features are extracted ONCE per app, then metric columns are blanked
in-memory per scenario — no re-extraction needed.

Usage:
    python scripts/evaluation/robustness_test.py
    python scripts/evaluation/robustness_test.py --classifier rf
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import (
    DefaultBackend,
    _NON_METRIC_COLS,
    _fill_metric_nans,
    _build_window_dataframe,
    EXCLUDE_METRIC_PREFIXES,
    EXCLUDE_METRIC_COLS,
    BASIC_FC_PARAMETERS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "labelled_data" / "training_set"

FIXED_SCENARIOS: dict[str, list[str]] = {
    "no_L3":           ["cache_*L3*"],
    "no_L2_L3":        ["cache_*L2*", "cache_*L3*"],
    "no_branch":       ["cpu_Branch*"],
    "no_memory":       ["memory_*"],
    "no_cache_memory": ["cache_*", "memory_*"],
    "no_cpu":          ["cpu_*"],
}

RANDOM_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
N_RANDOM = 30
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_app_features(
    csv_path: str,
    window_size: int,
    step_size: int,
    fc_params: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build tsfresh feature matrix for all windows in csv_path.
    Returns (X, metric_cols) where metric_cols are the raw input metrics.
    """
    df = pd.read_csv(csv_path)
    metric_cols = [
        c for c in df.columns
        if c not in _NON_METRIC_COLS
        and not any(c.startswith(p) for p in EXCLUDE_METRIC_PREFIXES)
        and c not in EXCLUDE_METRIC_COLS
    ]

    fragments: list[pd.DataFrame] = []
    window_ids: list[str] = []

    for job_id, job_df in df.groupby("id"):
        job_df = job_df.sort_values("time").dropna(subset=metric_cols).reset_index(drop=True)
        long_df, wids = _build_window_dataframe(
            job_df, metric_cols, str(job_id), window_size, step_size
        )
        fragments.append(long_df)
        window_ids.extend(wids)

    if not fragments:
        return pd.DataFrame(), metric_cols

    tsfresh_df = _fill_metric_nans(pd.concat(fragments, ignore_index=True))
    X = extract_features(
        tsfresh_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_params,
        impute_function=impute,
        disable_progressbar=True,
    )
    X = X.reindex(window_ids).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X, metric_cols


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _tsfresh_cols_for_metrics(X: pd.DataFrame, metric_patterns: list[str]) -> list[str]:
    if not metric_patterns:
        return []
    return [
        col for col in X.columns
        if any(fnmatch.fnmatch(col.split("__")[0], p) for p in metric_patterns)
    ]


def _predict_all(backend: DefaultBackend, X: pd.DataFrame) -> pd.DataFrame:
    """
    Return a boolean DataFrame (windows × bottleneck_types) indicating
    which classes fire above threshold for each window.
    """
    results = {}
    for bt, clf in backend._models.items():
        X_aligned = X.reindex(columns=backend._feature_cols[bt], fill_value=backend._missing_fill_value)
        probs = clf.predict_proba(X_aligned)[:, 1]
        thr = backend._thresholds.get(bt, 0.5)
        results[bt] = probs >= thr
    return pd.DataFrame(results, index=X.index)


def _blank_and_predict(
    backend: DefaultBackend,
    X: pd.DataFrame,
    blank_cols: list[str],
) -> pd.DataFrame:
    Xc = X.copy()
    if blank_cols:
        present = [c for c in blank_cols if c in Xc.columns]
        Xc[present] = backend._missing_fill_value
    return _predict_all(backend, Xc)


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _stability(full_preds: pd.DataFrame, drop_preds: pd.DataFrame) -> dict[str, float]:
    """Per-class fraction of windows where prediction is unchanged."""
    result = {}
    for bt in full_preds.columns:
        if bt not in drop_preds.columns:
            continue
        agree = (full_preds[bt] == drop_preds[bt]).mean()
        result[bt] = float(agree)
    return result


def _overall_stability(full_preds: pd.DataFrame, drop_preds: pd.DataFrame) -> float:
    """Fraction of windows where ALL class predictions are unchanged."""
    same = (full_preds == drop_preds).all(axis=1)
    return float(same.mean())


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_sweep(
    sweep_data: dict,
    bt_cols: list[str],
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    fracs = sorted(sweep_data.keys())
    pct = [f * 100 for f in fracs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall stability with ±1 std band
    ax = axes[0]
    means = [sweep_data[f]["overall_mean"] for f in fracs]
    stds  = [sweep_data[f]["overall_std"]  for f in fracs]
    ax.plot(pct, means, "o-", color="steelblue", linewidth=2, label="Overall")
    ax.fill_between(pct,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.25, color="steelblue")
    ax.set_xlabel("Metrics removed (%)")
    ax.set_ylabel("Prediction stability")
    ax.set_title("Overall stability (all classes agree)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Right: per-class stability (mean only, cleaner)
    ax = axes[1]
    colors = plt.cm.tab10.colors
    for i, bt in enumerate(bt_cols):
        means = [sweep_data[f]["per_class_mean"].get(bt, float("nan")) for f in fracs]
        stds  = [sweep_data[f]["per_class_std"].get(bt, float("nan"))  for f in fracs]
        label = bt.replace("_", " ").title()
        ax.plot(pct, means, "o-", color=colors[i % len(colors)], linewidth=2, label=label)
        ax.fill_between(pct,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.15, color=colors[i % len(colors)])
    ax.set_xlabel("Metrics removed (%)")
    ax.set_ylabel("Prediction stability")
    ax.set_title("Per-class stability")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    classifier = args.classifier
    fig.suptitle(f"Robustness to Missing Metrics — {classifier.upper()}", fontsize=13)
    fig.tight_layout()

    out = ROOT / "results" / f"robustness_{classifier}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    backend = DefaultBackend.load(str(ROOT / args.model))
    if args.classifier == "rf":
        backend._missing_fill_value = 0.0

    csv_paths = sorted(Path(DATA_DIR).rglob("*.csv"))
    logger.info("Found %d CSVs in %s", len(csv_paths), DATA_DIR)

    # ── Pre-extract features once per app ────────────────────────────────────
    window_size = backend._window_size or args.window_size
    all_X: list[pd.DataFrame] = []
    all_metric_cols: list[str] = []

    for csv_path in csv_paths:
        logger.info("Extracting features: %s", csv_path.name)
        X, metric_cols = _extract_app_features(
            str(csv_path), window_size, window_size, backend._fc_params
        )
        if X.empty:
            continue
        all_X.append(X)
        if not all_metric_cols:
            all_metric_cols = metric_cols

    X_all = pd.concat(all_X)
    n_windows = len(X_all)
    logger.info("Total windows: %d", n_windows)

    # ── Full-metric baseline predictions ─────────────────────────────────────
    full_preds = _predict_all(backend, X_all)
    bt_cols = list(full_preds.columns)

    # ── Fixed scenarios ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  FIXED METRIC-DROP SCENARIOS  ({args.classifier.upper()}, {n_windows} windows)")
    print(f"  Stability = fraction of windows where prediction is unchanged vs full")
    print(f"{'='*80}\n")

    col_w = 13
    print(f"  {'Scenario':<18}  {'Overall':>{col_w}}", end="")
    for bt in bt_cols:
        short = bt.replace("_", " ")[:col_w]
        print(f"  {short:>{col_w}}", end="")
    print()
    print("  " + "-" * (18 + (col_w + 2) * (1 + len(bt_cols)) + 2))

    for scenario, patterns in FIXED_SCENARIOS.items():
        blank = _tsfresh_cols_for_metrics(X_all, patterns)
        n_blanked = len(blank)
        drop_preds = _blank_and_predict(backend, X_all, blank)

        overall = _overall_stability(full_preds, drop_preds)
        per_class = _stability(full_preds, drop_preds)

        print(f"  {scenario:<18}  {overall:>{col_w}.3f}", end="")
        for bt in bt_cols:
            print(f"  {per_class.get(bt, float('nan')):>{col_w}.3f}", end="")
        print(f"  ({n_blanked} tsfresh cols blanked)")

    # ── Random sweep ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  RANDOM METRIC REMOVAL SWEEP  ({N_RANDOM} trials per fraction)")
    print(f"  Overall stability = fraction of windows with ALL predictions unchanged")
    print(f"{'='*80}\n")

    rng = random.Random(RANDOM_SEED)
    n_metrics = len(all_metric_cols)

    print(f"  {'Fraction':<10}  {'Overall mean±std':>18}", end="")
    for bt in bt_cols:
        short = bt.replace("_", " ")[:16]
        print(f"  {short:>16}", end="")
    print(f"  (of {n_metrics} metrics)")
    print("  " + "-" * (10 + 20 + 18 * len(bt_cols) + 16))

    sweep_data: dict[str, dict] = {}  # frac -> {overall_mean, overall_std, per_class_mean, per_class_std}

    for frac in RANDOM_FRACTIONS:
        n_drop = max(1, int(frac * n_metrics))
        overall_trials: list[float] = []
        per_class_trials: dict[str, list[float]] = {bt: [] for bt in bt_cols}

        for _ in range(N_RANDOM):
            dropped = rng.sample(all_metric_cols, n_drop)
            blank = _tsfresh_cols_for_metrics(X_all, dropped)
            drop_preds = _blank_and_predict(backend, X_all, blank)
            overall_trials.append(_overall_stability(full_preds, drop_preds))
            for bt, v in _stability(full_preds, drop_preds).items():
                per_class_trials[bt].append(v)

        overall_mean = float(np.mean(overall_trials))
        overall_std  = float(np.std(overall_trials))
        sweep_data[frac] = {
            "overall_mean": overall_mean,
            "overall_std":  overall_std,
            "per_class_mean": {bt: float(np.mean(v)) for bt, v in per_class_trials.items()},
            "per_class_std":  {bt: float(np.std(v))  for bt, v in per_class_trials.items()},
        }

        print(f"  {frac*100:>5.0f}%      {overall_mean:>7.3f} ± {overall_std:.3f}  ", end="")
        for bt in bt_cols:
            m = sweep_data[frac]["per_class_mean"].get(bt, float("nan"))
            s = sweep_data[frac]["per_class_std"].get(bt, float("nan"))
            print(f"  {m:>6.3f}±{s:.3f} ", end="")
        print(f"  (drop {n_drop})")

    print()
    _plot_sweep(sweep_data, bt_cols, args)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/xgboost.pkl",
                   help="Path to trained .pkl (relative to repo root).")
    p.add_argument("--classifier", choices=["xgboost", "rf"], default="xgboost",
                   help="Controls missing fill value: xgboost=NaN, rf=0.0.")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--window-size", type=int, default=12,
                   help="Fallback window size if not stored in model.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
