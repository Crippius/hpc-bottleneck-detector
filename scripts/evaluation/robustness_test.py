"""
Robustness Test - Prediction Stability Under Missing Metrics

Loads a trained model and the apps it was trained on, then checks whether
predictions stay the same when metric groups are removed at inference time.

Usage:
    python scripts/evaluation/robustness_test.py
    python scripts/evaluation/robustness_test.py --classifier rf
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
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
from hpc_bottleneck_detector.ml.backends.amllibrary_backend import (
    AMLLibraryBackend,
    _fill_metric_nans as _aml_fill_nans,
    _EXCLUDE_COLS as _AML_EXCLUDE_COLS,
    _EXCLUDE_PREFIXES as _AML_EXCLUDE_PREFIXES,
    _LABEL_COLS as _AML_LABEL_COLS,
    _ensure_aml_on_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data" / "labelled_data" / "training_set"

FIXED_SCENARIOS: dict[str, list[str]] = {
    "no_L3":        ["cache_*L3*"],
    "no_L2_L3":     ["cache_*L2*", "cache_*L3*"],
    "no_branch":    ["cpu_Branch*"],
    # Metrics present in training set but missing in the demo system
    "missing_demo": [
        "cache_Miss Rate_L3*", "cache_Miss Ratio_L3*", "cache_Request Rate_L3*",
        "disk_*",
    ],
    # Intel-specific metrics not available on AMD EPYC (Genoa)
    "missing_amd":  [
        "cache_Bandwidth_L2D*", "cache_Bandwidth_L3_evict*", "cache_Bandwidth_L3_load*",
        "cache_Data Volume_L2D*", "cache_Data Volume_L3_load*",
        "cpu_Cycles w/o Execution*", "cpu_Stall*",
        "cpu_FLOPS_AVX*", "cpu_SSE Operations*", "cpu_Vectorization*",
        "energy_CPU Temperature*", "energy_DRAM Power*",
        "memory_Bandwidth_write", "memory_Data Volume_write", "memory_UPI*",
    ],
}


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
    Return a boolean DataFrame (windows x bottleneck_types) indicating
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
# Main logic
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    backend = DefaultBackend.load(str(ROOT / args.model))
    if args.classifier == "rf":
        backend._missing_fill_value = 0.0

    csv_paths = sorted(Path(DATA_DIR).rglob("*.csv"))
    logger.info("Found %d CSVs in %s", len(csv_paths), DATA_DIR)

    # --- Pre-extract features once per app ------------------------------------------------------
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

    # --- Full-metric baseline predictions -------------------------------------------------------
    full_preds = _predict_all(backend, X_all)
    bt_cols = list(full_preds.columns)

    # --- Fixed scenarios ----------------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# AMLLibrary robustness helpers
# ---------------------------------------------------------------------------

def _aml_metric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in (set(_AML_LABEL_COLS) | {"id", "time"})
        and not any(c.startswith(pfx) for pfx in _AML_EXCLUDE_PREFIXES)
        and c not in _AML_EXCLUDE_COLS
    ]


def _aml_predict_jobs(
    backend: AMLLibraryBackend,
    df: pd.DataFrame,
    metric_cols: list[str],
    window_size: int,
) -> pd.DataFrame:
    """
    Run all trained regressors on a CSV DataFrame and return per-window binary
    predictions as a boolean DataFrame (index = sequential window id).
    """
    _ensure_aml_on_path()
    rows: list[dict] = []
    win_idx = 0

    for _, job_df in df.groupby("id"):
        job_df = job_df.sort_values("time").reset_index(drop=True)
        if len(job_df) < window_size:
            continue
        job_metrics = _aml_fill_nans(job_df[metric_cols].copy())

        win_preds: dict[str, list] = {}
        for bt, reg in backend._regressors.items():
            try:
                preds = np.asarray(reg.predict(job_metrics)).flatten()
            except Exception:
                preds = np.full(max(0, len(job_df) - window_size + 1), np.nan)
            thr = backend._thresholds.get(bt, 0.5)
            win_preds[bt] = (np.clip(preds, 0.0, 1.0) >= thr).tolist()

        n_wins = min(len(v) for v in win_preds.values()) if win_preds else 0
        for i in range(n_wins):
            rows.append({bt: win_preds[bt][i] for bt in win_preds})
            win_idx += 1

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_amllibrary(args: argparse.Namespace) -> None:
    """Robustness test for AMLLibraryBackend — blanks raw metric columns."""
    backend = AMLLibraryBackend.load(str(ROOT / args.model))
    window_size = backend._window_size or args.window_size

    csv_paths = sorted(Path(args.data_dir).rglob("*.csv"))
    logger.info("Found %d CSVs in %s", len(csv_paths), args.data_dir)

    dfs: list[pd.DataFrame] = []
    all_metric_cols: list[str] = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if not all_metric_cols:
            all_metric_cols = _aml_metric_cols(df)
        dfs.append(df)

    n_metrics = len(all_metric_cols)
    logger.info("Total metric columns: %d", n_metrics)

    # --- Full-metric baseline ---
    full_frames: list[pd.DataFrame] = []
    for df in dfs:
        mc = _aml_metric_cols(df)
        fp = _aml_predict_jobs(backend, df, mc, window_size)
        if not fp.empty:
            full_frames.append(fp)
    full_preds = pd.concat(full_frames, ignore_index=True) if full_frames else pd.DataFrame()
    bt_cols = list(full_preds.columns) if not full_preds.empty else []
    n_windows = len(full_preds)

    print(f"\n{'='*80}")
    print(f"  FIXED METRIC-DROP SCENARIOS  (amllibrary, {n_windows} windows)")
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
        drop_frames: list[pd.DataFrame] = []
        for df in dfs:
            mc = _aml_metric_cols(df)
            df_zeroed = df.copy()
            blanked = [c for c in mc if any(fnmatch.fnmatch(c, p) for p in patterns)]
            if blanked:
                df_zeroed[blanked] = 0.0
            dp = _aml_predict_jobs(backend, df_zeroed, mc, window_size)
            if not dp.empty:
                drop_frames.append(dp)
        drop_preds = pd.concat(drop_frames, ignore_index=True) if drop_frames else pd.DataFrame()

        overall = _overall_stability(full_preds, drop_preds)
        per_class = _stability(full_preds, drop_preds)
        print(f"  {scenario:<18}  {overall:>{col_w}.3f}", end="")
        for bt in bt_cols:
            print(f"  {per_class.get(bt, float('nan')):>{col_w}.3f}", end="")
        n_blanked = len([c for c in all_metric_cols if any(fnmatch.fnmatch(c, p) for p in patterns)])
        print(f"  ({n_blanked} raw metric cols zeroed)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/xgboost.pkl",
                   help="Path to trained .pkl (relative to repo root).")
    p.add_argument("--classifier", choices=["xgboost", "rf"], default="xgboost",
                   help="Controls missing fill value: xgboost=NaN, rf=0.0.")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--window-size", type=int, default=12,
                   help="Fallback window size if not stored in model.")
    p.add_argument("--backend", choices=["default", "amllibrary"], default="default",
                   help="ML backend to evaluate (default: default).")
    args = p.parse_args()
    if args.backend == "amllibrary":
        run_amllibrary(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
