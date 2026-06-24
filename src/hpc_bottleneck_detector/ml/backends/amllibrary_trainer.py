"""
AMLLibrary Trainer

Implements :class:`IMLTrainer` using aMLLibrary's automated model selection.
Produces an :class:`AMLLibraryBackend` ready for inference.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..backend_interface import IMLTrainer
from .amllibrary_backend import (
    AMLLibraryBackend,
    _EXCLUDE_COLS,
    _EXCLUDE_PREFIXES,
    _LABEL_COLS,
    _TECHNIQUES,
    _TECHNIQUE_HPARAMS,
    _WINDOW_FEATURES,
    _ensure_aml_on_path,
    _fill_metric_nans,
)

logger = logging.getLogger(__name__)


def _train_one(
    merged_df: pd.DataFrame,
    target_col: str,
    window_size: int,
    step_size: int,
    out_dir: str,
) -> object:
    """
    Run one aMLLibrary campaign for *target_col* and return the MTSRegressor.
    """
    _ensure_aml_on_path()
    import sequence_data_processing as sdp  # type: ignore[import]

    config: dict = {
        "General": {
            "run_num": 1,
            "techniques": list(_TECHNIQUES),
            "hp_selection": "All",
            "validation": "KFold",
            "folds": 5,
            "y": target_col,
            "metric": "MAE",
        },
        "DataPreparation": {
            "input_path": merged_df,
            "time_column": "time",
            "series_id_column": "id",
            "window_size": window_size,
            "stride": step_size,
        },
        "WindowFeatures": {
            "features": _WINDOW_FEATURES,
            "y_window_position": "mean",
        },
        "FeatureSelection": {
            "method": "XGBoost",
            "max_features": 50,
            "XGBoost_tolerance": 0.9,
        },
        **_TECHNIQUE_HPARAMS,
    }

    processor = sdp.SequenceDataProcessing(config, output=out_dir)
    return processor.process()


class AMLLibraryTrainer(IMLTrainer):
    """
    Trainer for :class:`AMLLibraryBackend`.

    Runs an aMLLibrary campaign per ``BottleneckType`` (LRRidge, RandomForest,
    XGBoost with HoldOut validation) and returns a fitted backend.
    """

    def train(
        self,
        labelled_csv_paths: list[str],
        window_size: int,
        step_size: int,
        severity_threshold: float = 0.0,
    ) -> AMLLibraryBackend:
        """Train one MTSRegressor per BottleneckType and return a fitted backend."""
        t_total_start = time.perf_counter()
        frames = [pd.read_csv(p) for p in labelled_csv_paths]
        if not frames:
            raise ValueError("No data loaded - check labelled_csv_paths.")
        merged = pd.concat(frames, ignore_index=True)

        metric_cols = [
            c for c in merged.columns
            if c not in (set(_LABEL_COLS) | {"id", "time"})
            and not any(c.startswith(pfx) for pfx in _EXCLUDE_PREFIXES)
            and c not in _EXCLUDE_COLS
        ]

        backend = AMLLibraryBackend()
        per_type_times: dict[str, float] = {}

        for col in _LABEL_COLS:
            if col not in merged.columns:
                logger.warning("Column %s missing from labelled data; skipping.", col)
                continue

            train_df = merged[["id", "time"] + metric_cols + [col]].copy()
            train_df = train_df[train_df[col].notna()].reset_index(drop=True)
            train_df = _fill_metric_nans(train_df)

            n_total = len(train_df)
            n_pos = int((train_df[col] > severity_threshold).sum())
            logger.info(
                "Training %s - %d labelled intervals (%d positive at threshold=%.2f).",
                col, n_total, n_pos, severity_threshold,
            )

            if n_total < window_size:
                logger.warning(
                    "Skipping %s - fewer labelled intervals (%d) than window_size (%d).",
                    col, n_total, window_size,
                )
                continue

            tmp_root = tempfile.mkdtemp(prefix=f"aml_{col}_")
            out_dir = os.path.join(tmp_root, "output")
            t_type_start = time.perf_counter()
            try:
                reg = _train_one(train_df, col, window_size, step_size, out_dir)
                backend._regressors[col] = reg
                per_type_times[col] = time.perf_counter() - t_type_start
                logger.info("MTSRegressor trained for %s.", col)
            except Exception as exc:
                logger.warning("Failed to train regressor for %s: %s", col, exc, exc_info=True)
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)

        backend._window_size = window_size
        if not backend._regressors:
            raise RuntimeError(
                "No regressors were trained. "
                "Check that the labelled CSVs contain enough data per BottleneckType."
            )
        backend._training_meta = {
            "total_training_time_s": time.perf_counter() - t_total_start,
            "per_type_campaign_time_s": per_type_times,
            "n_apps": len(labelled_csv_paths),
        }
        return backend

    def calibrate_thresholds_cv(
        self,
        backend: AMLLibraryBackend,
        labelled_csv_paths: list[str],
        window_size: int,
        step_size: int,
        severity_threshold: float = 0.2,
        n_splits: int = 5,
        default_threshold: float = 0.5,
    ) -> dict[str, float]:
        """GroupKFold over apps to calibrate per-class decision thresholds.

        Uses already-trained regressors (no retraining per fold) — sweeps
        probability thresholds on held-out app groups and averages across folds.
        """
        _ensure_aml_on_path()

        n_apps = len(labelled_csv_paths)
        n_folds = min(n_splits, n_apps)
        gkf = GroupKFold(n_splits=n_folds)
        indices = np.arange(n_apps)
        threshold_grid = list(np.arange(0.05, 0.91, 0.05))
        thr_lists: dict[str, list[float]] = {col: [] for col in _LABEL_COLS}

        for _, val_idx in gkf.split(indices, groups=indices):
            val_paths = [labelled_csv_paths[i] for i in val_idx]

            for col, reg in backend._regressors.items():
                all_preds: list[float] = []
                all_labels: list[int] = []

                for csv_path in val_paths:
                    df = pd.read_csv(csv_path)
                    if col not in df.columns:
                        continue
                    metric_cols = [
                        c for c in df.columns
                        if c not in (set(_LABEL_COLS) | {"id", "time"})
                        and not any(c.startswith(pfx) for pfx in _EXCLUDE_PREFIXES)
                        and c not in _EXCLUDE_COLS
                    ]
                    for _, job_df in df.groupby("id"):
                        job_df = job_df.sort_values("time").reset_index(drop=True)
                        if len(job_df) < window_size:
                            continue
                        job_metrics = _fill_metric_nans(job_df[metric_cols].copy())
                        job_with_label = job_metrics.copy()
                        job_with_label[col] = job_df[col].values
                        try:
                            preds = np.asarray(reg.predict(job_metrics)).flatten()
                            labels = np.asarray(reg.get_true_y(job_with_label)).flatten()
                            binary = (labels > severity_threshold).astype(int)
                            n = min(len(preds), len(binary))
                            if n > 0:
                                all_preds.extend(preds[:n].tolist())
                                all_labels.extend(binary[:n].tolist())
                        except Exception as exc:
                            logger.warning("Calibration predict failed for %s: %s", col, exc)

                if len(all_preds) < 2 or len(set(all_labels)) < 2:
                    continue

                y_true = np.array(all_labels)
                preds_arr = np.clip(np.array(all_preds), 0.0, 1.0)
                best_thr, best_f1 = default_threshold, -1.0
                for thr in threshold_grid:
                    y_pred = (preds_arr >= thr).astype(int)
                    tp = int(((y_pred == 1) & (y_true == 1)).sum())
                    fp = int(((y_pred == 1) & (y_true == 0)).sum())
                    fn = int(((y_pred == 0) & (y_true == 1)).sum())
                    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
                    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
                    if math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0:
                        continue
                    f1 = 2 * prec * rec / (prec + rec)
                    if f1 > best_f1:
                        best_f1, best_thr = f1, thr
                thr_lists[col].append(best_thr)

        thresholds = {
            col: float(np.nanmean(v)) if v else default_threshold
            for col, v in thr_lists.items()
        }
        backend._thresholds = thresholds
        logger.info("Calibrated thresholds: %s", {k: f"{v:.3f}" for k, v in thresholds.items()})
        return thresholds
