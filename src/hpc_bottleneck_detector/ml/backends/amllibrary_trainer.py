"""
AMLLibrary Trainer

Implements :class:`IMLTrainer` using aMLLibrary's automated model selection.
Produces an :class:`AMLLibraryBackend` ready for inference.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

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
            "validation": "HoldOut",
            "hold_out_ratio": 0.2,
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
            try:
                reg = _train_one(train_df, col, window_size, step_size, out_dir)
                backend._regressors[col] = reg
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
        return backend
