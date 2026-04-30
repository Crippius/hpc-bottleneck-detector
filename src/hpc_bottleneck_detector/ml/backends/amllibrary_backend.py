"""
AMLLibrary Backend

Implements IMLBackend using aMLLibrary's MTS regression pipeline for automated
model selection. One MTSRegressor is trained per BottleneckType using a regression
proxy: severity (continuous 0-1) is used directly as the regression target.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..backend_interface import IMLBackend
from ...utils.labeling import BOTTLENECK_COLUMNS

logger = logging.getLogger(__name__)

_LABEL_COLS: list[str] = [bt.value for bt in BOTTLENECK_COLUMNS]
_EXCLUDE_PREFIXES: tuple[str, ...] = ("gpu_",)

_AML_DIR = Path(__file__).parents[4] / "aMLLibrary"

_WINDOW_FEATURES: list = [
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
    "range",
    "slope",
    "skewness",
    "kurtosis",
    {"quantile": {"q": 0.25}},
    {"quantile": {"q": 0.75}},
    {"autocorrelation": {"f_agg": "mean", "maxlag": 3}},
]

_TECHNIQUES: list[str] = ["LRRidge", "RandomForest", "XGBoost"]

# Per-technique hyperparameter search spaces passed as campaign config sections.
_TECHNIQUE_HPARAMS: dict = {
    "LRRidge": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "criterion": ["squared_error"],
        "max_depth": [None, 10],
        "max_features": ["sqrt"],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
    },
    "XGBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 6],
        "gamma": [0],
        "min_child_weight": [1],
        "lambda": [1],
        "alpha": [0],
    },
}


def _ensure_aml_on_path() -> None:
    aml_str = str(_AML_DIR)
    if aml_str not in sys.path:
        sys.path.insert(0, aml_str)


def _fill_metric_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in metric columns so aMLLibrary's DataCheck doesn't crash.

    Strategy: forward-fill then backward-fill within each id group (preserves
    temporal order), then fill any remaining NaN with 0.
    """
    non_metric = {"id", "time"} | set(_LABEL_COLS)
    metric_cols = [c for c in df.columns if c not in non_metric]
    if not metric_cols:
        return df
    df = df.copy()
    if "id" in df.columns:
        df[metric_cols] = (
            df.groupby("id", sort=False)[metric_cols]
            .transform(lambda s: s.ffill().bfill())
            .fillna(0.0)
        )
    else:
        df[metric_cols] = df[metric_cols].ffill().bfill().fillna(0.0)
    return df


def _train_one(
    merged_df: pd.DataFrame,
    target_col: str,
    window_size: int,
    step_size: int,
    out_dir: str,
) -> object:
    """
    Run one aMLLibrary campaign for *target_col* and return the MTSRegressor.

    The DataFrame passed in already has all other label columns removed, so
    aMLLibrary sees only [id, time, metrics, target_col]
    """
    _ensure_aml_on_path()
    import sequence_data_processing as sdp  # type: ignore[import]

    config: dict = {
        "General": {
            "run_num": 1,
            "techniques": _TECHNIQUES,
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


class AMLLibraryBackend(IMLBackend):
    """
    ML backend using aMLLibrary's automated model selection.

    For each BottleneckType, aMLLibrary trains several regression techniques
    (LRRidge, RandomForest, XGBoost) with HoldOut validation and picks the
    best one.  

    The predicted severity value is clipped to [0, 1] and used directly as
    a probability.
    """

    def __init__(self) -> None:
        self._regressors: dict[str, object] = {}

    def train(
        self,
        labelled_csv_paths: list[str],
        window_size: int,
        step_size: int,
        severity_threshold: float = 0.0,
    ) -> None:
        """Train one MTSRegressor per BottleneckType via aMLLibrary."""
        frames = [pd.read_csv(p) for p in labelled_csv_paths]
        if not frames:
            raise ValueError("No data loaded — check labelled_csv_paths.")
        merged = pd.concat(frames, ignore_index=True)

        metric_cols = [
            c for c in merged.columns
            if c not in (set(_LABEL_COLS) | {"id", "time"})
            and not any(c.startswith(pfx) for pfx in _EXCLUDE_PREFIXES)
        ]

        self._regressors = {}

        for col in _LABEL_COLS:
            if col not in merged.columns:
                logger.warning("Column %s missing from labelled data; skipping.", col)
                continue

            # Build a per-target DataFrame: keep only id, time, metrics, target.
            # This prevents other label columns from leaking as features
            train_df = merged[["id", "time"] + metric_cols + [col]].copy()
            train_df = train_df[train_df[col].notna()].reset_index(drop=True)
            train_df = _fill_metric_nans(train_df)

            n_total = len(train_df)
            n_pos = int((train_df[col] > severity_threshold).sum())
            logger.info(
                "Training %s — %d labelled intervals (%d positive at threshold=%.2f).",
                col, n_total, n_pos, severity_threshold,
            )

            if n_total < window_size:
                logger.warning(
                    "Skipping %s — fewer labelled intervals (%d) than window_size (%d).",
                    col, n_total, window_size,
                )
                continue

            tmp_root = tempfile.mkdtemp(prefix=f"aml_{col}_")
            out_dir = os.path.join(tmp_root, "output")
            try:
                reg = _train_one(train_df, col, window_size, step_size, out_dir)
                self._regressors[col] = reg
                logger.info("MTSRegressor trained for %s.", col)
            except Exception as exc:
                logger.warning("Failed to train regressor for %s: %s", col, exc, exc_info=True)
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)

        if not self._regressors:
            raise RuntimeError(
                "No regressors were trained. "
                "Check that the labelled CSVs contain enough data per BottleneckType."
            )

    def predict_probabilities(self, window_df: pd.DataFrame) -> dict[str, float]:
        """
        Return per-bottleneck severity estimates for a single pre-windowed DataFrame.

        Drops id, time, and any label columns present, then calls each MTSRegressor
        which re-applies windowing + feature extraction internally.

        Args:
            window_df: Raw window DataFrame from DataManager.get_flat_dataframe().
                       Contains id, time, and metric columns.

        Returns:
            {BottleneckType.value: probability in [0, 1]} for every trained type.
        """
        if not self._regressors:
            raise RuntimeError("Backend has not been trained or loaded yet.")

        _ensure_aml_on_path()

        drop_cols = {"id", "time"} | (set(_LABEL_COLS) & set(window_df.columns))
        infer_df = window_df.drop(columns=list(drop_cols), errors="ignore")
        infer_df = _fill_metric_nans(infer_df)

        result: dict[str, float] = {}
        for col, reg in self._regressors.items():
            preds = reg.predict(infer_df)
            val = float(np.asarray(preds).flatten()[0])
            result[col] = float(np.clip(val, 0.0, 1.0))

        return result

    def save(self, path: str) -> None:
        """Serialize all trained MTSRegressors to *path* (.pkl)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._regressors, out)
        logger.info("AMLLibraryBackend saved to %s (%d regressors).", out, len(self._regressors))

    @classmethod
    def load(cls, path: str) -> "AMLLibraryBackend":
        """
        Restore a backend previously saved with :meth:`save`.

        aMLLibrary must be importable (the module path is added automatically).
        """
        _ensure_aml_on_path()
        backend = cls()
        backend._regressors = joblib.load(path)
        logger.info(
            "AMLLibraryBackend loaded from %s (%d regressors).",
            path, len(backend._regressors),
        )
        return backend
