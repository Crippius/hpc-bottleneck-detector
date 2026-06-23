"""
AMLLibrary Backend

Implements :class:`IMLBackend` using aMLLibrary's MTS regression pipeline.
One MTSRegressor is stored per BottleneckType; predicted severity (0-1) is
used directly as a probability.

Use :class:`~hpc_bottleneck_detector.ml.backends.AMLLibraryTrainer` to build
a backend from labelled data.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..backend_interface import IMLBackend
from ...utils.labeling import BOTTLENECK_COLUMNS

logger = logging.getLogger(__name__)

_LABEL_COLS: list[str] = [bt.value for bt in BOTTLENECK_COLUMNS]
_EXCLUDE_PREFIXES: tuple[str, ...] = ("gpu_",)
_EXCLUDE_COLS: frozenset[str] = frozenset({"INTER_NODE_LOAD_IMBALANCE"})

_AML_DIR = Path(__file__).parents[4] / "aMLLibrary"

_WINDOW_FEATURES: list = [
    # Simple stats (matches tsfresh BASIC_FC_PARAMETERS)
    "minimum",
    "maximum",
    "mean",
    "standard_deviation",
    {"quantile": {"q": 0.05}},
    {"quantile": {"q": 0.25}},
    {"quantile": {"q": 0.50}},
    {"quantile": {"q": 0.75}},
    {"quantile": {"q": 0.95}},
    "skewness",
    "kurtosis",
    {"autocorrelation": {"f_agg": "mean",   "maxlag": 3}},
    {"autocorrelation": {"f_agg": "median", "maxlag": 3}},
    {"autocorrelation": {"f_agg": "var",    "maxlag": 3}},
    {"agg_linear_trend": {"attr": ["slope", "intercept", "rvalue"], "chunk_len": [5], "f_agg": "mean"}},
    {"agg_linear_trend": {"attr": ["slope", "intercept"],           "chunk_len": [2], "f_agg": "mean"}},
    "absolute_sum_of_changes",
]

_TECHNIQUES: list[str] = ["RandomForest", "XGBoost"]

_TECHNIQUE_HPARAMS: dict = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "criterion": ["squared_error"],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt"],
        "min_samples_split": [2],
        "min_samples_leaf": [1, 2],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5, 8],
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
    """
    non_metric = {"id", "time"} | set(_LABEL_COLS)
    metric_cols = [c for c in df.columns if c not in non_metric]
    if not metric_cols:
        return df
    df = df.copy()
    df[metric_cols] = df[metric_cols].ffill().bfill().fillna(0.0)
    return df


class AMLLibraryBackend(IMLBackend):
    """
    Fitted inference backend using aMLLibrary regressors.

    Build with :class:`~hpc_bottleneck_detector.ml.backends.AMLLibraryTrainer`,
    or restore a saved one with :meth:`load`.

    Attributes:
        _regressors: ``{bottleneck_type_name: MTSRegressor}``
    """

    def __init__(self) -> None:
        self._regressors: dict[str, object] = {}
        self._window_size: int = 12
        self._thresholds: dict[str, float] = {}
        self._training_meta: dict = {}

    def predict_probabilities(self, window_df: pd.DataFrame) -> dict[str, float]:
        """
        Return per-bottleneck severity estimates clipped to [0, 1].

        Args:
            window_df: Raw window DataFrame from DataManager.get_flat_dataframe().

        Returns:
            ``{BottleneckType.value: probability}`` for every trained type.
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
        joblib.dump({"regressors": self._regressors, "window_size": self._window_size, "thresholds": self._thresholds, "training_meta": self._training_meta}, out)
        logger.info(
            "AMLLibraryBackend saved to %s (%d regressors).", out, len(self._regressors)
        )

    @classmethod
    def load(cls, path: str) -> "AMLLibraryBackend":
        """
        Restore a backend previously saved with :meth:`save`.

        aMLLibrary must be importable (the module path is added automatically).
        """
        _ensure_aml_on_path()
        backend = cls()
        data = joblib.load(path)
        if isinstance(data, dict) and "regressors" in data:
            backend._regressors = data["regressors"]
            backend._window_size = data.get("window_size", 12)
            backend._thresholds = data.get("thresholds", {})
            backend._training_meta = data.get("training_meta", {})
        else:
            backend._regressors = data
        logger.info(
            "AMLLibraryBackend loaded from %s (%d regressors).",
            path, len(backend._regressors),
        )
        return backend
