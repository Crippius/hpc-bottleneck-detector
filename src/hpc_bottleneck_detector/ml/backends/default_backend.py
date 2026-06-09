"""
Default Backend

Implements :class:`IMLBackend` using tsfresh feature extraction and a
configurable scikit-learn classifier (one per ``BottleneckType``).

Use :class:`~hpc_bottleneck_detector.ml.backends.DefaultTrainer` to build a
backend from labelled data.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

# FDR significance level for tsfresh feature selection.
_FDR_LEVEL = 0.01
# Number of features to keep when tsfresh's FDR test selects nothing.
_FALLBACK_K_FEATURES = 100
# Minimum RF feature importance to keep after FDR selection.
_IMPORTANCE_THRESHOLD = 1e-6

from ..backend_interface import IMLBackend
from ...utils.labeling import BOTTLENECK_COLUMNS
from .config import BASIC_FC_PARAMETERS

logger = logging.getLogger(__name__)

_LABEL_COLS = [bt.value for bt in BOTTLENECK_COLUMNS]
_NON_METRIC_COLS = {"id", "time"} | set(_LABEL_COLS)
EXCLUDE_METRIC_PREFIXES: tuple[str, ...] = ("gpu_",)
EXCLUDE_METRIC_COLS: frozenset[str] = frozenset({"INTER_NODE_LOAD_IMBALANCE"})

_DEFAULT_CLASSIFIER = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Module-level helpers (used by inference, training, and external scripts)
# ---------------------------------------------------------------------------

def _fill_metric_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in metric columns so tsfresh can process the DataFrame.

    Forward-fill then backward-fill within each id group, then zero-fill any
    remaining NaN (metric was never observed in this window).
    """
    meta_cols = {"id", "time"}
    metric_cols = [c for c in df.columns if c not in meta_cols]
    if not metric_cols:
        return df
    df = df.copy()
    df[metric_cols] = (
        df.groupby("id", sort=False)[metric_cols]
        .transform(lambda s: s.ffill().bfill())
        .fillna(0.0)
    )
    return df


def _build_window_dataframe(
    job_df: pd.DataFrame,
    metric_cols: list[str],
    job_id: str,
    window_size: int,
    step_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Slide windows over job_df and return a long-format tsfresh DataFrame
    together with a list of window IDs in window order.
    """
    n = len(job_df)
    fragments: list[pd.DataFrame] = []
    window_ids: list[str] = []

    start = 0
    win_idx = 0
    while start < n:
        end = min(start + window_size, n)
        win_id = f"{job_id}_w{win_idx}"

        fragment = job_df.iloc[start:end][["time"] + metric_cols].copy()
        fragment["id"] = win_id
        fragments.append(fragment)
        window_ids.append(win_id)

        if end == n:
            break
        start += step_size
        win_idx += 1

    long_df = pd.concat(fragments, ignore_index=True)
    return long_df, window_ids


def _window_labels(
    job_df: pd.DataFrame,
    window_size: int,
    step_size: int,
    severity_threshold: float,
) -> dict[str, list[Optional[float]]]:
    """
    For each sliding window compute a binary label per bottleneck column.

    Returns ``{col_name: [label, …]}`` where each label is 1, 0, or NaN.
    """
    n = len(job_df)
    result: dict[str, list[Optional[float]]] = {col: [] for col in _LABEL_COLS}

    start = 0
    while start < n:
        end = min(start + window_size, n)
        window_rows = job_df.iloc[start:end]

        for col in _LABEL_COLS:
            vals = window_rows[col].values
            real_vals = [v for v in vals if not math.isnan(v)]
            if real_vals:
                result[col].append(1 if max(real_vals) > severity_threshold else 0)
            else:
                result[col].append(float("nan"))

        if end == n:
            break
        start += step_size

    return result


def _merge_app_y(y_dicts: list[dict[str, pd.Series]]) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for col in _LABEL_COLS:
        parts = [yd[col] for yd in y_dicts if col in yd]
        if parts:
            result[col] = pd.concat(parts)
    return result


def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if math.isnan(p) or math.isnan(r) or (p + r) == 0:
        return float("nan")
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Backend (inference + persistence)
# ---------------------------------------------------------------------------

class DefaultBackend(IMLBackend):
    """
    Fitted inference backend: tsfresh feature extraction + per-type sklearn
    classifiers.

    Build with :class:`~hpc_bottleneck_detector.ml.backends.DefaultTrainer`,
    or restore a saved one with :meth:`load`.

    Attributes:
        _models:       ``{bt_name: fitted classifier}``
        _feature_cols: ``{bt_name: list of selected feature column names}``
        _thresholds:   ``{bt_name: probability threshold}``
        _fc_params:    tsfresh feature-calculation parameters.
        _window_size:  Window size used during training.
    """

    def __init__(self) -> None:
        self._models: dict[str, ClassifierMixin] = {}
        self._feature_cols: dict[str, list[str]] = {}
        self._thresholds: dict[str, float] = {}
        self._fc_params = BASIC_FC_PARAMETERS
        self._window_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Threshold calibration (post-training, pre-deployment)
    # ------------------------------------------------------------------

    def calibrate_thresholds(
        self,
        X_val: pd.DataFrame,
        y_dict_val: dict[str, pd.Series],
        threshold_grid: list[float] | None = None,
    ) -> dict[str, float]:
        """
        Sweep probability thresholds on X_val / y_dict_val and store
        the best F1-maximising threshold per class in ``self._thresholds``.
        """
        if threshold_grid is None:
            threshold_grid = list(np.arange(0.05, 0.91, 0.05))

        for col, clf in self._models.items():
            if col not in y_dict_val or y_dict_val[col].nunique() < 2:
                continue

            y_val = y_dict_val[col]
            X_aligned = X_val.reindex(
                index=y_val.index,
                columns=self._feature_cols[col],
                fill_value=0.0,
            )
            probs = clf.predict_proba(X_aligned)[:, 1]
            y_true = y_val.values

            best_thr, best_f1 = 0.5, -1.0
            for thr in threshold_grid:
                f1 = _f1_score(y_true, (probs >= thr).astype(int))
                if not math.isnan(f1) and f1 > best_f1:
                    best_f1, best_thr = f1, thr

            self._thresholds[col] = best_thr

        return dict(self._thresholds)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_probabilities(self, window_df: pd.DataFrame) -> dict[str, float]:
        """
        Extract tsfresh features from *window_df* and return per-type
        probabilities.

        Args:
            window_df: Raw window DataFrame from
                       ``DataManager.get_flat_dataframe()``.

        Returns:
            ``{BottleneckType.value: probability}`` for every trained type.
        """
        if not self._models:
            raise RuntimeError("Backend has not been trained or loaded yet.")

        if self._window_size is not None and len(window_df) != self._window_size:
            raise ValueError(
                f"Window has {len(window_df)} interval(s) but the model was trained "
                f"with window_size={self._window_size}. Re-train or adjust the "
                "orchestrator window_size to match."
            )

        infer_df = window_df.drop(
            columns=[c for c in _LABEL_COLS if c in window_df.columns]
        )
        infer_df = _fill_metric_nans(infer_df)

        X = extract_features(
            infer_df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=self._fc_params,
            impute_function=impute,
            disable_progressbar=True,
            n_jobs=1,
        )
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        result: dict[str, float] = {}
        for col, clf in self._models.items():
            X_aligned = X.reindex(columns=self._feature_cols[col], fill_value=0.0)
            prob = float(clf.predict_proba(X_aligned)[0, 1])
            result[col] = prob

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save models and feature pipeline to *path* (.pkl)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": self._models,
                "feature_cols": self._feature_cols,
                "fc_params": self._fc_params,
                "window_size": self._window_size,
                "thresholds": self._thresholds,
            },
            out,
        )
        logger.info("Backend saved to %s", out)

    @classmethod
    def load(cls, path: str) -> "DefaultBackend":
        """
        Restore a backend saved with :meth:`save`.

        Returns:
            Fully restored :class:`DefaultBackend` ready for inference.
        """
        data = joblib.load(path)
        backend = cls()
        backend._models = data["models"]
        backend._feature_cols = data["feature_cols"]
        backend._thresholds = data.get("thresholds", {})
        backend._fc_params = data.get("fc_params", BASIC_FC_PARAMETERS)
        backend._window_size = data.get("window_size")
        logger.info(
            "Backend loaded from %s (%d classifiers).", path, len(backend._models)
        )
        return backend
