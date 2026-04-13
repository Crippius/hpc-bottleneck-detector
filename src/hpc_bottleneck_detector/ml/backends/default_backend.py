"""
Default Backend

Implements :class:`IMLBackend` using tsfresh for feature extraction and selection
and a configurable scikit-learn classifier (one per
``BottleneckType``) for classification.

Any sklearn-compatible classifier that exposes ``predict_proba()`` is
supported.  Pass it as *classifier* to :meth:`DefaultBackend.__init__`::

    from sklearn.ensemble import GradientBoostingClassifier
    backend = DefaultBackend(classifier=GradientBoostingClassifier(n_estimators=100))

The default is ``RandomForestClassifier(n_estimators=200, class_weight='balanced')``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

# Number of features to keep when tsfresh's FDR test selects nothing.
_FALLBACK_K_FEATURES = 100

from ..backend_interface import IMLBackend
from ...utils.labeling import BOTTLENECK_COLUMNS

logger = logging.getLogger(__name__)

# Names of the label columns as they appear in labelled CSVs.
_LABEL_COLS = [bt.value for bt in BOTTLENECK_COLUMNS]
# Columns that are never metric features.
_NON_METRIC_COLS = {"id", "time"} | set(_LABEL_COLS)

# ---------------------------------------------------------------------------
# Feature-extraction parameter sets
# ---------------------------------------------------------------------------
# Switch between sets by uncommenting the desired assignment for _fc_params in
# DefaultBackend.__init__ (and mirroring the change in train_ml_model.py).

# 1. Basic — lightweight descriptive statistics, suitable for fast iteration.
BASIC_FC_PARAMETERS: dict = {
    "minimum": None,
    "maximum": None,
    "mean": None,
    "standard_deviation": None,
    "quantile": [
        {"q": 0.05}, {"q": 0.25}, {"q": 0.50}, {"q": 0.75}, {"q": 0.95},
    ],
    "skewness": None,
    "kurtosis": None,
    "agg_autocorrelation": [
        {"f_agg": "mean",   "maxlag": 40},
        {"f_agg": "median", "maxlag": 40},
        {"f_agg": "var",    "maxlag": 40},
    ],
    "agg_linear_trend": [
        {"attr": "slope",     "chunk_len": 5,  "f_agg": "mean"},
        {"attr": "intercept", "chunk_len": 5,  "f_agg": "mean"},
        {"attr": "rvalue",    "chunk_len": 5,  "f_agg": "mean"},
        {"attr": "slope",     "chunk_len": 10, "f_agg": "mean"},
        {"attr": "intercept", "chunk_len": 10, "f_agg": "mean"},
        {"attr": "rvalue",    "chunk_len": 10, "f_agg": "mean"},
    ],
}

# 2. Basic + Advanced — adds thresholding, energy, complexity, and frequency
#    features on top of the basic set.
# BASIC_ADVANCED_FC_PARAMETERS: dict = {
#     **BASIC_FC_PARAMETERS,
#     "count_above_mean": None,
#     "count_below_mean": None,
#     "ratio_beyond_r_sigma": [
#         {"r": 0.5}, {"r": 1.0}, {"r": 1.5}, {"r": 2.0}, {"r": 2.5}, {"r": 3.0},
#     ],
#     "first_location_of_maximum": None,
#     "first_location_of_minimum": None,
#     "abs_energy": None,
#     "absolute_sum_of_changes": None,
#     "cid_ce": [{"normalize": True}, {"normalize": False}],
#     "c3": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
#     "ar_coefficient": [
#         {"coeff": 0, "k": 10}, {"coeff": 1, "k": 10}, {"coeff": 2, "k": 10},
#         {"coeff": 3, "k": 10}, {"coeff": 4, "k": 10},
#     ],
#     "augmented_dickey_fuller": [
#         {"attr": "teststat", "autolag": "AIC"},
#         {"attr": "pvalue",   "autolag": "AIC"},
#     ],
#     "fft_coefficient": [
#         {"coeff": 0, "attr": "real"},
#         {"coeff": 1, "attr": "real"}, {"coeff": 1, "attr": "imag"},
#         {"coeff": 2, "attr": "real"}, {"coeff": 2, "attr": "imag"},
#         {"coeff": 3, "attr": "real"}, {"coeff": 3, "attr": "imag"},
#     ],
# }

# 3. Basic + Advanced + High-Cost — adds entropy and spectral density features;
#    significantly slower to compute, use only when compute budget allows.
# BASIC_ADVANCED_HIGH_COST_FC_PARAMETERS: dict = {
#     **BASIC_ADVANCED_FC_PARAMETERS,
#     "approximate_entropy": [
#         {"m": 2, "r": 0.1}, {"m": 2, "r": 0.3},
#         {"m": 2, "r": 0.5}, {"m": 2, "r": 0.7},
#     ],
#     "spkt_welch_density": [{"coeff": 1}, {"coeff": 2}, {"coeff": 5}],
#     "variation_coefficient": None,
# }


def _fill_metric_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in metric columns so tsfresh can process the DataFrame.

    Strategy: forward-fill then backward-fill within each id group (preserves
    temporal continuity), then fill any remaining NaN with 0 (metric was never
    observed in this window).
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

    Each window gets a unique id ``"<job_id>_w<n>"``.
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

    Returns a dict ``{col_name: [label, …]}`` where each label is:
    - ``1``   — max severity in window > threshold (bottleneck present)
    - ``0``   — all real severities ≤ threshold (no bottleneck)
    - ``NaN`` — every interval in the window had NaN label (unknown)
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
                # result[col].append(max(real_vals)) # heuristic severity ≈ ml confidence
            else:
                result[col].append(float("nan"))

        if end == n:
            break
        start += step_size

    return result


_DEFAULT_CLASSIFIER = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


class DefaultBackend(IMLBackend):
    """
    ML backend based on tsfresh feature extraction and a configurable
    sklearn classifier (one per ``BottleneckType``).

    Args:
        classifier: Any sklearn-compatible classifier with ``predict_proba()``.
                    A fresh clone is fitted per ``BottleneckType`` so the same
                    instance can be safely reused.  Defaults to
                    ``RandomForestClassifier(n_estimators=200, class_weight='balanced')``.

    Example::

        from sklearn.linear_model import LogisticRegression
        backend = DefaultBackend(classifier=LogisticRegression(max_iter=1000))
        backend.train(csv_paths, window_size=10, step_size=10)

    Attributes:
        _classifier:   Unfitted prototype classifier (cloned for each type).
        _models:       ``{bt_name: fitted classifier}``
        _feature_cols: ``{bt_name: list of selected feature column names}``
        _fc_params:    tsfresh feature-calculation parameters.
    """

    def __init__(self, classifier: ClassifierMixin = _DEFAULT_CLASSIFIER) -> None:
        self._classifier = classifier
        self._models: dict[str, ClassifierMixin] = {}
        self._feature_cols: dict[str, list[str]] = {}
        self._fc_params = BASIC_FC_PARAMETERS
        # self._fc_params = BASIC_ADVANCED_FC_PARAMETERS
        # self._fc_params = BASIC_ADVANCED_HIGH_COST_FC_PARAMETERS

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        labelled_csv_paths: list[str],
        window_size: int,
        step_size: int,
        severity_threshold: float = 0.0,
    ) -> None:
        """Train one classifier per ``BottleneckType`` using ``self._classifier`` as prototype."""
        all_fragments: list[pd.DataFrame] = []
        all_window_ids: list[str] = []
        all_labels: dict[str, list[Optional[float]]] = {
            col: [] for col in _LABEL_COLS
        }

        for csv_path in labelled_csv_paths:
            logger.info("Loading labelled CSV: %s", csv_path)
            df = pd.read_csv(csv_path)

            metric_cols = [c for c in df.columns if c not in _NON_METRIC_COLS]

            for job_id, job_df in df.groupby("id"):
                job_df = job_df.sort_values("time").reset_index(drop=True)

                long_df, window_ids = _build_window_dataframe(
                    job_df, metric_cols, str(job_id), window_size, step_size
                )
                labels = _window_labels(
                    job_df, window_size, step_size, severity_threshold
                )

                all_fragments.append(long_df)
                all_window_ids.extend(window_ids)
                for col in _LABEL_COLS:
                    all_labels[col].extend(labels[col])

        if not all_fragments:
            raise ValueError("No data loaded — check labelled_csv_paths.")

        tsfresh_df = _fill_metric_nans(pd.concat(all_fragments, ignore_index=True))

        logger.info(
            "Extracting tsfresh features for %d windows…", len(all_window_ids)
        )
        X_full = extract_features(
            tsfresh_df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=self._fc_params,
            impute_function=impute,
            disable_progressbar=False,
        )
        # Reindex to guarantee row order matches all_window_ids
        X_full = X_full.reindex(all_window_ids)

        self._models = {}
        self._feature_cols = {}

        for col in _LABEL_COLS:
            y = pd.Series(all_labels[col], index=all_window_ids, dtype=float)

            # Drop windows with unknown labels
            valid_mask = ~y.isna()
            y_clean = y[valid_mask].astype(int)
            X_clean = X_full.loc[y_clean.index]

            n_pos = int(y_clean.sum())
            n_neg = int((y_clean == 0).sum())
            logger.info(
                "  %s — %d windows (%d pos, %d neg)",
                col, len(y_clean), n_pos, n_neg,
            )

            if y_clean.nunique() < 2:
                logger.warning(
                    "  Skipping %s — only one class present in training data.", col
                )
                continue

            # Feature selection — tsfresh FDR test first, then SelectKBest fallback.
            try:
                X_selected = select_features(X_clean, y_clean)
            except Exception as exc:
                logger.warning(
                    "  tsfresh select_features failed for %s (%s); falling back to SelectKBest.", col, exc
                )
                X_selected = pd.DataFrame()

            if X_selected.shape[1] == 0:
                k = min(_FALLBACK_K_FEATURES, X_clean.shape[1])
                logger.warning(
                    "  tsfresh selected 0 features for %s; using SelectKBest(k=%d).", col, k
                )
                selector = SelectKBest(f_classif, k=k)
                selector.fit(X_clean, y_clean)
                X_selected = X_clean.iloc[:, selector.get_support(indices=True)]

            self._feature_cols[col] = X_selected.columns.tolist()

            clf = clone(self._classifier)
            clf.fit(X_selected, y_clean)
            self._models[col] = clf
            logger.info(
                "  Trained %s for %s (%d features).",
                type(clf).__name__, col, X_selected.shape[1],
            )

        if not self._models:
            raise RuntimeError(
                "No classifiers were trained. "
                "Check that the labelled CSVs contain at least two classes for "
                "at least one BottleneckType."
            )

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

        # Drop label columns if they happen to be present (e.g. from labelled CSV)
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
            n_jobs=1,  # avoid multiprocessing pool teardown errors
        )

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        result: dict[str, float] = {}
        for col, clf in self._models.items():
            feature_cols = self._feature_cols[col]
            X_aligned = X.reindex(columns=feature_cols, fill_value=0.0)
            prob = float(clf.predict_proba(X_aligned)[0, 1])
            result[col] = prob

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save models and feature columns to *path* (a ``.pkl`` file).

        Args:
            path: Destination file path (created including parent dirs).
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self._classifier,
                "models": self._models,
                "feature_cols": self._feature_cols,
                "fc_params": self._fc_params,
            },
            out,
        )
        logger.info("Backend saved to %s", out)

    @classmethod
    def load(cls, path: str) -> "DefaultBackend":
        """
        Restore a backend saved with :meth:`save`.

        Args:
            path: Path previously passed to :meth:`save`.

        Returns:
            Fully restored :class:`DefaultBackend` ready for inference.
        """
        data = joblib.load(path)
        backend = cls(classifier=data.get("classifier", _DEFAULT_CLASSIFIER))
        backend._models = data["models"]
        backend._feature_cols = data["feature_cols"]
        backend._fc_params = data.get("fc_params", BASIC_FC_PARAMETERS)
        logger.info(
            "Backend loaded from %s (%d classifiers).", path, len(backend._models)
        )
        return backend
