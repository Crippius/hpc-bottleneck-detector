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

import itertools
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
from sklearn.model_selection import GroupKFold
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

# FDR significance level for tsfresh feature selection. Lower (0.01) = stricter, Higher (0.20) = lenient. Default tsfresh value is 0.05
_FDR_LEVEL = 0.01
# Number of features to keep when tsfresh's FDR test selects nothing.
_FALLBACK_K_FEATURES = 100
# Minimum RF feature importance to keep after FDR selection.
_IMPORTANCE_THRESHOLD = 1e-6

from ..backend_interface import IMLBackend
from ...utils.labeling import BOTTLENECK_COLUMNS
from .config import BASIC_FC_PARAMETERS, _PARAM_GRIDS

logger = logging.getLogger(__name__)

# Names of the label columns as they appear in labelled CSVs.
_LABEL_COLS = [bt.value for bt in BOTTLENECK_COLUMNS]
# Columns that are never metric features.
_NON_METRIC_COLS = {"id", "time"} | set(_LABEL_COLS)
# Metric column prefixes to exclude from feature extraction.
EXCLUDE_METRIC_PREFIXES: tuple[str, ...] = ("gpu_",)
# Exact metric column names to exclude from feature extraction.
EXCLUDE_METRIC_COLS: frozenset[str] = frozenset({"INTER_NODE_LOAD_IMBALANCE"})

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
    - ``1``   - max severity in window > threshold (bottleneck present)
    - ``0``   - all real severities ≤ threshold (no bottleneck)
    - ``NaN`` - every interval in the window had NaN label (unknown)
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


def _sample_params(grid: dict, n_iter: int, rng: np.random.Generator) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    all_combos = list(itertools.product(*[grid[k] for k in keys]))
    n_sample = min(n_iter, len(all_combos))
    indices = rng.choice(len(all_combos), size=n_sample, replace=False)
    return [dict(zip(keys, all_combos[i])) for i in indices]


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
        backend.train(csv_paths, window_size=12, step_size=12)

    Attributes:
        _classifier:   Unfitted prototype classifier (cloned for each type).
        _models:       ``{bt_name: fitted classifier}``
        _feature_cols: ``{bt_name: list of selected feature column names}``
        _fc_params:    tsfresh feature-calculation parameters.
    """

    def __init__(
        self,
        classifier: ClassifierMixin = _DEFAULT_CLASSIFIER,
        use_fdr: bool = True,
        use_importance_pruning: bool = True,
    ) -> None:
        self._classifier = classifier
        self._use_fdr = use_fdr
        self._use_importance_pruning = use_importance_pruning
        self._models: dict[str, ClassifierMixin] = {}
        self._feature_cols: dict[str, list[str]] = {}
        self._thresholds: dict[str, float] = {}
        self._fc_params = BASIC_FC_PARAMETERS
        self._window_size: Optional[int] = None

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

            metric_cols = [
                c for c in df.columns
                if c not in _NON_METRIC_COLS
                and not any(c.startswith(p) for p in EXCLUDE_METRIC_PREFIXES)
                and c not in EXCLUDE_METRIC_COLS
            ]

            for job_id, job_df in df.groupby("id"):
                job_df = job_df.sort_values("time").dropna(subset=metric_cols).reset_index(drop=True)

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

        self._window_size = window_size

        if not all_fragments:
            raise ValueError("No data loaded - check labelled_csv_paths.")

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
                "  %s - %d windows (%d pos, %d neg)",
                col, len(y_clean), n_pos, n_neg,
            )

            if y_clean.nunique() < 2:
                logger.warning(
                    "  Skipping %s - only one class present in training data.", col
                )
                continue

            X_selected = X_clean

            # tsfresh FDR test: remove statistically irrelevant features.
            if self._use_fdr:
                try:
                    X_selected = select_features(X_clean, y_clean, fdr_level=_FDR_LEVEL)
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

                logger.info("  FDR: %d → %d features.", X_clean.shape[1], X_selected.shape[1])

            # importance-based pruning: fit a preliminary clf,
            # drop features below _IMPORTANCE_THRESHOLD, then refit on pruned set.
            if self._use_importance_pruning:
                prelim_clf = clone(self._classifier)
                prelim_clf.fit(X_selected, y_clean)
                mask = prelim_clf.feature_importances_ >= _IMPORTANCE_THRESHOLD
                n_before = X_selected.shape[1]
                X_selected = X_selected.loc[:, mask]
                logger.info(
                    "  Importance pruning: %d → %d features (threshold=%.2e).",
                    n_before, X_selected.shape[1], _IMPORTANCE_THRESHOLD,
                )

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
    # Alternative constructor: fit on pre-extracted features
    # ------------------------------------------------------------------

    @classmethod
    def from_preextracted_features(
        cls,
        X_train: pd.DataFrame,
        y_dict_train: dict[str, pd.Series],
        classifier: ClassifierMixin = _DEFAULT_CLASSIFIER,
        use_fdr: bool = True,
        use_importance_pruning: bool = True,
    ) -> "DefaultBackend":
        """
        Build a trained backend from already-extracted tsfresh features,
        skipping the extraction step entirely.
        """
        backend = cls(
            classifier=classifier,
            use_fdr=use_fdr,
            use_importance_pruning=use_importance_pruning,
        )

        for col in _LABEL_COLS:
            if col not in y_dict_train:
                continue
            y = y_dict_train[col]
            X_clean = X_train.reindex(index=y.index, fill_value=0.0).fillna(0.0)

            if y.nunique() < 2:
                continue

            X_selected = X_clean

            if backend._use_fdr:
                try:
                    X_selected = select_features(X_clean, y, fdr_level=_FDR_LEVEL)
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
                    selector.fit(X_clean, y)
                    X_selected = X_clean.iloc[:, selector.get_support(indices=True)]

            if backend._use_importance_pruning and X_selected.shape[1] > 0:
                prelim_clf = clone(backend._classifier)
                prelim_clf.fit(X_selected, y)
                mask = prelim_clf.feature_importances_ >= _IMPORTANCE_THRESHOLD
                n_before = X_selected.shape[1]
                X_selected = X_selected.loc[:, mask]
                logger.info(
                    "  Importance pruning: %d → %d features (threshold=%.2e).",
                    n_before, X_selected.shape[1], _IMPORTANCE_THRESHOLD,
                )

            if X_selected.shape[1] == 0:
                logger.warning("  No features left for %s after selection; skipping.", col)
                continue

            backend._feature_cols[col] = X_selected.columns.tolist()
            clf = clone(backend._classifier)
            clf.fit(X_selected, y)
            backend._models[col] = clf

        return backend

    # ------------------------------------------------------------------
    # Threshold calibration
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
                self._thresholds.setdefault(col, 0.5)
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
    # Joint hyperparameter + threshold CV tuning (optional)
    # ------------------------------------------------------------------

    @classmethod
    def tune(
        cls,
        app_features: list[tuple[pd.DataFrame, dict[str, pd.Series]]],
        classifier: ClassifierMixin = _DEFAULT_CLASSIFIER,
        param_grid: dict | None = None,
        n_splits: int = 5,
        n_iter: int = 20,
        threshold_grid: list[float] | None = None,
        seed: int = 42,
    ) -> tuple[ClassifierMixin, dict[str, float]]:
        """
        Jointly search for the best classifier hyperparameters and probability
        thresholds via GroupKFold CV (one group = one application).

        This is computationally expensive (n_iter x n_splits x per-label fits)
        and is intended as an optional step before the main training run.
        """
        clf_name = type(classifier).__name__
        if param_grid is None:
            param_grid = _PARAM_GRIDS.get(clf_name, {})
            if not param_grid:
                logger.warning(
                    "No param grid found for %s — running threshold calibration only.",
                    clf_name,
                )

        rng = np.random.default_rng(seed)
        sampled = _sample_params(param_grid, n_iter, rng)

        n_apps = len(app_features)
        app_indices = np.arange(n_apps)
        n_folds = min(n_splits, n_apps)
        gkf = GroupKFold(n_splits=n_folds)

        best_score = -1.0
        best_params: dict = sampled[0] if sampled else {}
        best_thresholds: dict[str, float] = {}

        for params in sampled:
            fold_scores: list[float] = []
            fold_thr_lists: dict[str, list[float]] = {col: [] for col in _LABEL_COLS}

            for train_idx, val_idx in gkf.split(app_indices, groups=app_indices):
                X_train = pd.concat(
                    [app_features[i][0] for i in train_idx]
                ).fillna(0.0)
                y_dict_train = _merge_app_y([app_features[i][1] for i in train_idx])
                X_val = pd.concat(
                    [app_features[i][0] for i in val_idx]
                ).fillna(0.0)
                y_dict_val = _merge_app_y([app_features[i][1] for i in val_idx])

                clf = clone(classifier).set_params(**params) if params else clone(classifier)
                backend_fold = cls.from_preextracted_features(
                    X_train, y_dict_train, clf
                )
                if not backend_fold._models:
                    continue

                backend_fold.calibrate_thresholds(X_val, y_dict_val, threshold_grid)

                f1_vals: list[float] = []
                for col in backend_fold._models:
                    if col not in y_dict_val:
                        continue
                    y_true = y_dict_val[col].values
                    X_aligned = X_val.reindex(
                        index=y_dict_val[col].index,
                        columns=backend_fold._feature_cols[col],
                        fill_value=0.0,
                    )
                    probs = backend_fold._models[col].predict_proba(X_aligned)[:, 1]
                    thr = backend_fold._thresholds.get(col, 0.5)
                    f1 = _f1_score(y_true, (probs >= thr).astype(int))
                    if not math.isnan(f1):
                        f1_vals.append(f1)

                if f1_vals:
                    fold_scores.append(float(np.mean(f1_vals)))
                for col in _LABEL_COLS:
                    thr = backend_fold._thresholds.get(col)
                    if thr is not None:
                        fold_thr_lists[col].append(thr)

            score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
            logger.info("Params %s → CV score=%.4f", params, score)

            if not math.isnan(score) and score > best_score:
                best_score = score
                best_params = params
                best_thresholds = {
                    col: float(np.nanmean(v)) if v else 0.5
                    for col, v in fold_thr_lists.items()
                }

        logger.info("Best params: %s (CV score=%.4f)", best_params, best_score)
        tuned_clf = clone(classifier).set_params(**best_params) if best_params else clone(classifier)
        return tuned_clf, best_thresholds

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
                "use_fdr": self._use_fdr,
                "use_importance_pruning": self._use_importance_pruning,
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

        Args:
            path: Path previously passed to :meth:`save`.

        Returns:
            Fully restored :class:`DefaultBackend` ready for inference.
        """
        data = joblib.load(path)
        backend = cls(
            classifier=data.get("classifier", _DEFAULT_CLASSIFIER),
            use_fdr=data.get("use_fdr", True),
            use_importance_pruning=data.get("use_importance_pruning", True),
        )
        backend._models = data["models"]
        backend._feature_cols = data["feature_cols"]
        backend._thresholds = data.get("thresholds", {})
        backend._fc_params = data.get("fc_params", BASIC_FC_PARAMETERS)
        backend._window_size = data.get("window_size")
        logger.info(
            "Backend loaded from %s (%d classifiers).", path, len(backend._models)
        )
        return backend
