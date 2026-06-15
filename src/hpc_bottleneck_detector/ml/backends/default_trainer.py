"""
Default Trainer

Implements :class:`IMLTrainer` using tsfresh for feature extraction/selection
and a configurable scikit-learn classifier.  Produces a :class:`DefaultBackend`
ready for inference.
"""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import GroupKFold

from ..backend_interface import IMLTrainer
from .default_backend import (
    DefaultBackend,
    _DEFAULT_CLASSIFIER,
    EXCLUDE_METRIC_COLS,
    EXCLUDE_METRIC_PREFIXES,
    _FALLBACK_K_FEATURES,
    _FDR_LEVEL,
    _IMPORTANCE_THRESHOLD,
    _LABEL_COLS,
    _NON_METRIC_COLS,
    _build_window_dataframe,
    _f1_score,
    _fill_metric_nans,
    _merge_app_y,
    _window_labels,
)
import yaml

from .config import BASIC_FC_PARAMETERS

_PARAM_GRIDS_PATH = Path(__file__).parents[4] / "configs" / "param_grids.yaml"


def _load_param_grids() -> dict:
    with open(_PARAM_GRIDS_PATH) as f:
        return yaml.safe_load(f)

logger = logging.getLogger(__name__)


def _sample_params(grid: dict, n_iter: int, rng: np.random.Generator) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    all_combos = list(itertools.product(*[grid[k] for k in keys]))
    n_sample = min(n_iter, len(all_combos))
    indices = rng.choice(len(all_combos), size=n_sample, replace=False)
    return [dict(zip(keys, all_combos[i])) for i in indices]


class DefaultTrainer(IMLTrainer):
    """
    Trainer for :class:`DefaultBackend`.

    Holds training configuration and produces a fitted backend via
    :meth:`train` or :meth:`from_preextracted_features`.

    Args:
        classifier: Any sklearn-compatible classifier with ``predict_proba()``.
                    A fresh clone is fitted per ``BottleneckType``.
        use_fdr:    Apply tsfresh FDR feature selection during training.
        use_importance_pruning: Drop features below the RF importance threshold.
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

    # ------------------------------------------------------------------
    # IMLTrainer
    # ------------------------------------------------------------------

    def train(
        self,
        labelled_csv_paths: list[str],
        window_size: int,
        step_size: int,
        severity_threshold: float = 0.0,
    ) -> DefaultBackend:
        """Train one classifier per ``BottleneckType`` and return a fitted backend."""
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import impute
        from sklearn.feature_selection import SelectKBest, f_classif

        all_fragments: list[pd.DataFrame] = []
        all_window_ids: list[str] = []
        all_labels: dict[str, list[Optional[float]]] = {col: [] for col in _LABEL_COLS}

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
                labels = _window_labels(job_df, window_size, step_size, severity_threshold)

                all_fragments.append(long_df)
                all_window_ids.extend(window_ids)
                for col in _LABEL_COLS:
                    all_labels[col].extend(labels[col])

        if not all_fragments:
            raise ValueError("No data loaded - check labelled_csv_paths.")

        tsfresh_df = _fill_metric_nans(pd.concat(all_fragments, ignore_index=True))

        logger.info("Extracting tsfresh features for %d windows...", len(all_window_ids))
        X_full = extract_features(
            tsfresh_df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=BASIC_FC_PARAMETERS,
            impute_function=impute,
            disable_progressbar=False,
        )
        X_full = X_full.reindex(all_window_ids)

        backend = DefaultBackend()
        backend._window_size = window_size

        for col in _LABEL_COLS:
            y = pd.Series(all_labels[col], index=all_window_ids, dtype=float)

            valid_mask = ~y.isna()
            y_clean = y[valid_mask].astype(int)
            X_clean = X_full.loc[y_clean.index]

            n_pos = int(y_clean.sum())
            n_neg = int((y_clean == 0).sum())
            logger.info(
                "  %s - %d windows (%d pos, %d neg)", col, len(y_clean), n_pos, n_neg,
            )

            if y_clean.nunique() < 2:
                logger.warning("  Skipping %s - only one class present.", col)
                continue

            X_selected = self._select_features(X_clean, y_clean, col, SelectKBest, f_classif)
            if X_selected is None or X_selected.shape[1] == 0:
                continue

            backend._feature_cols[col] = X_selected.columns.tolist()
            clf = clone(self._classifier)
            clf.fit(X_selected, y_clean)
            backend._models[col] = clf
            logger.info(
                "  Trained %s for %s (%d features).",
                type(clf).__name__, col, X_selected.shape[1],
            )

        if not backend._models:
            raise RuntimeError(
                "No classifiers were trained. "
                "Check that the labelled CSVs contain at least two classes for "
                "at least one BottleneckType."
            )
        return backend

    # ------------------------------------------------------------------
    # Alternative constructor: fit on pre-extracted features
    # ------------------------------------------------------------------

    def from_preextracted_features(
        self,
        X_train: pd.DataFrame,
        y_dict_train: dict[str, pd.Series],
    ) -> DefaultBackend:
        """
        Build a fitted backend from already-extracted tsfresh features,
        skipping the extraction step.
        """
        from sklearn.feature_selection import SelectKBest, f_classif

        backend = DefaultBackend()

        for col in _LABEL_COLS:
            if col not in y_dict_train:
                continue
            y = y_dict_train[col]
            X_clean = X_train.reindex(index=y.index, fill_value=0.0).fillna(0.0)

            if y.nunique() < 2:
                continue

            X_selected = self._select_features(X_clean, y, col, SelectKBest, f_classif)
            if X_selected is None or X_selected.shape[1] == 0:
                logger.warning("  No features left for %s after selection; skipping.", col)
                continue

            backend._feature_cols[col] = X_selected.columns.tolist()
            clf = clone(self._classifier)
            clf.fit(X_selected, y)
            backend._models[col] = clf

        return backend

    # ------------------------------------------------------------------
    # Threshold calibration via GroupKFold CV
    # ------------------------------------------------------------------

    def calibrate_thresholds_cv(
        self,
        app_features: list[tuple[pd.DataFrame, dict[str, pd.Series]]],
        n_splits: int = 5,
        default_threshold: float = 0.5,
    ) -> dict[str, float]:
        """GroupKFold CV over apps to calibrate per-class probability thresholds."""
        n_folds = min(n_splits, len(app_features))
        gkf = GroupKFold(n_splits=n_folds)
        indices = np.arange(len(app_features))
        thr_lists: dict[str, list[float]] = {col: [] for col in _LABEL_COLS}

        for tr_idx, va_idx in gkf.split(indices, groups=indices):
            X_tr = pd.concat([app_features[i][0] for i in tr_idx]).fillna(0.0)
            y_tr = _merge_app_y([app_features[i][1] for i in tr_idx])
            X_va = pd.concat([app_features[i][0] for i in va_idx]).fillna(0.0)
            y_va = _merge_app_y([app_features[i][1] for i in va_idx])

            fold_backend = self.from_preextracted_features(X_tr, y_tr)
            fold_backend.calibrate_thresholds(X_va, y_va)
            for col in _LABEL_COLS:
                thr = fold_backend._thresholds.get(col)
                if thr is not None:
                    thr_lists[col].append(thr)

        return {
            col: float(np.nanmean(v)) if v else default_threshold
            for col, v in thr_lists.items()
        }

    # ------------------------------------------------------------------
    # Joint hyperparameter + threshold CV tuning (optional, expensive)
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

        Returns the best ``(classifier, thresholds)`` pair.
        """
        clf_name = type(classifier).__name__
        if param_grid is None:
            param_grid = _load_param_grids().get(clf_name, {})
            if not param_grid:
                logger.warning(
                    "No param grid found for %s - running threshold calibration only.",
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
                trainer_fold = cls(classifier=clf)
                backend_fold = trainer_fold.from_preextracted_features(X_train, y_dict_train)
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
            logger.info("Params %s -> CV score=%.4f", params, score)

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
    # Internal: shared feature selection logic
    # ------------------------------------------------------------------

    def _select_features(
        self,
        X_clean: pd.DataFrame,
        y_clean: pd.Series,
        col: str,
        SelectKBest,
        f_classif,
    ) -> pd.DataFrame | None:
        """Apply FDR test and/or importance pruning; return selected feature DataFrame."""
        from tsfresh import select_features

        X_selected = X_clean

        if self._use_fdr:
            try:
                X_selected = select_features(X_clean, y_clean, fdr_level=_FDR_LEVEL)
            except Exception as exc:
                logger.warning(
                    "  tsfresh select_features failed for %s (%s); falling back to SelectKBest.",
                    col, exc,
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

            logger.info("  FDR: %d -> %d features.", X_clean.shape[1], X_selected.shape[1])

        if self._use_importance_pruning and X_selected.shape[1] > 0:
            prelim_clf = clone(self._classifier)
            prelim_clf.fit(X_selected, y_clean)
            mask = prelim_clf.feature_importances_ >= _IMPORTANCE_THRESHOLD
            n_before = X_selected.shape[1]
            X_selected = X_selected.loc[:, mask]
            logger.info(
                "  Importance pruning: %d -> %d features (threshold=%.2e).",
                n_before, X_selected.shape[1], _IMPORTANCE_THRESHOLD,
            )

        return X_selected
