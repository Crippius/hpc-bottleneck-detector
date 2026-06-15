"""
Shared feature-extraction helpers used across training and evaluation scripts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backends.default_backend import (
    BASIC_FC_PARAMETERS,
    _LABEL_COLS,
    _NON_METRIC_COLS,
    _build_window_dataframe,
    _fill_metric_nans,
    _window_labels,
)


def find_labelled_csvs(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.rglob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No labelled CSVs found in '{data_dir}'.")
    return paths


def extract_features_for_app(
    csv_path: Path,
    window_size: int,
    step_size: int,
    severity_threshold: float,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Extract tsfresh features and labels for a single labelled CSV."""
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute

    df = pd.read_csv(csv_path)
    metric_cols = [c for c in df.columns if c not in _NON_METRIC_COLS]

    all_fragments:  list[pd.DataFrame] = []
    all_window_ids: list[str] = []
    all_labels: dict[str, list] = {col: [] for col in _LABEL_COLS}

    for job_id, job_df in df.groupby("id"):
        job_df    = job_df.sort_values("time").reset_index(drop=True)
        unique_id = f"{csv_path.stem}__{job_id}"

        long_df, window_ids = _build_window_dataframe(
            job_df, metric_cols, unique_id, window_size, step_size
        )
        labels = _window_labels(job_df, window_size, step_size, severity_threshold)

        all_fragments.append(long_df)
        all_window_ids.extend(window_ids)
        for col in _LABEL_COLS:
            all_labels[col].extend(labels[col])

    tsfresh_df = _fill_metric_nans(pd.concat(all_fragments, ignore_index=True))

    X = extract_features(
        tsfresh_df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=BASIC_FC_PARAMETERS,
        impute_function=impute,
        disable_progressbar=True,
    )
    X = X.reindex(all_window_ids)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    y_dict: dict[str, pd.Series] = {}
    for col in _LABEL_COLS:
        y = pd.Series(all_labels[col], index=all_window_ids, dtype=float)
        valid = ~y.isna()
        if valid.sum() > 0:
            y_dict[col] = y[valid].astype(int)

    return X, y_dict
