"""
Data Manager Module

This module provides the DataManager class that serves as the main interface
for accessing job metrics data.
"""

import logging

import pandas as pd
import numpy as np
from typing import Dict, Generator, List, Optional, Tuple, Union

from .job_context import JobContext

logger = logging.getLogger(__name__)

# Maps flat-DataFrame column names to benchmark keys in JobContext.
# Columns present in this map are divided by the corresponding peak value
# when a JobContext is available.  Adjust keys to match your XBAT instance.
METRIC_BENCHMARK_MAP: dict[str, str] = {
    "cpu_FLOPS_SP":                  "peakflops_sp",
    "cpu_FLOPS_DP":                  "peakflops_dp",
    "cpu_FLOPS_AVX_SP":              "peakflops_avx_sp",
    "cpu_FLOPS_AVX_DP":              "peakflops_avx_dp",
    "cpu_FLOPS_AVX512_SP":           "peakflops_avx512_sp",
    "cpu_FLOPS_AVX512_DP":           "peakflops_avx512_dp",
    "memory_Bandwidth_total":        "bandwidth_mem",
    "memory_Bandwidth_read":         "bandwidth_mem",
    "memory_Bandwidth_write":        "bandwidth_mem",
    "memory_UPI Bandwidth_total":    "bandwidth_upi",
}


class DataManager:
    """
    DataManager for accessing job metrics.

    Attributes:
        job_data:    DataFrame containing all job metrics (time-series rows).
        job_id:      The job identifier extracted from the data.
        job_context: Optional :class:`JobContext` carrying static hardware
                     metadata (benchmarks, CPU/memory specs, job runtime ...).
                     ``None`` when the data source cannot provide it.
    """

    def __init__(self, job_data: pd.DataFrame, job_context: Optional[JobContext] = None):
        """
        Initialize the DataManager with job data.

        Args:
            job_data:    DataFrame with columns: jobId, group, metric, trace,
                         interval 0, interval 1, ...
            job_context: Optional static job / hardware context.
        """
        self.job_data = job_data
        self.job_context = job_context

        # Extract job ID from the data
        if not job_data.empty:
            self.job_id = str(job_data['jobId'].iloc[0])
        else:
            self.job_id = None
    
    def get_metric(self, group: str, metric: str, trace: Optional[str] = None) -> pd.Series:
        """
        Get time series data for a specific metric.
        
        Args:
            group: Metric group (e.g., 'cpu', 'memory')
            metric: Metric name (e.g., 'Branching')
            trace: Trace name (optional, e.g., 'branch rate')
            
        Returns:
            Series containing the time series values (interval 0, interval 1, ...)
            
        Raises:
            ValueError: If the metric is not found
        """
        # Build filter conditions
        condition = (self.job_data['group'] == group) & (self.job_data['metric'] == metric)
        
        if trace is not None:
            condition = condition & (self.job_data['trace'] == trace)
        
        # Filter the data
        metric_data = self.job_data[condition]
        
        if metric_data.empty:
            raise ValueError(f"Metric not found: group='{group}', metric='{metric}', trace='{trace}'")
        
        # Extract interval columns
        interval_cols = [col for col in self.job_data.columns if col.startswith('interval ')]
        
        # Return the first matching row (should be only one)
        return metric_data[interval_cols].iloc[0]
    
    def get_metrics(self, metric_specs: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Get multiple metrics at once.
        
        Args:
            metric_specs: List of dicts with keys 'group', 'metric', and optionally 'trace'
                Example: [{'group': 'cpu', 'metric': 'Branching', 'trace': 'branch rate'}]
            
        Returns:
            DataFrame where each row is a metric time series, indexed by descriptive names
        """
        result = {}
        
        for spec in metric_specs:
            group = spec['group']
            metric = spec['metric']
            trace = spec.get('trace')
            
            try:
                series = self.get_metric(group, metric, trace)
                
                # Create a descriptive key
                key = f"{group}_{metric}"
                if trace:
                    key += f"_{trace.replace(' ', '_')}"
                
                result[key] = series
            except ValueError:
                # Skip metrics that are not found
                continue
        
        return pd.DataFrame(result).T
    
    def list_available_metrics(self) -> pd.DataFrame:
        """
        List all available metrics in the job data.
        
        Returns:
            DataFrame with columns: group, metric, trace
        """
        return self.job_data[['group', 'metric', 'trace']].drop_duplicates().reset_index(drop=True)

    def has_metric(self, group: str, metric: str, trace: Optional[str] = None) -> bool:
        """Return True when the requested metric exists in the job data."""
        try:
            self.get_metric(group, metric, trace)
            return True
        except ValueError:
            return False
    
    def get_time_series_length(self) -> int:
        """
        Get the length of the time series (number of intervals).

        Returns:
            Number of time intervals
        """
        interval_cols = [col for col in self.job_data.columns if col.startswith('interval ')]
        return len(interval_cols)

    @property
    def sampling_interval(self) -> Optional[int]:
        """
        Infer the sampling period in seconds from job context.

        Uses ``round(runtime_seconds / n_intervals)`` so a small trailing gap
        between the last captured metric and the actual job end does not skew
        the result.  Returns ``None`` when no job context is available or when
        the runtime cannot be determined.

        ``runtime`` from XBAT is a ``"H:MM:SS"`` string; it is parsed to
        seconds before dividing.
        """
        if self.job_context is None:
            return None
        runtime = self.job_context.get_metadata("runtime")
        if runtime is None:
            return None
        n = self.get_time_series_length()
        if n == 0:
            return None
        if isinstance(runtime, str):
            parts = runtime.split(":")
            try:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                runtime = h * 3600 + m * 60 + s
            except (ValueError, IndexError):
                return None
        return round(runtime / n)
    
    def get_all_time_series(self) -> pd.DataFrame:
        """
        Get all metrics as a DataFrame with one row per metric.
        
        Returns:
            DataFrame where each row is a metric, with columns:
                - group, metric, trace (identifiers)
                - interval 0, interval 1, ... (time series values)
        """
        return self.job_data.copy()

    def get_flat_dataframe(self, interval_seconds: Optional[int] = None) -> pd.DataFrame:
        """
        Return a flat (wide) DataFrame version of data.

        The returned DataFrame has one row per time interval and the
        following columns:

        - ``id``   : job identifier (same value for every row).
        - ``time`` : elapsed time in seconds (0, 5, 10, ...), stepping by
                     *interval_seconds* per interval.
        - One column per metric, named ``<group>_<metric>`` or
          ``<group>_<metric>_<trace>`` (spaces replaced by underscores).

        When a :class:`JobContext` is attached, columns listed in
        :data:`METRIC_BENCHMARK_MAP` are divided by their corresponding
        hardware peak (obtained via :meth:`JobContext.get_benchmark`).
        Columns whose benchmark value is unavailable or zero are left as-is.

        To extract features with tsfresh pass::

            tsfresh.extract_features(
                df,
                column_id="id",
                column_sort="time",
            )

        Args:
            interval_seconds: Duration of each interval in seconds.  When
                              ``None`` (default), the value is inferred from
                              ``job_context.runtime`` via the
                              :attr:`sampling_interval` property.

        Raises:
            ValueError: If *interval_seconds* is ``None`` and the sampling
                        interval cannot be inferred from the job context.

        Returns:
            DataFrame with shape ``(n_intervals, 2 + n_metrics)``.
        """
        if interval_seconds is None:
            interval_seconds = self.sampling_interval
            if interval_seconds is None:
                raise ValueError(
                    "interval_seconds must be provided when no job context "
                    "with runtime information is available."
                )

        interval_cols = self._interval_columns()
        n_intervals = len(interval_cols)

        cols: dict = {
            "id":   [self.job_id] * n_intervals,
            "time": [i * interval_seconds for i in range(n_intervals)],
        }

        for _, row in self.job_data.iterrows():
            group = row["group"]
            metric = row["metric"]
            trace = row.get("trace")

            col_name = f"{group}_{metric}"
            if pd.notna(trace) and str(trace).strip():
                col_name += f"_{str(trace).replace(' ', '_')}"

            cols[col_name] = row[interval_cols].values

        df = pd.DataFrame(cols)

        if self.job_context is not None:
            df = self._apply_benchmark_normalization(df)

        return df

    def _apply_benchmark_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Divide metric columns by their hardware peak from ``job_context``.

        Only columns listed in :data:`METRIC_BENCHMARK_MAP` are touched.
        If the benchmark value is absent or zero, the column is left unchanged.
        """
        df = df.copy()
        normalized: list[str] = []
        skipped: list[str] = []

        for col, benchmark_key in METRIC_BENCHMARK_MAP.items():
            if col not in df.columns:
                continue
            peak = self.job_context.get_benchmark(benchmark_key)
            if peak and peak > 0:
                df[col] = df[col] / peak
                normalized.append(col)
            else:
                skipped.append(col)

        if normalized:
            logger.debug("Normalized by benchmark peak: %s", normalized)
        if skipped:
            logger.debug(
                "Benchmark peak unavailable for (left raw): %s", skipped
            )

        return df

    # ------------------------------------------------------------------
    # Windowing helpers
    # ------------------------------------------------------------------

    def _interval_columns(self) -> List[str]:
        """Return the ordered list of interval column names."""
        return [col for col in self.job_data.columns if col.startswith("interval ")]

    def slice_window(self, start: int, end: int) -> "DataManager":
        """
        Return a new DataManager containing only intervals [start, end).

        The interval columns in the returned instance are **renumbered**
        starting from 0 so that strategies can treat every window
        uniformly.

        Args:
            start: Inclusive start index (0-based over interval columns).
            end:   Exclusive end index.

        Returns:
            A new :class:`DataManager` scoped to the requested interval slice.
        """
        all_interval_cols = self._interval_columns()
        slice_cols = all_interval_cols[start:end]

        id_cols = [c for c in self.job_data.columns if not c.startswith("interval ")]
        sliced = self.job_data[id_cols + slice_cols].copy()

        # Renumber interval columns: interval 0, interval 1, ...
        rename_map = {
            old: f"interval {i}" for i, old in enumerate(slice_cols)
        }
        sliced.rename(columns=rename_map, inplace=True)

        return DataManager(sliced, job_context=self.job_context)

    def iterate_windows(
        self,
        window_size: int,
        step_size: int,
    ) -> Generator[Tuple[int, int, "DataManager"], None, None]:
        """
        Slide a window over the time series and yield sub-DataManagers.

        Args:
            window_size: Number of intervals per window.
            step_size:   Number of intervals to advance between windows.
                         Setting ``step_size == window_size`` gives
                         tumbling (non-overlapping) windows; smaller
                         values give sliding (overlapping) windows.

        Yields:
            Tuples of ``(start_interval, end_interval, window_data_manager)``
            where *end_interval* is the **inclusive** last interval index in
            the original time series.
        """
        n_intervals = self.get_time_series_length()

        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")

        start = 0
        while start < n_intervals:
            end = min(start + window_size, n_intervals)
            yield start, end - 1, self.slice_window(start, end)
            if end == n_intervals:
                break
            start += step_size
