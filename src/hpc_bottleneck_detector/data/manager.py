"""
Data Manager Module

This module provides the DataManager class that serves as the main interface
for accessing job metrics data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from .job_context import JobContext


class DataManager:
    """
    DataManager for accessing job metrics.

    Attributes:
        job_data:    DataFrame containing all job metrics (time-series rows).
        job_id:      The job identifier extracted from the data.
        job_context: Optional :class:`JobContext` carrying static hardware
                     metadata (benchmarks, CPU/memory specs, job runtime …).
                     ``None`` when the data source cannot provide it.
    """

    def __init__(self, job_data: pd.DataFrame, job_context: Optional[JobContext] = None):
        """
        Initialize the DataManager with job data.

        Args:
            job_data:    DataFrame with columns: jobId, group, metric, trace,
                         interval 0, interval 1, …
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
    
    def get_time_series_length(self) -> int:
        """
        Get the length of the time series (number of intervals).
        
        Returns:
            Number of time intervals
        """
        interval_cols = [col for col in self.job_data.columns if col.startswith('interval ')]
        return len(interval_cols)
    
    def get_all_time_series(self) -> pd.DataFrame:
        """
        Get all metrics as a DataFrame with one row per metric.
        
        Returns:
            DataFrame where each row is a metric, with columns:
                - group, metric, trace (identifiers)
                - interval 0, interval 1, ... (time series values)
        """
        return self.job_data.copy()
