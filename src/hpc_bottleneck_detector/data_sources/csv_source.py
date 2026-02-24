"""
CSV Data Source Implementation

This module provides a CSV-based data source for reading HPC job metrics
from CSV files (e.g., exported from XBAT).
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from .interface import IDataSource
from ..data.manager import DataManager


class CSVDataSource(IDataSource):
    """
    Data source implementation that reads job metrics from CSV files.

    This is designed to work with CSV files exported from XBAT that have
    the format:
        jobId, group, metric, trace, interval 0, interval 1, ..., interval N

    .. note::
        CSV files do not carry hardware metadata, so the
        :attr:`~hpc_bottleneck_detector.data.manager.DataManager.job_context`
        attribute of the returned :class:`DataManager` is always ``None``.
        Use :class:`~hpc_bottleneck_detector.data_sources.xbat_source.XBATDataSource`
        to obtain a :class:`~hpc_bottleneck_detector.data.job_context.JobContext`
        populated with hardware benchmarks and node specs.

    Attributes:
        file_path: Path to the CSV file
        delimiter: Delimiter used in the CSV file (default: ',')
    """
    
    def __init__(self, file_path: str, delimiter: str = ','):
        """
        Initialize the CSV data source.
        
        Args:
            file_path: Path to the CSV file containing job metrics
            delimiter: CSV delimiter (default: ',')
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
        """
        self.file_path = Path(file_path)
        self.delimiter = delimiter
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    def fetch_job_data(self, job_id: str) -> DataManager:
        """
        Fetch job metrics data from the CSV file for a given job ID.

        Args:
            job_id: The unique identifier for the job

        Returns:
            :class:`DataManager` instance containing all metrics for the
            specified job.  The ``job_context`` attribute is ``None`` because
            CSV files do not contain hardware or job-execution metadata.

        Raises:
            ValueError: If job_id is not found in the CSV
            IOError: If the CSV cannot be read
        """
        try:
            # Read the CSV file
            df = pd.read_csv(self.file_path, delimiter=self.delimiter)
            
            # Filter by job ID
            job_data = df[df['jobId'].astype(str) == str(job_id)]
            
            if job_data.empty:
                raise ValueError(f"Job ID '{job_id}' not found in {self.file_path}")
            
            # Create and return a DataManager with the job data
            return DataManager(job_data.reset_index(drop=True))
            
        except pd.errors.ParserError as e:
            raise IOError(f"Failed to parse CSV file {self.file_path}: {e}")
        except Exception as e:
            raise IOError(f"Failed to read CSV file {self.file_path}: {e}")
