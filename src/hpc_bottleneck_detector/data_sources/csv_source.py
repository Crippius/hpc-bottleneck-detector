"""
CSV Data Source Implementation

This module provides a CSV-based data source for reading HPC job metrics
from CSV files (e.g., exported from XBAT).
"""

import csv
import io
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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

    def _read_csv_robust(self) -> pd.DataFrame:
        """
        Parse XBAT CSV robustly across schema differences.

        Current production behavior can produce csv where a subset of metric
        rows contains one additional trailing interval value. Drop conservatively
        the trailing overflow values so every metric shares the same interval count.
        """
        raw_text = self.file_path.read_text()
        rows = list(csv.reader(io.StringIO(raw_text), delimiter=self.delimiter))

        if not rows:
            raise IOError(f"CSV file is empty: {self.file_path}")

        header = rows[0]
        expected_cols = len(header)
        parsed_rows: list[list[str]] = []

        for row in rows[1:]:
            if not row:
                continue

            if row[0] == "jobId":
                continue

            if len(row) > expected_cols:
                row = row[:expected_cols]
            elif len(row) < expected_cols:
                row = row + [""] * (expected_cols - len(row))

            parsed_rows.append(row)

        if not parsed_rows:
            raise IOError(f"No data rows found in CSV file: {self.file_path}")
        df = pd.DataFrame(parsed_rows, columns=header)

        interval_cols = [col for col in df.columns if col.startswith("interval ")]
        for column in interval_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        return df
    
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
            # Read the CSV file (robust against XBAT export format drifts)
            df = self._read_csv_robust()
            
            # Filter by job ID
            job_data = df[df['jobId'].astype(str) == str(job_id)]
            
            if job_data.empty:
                raise ValueError(f"Job ID '{job_id}' not found in {self.file_path}")
            
            # Compute intra-node imbalance from core-level traces if present.
            intra_row = self._intra_node_imbalance_row(str(job_id), job_data)
            if intra_row is not None:
                job_data = pd.concat(
                    [job_data, pd.DataFrame([intra_row])],
                    ignore_index=True,
                )

            return DataManager(job_data.reset_index(drop=True))

        except ValueError:
            raise
        except pd.errors.ParserError as e:
            raise IOError(f"Failed to parse CSV file {self.file_path}: {e}")
        except Exception as e:
            raise IOError(f"Failed to read CSV file {self.file_path}: {e}")

    @staticmethod
    def _intra_node_imbalance_row(
        job_id: str,
        df: pd.DataFrame,
    ) -> Optional[dict]:
        """
        Compute per-interval ``max - min`` total FLOPS/s across all cores.

        Operates on traces with the pattern ``<type> c<N>`` (e.g. ``SP c0``,
        ``AVX512 DP c3``).  Returns ``None`` when fewer than two distinct
        cores are found.
        """
        interval_cols: List[str] = [
            c for c in df.columns if c.startswith("interval ")
        ]
        flops_df = df[(df["group"] == "cpu") & (df["metric"] == "FLOPS")]
        if flops_df.empty:
            return None

        core_totals: Dict[str, np.ndarray] = {}
        for _, row in flops_df.iterrows():
            m = re.search(r"\bc(\d+)$", str(row["trace"]))
            if not m:
                continue
            core_id = m.group(0)
            values = np.zeros(len(interval_cols))
            for i, col in enumerate(interval_cols):
                v = row.get(col, 0.0)
                values[i] = float(v) if pd.notna(v) else 0.0
            if core_id not in core_totals:
                core_totals[core_id] = values.copy()
            else:
                core_totals[core_id] += values

        if len(core_totals) < 2:
            return None

        matrix = np.stack(list(core_totals.values()))  # (n_cores, n_intervals)
        imbalance = matrix.max(axis=0) - matrix.min(axis=0)

        row_dict: dict = {
            "jobId": job_id,
            "group": "load_imbalance",
            "metric": "FLOPS",
            "trace": "intra_node",
        }
        for col, val in zip(interval_cols, imbalance):
            row_dict[col] = float(val)
        return row_dict
