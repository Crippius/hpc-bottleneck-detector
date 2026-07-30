"""
CSV Data Source Implementation

This module provides a CSV-based data source for reading HPC job metrics
from CSV files (e.g., exported from XBAT).
"""

import csv
import io
from pathlib import Path

import pandas as pd

from .interface import IDataSource
from ._load_imbalance import intra_node_imbalance_row
from ..data.manager import DataManager


class CSVDataSource(IDataSource):
    """
    Reads job metrics from CSV files exported from XBAT, with the format
    ``jobId, group, metric, trace, interval 0, interval 1, ..., interval N``.
    """
    
    def __init__(self, file_path: str, delimiter: str = ','):
        """Raises FileNotFoundError if file_path doesn't exist."""
        self.file_path = Path(file_path)
        self.delimiter = delimiter
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

    def _read_csv_robust(self) -> pd.DataFrame:
        """
        Parse XBAT CSV robustly across schema differences.
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
        Fetch job metrics data from the CSV file for job_id. Raises ValueError
        if not found in the CSV, IOError if the CSV can't be read.
        """
        try:
            # Read the CSV file (robust against XBAT export format drifts)
            df = self._read_csv_robust()
            
            # Filter by job ID
            job_data = df[df['jobId'].astype(str) == str(job_id)]
            
            if job_data.empty:
                raise ValueError(f"Job ID '{job_id}' not found in {self.file_path}")
            
            # Compute intra-node imbalance from core-level traces if present.
            intra_row = intra_node_imbalance_row(str(job_id), job_data)
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
