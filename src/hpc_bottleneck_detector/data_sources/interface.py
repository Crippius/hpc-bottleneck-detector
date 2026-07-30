"""
Data Source Interface Module

This module defines the interface that all data sources must implement.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.manager import DataManager


class IDataSource(ABC):
    """
    Interface for data sources that provide HPC job metrics.
    
    All data source implementations must inherit from this interface
    and implement the fetch_job_data method.
    """
    
    @abstractmethod
    def fetch_job_data(self, job_id: str) -> "DataManager":
        """
        Fetch job metrics data for job_id. Implementations should raise
        ValueError if job_id is invalid/not found, IOError if data can't be
        retrieved.
        """
        pass
