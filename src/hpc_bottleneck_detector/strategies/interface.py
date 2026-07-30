"""
Analysis Strategy Interface

All concrete strategies (heuristic, supervised-ML) must implement
:class:`IAnalysisStrategy`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..data.manager import DataManager
    from ..output.models import Diagnosis


class IAnalysisStrategy(ABC):
    """
    Interface for bottleneck-detection strategies.

    The orchestrator calls :meth:`diagnose` once per analysis window,
    passing a :class:`~hpc_bottleneck_detector.data.manager.DataManager`
    that contains only the intervals belonging to that window.
    """

    @abstractmethod
    def diagnose(self, data_mgr: "DataManager") -> "List[Diagnosis]":
        """
        Analyse a single window and return a list of
        :class:`~hpc_bottleneck_detector.output.models.Diagnosis` objects — a
        single ``NONE`` diagnosis (or an empty list) if no bottleneck is detected.
        """
        ...

    @abstractmethod
    def get_required_metrics(self) -> List[str]:
        """
        Return the list of metric keys needed by this strategy.

        The orchestrator may use this to validate that the data source
        provides all required metrics before starting the pipeline.
        """
        ...
