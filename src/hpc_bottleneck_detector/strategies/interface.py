"""
Analysis Strategy Interface

All concrete strategies (heuristic, supervised-ML, hybrid) must implement
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

    Implementations are free to run multiple internal detectors and return
    one :class:`~hpc_bottleneck_detector.output.models.Diagnosis` per
    detected bottleneck (or a single ``NONE`` diagnosis when the window
    is healthy).
    """

    @abstractmethod
    def diagnose(self, data_mgr: "DataManager") -> "List[Diagnosis]":
        """
        Analyse a single window and return a list of diagnoses.

        Args:
            data_mgr: DataManager scoped to the current analysis window.

        Returns:
            A list of :class:`~hpc_bottleneck_detector.output.models.Diagnosis`
            objects.  Return a list with a single ``NONE`` diagnosis (or an
            empty list) if no bottleneck is detected.
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
