"""
ML Backend Interface

All concrete ML backends must implement :class:`IMLBackend`.
A backend owns the full pipeline from a raw window DataFrame to per-class probabilities: 
feature extraction, feature selection, and model inference all live inside the backend
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class IMLBackend(ABC):
    """
    Abstract interface for ML backends used by :class:`SupervisedMLStrategy`.

    Contract
    --------
    Training — call :meth:`train` once with the paths to labelled CSVs
    (produced by ``label_job()``), the window size, and the step size.
    The backend extracts features, selects the most relevant ones, trains one
    binary classifier per ``BottleneckType``, and stores everything in memory.

    Persistence — call :meth:`save` to save  to disk models and the selected feature columns).  
    Use :meth:`load` to restore a previously saved backend.

    Inference — call :meth:`predict_probabilities` with a raw window
    DataFrame (``DataManager.get_flat_dataframe()`` output).  The backend
    applies the same feature-extraction and column-alignment steps used during
    training and returns a probability per bottleneck type.
    """

    @abstractmethod
    def train(
        self,
        labelled_csv_paths: list[str],
        window_size: int,
        step_size: int,
        severity_threshold: float = 0.0,
    ) -> None:
        """
        Train one binary classifier per ``BottleneckType``.

        Args:
            labelled_csv_paths: Paths to CSVs produced by ``label_job()``.
                                 Each file may contain one or more jobs.
            window_size:         Number of intervals per analysis window.
            step_size:           Interval advance between successive windows.
            severity_threshold:  Intervals with severity > this value are
                                 labelled positive (1); others are labelled 0.
                                 Rows where the label is NaN are dropped.
        """
        ...

    @abstractmethod
    def predict_probabilities(self, window_df: "pd.DataFrame") -> dict[str, float]:
        """
        Predict bottleneck probabilities for a single window.

        Args:
            window_df: Raw window DataFrame as returned by
                       ``DataManager.get_flat_dataframe()``.
                       Must contain ``id`` and ``time`` columns plus metric
                       columns.

        Returns:
            Mapping ``{BottleneckType.value: probability}`` for every type
            that has a trained model.  Probabilities are in ``[0, 1]``.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Serialise the trained backend (models + feature pipeline) to *path*.

        Args:
            path: File or directory path.  The backend decides the exact layout.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "IMLBackend":
        """
        Restore a backend that was previously saved with :meth:`save`.

        Args:
            path: Same path that was passed to :meth:`save`.

        Returns:
            A fully-restored backend instance ready for inference.
        """
        ...
