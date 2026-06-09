"""
ML Backend Interface

Backends (:class:`IMLBackend`) hold fitted state and perform inference.
Trainers (:class:`IMLTrainer`) consume labelled data and produce a fitted backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class IMLBackend(ABC):
    """
    Fitted inference backend used by :class:`SupervisedMLStrategy`.

    Holds the trained models and feature pipeline; performs inference and
    persistence.  Use a corresponding :class:`IMLTrainer` to build one from
    labelled data.
    """

    @abstractmethod
    def predict_probabilities(self, window_df: "pd.DataFrame") -> dict[str, float]:
        """
        Predict bottleneck probabilities for a single window.

        Args:
            window_df: Raw window DataFrame from ``DataManager.get_flat_dataframe()``.

        Returns:
            ``{BottleneckType.value: probability}`` for every trained type.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialise the fitted backend to *path*."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "IMLBackend":
        """Restore a backend previously saved with :meth:`save`."""
        ...


class IMLTrainer(ABC):
    """
    Trainer that produces a fitted :class:`IMLBackend` from labelled data.

    Holds training configuration (classifier choice, feature-selection flags,
    etc.) and exposes a single :meth:`train` method.
    """

    @abstractmethod
    def train(
        self,
        labelled_csv_paths: list[str],
        window_size: int,
        step_size: int,
        severity_threshold: float = 0.0,
    ) -> IMLBackend:
        """
        Train on labelled CSVs and return a fitted backend ready for inference.

        Args:
            labelled_csv_paths: Paths to CSVs produced by ``label_job()``.
            window_size:         Number of intervals per analysis window.
            step_size:           Interval advance between successive windows.
            severity_threshold:  Intervals with severity > this value are
                                 labelled positive (1); others are 0.
                                 Windows where every label is NaN are dropped.
        """
        ...
