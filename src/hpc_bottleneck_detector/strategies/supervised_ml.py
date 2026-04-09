"""
Supervised ML Strategy

An :class:`~hpc_bottleneck_detector.strategies.interface.IAnalysisStrategy`
that delegates bottleneck detection to a trained :class:`IMLBackend`.
"""

from __future__ import annotations

import logging
from typing import List

from .interface import IAnalysisStrategy
from ..data.manager import DataManager
from ..ml.backend_interface import IMLBackend
from ..output.models import BottleneckType, Diagnosis

logger = logging.getLogger(__name__)


class SupervisedMLStrategy(IAnalysisStrategy):
    """
    Bottleneck-detection strategy backed by a trained ML model.

    Attributes:
        backend:                Trained :class:`IMLBackend` instance.
        significance_threshold: Minimum probability to emit a
                                :class:`Diagnosis`.  Defaults to ``0.3``.
    """

    def __init__(
        self,
        backend: IMLBackend,
        significance_threshold: float = 0.3,
    ) -> None:
        self.backend = backend
        self.significance_threshold = significance_threshold

    # ------------------------------------------------------------------
    # IAnalysisStrategy
    # ------------------------------------------------------------------

    def diagnose(self, data_mgr: DataManager) -> List[Diagnosis]:
        """
        Run ML inference on a DataManager's window and return diagnoses.

        Returns a :class:`Diagnosis` for every ``BottleneckType`` whose
        predicted probability is ≥ ``significance_threshold``.  Returns a
        single ``NONE`` diagnosis when no type exceeds the threshold.
        """
        window_df = data_mgr.get_flat_dataframe()

        try:
            probs = self.backend.predict_probabilities(window_df)
        except Exception as exc:
            logger.warning("ML inference failed: %s — returning UNKNOWN.", exc)
            return [
                Diagnosis(
                    bottleneck_type=BottleneckType.UNKNOWN,
                    severity_score=0.0,
                    confidence=0.0,
                    source="ml",
                )
            ]

        diagnoses: List[Diagnosis] = []
        for bt_name, prob in probs.items():
            if prob >= self.significance_threshold:
                try:
                    bt = BottleneckType[bt_name]
                except KeyError:
                    logger.warning("Unknown BottleneckType from backend: %s", bt_name)
                    continue
                diagnoses.append(
                    Diagnosis(
                        bottleneck_type=bt,
                        severity_score=prob,
                        confidence=prob,
                        source="ml",
                    )
                )

        if not diagnoses:
            return [
                Diagnosis(
                    bottleneck_type=BottleneckType.NONE,
                    severity_score=0.0,
                    confidence=1.0,
                    source="ml",
                )
            ]

        return diagnoses

    def get_required_metrics(self) -> List[str]:
        """
        Returns an empty list — (ML backend determines its own feature set at runtime)
        """
        return []
