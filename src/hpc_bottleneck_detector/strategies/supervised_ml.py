"""
Supervised ML Strategy

An :class:`~hpc_bottleneck_detector.strategies.interface.IAnalysisStrategy`
that delegates bottleneck detection to a trained :class:`IMLBackend`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import yaml

from .interface import IAnalysisStrategy
from ..data.manager import DataManager
from ..ml.backend_interface import IMLBackend
from ..output.models import BottleneckType, Diagnosis

logger = logging.getLogger(__name__)

# General, class-level recommendations for the ML branch, keyed by BottleneckType.
_ML_RECOMMENDATIONS_PATH = Path(__file__).resolve().parents[3] / "configs" / "ml_recommendations.yaml"


def _load_ml_recommendations() -> dict[BottleneckType, str]:
    if not _ML_RECOMMENDATIONS_PATH.exists():
        logger.warning("ML recommendations file not found: %s", _ML_RECOMMENDATIONS_PATH)
        return {}
    with _ML_RECOMMENDATIONS_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    recommendations: dict[BottleneckType, str] = {}
    for name, text in raw.items():
        try:
            recommendations[BottleneckType[name]] = text
        except KeyError:
            logger.warning("Unknown BottleneckType in ml_recommendations.yaml: %s", name)
    return recommendations


_ML_RECOMMENDATIONS: dict[BottleneckType, str] = _load_ml_recommendations()


class SupervisedMLStrategy(IAnalysisStrategy):
    """Bottleneck-detection strategy backed by a trained ML model."""

    def __init__(
        self,
        backend: IMLBackend,
        significance_threshold: float = 0.3,
    ) -> None:
        self.backend = backend
        self.significance_threshold = significance_threshold

    # --- IAnalysisStrategy ---------------------------------------------------

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
            logger.warning("ML inference failed: %s - returning UNKNOWN.", exc)
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
                        recommendation=_ML_RECOMMENDATIONS.get(bt),
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
        Returns an empty list - (ML backend determines its own feature set at runtime)
        """
        return []
