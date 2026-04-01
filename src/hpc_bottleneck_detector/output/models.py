"""
Output Models

Domain objects shared across the whole project:

- :class:`MacroCategoryType` - coarse bottleneck family (compute, memory, …)
- :class:`BottleneckType`    - fine-grained bottleneck kind
- :class:`Diagnosis`         - single bottleneck finding for one analysis window
- :class:`WindowDiagnosis`   - aggregated findings for one time window
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MacroCategoryType(Enum):
    """Coarse family a bottleneck belongs to."""

    COMPUTE_BOUND  = "COMPUTE_BOUND"
    MEMORY_BOUND   = "MEMORY_BOUND"
    LOAD_IMBALANCE = "LOAD_IMBALANCE"
    # IO_BOUND       = "IO_BOUND"
    # NETWORK_BOUND  = "NETWORK_BOUND"
    OTHER          = "OTHER"
    NONE           = "NONE"
    UNKNOWN       = "UNKNOWN"  # analysis could not be performed (missing metrics)


class BottleneckType(Enum):
    """Fine-grained bottleneck classification."""

    # Compute-bound — pipeline / instruction efficiency
    PIPELINE_STALL                  = "PIPELINE_STALL"
    COMPUTE_UNDERUTILIZATION        = "COMPUTE_UNDERUTILIZATION"
    PRECISION_WASTE                 = "PRECISION_WASTE"

    # Compute-bound — vectorisation / data-level parallelism
    BRANCH_MISPREDICTION            = "BRANCH_MISPREDICTION"

    # Memory-bound
    CACHE_PRESSURE                  = "CACHE_PRESSURE"
    MEMORY_BANDWIDTH                = "MEMORY_BANDWIDTH"

    # Communication / load balance
    INTRA_NODE_LOAD_IMBALANCE       = "INTRA_NODE_LOAD_IMBALANCE"
    INTER_NODE_LOAD_IMBALANCE       = "INTER_NODE_LOAD_IMBALANCE"

    # Catch-all / sentinel
    NONE    = "NONE"
    UNKNOWN = "UNKNOWN"  # analysis could not be performed (e.g. missing metrics)

    # ------------------------------------------------------------------
    def get_macro_category(self) -> MacroCategoryType:
        """Return the macro category this bottleneck belongs to."""
        _MAP: dict[BottleneckType, MacroCategoryType] = {
            BottleneckType.PIPELINE_STALL:                MacroCategoryType.COMPUTE_BOUND,
            BottleneckType.COMPUTE_UNDERUTILIZATION:      MacroCategoryType.COMPUTE_BOUND,
            BottleneckType.PRECISION_WASTE:               MacroCategoryType.COMPUTE_BOUND,
            BottleneckType.BRANCH_MISPREDICTION:          MacroCategoryType.COMPUTE_BOUND,
            BottleneckType.CACHE_PRESSURE:                MacroCategoryType.MEMORY_BOUND,
            BottleneckType.MEMORY_BANDWIDTH:              MacroCategoryType.MEMORY_BOUND,
            BottleneckType.INTRA_NODE_LOAD_IMBALANCE:     MacroCategoryType.LOAD_IMBALANCE,
            BottleneckType.INTER_NODE_LOAD_IMBALANCE:     MacroCategoryType.LOAD_IMBALANCE,
            BottleneckType.NONE:                          MacroCategoryType.NONE,
            BottleneckType.UNKNOWN:                       MacroCategoryType.UNKNOWN,
        }
        return _MAP.get(self, MacroCategoryType.NONE)


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

@dataclass
class Diagnosis:
    """
    A single bottleneck finding produced by a strategy tree or ML model.

    Attributes:
        bottleneck_type:    The classified bottleneck (``NONE`` = healthy).
        severity_score:     0.0 - 1.0; 0 = negligible, 1 = critical.
        confidence:         0.0 - 1.0; how confident the detector is.
        recommendation:     Human-readable remediation hint.
        source:             Name of the strategy / tree that produced this.
        triggered_metrics:  Metric keys that drove the decision.
    """

    bottleneck_type:   BottleneckType
    severity_score:    float
    confidence:        float
    recommendation:    Optional[str]        = field(default=None)
    source:            str                  = field(default="")
    triggered_metrics: List[str]            = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def is_healthy(self) -> bool:
        """True when no bottleneck was detected."""
        return self.bottleneck_type is BottleneckType.NONE

    @property
    def is_unknown(self) -> bool:
        """True when the analysis could not determine whether a bottleneck exists."""
        return self.bottleneck_type is BottleneckType.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "bottleneck_type":   self.bottleneck_type.value,
            "severity_score":    round(self.severity_score, 4),
            "confidence":        round(self.confidence, 4),
            "recommendation":    self.recommendation,
            "source":            self.source,
            "triggered_metrics": self.triggered_metrics,
        }


# ---------------------------------------------------------------------------
# WindowDiagnosis
# ---------------------------------------------------------------------------

@dataclass
class WindowDiagnosis:
    """
    Aggregated diagnoses for a single analysis window.

    Attributes:
        window_index:  Zero-based index of the window in the job timeline.
        start_interval: First interval index included in this window.
        end_interval:   Last interval index included in this window (inclusive).
        diagnoses:      All :class:`Diagnosis` objects produced for this window.
    """

    window_index:   int
    start_interval: int
    end_interval:   int
    diagnoses:      List[Diagnosis] = field(default_factory=list)

    # ------------------------------------------------------------------
    def has_bottlenecks(self) -> bool:
        """Return True when at least one real (non-NONE, non-UNKNOWN) diagnosis is present."""
        return any(not d.is_healthy and not d.is_unknown for d in self.diagnoses)

    def has_unknowns(self) -> bool:
        """Return True when at least one UNKNOWN diagnosis is present."""
        return any(d.is_unknown for d in self.diagnoses)

    def worst_severity(self) -> float:
        """Return the maximum severity score among all diagnoses (0.0 if empty)."""
        if not self.diagnoses:
            return 0.0
        return max(d.severity_score for d in self.diagnoses)

    def to_dict(self) -> dict:
        _excluded = {BottleneckType.NONE, BottleneckType.UNKNOWN}
        detected = {
            d.bottleneck_type: d
            for d in self.diagnoses
            if d.bottleneck_type not in _excluded
        }
        full_diagnoses = []
        for bt in BottleneckType:
            if bt in _excluded:
                continue
            if bt in detected:
                full_diagnoses.append(detected[bt].to_dict())
            else:
                full_diagnoses.append({
                    "bottleneck_type":   bt.value,
                    "severity_score":    0.0,
                    "confidence":        1.0,
                    "recommendation":    None,
                    "source":            "",
                    "triggered_metrics": [],
                })
        return {
            "window_index":    self.window_index,
            "start_interval":  self.start_interval,
            "end_interval":    self.end_interval,
            "has_bottlenecks": self.has_bottlenecks(),
            "worst_severity":  round(self.worst_severity(), 4),
            "diagnoses":       full_diagnoses,
        }
