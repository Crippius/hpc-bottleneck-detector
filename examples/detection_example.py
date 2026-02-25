"""
Detection Algorithm Example

Demonstrates the pipeline from data acquisition to bottleneck detection:

Two toy detectors are shown:

  1. MemoryBandwidthBottleneckDetector
       Normalises the measured memory-bandwidth trace against the node's
       theoretical peak and flags intervals
       where utilisation exceeds a configurable threshold.

  2. BranchMispredictionDetector
       Flags intervals where the branch-misprediction rate exceeds a fixed
       threshold, independent of hardware context.

Usage:
    python detection_example.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ── path setup for development (no install required) ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpc_bottleneck_detector.data import DataManager
from hpc_bottleneck_detector.data.job_context import JobContext
from hpc_bottleneck_detector.data_sources.xbat_source import XBATDataSource


# =============================================================================
# Shared result type
# =============================================================================

@dataclass
class DetectionResult:
    """Result produced by a single detector run."""

    detector_name: str
    job_id: str
    bottleneck_detected: bool
    flagged_intervals: List[int] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"[{self.detector_name}]  job={self.job_id}",
            f"  bottleneck detected : {self.bottleneck_detected}",
        ]
        if self.flagged_intervals:
            lines.append(
                f"  flagged intervals   : {len(self.flagged_intervals)} "
                f"(first 5: {self.flagged_intervals[:5]})"
            )
        for key, value in self.details.items():
            lines.append(f"  {key:<20}: {value}")
        return "\n".join(lines)


# =============================================================================
# Detector 1 – Memory bandwidth utilisation
# =============================================================================

class MemoryBandwidthBottleneckDetector:
    """
    Flags intervals where measured memory bandwidth exceeds *threshold* x peak.

    The peak bandwidth is taken from the hardware benchmarks stored in
    JobContext (``bandwidth_mem``).  If no context is available the metric
    is compared against an absolute fallback value.

    Args:
        threshold:          Utilisation ratio above which a bottleneck is
                            flagged (0 - 1), default 0.7.
        fallback_peak_bw:   Absolute peak bandwidth (bytes/s) used when
                            JobContext is unavailable, default 400 GB/s.
    """

    def __init__(
        self,
        threshold: float = 0.70,
        fallback_peak_bw: float = 400e9,
    ) -> None:
        self.threshold = threshold
        self.fallback_peak_bw = fallback_peak_bw

    def analyze(self, dm: DataManager) -> DetectionResult:
        """Run the detector on a DataManager."""

        # ── resolve peak bandwidth ────────────────────────────────────────────
        ctx: Optional[JobContext] = dm.job_context
        if ctx is not None:
            peak_bw = ctx.get_benchmark("bandwidth_mem", aggregate="min")
            bw_source = "hardware benchmark (min across nodes)"
        else:
            peak_bw = None
            bw_source = "fallback constant"

        if peak_bw is None:
            peak_bw = self.fallback_peak_bw
            bw_source = "fallback constant"

        # ── fetch the time-series ─────────────────────────────────────────────
        try:
            ts: pd.Series = dm.get_metric(
                group="memory",
                metric="Bandwidth",
                trace="total",
            )
        except ValueError:
            return DetectionResult(
                detector_name=self.__class__.__name__,
                job_id=dm.job_id or "?",
                bottleneck_detected=False,
                details={"error": "metric 'memory/Bandwidth' not found"},
            )

        values = ts.values.astype(float)

        # ── compute utilisation ───────────────────────────────────────────────
        utilisation = values / peak_bw  # dimensionless ratio

        flagged = [i for i, u in enumerate(utilisation) if u >= self.threshold]
        mean_util = float(np.mean(utilisation))
        max_util = float(np.max(utilisation))

        # ── hardware annotations from context ─────────────────────────────────
        extra = {}
        if ctx is not None:
            extra["cpu_model"] = ctx.get_cpu_info("Model name")
            extra["mem_type"] = ctx.get_memory_info("Type")
            extra["mem_speed"] = ctx.get_memory_info("Speed")
            extra["runtime"] = ctx.get_metadata("runtime")
            extra["job_state"] = ctx.get_metadata("jobState")

        return DetectionResult(
            detector_name=self.__class__.__name__,
            job_id=dm.job_id or "?",
            bottleneck_detected=bool(flagged),
            flagged_intervals=flagged,
            details={
                "peak_bw_source": bw_source,
                "peak_bw_GB/s": f"{peak_bw / 1e9:.1f}",
                "threshold": f"{self.threshold:.0%}",
                "mean_utilisation": f"{mean_util:.1%}",
                "max_utilisation": f"{max_util:.1%}",
                **extra,
            },
        )


# =============================================================================
# Detector 2 – Branch misprediction rate
# =============================================================================

class BranchMispredictionDetector:
    """
    Flags intervals where the branch-misprediction rate exceeds *threshold*.

    This detector does not require hardware context; the threshold is a
    dimensionless rate (mispredicted branches / total branches).

    Args:
        threshold: Rate above which an interval is flagged, default 0.05 (5 %).
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def analyze(self, dm: DataManager) -> DetectionResult:
        """Run the detector on a DataManager."""

        try:
            ts: pd.Series = dm.get_metric(
                group="cpu",
                metric="Branching",
                trace="branch misprediction rate",
            )
        except ValueError:
            return DetectionResult(
                detector_name=self.__class__.__name__,
                job_id=dm.job_id or "?",
                bottleneck_detected=False,
                details={"error": "metric 'cpu/Branching/branch misprediction rate' not found"},
            )

        values = ts.values.astype(float)
        flagged = [i for i, v in enumerate(values) if v >= self.threshold]
        mean_rate = float(np.mean(values))
        max_rate = float(np.max(values))

        # Enrich with context if available
        extra = {}
        ctx: Optional[JobContext] = dm.job_context
        if ctx is not None:
            extra["cpu_model"] = ctx.get_cpu_info("Model name")
            extra["cores"] = ctx.get_cpu_info("CPU(s)")
            extra["peak_flops_avx512_fma"] = ctx.get_benchmark(
                "peakflops_avx512_fma", aggregate="mean"
            )

        return DetectionResult(
            detector_name=self.__class__.__name__,
            job_id=dm.job_id or "?",
            bottleneck_detected=bool(flagged),
            flagged_intervals=flagged,
            details={
                "threshold": f"{self.threshold:.0%}",
                "mean_misprediction_rate": f"{mean_rate:.3%}",
                "max_misprediction_rate": f"{max_rate:.3%}",
                **extra,
            },
        )


# =============================================================================
# Main demo
# =============================================================================

def print_section(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> None:

    # ── 1. Create the data source (demo credentials are the defaults) ─────────
    print_section("1. Creating XBATDataSource")
    source = XBATDataSource()          # connects to demo.xbat.dev
    print(f"  Connected to : {source.api_base}")
    print(f"  Level        : {source.level}")

    # ── 2. Fetch a DataManager for a specific job ─────────────────────────────
    print_section("2. Fetching DataManager for job 234650")
    job_id = "234650"
    dm: DataManager = source.fetch_job_data(job_id)

    print(f"  job_id               : {dm.job_id}")
    print(f"  time-series length   : {dm.get_time_series_length()} intervals")
    print(f"  total metric rows    : {len(dm.job_data)}")
    print(f"  job_context present  : {dm.job_context is not None}")

    # ── 3. Inspect the JobContext ─────────────────────────────────────────────
    print_section("3. Inspecting JobContext (hardware metadata)")
    ctx: Optional[JobContext] = dm.job_context
    if ctx is None:
        print("  JobContext is None - hardware metadata unavailable.")
    else:
        print(f"  {ctx}")
        print()
        print(f"  Job state      : {ctx.get_metadata('jobState')}")
        print(f"  Runtime        : {ctx.get_metadata('runtime')}")
        print(f"  Variant        : {ctx.get_metadata('variantName')}")
        print(f"  Node hashes    : {ctx.get_node_hashes()}")
        print()
        print("  CPU (first node)")
        print(f"    Model        : {ctx.get_cpu_info('Model name')}")
        print(f"    Sockets      : {ctx.get_cpu_info('Socket(s)')}")
        print(f"    Cores/socket : {ctx.get_cpu_info('Core(s) per socket')}")
        print(f"    Threads/core : {ctx.get_cpu_info('Thread(s) per core')}")
        print(f"    NUMA nodes   : {ctx.get_cpu_info('NUMA node(s)')}")
        print(f"    L3 cache     : {ctx.get_cpu_info('L3 cache')}")
        print()
        print("  Memory (first node)")
        print(f"    Type         : {ctx.get_memory_info('Type')}")
        print(f"    Size / dimm  : {ctx.get_memory_info('Size')}")
        print(f"    Speed        : {ctx.get_memory_info('Speed')}")
        print()
        print("  Benchmarks (mean across nodes)")
        bmarks = [
            ("bandwidth_mem",         "Memory BW"),
            ("bandwidth_l3",          "L3 BW"),
            ("bandwidth_l2",          "L2 BW"),
            ("bandwidth_l1",          "L1 BW"),
            ("peakflops_avx512_fma",  "Peak flops (AVX-512 FMA)"),
        ]
        for key, label in bmarks:
            val = ctx.get_benchmark(key)
            if val is not None:
                print(f"    {label:<26}: {val / 1e9:>10.1f} GB/s or GFlops/s")

    # ── 4. Browse available metrics ───────────────────────────────────────────
    print_section("4. Available metrics (first 10)")
    available = dm.list_available_metrics()
    print(f"  Total: {len(available)}")
    # available = available[available["group"] == "memory"]
    for _, row in available.head(10).iterrows():
        print(f"    {row['group']:<12} {row['metric']:<30} {row['trace']}")

    # ── 5. Fetch a single metric time series ──────────────────────────────────
    print_section("5. Fetching a single metric time series")
    try:
        ts = dm.get_metric("cpu", "Branching", "branch rate")
        print(f"  cpu/Branching/branch rate  ({len(ts)} intervals)")
        print(f"    mean  : {ts.mean():.4f}")
        print(f"    std   : {ts.std():.4f}")
        print(f"    min   : {ts.min():.4f}")
        print(f"    max   : {ts.max():.4f}")
    except ValueError as exc:
        print(f"  {exc}")

    # ── 6. Run detection algorithms ───────────────────────────────────────────
    print_section("6. Running detection algorithms")

    detectors = [
        MemoryBandwidthBottleneckDetector(threshold=0.70),
        BranchMispredictionDetector(threshold=0.15),
    ]

    for detector in detectors:
        result: DetectionResult = detector.analyze(dm)
        print()
        print(result.summary())

    print()
    print("=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
