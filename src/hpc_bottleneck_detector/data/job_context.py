"""
Job Context Module

Holds static metadata about an HPC job and the hardware nodes it ran on.
This context complements the time-series measurements stored in DataManager
with information that is fixed for the lifetime of the job (hardware specs,
job configuration, benchmark capabilities of the nodes).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Keys extracted from the node hardware response that are relevant for
# relating hardware capabilities to time-series observations.
_CPU_KEYS = (
    "Model name",
    "CPU(s)",
    "Core(s) per socket",
    "Socket(s)",
    "Thread(s) per core",
    "NUMA node(s)",
    "CPU max MHz",
    "L1d cache",
    "L1i cache",
    "L2 cache",
    "L3 cache",
    "Architecture",
    "Vendor ID",
)

_MEMORY_KEYS = (
    "Type",
    "Size",
    "Speed",
    "Maximum Capacity",
    "Number Of Installed Devices",
    "Error Correction Type",
)


def _filter_cpu(cpu_raw: dict) -> dict:
    """Return only the CPU fields relevant for performance analysis."""
    return {k: cpu_raw[k] for k in _CPU_KEYS if k in cpu_raw}


def _filter_memory(mem_raw: dict) -> dict:
    """Return only the memory fields relevant for performance analysis."""
    return {k: mem_raw[k] for k in _MEMORY_KEYS if k in mem_raw}


def _extract_node_info(node_raw: dict) -> dict:
    """
    Extract the hardware fields relevant for correlating with time-series data.

    Keeps:
    - ``cpu``        - model, core counts, NUMA topology, cache sizes
    - ``memory``     - type, size, speed
    - ``benchmarks`` - theoretical peak bandwidth / flops
    - ``os``         - distro and kernel
    """
    info: dict = {}

    if "cpu" in node_raw:
        info["cpu"] = _filter_cpu(node_raw["cpu"])

    if "memory" in node_raw:
        info["memory"] = _filter_memory(node_raw["memory"])

    if "benchmarks" in node_raw:
        info["benchmarks"] = dict(node_raw["benchmarks"])

    if "os" in node_raw:
        os_raw = node_raw["os"]
        info["os"] = {
            "distro": os_raw.get("distro"),
            "kernel": os_raw.get("kernel"),
            "architecture": os_raw.get("architecture"),
        }

    return info


class JobContext:
    """
    Static context for a single HPC job.

    Attributes:
        job_id:        String job identifier.
        job_metadata:  Dict extracted from the ``/api/v1/jobs`` response for
                       this job (runtime, capturetime, jobState, node
                       hostnames, configuration variant).
        node_hardware: Dict mapping node hash -> filtered hardware info
                       (cpu, memory, benchmarks, os).  Nodes that share the
                       same hardware hash appear under a single entry.
    """

    def __init__(
        self,
        job_id: str,
        job_metadata: Dict[str, Any],
        node_hardware: Dict[str, Dict[str, Any]],
    ) -> None:
        self.job_id = job_id
        self.job_metadata = job_metadata
        self.node_hardware = node_hardware

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_job_id(self) -> str:
        """Return the job identifier."""
        return self.job_id

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Return a top-level field from the job metadata."""
        return self.job_metadata.get(key, default)

    def get_node_hashes(self) -> List[str]:
        """Return the list of unique node hardware hashes used by this job."""
        return list(self.node_hardware.keys())

    def get_node_info(self, node_hash: str) -> Optional[Dict[str, Any]]:
        """Return the hardware info dict for a given node hash."""
        return self.node_hardware.get(node_hash)

    def get_benchmark(self, key: str, aggregate: str = "mean") -> Optional[float]:
        """
        Return an aggregated benchmark value across all nodes.

        Args:
            key:       Benchmark name, e.g. ``'bandwidth_mem'``,
                       ``'peakflops_avx512_fma'``.
            aggregate: ``'mean'`` (default), ``'min'``, or ``'max'``.

        Returns:
            Aggregated float, or ``None`` if the key is absent on all nodes.
        """
        values = [
            info["benchmarks"][key]
            for info in self.node_hardware.values()
            if "benchmarks" in info and key in info["benchmarks"]
        ]
        if not values:
            return None
        if aggregate == "min":
            return min(values)
        if aggregate == "max":
            return max(values)
        return sum(values) / len(values)

    def get_cpu_info(self, key: str) -> Optional[Any]:
        """
        Return a CPU property from the first node.

        For homogeneous clusters all nodes share the same hardware, so the
        first entry is representative.  Use ``node_hardware`` directly for
        heterogeneous jobs.
        """
        for info in self.node_hardware.values():
            if "cpu" in info:
                return info["cpu"].get(key)
        return None

    def get_memory_info(self, key: str) -> Optional[Any]:
        """Return a memory property from the first node (representative)."""
        for info in self.node_hardware.values():
            if "memory" in info:
                return info["memory"].get(key)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_nodes = len(self.job_metadata.get("nodes", {}))
        state = self.job_metadata.get("jobState", "unknown")
        return (
            f"JobContext(job_id={self.job_id!r}, nodes={n_nodes}, "
            f"jobState={state!r}, "
            f"node_hashes={len(self.node_hardware)})"
        )

    @classmethod
    def from_xbat(
        cls,
        job_id: str,
        job_entry: dict,
        node_hardware_raw: Dict[str, dict],
    ) -> "JobContext":
        """
        Build a ``JobContext`` from raw XBAT API responses.

        Args:
            job_id:            String job identifier.
            job_entry:         Single element from the ``/api/v1/jobs`` data
                               list matching this job.
            node_hardware_raw: Full response from ``/api/v1/nodes``, keyed
                               by hash.

        Returns:
            Populated ``JobContext`` instance.
        """
        # Flatten the fields that are useful downstream
        job_info = job_entry.get("jobInfo", {})
        nodes_raw: dict = job_entry.get("nodes", {})
        configuration: dict = job_entry.get("configuration", {})
        variant = (
            configuration.get("jobscript", {}).get("variantName")
            or configuration.get("variantName")
        )

        job_metadata = {
            "runtime": job_entry.get("runtime"),
            "capturetime": job_entry.get("capturetime"),
            "jobState": job_info.get("jobState"),
            "runNr": job_entry.get("runNr"),
            "iteration": job_entry.get("iteration"),
            "variantName": variant,
            # Map hostname → hash for reference
            "nodes": {
                hostname: meta.get("hash")
                for hostname, meta in nodes_raw.items()
            },
        }

        node_hardware = {
            hash_: _extract_node_info(node_hardware_raw[hash_])
            for _, hash_ in job_metadata["nodes"].items()
            if hash_ and hash_ in node_hardware_raw
        }

        return cls(
            job_id=job_id,
            job_metadata=job_metadata,
            node_hardware=node_hardware,
        )
