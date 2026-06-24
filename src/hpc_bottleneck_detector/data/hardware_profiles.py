"""
Hardware Profile Loader

Loads per-architecture hardware profiles from a directory of YAML files.

Profile files must live under the configured ``profiles_dir`` and follow
this structure::

    name: Intel Xeon Platinum 8360Y (Ice Lake-SP)
    cpu_model_pattern: "Intel.*8360Y"   # Python regex, case-insensitive
    benchmarks:
      bandwidth_upi: 55600              # MB/s, per-direction, MLC-measured
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class HardwareProfileLoader:
    """
    Loads hardware profiles from a directory and matches them by CPU model.
    """

    def __init__(self, profiles_dir: Optional[str | Path] = None) -> None:
        self._profiles: list[tuple[re.Pattern, dict]] = []
        if profiles_dir is not None:
            p = Path(profiles_dir)
            if p.is_file():
                self._profiles.append((re.compile(".*"), self._load_benchmarks(p)))
                logger.info("Forced hardware profile from %s.", p.name)
            else:
                self._load_dir(p)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_benchmarks(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as fh:
            profile = yaml.safe_load(fh)
        benchmarks = profile.get("benchmarks", {})
        if not benchmarks:
            raise ValueError(f"Profile {path.name} has no benchmarks.")
        return dict(benchmarks)

    def _load_dir(self, directory: Path) -> None:
        if not directory.is_dir():
            logger.warning(
                "Hardware profiles directory not found: %s - no profiles loaded.",
                directory,
            )
            return

        loaded = 0
        for path in sorted(directory.glob("*.yaml")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    profile = yaml.safe_load(fh)
                pattern_str = profile.get("cpu_model_pattern", "")
                if not pattern_str:
                    logger.warning("Profile %s has no cpu_model_pattern - skipped.", path.name)
                    continue
                benchmarks = profile.get("benchmarks", {})
                if not benchmarks:
                    logger.warning("Profile %s has no benchmarks - skipped.", path.name)
                    continue
                compiled = re.compile(pattern_str, re.IGNORECASE)
                self._profiles.append((compiled, dict(benchmarks)))
                loaded += 1
                logger.debug(
                    "Loaded hardware profile '%s' (%d benchmark(s)) from %s.",
                    profile.get("name", path.stem),
                    len(benchmarks),
                    path.name,
                )

            except Exception as exc:
                logger.warning("Failed to load hardware profile %s: %s", path, exc)

        logger.info("Loaded %d hardware profile(s) from %s.", loaded, directory)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, cpu_model: str) -> dict:
        """
        Return benchmark values from the first profile whose
        ``cpu_model_pattern`` matches *cpu_model*.
        """
        for pattern, benchmarks in self._profiles:
            if pattern.search(cpu_model):
                logger.debug(
                    "Hardware profile matched '%s' for CPU '%s'.",
                    pattern.pattern,
                    cpu_model,
                )
                return dict(benchmarks)
        logger.debug("No hardware profile matched CPU model '%s'.", cpu_model)
        return {}

    def __len__(self) -> int:
        return len(self._profiles)
