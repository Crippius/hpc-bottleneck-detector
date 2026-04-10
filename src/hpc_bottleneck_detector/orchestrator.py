"""
Analysis Orchestrator

The :class:`AnalysisOrchestrator` ties every component together:

1. Reads a YAML configuration file.
2. Instantiates the appropriate :class:`~hpc_bottleneck_detector.data_sources.interface.IDataSource`.
3. Instantiates the chosen :class:`~hpc_bottleneck_detector.strategies.interface.IAnalysisStrategy`.
4. Exposes :meth:`run_pipeline` which fetches job data, slices it into
   windows, invokes the strategy on every window and returns formatted
   :class:`~hpc_bottleneck_detector.output.models.WindowDiagnosis` results.

Configuration file format
-------------------------
See ``configs/orchestrator.yaml`` for the canonical example.  The top-level
keys are ``pipeline``, ``data_source``, ``strategy`` and ``output``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import yaml

from .data.manager import DataManager
from .data_sources.interface import IDataSource
from .data_sources.csv_source import CSVDataSource
from .data_sources.xbat_source import XBATDataSource
from .output.models import (
    BottleneckType,
    Diagnosis,
    WindowDiagnosis,
)
from .output.formatter import format_results
from .strategies.interface import IAnalysisStrategy
from .strategies.heuristic import HeuristicStrategy

logger = logging.getLogger(__name__)


class AnalysisOrchestrator:
    """
    Central coordinator for the HPC bottleneck-detection pipeline.

    Attributes:
        config:       Parsed configuration dictionary.
        data_source:  Configured data source instance.
        strategy:     Configured analysis strategy instance.
        window_size:  Number of intervals per analysis window.
        step_size:    Interval advance between successive windows.
        output_cfg:   Output-related configuration sub-dict.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        data_source: IDataSource,
        strategy: IAnalysisStrategy,
        window_size: int = 10,
        step_size: int = 10,
        output_cfg: Optional[dict] = None,
    ) -> None:
        self.data_source = data_source
        self.strategy = strategy
        self.window_size = window_size
        self.step_size = step_size
        self.output_cfg: dict = output_cfg or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str) -> "AnalysisOrchestrator":
        """
        Build an :class:`AnalysisOrchestrator` from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            A fully-configured :class:`AnalysisOrchestrator` instance.

        Raises:
            FileNotFoundError: If *config_path* does not exist.
            ValueError: If a required configuration key is missing or
                        an unsupported type is specified.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        logger.info("Loaded configuration from '%s'.", config_path)

        # ── pipeline ──────────────────────────────────────────────────
        pipeline_cfg = config.get("pipeline", {})
        window_size = int(pipeline_cfg.get("window_size", 10))
        step_size   = int(pipeline_cfg.get("step_size",   window_size))

        # ── data source ───────────────────────────────────────────────
        ds_cfg = config.get("data_source", {})
        data_source = cls._build_data_source(ds_cfg)

        # ── strategy ──────────────────────────────────────────────────
        strat_cfg = config.get("strategy", {})
        strategy = cls._build_strategy(strat_cfg)

        # ── output ────────────────────────────────────────────────────
        output_cfg = config.get("output", {})

        return cls(
            data_source=data_source,
            strategy=strategy,
            window_size=window_size,
            step_size=step_size,
            output_cfg=output_cfg,
        )

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self, job_id: str) -> List[WindowDiagnosis]:
        """
        Execute the full analysis pipeline for *job_id*.

        Steps:
            1. Fetch job data from the data source.
            2. Slide the analysis window over the time series.
            3. Call the strategy's ``diagnose`` method for each window.
            4. Apply output filters (min_severity, min_confidence,
               show_healthy_windows).
            5. Render and optionally save results.

        Args:
            job_id: The identifier of the job to analyse.

        Returns:
            List of :class:`~hpc_bottleneck_detector.output.models.WindowDiagnosis`
            objects (after filtering).
        """
        logger.info("Starting pipeline for job '%s'.", job_id)

        # ── 1. Fetch data ─────────────────────────────────────────────
        data_mgr = self.data_source.fetch_job_data(job_id)
        n = data_mgr.get_time_series_length()
        logger.info(
            "Fetched %d interval(s) for job '%s'.",
            n,
            job_id,
        )

        # ── 2 & 3. Window iteration + strategy ────────────────────────
        window_diagnoses: List[WindowDiagnosis] = []

        for win_idx, (start, end, win_dm) in enumerate(
            data_mgr.iterate_windows(self.window_size, self.step_size)
        ):
            diagnoses = self.strategy.diagnose(win_dm)
            wd = WindowDiagnosis(
                window_index=win_idx,
                start_interval=start,
                end_interval=end,
                diagnoses=diagnoses,
            )
            window_diagnoses.append(wd)

        logger.info(
            "Analysis complete: %d window(s) processed.", len(window_diagnoses)
        )

        # ── 4. Filter ─────────────────────────────────────────────────
        window_diagnoses = self._apply_filters(window_diagnoses)

        # ── 5. Format / save ──────────────────────────────────────────
        fmt       = self.output_cfg.get("format", "print")
        save_path = self.output_cfg.get("save_path")
        format_results(window_diagnoses, fmt=fmt, save_path=save_path)

        return window_diagnoses

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_filters(
        self, window_diagnoses: List[WindowDiagnosis]
    ) -> List[WindowDiagnosis]:
        """
        Apply min_severity, min_confidence and show_healthy_windows filters.
        """
        min_severity  = float(self.output_cfg.get("min_severity",  0.0))
        min_confidence = float(self.output_cfg.get("min_confidence", 0.0))
        show_healthy  = bool(self.output_cfg.get("show_healthy_windows", True))

        filtered: List[WindowDiagnosis] = []

        for wd in window_diagnoses:
            # Filter individual diagnoses
            kept = [
                d for d in wd.diagnoses
                if (
                    d.bottleneck_type is BottleneckType.NONE
                    or (
                        d.severity_score  >= min_severity
                        and d.confidence  >= min_confidence
                    )
                )
            ]
            wd.diagnoses = kept

            # Drop healthy windows if requested
            if not show_healthy and not wd.has_bottlenecks():
                continue

            filtered.append(wd)

        return filtered

    # ------------------------------------------------------------------
    # Static builder helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_data_source(cfg: dict) -> IDataSource:
        """Instantiate a data source from its configuration block."""
        ds_type = cfg.get("type", "").lower()

        if ds_type == "csv":
            return CSVDataSource(
                file_path=cfg["file_path"],
                delimiter=cfg.get("delimiter", ","),
            )

        if ds_type == "xbat":
            # If env_file is set, load credentials from the .env file;
            # otherwise fall back to values spelled out in the YAML.
            # Build proxies dict from config when present
            proxy_url = cfg.get("proxy", "")
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
            verify_ssl = cfg.get("verify_ssl", True)

            if "env_file" in cfg:
                return XBATDataSource.from_env(
                    env_file=cfg["env_file"],
                    group=cfg.get("group", ""),
                    metric=cfg.get("metric", ""),
                    level=cfg.get("level", "job"),
                    node=cfg.get("node", ""),
                    token_file=cfg.get("token_file", ".env.xbat"),
                    proxies=proxies,  # None -> from_env reads XBAT_PROXY instead
                    verify_ssl=verify_ssl,
                )
            return XBATDataSource(
                api_base=cfg.get("api_base",   "https://demo.xbat.dev"),
                username=cfg.get("username",   "demo"),
                password=cfg.get("password",   "demo"),
                client_id=cfg.get("client_id", "demo"),
                group=cfg.get("group", ""),
                metric=cfg.get("metric", ""),
                level=cfg.get("level", "job"),
                node=cfg.get("node", ""),
                token_file=cfg.get("token_file", ".env.xbat"),
                proxies=proxies,
                verify_ssl=verify_ssl,
            )

        raise ValueError(
            f"Unsupported data_source type: '{ds_type}'. "
            "Expected 'csv' or 'xbat'."
        )

    @staticmethod
    def _build_strategy(cfg: dict) -> IAnalysisStrategy:
        """Instantiate an analysis strategy from its configuration block."""
        strat_type = cfg.get("type", "").lower()

        if strat_type == "heuristic":
            return HeuristicStrategy(
                strategy_folder=cfg.get("strategy_folder")
            )

        if strat_type == "supervised_ml":
            # Lazy import so that sklearn/tsfresh are only required when
            # this strategy type is actually requested.
            from .strategies.supervised_ml import SupervisedMLStrategy
            from .ml.backends.default_backend import DefaultBackend

            model_path = cfg.get("model_path")
            if not model_path:
                raise ValueError(
                    "strategy.model_path must be set for supervised_ml strategy."
                )
            backend_type = cfg.get("backend", "tsfresh_sklearn").lower()
            if backend_type == "tsfresh_sklearn":
                backend = DefaultBackend.load(model_path)
            else:
                raise ValueError(
                    f"Unsupported ML backend: '{backend_type}'. "
                    "Expected 'tsfresh_sklearn'."
                )
            return SupervisedMLStrategy(
                backend=backend,
                significance_threshold=float(
                    cfg.get("significance_threshold", 0.3)
                ),
            )

        raise ValueError(
            f"Unsupported strategy type: '{strat_type}'. "
            "Expected 'heuristic' or 'supervised_ml'."
        )
