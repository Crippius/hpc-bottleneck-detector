"""
Detector Scaling Benchmark

Measures wall-clock time and peak RSS memory of different detectors
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("benchmark_detector_scaling")

_ELAPSED_RE = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)")
_MAXRSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def _parse_elapsed(value: str) -> float:
    parts = [float(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def _benchmark_config(base_config: Path) -> Path:
    import yaml
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("output", {})
    cfg["output"]["show_healthy_windows"] = True
    cfg["output"]["min_severity"] = 0.0
    cfg["output"]["min_confidence"] = 0.0

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return Path(tmp.name)


def run_one(python_exe: str, config: Path, job_id: str, model_path: Path | None = None) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f, \
         tempfile.NamedTemporaryFile(suffix=".time", delete=False) as time_f:
        out_path = Path(out_f.name)
        time_path = Path(time_f.name)

    cmd = [
        "/usr/bin/time", "-v", "-o", str(time_path),
        python_exe, "-m", "hpc_bottleneck_detector.cli",
        "--job-id", str(job_id),
        "--config", str(config),
        "--format", "json",
        "--output", str(out_path),
        "--quiet",
    ]
    if model_path is not None:
        cmd += ["--model-path", str(model_path)]

    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,  # cli.py resolves config paths relative to cwd
            capture_output=True,
            text=True,
        )

        time_report = time_path.read_text() if time_path.exists() else ""
        elapsed_match = _ELAPSED_RE.search(time_report)
        maxrss_match = _MAXRSS_RE.search(time_report)

        elapsed_seconds = _parse_elapsed(elapsed_match.group(1)) if elapsed_match else float("nan")
        max_rss_mb = (int(maxrss_match.group(1)) / 1024) if maxrss_match else float("nan")

        n_windows = -1
        if proc.returncode == 0:
            try:
                n_windows = len(json.loads(out_path.read_text()))
            except (json.JSONDecodeError, OSError):
                n_windows = -1

        if proc.returncode != 0:
            log.warning(
                "job %s exited %d: %s", job_id, proc.returncode,
                (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else "(no stderr)",
            )

        return {
            "elapsed_seconds": elapsed_seconds,
            "max_rss_mb": round(max_rss_mb, 1) if max_rss_mb == max_rss_mb else max_rss_mb,  # keep NaN as NaN
            "n_windows": n_windows,
        }
    finally:
        out_path.unlink(missing_ok=True)
        time_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True,
                         help="Sweep manifest CSV with job_id and target_minutes columns")
    parser.add_argument("--strategies", nargs="+", default=["heuristic", "rf", "xgboost"],
                         choices=["heuristic", "rf", "xgboost"])
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "xbat_cli.yaml"),
                         help="Base config; strategy is overridden per-run via --strategy/--model-path")
    parser.add_argument("--rf-model", default=str(REPO_ROOT / "models" / "rf.pkl"),
                         help="rf is skipped if this model file doesn't exist")
    parser.add_argument("--xgboost-model", default=str(REPO_ROOT / "models" / "xgboost.pkl"),
                         help="xgboost is skipped if this model file doesn't exist")
    parser.add_argument("--output-csv", default=None,
                         help="default: results/scalability/detector_scaling_<manifest-name>.csv")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to run the CLI with")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        parser.error(f"manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    if "target_minutes" not in manifest_df.columns:
        parser.error(f"manifest {manifest_path} has no target_minutes column")
    job_ids = manifest_df["job_id"].tolist()
    runtime_by_job = dict(zip(manifest_df["job_id"], manifest_df["target_minutes"]))

    output_csv = Path(args.output_csv) if args.output_csv else (
        REPO_ROOT / "results" / "scalability" / f"detector_scaling_{manifest_path.stem}.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["strategy", "elapsed_seconds", "max_rss_mb", "n_windows", "runtime"]
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0

    log.info("manifest=%s  jobs=%d  strategies=%s  output=%s", manifest_path, len(job_ids), args.strategies, output_csv)

    with output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        model_path_for_strategy = {
            "rf": Path(args.rf_model),
            "xgboost": Path(args.xgboost_model),
        }

        bench_config = _benchmark_config(Path(args.config))
        try:
            for strategy in args.strategies:
                model_path = model_path_for_strategy.get(strategy)
                if model_path is not None and not model_path.is_file():
                    log.warning("skipping %s: no model at %s (train one with "
                                "scripts/training/train_ml_model.py --classifier %s)",
                                strategy, model_path, strategy)
                    continue

                for job_id in job_ids:
                    log.info("running job=%s strategy=%s", job_id, strategy)
                    row = run_one(args.python, bench_config, job_id, model_path=model_path)
                    row["strategy"] = strategy
                    row["runtime"] = runtime_by_job[job_id]
                    writer.writerow(row)
                    f.flush()
                    log.info("  elapsed=%.2fs  max_rss=%.1fMB  windows=%d",
                             row["elapsed_seconds"], row["max_rss_mb"], row["n_windows"])
        finally:
            bench_config.unlink(missing_ok=True)

    log.info("done. results: %s", output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
