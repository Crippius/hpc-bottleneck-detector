"""Run trained ML model on HPAS fault-injection windows and check predictions."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import DefaultBackend

FAULT_TO_EXPECTED = {
    "cpuoccupy": "COMPUTE_UNDERUTILIZATION",
    "cachecopy": "CACHE_PRESSURE",
    "branchmiss": "BRANCH_MISPREDICTION",
    "pipestall": "PIPELINE_STALL",
    "precwaste": "PRECISION_WASTE",
    "loadimb": "INTRA_NODE_LOAD_IMBALANCE",
}

DATA_DIR = ROOT / "data" / "labelled_data" / "hpas"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="calibrated")
    p.add_argument("--metrics", default=str(DATA_DIR / "hpas_metrics.csv"))
    p.add_argument("--timings", default=str(DATA_DIR / "finj_timings.csv"))
    return p.parse_args()


def parse_timings(path: str) -> tuple[int, list[dict]]:
    df = pd.read_csv(path, sep=";")
    job_start_ts = int(df.loc[df["type"] == "command_session_s", "timestamp"].iloc[0])

    faults = []
    starts = df[df["type"] == "status_start"].copy()
    ends = df[df["type"] == "status_end"].copy()

    for _, row in starts.iterrows():
        m = re.search(r"hpas\s+(\w+)", str(row["args"]))
        if not m:
            continue
        fault_name = m.group(1)
        if fault_name not in FAULT_TO_EXPECTED:
            continue
        end_row = ends[ends["args"] == row["args"]]
        end_ts = int(end_row["timestamp"].iloc[0]) if not end_row.empty else None
        faults.append({
            "fault": fault_name,
            "start_ts": int(row["timestamp"]),
            "end_ts": end_ts,
            "expected": FAULT_TO_EXPECTED[fault_name],
        })

    return job_start_ts, faults


def main() -> None:
    args = parse_args()

    backend = DefaultBackend.load(str(ROOT / "models" / f"{args.model}.pkl"))
    window_size = backend._window_size

    metrics = pd.read_csv(args.metrics)
    job_start_ts, faults = parse_timings(args.timings)

    summary_rows = []
    for fault in faults:
        rel_start = fault["start_ts"] - job_start_ts
        window_start = round((rel_start + 10) / 5) * 5
        window_df = metrics[metrics["time"] >= window_start].head(window_size).copy()
        window_df["id"] = f"{fault['fault']}_w0"

        probs = backend.predict_probabilities(window_df)
        expected = fault["expected"]
        prob = probs.get(expected, 0.0)
        threshold = backend._thresholds.get(expected, 0.5)
        detected = prob >= threshold
        summary_rows.append({
            "fault": fault["fault"],
            "expected": expected,
            "prob": prob,
            "threshold": threshold,
            "correct": "[v]" if detected else "[x]",
            "probs": probs,
        })

    # Summary table
    header = f"{'Fault':<14} {'Expected':<32} {'Prob':>6}  {'Thr':>6}  {'OK'}"
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        print(f"{r['fault']:<14} {r['expected']:<32} {r['prob']:>6.4f}  {r['threshold']:>6.4f}  {r['correct']}")

    # Per-fault probability breakdown
    print()
    for r in summary_rows:
        print(f"--- {r['fault']} (expected: {r['expected']}) ---")
        for bt, p in sorted(r["probs"].items(), key=lambda x: -x[1]):
            marker = " <-- expected" if bt == r["expected"] else ""
            print(f"  {bt:<36} {p:.4f}{marker}")
        print()


if __name__ == "__main__":
    main()
