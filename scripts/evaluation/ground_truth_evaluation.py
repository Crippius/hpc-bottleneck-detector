"""
Ground Truth Evaluation

Runs the trained ML model and the heuristic strategy on HPAS fault-injection
windows and checks predictions/diagnoses against the expected fault.

Usage
-----
    python scripts/evaluation/ground_truth_evaluation.py
    python scripts/evaluation/ground_truth_evaluation.py --mode specificity
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hpc_bottleneck_detector.ml.backends.default_backend import DefaultBackend
from hpc_bottleneck_detector.data.manager import DataManager
from hpc_bottleneck_detector.data.job_context import JobContext
from hpc_bottleneck_detector.data.hardware_profiles import HardwareProfileLoader
from hpc_bottleneck_detector.strategies import HeuristicStrategy
from hpc_bottleneck_detector.utils.labeling import _build_tree_type_map, _window_severity

STRATEGY_FOLDER = ROOT / "configs" / "strategies" / "persyst_strategy"
HW_PROFILES_DIR = ROOT / "configs" / "hardware_profiles"


def _load_backend(path: str):
    return DefaultBackend.load(path)


def _collect_required_specs(strategy: HeuristicStrategy) -> list[dict]:
    """Union of {group, metric, trace} specs required by every loaded tree."""
    seen: set[tuple] = set()
    specs: list[dict] = []
    for tree in strategy._strategy_trees:
        for spec in tree._required_metrics:
            key = (spec.get("group"), spec.get("metric"), spec.get("trace"))
            if key not in seen:
                seen.add(key)
                specs.append(spec)
    return specs


def _flat_col_name(group: str, metric: str, trace: str | None) -> str:
    col = f"{group}_{metric}"
    if trace:
        col += f"_{str(trace).replace(' ', '_')}"
    return col


def build_window_datamanager(
    window_df: pd.DataFrame,
    specs: list[dict],
    supplemental_benchmarks: dict | None = None,
) -> DataManager:
    """
    Reconstruct a long-format DataManager (jobId/group/metric/trace/interval N)
    from a flat window DataFrame (the same format get_flat_dataframe() produces),
    covering only the metrics the loaded strategy trees actually need.

    supplemental_benchmarks (from a HardwareProfileLoader match) mirrors what
    Orchestrator.inject_supplemental_benchmarks() attaches in the live pipeline,
    so benchmark-normalized thresholds (e.g. compute_bound_cache's L3 bandwidth
    gate) resolve instead of always returning UNKNOWN. Without it, any tree
    needing a benchmark falls back to UNKNOWN, same as a live CSV-only source
    with no hardware profile.
    """
    job_id = window_df["id"].iloc[0]
    rows: list[dict] = []
    for spec in specs:
        group, metric, trace = spec.get("group"), spec.get("metric"), spec.get("trace")
        col = _flat_col_name(group, metric, trace)
        if col not in window_df.columns:
            continue
        row = {"jobId": job_id, "group": group, "metric": metric, "trace": trace}
        for i, v in enumerate(window_df[col].to_numpy()):
            row[f"interval {i}"] = v
        rows.append(row)

    job_context = None
    if supplemental_benchmarks:
        job_context = JobContext(
            job_id=str(job_id),
            job_metadata={},
            node_hardware={},
            supplemental_benchmarks=supplemental_benchmarks,
        )
    return DataManager(pd.DataFrame(rows), job_context=job_context)

FAULT_TO_EXPECTED = {
    "cpuoccupy": "COMPUTE_UNDERUTILIZATION",
    "cachecopy": "CACHE_PRESSURE",
    "branchmiss": "BRANCH_MISPREDICTION",
    "pipestall": "PIPELINE_STALL",
    "precwaste": "PRECISION_WASTE",
    "loadimb": "INTRA_NODE_LOAD_IMBALANCE",
}

DATA_DIR = ROOT / "data" / "hpas_fault_injection"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="calibrated")
    p.add_argument(
        "--mode",
        choices=["sensitivity", "specificity"],
        default="sensitivity",
        help=(
            "sensitivity: expect the injected fault to be DETECTED in hpas_metrics.csv. "
            "specificity: expect the fault to be ABSENT (fixed) in hpas_fixed_metrics.csv, "
            "i.e. the expected bottleneck should NOT fire."
        ),
    )
    p.add_argument("--metrics", default=None)
    p.add_argument("--timings", default=str(DATA_DIR / "finj_timings.csv"))
    p.add_argument(
        "--hardware-profile", default=None, metavar="NAME_OR_PATH",
        help="Hardware profile to attach for benchmark-normalized heuristic thresholds "
             "(e.g. 'amd_epyc_9654' or 'intel_xeon_8360Y'). Name stem is resolved against "
             "configs/hardware_profiles/. Without it, benchmark-dependent trees (cache "
             "pressure, vectorization, ...) report UNKNOWN.",
    )
    p.add_argument(
        "--output", default=None,
        help="Optional path to save per-fault results as CSV (fault, expected, prob, "
             "threshold, ML/heuristic correctness, heuristic severity, plus the full "
             "per-class probability breakdown as extra columns).",
    )
    args = p.parse_args()
    if args.metrics is None:
        default_name = "hpas_metrics.csv" if args.mode == "sensitivity" else "hpas_fixed_metrics.csv"
        args.metrics = str(DATA_DIR / default_name)
    return args


def parse_timings(path: str) -> list[dict]:
    """Each row is one fault: 0-based timestamp/end_timestamp (seconds since
    job start), already resolved relative to the session start marker."""
    df = pd.read_csv(path)

    faults = []
    for _, row in df.iterrows():
        fault_name = row["fault"]
        if fault_name not in FAULT_TO_EXPECTED:
            continue
        faults.append({
            "fault": fault_name,
            "start_ts": int(row["timestamp"]),
            "end_ts": int(row["end_timestamp"]),
            "expected": FAULT_TO_EXPECTED[fault_name],
        })

    return faults


def _heuristic_mark(severity: float, mode: str) -> str:
    if math.isnan(severity):
        return "[?]"  # tree couldn't run (missing metric) - abstains, not wrong
    correct = severity > 0.0 if mode == "sensitivity" else severity == 0.0
    return "[v]" if correct else "[x]"


def main() -> None:
    args = parse_args()

    backend = _load_backend(str(ROOT / "models" / f"{args.model}.pkl"))
    window_size = backend._window_size

    strategy = HeuristicStrategy(str(STRATEGY_FOLDER))
    tree_type_map = _build_tree_type_map(strategy)
    required_specs = _collect_required_specs(strategy)

    supplemental_benchmarks: dict = {}
    if args.hardware_profile:
        profile_path = Path(args.hardware_profile)
        if not profile_path.suffix and not profile_path.exists():
            profile_path = HW_PROFILES_DIR / f"{args.hardware_profile}.yaml"
        hw_loader = HardwareProfileLoader(profile_path)
        supplemental_benchmarks = hw_loader.match("")  # single-file loader matches unconditionally
        if not supplemental_benchmarks:
            print(f"[WARN] No benchmarks resolved from hardware profile '{args.hardware_profile}'.")

    metrics = pd.read_csv(args.metrics)
    faults = parse_timings(args.timings)

    summary_rows = []
    for fault in faults:
        window_start = round((fault["start_ts"] + 30) / 5) * 5
        window_df = metrics[metrics["time"] >= window_start].head(window_size).copy()
        window_df["id"] = f"{fault['fault']}_w0"

        probs = backend.predict_probabilities(window_df)
        expected = fault["expected"]
        prob = probs.get(expected, 0.0)
        threshold = backend._thresholds.get(expected, 0.5)
        detected = prob >= threshold
        correct = detected if args.mode == "sensitivity" else not detected

        data_mgr = build_window_datamanager(window_df, required_specs, supplemental_benchmarks)
        heur_severity = _window_severity(strategy.diagnose(data_mgr), tree_type_map)[expected]

        summary_rows.append({
            "fault": fault["fault"],
            "expected": expected,
            "prob": prob,
            "threshold": threshold,
            "correct": "[v]" if correct else "[x]",
            "probs": probs,
            "heur_severity": heur_severity,
            "heur_mark": _heuristic_mark(heur_severity, args.mode),
        })

    # Summary table
    goal = "detected" if args.mode == "sensitivity" else "NOT detected"
    header = (
        f"{'Fault':<14} {'Expected (' + goal + ')':<46} {'Prob':>6}  {'Thr':>6}  {'ML'}  "
        f"{'Heur':>6}  {'Heur'}"
    )
    print(header)
    print("-" * len(header))
    for r in summary_rows:
        heur_str = "   NaN" if math.isnan(r["heur_severity"]) else f"{r['heur_severity']:>6.4f}"
        print(
            f"{r['fault']:<14} {r['expected']:<46} {r['prob']:>6.4f}  {r['threshold']:>6.4f}  "
            f"{r['correct']}  {heur_str}  {r['heur_mark']}"
        )

    n_correct = sum(1 for r in summary_rows if r["correct"] == "[v]")
    n_heur_correct = sum(1 for r in summary_rows if r["heur_mark"] == "[v]")
    n_heur_unknown = sum(1 for r in summary_rows if r["heur_mark"] == "[?]")
    metric_name = "Sensitivity" if args.mode == "sensitivity" else "Specificity"
    print(f"\n{metric_name} (ML):        {n_correct}/{len(summary_rows)} = {n_correct / len(summary_rows):.3f}")
    print(
        f"{metric_name} (Heuristic): {n_heur_correct}/{len(summary_rows)} = "
        f"{n_heur_correct / len(summary_rows):.3f}  "
        f"({n_heur_unknown} abstained - required metric unavailable)"
    )

    # Per-fault probability breakdown
    print()
    for r in summary_rows:
        heur_str = "NaN (tree could not run)" if math.isnan(r["heur_severity"]) else f"{r['heur_severity']:.4f}"
        print(f"--- {r['fault']} (expected: {r['expected']}) --- heuristic severity: {heur_str}")
        for bt, p in sorted(r["probs"].items(), key=lambda x: -x[1]):
            marker = " <-- expected" if bt == r["expected"] else ""
            print(f"  {bt:<36} {p:.4f}{marker}")
        print()

    if args.output:
        out_rows = []
        for r in summary_rows:
            row = {
                "fault": r["fault"],
                "expected": r["expected"],
                "prob": r["prob"],
                "threshold": r["threshold"],
                "ml_correct": r["correct"] == "[v]",
                "heur_severity": r["heur_severity"],
                "heur_correct": r["heur_mark"] == "[v]" if r["heur_mark"] != "[?]" else None,
            }
            for bt, p in r["probs"].items():
                row[f"prob_{bt}"] = p
            out_rows.append(row)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(out_path, index=False)
        print(f"[INFO] Per-fault results saved to: {out_path}")


if __name__ == "__main__":
    main()
