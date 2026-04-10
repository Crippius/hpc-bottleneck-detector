"""
Labeling Example

Demonstrates how to use label_job() to produce a flat, labelled DataFrame
where each row is a time interval and each bottleneck type has its own
severity column.  The result is saved as a CSV file ready for ML training.

Usage:
    python examples/labeling_example.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hpc_bottleneck_detector.data_sources.xbat_source import XBATDataSource
from hpc_bottleneck_detector.strategies import HeuristicStrategy
from hpc_bottleneck_detector.utils.labeling import label_job, BOTTLENECK_COLUMNS


STRATEGY_FOLDER = Path(__file__).parent.parent / "configs" / "strategies" / "persyst_strategy"
ENV_FILE        = ".env"

JOB_ID = 43141

WINDOW_SIZE      = 10   # intervals per analysis window
STEP_SIZE        = 10   # tumbling windows (set < WINDOW_SIZE for sliding)
INTERVAL_SECONDS = 5    # seconds per interval


def main() -> None:
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"[INFO] Connecting via XBATDataSource (env={ENV_FILE})")
    source = XBATDataSource.from_env(env_file=ENV_FILE)
    print(f"       api_base : {source.api_base}")
    print(f"       level    : {source.level}")

    print(f"\n[INFO] Fetching job {JOB_ID} …")
    dm = source.fetch_job_data(JOB_ID)
    print(f"       job_id    : {dm.job_id}")
    print(f"       intervals : {dm.get_time_series_length()}")
    print(f"       metrics   : {len(dm.job_data)}")
    print(f"       context   : {dm.job_context is not None}")

    # ── 2. Load heuristic strategy ────────────────────────────────────────────
    print(f"\n[INFO] Loading strategy trees from {STRATEGY_FOLDER.name}/")
    strategy = HeuristicStrategy(str(STRATEGY_FOLDER))
    print(f"       trees loaded : {len(strategy._strategy_trees)}")
    print(f"       tree names   : {[t.tree_name for t in strategy._strategy_trees]}")

    # ── 3. Label the job ──────────────────────────────────────────────────────
    print(f"\n[INFO] Labelling (window={WINDOW_SIZE}, step={STEP_SIZE}) …")
    labelled = label_job(
        data_mgr=dm,
        strategy=strategy,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        interval_seconds=INTERVAL_SECONDS,
    )
    print(f"       output shape : {labelled.shape}  (rows=intervals, cols=metrics+labels)")

    # ── 4. Inspect label columns ──────────────────────────────────────────────
    print("\n[INFO] Bottleneck label summary:")
    label_cols = [bt.value for bt in BOTTLENECK_COLUMNS]
    for col in label_cols:
        total     = len(labelled)
        n_unknown = labelled[col].isna().sum()
        n_active  = (labelled[col] > 0).sum()
        mean_sev  = labelled[col].mean()
        print(
            f"       {col:<35}  "
            f"active={n_active:>3}/{total}  "
            f"unknown={n_unknown:>3}  "
            f"mean_sev={mean_sev:.3f}"
        )

    # ── 5. Show a few labelled rows ───────────────────────────────────────────
    print("\n[INFO] First 5 rows (id, time, label columns):")
    preview_cols = ["id", "time"] + label_cols
    print(labelled[preview_cols].head().to_string(index=False))

    # ── 6. Save to CSV ────────────────────────────────────────────────────────
    out_path = Path(__file__).parent.parent / "data" / "labelled_data" / f"{dm.job_id}_labelled.csv"
    labelled.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved labelled CSV → {out_path}")

    print("\n" + "=" * 70)
    print("[INFO] Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
