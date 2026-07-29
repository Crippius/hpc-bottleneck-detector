<div align="center">
  <img src="media/xbat-logo.svg" alt="XBAT Logo" height="80"/>
</div>

# HPC Bottleneck Detector

Automated performance analysis and bottleneck detection for HPC applications.

This tool identifies performance bottlenecks in HPC jobs using system-wide time-series metrics collected by [XBAT](https://github.com/MEGWARE-HPC/xbat). It requires no source code instrumentation and supports two detection strategies: a rule-based heuristic approach and a weakly-supervised ML approach. Results are per-window diagnoses with severity scores, confidence scores, and actionable recommendations.

---

## Features

- **Two detection strategies**: rule-based YAML decision trees (heuristic) and weakly-supervised ML
- **No source code instrumentation** required: works from system-level metrics alone
- **Sliding-window analysis**: processes time series in overlapping windows, producing per-window diagnoses
- **Interpretable output**: each diagnosis includes severity, confidence, triggered metrics, and a recommendation

---

## Bottleneck Types

| Bottleneck Type             | Category       | Description                                              |
| --------------------------- | -------------- | -------------------------------------------------------- |
| `PIPELINE_STALL`            | Compute-bound  | High CPI due to instruction pipeline stalls              |
| `COMPUTE_UNDERUTILIZATION`  | Compute-bound  | Low utilization of available FP compute throughput       |
| `PRECISION_WASTE`           | Compute-bound  | Using double precision where single precision suffices   |
| `BRANCH_MISPREDICTION`      | Compute-bound  | Frequent branch mispredictions stalling the pipeline     |
| `CACHE_PRESSURE`            | Memory-bound   | High cache miss rate causing memory-bandwidth bottleneck |
| `INTRA_NODE_LOAD_IMBALANCE` | Load imbalance | Uneven load distribution across cores within a node      |
| `INTER_NODE_LOAD_IMBALANCE` | Load imbalance | Uneven load distribution across nodes in the job         |

---

## Installation

```bash
git clone https://github.com/Crippius/hpc-bottleneck-detector.git
cd hpc-bottleneck-detector
uv sync
```

This installs the package into a local `.venv` along with the `bottleneck-detect` CLI command. For notebook/analysis-script/DVC dependencies too, run `uv sync --extra dev` instead.

Copy the credentials template and fill in your XBAT details:

```bash
cp .env.example .env
# edit .env: set XBAT_API_BASE, USERNAME, PASSWORD, CLIENT_ID
```

---

## Quick Start

```bash
uv run bottleneck-detect --job-id <JOB_ID>
```

Run this from the repository root (config paths are resolved relative to the current working directory). This fetches job data from XBAT, slides a 10-interval analysis window over the time series, runs the strategy, and prints the diagnoses as JSON to stdout. Use `--format print` for a human-readable severity heatmap instead, or `--output PATH` to write results to a file. See `bottleneck-detect --help` for all options.

---

## Configuration

The main config is `configs/xbat_cli.yaml`. Key options:

| Key                           | Default     | Description                                               |
| ----------------------------- | ----------- | --------------------------------------------------------- |
| `pipeline.window_size`        | `10`        | Number of intervals per analysis window                   |
| `pipeline.step_size`          | `10`        | Intervals to advance between windows                      |
| `strategy.type`               | `heuristic` | `heuristic` or `supervised_ml`                            |
| `output.min_severity`         | `0.3`       | Suppress diagnoses below this severity (heuristic output) |
| `output.min_confidence`       | `0.3`       | Suppress diagnoses below this confidence (ML output)      |
| `output.show_healthy_windows` | `false`     | Whether to print windows with no bottlenecks              |

To switch to the ML strategy, either edit `strategy` in the config:

```yaml
strategy:
  type: supervised_ml
  model_path: models/default.pkl
  significance_threshold: 0.3
```

or override it from the command line without touching the file:

```bash
uv run bottleneck-detect --job-id <JOB_ID> --model-path models/default.pkl
```

To run against an offline/archived CSV export instead of a live XBAT connection, set `data_source.type: csv` with a `file_path` pointing at a job CSV in the same format as `data/example.csv`.

(`--model-path` implies `--strategy supervised_ml`; pass `--strategy heuristic` to go back.)

---

## Detection Strategies

### Heuristic (rule-based)

YAML decision trees in `configs/strategies/persyst_strategy/`. Each file encodes one bottleneck type as a nested compare-then-branch tree. Inner nodes aggregate a metric (mean, max, ...) and compare it against a threshold. Thresholds can be absolute or hardware-relative (e.g. a fraction of peak FLOPS from the CPU hardware profile).

### Supervised ML (weak supervision)

The ML pipeline uses the heuristic strategy itself to generate weak training labels (`scripts/training/label_jobs.py`). Features are extracted from raw time series using [tsfresh](https://tsfresh.readthedocs.io), filtered with FDR-based selection, and fed into a Random Forest classifier, one per bottleneck type. This lets the model generalize the heuristic rules to jobs it has never seen.

---

## Strategy Trees

The full decision logic for each bottleneck category:

**Compute-bound**
![Compute bound strategy tree](media/strategy_trees/compute_bound_analysis.png)

**Memory-bound**
![Memory bound strategy tree](media/strategy_trees/memory_bound_analysis.png)

**Load imbalance**
![Load imbalance strategy tree](media/strategy_trees/load_imbalance_analysis.png)

For an interactive view, open [`persyst_strategy_trees.html`](persyst_strategy_trees.html) in a browser.

---

## Training Your Own Model

```bash
# 1. Generate weak labels using the heuristic strategy
python scripts/training/label_jobs.py --job-ids 100 101 102 --output data/labels/

# 2. Train the ML model
python scripts/training/train_ml_model.py --data-dir data/labels/ -o models/my_model.pkl
```

---

## Project Structure

```
src/hpc_bottleneck_detector/
├── cli.py               # bottleneck-detect CLI entry point
├── orchestrator.py      # AnalysisOrchestrator: top-level pipeline coordinator
├── data_sources/        # XBAT REST API and CSV data source implementations
├── data/                # DataManager, metric access, hardware profiles
├── strategies/          # IAnalysisStrategy, HeuristicStrategy, SupervisedMLStrategy
├── ml/                  # ML backends: tsfresh feature extraction, classifiers
├── output/              # Diagnosis and WindowDiagnosis domain models
└── utils/               # Shared utilities
```

---

## Dataset

The labelled dataset used to train and evaluate this project (the 20-application
training corpus, the HSUper held-out generalization set, and the HPAS
fault-injection ground truth) is published separately on
Zenodo:

**[HPC Bottleneck Detection Dataset: Labelled Performance Counter Traces](https://zenodo.org/records/21679739)**
— DOI: [10.5281/zenodo.21679739](https://doi.org/10.5281/zenodo.21679739)

---

## License

Code is licensed under MIT — see [LICENSE](LICENSE) for details.

The dataset is licensed separately under CC BY 4.0 - see the
[Zenodo record](https://zenodo.org/records/21679739).
