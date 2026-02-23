"""
Example usage of the data source layer.

This script demonstrates how to use the CSVDataSource and DataManager
to load and access HPC job metrics from XBAT CSV files.

Usage:
    python examples/data_source_example.py
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hpc_bottleneck_detector.data_sources import CSVDataSource
from hpc_bottleneck_detector.data import DataManager


def main():
    # Path to the CSV file (adjust as needed)
    # Try both the project sandbox and the parent sandbox
    csv_path = Path(__file__).parent.parent / 'data' / '234650_all_job.csv'
    
    if not csv_path.exists():
        print(f"[ERROR] CSV file not found")
        print(f"Path given: {csv_path}")
        print("Please ensure the XBAT CSV file is available.")
        print("Expected location:")
        print("  - hpc-bottleneck-detector/data/234650_all_job.csv")
        print("If not present look at other csv files in 'data/' folder")
        return
    
    # Step 1: Create a CSV data source
    print(f"[INFO] Loading CSV file: {csv_path.name}")
    data_source = CSVDataSource(str(csv_path))
    print("[INFO] Data source created successfully")
    print()
    
    # Step 2: Fetch job data
    job_id = "234650"
    print(f"[INFO] Fetching data for Job ID: {job_id}")
    try:
        data_manager = data_source.fetch_job_data(job_id)
        print(f"   Job ID: {data_manager.job_id}")
        print(f"   Time series length: {data_manager.get_time_series_length()} intervals")
        print()
    except ValueError as e:
        print(f"[ERROR]: {e}")
        return
    
    # Step 3: List available metrics
    print("[INFO] Available metrics:")
    metrics = data_manager.list_available_metrics()
    for idx, row in metrics.iterrows():
        print(f"   - {row['group']}/{row['metric']}: {row['trace']}")
    print()
    
    # Step 4: Access a specific metric
    print("[INFO] Accessing specific metric:")
    group = "cpu"
    metric = "Branching"
    trace = "branch rate"
    
    try:
        time_series = data_manager.get_metric(group, metric, trace)
        print(f"   Metric: {group}/{metric}/{trace}")
        print(f"   Number of values: {len(time_series)}")
        print(f"   First 5 values: {time_series.head().values}")
        print(f"   Mean: {time_series.mean():.4f}")
        print(f"   Std Dev: {time_series.std():.4f}")
        print(f"   Min: {time_series.min():.4f}")
        print(f"   Max: {time_series.max():.4f}")
        print()
    except ValueError as e:
        print(f"[ERROR]: {e}")
        print()
    
    # Step 5: Get multiple metrics at once
    print("[INFO] Fetching multiple metrics:")
    metric_specs = [
        {'group': 'cpu', 'metric': 'Branching', 'trace': 'branch rate'},
        {'group': 'cpu', 'metric': 'Branching', 'trace': 'branch misprediction rate'},
    ]
    
    multi_metrics = data_manager.get_metrics(metric_specs)
    print(f"   Retrieved {len(multi_metrics)} metrics")
    print(f"   Metrics: {list(multi_metrics.index)}")
    print(f"   Shape: {multi_metrics.shape}")
    print()
    
    # Step 6: Get all time series data
    print("[INFO] Full dataset:")
    all_data = data_manager.get_all_time_series()
    print(f"   Total metrics: {len(all_data)}")
    print(f"   Columns: {list(all_data.columns[:5])}... (showing first 5)")
    print()
    
    print("=" * 70)
    print("[INFO] Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
