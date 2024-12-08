from pathlib import Path

import pandas as pd

# Get the project root directory (where pyproject.toml is located)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Set the data folder path relative to the project root
DATA_DIR = PROJECT_ROOT / "cmf" / "data" / "train"

# Create directory if it doesn't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets with error handling and provide more informative messages
datasets = {}
required_files = [
    "agg_trades.parquet",
    "orderbook_embedding.parquet",
    "orderbook_solusdt.parquet",
    "target_data_solusdt.parquet",
    "target_solusdt.parquet"
]

for file_name in required_files:
    file_path = DATA_DIR / file_name
    try:
        datasets[file_name.replace('.parquet', '')] = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Warning: Required data file not found: {file_path}")
        print(f"Please ensure the file exists in: {DATA_DIR}")
        datasets[file_name.replace('.parquet', '')] = None

# Constants for time conversions
NANOSECOND = 1
MICROSECOND = 1000
MILLISECOND = 1000000
SECOND = 1000000000
