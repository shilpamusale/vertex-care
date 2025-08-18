# In vertexcare/src/data_processing/ingestion.py

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import yaml


def setup_logging(project_root: Path) -> None:
    """Sets up logging to both console and a file."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{timestamp}_ingestion.log"
    log_filepath = log_dir / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging initialized. Log file at: {log_filepath}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads the main configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info("Configuration file loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading configuration file: {e}")
        raise


def run_ingestion(config: Dict[str, Any], project_root: Path) -> None:
    """Runs the data ingestion process."""
    logging.info("Starting data ingestion process...")

    raw_file_config = config["data_paths"]["raw_data_dir"]
    raw_data_dir = project_root / raw_file_config
    intermediate_config = config["data_paths"]["intermediate_data_dir"]
    intermediate_dir = project_root / intermediate_config

    intermediate_dir.mkdir(parents=True, exist_ok=True)

    raw_data_file = raw_data_dir / "mock_data_with_notes.csv"
    output_file = intermediate_dir / "ingested_data.parquet"

    if not raw_data_file.exists():
        msg = f"Raw data file not found: {raw_data_file}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    logging.info(f"Reading raw data from {raw_data_file}...")
    df = pd.read_csv(raw_data_file)

    df.columns = [col.lower().strip() for col in df.columns]
    logging.info("Standardized column names.")

    if "chw_notes" in df.columns:
        df["chw_notes"] = df["chw_notes"].fillna("")
        logging.info("Filled missing values in 'chw_notes' column.")

    logging.info(f"Saving ingested data to {output_file}...")
    df.to_parquet(output_file, index=False)

    logging.info("Data ingestion process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data ingestion script for VertexCare."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/main_config.yaml",
        help="Path to the main configuration file.",
    )
    args = parser.parse_args()

    # This assumes the script is run from the top-level project directory
    project_root_path = Path.cwd()

    setup_logging(project_root_path)

    config_path = project_root_path / args.config
    config_data = load_config(config_path)
    run_ingestion(config_data, project_root_path)
