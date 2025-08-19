# vertexcare/data_processing/ingestion.py

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd
import yaml


def setup_logging(module_root: Path, log_name: str) -> None:
    """Sets up a robust logger inside the module."""
    log_dir = module_root / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{timestamp}_{log_name}.log"
    log_filepath = log_dir / log_filename

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logging.info(f"Logging initialized. Log file at: {log_filepath}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads a configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Config file loaded successfully from {config_path}.")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found at: {config_path}")
        raise


def run_ingestion(config: Dict[str, Any], module_root: Path) -> None:
    """Runs the data ingestion process using
    paths relative to the module root.
    """
    logging.info("Starting data ingestion process...")

    raw_data_dir = module_root / config["data_paths"]["raw_data_dir"]
    intermediate_path = config["data_paths"]["intermediate_data_dir"]
    intermediate_dir = module_root / intermediate_path
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    raw_data_file = raw_data_dir / "mock_data_with_notes.csv"
    output_file = intermediate_dir / "ingested_data.parquet"

    if not raw_data_file.exists():
        msg = f"Raw data file not found: {raw_data_file}"
        logging.error(msg)
        raise FileNotFoundError(msg)

    logging.info(f"Reading raw data from {raw_data_file}...")
    df = pd.read_csv(raw_data_file)
    df.columns = [col.lower().strip().replace(".", "_") for col in df.columns]
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

    project_root_path = Path.cwd()
    module_root_path = project_root_path / "vertexcare"

    setup_logging(module_root_path, "ingestion")

    config_path = module_root_path / args.config
    config_data = load_config(config_path)
    run_ingestion(config_data, module_root_path)
