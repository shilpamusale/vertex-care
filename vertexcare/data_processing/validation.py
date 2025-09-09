# vertexcare/data_processing/validation.py

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

from vertexcare.data_processing.ingestion import setup_logging, load_config


def load_data(file_path: Path) -> pd.DataFrame:
    """Loads data from a parquet file."""
    logging.info(f"Loading ingested data from {file_path}...")
    if not file_path.exists():
        msg = f"Ingested data file not found: {file_path}"
        logging.error(msg)
        raise FileNotFoundError(msg)
    return pd.read_parquet(file_path)


def load_schema(schema_path: Path) -> Dict[str, Any]:
    """Loads the data validation schema from a YAML file."""
    logging.info(f"Loading data schema from {schema_path}...")
    try:
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)
        return schema
    except FileNotFoundError:
        msg = f"Schema file not found: {schema_path}"
        logging.error(msg)
        raise FileNotFoundError(msg)


def validate_data(df: pd.DataFrame, schema: Dict[str, Any]) -> bool:
    """Validates the dataframe against a defined schema."""
    is_valid = True
    logging.info("Starting data validation...")

    expected_columns = set(schema["columns"].keys())
    actual_columns = set(df.columns)
    if not expected_columns.issubset(actual_columns):
        missing_columns = expected_columns - actual_columns
        logging.error(f"Validation failed: Missing columns {missing_columns}")
        is_valid = False
    else:
        logging.info("Column presence check passed.")

    for col, expected_type in schema["columns"].items():
        if col in df.columns and str(df[col].dtype) != expected_type:
            logging.error(f"Validation failed: Column '{col}' has type " f"{df[col].dtype}, expected {expected_type}")
            is_valid = False
    if is_valid:
        logging.info("Data type check passed.")

    critical_columns = schema.get("not_null_columns", [])
    for col in critical_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            logging.error(f"Validation failed: Column '{col}' contains null.")
            is_valid = False
    if is_valid:
        logging.info("Null value check passed.")

    if is_valid:
        logging.info("Data validation completed successfully.")
    else:
        logging.error("Data validation failed.")
    return is_valid


def run_validation(config: Dict[str, Any]) -> None:
    """Main function to run the entire validation process."""
    # directory = config["data_paths"]["intermediate_data_dir"]
    intermediate_dir = Path(config["data_paths"]["intermediate_data_dir"])
    input_file = intermediate_dir / "ingested_data.parquet"

    schema_path = Path("configs") / "data_schema.yaml"
    schema = load_schema(schema_path)

    df = load_data(input_file)
    if not validate_data(df, schema):
        raise ValueError("Data validation failed. Halting pipeline.")


if __name__ == "__main__":
    project_root_path = Path.cwd()
    setup_logging("validation")

    config_path = project_root_path / "configs" / "main_config.yaml"
    config_data = load_config(config_path)

    try:
        run_validation(config_data)
    except (FileNotFoundError, ValueError) as e:
        logging.critical(f"A critical error occurred: {e}")
