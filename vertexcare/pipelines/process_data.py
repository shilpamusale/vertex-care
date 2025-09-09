# vertexcare/pipelines/process_data.py

import logging
from pathlib import Path

from vertexcare.data_processing.ingestion import (
    run_ingestion,
    setup_logging,
    load_config,
)
from vertexcare.data_processing.validation import run_validation
from vertexcare.feature_engineering.build_features import run_feature_engineering


def main():
    """Main pipeline script to process data."""

    project_root_path = Path.cwd()
    setup_logging(project_root_path, "pipeline")

    logging.info("=======================================")
    logging.info("  STARTING VERTEXCARE DATA PIPELINE    ")
    logging.info("=======================================")

    try:
        config_path = Path("configs") / "main_config.yaml"
        config = load_config(config_path)

        logging.info("--- Step 1: Running Data Ingestion ---")
        run_ingestion(config)
        logging.info("--- Data Ingestion complete. ---")

        logging.info("--- Step 2: Running Data Validation ---")
        run_validation(config)
        logging.info("--- Data Validation complete. ---")

        logging.info("--- Step 3: Running Feature Engineering ---")
        run_feature_engineering(config)
        logging.info("--- Feature Engineering complete. ---")

        logging.info("=======================================")
        logging.info("  VERTEXCARE DATA PIPELINE SUCCEEDED   ")
        logging.info("=======================================")

    except (FileNotFoundError, ValueError) as e:
        logging.critical(f"Pipeline failed due to a critical error: {e}")
        logging.info("=======================================")
        logging.info("    VERTEXCARE DATA PIPELINE FAILED      ")
        logging.info("=======================================")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        logging.info("=======================================")
        logging.info("    VERTEXCARE DATA PIPELINE FAILED      ")
        logging.info("=======================================")


if __name__ == "__main__":
    main()
