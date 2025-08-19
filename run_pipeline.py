# run_pipeline.py

import logging
from pathlib import Path

# Import all the necessary components from our package

from vertexcare.data_processing.ingestion import (
    run_ingestion,
    setup_logging,
    load_config,
)

from vertexcare.data_processing.validation import run_validation

from vertexcare.features.build_features import run_feature_engineering

from vertexcare.models.train_model import run_training


def main():
    """Main script to run the full ML pipeline."""

    # --- 1. Setup ---
    project_root_path = Path.cwd()
    module_root_path = project_root_path / "vertexcare"
    setup_logging(module_root_path, "full_pipeline")

    logging.info("=======================================")
    logging.info("   STARTING VERTEXCARE ML PIPELINE     ")
    logging.info("=======================================")

    try:
        # --- 2. Load Configuration ---
        config_path = module_root_path / "configs" / "main_config.yaml"
        config = load_config(config_path)

        # --- 3. Run Data Processing Pipeline ---
        logging.info("--- Running Data Processing ---")
        run_ingestion(config, module_root_path)
        run_validation(config, module_root_path)
        logging.info("--- Data Processing complete. ---")

        # --- 4. Run Feature Engineering ---
        logging.info("--- Running Feature Engineering ---")
        run_feature_engineering(config, module_root_path)
        logging.info("--- Feature Engineering complete. ---")

        # --- 5. Run Model Training ---
        logging.info("--- Running Model Training ---")
        # UPDATED: Pass the config_path to the training function
        run_training(config, module_root_path, config_path)
        logging.info("--- Model Training complete. ---")

        logging.info("=======================================")
        logging.info("   VERTEXCARE ML PIPELINE SUCCEEDED    ")
        logging.info("=======================================")

    except Exception as e:
        logging.exception(f"Pipeline failed due to an unexpected error: {e}")
        logging.info("=======================================")
        logging.info("     VERTEXCARE ML PIPELINE FAILED       ")
        logging.info("=======================================")


if __name__ == "__main__":
    main()
