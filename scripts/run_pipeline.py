# run_pipeline.py

import logging
from pathlib import Path
import argparse
import asyncio

# Import all the necessary components from our package
from vertexcare.data_processing.ingestion import (
    run_ingestion,
    setup_logging,
    load_config,
)
from vertexcare.data_processing.validation import run_validation

# Add the new import for the LLM script
from vertexcare.feature_engineering.llm_feature_extractor import run_llm_feature_extraction
from vertexcare.feature_engineering.build_features import run_feature_engineering
from vertexcare.model_training.train_model import run_training


def main():
    """Main script to run the full ML pipeline."""
    parser = argparse.ArgumentParser(description="Run the full VertexCare ML pipeline.")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "xgboost"],
        help="The model to train.",
    )
    args = parser.parse_args()
    model_name = args.model

    setup_logging(f"full_pipeline_{model_name}")

    logging.info("=======================================")
    logging.info(f"   STARTING VERTEXCARE ML PIPELINE: {model_name.upper()}   ")
    logging.info("=======================================")

    try:
        config_path = Path("configs") / "main_config.yaml"
        model_params_path = Path("configs") / "model_params.yaml"
        config = load_config(config_path)
        model_params_config = load_config(model_params_path)

        logging.info("--- Running Data Processing ---")
        run_ingestion(config)
        run_validation(config)
        logging.info("--- Data Processing complete. ---")

        # --- ADD THIS NEW STEP ---
        logging.info("--- Running LLM Feature Extraction ---")
        asyncio.run(run_llm_feature_extraction(config))
        logging.info("--- LLM Feature Extraction complete. ---")

        logging.info("--- Running Feature Engineering ---")
        run_feature_engineering(config)
        logging.info("--- Feature Engineering complete. ---")

        logging.info(f"--- Running Model Training for {model_name} ---")
        run_training(
            config,
            model_params_config,
            config_path,
            model_params_path,
            model_name,
        )
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
