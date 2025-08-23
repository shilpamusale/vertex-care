# run_clustering_pipeline.py

import logging
from pathlib import Path

from vertexcare.data_processing.ingestion import (
    setup_logging,
    load_config,
)
from vertexcare.models.cluster_patients import run_clustering


def main():
    """Main script to run the patient clustering pipeline."""

    project_root_path = Path.cwd()
    module_root_path = project_root_path / "vertexcare"
    setup_logging(module_root_path, "clustering_pipeline")

    logging.info("=======================================")
    logging.info("   STARTING VERTEXCARE CLUSTERING PIPELINE   ")
    logging.info("=======================================")

    try:
        # Load both configuration files
        config_path = module_root_path / "configs" / "main_config.yaml"
        model_params_path = module_root_path / "configs" / "model_params.yaml"
        config = load_config(config_path)
        model_params_config = load_config(model_params_path)

        # Combine the configs into a single dictionary for the pipeline
        # This makes it easy to pass all parameters down
        full_config = {**config, **model_params_config}

        run_clustering(full_config, module_root_path)

        logging.info("=======================================")
        logging.info("   VERTEXCARE CLUSTERING PIPELINE SUCCEEDED   ")
        logging.info("=======================================")

    except Exception as e:
        logging.exception(f"Pipeline failed due to an unexpected error: {e}")
        logging.info("=======================================")
        logging.info("     VERTEXCARE CLUSTERING PIPELINE FAILED      ")
        logging.info("=======================================")


if __name__ == "__main__":
    main()
