# run_routing_pipeline.py

import logging
from pathlib import Path

from vertexcare.data_processing.ingestion import (
    setup_logging,
    load_config,
)
from vertexcare.api.routing_engine import run_routing


def main():
    """Main script to run the patient routing pipeline."""

    project_root_path = Path.cwd()
    module_root_path = project_root_path / "vertexcare"
    setup_logging(module_root_path, "routing_pipeline")

    logging.info("=======================================")
    logging.info("   STARTING VERTEXCARE ROUTING PIPELINE    ")
    logging.info("=======================================")

    try:
        # Load all necessary configuration files
        config_path = module_root_path / "configs" / "main_config.yaml"
        cluster_config_path = module_root_path / "configs" / "cluster_config.yaml"
        policy_config_path = module_root_path / "configs" / "routing_policy.yaml"
        config = load_config(config_path)
        cluster_config = load_config(cluster_config_path)
        policy_config = load_config(policy_config_path)

        run_routing(config, cluster_config, policy_config, module_root_path)

        logging.info("=======================================")
        logging.info("   VERTEXCARE ROUTING PIPELINE SUCCEEDED   ")
        logging.info("=======================================")

    except Exception as e:
        logging.exception(f"Pipeline failed due to an unexpected error: {e}")
        logging.info("=======================================")
        logging.info("     VERTEXCARE ROUTING PIPELINE FAILED      ")
        logging.info("=======================================")


if __name__ == "__main__":
    main()
