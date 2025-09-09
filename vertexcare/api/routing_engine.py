# vertexcare/api/routing_engine.py

import logging
import pandas as pd

# import joblib

from pathlib import Path
from typing import Dict, Any

# from vertexcare.data_processing.ingestion import setup_logging, load_config


def load_clustered_data(primary_dir: Path) -> pd.DataFrame:
    """
    Loads the with patient clsuter labels.
    """
    logging.info("Loading clustered patient data...")
    input_file = primary_dir / "clustered_patients.parquet"
    if not input_file.exists():
        msg = f"Clustered data file not found: {input_file}"
        logging.error(msg)
        raise FileNotFoundError(msg)
    return pd.read_parquet(input_file)


def assign_cluster_names(df: pd.DataFrame, cluster_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Assigns descriptive names to the numeric cluster labels from a config.
    """

    cluster_name_map = cluster_config["cluster_names"]
    df["cluster_name"] = df["cluster"].map(cluster_name_map)

    logging.info("Assigned descriptive names to clusters.")
    return df


def apply_routing_policy(patient_row: pd.Series, policy: Dict[str, Any]) -> Dict[str, str]:
    """
    Applies a set of rules to a patient's data to recommend an action.
    """
    cluster_name = patient_row["cluster_name"]
    # Look up the action for the patient's cluster, or use the default.
    return policy.get(cluster_name, policy["default"])


def run_routing(
    config: Dict[str, Any],
    cluster_config: Dict[str, Any],
    policy_config: Dict[str, Any],
) -> None:
    """
    Main function to run the patient routing pipeline.
    """
    primary_data_dir = Path(config["data_paths"]["primary_data_dir"])

    df = load_clustered_data(primary_data_dir)
    df = assign_cluster_names(df, cluster_config)

    logging.info("Applying routing policy to all patients...")
    routing_decisions = df.apply(apply_routing_policy, axis=1, policy=policy_config)

    routing_df = pd.json_normalize(routing_decisions)

    final_df = pd.concat([df, routing_df], axis=1)

    output_path = primary_data_dir / "routed_patients.parquet"
    final_df.to_parquet(output_path, index=False)
    logging.info(f"Saved routed patient data to {output_path}")
    logging.info("Patient routing pipeline completed successfully.")
