# In vertexcare/models/cluster_patients.py

import logging
import joblib
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans

# from typing import Any


def run_clustering(config: dict):
    """
    This function runs the patient clustering pipeline using K-Means.

    It loads the preprocessed training data, fits a K-Means model
    to identify patient clusters, and saves both the trained model and the
    data with cluster assignments.
    """
    logging.info("--- Starting Patient Clustering Pipeline ---")

    # --- 1. Load Data ---
    primary_data_path = Path(config["data_paths"]["primary_data_dir"])
    X_train_path = primary_data_path / "X_train.parquet"
    if not X_train_path.exists():
        logging.error(f"Training data not found at {X_train_path}")
        raise FileNotFoundError("X_train.parquet not found.Please run the main pipeline first.")
    X_train = pd.read_parquet(X_train_path)
    logging.info(f"Loaded training data for clustering from {X_train_path}.")

    # --- 2. Run K-Means Clustering ---
    # --- CHANGE: Corrected the config key to 'clustering_params' ---
    n_clusters = config["clustering_params"]["n_clusters"]
    random_state = config["model_params"]["random_state"]
    logging.info(f"Fitting K-Means model with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = kmeans.fit_predict(X_train)
    logging.info(f"Successfully created {len(set(clusters))} patient clusters.")

    # --- 3. Save Outputs ---
    model_path = Path("models") / "patient_cluster_model.joblib"
    joblib.dump(kmeans, model_path)
    logging.info(f"Clustering model saved to {model_path}")

    X_train_with_clusters = X_train.copy()
    X_train_with_clusters["cluster"] = clusters
    # cluster_profiles_path = primary_data_path / "patient_profiles.parquet"
    cluster_profiles_path = primary_data_path / "clustered_patients.parquet"
    X_train_with_clusters.to_parquet(cluster_profiles_path)
    logging.info("Patient profiles with cluster assignments saved to " + f"{cluster_profiles_path}")

    logging.info("--- Patient Clustering Pipeline Complete ---")
