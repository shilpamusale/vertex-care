# vertexcare/models/train_model.py

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import shutil

import pandas as pd
import xgboost as xgb
import joblib
import json
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

# Import from our project
from vertexcare.data_processing.ingestion import setup_logging, load_config


def load_processed_data(
    primary_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads the processed training and testing data sets."""
    logging.info("Loading processed data...")
    X_train = pd.read_parquet(primary_dir / "X_train.parquet")
    X_test = pd.read_parquet(primary_dir / "X_test.parquet")
    y_train = pd.read_parquet(primary_dir / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(primary_dir / "y_test.parquet").squeeze()
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, model_params: Dict[str, Any]
) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier.

    Args:
        X_train: The training features.
        y_train: The training target variable.
        model_params: Hyperparameters for the XGBoost model.

    Returns:
        The trained XGBoost classifier object.
    """
    logging.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        **model_params, use_label_encoder=False, eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model


def evaluate_model(
    model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluates the model on the test set and returns the metrics.
    """
    logging.info("Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    logging.info(f"Model Metrics: {json.dumps(metrics, indent=2)}")
    return metrics


def save_experiment(
    model: xgb.XGBClassifier,
    metrics: Dict[str, float],
    config_path: Path,
    output_dir: Path,
):
    """Saves the model, metrics, and config to a timestamped folder."""
    # Create a unique, timestamped directory for this experiment
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = output_dir / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save the model
    model_path = experiment_dir / "xgboost_model.joblib"
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

    # Save the metrics
    metrics_path = experiment_dir / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")

    # Save a copy of the config file used for this run
    shutil.copy(config_path, experiment_dir / "main_config.yaml")
    logging.info(f"Config file saved to {experiment_dir}")


def run_training(
    config: Dict[str, Any], module_root: Path, config_path: Path
) -> None:  # noqa: E501
    """Main function to run the model training pipeline."""
    primary_data_dir = module_root / config["data_paths"]["primary_data_dir"]
    model_output_dir = module_root / "models"

    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(primary_data_dir)

    # Train model
    model = train_model(X_train, y_train, config["xgboost_params"])

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Save all experiment artifacts
    save_experiment(model, metrics, config_path, model_output_dir)
    logging.info("Model training pipeline completed successfully.")


if __name__ == "__main__":
    module_root_path = Path.cwd() / "vertexcare"
    setup_logging(module_root_path, "training")

    config_path = module_root_path / "configs" / "main_config.yaml"
    config_data = load_config(config_path)

    try:
        # Pass the config_path to the training runner
        run_training(config_data, module_root_path, config_path)
    except Exception as e:
        logging.critical(f"A critical error occurred during training: {e}")
