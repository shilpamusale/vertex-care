# vertexcare/models/train_model.py

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from datetime import datetime
import shutil
import argparse

import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    recall_score,
    precision_score,
)

from vertexcare.data_processing.ingestion import setup_logging, load_config


def load_processed_data(
    primary_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads the processed training and testing data sets."""
    logging.info("Loading the processed data...")
    X_train = pd.read_parquet(primary_dir / "X_train.parquet")
    X_test = pd.read_parquet(primary_dir / "X_test.parquet")
    y_train = pd.read_parquet(primary_dir / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(primary_dir / "y_test.parquet").squeeze()
    return X_train, X_test, y_train, y_test


def get_model(model_name: str, model_params: Dict[str, Any]) -> Any:
    """Initializes a model based on the provided name."""
    logging.info(f" Initailizing {model_name} model ...")
    if model_name == "logistic_regression":
        return LogisticRegression(**model_params)
    elif model_name == "xgboost":
        return xgb.XGBClassifier(
            **model_params, use_label_encoder=False, eval_metric="logloss"
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Trains a given model.
    """
    logging.info(f"Training {model.__class__.__name__} model...")
    model.fit(X_train, y_train)
    logging.info("Model training complete.")
    return model


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
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
    model: Any,
    metrics: Dict[str, float],
    config_path: Path,
    model_params_path: Path,
    output_dir: Path,
    model_name: str,
):
    """
    Saves all experiment artifacts to a timestamped folder.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = output_dir / f"{timestamp}_{model_name}"
    experiment_dir.mkdir(exist_ok=True, parents=True)

    model_path = experiment_dir / f"{model_name}_model.joblib"
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

    metrics_path = experiment_dir / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")

    shutil.copy(config_path, experiment_dir / "main_config.yaml")
    shutil.copy(model_params_path, experiment_dir / "model_params.yaml")
    logging.info(f"Config files saved to {experiment_dir}")


def run_training(
    config: Dict[str, Any],
    model_params_config: Dict[str, Any],
    module_root: Path,
    config_path: Path,
    model_params_path: Path,
    model_name: str,
) -> None:
    """
    Main function to run the model training pipeline.
    """
    primary_data_dir = module_root / config["data_paths"]["primary_data_dir"]
    model_output_dir = module_root / "models"

    X_train, X_test, y_train, y_test = load_processed_data(primary_data_dir)
    model_params = model_params_config[f"{model_name}_params"]
    model = get_model(model_name, model_params)

    trained_model = train_model(model, X_train, y_train)
    mertics = evaluate_model(trained_model, X_test, y_test)
    save_experiment(
        trained_model,
        mertics,
        config_path,
        model_params_path,
        model_output_dir,
        model_name,
    )
    logging.info("Model training pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model training script for VertexCare."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["logistic_regression", "random_forest", "xgboost"],
        help="The model to train.",
    )
    args = parser.parse_args()

    module_root_path = Path.cwd() / "vertexcare"
    setup_logging(module_root_path, f"training_{args.model}")

    config_path = module_root_path / "configs" / "main_config.yaml"
    model_params_path = module_root_path / "configs" / "model_params.yaml"

    config_data = load_config(config_path)
    model_params_data = load_config(model_params_path)

    try:
        run_training(
            config_data,
            model_params_data,
            module_root_path,
            config_path,
            model_params_path,
            args.model,
        )
    except Exception as e:
        logging.critical(
            f"A critical error occurred during training: {e}", exc_info=True
        )
