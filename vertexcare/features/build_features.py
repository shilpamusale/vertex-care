# vertexcare/features/build_features.py

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new features from the raw data.

    Args:
        df: The input DataFrame after initial processing.

    Returns:
        DataFrame with new, engineered features.
    """
    logging.info("Starting feature engineering...")

    # --- Datetime Feature Engineering ---
    # Convert timestamp to a more useful binary feature.
    if "chw_interaction_end_time" in df.columns:
        df["chw_interaction_end_time"] = pd.to_datetime(
            df["chw_interaction_end_time"], errors="coerce"
        )
        is_ended = df["chw_interaction_end_time"].notna()
        df["has_interaction_ended"] = is_ended.astype(int)
        # Drop the original column as it's no longer needed
        df = df.drop(columns=["chw_interaction_end_time"])
        logging.info("Created 'has_interaction_ended' feature.")

    # In the future, more complex feature engineering can be added here.
    # For example, scaling numerical features or creating interaction terms.

    logging.info("Feature engineering complete.")
    return df


def split_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into training and testing sets.

    Args:
        df: The full DataFrame with features.
        config: The project configuration dictionary.

    Returns:
        A tuple containing X_train, X_test, y_train, y_test.
    """
    logging.info("Splitting data into training and testing sets...")

    target_column = config["model_params"]["target_column"]
    test_size = config["model_params"]["test_size"]
    random_state = config["model_params"]["random_state"]

    X = df.drop(columns=[target_column, "chw_notes"])  # Drop notes for now
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logging.info(
        f"Data split complete. Train set size: {len(X_train)}"
        f", Test set size: {len(X_test)}"
    )
    return X_train, X_test, y_train, y_test


def run_feature_engineering(config: Dict[str, Any], module_root: Path) -> None:
    """Main function to run the feature engineering process."""
    logging.info("Starting feature engineering pipeline...")
    intermediate_path = config["data_paths"]["intermediate_data_dir"]
    intermediate_dir = module_root / intermediate_path
    primary_dir = module_root / config["data_paths"]["primary_data_dir"]
    primary_dir.mkdir(parents=True, exist_ok=True)

    input_file = intermediate_dir / "ingested_data.parquet"

    logging.info(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)

    # Create features
    df_featured = create_features(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df_featured, config)

    # Save the processed data sets
    logging.info("Saving processed data sets...")
    X_train.to_parquet(primary_dir / "X_train.parquet", index=False)
    X_test.to_parquet(primary_dir / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(primary_dir / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(primary_dir / "y_test.parquet", index=False)

    logging.info("Feature engineering pipeline completed successfully.")
