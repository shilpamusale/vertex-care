# vertexcare/features/build_features.py

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers new features from the raw data."""
    logging.info("Starting feature engineering...")

    if "chw_interaction_end_time" in df.columns:
        df["chw_interaction_end_time"] = pd.to_datetime(
            df["chw_interaction_end_time"], errors="coerce"
        )
        is_ended = df["chw_interaction_end_time"].notna()
        df["has_interaction_ended"] = is_ended.astype(int)
        df = df.drop(columns=["chw_interaction_end_time"])
        logging.info("Created 'has_interaction_ended' feature.")

    logging.info("Feature engineering complete.")
    return df


def split_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the data into training and testing sets."""
    logging.info("Splitting data into training and testing sets...")

    target_column = config["model_params"]["target_column"]
    test_size = config["model_params"]["test_size"]
    random_state = config["model_params"]["random_state"]

    cols_to_drop = [target_column, "chw_notes", "visit_date"]
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(
        f"Data split complete. Train set size: {len(X_train)}"
        f", Test set size: {len(X_test)}"
    )
    return X_train, X_test, y_train, y_test


def impute_missing_values(
    X_train: pd.DataFrame, X_test: pd.DataFrame, module_root: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Imputes missing values using the median strategy."""
    logging.info("Imputing missing values with median...")

    imputer = SimpleImputer(strategy="median")

    # Fit on the training data and transform it
    X_train_imputed_np = imputer.fit_transform(X_train)

    # Only transform the test data
    X_test_imputed_np = imputer.transform(X_test)

    # Get the correct column names from the imputer itself
    # This handles cases where the imputer might drop a column
    imputed_columns = imputer.get_feature_names_out()

    # Recreate the DataFrames with the correct columns
    X_train_imputed = pd.DataFrame(X_train_imputed_np, columns=imputed_columns)
    X_test_imputed = pd.DataFrame(X_test_imputed_np, columns=imputed_columns)

    # Save the imputer so we can use it on new data later
    imputer_path = module_root / "models" / "imputer.joblib"
    joblib.dump(imputer, imputer_path)
    logging.info(f"Imputer saved to {imputer_path}")

    return X_train_imputed, X_test_imputed


def run_feature_engineering(config: Dict[str, Any], module_root: Path) -> None:
    """Main function to run the feature engineering process."""
    logging.info("Starting feature engineering pipeline...")
    intermediate_path = config["data_paths"]["intermediate_data_dir"]
    intermediate_dir = module_root / intermediate_path
    primary_dir = module_root / config["data_paths"]["primary_data_dir"]
    primary_dir.mkdir(parents=True, exist_ok=True)

    input_file = intermediate_dir / "ingested_data.parquet"
    df = pd.read_parquet(input_file)
    df_featured = create_features(df)
    X_train, X_test, y_train, y_test = split_data(df_featured, config)

    # Add the imputation step
    X_train, X_test = impute_missing_values(X_train, X_test, module_root)

    logging.info("Saving processed data sets...")
    X_train.to_parquet(primary_dir / "X_train.parquet", index=False)
    X_test.to_parquet(primary_dir / "X_test.parquet", index=False)
    y_train.to_frame().to_parquet(primary_dir / "y_train.parquet", index=False)
    y_test.to_frame().to_parquet(primary_dir / "y_test.parquet", index=False)

    logging.info("Feature engineering pipeline completed successfully.")
