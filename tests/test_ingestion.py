# tests/test_ingestion.py

import pandas as pd
import pytest
from pathlib import Path

# Import the function we want to test
from vertexcare.data_processing.ingestion import run_ingestion


# Pytest fixtures are a powerful way to provide
# #data and resources to your tests.
# This fixture creates a temporary
# directory structure for our test to run in,
# so we don't have to use our actual
# project's data folders.
@pytest.fixture
def mock_project_dirs(tmp_path: Path) -> Path:
    """Creates a temporary directory structure mimicking our project."""
    module_root = tmp_path / "vertexcare"
    (module_root / "data" / "01_raw").mkdir(parents=True)
    (module_root / "data" / "02_intermediate").mkdir(parents=True)
    (module_root / "configs").mkdir(parents=True)
    return tmp_path


def test_run_ingestion_success(mock_project_dirs: Path):
    """
    Tests the happy path of the run_ingestion function.

    It checks for:
    1. Correct file creation.
    2. Column name standardization.
    3. Handling of null values in the notes column.
    """
    # --- 1. Arrange ---
    # Setup the necessary inputs for our function.
    project_root = mock_project_dirs
    module_root = project_root / "vertexcare"
    raw_data_dir = module_root / "data" / "01_raw"

    # Create a dummy config dictionary
    mock_config = {
        "data_paths": {
            "raw_data_dir": "data/01_raw/",
            "intermediate_data_dir": "data/02_intermediate/",
        }
    }

    # Create a dummy raw CSV file with messy column names and a missing note
    mock_raw_data = pd.DataFrame(
        {
            "Patient.ID": [1, 2, 3],
            "Age ": [65, 72, 55],
            "CHW_Notes": ["Test note 1", None, "Test note 3"],
        }
    )
    mock_raw_csv_path = raw_data_dir / "mock_data_with_notes.csv"
    mock_raw_data.to_csv(mock_raw_csv_path, index=False)

    # --- 2. Act ---
    # Run the function we are testing.
    run_ingestion(mock_config, module_root)

    # --- 3. Assert ---
    # Check that the outcomes are what we expect.
    ingested_path = "data" / "02_intermediate" / "ingested_data.parquet"
    output_file = module_root / ingested_path

    # Assert that the output file was actually created.
    assert output_file.exists(), "Output parquet file was not created."

    # Assert that the data inside the file is correct.
    result_df = pd.read_parquet(output_file)

    # Check that column names were standardized (lowercase, no spaces/dots).
    expected_columns = ["patient_id", "age", "chw_notes"]
    assert (
        list(result_df.columns) == expected_columns
    ), "Column names not standardized correctly."

    # Check that the NaN in the notes column was filled with an empty string.
    assert result_df["chw_notes"].iloc[1] == "", "Null value in notes."
