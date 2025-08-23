# tests/test_ingestion.py

import pandas as pd
import pytest
from pathlib import Path

# Import the function we want to test
from vertexcare.data_processing.ingestion import run_ingestion


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
    project_root = mock_project_dirs
    module_root = project_root / "vertexcare"
    raw_data_dir = module_root / "data" / "01_raw"

    mock_config = {
        "data_paths": {
            "raw_data_dir": "data/01_raw/",
            "intermediate_data_dir": "data/02_intermediate/",
        }
    }

    # CORRECTED: Renamed 'Patient.ID' to 'record_id' to avoid conflict
    mock_raw_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "Age ": [65, 72, 55],
            "CHW_Notes": ["Test note 1", None, "Test note 3"],
        }
    )
    mock_raw_csv_path = raw_data_dir / "mock_data_with_notes.csv"
    mock_raw_data.to_csv(mock_raw_csv_path, index=False)

    # --- 2. Act ---
    run_ingestion(mock_config, module_root)

    # --- 3. Assert ---
    intermediate_dir = module_root / "data" / "02_intermediate"
    output_file = intermediate_dir / "ingested_data.parquet"

    assert output_file.exists(), "Output parquet file was not created."

    result_df = pd.read_parquet(output_file)

    # Check that the new 'patient_id' was created and others were standardized
    expected_columns = ["patient_id", "record_id", "age", "chw_notes"]
    assert (
        list(result_df.columns) == expected_columns
    ), "Column names not standardized correctly."

    assert result_df["chw_notes"].iloc[1] == "", "Null value in notes was not filled."
