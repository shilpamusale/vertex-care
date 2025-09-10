# vertexcare/api/main.py

import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd

from vertexcare.agents.chw_agent import run_agent

# --- Configure Logging ---

# logging is configured as soon as the file is imported.
logging.basicConfig(level=logging.INFO)


def find_latest_model_dir(models_base_path: Path) -> Path:
    """
    Scans a directory for folders matching the model pattern and returns the latest one.
    """
    # Find all directories that match the naming pattern
    search_pattern = "*_logistic_regression"
    matching_dirs = [p for p in models_base_path.glob(search_pattern) if p.is_dir()]

    # Raise an error if no model directories are found
    if not matching_dirs:
        raise FileNotFoundError(f"No model directories found in {models_base_path}")

    # Sort the directories by name (YYYY-MM-DD format ensures this works)
    # and return the last one, which is the most recent.
    latest_dir = sorted(matching_dirs)[-1]
    logging.info(f"Found latest model directory: {latest_dir.name}")
    return latest_dir


# --- Configuration ---
# LATEST_MODEL_DIR = "2025-08-26_00-29-56_logistic_regression"
ml_assets = {}


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML artifacts when the API server starts."""
    logging.info("--- Loading ML artifacts... ---")
    try:
        base_path = Path("/app")
        if not base_path.exists():
            base_path = Path(__file__).resolve().parent.parent.parent

        # 1. Dynamically find the latest model directory
        models_dir = base_path / "models"
        latest_model_dir_path = find_latest_model_dir(models_dir)

        # 2. Construct the full path to the model file
        model_path = latest_model_dir_path / "logistic_regression_model.joblib"
        # intermediate_dir = Path(config["data_paths"]["intermediate_data_dir"])
        imputer_path = base_path / "models" / "imputer.joblib"
        data_path = base_path / "data" / "02_intermediate" / "data_with_llm_features.parquet"

        ml_assets["model"] = joblib.load(model_path)
        ml_assets["imputer"] = joblib.load(imputer_path)

        patient_data = pd.read_parquet(data_path)
        if "patient_id" not in patient_data.columns and patient_data.index.name == "patient_id":
            patient_data.reset_index(inplace=True)
        ml_assets["patient_data"] = patient_data

        logging.info("--- ML artifacts loaded successfully. ---")
    except Exception as e:
        logging.critical(f"Failed to load ML artifacts on startup: {e}", exc_info=True)

    yield

    logging.info("--- Clearing ML artifacts... ---")
    ml_assets.clear()


# --- FastAPI App Initialization ---
app = FastAPI(
    title="VertexCare Agent API",
    description="An API to run the CHW Intervention Agent.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic Model ---
class PatientRequest(BaseModel):
    patient_id: int


# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the VertexCare API!"}


@app.post("/generate_plan")
async def generate_plan(request: PatientRequest):
    """Runs the CHW Intervention Agent for a patient."""
    if not ml_assets:
        raise HTTPException(
            status_code=503,
            detail="ML artifacts are not loaded. The service is unavailable.",
        )
    try:
        logging.info(f"API: Received request for patient ID: {request.patient_id}")

        plan = await run_agent(
            patient_id=request.patient_id,
            model=ml_assets["model"],
            imputer=ml_assets["imputer"],
            patient_data_df=ml_assets["patient_data"],
        )

        if "error" in plan:
            raise HTTPException(status_code=500, detail=plan["error"])
        return plan
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"API: An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


if __name__ == "__main__":
    # This block is now only for direct execution, not for Uvicorn/Gunicorn.
    uvicorn.run(app, host="0.0.0.0", port=8000)
