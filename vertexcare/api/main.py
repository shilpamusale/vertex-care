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
# MENTOR NOTE: This is the critical fix. By placing this here,
# logging is configured as soon as the file is imported.
logging.basicConfig(level=logging.INFO)


# --- Configuration ---
LATEST_MODEL_DIR = "2025-08-26_00-29-56_logistic_regression"
ml_assets = {}


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML artifacts when the API server starts."""
    logging.info("--- Loading ML artifacts... ---")
    try:
        base_path = Path("/app/vertexcare")
        if not base_path.exists():
            base_path = Path(__file__).resolve().parent.parent

        model_path = (
            base_path / "models" / LATEST_MODEL_DIR / "logistic_regression_model.joblib"
        )
        imputer_path = base_path / "models" / "imputer.joblib"
        data_path = base_path / "data" / "03_primary" / "data_with_llm_features.parquet"

        ml_assets["model"] = joblib.load(model_path)
        ml_assets["imputer"] = joblib.load(imputer_path)

        patient_data = pd.read_parquet(data_path)
        if (
            "patient_id" not in patient_data.columns
            and patient_data.index.name == "patient_id"
        ):
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
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )


if __name__ == "__main__":
    # This block is now only for direct execution, not for Uvicorn/Gunicorn.
    uvicorn.run(app, host="0.0.0.0", port=8000)
