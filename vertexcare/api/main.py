# vertexcare/api/main.py

import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd

from vertexcare.agents.chw_agent import run_agent

# --- Configuration ---
# IMPORTANT: Update this to the timestamped folder of your best model
LATEST_MODEL_DIR = "2025-08-26_00-29-56_logistic_regression"


# --- Pre-load all necessary artifacts on startup ---
# This ensures all files are loaded once, efficiently.
MODEL = None
IMPUTER = None
PATIENT_DATA_DF = None

app = FastAPI(
    title="VertexCare Agent API",
    description="An API to run the CHW Intervention Agent.",
    version="1.0.0",
)


@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "ok", "message": "Welcome to the VertexCare API!"}


@app.on_event("startup")
def load_artifacts():
    """Load all ML artifacts when the API server starts."""
    global MODEL, IMPUTER, PATIENT_DATA_DF
    logging.info("Loading ML artifacts...")
    try:
        # Inside Docker, the working directory is /app
        base_path = Path("/app/vertexcare")
        if not base_path.exists():
            # Fallback for local development
            base_path = Path(__file__).resolve().parent.parent

        model_path = (
            base_path / "models" / LATEST_MODEL_DIR / "logistic_regression_model.joblib"
        )
        imputer_path = base_path / "models" / "imputer.joblib"
        data_path = base_path / "data" / "03_primary" / "data_with_llm_features.parquet"

        MODEL = joblib.load(model_path)
        IMPUTER = joblib.load(imputer_path)
        PATIENT_DATA_DF = pd.read_parquet(data_path)
        logging.info("ML artifacts loaded successfully.")
    except Exception as e:
        logging.critical(f"Failed to load ML artifacts on startup: {e}", exc_info=True)


class PatientRequest(BaseModel):
    patient_id: int


@app.post("/generate_plan")
async def generate_plan(request: PatientRequest):
    """
    This endpoint runs the CHW Intervention Agent for a patient.
    """
    if not all([MODEL, IMPUTER, PATIENT_DATA_DF is not None]):
        raise HTTPException(
            status_code=503,
            detail="ML artifacts are not loaded. The service is unavailable.",
        )
    try:
        logging.info(f"API: Received request for patient ID: {request.patient_id}")
        # Inject the pre-loaded artifacts into the agent
        plan = await run_agent(request.patient_id, MODEL, IMPUTER, PATIENT_DATA_DF)
        if "error" in plan:
            raise HTTPException(status_code=500, detail=plan["error"])
        return plan
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"API: An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
