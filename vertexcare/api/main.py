# vertexcare/api/main.py

import logging
import uvicorn

# from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from vertexcare.agents.chw_agent import run_agent

# Initailize the FastAPI runner
app = FastAPI(
    title="VertextCare Agent API",
    description="An API to run the CHW intervention Agent.",
    version="1.0.0",
)


class PatientRequest(BaseModel):
    patient_id: int


@app.post("/generate_plan")
async def generate_plan(request: PatientRequest):
    """
    This is the endpoint that takes a patient id as input,
    runs the CHW intervention agent, and returns the generated
    intervention steps.
    """

    try:
        logging.info(f"API: Received request for the patient ID: {request.patient_id}")

        plan = await run_agent(request.patient_id)
        if "error" in plan:
            raise HTTPException(status_code=500, detail=plan["error"])
        return plan
    except ValueError as e:
        # "Patient not found"
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"API: An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500, detail="An internal serevr error occurred."
        )


# This allows running the server directly for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
