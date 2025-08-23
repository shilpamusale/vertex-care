# vertexcare/agents/agent_tools.py

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import joblib
import shap

# --- Configuration ---

LATEST_MODEL_DIR = "2025-08-23_22-49-59_logistic_regression"
# --- End Configuration ---


# --- Pre-load all necessary artifacts ---

MODULE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    MODULE_ROOT / "models" / LATEST_MODEL_DIR / "logistic_regression_model.joblib"
)
IMPUTER_PATH = MODULE_ROOT / "models" / "imputer.joblib"
DATA_PATH = MODULE_ROOT / "data" / "03_primary" / "data_with_llm_features.parquet"

try:
    MODEL = joblib.load(MODEL_PATH)
    IMPUTER = joblib.load(IMPUTER_PATH)
    PATIENT_DATA_DF = pd.read_parquet(DATA_PATH)
    logging.info("Successfully pre-loaded model, imputer, and patient data.")
except Exception as e:
    logging.error(f"Failed to pre-load artifacts: {e}")

    MODEL, IMPUTER, PATIENT_DATA_DF = None, None, pd.DataFrame()


# --- Tool Implementations ---


def get_patient_data(patient_id: int) -> pd.DataFrame:
    """A helper function to retrieve a single patient's data."""
    if PATIENT_DATA_DF.empty:
        raise RuntimeError("Patient data is not loaded.")

    patient_df = PATIENT_DATA_DF[PATIENT_DATA_DF["patient_id"] == patient_id]

    if patient_df.empty:
        raise ValueError(f"Patient with ID {patient_id} not found.")
    return patient_df


def prediction_tool(patient_id: int) -> Dict[str, Any]:
    """
    Tool to get the readmission risk prediction for a single patient.
    """
    logging.info(f"PREDICTION TOOL: Getting risk score for patient {patient_id}.")
    try:
        patient_data = get_patient_data(patient_id)

        features = [
            col
            for col in IMPUTER.get_feature_names_out()
            if col in patient_data.columns
        ]
        patient_features = patient_data[features]

        imputed_features = IMPUTER.transform(patient_features)

        prediction_proba = MODEL.predict_proba(imputed_features)[0][1]

        return {
            "patient_id": patient_id,
            "readmission_risk_score": round(prediction_proba, 3),
            "status": "Success",
        }
    except Exception as e:
        logging.error(f"PREDICTION TOOL: Error for patient {patient_id}: {e}")
        return {"patient_id": patient_id, "error": str(e), "status": "Error"}


def notes_tool(patient_id: int) -> Dict[str, Any]:
    """
    Tool to retrieve the latest CHW notes for a single patient.
    """
    logging.info(f"NOTES TOOL: Getting notes for patient {patient_id}.")
    try:
        patient_data = get_patient_data(patient_id)
        notes = patient_data["chw_notes"].iloc[0]

        return {
            "patient_id": patient_id,
            "notes": notes if notes else "No notes found for this patient.",
            "status": "Success",
        }
    except Exception as e:
        logging.error(f"NOTES TOOL: Error for patient {patient_id}: {e}")
        return {"patient_id": patient_id, "error": str(e), "status": "Error"}


def explanation_tool(patient_id: int) -> Dict[str, Any]:
    """
    Tool to get the top risk factors (explanation) for a patient's prediction.
    """
    logging.info(f"EXPLANATION TOOL: Getting risk factors for patient {patient_id}.")
    try:
        patient_data = get_patient_data(patient_id)

        features = [
            col
            for col in IMPUTER.get_feature_names_out()
            if col in patient_data.columns
        ]
        patient_features = patient_data[features]
        imputed_features_df = pd.DataFrame(
            IMPUTER.transform(patient_features), columns=features
        )

        explainer = shap.Explainer(MODEL, imputed_features_df)
        shap_values = explainer(imputed_features_df)

        feature_importances = pd.DataFrame(
            {"feature": features, "importance": shap_values.values[0]}
        ).sort_values(by="importance", ascending=False)

        top_risk_factors = feature_importances.head(3)["feature"].tolist()

        return {
            "patient_id": patient_id,
            "top_risk_factors": top_risk_factors,
            "status": "Success",
        }
    except Exception as e:
        logging.error(f"EXPLANATION TOOL: Error for patient {patient_id}: {e}")
        return {"patient_id": patient_id, "error": str(e), "status": "Error"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_patient_id = 101

    print("--- Testing Prediction Tool ---")
    prediction_result = prediction_tool(test_patient_id)
    print(prediction_result)

    print("\n--- Testing Notes Tool ---")
    notes_result = notes_tool(test_patient_id)
    print(notes_result)

    print("\n--- Testing Explanation Tool ---")
    explanation_result = explanation_tool(test_patient_id)
    print(explanation_result)
