# vertexcare/agents/agent_tools.py

import logging
from typing import Dict, Any

import pandas as pd
import shap


def get_patient_data(patient_id: int, patient_data_df: pd.DataFrame) -> pd.DataFrame:
    """A helper function to retrieve a single patient's data."""
    if patient_data_df.empty:
        raise RuntimeError("Patient data is not loaded.")

    patient_df = patient_data_df[patient_data_df["patient_id"] == patient_id]

    if patient_df.empty:
        raise ValueError(f"Patient with ID {patient_id} not found.")
    return patient_df


def prediction_tool(patient_id: int, model: Any, imputer: Any, patient_data_df: pd.DataFrame) -> Dict[str, Any]:
    """Tool to get the readmission risk prediction for a single patient."""
    logging.info(f"PREDICTION TOOL: Getting risk score for patient {patient_id}.")
    try:
        patient_data = get_patient_data(patient_id, patient_data_df)

        features = [col for col in imputer.get_feature_names_out() if col in patient_data.columns]
        patient_features = patient_data[features]

        imputed_features = imputer.transform(patient_features)

        prediction_proba = model.predict_proba(imputed_features)[0][1]

        return {
            "patient_id": patient_id,
            "readmission_risk_score": float(round(prediction_proba, 3)),
            "status": "Success",
        }
    except Exception as e:
        logging.error(f"PREDICTION TOOL: Error for patient {patient_id}: {e}")
        return {"patient_id": patient_id, "error": str(e), "status": "Error"}


def notes_tool(patient_id: int, patient_data_df: pd.DataFrame) -> Dict[str, Any]:
    """Tool to retrieve the latest CHW notes for a single patient."""
    logging.info(f"NOTES TOOL: Getting notes for patient {patient_id}.")
    try:
        patient_data = get_patient_data(patient_id, patient_data_df)
        notes = patient_data["chw_notes"].iloc[0]

        return {
            "patient_id": patient_id,
            "notes": notes if notes else "No notes found for this patient.",
            "status": "Success",
        }
    except Exception as e:
        logging.error(f"NOTES TOOL: Error for patient {patient_id}: {e}")
        return {"patient_id": patient_id, "error": str(e), "status": "Error"}


def explanation_tool(patient_id: int, model: Any, imputer: Any, patient_data_df: pd.DataFrame) -> Dict[str, Any]:
    """Tool to get the top risk factors for a patient's prediction."""
    logging.info(f"EXPLANATION TOOL: Getting risk factors for patient {patient_id}.")
    try:
        patient_data = get_patient_data(patient_id, patient_data_df)

        features = [col for col in imputer.get_feature_names_out() if col in patient_data.columns]
        patient_features = patient_data[features]
        imputed_features_df = pd.DataFrame(imputer.transform(patient_features), columns=features)

        explainer = shap.Explainer(model, imputed_features_df)
        shap_values = explainer(imputed_features_df)

        feature_importances = pd.DataFrame({"feature": features, "importance": shap_values.values[0]}).sort_values(
            by="importance", ascending=False
        )

        top_risk_factors = feature_importances.head(3)["feature"].tolist()

        return {
            "patient_id": patient_id,
            "top_risk_factors": top_risk_factors,
            "status": "Success",
        }
    except Exception as e:
        logging.error(f"EXPLANATION TOOL: Error for patient {patient_id}: {e}")
        return {"patient_id": patient_id, "error": str(e), "status": "Error"}
