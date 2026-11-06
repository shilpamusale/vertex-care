# vertexcare/agents/agent_tools.py

import logging
from typing import Dict, Any

import pandas as pd
import shap

from vertexcare.agents.feature_glossary import FACTOR_CAVEAT, describe_factor


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


# Number of reference patients SHAP compares against. Fixed seed keeps explanations reproducible.
SHAP_BACKGROUND_SIZE = 100


def _impute(df: pd.DataFrame, features: list, imputer: Any) -> pd.DataFrame:
    return pd.DataFrame(imputer.transform(df[features]), columns=features)


def shap_contributions(patient_id: int, model: Any, imputer: Any, patient_data_df: pd.DataFrame) -> pd.Series:
    """Per-feature SHAP contributions to one patient's predicted risk, largest first.

    SHAP explains a prediction relative to a background (reference) population, which must be
    other patients: explaining a patient against themselves makes every contribution zero.
    """
    patient_data = get_patient_data(patient_id, patient_data_df)
    features = [col for col in imputer.get_feature_names_out() if col in patient_data.columns]
    background = patient_data_df.sample(n=min(SHAP_BACKGROUND_SIZE, len(patient_data_df)), random_state=0)
    explainer = shap.Explainer(model, _impute(background, features, imputer))
    values = explainer(_impute(patient_data, features, imputer)).values[0]
    if values.ndim == 2:  # some explainers return one column per class; keep the positive class
        values = values[:, 1]
    return pd.Series(values, index=features).sort_values(ascending=False)


def explanation_tool(patient_id: int, model: Any, imputer: Any, patient_data_df: pd.DataFrame) -> Dict[str, Any]:
    """Tool to get the top risk factors for a patient's prediction."""
    logging.info(f"EXPLANATION TOOL: Getting risk factors for patient {patient_id}.")
    try:
        contributions = shap_contributions(patient_id, model, imputer, patient_data_df)
        patient_row = get_patient_data(patient_id, patient_data_df).iloc[0]
        top_risk_factors = [describe_factor(f, patient_row[f], c) for f, c in contributions.head(3).items()]

        return {
            "patient_id": patient_id,
            "top_risk_factors": top_risk_factors,
            "caveat": FACTOR_CAVEAT,
            "status": "Success",
        }
    except Exception as e:
        logging.error(f"EXPLANATION TOOL: Error for patient {patient_id}: {e}")
        return {"patient_id": patient_id, "error": str(e), "status": "Error"}
