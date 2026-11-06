# vertexcare/agents/feature_glossary.py
"""Plain-language descriptions of model features for the agent's explanation tool.

A raw feature name such as `sdoh_pcp_1_0` is ambiguous to an LLM: it cannot tell whether the
patient has a PCP or lacks one, or whether the factor raises or lowers risk. Each entry states
what the feature means and whether it may inform a care plan. Descriptions follow the study
data dictionary; update both together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

PROTECTED = "Protected attribute. Do not use in care planning."
FACTOR_CAVEAT = (
    "These factors explain the model's risk prediction. They are statistical associations, "
    "not confirmed patient needs; confirm needs from the CHW notes."
)
_UNMAPPED = "No glossary entry. Do not interpret this factor."


@dataclass(frozen=True)
class FeatureInfo:
    description: str
    numeric: bool = False  # False: a 0/1 flag, shown as Yes/No
    use_in_care_plan: bool = True
    note: str = ""


FEATURE_GLOSSARY: Dict[str, FeatureInfo] = {
    "age": FeatureInfo("Age in years", numeric=True, note="Not modifiable."),
    "sex": FeatureInfo("Sex", use_in_care_plan=False, note=PROTECTED),
    "race_black": FeatureInfo("Race: Black", use_in_care_plan=False, note=PROTECTED),
    "sumcomorbidities": FeatureInfo("Number of chronic conditions", numeric=True),
    "hypertension": FeatureInfo("Has hypertension"),
    "diabetes": FeatureInfo("Has diabetes"),
    "asthma": FeatureInfo("Has asthma"),
    "engaged": FeatureInfo("Engaged with the CHW program"),
    "time_spent_total": FeatureInfo("Total CHW time spent with the patient", numeric=True),
    "referral_type_ed": FeatureInfo("Referred from the emergency department"),
    "insurance_public": FeatureInfo("Has public insurance"),
    "sdoh_pcp_1_0": FeatureInfo(
        "Reported having a regular primary care physician",
        note="No covers survey answers of No, Don't know and Refused.",
    ),
    "llm_transportation_issue": FeatureInfo("Transportation issue mentioned in CHW notes"),
    "llm_financial_concern": FeatureInfo("Financial concern mentioned in CHW notes"),
    "llm_medication_adherence_mentioned": FeatureInfo("Medication adherence mentioned in CHW notes"),
}


def _format_value(value: Any, numeric: bool) -> Any:
    if pd.isna(value):
        return "Missing"
    if not numeric and value in (0, 1):
        return "Yes" if value == 1 else "No"
    number = round(float(value), 2)
    return int(number) if number.is_integer() else number


def describe_factor(feature: str, patient_value: Any, contribution: float) -> Dict[str, Any]:
    """Turn one SHAP contribution into a self-explanatory, JSON-safe factor for the agent."""
    info = FEATURE_GLOSSARY.get(feature) or FeatureInfo(feature, numeric=True, use_in_care_plan=False, note=_UNMAPPED)
    if contribution > 0:
        effect = "raises predicted risk"
    elif contribution < 0:
        effect = "lowers predicted risk"
    else:
        effect = "no effect on predicted risk"
    factor = {
        "feature": feature,
        "description": info.description,
        "patient_value": _format_value(patient_value, info.numeric),
        "effect": effect,
        "use_in_care_plan": info.use_in_care_plan,
    }
    if info.note:
        factor["note"] = info.note
    return factor
