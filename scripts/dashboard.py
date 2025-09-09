# dashboard.py

import streamlit as st
import requests
import os
from requests.exceptions import JSONDecodeError

# --- Page Configuration ---
st.set_page_config(
    page_title="VertexCare Intervention Planner",
    layout="wide",
)

# --- UI Components ---
st.title("VertexCare Intervention Planner")
st.markdown(
    "Enter a patient ID below to generate a " "prioritized intervention plan using the " "AI-powered CHW Agent."
)

patient_id = st.number_input("Enter Patient ID:", min_value=101, step=1, value=101)

if st.button("Generate Plan", type="primary"):
    with st.spinner("The AI agent is analyzing the case..."):
        try:
            # --- API Call ---

            # # To test the deployed version, uncomment the other URL.
            # api_url = "http://127.0.0.1:8000/generate_plan"
            # # api_url = (
            # #     "https://vertexcare-api-678532812483.us-central1.run.app/generate_plan"
            # # )

            # --- Dynamic API Call ---
            # Get the base URL from an environment variable, defaulting to localhost.
            API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
            api_url = f"{API_BASE_URL}/generate_plan"

            # Optional: Display which API is being used, for easier debugging.
            # st.info(f"Connecting to API at: {API_BASE_URL}")
            response = requests.post(api_url, json={"patient_id": patient_id})

            response.raise_for_status()

            plan = response.json()

            # --- Display Results ---
            st.success("Intervention plan generated successfully!")

            st.subheader("Patient Risk Assessment")
            col1, col2 = st.columns(2)
            col1.metric("Patient ID", plan.get("patient_id"))

            # The key "readmission_risk_score" now
            # exactly matches the JSON output from the agent.
            risk_score = plan.get("readmission_risk_score", "N/A")

            if isinstance(risk_score, (int, float)):
                col2.metric("Readmission Risk Score", f"{risk_score * 100:.1f}%")
            else:
                col2.metric("Readmission Risk Score", "N/A")

            st.subheader("Risk Summary")
            st.info(plan.get("risk_summary", "No summary available."))

            st.subheader("Recommended Actions")
            actions = plan.get("recommended_actions", [])
            if actions:
                for action in actions:
                    st.markdown(f"- **{action.get('action')}** " f"(Priority: `{action.get('priority')}`)")
            else:
                st.write("No specific actions recommended at this time.")

        except requests.exceptions.HTTPError as e:
            error_message = "An unknown error occurred on the server."
            try:
                error_message = e.response.json().get("detail", "No detail key in JSON response.")
            except JSONDecodeError:
                error_message = e.response.text if e.response.text else "The API returned an empty error response."
            st.error(f"API Error (Status Code: {e.response.status_code}): {error_message}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
