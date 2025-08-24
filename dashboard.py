# dashboard.py

import streamlit as st
import requests

# import json

# --- Page Configuration ---
st.set_page_config(
    page_title="VertexCare Intervention Planner",
    layout="wide",
)

# --- UI Components ---
st.title("VertexCare Intervention Planner")
st.markdown(
    "Enter a patient ID below to generate a "
    "prioritized intervention plan "
    "using the AI-powered CHW Agent."
)

patient_id = st.number_input("Enter Patient ID:", min_value=101, step=1, value=101)

if st.button("Generate Plan", type="primary"):
    with st.spinner("The AI agent is analyzing the case..."):
        try:
            # --- API Call ---
            api_url = "http://127.0.0.1:8000/generate_plan"
            response = requests.post(api_url, json={"patient_id": patient_id})

            response.raise_for_status()

            plan = response.json()

            # --- Display Results ---
            st.success("Intervention plan generated successfully!")

            st.subheader("Patient Risk Assessment")
            col1, col2 = st.columns(2)
            col1.metric("Patient ID", plan.get("patient_id"))

            risk_score = plan.get("risk_score", "N/A")
            if isinstance(risk_score, (int, float)):
                col2.metric("Readmission Risk Score", f"{risk_score:.1%}")
            else:
                col2.metric("Readmission Risk Score", "N/A")

            st.subheader("Risk Summary")
            st.info(plan.get("risk_summary", "No summary available."))

            st.subheader("Recommended Actions")
            actions = plan.get("recommended_actions", [])
            if actions:
                for action in actions:
                    action_text = action.get("action", "N/A")
                    priority = action.get("priority", "N/A")
                    st.markdown(f"- **{action_text}** (Priority: `{priority}`)")
            else:
                st.write("No specific actions recommended at this time.")

        except requests.exceptions.HTTPError as e:
            st.error(
                f"Error from API: {e.response.json().get('detail', 'Unknown error')}"
            )
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
