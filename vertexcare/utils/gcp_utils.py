# vertexcare/utils/gcp_utils.py
import os
from google.cloud import secretmanager


def get_gemini_api_key() -> str:
    """Fetches the Gemini API key from Google Cloud Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.environ.get("GCP_PROJECT", "vertexcare")
        secret_name = f"projects/{project_id}/secrets/gemini-api-key/versions/latest"

        response = client.access_secret_version(request={"name": secret_name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        # In a real application, you'd have more robust error handling
        print(f"FATAL: Could not access Gemini API key from Secret Manager. Error: {e}")
        return None
