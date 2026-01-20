import json
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"

        if "run.app" in ip:
            self.base_url = f"https://{ip}"

        logger.info(f"Initializing client; base URL: {self.base_url}")

        self.features = features
        self.model = None  # internal state tracker for the currently loaded model name.
        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.

        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        # Check if non-empty data
        if X.empty:
            raise ValueError('Input DataFrame is empty.')

        # Convert dataframe to a JSON format that Flask understand
        payload = X.to_dict(orient='list')  # list of dicts

        # POST request to the /predict endpoint.
        try:
            response = requests.post(f"{self.base_url}/predict", json=payload)
            # Catch server errors
            response.raise_for_status()
            data = response.json()

            # Prediction service return list of predictions
            # Convert back into dataframe using original input index for alignment
            preds = pd.DataFrame(data['predictions'], index=X.index, columns=['prediction'])
            return preds

        except requests.RequestException as e:
            logger.error(f"Prediction request failed: {e}")
            raise

    def logs(self) -> dict:
        """Get server logs"""

        try:
            # GET request to fetch logs
            response = requests.get(f"{self.base_url}/logs")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Logs request failed: {e}")
            raise

    def download_registry_model(self, entity: str, artifact_name: str, project: str = "IFT6758.2025-A03",
                                version: str = "latest") -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it.

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model

        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """
        # Construct the payload containing all necessary metadata for Wandb
        payload = {
            "entity": entity,
            "project": project,
            "artifact_name": artifact_name,
            "version": version
        }

        try:
            # POST request to trigger the model download/load logic on the server
            response = requests.post(f"{self.base_url}/download_registry_model", json=payload)
            response.raise_for_status()
            result = response.json()

            # Update internal state if ther model swap succeeded.
            # This is useful for UI feedback (modified by Remi to do the streamlit app)
            if result.get("success"):
                self.model = result["model"]  # store the full model name if available
                self.features = result.get("features")  # update feature list if the server returns it
            return result
        except requests.RequestException as e:
            logger.error(f"Download registry model request failed: {e}")
            raise