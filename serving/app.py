import os
import sys
import logging
from flask import Flask, jsonify, request, abort
import pandas as pd
import joblib
import wandb
from dotenv import load_dotenv

# Ajoute le repertoire racine du projet au pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

#Log file location
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")
MODELS_DIR = "models" #Directory where downloaded models will be stored
app = Flask(__name__) #Initialize flask app

#Variable globale pour le gestionnaire de modeles
current_model = None
current_model_name = None
current_features = None
# Comme mon ordi est un windows, jutilise waitress donc je mets le code d'initialisation ici
load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

#Configuring logging both to file and to console
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

app.logger.info("Flask service starting.")
os.makedirs(MODELS_DIR, exist_ok=True)

@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    app.logger.info("GET request received on /logs endpoint.")

    #If log file does not exist yet, return an empty response
    if not os.path.exists(LOG_FILE):
        app.logger.warning(f"Log file {LOG_FILE} does not exist yet")
        return jsonify({"logs": [], "message": "No logs yet"}), 200

    try:
        #Read full log content
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()

        app.logger.info(f"Successfully read {len(lines)} log lines")

        return jsonify({
            "logs": lines[-100:], #return last 100 lines
            "total": len(lines)
        })
    # In case of a problem, we log the error
    except Exception as e:
        app.logger.error(f"Error reading log file: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model
    """

    global current_model, current_model_name, current_features

    # Parse incoming POST request data
    data = request.get_json()
    app.logger.info(f"POST request received on /download_registry_model with data: {data}")

    # Required parameters to locate a model in Wandb
    entity = data.get("entity")
    project = data.get("project")
    artifact_name = data.get("artifact_name")
    version = data.get("version", "latest")

    # Basic validation
    if not (entity and project and artifact_name):
        app.logger.error("Missing required parameters: entity, project or artifact_name")
        return jsonify({"success": False, "error": "Missing parameters"}), 400

    # Build local path
    full_name = f"{entity}/{project}/{artifact_name}:{version}"
    local_filename = f"{artifact_name.replace('/', '_')}_{version}.pkl"
    local_path = os.path.join(MODELS_DIR, local_filename)

    app.logger.info(f"Request to load model: {full_name}")

    try:
        # Try load locally
        if os.path.exists(local_path):
            app.logger.info(f"Loading local file: {local_path}")
            model_obj = joblib.load(local_path)
        else:
            # Download from WandB
            app.logger.info(f"Downloading from WandB...")
            api = wandb.Api()
            artifact = api.artifact(full_name)
            artifact_dir = artifact.download()

            # Find the model file (pkl or joblib)
            model_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pkl") or f.endswith(".joblib")]
            if not model_files:
                raise FileNotFoundError("No .pkl or .joblib file found in artifact")

            src = os.path.join(artifact_dir, model_files[0])

            # Save to local cache
            model_obj = joblib.load(src)
            joblib.dump(model_obj, local_path)
            app.logger.info(f"Downloaded and cached at {local_path}")

        # Update global state
        current_model = model_obj
        current_model_name = "XGBoost"
        current_features = list(current_model.feature_names_in_)
        app.logger.info(f"Detected features: {current_features}")

        return jsonify({"success": True, "model": full_name, "features": current_features})

    except Exception as e:
        app.logger.error(f"Failed to load model: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    global current_model, current_features

    app.logger.info("POST request received on /predict endpoint")

    #Ensure a model is loaded before handling predictions
    if current_model is None:
        app.logger.info("No model loaded. Please load a model using /download_registry_model first.")
        return jsonify({"error": "No model loaded"}), 50

    # Get data
    data = request.get_json()
    if data is None:
        app.logger.info("No data provided in request body")
        return jsonify({"error": "No data provided"}), 400

    try:
        # Input must be a dict of lists matching training features
        X = pd.DataFrame.from_dict(data)
        app.logger.info("Received data for prediction.")

    except Exception as e:
        app.logger.info(f"Invalid JSON format: {str(e)}")
        return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    try:
        # Prefer predict_proba if available (classification model)
        if hasattr(current_model, "predict_proba"):
            app.logger.info("Using predict_proba() for predictions.")
            # Predict goal probability
            preds = current_model.predict_proba(X)
            # If binary classification, take the probability of class 1 (Goal)
            if preds.ndim > 1:
                preds = preds[:, 1]
        else:
            #Fallback for models without predict_proba
            app.logger.info("Using predict() for predictions (predict_proba not available)")
            preds = current_model.predict(X).tolist()

        app.logger.info(f"PREDICTION SUCCESS: Generated {len(preds)} predictions using model {current_model_name}")

        #Construct the final JSON response with the results
        response = {
            "predictions": preds.tolist(),
            "n_samples": len(preds),
            "model": current_model_name
        }

        return jsonify(response)

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        app.logger.error(error_msg)
        app.logger.error(f"Error type: {type(e).__name__}")

        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "model": current_model_name
        }), 500