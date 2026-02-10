import json
import mlflow
import logging
import os

#mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_tracking_uri("file:./mlruns") 
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler =logging.StreamHandler()
console_handler.setLevel("DEBUG")
file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path,'r') as file:
            model_info = json.load(file)
        logger.debug(f"Model info loaded from : {file_path}")

        return model_info
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the model info: {e}") 

def register_model(model_name: str,model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri,model_name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage='Staging'
        )

        logger.debug(f"Model {model_name} version {model_version.version} registered and transitioned to Staging.")
    
    except Exception as e:
        logger.error(f"Error during model registration: {e}")
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        model_name = "yt_chrome_plugin_model"
        register_model(model_name,model_info)

    except Exception as e:
        logger.error(f"Failed to complete the model registration process: {e}")

if __name__ == '__main__':
    main()