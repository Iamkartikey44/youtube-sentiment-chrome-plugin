import json
import mlflow
import logging
import os
import dagshub

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set up MLflow tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/Iamkartikey44/youtube-sentiment-chrome-plugin.mlflow"
)

# dagshub.init(
#     repo_owner='Iamkartikey44',
#     repo_name='youtube-sentiment-chrome-plugin',
#     mlflow=True
# )



# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        #model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        #model_uri = f"models:/{model_info['model_path']}/staging"
        
        # Register the model
        #model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(model_name,stages=["None"])
        version = model_version[0].version
        print(f"Model Version: {version}")

        #version = model_version[0].version

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
        
        logger.debug(f'Model {model_name} version {model_version[0].version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "lgbm_model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()