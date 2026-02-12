import mlflow
import mlflow.pyfunc
import pytest
import dagshub
from mlflow.tracking import MlflowClient


# Set MLflow tracking
mlflow.set_tracking_uri(
    "https://dagshub.com/Iamkartikey44/youtube-sentiment-chrome-plugin.mlflow"
)

dagshub.init(
    repo_owner='Iamkartikey44',
    repo_name='youtube-sentiment-chrome-plugin',
    mlflow=True
)


@pytest.mark.parametrize("model_name,stage", [
    ("lgbm_model", "staging"),
])
def test_load_latest_staging_model(model_name, stage):

    print("\n================ MODEL LOAD TEST STARTED ================")
    print(f"Model Name : {model_name}")
    print(f"Stage      : {stage}")

    try:
        client = MlflowClient()

        print("\n🔍 Fetching latest model version from MLflow...")
        latest_version_info = client.get_latest_versions(
            model_name,
            stages=[stage]
        )

        print(f"Raw Version Info: {latest_version_info}")

        if not latest_version_info:
            pytest.fail(
                f"❌ No model found in stage '{stage}' for '{model_name}'. "
                f"Available versions may not be assigned to this stage."
            )

        latest_version = latest_version_info[0].version
        print(f"✅ Latest Version in '{stage}' stage: {latest_version}")

        # -----------------------------------------
        # Load Model
        # -----------------------------------------
        model_uri = f"models:/{model_name}/{latest_version}"
        print(f"\n📦 Loading model from URI: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)

        assert model is not None, "Model object is None"

        print(f"✅ Model '{model_name}' version {latest_version} loaded successfully!")
        print("================ MODEL LOAD TEST PASSED ================\n")

    except Exception as e:
        print("\n❌ ERROR DURING MODEL LOAD TEST")
        print("Error Type:", type(e).__name__)
        print("Error Message:", str(e))
        pytest.fail(f"Model loading failed with error: {e}")
