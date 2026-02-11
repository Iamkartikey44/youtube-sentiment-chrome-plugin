import mlflow
import pandas as pd
import os
import pickle
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import dagshub
# ----------------------------
# MLflow Setup
# ----------------------------
mlflow.set_tracking_uri(
    "https://dagshub.com/Iamkartikey44/youtube-sentiment-chrome-plugin.mlflow"
)
dagshub.init(
    repo_owner='Iamkartikey44',
    repo_name='youtube-sentiment-chrome-plugin',
    mlflow=True
)

mlflow.set_experiment("dvc-pipeline-runs-v1")

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

with open(os.path.join(root_dir, 'lgbm_model.pkl'), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(root_dir, 'tfidf_vectorizer.pkl'), "rb") as f:
    vectorizer = pickle.load(f)

# ----------------------------
# Start MLflow Run
# ----------------------------
with mlflow.start_run() as run:

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Run ID:", run.info.run_id)
    print("Model type:", type(model))

    # Load test data
    test_data = pd.read_csv(
        os.path.join(root_dir, 'data/interim/test_processed.csv')
    )
    test_data.fillna('', inplace=True)

    X_test_tfidf = vectorizer.transform(test_data['clean_comment'])
    y_test = test_data['category'].values

    input_example = pd.DataFrame(
        X_test_tfidf.toarray()[:5],
        columns=vectorizer.get_feature_names_out()
    )

    signature = infer_signature(
        input_example,
        model.predict(X_test_tfidf[:5])
    )

    # 🔥 Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",  # IMPORTANT for MLflow 3.x
        signature=signature,
        input_example=input_example
    )

    print("Model logged successfully!")

    # ----------------------------
    # Register Model Automatically
    # ----------------------------
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_name = "yt_chrome_plugin_model"

    print("Registering model from:", model_uri)

    model_version = mlflow.register_model(model_uri, model_name)

    print(f"Model version {model_version.version} registered!")

    # ----------------------------
    # Assign Alias (Modern Way)
    # ----------------------------
    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias="staging",
        version=model_version.version
    )

    print("Alias 'staging' assigned successfully!")
