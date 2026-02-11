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

vectorizer_path = os.path.join(root_dir, 'tfidf_vectorizer.pkl')

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# ----------------------------
# Start MLflow Run
# ----------------------------
with mlflow.start_run() as run:

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Run ID:", run.info.run_id)
    print("Model type:", type(model))

    # ----------------------------
    # Log Vectorizer FIRST (safe)
    # ----------------------------
    mlflow.log_artifact(
        vectorizer_path,
        artifact_path="preprocessing"   # 👈 Put inside folder
    )

    # ----------------------------
    # Prepare test data
    # ----------------------------
    test_data = pd.read_csv(
        os.path.join(root_dir, 'data/interim/test_processed.csv')
    )
    test_data.fillna('', inplace=True)

    X_test_tfidf = vectorizer.transform(test_data['clean_comment'])

    input_example = pd.DataFrame(
        X_test_tfidf.toarray()[:5],
        columns=vectorizer.get_feature_names_out()
    )

    signature = infer_signature(
        input_example,
        model.predict(X_test_tfidf[:5])
    )

    # ----------------------------
    # Log Model LAST
    # ----------------------------
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",   # 🔥 use name in MLflow 3.x
        signature=signature,
        input_example=input_example
    )

    print("Model logged successfully!")

    # ----------------------------
    # Register Model
    # ----------------------------
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_name = "model"

    print("Registering model from:", model_uri)

    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(model_name,stages=["None"])
    #version = model_version[0].version

    client.transition_model_version_stage(
        name=model_name,
        version=model_version[0].version,
        stage="Staging"
    )