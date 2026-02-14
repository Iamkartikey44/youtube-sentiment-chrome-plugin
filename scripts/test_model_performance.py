import mlflow
import pytest
import pickle
import pandas as pd
import os
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri("https://dagshub.com/Iamkartikey44/youtube-sentiment-chrome-plugin.mlflow")


@pytest.mark.parametrize("model_name,stage,holdout_data_path, vectorizer_path", [
    ("lgbm_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl"),
])
def test_model_performance(model_name, stage, holdout_data_path, vectorizer_path):

    print("\n================ MODEL PERFORMANCE TEST STARTED ================")

    try:
        # -------------------------------
        # Step 1: Get Latest Model Version
        # -------------------------------
        client = mlflow.tracking.MlflowClient()
        latest_version_info = client.get_latest_versions(model_name, stages=[stage])
        latest_version = latest_version_info[0].version if latest_version_info else None

        assert latest_version is not None, f"No model found in stage '{stage}'"

        # -------------------------------
        # Step 2: Load Model
        # -------------------------------
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # -------------------------------
        # Step 3: Load Vectorizer
        # -------------------------------
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)

        # -------------------------------
        # Step 4: Load Holdout Data
        # -------------------------------
        holdout_data = pd.read_csv(holdout_data_path)
        print(f"Original Dataset Shape: {holdout_data.shape}")

        text_column = "clean_comment"
        label_column = "category"

        # Drop NaNs in text column
        holdout_data = holdout_data.dropna(subset=[text_column])

        # Remove empty strings
        holdout_data = holdout_data[
            holdout_data[text_column].astype(str).str.strip() != ""
        ]

        print(f"Dataset Shape After Cleaning: {holdout_data.shape}")

        # Extract cleaned text + labels
        X_holdout_raw = holdout_data[text_column].astype(str)
        y_holdout = holdout_data[label_column]

        # Final safety check
        assert X_holdout_raw.isna().sum() == 0, "Still contains NaN values!"

        # -------------------------------
        # Step 5: Transform Text
        # -------------------------------
        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)

        X_holdout_tfidf_df = pd.DataFrame(
            X_holdout_tfidf.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        # -------------------------------
        # Step 6: Predict
        # -------------------------------
        y_pred_new = model.predict(X_holdout_tfidf_df)

        # -------------------------------
        # Step 7: Calculate Metrics
        # -------------------------------
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        recall_new = recall_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        f1_new = f1_score(y_holdout, y_pred_new, average='weighted', zero_division=1)

        print(f"Accuracy  : {accuracy_new:.4f}")
        print(f"Precision : {precision_new:.4f}")
        print(f"Recall    : {recall_new:.4f}")
        print(f"F1 Score  : {f1_new:.4f}")

        # -------------------------------
        # Step 8: Threshold Check
        # -------------------------------
        assert accuracy_new >= 0.40
        assert precision_new >= 0.40
        assert recall_new >= 0.40
        assert f1_new >= 0.40

        print("\n✅ MODEL PERFORMANCE TEST PASSED")

    except Exception as e:
        print("\n❌ ERROR OCCURRED DURING MODEL TEST")
        print(str(e))
        pytest.fail(f"Model performance test failed: {e}")
