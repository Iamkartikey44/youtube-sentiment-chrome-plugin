import mlflow
import pytest
import pickle
import pandas as pd
import os
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token


mlflow.set_tracking_uri("https://dagshub.com/Iamkartikey44/youtube-sentiment-chrome-plugin.mlflow")


@pytest.mark.parametrize("model_name,stage,holdout_data_path, vectorizer_path", [
    ("lgbm_model", "staging", "data/interim/test_processed.csv", "tfidf_vectorizer.pkl"),
])
def test_model_performance(model_name, stage, holdout_data_path, vectorizer_path):

    print("\n================ MODEL PERFORMANCE TEST STARTED ================")
    print(f"Model Name: {model_name}")
    print(f"Stage: {stage}")
    print(f"Holdout Data Path: {holdout_data_path}")
    print(f"Vectorizer Path: {vectorizer_path}")

    try:
        # -------------------------------
        # Step 1: Get Latest Model Version
        # -------------------------------
        print("\n🔍 Fetching latest model version from MLflow...")
        client = mlflow.tracking.MlflowClient()
        latest_version_info = client.get_latest_versions(model_name, stages=[stage])
        latest_version = latest_version_info[0].version if latest_version_info else None

        print(f"Latest version info: {latest_version_info}")

        assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"
        print(f"✅ Found Model Version: {latest_version}")

        # -------------------------------
        # Step 2: Load Model
        # -------------------------------
        model_uri = f"models:/{model_name}/{latest_version}"
        print(f"\n📦 Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Model loaded successfully")

        # -------------------------------
        # Step 3: Load Vectorizer
        # -------------------------------
        print("\n📦 Loading TF-IDF vectorizer...")
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        print("✅ Vectorizer loaded successfully")

        # -------------------------------
        # Step 4: Load Holdout Data
        # -------------------------------
        print("\n📊 Loading holdout dataset...")
        holdout_data = pd.read_csv(holdout_data_path)
        print(f"Dataset Shape: {holdout_data.shape}")
        print("First 5 rows:\n", holdout_data.head())

        X_holdout_raw = holdout_data.iloc[:, :-1].squeeze()
        y_holdout = holdout_data.iloc[:, -1]

        print(f"Text Samples Count: {len(X_holdout_raw)}")
        print(f"Label Distribution:\n{y_holdout.value_counts()}")
        # Identify text column (first column)
        text_column = holdout_data.columns[0]

        # Check for NaN values
        nan_count = holdout_data[text_column].isna().sum()

        print(f"NaN values found in '{text_column}': {nan_count}")

        # Drop NaN rows only if present
        if nan_count > 0:
            print("⚠ Dropping rows with NaN text values...")
            holdout_data = holdout_data.dropna(subset=[text_column])
            print(f"Dataset Shape (After Dropping NaNs): {holdout_data.shape}")
        else:
            print("✅ No NaN values found in text column.")

        # -------------------------------
        # Step 5: Transform Text
        # -------------------------------
        print("\n🔄 Transforming text using TF-IDF...")
        X_holdout_tfidf = vectorizer.transform(X_holdout_raw)
        X_holdout_tfidf_df = pd.DataFrame(
            X_holdout_tfidf.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        print(f"TF-IDF Shape: {X_holdout_tfidf_df.shape}")

        # -------------------------------
        # Step 6: Predict
        # -------------------------------
        print("\n🤖 Making predictions...")
        y_pred_new = model.predict(X_holdout_tfidf_df)

        print("Sample Predictions:", y_pred_new[:10])
        print("Sample True Labels:", y_holdout.values[:10])

        # -------------------------------
        # Step 7: Calculate Metrics
        # -------------------------------
        print("\n📈 Calculating performance metrics...")

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        recall_new = recall_score(y_holdout, y_pred_new, average='weighted', zero_division=1)
        f1_new = f1_score(y_holdout, y_pred_new, average='weighted', zero_division=1)

        print(f"Accuracy  : {accuracy_new:.4f}")
        print(f"Precision : {precision_new:.4f}")
        print(f"Recall    : {recall_new:.4f}")
        print(f"F1 Score  : {f1_new:.4f}")

        # -------------------------------
        # Step 8: Define Thresholds
        # -------------------------------
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        print("\n🎯 Expected Thresholds:")
        print(f"Accuracy >= {expected_accuracy}")
        print(f"Precision >= {expected_precision}")
        print(f"Recall >= {expected_recall}")
        print(f"F1 >= {expected_f1}")

        # -------------------------------
        # Step 9: Assertions
        # -------------------------------
        assert accuracy_new >= expected_accuracy, \
            f'Accuracy should be at least {expected_accuracy}, got {accuracy_new}'

        assert precision_new >= expected_precision, \
            f'Precision should be at least {expected_precision}, got {precision_new}'

        assert recall_new >= expected_recall, \
            f'Recall should be at least {expected_recall}, got {recall_new}'

        assert f1_new >= expected_f1, \
            f'F1 score should be at least {expected_f1}, got {f1_new}'

        print(f"\n✅ Performance test PASSED for model '{model_name}' version {latest_version}")
        print("================ MODEL PERFORMANCE TEST COMPLETED ================\n")

    except Exception as e:
        print("\n❌ ERROR OCCURRED DURING MODEL TEST")
        print(str(e))
        pytest.fail(f"Model performance test failed with error: {e}")
