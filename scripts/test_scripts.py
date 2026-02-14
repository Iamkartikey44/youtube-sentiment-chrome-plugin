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
dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

model_name="lgbm_model"
stage ="staging"
vectorizer_path ="tfidf_vectorizer.pkl"
holdout_data_path="data/interim/test_processed.csv"

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