import pandas as pd
import joblib

# -------------------------------
# Load model + vectorizer locally
# -------------------------------
def load_model_and_vectorizer_local(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


# Load assets
model, vectorizer = load_model_and_vectorizer_local(
    "./lgbm_model.pkl",
    "./tfidf_vectorizer.pkl"
)

# Cache feature names ONCE (important)
FEATURE_NAMES = vectorizer.get_feature_names_out()

# -------------------------------
# Prepare input
# -------------------------------
text = "nice video bro"

X_sparse = vectorizer.transform([text])
print(f"X_spare shape : {X_sparse.shape}")

X_df = pd.DataFrame(
    X_sparse.toarray(),
    columns=FEATURE_NAMES
)

print(f"X_df shape: {X_df.shape}")


# -------------------------------
# Predict
# -------------------------------
prediction = model.predict(X_df)

print(f"Prediction: {prediction}")


###MLFLOW 
#Load the model and vectorizer from the model registry and local storage
# def load_model_and_vectorizer(model_name,model_version,vectorizer_path):
#     #Set MLFlow tracking URI to you server
#     mlflow.set_tracking_uri('')
#     client = MlflowClient()
#     model_uri = f"models:/{model_name}/{model_version}"
#     model = mlflow.pyfunc.load_model(model_uri)
#     vectorizer = joblib.load(vectorizer_path)

#     return model,vectorizer

# model,vectorizer = load_model_and_vectorizer("my_model","1","./tfidf_vectorizer.pkl")