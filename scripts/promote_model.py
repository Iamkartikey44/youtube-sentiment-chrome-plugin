import os
import mlflow


def promote_model():

    mlflow.set_tracking_uri()

    client = mlflow.MlflowClient()

    model_name = 'yt_chrome_plugin_model'
    latest_version_staging = client.get_latest_versions(model_name,stages=["Staging"])[0].version

    prod_version = client.get_latest_versions(model_name,stages=['Production'])

    #Archive the current production model
    for version in prod_version:
        client.transition_model_version_stage(name=model_name,version=version.version,stage="Archived")
    
    #Promote the new model to production
    client.transition_model_version_stage(name=model_name,version=latest_version_staging,stage='Production')

    print(f"Model version {latest_version_staging} promoted to Production")

if __name__ == '__main__':
    promote_model()