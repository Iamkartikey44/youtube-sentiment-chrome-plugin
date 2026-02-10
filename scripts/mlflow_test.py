import mlflow
import random
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/Iamkartikey44/youtube-sentiment-chrome-plugin.mlflow")
dagshub.init(repo_owner='Iamkartikey44', repo_name='youtube-sentiment-chrome-plugin', mlflow=True)

with mlflow.start_run():
    mlflow.log_param("param1",random.randint(1,100))
    mlflow.log_param("param2",random.random())

    mlflow.log_metric("metric1",random.random())
    mlflow.log_metric("metric2",random.uniform(0.5,1.5))

    print("Logged random parameters and metrics.")