import mlflow
import random

mlflow.set_tracking_uri("")

with mlflow.start_run():
    mlflow.log_param("param1",random.randint(1,100))
    mlflow.log_param("param2",random.random())

    mlflow.log_metric("metric1",random.random())
    mlflow.log_metric("metric2",random.uniform(0.5,1.5))

    print("Logged random parameters and metrics.")