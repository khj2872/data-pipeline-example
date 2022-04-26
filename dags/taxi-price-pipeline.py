from datetime import datetime

from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

spark_master = "spark://spark-master:7077"
application_default_path = "/usr/local/spark/app"

default_args = {
    'start_date': datetime(2022, 1, 1),
}

with DAG(dag_id='taxi-price-pipeline',
         schedule_interval='@monthly',
         default_args=default_args,
         catchup=False) as dag:

    preprocess = SparkSubmitOperator(
        application=f"{application_default_path}/preprocess.py",
        task_id="preprocess",
        conn_id="spark_local",
        verbose=True,
        # kwargs={"deploy_mode": "cluster"}
    )

    train_model = SparkSubmitOperator(
        application=f"{application_default_path}/train_model.py",
        task_id="train_model",
        conn_id="spark_local"
    )

    preprocess >> train_model
