ARG AIRFLOW_BASE_IMAGE="apache/airflow:2.2.5-python3.8"
FROM ${AIRFLOW_BASE_IMAGE}

RUN pip install --user --no-cache-dir \
    minio==5.0.10 \
    apache-airflow-providers-amazon~=1.0.0 \
    pyspark==3.2.1 \
    apache-airflow-providers-apache-spark==2.1.3 \