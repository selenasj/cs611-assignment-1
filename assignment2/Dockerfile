FROM apache/airflow:2.10.5-python3.11

COPY requirements.txt /requirements.txt
USER airflow
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /requirements.txt

WORKDIR /opt/airflow/project
COPY --chown=airflow:0 . /opt/airflow/project
ENV PYTHONPATH=/opt/airflow/project
