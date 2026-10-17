"""CS611 Assignment 2 monthly backfill DAG.

The logical date is a simulated production date. A run for February 2025 has
no new application cohort, but still refreshes monitoring when older labels
become observable.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_pipeline import prepare_datamart
from utils.model_pipeline import monitor_as_of, score_cohort, train_and_select
from utils.reporting import render_monitoring


with DAG(
    dag_id="cs611_assignment2_loan_default",
    description="Application-time loan model training, scoring and delayed-outcome monitoring",
    start_date=datetime(2024, 6, 1),
    end_date=datetime(2025, 8, 1),
    schedule_interval="0 0 1 * *",
    catchup=True,
    is_paused_upon_creation=False,
    max_active_runs=1,
    default_args={"owner": "cs611", "retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["cs611", "assignment2", "loan-risk"],
) as dag:
    prepare = PythonOperator(task_id="prepare_assignment1_datamart", python_callable=prepare_datamart)
    train = PythonOperator(task_id="train_compare_select_model", python_callable=train_and_select,
                           op_kwargs={"as_of": "{{ ds }}"})
    score = PythonOperator(task_id="score_application_cohort", python_callable=score_cohort,
                           op_kwargs={"as_of": "{{ ds }}"})
    monitor = PythonOperator(task_id="write_monitoring_gold", python_callable=monitor_as_of,
                             op_kwargs={"as_of": "{{ ds }}"})
    chart = PythonOperator(task_id="render_monitoring_chart", python_callable=render_monitoring,
                           op_kwargs={"as_of": "{{ ds }}"})

    prepare >> train >> score >> monitor >> chart
