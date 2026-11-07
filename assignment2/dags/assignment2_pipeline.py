"""CS611 Assignment 2 monthly backfill DAG.

The logical date is a simulated production date. A run for February 2025 has
no new application cohort, but still refreshes monitoring when older labels
become observable.

Graph (one task = one job, so a failure points straight at its stage):

  check_source_files
    -> bronze_<source> -> silver_<source>            (4 sources)
    -> gold_label_store / gold_feature_store
    -> validate_gold_stores                          (data pipeline done)
    -> train_tune_select_model -> review_and_promote_model
    -> score_monthly_model -> monitor_monthly_model  (in parallel with)
       score_fixed_model   -> monitor_fixed_model
    -> render_monitoring_report
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

from utils.config import SOURCE_FILES
from utils.data_pipeline import (build_bronze, build_feature_store, build_label_store, build_silver,
                                 check_source_files, validate_gold_stores)
from utils.governance import simulate_review_and_promote
from utils.model_pipeline import (monitor_fixed_model, monitor_monthly_model, score_cohort,
                                  score_fixed_model, train_and_select)
from utils.reporting import render_monitoring

AS_OF = {"as_of": "{{ ds }}"}

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
    # ---- Data pipeline: raw CSV -> Bronze -> Silver -> Gold feature and label stores
    check_sources = PythonOperator(task_id="check_source_files", python_callable=check_source_files)

    gold_labels = PythonOperator(task_id="gold_label_store", python_callable=build_label_store)
    gold_features = PythonOperator(task_id="gold_feature_store", python_callable=build_feature_store)
    validate_gold = PythonOperator(task_id="validate_gold_stores", python_callable=validate_gold_stores)

    for source in SOURCE_FILES:
        bronze = PythonOperator(task_id=f"bronze_{source}", python_callable=build_bronze,
                                op_kwargs={"source": source})
        silver = PythonOperator(task_id=f"silver_{source}", python_callable=build_silver,
                                op_kwargs={"source": source})
        check_sources >> bronze >> silver >> gold_features
        if source == "lms_loan_daily":
            silver >> gold_labels

    [gold_labels, gold_features] >> validate_gold

    # ---- Model pipeline: train and tune -> approval gates -> score -> monitor -> report
    train = PythonOperator(task_id="train_tune_select_model", python_callable=train_and_select,
                           op_kwargs=AS_OF)
    review = PythonOperator(task_id="review_and_promote_model", python_callable=simulate_review_and_promote,
                            op_kwargs=AS_OF)
    score_monthly = PythonOperator(task_id="score_monthly_model", python_callable=score_cohort,
                                   op_kwargs=AS_OF)
    score_fixed = PythonOperator(task_id="score_fixed_model", python_callable=score_fixed_model,
                                 op_kwargs=AS_OF)
    monitor_monthly = PythonOperator(task_id="monitor_monthly_model", python_callable=monitor_monthly_model,
                                     op_kwargs=AS_OF)
    monitor_fixed = PythonOperator(task_id="monitor_fixed_model", python_callable=monitor_fixed_model,
                                   op_kwargs=AS_OF)
    report = PythonOperator(task_id="render_monitoring_report", python_callable=render_monitoring,
                            op_kwargs=AS_OF)

    validate_gold >> train >> review >> [score_monthly, score_fixed]
    score_monthly >> monitor_monthly
    score_fixed >> monitor_fixed
    [monitor_monthly, monitor_fixed] >> report
