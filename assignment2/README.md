# CS611 Assignment 2: loan default ML pipeline

This project continues Su Jiehong's Assignment 1 application-time feature and
MOB 6 label design. The supplied four CSV files are the same inputs as in A1.
The data preparation code carries forward the same population, joins, 30+ DPD
label, and feature policy in a pandas/Parquet implementation so that one
Airflow container can execute the full pipeline.

The eLearn submission zip includes the four supplied CSV files. The GitHub
source branch omits these class data files; place them in `data/` before
running a checkout from GitHub alone.

## Run

```bash
docker compose build
docker compose up
```

Open <http://localhost:8080> and sign in with `admin` / `admin`. The DAG is
`cs611_assignment2_loan_default`. Its 15 monthly logical dates run from
1 June 2024 through 1 August 2025. The first eight dates have application
cohorts; later dates add monitoring as their month-six labels become available.
The DAG starts its historical catchup when the scheduler loads it. You can also
trigger a single date from the UI.

For a local run without Airflow:

```bash
python main.py backfill --start 2024-06-01 --end 2025-08-01
python -m unittest discover -s tests -v
python experiments/feature_ablation.py
```

## Outputs

- `datamart/bronze` and `datamart/silver`: the four supplied sources and cleaned tables.
- `datamart/gold/feature_store.parquet` and `label_store.parquet`: one row per loan application.
- `model_bank/credit_model_YYYY_MM_DD.joblib`: selected estimator, fitted preprocessing and monitoring reference distributions; companion JSON records the selection evidence.
- `datamart/gold/model_predictions/cohort_date=...`: per-loan probability, model version and date.
- `datamart/gold/model_monitoring/as_of_date=...`: performance and stability by scored cohort.
- `reports/monitoring_as_of_...png` and `.csv`: visual and tabular evidence.

All writes are idempotent for a given date. The data stage skips unchanged
source files by SHA-256. Repeated scoring and monitoring overwrite the same
date partition rather than appending duplicates.

## Model decisions

At each application month, the training job uses only prior application
cohorts whose MOB 6 label has been observed by that date. It validates on the
latest two mature cohorts, compares logistic regression with histogram
gradient boosting by ROC AUC, then refits the winner on all available mature
rows. Median imputation and scaling are fitted inside model pipelines, after
the temporal split. The 16 numeric features come from the approved A1 feature
policy. The anonymous clickstream fields are excluded because the supplied
clickstream ends before 3,526 later applications.

### Feature ideas from the three A1 samples

We tested three sample-inspired additions with the same train/validation
dates and then measured AUC on the next, previously unseen application cohort.
The experiment covers eight June 2024–January 2025 cohorts; the detailed
per-cohort results are in `experiments/feature_ablation_results.csv`.

| Feature set | Mean next-cohort AUC | Cohorts better than A1 baseline |
| --- | ---: | ---: |
| Own A1 16-feature baseline | 0.78880 | — |
| Sample 1: occupation and credit-mix encoding | 0.79009 | 4 / 8 |
| Sample 2: product counts and financial ratios | 0.78585 | 1 / 8 |
| Sample 3: log income and balance/inquiry ratios | 0.78718 | 2 / 8 |
| All sample ideas together | 0.78617 | 4 / 8 |

The small category gain was inconsistent across cohorts (mean AUC +0.00129,
4 / 8 wins). The deployed model therefore retains the reproducible A1 feature
set. Category encoding is fitted inside the experiment's training pipeline,
avoiding use of validation or future-cohort distributions.

Monitoring writes ROC AUC and Brier score only when labels have matured.
Prediction PSI and three feature CSI values can be computed immediately.
PSI/CSI at 0.10 prompts review; 0.25 prompts an alert. An observed AUC more
than 0.05 below that version's temporal validation AUC also prompts review.
These thresholds are documented decisions, not claims that drift occurred.

## Data and deployment limits

`loan_start_date` proxies for the application timestamp. The design assumes
same-day attributes and financials were available before the credit decision.
The target is 30+ DPD at MOB 6, not default at any earlier point. The supplied
feature data ends in January 2025, so later DAG runs produce monitoring only.

The provided Docker Compose uses Airflow's SQLite metadata database and
SequentialExecutor for a single-machine coursework run. A production deployment
would use a supported server database, an approval gate for champion promotion,
and a previous-model rollback path.
