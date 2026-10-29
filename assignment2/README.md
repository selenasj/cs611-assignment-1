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

If data or training code changes after model artifacts have been saved, the
pipeline stops rather than silently reusing them. Refresh every affected model
and its downstream predictions and monitoring with:

```bash
python main.py backfill --start 2024-06-01 --end 2025-08-01 --force-train
```

## Outputs

- `datamart/bronze` and `datamart/silver`: the four supplied sources and cleaned tables.
- `datamart/gold/feature_store.parquet` and `label_store.parquet`: one row per loan application.
- `model_bank/credit_model_YYYY_MM_DD.joblib`: selected estimator, fitted preprocessing and monitoring reference distributions; companion JSON records the selection evidence.
- `datamart/gold/model_predictions/cohort_date=...`: per-loan probability, model version and date.
- `datamart/gold/model_monitoring/as_of_date=...`: performance, stability, and model-feature coverage by scored cohort.
- `reports/monitoring_as_of_...png` and `.csv`: visual and tabular evidence.

All writes are idempotent for a given date. The data stage skips unchanged
source files and unchanged preparation code by SHA-256. Each saved model also
records fingerprints of its Gold inputs, feature/training code and library
versions. A mismatch blocks reuse until `--force-train` rebuilds the models.
Repeated scoring and monitoring overwrite the same date partition rather than
appending duplicates.

## Model decisions

At each application month, the training job uses only prior application
cohorts whose MOB 6 label has been observed by that date. It validates on the
latest two mature cohorts, compares logistic regression with histogram
gradient boosting by ROC AUC, then refits the winner on all available mature
rows. Median imputation and scaling are fitted inside model pipelines, after
the temporal split. The 16 numeric features come from the approved A1 feature
policy. The anonymous clickstream fields are excluded because the supplied
clickstream ends before 3,526 later applications.

Silver and Gold retain source values for audit. At the model input boundary,
impossible negative counts (including 500 `num_of_loan=-100` and four
`num_bank_accounts=-1` values) become missing values. The fitted median
imputer then handles them. The feature experiment uses the same rule before
constructing its count-based ratios.

### Feature ideas from the three A1 samples

We compared three sample-inspired additions retrospectively, using the same
train/validation dates and then measuring AUC on the next application cohort.
Feature development uses seven June–December 2024 cohorts only. Before scoring
January 2025 in the revised workflow, the feature policy requires at least
+0.005 mean AUC and wins in at least 60% of development cohorts to justify
additional features. This is a practical complexity hurdle, not a statistical
significance claim. The per-cohort development results are in
`experiments/feature_ablation_results.csv`, and the locked choice is recorded
in `experiments/feature_selection_summary.json`.

| Feature set | Mean development AUC | Development cohorts better than A1 baseline |
| --- | ---: | ---: |
| Own A1 16-feature baseline | 0.79095 | — |
| Sample 1: occupation and credit-mix encoding | 0.79297 | 5 / 7 |
| Sample 2: product counts and financial ratios | 0.78958 | 3 / 7 |
| Sample 3: log income and balance/inquiry ratios | 0.79107 | 4 / 7 |
| All sample ideas together | 0.79072 | 4 / 7 |

The strongest category idea gained only +0.00202 mean AUC, below the +0.005
adoption hurdle, so the deployed model retains the A1 16-feature set. The
January 2025 cohort is excluded from this comparison. After locking the
baseline, the revised experiment evaluates that cohort once: 526 loans,
ROC AUC 0.76655. Its result is stored separately in
`experiments/feature_holdout_confirmation.csv`. Category encoding is fitted
inside the experiment's training pipeline, avoiding use of validation or
future-cohort distributions. The experiment allows up to 5,000 logistic
regression iterations and treats a convergence warning as an error; the clean
rerun completed without one and did not change the reported AUC values.

This is a **procedurally separated, retrospective OOT confirmation**, not a
historically blind test: an earlier eight-cohort exploratory run had already
shown January outcomes. The final monitoring table's eight-cohort average is
descriptive and must not be presented as independent feature-selection evidence.

Monitoring writes ROC AUC and Brier score only when labels have matured.
Prediction PSI and three feature CSI values can be computed immediately.
It also stores the mean and lowest nonmissing coverage across the 16 model
features before imputation, so a deteriorating source can be investigated.
PSI/CSI at 0.10 prompts review; 0.25 prompts an alert. An observed AUC more
than 0.05 below that version's temporal validation AUC also prompts review.
These thresholds are documented decisions, not claims that drift occurred.

## Data and deployment limits

`loan_start_date` proxies for the application timestamp. The design assumes
same-day attributes and financials were available before the credit decision.
Clickstream snapshots must be dated on or before the application date; all
8,974 matched snapshots happen to be same-day. Their intraday availability is
unverified. Clickstream is not used by the current 16-feature model.
The target is 30+ DPD at MOB 6, not default at any earlier point. The supplied
feature data ends in January 2025, so later DAG runs produce monitoring only.
The `high_risk_flag` cutoff of 0.5 is illustrative; a production threshold
would be set from lending costs, approval capacity, and recall targets.

The provided Docker Compose uses Airflow's SQLite metadata database and
SequentialExecutor for a single-machine coursework run. A production deployment
would use a supported server database, an approval gate for champion promotion,
and a previous-model rollback path.
