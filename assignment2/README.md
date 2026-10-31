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
The DAG starts its historical catchup when the scheduler loads it. Its six tasks
prepare data, train/register a candidate, simulate review/promotion, score both
strategies, monitor both strategies, and render evidence. A clean full catchup
is required before running an isolated later date, because monitoring consumes
earlier prediction cohorts.

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
- `datamart/gold/model_monitoring/as_of_date=...`: monthly-version performance, PSI, all 16 CSI values, all 16 missing rates, maximum CSI and review flags.
- `datamart/gold/model_feature_monitoring/as_of_date=...`: one row per cohort and input, including CSI, reference/current missing rates and their difference.
- `datamart/gold/fixed_model_predictions`, `fixed_model_monitoring` and `fixed_model_feature_monitoring`: the separate frozen-June comparator.
- `model_bank/registry/historical.json`: explicitly simulated approvals and deployment history.
- `model_bank/registry/production.json`: separate manually recorded approvals and deployment history; historical decisions cannot authorize this scope.
- `reports/monitoring_as_of_...png`, `feature_monitoring_as_of_...png` and CSVs: strategy comparisons, complete input monitoring and auditable tabular evidence.

Prediction/monitoring writes and repeated governance decisions are idempotent for a given date. The data stage skips unchanged
source files and unchanged preparation code by SHA-256. Each saved model also
records fingerprints of its Gold inputs, feature/training code and library
versions. A mismatch blocks reuse until `--force-train` rebuilds the models.
Repeated scoring and monitoring overwrite the same date partition rather than
appending duplicates. A forced full replay archives and resets simulated
governance decisions. It refuses to overwrite an artifact registered in the
production scope; production changes require a new version.

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
Prediction PSI and all 16 model-input CSI values can be computed immediately.
References and current inputs use the same domain masking before imputation;
negative counts and nonfinite values are missing, with an explicit missing bin.
Constant and entirely missing references are supported. The monitoring Gold
tables retain every CSI and missing rate, maximum CSI and its feature name,
and the greatest missing-rate increase. An increase of at least 5 percentage
points versus that version's training reference prompts a source-quality review.
It also stores the mean and lowest nonmissing coverage across the 16 model
features before imputation, so a deteriorating source can be investigated.
PSI/CSI at 0.10 prompts review; 0.25 prompts an alert. An observed AUC more
than 0.05 below that version's temporal validation AUC also prompts review.
These thresholds are documented project decisions, not claims that drift occurred.
The candidate validation floor of ROC AUC 0.65 is also a coursework policy,
not an independently established lending acceptance threshold.

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
would use a supported server database and authenticated, role-based approval
controls. The local approval, promotion and rollback commands below demonstrate
the version-selection behavior without providing those security controls.

## Fixed model comparison

The comparator is fixed in advance in code as `credit_model_2024_06_01`, the
first model in the replay. It is never refreshed or selected from later cohort
results. The monthly strategy uses the approved version effective at each
cohort date. Both score the same 4,024 applications, write to separate Gold
partitions, and wait for the same MOB 6 observation dates before calculating AUC
and Brier score. Each strategy uses its own model's training reference for PSI
and CSI. The fixed comparator consequently has a single fixed reference across
all eight application cohorts.

`reports/strategy_comparison_as_of_2025-08-01.csv` reports the paired results.
`experiments/fixed_model_comparison.csv` is a checked-in snapshot of that clean
container replay; rerunning the DAG regenerates the reports version.
The cohort-size-weighted AUC summaries are descriptive means of cohort AUCs,
not pooled AUCs or evidence of a statistically significant strategy advantage.
This additional retrospective comparison does not restore a blind holdout:
January outcomes had already been seen, as disclosed above. No comparator
result changes feature selection, candidate approval or the deployed strategy.

## Approval, promotion and rollback

Historical replay and production are separate registry scopes. The historical
DAG records `decision_mode=policy_simulation` with the reviewer
`historical_policy_simulation`. It reproduces an approval-policy decision for
coursework; it does not claim that a human approved a 2024 release. Simulated
approvals never authorize production scoring.

Every trained artifact starts as a candidate. Approval checks the training
fingerprint, mature-label chronology, exact feature contract, all 16 monitoring
references, validation AUC >= 0.65, finite probabilities and current input
availability. PSI or any CSI >= 0.25 blocks approval/promotion. Values >= 0.10
require review and are included in the recorded assessment; the historical
review is explicitly simulated. A fully missing input also blocks promotion.
Promotion rechecks these gates. Ordinary monitoring additionally flags a
5-percentage-point missingness increase and an AUC drop > 0.05 from validation.
Monitoring flags request investigation; they do not silently authorize a release.

Standalone `score` defaults to the production scope. It fails until an approved
model has been explicitly promoted, and writes to `production_predictions`
without overwriting historical results. Both approval and deployment commands
require a nonempty reviewer and reason. For example, after the historical replay:

```bash
docker compose exec airflow-scheduler python main.py approve --model-version credit_model_2024_06_01 --as-of 2024-06-01 --reviewer course_reviewer --reason "Reviewed validation and input monitoring"
docker compose exec airflow-scheduler python main.py promote --model-version credit_model_2024_06_01 --as-of 2024-06-01 --reviewer course_operator --reason "Deploy approved release"
docker compose exec airflow-scheduler python main.py score --as-of 2024-06-01 --scope production
```

To demonstrate a later deployment and an explicit rollback:

```bash
docker compose exec airflow-scheduler python main.py approve --model-version credit_model_2024_07_01 --as-of 2024-07-01 --reviewer course_reviewer --reason "Reviewed July candidate"
docker compose exec airflow-scheduler python main.py promote --model-version credit_model_2024_07_01 --as-of 2024-07-01 --reviewer course_operator --reason "Deploy approved July release"
docker compose exec airflow-scheduler python main.py rollback --model-version credit_model_2024_06_01 --as-of 2024-08-01 --reviewer course_operator --reason "Restore previously deployed approved version"
docker compose exec airflow-scheduler python main.py registry-status --scope production
```

The inference resolver reads the deployment effective at the scoring date, so
the rollback changes subsequent version selection while preserving July's
history. Rollback requires a different, previously deployed approved artifact;
it does not erase existing predictions. Explicitly rerun affected scoring dates
if predictions should be regenerated. The audit trail records scope, artifact
hash, reviewer, reason, effective date and actual UTC recording time. Replacing
a registered artifact or inserting a new deployment before the latest effective
date is rejected. Registry writes use a file lock and atomic replacement.

The reviewer is entered manually; this local coursework registry does not
provide authentication, role-based authorization, cryptographic signatures or
a production-grade model-serving system. Batch scoring, the 0.5 demonstration
cutoff and same-day feature availability remain the operating limits described
above. The manual commands demonstrate controlled release and rollback; they
do not automate business approval.
