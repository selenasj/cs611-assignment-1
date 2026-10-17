"""Compare sample-inspired features on later application cohorts only."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.feature_policy import APPROVED_NUMERIC_FEATURES

features = pd.read_parquet(PROJECT / 'datamart/gold/feature_store.parquet')
labels = pd.read_parquet(PROJECT / 'datamart/gold/label_store.parquet')
frame = features.merge(labels[['loan_id','label_observation_date','label']], on='loan_id', validate='one_to_one')

def div(num, den):
    n = pd.to_numeric(num, errors='coerce')
    d = pd.to_numeric(den, errors='coerce')
    return (n / d.where(d > 0)).replace([np.inf,-np.inf], np.nan)

# Sample 2: product counts, ratios, and disposable monthly cash.
frame['num_financial_products'] = frame['num_bank_accounts'] + frame['num_credit_card'] + frame['num_of_loan']
frame['loans_per_credit_item'] = div(frame['num_of_loan'], frame['num_bank_accounts'] + frame['num_credit_card'] + 1)
frame['debt_to_monthly_salary'] = div(frame['outstanding_debt'], frame['monthly_inhand_salary'] + 1)
frame['repayment_ability'] = frame['monthly_inhand_salary'] - frame['total_emi_per_month']
# Sample 3: skew treatment, balance/debt, and inquiry intensity.
frame['log_annual_income'] = np.log1p(pd.to_numeric(frame['annual_income'], errors='coerce').clip(lower=0))
frame['balance_to_debt'] = div(frame['monthly_balance'] + 1, frame['outstanding_debt'] + 1)
frame['inquiries_per_loan'] = div(frame['num_credit_inquiries'], frame['num_of_loan'] + 1)
# Sample 1: train-only one-hot encoding of application-time categories.
for cat in ['occupation','credit_mix']:
    frame[cat] = frame[cat].fillna('Unknown').astype(str)

sample2 = ['num_financial_products','loans_per_credit_item','debt_to_monthly_salary','repayment_ability']
sample3 = ['log_annual_income','balance_to_debt','inquiries_per_loan']
sets = {
    'assignment1_baseline': (APPROVED_NUMERIC_FEATURES, []),
    'sample2_financial_ratios': (APPROVED_NUMERIC_FEATURES + sample2, []),
    'sample3_log_ratios': (APPROVED_NUMERIC_FEATURES + sample3, []),
    'sample1_categories': (APPROVED_NUMERIC_FEATURES, ['occupation','credit_mix']),
    'all_sample_ideas': (APPROVED_NUMERIC_FEATURES + sample2 + sample3, ['occupation','credit_mix']),
}

def make_model(algorithm, numeric, categorical):
    numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
    if algorithm == 'logistic_regression':
        numeric_steps.append(('scaler', RobustScaler()))
    transformers = [('numeric', Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append(('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categorical))
    classifier = (LogisticRegression(max_iter=2000, random_state=88) if algorithm == 'logistic_regression'
                  else HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                      learning_rate=0.05, l2_regularization=2.0, random_state=88))
    return Pipeline([('prep', ColumnTransformer(transformers, sparse_threshold=0)), ('classifier', classifier)])

rows = []
for date in sorted(frame['feature_as_of_date'].unique()):
    date = pd.Timestamp(date)
    if not (pd.Timestamp('2024-06-01') <= date <= pd.Timestamp('2025-01-01')):
        continue
    past = frame[(frame['feature_as_of_date'] < date) & (frame['label_observation_date'] <= date)]
    cohort = frame[frame['feature_as_of_date'] == date]
    validation_cohorts = sorted(past['feature_as_of_date'].unique())[-2:]
    train = past[~past['feature_as_of_date'].isin(validation_cohorts)]
    valid = past[past['feature_as_of_date'].isin(validation_cohorts)]
    print(date.strftime('%Y-%m-%d'), 'train', len(train), 'validation', len(valid), 'future', len(cohort), flush=True)
    for name, (numeric, categorical) in sets.items():
        fields = numeric + categorical
        evaluations = []
        for algorithm in ['logistic_regression', 'hist_gradient_boosting']:
            model = make_model(algorithm, numeric, categorical)
            model.fit(train[fields], train['label'])
            score = roc_auc_score(valid['label'], model.predict_proba(valid[fields])[:, 1])
            evaluations.append((score, algorithm))
        validation_auc, selected = max(evaluations)
        model = make_model(selected, numeric, categorical)
        model.fit(past[fields], past['label'])
        future_auc = roc_auc_score(cohort['label'], model.predict_proba(cohort[fields])[:, 1])
        rows.append({'cohort_date': date.strftime('%Y-%m-%d'), 'feature_set': name,
                     'selected_algorithm': selected, 'validation_auc': validation_auc,
                     'future_cohort_auc': future_auc, 'future_rows': len(cohort)})
        print(' ', name, round(validation_auc, 4), round(future_auc, 4), flush=True)

output = Path(__file__).parent / 'feature_ablation_results.csv'
with output.open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
results = pd.DataFrame(rows)
base = results[results['feature_set'] == 'assignment1_baseline'].set_index('cohort_date')['future_cohort_auc']
for name, group in results.groupby('feature_set'):
    delta = group.set_index('cohort_date')['future_cohort_auc'] - base
    print(name, 'mean_future_auc', round(group['future_cohort_auc'].mean(), 5),
          'mean_delta', round(delta.mean(), 5), 'positive_cohorts', int((delta > 0).sum()), flush=True)
print('saved', output)
