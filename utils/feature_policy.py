"""Feature inclusion decisions used for the Gold store and the sanity check.

The Gold table keeps application-time fields so that their source and timing
can be checked.  The sanity model uses only clearly defined numeric fields that
exist across all cohorts.  Categorical and time-limited clickstream fields stay
outside this check because they need separate encoding or coverage decisions.
"""


APPROVED_NUMERIC_FEATURES = [
    "loan_amt",
    "tenure",
    "age",
    "annual_income",
    "monthly_inhand_salary",
    "num_bank_accounts",
    "num_credit_card",
    "num_of_loan",
    "num_credit_inquiries",
    "outstanding_debt",
    "credit_utilization_ratio",
    "credit_history_months",
    "total_emi_per_month",
    "monthly_balance",
    "debt_to_annual_income",
    "emi_to_monthly_income",
]

APPROVED_CATEGORICAL_FEATURES = [
    "occupation",
    "credit_mix",
]

CLICKSTREAM_FEATURES = [f"fe_{index}" for index in range(1, 21)]

# These columns remain in Gold for traceability or later investigation, but the
# reasons below keep them out of the current sanity model.
EXCLUDED_FROM_SANITY_MODEL = {
    "loan_id": "join key, not a predictive feature",
    "customer_id": "entity identifier, not a predictive feature",
    "feature_as_of_date": "cohort field used for temporal splitting",
    "attribute_snapshot_date": "audit metadata",
    "financial_snapshot_date": "audit metadata",
    "clickstream_snapshot_date": "audit metadata",
    "clickstream_available": "perfectly associated with later source coverage cohorts",
    "age_invalid_flag": "data-quality metadata rather than customer behaviour",
    "occupation_unknown_flag": "data-quality metadata",
    "financial_parse_error_flag": "data-quality metadata",
    "financial_domain_warning_flag": "data-quality metadata",
    "income_consistency_warning_flag": "data-quality metadata",
    "interest_rate": "business timing is unclear from the supplied data",
    "delay_from_due_date": "historical meaning is not documented",
    "num_of_delayed_payment": "historical meaning is not documented",
    "changed_credit_limit": "business timing is not documented",
    "type_of_loan": "free-text multi-value field requiring a separate encoding design",
    "payment_behaviour": "business timing is not documented",
    "payment_of_min_amount": "business timing is not documented",
    "amount_invested_monthly": "contains unresolved source placeholders",
    "occupation": "categorical encoding belongs in the model pipeline",
    "credit_mix": "categorical encoding belongs in the model pipeline",
    **{f"fe_{index}": "not available for the 3,526 latest applications" for index in range(1, 21)},
}
