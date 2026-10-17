"""Assignment 1 data product, adapted to run without a Spark service.

The source grain, application-time joins, 30+ DPD at MOB 6 label, and
feature-eligibility policy are carried forward from the submitted A1 work.
The four supplied CSVs are immutable inputs. Every write is deterministic.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import BRONZE_DIR, DATA_DIR, DATAMART_DIR, GOLD_DIR, SILVER_DIR, SOURCE_FILES


NUMERIC_FINANCIAL_COLUMNS = {
    "Annual_Income": "annual_income",
    "Monthly_Inhand_Salary": "monthly_inhand_salary",
    "Num_Bank_Accounts": "num_bank_accounts",
    "Num_Credit_Card": "num_credit_card",
    "Interest_Rate": "interest_rate",
    "Num_of_Loan": "num_of_loan",
    "Delay_from_due_date": "delay_from_due_date",
    "Num_of_Delayed_Payment": "num_of_delayed_payment",
    "Changed_Credit_Limit": "changed_credit_limit",
    "Num_Credit_Inquiries": "num_credit_inquiries",
    "Outstanding_Debt": "outstanding_debt",
    "Credit_Utilization_Ratio": "credit_utilization_ratio",
    "Total_EMI_per_month": "total_emi_per_month",
    "Amount_invested_monthly": "amount_invested_monthly",
    "Monthly_Balance": "monthly_balance",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    df.to_parquet(temporary, index=False)
    temporary.replace(path)


def _date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise ValueError("A required source date could not be parsed")
    return parsed


def _numeric(series: pd.Series) -> pd.Series:
    # A1 source values occasionally have a trailing underscore.
    return pd.to_numeric(series.astype("string").str.strip().str.replace(r"_$", "", regex=True), errors="coerce")


def _unknown(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned.isna() | cleaned.eq("") | cleaned.str.fullmatch(r"_+", na=False), "Unknown")


def _clean_attributes(raw: pd.DataFrame) -> pd.DataFrame:
    age = _numeric(raw["Age"])
    occupation = _unknown(raw["Occupation"])
    return pd.DataFrame({
        "customer_id": raw["Customer_ID"].astype("string"),
        "snapshot_date": _date(raw["snapshot_date"]),
        "age": age.where(age.between(18, 100)),
        "age_invalid_flag": (~age.between(18, 100)).astype("int8"),
        "occupation": occupation,
        "occupation_unknown_flag": occupation.eq("Unknown").astype("int8"),
    })


def _clean_financials(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "customer_id": raw["Customer_ID"].astype("string"),
        "snapshot_date": _date(raw["snapshot_date"]),
    })
    parse_flags = []
    for original, target in NUMERIC_FINANCIAL_COLUMNS.items():
        value = _numeric(raw[original])
        out[target] = value
        nonempty = raw[original].astype("string").str.strip().notna() & raw[original].astype("string").str.strip().ne("")
        parse_flags.append((nonempty & value.isna()).fillna(False))

    years = pd.to_numeric(raw["Credit_History_Age"].astype("string").str.extract(r"(\d+)\s+Years?", expand=False), errors="coerce")
    months = pd.to_numeric(raw["Credit_History_Age"].astype("string").str.extract(r"(\d+)\s+Months?", expand=False), errors="coerce")
    out["credit_history_months"] = years * 12 + months
    out["credit_mix"] = _unknown(raw["Credit_Mix"])
    out["type_of_loan"] = raw["Type_of_Loan"].astype("string")
    out["payment_behaviour"] = _unknown(raw["Payment_Behaviour"])
    out["payment_of_min_amount"] = _unknown(raw["Payment_of_Min_Amount"])
    out["financial_parse_error_flag"] = pd.concat(parse_flags, axis=1).any(axis=1).astype("int8")

    nonnegative = ["annual_income", "monthly_inhand_salary", "num_bank_accounts", "num_credit_card",
                   "interest_rate", "num_of_loan", "num_of_delayed_payment", "num_credit_inquiries",
                   "outstanding_debt", "credit_utilization_ratio", "total_emi_per_month",
                   "amount_invested_monthly", "monthly_balance"]
    out["financial_domain_warning_flag"] = out[nonnegative].lt(0).any(axis=1).astype("int8")
    ratio = out["annual_income"] / (out["monthly_inhand_salary"] * 12)
    out["income_consistency_warning_flag"] = (
        out["annual_income"].gt(0) & out["monthly_inhand_salary"].gt(0)
        & (ratio.lt(0.25) | ratio.gt(4.0))
    ).astype("int8")
    return out


def _clean_clickstream(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "customer_id": raw["Customer_ID"].astype("string"),
        "snapshot_date": _date(raw["snapshot_date"]),
    })
    for index in range(1, 21):
        out[f"fe_{index}"] = _numeric(raw[f"fe_{index}"])
    return out


def _clean_loans(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "loan_id": raw["loan_id"].astype("string"),
        "customer_id": raw["Customer_ID"].astype("string"),
        "loan_start_date": _date(raw["loan_start_date"]),
        "snapshot_date": _date(raw["snapshot_date"]),
    })
    for column in ["tenure", "installment_num", "loan_amt", "due_amt", "paid_amt", "overdue_amt", "balance"]:
        out[column] = _numeric(raw[column])
    out["mob"] = out["installment_num"].astype("int16")
    missed = np.where(out["overdue_amt"].gt(0) & out["due_amt"].gt(0),
                      np.ceil(out["overdue_amt"] / out["due_amt"]), 0).astype("int16")
    out["installments_missed"] = missed
    if not out["snapshot_date"].dt.day.eq(1).all():
        raise ValueError("The month-based DPD calculation requires first-day snapshots")
    month_number = out["snapshot_date"].dt.year * 12 + out["snapshot_date"].dt.month - 1 - missed
    first_missed = pd.to_datetime({
        "year": month_number // 12,
        "month": month_number % 12 + 1,
        "day": 1,
    })
    out["dpd"] = np.where(out["overdue_amt"].gt(0),
                          (out["snapshot_date"] - first_missed).dt.days, 0).astype("int16")
    return out


def _build_gold(attributes: pd.DataFrame, financials: pd.DataFrame,
                clickstream: pd.DataFrame, loans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    applications = loans.loc[loans["mob"].eq(0),
                             ["loan_id", "customer_id", "loan_start_date", "loan_amt", "tenure"]].copy()
    applications = applications.rename(columns={"loan_start_date": "feature_as_of_date"})
    attr = attributes.rename(columns={"snapshot_date": "attribute_snapshot_date"})
    fin = financials.rename(columns={"snapshot_date": "financial_snapshot_date"})
    features = applications.merge(attr, how="left", left_on=["customer_id", "feature_as_of_date"],
                                  right_on=["customer_id", "attribute_snapshot_date"], validate="one_to_one")
    features = features.merge(fin, how="left", left_on=["customer_id", "feature_as_of_date"],
                              right_on=["customer_id", "financial_snapshot_date"], validate="one_to_one")

    clickstream = clickstream.rename(columns={"snapshot_date": "clickstream_snapshot_date"})
    features = pd.merge_asof(
        features.sort_values("feature_as_of_date"),
        clickstream.sort_values("clickstream_snapshot_date"),
        left_on="feature_as_of_date", right_on="clickstream_snapshot_date",
        by="customer_id", direction="backward",
    ).sort_values(["feature_as_of_date", "loan_id"]).reset_index(drop=True)
    features["clickstream_available"] = features["clickstream_snapshot_date"].notna().astype("int8")
    features["debt_to_annual_income"] = (features["outstanding_debt"] / features["annual_income"].where(features["annual_income"].gt(0)))
    features["emi_to_monthly_income"] = (features["total_emi_per_month"] / features["monthly_inhand_salary"].where(features["monthly_inhand_salary"].gt(0)))

    labels = loans.loc[loans["mob"].eq(6),
                       ["loan_id", "customer_id", "loan_start_date", "snapshot_date", "dpd"]].copy()
    labels = labels.rename(columns={"loan_start_date": "feature_as_of_date",
                                    "snapshot_date": "label_observation_date"})
    labels["label"] = labels["dpd"].ge(30).astype("int8")
    labels["label_def"] = "30dpd_6mob"
    labels = labels.drop(columns="dpd").sort_values(["feature_as_of_date", "loan_id"]).reset_index(drop=True)
    return features, labels


def _validate(features: pd.DataFrame, labels: pd.DataFrame) -> dict:
    if len(features) != 12500 or len(labels) != 12500:
        raise ValueError(f"Expected 12,500 application rows in each Gold table, got {len(features)} and {len(labels)}")
    if features["loan_id"].duplicated().any() or labels["loan_id"].duplicated().any():
        raise ValueError("Gold loan_id values must be unique")
    if set(features["loan_id"]) != set(labels["loan_id"]):
        raise ValueError("Feature and label populations differ")
    if (features["clickstream_snapshot_date"] > features["feature_as_of_date"]).any():
        raise ValueError("Future clickstream data leaked into features")
    if (labels["label_observation_date"] <= labels["feature_as_of_date"]).any():
        raise ValueError("Labels must be observed after applications")
    if {"name", "ssn"} & set(features.columns):
        raise ValueError("PII leaked into Gold features")
    available = int(features["clickstream_available"].sum())
    if available != 8974:
        raise ValueError(f"Expected 8,974 applications with clickstream, got {available}")
    return {"feature_rows": len(features), "label_rows": len(labels),
            "clickstream_available": available, "clickstream_missing": len(features) - available,
            "bad_rate": float(labels["label"].mean()),
            "feature_min_date": str(features["feature_as_of_date"].min().date()),
            "feature_max_date": str(features["feature_as_of_date"].max().date())}


def prepare_datamart(force: bool = False) -> dict:
    fingerprints = {name: _sha256(path) for name, path in SOURCE_FILES.items()}
    manifest_path = DATAMART_DIR / "data_manifest.json"
    expected_outputs = [GOLD_DIR / "feature_store.parquet", GOLD_DIR / "label_store.parquet"]
    if not force and manifest_path.exists() and all(path.exists() for path in expected_outputs):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source_sha256") == fingerprints:
            return manifest

    raw = {name: pd.read_csv(path, dtype="string", keep_default_na=False)
           for name, path in SOURCE_FILES.items()}
    for name, frame in raw.items():
        _save(frame, BRONZE_DIR / f"{name}.parquet")

    silver = {
        "features_attributes": _clean_attributes(raw["features_attributes"]),
        "features_financials": _clean_financials(raw["features_financials"]),
        "feature_clickstream": _clean_clickstream(raw["feature_clickstream"]),
        "loan_daily": _clean_loans(raw["lms_loan_daily"]),
    }
    for name, frame in silver.items():
        _save(frame, SILVER_DIR / f"{name}.parquet")

    features, labels = _build_gold(silver["features_attributes"], silver["features_financials"],
                                   silver["feature_clickstream"], silver["loan_daily"])
    summary = _validate(features, labels)
    _save(features, expected_outputs[0])
    _save(labels, expected_outputs[1])
    manifest = {"source_sha256": fingerprints, "summary": summary,
                "lineage": "CS611 Assignment 1: application-time features and 30+ DPD at MOB 6"}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
