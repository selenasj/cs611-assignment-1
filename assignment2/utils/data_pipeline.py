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


def _build_features(attributes: pd.DataFrame, financials: pd.DataFrame,
                    clickstream: pd.DataFrame, loans: pd.DataFrame) -> pd.DataFrame:
    """One row per loan application, using only data dated on or before the application."""
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
    return features


def _build_labels(loans: pd.DataFrame) -> pd.DataFrame:
    """Label = 1 when the loan is 30+ days past due at month-on-book 6."""
    labels = loans.loc[loans["mob"].eq(6),
                       ["loan_id", "customer_id", "loan_start_date", "snapshot_date", "dpd"]].copy()
    labels = labels.rename(columns={"loan_start_date": "feature_as_of_date",
                                    "snapshot_date": "label_observation_date"})
    labels["label"] = labels["dpd"].ge(30).astype("int8")
    labels["label_def"] = "30dpd_6mob"
    return labels.drop(columns="dpd").sort_values(["feature_as_of_date", "loan_id"]).reset_index(drop=True)


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


# ---------------------------------------------------------------------------
# Pipeline stages. Each Airflow task calls exactly one of these functions.
# A stage is skipped when its inputs and this file are unchanged since its last
# successful run, so repeated monthly runs never rewrite identical Gold files.
# ---------------------------------------------------------------------------

SILVER_NAMES = {
    "features_attributes": "features_attributes",
    "features_financials": "features_financials",
    "feature_clickstream": "feature_clickstream",
    "lms_loan_daily": "loan_daily",
}
SILVER_CLEANERS = {
    "features_attributes": _clean_attributes,
    "features_financials": _clean_financials,
    "feature_clickstream": _clean_clickstream,
    "lms_loan_daily": _clean_loans,
}
STAGE_DIR = DATAMART_DIR / "stage_manifests"
FEATURE_STORE = GOLD_DIR / "feature_store.parquet"
LABEL_STORE = GOLD_DIR / "label_store.parquet"


def _bronze_path(source: str) -> Path:
    return BRONZE_DIR / f"{source}.parquet"


def _silver_path(source: str) -> Path:
    return SILVER_DIR / f"{SILVER_NAMES[source]}.parquet"


def _run_stage(stage: str, inputs: list[Path], outputs: list[Path], build, force: bool = False) -> dict:
    state = {"inputs": {path.name: _sha256(path) for path in inputs},
             "transform_code_sha256": _sha256(Path(__file__))}
    manifest = STAGE_DIR / f"{stage}.json"
    if (not force and manifest.exists() and all(path.exists() for path in outputs)
            and json.loads(manifest.read_text()).get("state") == state):
        print(f"[{stage}] inputs and code unchanged; reusing {len(outputs)} output file(s)")
        return {"stage": stage, "status": "up_to_date"}
    summary = build()
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"state": state, "summary": summary}, indent=2, default=str))
    print(f"[{stage}] rebuilt: {summary}")
    return {"stage": stage, "status": "rebuilt", **summary}


def check_source_files() -> dict:
    """Stop the run early if a supplied CSV is missing, empty or lacks its key columns."""
    required = {"features_attributes": {"Customer_ID", "snapshot_date"},
                "features_financials": {"Customer_ID", "snapshot_date"},
                "feature_clickstream": {"Customer_ID", "snapshot_date"},
                "lms_loan_daily": {"loan_id", "Customer_ID", "loan_start_date", "snapshot_date"}}
    report = {}
    for source, path in SOURCE_FILES.items():
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Source file missing or empty: {path.name}")
        header = set(pd.read_csv(path, nrows=0).columns)
        missing = required[source] - header
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        report[source] = {"bytes": path.stat().st_size}
    print(f"[check_source_files] all {len(report)} source files present with key columns")
    return report


def build_bronze(source: str, force: bool = False) -> dict:
    """Bronze keeps the supplied CSV exactly as text."""
    def build():
        raw = pd.read_csv(SOURCE_FILES[source], dtype="string", keep_default_na=False)
        _save(raw, _bronze_path(source))
        return {"rows": len(raw)}
    return _run_stage(f"bronze_{source}", [SOURCE_FILES[source]], [_bronze_path(source)], build, force)


def build_silver(source: str, force: bool = False) -> dict:
    """Silver parses types, cleans placeholders and adds warning flags."""
    def build():
        clean = SILVER_CLEANERS[source](pd.read_parquet(_bronze_path(source)))
        _save(clean, _silver_path(source))
        return {"rows": len(clean)}
    return _run_stage(f"silver_{source}", [_bronze_path(source)], [_silver_path(source)], build, force)


def build_label_store(force: bool = False) -> dict:
    def build():
        labels = _build_labels(pd.read_parquet(_silver_path("lms_loan_daily")))
        _save(labels, LABEL_STORE)
        return {"rows": len(labels), "bad_rate": float(labels["label"].mean())}
    return _run_stage("gold_label_store", [_silver_path("lms_loan_daily")], [LABEL_STORE], build, force)


def build_feature_store(force: bool = False) -> dict:
    inputs = [_silver_path(source) for source in SOURCE_FILES]

    def build():
        silver = {source: pd.read_parquet(_silver_path(source)) for source in SOURCE_FILES}
        features = _build_features(silver["features_attributes"], silver["features_financials"],
                                   silver["feature_clickstream"], silver["lms_loan_daily"])
        _save(features, FEATURE_STORE)
        return {"rows": len(features)}
    return _run_stage("gold_feature_store", inputs, [FEATURE_STORE], build, force)


def validate_gold_stores() -> dict:
    """Quality gate between the data pipeline and the model pipeline."""
    summary = _validate(pd.read_parquet(FEATURE_STORE), pd.read_parquet(LABEL_STORE))
    manifest = {"source_sha256": {name: _sha256(path) for name, path in SOURCE_FILES.items()},
                "transform_code_sha256": _sha256(Path(__file__)),
                "summary": summary,
                "lineage": "CS611 Assignment 1: application-time features and 30+ DPD at MOB 6"}
    manifest_path = DATAMART_DIR / "data_manifest.json"
    if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[validate_gold_stores] passed: {summary}")
    return manifest


def prepare_datamart(force: bool = False) -> dict:
    """Run every data stage in DAG order (used by main.py, tests and experiments)."""
    check_source_files()
    for source in SOURCE_FILES:
        build_bronze(source, force)
        build_silver(source, force)
    build_label_store(force)
    build_feature_store(force)
    return validate_gold_stores()
