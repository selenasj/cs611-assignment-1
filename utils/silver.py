from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, IntegerType, StringType

from utils.config import BRONZE_DIR, SILVER_DIR
from utils.io_helpers import write_partitioned_parquet


def _clean_numeric(column_name: str, data_type=DoubleType()):
    value = F.trim(F.col(column_name))
    value = F.regexp_replace(value, r"_$", "")
    return value.cast(data_type)


def _parse_failure_flag(column_name: str, cleaned_column) -> F.Column:
    raw = F.trim(F.col(column_name))
    return (
        F.when(raw.isNotNull() & (raw != "") & cleaned_column.isNull(), F.lit(1))
        .otherwise(F.lit(0))
        .cast(IntegerType())
    )


def _read_bronze(spark: SparkSession, table_name: str) -> DataFrame:
    return spark.read.parquet(str(BRONZE_DIR / table_name))


def clean_attributes(spark: SparkSession) -> DataFrame:
    df = _read_bronze(spark, "features_attributes")
    raw_age = _clean_numeric("Age", IntegerType())

    occupation = F.trim(F.col("Occupation"))
    occupation_invalid = occupation.isNull() | (occupation == "") | occupation.rlike(r"^_+$")

    return df.select(
        F.col("Customer_ID").cast(StringType()).alias("customer_id"),
        F.to_date("snapshot_date").cast(DateType()).alias("snapshot_date"),
        F.col("Name").cast(StringType()).alias("name"),
        F.col("SSN").cast(StringType()).alias("ssn"),
        F.when(raw_age.between(18, 100), raw_age).otherwise(F.lit(None).cast(IntegerType())).alias("age"),
        F.when(~raw_age.between(18, 100) | raw_age.isNull(), 1).otherwise(0).cast(IntegerType()).alias("age_invalid_flag"),
        F.when(occupation_invalid, F.lit("Unknown")).otherwise(occupation).alias("occupation"),
        F.when(occupation_invalid, 1).otherwise(0).cast(IntegerType()).alias("occupation_unknown_flag"),
    )


def clean_financials(spark: SparkSession) -> DataFrame:
    df = _read_bronze(spark, "features_financials")

    numeric_columns = {
        "Annual_Income": ("annual_income", DoubleType()),
        "Monthly_Inhand_Salary": ("monthly_inhand_salary", DoubleType()),
        "Num_Bank_Accounts": ("num_bank_accounts", IntegerType()),
        "Num_Credit_Card": ("num_credit_card", IntegerType()),
        "Interest_Rate": ("interest_rate", DoubleType()),
        "Num_of_Loan": ("num_of_loan", IntegerType()),
        "Delay_from_due_date": ("delay_from_due_date", IntegerType()),
        "Num_of_Delayed_Payment": ("num_of_delayed_payment", IntegerType()),
        "Changed_Credit_Limit": ("changed_credit_limit", DoubleType()),
        "Num_Credit_Inquiries": ("num_credit_inquiries", DoubleType()),
        "Outstanding_Debt": ("outstanding_debt", DoubleType()),
        "Credit_Utilization_Ratio": ("credit_utilization_ratio", DoubleType()),
        "Total_EMI_per_month": ("total_emi_per_month", DoubleType()),
        "Amount_invested_monthly": ("amount_invested_monthly", DoubleType()),
        "Monthly_Balance": ("monthly_balance", DoubleType()),
    }

    selections = [
        F.col("Customer_ID").cast(StringType()).alias("customer_id"),
        F.to_date("snapshot_date").cast(DateType()).alias("snapshot_date"),
    ]
    parse_flags = []
    cleaned_columns = {}
    for source_name, (target_name, data_type) in numeric_columns.items():
        cleaned = _clean_numeric(source_name, data_type)
        cleaned_columns[target_name] = cleaned
        selections.append(cleaned.alias(target_name))
        parse_flag = _parse_failure_flag(source_name, cleaned)
        parse_flags.append(parse_flag)
        selections.append(parse_flag.alias(f"{target_name}_parse_error_flag"))

    credit_mix = F.trim(F.col("Credit_Mix"))
    payment_behaviour = F.trim(F.col("Payment_Behaviour"))
    credit_history_years = F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Years?", 1).cast(IntegerType())
    credit_history_months = F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Months?", 1).cast(IntegerType())
    payment_minimum = F.trim(F.col("Payment_of_Min_Amount"))

    # Flag values that conflict with a clear non-negative definition.  No upper
    # limit is imposed because the supplied material does not define one.
    nonnegative_fields = [
        "annual_income",
        "monthly_inhand_salary",
        "num_bank_accounts",
        "num_credit_card",
        "interest_rate",
        "num_of_loan",
        "num_of_delayed_payment",
        "num_credit_inquiries",
        "outstanding_debt",
        "credit_utilization_ratio",
        "total_emi_per_month",
        "amount_invested_monthly",
        "monthly_balance",
    ]
    domain_warnings = [
        F.when(cleaned_columns[name] < 0, 1).otherwise(0) for name in nonnegative_fields
    ]

    annual_to_monthly_ratio = cleaned_columns["annual_income"] / (
        cleaned_columns["monthly_inhand_salary"] * 12
    )
    income_consistency_warning = (
        F.when(
            (cleaned_columns["annual_income"] > 0)
            & (cleaned_columns["monthly_inhand_salary"] > 0)
            & ((annual_to_monthly_ratio < 0.25) | (annual_to_monthly_ratio > 4.0)),
            1,
        )
        .otherwise(0)
        .cast(IntegerType())
    )

    selections.extend(
        [
            F.col("Type_of_Loan").cast(StringType()).alias("type_of_loan"),
            F.when(credit_mix.isNull() | (credit_mix == "") | credit_mix.rlike(r"^_+$"), "Unknown")
            .otherwise(credit_mix)
            .alias("credit_mix"),
            F.when(
                payment_behaviour.isNull()
                | (payment_behaviour == "")
                | payment_behaviour.rlike(r"^[^A-Za-z]+$"),
                "Unknown",
            )
            .otherwise(payment_behaviour)
            .alias("payment_behaviour"),
            F.when(
                payment_minimum.isNull()
                | (payment_minimum == "")
                | (F.upper(payment_minimum) == "NM"),
                "Unknown",
            )
            .otherwise(payment_minimum)
            .alias("payment_of_min_amount"),
            (credit_history_years * 12 + credit_history_months).cast(IntegerType()).alias("credit_history_months"),
            F.greatest(*parse_flags).cast(IntegerType()).alias("financial_parse_error_flag"),
            F.greatest(*domain_warnings).cast(IntegerType()).alias("financial_domain_warning_flag"),
            income_consistency_warning.alias("income_consistency_warning_flag"),
        ]
    )
    return df.select(*selections)


def clean_clickstream(spark: SparkSession) -> DataFrame:
    df = _read_bronze(spark, "feature_clickstream")
    selections = [
        F.col("Customer_ID").cast(StringType()).alias("customer_id"),
        F.to_date("snapshot_date").cast(DateType()).alias("snapshot_date"),
    ]
    # fe_1 to fe_20 are anonymous.  Cast them to numeric values, but do not treat
    # negatives as errors without a data dictionary that defines their meaning.
    selections.extend(F.col(f"fe_{i}").cast(DoubleType()).alias(f"fe_{i}") for i in range(1, 21))
    return df.select(*selections)


def clean_loans(spark: SparkSession) -> DataFrame:
    df = _read_bronze(spark, "lms_loan_daily")
    cleaned = df.select(
        F.col("loan_id").cast(StringType()).alias("loan_id"),
        F.col("Customer_ID").cast(StringType()).alias("customer_id"),
        F.to_date("loan_start_date").cast(DateType()).alias("loan_start_date"),
        F.col("tenure").cast(IntegerType()).alias("tenure"),
        F.col("installment_num").cast(IntegerType()).alias("installment_num"),
        F.col("loan_amt").cast(DoubleType()).alias("loan_amt"),
        F.col("due_amt").cast(DoubleType()).alias("due_amt"),
        F.col("paid_amt").cast(DoubleType()).alias("paid_amt"),
        F.col("overdue_amt").cast(DoubleType()).alias("overdue_amt"),
        F.col("balance").cast(DoubleType()).alias("balance"),
        F.to_date("snapshot_date").cast(DateType()).alias("snapshot_date"),
    )

    cleaned = cleaned.withColumn("mob", F.col("installment_num"))
    cleaned = cleaned.withColumn(
        "installments_missed",
        F.when(
            (F.col("overdue_amt") > 0) & (F.col("due_amt") > 0),
            F.ceil(F.col("overdue_amt") / F.col("due_amt")),
        )
        .otherwise(0)
        .cast(IntegerType()),
    )
    cleaned = cleaned.withColumn(
        "first_missed_date",
        F.when(
            F.col("installments_missed") > 0,
            F.add_months(F.col("snapshot_date"), -F.col("installments_missed")),
        ).cast(DateType()),
    )
    return cleaned.withColumn(
        "dpd",
        F.when(
            F.col("overdue_amt") > 0,
            F.datediff(F.col("snapshot_date"), F.col("first_missed_date")),
        )
        .otherwise(0)
        .cast(IntegerType()),
    )


def build_silver_tables(spark: SparkSession) -> dict[str, int]:
    builders = {
        "features_attributes": clean_attributes,
        "features_financials": clean_financials,
        "feature_clickstream": clean_clickstream,
        "loan_daily": clean_loans,
    }
    counts: dict[str, int] = {}
    for table_name, builder in builders.items():
        df = builder(spark)
        counts[table_name] = df.count()
        write_partitioned_parquet(df, SILVER_DIR / table_name, "snapshot_date")
        print(f"Silver {table_name}: {counts[table_name]:,} rows")
    return counts
