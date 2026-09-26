from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType

from utils.asof_join import attach_latest_clickstream
from utils.config import GOLD_DIR, SILVER_DIR
from utils.contracts import GOLD_CONTRACTS
from utils.io_helpers import write_partitioned_parquet
from utils.validation import validate_gold_outputs


def _read_silver(spark: SparkSession, table_name: str) -> DataFrame:
    return spark.read.parquet(str(SILVER_DIR / table_name))


def _safe_ratio(numerator: str, denominator: str):
    return F.when(
        F.col(denominator).isNotNull() & (F.col(denominator) > 0),
        F.col(numerator) / F.col(denominator),
    )


def build_feature_store(spark: SparkSession) -> DataFrame:
    loans = _read_silver(spark, "loan_daily")
    attributes = _read_silver(spark, "features_attributes")
    financials = _read_silver(spark, "features_financials")
    clickstream = _read_silver(spark, "feature_clickstream")

    applications = loans.filter(F.col("mob") == 0).select(
        "loan_id",
        "customer_id",
        F.col("loan_start_date").alias("feature_as_of_date"),
        "loan_amt",
        "tenure",
    )

    attr = attributes.drop("name", "ssn").withColumnRenamed("snapshot_date", "attribute_snapshot_date")
    fin = financials.withColumnRenamed("snapshot_date", "financial_snapshot_date")

    features = applications.join(
        attr,
        (applications.customer_id == attr.customer_id)
        & (applications.feature_as_of_date == attr.attribute_snapshot_date),
        "left",
    ).drop(attr.customer_id)

    features = features.join(
        fin,
        (features.customer_id == fin.customer_id)
        & (features.feature_as_of_date == fin.financial_snapshot_date),
        "left",
    ).drop(fin.customer_id)

    features = attach_latest_clickstream(features, clickstream)

    features = (
        features.withColumn("debt_to_annual_income", _safe_ratio("outstanding_debt", "annual_income"))
        .withColumn("emi_to_monthly_income", _safe_ratio("total_emi_per_month", "monthly_inhand_salary"))
    )

    # Keep source dates so the timing of every join can be checked later.
    return features


def derive_label_store(loans: DataFrame, dpd_threshold: int = 30, mob: int = 6) -> DataFrame:
    """Create the point-in-time outcome observed at one specified MOB."""
    return (
        loans.filter(F.col("mob") == mob)
        .withColumn(
            "label",
            F.when(F.col("dpd") >= dpd_threshold, 1).otherwise(0).cast(IntegerType()),
        )
        .withColumn("label_def", F.lit(f"{dpd_threshold}dpd_{mob}mob").cast(StringType()))
        .select(
            "loan_id",
            "customer_id",
            F.col("loan_start_date").alias("feature_as_of_date"),
            F.col("snapshot_date").alias("label_observation_date"),
            "label",
            "label_def",
        )
    )


def build_label_store(spark: SparkSession, dpd_threshold: int = 30, mob: int = 6) -> DataFrame:
    return derive_label_store(_read_silver(spark, "loan_daily"), dpd_threshold, mob)


def build_gold_tables(spark: SparkSession) -> dict:
    feature_store = build_feature_store(spark).cache()
    label_store = build_label_store(spark).cache()
    result = validate_gold_outputs(
        feature_store,
        label_store,
        GOLD_CONTRACTS["feature_store"],
        GOLD_CONTRACTS["label_store"],
    )

    write_partitioned_parquet(feature_store, GOLD_DIR / "feature_store", "feature_as_of_date")
    write_partitioned_parquet(label_store, GOLD_DIR / "label_store", "label_observation_date")

    print(f"Gold feature_store: {result['tables']['feature_store']['row_count']:,} rows")
    print(f"Gold label_store: {result['tables']['label_store']['row_count']:,} rows")
    print(f"Applications with clickstream: {result['clickstream_coverage']['available']:,}")

    feature_store.unpersist()
    label_store.unpersist()
    return result
