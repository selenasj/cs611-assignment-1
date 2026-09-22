from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType

from utils.config import GOLD_DIR, SILVER_DIR
from utils.io_helpers import write_partitioned_parquet


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

    # Restrict clickstream to information available by the application date.
    cs = clickstream.withColumnRenamed("snapshot_date", "clickstream_snapshot_date")
    features = features.join(
        cs,
        (features.customer_id == cs.customer_id)
        & (cs.clickstream_snapshot_date <= features.feature_as_of_date),
        "left",
    ).drop(cs.customer_id)

    latest_clickstream = Window.partitionBy("loan_id").orderBy(
        F.col("clickstream_snapshot_date").desc_nulls_last()
    )
    features = (
        features.withColumn("_clickstream_rank", F.row_number().over(latest_clickstream))
        .filter(F.col("_clickstream_rank") == 1)
        .drop("_clickstream_rank")
    )

    features = (
        features.withColumn(
            "clickstream_available",
            F.when(F.col("clickstream_snapshot_date").isNotNull(), 1).otherwise(0).cast(IntegerType()),
        )
        .withColumn("debt_to_annual_income", _safe_ratio("outstanding_debt", "annual_income"))
        .withColumn("emi_to_monthly_income", _safe_ratio("total_emi_per_month", "monthly_inhand_salary"))
    )

    # Keep source dates so the timing of every join can be checked later.
    return features


def build_label_store(spark: SparkSession, dpd_threshold: int = 30, mob: int = 6) -> DataFrame:
    loans = _read_silver(spark, "loan_daily")
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


def _assert_unique(df: DataFrame, columns: list[str], table_name: str) -> None:
    duplicates = df.groupBy(*columns).count().filter(F.col("count") > 1).limit(1).count()
    if duplicates:
        raise ValueError(f"{table_name} is not unique on {columns}")


def validate_gold_tables(feature_store: DataFrame, label_store: DataFrame) -> None:
    feature_count = feature_store.count()
    label_count = label_store.count()
    _assert_unique(feature_store, ["loan_id"], "feature_store")
    _assert_unique(label_store, ["loan_id"], "label_store")

    application_count = feature_store.select("loan_id").distinct().count()
    if feature_count != application_count:
        raise ValueError("Feature-store join unexpectedly changed the application grain")
    if feature_count != label_count:
        raise ValueError(
            f"Feature/label row counts differ: feature={feature_count}, label={label_count}"
        )

    bad_time_order = label_store.filter(
        F.col("feature_as_of_date") >= F.col("label_observation_date")
    ).limit(1).count()
    if bad_time_order:
        raise ValueError("A label observation date is not later than its feature date")

    future_clickstream = feature_store.filter(
        F.col("clickstream_snapshot_date") > F.col("feature_as_of_date")
    ).limit(1).count()
    if future_clickstream:
        raise ValueError("Future clickstream data leaked into the feature store")


def build_gold_tables(spark: SparkSession) -> dict[str, int]:
    feature_store = build_feature_store(spark).cache()
    label_store = build_label_store(spark).cache()
    validate_gold_tables(feature_store, label_store)

    counts = {
        "feature_store": feature_store.count(),
        "label_store": label_store.count(),
        "clickstream_available": feature_store.filter(F.col("clickstream_available") == 1).count(),
    }

    write_partitioned_parquet(feature_store, GOLD_DIR / "feature_store", "feature_as_of_date")
    write_partitioned_parquet(label_store, GOLD_DIR / "label_store", "label_observation_date")

    print(f"Gold feature_store: {counts['feature_store']:,} rows")
    print(f"Gold label_store: {counts['label_store']:,} rows")
    print(f"Applications with clickstream: {counts['clickstream_available']:,}")

    feature_store.unpersist()
    label_store.unpersist()
    return counts
