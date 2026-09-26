from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


def attach_latest_clickstream(applications: DataFrame, clickstream: DataFrame) -> DataFrame:
    """Left join the latest customer snapshot available by each application date."""
    required_app = {"loan_id", "customer_id", "feature_as_of_date"}
    required_clickstream = {"customer_id", "snapshot_date"}
    if not required_app.issubset(applications.columns):
        raise ValueError("Applications are missing fields required for the as-of join")
    if not required_clickstream.issubset(clickstream.columns):
        raise ValueError("Clickstream is missing fields required for the as-of join")

    cs = clickstream.withColumnRenamed("snapshot_date", "clickstream_snapshot_date")
    joined = applications.join(
        cs,
        (applications.customer_id == cs.customer_id)
        & (cs.clickstream_snapshot_date <= applications.feature_as_of_date),
        "left",
    ).drop(cs.customer_id)

    latest_available = Window.partitionBy("loan_id").orderBy(
        F.col("clickstream_snapshot_date").desc_nulls_last()
    )
    return (
        joined.withColumn("_clickstream_rank", F.row_number().over(latest_available))
        .filter(F.col("_clickstream_rank") == 1)
        .drop("_clickstream_rank")
        .withColumn(
            "clickstream_available",
            F.when(F.col("clickstream_snapshot_date").isNotNull(), 1)
            .otherwise(0)
            .cast(IntegerType()),
        )
    )
