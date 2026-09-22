from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def require_source_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file not found: {path}")


def read_raw_csv(spark: SparkSession, path: Path) -> DataFrame:
    """Read every source value as a string so Bronze preserves the raw representation."""
    require_source_file(path)
    return (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("mode", "FAILFAST")
        .csv(str(path))
    )


def write_partitioned_parquet(
    df: DataFrame,
    output_path: Path,
    partition_column: str | None = None,
) -> None:
    writer = df.write.mode("overwrite")
    if partition_column:
        writer = writer.partitionBy(partition_column)
    writer.parquet(str(output_path))


def add_ingestion_metadata(df: DataFrame, source_name: str) -> DataFrame:
    return (
        df.withColumn("_source_name", F.lit(source_name))
        .withColumn("_ingested_at_utc", F.current_timestamp())
    )

