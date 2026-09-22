from pyspark.sql import SparkSession

from utils.config import BRONZE_DIR, SOURCE_FILES
from utils.io_helpers import add_ingestion_metadata, read_raw_csv, write_partitioned_parquet


def build_bronze_tables(spark: SparkSession) -> dict[str, int]:
    """Land raw sources as string-valued, date-partitioned Bronze tables."""
    counts: dict[str, int] = {}

    for source_name, source_path in SOURCE_FILES.items():
        df = add_ingestion_metadata(read_raw_csv(spark, source_path), source_name)
        counts[source_name] = df.count()
        write_partitioned_parquet(
            df,
            BRONZE_DIR / source_name,
            partition_column="snapshot_date",
        )
        print(f"Bronze {source_name}: {counts[source_name]:,} rows")

    return counts

