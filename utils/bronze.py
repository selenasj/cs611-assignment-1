from pyspark.sql import SparkSession

from utils.config import BRONZE_DIR, SOURCE_FILES
from utils.contracts import BRONZE_CONTRACTS
from utils.io_helpers import add_ingestion_metadata, read_raw_csv, write_partitioned_parquet
from utils.validation import validate_table_contract


def build_bronze_tables(spark: SparkSession) -> dict[str, dict]:
    """Land raw sources as string-valued, date-partitioned Bronze tables."""
    results: dict[str, dict] = {}

    for source_name, source_path in SOURCE_FILES.items():
        df = add_ingestion_metadata(read_raw_csv(spark, source_path), source_name).cache()
        result = validate_table_contract(df, BRONZE_CONTRACTS[source_name])
        results[source_name] = result.to_dict()
        write_partitioned_parquet(
            df,
            BRONZE_DIR / source_name,
            partition_column="snapshot_date",
        )
        print(f"Bronze {source_name}: {result.row_count:,} rows")
        df.unpersist()

    return results
