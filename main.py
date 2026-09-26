from pyspark.sql import SparkSession

from utils.bronze import build_bronze_tables
from utils.config import DATAMART_DIR
from utils.gold import build_gold_tables
from utils.manifest import PipelineManifest
from utils.silver import build_silver_tables


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("cs611-assignment-1")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def print_stage_summary(stage_name: str, result: dict) -> None:
    tables = result.get("tables", result)
    for table_name, details in tables.items():
        if not isinstance(details, dict) or "row_count" not in details:
            continue
        print(
            f"{stage_name:<6} | {table_name:<22} | "
            f"rows={details['row_count']:,} | "
            f"range={details['minimum_partition']} to {details['maximum_partition']} | "
            f"status={details['status']}"
        )
    if "clickstream_coverage" in result:
        coverage = result["clickstream_coverage"]
        print(
            "Gold   | clickstream coverage   | "
            f"available={coverage['available']:,} | missing={coverage['missing']:,}"
        )


def main() -> None:
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    manifest = PipelineManifest()
    manifest_path = DATAMART_DIR / "run_manifest.json"

    try:
        print("\n=== Building Bronze tables ===")
        bronze_result = build_bronze_tables(spark)
        manifest.record_stage("bronze", bronze_result)

        print("\n=== Building Silver tables ===")
        silver_result = build_silver_tables(spark)
        manifest.record_stage("silver", silver_result)

        print("\n=== Building Gold tables ===")
        gold_result = build_gold_tables(spark)
        manifest.record_stage("gold", gold_result)

        manifest.complete()
        manifest.write(manifest_path)

        print("\n=== Pipeline completed successfully ===")
        print_stage_summary("Bronze", bronze_result)
        print_stage_summary("Silver", silver_result)
        print_stage_summary("Gold", gold_result)
        print(f"Run manifest: {manifest_path}")
    except Exception as error:
        manifest.fail(error)
        manifest.write(manifest_path)
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
