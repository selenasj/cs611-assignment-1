from pyspark.sql import SparkSession

from utils.bronze import build_bronze_tables
from utils.gold import build_gold_tables
from utils.silver import build_silver_tables


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("cs611-assignment-1")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def main() -> None:
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("\n=== Building Bronze tables ===")
        build_bronze_tables(spark)

        print("\n=== Building Silver tables ===")
        build_silver_tables(spark)

        print("\n=== Building Gold tables ===")
        counts = build_gold_tables(spark)

        print("\n=== Pipeline completed successfully ===")
        print(counts)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

