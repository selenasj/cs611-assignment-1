import tempfile
import unittest
from datetime import date
from pathlib import Path

from pyspark.sql import SparkSession

from utils.asof_join import attach_latest_clickstream
from utils.gold import derive_label_store
from utils.io_helpers import write_partitioned_parquet


class PipelineBehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("cs611-pipeline-tests")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_mob6_label_is_a_point_in_time_definition(self) -> None:
        loans = self.spark.createDataFrame(
            [
                ("L1", "C1", date(2023, 1, 1), date(2023, 6, 1), 5, 99),
                ("L1", "C1", date(2023, 1, 1), date(2023, 7, 1), 6, 29),
                ("L2", "C2", date(2023, 1, 1), date(2023, 7, 1), 6, 30),
            ],
            ["loan_id", "customer_id", "loan_start_date", "snapshot_date", "mob", "dpd"],
        )

        labels = {
            row["loan_id"]: (row["label"], row["label_def"])
            for row in derive_label_store(loans).collect()
        }

        self.assertEqual(labels, {"L1": (0, "30dpd_6mob"), "L2": (1, "30dpd_6mob")})

    def test_asof_join_retains_customers_without_clickstream(self) -> None:
        applications = self.spark.createDataFrame(
            [
                ("L1", "C1", date(2023, 3, 1)),
                ("L2", "C2", date(2023, 3, 1)),
            ],
            ["loan_id", "customer_id", "feature_as_of_date"],
        )
        clickstream = self.spark.createDataFrame(
            [
                ("C1", date(2023, 2, 1), 10.0),
                ("C1", date(2023, 4, 1), 999.0),
            ],
            ["customer_id", "snapshot_date", "fe_1"],
        )

        rows = {
            row["loan_id"]: row.asDict()
            for row in attach_latest_clickstream(applications, clickstream).collect()
        }

        self.assertEqual(set(rows), {"L1", "L2"})
        self.assertEqual(rows["L1"]["fe_1"], 10.0)
        self.assertEqual(rows["L1"]["clickstream_available"], 1)
        self.assertEqual(rows["L2"]["clickstream_available"], 0)

    def test_partitioned_write_is_safe_to_rerun(self) -> None:
        df = self.spark.createDataFrame(
            [("L1", date(2023, 1, 1)), ("L2", date(2023, 2, 1))],
            ["loan_id", "snapshot_date"],
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "rerun_check"
            write_partitioned_parquet(df, output_path, "snapshot_date")
            write_partitioned_parquet(df, output_path, "snapshot_date")
            self.assertEqual(self.spark.read.parquet(str(output_path)).count(), 2)


if __name__ == "__main__":
    unittest.main()
