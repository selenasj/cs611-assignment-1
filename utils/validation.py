from dataclasses import asdict, dataclass
from typing import Iterable

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from utils.contracts import (
    EXPECTED_CLICKSTREAM_AVAILABLE,
    EXPECTED_CLICKSTREAM_MISSING,
    TableContract,
)


@dataclass(frozen=True)
class TableValidationResult:
    layer: str
    table: str
    row_count: int
    expected_rows: int
    minimum_partition: str | None
    maximum_partition: str | None
    duplicate_keys: int
    null_keys: int
    status: str = "passed"

    def to_dict(self) -> dict:
        return asdict(self)


def _require_columns(df: DataFrame, required: Iterable[str], table_name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def validate_table_contract(df: DataFrame, contract: TableContract) -> TableValidationResult:
    """Validate grain, keys, row volume and partition range against a contract."""
    _require_columns(
        df,
        (*contract.primary_key, contract.partition_column, *contract.required_columns),
        f"{contract.layer}.{contract.name}",
    )

    summary = df.agg(
        F.count(F.lit(1)).alias("row_count"),
        F.min(contract.partition_column).alias("minimum_partition"),
        F.max(contract.partition_column).alias("maximum_partition"),
    ).first()
    row_count = int(summary["row_count"])

    null_condition = None
    for column_name in contract.primary_key:
        condition = F.col(column_name).isNull()
        null_condition = condition if null_condition is None else null_condition | condition
    null_keys = df.filter(null_condition).count() if null_condition is not None else 0
    duplicate_keys = (
        df.groupBy(*contract.primary_key)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    errors = []
    if row_count != contract.expected_rows:
        errors.append(f"expected {contract.expected_rows:,} rows but found {row_count:,}")
    if null_keys:
        errors.append(f"found {null_keys:,} rows with a null primary key")
    if duplicate_keys:
        errors.append(f"found {duplicate_keys:,} duplicated primary-key values")
    if errors:
        raise ValueError(f"{contract.layer}.{contract.name}: " + "; ".join(errors))

    return TableValidationResult(
        layer=contract.layer,
        table=contract.name,
        row_count=row_count,
        expected_rows=contract.expected_rows,
        minimum_partition=str(summary["minimum_partition"]),
        maximum_partition=str(summary["maximum_partition"]),
        duplicate_keys=duplicate_keys,
        null_keys=null_keys,
    )


def assert_columns_absent(df: DataFrame, forbidden: Iterable[str], table_name: str) -> None:
    present = sorted(set(forbidden) & set(df.columns))
    if present:
        raise ValueError(f"{table_name} contains excluded columns: {present}")


def validate_label_timing(label_store: DataFrame) -> None:
    bad_rows = label_store.filter(
        F.col("feature_as_of_date") >= F.col("label_observation_date")
    ).limit(1).count()
    if bad_rows:
        raise ValueError("A label observation date is not later than its feature date")


def validate_clickstream_timing(feature_store: DataFrame) -> None:
    future_rows = feature_store.filter(
        F.col("clickstream_snapshot_date") > F.col("feature_as_of_date")
    ).limit(1).count()
    if future_rows:
        raise ValueError("Future clickstream data leaked into the feature store")


def validate_clickstream_coverage(feature_store: DataFrame) -> dict[str, int]:
    counts = {
        int(row["clickstream_available"]): int(row["count"])
        for row in feature_store.groupBy("clickstream_available").count().collect()
    }
    coverage = {
        "available": counts.get(1, 0),
        "missing": counts.get(0, 0),
    }
    expected = {
        "available": EXPECTED_CLICKSTREAM_AVAILABLE,
        "missing": EXPECTED_CLICKSTREAM_MISSING,
    }
    if coverage != expected:
        raise ValueError(f"Unexpected clickstream coverage: expected {expected}, found {coverage}")
    return coverage


def validate_gold_outputs(
    feature_store: DataFrame,
    label_store: DataFrame,
    feature_contract: TableContract,
    label_contract: TableContract,
) -> dict:
    feature_result = validate_table_contract(feature_store, feature_contract)
    label_result = validate_table_contract(label_store, label_contract)
    if feature_result.row_count != label_result.row_count:
        raise ValueError("Feature and label stores do not have the same application population")

    assert_columns_absent(feature_store, ("name", "ssn"), "gold.feature_store")
    validate_label_timing(label_store)
    validate_clickstream_timing(feature_store)
    coverage = validate_clickstream_coverage(feature_store)
    return {
        "tables": {
            "feature_store": feature_result.to_dict(),
            "label_store": label_result.to_dict(),
        },
        "clickstream_coverage": coverage,
    }
