from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TableContract:
    """Declarative expectations for one persisted pipeline table."""

    name: str
    layer: str
    grain: str
    primary_key: tuple[str, ...]
    partition_column: str
    expected_rows: int
    required_columns: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


BRONZE_CONTRACTS = {
    "features_attributes": TableContract(
        name="features_attributes",
        layer="bronze",
        grain="one application-time customer attribute snapshot",
        primary_key=("Customer_ID", "snapshot_date"),
        partition_column="snapshot_date",
        expected_rows=12_500,
        required_columns=("Customer_ID", "snapshot_date", "Name", "SSN", "Age", "Occupation"),
    ),
    "features_financials": TableContract(
        name="features_financials",
        layer="bronze",
        grain="one application-time customer financial snapshot",
        primary_key=("Customer_ID", "snapshot_date"),
        partition_column="snapshot_date",
        expected_rows=12_500,
        required_columns=("Customer_ID", "snapshot_date", "Annual_Income"),
    ),
    "feature_clickstream": TableContract(
        name="feature_clickstream",
        layer="bronze",
        grain="one monthly clickstream snapshot per customer",
        primary_key=("Customer_ID", "snapshot_date"),
        partition_column="snapshot_date",
        expected_rows=215_376,
        required_columns=("Customer_ID", "snapshot_date", "fe_1", "fe_20"),
    ),
    "lms_loan_daily": TableContract(
        name="lms_loan_daily",
        layer="bronze",
        grain="one monthly repayment snapshot per loan",
        primary_key=("loan_id", "snapshot_date"),
        partition_column="snapshot_date",
        expected_rows=137_500,
        required_columns=("loan_id", "Customer_ID", "loan_start_date", "snapshot_date"),
    ),
}


SILVER_CONTRACTS = {
    "features_attributes": TableContract(
        name="features_attributes",
        layer="silver",
        grain="one cleaned application-time customer attribute snapshot",
        primary_key=("customer_id", "snapshot_date"),
        partition_column="snapshot_date",
        expected_rows=12_500,
        required_columns=("customer_id", "snapshot_date", "age", "occupation"),
    ),
    "features_financials": TableContract(
        name="features_financials",
        layer="silver",
        grain="one cleaned application-time customer financial snapshot",
        primary_key=("customer_id", "snapshot_date"),
        partition_column="snapshot_date",
        expected_rows=12_500,
        required_columns=("customer_id", "snapshot_date", "annual_income", "outstanding_debt"),
    ),
    "feature_clickstream": TableContract(
        name="feature_clickstream",
        layer="silver",
        grain="one cleaned monthly clickstream snapshot per customer",
        primary_key=("customer_id", "snapshot_date"),
        partition_column="snapshot_date",
        expected_rows=215_376,
        required_columns=("customer_id", "snapshot_date", "fe_1", "fe_20"),
    ),
    "loan_daily": TableContract(
        name="loan_daily",
        layer="silver",
        grain="one cleaned monthly repayment snapshot per loan",
        primary_key=("loan_id", "snapshot_date"),
        partition_column="snapshot_date",
        expected_rows=137_500,
        required_columns=("loan_id", "customer_id", "loan_start_date", "snapshot_date", "mob", "dpd"),
    ),
}


GOLD_CONTRACTS = {
    "feature_store": TableContract(
        name="feature_store",
        layer="gold",
        grain="one feature row per loan application",
        primary_key=("loan_id",),
        partition_column="feature_as_of_date",
        expected_rows=12_500,
        required_columns=(
            "loan_id",
            "customer_id",
            "feature_as_of_date",
            "clickstream_snapshot_date",
            "clickstream_available",
        ),
    ),
    "label_store": TableContract(
        name="label_store",
        layer="gold",
        grain="one MOB 6 outcome per loan application",
        primary_key=("loan_id",),
        partition_column="label_observation_date",
        expected_rows=12_500,
        required_columns=(
            "loan_id",
            "customer_id",
            "feature_as_of_date",
            "label_observation_date",
            "label",
            "label_def",
        ),
    ),
}


EXPECTED_CLICKSTREAM_AVAILABLE = 8_974
EXPECTED_CLICKSTREAM_MISSING = 3_526


def contracts_as_dict() -> dict[str, dict[str, dict]]:
    return {
        "bronze": {name: contract.to_dict() for name, contract in BRONZE_CONTRACTS.items()},
        "silver": {name: contract.to_dict() for name, contract in SILVER_CONTRACTS.items()},
        "gold": {name: contract.to_dict() for name, contract in GOLD_CONTRACTS.items()},
    }
