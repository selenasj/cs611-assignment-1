import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from utils.feature_policy import APPROVED_NUMERIC_FEATURES


OUTPUT_DIR = Path("outputs/slide_assets")
TEMPORAL_CUTOFF = pd.Timestamp("2024-06-01")


def clip_from_training_bounds(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Winsorise using bounds learned only from the training period."""
    lower = train.quantile(0.01)
    upper = train.quantile(0.99)
    bounds = {
        column: {"lower": float(lower[column]), "upper": float(upper[column])}
        for column in train.columns
        if pd.notna(lower[column]) and pd.notna(upper[column])
    }
    return train.clip(lower=lower, upper=upper, axis=1), test.clip(
        lower=lower, upper=upper, axis=1
    ), bounds


def main() -> None:
    spark = (
        SparkSession.builder.appName("cs611-model-sanity-check")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        features = spark.read.parquet("datamart/gold/feature_store")
        labels = spark.read.parquet("datamart/gold/label_store").select(
            "loan_id", "label", "label_observation_date"
        )
        data = features.join(labels, on="loan_id", how="inner").select(
            "loan_id", "feature_as_of_date", "label", *APPROVED_NUMERIC_FEATURES
        )
        pdf = data.toPandas()
    finally:
        spark.stop()

    pdf["feature_as_of_date"] = pd.to_datetime(pdf["feature_as_of_date"])
    train = pdf[pdf["feature_as_of_date"] <= TEMPORAL_CUTOFF].copy()
    oot = pdf[pdf["feature_as_of_date"] > TEMPORAL_CUTOFF].copy()

    if train.empty or oot.empty:
        raise ValueError("Temporal train/OOT split produced an empty dataset")

    x_train = train[APPROVED_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    x_oot = oot[APPROVED_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    y_train = train["label"].astype(int)
    y_oot = oot["label"].astype(int)

    x_train, x_oot, clip_bounds = clip_from_training_bounds(x_train, x_oot)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=88),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    probability = pipeline.predict_proba(x_oot)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(y_oot, prediction)

    metrics = {
        "model": "logistic_regression",
        "purpose": "feature-store compatibility check only",
        "temporal_cutoff": TEMPORAL_CUTOFF.strftime("%Y-%m-%d"),
        "train_rows": int(len(train)),
        "oot_rows": int(len(oot)),
        "train_positive_rate": float(y_train.mean()),
        "oot_positive_rate": float(y_oot.mean()),
        "oot_roc_auc": float(roc_auc_score(y_oot, probability)),
        "oot_accuracy": float(accuracy_score(y_oot, prediction)),
        "oot_precision": float(precision_score(y_oot, prediction, zero_division=0)),
        "oot_recall": float(recall_score(y_oot, prediction, zero_division=0)),
        "oot_f1": float(f1_score(y_oot, prediction, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "feature_count": len(APPROVED_NUMERIC_FEATURES),
        "features": APPROVED_NUMERIC_FEATURES,
        "preprocessing": [
            "1st/99th percentile clipping fitted on training period",
            "median imputation fitted on training period",
            "robust scaling fitted on training period",
        ],
        "clip_bounds": clip_bounds,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "model_sanity_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({key: value for key, value in metrics.items() if key != "clip_bounds"}, indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()

