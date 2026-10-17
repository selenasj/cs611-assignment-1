"""Time-aware model training, selection, inference, and monitoring.

Each date is the simulated production as-of date. A label is eligible for
training or performance monitoring only after its observation date.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from utils.config import DATAMART_DIR, GOLD_DIR, PROJECT_ROOT
from utils.feature_policy import APPROVED_NUMERIC_FEATURES


MODEL_BANK = PROJECT_ROOT / "model_bank"
PREDICTIONS_DIR = GOLD_DIR / "model_predictions"
MONITORING_DIR = GOLD_DIR / "model_monitoring"
MONITORED_FEATURES = ["age", "annual_income", "debt_to_annual_income"]
SCORE_THRESHOLD = 0.5


def _day(value: str | pd.Timestamp) -> pd.Timestamp:
    date = pd.Timestamp(value).normalize()
    if date.day != 1:
        raise ValueError("Expected a first-of-month as-of date")
    return date


def _load_gold() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_path = GOLD_DIR / "feature_store.parquet"
    label_path = GOLD_DIR / "label_store.parquet"
    if not feature_path.exists() or not label_path.exists():
        raise FileNotFoundError("Gold feature and label stores are missing; run prepare_datamart")
    return pd.read_parquet(feature_path), pd.read_parquet(label_path)


def _matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[APPROVED_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")


def _candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=88)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=15, learning_rate=0.05,
                l2_regularization=2.0, random_state=88,
            )),
        ]),
    }


def _reference(values: np.ndarray | pd.Series, bins: int = 10) -> dict:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        raise ValueError("Cannot make a monitoring reference from all-missing values")
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        pivot = float(edges[0])
        edges = np.array([pivot - 0.5, pivot + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    counts = np.histogram(finite, bins=edges)[0].astype(float)
    counts = np.append(counts, len(array) - len(finite))
    return {"edges": edges.tolist(), "proportions": (counts / len(array)).tolist(),
            "n": int(len(array))}


def population_stability_index(reference: dict, values: np.ndarray | pd.Series) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if len(array) == 0:
        return float("nan")
    finite = array[np.isfinite(array)]
    current = np.histogram(finite, bins=np.asarray(reference["edges"], dtype=float))[0].astype(float)
    current = np.append(current, len(array) - len(finite)) / len(array)
    expected = np.asarray(reference["proportions"], dtype=float)
    epsilon = 1e-4
    return float(np.sum((current - expected) * np.log((current + epsilon) / (expected + epsilon))))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _model_path(version: str) -> Path:
    return MODEL_BANK / f"{version}.joblib"


def train_and_select(as_of: str | pd.Timestamp) -> dict:
    date = _day(as_of)
    features, labels = _load_gold()
    scoring_cohort = features[features["feature_as_of_date"].eq(date)]
    if scoring_cohort.empty:
        return {"as_of_date": date.strftime("%Y-%m-%d"), "status": "no_application_cohort"}

    version = "credit_model_" + date.strftime("%Y_%m_%d")
    artifact_path = _model_path(version)
    if artifact_path.exists():
        return {"as_of_date": date.strftime("%Y-%m-%d"), "status": "already_trained",
                "model_version": version}

    matured = features.merge(labels[["loan_id", "label_observation_date", "label"]],
                             on="loan_id", how="inner", validate="one_to_one")
    matured = matured[
        matured["feature_as_of_date"].lt(date)
        & matured["label_observation_date"].le(date)
    ].copy()
    cohorts = sorted(matured["feature_as_of_date"].unique())
    if len(cohorts) < 6:
        raise ValueError(f"Only {len(cohorts)} mature cohorts available on {date.date()}")
    validation_dates = cohorts[-2:]
    train = matured[~matured["feature_as_of_date"].isin(validation_dates)].copy()
    validation = matured[matured["feature_as_of_date"].isin(validation_dates)].copy()
    if train["label"].nunique() < 2 or validation["label"].nunique() < 2:
        raise ValueError("Both classes are required for temporal AUC model selection")

    results: dict[str, float] = {}
    for name, pipeline in _candidates().items():
        pipeline.fit(_matrix(train), train["label"])
        probabilities = pipeline.predict_proba(_matrix(validation))[:, 1]
        results[name] = float(roc_auc_score(validation["label"], probabilities))
    selected_name = max(results, key=results.get)
    selected = _candidates()[selected_name]
    selected.fit(_matrix(matured), matured["label"])
    reference_scores = selected.predict_proba(_matrix(matured))[:, 1]
    references = {"score": _reference(reference_scores)}
    for feature in MONITORED_FEATURES:
        references[feature] = _reference(pd.to_numeric(matured[feature], errors="coerce"))

    metadata = {
        "model_version": version,
        "as_of_date": date.strftime("%Y-%m-%d"),
        "selected_algorithm": selected_name,
        "candidate_validation_auc": results,
        "selected_validation_auc": results[selected_name],
        "train_rows_for_selection": int(len(train)),
        "validation_rows": int(len(validation)),
        "refit_rows": int(len(matured)),
        "latest_label_observation_date": matured["label_observation_date"].max().strftime("%Y-%m-%d"),
        "validation_cohorts": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in validation_dates],
        "feature_names": APPROVED_NUMERIC_FEATURES,
        "selection_metric": "temporal_validation_roc_auc",
        "label_definition": "30+ DPD at MOB 6",
    }
    MODEL_BANK.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": selected, "metadata": metadata, "references": references}, artifact_path)
    (MODEL_BANK / f"{version}.json").write_text(json.dumps(metadata, indent=2))
    return {"as_of_date": metadata["as_of_date"], "status": "trained", **metadata}


def score_cohort(as_of: str | pd.Timestamp) -> dict:
    date = _day(as_of)
    date_text = date.strftime("%Y-%m-%d")
    features, _ = _load_gold()
    cohort = features[features["feature_as_of_date"].eq(date)].copy()
    if cohort.empty:
        return {"as_of_date": date_text, "status": "no_application_cohort", "rows": 0}
    version = "credit_model_" + date.strftime("%Y_%m_%d")
    path = _model_path(version)
    if not path.exists():
        raise FileNotFoundError(f"Model was not trained for {date_text}: {path}")
    artifact = joblib.load(path)
    scores = artifact["pipeline"].predict_proba(_matrix(cohort))[:, 1]
    predictions = cohort[["loan_id", "customer_id", "feature_as_of_date"]].copy()
    predictions["model_version"] = version
    predictions["score_as_of_date"] = date
    predictions["default_probability"] = scores
    predictions["high_risk_flag"] = (scores >= SCORE_THRESHOLD).astype("int8")
    output = PREDICTIONS_DIR / f"cohort_date={date_text}" / "predictions.parquet"
    _write_parquet(predictions, output)
    return {"as_of_date": date_text, "status": "scored", "rows": int(len(cohort)),
            "model_version": version, "path": str(output)}


def monitor_as_of(as_of: str | pd.Timestamp) -> dict:
    date = _day(as_of)
    date_text = date.strftime("%Y-%m-%d")
    features, labels = _load_gold()
    prediction_paths = sorted(PREDICTIONS_DIR.glob("cohort_date=*/predictions.parquet"))
    prediction_paths = [p for p in prediction_paths if p.parent.name.split("=", 1)[1] <= date_text]
    if not prediction_paths:
        raise FileNotFoundError("No predictions exist yet for monitoring")
    labels = labels[["loan_id", "label_observation_date", "label"]]
    context = features[["loan_id", *MONITORED_FEATURES]]
    rows = []
    for path in prediction_paths:
        predictions = pd.read_parquet(path)
        cohort = predictions.merge(labels, on="loan_id", how="left", validate="one_to_one")
        cohort = cohort.merge(context, on="loan_id", how="left", validate="one_to_one")
        observed = cohort[cohort["label_observation_date"].le(date)].copy()
        version = str(cohort["model_version"].iloc[0])
        artifact = joblib.load(_model_path(version))
        references = artifact["references"]
        auc = float(roc_auc_score(observed["label"], observed["default_probability"])) if observed["label"].nunique() == 2 else math.nan
        brier = float(brier_score_loss(observed["label"], observed["default_probability"])) if not observed.empty else math.nan
        psi = population_stability_index(references["score"], cohort["default_probability"])
        csi = {f"csi_{name}": population_stability_index(references[name], cohort[name])
               for name in MONITORED_FEATURES}
        baseline_auc = float(artifact["metadata"]["selected_validation_auc"])
        performance_alert = bool(math.isfinite(auc) and auc < baseline_auc - 0.05)
        stability_alert = bool(psi >= 0.25 or any(value >= 0.25 for value in csi.values()))
        rows.append({
            "monitoring_as_of_date": date,
            "cohort_date": cohort["feature_as_of_date"].iloc[0],
            "model_version": version,
            "algorithm": artifact["metadata"]["selected_algorithm"],
            "n_predictions": int(len(cohort)),
            "n_observed_labels": int(len(observed)),
            "observed_bad_rate": float(observed["label"].mean()) if not observed.empty else math.nan,
            "roc_auc": auc,
            "brier_score": brier,
            "reference_validation_auc": baseline_auc,
            "mean_predicted_risk": float(cohort["default_probability"].mean()),
            "psi_score": psi,
            **csi,
            "performance_alert": performance_alert,
            "stability_alert": stability_alert,
            "refresh_review_required": performance_alert or stability_alert,
        })
    monitoring = pd.DataFrame(rows).sort_values("cohort_date")
    output = MONITORING_DIR / f"as_of_date={date_text}" / "metrics.parquet"
    _write_parquet(monitoring, output)
    return {"as_of_date": date_text, "status": "monitored", "cohorts": int(len(monitoring)),
            "mature_cohorts": int(monitoring["roc_auc"].notna().sum()),
            "alerts": int(monitoring["refresh_review_required"].sum()), "path": str(output)}
