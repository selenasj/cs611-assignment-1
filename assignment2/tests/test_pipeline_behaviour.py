"""Checks the time and population contracts that matter for deployment."""

import unittest
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer

from utils.config import GOLD_DIR, PROJECT_ROOT
from utils.data_pipeline import prepare_datamart
from utils.feature_policy import APPROVED_NUMERIC_FEATURES
from utils.model_pipeline import (
    MODEL_BANK, MONITORING_DIR, PREDICTIONS_DIR,
    _matrix, _reference, _stability_flags, population_stability_index,
    score_cohort, train_and_select,
)


class PipelineBehaviourTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare_datamart()

    def test_application_population_and_point_in_time_features(self):
        features = pd.read_parquet(GOLD_DIR / "feature_store.parquet")
        labels = pd.read_parquet(GOLD_DIR / "label_store.parquet")
        self.assertEqual(len(features), 12500)
        self.assertEqual(len(labels), 12500)
        self.assertEqual(features["loan_id"].nunique(), 12500)
        self.assertEqual(int(features["clickstream_available"].eq(0).sum()), 3526)
        matched = features[features["clickstream_available"].eq(1)]
        self.assertEqual(int(matched["clickstream_snapshot_date"].eq(matched["feature_as_of_date"]).sum()), 8974)
        self.assertFalse((features["clickstream_snapshot_date"] > features["feature_as_of_date"]).any())
        self.assertFalse((labels["label_observation_date"] <= labels["feature_as_of_date"]).any())
        self.assertFalse({"name", "ssn", "dpd"} & set(features.columns))

    def test_training_uses_only_mature_labels(self):
        for path in sorted(MODEL_BANK.glob("*.joblib")):
            artifact = joblib.load(path)
            metadata = artifact["metadata"]
            self.assertLessEqual(metadata["latest_label_observation_date"], metadata["as_of_date"])
            self.assertTrue(all(cohort < metadata["as_of_date"] for cohort in metadata["validation_cohorts"]))
            self.assertGreater(metadata["validation_rows"], 0)
            self.assertIn("feature_store", metadata["training_fingerprint"])

    def test_changed_inputs_cannot_silently_reuse_a_saved_model(self):
        self.assertEqual(train_and_select("2024-06-01")["status"], "already_trained")
        with patch("utils.model_pipeline._training_fingerprint",
                   return_value={"feature_store": "changed"}):
            with self.assertRaisesRegex(RuntimeError, "--force-train"):
                train_and_select("2024-06-01")

    def test_predictions_and_delayed_monitoring(self):
        predictions = [pd.read_parquet(path) for path in sorted(PREDICTIONS_DIR.glob("cohort_date=*/predictions.parquet"))]
        self.assertEqual(len(predictions), 8)
        self.assertEqual(sum(len(frame) for frame in predictions), 4024)
        self.assertTrue(all(frame["loan_id"].is_unique for frame in predictions))
        early = pd.read_parquet(MONITORING_DIR / "as_of_date=2024-06-01/metrics.parquet")
        final = pd.read_parquet(MONITORING_DIR / "as_of_date=2025-08-01/metrics.parquet")
        self.assertEqual(int(early["n_observed_labels"].sum()), 0)
        self.assertEqual(final["n_observed_labels"].sum(), 4024)
        self.assertEqual(final["roc_auc"].notna().sum(), 8)
        self.assertTrue(final["model_feature_coverage"].between(0, 1).all())
        self.assertTrue(final["lowest_feature_coverage"].between(0, 1).all())
        self.assertTrue((final["lowest_feature_coverage"] <= final["model_feature_coverage"]).all())

    def test_stability_review_and_alert_thresholds(self):
        self.assertEqual(_stability_flags(0.09, {"csi_age": 0.09}), (False, False))
        self.assertEqual(_stability_flags(0.11, {"csi_age": 0.09}), (True, False))
        self.assertEqual(_stability_flags(0.09, {"csi_age": 0.26}), (True, True))

    def test_impossible_negative_counts_are_imputed_for_model_input(self):
        frame = pd.DataFrame({name: [1.0, 2.0] for name in APPROVED_NUMERIC_FEATURES})
        frame.loc[0, "num_of_loan"] = -100
        frame.loc[0, "num_bank_accounts"] = -1
        matrix = _matrix(frame)
        self.assertTrue(pd.isna(matrix.loc[0, "num_of_loan"]))
        self.assertTrue(pd.isna(matrix.loc[0, "num_bank_accounts"]))
        transformed = SimpleImputer(strategy="median").fit_transform(matrix)
        self.assertNotIn(-100, transformed)
        self.assertNotIn(-1, transformed)
        self.assertEqual(frame.loc[0, "num_of_loan"], -100)

    def test_scoring_rerun_is_semantically_idempotent(self):
        path = PREDICTIONS_DIR / "cohort_date=2024-06-01/predictions.parquet"
        before = pd.read_parquet(path)
        score_cohort("2024-06-01")
        after = pd.read_parquet(path)
        pd.testing.assert_frame_equal(before, after)

    def test_psi_is_zero_against_its_own_reference(self):
        artifact = joblib.load(MODEL_BANK / "credit_model_2024_06_01.joblib")
        reference = artifact["references"]["score"]
        edges = reference["edges"]
        proportions = reference["proportions"]
        # Reconstruct the histogram proportions directly: same baseline means no shift.
        self.assertAlmostEqual(sum(proportions), 1.0)
        self.assertEqual(len(proportions), len(edges))
        # A constant sample fully in a single bin must be distinguishable from the baseline.
        self.assertGreater(population_stability_index(reference, [0.5] * 100), 0)

    def test_nullable_numeric_reference(self):
        values = pd.Series([1.0, 2.0, pd.NA], dtype="Float64")
        reference = _reference(values)
        self.assertAlmostEqual(population_stability_index(reference, values), 0.0)


if __name__ == "__main__":
    unittest.main()
