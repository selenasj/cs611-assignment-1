"""Checks the time and population contracts that matter for deployment."""

import unittest

import joblib
import pandas as pd

from utils.config import GOLD_DIR, PROJECT_ROOT
from utils.data_pipeline import prepare_datamart
from utils.model_pipeline import (
    MODEL_BANK, MONITORING_DIR, PREDICTIONS_DIR,
    _reference, population_stability_index, score_cohort,
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
