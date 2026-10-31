"""Behavioral checks for complete monitoring and real deployment gates."""
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from utils import governance as gov
from utils import model_pipeline as model
from utils.feature_policy import APPROVED_NUMERIC_FEATURES
from utils.model_pipeline import (MODEL_BANK, MONITORING_DIR, FEATURE_MONITORING_DIR,
    FIXED_PREDICTIONS_DIR, PREDICTIONS_DIR, FIXED_MODEL_VERSION,
    _reference, _matrix, feature_monitoring, population_stability_index, score_cohort)


class CompleteMonitoringTest(unittest.TestCase):
    def test_all_model_inputs_have_references_csi_and_missingness(self):
        artifact = joblib.load(MODEL_BANK / f"{FIXED_MODEL_VERSION}.joblib")
        final = pd.read_parquet(MONITORING_DIR / "as_of_date=2025-08-01/metrics.parquet")
        for name in APPROVED_NUMERIC_FEATURES:
            self.assertIn(name, artifact["references"])
            self.assertIn(f"csi_{name}", final)
            self.assertTrue(final[f"missing_rate_{name}"].between(0, 1).all())
        expected = final[[f"csi_{name}" for name in APPROVED_NUMERIC_FEATURES]].max(axis=1)
        np.testing.assert_allclose(final["max_csi"], expected)
        detail = pd.read_parquet(FEATURE_MONITORING_DIR / "as_of_date=2025-08-01/metrics.parquet")
        self.assertFalse(detail.duplicated(["cohort_date", "feature_name"]).any())
        self.assertEqual(set(detail.feature_name), set(APPROVED_NUMERIC_FEATURES))
        self.assertTrue(detail.groupby("cohort_date").size().eq(16).all())

    def test_fixed_model_scores_same_population_without_refresh(self):
        for current in sorted(PREDICTIONS_DIR.glob("*/predictions.parquet")):
            monthly = pd.read_parquet(current)
            fixed = pd.read_parquet(FIXED_PREDICTIONS_DIR / current.parent.name / current.name)
            self.assertEqual(set(monthly.loan_id), set(fixed.loan_id))
            self.assertEqual(set(fixed.model_version), {FIXED_MODEL_VERSION})
            self.assertEqual(set(fixed.scoring_strategy), {"fixed_june_2024"})
            if current.parent.name.endswith("2024-06-01"):
                np.testing.assert_allclose(monthly.default_probability, fixed.default_probability)

    def test_shift_in_previously_unmonitored_tenure_is_detected(self):
        base = pd.DataFrame({name: [10.0] * 100 for name in APPROVED_NUMERIC_FEATURES})
        refs = {name: _reference(base[name]) for name in APPROVED_NUMERIC_FEATURES}
        changed = base.copy()
        changed["tenure"] = 20.0
        csi, _, details = feature_monitoring(refs, changed)
        self.assertGreater(csi["csi_tenure"], 0.25)
        self.assertTrue(next(d for d in details if d["feature_name"] == "tenure")["stability_alert"])
        self.assertAlmostEqual(csi["csi_age"], 0)

    def test_missingness_and_csi_use_the_model_domain_policy(self):
        base = pd.DataFrame({name: [1.0, 2.0] for name in APPROVED_NUMERIC_FEATURES})
        base.loc[0, "num_of_loan"] = -100
        matrix = _matrix(base)
        refs = {name: _reference(matrix[name]) for name in APPROVED_NUMERIC_FEATURES}
        csi, missing, _ = feature_monitoring(refs, base)
        self.assertEqual(missing["missing_rate_num_of_loan"], 0.5)
        self.assertAlmostEqual(csi["csi_num_of_loan"], 0)
        changed = base.copy()
        changed["age"] = np.nan
        _, _, details = feature_monitoring(refs, changed)
        row = next(d for d in details if d["feature_name"] == "age")
        self.assertTrue(row["missingness_review"])
        self.assertEqual(row["missing_rate_change"], 1)

    def test_all_missing_reference_detects_newly_present_values(self):
        reference = _reference([np.nan] * 20)
        self.assertAlmostEqual(population_stability_index(reference, [np.nan] * 20), 0)
        self.assertGreater(population_stability_index(reference, [1] * 20), 0.25)

    def test_historical_approval_cannot_authorize_production_scoring(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(gov, "REGISTRY_DIR", Path(directory)):
            with self.assertRaisesRegex(RuntimeError, "No approved model"):
                score_cohort("2024-06-01", scope="production")

    def test_real_artifacts_follow_manual_gates_and_rollback(self):
        # Use real fitted models, data gates and inference, with isolated registry
        # and Gold output paths; the replay's decisions and outputs stay intact.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "gold"
            gold.mkdir()
            for name in ["feature_store.parquet", "label_store.parquet"]:
                shutil.copy2(model.GOLD_DIR / name, gold / name)
            with patch.object(gov, "REGISTRY_DIR", root / "registry"), patch.object(model, "GOLD_DIR", gold):
                june, july = FIXED_MODEL_VERSION, "credit_model_2024_07_01"
                with self.assertRaisesRegex(RuntimeError, "No approved model"):
                    score_cohort("2024-06-01", scope="production")
                for version, date in [(june, "2024-06-01"), (july, "2024-07-01")]:
                    gov.approve_model(version, date, "integration_test", "Exercise actual approval gates")
                    gov.promote_model(version, date, "integration_test", "Exercise approved release")
                before = score_cohort("2024-07-01", scope="production")
                self.assertEqual(before["model_version"], july)
                gov.rollback_model(june, "2024-08-01", "integration_test", "Restore prior approved artifact")
                restored = score_cohort("2024-08-01", scope="production")
                self.assertEqual(restored["model_version"], june)
                actual = pd.read_parquet(restored["path"])
                expected = pd.read_parquet(FIXED_PREDICTIONS_DIR / "cohort_date=2024-08-01/predictions.parquet")
                np.testing.assert_allclose(actual.default_probability, expected.default_probability)
                self.assertEqual(gov.resolve_approved_model("2024-07-01"), july)
                with self.assertRaisesRegex(RuntimeError, "immutable"):
                    model.train_and_select("2024-06-01", force=True)
                with self.assertRaisesRegex(RuntimeError, "immutable"):
                    gov.prepare_historical_rebuild()


class GovernanceControlsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bank = self.root / "models"
        self.bank.mkdir()
        self.patches = [patch.object(gov, "MODEL_BANK", self.bank),
                        patch.object(gov, "REGISTRY_DIR", self.root / "registry"),
                        patch.object(gov, "assess_candidate", return_value={
                            "promotion_eligible": True, "blocked_reasons": [], "review_required": False})]
        for item in self.patches:
            item.start()
        self.first = "credit_model_2024_06_01"
        self.second = "credit_model_2024_07_01"
        for version, date in [(self.first, "2024-06-01"), (self.second, "2024-07-01")]:
            joblib.dump({"metadata": {"as_of_date": date}}, self.bank / f"{version}.joblib")
        self.addCleanup(self.temp.cleanup)
        for item in self.patches:
            self.addCleanup(item.stop)

    def approve(self, version, date, scope="production"):
        return gov.approve_model(version, date, "coursework_reviewer", "Reviewed evidence", scope)

    def promote(self, version, date, scope="production"):
        return gov.promote_model(version, date, "coursework_operator", "Approved release", scope)

    def test_candidate_and_approval_alone_cannot_serve_predictions(self):
        gov.register_candidate(self.first, "production")
        with self.assertRaisesRegex(RuntimeError, "Only an approved"):
            self.promote(self.first, "2024-06-01")
        self.approve(self.first, "2024-06-01")
        with self.assertRaisesRegex(RuntimeError, "No approved model"):
            gov.resolve_approved_model("2024-06-01")
        self.promote(self.first, "2024-06-01")
        self.assertEqual(gov.resolve_approved_model("2024-06-01"), self.first)

    def test_validation_gate_blocks_approval_and_promotion(self):
        with patch.object(gov, "assess_candidate", return_value={
                "promotion_eligible": False, "blocked_reasons": ["psi_or_csi_at_least_0.25"],
                "review_required": True}):
            with self.assertRaisesRegex(RuntimeError, "Approval blocked"):
                self.approve(self.first, "2024-06-01")
        self.assertEqual(gov.registry_status()["models"][self.first]["status"], "candidate")
        self.approve(self.first, "2024-06-01")
        with patch.object(gov, "assess_candidate", return_value={"promotion_eligible": False}):
            with self.assertRaisesRegex(RuntimeError, "Promotion blocked"):
                self.promote(self.first, "2024-06-01")

    def test_rollback_restores_prior_version_and_keeps_as_of_history(self):
        self.approve(self.first, "2024-06-01")
        self.promote(self.first, "2024-06-01")
        self.approve(self.second, "2024-07-01")
        self.promote(self.second, "2024-07-01")
        gov.rollback_model(self.first, "2024-08-01", "operator", "Restore prior approved release")
        self.assertEqual(gov.resolve_approved_model("2024-07-01"), self.second)
        self.assertEqual(gov.resolve_approved_model("2024-08-01"), self.first)
        last = gov.registry_status()["events"][-1]
        self.assertEqual(last["action"], "rollback")
        self.assertEqual(last["reason"], "Restore prior approved release")

    def test_simulated_decisions_are_isolated_from_production(self):
        self.approve(self.first, "2024-06-01", "historical")
        self.promote(self.first, "2024-06-01", "historical")
        self.assertEqual(gov.registry_status("historical")["events"][-1]["decision_mode"], "policy_simulation")
        with self.assertRaisesRegex(RuntimeError, "No approved model"):
            gov.resolve_approved_model("2024-06-01", "production")

    def test_registered_artifact_cannot_be_silently_replaced(self):
        self.approve(self.first, "2024-06-01")
        self.promote(self.first, "2024-06-01")
        (self.bank / f"{self.first}.joblib").write_bytes(b"different artifact")
        with self.assertRaisesRegex(RuntimeError, "Artifact changed"):
            gov.resolve_approved_model("2024-06-01")

    def test_effective_dates_and_previous_deployment_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "cannot precede training"):
            self.approve(self.second, "2024-06-01")
        self.approve(self.first, "2024-06-01")
        self.promote(self.first, "2024-06-01")
        self.approve(self.second, "2024-07-01")
        with self.assertRaisesRegex(RuntimeError, "previously deployed"):
            gov.rollback_model(self.second, "2024-07-01", "operator", "Invalid rollback target")
        self.promote(self.second, "2024-07-01")
        with self.assertRaisesRegex(ValueError, "cannot rewrite earlier"):
            self.promote(self.first, "2024-06-15")

    def test_repeat_operations_do_not_duplicate_audit_events(self):
        for _ in range(2):
            self.approve(self.first, "2024-06-01")
            self.promote(self.first, "2024-06-01")
        self.assertEqual([e["action"] for e in gov.registry_status()["events"]],
                         ["register_candidate", "approve", "promote"])
        with self.assertRaisesRegex(ValueError, "reviewer and reason"):
            gov.approve_model(self.second, "2024-07-01", "", "")


if __name__ == "__main__":
    unittest.main()
