# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest

from experiments.scoring_reconciliation_lab import LAB_VERSION, run_lab


class ScoringReconciliationLabTests(unittest.TestCase):
    def test_stranded_callbacks_reconcile_without_retry_or_mastery(
        self,
    ) -> None:
        result = run_lab()

        self.assertEqual(result["lab_version"], LAB_VERSION)
        self.assertTrue(result["ok"], result["failures"])
        self.assertTrue(result["deterministic_rerun"])
        signature = result["stable_signature"]
        self.assertEqual(signature["unknown_status"], "unknown")
        self.assertFalse(signature["unknown_terminal"])
        self.assertEqual(signature["repeated_unknown_count"], 2)
        self.assertEqual(signature["unknown_lookup_calls"], 2)
        self.assertEqual(signature["completed_status"], "completed")
        self.assertTrue(signature["completed_terminal"])
        self.assertEqual(
            signature["absent_status"],
            "definitely_absent",
        )
        self.assertTrue(signature["absent_terminal"])
        self.assertEqual(signature["completed_provider_calls"], 1)
        self.assertEqual(signature["absent_provider_calls"], 1)
        self.assertEqual(signature["completed_lookup_calls"], 1)
        self.assertEqual(signature["absent_lookup_calls"], 1)
        self.assertTrue(signature["unknown_projection_unchanged"])
        self.assertTrue(signature["recovery_projection_unchanged"])
        self.assertTrue(signature["absence_projection_unchanged"])
        self.assertEqual(signature["completed_evaluation_count"], 1)
        self.assertEqual(signature["absent_evaluation_count"], 0)
        self.assertEqual(signature["recovered_shadow_weight"], 0.0)
        self.assertTrue(signature["recovery_events_session_null"])
        self.assertTrue(signature["automatic_retry_disabled"])
        self.assertTrue(signature["projection_disabled"])
        self.assertTrue(signature["certification_disabled"])
        self.assertTrue(signature["skill_authority_disabled"])
        self.assertTrue(signature["observational_only"])
        self.assertEqual(
            signature["event_counts"],
            {
                "PerformanceScoringClaimed": 2,
                "PerformanceScoringReconciled": 4,
                "ShadowEvidenceReduced": 1,
                "TaskEvaluationRecorded": 1,
            },
        )
        self.assertTrue(signature["completed_replay_ok"])
        self.assertTrue(signature["absent_replay_ok"])
        self.assertTrue(signature["performance_replay_ok"])
        self.assertTrue(signature["integrity_ok"])


if __name__ == "__main__":
    unittest.main()
