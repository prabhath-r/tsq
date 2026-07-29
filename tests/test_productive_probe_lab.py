# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest

from experiments.productive_probe_lab import LAB_VERSION, run_lab


class ProductiveProbeLabTests(unittest.TestCase):
    def test_synthetic_probe_ranking_is_quarantined_and_non_authoritative(
        self,
    ) -> None:
        result = run_lab()

        self.assertEqual(result["lab_version"], LAB_VERSION)
        self.assertTrue(result["ok"], result["failures"])
        self.assertTrue(result["deterministic_rerun"])
        signature = result["stable_signature"]
        self.assertEqual(
            signature["attempt_status"], "not_started_quarantined"
        )
        self.assertEqual(signature["shadow_weight"], 0.0)
        self.assertTrue(signature["projection_unchanged"])
        self.assertTrue(signature["family_constraint"])
        self.assertEqual(signature["normal_task_ids"], [])
        self.assertTrue(signature["normal_start_rejected"])
        self.assertFalse(signature["synthetic_startable"])
        self.assertEqual(
            signature["synthetic_blocker_codes"],
            ["synthetic_quarantine"],
        )
        self.assertEqual(
            signature["synthetic_selection_scope"],
            "synthetic_quarantined_lab",
        )
        self.assertEqual(
            signature["release_declaration_kind"], "synthetic_lab"
        )
        self.assertFalse(signature["release_human_reviewed"])
        self.assertFalse(signature["release_activation_authority"])
        self.assertEqual(
            signature["performance_counts"],
            {"attempts": 0, "bundles": 0, "evaluations": 0, "events": 0},
        )
        self.assertEqual(signature["session_productive_attempts"], 0)
        self.assertEqual(signature["pending_blocker_codes"], ["pending_question"])
        self.assertTrue(signature["artifact_digest_matches_submission"])
        self.assertTrue(signature["artifact_digest_matches_bytes"])
        self.assertTrue(signature["artifact_private_material_absent"])
        self.assertFalse(signature["artifact_executed"])
        self.assertTrue(signature["performance_projection_matches_replay"])
        self.assertTrue(signature["replay_ok"])
        self.assertTrue(signature["integrity_ok"])


if __name__ == "__main__":
    unittest.main()
