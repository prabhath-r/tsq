# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest

from experiments.productive_probe_lab import LAB_VERSION, run_lab


class ProductiveProbeLabTests(unittest.TestCase):
    def test_mixed_mcq_and_productive_path_is_replayable_and_shadow_only(
        self,
    ) -> None:
        result = run_lab()

        self.assertEqual(result["lab_version"], LAB_VERSION)
        self.assertTrue(result["ok"], result["failures"])
        self.assertTrue(result["deterministic_rerun"])
        signature = result["stable_signature"]
        self.assertEqual(signature["attempt_status"], "submitted")
        self.assertEqual(signature["shadow_weight"], 0.0)
        self.assertTrue(signature["projection_unchanged"])
        self.assertTrue(signature["family_constraint"])
        self.assertEqual(signature["pending_blocker_codes"], ["pending_question"])
        self.assertTrue(signature["performance_projection_matches_replay"])
        self.assertTrue(signature["replay_ok"])
        self.assertTrue(signature["integrity_ok"])


if __name__ == "__main__":
    unittest.main()
