# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest

from experiments.policy_shadow_comparison_lab import (
    LAB_VERSION,
    oracle_values,
    run_lab,
)


class PolicyShadowComparisonLabTests(unittest.TestCase):
    def test_oracle_values_match_a_hand_calculated_frontier(self) -> None:
        frontier = [
            {"question_id": "q_a", "logging_probability": 0.75},
            {"question_id": "q_b", "logging_probability": 0.25},
        ]

        result = oracle_values(
            frontier,
            live_question_id="q_b",
            challenger_question_id="q_a",
            probabilities={"q_a": 0.20, "q_b": 0.80},
        )

        self.assertAlmostEqual(result["behavior"], 0.35)
        self.assertAlmostEqual(result["uniform"], 0.50)
        self.assertAlmostEqual(result["greedy"], 0.20)
        self.assertAlmostEqual(result["live_action"], 0.80)

    def test_oracle_values_reject_missing_support(self) -> None:
        frontier = [
            {"question_id": "q_a", "logging_probability": 1.0},
        ]
        with self.assertRaisesRegex(ValueError, "challenger"):
            oracle_values(
                frontier,
                live_question_id="q_a",
                challenger_question_id="q_b",
                probabilities={"q_a": 0.5},
            )

    def test_real_engine_lab_is_replayable_read_only_and_deterministic(
        self,
    ) -> None:
        result = run_lab(trials_per_profile=1, replicate=True)

        self.assertEqual(result["lab_version"], LAB_VERSION)
        self.assertTrue(result["ok"], result["failures"])
        self.assertTrue(result["deterministic_rerun"])
        signature = result["stable_signature"]
        self.assertEqual(signature["observation_count"], 6)
        self.assertEqual(
            signature["phase_counts"],
            {"diagnose": 2, "learn": 2, "review": 2},
        )
        self.assertTrue(signature["invariants"]["integrity_ok"])
        self.assertTrue(
            signature["invariants"]["production_arithmetic_matches"]
        )
        self.assertTrue(signature["invariants"]["replay_ok"])
        self.assertTrue(
            signature["invariants"]["source_database_unchanged_by_analysis"]
        )
        self.assertTrue(
            signature["invariants"]["one_decision_per_fresh_learner"]
        )
        self.assertTrue(signature["invariants"]["all_profiles_observed"])
        self.assertTrue(
            signature["invariants"]["all_declared_phases_observed"]
        )
        self.assertEqual(
            sorted(signature["stratified"]["phases"]),
            ["diagnose", "learn", "review"],
        )
        self.assertEqual(
            signature["assessments"]["uniform_safe_frontier_ips"][
                "assessment"
            ],
            "inconclusive",
        )

    def test_adequate_uniform_run_recovers_synthetic_oracle_within_bound(
        self,
    ) -> None:
        result = run_lab(trials_per_profile=6, replicate=False)

        self.assertTrue(result["ok"], result["failures"])
        assessment = result["stable_signature"]["assessments"][
            "uniform_safe_frontier_ips"
        ]
        self.assertTrue(assessment["adequate_information"])
        self.assertEqual(
            assessment["assessment"],
            "not_falsified_within_predeclared_bound",
        )
        findings = result["stable_signature"]["findings"]
        self.assertEqual(
            len(findings["underpowered_greedy_profile_ids"]), 6
        )
        self.assertEqual(
            findings["underpowered_greedy_phases"],
            ["diagnose", "learn", "review"],
        )


if __name__ == "__main__":
    unittest.main()
