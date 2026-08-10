# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.objective_discovery_lab import (
    LAB_VERSION,
    ObjectivePatternLearner,
    bounded_family_recovery_probe,
    build_report,
    discovery_metrics,
    persistent_gap_episode_audit,
    schedule_regime,
)


class ObjectiveDiscoveryLabTests(unittest.TestCase):
    def test_persistent_gap_episode_audit_requires_an_interleaved_family(self) -> None:
        valid = {
            "index": 1,
            "trace": [
                {"objective_id": "lo_gap"},
                {"objective_id": "lo_other"},
                {"objective_id": "lo_gap"},
            ],
            "persistent_gap_episode_spends": [
                {
                    "objective_id": "lo_gap",
                    "family_id": "family_a",
                    "response_index": 1,
                    "spend": 1,
                    "budget": 2,
                },
                {
                    "objective_id": "lo_gap",
                    "family_id": "family_b",
                    "response_index": 3,
                    "spend": 2,
                    "budget": 2,
                },
            ],
        }
        result = persistent_gap_episode_audit((valid,))
        self.assertTrue(result["all_contracts_passed"])
        self.assertEqual(result["maximum_spends_per_session_objective"], 2)

        adjacent = {
            **valid,
            "trace": [
                {"objective_id": "lo_gap"},
                {"objective_id": "lo_gap"},
            ],
            "persistent_gap_episode_spends": [
                valid["persistent_gap_episode_spends"][0],
                {
                    **valid["persistent_gap_episode_spends"][1],
                    "response_index": 2,
                },
            ],
        }
        failed = persistent_gap_episode_audit((adjacent,))
        self.assertFalse(failed["all_contracts_passed"])
        self.assertEqual(len(failed["violations"]), 1)

    def test_bounded_family_recovery_probe_covers_every_released_family(self) -> None:
        result = bounded_family_recovery_probe(
            objective_id="lo_attention_logit_scaling",
            max_correct_retests=2,
        )

        self.assertTrue(result["all_families_recovered"], result["failed_cases"])
        self.assertGreater(result["case_count"], 1)
        self.assertLess(result["case_count"], result["eligible_question_count"])
        self.assertFalse(result["failed_cases"])
        self.assertEqual(
            len({case["family_id"] for case in result["cases"]}),
            result["case_count"],
        )
        self.assertEqual(
            sum(case["member_count"] for case in result["cases"]),
            result["eligible_question_count"],
        )
        for case in result["cases"]:
            self.assertEqual(
                case["question_id"], case["representative_question_id"]
            )
            self.assertIn(
                case["representative_question_id"],
                case["member_question_ids"],
            )
            self.assertLess(case["after_wrong_mastery"], case["prior_mastery"])
            self.assertIsNotNone(case["recovered_after_correct_retests"])
            self.assertLessEqual(case["recovered_after_correct_retests"], 2)
            self.assertGreaterEqual(case["recovery_fraction"], 0.75)

        cases_by_family = {
            case["family_id"]: case for case in result["cases"]
        }
        variance = cases_by_family["f_attention_scaling_variance"]
        self.assertEqual(
            variance["representative_question_id"],
            "q_attention_scaling_variance_001",
        )
        self.assertIn(
            "q_attention_scaling_head_dimension_comparison_001",
            variance["member_question_ids"],
        )
        self.assertIn(
            "q_attention_scaling_variance_warmup_001",
            variance["member_question_ids"],
        )
        self.assertIn(
            "f_attention_scaled_variance_nonunit",
            variance["published_family_ids"],
        )
        rank = cases_by_family["f_attention_scaling_rank"]
        self.assertEqual(
            rank["representative_question_id"],
            "q_attention_scaling_rank_001",
        )
        self.assertIn(
            "q_attention_scaling_rank_warmup_001",
            rank["member_question_ids"],
        )

        deliberately_too_short = bounded_family_recovery_probe(
            objective_id="lo_attention_logit_scaling",
            max_correct_retests=1,
        )
        self.assertFalse(deliberately_too_short["all_families_recovered"])
        self.assertTrue(deliberately_too_short["failed_cases"])

    def test_schedule_regimes_are_explicit_and_monotone(self) -> None:
        self.assertEqual(
            schedule_regime((True, True)), ("weak_control", None)
        )
        self.assertEqual(
            schedule_regime((False, False)), ("strong_control", None)
        )
        self.assertEqual(
            schedule_regime((True, True, False, False)),
            ("weak_to_strong_recovery", 2),
        )
        with self.assertRaises(ValueError):
            schedule_regime(())
        with self.assertRaises(ValueError):
            schedule_regime((True, False, True))

    def test_discovery_metric_requires_exposure_rank_and_separation(self) -> None:
        snapshots = (
            {
                "lo_weak": {
                    "mastery_probability": 0.18,
                    "observed_response_families": 1,
                },
                "lo_strong_a": {
                    "mastery_probability": 0.30,
                    "observed_response_families": 1,
                },
                "lo_strong_b": {
                    "mastery_probability": 0.32,
                    "observed_response_families": 1,
                },
            },
            {
                "lo_weak": {
                    "mastery_probability": 0.10,
                    "observed_response_families": 2,
                },
                "lo_strong_a": {
                    "mastery_probability": 0.40,
                    "observed_response_families": 2,
                },
                "lo_strong_b": {
                    "mastery_probability": 0.35,
                    "observed_response_families": 1,
                },
            },
        )

        result = discovery_metrics(
            snapshots,
            target_objective_id="lo_weak",
            baseline_snapshot={
                "lo_weak": {"mastery_probability": 0.20},
                "lo_strong_a": {"mastery_probability": 0.20},
                "lo_strong_b": {"mastery_probability": 0.20},
            },
        )

        self.assertTrue(result["hypothesis_passed"])
        self.assertEqual(result["target_rank_lowest_first"], 1)
        self.assertEqual(result["first_detected_session"], 2)
        self.assertAlmostEqual(result["separation"], 0.275)
        self.assertTrue(result["specificity_passed"])
        self.assertEqual(result["non_target_regressions"], [])

    def test_diagnostic_distractor_does_not_redefine_primary_weakness(self) -> None:
        correct = SimpleNamespace(
            id="correct",
            correct=True,
            misconception_id=None,
            diagnostic_objective_id=None,
        )
        diagnostic = SimpleNamespace(
            id="diagnostic",
            correct=False,
            misconception_id="m_target",
            diagnostic_objective_id="lo_target",
        )
        question = SimpleNamespace(
            id="q_cross_objective",
            objective_id="lo_other",
            options=(correct, diagnostic),
            correct_option=correct,
        )
        learner = ObjectivePatternLearner(
            name="isolated",
            target_objective_id="lo_target",
            target_is_weak=True,
        )

        answer = learner.answer(
            SimpleNamespace(question=question),
            simulation_seed=1,
            trial_index=0,
            encounter=0,
        )

        self.assertTrue(answer.correct)
        self.assertEqual(answer.selected_option_id, "correct")

    def test_real_engine_localizes_one_stark_objective_gap(self) -> None:
        report = build_report(
            targets=("lo_causal_visibility",),
            sessions=2,
            steps_per_session=4,
            seed=719,
            include_recovery=False,
        )

        self.assertEqual(report["lab_version"], "objective-discovery-lab-v6")
        self.assertEqual(report["lab_version"], LAB_VERSION)
        self.assertEqual(report["findings"]["case_count"], 1)
        self.assertEqual(
            report["findings"]["cases_passing_discovery_hypothesis"], 1
        )
        self.assertTrue(
            report["findings"]["all_integrity_and_replay_checks_passed"]
        )
        self.assertTrue(
            report["findings"]["bounded_family_recovery_observed"]
        )
        self.assertTrue(
            report["findings"][
                "all_persistent_gap_episode_contracts_passed"
            ]
        )
        self.assertFalse(
            report["findings"]["certificate_inflation_observed"]
        )
        case = report["cases"][0]
        self.assertTrue(case["discovery"]["hypothesis_passed"])
        self.assertTrue(case["projection_replay"]["ok"])
        self.assertTrue(case["integrity"]["ok"])


if __name__ == "__main__":
    unittest.main()
