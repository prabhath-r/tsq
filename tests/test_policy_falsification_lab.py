# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest
from types import SimpleNamespace

from experiments.policy_falsification_lab import (
    ObjectiveBehaviorLearner,
    ProfileSpec,
    SessionPattern,
    build_report,
    canonical_hash,
    localization_metrics,
)
from tsq.models import SessionPhase


def _snapshot(
    mastery: float,
    families: int,
    *,
    successful: int = 0,
) -> dict[str, float | int | str]:
    return {
        "mastery_probability": mastery,
        "expected_competence": mastery,
        "uncertainty": 0.5,
        "evidence_mass": float(families),
        "observed_response_families": families,
        "successful_retrieval_families": successful,
        "delayed_retrievals": 0,
        "active_misconception_probability": 0.0,
        "state": "uncertain",
    }


class PolicyFalsificationLabTests(unittest.TestCase):
    def test_objective_behavior_keeps_wrong_and_uncertain_semantics_distinct(
        self,
    ) -> None:
        correct = SimpleNamespace(
            id="correct",
            correct=True,
            misconception_id=None,
            diagnostic_objective_id=None,
        )
        misconception = SimpleNamespace(
            id="misconception",
            correct=False,
            misconception_id="m_named",
            diagnostic_objective_id="lo_target",
        )
        question = SimpleNamespace(
            id="q_target",
            objective_id="lo_target",
            options=(correct, misconception),
            correct_option=correct,
        )
        presentation = SimpleNamespace(
            question=question,
            phase=SessionPhase.LEARN,
        )

        wrong = ObjectiveBehaviorLearner(
            name="wrong",
            pattern=SessionPattern(
                objective_modes=(("lo_target", "confident_wrong"),)
            ),
        ).answer(
            presentation,
            simulation_seed=1,
            trial_index=0,
            encounter=1,
        )
        uncertain = ObjectiveBehaviorLearner(
            name="uncertain",
            pattern=SessionPattern(
                objective_modes=(("lo_target", "uncertain_abstain"),)
            ),
        ).answer(
            presentation,
            simulation_seed=1,
            trial_index=0,
            encounter=1,
        )
        support = ObjectiveBehaviorLearner(
            name="support",
            pattern=SessionPattern(
                objective_modes=(
                    ("lo_target", "wrong_main_correct_support"),
                )
            ),
        ).answer(
            SimpleNamespace(
                question=question,
                phase=SessionPhase.VERIFY,
            ),
            simulation_seed=1,
            trial_index=0,
            encounter=1,
        )

        self.assertFalse(wrong.correct)
        self.assertEqual(wrong.selected_option_id, "misconception")
        self.assertEqual(wrong.confidence, 0.95)
        self.assertIsNone(uncertain.selected_option_id)
        self.assertEqual(uncertain.confidence, 0.20)
        self.assertTrue(support.correct)
        self.assertEqual(support.response_ms, 45_000)

    def test_localization_metric_distinguishes_failure_from_missing_exposure(
        self,
    ) -> None:
        baseline = {
            "lo_weak": _snapshot(0.20, 0),
            "lo_strong_a": _snapshot(0.20, 0),
            "lo_strong_b": _snapshot(0.20, 0),
            "lo_control_thin": _snapshot(0.20, 0),
        }
        separated = {
            "lo_weak": _snapshot(0.10, 2),
            "lo_strong_a": _snapshot(0.40, 2),
            "lo_strong_b": _snapshot(0.35, 2),
            "lo_control_thin": _snapshot(0.30, 1),
        }
        supported = localization_metrics(
            (separated,),
            baseline=baseline,
            target_objective_ids=("lo_weak",),
        )
        missing = localization_metrics(
            (
                {
                    **separated,
                    "lo_weak": _snapshot(0.10, 1),
                },
            ),
            baseline=baseline,
            target_objective_ids=("lo_weak",),
        )

        self.assertTrue(supported["criterion_passed"])
        self.assertEqual(supported["target_ranks_lowest_first"]["lo_weak"], 1)
        self.assertAlmostEqual(
            supported["median_strong_minus_target_mastery"], 0.275
        )
        self.assertEqual(
            supported["underpowered_control_objective_ids"],
            ["lo_control_thin"],
        )
        self.assertFalse(missing["sufficient_exposure"])
        self.assertFalse(missing["criterion_passed"])

    def test_localization_metric_requires_symmetric_control_exposure(
        self,
    ) -> None:
        baseline = {
            "lo_weak": _snapshot(0.20, 0),
            "lo_control_a": _snapshot(0.20, 0),
            "lo_control_b": _snapshot(0.20, 0),
            "lo_cold": _snapshot(0.20, 0),
        }
        underpowered = {
            "lo_weak": _snapshot(0.10, 2),
            "lo_control_a": _snapshot(0.40, 2),
            "lo_control_b": _snapshot(0.35, 1),
            # A tiny cold-state numerical difference must not affect rank.
            "lo_cold": _snapshot(0.099999, 0),
        }

        result = localization_metrics(
            (underpowered,),
            baseline=baseline,
            target_objective_ids=("lo_weak",),
        )

        self.assertFalse(result["sufficient_exposure"])
        self.assertFalse(result["criterion_passed"])
        self.assertEqual(
            result["underpowered_control_objective_ids"],
            ["lo_control_b"],
        )
        self.assertEqual(
            result["observed_strong_objective_ids"], ["lo_control_a"]
        )
        self.assertEqual(result["target_ranks_lowest_first"], {"lo_weak": 1})
        self.assertAlmostEqual(
            result["median_strong_minus_target_mastery"], 0.30
        )

    def test_real_engine_profile_is_disposable_replayable_and_deterministic(
        self,
    ) -> None:
        repair = SessionPattern(
            objective_modes=(
                ("lo_causal_visibility", "wrong_main_correct_support"),
            )
        )
        spec = ProfileSpec(
            id="test_localized_repair",
            hypothesis="Production routing resolves only after fresh verification.",
            sessions=(repair, repair),
            target_objective_ids=("lo_causal_visibility",),
            require_independent_verification=True,
        )

        first = build_report(
            specs=(spec,),
            steps_per_session=4,
            seed=8_119,
        )
        second = build_report(
            specs=(spec,),
            steps_per_session=4,
            seed=8_119,
        )

        self.assertEqual(first["artifact_sha256"], second["artifact_sha256"])
        self.assertEqual(
            first["artifact_sha256"],
            canonical_hash(
                {
                    key: value
                    for key, value in first.items()
                    if key != "artifact_sha256"
                }
            ),
        )
        profile = first["profiles"][0]
        self.assertTrue(profile["infrastructure_ok"])
        self.assertFalse(profile["blockers"])
        self.assertTrue(profile["projection_replay"]["ok"])
        self.assertTrue(
            profile["remediation_and_verification"][
                "verification_contract_passed"
            ]
        )
        self.assertEqual(
            profile["remediation_and_verification"][
                "resolved_episode_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
