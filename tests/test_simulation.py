# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import MAX_REMEDIATION_DEPTH, AdaptiveEngine
from tsq.simulation import BehavioralSimulator, SyntheticLearner
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2100, 2, 3, 10, 0, tzinfo=timezone.utc)


def make_simulator(directory: str, filename: str = "simulation.db") -> BehavioralSimulator:
    database = Database(Path(directory) / filename)
    database.initialize()
    database.import_corpus(*read_and_parse(CORPUS))
    return BehavioralSimulator(AdaptiveEngine(database))


class BehavioralSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.simulator = make_simulator(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def always_wrong() -> SyntheticLearner:
        return SyntheticLearner(
            "always-wrong",
            default_ability=0.0,
            slip_probability=1.0,
            guess_probability=0.0,
            seed=73,
        )

    def test_replay_is_deterministic_for_profile_clock_and_seed(self) -> None:
        profile = SyntheticLearner(
            "intermediate",
            default_ability=0.58,
            concept_abilities={
                "c_clustering": 0.46,
                "c_feature_scaling": 0.72,
            },
            misconception_strengths={
                "m_internal_metric_proves_true_clusters": 0.85,
                "m_feature_scale_does_not_affect_kmeans": 0.60,
            },
            slip_probability=0.03,
            guess_probability=0.01,
            seed=991,
        )
        first = self.simulator.run(
            profile,
            learner_id="replay-a",
            root_concept_id="c_clustering",
            policy_seed=31,
            max_steps=10,
            start_at=START,
        )

        # A fresh engine proves determinism does not come from reusing a pending
        # decision or learner projection. Random database IDs are intentionally
        # absent from the behavior signature.
        second_simulator = make_simulator(self.tempdir.name, "simulation-2.db")
        second = second_simulator.run(
            profile,
            learner_id="replay-b",
            root_concept_id="c_clustering",
            policy_seed=31,
            max_steps=10,
            start_at=START,
        )

        self.assertGreater(first.attempted, 0)
        self.assertEqual(first.behavior_signature(), second.behavior_signature())
        self.assertEqual(first.summary()["calibration"], second.summary()["calibration"])

    def test_remediation_episode_never_reuses_trigger_item_or_family(self) -> None:
        report = self.simulator.run(
            self.always_wrong(),
            learner_id="repeat-audit",
            root_concept_id="c_clustering",
            policy_seed=11,
            max_steps=12,
            start_at=START,
        )

        self.assertTrue(report.focus_episodes, report.summary())
        self.assertEqual(report.remediation_exact_repeat_count, 0, report.summary())
        self.assertEqual(report.remediation_family_repeat_count, 0, report.summary())
        for episode in report.focus_episodes:
            self.assertNotIn(episode.trigger_question_id, episode.question_ids)
            self.assertNotIn(episode.trigger_family_id, episode.family_ids)
            self.assertEqual(len(episode.question_ids), len(set(episode.question_ids)))
            self.assertEqual(len(episode.family_ids), len(set(episode.family_ids)))

    def test_persistent_failure_exits_a_bounded_tunnel(self) -> None:
        report = self.simulator.run(
            self.always_wrong(),
            learner_id="failure-bound",
            root_concept_id="c_clustering",
            policy_seed=11,
            max_steps=12,
            start_at=START,
        )
        bounded_exits = [
            episode
            for episode in report.focus_episodes
            if episode.outcome == "bounded_failure_exit"
        ]

        self.assertTrue(bounded_exits, report.summary())
        self.assertTrue(
            all(episode.length < MAX_REMEDIATION_DEPTH for episode in bounded_exits),
            report.summary(),
        )
        # If the next episode hits a real corpus hole, it must be explicit in the
        # report rather than silently selecting an unrelated fallback item.
        self.assertEqual(report.has_blockers, bool(report.summary()["blockers"]))

    def test_stronger_ground_truth_learners_outperform_over_paired_trials(self) -> None:
        strong = SyntheticLearner(
            "strong",
            default_ability=0.995,
            slip_probability=0.0,
            guess_probability=0.0,
            seed=404,
        )
        weak = SyntheticLearner(
            "weak",
            default_ability=0.005,
            slip_probability=1.0,
            guess_probability=0.0,
            seed=404,
        )
        seeds = tuple(range(20, 30))
        strong_report = self.simulator.evaluate(
            strong,
            learner_id_prefix="paired-strong",
            root_concept_id="c_clustering",
            policy_seeds=seeds,
            max_steps=8,
            start_at=START,
        )
        weak_report = self.simulator.evaluate(
            weak,
            learner_id_prefix="paired-weak",
            root_concept_id="c_clustering",
            policy_seeds=seeds,
            max_steps=8,
            start_at=START,
        )

        self.assertEqual(len(strong_report.trials), len(seeds))
        self.assertEqual(len(weak_report.trials), len(seeds))
        self.assertGreater(strong_report.attempted, 30)
        self.assertGreater(weak_report.attempted, 30)
        self.assertGreater(strong_report.accuracy, weak_report.accuracy + 0.60)
        self.assertEqual(strong_report.calibration.count, strong_report.attempted)
        self.assertEqual(weak_report.calibration.count, weak_report.attempted)

    def test_abstaining_low_confidence_profile_reaches_true_uncertainty_path(self) -> None:
        report = self.simulator.run(
            SyntheticLearner(
                "uncertain",
                default_ability=0.55,
                abstain_probability=1.0,
                confidence_override=0.20,
                seed=44,
            ),
            learner_id="uncertain-path",
            root_concept_id="c_ai_learning_systems",
            policy_seed=44,
            max_steps=8,
            start_at=START,
        )

        self.assertEqual(report.attempted, 8, report.summary())
        self.assertTrue(all(step.selected_option_id is None for step in report.steps))
        self.assertTrue(all(step.confidence == 0.20 for step in report.steps))
        self.assertEqual(report.summary()["answer_patterns"]["abstained"], 8)
        self.assertEqual(report.summary()["answer_patterns"]["low_confidence"], 8)
        self.assertTrue(self.simulator.engine.database.verify_integrity()["ok"])

    def test_fast_and_slow_correct_answers_follow_different_evidence_paths(self) -> None:
        fast_report = self.simulator.run(
            SyntheticLearner(
                "fast-correct",
                default_ability=0.9,
                forced_correctness=True,
                confidence_override=0.95,
                base_response_ms=120,
                seed=71,
            ),
            learner_id="fast-correct",
            root_concept_id="c_feature_scaling",
            policy_seed=71,
            max_steps=1,
            start_at=START,
        )
        slow_simulator = make_simulator(self.tempdir.name, "slow.db")
        slow_report = slow_simulator.run(
            SyntheticLearner(
                "slow-correct",
                default_ability=0.9,
                forced_correctness=True,
                confidence_override=0.95,
                base_response_ms=12_000,
                seed=71,
            ),
            learner_id="slow-correct",
            root_concept_id="c_feature_scaling",
            policy_seed=71,
            max_steps=1,
            start_at=START,
        )
        self.assertEqual(
            fast_report.steps[0].question_id, slow_report.steps[0].question_id
        )
        self.assertLess(fast_report.steps[0].response_ms, 250)
        self.assertGreaterEqual(slow_report.steps[0].response_ms, 500)

        def evidence(database, learner_id):
            with database.read() as connection:
                mass = connection.execute(
                    """SELECT SUM(evidence_mass) AS mass FROM skill_states
                       WHERE learner_id=?""",
                    (learner_id,),
                ).fetchone()["mass"]
                families = connection.execute(
                    """SELECT COUNT(*) AS n FROM learner_skill_families
                       WHERE learner_id=?""",
                    (learner_id,),
                ).fetchone()["n"]
            return mass, families

        fast_mass, fast_families = evidence(
            self.simulator.engine.database, "fast-correct"
        )
        slow_mass, slow_families = evidence(
            slow_simulator.engine.database, "slow-correct"
        )
        self.assertLess(fast_mass, slow_mass)
        self.assertEqual(fast_families, 0)
        self.assertEqual(slow_families, 1)


if __name__ == "__main__":
    unittest.main()
