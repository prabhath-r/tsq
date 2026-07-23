# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import MAX_REMEDIATION_DEPTH, AdaptiveEngine
from tsq.errors import ValidationError
from tsq.learner import CONCEPT_MODEL_VERSION, LearnerModel
from tsq.simulation import (
    BehavioralSimulator,
    SyntheticAnswer,
    SyntheticLearner,
    evidence_anchor_concept_id,
)
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2100, 2, 3, 10, 0, tzinfo=timezone.utc)


def make_simulator(directory: str, filename: str = "simulation.db") -> BehavioralSimulator:
    database = Database(Path(directory) / filename)
    database.initialize()
    database.import_corpus(*read_and_parse(CORPUS, include_catalog=True))
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
        with self.simulator.engine.database.read() as connection:
            stored_session = connection.execute(
                "SELECT created_at FROM sessions WHERE learner_id='replay-a'"
            ).fetchone()
            started_event = connection.execute(
                """SELECT occurred_at FROM events
                   WHERE learner_id='replay-a' AND event_type='SessionStarted'"""
            ).fetchone()
        self.assertEqual(stored_session["created_at"], START.isoformat())
        self.assertEqual(started_event["occurred_at"], START.isoformat())

        objective_tamper = replace(
            first,
            steps=(
                replace(
                    first.steps[0],
                    learning_objective_id="lo_signature_tamper",
                    focus_objective_after="lo_focus_tamper",
                ),
                *first.steps[1:],
            ),
        )
        self.assertNotEqual(
            first.behavior_signature(), objective_tamper.behavior_signature()
        )

        missing_confidence = replace(
            first,
            steps=(
                replace(first.steps[0], confidence=None),
                *first.steps[1:],
            ),
        )
        self.assertNotEqual(
            first.behavior_signature(), missing_confidence.behavior_signature()
        )
        self.assertEqual(
            missing_confidence.summary()["answer_patterns"][
                "missing_confidence"
            ],
            1,
        )

    def test_objective_ability_replaces_surface_concept_for_objective_items(self) -> None:
        database = self.simulator.engine.database
        release_id = database.get_active_release_id()
        question = database.get_question(
            "q_transformer_mask_direction_001", release_id=release_id
        )
        self.assertEqual(question.primary_concept_id, "c_transformers")
        self.assertEqual(
            evidence_anchor_concept_id(question), "c_causal_masking"
        )

        weak_objective = SyntheticLearner(
            "weak-objective",
            concept_abilities={"c_transformers": 0.99},
            objective_abilities={"lo_causal_visibility": 0.01},
            slip_probability=0.0,
            guess_probability=0.0,
        )
        strong_objective = SyntheticLearner(
            "strong-objective",
            concept_abilities={"c_transformers": 0.01},
            objective_abilities={"lo_causal_visibility": 0.99},
            slip_probability=0.0,
            guess_probability=0.0,
        )

        self.assertLess(
            weak_objective.probability_correct(question),
            strong_objective.probability_correct(question) - 0.50,
        )

        with database.read() as connection:
            legacy_id = connection.execute(
                """SELECT question.id
                   FROM release_questions membership
                   JOIN questions question ON question.id = membership.question_id
                   LEFT JOIN release_question_objectives objective
                     ON objective.release_id = membership.release_id
                    AND objective.question_id = question.id
                   WHERE membership.release_id = ?
                     AND objective.objective_id IS NULL
                   ORDER BY question.id LIMIT 1""",
                (release_id,),
            ).fetchone()["id"]
        legacy = database.get_question(legacy_id, release_id=release_id)
        baseline = SyntheticLearner(
            "legacy-baseline",
            default_ability=0.61,
            slip_probability=0.0,
            guess_probability=0.0,
        )
        irrelevant_objective = SyntheticLearner(
            "legacy-objective-map",
            default_ability=0.61,
            objective_abilities={"lo_causal_visibility": 0.01},
            slip_probability=0.0,
            guess_probability=0.0,
        )
        self.assertEqual(
            baseline.probability_correct(legacy),
            irrelevant_objective.probability_correct(legacy),
        )

    def test_misspecified_generators_are_explicit_and_distinct(self) -> None:
        database = self.simulator.engine.database
        question = database.get_question(
            "q_transformer_mask_direction_001",
            release_id=database.get_active_release_id(),
        )
        common = {
            "default_objective_ability": 0.50,
            "slip_probability": 0.0,
            "guess_probability": 0.0,
        }
        smooth = SyntheticLearner("smooth", **common)
        threshold = SyntheticLearner(
            "threshold", **common, response_model="discontinuous_threshold"
        )
        ability_only = SyntheticLearner(
            "ability-only", **common, response_model="ability_only"
        )

        self.assertNotEqual(
            smooth.probability_correct(question),
            threshold.probability_correct(question),
        )
        self.assertEqual(ability_only.probability_correct(question), 0.50)
        with self.assertRaisesRegex(ValidationError, "response model"):
            SyntheticLearner("invalid", response_model="engine_clone_v99")

    def test_objective_coverage_uses_evidence_anchors_and_family_pairs(self) -> None:
        report = self.simulator.run(
            SyntheticLearner(
                "coverage",
                default_objective_ability=0.70,
                forced_correctness=True,
                confidence_override=0.90,
                seed=18,
            ),
            learner_id="objective-coverage",
            root_concept_id="t_transformers",
            policy_seed=18,
            max_steps=4,
            start_at=START,
        )

        self.assertEqual(report.coverage.scope_objectives, 8)
        self.assertEqual(report.coverage.eligible_objectives, 8)
        self.assertGreater(
            report.coverage.eligible_evidence_families, 0
        )
        self.assertEqual(
            set(report.coverage.observed_objectives),
            {
                step.learning_objective_id
                for step in report.steps
                if step.learning_objective_id is not None
            },
        )
        release_id = self.simulator.engine.database.get_active_release_id()
        for step in report.steps:
            question = self.simulator.engine.database.get_question(
                step.question_id, release_id=release_id
            )
            self.assertEqual(
                step.surface_primary_concept_id, question.primary_concept_id
            )
            self.assertEqual(
                step.evidence_anchor_concept_id,
                evidence_anchor_concept_id(question),
            )

    def test_idempotent_retries_across_two_sessions_do_not_duplicate_state(self) -> None:
        profile = SyntheticLearner(
            "longitudinal",
            default_objective_ability=0.85,
            forced_correctness=True,
            confidence_override=0.90,
            base_response_ms=2_000,
            seed=303,
        )
        first = self.simulator.run(
            profile,
            learner_id="longitudinal",
            root_concept_id="t_transformers",
            policy_seed=303,
            max_steps=2,
            start_at=START,
            verify_idempotency=True,
        )
        second = self.simulator.run(
            profile,
            learner_id="longitudinal",
            root_concept_id="t_transformers",
            policy_seed=304,
            max_steps=2,
            start_at=first.ended_at + timedelta(days=7),
            trial_index=1,
            require_fresh_learner=False,
            verify_idempotency=True,
        )

        self.assertEqual(first.idempotent_retries_verified, first.attempted)
        self.assertEqual(second.idempotent_retries_verified, second.attempted)
        expected = first.attempted + second.attempted
        with self.simulator.engine.database.read() as connection:
            counts = connection.execute(
                """SELECT learner.revision,
                          (SELECT COUNT(*) FROM sessions
                           WHERE learner_id=learner.id) AS sessions,
                          (SELECT COUNT(*) FROM attempts
                           WHERE learner_id=learner.id) AS attempts,
                          (SELECT COUNT(*) FROM events
                           WHERE learner_id=learner.id
                             AND event_type='ResponseSubmitted') AS responses
                   FROM learners learner WHERE learner.id='longitudinal'"""
            ).fetchone()
        self.assertEqual(counts["sessions"], 2)
        self.assertEqual(counts["attempts"], expected)
        self.assertEqual(counts["responses"], expected)
        self.assertEqual(counts["revision"], expected)
        self.assertTrue(
            self.simulator.engine.database.verify_integrity()["ok"]
        )

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
            root_concept_id="t_machine_learning",
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

    def test_legacy_event_contract_resolves_focus_without_confidence(self) -> None:
        class LegacyScriptedLearner:
            name = "legacy-scripted"
            response_model = "scripted"

            def __init__(self) -> None:
                self.calls = 0

            def answer(self, presentation, **_context) -> SyntheticAnswer:
                self.calls += 1
                correct = self.calls != 1
                selected = (
                    presentation.question.correct_option
                    if correct
                    else next(
                        option
                        for option in presentation.question.options
                        if not option.correct
                    )
                )
                return SyntheticAnswer(
                    selected_option_id=selected.id,
                    correct=correct,
                    ground_truth_probability=1.0 if correct else 0.0,
                    confidence=None,
                    response_ms=4_000,
                    hint_count=0,
                )

        database = Database(Path(self.tempdir.name) / "legacy-simulation.db")
        database.initialize()
        database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        simulator = BehavioralSimulator(
            AdaptiveEngine(
                database,
                LearnerModel(CONCEPT_MODEL_VERSION),
            )
        )

        report = simulator.run(
            LegacyScriptedLearner(),
            learner_id="legacy-focus-resolution",
            root_concept_id="c_clustering",
            policy_seed=11,
            max_steps=3,
            start_at=START,
        )

        self.assertEqual(
            [
                (
                    step.phase_before.value,
                    step.phase_after.value,
                    step.actual_correct,
                )
                for step in report.steps
            ],
            [
                ("learn", "remediate", False),
                ("remediate", "verify", True),
                ("verify", "learn", True),
            ],
        )
        self.assertEqual(
            [episode.outcome for episode in report.focus_episodes],
            ["resolved"],
        )
        with database.read() as connection:
            model_versions = [
                json.loads(row["metadata_json"])[
                    "learner_model_version"
                ]
                for row in connection.execute(
                    """SELECT metadata_json FROM events
                       WHERE learner_id = ?
                         AND event_type = 'ResponseSubmitted'
                       ORDER BY stream_version""",
                    ("legacy-focus-resolution",),
                ).fetchall()
            ]
        self.assertEqual(
            model_versions,
            [CONCEPT_MODEL_VERSION] * 3,
        )

    def test_verified_prerequisite_returns_to_an_independent_parent_check(self) -> None:
        report = self.simulator.run(
            SyntheticLearner(
                "intermediate",
                default_ability=0.55,
                slip_probability=0.04,
                guess_probability=0.02,
                seed=17,
            ),
            learner_id="verified-boundary",
            root_concept_id="t_large_language_models",
            policy_seed=21,
            trial_index=4,
            max_steps=16,
            start_at=START,
        )

        self.assertFalse(report.has_blockers, report.summary())
        with self.simulator.engine.database.read() as connection:
            session_id = connection.execute(
                "SELECT id FROM sessions WHERE learner_id='verified-boundary'"
            ).fetchone()["id"]
        session_report = self.simulator.engine.session_report(
            session_id, now=report.ended_at
        )
        reasons = [
            step["transition_reason"]
            for step in session_report["adaptive_path"]
        ]
        self.assertIn("descend_to_evidence_boundary", reasons)
        self.assertIn("prerequisite_verified_resume_parent", reasons)
        path = session_report["adaptive_path"]
        resume_index = reasons.index(
            "prerequisite_verified_resume_parent"
        )
        self.assertEqual(path[resume_index]["to_phase"], "verify")
        self.assertEqual(
            path[resume_index + 1]["primary_concept_id"],
            path[resume_index]["focus_after"],
        )
        self.assertEqual(
            path[resume_index + 1]["pedagogical_role"], "verification"
        )

    def test_recursive_planner_avoids_an_exhausted_boundary(self) -> None:
        report = self.simulator.run(
            SyntheticLearner(
                "intermediate",
                default_ability=0.55,
                slip_probability=0.04,
                guess_probability=0.02,
                seed=15,
            ),
            learner_id="four-family-boundary",
            root_concept_id="t_large_language_models",
            policy_seed=17,
            max_steps=30,
            start_at=START,
        )

        self.assertFalse(report.has_blockers, report.summary())
        with self.simulator.engine.database.read() as connection:
            session_id = connection.execute(
                "SELECT id FROM sessions WHERE learner_id='four-family-boundary'"
            ).fetchone()["id"]
        session_report = self.simulator.engine.session_report(
            session_id, now=report.ended_at
        )
        reasons = [
            step["transition_reason"]
            for step in session_report["adaptive_path"]
        ]
        self.assertIn("descend_to_evidence_boundary", reasons)
        self.assertTrue(
            {
                "no_serviceable_prerequisite_boundary",
                "prerequisite_verified_parent_deferred",
                "verified_prerequisite_not_reopened",
                "persistent_prerequisite_verified_parent_deferred",
            }
            & set(reasons),
            reasons,
        )
        self.assertGreaterEqual(
            session_report["adaptive_routing"]["capacity_exits"]
            + session_report["adaptive_routing"]["prevented_reopenings"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
