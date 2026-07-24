# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.capacity import VERIFICATION_KINDS
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.learner import (
    MODEL_VERSION,
    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
    LearnerModel,
)
from tsq.models import SessionPhase
from tsq.policy import AdaptivePolicy
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2102, 3, 4, 9, 0, tzinfo=timezone.utc)


class PersistentObjectiveRemediationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "persistent.db")
        self.database.initialize()
        parsed = read_and_parse(CORPUS, include_catalog=True)
        self.questions = tuple(
            question
            for question in parsed[4]
            if question.status.eligible_for_adaptation
        )
        self.release_id = self.database.import_corpus(*parsed)["release_id"]
        self.objectives = {
            objective.id: objective
            for objective in self.database.get_learning_objectives(
                self.release_id
            )
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _engine(self, model_version: str = MODEL_VERSION) -> AdaptiveEngine:
        return AdaptiveEngine(self.database, LearnerModel(model_version))

    def _set_objective_focus(
        self,
        session_id: str,
        objective_id: str,
        *,
        phase: SessionPhase,
        depth: int,
        path: list[dict[str, object]],
    ) -> None:
        objective = self.objectives[objective_id]
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE sessions
                   SET phase = ?, focus_concept_id = ?,
                       focus_misconception_id = NULL,
                       focus_objective_id = ?, remediation_depth = ?,
                       remediation_path_json = ?, updated_at = ?
                   WHERE id = ?""",
                (
                    phase.value,
                    objective.primary_concept_id,
                    objective_id,
                    depth,
                    json.dumps(path, sort_keys=True, separators=(",", ":")),
                    START.isoformat(),
                    session_id,
                ),
            )

    def _set_legacy_focus(
        self,
        session_id: str,
        concept_id: str,
        *,
        phase: SessionPhase,
        depth: int,
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE sessions
                   SET phase = ?, focus_concept_id = ?,
                       focus_misconception_id = NULL,
                       focus_objective_id = NULL, remediation_depth = ?,
                       remediation_path_json = '[]', updated_at = ?
                   WHERE id = ?""",
                (
                    phase.value,
                    concept_id,
                    depth,
                    START.isoformat(),
                    session_id,
                ),
            )

    def test_persistent_certificate_is_release_scoped_and_ready(self) -> None:
        learner_id = "persistent-prerequisite"
        engine = self._engine(OBJECTIVE_GAUSSIAN_MODEL_VERSION)
        engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id, "t_transformers", seed=0, now=START
        )
        prerequisite_id = "lo_transformer_information_paths"
        parent_id = "lo_transformer_sublayer_composition"
        prerequisite_families = []
        for question in self.questions:
            if (
                question.objective_id == prerequisite_id
                and question.kind in VERIFICATION_KINDS
                and question.family_id not in {
                    family_id for family_id, _ in prerequisite_families
                }
            ):
                prerequisite_families.append(
                    (question.family_id, question.kind.value)
                )
        self.assertGreaterEqual(len(prerequisite_families), 2)

        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO objective_states(
                       learner_id, objective_id, mean, variance,
                       stability_hours, exposures, last_seen_at,
                       next_review_at, evidence_mass, as_of_event_id,
                       model_version
                   ) VALUES (?, ?, -1.0, 0.10, 720.0, 3, ?, ?, 2.0, NULL, ?)""",
                (
                    learner_id,
                    prerequisite_id,
                    START.isoformat(),
                    (START + timedelta(days=30)).isoformat(),
                    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
                ),
            )
            connection.executemany(
                """INSERT INTO learner_objective_families(
                       learner_id, objective_id, family_id, kind,
                       first_unguided_correct_at,
                       last_unguided_correct_at,
                       delayed_unguided_correct_at
                   ) VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                (
                    (
                        learner_id,
                        prerequisite_id,
                        family_id,
                        kind,
                        (START - timedelta(days=offset + 2)).isoformat(),
                        (START - timedelta(days=offset + 1)).isoformat(),
                    )
                    for offset, (family_id, kind) in enumerate(
                        (
                            prerequisite_families[0],
                            ("f_not_in_pinned_release", "transfer"),
                        )
                    )
                ),
            )

        with self.database.read() as connection:
            not_yet_verified = engine._declared_objective_boundary(
                connection,
                session_id=session["id"],
                learner_id=learner_id,
                release_id=self.release_id,
                focus_objective_id=parent_id,
                now=START,
            )
        self.assertEqual(
            not_yet_verified["selected_objective_id"], prerequisite_id
        )
        self.assertNotIn(prerequisite_id, not_yet_verified["verified"])

        second_family, second_kind = prerequisite_families[1]
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO learner_objective_families(
                       learner_id, objective_id, family_id, kind,
                       first_unguided_correct_at,
                       last_unguided_correct_at,
                       delayed_unguided_correct_at
                   ) VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                (
                    learner_id,
                    prerequisite_id,
                    second_family,
                    second_kind,
                    (START - timedelta(days=4)).isoformat(),
                    (START - timedelta(days=3)).isoformat(),
                ),
            )

        with self.database.read() as connection:
            still_not_ready = engine._declared_objective_boundary(
                connection,
                session_id=session["id"],
                learner_id=learner_id,
                release_id=self.release_id,
                focus_objective_id=parent_id,
                now=START,
            )
        self.assertEqual(
            still_not_ready["selected_objective_id"], prerequisite_id
        )
        self.assertNotIn(prerequisite_id, still_not_ready["verified"])

        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE objective_states SET mean = 3.0
                   WHERE learner_id = ? AND objective_id = ?""",
                (learner_id, prerequisite_id),
            )
        with self.database.read() as connection:
            verified = engine._declared_objective_boundary(
                connection,
                session_id=session["id"],
                learner_id=learner_id,
                release_id=self.release_id,
                focus_objective_id=parent_id,
                now=START,
            )
            current_attempts = connection.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE session_id = ?",
                (session["id"],),
            ).fetchone()["n"]
        self.assertIsNone(verified["selected_objective_id"])
        self.assertEqual(verified["verified"], {prerequisite_id})
        self.assertEqual(current_attempts, 0)

    def test_declared_chain_serves_grand_prerequisite_and_unwinds_frames(self) -> None:
        learner_id = "multi-hop-objective"
        engine = self._engine()
        engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id, "t_transformers", seed=17, now=START
        )
        root_id = "lo_transformer_sublayer_composition"
        child_id = "lo_transformer_information_paths"
        grandchild_id = "lo_attention_value_routing"
        self._set_objective_focus(
            session["id"],
            root_id,
            phase=SessionPhase.REMEDIATE,
            depth=1,
            path=[],
        )
        clock = START + timedelta(minutes=1)

        root_repair = engine.next_question(session["id"], now=clock)
        self.assertEqual(root_repair.question.objective_id, root_id)
        root_wrong = next(
            option for option in root_repair.question.options if not option.correct
        )
        clock += timedelta(minutes=1)
        first_descent = engine.submit_answer(
            root_repair.decision_id,
            root_wrong.id,
            confidence=0.9,
            response_ms=1200,
            now=clock,
        )
        self.assertEqual(first_descent.focus_objective_id, child_id)
        self.assertEqual(
            [
                frame["objective_id"]
                for frame in self.database.get_session(session["id"])[
                    "remediation_path"
                ]
            ],
            [root_id],
        )

        clock += timedelta(minutes=1)
        child_repair = engine.next_question(session["id"], now=clock)
        self.assertEqual(child_repair.question.objective_id, child_id)
        child_wrong = next(
            option for option in child_repair.question.options if not option.correct
        )
        clock += timedelta(minutes=1)
        second_descent = engine.submit_answer(
            child_repair.decision_id,
            child_wrong.id,
            confidence=0.9,
            response_ms=1200,
            now=clock,
        )
        self.assertEqual(second_descent.focus_objective_id, grandchild_id)
        second_descent_session = self.database.get_session(session["id"])
        self.assertEqual(second_descent_session["remediation_depth"], 3)
        self.assertEqual(
            [
                frame["objective_id"]
                for frame in second_descent_session["remediation_path"]
            ],
            [root_id, child_id],
        )

        clock += timedelta(minutes=1)
        grandchild_repair = engine.next_question(session["id"], now=clock)
        self.assertEqual(grandchild_repair.question.objective_id, grandchild_id)
        clock += timedelta(minutes=1)
        repaired = engine.submit_answer(
            grandchild_repair.decision_id,
            grandchild_repair.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            now=clock,
        )
        self.assertEqual(repaired.next_phase, SessionPhase.VERIFY)
        self.assertEqual(repaired.focus_objective_id, grandchild_id)

        clock += timedelta(minutes=1)
        grandchild_verify = engine.next_question(session["id"], now=clock)
        self.assertEqual(grandchild_verify.question.objective_id, grandchild_id)
        self.assertNotEqual(
            grandchild_verify.question.family_id,
            grandchild_repair.question.family_id,
        )
        clock += timedelta(minutes=1)
        resumed_child = engine.submit_answer(
            grandchild_verify.decision_id,
            grandchild_verify.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            now=clock,
        )
        self.assertEqual(
            resumed_child.transition_reason,
            "prerequisite_verified_resume_parent",
        )
        self.assertEqual(resumed_child.focus_objective_id, child_id)
        self.assertEqual(
            [
                frame["objective_id"]
                for frame in self.database.get_session(session["id"])[
                    "remediation_path"
                ]
            ],
            [root_id],
        )

        clock += timedelta(minutes=1)
        child_verify = engine.next_question(session["id"], now=clock)
        self.assertEqual(child_verify.question.objective_id, child_id)
        clock += timedelta(minutes=1)
        resumed_root = engine.submit_answer(
            child_verify.decision_id,
            child_verify.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            now=clock,
        )
        self.assertEqual(resumed_root.focus_objective_id, root_id)
        self.assertEqual(
            self.database.get_session(session["id"])["remediation_path"],
            [],
        )

        clock += timedelta(minutes=1)
        root_verify = engine.next_question(session["id"], now=clock)
        self.assertEqual(root_verify.question.objective_id, root_id)
        clock += timedelta(minutes=1)
        completed = engine.submit_answer(
            root_verify.decision_id,
            root_verify.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            now=clock,
        )
        self.assertEqual(completed.next_phase, SessionPhase.LEARN)
        self.assertIsNone(completed.focus_objective_id)

    def test_prior_transfer_plus_repair_resumes_without_retesting(self) -> None:
        learner_id = "persistent-resume"
        engine = self._engine(OBJECTIVE_GAUSSIAN_MODEL_VERSION)
        engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id, "t_transformers", seed=19, now=START
        )
        parent_id = "lo_transformer_sublayer_composition"
        prerequisite_id = "lo_transformer_information_paths"
        parent_frame = {
            "concept_id": self.objectives[parent_id].primary_concept_id,
            "objective_id": parent_id,
            "misconception_id": None,
        }
        self._set_objective_focus(
            session["id"],
            prerequisite_id,
            phase=SessionPhase.REMEDIATE,
            depth=2,
            path=[parent_frame],
        )
        repair = engine.next_question(session["id"], now=START)
        prior = next(
            question
            for question in self.questions
            if question.objective_id == prerequisite_id
            and question.family_id != repair.question.family_id
            and question.kind in VERIFICATION_KINDS
        )
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO objective_states(
                       learner_id, objective_id, mean, variance,
                       stability_hours, exposures, last_seen_at,
                       next_review_at, evidence_mass, as_of_event_id,
                       model_version
                   ) VALUES (?, ?, 3.0, 0.10, 720.0, 2, ?, ?, 1.5, NULL, ?)""",
                (
                    learner_id,
                    prerequisite_id,
                    (START - timedelta(days=2)).isoformat(),
                    (START + timedelta(days=30)).isoformat(),
                    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
                ),
            )
            connection.execute(
                """INSERT INTO learner_objective_families(
                       learner_id, objective_id, family_id, kind,
                       first_unguided_correct_at,
                       last_unguided_correct_at,
                       delayed_unguided_correct_at
                   ) VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                (
                    learner_id,
                    prerequisite_id,
                    prior.family_id,
                    prior.kind.value,
                    (START - timedelta(days=3)).isoformat(),
                    (START - timedelta(days=2)).isoformat(),
                ),
            )

        resumed = engine.submit_answer(
            repair.decision_id,
            repair.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            now=START + timedelta(minutes=1),
        )

        self.assertEqual(resumed.next_phase, SessionPhase.VERIFY)
        self.assertEqual(resumed.focus_objective_id, parent_id)
        self.assertEqual(
            resumed.transition_reason,
            "persistent_prerequisite_verification_resume_parent",
        )
        current = self.database.get_session(session["id"])
        self.assertEqual(current["remediation_path"], [])
        self.assertEqual(current["remediation_depth"], 1)

    def test_v6_missing_timing_cannot_certify_but_v5_remains_compatible(self) -> None:
        for index, (model_version, expected_reason) in enumerate(
            (
                (MODEL_VERSION, "noncredible_success_requires_verification"),
                (OBJECTIVE_GAUSSIAN_MODEL_VERSION, "credible_main_success"),
            )
        ):
            learner_id = f"missing-timing-{index}"
            engine = self._engine(model_version)
            engine.create_learner(learner_id)
            session = engine.start_session(
                learner_id, "c_clustering", seed=index, now=START
            )
            selected = engine.next_question(session["id"], now=START)
            result = engine.submit_answer(
                selected.decision_id,
                selected.question.correct_option.id,
                confidence=0.9,
                now=START + timedelta(minutes=1),
            )
            self.assertEqual(result.transition_reason, expected_reason)

    def test_v7_missing_telemetry_cannot_satisfy_same_session_prerequisite(self) -> None:
        learner_id = "missing-prerequisite-telemetry"
        engine = self._engine()
        engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id, "t_transformers", seed=29, now=START
        )
        parent_id = "lo_transformer_sublayer_composition"
        prerequisite_id = "lo_transformer_information_paths"
        clock = START

        self._set_objective_focus(
            session["id"],
            prerequisite_id,
            phase=SessionPhase.REMEDIATE,
            depth=1,
            path=[],
        )
        clock += timedelta(minutes=1)
        repair = engine.next_question(session["id"], now=clock)
        clock += timedelta(minutes=1)
        noncertifying_repair = engine.submit_answer(
            repair.decision_id,
            repair.question.correct_option.id,
            confidence=None,
            response_ms=1200,
            now=clock,
        )
        self.assertEqual(
            noncertifying_repair.transition_reason,
            "noncredible_repair_requires_another_probe",
        )

        self._set_objective_focus(
            session["id"],
            prerequisite_id,
            phase=SessionPhase.VERIFY,
            depth=2,
            path=[],
        )
        clock += timedelta(minutes=1)
        verification = engine.next_question(session["id"], now=clock)
        self.assertNotEqual(
            verification.question.family_id, repair.question.family_id
        )
        clock += timedelta(minutes=1)
        noncertifying_verification = engine.submit_answer(
            verification.decision_id,
            verification.question.correct_option.id,
            confidence=0.9,
            response_ms=None,
            now=clock,
        )
        self.assertEqual(
            noncertifying_verification.transition_reason,
            "noncredible_verification_bounded_exit",
        )

        with self.database.read() as connection:
            certified_families = connection.execute(
                """SELECT COUNT(*) AS n
                   FROM learner_objective_families
                   WHERE learner_id=? AND objective_id=?""",
                (learner_id, prerequisite_id),
            ).fetchone()["n"]
        self.assertEqual(certified_families, 0)

        self._set_objective_focus(
            session["id"],
            parent_id,
            phase=SessionPhase.REMEDIATE,
            depth=1,
            path=[],
        )
        clock += timedelta(minutes=1)
        parent_repair = engine.next_question(session["id"], now=clock)
        parent_wrong = next(
            option for option in parent_repair.question.options
            if not option.correct
        )
        clock += timedelta(minutes=1)
        bounded_exit = engine.submit_answer(
            parent_repair.decision_id,
            parent_wrong.id,
            confidence=0.9,
            response_ms=1200,
            now=clock,
        )
        self.assertEqual(
            bounded_exit.transition_reason,
            "no_serviceable_prerequisite_boundary",
        )
        self.assertEqual(bounded_exit.next_phase, SessionPhase.LEARN)
        self.assertIsNone(bounded_exit.focus_objective_id)

    def test_v7_missing_telemetry_cannot_satisfy_legacy_concept_prerequisite(self) -> None:
        learner_id = "missing-legacy-prerequisite-telemetry"
        engine = self._engine()
        engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id, "c_clustering", seed=31, now=START
        )
        prerequisite_id = "c_feature_scaling"
        clock = START

        self._set_legacy_focus(
            session["id"],
            prerequisite_id,
            phase=SessionPhase.REMEDIATE,
            depth=1,
        )
        clock += timedelta(minutes=1)
        repair = engine.next_question(session["id"], now=clock)
        clock += timedelta(minutes=1)
        engine.submit_answer(
            repair.decision_id,
            repair.question.correct_option.id,
            confidence=None,
            response_ms=1200,
            now=clock,
        )

        self._set_legacy_focus(
            session["id"],
            prerequisite_id,
            phase=SessionPhase.VERIFY,
            depth=2,
        )
        clock += timedelta(minutes=1)
        verification = engine.next_question(session["id"], now=clock)
        self.assertNotEqual(
            verification.question.family_id, repair.question.family_id
        )
        clock += timedelta(minutes=1)
        engine.submit_answer(
            verification.decision_id,
            verification.question.correct_option.id,
            confidence=0.9,
            response_ms=None,
            now=clock,
        )

        with self.database.read() as connection:
            certified_families = connection.execute(
                """SELECT COUNT(*) AS n
                   FROM learner_skill_families
                   WHERE learner_id=? AND concept_id=?""",
                (learner_id, prerequisite_id),
            ).fetchone()["n"]
        self.assertEqual(certified_families, 0)

        self._set_legacy_focus(
            session["id"],
            "c_clustering",
            phase=SessionPhase.REMEDIATE,
            depth=1,
        )
        clock += timedelta(minutes=1)
        parent_repair = engine.next_question(session["id"], now=clock)
        parent_wrong = next(
            option for option in parent_repair.question.options
            if not option.correct
        )
        clock += timedelta(minutes=1)
        descended = engine.submit_answer(
            parent_repair.decision_id,
            parent_wrong.id,
            confidence=0.9,
            response_ms=1200,
            now=clock,
        )
        self.assertEqual(
            descended.transition_reason, "descend_to_evidence_boundary"
        )
        self.assertEqual(descended.focus_concept_id, prerequisite_id)
        self.assertIsNone(descended.focus_objective_id)

    def test_exploration_gate_uses_the_same_model_timing_contract(self) -> None:
        session = {
            "topic_id": "t_transformers",
            "exploration_mode": "adaptive",
            "focus_concept_id": None,
            "focus_misconception_id": None,
            "focus_objective_id": None,
            "step": 3,
        }
        def recent(model_version: str) -> list[dict]:
            return [
                {
                    "correct": True,
                    "pedagogical_role": "main",
                    "hint_count": 0,
                    "confidence": 0.9,
                    "response_ms": None,
                    "selected_option_id": "option-a",
                    "learner_model_version": model_version,
                }
                for _ in range(3)
            ]

        self.assertFalse(
            AdaptivePolicy(
                self.database, LearnerModel(MODEL_VERSION)
            )._should_explore(
                session,
                SessionPhase.LEARN,
                recent(MODEL_VERSION),
            )
        )
        self.assertTrue(
            AdaptivePolicy(
                self.database,
                LearnerModel(OBJECTIVE_GAUSSIAN_MODEL_VERSION),
            )._should_explore(
                session,
                SessionPhase.LEARN,
                recent(OBJECTIVE_GAUSSIAN_MODEL_VERSION),
            )
        )


if __name__ == "__main__":
    unittest.main()
