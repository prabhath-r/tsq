# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, ValidationError
from tsq.learner import (
    CONCEPT_MODEL_VERSION,
    MODEL_VERSION,
    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
    OBJECTIVE_GRID_V6_MODEL_VERSION,
    OBJECTIVE_GRID_V7_MODEL_VERSION,
    LearnerModel,
)
from tsq.models import SessionPhase
from tsq.policy import POLICY_VERSION
from tsq.replay import ProjectionReplay
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2101, 2, 3, 9, 0, tzinfo=timezone.utc)


class LearnerModelBoundaryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "model-boundary.db")
        self.database.initialize()
        parsed = read_and_parse(CORPUS, include_catalog=True)
        self.questions = parsed[4]
        self.release_id = self.database.import_corpus(*parsed)["release_id"]

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _engine(self, model_version: str) -> AdaptiveEngine:
        return AdaptiveEngine(self.database, LearnerModel(model_version))

    def _pending_objective(
        self, learner_id: str, model_version: str, *, seed: int = 0
    ):
        engine = self._engine(model_version)
        engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id,
            "t_transformers",
            mode="learn",
            seed=seed,
            now=START,
        )
        presentation = engine.next_question(session["id"], now=START)
        self.assertIsNotNone(presentation.question.objective_id)
        return engine, session, presentation

    def test_policy_invalidates_cross_model_pending_choice_before_reuse(self) -> None:
        _, session, selected_v5 = self._pending_objective(
            "policy-model-change", OBJECTIVE_GAUSSIAN_MODEL_VERSION
        )

        current = self._engine(MODEL_VERSION)
        selected_v6 = current.next_question(
            session["id"], now=START + timedelta(seconds=1)
        )

        self.assertNotEqual(selected_v5.decision_id, selected_v6.decision_id)
        with self.database.read() as connection:
            stale = connection.execute(
                "SELECT * FROM decisions WHERE id = ?",
                (selected_v5.decision_id,),
            ).fetchone()
            invalidation = connection.execute(
                """SELECT payload_json FROM events
                   WHERE event_type = 'DecisionInvalidated'
                     AND causation_id = ?""",
                (selected_v5.decision_id,),
            ).fetchone()
            selections = connection.execute(
                """SELECT metadata_json FROM events
                   WHERE event_type = 'QuestionSelected'
                     AND session_id = ? ORDER BY stream_version""",
                (session["id"],),
            ).fetchall()
        self.assertEqual(stale["invalidation_reason"], "learner_model_changed")
        self.assertIsNotNone(stale["invalidated_at"])
        self.assertEqual(
            json.loads(invalidation["payload_json"])["reason"],
            "learner_model_changed",
        )
        self.assertEqual(
            [
                json.loads(row["metadata_json"])["learner_model_version"]
                for row in selections
            ],
            [OBJECTIVE_GAUSSIAN_MODEL_VERSION, MODEL_VERSION],
        )
        self.assertEqual(
            {
                json.loads(row["metadata_json"])["policy_version"]
                for row in selections
            },
            {POLICY_VERSION},
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_submission_durably_invalidates_cross_model_pending_choice(self) -> None:
        _, _, selected_v5 = self._pending_objective(
            "submit-model-change", OBJECTIVE_GAUSSIAN_MODEL_VERSION
        )

        with self.assertRaisesRegex(ConflictError, "stale decision was invalidated"):
            self._engine(MODEL_VERSION).submit_answer(
                selected_v5.decision_id,
                selected_v5.question.correct_option.id,
                confidence=0.8,
                response_ms=900,
                now=START + timedelta(minutes=1),
            )

        with self.database.read() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id = ?",
                (selected_v5.decision_id,),
            ).fetchone()
            attempts = connection.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE learner_id = ?",
                ("submit-model-change",),
            ).fetchone()["n"]
            revision = connection.execute(
                "SELECT revision FROM learners WHERE id = ?",
                ("submit-model-change",),
            ).fetchone()["revision"]
        self.assertEqual(decision["invalidation_reason"], "learner_model_changed")
        self.assertIsNone(decision["consumed_at"])
        self.assertEqual(attempts, 0)
        self.assertEqual(revision, 0)
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_completed_answer_replays_idempotently_across_model_upgrade(self) -> None:
        engine_v5, _, selected = self._pending_objective(
            "model-upgrade-replay", OBJECTIVE_GAUSSIAN_MODEL_VERSION
        )
        first = engine_v5.submit_answer(
            selected.decision_id,
            selected.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            idempotency_key="answer-before-model-upgrade",
            now=START + timedelta(minutes=1),
        )

        replayed = self._engine(MODEL_VERSION).submit_answer(
            selected.decision_id,
            selected.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            idempotency_key="answer-before-model-upgrade",
            now=START + timedelta(minutes=2),
        )

        self.assertEqual(replayed.interaction_id, first.interaction_id)
        self.assertTrue(replayed.idempotent_replay)
        with self.database.read() as connection:
            decision = connection.execute(
                "SELECT invalidated_at FROM decisions WHERE id = ?",
                (selected.decision_id,),
            ).fetchone()
        self.assertIsNone(decision["invalidated_at"])

    def test_completed_objective_answer_retries_under_concept_model(self) -> None:
        engine_v5, _, selected = self._pending_objective(
            "objective-retry-under-v4", OBJECTIVE_GAUSSIAN_MODEL_VERSION
        )
        first = engine_v5.submit_answer(
            selected.decision_id,
            selected.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            idempotency_key="objective-before-concept-runtime",
            now=START + timedelta(minutes=1),
        )

        replayed = self._engine(CONCEPT_MODEL_VERSION).submit_answer(
            selected.decision_id,
            selected.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            idempotency_key="objective-before-concept-runtime",
            now=START + timedelta(minutes=2),
        )

        self.assertEqual(replayed.interaction_id, first.interaction_id)
        self.assertTrue(replayed.idempotent_replay)

    def test_policy_upgrade_invalidates_pending_choice_before_reuse(self) -> None:
        prior_policy = "recursive-evidence-graph-v12"
        with patch("tsq.policy.POLICY_VERSION", prior_policy):
            _, session, stale = self._pending_objective(
                "policy-version-change", MODEL_VERSION
            )

        replacement = self._engine(MODEL_VERSION).next_question(
            session["id"], now=START + timedelta(seconds=1)
        )

        self.assertNotEqual(stale.decision_id, replacement.decision_id)
        with self.database.read() as connection:
            stale_row = connection.execute(
                """SELECT invalidation_reason, invalidated_at
                   FROM decisions WHERE id = ?""",
                (stale.decision_id,),
            ).fetchone()
            selected_events = connection.execute(
                """SELECT metadata_json FROM events
                   WHERE event_type = 'QuestionSelected'
                     AND session_id = ?
                   ORDER BY stream_version""",
                (session["id"],),
            ).fetchall()
            invalidation = connection.execute(
                """SELECT payload_json, metadata_json FROM events
                   WHERE event_type = 'DecisionInvalidated'
                     AND causation_id = ?""",
                (stale.decision_id,),
            ).fetchone()
        self.assertEqual(stale_row["invalidation_reason"], "policy_changed")
        self.assertIsNotNone(stale_row["invalidated_at"])
        self.assertEqual(
            [
                json.loads(row["metadata_json"])["policy_version"]
                for row in selected_events
            ],
            [prior_policy, POLICY_VERSION],
        )
        self.assertEqual(
            json.loads(invalidation["payload_json"])["reason"],
            "policy_changed",
        )
        self.assertEqual(
            json.loads(invalidation["metadata_json"])["policy_version"],
            POLICY_VERSION,
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_submission_durably_invalidates_cross_policy_pending_choice(
        self,
    ) -> None:
        prior_policy = "recursive-evidence-graph-v12"
        with patch("tsq.policy.POLICY_VERSION", prior_policy):
            _, _, stale = self._pending_objective(
                "submit-policy-change", MODEL_VERSION
            )

        with self.assertRaisesRegex(ConflictError, "adaptive policy changed"):
            self._engine(MODEL_VERSION).submit_answer(
                stale.decision_id,
                stale.question.correct_option.id,
                confidence=0.9,
                response_ms=900,
                now=START + timedelta(minutes=1),
            )

        with self.database.read() as connection:
            decision = connection.execute(
                """SELECT invalidation_reason, consumed_at
                   FROM decisions WHERE id = ?""",
                (stale.decision_id,),
            ).fetchone()
            attempt_count = connection.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE learner_id = ?",
                ("submit-policy-change",),
            ).fetchone()["n"]
        self.assertEqual(decision["invalidation_reason"], "policy_changed")
        self.assertIsNone(decision["consumed_at"])
        self.assertEqual(attempt_count, 0)
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_completed_answer_replays_idempotently_across_policy_upgrade(
        self,
    ) -> None:
        prior_policy = "recursive-evidence-graph-v12"
        with (
            patch("tsq.policy.POLICY_VERSION", prior_policy),
            patch("tsq.engine.POLICY_VERSION", prior_policy),
        ):
            engine, _, selected = self._pending_objective(
                "policy-upgrade-replay", MODEL_VERSION
            )
            first = engine.submit_answer(
                selected.decision_id,
                selected.question.correct_option.id,
                confidence=0.9,
                response_ms=900,
                idempotency_key="answer-before-policy-upgrade",
                now=START + timedelta(minutes=1),
            )

        replayed = self._engine(MODEL_VERSION).submit_answer(
            selected.decision_id,
            selected.question.correct_option.id,
            confidence=0.9,
            response_ms=900,
            idempotency_key="answer-before-policy-upgrade",
            now=START + timedelta(minutes=2),
        )

        self.assertEqual(replayed.interaction_id, first.interaction_id)
        self.assertTrue(replayed.idempotent_replay)
        with self.database.read() as connection:
            decision = connection.execute(
                """SELECT invalidated_at FROM decisions WHERE id = ?""",
                (selected.decision_id,),
            ).fetchone()
            invalidations = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type = 'DecisionInvalidated'
                     AND causation_id = ?""",
                (selected.decision_id,),
            ).fetchone()["n"]
        self.assertIsNone(decision["invalidated_at"])
        self.assertEqual(invalidations, 0)

    def test_omitted_feedback_flag_cannot_create_acquisition(self) -> None:
        engine, _, selected = self._pending_objective(
            "no-implicit-feedback", MODEL_VERSION
        )

        result = engine.submit_answer(
            selected.decision_id,
            selected.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            now=START + timedelta(minutes=1),
        )

        self.assertTrue(result.correct)
        self.assertTrue(result.state_changes)
        self.assertTrue(
            all(
                change["feedback_transition"] == 0.0
                for change in result.state_changes
            )
        )
        with self.database.read() as connection:
            attempt = connection.execute(
                "SELECT feedback_shown FROM attempts WHERE decision_id = ?",
                (selected.decision_id,),
            ).fetchone()
            response = connection.execute(
                """SELECT payload_json FROM events
                   WHERE event_type = 'ResponseSubmitted'
                     AND causation_id = ?""",
                (selected.decision_id,),
            ).fetchone()
        self.assertEqual(attempt["feedback_shown"], 0)
        self.assertIs(
            json.loads(response["payload_json"])["feedback_shown"], False
        )

        explicit_engine, _, explicit_selected = self._pending_objective(
            "truthful-explicit-feedback", MODEL_VERSION, seed=1
        )
        explicit = explicit_engine.submit_answer(
            explicit_selected.decision_id,
            explicit_selected.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            feedback_shown=True,
            now=START + timedelta(minutes=2),
        )
        self.assertTrue(
            any(
                change["feedback_transition"] > 0.0
                for change in explicit.state_changes
            )
        )

    def test_objective_projection_versions_follow_the_model_boundary(self) -> None:
        expected = (
            (OBJECTIVE_GAUSSIAN_MODEL_VERSION, 3, 2),
            (OBJECTIVE_GRID_V6_MODEL_VERSION, 4, 3),
            (MODEL_VERSION, 4, 3),
        )
        for index, (model_version, event_schema, hash_schema) in enumerate(
            expected
        ):
            learner_id = f"projection-model-{index}"
            engine, _, selected = self._pending_objective(
                learner_id, model_version, seed=index
            )
            engine.submit_answer(
                selected.decision_id,
                selected.question.correct_option.id,
                confidence=0.8,
                response_ms=900,
                now=START + timedelta(minutes=index + 1),
            )
            with self.database.read() as connection:
                event = connection.execute(
                    """SELECT schema_version, payload_json, metadata_json
                       FROM events WHERE learner_id = ?
                         AND event_type = 'LearnerProjectionAdvanced'""",
                    (learner_id,),
                ).fetchone()
            self.assertEqual(event["schema_version"], event_schema)
            self.assertEqual(
                json.loads(event["payload_json"])["projection_hash_version"],
                hash_schema,
            )
            self.assertEqual(
                json.loads(event["metadata_json"])["learner_model_version"],
                model_version,
            )

        concept_engine = self._engine(MODEL_VERSION)
        concept_engine.create_learner("concept-projection-version")
        concept_session = concept_engine.start_session(
            "concept-projection-version",
            "c_clustering",
            seed=0,
            now=START,
        )
        concept_selected = concept_engine.next_question(
            concept_session["id"], now=START
        )
        self.assertIsNone(concept_selected.question.objective_id)
        concept_engine.submit_answer(
            concept_selected.decision_id,
            concept_selected.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            now=START + timedelta(minutes=3),
        )
        with self.database.read() as connection:
            legacy_shape = connection.execute(
                """SELECT schema_version, payload_json FROM events
                   WHERE learner_id = ?
                     AND event_type = 'LearnerProjectionAdvanced'""",
                ("concept-projection-version",),
            ).fetchone()
        self.assertEqual(legacy_shape["schema_version"], 2)
        self.assertNotIn(
            "projection_hash_version",
            json.loads(legacy_shape["payload_json"]),
        )

    def test_v8_selection_schema_binds_the_decision_clock(self) -> None:
        for index, (model_version, expected_schema) in enumerate(
            (
                (OBJECTIVE_GRID_V7_MODEL_VERSION, 2),
                (MODEL_VERSION, 3),
            )
        ):
            learner_id = f"selection-schema-{index}"
            _, _, selected = self._pending_objective(
                learner_id, model_version, seed=index
            )
            with self.database.read() as connection:
                row = connection.execute(
                    """SELECT selection.schema_version,
                              selection.occurred_at,
                              selection.metadata_json,
                              decision.created_at
                       FROM decisions decision
                       JOIN events selection
                         ON selection.event_type='QuestionSelected'
                        AND json_extract(
                            selection.payload_json, '$.decision_id'
                        )=decision.id
                       WHERE decision.id=?""",
                    (selected.decision_id,),
                ).fetchone()
            self.assertEqual(row["schema_version"], expected_schema)
            self.assertEqual(
                json.loads(row["metadata_json"])["learner_model_version"],
                model_version,
            )
            if expected_schema == 3:
                self.assertEqual(row["occurred_at"], row["created_at"])

    def test_integrity_and_replay_reject_unknown_response_and_projection_models(
        self,
    ) -> None:
        engine, _, selected = self._pending_objective(
            "unknown-event-model", MODEL_VERSION
        )
        engine.submit_answer(
            selected.decision_id,
            selected.question.correct_option.id,
            confidence=0.9,
            response_ms=900,
            now=START + timedelta(minutes=1),
        )
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            events = connection.execute(
                """SELECT event_id, metadata_json FROM events
                   WHERE learner_id = ?
                     AND event_type IN (
                         'ResponseSubmitted', 'LearnerProjectionAdvanced'
                     )
                   ORDER BY stream_version""",
                ("unknown-event-model",),
            ).fetchall()
            self.assertEqual(len(events), 2)
            for event in events:
                metadata = json.loads(event["metadata_json"])
                metadata["learner_model_version"] = "future-model-v99"
                connection.execute(
                    """UPDATE events SET metadata_json = ?
                       WHERE event_id = ?""",
                    (
                        json.dumps(
                            metadata,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        event["event_id"],
                    ),
                )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        unsupported = [
            error
            for error in integrity["errors"]
            if "unsupported learner model 'future-model-v99'" in error
        ]
        self.assertEqual(len(unsupported), 2, integrity["errors"])
        with self.assertRaisesRegex(
            ValidationError, "unsupported learner model 'future-model-v99'"
        ):
            ProjectionReplay(self.database).check("unknown-event-model")

    def test_v8_integrity_and_replay_reject_impossible_response_window(
        self,
    ) -> None:
        engine, _, selected = self._pending_objective(
            "impossible-response-window", MODEL_VERSION
        )
        with patch(
            "tsq.engine.response_window",
            return_value=SimpleNamespace(consistent=True),
        ):
            engine.submit_answer(
                selected.decision_id,
                selected.question.correct_option.id,
                confidence=0.9,
                response_ms=2_000,
                now=START + timedelta(seconds=1),
            )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "authoritative selection-to-answer window" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        with self.assertRaisesRegex(
            ValidationError, "authoritative selection-to-answer window"
        ):
            ProjectionReplay(self.database).check(
                "impossible-response-window"
            )

    def test_legacy_replay_preserves_pre_v8_response_window_semantics(
        self,
    ) -> None:
        for index, model_version in enumerate(
            (
                OBJECTIVE_GAUSSIAN_MODEL_VERSION,
                OBJECTIVE_GRID_V6_MODEL_VERSION,
                OBJECTIVE_GRID_V7_MODEL_VERSION,
            )
        ):
            learner_id = f"legacy-response-window-{index}"
            engine, _, selected = self._pending_objective(
                learner_id, model_version
            )
            engine.submit_answer(
                selected.decision_id,
                selected.question.correct_option.id,
                confidence=0.9,
                response_ms=2_000,
                now=START + timedelta(seconds=1),
            )

        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])
        for index in range(3):
            replay = ProjectionReplay(self.database).check(
                f"legacy-response-window-{index}"
            )
            self.assertTrue(replay["ok"], replay["errors"])

    def test_v8_submission_enforces_authoritative_response_window(self) -> None:
        engine, _, selected = self._pending_objective(
            "v8-live-response-window", MODEL_VERSION
        )
        with self.assertRaisesRegex(
            ValidationError, "authoritative time"
        ):
            engine.submit_answer(
                selected.decision_id,
                selected.question.correct_option.id,
                confidence=0.9,
                response_ms=2_000,
                now=START + timedelta(seconds=1),
            )

    def test_model_upgrade_uses_v6_checkpoint_for_later_concept_answer(self) -> None:
        learner_id = "mixed-projection-boundary"
        engine_v5, _, objective_selected = self._pending_objective(
            learner_id, OBJECTIVE_GAUSSIAN_MODEL_VERSION
        )
        engine_v5.submit_answer(
            objective_selected.decision_id,
            objective_selected.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            now=START + timedelta(minutes=1),
        )

        engine_v6 = self._engine(MODEL_VERSION)
        concept_session = engine_v6.start_session(
            learner_id,
            "c_clustering",
            seed=0,
            now=START + timedelta(days=1),
        )
        concept_selected = engine_v6.next_question(
            concept_session["id"], now=START + timedelta(days=1)
        )
        self.assertIsNone(concept_selected.question.objective_id)
        engine_v6.submit_answer(
            concept_selected.decision_id,
            concept_selected.question.correct_option.id,
            confidence=0.8,
            response_ms=900,
            now=START + timedelta(days=1, minutes=1),
        )

        with self.database.read() as connection:
            checkpoints = connection.execute(
                """SELECT schema_version, payload_json, metadata_json
                   FROM events WHERE learner_id = ?
                     AND event_type = 'LearnerProjectionAdvanced'
                   ORDER BY stream_version""",
                (learner_id,),
            ).fetchall()
        self.assertEqual(
            [row["schema_version"] for row in checkpoints], [3, 4]
        )
        self.assertEqual(
            [
                json.loads(row["payload_json"])["projection_hash_version"]
                for row in checkpoints
            ],
            [2, 3],
        )
        self.assertEqual(
            [
                json.loads(row["metadata_json"])["learner_model_version"]
                for row in checkpoints
            ],
            [OBJECTIVE_GAUSSIAN_MODEL_VERSION, MODEL_VERSION],
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_v6_planned_information_discounts_a_repeated_family(self) -> None:
        question = next(
            question
            for question in self.questions
            if question.objective_id is not None
        )
        graph = self.database.get_graph(self.release_id)
        session = {
            "learner_id": "planned-information",
            "topic_id": None,
            "focus_concept_id": None,
            "focus_misconception_id": None,
            "focus_objective_id": None,
        }
        readiness = {
            question.primary_concept_id: SimpleNamespace(
                bottleneck_concept_id=None,
                prerequisite_support=1.0,
            )
        }

        def planned_information(model_version: str, family_count: int) -> float:
            learner_model = LearnerModel(model_version)
            policy = self._engine(model_version).policy
            stored_states = {
                mapping.concept_id: learner_model.initial_state(
                    session["learner_id"], graph.concepts[mapping.concept_id]
                )
                for mapping in question.concepts
            }
            score = policy._score(
                question,
                session=session,
                phase=SessionPhase.LEARN,
                prerequisite_distances={question.primary_concept_id: 0},
                concepts=graph.concepts,
                stored_states=stored_states,
                objective_states={},
                beliefs={},
                exposure={
                    "questions": {},
                    "families": {
                        question.family_id: {"count": family_count}
                    },
                },
                recent_families=[],
                last_primary_concept=None,
                topic_by_concept={},
                base_scope={question.primary_concept_id},
                connected_pairs=set(),
                readiness=readiness,
                now=START,
            )
            return score.information_gain

        unseen_v6 = planned_information(MODEL_VERSION, 0)
        repeated_v6 = planned_information(MODEL_VERSION, 1)
        self.assertLess(repeated_v6, unseen_v6)

        unseen_v5 = planned_information(OBJECTIVE_GAUSSIAN_MODEL_VERSION, 0)
        repeated_v5 = planned_information(
            OBJECTIVE_GAUSSIAN_MODEL_VERSION, 1
        )
        self.assertEqual(repeated_v5, unseen_v5)

    def test_pre_objective_model_still_fails_closed(self) -> None:
        engine = self._engine(CONCEPT_MODEL_VERSION)
        engine.create_learner("pre-objective-boundary")
        session = engine.start_session(
            "pre-objective-boundary",
            "t_transformers",
            seed=0,
            now=START,
        )

        with self.assertRaisesRegex(
            ValidationError, "cannot select objective-aware questions"
        ):
            engine.next_question(session["id"], now=START)

        with self.database.read() as connection:
            decision_count = connection.execute(
                "SELECT COUNT(*) AS n FROM decisions WHERE learner_id = ?",
                ("pre-objective-boundary",),
            ).fetchone()["n"]
        self.assertEqual(decision_count, 0)


if __name__ == "__main__":
    unittest.main()
