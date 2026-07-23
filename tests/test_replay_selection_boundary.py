# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.learner import LearnerModel
from tsq.replay import ProjectionReplay
from tsq.store import Database
from tsq.versions import (
    BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION,
    OBJECTIVE_GRID_V7_MODEL_VERSION,
    OBJECTIVE_GRID_V8_MODEL_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2102, 6, 7, 9, 0, tzinfo=timezone.utc)
DIGEST = "a" * 64


class ReplaySelectionBoundaryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "selection.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _pending(self, learner_id: str, model_version: str):
        engine = AdaptiveEngine(
            self.database, LearnerModel(model_version)
        )
        engine.create_learner(learner_id)
        session = engine.start_session(
            learner_id,
            "t_transformers",
            mode="learn",
            seed=17,
            now=START,
        )
        selected = engine.next_question(session["id"], now=START)
        self.assertIsNotNone(selected.question.objective_id)
        return engine, selected

    def _answer(
        self,
        learner_id: str,
        model_version: str,
        *,
        response_ms: int = 900,
        answered_at: datetime | None = None,
    ):
        engine, selected = self._pending(learner_id, model_version)
        engine.submit_answer(
            selected.decision_id,
            selected.question.correct_option.id,
            confidence=0.9,
            response_ms=response_ms,
            now=answered_at or START + timedelta(minutes=1),
        )
        return selected

    def _drop_event_update_guard(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")

    def test_v8_schema_three_replays_response_and_pre_response_action(self) -> None:
        engine, selected = self._pending(
            "v8-selection-action", OBJECTIVE_GRID_V8_MODEL_VERSION
        )
        engine.record_action(
            selected.decision_id,
            "answer_revised",
            {"answer_digest": DIGEST},
            now=START + timedelta(seconds=2),
        )
        engine.submit_answer(
            selected.decision_id,
            selected.question.correct_option.id,
            confidence=0.9,
            response_ms=900,
            now=START + timedelta(minutes=1),
        )

        with self.database.read() as connection:
            selection = connection.execute(
                """SELECT schema_version, occurred_at
                   FROM events
                   WHERE event_type='QuestionSelected'
                     AND learner_id='v8-selection-action'"""
            ).fetchone()
            decision_clock = connection.execute(
                "SELECT created_at FROM decisions WHERE id=?",
                (selected.decision_id,),
            ).fetchone()["created_at"]
        self.assertEqual(
            selection["schema_version"],
            BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION,
        )
        self.assertEqual(selection["occurred_at"], decision_clock)

        report = ProjectionReplay(self.database).check(
            "v8-selection-action"
        )
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["action_event_count"], 1)
        self.assertTrue(report["action_projection_matches_replay"])

    def test_v8_replay_rejects_mutated_decision_clock(self) -> None:
        selected = self._answer(
            "v8-clock-mutation", OBJECTIVE_GRID_V8_MODEL_VERSION
        )
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE decisions SET created_at=? WHERE id=?",
                (
                    (START - timedelta(days=30)).isoformat(),
                    selected.decision_id,
                ),
            )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "selection event time is not bound to decision" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        with self.assertRaisesRegex(
            ValidationError, "does not match its decision clock"
        ):
            ProjectionReplay(self.database).check("v8-clock-mutation")

    def test_v8_replay_binds_response_model_to_selection_model(self) -> None:
        self._answer(
            "v8-model-mutation", OBJECTIVE_GRID_V8_MODEL_VERSION
        )
        self._drop_event_update_guard()
        with self.database.transaction() as connection:
            events = connection.execute(
                """SELECT event_id, metadata_json FROM events
                   WHERE learner_id='v8-model-mutation'
                     AND event_type IN (
                         'ResponseSubmitted', 'LearnerProjectionAdvanced'
                     )"""
            ).fetchall()
            self.assertEqual(len(events), 2)
            for event in events:
                metadata = json.loads(event["metadata_json"])
                metadata["learner_model_version"] = (
                    OBJECTIVE_GRID_V7_MODEL_VERSION
                )
                connection.execute(
                    "UPDATE events SET metadata_json=? WHERE event_id=?",
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
        self.assertTrue(
            any(
                "response learner model does not match selection" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        with self.assertRaisesRegex(
            ValidationError,
            "learner model does not match its QuestionSelected event",
        ):
            ProjectionReplay(self.database).check("v8-model-mutation")

    def test_schema_three_selection_payload_rejects_unknown_fields(self) -> None:
        self._answer(
            "v8-payload-mutation", OBJECTIVE_GRID_V8_MODEL_VERSION
        )
        self._drop_event_update_guard()
        with self.database.transaction() as connection:
            event = connection.execute(
                """SELECT event_id, payload_json FROM events
                   WHERE learner_id='v8-payload-mutation'
                     AND event_type='QuestionSelected'"""
            ).fetchone()
            payload = json.loads(event["payload_json"])
            payload["unversioned_clock_hint"] = "accept-me"
            connection.execute(
                "UPDATE events SET payload_json=? WHERE event_id=?",
                (
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    event["event_id"],
                ),
                )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "selection event: unexpected fields" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        with self.assertRaisesRegex(
            ValidationError,
            "QuestionSelected event .* payload has incompatible fields.*unknown",
        ):
            ProjectionReplay(self.database).check("v8-payload-mutation")

    def test_selection_metadata_rejects_unversioned_fields(self) -> None:
        self._answer(
            "v8-metadata-mutation", OBJECTIVE_GRID_V8_MODEL_VERSION
        )
        self._drop_event_update_guard()
        with self.database.transaction() as connection:
            event = connection.execute(
                """SELECT event_id, metadata_json FROM events
                   WHERE learner_id='v8-metadata-mutation'
                     AND event_type='QuestionSelected'"""
            ).fetchone()
            metadata = json.loads(event["metadata_json"])
            metadata["selection_clock_source"] = "decision"
            connection.execute(
                "UPDATE events SET metadata_json=? WHERE event_id=?",
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
        self.assertTrue(
            any(
                "selection metadata: unexpected fields" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        with self.assertRaisesRegex(
            ValidationError,
            "QuestionSelected event .* metadata has incompatible fields.*unknown",
        ):
            ProjectionReplay(self.database).check("v8-metadata-mutation")

    def test_v7_replay_preserves_legacy_unbound_decision_clock(self) -> None:
        selected = self._answer(
            "v7-unbound-clock",
            OBJECTIVE_GRID_V7_MODEL_VERSION,
            response_ms=2_000,
            answered_at=START + timedelta(seconds=1),
        )
        with self.database.transaction() as connection:
            selection_schema = connection.execute(
                """SELECT schema_version FROM events
                   WHERE learner_id='v7-unbound-clock'
                     AND event_type='QuestionSelected'"""
            ).fetchone()["schema_version"]
            connection.execute(
                "UPDATE decisions SET created_at=? WHERE id=?",
                (
                    (START - timedelta(days=30)).isoformat(),
                    selected.decision_id,
                ),
            )
        self.assertEqual(selection_schema, 2)

        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])
        report = ProjectionReplay(self.database).check("v7-unbound-clock")
        self.assertTrue(report["ok"], report["errors"])


if __name__ == "__main__":
    unittest.main()
