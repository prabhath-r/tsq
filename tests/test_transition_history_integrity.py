# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.learner import LearnerModel
from tsq.store import Database, _content_hash
from tsq.versions import LEGACY_MODEL_VERSION


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
START = datetime(2108, 2, 3, 9, 0, tzinfo=timezone.utc)


class TransitionHistoryIntegrityTestCase(unittest.TestCase):
    """Pin event-ordered remediation replay across frozen schemas."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(
            Path(self.tempdir.name) / "transition-history.db"
        )
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @staticmethod
    def _rehash_streams(connection: sqlite3.Connection) -> None:
        stream_ids = [
            row["stream_id"]
            for row in connection.execute(
                "SELECT DISTINCT stream_id FROM events ORDER BY stream_id"
            )
        ]
        for stream_id in stream_ids:
            previous_hash = None
            tail_version = 0
            for event in connection.execute(
                """SELECT * FROM events
                   WHERE stream_id=? ORDER BY stream_version""",
                (stream_id,),
            ).fetchall():
                envelope = {
                    "event_id": event["event_id"],
                    "stream_id": event["stream_id"],
                    "stream_version": event["stream_version"],
                    "event_type": event["event_type"],
                    "schema_version": event["schema_version"],
                    "occurred_at": event["occurred_at"],
                    "recorded_at": event["recorded_at"],
                    "learner_id": event["learner_id"],
                    "session_id": event["session_id"],
                    "correlation_id": event["correlation_id"],
                    "causation_id": event["causation_id"],
                    "idempotency_key": event["idempotency_key"],
                    "payload": json.loads(event["payload_json"]),
                    "metadata": json.loads(event["metadata_json"]),
                    "previous_hash": previous_hash,
                }
                payload_hash = _content_hash(envelope)
                connection.execute(
                    """UPDATE events
                       SET previous_hash=?, payload_hash=?
                       WHERE event_id=?""",
                    (
                        previous_hash,
                        payload_hash,
                        event["event_id"],
                    ),
                )
                previous_hash = payload_hash
                tail_version = event["stream_version"]
            connection.execute(
                """UPDATE stream_heads
                   SET stream_version=?, payload_hash=?
                   WHERE stream_id=?""",
                (tail_version, previous_hash, stream_id),
            )

    @staticmethod
    def _drop_event_update_trigger(
        connection: sqlite3.Connection,
    ) -> str:
        row = connection.execute(
            """SELECT sql FROM sqlite_master
               WHERE type='trigger' AND name='events_no_update'"""
        ).fetchone()
        if row is None:
            raise AssertionError("Missing events_no_update trigger.")
        connection.execute("DROP TRIGGER events_no_update")
        return row["sql"]

    def test_equal_answer_timestamps_replay_in_stream_order(self) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("equal-clock")
        session = engine.start_session(
            "equal-clock",
            "t_transformers",
            mode="learn",
            seed=17,
            now=START,
        )
        first = engine.next_question(session["id"], now=START)
        wrong = next(
            option for option in first.question.options if not option.correct
        )
        answered_at = START + timedelta(seconds=1)
        with patch(
            "tsq.engine.new_id", return_value="att_z_equal_clock"
        ):
            engine.submit_answer(
                first.decision_id,
                wrong.id,
                confidence=0.9,
                response_ms=1_000,
                now=answered_at,
            )

        second = engine.next_question(session["id"], now=answered_at)
        with patch(
            "tsq.engine.new_id", return_value="att_a_equal_clock"
        ):
            engine.submit_answer(
                second.decision_id,
                second.question.correct_option.id,
                confidence=0.9,
                response_ms=0,
                now=answered_at,
            )

        with self.database.read() as connection:
            id_order = [
                row["id"]
                for row in connection.execute(
                    """SELECT id FROM attempts
                       ORDER BY answered_at, id"""
                )
            ]
        self.assertEqual(
            id_order, ["att_a_equal_clock", "att_z_equal_clock"]
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_schema_one_transition_payload_is_verified(self) -> None:
        engine = AdaptiveEngine(
            self.database, LearnerModel(LEGACY_MODEL_VERSION)
        )
        engine.create_learner("legacy-transition")
        session = engine.start_session(
            "legacy-transition",
            "t_machine_learning",
            mode="learn",
            seed=7,
            now=START,
        )
        selected = engine.next_question(session["id"], now=START)
        self.assertIsNone(selected.question.objective_id)
        wrong = next(
            option
            for option in selected.question.options
            if not option.correct
        )
        result = engine.submit_answer(
            selected.decision_id,
            wrong.id,
            now=START + timedelta(seconds=1),
        )
        self.assertEqual(result.next_phase.value, "remediate")

        with closing(sqlite3.connect(self.database.path)) as connection:
            connection.row_factory = sqlite3.Row
            trigger_sql = self._drop_event_update_trigger(connection)
            projection = connection.execute(
                """SELECT * FROM events
                   WHERE learner_id='legacy-transition'
                     AND event_type='LearnerProjectionAdvanced'"""
            ).fetchone()
            transition = connection.execute(
                """SELECT * FROM events
                   WHERE learner_id='legacy-transition'
                     AND event_type='RemediationTransitioned'"""
            ).fetchone()
            projection_payload = json.loads(projection["payload_json"])
            projection_payload.pop("transition_reason")
            projection_payload.pop("boundary_decision")
            connection.execute(
                """UPDATE events
                   SET schema_version=1, payload_json=?
                   WHERE event_id=?""",
                (
                    json.dumps(
                        projection_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    projection["event_id"],
                ),
            )
            transition_payload = json.loads(transition["payload_json"])
            transition_payload.pop("transition_reason")
            transition_payload.pop("boundary_decision")
            connection.execute(
                """UPDATE events
                   SET schema_version=1, payload_json=?
                   WHERE event_id=?""",
                (
                    json.dumps(
                        transition_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    transition["event_id"],
                ),
            )
            self._rehash_streams(connection)
            connection.execute(trigger_sql)
            connection.commit()

        clean_legacy = self.database.verify_integrity()
        self.assertTrue(clean_legacy["ok"], clean_legacy["errors"])

        with closing(sqlite3.connect(self.database.path)) as connection:
            connection.row_factory = sqlite3.Row
            trigger_sql = self._drop_event_update_trigger(connection)
            transition = connection.execute(
                """SELECT * FROM events
                   WHERE learner_id='legacy-transition'
                     AND event_type='RemediationTransitioned'"""
            ).fetchone()
            transition_payload = json.loads(transition["payload_json"])
            transition_payload["from_phase"] = "review"
            connection.execute(
                """UPDATE events SET payload_json=? WHERE event_id=?""",
                (
                    json.dumps(
                        transition_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    transition["event_id"],
                ),
            )
            self._rehash_streams(connection)
            connection.execute(trigger_sql)
            connection.commit()

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"], integrity["errors"])
        self.assertTrue(
            any(
                "remediation transition: from_phase mismatch" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )

    def test_session_start_boundary_is_bound_to_session(self) -> None:
        engine = AdaptiveEngine(self.database)
        engine.create_learner("session-owner")
        engine.create_learner("wrong-owner")
        session = engine.start_session(
            "session-owner",
            "t_transformers",
            mode="learn",
            seed=11,
            now=START,
        )

        with closing(sqlite3.connect(self.database.path)) as connection:
            connection.row_factory = sqlite3.Row
            trigger_sql = self._drop_event_update_trigger(connection)
            started = connection.execute(
                """SELECT * FROM events
                   WHERE event_type='SessionStarted' AND session_id=?""",
                (session["id"],),
            ).fetchone()
            payload = json.loads(started["payload_json"])
            payload["session_id"] = "ses_false_payload"
            connection.execute(
                """UPDATE events
                   SET learner_id=?, payload_json=?
                   WHERE event_id=?""",
                (
                    "wrong-owner",
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    started["event_id"],
                ),
            )
            self._rehash_streams(connection)
            connection.execute(trigger_sql)
            connection.commit()

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"], integrity["errors"])


if __name__ == "__main__":
    unittest.main()
