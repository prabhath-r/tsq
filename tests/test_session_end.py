# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.replay import ProjectionReplay
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"


class SessionEndIntegrityTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "session-end.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)
        self.engine.create_learner("session-end", "Session End")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _pending_session(self) -> tuple[dict[str, object], str]:
        session = self.engine.start_session(
            "session-end", "c_attention", seed=43
        )
        presentation = self.engine.next_question(session["id"])
        return self.database.get_session(session["id"]), presentation.decision_id

    def test_end_invalidates_pending_decision_before_session_boundary(self) -> None:
        session, decision_id = self._pending_session()

        ended = self.engine.end_session(
            session["id"],
            status="completed",
            reason="learner_finished",
            idempotency_key="finish-with-pending",
        )
        replayed = self.engine.end_session(
            session["id"],
            status="completed",
            reason="learner_finished",
            idempotency_key="finish-with-pending",
        )

        self.assertEqual(ended, replayed)
        self.assertEqual(ended["status"], "completed")
        self.assertEqual(ended["revision"], session["revision"] + 1)
        self.assertIsNone(self.database.pending_presentation(session["id"]))
        with self.database.read() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
            boundaries = connection.execute(
                """SELECT * FROM events
                   WHERE session_id = ?
                     AND event_type IN ('DecisionInvalidated', 'SessionEnded')
                   ORDER BY stream_version""",
                (session["id"],),
            ).fetchall()
            learner_revision = connection.execute(
                "SELECT revision FROM learners WHERE id = 'session-end'"
            ).fetchone()["revision"]

        self.assertIsNotNone(decision["invalidated_at"])
        self.assertEqual(decision["invalidation_reason"], "session_completed")
        self.assertEqual(
            [boundary["event_type"] for boundary in boundaries],
            ["DecisionInvalidated", "SessionEnded"],
        )
        invalidation = boundaries[0]
        invalidation_payload = json.loads(invalidation["payload_json"])
        self.assertEqual(invalidation["causation_id"], decision_id)
        self.assertEqual(invalidation["occurred_at"], decision["invalidated_at"])
        self.assertEqual(
            invalidation_payload,
            {
                "decision_id": decision_id,
                "reason": "session_completed",
                "selection_learner_revision": decision["learner_revision"],
                "current_learner_revision": learner_revision,
            },
        )
        self.assertEqual(
            json.loads(boundaries[1]["payload_json"]),
            {
                "session_id": session["id"],
                "status": "completed",
                "reason": "learner_finished",
            },
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])
        projection = ProjectionReplay(self.database).check("session-end")
        self.assertTrue(projection["ok"], projection["errors"])

    def test_abandoned_session_records_an_honest_invalidation_reason(self) -> None:
        session, decision_id = self._pending_session()

        self.engine.end_session(session["id"], status="abandoned")

        with self.database.read() as connection:
            decision = connection.execute(
                "SELECT invalidation_reason FROM decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
            event = connection.execute(
                """SELECT payload_json FROM events
                   WHERE event_type = 'DecisionInvalidated'
                     AND causation_id = ?""",
                (decision_id,),
            ).fetchone()
        self.assertEqual(decision["invalidation_reason"], "session_abandoned")
        self.assertEqual(
            json.loads(event["payload_json"])["reason"], "session_abandoned"
        )

    def test_session_end_rolls_back_invalidation_if_boundary_append_fails(self) -> None:
        session, decision_id = self._pending_session()
        with self.database.read() as connection:
            before_events = connection.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]
        original_append = self.database.append_event

        def fail_on_session_end(connection, **kwargs):
            if kwargs["event_type"] == "SessionEnded":
                raise RuntimeError("synthetic session boundary failure")
            return original_append(connection, **kwargs)

        with patch.object(
            self.database, "append_event", side_effect=fail_on_session_end
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic session"):
                self.engine.end_session(
                    session["id"],
                    status="completed",
                    idempotency_key="atomic-session-end",
                )

        current = self.database.get_session(session["id"])
        self.assertEqual(current["status"], "active")
        self.assertEqual(current["revision"], session["revision"])
        with self.database.read() as connection:
            event_count = connection.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]
            decision = connection.execute(
                "SELECT invalidated_at, invalidation_reason FROM decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
            invalidations = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type = 'DecisionInvalidated'
                     AND causation_id = ?""",
                (decision_id,),
            ).fetchone()["n"]
        self.assertEqual(event_count, before_events)
        self.assertIsNone(decision["invalidated_at"])
        self.assertIsNone(decision["invalidation_reason"])
        self.assertEqual(invalidations, 0)
        self.assertIsNotNone(self.database.pending_presentation(session["id"]))

        ended = self.engine.end_session(
            session["id"],
            status="completed",
            idempotency_key="atomic-session-end",
        )
        self.assertEqual(ended["status"], "completed")
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_v10_migration_appends_boundary_for_legacy_stale_decision(self) -> None:
        first, stale_decision_id = self._pending_session()
        second = self.engine.start_session(
            "session-end", "c_attention", seed=71
        )
        started = datetime(2101, 1, 1, tzinfo=timezone.utc)
        presentation = self.engine.next_question(second["id"], now=started)
        self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.9,
            response_ms=1200,
            now=started + timedelta(minutes=1),
        )
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE meta SET value = '9' WHERE key = 'schema_version'"
            )

        migrated = Database(self.database.path)
        migrated.initialize()

        with migrated.read() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id = ?",
                (stale_decision_id,),
            ).fetchone()
            invalidation = connection.execute(
                """SELECT * FROM events
                   WHERE event_type = 'DecisionInvalidated'
                     AND causation_id = ?""",
                (stale_decision_id,),
            ).fetchone()
            schema_version = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()["value"]

        self.assertEqual(schema_version, "13")
        self.assertEqual(
            decision["invalidation_reason"], "learner_projection_advanced"
        )
        self.assertEqual(
            invalidation["occurred_at"], decision["invalidated_at"]
        )
        payload = json.loads(invalidation["payload_json"])
        self.assertEqual(payload["decision_id"], stale_decision_id)
        self.assertGreater(
            payload["current_learner_revision"],
            payload["selection_learner_revision"],
        )
        self.assertEqual(
            migrated.get_session(first["id"])["status"], "active"
        )
        integrity = migrated.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])


if __name__ == "__main__":
    unittest.main()
