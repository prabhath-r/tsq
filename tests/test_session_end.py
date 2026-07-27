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
from tsq.errors import ValidationError
from tsq.replay import ProjectionReplay
from tsq.store import Database

from tests.test_scoring_claim_history_upgrade import restore_pre_shadow_schema


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

    def test_future_domain_session_closes_at_explicit_simulation_time(self) -> None:
        started_at = datetime(2101, 2, 3, 9, 0, tzinfo=timezone.utc)
        selected_at = started_at + timedelta(minutes=1)
        answered_at = selected_at + timedelta(minutes=2)
        ended_at = answered_at + timedelta(minutes=3)
        session = self.engine.start_session(
            "session-end",
            "c_attention",
            seed=89,
            now=started_at,
        )
        presentation = self.engine.next_question(
            session["id"], now=selected_at
        )
        self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.8,
            response_ms=90_000,
            now=answered_at,
        )

        ended = self.engine.end_session(
            session["id"],
            completed=True,
            reason="simulation complete",
            idempotency_key="future-session-end",
            now=ended_at,
        )
        # Domain time is not an idempotency input. A retry returns the same
        # durable result even when its supplied clock is earlier.
        replayed = self.engine.end_session(
            session["id"],
            completed=True,
            reason="simulation complete",
            idempotency_key="future-session-end",
            now=started_at,
        )
        self.assertEqual(ended, replayed)
        self.assertEqual(ended["updated_at"], ended_at.isoformat())
        with self.database.read() as connection:
            boundary = connection.execute(
                """SELECT occurred_at FROM events
                   WHERE event_type = 'SessionEnded' AND session_id = ?""",
                (session["id"],),
            ).fetchone()
        self.assertEqual(boundary["occurred_at"], ended_at.isoformat())
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_end_rejects_time_before_latest_session_event_atomically(self) -> None:
        started_at = datetime(2031, 4, 5, 10, 0, tzinfo=timezone.utc)
        selected_at = started_at + timedelta(minutes=4)
        session = self.engine.start_session(
            "session-end",
            "c_attention",
            seed=97,
            now=started_at,
        )
        presentation = self.engine.next_question(
            session["id"], now=selected_at
        )

        with self.assertRaisesRegex(
            ValidationError, "latest recorded event"
        ):
            self.engine.end_session(
                session["id"],
                status="abandoned",
                now=selected_at - timedelta(seconds=1),
            )

        current = self.database.get_session(session["id"])
        self.assertEqual(current["status"], "active")
        self.assertEqual(current["revision"], session["revision"] + 1)
        pending = self.database.pending_presentation(session["id"])
        self.assertIsNotNone(pending)
        self.assertEqual(pending.decision_id, presentation.decision_id)
        with self.database.read() as connection:
            ended_count = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type = 'SessionEnded' AND session_id = ?""",
                (session["id"],),
            ).fetchone()["n"]
        self.assertEqual(ended_count, 0)

        ended_at = selected_at + timedelta(seconds=1)
        ended = self.engine.end_session(
            session["id"],
            status="abandoned",
            now=ended_at,
        )
        self.assertEqual(ended["updated_at"], ended_at.isoformat())
        with self.database.read() as connection:
            times = connection.execute(
                """SELECT event_type, occurred_at FROM events
                   WHERE session_id = ?
                     AND event_type IN (
                         'DecisionInvalidated', 'SessionEnded'
                     )
                   ORDER BY stream_version""",
                (session["id"],),
            ).fetchall()
        self.assertEqual(
            [(row["event_type"], row["occurred_at"]) for row in times],
            [
                ("DecisionInvalidated", ended_at.isoformat()),
                ("SessionEnded", ended_at.isoformat()),
            ],
        )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_end_rejects_naive_domain_time(self) -> None:
        session = self.engine.start_session(
            "session-end", "c_attention", seed=101
        )
        with self.assertRaisesRegex(ValidationError, "timezone-aware"):
            self.engine.end_session(
                session["id"],
                now=datetime(2030, 1, 1),
            )
        self.assertEqual(
            self.database.get_session(session["id"])["status"], "active"
        )

    def test_integrity_detects_closed_session_projection_corruption(self) -> None:
        ended_at = datetime(2032, 5, 6, 12, 0, tzinfo=timezone.utc)
        session = self.engine.start_session(
            "session-end",
            "c_attention",
            seed=103,
            now=ended_at - timedelta(hours=1),
        )
        self.engine.end_session(session["id"], now=ended_at)
        self.assertTrue(self.database.verify_integrity()["ok"])

        with self.database.transaction() as connection:
            connection.execute(
                """UPDATE sessions SET status = 'active',
                       updated_at = created_at
                   WHERE id = ?""",
                (session["id"],),
            )
        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "active status has a SessionEnded boundary" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        self.assertTrue(
            any(
                "SessionEnded payload: status mismatch" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        self.assertTrue(
            any(
                "updated_at does not match SessionEnded occurrence" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )

    def test_integrity_detects_session_end_before_start(self) -> None:
        started_at = datetime(2033, 6, 7, 8, 0, tzinfo=timezone.utc)
        session = self.engine.start_session(
            "session-end",
            "c_attention",
            seed=107,
            now=started_at,
        )
        self.engine.end_session(
            session["id"], now=started_at + timedelta(minutes=1)
        )
        impossible_end = started_at - timedelta(seconds=1)
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            connection.execute(
                """UPDATE events SET occurred_at = ?
                   WHERE event_type = 'SessionEnded' AND session_id = ?""",
                (impossible_end.isoformat(), session["id"]),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (impossible_end.isoformat(), session["id"]),
            )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "SessionEnded occurred before SessionStarted" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )

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
            restore_pre_shadow_schema(connection)
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

        self.assertEqual(schema_version, "19")
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
