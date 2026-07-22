# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.learner import MODEL_VERSION
from tsq.replay import ProjectionReplay
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
GOLDEN = ROOT / "tests" / "fixtures" / "action_replay_expected.json"
START = datetime(2101, 6, 7, 8, 0, tzinfo=timezone.utc)
DIGEST_A = "1" * 64
DIGEST_B = "2" * 64


class FixedIds:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)

    def __call__(self, prefix: str) -> str:
        self.counts[prefix] += 1
        return f"{prefix}_action_replay_{self.counts[prefix]:03d}"


class FixedDateTime(datetime):
    ticks = 0

    @classmethod
    def now(cls, tz: timezone | None = None) -> "FixedDateTime":
        microsecond = cls.ticks
        cls.ticks += 1
        value = cls(
            START.year,
            START.month,
            START.day,
            START.hour,
            START.minute,
            START.second,
            microsecond,
            tzinfo=timezone.utc,
        )
        return value if tz is None else value.astimezone(tz)


def build_action_database(path: Path) -> tuple[Database, str]:
    identifiers = FixedIds()
    FixedDateTime.ticks = 0
    with (
        patch("tsq.store.datetime", FixedDateTime),
        patch("tsq.store.new_id", side_effect=identifiers),
        patch("tsq.engine.new_id", side_effect=identifiers),
        patch("tsq.policy.new_id", side_effect=identifiers),
    ):
        database = Database(path)
        database.initialize()
        database.import_corpus(*read_and_parse(CORPUS, include_catalog=True))
        engine = AdaptiveEngine(database)
        engine.create_learner("action-replay", "Action Replay Fixture")
        session = engine.start_session(
            "action-replay", "c_attention", mode="learn", seed=17
        )
        selected_at = START + timedelta(minutes=1)
        presentation = engine.next_question(session["id"], now=selected_at)
        decision_id = presentation.decision_id
        engine.record_action(
            decision_id,
            "started",
            {},
            idempotency_key="action-started",
            now=selected_at + timedelta(seconds=1),
        )
        engine.record_action(
            decision_id,
            "hint_requested",
            {"hint_id": "attention_scale_hint", "level": 1},
            stage="assisted",
            idempotency_key="action-hint",
            now=selected_at + timedelta(seconds=2),
        )
        artifact = {
            "sha256": DIGEST_A,
            "size_bytes": 2048,
            "media_type": "application/octet-stream",
        }
        engine.record_action(
            decision_id,
            "artifact_checkpoint",
            {"artifact_digest": DIGEST_A, "artifact_kind": "implementation"},
            stage="assisted",
            artifact=artifact,
            idempotency_key="action-artifact",
            now=selected_at + timedelta(seconds=3),
        )
        engine.record_action(
            decision_id,
            "submitted",
            {"submission_digest": DIGEST_A},
            stage="assisted",
            artifact=artifact,
            idempotency_key="action-submitted",
            now=selected_at + timedelta(seconds=4),
        )
        engine.submit_answer(
            decision_id,
            presentation.question.correct_option.id,
            confidence=0.84,
            response_ms=5_000,
            hint_count=1,
            idempotency_key="action-answer",
            now=selected_at + timedelta(seconds=5),
        )
        feedback_artifact = {
            "sha256": DIGEST_B,
            "size_bytes": 512,
            "media_type": "text/plain",
        }
        engine.record_action(
            decision_id,
            "feedback_shown",
            {"feedback_digest": DIGEST_B},
            stage="post_feedback",
            artifact=feedback_artifact,
            idempotency_key="action-feedback",
            now=selected_at + timedelta(seconds=6),
        )
        engine.record_action(
            decision_id,
            "explanation_checkpoint",
            {"explanation_digest": DIGEST_B},
            stage="post_feedback",
            artifact=feedback_artifact,
            idempotency_key="action-explanation",
            now=selected_at + timedelta(seconds=7),
        )
    return database, decision_id


def golden_action_projection(report: dict) -> dict:
    return {
        "format_version": report["format_version"],
        "action_event_count": report["action_event_count"],
        "artifact_count": report["artifact_count"],
        "action_projection_hash": report["reconstructed_action_projection_hash"],
        "checkpoints": report["action_checkpoints"],
    }


class ActionProjectionReplayTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database, self.decision_id = build_action_database(
            Path(self.tempdir.name) / "source.db"
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def action_snapshot(self, database: Database) -> dict[str, list[tuple]]:
        with database.read() as connection:
            return {
                "actions": [
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM learning_actions "
                        "ORDER BY decision_id, sequence"
                    )
                ],
                "artifacts": [
                    tuple(row)
                    for row in connection.execute(
                        "SELECT * FROM learning_artifacts ORDER BY id"
                    )
                ],
            }

    def test_reconstructs_exact_golden_action_projection_without_source_writes(
        self,
    ) -> None:
        before = self.action_snapshot(self.database)

        report = ProjectionReplay(self.database).check("action-replay")

        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(report["action_projection_matches_replay"])
        self.assertEqual(
            report["source_action_projection_hash"],
            report["reconstructed_action_projection_hash"],
        )
        self.assertEqual(self.action_snapshot(self.database), before)
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(golden_action_projection(report), expected)

    def test_check_detects_projection_corruption_and_copy_rebuild_repairs_it(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER learning_actions_no_update")
            connection.execute(
                """UPDATE learning_actions SET command_hash=?
                   WHERE decision_id=? AND sequence=3""",
                ("0" * 64, self.decision_id),
            )
        corrupted = self.action_snapshot(self.database)

        report = ProjectionReplay(self.database).check("action-replay")

        self.assertFalse(report["ok"])
        self.assertFalse(report["action_projection_matches_replay"])
        self.assertIn(
            "stored learning-action projection differs from deterministic replay",
            report["errors"],
        )
        self.assertEqual(self.action_snapshot(self.database), corrupted)

        target = Path(self.tempdir.name) / "rebuilt.db"
        rebuilt_report = ProjectionReplay(self.database).rebuild_copy(
            "action-replay", target
        )
        self.assertTrue(rebuilt_report["source_action_projection_was_repaired"])
        self.assertTrue(rebuilt_report["ok"])
        self.assertEqual(self.action_snapshot(self.database), corrupted)
        rebuilt = Database(target)
        self.assertTrue(rebuilt.verify_integrity()["ok"])
        self.assertTrue(ProjectionReplay(rebuilt).check("action-replay")["ok"])

    def test_missing_action_projection_is_recovered_from_event_history(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER learning_actions_no_delete")
            connection.execute(
                "DELETE FROM learning_actions WHERE decision_id=? AND sequence=6",
                (self.decision_id,),
            )

        report = ProjectionReplay(self.database).check("action-replay")

        self.assertFalse(report["ok"])
        self.assertEqual(report["action_event_count"], 6)
        self.assertFalse(report["action_projection_matches_replay"])
        self.assertTrue(
            any(
                "LearnerActionRecorded has no action projection" in error
                for error in report["recoverable_source_integrity_errors"]
            )
        )

    def test_malformed_action_event_fails_closed_and_source_stays_unchanged(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            row = connection.execute(
                """SELECT event_id, payload_json FROM events
                   WHERE event_type='LearnerActionRecorded'
                   ORDER BY stream_version LIMIT 1"""
            ).fetchone()
            payload = json.loads(row["payload_json"])
            payload["raw_answer"] = "must never enter semantic telemetry"
            connection.execute(
                "UPDATE events SET payload_json=? WHERE event_id=?",
                (json.dumps(payload), row["event_id"]),
            )
        before = self.action_snapshot(self.database)

        with self.assertRaisesRegex(ValidationError, "incompatible fields.*unknown"):
            ProjectionReplay(self.database).check("action-replay")

        self.assertEqual(self.action_snapshot(self.database), before)

    def test_conflicting_artifact_declaration_in_history_fails_closed(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            row = connection.execute(
                """SELECT event_id, payload_json FROM events
                   WHERE event_type='LearnerActionRecorded'
                     AND json_extract(payload_json, '$.artifact.sha256')=?
                   ORDER BY stream_version DESC LIMIT 1""",
                (DIGEST_A,),
            ).fetchone()
            payload = json.loads(row["payload_json"])
            payload["artifact"]["size_bytes"] += 1
            connection.execute(
                "UPDATE events SET payload_json=? WHERE event_id=?",
                (json.dumps(payload), row["event_id"]),
            )

        with self.assertRaisesRegex(
            ValidationError, "artifact digest with different metadata"
        ):
            ProjectionReplay(self.database).check("action-replay")

    def test_action_after_session_end_fails_closed(self) -> None:
        with self.database.transaction() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id=?", (self.decision_id,)
            ).fetchone()
            self.database.append_event(
                connection,
                stream_id=f"learner:{decision['learner_id']}",
                event_type="SessionEnded",
                payload={
                    "session_id": decision["session_id"],
                    "status": "completed",
                    "reason": "fixture_end",
                },
                metadata={"corpus_release_id": decision["corpus_release_id"]},
                learner_id=decision["learner_id"],
                session_id=decision["session_id"],
                occurred_at=START + timedelta(minutes=1, seconds=8),
            )
            self.database.append_event(
                connection,
                stream_id=f"learner:{decision['learner_id']}",
                event_type="LearnerActionRecorded",
                payload={
                    "action_id": "act_after_session_end",
                    "decision_id": self.decision_id,
                    "sequence": 7,
                    "stage": "post_feedback",
                    "action_type": "tool_used",
                    "payload": {
                        "tool_id": "reviewer",
                        "purpose_code": "post_session_mutation",
                    },
                    "artifact": None,
                },
                metadata={
                    "action_schema_version": 1,
                    "observational_only": True,
                    "corpus_release_id": decision["corpus_release_id"],
                },
                learner_id=decision["learner_id"],
                session_id=decision["session_id"],
                causation_id=self.decision_id,
                occurred_at=START + timedelta(minutes=1, seconds=9),
            )
        with self.assertRaisesRegex(
            ValidationError, "outside its session-active event interval"
        ):
            ProjectionReplay(self.database).check("action-replay")

    def test_action_recorded_after_emergency_revocation_fails_closed(self) -> None:
        with self.database.read() as connection:
            decision = dict(
                connection.execute(
                    "SELECT * FROM decisions WHERE id=?", (self.decision_id,)
                ).fetchone()
            )
        with patch("tsq.store.datetime", FixedDateTime):
            self.database.revoke_question(
                decision["question_id"], "fixture safety revocation"
            )

        pre_revocation_report = ProjectionReplay(self.database).check("action-replay")
        self.assertTrue(pre_revocation_report["ok"], pre_revocation_report["errors"])

        with (
            patch("tsq.store.datetime", FixedDateTime),
            self.database.transaction() as connection,
        ):
            self.database.append_event(
                connection,
                stream_id=f"learner:{decision['learner_id']}",
                event_type="LearnerActionRecorded",
                payload={
                    "action_id": "act_after_revocation",
                    "decision_id": self.decision_id,
                    "sequence": 7,
                    "stage": "post_feedback",
                    "action_type": "tool_used",
                    "payload": {
                        "tool_id": "reviewer",
                        "purpose_code": "revoked_question_mutation",
                    },
                    "artifact": None,
                },
                metadata={
                    "action_schema_version": 1,
                    "observational_only": True,
                    "corpus_release_id": decision["corpus_release_id"],
                },
                learner_id=decision["learner_id"],
                session_id=decision["session_id"],
                causation_id=self.decision_id,
                occurred_at=START + timedelta(minutes=1, seconds=10),
            )

        with self.assertRaisesRegex(
            ValidationError, "recorded at or after emergency revocation"
        ):
            ProjectionReplay(self.database).check("action-replay")

    def test_pre_response_action_for_stale_parallel_decision_fails_closed(self) -> None:
        database = Database(Path(self.tempdir.name) / "stale.db")
        database.initialize()
        database.import_corpus(*read_and_parse(CORPUS, include_catalog=True))
        engine = AdaptiveEngine(database)
        engine.create_learner("stale-action")
        first_session = engine.start_session(
            "stale-action", "c_attention", seed=41
        )
        first = engine.next_question(first_session["id"], now=START)
        second_session = engine.start_session(
            "stale-action", "c_attention", seed=42
        )
        second = engine.next_question(
            second_session["id"], now=START + timedelta(seconds=1)
        )
        engine.submit_answer(
            second.decision_id,
            second.question.correct_option.id,
            confidence=0.9,
            response_ms=700,
            now=START + timedelta(seconds=2),
        )
        with database.transaction() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id=?", (first.decision_id,)
            ).fetchone()
            database.append_event(
                connection,
                stream_id="learner:stale-action",
                event_type="LearnerActionRecorded",
                payload={
                    "action_id": "act_stale_parallel",
                    "decision_id": first.decision_id,
                    "sequence": 1,
                    "stage": "unassisted",
                    "action_type": "started",
                    "payload": {},
                    "artifact": None,
                },
                metadata={
                    "action_schema_version": 1,
                    "observational_only": True,
                    "corpus_release_id": decision["corpus_release_id"],
                },
                learner_id="stale-action",
                session_id=first_session["id"],
                causation_id=first.decision_id,
                occurred_at=START + timedelta(seconds=3),
            )

        with self.assertRaisesRegex(
            ValidationError, "pre-response work for a stale decision"
        ):
            ProjectionReplay(database).check("stale-action")

    def test_action_after_decision_invalidation_fails_but_history_before_it_replays(
        self,
    ) -> None:
        database = Database(Path(self.tempdir.name) / "invalidated.db")
        database.initialize()
        database.import_corpus(*read_and_parse(CORPUS, include_catalog=True))
        engine = AdaptiveEngine(database)
        engine.create_learner("invalidated-action")
        session = engine.start_session(
            "invalidated-action", "c_attention", seed=51
        )
        presentation = engine.next_question(session["id"], now=START)
        engine.record_action(
            presentation.decision_id,
            "started",
            {},
            now=START + timedelta(seconds=1),
        )
        invalidated_at = START + timedelta(seconds=2)
        with database.transaction() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id=?",
                (presentation.decision_id,),
            ).fetchone()
            connection.execute(
                """UPDATE decisions
                   SET invalidated_at=?, invalidation_reason=? WHERE id=?""",
                (
                    invalidated_at.isoformat(),
                    "question_emergency_revoked",
                    presentation.decision_id,
                ),
            )
            database.append_event(
                connection,
                stream_id="learner:invalidated-action",
                event_type="DecisionInvalidated",
                payload={
                    "decision_id": presentation.decision_id,
                    "reason": "question_emergency_revoked",
                    "selection_learner_revision": decision["learner_revision"],
                    "current_learner_revision": decision["learner_revision"],
                },
                metadata={
                    "policy_version": decision["policy_version"],
                    "learner_model_version": MODEL_VERSION,
                    "corpus_release_id": decision["corpus_release_id"],
                },
                learner_id="invalidated-action",
                session_id=session["id"],
                causation_id=presentation.decision_id,
                occurred_at=invalidated_at,
            )

        historic_report = ProjectionReplay(database).check("invalidated-action")
        self.assertTrue(historic_report["ok"], historic_report["errors"])

        with database.transaction() as connection:
            database.append_event(
                connection,
                stream_id="learner:invalidated-action",
                event_type="LearnerActionRecorded",
                payload={
                    "action_id": "act_after_invalidation",
                    "decision_id": presentation.decision_id,
                    "sequence": 2,
                    "stage": "unassisted",
                    "action_type": "answer_revised",
                    "payload": {"answer_digest": DIGEST_A},
                    "artifact": None,
                },
                metadata={
                    "action_schema_version": 1,
                    "observational_only": True,
                    "corpus_release_id": decision["corpus_release_id"],
                },
                learner_id="invalidated-action",
                session_id=session["id"],
                causation_id=presentation.decision_id,
                occurred_at=START + timedelta(seconds=3),
            )

        with self.assertRaisesRegex(
            ValidationError, "appended at or after decision invalidation"
        ):
            ProjectionReplay(database).check("invalidated-action")


if __name__ == "__main__":
    unittest.main()
