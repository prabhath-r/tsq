# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, NotFoundError, ValidationError
from tsq.replay import ProjectionReplay
from tsq.store import SCHEMA_VERSION, Database

from tests.schema_upgrade_helpers import restore_pre_shadow_schema


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2101, 5, 6, 9, 0, tzinfo=timezone.utc)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


class ActionLedgerTestCase(unittest.TestCase):
    """Contract tests for observational learner-action checkpoints.

    ``record_action`` returns the durable action projection as a dictionary.
    JSON fields are decoded, ``sequence`` is assigned by the backend, and an
    exact idempotent retry returns the same IDs with ``idempotent_replay`` set.
    Merely recording an action is never learner evidence.
    """

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "actions.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)
        self.engine.create_learner("action-learner", "Action Learner")
        self.session = self.engine.start_session(
            "action-learner", "t_machine_learning", seed=37
        )
        self.presentation = self.engine.next_question(
            self.session["id"], now=START
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def record(
        self,
        action_type: str,
        payload: dict,
        *,
        seconds: int,
        key: str | None = None,
        decision_id: str | None = None,
    ) -> dict:
        return self.engine.record_action(
            decision_id or self.presentation.decision_id,
            action_type,
            payload,
            idempotency_key=key,
            now=START + timedelta(seconds=seconds),
        )

    def _drop_action_guards(self) -> None:
        with self.database.transaction() as connection:
            trigger_names = [
                row["name"]
                for row in connection.execute(
                    """SELECT name FROM sqlite_master
                       WHERE type='trigger' AND tbl_name='learning_actions'"""
                )
            ]
            self.assertTrue(trigger_names, "learning_actions must have guards")
            for trigger_name in trigger_names:
                escaped = trigger_name.replace('"', '""')
                connection.execute(f'DROP TRIGGER "{escaped}"')

    def _insert_raw_action(
        self,
        action_type: str,
        payload: dict,
        *,
        stage: str,
        occurred_at: datetime,
        artifact: dict | None = None,
        drop_validation_trigger: bool = False,
    ) -> dict:
        """Exercise the database trust boundary without the engine validator."""

        with self.database.transaction() as connection:
            if drop_validation_trigger:
                connection.execute(
                    "DROP TRIGGER IF EXISTS learning_actions_validate_insert"
                )
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id=?",
                (self.presentation.decision_id,),
            ).fetchone()
            prior = connection.execute(
                """SELECT MAX(sequence) AS sequence FROM learning_actions
                   WHERE decision_id=?""",
                (self.presentation.decision_id,),
            ).fetchone()
            sequence = int(prior["sequence"] or 0) + 1
            action_id = f"act_raw_{action_type}_{sequence}"
            artifact_id = None
            if artifact is not None:
                artifact_id = f"art_{artifact['sha256']}"
                connection.execute(
                    """INSERT INTO learning_artifacts(
                           id, sha256, size_bytes, media_type, created_at
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        artifact_id,
                        artifact["sha256"],
                        artifact["size_bytes"],
                        artifact["media_type"],
                        occurred_at.isoformat(),
                    ),
                )
            event_payload = {
                "action_id": action_id,
                "decision_id": self.presentation.decision_id,
                "sequence": sequence,
                "stage": stage,
                "action_type": action_type,
                "payload": payload,
                "artifact": artifact,
            }
            event = self.database.append_event(
                connection,
                stream_id="learner:action-learner",
                event_type="LearnerActionRecorded",
                schema_version=1,
                payload=event_payload,
                metadata={
                    "action_schema_version": 1,
                    "observational_only": True,
                    "corpus_release_id": decision["corpus_release_id"],
                },
                learner_id="action-learner",
                session_id=self.session["id"],
                causation_id=self.presentation.decision_id,
                occurred_at=occurred_at,
            )
            command_material = {
                "decision_id": self.presentation.decision_id,
                "stage": stage,
                "action_type": action_type,
                "payload": payload,
                "artifact": artifact,
            }
            command_hash = hashlib.sha256(
                json.dumps(
                    command_material,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            connection.execute(
                """INSERT INTO learning_actions(
                       id, event_id, decision_id, session_id, learner_id,
                       sequence, stage, action_type, payload_json, artifact_id,
                       occurred_at, recorded_at, command_hash
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    action_id,
                    event["event_id"],
                    self.presentation.decision_id,
                    self.session["id"],
                    "action-learner",
                    sequence,
                    stage,
                    action_type,
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    artifact_id,
                    occurred_at.isoformat(),
                    event["recorded_at"],
                    command_hash,
                ),
            )
        return {"id": action_id, "event_id": event["event_id"]}

    def test_v7_to_current_migration_preserves_release_and_event_history(self) -> None:
        with self.database.transaction() as connection:
            restore_pre_shadow_schema(connection)
            for name in (
                "learning_artifacts_no_update",
                "learning_artifacts_no_delete",
                "learning_actions_validate_insert",
                "learning_actions_no_update",
                "learning_actions_no_delete",
            ):
                connection.execute(f"DROP TRIGGER IF EXISTS {name}")
            connection.execute("DROP TABLE learning_actions")
            connection.execute("DROP TABLE learning_artifacts")
            connection.execute(
                "UPDATE meta SET value='7' WHERE key='schema_version'"
            )

        with self.database.read() as connection:
            before_events = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM events ORDER BY stream_id, stream_version"
                )
            ]
            before_releases = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM corpus_releases ORDER BY id"
                )
            ]

        self.database.initialize()

        with self.database.read() as connection:
            schema_version = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()["value"]
            after_events = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM events ORDER BY stream_id, stream_version"
                )
            ]
            after_releases = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM corpus_releases ORDER BY id"
                )
            ]
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        self.assertEqual(schema_version, str(SCHEMA_VERSION))
        self.assertEqual(after_events, before_events)
        self.assertEqual(after_releases, before_releases)
        self.assertIn("learning_actions", tables)
        self.assertIn("learning_artifacts", tables)
        self.assertIn("learning_objectives", tables)
        self.assertIn("objective_states", tables)
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_fresh_schema_records_and_lists_a_canonical_action(self) -> None:
        with self.database.read() as connection:
            tables = {
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        self.assertIn("learning_actions", tables)

        action = self.record(
            "answer_revised",
            {"answer_digest": DIGEST_A},
            seconds=2,
            key="fresh-action",
        )
        listed = self.engine.list_actions(self.presentation.decision_id)

        self.assertEqual(len(listed), 1)
        self.assertEqual(action["id"], listed[0]["id"])
        self.assertEqual(action["event_id"], listed[0]["event_id"])
        self.assertEqual(action["decision_id"], self.presentation.decision_id)
        self.assertEqual(action["sequence"], 1)
        self.assertEqual(action["stage"], "unassisted")
        self.assertEqual(action["action_type"], "answer_revised")
        self.assertEqual(
            action["payload"],
            {"answer_digest": DIGEST_A},
        )
        self.assertIsNone(action["artifact"])
        self.assertFalse(action["idempotent_replay"])

    def test_exact_idempotent_retry_returns_the_same_projection(self) -> None:
        payload = {"tool_id": "notes", "purpose_code": "compare_hypotheses"}
        first = self.record(
            "tool_used", payload, seconds=3, key="same-action-command"
        )
        replay = self.record(
            "tool_used", payload, seconds=3, key="same-action-command"
        )

        self.assertEqual(replay["id"], first["id"])
        self.assertEqual(replay["event_id"], first["event_id"])
        self.assertEqual(replay["sequence"], first["sequence"])
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(
            len(self.engine.list_actions(self.presentation.decision_id)), 1
        )
        with self.database.read() as connection:
            event_count = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE idempotency_key='same-action-command'"""
            ).fetchone()["n"]
        self.assertEqual(event_count, 1)

    def test_concurrent_exact_retries_create_one_action_projection(self) -> None:
        barrier = Barrier(2)

        def record_once() -> dict:
            barrier.wait()
            return AdaptiveEngine(Database(self.database.path)).record_action(
                self.presentation.decision_id,
                "answer_revised",
                {"answer_digest": DIGEST_A},
                idempotency_key="parallel-action-command",
                now=START + timedelta(seconds=3),
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: record_once(), range(2)))

        self.assertEqual({result["id"] for result in results}, {results[0]["id"]})
        self.assertEqual(
            {result["event_id"] for result in results}, {results[0]["event_id"]}
        )
        self.assertEqual(
            sorted(result["idempotent_replay"] for result in results),
            [False, True],
        )
        self.assertEqual(
            len(self.engine.list_actions(self.presentation.decision_id)), 1
        )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_idempotency_key_reuse_with_changed_payload_is_rejected(self) -> None:
        self.record(
            "tool_used",
            {"tool_id": "notes", "purpose_code": "compare_hypotheses"},
            seconds=4,
            key="changed-action",
        )

        with self.assertRaisesRegex(ConflictError, "[Ii]dempotency"):
            self.record(
                "tool_used",
                {"tool_id": "documentation", "purpose_code": "lookup"},
                seconds=4,
                key="changed-action",
            )

        self.assertEqual(
            len(self.engine.list_actions(self.presentation.decision_id)), 1
        )

    def test_closed_payload_contract_rejects_raw_unexpected_nonfinite_and_oversize_data(
        self,
    ) -> None:
        check_payload = {
            "check_set_id": "unit-tests",
            "passed": float("nan"),
            "failed": 0,
            "errored": 0,
            "skipped": 0,
            "result_digest": DIGEST_A,
        }
        invalid_commands = (
            (
                "raw payload",
                "answer_revised",
                "the learner's unredacted answer",
            ),
            (
                "unexpected raw field",
                "answer_revised",
                {
                    "answer_digest": DIGEST_A,
                    "raw_answer": "the learner's unredacted answer",
                },
            ),
            ("non-finite counter", "check_run", check_payload),
            (
                "unbounded counter",
                "check_run",
                {
                    **check_payload,
                    "passed": 10**5000,
                },
            ),
            (
                "oversize identifier",
                "abandoned",
                {"reason_code": "x" * 20_000},
            ),
            ("unknown action", "raw_keystroke", {}),
        )

        for label, action_type, payload in invalid_commands:
            with self.subTest(label=label):
                with self.assertRaises(ValidationError):
                    self.engine.record_action(
                        self.presentation.decision_id,
                        action_type,
                        payload,  # type: ignore[arg-type]
                        now=START + timedelta(seconds=2),
                    )

        self.assertEqual(
            self.engine.list_actions(self.presentation.decision_id), []
        )
        with self.database.read() as connection:
            action_events = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type='LearnerActionRecorded'"""
            ).fetchone()["n"]
        self.assertEqual(action_events, 0)

    def test_lifecycle_rejects_duplicate_started_and_actions_after_submission(
        self,
    ) -> None:
        self.record("started", {}, seconds=1)
        self.record(
            "answer_revised", {"answer_digest": DIGEST_A}, seconds=2
        )

        with self.assertRaisesRegex(ValidationError, "lifecycle"):
            self.record("started", {}, seconds=3)

        self.record(
            "submitted", {"submission_digest": DIGEST_B}, seconds=4
        )
        with self.assertRaisesRegex(ValidationError, "lifecycle"):
            self.record(
                "tool_used",
                {"tool_id": "notes", "purpose_code": "revise"},
                seconds=5,
            )

        self.assertEqual(
            [item["action_type"] for item in self.engine.list_actions(
                self.presentation.decision_id
            )],
            ["started", "answer_revised", "submitted"],
        )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_abandoned_trace_is_terminal_and_cannot_be_answered(self) -> None:
        abandoned = self.record(
            "abandoned",
            {"reason_code": "learner_left"},
            seconds=2,
            key="abandon-trace",
        )
        replayed = self.record(
            "abandoned",
            {"reason_code": "learner_left"},
            seconds=2,
            key="abandon-trace",
        )
        self.assertEqual(replayed["id"], abandoned["id"])
        self.assertTrue(replayed["idempotent_replay"])

        with self.assertRaisesRegex(ConflictError, "invalidated"):
            self.record(
                "answer_revised", {"answer_digest": DIGEST_A}, seconds=3
            )
        with self.assertRaisesRegex(ConflictError, "abandoned"):
            self.engine.submit_answer(
                self.presentation.decision_id,
                self.presentation.question.correct_option.id,
                response_ms=1000,
                now=START + timedelta(seconds=4),
            )

        self.assertEqual(
            [item["action_type"] for item in self.engine.list_actions(
                self.presentation.decision_id
            )],
            ["abandoned"],
        )
        replacement = self.engine.next_question(
            self.session["id"], now=START + timedelta(seconds=5)
        )
        self.assertNotEqual(
            replacement.decision_id, self.presentation.decision_id
        )
        with self.database.read() as connection:
            invalidated = connection.execute(
                "SELECT * FROM decisions WHERE id=?",
                (self.presentation.decision_id,),
            ).fetchone()
        self.assertEqual(
            invalidated["invalidation_reason"], "learner_abandoned_trace"
        )
        self.assertIsNotNone(invalidated["invalidated_at"])
        replay_report = ProjectionReplay(self.database).check("action-learner")
        self.assertTrue(replay_report["ok"], replay_report["errors"])
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_submit_answer_rejects_integers_outside_storage_bounds(self) -> None:
        invalid_commands = (
            {"response_ms": 2**100},
            {"hint_count": 2**100},
        )
        for command in invalid_commands:
            with self.subTest(command=command):
                with self.assertRaises(ValidationError):
                    self.engine.submit_answer(
                        self.presentation.decision_id,
                        self.presentation.question.correct_option.id,
                        now=START + timedelta(seconds=2),
                        **command,
                    )

        with self.database.read() as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) AS n FROM attempts").fetchone()[
                    "n"
                ],
                0,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE event_type='ResponseSubmitted'"""
                ).fetchone()["n"],
                0,
            )

    def test_action_trigger_and_verifier_enforce_emergency_revocation_order(
        self,
    ) -> None:
        self.database.revoke_question(
            self.presentation.question.id,
            "Emergency action-boundary test.",
            idempotency_key="revoke-before-action",
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "revocation"):
            self._insert_raw_action(
                "started",
                {},
                stage="unassisted",
                occurred_at=START + timedelta(seconds=2),
            )

        self._insert_raw_action(
            "started",
            {},
            stage="unassisted",
            occurred_at=START + timedelta(seconds=2),
            drop_validation_trigger=True,
        )
        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("emergency revocation" in error for error in report["errors"]),
            report["errors"],
        )

    def test_action_trigger_and_verifier_reject_stale_parallel_decision(
        self,
    ) -> None:
        parallel_session = self.engine.start_session(
            "action-learner", "t_machine_learning", seed=41
        )
        parallel = self.engine.next_question(parallel_session["id"], now=START)
        self.engine.submit_answer(
            parallel.decision_id,
            parallel.question.correct_option.id,
            response_ms=1000,
            now=START + timedelta(seconds=1),
        )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "projection"):
            self._insert_raw_action(
                "started",
                {},
                stage="unassisted",
                occurred_at=START + timedelta(seconds=2),
            )

        self._insert_raw_action(
            "started",
            {},
            stage="unassisted",
            occurred_at=START + timedelta(seconds=2),
            drop_validation_trigger=True,
        )
        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "learner projection advance" in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_integrity_rejects_action_appended_after_session_end(self) -> None:
        self.database.end_session(
            self.session["id"],
            status="completed",
            now=START + timedelta(seconds=1),
        )
        self._insert_raw_action(
            "started",
            {},
            stage="unassisted",
            occurred_at=START + timedelta(seconds=2),
            drop_validation_trigger=True,
        )

        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "session-active interval" in error for error in report["errors"]
            ),
            report["errors"],
        )

    def test_integrity_reduces_the_whole_action_lifecycle(self) -> None:
        self.record("started", {}, seconds=1)
        self._insert_raw_action(
            "started",
            {},
            stage="unassisted",
            occurred_at=START + timedelta(seconds=2),
            drop_validation_trigger=True,
        )

        report = self.database.verify_integrity()

        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "invalid learning-action lifecycle" in error
                and "repeat singleton" in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_artifact_checkpoint_is_content_addressed_and_immutable(self) -> None:
        artifact = {
            "sha256": DIGEST_A,
            "size_bytes": 321,
            "media_type": "text/x-python",
        }
        action = self.engine.record_action(
            self.presentation.decision_id,
            "artifact_checkpoint",
            {
                "artifact_digest": DIGEST_A,
                "artifact_kind": "source_code",
            },
            artifact=artifact,
            now=START + timedelta(seconds=2),
        )

        self.assertEqual(action["artifact"], artifact)
        with self.database.read() as connection:
            stored = connection.execute(
                "SELECT * FROM learning_artifacts WHERE sha256=?", (DIGEST_A,)
            ).fetchone()
        self.assertIsNotNone(stored)
        self.assertEqual(stored["id"], f"art_{DIGEST_A}")
        self.assertEqual(stored["size_bytes"], artifact["size_bytes"])
        self.assertEqual(stored["media_type"], artifact["media_type"])

        with self.assertRaisesRegex(ValidationError, "must match"):
            self.engine.record_action(
                self.presentation.decision_id,
                "artifact_checkpoint",
                {
                    "artifact_digest": DIGEST_B,
                    "artifact_kind": "source_code",
                },
                artifact=artifact,
                now=START + timedelta(seconds=3),
            )
        with self.assertRaisesRegex(ConflictError, "different metadata"):
            self.engine.record_action(
                self.presentation.decision_id,
                "artifact_checkpoint",
                {
                    "artifact_digest": DIGEST_A,
                    "artifact_kind": "source_code",
                },
                artifact={**artifact, "size_bytes": 322},
                now=START + timedelta(seconds=4),
            )

        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE learning_artifacts SET size_bytes=322 WHERE id=?",
                    (stored["id"],),
                )
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    "DELETE FROM learning_artifacts WHERE id=?", (stored["id"],)
                )

        self.assertEqual(
            len(self.engine.list_actions(self.presentation.decision_id)), 1
        )
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM learning_artifacts"
                ).fetchone()["n"],
                1,
            )

    def test_unknown_consumed_and_invalidated_decisions_reject_actions(self) -> None:
        with self.assertRaises(NotFoundError):
            self.record(
                "answer_revised",
                {"answer_digest": DIGEST_A},
                seconds=2,
                decision_id="dec_missing",
            )

        self.engine.submit_answer(
            self.presentation.decision_id,
            self.presentation.question.correct_option.id,
            response_ms=900,
            idempotency_key="consume-action-decision",
            now=START + timedelta(seconds=10),
        )
        with self.assertRaisesRegex(ConflictError, "answered|consumed|pending"):
            self.record(
                "answer_revised", {"answer_digest": DIGEST_B}, seconds=11
            )

        first_session = self.engine.start_session(
            "action-learner", "t_machine_learning", seed=41
        )
        stale_session = self.engine.start_session(
            "action-learner", "t_machine_learning", seed=43
        )
        first = self.engine.next_question(
            first_session["id"], now=START + timedelta(minutes=1)
        )
        stale = self.engine.next_question(
            stale_session["id"], now=START + timedelta(minutes=1)
        )
        self.engine.submit_answer(
            first.decision_id,
            first.question.correct_option.id,
            response_ms=1000,
            idempotency_key="advance-before-stale-action",
            now=START + timedelta(minutes=1, seconds=10),
        )
        replacement = self.engine.next_question(
            stale_session["id"], now=START + timedelta(minutes=1, seconds=20)
        )
        self.assertNotEqual(replacement.decision_id, stale.decision_id)

        with self.assertRaisesRegex(ConflictError, "invalidated|pending|stale"):
            self.engine.record_action(
                stale.decision_id,
                "answer_revised",
                {"answer_digest": DIGEST_C},
                now=START + timedelta(minutes=1, seconds=30),
            )

    def test_sequence_and_occurrence_time_are_monotonic(self) -> None:
        first = self.record(
            "answer_revised", {"answer_digest": DIGEST_A}, seconds=2
        )
        second = self.record(
            "answer_revised", {"answer_digest": DIGEST_B}, seconds=5
        )
        actions = self.engine.list_actions(self.presentation.decision_id)

        self.assertEqual([row["sequence"] for row in actions], [1, 2])
        self.assertEqual(first["sequence"], 1)
        self.assertEqual(second["sequence"], 2)
        self.assertLess(
            datetime.fromisoformat(actions[0]["occurred_at"]),
            datetime.fromisoformat(actions[1]["occurred_at"]),
        )

        with self.assertRaisesRegex(
            ValidationError, "monotonic|out.of.order|before"
        ):
            self.record(
                "answer_revised", {"answer_digest": DIGEST_C}, seconds=4
            )
        self.assertEqual(
            [row["sequence"] for row in self.engine.list_actions(
                self.presentation.decision_id
            )],
            [1, 2],
        )

    def test_recording_actions_does_not_advance_session_or_learner(self) -> None:
        with self.database.read() as connection:
            before_learner = connection.execute(
                "SELECT revision FROM learners WHERE id='action-learner'"
            ).fetchone()["revision"]
            before_session = connection.execute(
                "SELECT revision FROM sessions WHERE id=?", (self.session["id"],)
            ).fetchone()["revision"]
            before_projection = self.database.learner_projection_hash(
                "action-learner", connection
            )

        self.record(
            "answer_revised", {"answer_digest": DIGEST_A}, seconds=2
        )
        self.record(
            "tool_used",
            {"tool_id": "scratchpad", "purpose_code": "reasoning_aid"},
            seconds=3,
        )

        with self.database.read() as connection:
            after_learner = connection.execute(
                "SELECT revision FROM learners WHERE id='action-learner'"
            ).fetchone()["revision"]
            after_session = connection.execute(
                "SELECT revision FROM sessions WHERE id=?", (self.session["id"],)
            ).fetchone()["revision"]
            after_projection = self.database.learner_projection_hash(
                "action-learner", connection
            )
        self.assertEqual(after_learner, before_learner)
        self.assertEqual(after_session, before_session)
        self.assertEqual(after_projection, before_projection)

    def test_hint_actions_are_authoritative_for_submitted_hint_count(self) -> None:
        self.record(
            "hint_requested",
            {"hint_id": "conceptual-1", "level": 1},
            seconds=2,
        )
        self.record(
            "hint_requested",
            {"hint_id": "structural-2", "level": 2},
            seconds=3,
        )

        self.engine.submit_answer(
            self.presentation.decision_id,
            self.presentation.question.correct_option.id,
            # The durable trace, not an untrusted caller counter, is authoritative.
            hint_count=0,
            response_ms=1200,
            idempotency_key="answer-after-traced-hints",
            now=START + timedelta(seconds=10),
        )

        with self.database.read() as connection:
            attempt = connection.execute(
                "SELECT hint_count, event_id FROM attempts WHERE decision_id=?",
                (self.presentation.decision_id,),
            ).fetchone()
            event_payload = json.loads(
                connection.execute(
                    "SELECT payload_json FROM events WHERE event_id=?",
                    (attempt["event_id"],),
                ).fetchone()["payload_json"]
            )
        self.assertEqual(attempt["hint_count"], 2)
        self.assertEqual(event_payload["hint_count"], 2)

    def test_feedback_shown_requires_post_feedback_answer_in_engine_and_trigger(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValidationError, "post_feedback"):
            self.engine.record_action(
                self.presentation.decision_id,
                "feedback_shown",
                {"feedback_digest": DIGEST_A},
                stage="unassisted",
                now=START + timedelta(seconds=2),
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "post_feedback"):
            self._insert_raw_action(
                "feedback_shown",
                {"feedback_digest": DIGEST_A},
                stage="unassisted",
                occurred_at=START + timedelta(seconds=2),
            )

        self.assertEqual(
            self.engine.list_actions(self.presentation.decision_id), []
        )
        self.assertTrue(self.database.verify_integrity()["ok"])

        self._insert_raw_action(
            "feedback_shown",
            {"feedback_digest": DIGEST_A},
            stage="unassisted",
            occurred_at=START + timedelta(seconds=2),
            drop_validation_trigger=True,
        )
        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "feedback_shown action is not post_feedback" in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_post_feedback_actions_require_an_answer_and_remain_observational(
        self,
    ) -> None:
        with self.assertRaisesRegex(ConflictError, "requires an answered"):
            self.engine.record_action(
                self.presentation.decision_id,
                "feedback_shown",
                {"feedback_digest": DIGEST_A},
                stage="post_feedback",
                now=START + timedelta(seconds=2),
            )

        self.engine.submit_answer(
            self.presentation.decision_id,
            self.presentation.question.correct_option.id,
            response_ms=900,
            idempotency_key="answer-before-feedback-action",
            now=START + timedelta(seconds=10),
        )
        with self.database.read() as connection:
            learner_revision = connection.execute(
                "SELECT revision FROM learners WHERE id='action-learner'"
            ).fetchone()["revision"]
            session_revision = connection.execute(
                "SELECT revision FROM sessions WHERE id=?", (self.session["id"],)
            ).fetchone()["revision"]
            projection_hash = self.database.learner_projection_hash(
                "action-learner", connection
            )

        action = self.engine.record_action(
            self.presentation.decision_id,
            "feedback_shown",
            {"feedback_digest": DIGEST_B},
            stage="post_feedback",
            now=START + timedelta(seconds=11),
        )

        self.assertEqual(action["stage"], "post_feedback")
        self.assertEqual(action["sequence"], 1)
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT revision FROM learners WHERE id='action-learner'"
                ).fetchone()["revision"],
                learner_revision,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT revision FROM sessions WHERE id=?",
                    (self.session["id"],),
                ).fetchone()["revision"],
                session_revision,
            )
            self.assertEqual(
                self.database.learner_projection_hash(
                    "action-learner", connection
                ),
                projection_hash,
            )

        behavior = self.engine.session_report(
            self.session["id"], now=START + timedelta(seconds=12)
        )["behavior_trace"]
        self.assertTrue(behavior["observational_only"])
        self.assertEqual(behavior["actions"], 1)
        self.assertEqual(behavior["by_type"], {"feedback_shown": 1})
        self.assertEqual(behavior["by_stage"], {"post_feedback": 1})

    def test_projection_replay_accepts_pre_and_post_answer_action_events(self) -> None:
        self.record(
            "answer_revised", {"answer_digest": DIGEST_A}, seconds=2
        )
        self.engine.submit_answer(
            self.presentation.decision_id,
            self.presentation.question.correct_option.id,
            response_ms=1000,
            idempotency_key="answer-between-action-events",
            now=START + timedelta(seconds=10),
        )
        self.engine.record_action(
            self.presentation.decision_id,
            "feedback_shown",
            {"feedback_digest": DIGEST_B},
            stage="post_feedback",
            now=START + timedelta(seconds=11),
        )
        before_actions = self.engine.list_actions(self.presentation.decision_id)
        before_hash = self.database.learner_projection_hash("action-learner")

        report = ProjectionReplay(self.database).check("action-learner")

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["response_count"], 1)
        self.assertTrue(report["source_projection_matches_replay"])
        self.assertTrue(report["commitment_matches_replay"])
        self.assertTrue(all(item["hash_matches"] for item in report["checkpoints"]))
        self.assertEqual(
            self.engine.list_actions(self.presentation.decision_id), before_actions
        )
        self.assertEqual(
            self.database.learner_projection_hash("action-learner"), before_hash
        )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_action_rows_are_immutable(self) -> None:
        action = self.record(
            "answer_revised", {"answer_digest": DIGEST_A}, seconds=2
        )

        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE learning_actions SET action_type='tool_used' WHERE id=?",
                    (action["id"],),
                )
        with self.assertRaises(sqlite3.IntegrityError):
            with self.database.transaction() as connection:
                connection.execute(
                    "DELETE FROM learning_actions WHERE id=?", (action["id"],)
                )

        listed = self.engine.list_actions(self.presentation.decision_id)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["action_type"], "answer_revised")

    def test_integrity_detects_action_projection_tampering(self) -> None:
        action = self.record(
            "answer_revised", {"answer_digest": DIGEST_A}, seconds=2
        )
        self._drop_action_guards()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE learning_actions SET payload_json=? WHERE id=?",
                ('{"answer_digest":"' + DIGEST_C + '"}', action["id"]),
            )

        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("action" in error.casefold() for error in report["errors"]),
            report["errors"],
        )

    def test_integrity_detects_action_event_tampering(self) -> None:
        action = self.record(
            "answer_revised", {"answer_digest": DIGEST_A}, seconds=2
        )
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            connection.execute(
                "UPDATE events SET payload_json=? WHERE event_id=?",
                (
                    json.dumps(
                        {
                            "action_id": action["id"],
                            "decision_id": self.presentation.decision_id,
                            "sequence": 1,
                            "stage": "unassisted",
                            "action_type": "answer_revised",
                            "payload": {"answer_digest": DIGEST_C},
                            "artifact": None,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    action["event_id"],
                ),
            )

        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "payload hash mismatch" in error
                or "action" in error.casefold()
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_artifact_payload_binding_is_enforced_and_verified(self) -> None:
        artifact = {
            "sha256": DIGEST_B,
            "size_bytes": 19,
            "media_type": "text/plain",
        }
        with self.assertRaisesRegex(sqlite3.IntegrityError, "artifact"):
            self._insert_raw_action(
                "answer_revised",
                {"answer_digest": DIGEST_A},
                stage="unassisted",
                occurred_at=START + timedelta(seconds=2),
                artifact=artifact,
            )

        self._insert_raw_action(
            "answer_revised",
            {"answer_digest": DIGEST_A},
            stage="unassisted",
            occurred_at=START + timedelta(seconds=2),
            artifact=artifact,
            drop_validation_trigger=True,
        )
        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "artifact digest does not match payload" in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_integrity_rejects_backdated_pre_response_event_after_answer(
        self,
    ) -> None:
        self.engine.submit_answer(
            self.presentation.decision_id,
            self.presentation.question.correct_option.id,
            response_ms=1000,
            now=START + timedelta(seconds=10),
        )
        self._insert_raw_action(
            "started",
            {},
            stage="unassisted",
            occurred_at=START + timedelta(seconds=2),
            drop_validation_trigger=True,
        )

        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "pre-response event does not precede response event" in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_post_feedback_event_must_follow_response_event(self) -> None:
        action_id = "act_raw_early_feedback_1"
        payload = {"feedback_digest": DIGEST_A}
        occurred_at = START + timedelta(seconds=11)
        with self.database.transaction() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id=?",
                (self.presentation.decision_id,),
            ).fetchone()
            event = self.database.append_event(
                connection,
                stream_id="learner:action-learner",
                event_type="LearnerActionRecorded",
                schema_version=1,
                payload={
                    "action_id": action_id,
                    "decision_id": self.presentation.decision_id,
                    "sequence": 1,
                    "stage": "post_feedback",
                    "action_type": "feedback_shown",
                    "payload": payload,
                    "artifact": None,
                },
                metadata={
                    "action_schema_version": 1,
                    "observational_only": True,
                    "corpus_release_id": decision["corpus_release_id"],
                },
                learner_id="action-learner",
                session_id=self.session["id"],
                causation_id=self.presentation.decision_id,
                occurred_at=occurred_at,
            )

        self.engine.submit_answer(
            self.presentation.decision_id,
            self.presentation.question.correct_option.id,
            response_ms=1000,
            now=START + timedelta(seconds=10),
        )
        command_material = {
            "decision_id": self.presentation.decision_id,
            "stage": "post_feedback",
            "action_type": "feedback_shown",
            "payload": payload,
            "artifact": None,
        }
        command_hash = hashlib.sha256(
            json.dumps(
                command_material,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

        def insert_projection(*, drop_guard: bool) -> None:
            with self.database.transaction() as connection:
                if drop_guard:
                    connection.execute(
                        "DROP TRIGGER IF EXISTS learning_actions_validate_insert"
                    )
                connection.execute(
                    """INSERT INTO learning_actions(
                           id, event_id, decision_id, session_id, learner_id,
                           sequence, stage, action_type, payload_json, artifact_id,
                           occurred_at, recorded_at, command_hash
                       ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, NULL, ?, ?, ?)""",
                    (
                        action_id,
                        event["event_id"],
                        self.presentation.decision_id,
                        self.session["id"],
                        "action-learner",
                        "post_feedback",
                        "feedback_shown",
                        json.dumps(payload, sort_keys=True, separators=(",", ":")),
                        occurred_at.isoformat(),
                        event["recorded_at"],
                        command_hash,
                    ),
                )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "answered decision"):
            insert_projection(drop_guard=False)
        insert_projection(drop_guard=True)

        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "post-feedback event does not follow response event" in error
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_integrity_reports_naive_action_time_without_crashing(self) -> None:
        action = self.record("started", {}, seconds=2)
        self._drop_action_guards()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE learning_actions SET occurred_at=? WHERE id=?",
                ("2101-05-06T09:00:02", action["id"]),
            )

        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any("timestamp is timezone-naive" in error for error in report["errors"]),
            report["errors"],
        )


if __name__ == "__main__":
    unittest.main()
