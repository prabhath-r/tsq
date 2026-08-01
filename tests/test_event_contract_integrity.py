# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.learner import LearnerModel
from tsq.replay import ProjectionReplay
from tsq.store import Database, _content_hash
from tsq.versions import (
    CONCEPT_MODEL_VERSION,
    OBJECTIVE_GRID_V7_MODEL_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
START = datetime(2107, 8, 9, 9, 0, tzinfo=timezone.utc)


class EventContractIntegrityTestCase(unittest.TestCase):
    """Adversarial checks for immutable response and projection boundaries."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.base_path = Path(cls.tempdir.name) / "event-contract-base.db"
        database = Database(cls.base_path)
        database.initialize()
        database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )

        engine = AdaptiveEngine(database)
        engine.create_learner("contract-v8")
        session = engine.start_session(
            "contract-v8",
            "t_transformers",
            mode="learn",
            seed=17,
            now=START,
        )
        selected = engine.next_question(session["id"], now=START)
        if selected.question.objective_id is None:
            raise AssertionError("The v8 fixture must select an objective item.")
        engine.submit_answer(
            selected.decision_id,
            selected.question.correct_option.id,
            confidence=1.0,
            response_ms=1,
            hint_count=0,
            feedback_shown=False,
            now=START + timedelta(seconds=1),
        )

        v7 = AdaptiveEngine(
            database, LearnerModel(OBJECTIVE_GRID_V7_MODEL_VERSION)
        )
        v7.create_learner("contract-v7-pending")
        v7_session = v7.start_session(
            "contract-v7-pending",
            "t_transformers",
            mode="learn",
            seed=23,
            now=START + timedelta(days=1),
        )
        v7_selected = v7.next_question(
            v7_session["id"], now=START + timedelta(days=1)
        )
        if v7_selected.question.objective_id is None:
            raise AssertionError("The v7 fixture must select an objective item.")

        engine.create_learner("negative-zero")
        zero_start = START + timedelta(days=2)
        zero_session = engine.start_session(
            "negative-zero",
            "t_transformers",
            mode="learn",
            seed=29,
            now=zero_start,
        )
        zero_selected = engine.next_question(
            zero_session["id"], now=zero_start
        )
        engine.submit_answer(
            zero_selected.decision_id,
            zero_selected.question.correct_option.id,
            confidence=-0.0,
            response_ms=900,
            now=zero_start + timedelta(seconds=1),
        )

        integrity = database.verify_integrity()
        if not integrity["ok"]:
            raise AssertionError(integrity["errors"])
        replay = ProjectionReplay(database).check("contract-v8")
        if not replay["ok"]:
            raise AssertionError(replay["errors"])

    def _copy_base(self, name: str) -> Path:
        target = Path(self.tempdir.name) / f"{name}.db"
        source_connection = sqlite3.connect(self.base_path)
        target_connection = sqlite3.connect(target)
        try:
            source_connection.backup(target_connection)
        finally:
            target_connection.close()
            source_connection.close()
        return target

    @staticmethod
    def _rehash_streams(connection: sqlite3.Connection) -> None:
        streams = [
            row["stream_id"]
            for row in connection.execute(
                "SELECT DISTINCT stream_id FROM events ORDER BY stream_id"
            ).fetchall()
        ]
        for stream_id in streams:
            previous_hash = None
            tail_version = None
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

    def _mutate(
        self,
        path: Path,
        operation: Callable[[sqlite3.Connection], None],
        *,
        rehash: bool = True,
        mutate_attempt: bool = False,
    ) -> None:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        trigger_names = ["events_no_update"]
        if mutate_attempt:
            trigger_names.append("attempts_no_update")
        trigger_sql: list[str] = []
        try:
            for trigger_name in trigger_names:
                row = connection.execute(
                    """SELECT sql FROM sqlite_master
                       WHERE type='trigger' AND name=?""",
                    (trigger_name,),
                ).fetchone()
                self.assertIsNotNone(row, trigger_name)
                trigger_sql.append(row["sql"])
                connection.execute(f'DROP TRIGGER "{trigger_name}"')
            operation(connection)
            if rehash:
                self._rehash_streams(connection)
            for definition in trigger_sql:
                connection.execute(definition)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _event_json(
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        event_type: str,
        column: str = "payload_json",
    ) -> tuple[str, dict]:
        event = connection.execute(
            f"""SELECT event_id, {column} FROM events
                WHERE learner_id=? AND event_type=?""",
            (learner_id, event_type),
        ).fetchone()
        if event is None:
            raise AssertionError(
                f"Missing {event_type} event for {learner_id}."
            )
        return event["event_id"], json.loads(event[column])

    @staticmethod
    def _write_event_json(
        connection: sqlite3.Connection,
        event_id: str,
        value: dict,
        *,
        column: str = "payload_json",
    ) -> None:
        connection.execute(
            f"UPDATE events SET {column}=? WHERE event_id=?",
            (
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                event_id,
            ),
        )

    def _assert_integrity_and_replay_reject(
        self,
        path: Path,
        learner_id: str,
        *,
        integrity_fragment: str | None = None,
    ) -> None:
        database = Database(path)
        integrity = database.verify_integrity()
        self.assertFalse(integrity["ok"], integrity["errors"])
        if integrity_fragment is not None:
            self.assertTrue(
                any(
                    integrity_fragment in error
                    for error in integrity["errors"]
                ),
                integrity["errors"],
            )
        try:
            replay = ProjectionReplay(database).check(learner_id)
        except ValidationError:
            return
        self.assertFalse(replay["ok"], replay["errors"])

    def test_projection_payload_mutations_fail_closed(self) -> None:
        def mutate_projection(
            field: str, replacement: object
        ) -> Callable[[sqlite3.Connection], None]:
            def operation(connection: sqlite3.Connection) -> None:
                event_id, payload = self._event_json(
                    connection,
                    learner_id="contract-v8",
                    event_type="LearnerProjectionAdvanced",
                )
                if field == "state_changes":
                    self.assertNotEqual(payload[field], replacement)
                payload[field] = replacement
                self._write_event_json(connection, event_id, payload)

            return operation

        cases = (
            ("phase", "diagnose", "projection event: phase mismatch"),
            (
                "response_event_id",
                "evt_missing_response",
                "expected one LearnerProjectionAdvanced event",
            ),
            (
                "state_changes",
                [],
                "projection event: state_changes mismatch",
            ),
        )
        for index, (field, replacement, fragment) in enumerate(cases):
            with self.subTest(field=field):
                path = self._copy_base(f"projection-field-{index}")
                self._mutate(
                    path, mutate_projection(field, replacement)
                )
                self._assert_integrity_and_replay_reject(
                    path,
                    "contract-v8",
                    integrity_fragment=fragment,
                )

    def test_coordinated_transition_claims_cannot_diverge_from_session(self) -> None:
        valid_frame = {
            "concept_id": "c_clustering",
            "misconception_id": None,
        }

        def coordinated_phase(connection: sqlite3.Connection) -> None:
            attempt = connection.execute(
                """SELECT id, outcome_json FROM attempts
                   WHERE learner_id='contract-v8'"""
            ).fetchone()
            outcome = json.loads(attempt["outcome_json"])
            outcome["next_phase"] = (
                "diagnose"
                if outcome["next_phase"] != "diagnose"
                else "review"
            )
            connection.execute(
                "UPDATE attempts SET outcome_json=? WHERE id=?",
                (
                    json.dumps(
                        outcome,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    attempt["id"],
                ),
            )
            event_id, projection = self._event_json(
                connection,
                learner_id="contract-v8",
                event_type="LearnerProjectionAdvanced",
            )
            projection["phase"] = outcome["next_phase"]
            self._write_event_json(connection, event_id, projection)

        def coordinated_path(connection: sqlite3.Connection) -> None:
            attempt = connection.execute(
                """SELECT id, outcome_json FROM attempts
                   WHERE learner_id='contract-v8'"""
            ).fetchone()
            outcome = json.loads(attempt["outcome_json"])
            outcome["remediation_depth"] = 1
            outcome["remediation_path"] = [valid_frame]
            connection.execute(
                "UPDATE attempts SET outcome_json=? WHERE id=?",
                (
                    json.dumps(
                        outcome,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    attempt["id"],
                ),
            )
            event_id, projection = self._event_json(
                connection,
                learner_id="contract-v8",
                event_type="LearnerProjectionAdvanced",
            )
            projection["remediation_depth"] = 1
            projection["remediation_path"] = [valid_frame]
            self._write_event_json(connection, event_id, projection)

        for name, operation in (
            ("phase", coordinated_phase),
            ("path-depth", coordinated_path),
        ):
            with self.subTest(field=name):
                path = self._copy_base(f"coordinated-{name}")
                self._mutate(
                    path,
                    operation,
                    mutate_attempt=True,
                )
                self._assert_integrity_and_replay_reject(
                    path,
                    "contract-v8",
                    integrity_fragment="current remediation state",
                )

    def test_json_bool_numeric_collisions_fail_closed(self) -> None:
        cases = (
            ("QuestionSelected", "question_version", True),
            ("QuestionSelected", "session_revision", False),
            ("ResponseSubmitted", "question_version", True),
            ("ResponseSubmitted", "confidence", True),
            ("ResponseSubmitted", "response_ms", True),
            ("ResponseSubmitted", "hint_count", False),
            ("ResponseSubmitted", "feedback_shown", 0),
            ("ResponseSubmitted", "is_correct", 1),
        )
        for index, (event_type, field, replacement) in enumerate(cases):
            with self.subTest(event_type=event_type, field=field):
                path = self._copy_base(f"json-type-{index}")

                def operation(
                    connection: sqlite3.Connection,
                    *,
                    event_type: str = event_type,
                    field: str = field,
                    replacement: object = replacement,
                ) -> None:
                    event_id, payload = self._event_json(
                        connection,
                        learner_id="contract-v8",
                        event_type=event_type,
                    )
                    payload[field] = replacement
                    self._write_event_json(
                        connection, event_id, payload
                    )

                self._mutate(path, operation)
                self._assert_integrity_and_replay_reject(
                    path, "contract-v8"
                )

    def test_objective_selection_rejects_pre_objective_model(self) -> None:
        path = self._copy_base("objective-v4-selection")

        def operation(connection: sqlite3.Connection) -> None:
            event_id, metadata = self._event_json(
                connection,
                learner_id="contract-v7-pending",
                event_type="QuestionSelected",
                column="metadata_json",
            )
            metadata["learner_model_version"] = CONCEPT_MODEL_VERSION
            self._write_event_json(
                connection,
                event_id,
                metadata,
                column="metadata_json",
            )

        self._mutate(path, operation)
        self._assert_integrity_and_replay_reject(
            path,
            "contract-v7-pending",
            integrity_fragment="cannot select objective-aware questions",
        )

    def test_non_finite_and_huge_json_numbers_fail_without_crashing(self) -> None:
        overflow_path = self._copy_base("json-overflow")

        def overflow_operation(connection: sqlite3.Connection) -> None:
            event = connection.execute(
                """SELECT event_id, payload_json FROM events
                   WHERE learner_id='contract-v8'
                     AND event_type='ResponseSubmitted'"""
            ).fetchone()
            mutated = event["payload_json"].replace(
                '"confidence":1.0', '"confidence":1e999'
            )
            self.assertNotEqual(mutated, event["payload_json"])
            connection.execute(
                "UPDATE events SET payload_json=? WHERE event_id=?",
                (mutated, event["event_id"]),
            )

        self._mutate(
            overflow_path, overflow_operation, rehash=False
        )
        self._assert_integrity_and_replay_reject(
            overflow_path,
            "contract-v8",
            integrity_fragment="non-finite JSON number 1e999",
        )

        huge_path = self._copy_base("json-huge-integer")
        huge_integer = 10**400

        def huge_operation(connection: sqlite3.Connection) -> None:
            event_id, payload = self._event_json(
                connection,
                learner_id="contract-v8",
                event_type="QuestionSelected",
            )
            payload["question_version"] = huge_integer
            self._write_event_json(connection, event_id, payload)

        self._mutate(huge_path, huge_operation)
        self._assert_integrity_and_replay_reject(
            huge_path,
            "contract-v8",
            integrity_fragment="question_version mismatch",
        )

    def test_negative_zero_is_canonical_on_the_live_path(self) -> None:
        database = Database(self.base_path)
        with database.read() as connection:
            response = connection.execute(
                """SELECT payload_json FROM events
                   WHERE learner_id='negative-zero'
                     AND event_type='ResponseSubmitted'"""
            ).fetchone()
            attempt = connection.execute(
                """SELECT confidence FROM attempts
                   WHERE learner_id='negative-zero'"""
            ).fetchone()
        self.assertNotIn("-0.0", response["payload_json"])
        self.assertEqual(attempt["confidence"], 0.0)
        integrity = database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])
        replay = ProjectionReplay(database).check("negative-zero")
        self.assertTrue(replay["ok"], replay["errors"])


if __name__ == "__main__":
    unittest.main()
