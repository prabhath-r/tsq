# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError
from tsq.models import CandidateScore
from tsq.policy_shadow import (
    GREEDY_POLICY_DEFINITION_DIGEST,
    POLICY_SHADOW_EVENT_SCHEMA_VERSION,
    POLICY_SHADOW_PROJECTION_COLUMNS,
    build_policy_shadow_evaluation,
    policy_shadow_logging_probabilities,
)
from tsq.store import (
    SCHEMA_VERSION,
    Database,
    _capture_current_schema_contract,
    _expected_current_schema_contract,
    _expected_v17_schema_contract,
)

from tests.schema_upgrade_helpers import (
    durable_database_fingerprint,
    restore_pre_shadow_schema,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
POLICY_SHADOW_TABLE = "policy_shadow_evaluations"
POLICY_SHADOW_TRIGGERS = frozenset(
    {
        "policy_shadow_evaluations_validate_insert",
        "policy_shadow_evaluations_no_update",
        "policy_shadow_evaluations_no_delete",
    }
)
EVALUATED_AT = datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc)


def schema_contract(database: Database):
    with database.read() as connection:
        return _capture_current_schema_contract(connection)


def event_snapshot(database: Database) -> tuple[tuple[object, ...], ...]:
    with database.read() as connection:
        return tuple(
            tuple(row)
            for row in connection.execute(
                """SELECT * FROM events
                   ORDER BY stream_id, stream_version"""
            )
        )


def historical_data_snapshot(
    database: Database,
) -> tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]:
    """Capture historical rows without prospective empty projections."""

    with database.read() as connection:
        table_names = [
            row["name"]
            for row in connection.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table'
                     AND name NOT LIKE 'sqlite_%'
                     AND name NOT IN (?, ?, ?, ?)
                   ORDER BY name""",
                (
                    POLICY_SHADOW_TABLE,
                    "performance_scoring_reconciliations",
                    "performance_artifact_run_claims",
                    "performance_artifact_run_receipts",
                ),
            ).fetchall()
        ]
        snapshot: list[
            tuple[str, tuple[tuple[object, ...], ...]]
        ] = []
        for table_name in table_names:
            quoted = '"' + table_name.replace('"', '""') + '"'
            rows = [
                tuple(row)
                for row in connection.execute(
                    f"SELECT * FROM {quoted}"
                ).fetchall()
            ]
            if table_name == "meta":
                rows = [
                    row
                    for row in rows
                    if not row or row[0] != "schema_version"
                ]
            snapshot.append(
                (table_name, tuple(sorted(rows, key=repr)))
            )
        return tuple(snapshot)


def remove_tail_policy_shadow_history(
    connection: sqlite3.Connection,
) -> None:
    """Remove prospective v18 shadow tails while preserving the event chain."""

    shadow_events = connection.execute(
        """SELECT event_id, stream_id, stream_version
           FROM events
           WHERE event_type='PolicyShadowEvaluated'
           ORDER BY stream_id, stream_version"""
    ).fetchall()
    if not shadow_events:
        return
    for event in shadow_events:
        successor = connection.execute(
            """SELECT event_id FROM events
               WHERE stream_id=? AND stream_version>?
               ORDER BY stream_version LIMIT 1""",
            (event["stream_id"], event["stream_version"]),
        ).fetchone()
        if successor is not None:
            raise AssertionError(
                "Exact-v17 reconstruction only removes policy-shadow "
                "events at a stream tail."
            )
    guard_rows = connection.execute(
        """SELECT name, sql FROM sqlite_master
           WHERE type='trigger' AND name IN (
               'events_no_delete',
               'policy_shadow_evaluations_no_delete'
           )
           ORDER BY name"""
    ).fetchall()
    if {row["name"] for row in guard_rows} != {
        "events_no_delete",
        "policy_shadow_evaluations_no_delete",
    } or any(not row["sql"] for row in guard_rows):
        raise AssertionError(
            "Exact-v17 reconstruction lacks its immutable delete guards."
        )
    for guard in guard_rows:
        connection.execute(f'DROP TRIGGER "{guard["name"]}"')
    connection.execute("DELETE FROM policy_shadow_evaluations")
    connection.execute(
        "DELETE FROM events WHERE event_type='PolicyShadowEvaluated'"
    )
    for stream_id in sorted(
        {event["stream_id"] for event in shadow_events}
    ):
        tail = connection.execute(
            """SELECT stream_version, payload_hash, recorded_at
               FROM events WHERE stream_id=?
               ORDER BY stream_version DESC LIMIT 1""",
            (stream_id,),
        ).fetchone()
        if tail is None:
            connection.execute(
                "DELETE FROM stream_heads WHERE stream_id=?",
                (stream_id,),
            )
        else:
            updated = connection.execute(
                """UPDATE stream_heads
                   SET stream_version=?, payload_hash=?, updated_at=?
                   WHERE stream_id=?""",
                (
                    tail["stream_version"],
                    tail["payload_hash"],
                    tail["recorded_at"],
                    stream_id,
                ),
            )
            if updated.rowcount != 1:
                raise AssertionError(
                    "Exact-v17 reconstruction could not repair a stream head."
                )
    for guard in guard_rows:
        connection.execute(guard["sql"])


def downgrade_to_exact_v17(database: Database) -> None:
    """Remove only the prospective v18 projection and record v17."""

    with database.transaction() as connection:
        restore_pre_shadow_schema(connection)
        connection.execute(
            """UPDATE meta SET value='17'
               WHERE key='schema_version'"""
        )
    if schema_contract(database) != _expected_v17_schema_contract():
        raise AssertionError(
            "Test fixture is not the exact supported v17 contract."
        )


def build_exact_v17(
    path: Path,
    *,
    with_decision: bool = False,
) -> tuple[Database, str | None]:
    database = Database(path)
    database.initialize()
    decision_id: str | None = None
    if with_decision:
        database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        engine = AdaptiveEngine(database)
        engine.create_learner(
            "schema-v18-learner",
            "Schema v18 Learner",
        )
        session = engine.start_session(
            "schema-v18-learner",
            "t_machine_learning",
            seed=1818,
            now=EVALUATED_AT,
        )
        decision_id = engine.next_question(
            session["id"],
            now=EVALUATED_AT,
        ).decision_id
    else:
        database.ensure_learner(
            "schema-v18-learner",
            "Schema v18 Learner",
        )
    downgrade_to_exact_v17(database)
    return database, decision_id


class PolicyShadowSchemaTests(unittest.TestCase):
    def test_exact_v17_upgrade_adds_only_empty_shadow_projection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exact-v17.db"
            database, decision_id = build_exact_v17(
                path,
                with_decision=True,
            )
            before_data = historical_data_snapshot(database)
            before_events = event_snapshot(database)

            self.assertEqual(SCHEMA_VERSION, 23)
            self.assertIsNotNone(decision_id)
            self.assertEqual(
                schema_contract(database),
                _expected_v17_schema_contract(),
            )

            database.initialize()

            self.assertEqual(
                historical_data_snapshot(database),
                before_data,
            )
            self.assertEqual(event_snapshot(database), before_events)
            self.assertEqual(
                schema_contract(database),
                _expected_current_schema_contract(),
            )
            with database.read() as connection:
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
                shadow_count = connection.execute(
                    "SELECT COUNT(*) FROM policy_shadow_evaluations"
                ).fetchone()[0]
                trigger_names = {
                    row["name"]
                    for row in connection.execute(
                        """SELECT name FROM sqlite_master
                           WHERE type='trigger'
                             AND tbl_name='policy_shadow_evaluations'"""
                    ).fetchall()
                }
            self.assertEqual(version, str(SCHEMA_VERSION))
            self.assertEqual(shadow_count, 0)
            self.assertEqual(trigger_names, POLICY_SHADOW_TRIGGERS)
            database.validate_current_schema()
            integrity = database.verify_integrity()
            self.assertTrue(integrity["ok"], integrity["errors"])

    def test_reopening_current_schema_is_semantically_idempotent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reopen-current.db"
            database, _decision_id = build_exact_v17(path)
            database.initialize()
            before_data = historical_data_snapshot(database)
            before_events = event_snapshot(database)
            before_schema = schema_contract(database)

            database.initialize()

            self.assertEqual(
                historical_data_snapshot(database),
                before_data,
            )
            self.assertEqual(event_snapshot(database), before_events)
            self.assertEqual(schema_contract(database), before_schema)
            Database(path, read_only=True).validate_current_schema()

    def test_tampered_v17_fails_before_any_durable_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered-v17.db"
            database, _decision_id = build_exact_v17(path)
            with database.transaction() as connection:
                connection.execute(
                    """CREATE INDEX unsupported_v17_learner_name
                       ON learners(display_name)"""
                )
            self.assertNotEqual(
                schema_contract(database),
                _expected_v17_schema_contract(),
            )
            before_files = durable_database_fingerprint(path)
            before_data = historical_data_snapshot(database)
            before_events = event_snapshot(database)

            with self.assertRaisesRegex(
                ConflictError,
                (
                    "Schema v17 structure is not the exact supported "
                    "v18 migration source"
                ),
            ):
                database.initialize()

            self.assertEqual(durable_database_fingerprint(path), before_files)
            self.assertEqual(
                historical_data_snapshot(database),
                before_data,
            )
            self.assertEqual(event_snapshot(database), before_events)
            with database.read() as connection:
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
                table = connection.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type='table'
                         AND name='policy_shadow_evaluations'"""
                ).fetchone()
            self.assertEqual(version, "17")
            self.assertIsNone(table)

    def test_v17_schema_change_after_preflight_is_not_promoted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v17-preflight-race.db"
            database, _decision_id = build_exact_v17(path)
            before_data = historical_data_snapshot(database)
            before_events = event_snapshot(database)
            original_connect = Database.connect
            raced = False

            def connect_after_race(instance: Database):
                nonlocal raced
                if instance is database and not instance.read_only and not raced:
                    # Simulate another writer changing the exact-v17 schema
                    # after read-only inspection but before the migration
                    # writer acquires its lock and repeats admission checks.
                    with closing(sqlite3.connect(path)) as competing:
                        competing.execute(
                            """CREATE INDEX unsupported_v17_race_index
                               ON learners(display_name)"""
                        )
                        competing.commit()
                    raced = True
                return original_connect(instance)

            with patch.object(
                Database,
                "connect",
                new=connect_after_race,
            ):
                with self.assertRaisesRegex(
                    ConflictError,
                    (
                        "Schema v17 structure is not the exact supported "
                        "v18 migration source"
                    ),
                ):
                    database.initialize()

            self.assertTrue(raced)
            self.assertEqual(
                historical_data_snapshot(database),
                before_data,
            )
            self.assertEqual(event_snapshot(database), before_events)
            self.assertNotEqual(
                schema_contract(database),
                _expected_v17_schema_contract(),
            )
            with database.read() as connection:
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
                raced_index = connection.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type='index'
                         AND name='unsupported_v17_race_index'"""
                ).fetchone()
                shadow_table = connection.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type='table'
                         AND name='policy_shadow_evaluations'"""
                ).fetchone()
            self.assertEqual(version, "17")
            self.assertIsNotNone(raced_index)
            self.assertIsNone(shadow_table)

    def test_orphaned_pre_v18_shadow_event_is_not_backfilled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orphan-shadow-v17.db"
            database, _decision_id = build_exact_v17(path)
            with database.transaction() as connection:
                event = database.append_event(
                    connection,
                    stream_id="migration:policy-shadow",
                    event_type="PolicyShadowEvaluated",
                    schema_version=1,
                    payload={"legacy": True},
                    metadata={},
                    idempotency_key="legacy-policy-shadow-event",
                    occurred_at=EVALUATED_AT,
                )
            before_events = event_snapshot(database)

            with self.assertRaisesRegex(
                ConflictError,
                (
                    "contains a PolicyShadowEvaluated event without an "
                    "event-backed projection"
                ),
            ):
                database.initialize()

            self.assertEqual(event_snapshot(database), before_events)
            self.assertEqual(before_events[-1][0], event["event_id"])
            self.assertEqual(
                schema_contract(database),
                _expected_v17_schema_contract(),
            )
            with database.read() as connection:
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
            self.assertEqual(version, "17")

    def test_v18_labeled_v17_decision_fails_before_durable_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future-policy-v17.db"
            database, decision_id = build_exact_v17(
                path,
                with_decision=True,
            )
            self.assertIsNotNone(decision_id)
            with database.transaction() as connection:
                connection.execute(
                    """UPDATE decisions SET policy_version=?
                       WHERE id=?""",
                    ("recursive-evidence-graph-v18", decision_id),
                )
            before_files = durable_database_fingerprint(path)
            before_data = historical_data_snapshot(database)
            before_events = event_snapshot(database)

            with self.assertRaisesRegex(
                ConflictError,
                (
                    "decision labeled with the v18 policy without its "
                    "required prospective shadow evidence"
                ),
            ):
                database.initialize()

            self.assertEqual(durable_database_fingerprint(path), before_files)
            self.assertEqual(
                historical_data_snapshot(database),
                before_data,
            )
            self.assertEqual(event_snapshot(database), before_events)
            self.assertEqual(
                schema_contract(database),
                _expected_v17_schema_contract(),
            )
            with database.read() as connection:
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
            self.assertEqual(version, "17")

    def test_shadow_projection_requires_matching_event_and_is_immutable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow-guards.db"
            database = Database(path)
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(database)
            engine.create_learner(
                "shadow-guard-learner",
                "Shadow Guard Learner",
            )
            session = engine.start_session(
                "shadow-guard-learner",
                "t_machine_learning",
                seed=1819,
                now=EVALUATED_AT,
            )
            presentation = engine.next_question(
                session["id"],
                now=EVALUATED_AT,
            )

            with database.transaction() as connection:
                remove_tail_policy_shadow_history(connection)
                decision = connection.execute(
                    "SELECT * FROM decisions WHERE id=?",
                    (presentation.decision_id,),
                ).fetchone()
                selection_event = connection.execute(
                    """SELECT * FROM events
                       WHERE event_type='QuestionSelected'
                         AND json_extract(
                             payload_json, '$.decision_id'
                         )=?
                       ORDER BY stream_version DESC LIMIT 1""",
                    (presentation.decision_id,),
                ).fetchone()
                selection_metadata = json.loads(
                    selection_event["metadata_json"]
                )
                logged = json.loads(
                    decision["top_candidates_json"]
                )
                frontier_scores = tuple(
                    CandidateScore(
                        question_id=candidate["question_id"],
                        **{
                            field: value
                            for field, value in candidate.items()
                            if field != "question_id"
                        },
                    )
                    for candidate in logged[
                        : min(5, decision["candidate_count"])
                    ]
                )
                probabilities = policy_shadow_logging_probabilities(
                    frontier_scores
                )
                built = build_policy_shadow_evaluation(
                    decision_id=decision["id"],
                    logging_policy_version=decision["policy_version"],
                    learner_model_version=selection_metadata[
                        "learner_model_version"
                    ],
                    corpus_release_id=decision["corpus_release_id"],
                    candidate_count=decision["candidate_count"],
                    candidate_digest=decision["candidate_digest"],
                    frontier=tuple(
                        zip(
                            frontier_scores,
                            probabilities,
                            strict=True,
                        )
                    ),
                    live_question_id=decision["question_id"],
                    evaluated_at=selection_event["occurred_at"],
                )
                event = database.append_event(
                    connection,
                    stream_id=selection_event["stream_id"],
                    event_type="PolicyShadowEvaluated",
                    schema_version=POLICY_SHADOW_EVENT_SCHEMA_VERSION,
                    payload=built.payload,
                    metadata=built.metadata,
                    learner_id=selection_event["learner_id"],
                    session_id=selection_event["session_id"],
                    idempotency_key=(
                        "policy-shadow:v1:"
                        + decision["id"]
                        + ":"
                        + GREEDY_POLICY_DEFINITION_DIGEST
                    ),
                    correlation_id=decision["id"],
                    causation_id=selection_event["event_id"],
                    occurred_at=datetime.fromisoformat(
                        selection_event["occurred_at"]
                    ),
                )
                projection = built.projection_row(
                    event_id=event["event_id"],
                    recorded_at=event["recorded_at"],
                )
                insert_sql = (
                    "INSERT INTO policy_shadow_evaluations("
                    + ", ".join(POLICY_SHADOW_PROJECTION_COLUMNS)
                    + ") VALUES ("
                    + ", ".join(
                        f":{column}"
                        for column in POLICY_SHADOW_PROJECTION_COLUMNS
                    )
                    + ")"
                )
                original_propensity = decision["propensity"]
                connection.execute(
                    """UPDATE decisions SET propensity=? WHERE id=?""",
                    (original_propensity / 2.0, decision["id"]),
                )
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "does not match its decision/event",
                ):
                    connection.execute(insert_sql, projection)
                connection.execute(
                    """UPDATE decisions SET propensity=? WHERE id=?""",
                    (original_propensity, decision["id"]),
                )

                original_score_json = decision["selected_score_json"]
                altered_score = json.loads(original_score_json)
                altered_score["predicted_correct"] = (
                    0.99
                    if altered_score["predicted_correct"] < 0.99
                    else 0.01
                )
                connection.execute(
                    """UPDATE decisions SET selected_score_json=? WHERE id=?""",
                    (json.dumps(altered_score, sort_keys=True), decision["id"]),
                )
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "does not match its decision/event",
                ):
                    connection.execute(insert_sql, projection)
                connection.execute(
                    """UPDATE decisions SET selected_score_json=? WHERE id=?""",
                    (original_score_json, decision["id"]),
                )
                connection.execute(insert_sql, projection)
                evaluation_id = built.evaluation_id

            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "policy shadow evaluations are immutable",
            ):
                with database.transaction() as connection:
                    connection.execute(
                        """UPDATE policy_shadow_evaluations
                           SET evaluated_at=evaluated_at
                           WHERE id=?""",
                        (evaluation_id,),
                    )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "policy shadow evaluations are immutable",
            ):
                with database.transaction() as connection:
                    connection.execute(
                        "DELETE FROM policy_shadow_evaluations WHERE id=?",
                        (evaluation_id,),
                    )
            with database.read() as connection:
                stored = connection.execute(
                    """SELECT shadow_only, selection_applied,
                              mastery_applied
                       FROM policy_shadow_evaluations WHERE id=?""",
                    (evaluation_id,),
                ).fetchone()
            self.assertEqual(tuple(stored), (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
