# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from tsq.errors import ConflictError
from tsq.replay import ProjectionReplay
from tsq.store import (
    SCHEMA_VERSION,
    Database,
    _capture_current_schema_contract,
    _expected_current_schema_contract,
    _expected_v16_schema_contract,
)

from tests.test_scoring_claim_history_upgrade import (
    START as SCORING_START,
    _build_two_claim_database,
    _downgrade_exact_v15,
    rehash_event_streams,
)


CLAIM_TRIGGER = "performance_scoring_claims_validate_insert"
V17_SESSION_CLAUSE = """AND (
                          (
                              claim_event.event_type =
                                  'PerformanceScoringClaimed'
                              AND claim_event.session_id = attempt.session_id
                          )
                          OR (
                              claim_event.event_type =
                                  'PerformanceScoringClaimMigrated'
                              AND (
                                  claim_event.session_id IS NULL
                                  OR claim_event.session_id =
                                      attempt.session_id
                              )
                          )
                      )"""
V16_SESSION_CLAUSE = (
    "AND claim_event.session_id = attempt.session_id"
)


def durable_database_fingerprint(path: Path) -> tuple[tuple[str, int, str], ...]:
    """Fingerprint every non-empty durable SQLite file for one database."""

    result: list[tuple[str, int, str]] = []
    # SQLite may create a transient shared-memory coordination file while
    # opening a WAL database read-only. It is not durable database content.
    for suffix in ("", "-wal", "-journal"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists() and candidate.stat().st_size:
            material = candidate.read_bytes()
            result.append(
                (
                    suffix or "main",
                    len(material),
                    hashlib.sha256(material).hexdigest(),
                )
            )
    return tuple(result)


def data_snapshot(
    database: Database,
    *,
    omit_schema_version: bool = False,
) -> tuple[tuple[str, tuple[tuple[object, ...], ...]], ...]:
    """Capture all application table rows independently of row order."""

    with database.read() as connection:
        tables = [
            row["name"]
            for row in connection.execute(
                """SELECT name FROM sqlite_master
                   WHERE type='table' AND name NOT LIKE 'sqlite_%'
                   ORDER BY name"""
            )
        ]
        result: list[tuple[str, tuple[tuple[object, ...], ...]]] = []
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            rows = [
                tuple(row)
                for row in connection.execute(f"SELECT * FROM {quoted}")
            ]
            if omit_schema_version and table == "meta":
                rows = [
                    row
                    for row in rows
                    if not row or row[0] != "schema_version"
                ]
            result.append(
                (table, tuple(sorted(rows, key=repr)))
            )
        return tuple(result)


def event_snapshot(database: Database) -> tuple[tuple[object, ...], ...]:
    """Capture the immutable event history in canonical stream order."""

    with database.read() as connection:
        return tuple(
            tuple(row)
            for row in connection.execute(
                """SELECT * FROM events
                   ORDER BY stream_id, stream_version"""
            )
        )


def schema_contract(database: Database):
    with database.read() as connection:
        return _capture_current_schema_contract(connection)


def claim_trigger_sql(database: Database) -> str:
    with database.read() as connection:
        row = connection.execute(
            """SELECT sql FROM sqlite_master
               WHERE type='trigger' AND name=?""",
            (CLAIM_TRIGGER,),
        ).fetchone()
    if row is None or not row["sql"]:
        raise AssertionError("Scoring-claim trigger is absent.")
    return row["sql"]


def downgrade_to_exact_v16(database: Database) -> None:
    """Reconstruct the one historical v16 structural boundary."""

    current_sql = claim_trigger_sql(database)
    if current_sql.count(V17_SESSION_CLAUSE) != 1:
        raise AssertionError(
            "Current scoring-claim trigger has an unexpected session clause."
        )
    legacy_sql = current_sql.replace(
        V17_SESSION_CLAUSE,
        V16_SESSION_CLAUSE,
    )
    with database.transaction() as connection:
        connection.execute(f'DROP TRIGGER "{CLAIM_TRIGGER}"')
        connection.execute(legacy_sql)
        connection.execute(
            """UPDATE meta SET value='16'
               WHERE key='schema_version'"""
        )


def remove_scoring_claim_fk_deferrability(database: Database) -> str:
    """Rebuild one table with an immediate rather than deferred attempt FK."""

    with database.transaction() as connection:
        row = connection.execute(
            """SELECT sql FROM sqlite_master
               WHERE type='table'
                 AND name='performance_scoring_claims'"""
        ).fetchone()
        if row is None or not row["sql"]:
            raise AssertionError("Fixture lacks the scoring-claims table.")
        canonical_sql = row["sql"]
        deferred_clause = "DEFERRABLE INITIALLY DEFERRED"
        if canonical_sql.upper().count(deferred_clause) != 1:
            raise AssertionError(
                "Scoring-claims table has an unexpected deferred FK shape."
            )
        altered_sql = canonical_sql.replace(
            deferred_clause,
            "",
        )
        for trigger in (
            "performance_scoring_claims_validate_insert",
            "performance_scoring_claims_no_update",
            "performance_scoring_claims_no_delete",
            "events_respect_performance_scoring_claim",
            "task_evaluations_validate_scoring_claim",
            "task_evaluations_validate_insert",
        ):
            connection.execute(f'DROP TRIGGER "{trigger}"')
        connection.execute(
            """ALTER TABLE performance_scoring_claims
               RENAME TO _deferred_scoring_claims"""
        )
        connection.execute(altered_sql)
        connection.execute(
            """INSERT INTO performance_scoring_claims(
                   id, event_id, idempotency_key, attempt_id, evaluation_id,
                   through_sequence, provider_id, provider_version,
                   action_trace_digest, command_hash, claimed_at
               )
               SELECT id, event_id, idempotency_key, attempt_id, evaluation_id,
                      through_sequence, provider_id, provider_version,
                      action_trace_digest, command_hash, claimed_at
               FROM _deferred_scoring_claims"""
        )
        connection.execute("DROP TABLE _deferred_scoring_claims")
        database._install_current_performance_scoring_triggers(connection)
    return altered_sql


def build_exact_v16(path: Path) -> Database:
    """Create a populated v17 database and reconstruct exact v16 structure."""

    database = Database(path)
    database.initialize()
    database.ensure_learner(
        "schema-v17-migration-learner",
        "Schema v17 Migration Learner",
    )
    downgrade_to_exact_v16(database)
    actual = schema_contract(database)
    if actual != _expected_v16_schema_contract():
        raise AssertionError(
            "Test fixture does not match the exact supported v16 contract."
        )
    return database


def bind_migrated_claim_to_historical_session(
    database: Database,
) -> tuple[dict[str, object], ...]:
    """Reconstruct the session envelope emitted by the old v16 migrator."""

    with database.transaction() as connection:
        rows = connection.execute(
            """SELECT event.event_id, event.session_id AS event_session_id,
                      event.stream_version, attempt.id AS attempt_id,
                      attempt.session_id, attempt.learner_id
               FROM performance_scoring_claims claim
               JOIN events event ON event.event_id=claim.event_id
               JOIN performance_attempts attempt
                 ON attempt.id=claim.attempt_id
               WHERE event.event_type='PerformanceScoringClaimMigrated'
               ORDER BY event.event_id"""
        ).fetchall()
        if not rows or any(
            row["event_session_id"] is not None for row in rows
        ):
            raise AssertionError(
                "Lifecycle fixture requires unbound migrated claims."
            )
        trigger = connection.execute(
            """SELECT sql FROM sqlite_master
               WHERE type='trigger' AND name='events_no_update'"""
        ).fetchone()
        if trigger is None or not trigger["sql"]:
            raise AssertionError("Fixture lacks the event immutability guard.")
        events_no_update_sql = trigger["sql"]
        connection.execute("DROP TRIGGER events_no_update")
        connection.executemany(
            """UPDATE events SET session_id=?
               WHERE event_id=?""",
            (
                (row["session_id"], row["event_id"])
                for row in rows
            ),
        )
        rehash_event_streams(connection)
        connection.execute(events_no_update_sql)
        return tuple(dict(row) for row in rows)


def build_session_bound_migrated_v16(
    path: Path,
    *,
    migration_after_session_end: bool,
) -> tuple[Database, dict[str, object]]:
    """Build the valid or buggy historical session-bound v16 shape."""

    database, scoring_fixture = _build_two_claim_database(path)
    attempt_ids = (
        scoring_fixture["completed_attempt_id"],
        scoring_fixture["unfinished_attempt_id"],
    )
    with database.read() as connection:
        attempts = [
            dict(row)
            for row in connection.execute(
                """SELECT id, learner_id, session_id
                   FROM performance_attempts
                   WHERE id IN (?, ?) ORDER BY id""",
                attempt_ids,
            ).fetchall()
        ]
    if len(attempts) != 2:
        raise AssertionError("Lifecycle fixture lacks its two scoring attempts.")
    if migration_after_session_end:
        for index, attempt in enumerate(attempts):
            database.end_session(
                attempt["session_id"],
                idempotency_key=(
                    f"schema-v17-historical-end-{index}"
                ),
                now=SCORING_START + timedelta(minutes=10 + index),
            )
    _downgrade_exact_v15(database)
    database.initialize()
    migrated = bind_migrated_claim_to_historical_session(database)
    if not migration_after_session_end:
        for index, attempt in enumerate(attempts):
            database.end_session(
                attempt["session_id"],
                idempotency_key=(
                    f"schema-v17-historical-end-{index}"
                ),
                now=SCORING_START + timedelta(minutes=10 + index),
            )
    downgrade_to_exact_v16(database)
    actual = schema_contract(database)
    if actual != _expected_v16_schema_contract():
        raise AssertionError(
            "Historical lifecycle fixture is not exact v16 structure."
        )
    target_attempt = next(
        attempt
        for attempt in attempts
        if attempt["id"] == scoring_fixture["completed_attempt_id"]
    )
    target_migration = next(
        row
        for row in migrated
        if row["attempt_id"] == target_attempt["id"]
    )
    return database, {
        **target_attempt,
        "migration_event_id": target_migration["event_id"],
    }


class MigrationEventLifecycleTests(unittest.TestCase):
    def test_exact_v16_upgrade_changes_only_version_and_claim_trigger(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exact-v16.db"
            database = build_exact_v16(path)
            before_data = data_snapshot(
                database,
                omit_schema_version=True,
            )
            before_events = event_snapshot(database)
            before_schema = schema_contract(database)

            self.assertEqual(SCHEMA_VERSION, 17)
            self.assertEqual(
                before_schema,
                _expected_v16_schema_contract(),
            )
            self.assertIn(
                V16_SESSION_CLAUSE,
                claim_trigger_sql(database),
            )
            self.assertNotIn(
                "claim_event.session_id IS NULL",
                claim_trigger_sql(database),
            )

            database.initialize()

            self.assertEqual(
                data_snapshot(database, omit_schema_version=True),
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
            self.assertEqual(version, "17")
            upgraded_trigger = claim_trigger_sql(database)
            self.assertIn(V17_SESSION_CLAUSE, upgraded_trigger)
            self.assertIn(
                "claim_event.session_id IS NULL",
                upgraded_trigger,
            )
            database.validate_current_schema()
            integrity = database.verify_integrity()
            self.assertTrue(integrity["ok"], integrity["errors"])

    def test_reopening_upgraded_v17_is_semantically_idempotent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reopen-v17.db"
            database = build_exact_v16(path)
            database.initialize()
            before_data = data_snapshot(database)
            before_events = event_snapshot(database)
            before_schema = schema_contract(database)

            database.initialize()

            self.assertEqual(data_snapshot(database), before_data)
            self.assertEqual(event_snapshot(database), before_events)
            self.assertEqual(schema_contract(database), before_schema)
            Database(path, read_only=True).validate_current_schema()

    def test_tampered_v16_fails_before_any_durable_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered-v16.db"
            database = build_exact_v16(path)
            legacy_sql = claim_trigger_sql(database)
            tampered_sql = legacy_sql.replace(
                V16_SESSION_CLAUSE,
                (
                    "AND (claim_event.session_id = attempt.session_id "
                    "OR claim_event.session_id IS NULL)"
                ),
            )
            self.assertNotEqual(tampered_sql, legacy_sql)
            with database.transaction() as connection:
                connection.execute(f'DROP TRIGGER "{CLAIM_TRIGGER}"')
                connection.execute(tampered_sql)

            self.assertNotEqual(
                schema_contract(database),
                _expected_v16_schema_contract(),
            )
            before_files = durable_database_fingerprint(path)
            before_data = data_snapshot(database)
            before_trigger = claim_trigger_sql(database)

            with self.assertRaisesRegex(
                ConflictError,
                (
                    "Schema v16 structure is not the exact supported "
                    "v17 migration source"
                ),
            ):
                database.initialize()

            self.assertEqual(durable_database_fingerprint(path), before_files)
            self.assertEqual(data_snapshot(database), before_data)
            self.assertEqual(claim_trigger_sql(database), before_trigger)
            with database.read() as connection:
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
            self.assertEqual(version, "16")

    def test_session_bound_migration_before_end_upgrades_and_replays(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical-valid-v16.db"
            database, fixture = build_session_bound_migrated_v16(
                path,
                migration_after_session_end=False,
            )
            with database.read() as connection:
                boundaries = connection.execute(
                    """SELECT migrated.session_id,
                              migrated.stream_version AS migrated_version,
                              ended.stream_version AS ended_version
                       FROM events migrated
                       JOIN events ended
                         ON ended.session_id=migrated.session_id
                        AND ended.event_type='SessionEnded'
                       WHERE migrated.event_id=?""",
                    (fixture["migration_event_id"],),
                ).fetchone()
            self.assertIsNotNone(boundaries)
            self.assertEqual(
                boundaries["session_id"],
                fixture["session_id"],
            )
            self.assertLess(
                boundaries["migrated_version"],
                boundaries["ended_version"],
            )
            before_events = event_snapshot(database)

            database.initialize()

            self.assertEqual(event_snapshot(database), before_events)
            database.validate_current_schema()
            integrity = database.verify_integrity()
            self.assertTrue(integrity["ok"], integrity["errors"])
            replay = ProjectionReplay(database).check(
                fixture["learner_id"]
            )
            self.assertTrue(replay["ok"], replay["errors"])
            self.assertTrue(replay["rebuild_safe"], replay["errors"])
            self.assertTrue(
                replay["performance_projection_matches_replay"]
            )

    def test_session_bound_migration_after_end_fails_without_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical-invalid-v16.db"
            database, fixture = build_session_bound_migrated_v16(
                path,
                migration_after_session_end=True,
            )
            with database.read() as connection:
                boundaries = connection.execute(
                    """SELECT migrated.session_id,
                              migrated.stream_version AS migrated_version,
                              ended.stream_version AS ended_version
                       FROM events migrated
                       JOIN events ended
                         ON ended.session_id=migrated.session_id
                        AND ended.event_type='SessionEnded'
                       WHERE migrated.event_id=?""",
                    (fixture["migration_event_id"],),
                ).fetchone()
            self.assertIsNotNone(boundaries)
            self.assertEqual(
                boundaries["session_id"],
                fixture["session_id"],
            )
            self.assertGreater(
                boundaries["migrated_version"],
                boundaries["ended_version"],
            )
            before_files = durable_database_fingerprint(path)
            before_data = data_snapshot(database)
            before_schema = schema_contract(database)

            with self.assertRaisesRegex(
                ConflictError,
                (
                    "Schema v16 contains a migration observation after "
                    "SessionEnded"
                ),
            ):
                database.initialize()

            self.assertEqual(durable_database_fingerprint(path), before_files)
            self.assertEqual(data_snapshot(database), before_data)
            self.assertEqual(schema_contract(database), before_schema)
            with database.read() as connection:
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
            self.assertEqual(version, "16")

    def test_v16_schema_change_after_preflight_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v16-preflight-race.db"
            database = build_exact_v16(path)
            before_data = data_snapshot(database)
            original_connect = Database.connect
            raced = False
            tampered_sql: str | None = None

            def connect_after_race(instance: Database):
                nonlocal raced, tampered_sql
                if instance is database and not instance.read_only and not raced:
                    # Database.initialize() has completed its read-only exact-v16
                    # preflight when it asks the original writable object for
                    # this connection. Simulate a second process changing the
                    # schema in that gap.
                    with closing(sqlite3.connect(path)) as competing:
                        row = competing.execute(
                            """SELECT sql FROM sqlite_master
                               WHERE type='trigger' AND name=?""",
                            (CLAIM_TRIGGER,),
                        ).fetchone()
                        if row is None or not row[0]:
                            raise AssertionError(
                                "Race fixture lacks the scoring-claim trigger."
                            )
                        tampered_sql = row[0].replace(
                            V16_SESSION_CLAUSE,
                            (
                                "AND (claim_event.session_id = "
                                "attempt.session_id OR "
                                "claim_event.session_id IS NULL)"
                            ),
                        )
                        if tampered_sql == row[0]:
                            raise AssertionError(
                                "Race fixture did not alter the v16 trigger."
                            )
                        competing.execute(
                            f'DROP TRIGGER "{CLAIM_TRIGGER}"'
                        )
                        competing.execute(tampered_sql)
                        competing.commit()
                    raced = True
                return original_connect(instance)

            with patch.object(
                Database,
                "connect",
                new=connect_after_race,
            ):
                with self.assertRaises(ConflictError):
                    database.initialize()

            self.assertTrue(raced)
            self.assertIsNotNone(tampered_sql)
            self.assertEqual(data_snapshot(database), before_data)
            self.assertEqual(claim_trigger_sql(database), tampered_sql)
            self.assertNotEqual(
                schema_contract(database),
                _expected_v16_schema_contract(),
            )
            with database.read() as connection:
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
            self.assertEqual(version, "16")

    def test_v17_schema_change_before_safety_writer_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v17-safety-writer-race.db"
            database = build_exact_v16(path)
            database.initialize()
            before_data = data_snapshot(database)
            before_events = event_snapshot(database)
            original_connect = Database.connect
            raced = False
            tampered_sql: str | None = None

            def connect_after_validation(instance: Database):
                nonlocal raced, tampered_sql
                if instance is database and not instance.read_only and not raced:
                    # A current-schema reopen reaches this writable connection
                    # only after its independent read-only structural
                    # validation, when generated-content safety enforcement is
                    # about to begin.
                    with closing(sqlite3.connect(path)) as competing:
                        row = competing.execute(
                            """SELECT sql FROM sqlite_master
                               WHERE type='trigger' AND name=?""",
                            (CLAIM_TRIGGER,),
                        ).fetchone()
                        if row is None or not row[0]:
                            raise AssertionError(
                                "Race fixture lacks the scoring-claim trigger."
                            )
                        tampered_sql = row[0].replace(
                            V17_SESSION_CLAUSE,
                            "AND 1 = 1",
                        )
                        if tampered_sql == row[0]:
                            raise AssertionError(
                                "Race fixture did not alter the v17 trigger."
                            )
                        competing.execute(
                            f'DROP TRIGGER "{CLAIM_TRIGGER}"'
                        )
                        competing.execute(tampered_sql)
                        competing.commit()
                    raced = True
                return original_connect(instance)

            with patch.object(
                Database,
                "connect",
                new=connect_after_validation,
            ):
                with self.assertRaises(ConflictError):
                    database.initialize()

            self.assertTrue(raced)
            self.assertIsNotNone(tampered_sql)
            self.assertEqual(data_snapshot(database), before_data)
            self.assertEqual(event_snapshot(database), before_events)
            self.assertEqual(claim_trigger_sql(database), tampered_sql)
            self.assertNotEqual(
                schema_contract(database),
                _expected_current_schema_contract(),
            )
            with database.read() as connection:
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
            self.assertEqual(version, "17")

    def test_scoring_claim_fk_deferrability_is_part_of_schema_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "immediate-claim-fk.db"
            database = build_exact_v16(path)
            database.initialize()
            altered_sql = remove_scoring_claim_fk_deferrability(database)
            self.assertNotIn(
                "DEFERRABLE INITIALLY DEFERRED",
                altered_sql.upper(),
            )
            before_data = data_snapshot(database)
            before_events = event_snapshot(database)

            with self.assertRaises(ConflictError):
                database.validate_current_schema()
            with self.assertRaises(ConflictError):
                database.initialize()

            self.assertEqual(data_snapshot(database), before_data)
            self.assertEqual(event_snapshot(database), before_events)
            with database.read() as connection:
                row = connection.execute(
                    """SELECT sql FROM sqlite_master
                       WHERE type='table'
                         AND name='performance_scoring_claims'"""
                ).fetchone()
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
            self.assertIsNotNone(row)
            self.assertNotIn(
                "DEFERRABLE INITIALLY DEFERRED",
                row["sql"].upper(),
            )
            self.assertEqual(version, "17")

    def test_current_v17_fk_corruption_fails_before_safety_writes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current-v17-fk-corrupt.db"
            database = Database(path)
            database.initialize()
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("PRAGMA foreign_keys=OFF")
                connection.execute(
                    """INSERT INTO concept_edges(
                           source_id, target_id, relation, weight
                       ) VALUES(
                           'missing-source', 'missing-target',
                           'prerequisite', 1.0
                       )"""
                )
                connection.commit()

            Database(path, read_only=True).validate_current_schema()
            before_files = durable_database_fingerprint(path)
            before_data = data_snapshot(database)
            before_events = event_snapshot(database)
            before_schema = schema_contract(database)
            with database.read() as connection:
                before_violations = tuple(
                    tuple(row)
                    for row in connection.execute(
                        "PRAGMA foreign_key_check"
                    )
                )
            self.assertEqual(len(before_violations), 2)

            with patch.object(
                Database,
                "_revoke_historically_active_unreviewed_generated_questions",
                side_effect=AssertionError(
                    "safety writer ran before foreign-key rejection"
                ),
            ):
                with self.assertRaises(ConflictError):
                    database.initialize()

            self.assertEqual(durable_database_fingerprint(path), before_files)
            self.assertEqual(data_snapshot(database), before_data)
            self.assertEqual(event_snapshot(database), before_events)
            self.assertEqual(schema_contract(database), before_schema)
            with database.read() as connection:
                after_violations = tuple(
                    tuple(row)
                    for row in connection.execute(
                        "PRAGMA foreign_key_check"
                    )
                )
                version = connection.execute(
                    """SELECT value FROM meta
                       WHERE key='schema_version'"""
                ).fetchone()["value"]
            self.assertEqual(after_violations, before_violations)
            self.assertEqual(version, "17")


if __name__ == "__main__":
    unittest.main()
