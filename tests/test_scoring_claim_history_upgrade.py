# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, ValidationError
from tsq.evidence import ActionPhase, EvaluationStatus
from tsq.performance import (
    ImportedCriterionResult,
    ImportedEvaluation,
    ScoringProviderRegistry,
    SyntheticDeterministicProvider,
)
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    read_task_release,
)
from tsq.store import (
    SCHEMA_VERSION,
    Database,
    performance_scoring_claim_event_key,
)

from tests.schema_upgrade_helpers import (
    rehash_event_streams,
    restore_pre_shadow_schema,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
TASK_RELEASE = ROOT / "tests" / "fixtures" / "reviewed_productive_task_release.json"
TASK_RELEASE_CORPUS_PLACEHOLDER = "rel_fixture_requires_explicit_pinning"
START = datetime(2116, 4, 5, 10, 0, tzinfo=timezone.utc)
SUBMISSION_DIGEST = "9" * 64


def _fingerprint_material(lines: list[str]) -> tuple[int, str]:
    material = "\n".join(lines).encode("utf-8")
    return len(material), hashlib.sha256(material).hexdigest()


def _complete_durable_fingerprint(path: Path) -> dict[str, object]:
    """Fingerprint every durable byte plus all schema, data, and event state."""

    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        schema = [
            repr(tuple(row))
            for row in connection.execute(
                """SELECT type, name, tbl_name, sql
                   FROM sqlite_master
                   ORDER BY type, name, tbl_name, sql"""
            )
        ]
        dump = list(connection.iterdump())
        events = [
            repr(tuple(row))
            for row in connection.execute(
                """SELECT * FROM events
                   ORDER BY stream_id, stream_version, event_id"""
            )
        ]
        meta = [
            repr(tuple(row))
            for row in connection.execute(
                "SELECT key, value FROM meta ORDER BY key"
            )
        ]
    finally:
        connection.close()

    files: list[tuple[str, int, str]] = []
    for suffix in ("", "-wal", "-journal"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists() and candidate.stat().st_size:
            material = candidate.read_bytes()
            files.append(
                (
                    suffix or "main",
                    len(material),
                    hashlib.sha256(material).hexdigest(),
                )
            )
    return {
        "durable_files": tuple(files),
        "schema": _fingerprint_material(schema),
        "all_schema_and_data": _fingerprint_material(dump),
        "events": _fingerprint_material(events),
        "meta": _fingerprint_material(meta),
    }


def declared_task_release_fixture(
    corpus_release_id: str,
) -> PerformanceTaskRelease:
    """Bind the portable fixture definition to one imported corpus release."""

    template = read_task_release(TASK_RELEASE)
    if template.corpus_release_id != TASK_RELEASE_CORPUS_PLACEHOLDER:
        raise AssertionError(
            "Productive-task fixture must use its explicit corpus placeholder."
        )
    return replace(template, corpus_release_id=corpus_release_id)


def _registered_evaluation(
    submission_id: str,
    outcome_code: str,
) -> ImportedEvaluation:
    return ImportedEvaluation(
        criteria=(
            ImportedCriterionResult(
                criterion_id="criterion_cli_mask_invariant",
                status=EvaluationStatus.VALID,
                score=0.8,
                outcome_code=outcome_code,
                phase=ActionPhase.UNASSISTED,
                source_action_ids=(submission_id,),
            ),
        )
    )


def _build_two_claim_database(path: Path) -> tuple[Database, dict[str, object]]:
    database = Database(path)
    database.initialize()
    corpus_report = database.import_corpus(
        *read_and_parse(CORPUS, include_catalog=True)
    )
    engine = AdaptiveEngine(database)
    ledger = PerformanceLedger(database)
    release = declared_task_release_fixture(corpus_report["release_id"])
    release_report = ledger.publish_release(release, now=START)
    task = release.tasks[0][1]

    def submitted_attempt(
        learner_id: str,
        seed: int,
        key_prefix: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        engine.create_learner(learner_id, learner_id)
        session = engine.start_session(
            learner_id, "t_transformers", seed=seed
        )
        attempt = ledger.start_attempt(
            session["id"],
            task.id,
            task_version=task.version,
            task_release_id=release_report["release_id"],
            idempotency_key=f"{key_prefix}-start",
            now=START + timedelta(minutes=1),
        )
        submitted = ledger.record_action(
            attempt["id"],
            "submitted",
            {"submission_digest": SUBMISSION_DIGEST},
            idempotency_key=f"{key_prefix}-submit",
            now=START + timedelta(minutes=2),
        )
        return attempt, submitted

    complete_attempt, complete_submission = submitted_attempt(
        "schema-v16-completed", 1601, "schema-v16-completed"
    )
    complete_provider = SyntheticDeterministicProvider(
        _registered_evaluation(
            complete_submission["id"], "schema_v15_completed"
        ),
        provider_id="synthetic.schema-v15-completed-scorer",
    )
    complete_registry = ScoringProviderRegistry(allow_synthetic=True)
    complete_registry.register(
        complete_provider, complete_provider.authority_binding
    )
    completed = ledger.score_attempt(
        complete_attempt["id"],
        complete_registry,
        complete_provider.provider_id,
        complete_provider.provider_version,
        idempotency_key="schema-v15-completed-score",
        now=START + timedelta(minutes=3),
    )

    unfinished_attempt, unfinished_submission = submitted_attempt(
        "schema-v16-unfinished", 1602, "schema-v16-unfinished"
    )

    class FailingProvider(SyntheticDeterministicProvider):
        def score(self, request):
            raise RuntimeError("schema-v15 unfinished callback probe")

    unfinished_provider = FailingProvider(
        _registered_evaluation(
            unfinished_submission["id"], "schema_v15_unfinished"
        ),
        provider_id="synthetic.schema-v15-unfinished-scorer",
    )
    unfinished_registry = ScoringProviderRegistry(allow_synthetic=True)
    unfinished_registry.register(
        unfinished_provider, unfinished_provider.authority_binding
    )
    try:
        ledger.score_attempt(
            unfinished_attempt["id"],
            unfinished_registry,
            unfinished_provider.provider_id,
            unfinished_provider.provider_version,
            idempotency_key="schema-v15-unfinished-score",
            now=START + timedelta(minutes=3),
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("Fixture provider unexpectedly completed scoring.")

    return database, {
        "completed_attempt_id": complete_attempt["id"],
        "completed_evaluation_id": completed["evaluation"]["id"],
        "completed_key": "schema-v15-completed-score",
        "completed_provider_id": complete_provider.provider_id,
        "completed_provider_version": complete_provider.provider_version,
        "completed_submission_id": complete_submission["id"],
        "unfinished_attempt_id": unfinished_attempt["id"],
        "unfinished_key": "schema-v15-unfinished-score",
        "unfinished_provider_id": unfinished_provider.provider_id,
        "unfinished_provider_version": unfinished_provider.provider_version,
        "unfinished_submission_id": unfinished_submission["id"],
    }


def _downgrade_exact_v15(database: Database) -> list[dict[str, object]]:
    """Remove v16-only claim events and restore the exact v15 claim shape."""

    with database.transaction() as connection:
        restore_pre_shadow_schema(connection)
        claims = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM performance_scoring_claims ORDER BY id"
            ).fetchall()
        ]
        for trigger in (
            "events_no_update",
            "events_no_delete",
            "performance_scoring_claims_validate_insert",
            "performance_scoring_claims_no_update",
            "performance_scoring_claims_no_delete",
            "events_respect_performance_scoring_claim",
            "task_evaluations_validate_scoring_claim",
            "task_evaluations_validate_insert",
        ):
            connection.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')

        connection.execute(
            """ALTER TABLE performance_scoring_claims
               RENAME TO _schema_v16_scoring_claims"""
        )
        connection.execute(
            """
            CREATE TABLE performance_scoring_claims (
                id TEXT PRIMARY KEY CHECK(length(id) BETWEEN 1 AND 128),
                idempotency_key TEXT UNIQUE CHECK(
                    idempotency_key IS NULL
                    OR (
                        length(idempotency_key) BETWEEN 1 AND 256
                        AND idempotency_key = trim(idempotency_key)
                    )
                ),
                attempt_id TEXT NOT NULL,
                evaluation_id TEXT NOT NULL UNIQUE CHECK(
                    length(evaluation_id) BETWEEN 1 AND 128
                ),
                through_sequence INTEGER NOT NULL CHECK(through_sequence >= 0),
                provider_id TEXT NOT NULL CHECK(
                    length(trim(provider_id)) BETWEEN 1 AND 128
                ),
                provider_version TEXT NOT NULL CHECK(
                    length(trim(provider_version)) BETWEEN 1 AND 128
                ),
                action_trace_digest TEXT NOT NULL CHECK(
                    length(action_trace_digest) = 64
                    AND action_trace_digest NOT GLOB '*[^0-9a-f]*'
                ),
                command_hash TEXT NOT NULL UNIQUE CHECK(
                    length(command_hash) = 64
                    AND command_hash NOT GLOB '*[^0-9a-f]*'
                ),
                claimed_at TEXT NOT NULL,
                FOREIGN KEY(attempt_id) REFERENCES performance_attempts(id)
                    DEFERRABLE INITIALLY DEFERRED
            )
            """
        )
        connection.executemany(
            """INSERT INTO performance_scoring_claims(
                   id, idempotency_key, attempt_id, evaluation_id,
                   through_sequence, provider_id, provider_version,
                   action_trace_digest, command_hash, claimed_at
               ) VALUES (
                   :id, :idempotency_key, :attempt_id, :evaluation_id,
                   :through_sequence, :provider_id, :provider_version,
                   :action_trace_digest, :command_hash, :claimed_at
               )""",
            claims,
        )
        connection.execute("DROP TABLE _schema_v16_scoring_claims")

        for claim in claims:
            evaluation = connection.execute(
                "SELECT event_id FROM task_evaluations WHERE id=?",
                (claim["evaluation_id"],),
            ).fetchone()
            if evaluation is not None:
                connection.execute(
                    """UPDATE events SET causation_id=?
                       WHERE event_id=?""",
                    (claim["attempt_id"], evaluation["event_id"]),
                )
            connection.execute(
                "DELETE FROM events WHERE event_id=?",
                (claim["event_id"],),
            )

        for stream in connection.execute(
            "SELECT DISTINCT stream_id FROM events ORDER BY stream_id"
        ).fetchall():
            rows = connection.execute(
                """SELECT event_id FROM events
                   WHERE stream_id=? ORDER BY stream_version""",
                (stream["stream_id"],),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """UPDATE events
                       SET stream_version=stream_version+1000000
                       WHERE event_id=?""",
                    (row["event_id"],),
                )
            for stream_version, row in enumerate(rows, start=1):
                connection.execute(
                    "UPDATE events SET stream_version=? WHERE event_id=?",
                    (stream_version, row["event_id"]),
                )
        rehash_event_streams(connection)
        connection.execute(
            "UPDATE meta SET value='15' WHERE key='schema_version'"
        )
    return claims


class ScoringClaimHistoryUpgradeTests(unittest.TestCase):
    def test_exact_v15_migrates_completed_and_unfinished_claims_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "exact-v15.db"
            )
            legacy_claims = _downgrade_exact_v15(database)
            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()["value"],
                    "15",
                )
                self.assertNotIn(
                    "event_id",
                    {
                        row["name"]
                        for row in connection.execute(
                            "PRAGMA table_info(performance_scoring_claims)"
                        )
                    },
                )
                self.assertEqual(
                    connection.execute(
                        """SELECT COUNT(*) AS n FROM events
                           WHERE event_type IN (
                               'PerformanceScoringClaimed',
                               'PerformanceScoringClaimMigrated'
                           )"""
                    ).fetchone()["n"],
                    0,
                )
                self.assertEqual(len(legacy_claims), 2)

            database.initialize()

            self.assertEqual(SCHEMA_VERSION, 22)
            database.validate_current_schema()
            integrity = database.verify_integrity()
            self.assertTrue(integrity["ok"], integrity["errors"])
            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()["value"],
                    str(SCHEMA_VERSION),
                )
                migrated = connection.execute(
                    """SELECT claim.*, event.event_type, event.stream_id,
                              event.learner_id, event.session_id,
                              event.correlation_id, event.causation_id,
                              event.idempotency_key AS event_key,
                              event.occurred_at, event.payload_json,
                              event.metadata_json
                       FROM performance_scoring_claims claim
                       JOIN events event ON event.event_id=claim.event_id
                       ORDER BY claim.id"""
                ).fetchall()
                self.assertEqual(len(migrated), 2)
                self.assertEqual(
                    connection.execute(
                        """SELECT COUNT(*) AS n FROM events
                           WHERE event_type='PerformanceScoringClaimMigrated'"""
                    ).fetchone()["n"],
                    2,
                )
                self.assertEqual(
                    connection.execute(
                        """SELECT COUNT(*) AS n FROM events
                           WHERE event_type='PerformanceScoringClaimed'"""
                    ).fetchone()["n"],
                    0,
                )
                evaluation_ids = {
                    row["id"]
                    for row in connection.execute(
                        "SELECT id FROM task_evaluations"
                    )
                }

            by_id = {claim["id"]: claim for claim in legacy_claims}
            for row in migrated:
                with self.subTest(claim_id=row["id"]):
                    original = by_id[row["id"]]
                    self.assertEqual(
                        {
                            key: row[key]
                            for key in (
                                "id",
                                "idempotency_key",
                                "attempt_id",
                                "evaluation_id",
                                "through_sequence",
                                "provider_id",
                                "provider_version",
                                "action_trace_digest",
                                "command_hash",
                                "claimed_at",
                            )
                        },
                        {
                            key: original[key]
                            for key in (
                                "id",
                                "idempotency_key",
                                "attempt_id",
                                "evaluation_id",
                                "through_sequence",
                                "provider_id",
                                "provider_version",
                                "action_trace_digest",
                                "command_hash",
                                "claimed_at",
                            )
                        },
                    )
                    self.assertEqual(
                        row["event_type"], "PerformanceScoringClaimMigrated"
                    )
                    self.assertEqual(
                        row["event_key"],
                        performance_scoring_claim_event_key(
                            row["command_hash"]
                        ),
                    )
                    self.assertEqual(row["correlation_id"], row["attempt_id"])
                    self.assertEqual(row["causation_id"], row["attempt_id"])
                    self.assertIsNone(row["session_id"])
                    self.assertEqual(row["occurred_at"], row["claimed_at"])
                    payload = json.loads(row["payload_json"])
                    metadata = json.loads(row["metadata_json"])
                    self.assertEqual(
                        payload,
                        {
                            "claim_id": row["id"],
                            "caller_idempotency_key": row[
                                "idempotency_key"
                            ],
                            "attempt_id": row["attempt_id"],
                            "evaluation_id": row["evaluation_id"],
                            "through_sequence": row["through_sequence"],
                            "provider_id": row["provider_id"],
                            "provider_version": row["provider_version"],
                            "action_trace_digest": row[
                                "action_trace_digest"
                            ],
                            "command_hash": row["command_hash"],
                            "claimed_at": row["claimed_at"],
                        },
                    )
                    self.assertEqual(
                        metadata,
                        {
                            "claim_schema_version": 1,
                            "admission_mode": "legacy_projection_migration",
                            "source_schema_version": 15,
                            "shadow_only": True,
                        },
                    )
            self.assertIn(
                fixture["completed_evaluation_id"], evaluation_ids
            )
            self.assertEqual(
                sum(
                    row["evaluation_id"] not in evaluation_ids
                    for row in migrated
                ),
                1,
            )

            class CountingProvider(SyntheticDeterministicProvider):
                calls = 0

                def score(self, request):
                    self.calls += 1
                    return super().score(request)

            complete_provider = CountingProvider(
                _registered_evaluation(
                    fixture["completed_submission_id"],
                    "migration_retry_must_not_run",
                ),
                provider_id=fixture["completed_provider_id"],
                provider_version=fixture["completed_provider_version"],
            )
            unfinished_provider = CountingProvider(
                _registered_evaluation(
                    fixture["unfinished_submission_id"],
                    "migration_retry_must_not_run",
                ),
                provider_id=fixture["unfinished_provider_id"],
                provider_version=fixture["unfinished_provider_version"],
            )
            registry = ScoringProviderRegistry(allow_synthetic=True)
            registry.register(
                complete_provider, complete_provider.authority_binding
            )
            registry.register(
                unfinished_provider, unfinished_provider.authority_binding
            )
            ledger = PerformanceLedger(database)
            replay = ledger.score_attempt(
                fixture["completed_attempt_id"],
                registry,
                complete_provider.provider_id,
                complete_provider.provider_version,
                idempotency_key=fixture["completed_key"],
                now=START + timedelta(minutes=4),
            )
            self.assertTrue(replay["idempotent_replay"])
            for attempt_id, provider, original_key in (
                (
                    fixture["completed_attempt_id"],
                    complete_provider,
                    fixture["completed_key"],
                ),
                (
                    fixture["unfinished_attempt_id"],
                    unfinished_provider,
                    fixture["unfinished_key"],
                ),
            ):
                for retry_key in (
                    original_key
                    if attempt_id == fixture["unfinished_attempt_id"]
                    else f"{original_key}-different",
                    f"{original_key}-other",
                    None,
                ):
                    with self.subTest(
                        attempt_id=attempt_id, retry_key=retry_key
                    ):
                        with self.assertRaisesRegex(
                            ConflictError, "callback will not be repeated"
                        ):
                            ledger.score_attempt(
                                attempt_id,
                                registry,
                                provider.provider_id,
                                provider.provider_version,
                                idempotency_key=retry_key,
                                now=START + timedelta(minutes=4),
                            )
            self.assertEqual(complete_provider.calls, 0)
            self.assertEqual(unfinished_provider.calls, 0)

    def test_v15_offset_claim_survives_calendar_guard_upgrade(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "offset-v15.db"
            )
            _downgrade_exact_v15(database)
            claimed_at = "2117-01-01T00:30:00+01:00"
            with database.transaction() as connection:
                connection.execute(
                    """UPDATE performance_scoring_claims
                       SET claimed_at=? WHERE attempt_id=?""",
                    (claimed_at, fixture["unfinished_attempt_id"]),
                )

            database.initialize()

            database.validate_current_schema()
            integrity = database.verify_integrity()
            self.assertTrue(integrity["ok"], integrity["errors"])
            with database.read() as connection:
                row = connection.execute(
                    """SELECT claim.claimed_at,
                              claim.claim_schema_version,
                              event.event_type, event.occurred_at,
                              event.payload_json, event.metadata_json
                       FROM performance_scoring_claims claim
                       JOIN events event ON event.event_id=claim.event_id
                       WHERE claim.attempt_id=?""",
                    (fixture["unfinished_attempt_id"],),
                ).fetchone()
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["claim_schema_version"], 1)
            self.assertEqual(row["claimed_at"], claimed_at)
            self.assertEqual(
                row["event_type"], "PerformanceScoringClaimMigrated"
            )
            self.assertEqual(
                row["occurred_at"], "2116-12-31T23:30:00+00:00"
            )
            self.assertEqual(
                json.loads(row["payload_json"])["claimed_at"], claimed_at
            )
            self.assertEqual(
                json.loads(row["metadata_json"]),
                {
                    "claim_schema_version": 1,
                    "admission_mode": "legacy_projection_migration",
                    "source_schema_version": 15,
                    "shadow_only": True,
                },
            )

    def test_schema_v15_registered_evaluation_without_claim_fails_closed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-v15-claim.db"
            database, fixture = _build_two_claim_database(path)
            _downgrade_exact_v15(database)
            with database.transaction() as connection:
                connection.execute(
                    """DELETE FROM performance_scoring_claims
                       WHERE evaluation_id=?""",
                    (fixture["completed_evaluation_id"],),
                )

            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()["value"],
                    "15",
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM performance_scoring_claims"
                    ).fetchone()["n"],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        """SELECT COUNT(*) AS n
                           FROM task_evaluations evaluation
                           WHERE json_extract(
                               evaluation.authority_json,
                               '$.normalized_result.normalization_mode'
                           )='registered_provider'
                             AND NOT EXISTS (
                                 SELECT 1
                                 FROM performance_scoring_claims claim
                                 WHERE claim.evaluation_id=evaluation.id
                                   AND claim.attempt_id=evaluation.attempt_id
                                   AND claim.command_hash=evaluation.command_hash
                             )"""
                    ).fetchone()["n"],
                    1,
                )
            before = _complete_durable_fingerprint(path)

            with self.assertRaisesRegex(
                ConflictError,
                "registered-provider evaluation without its required scoring claim",
            ):
                database.initialize()
            self.assertEqual(_complete_durable_fingerprint(path), before)

    def test_exact_v14_registered_evaluation_gets_explicit_exception(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "exact-v14-exception.db"
            )
            _downgrade_exact_v15(database)
            with database.transaction() as connection:
                connection.execute("DROP TABLE performance_scoring_claims")
                connection.execute(
                    "UPDATE meta SET value='14' WHERE key='schema_version'"
                )

            database.initialize()

            database.validate_current_schema()
            integrity = database.verify_integrity()
            self.assertTrue(integrity["ok"], integrity["errors"])
            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM performance_scoring_claims"
                    ).fetchone()["n"],
                    0,
                )
                exemptions = connection.execute(
                    """SELECT * FROM events
                       WHERE event_type='PerformanceScoringLegacyExempted'"""
                ).fetchall()
                self.assertEqual(len(exemptions), 1)
                event = exemptions[0]
                payload = json.loads(event["payload_json"])
                metadata = json.loads(event["metadata_json"])
            self.assertEqual(
                payload,
                {
                    "evaluation_id": fixture["completed_evaluation_id"],
                    "attempt_id": fixture["completed_attempt_id"],
                    "command_hash": payload["command_hash"],
                    "reason": "schema_v14_predates_callback_claims",
                },
            )
            self.assertEqual(
                metadata,
                {
                    "migration_from_schema_version": 14,
                    "shadow_only": True,
                },
            )
            self.assertEqual(
                event["idempotency_key"],
                "performance-score-legacy:v1:"
                + fixture["completed_evaluation_id"],
            )
            self.assertIsNone(event["session_id"])

    def test_v15_claim_migration_does_not_append_inside_ended_sessions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, _fixture = _build_two_claim_database(
                Path(directory) / "ended-v15.db"
            )
            with database.read() as connection:
                session_ids = [
                    row["session_id"]
                    for row in connection.execute(
                        """SELECT DISTINCT session_id
                           FROM performance_attempts ORDER BY session_id"""
                    )
                ]
            for index, session_id in enumerate(session_ids):
                database.end_session(
                    session_id,
                    now=START + timedelta(minutes=10 + index),
                )
            _downgrade_exact_v15(database)

            database.initialize()

            report = database.verify_integrity()
            self.assertTrue(report["ok"], report["errors"])
            with database.read() as connection:
                migrated = connection.execute(
                    """SELECT session_id FROM events
                       WHERE event_type='PerformanceScoringClaimMigrated'"""
                ).fetchall()
                self.assertEqual(len(migrated), 2)
                self.assertTrue(
                    all(row["session_id"] is None for row in migrated)
                )
                for session_id in session_ids:
                    final = connection.execute(
                        """SELECT event_type FROM events
                           WHERE session_id=? ORDER BY stream_version DESC
                           LIMIT 1""",
                        (session_id,),
                    ).fetchone()
                    self.assertEqual(final["event_type"], "SessionEnded")

    def test_v14_exemption_does_not_append_inside_ended_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "ended-v14.db"
            )
            with database.read() as connection:
                session_id = connection.execute(
                    """SELECT session_id FROM performance_attempts
                       WHERE id=?""",
                    (fixture["completed_attempt_id"],),
                ).fetchone()["session_id"]
            database.end_session(
                session_id,
                now=START + timedelta(minutes=10),
            )
            _downgrade_exact_v15(database)
            with database.transaction() as connection:
                connection.execute("DROP TABLE performance_scoring_claims")
                connection.execute(
                    "UPDATE meta SET value='14' WHERE key='schema_version'"
                )

            database.initialize()

            report = database.verify_integrity()
            self.assertTrue(report["ok"], report["errors"])
            with database.read() as connection:
                exemption = connection.execute(
                    """SELECT session_id FROM events
                       WHERE event_type='PerformanceScoringLegacyExempted'
                         AND json_extract(payload_json, '$.evaluation_id')=?""",
                    (fixture["completed_evaluation_id"],),
                ).fetchone()
                final = connection.execute(
                    """SELECT event_type FROM events
                       WHERE session_id=? ORDER BY stream_version DESC
                       LIMIT 1""",
                    (session_id,),
                ).fetchone()
            self.assertIsNone(exemption["session_id"])
            self.assertEqual(final["event_type"], "SessionEnded")


if __name__ == "__main__":
    unittest.main()
