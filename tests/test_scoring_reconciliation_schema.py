# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from tsq.evidence import canonical_json
from tsq.errors import ConflictError
from tsq.performance_ledger import PerformanceLedger
from tsq.reconciliation import (
    ReconciliationObservation,
    ReconciliationOutcome,
    ScoringReconciliationReceipt,
    ScoringReconciliationRegistry,
    ScoringReconciliationRequest,
    SyntheticReconciliationAdapter,
)
from tsq.store import (
    Database,
    _capture_current_schema_contract,
    _expected_current_schema_contract,
    _expected_v18_schema_contract,
    from_timestamp,
    performance_scoring_claim_payload,
    to_timestamp,
)

from tests.test_scoring_claim_history_upgrade import (
    _build_two_claim_database,
    _registered_evaluation,
    rehash_event_streams,
)
from tests.test_migration_event_lifecycle import durable_database_fingerprint


_RECONCILER_ID = "synthetic.schema-v19-reconciler"
_RECONCILER_VERSION = "test-v1"


def _schema_contract(database: Database):
    with database.read() as connection:
        return _capture_current_schema_contract(connection)


def _events(database: Database) -> tuple[tuple[object, ...], ...]:
    with database.read() as connection:
        return tuple(
            tuple(row)
            for row in connection.execute(
                """SELECT * FROM events
                   ORDER BY stream_id, stream_version"""
            )
        )


def _drop_current_scoring_guards(
    connection: sqlite3.Connection,
) -> None:
    for trigger in (
        "performance_scoring_claims_validate_insert",
        "performance_scoring_claims_no_update",
        "performance_scoring_claims_no_delete",
        "events_respect_performance_scoring_claim",
        "performance_scoring_reconciliations_validate_insert",
        "performance_scoring_reconciliations_no_update",
        "performance_scoring_reconciliations_no_delete",
        "events_respect_performance_scoring_reconciliation",
        "task_evaluations_validate_scoring_claim",
        "task_evaluations_validate_insert",
        "shadow_evidence_bundles_validate_insert",
    ):
        connection.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')


def _downgrade_to_exact_v18(database: Database) -> None:
    """Construct the exact historical source without inventing v18 claims."""

    with database.transaction() as connection:
        Database._downgrade_v20_contract_to_v19(connection)
        if connection.execute(
            "SELECT 1 FROM performance_scoring_reconciliations LIMIT 1"
        ).fetchone() is not None:
            raise AssertionError("v18 fixture requires no reconciliation rows")
        claims = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM performance_scoring_claims ORDER BY id"
            )
        ]
        event_guard = connection.execute(
            """SELECT sql FROM sqlite_master
               WHERE type='trigger' AND name='events_no_update'"""
        ).fetchone()
        if event_guard is None or not event_guard["sql"]:
            raise AssertionError("fixture lacks the immutable event guard")
        connection.execute('DROP TRIGGER "events_no_update"')
        _drop_current_scoring_guards(connection)

        for claim in claims:
            payload = performance_scoring_claim_payload(
                claim_id=claim["id"],
                caller_idempotency_key=claim["idempotency_key"],
                attempt_id=claim["attempt_id"],
                evaluation_id=claim["evaluation_id"],
                through_sequence=claim["through_sequence"],
                provider_id=claim["provider_id"],
                provider_version=claim["provider_version"],
                action_trace_digest_value=claim["action_trace_digest"],
                command_hash=claim["command_hash"],
                claimed_at=claim["claimed_at"],
            )
            connection.execute(
                """UPDATE events
                   SET schema_version=1, payload_json=?, metadata_json=?
                   WHERE event_id=?""",
                (
                    canonical_json(payload),
                    canonical_json(
                        {
                            "claim_schema_version": 1,
                            "admission_mode": "pre_callback",
                            "source_schema_version": None,
                            "shadow_only": True,
                        }
                    ),
                    claim["event_id"],
                ),
            )
        rehash_event_streams(connection)

        connection.execute(
            "DROP TABLE performance_scoring_reconciliations"
        )
        connection.execute(
            """ALTER TABLE performance_scoring_claims
               RENAME TO _schema_v19_scoring_claims"""
        )
        Database._create_v16_scoring_claim_table(connection)
        connection.executemany(
            """INSERT INTO performance_scoring_claims(
                   id, event_id, idempotency_key, attempt_id, evaluation_id,
                   through_sequence, provider_id, provider_version,
                   action_trace_digest, command_hash, claimed_at
               ) VALUES (
                   :id, :event_id, :idempotency_key, :attempt_id,
                   :evaluation_id, :through_sequence, :provider_id,
                   :provider_version, :action_trace_digest, :command_hash,
                   :claimed_at
               )""",
            claims,
        )
        connection.execute("DROP TABLE _schema_v19_scoring_claims")
        Database._install_v18_performance_scoring_triggers(connection)
        Database._install_v18_shadow_evidence_bundle_trigger(connection)
        connection.execute(event_guard["sql"])
        connection.execute(
            "UPDATE meta SET value='18' WHERE key='schema_version'"
        )
    if _schema_contract(database) != _expected_v18_schema_contract():
        raise AssertionError("fixture is not the exact supported v18 schema")


def _empty_v18(path: Path) -> Database:
    database = Database(path)
    database.initialize()
    with database.transaction() as connection:
        database._downgrade_v19_contract_to_v18(connection)
        connection.execute(
            "UPDATE meta SET value='18' WHERE key='schema_version'"
        )
    if _schema_contract(database) != _expected_v18_schema_contract():
        raise AssertionError("empty fixture is not exact schema v18")
    return database


def _request(row: sqlite3.Row) -> ScoringReconciliationRequest:
    return ScoringReconciliationRequest(
        claim_id=row["id"],
        attempt_id=row["attempt_id"],
        evaluation_id=row["evaluation_id"],
        through_sequence=row["through_sequence"],
        provider_id=row["provider_id"],
        provider_version=row["provider_version"],
        action_trace_digest=row["action_trace_digest"],
        command_hash=row["command_hash"],
        scoring_request_digest=row["scoring_request_digest"],
        provider_binding_digest=row["provider_binding_digest"],
        provider_operation_digest=row["provider_operation_digest"],
    )


def _registry(
    row: sqlite3.Row,
    *,
    outcome: ReconciliationOutcome,
    observed_offset: int,
    digest_character: str,
    imported=None,
) -> tuple[ScoringReconciliationRegistry, object]:
    request = _request(row)
    observed = from_timestamp(row["claimed_at"]) + timedelta(
        minutes=observed_offset
    )
    completed = (
        observed - timedelta(minutes=1)
        if outcome is ReconciliationOutcome.COMPLETED
        else None
    )
    receipt = ScoringReconciliationReceipt(
        claim_id=request.claim_id,
        attempt_id=request.attempt_id,
        evaluation_id=request.evaluation_id,
        through_sequence=request.through_sequence,
        provider_id=request.provider_id,
        provider_version=request.provider_version,
        reconciler_id=_RECONCILER_ID,
        reconciler_version=_RECONCILER_VERSION,
        action_trace_digest=request.action_trace_digest,
        command_hash=request.command_hash,
        scoring_request_digest=request.scoring_request_digest,
        provider_binding_digest=request.provider_binding_digest,
        outcome=outcome,
        observed_at=to_timestamp(observed),
        completed_at=None if completed is None else to_timestamp(completed),
        result_digest=(
            None
            if outcome is not ReconciliationOutcome.COMPLETED
            else imported.digest
        ),
        reason_code={
            ReconciliationOutcome.UNKNOWN: "provider_lookup_ambiguous",
            ReconciliationOutcome.DEFINITELY_ABSENT: (
                "provider_operation_never_accepted"
            ),
            ReconciliationOutcome.COMPLETED: "provider_result_recovered",
        }[outcome],
        provider_operation_digest=request.provider_operation_digest,
        provider_receipt_digest=digest_character * 64,
        attestation_digest=digest_character.swapcase().lower() * 64,
    )
    observation = ReconciliationObservation(
        receipt=receipt,
        imported_evaluation=imported,
    )
    adapter = SyntheticReconciliationAdapter(
        observation,
        reconciler_id=_RECONCILER_ID,
        reconciler_version=_RECONCILER_VERSION,
        can_prove_absence=(
            outcome is ReconciliationOutcome.DEFINITELY_ABSENT
        ),
    )
    registry = ScoringReconciliationRegistry(allow_synthetic=True)
    registry.register(adapter, adapter.authority_binding)
    return registry, observed


def _claim_row(database: Database, attempt_id: str) -> sqlite3.Row:
    with database.read() as connection:
        row = connection.execute(
            """SELECT * FROM performance_scoring_claims
               WHERE attempt_id=?""",
            (attempt_id,),
        ).fetchone()
    if row is None:
        raise AssertionError("claim fixture was not created")
    return row


def _fake_reconciliation_values(
    claim: sqlite3.Row,
    *,
    event_id: str,
    idempotency_key: str | None = None,
    command_character: str = "c",
) -> tuple[object, ...]:
    return (
        "reconciliation_fake_" + command_character,
        event_id,
        idempotency_key,
        claim["id"],
        claim["attempt_id"],
        claim["evaluation_id"],
        "unknown",
        claim["scoring_request_digest"],
        claim["provider_binding_digest"],
        claim["provider_operation_digest"],
        _RECONCILER_ID,
        _RECONCILER_VERSION,
        "b" * 64,
        "{}",
        "d" * 64,
        None,
        claim["claimed_at"],
        command_character * 64,
    )


_FAKE_RECONCILIATION_INSERT = """
    INSERT INTO performance_scoring_reconciliations(
        id, event_id, idempotency_key, claim_id, attempt_id, evaluation_id,
        outcome, scoring_request_digest, provider_binding_digest,
        provider_operation_digest, reconciler_id, reconciler_version,
        reconciliation_binding_digest, receipt_json, receipt_digest,
        normalized_result_digest, reconciled_at, command_hash
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_CLAIM_REINSERT = """
    INSERT INTO performance_scoring_claims(
        id, event_id, claim_schema_version, idempotency_key, attempt_id,
        evaluation_id, through_sequence, provider_id, provider_version,
        action_trace_digest, scoring_request_digest, provider_binding_digest,
        provider_operation_digest, command_hash, claimed_at
    ) VALUES (
        :id, :event_id, :claim_schema_version, :idempotency_key, :attempt_id,
        :evaluation_id, :through_sequence, :provider_id, :provider_version,
        :action_trace_digest, :scoring_request_digest,
        :provider_binding_digest, :provider_operation_digest, :command_hash,
        :claimed_at
    )
"""

_RECONCILIATION_REINSERT = """
    INSERT INTO performance_scoring_reconciliations(
        id, event_id, idempotency_key, claim_id, attempt_id, evaluation_id,
        outcome, scoring_request_digest, provider_binding_digest,
        provider_operation_digest, reconciler_id, reconciler_version,
        reconciliation_binding_digest, receipt_json, receipt_digest,
        normalized_result_digest, reconciled_at, command_hash
    ) VALUES (
        :id, :event_id, :idempotency_key, :claim_id, :attempt_id,
        :evaluation_id, :outcome, :scoring_request_digest,
        :provider_binding_digest, :provider_operation_digest, :reconciler_id,
        :reconciler_version, :reconciliation_binding_digest, :receipt_json,
        :receipt_digest, :normalized_result_digest, :reconciled_at,
        :command_hash
    )
"""


class ScoringReconciliationUpgradeTests(unittest.TestCase):
    def test_exact_v18_claim_bytes_migrate_as_v1_without_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, _fixture = _build_two_claim_database(
                Path(directory) / "claims-v18.db"
            )
            _downgrade_to_exact_v18(database)
            with database.read() as connection:
                before_claims = tuple(
                    tuple(row)
                    for row in connection.execute(
                        """SELECT * FROM performance_scoring_claims
                           ORDER BY id"""
                    )
                )
            before_events = _events(database)

            database.initialize()

            self.assertEqual(_events(database), before_events)
            self.assertEqual(
                _schema_contract(database),
                _expected_current_schema_contract(),
            )
            with database.read() as connection:
                version = connection.execute(
                    "SELECT value FROM meta WHERE key='schema_version'"
                ).fetchone()["value"]
                rows = connection.execute(
                    """SELECT id, event_id, idempotency_key, attempt_id,
                              evaluation_id, through_sequence, provider_id,
                              provider_version, action_trace_digest,
                              command_hash, claimed_at,
                              claim_schema_version,
                              scoring_request_digest,
                              provider_binding_digest,
                              provider_operation_digest
                       FROM performance_scoring_claims ORDER BY id"""
                ).fetchall()
                reconciliations = connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_scoring_reconciliations"""
                ).fetchone()["n"]
            self.assertEqual(version, "20")
            self.assertEqual(
                tuple(tuple(row)[:11] for row in rows),
                before_claims,
            )
            self.assertTrue(
                all(tuple(row)[11:] == (1, None, None, None) for row in rows)
            )
            self.assertEqual(reconciliations, 0)
            integrity = database.verify_integrity()
            self.assertTrue(integrity["ok"], integrity["errors"])
            migrated_v1 = _claim_row(
                database, _fixture["unfinished_attempt_id"]
            )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "does not match its claim/event/receipt",
            ), database.transaction() as connection:
                connection.execute(
                    _FAKE_RECONCILIATION_INSERT,
                    _fake_reconciliation_values(
                        migrated_v1,
                        event_id=migrated_v1["event_id"],
                        command_character="a",
                    ),
                )

    def test_tampered_v18_fails_before_durable_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered-v18.db"
            database = _empty_v18(path)
            with database.transaction() as connection:
                connection.execute(
                    """CREATE INDEX unsupported_v18_learner_name
                       ON learners(display_name)"""
                )
            before = durable_database_fingerprint(path)

            with self.assertRaisesRegex(
                ConflictError,
                "exact supported v19 migration source",
            ):
                database.initialize()

            self.assertEqual(durable_database_fingerprint(path), before)
            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()["value"],
                    "18",
                )
                self.assertIsNone(
                    connection.execute(
                        """SELECT 1 FROM sqlite_master
                           WHERE type='table' AND name=
                                 'performance_scoring_reconciliations'"""
                    ).fetchone()
                )

    def test_v18_writer_race_is_rechecked_before_migration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "race-v18.db"
            database = _empty_v18(path)
            original_connect = Database.connect
            raced = False

            def connect_after_race(instance: Database):
                nonlocal raced
                if instance is database and not instance.read_only and not raced:
                    with closing(sqlite3.connect(path)) as competing:
                        competing.execute(
                            """CREATE INDEX unsupported_v18_race
                               ON learners(display_name)"""
                        )
                        competing.commit()
                    raced = True
                return original_connect(instance)

            with patch.object(Database, "connect", new=connect_after_race):
                with self.assertRaisesRegex(
                    ConflictError,
                    "exact supported v19 migration source",
                ):
                    database.initialize()

            self.assertTrue(raced)
            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()["value"],
                    "18",
                )
                self.assertIsNotNone(
                    connection.execute(
                        """SELECT 1 FROM sqlite_master
                           WHERE type='index'
                             AND name='unsupported_v18_race'"""
                    ).fetchone()
                )

    def test_pre_v19_reconciliation_event_is_rejected_without_backfill(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orphan-reconciliation.db"
            database = _empty_v18(path)
            with database.transaction() as connection:
                database.append_event(
                    connection,
                    stream_id="learner:orphan-reconciliation",
                    event_type="PerformanceScoringReconciled",
                    schema_version=1,
                    payload={"historical": True},
                    metadata={"shadow_only": True},
                    learner_id="orphan-reconciliation",
                    session_id=None,
                )
            before_events = _events(database)

            with self.assertRaisesRegex(
                ConflictError,
                "PerformanceScoringReconciled event",
            ):
                database.initialize()

            self.assertEqual(_events(database), before_events)
            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()["value"],
                    "18",
                )
                self.assertIsNone(
                    connection.execute(
                        """SELECT 1 FROM sqlite_master
                           WHERE type='table' AND name=
                                 'performance_scoring_reconciliations'"""
                    ).fetchone()
                )


class ScoringReconciliationGuardTests(unittest.TestCase):
    def _assert_claim_envelope_rejected(
        self,
        database: Database,
        claim: sqlite3.Row,
        mutation,
    ) -> None:
        connection = database.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DROP TRIGGER performance_scoring_claims_no_delete"
            )
            connection.execute("DROP TRIGGER events_no_update")
            connection.execute(
                "DELETE FROM performance_scoring_claims WHERE id=?",
                (claim["id"],),
            )
            event = connection.execute(
                "SELECT * FROM events WHERE event_id=?",
                (claim["event_id"],),
            ).fetchone()
            projection = dict(claim)
            payload = json.loads(event["payload_json"])
            metadata = json.loads(event["metadata_json"])
            mutation(projection, payload, metadata)
            event_schema_version = projection.pop(
                "_event_schema_version", event["schema_version"]
            )
            event_occurred_at = projection.pop(
                "_event_occurred_at", event["occurred_at"]
            )
            connection.execute(
                """UPDATE events
                   SET schema_version=?, payload_json=?, metadata_json=?,
                       occurred_at=?
                   WHERE event_id=?""",
                (
                    event_schema_version,
                    canonical_json(payload),
                    canonical_json(metadata),
                    event_occurred_at,
                    event["event_id"],
                ),
            )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "scoring claim does not match its event",
            ):
                connection.execute(_CLAIM_REINSERT, projection)
        finally:
            connection.rollback()
            connection.close()

    def _assert_reconciliation_envelope_rejected(
        self,
        database: Database,
        reconciliation: sqlite3.Row,
        mutation,
        *,
        synchronize_receipt: bool = True,
    ) -> None:
        connection = database.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DROP TRIGGER performance_scoring_reconciliations_no_delete"
            )
            connection.execute("DROP TRIGGER events_no_update")
            connection.execute(
                """DELETE FROM performance_scoring_reconciliations
                   WHERE id=?""",
                (reconciliation["id"],),
            )
            event = connection.execute(
                "SELECT * FROM events WHERE event_id=?",
                (reconciliation["event_id"],),
            ).fetchone()
            projection = dict(reconciliation)
            payload = json.loads(event["payload_json"])
            metadata = json.loads(event["metadata_json"])
            receipt = json.loads(projection["receipt_json"])
            mutation(projection, payload, metadata, receipt)
            event_occurred_at = projection.pop(
                "_event_occurred_at", event["occurred_at"]
            )
            projection["receipt_json"] = canonical_json(receipt)
            if synchronize_receipt:
                payload["receipt"] = receipt
            connection.execute(
                """UPDATE events
                   SET payload_json=?, metadata_json=?, occurred_at=?
                   WHERE event_id=?""",
                (
                    canonical_json(payload),
                    canonical_json(metadata),
                    event_occurred_at,
                    event["event_id"],
                ),
            )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "does not match its claim/event/receipt",
            ):
                connection.execute(
                    _RECONCILIATION_REINSERT,
                    projection,
                )
        finally:
            connection.rollback()
            connection.close()

    def test_direct_sql_rejects_malformed_v2_claim_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "claim-envelope.db"
            )
            claim = _claim_row(
                database, fixture["unfinished_attempt_id"]
            )

            def metadata_number(_row, _payload, metadata):
                metadata["shadow_only"] = 1

            def metadata_extra(_row, _payload, metadata):
                metadata["unexpected"] = False

            def provider_extra(_row, payload, _metadata):
                payload["provider"]["unexpected"] = "authority"

            def provider_number(_row, payload, _metadata):
                payload["provider"]["verified"] = 0

            def provider_kind(_row, payload, _metadata):
                payload["provider"]["declared_kind"] = "magical"

            def provider_identity(row, payload, _metadata):
                row["provider_id"] = "synthetic.bad provider"
                payload["provider_id"] = row["provider_id"]
                payload["provider"]["provider_id"] = row["provider_id"]

            def provider_authority(row, payload, _metadata):
                row["provider_id"] = "provider.direct-sql"
                payload["provider_id"] = row["provider_id"]
                provider = payload["provider"]
                provider["provider_id"] = row["provider_id"]
                provider["declared_kind"] = "deterministic"
                provider["verified"] = True
                provider["synthetic"] = False
                provider["shadow_only"] = False
                provider["check_set_manifests"] = []
                provider["artifact_manifests"] = []

            def manifest_digest(_row, payload, _metadata):
                payload["provider"]["check_set_manifests"].append(
                    {
                        "check_set_id": "check_direct_sql",
                        "manifest_digest": "A" * 64,
                    }
                )

            def payload_type(_row, payload, _metadata):
                payload["through_sequence"] = str(
                    payload["through_sequence"]
                )

            def set_claimed_timestamp(row, payload, value):
                row["claimed_at"] = value
                row["_event_occurred_at"] = value
                payload["claimed_at"] = value

            def claimed_invalid_calendar(row, payload, _metadata):
                set_claimed_timestamp(
                    row, payload, "2117-02-30T12:00:00+00:00"
                )

            def claimed_invalid_hour(row, payload, _metadata):
                set_claimed_timestamp(
                    row, payload, "2117-01-01T24:00:00+00:00"
                )

            def claimed_zero_fraction(row, payload, _metadata):
                set_claimed_timestamp(
                    row, payload, "2117-01-01T12:00:00.000000+00:00"
                )

            def claimed_year_zero(row, payload, _metadata):
                set_claimed_timestamp(
                    row, payload, "0000-01-01T12:00:00+00:00"
                )

            def legacy_v1(row, payload, metadata):
                row["claim_schema_version"] = 1
                row["scoring_request_digest"] = None
                row["provider_binding_digest"] = None
                row["provider_operation_digest"] = None
                row["_event_schema_version"] = 1
                payload.clear()
                payload.update(
                    performance_scoring_claim_payload(
                        claim_id=row["id"],
                        caller_idempotency_key=row["idempotency_key"],
                        attempt_id=row["attempt_id"],
                        evaluation_id=row["evaluation_id"],
                        through_sequence=row["through_sequence"],
                        provider_id=row["provider_id"],
                        provider_version=row["provider_version"],
                        action_trace_digest_value=row[
                            "action_trace_digest"
                        ],
                        command_hash=row["command_hash"],
                        claimed_at=row["claimed_at"],
                    )
                )
                metadata.clear()
                metadata.update(
                    {
                        "claim_schema_version": 1,
                        "admission_mode": "pre_callback",
                        "source_schema_version": None,
                        "shadow_only": True,
                    }
                )

            for name, mutation in (
                ("metadata number", metadata_number),
                ("metadata extra", metadata_extra),
                ("provider extra", provider_extra),
                ("provider number", provider_number),
                ("provider kind", provider_kind),
                ("provider identity", provider_identity),
                ("provider authority", provider_authority),
                ("manifest digest", manifest_digest),
                ("payload type", payload_type),
                ("claimed invalid calendar", claimed_invalid_calendar),
                ("claimed invalid hour", claimed_invalid_hour),
                ("claimed zero fraction", claimed_zero_fraction),
                ("claimed year zero", claimed_year_zero),
                ("runtime legacy v1", legacy_v1),
            ):
                with self.subTest(name=name):
                    self._assert_claim_envelope_rejected(
                        database, claim, mutation
                    )

    def test_direct_sql_rejects_claim_before_submission_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "claim-order.db"
            )
            claim = _claim_row(
                database, fixture["unfinished_attempt_id"]
            )
            connection = database.connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DROP TRIGGER performance_scoring_claims_no_delete"
                )
                connection.execute("DROP TRIGGER events_no_update")
                connection.execute(
                    "DELETE FROM performance_scoring_claims WHERE id=?",
                    (claim["id"],),
                )
                claim_event = connection.execute(
                    "SELECT * FROM events WHERE event_id=?",
                    (claim["event_id"],),
                ).fetchone()
                submission_event = connection.execute(
                    """SELECT event.*
                       FROM performance_actions action
                       JOIN events event ON event.event_id=action.event_id
                       WHERE action.attempt_id=?
                         AND action.sequence=?
                         AND action.action_type='submitted'""",
                    (claim["attempt_id"], claim["through_sequence"]),
                ).fetchone()
                connection.execute(
                    "UPDATE events SET stream_version=? WHERE event_id=?",
                    (
                        claim_event["stream_version"] + 1_000_000,
                        submission_event["event_id"],
                    ),
                )
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "scoring claim does not match its event",
                ):
                    connection.execute(_CLAIM_REINSERT, dict(claim))
            finally:
                connection.rollback()
                connection.close()

    def test_direct_sql_rejects_duplicate_manifest_identity_keys(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "claim-manifest-shape.db"
            )
            claim = _claim_row(
                database, fixture["unfinished_attempt_id"]
            )
            for manifest_field, identity_field, identity_value in (
                (
                    "check_set_manifests",
                    "check_set_id",
                    "check_direct_sql",
                ),
                (
                    "artifact_manifests",
                    "artifact_kind",
                    "artifact_direct_sql",
                ),
            ):
                with self.subTest(manifest_field=manifest_field):
                    connection = database.connect()
                    try:
                        connection.execute("BEGIN IMMEDIATE")
                        connection.execute(
                            "DROP TRIGGER "
                            "performance_scoring_claims_no_delete"
                        )
                        connection.execute("DROP TRIGGER events_no_update")
                        connection.execute(
                            """DELETE FROM performance_scoring_claims
                               WHERE id=?""",
                            (claim["id"],),
                        )
                        event = connection.execute(
                            "SELECT * FROM events WHERE event_id=?",
                            (claim["event_id"],),
                        ).fetchone()
                        payload = json.loads(event["payload_json"])
                        member = {
                            identity_field: identity_value,
                            "manifest_digest": "a" * 64,
                        }
                        payload["provider"][manifest_field] = [member]
                        payload_json = canonical_json(payload)
                        canonical_member = canonical_json(member)
                        duplicate_member = (
                            "{"
                            f'"{identity_field}":'
                            f"{json.dumps(identity_value)},"
                            f'"{identity_field}":"duplicate"'
                            "}"
                        )
                        self.assertIn(canonical_member, payload_json)
                        payload_json = payload_json.replace(
                            canonical_member, duplicate_member, 1
                        )
                        connection.execute(
                            """UPDATE events SET payload_json=?
                               WHERE event_id=?""",
                            (payload_json, event["event_id"]),
                        )
                        with self.assertRaisesRegex(
                            sqlite3.IntegrityError,
                            "scoring claim does not match its event",
                        ):
                            connection.execute(
                                _CLAIM_REINSERT, dict(claim)
                            )
                    finally:
                        connection.rollback()
                        connection.close()

    def test_direct_sql_rejects_malformed_reconciliation_envelopes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "reconciliation-envelope.db"
            )
            claim = _claim_row(
                database, fixture["unfinished_attempt_id"]
            )
            registry, observed = _registry(
                claim,
                outcome=ReconciliationOutcome.UNKNOWN,
                observed_offset=2,
                digest_character="4",
            )
            PerformanceLedger(database).reconcile_scoring_claim(
                claim["id"],
                registry,
                _RECONCILER_ID,
                _RECONCILER_VERSION,
                idempotency_key="envelope-observation",
                now=observed,
            )
            with database.read() as connection:
                reconciliation = connection.execute(
                    """SELECT * FROM performance_scoring_reconciliations
                       WHERE claim_id=?""",
                    (claim["id"],),
                ).fetchone()

            before_claim = to_timestamp(
                from_timestamp(claim["claimed_at"])
                - timedelta(minutes=1)
            )
            after_record = to_timestamp(
                from_timestamp(reconciliation["reconciled_at"])
                + timedelta(minutes=1)
            )

            def metadata_number(_row, _payload, metadata, _receipt):
                metadata["observational_only"] = 1

            def metadata_extra(_row, _payload, metadata, _receipt):
                metadata["unexpected"] = False

            def payload_extra(_row, payload, _metadata, _receipt):
                payload["unexpected"] = "field"

            def reconciler_number(_row, payload, _metadata, _receipt):
                payload["reconciler"]["synthetic"] = 1

            def reconciler_extra(_row, payload, _metadata, _receipt):
                payload["reconciler"]["unexpected"] = False

            def reconciler_identity(row, payload, _metadata, receipt):
                row["reconciler_id"] = "synthetic.bad reconciler"
                payload["reconciler_id"] = row["reconciler_id"]
                payload["reconciler"]["reconciler_id"] = row[
                    "reconciler_id"
                ]
                receipt["reconciler_id"] = row["reconciler_id"]

            def receipt_extra(_row, _payload, _metadata, receipt):
                receipt["unexpected"] = "field"

            def receipt_type(_row, _payload, _metadata, receipt):
                receipt["through_sequence"] = str(
                    receipt["through_sequence"]
                )

            def receipt_reason(_row, _payload, _metadata, receipt):
                receipt["reason_code"] = 7

            def receipt_reason_prefix(
                _row, _payload, _metadata, receipt
            ):
                receipt["reason_code"] = ":invalid"

            def set_reconciled_timestamp(row, payload, value):
                row["reconciled_at"] = value
                row["_event_occurred_at"] = value
                payload["reconciled_at"] = value

            def reconciled_zero_fraction(
                row, payload, _metadata, _receipt
            ):
                set_reconciled_timestamp(
                    row,
                    payload,
                    "2116-04-05T10:02:00.000000+00:00",
                )

            def receipt_timestamp(_row, _payload, _metadata, receipt):
                receipt["observed_at"] = receipt["observed_at"].replace(
                    "+00:00", "Z"
                )

            def receipt_invalid_calendar(
                row, payload, _metadata, receipt
            ):
                set_reconciled_timestamp(
                    row, payload, "2116-05-01T10:02:00+00:00"
                )
                receipt["observed_at"] = "2116-04-31T10:01:00+00:00"

            def receipt_invalid_hour(
                row, payload, _metadata, receipt
            ):
                set_reconciled_timestamp(
                    row, payload, "2116-04-06T00:02:00+00:00"
                )
                receipt["observed_at"] = "2116-04-05T24:00:00+00:00"

            def receipt_zero_fraction(
                row, payload, _metadata, receipt
            ):
                set_reconciled_timestamp(
                    row, payload, "2116-04-05T10:03:00+00:00"
                )
                receipt["observed_at"] = (
                    "2116-04-05T10:02:00.000000+00:00"
                )

            def receipt_predates(_row, _payload, _metadata, receipt):
                receipt["observed_at"] = before_claim

            def receipt_after_record(
                _row, _payload, _metadata, receipt
            ):
                receipt["observed_at"] = after_record

            def receipt_lifecycle(_row, _payload, _metadata, receipt):
                receipt["completed_at"] = receipt["observed_at"]
                receipt["result_digest"] = "6" * 64

            def completed_after_observation(
                row, payload, _metadata, receipt
            ):
                row["outcome"] = "completed"
                row["normalized_result_digest"] = "7" * 64
                payload["outcome"] = "completed"
                payload["normalized_result_digest"] = "7" * 64
                receipt["outcome"] = "completed"
                receipt["completed_at"] = after_record
                receipt["result_digest"] = "8" * 64

            def set_completed(row, payload, receipt, completed_at):
                row["outcome"] = "completed"
                row["normalized_result_digest"] = "7" * 64
                payload["outcome"] = "completed"
                payload["normalized_result_digest"] = "7" * 64
                receipt["outcome"] = "completed"
                receipt["completed_at"] = completed_at
                receipt["result_digest"] = "7" * 64

            def completed_invalid_calendar(
                row, payload, _metadata, receipt
            ):
                set_reconciled_timestamp(
                    row, payload, "2116-05-01T10:02:00+00:00"
                )
                receipt["observed_at"] = "2116-05-01T10:01:00+00:00"
                set_completed(
                    row,
                    payload,
                    receipt,
                    "2116-04-31T10:01:00+00:00",
                )

            def completed_invalid_hour(
                row, payload, _metadata, receipt
            ):
                set_reconciled_timestamp(
                    row, payload, "2116-04-06T00:02:00+00:00"
                )
                receipt["observed_at"] = "2116-04-06T00:01:00+00:00"
                set_completed(
                    row,
                    payload,
                    receipt,
                    "2116-04-05T24:00:00+00:00",
                )

            def completed_zero_fraction(
                row, payload, _metadata, receipt
            ):
                set_completed(
                    row,
                    payload,
                    receipt,
                    "2116-04-05T10:01:00.000000+00:00",
                )

            def receipt_upper_digest(
                _row, _payload, _metadata, receipt
            ):
                receipt["provider_receipt_digest"] = "A" * 64

            for name, mutation in (
                ("metadata number", metadata_number),
                ("metadata extra", metadata_extra),
                ("payload extra", payload_extra),
                ("reconciler number", reconciler_number),
                ("reconciler extra", reconciler_extra),
                ("reconciler identity", reconciler_identity),
                ("receipt extra", receipt_extra),
                ("receipt type", receipt_type),
                ("receipt reason", receipt_reason),
                ("receipt reason prefix", receipt_reason_prefix),
                ("reconciled zero fraction", reconciled_zero_fraction),
                ("receipt timestamp", receipt_timestamp),
                ("receipt invalid calendar", receipt_invalid_calendar),
                ("receipt invalid hour", receipt_invalid_hour),
                ("receipt zero fraction", receipt_zero_fraction),
                ("receipt predates claim", receipt_predates),
                ("receipt after record", receipt_after_record),
                ("receipt lifecycle", receipt_lifecycle),
                ("completion after observation", completed_after_observation),
                ("completed invalid calendar", completed_invalid_calendar),
                ("completed invalid hour", completed_invalid_hour),
                ("completed zero fraction", completed_zero_fraction),
                ("receipt uppercase digest", receipt_upper_digest),
            ):
                with self.subTest(name=name):
                    self._assert_reconciliation_envelope_rejected(
                        database,
                        reconciliation,
                        mutation,
                    )

            def receipt_projection_only(
                _row, _payload, _metadata, receipt
            ):
                receipt["reason_code"] = "projection_changed"

            self._assert_reconciliation_envelope_rejected(
                database,
                reconciliation,
                receipt_projection_only,
                synchronize_receipt=False,
            )

    def test_direct_sql_rejects_projection_order_after_later_observation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "reconciliation-order.db"
            )
            claim = _claim_row(
                database, fixture["unfinished_attempt_id"]
            )
            ledger = PerformanceLedger(database)
            unknown_registry, unknown_at = _registry(
                claim,
                outcome=ReconciliationOutcome.UNKNOWN,
                observed_offset=2,
                digest_character="4",
            )
            ledger.reconcile_scoring_claim(
                claim["id"],
                unknown_registry,
                _RECONCILER_ID,
                _RECONCILER_VERSION,
                idempotency_key="order-unknown",
                now=unknown_at,
            )
            terminal_registry, terminal_at = _registry(
                claim,
                outcome=ReconciliationOutcome.DEFINITELY_ABSENT,
                observed_offset=3,
                digest_character="5",
            )
            ledger.reconcile_scoring_claim(
                claim["id"],
                terminal_registry,
                _RECONCILER_ID,
                _RECONCILER_VERSION,
                idempotency_key="order-terminal",
                now=terminal_at,
            )
            with database.read() as connection:
                rows = connection.execute(
                    """SELECT * FROM performance_scoring_reconciliations
                       WHERE claim_id=? ORDER BY reconciled_at""",
                    (claim["id"],),
                ).fetchall()
            unknown, terminal = rows

            connection = database.connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DROP TRIGGER "
                    "performance_scoring_reconciliations_no_delete"
                )
                connection.execute("DROP TRIGGER events_no_update")
                connection.execute(
                    """DELETE FROM performance_scoring_reconciliations
                       WHERE id=?""",
                    (terminal["id"],),
                )
                unknown_event = connection.execute(
                    "SELECT * FROM events WHERE event_id=?",
                    (unknown["event_id"],),
                ).fetchone()
                terminal_event = connection.execute(
                    "SELECT * FROM events WHERE event_id=?",
                    (terminal["event_id"],),
                ).fetchone()
                temporary_version = max(
                    unknown_event["stream_version"],
                    terminal_event["stream_version"],
                ) + 1_000_000
                connection.execute(
                    "UPDATE events SET stream_version=? WHERE event_id=?",
                    (temporary_version, terminal_event["event_id"]),
                )
                connection.execute(
                    "UPDATE events SET stream_version=? WHERE event_id=?",
                    (
                        terminal_event["stream_version"],
                        unknown_event["event_id"],
                    ),
                )
                connection.execute(
                    "UPDATE events SET stream_version=? WHERE event_id=?",
                    (
                        unknown_event["stream_version"],
                        terminal_event["event_id"],
                    ),
                )
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "does not match its claim/event/receipt",
                ):
                    connection.execute(
                        _RECONCILIATION_REINSERT, dict(terminal)
                    )
            finally:
                connection.rollback()
                connection.close()

    def test_unknowns_then_one_terminal_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "terminal.db"
            )
            claim = _claim_row(
                database, fixture["unfinished_attempt_id"]
            )
            ledger = PerformanceLedger(database)
            for offset, character, key in (
                (2, "1", "unknown-1"),
                (3, "2", "unknown-2"),
            ):
                registry, observed = _registry(
                    claim,
                    outcome=ReconciliationOutcome.UNKNOWN,
                    observed_offset=offset,
                    digest_character=character,
                )
                ledger.reconcile_scoring_claim(
                    claim["id"],
                    registry,
                    _RECONCILER_ID,
                    _RECONCILER_VERSION,
                    idempotency_key=key,
                    now=observed,
                )
            registry, observed = _registry(
                claim,
                outcome=ReconciliationOutcome.DEFINITELY_ABSENT,
                observed_offset=4,
                digest_character="3",
            )
            ledger.reconcile_scoring_claim(
                claim["id"],
                registry,
                _RECONCILER_ID,
                _RECONCILER_VERSION,
                idempotency_key="absent",
                now=observed,
            )
            with database.read() as connection:
                rows = connection.execute(
                    """SELECT * FROM performance_scoring_reconciliations
                       WHERE claim_id=? ORDER BY reconciled_at""",
                    (claim["id"],),
                ).fetchall()
            self.assertEqual(
                [row["outcome"] for row in rows],
                ["unknown", "unknown", "definitely_absent"],
            )

            with self.assertRaisesRegex(
                sqlite3.IntegrityError, "already reconciled"
            ), database.transaction() as connection:
                connection.execute(
                    _FAKE_RECONCILIATION_INSERT,
                    _fake_reconciliation_values(
                        claim,
                        event_id=claim["event_id"],
                        command_character="e",
                    ),
                )
            for statement in (
                """UPDATE performance_scoring_reconciliations
                   SET reconciled_at=reconciled_at WHERE id=?""",
                "DELETE FROM performance_scoring_reconciliations WHERE id=?",
            ):
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError, "immutable"
                ), database.transaction() as connection:
                    connection.execute(statement, (rows[0]["id"],))

    def test_completion_and_caller_key_are_first_wins_at_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "first-wins.db"
            )
            completed = _claim_row(
                database, fixture["completed_attempt_id"]
            )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError, "cannot be reconciled"
            ), database.transaction() as connection:
                connection.execute(
                    _FAKE_RECONCILIATION_INSERT,
                    _fake_reconciliation_values(
                        completed,
                        event_id=completed["event_id"],
                        command_character="6",
                    ),
                )

            unfinished = _claim_row(
                database, fixture["unfinished_attempt_id"]
            )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError, "belongs to a scoring claim"
            ), database.transaction() as connection:
                connection.execute(
                    _FAKE_RECONCILIATION_INSERT,
                    _fake_reconciliation_values(
                        unfinished,
                        event_id=unfinished["event_id"],
                        idempotency_key=unfinished["idempotency_key"],
                        command_character="7",
                    ),
                )

    def test_recovered_completion_binds_null_session_and_result_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "recovered.db"
            )
            claim = _claim_row(
                database, fixture["unfinished_attempt_id"]
            )
            imported = _registered_evaluation(
                fixture["unfinished_submission_id"],
                "schema_v19_recovered",
            )
            registry, observed = _registry(
                claim,
                outcome=ReconciliationOutcome.COMPLETED,
                observed_offset=3,
                digest_character="8",
                imported=imported,
            )
            result = PerformanceLedger(database).reconcile_scoring_claim(
                claim["id"],
                registry,
                _RECONCILER_ID,
                _RECONCILER_VERSION,
                idempotency_key="recovered",
                now=observed,
            )
            self.assertEqual(result["status"], "completed")

            with database.read() as connection:
                reconciliation = connection.execute(
                    """SELECT * FROM performance_scoring_reconciliations
                       WHERE claim_id=?""",
                    (claim["id"],),
                ).fetchone()
                evaluation = connection.execute(
                    "SELECT * FROM task_evaluations WHERE id=?",
                    (claim["evaluation_id"],),
                ).fetchone()
                evaluation_event = connection.execute(
                    "SELECT * FROM events WHERE event_id=?",
                    (evaluation["event_id"],),
                ).fetchone()
                bundle_event = connection.execute(
                    """SELECT event.*
                       FROM shadow_evidence_bundles bundle
                       JOIN events event ON event.event_id=bundle.event_id
                       WHERE bundle.evaluation_id=?""",
                    (claim["evaluation_id"],),
                ).fetchone()
            authority = json.loads(evaluation["authority_json"])
            self.assertIsNone(evaluation_event["session_id"])
            self.assertIsNone(bundle_event["session_id"])
            self.assertEqual(
                evaluation_event["causation_id"],
                reconciliation["event_id"],
            )
            self.assertEqual(
                authority["normalized_result_digest"],
                reconciliation["normalized_result_digest"],
            )

            connection = database.connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DROP TRIGGER shadow_evidence_bundles_no_delete"
                )
                connection.execute("DROP TRIGGER task_evaluations_no_delete")
                connection.execute(
                    """DELETE FROM shadow_evidence_bundles
                       WHERE evaluation_id=?""",
                    (evaluation["id"],),
                )
                connection.execute(
                    "DELETE FROM task_evaluations WHERE id=?",
                    (evaluation["id"],),
                )
                tampered = dict(authority)
                tampered["normalized_result_digest"] = "f" * 64
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "does not match its event",
                ):
                    connection.execute(
                        """INSERT INTO task_evaluations(
                               id, event_id, attempt_id, through_sequence,
                               evaluation_digest, evaluation_json,
                               authority_json, recorded_at, command_hash
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            evaluation["id"],
                            evaluation["event_id"],
                            evaluation["attempt_id"],
                            evaluation["through_sequence"],
                            evaluation["evaluation_digest"],
                            evaluation["evaluation_json"],
                            canonical_json(tampered),
                            evaluation["recorded_at"],
                            evaluation["command_hash"],
                        ),
                    )
                attempt = connection.execute(
                    "SELECT * FROM performance_attempts WHERE id=?",
                    (evaluation["attempt_id"],),
                ).fetchone()
                ordinary_event = database.append_event(
                    connection,
                    stream_id=f"learner:{attempt['learner_id']}",
                    event_type="TaskEvaluationRecorded",
                    schema_version=1,
                    payload={
                        "attempt_id": evaluation["attempt_id"],
                        "through_sequence": evaluation["through_sequence"],
                        "evaluation_digest": evaluation["evaluation_digest"],
                        "evaluation": json.loads(evaluation["evaluation_json"]),
                        "authority": authority,
                    },
                    metadata={
                        "command_hash": evaluation["command_hash"],
                        "task_release_id": attempt["task_release_id"],
                        "corpus_release_id": attempt["corpus_release_id"],
                        "shadow_only": True,
                        "projection_applied": False,
                        "certification_applied": False,
                    },
                    learner_id=attempt["learner_id"],
                    session_id=attempt["session_id"],
                    idempotency_key=claim["idempotency_key"],
                    correlation_id=attempt["id"],
                    causation_id=claim["event_id"],
                    occurred_at=from_timestamp(
                        evaluation_event["occurred_at"]
                    ),
                )
                with self.assertRaisesRegex(
                    sqlite3.IntegrityError,
                    "does not match its event",
                ):
                    connection.execute(
                        """INSERT INTO task_evaluations(
                               id, event_id, attempt_id, through_sequence,
                               evaluation_digest, evaluation_json,
                               authority_json, recorded_at, command_hash
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            evaluation["id"],
                            ordinary_event["event_id"],
                            evaluation["attempt_id"],
                            evaluation["through_sequence"],
                            evaluation["evaluation_digest"],
                            evaluation["evaluation_json"],
                            evaluation["authority_json"],
                            ordinary_event["recorded_at"],
                            evaluation["command_hash"],
                        ),
                    )
            finally:
                connection.rollback()
                connection.close()


if __name__ == "__main__":
    unittest.main()
