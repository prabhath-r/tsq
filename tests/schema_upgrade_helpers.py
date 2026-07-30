# SPDX-License-Identifier: MPL-2.0

"""Shared helpers for constructing and inspecting historical schemas."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from tsq.evidence import canonical_json
from tsq.store import (
    Database,
    _content_hash,
    performance_scoring_claim_payload,
)


def durable_database_fingerprint(
    path: Path,
) -> tuple[tuple[str, int, str], ...]:
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


def rehash_event_streams(connection: sqlite3.Connection) -> None:
    """Recompute event hashes and stream heads after a fixture rewrite."""

    for stream in connection.execute(
        "SELECT DISTINCT stream_id FROM events ORDER BY stream_id"
    ).fetchall():
        stream_id = stream["stream_id"]
        previous_hash = None
        tail_version = 0
        tail_recorded_at = None
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
                (previous_hash, payload_hash, event["event_id"]),
            )
            previous_hash = payload_hash
            tail_version = event["stream_version"]
            tail_recorded_at = event["recorded_at"]
        connection.execute(
            """UPDATE stream_heads
               SET stream_version=?, payload_hash=?, updated_at=?
               WHERE stream_id=?""",
            (
                tail_version,
                previous_hash,
                tail_recorded_at,
                stream_id,
            ),
        )


def restore_pre_reconciliation_schema(
    connection: sqlite3.Connection,
) -> None:
    """Restore the historical scoring boundary before reconciliation."""

    artifact_run_table = connection.execute(
        """SELECT 1 FROM sqlite_master
           WHERE type='table'
             AND name='performance_artifact_run_claims'"""
    ).fetchone()
    if artifact_run_table is not None:
        Database._downgrade_v20_contract_to_v19(connection)

    reconciliation_table = connection.execute(
        """SELECT 1 FROM sqlite_master
           WHERE type='table'
             AND name='performance_scoring_reconciliations'"""
    ).fetchone()
    if reconciliation_table is None:
        return
    if connection.execute(
        "SELECT 1 FROM performance_scoring_reconciliations LIMIT 1"
    ).fetchone() is not None:
        raise AssertionError(
            "Historical fixture cannot discard scoring reconciliation rows."
        )
    if connection.execute(
        """SELECT 1 FROM events
           WHERE event_type='PerformanceScoringReconciled'
           LIMIT 1"""
    ).fetchone() is not None:
        raise AssertionError(
            "Historical fixture cannot discard scoring reconciliation events."
        )

    claims = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM performance_scoring_claims ORDER BY id"
        ).fetchall()
    ]
    event_guard = connection.execute(
        """SELECT sql FROM sqlite_master
           WHERE type='trigger' AND name='events_no_update'"""
    ).fetchone()
    if event_guard is None or not event_guard["sql"]:
        raise AssertionError(
            "Historical fixture lacks the immutable event update guard."
        )
    connection.execute('DROP TRIGGER "events_no_update"')
    for trigger_name in (
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
        connection.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}"')

    for claim in claims:
        claim_event = connection.execute(
            "SELECT event_type FROM events WHERE event_id=?",
            (claim["event_id"],),
        ).fetchone()
        if claim_event is None or claim_event["event_type"] not in {
            "PerformanceScoringClaimed",
            "PerformanceScoringClaimMigrated",
        }:
            raise AssertionError(
                f"Historical claim {claim['id']} lacks its admission event."
            )
        migrated = (
            claim_event["event_type"] == "PerformanceScoringClaimMigrated"
        )
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
                        "admission_mode": (
                            "legacy_projection_migration"
                            if migrated
                            else "pre_callback"
                        ),
                        "source_schema_version": 15 if migrated else None,
                        "shadow_only": True,
                    }
                ),
                claim["event_id"],
            ),
        )
    rehash_event_streams(connection)

    connection.execute("DROP TABLE performance_scoring_reconciliations")
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


def restore_pre_shadow_schema(connection: sqlite3.Connection) -> None:
    """Strip policy-shadow and later prospective schema from a fixture."""

    restore_pre_reconciliation_schema(connection)
    table = connection.execute(
        """SELECT 1 FROM sqlite_master
           WHERE type='table' AND name='policy_shadow_evaluations'"""
    ).fetchone()
    if table is None:
        return

    shadow_events = connection.execute(
        """SELECT event_id, stream_id FROM events
           WHERE event_type='PolicyShadowEvaluated'
           ORDER BY stream_id, stream_version"""
    ).fetchall()
    shadow_policy_events = connection.execute(
        """SELECT event_id, stream_id, metadata_json FROM events
           WHERE json_extract(
                 metadata_json, '$.policy_version'
             )='recursive-evidence-graph-v18'
           ORDER BY stream_id, stream_version"""
    ).fetchall()
    event_guards = []
    if shadow_events or shadow_policy_events:
        event_guards = connection.execute(
            """SELECT name, sql FROM sqlite_master
               WHERE type='trigger'
                 AND name IN ('events_no_update', 'events_no_delete')
               ORDER BY name"""
        ).fetchall()
        for guard in event_guards:
            connection.execute(f'DROP TRIGGER "{guard["name"]}"')

    for trigger_name in (
        "policy_shadow_evaluations_validate_insert",
        "policy_shadow_evaluations_no_update",
        "policy_shadow_evaluations_no_delete",
    ):
        connection.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}"')
    connection.execute("DELETE FROM policy_shadow_evaluations")
    connection.execute("DROP TABLE policy_shadow_evaluations")

    connection.execute(
        """UPDATE decisions
           SET policy_version='recursive-evidence-graph-v17'
           WHERE policy_version='recursive-evidence-graph-v18'"""
    )
    for policy_event in shadow_policy_events:
        metadata = json.loads(policy_event["metadata_json"])
        metadata["policy_version"] = "recursive-evidence-graph-v17"
        connection.execute(
            "UPDATE events SET metadata_json=? WHERE event_id=?",
            (
                json.dumps(
                    metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                policy_event["event_id"],
            ),
        )

    if shadow_events:
        connection.execute(
            "DELETE FROM events WHERE event_type='PolicyShadowEvaluated'"
        )
        for stream_id in sorted(
            {row["stream_id"] for row in shadow_events}
        ):
            rows = connection.execute(
                """SELECT event_id FROM events
                   WHERE stream_id=? ORDER BY stream_version""",
                (stream_id,),
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
    if shadow_events or shadow_policy_events:
        rehash_event_streams(connection)
        for guard in event_guards:
            if not guard["sql"]:
                raise AssertionError(
                    f"Event guard {guard['name']} has no SQL."
                )
            connection.execute(guard["sql"])
