# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.artifact_runner import (
    ArtifactRunRequest,
    bundled_synthetic_binding,
)
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError
from tsq.evidence import (
    ActionKind,
    ScorerContract,
    ScorerKind,
    canonical_digest,
    canonical_json,
)
from tsq.performance_ledger import PerformanceLedger, _command_hash
from tsq.store import (
    Database,
    PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
    _capture_current_schema_contract,
    _expected_current_schema_contract,
    _expected_v19_schema_contract,
    performance_artifact_run_claim_event_key,
    performance_artifact_run_claim_payload,
    performance_artifact_run_observed_payload,
    performance_artifact_run_receipt_event_key,
    to_timestamp,
)

from tests.test_performance_ledger import (
    CORPUS,
    declared_task_release_fixture,
)
from tests.test_scoring_claim_history_upgrade import (
    _build_two_claim_database,
    rehash_event_streams,
)
from tests.test_migration_event_lifecycle import durable_database_fingerprint


START = datetime(2110, 7, 1, 9, 0, tzinfo=timezone.utc)
ARTIFACT_DIGEST = "3" * 64
_RUNNER_BINDING = bundled_synthetic_binding()
ARTIFACT_MANIFEST_DIGEST = _RUNNER_BINDING.artifact_manifest_digest
CHECK_SET_MANIFEST_DIGEST = _RUNNER_BINDING.check_set_manifest_digest

_CLAIM_INSERT = """
INSERT INTO performance_artifact_run_claims(
    id, event_id, idempotency_key, attempt_id, session_id,
    session_revision, artifact_action_id, through_sequence,
    task_release_id, task_id, task_version, task_digest,
    artifact_digest, artifact_kind, artifact_manifest_digest,
    check_set_id, check_set_manifest_digest, runner_id, runner_version,
    request_json, request_digest, binding_json, binding_digest,
    command_hash, claimed_at
) VALUES (
    :id, :event_id, :idempotency_key, :attempt_id, :session_id,
    :session_revision, :artifact_action_id, :through_sequence,
    :task_release_id, :task_id, :task_version, :task_digest,
    :artifact_digest, :artifact_kind, :artifact_manifest_digest,
    :check_set_id, :check_set_manifest_digest, :runner_id, :runner_version,
    :request_json, :request_digest, :binding_json, :binding_digest,
    :command_hash, :claimed_at
)
"""

_RECEIPT_INSERT = """
INSERT INTO performance_artifact_run_receipts(
    id, event_id, claim_id, check_action_id, outcome, result_json,
    result_digest, receipt_json, receipt_digest, started_at, completed_at
) VALUES (
    :id, :event_id, :claim_id, :check_action_id, :outcome, :result_json,
    :result_digest, :receipt_json, :receipt_digest, :started_at, :completed_at
)
"""


def _schema_contract(database: Database):
    with database.read() as connection:
        return _capture_current_schema_contract(connection)


def _event_snapshot(database: Database) -> tuple[tuple[object, ...], ...]:
    with database.read() as connection:
        return tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM events ORDER BY stream_id, stream_version"
            )
        )


def _seed_legacy_scoring_caller_key(
    fixture: dict[str, object],
    key: str,
    *,
    source: str,
) -> None:
    """Install one pre-v20 caller key without exercising current admission."""

    database = fixture["database"]
    attempt = fixture["attempt"]
    artifact = fixture["artifact"]
    assert isinstance(database, Database)
    assert isinstance(attempt, dict)
    assert isinstance(artifact, dict)
    if source not in {"claim", "reconciliation"}:
        raise AssertionError("unknown legacy scoring caller-key source")
    with database.transaction() as connection:
        trigger_sql = {
            row["name"]: row["sql"]
            for row in connection.execute(
                """SELECT name, sql FROM sqlite_master
                   WHERE type='trigger' AND name IN (
                       'performance_scoring_claims_validate_insert',
                       'performance_scoring_reconciliations_validate_insert'
                   )"""
            )
        }
        connection.execute(
            "DROP TRIGGER performance_scoring_claims_validate_insert"
        )
        connection.execute(
            "DROP TRIGGER performance_scoring_reconciliations_validate_insert"
        )
        connection.execute(
            """INSERT INTO performance_scoring_claims(
                   id, event_id, claim_schema_version, idempotency_key,
                   attempt_id, evaluation_id, through_sequence, provider_id,
                   provider_version, action_trace_digest,
                   scoring_request_digest, provider_binding_digest,
                   provider_operation_digest, command_hash, claimed_at
               ) VALUES (?, ?, 2, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "legacy-scoring-claim",
                attempt["event_id"],
                key if source == "claim" else "legacy-unrelated-key",
                attempt["id"],
                "legacy-scoring-evaluation",
                artifact["sequence"],
                "legacy.provider",
                "v1",
                "6" * 64,
                "7" * 64,
                "8" * 64,
                "9" * 64,
                "a" * 64,
                to_timestamp(START + timedelta(minutes=3)),
            ),
        )
        if source == "reconciliation":
            connection.execute(
                """INSERT INTO performance_scoring_reconciliations(
                       id, event_id, idempotency_key, claim_id, attempt_id,
                       evaluation_id, outcome, scoring_request_digest,
                       provider_binding_digest, provider_operation_digest,
                       reconciler_id, reconciler_version,
                       reconciliation_binding_digest, receipt_json,
                       receipt_digest, normalized_result_digest,
                       reconciled_at, command_hash
                   ) VALUES (
                       ?, ?, ?, ?, ?, ?, 'unknown', ?, ?, ?, ?, ?, ?, '{}',
                       ?, NULL, ?, ?
                   )""",
                (
                    "legacy-scoring-reconciliation",
                    artifact["event_id"],
                    key,
                    "legacy-scoring-claim",
                    attempt["id"],
                    "legacy-scoring-evaluation",
                    "7" * 64,
                    "8" * 64,
                    "9" * 64,
                    "legacy.reconciler",
                    "v1",
                    "b" * 64,
                    "c" * 64,
                    to_timestamp(START + timedelta(minutes=4)),
                    "d" * 64,
                ),
            )
        connection.execute(
            trigger_sql["performance_scoring_claims_validate_insert"]
        )
        connection.execute(
            trigger_sql[
                "performance_scoring_reconciliations_validate_insert"
            ]
        )


def _build_fixture(path: Path, suffix: str = "fixture") -> dict[str, object]:
    database = Database(path)
    database.initialize()
    database.import_corpus(*read_and_parse(CORPUS, include_catalog=True))
    engine = AdaptiveEngine(database)
    learner_id = f"artifact-run-learner-{suffix}"
    engine.create_learner(learner_id, "Artifact Run Learner")
    session = engine.start_session(
        learner_id,
        "t_transformers",
        seed=2601,
    )
    with database.read() as connection:
        corpus_release_id = connection.execute(
            "SELECT value FROM meta WHERE key='active_corpus_release'"
        ).fetchone()["value"]
    bundle = declared_task_release_fixture(corpus_release_id)
    status, task = bundle.tasks[0]
    contract = ScorerContract(
        kind=ScorerKind.DETERMINISTIC,
        scorer_id="checks.causal-mask-matrix",
        scorer_version="v1",
        authority_id="authority.synthetic.causal-mask",
        authority_manifest_digest="7" * 64,
        criterion_ids=tuple(
            criterion.id for criterion in task.criteria
        ),
        evidence_action_kinds=(
            ActionKind.ARTIFACT_CHECKPOINT,
            ActionKind.CHECK_RUN,
        ),
        check_set_manifests=(
            (
                _RUNNER_BINDING.check_set_id,
                _RUNNER_BINDING.check_set_manifest_digest,
            ),
        ),
        artifact_manifests=(
            (
                _RUNNER_BINDING.artifact_kind,
                _RUNNER_BINDING.artifact_manifest_digest,
            ),
        ),
    )
    bundle = replace(
        bundle,
        tasks=((status, replace(task, scorer_contracts=(contract,))),),
    )
    ledger = PerformanceLedger(database)
    release = ledger.publish_release(
        bundle,
        now=datetime.now(timezone.utc),
    )
    task = bundle.tasks[0][1]
    attempt = ledger.start_attempt(
        session["id"],
        task.id,
        task_version=task.version,
        task_release_id=release["release_id"],
        idempotency_key=f"artifact-run-start-{suffix}",
        now=START + timedelta(minutes=1),
    )
    artifact = ledger.record_action(
        attempt["id"],
        "artifact_checkpoint",
        {
            "artifact_digest": ARTIFACT_DIGEST,
            "artifact_kind": _RUNNER_BINDING.artifact_kind,
        },
        idempotency_key=f"artifact-run-checkpoint-{suffix}",
        now=START + timedelta(minutes=2),
    )
    with database.read() as connection:
        attempt_row = dict(
            connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?",
                (attempt["id"],),
            ).fetchone()
        )
        session_row = dict(
            connection.execute(
                "SELECT * FROM sessions WHERE id=?",
                (session["id"],),
            ).fetchone()
        )
        artifact_row = dict(
            connection.execute(
                "SELECT * FROM performance_actions WHERE id=?",
                (artifact["id"],),
            ).fetchone()
        )
    return {
        "database": database,
        "engine": engine,
        "ledger": ledger,
        "session": session_row,
        "attempt": attempt_row,
        "artifact": artifact_row,
        "suffix": suffix,
    }


def _claim_terms(fixture: dict[str, object]) -> dict[str, object]:
    attempt = fixture["attempt"]
    session = fixture["session"]
    artifact = fixture["artifact"]
    suffix = fixture["suffix"]
    assert isinstance(attempt, dict)
    assert isinstance(session, dict)
    assert isinstance(artifact, dict)
    assert isinstance(suffix, str)
    claim_id = f"artifact-run-claim-{suffix}"
    binding_value = _RUNNER_BINDING
    binding = binding_value.terms()
    binding_digest = binding_value.digest
    command_hash = _command_hash(
        {
            "operation": "run_artifact_check",
            "attempt_id": attempt["id"],
            "artifact_action_id": artifact["id"],
            "artifact_digest": ARTIFACT_DIGEST,
            "artifact_size_bytes": 0,
            "artifact_kind": binding_value.artifact_kind,
            "artifact_manifest_digest": (
                binding_value.artifact_manifest_digest
            ),
            "check_set_id": binding_value.check_set_id,
            "check_set_manifest_digest": (
                binding_value.check_set_manifest_digest
            ),
            "checker_id": binding_value.checker_id.value,
            "checker_version": binding_value.checker_version,
            "runner_id": binding_value.runner_id,
            "runner_version": binding_value.runner_version,
            "binding_digest": binding_digest,
        }
    )
    request_value = ArtifactRunRequest(
        run_id="arun_" + command_hash[:24],
        checker_id=binding_value.checker_id,
        checker_version=binding_value.checker_version,
        artifact_kind=binding_value.artifact_kind,
        artifact_manifest_digest=binding_value.artifact_manifest_digest,
        artifact_sha256=ARTIFACT_DIGEST,
        artifact_size_bytes=0,
        check_set_id=binding_value.check_set_id,
        check_set_manifest_digest=binding_value.check_set_manifest_digest,
        runner_binding_digest=binding_digest,
    )
    request = request_value.terms()
    request_digest = canonical_digest(request)
    return {
        "id": claim_id,
        "event_id": None,
        "idempotency_key": f"artifact-run-caller-{suffix}",
        "attempt_id": attempt["id"],
        "session_id": session["id"],
        "session_revision": session["revision"],
        "artifact_action_id": artifact["id"],
        "through_sequence": artifact["sequence"],
        "task_release_id": attempt["task_release_id"],
        "task_id": attempt["task_id"],
        "task_version": attempt["task_version"],
        "task_digest": attempt["task_digest"],
        "artifact_digest": ARTIFACT_DIGEST,
        "artifact_kind": binding_value.artifact_kind,
        "artifact_manifest_digest": ARTIFACT_MANIFEST_DIGEST,
        "check_set_id": binding_value.check_set_id,
        "check_set_manifest_digest": CHECK_SET_MANIFEST_DIGEST,
        "runner_id": binding_value.runner_id,
        "runner_version": binding_value.runner_version,
        "request": request,
        "request_json": canonical_json(request),
        "request_digest": request_digest,
        "binding": binding,
        "binding_json": canonical_json(binding),
        "binding_digest": binding_digest,
        "command_hash": command_hash,
        "claimed_at": to_timestamp(START + timedelta(minutes=3)),
    }


def _insert_claim(
    fixture: dict[str, object],
    *,
    mutate=None,
) -> dict[str, object]:
    database = fixture["database"]
    attempt = fixture["attempt"]
    artifact = fixture["artifact"]
    assert isinstance(database, Database)
    assert isinstance(attempt, dict)
    assert isinstance(artifact, dict)
    row = _claim_terms(fixture)
    payload = performance_artifact_run_claim_payload(
        claim_id=str(row["id"]),
        caller_idempotency_key=str(row["idempotency_key"]),
        attempt_id=str(row["attempt_id"]),
        session_id=str(row["session_id"]),
        session_revision=int(row["session_revision"]),
        artifact_action_id=str(row["artifact_action_id"]),
        through_sequence=int(row["through_sequence"]),
        task_release_id=str(row["task_release_id"]),
        task_id=str(row["task_id"]),
        task_version=int(row["task_version"]),
        task_digest=str(row["task_digest"]),
        artifact_digest=str(row["artifact_digest"]),
        artifact_kind=str(row["artifact_kind"]),
        artifact_manifest_digest=str(row["artifact_manifest_digest"]),
        check_set_id=str(row["check_set_id"]),
        check_set_manifest_digest=str(row["check_set_manifest_digest"]),
        runner_id=str(row["runner_id"]),
        runner_version=str(row["runner_version"]),
        request=row["request"],
        request_digest=str(row["request_digest"]),
        binding=row["binding"],
        binding_digest=str(row["binding_digest"]),
        command_hash=str(row["command_hash"]),
        claimed_at=str(row["claimed_at"]),
    )
    metadata = {
        "artifact_run_schema_version": 1,
        "admission_mode": "pre_runner",
        "automatic_retry_allowed": False,
        "shadow_only": True,
    }
    if mutate is not None:
        mutate(row, payload, metadata)
    with database.transaction() as connection:
        event = database.append_event(
            connection,
            stream_id=f"learner:{attempt['learner_id']}",
            event_type="PerformanceArtifactRunClaimed",
            schema_version=1,
            payload=payload,
            metadata=metadata,
            learner_id=str(attempt["learner_id"]),
            session_id=str(attempt["session_id"]),
            idempotency_key=performance_artifact_run_claim_event_key(
                str(row["command_hash"])
            ),
            correlation_id=str(attempt["id"]),
            causation_id=str(artifact["event_id"]),
            occurred_at=datetime.fromisoformat(str(row["claimed_at"])),
        )
        row["event_id"] = event["event_id"]
        connection.execute(_CLAIM_INSERT, row)
    return row


def _receipt_terms(
    claim: dict[str, object],
    *,
    outcome: str,
    check_action_id: str | None,
    result: dict[str, object] | None,
) -> dict[str, object]:
    started_at = to_timestamp(START + timedelta(minutes=4))
    completed_at = to_timestamp(START + timedelta(minutes=5))
    result_digest = None if result is None else canonical_digest(result)
    receipt = {
        "claim_id": claim["id"],
        "attempt_id": claim["attempt_id"],
        "artifact_action_id": claim["artifact_action_id"],
        "artifact_digest": claim["artifact_digest"],
        "artifact_kind": claim["artifact_kind"],
        "artifact_manifest_digest": claim["artifact_manifest_digest"],
        "check_set_id": claim["check_set_id"],
        "check_set_manifest_digest": claim["check_set_manifest_digest"],
        "runner_id": claim["runner_id"],
        "runner_version": claim["runner_version"],
        "outcome": outcome,
        "started_at": started_at,
        "completed_at": completed_at,
        "result_digest": result_digest,
        "request_digest": claim["request_digest"],
        "binding_digest": claim["binding_digest"],
        "schema_version": 1,
    }
    receipt_digest = canonical_digest(receipt)
    return {
        "id": f"artifact-run-receipt-{receipt_digest[:24]}",
        "event_id": None,
        "claim_id": claim["id"],
        "check_action_id": check_action_id,
        "outcome": outcome,
        "result": result,
        "result_json": None if result is None else canonical_json(result),
        "result_digest": result_digest,
        "receipt": receipt,
        "receipt_json": canonical_json(receipt),
        "receipt_digest": receipt_digest,
        "started_at": started_at,
        "completed_at": completed_at,
    }


def _insert_receipt(
    fixture: dict[str, object],
    claim: dict[str, object],
    outcome: str,
) -> dict[str, object]:
    database = fixture["database"]
    ledger = fixture["ledger"]
    attempt = fixture["attempt"]
    assert isinstance(database, Database)
    assert isinstance(ledger, PerformanceLedger)
    assert isinstance(attempt, dict)
    result = None
    check_action_id = None
    if outcome in {"completed", "invalid_artifact"}:
        result = {
            "errored": 0,
            "failed": 0 if outcome == "completed" else 1,
            "passed": 1 if outcome == "completed" else 0,
            "schema_version": 1,
            "skipped": 0,
        }
        result_digest = canonical_digest(result)
        action = ledger.record_action(
            str(attempt["id"]),
            "check_run",
            {
                "check_set_id": claim["check_set_id"],
                "passed": result["passed"],
                "failed": result["failed"],
                "errored": result["errored"],
                "skipped": result["skipped"],
                "result_digest": result_digest,
            },
            idempotency_key=(
                f"artifact-run-check-{fixture['suffix']}-{outcome}"
            ),
            now=START + timedelta(minutes=5),
        )
        check_action_id = action["id"]
    row = _receipt_terms(
        claim,
        outcome=outcome,
        check_action_id=check_action_id,
        result=result,
    )
    with database.transaction() as connection:
        event = database.append_event(
            connection,
            stream_id=f"learner:{attempt['learner_id']}",
            event_type="PerformanceArtifactRunObserved",
            schema_version=1,
            payload=performance_artifact_run_observed_payload(
                receipt_id=str(row["id"]),
                claim_id=str(row["claim_id"]),
                attempt_id=str(attempt["id"]),
                check_action_id=(
                    None
                    if row["check_action_id"] is None
                    else str(row["check_action_id"])
                ),
                outcome=outcome,
                result=row["result"],
                result_digest=(
                    None
                    if row["result_digest"] is None
                    else str(row["result_digest"])
                ),
                receipt=row["receipt"],
                receipt_digest=str(row["receipt_digest"]),
                started_at=str(row["started_at"]),
                completed_at=str(row["completed_at"]),
            ),
            metadata={
                "artifact_run_schema_version": 1,
                "observational_only": True,
                "projection_applied": False,
                "certification_applied": False,
                "skill_authority": False,
                "shadow_only": True,
            },
            learner_id=str(attempt["learner_id"]),
            session_id=None,
            idempotency_key=performance_artifact_run_receipt_event_key(
                str(row["receipt_digest"])
            ),
            correlation_id=str(attempt["id"]),
            causation_id=str(claim["event_id"]),
            occurred_at=datetime.fromisoformat(str(row["completed_at"])),
        )
        row["event_id"] = event["event_id"]
        connection.execute(_RECEIPT_INSERT, row)
    return row


class ArtifactRunUpgradeTests(unittest.TestCase):
    def test_v19_legacy_reserved_caller_key_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, fixture = _build_two_claim_database(
                Path(directory) / "legacy-caller-key.db"
            )
            with database.transaction() as connection:
                database._downgrade_v20_contract_to_v19(connection)
                connection.execute(
                    "UPDATE meta SET value='19' WHERE key='schema_version'"
                )
            self.assertEqual(
                _schema_contract(database),
                _expected_v19_schema_contract(),
            )

            legacy_key = (
                PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX + ("f" * 64)
            )
            with database.transaction() as connection:
                claim = connection.execute(
                    """SELECT * FROM performance_scoring_claims
                       WHERE attempt_id=?""",
                    (fixture["unfinished_attempt_id"],),
                ).fetchone()
                event = connection.execute(
                    "SELECT * FROM events WHERE event_id=?",
                    (claim["event_id"],),
                ).fetchone()
                trigger_sql = {
                    row["name"]: row["sql"]
                    for row in connection.execute(
                        """SELECT name, sql FROM sqlite_master
                           WHERE type='trigger' AND name IN (
                               'events_no_update',
                               'performance_scoring_claims_no_update'
                           )"""
                    )
                }
                connection.execute("DROP TRIGGER events_no_update")
                connection.execute(
                    "DROP TRIGGER performance_scoring_claims_no_update"
                )
                payload = json.loads(event["payload_json"])
                payload["caller_idempotency_key"] = legacy_key
                connection.execute(
                    """UPDATE performance_scoring_claims
                       SET idempotency_key=? WHERE id=?""",
                    (legacy_key, claim["id"]),
                )
                connection.execute(
                    "UPDATE events SET payload_json=? WHERE event_id=?",
                    (canonical_json(payload), event["event_id"]),
                )
                rehash_event_streams(connection)
                connection.execute(trigger_sql["events_no_update"])
                connection.execute(
                    trigger_sql["performance_scoring_claims_no_update"]
                )

            database.initialize()

            with database.read() as connection:
                preserved = connection.execute(
                    """SELECT idempotency_key
                       FROM performance_scoring_claims
                       WHERE id=?""",
                    (claim["id"],),
                ).fetchone()
            self.assertEqual(preserved["idempotency_key"], legacy_key)
            integrity = database.verify_integrity()
            self.assertTrue(integrity["ok"], integrity["errors"])

    def test_exact_v19_upgrade_is_empty_and_preserves_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = _build_fixture(
                Path(directory) / "upgrade.db",
                "upgrade",
            )["database"]
            assert isinstance(database, Database)
            before_events = _event_snapshot(database)
            with database.read() as connection:
                before_counts = tuple(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                    for table in (
                        "performance_attempts",
                        "performance_actions",
                        "task_evaluations",
                    )
                )
            with database.transaction() as connection:
                database._downgrade_v20_contract_to_v19(connection)
                connection.execute(
                    "UPDATE meta SET value='19' WHERE key='schema_version'"
                )
            self.assertEqual(
                _schema_contract(database),
                _expected_v19_schema_contract(),
            )

            database.initialize()

            self.assertEqual(
                _schema_contract(database),
                _expected_current_schema_contract(),
            )
            self.assertEqual(_event_snapshot(database), before_events)
            with database.read() as connection:
                after_counts = tuple(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                    for table in (
                        "performance_attempts",
                        "performance_actions",
                        "task_evaluations",
                    )
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()["value"],
                    "20",
                )
                self.assertEqual(
                    tuple(
                        connection.execute(
                            """SELECT
                                   (SELECT COUNT(*) FROM
                                    performance_artifact_run_claims),
                                   (SELECT COUNT(*) FROM
                                    performance_artifact_run_receipts)"""
                        ).fetchone()
                    ),
                    (0, 0),
                )
                self.assertEqual(connection.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall(), [])
            self.assertEqual(after_counts, before_counts)

    def test_v19_drift_and_orphan_history_fail_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for case in ("schema drift", "orphan event"):
                with self.subTest(case=case):
                    path = Path(directory) / f"{case.replace(' ', '-')}.db"
                    database = Database(path)
                    database.initialize()
                    with database.transaction() as connection:
                        database._downgrade_v20_contract_to_v19(connection)
                        connection.execute(
                            "UPDATE meta SET value='19' "
                            "WHERE key='schema_version'"
                        )
                        if case == "schema drift":
                            connection.execute(
                                "CREATE INDEX unsupported_v19_artifact_run "
                                "ON learners(display_name)"
                            )
                        else:
                            database.append_event(
                                connection,
                                stream_id="system:schema-v20-orphan",
                                event_type="PerformanceArtifactRunClaimed",
                                payload={},
                            )
                    before = durable_database_fingerprint(path)
                    message = (
                        "exact supported v20 migration source"
                        if case == "schema drift"
                        else "artifact-run event without its v20"
                    )
                    with self.assertRaisesRegex(ConflictError, message):
                        database.initialize()
                    self.assertEqual(durable_database_fingerprint(path), before)
                    with database.read() as connection:
                        self.assertEqual(
                            connection.execute(
                                """SELECT value FROM meta
                                   WHERE key='schema_version'"""
                            ).fetchone()["value"],
                            "19",
                        )
                        self.assertIsNone(
                            connection.execute(
                                """SELECT 1 FROM sqlite_master
                                   WHERE type='table' AND name=
                                   'performance_artifact_run_claims'"""
                            ).fetchone()
                        )


class ArtifactRunStorageGuardTests(unittest.TestCase):
    def test_artifact_technical_keys_cannot_collide_with_legacy_callers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            claim_fixture = _build_fixture(
                Path(directory) / "claim-collision.db",
                "claim-collision",
            )
            claim_terms = _claim_terms(claim_fixture)
            _seed_legacy_scoring_caller_key(
                claim_fixture,
                performance_artifact_run_claim_event_key(
                    str(claim_terms["command_hash"])
                ),
                source="claim",
            )
            claim_database = claim_fixture["database"]
            assert isinstance(claim_database, Database)
            claim_key = performance_artifact_run_claim_event_key(
                str(claim_terms["command_hash"])
            )
            with claim_database.read() as connection:
                with self.assertRaisesRegex(
                    ConflictError,
                    "historical scoring claim caller",
                ):
                    PerformanceLedger._validate_artifact_run_technical_event_key(
                        connection,
                        claim_key,
                    )

            receipt_fixture = _build_fixture(
                Path(directory) / "receipt-collision.db",
                "receipt-collision",
            )
            claim = _insert_claim(receipt_fixture)
            receipt = _receipt_terms(
                claim,
                outcome="runner_failed",
                check_action_id=None,
                result=None,
            )
            _seed_legacy_scoring_caller_key(
                receipt_fixture,
                performance_artifact_run_receipt_event_key(
                    str(receipt["receipt_digest"])
                ),
                source="reconciliation",
            )
            receipt_database = receipt_fixture["database"]
            assert isinstance(receipt_database, Database)
            receipt_key = performance_artifact_run_receipt_event_key(
                str(receipt["receipt_digest"])
            )
            with receipt_database.read() as connection:
                with self.assertRaisesRegex(
                    ConflictError,
                    "historical scoring reconciliation caller",
                ):
                    PerformanceLedger._validate_artifact_run_technical_event_key(
                        connection,
                        receipt_key,
                    )

    def test_scoring_sql_guards_reject_all_technical_namespaces(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(
                Path(directory) / "reserved-scoring.db",
                "reserved-scoring",
            )
            database = fixture["database"]
            attempt = fixture["attempt"]
            artifact = fixture["artifact"]
            assert isinstance(database, Database)
            assert isinstance(attempt, dict)
            assert isinstance(artifact, dict)
            for index, prefix in enumerate(
                (
                    PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
                    PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
                    PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
                    PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
                )
            ):
                key = prefix + ("e" * 64)
                with self.subTest(table="claim", prefix=prefix):
                    with self.assertRaisesRegex(
                        sqlite3.IntegrityError,
                        "reserved event namespace",
                    ):
                        with database.transaction() as connection:
                            connection.execute(
                                """INSERT INTO performance_scoring_claims(
                                       id, event_id, claim_schema_version,
                                       idempotency_key, attempt_id,
                                       evaluation_id, through_sequence,
                                       provider_id, provider_version,
                                       action_trace_digest, command_hash,
                                       claimed_at
                                   ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    f"reserved-scoring-claim-{index}",
                                    attempt["event_id"],
                                    key,
                                    attempt["id"],
                                    f"reserved-evaluation-{index}",
                                    artifact["sequence"],
                                    "reserved.provider",
                                    "v1",
                                    "1" * 64,
                                    f"{index + 1:x}" * 64,
                                    to_timestamp(START),
                                ),
                            )
                with self.subTest(table="reconciliation", prefix=prefix):
                    with self.assertRaisesRegex(
                        sqlite3.IntegrityError,
                        "reserved event namespace",
                    ):
                        with database.transaction() as connection:
                            connection.execute(
                                """INSERT INTO
                                       performance_scoring_reconciliations(
                                       id, event_id, idempotency_key, claim_id,
                                       attempt_id, evaluation_id, outcome,
                                       scoring_request_digest,
                                       provider_binding_digest,
                                       provider_operation_digest, reconciler_id,
                                       reconciler_version,
                                       reconciliation_binding_digest,
                                       receipt_json, receipt_digest,
                                       normalized_result_digest, reconciled_at,
                                       command_hash
                                   ) VALUES (
                                       ?, ?, ?, ?, ?, ?, 'unknown', ?, ?, ?, ?,
                                       ?, ?, '{}', ?, NULL, ?, ?
                                   )""",
                                (
                                    f"reserved-reconciliation-{index}",
                                    artifact["event_id"],
                                    key,
                                    "missing-claim",
                                    attempt["id"],
                                    f"reserved-evaluation-{index}",
                                    "2" * 64,
                                    "3" * 64,
                                    "4" * 64,
                                    "reserved.reconciler",
                                    "v1",
                                    "5" * 64,
                                    "6" * 64,
                                    to_timestamp(START),
                                    f"{index + 5:x}" * 64,
                                ),
                            )

    def test_closed_outcome_matrix_is_event_backed_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index, outcome in enumerate(
                (
                    "completed",
                    "invalid_artifact",
                    "runner_failed",
                    "timed_out",
                )
            ):
                with self.subTest(outcome=outcome):
                    fixture = _build_fixture(
                        Path(directory) / f"{outcome}.db",
                        f"outcome-{index}",
                    )
                    claim = _insert_claim(fixture)
                    receipt = _insert_receipt(fixture, claim, outcome)
                    database = fixture["database"]
                    assert isinstance(database, Database)
                    with database.read() as connection:
                        stored = connection.execute(
                            """SELECT * FROM
                               performance_artifact_run_receipts"""
                        ).fetchone()
                        self.assertEqual(stored["outcome"], outcome)
                        self.assertEqual(
                            stored["check_action_id"] is not None,
                            outcome in {"completed", "invalid_artifact"},
                        )
                        event = connection.execute(
                            "SELECT * FROM events WHERE event_id=?",
                            (receipt["event_id"],),
                        ).fetchone()
                        self.assertEqual(
                            event["causation_id"],
                            claim["event_id"],
                        )
                    for table, row_id in (
                        ("performance_artifact_run_claims", claim["id"]),
                        (
                            "performance_artifact_run_receipts",
                            receipt["id"],
                        ),
                    ):
                        with self.assertRaisesRegex(
                            sqlite3.IntegrityError,
                            "immutable",
                        ):
                            with database.transaction() as connection:
                                connection.execute(
                                    f"UPDATE {table} SET id=id WHERE id=?",
                                    (row_id,),
                                )
                        with self.assertRaisesRegex(
                            sqlite3.IntegrityError,
                            "immutable",
                        ):
                            with database.transaction() as connection:
                                connection.execute(
                                    f"DELETE FROM {table} WHERE id=?",
                                    (row_id,),
                                )

    def test_claim_guard_rejects_nonmatching_event_or_boundary(self) -> None:
        mutations = {
            "event payload": lambda row, payload, _metadata: payload.update(
                {"artifact_digest": "6" * 64}
            ),
            "session revision": lambda row, payload, _metadata: (
                row.update({"session_revision": row["session_revision"] + 1}),
                payload.update(
                    {"session_revision": payload["session_revision"] + 1}
                ),
            ),
            "metadata authority": lambda _row, _payload, metadata: (
                metadata.update({"projection_applied": False})
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for index, (case, mutation) in enumerate(mutations.items()):
                with self.subTest(case=case):
                    fixture = _build_fixture(
                        Path(directory) / f"claim-{index}.db",
                        f"claim-{index}",
                    )
                    with self.assertRaisesRegex(
                        sqlite3.IntegrityError,
                        "claim does not match",
                    ):
                        _insert_claim(fixture, mutate=mutation)
                    database = fixture["database"]
                    assert isinstance(database, Database)
                    with database.read() as connection:
                        self.assertEqual(
                            connection.execute(
                                """SELECT COUNT(*) FROM
                                   performance_artifact_run_claims"""
                            ).fetchone()[0],
                            0,
                        )
                        self.assertEqual(
                            connection.execute(
                                """SELECT COUNT(*) FROM events
                                   WHERE event_type=
                                   'PerformanceArtifactRunClaimed'"""
                            ).fetchone()[0],
                            0,
                        )

    def test_stale_attempt_and_session_leave_claim_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(
                Path(directory) / "stale.db",
                "stale",
            )
            claim = _insert_claim(fixture)
            ledger = fixture["ledger"]
            engine = fixture["engine"]
            attempt = fixture["attempt"]
            session = fixture["session"]
            database = fixture["database"]
            assert isinstance(ledger, PerformanceLedger)
            assert isinstance(engine, AdaptiveEngine)
            assert isinstance(attempt, dict)
            assert isinstance(session, dict)
            assert isinstance(database, Database)
            ledger.record_action(
                str(attempt["id"]),
                "submitted",
                {"submission_digest": ARTIFACT_DIGEST},
                idempotency_key="artifact-run-stale-submit",
                now=START + timedelta(minutes=4),
            )
            engine.end_session(
                str(session["id"]),
                status="completed",
                idempotency_key="artifact-run-stale-session",
                now=START + timedelta(minutes=5),
            )
            row = _receipt_terms(
                claim,
                outcome="runner_failed",
                check_action_id=None,
                result=None,
            )
            with self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "receipt does not match",
            ):
                with database.transaction() as connection:
                    event = database.append_event(
                        connection,
                        stream_id=f"learner:{attempt['learner_id']}",
                        event_type="PerformanceArtifactRunObserved",
                        payload=performance_artifact_run_observed_payload(
                            receipt_id=str(row["id"]),
                            claim_id=str(row["claim_id"]),
                            attempt_id=str(attempt["id"]),
                            check_action_id=None,
                            outcome="runner_failed",
                            result=None,
                            result_digest=None,
                            receipt=row["receipt"],
                            receipt_digest=str(row["receipt_digest"]),
                            started_at=str(row["started_at"]),
                            completed_at=str(row["completed_at"]),
                        ),
                        metadata={
                            "artifact_run_schema_version": 1,
                            "observational_only": True,
                            "projection_applied": False,
                            "certification_applied": False,
                            "skill_authority": False,
                            "shadow_only": True,
                        },
                        learner_id=str(attempt["learner_id"]),
                        session_id=None,
                        idempotency_key=(
                            performance_artifact_run_receipt_event_key(
                                str(row["receipt_digest"])
                            )
                        ),
                        correlation_id=str(attempt["id"]),
                        causation_id=str(claim["event_id"]),
                        occurred_at=datetime.fromisoformat(
                            str(row["completed_at"])
                        ),
                    )
                    row["event_id"] = event["event_id"]
                    connection.execute(_RECEIPT_INSERT, row)
            with database.read() as connection:
                self.assertEqual(
                    connection.execute(
                        """SELECT COUNT(*) FROM
                           performance_artifact_run_claims"""
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    connection.execute(
                        """SELECT COUNT(*) FROM
                           performance_artifact_run_receipts"""
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    connection.execute(
                        """SELECT COUNT(*) FROM events
                           WHERE event_type=
                           'PerformanceArtifactRunObserved'"""
                    ).fetchone()[0],
                    0,
                )
            with self.assertRaises(sqlite3.IntegrityError):
                with database.transaction() as connection:
                    connection.execute(
                        _CLAIM_INSERT,
                        {
                            **claim,
                            "id": "artifact-run-claim-automatic-retry",
                            "event_id": claim["event_id"],
                        },
                    )


if __name__ == "__main__":
    unittest.main()
