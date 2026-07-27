# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import PropertyMock, patch

from tsq.artifact_runner import (
    ArtifactProcessReceipt,
    ArtifactResultCode,
    ArtifactRunOutcome,
    ArtifactRunRequest,
    ArtifactRunnerBinding,
)
from tsq.errors import ConflictError, ValidationError
from tsq.evidence import (
    ActionKind,
    ActionPhase,
    EvaluationStatus,
    canonical_digest,
)
from tsq.performance import ImportedCriterionResult, ImportedEvaluation
from tsq.performance_ledger import _command_hash, derive_performance_projections
from tsq.replay import ProjectionReplay
from tsq.store import (
    PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
    Database,
)

from tests import test_artifact_run_ledger as ledger_fixture
from tests.test_artifact_run_schema import START, _build_fixture, _insert_claim


_RESERVED_CALLER_PREFIXES = (
    PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
)


def _new_ledger_fixture(test: unittest.TestCase, suffix: str):
    fixture = ledger_fixture.ArtifactRunLedgerTests(
        "test_success_commits_claim_system_action_and_exact_receipts"
    )
    fixture.setUp()
    test.addCleanup(fixture.tearDown)
    attempt, checkpoint, captured = fixture.start_with_artifact(
        ledger_fixture.artifact([[True]]),
        key_suffix=suffix,
    )
    return fixture, attempt, checkpoint, captured


def _strand_claim(
    test: unittest.TestCase,
    suffix: str,
    *,
    forged_digest_type: type | None = None,
) -> tuple[object, dict[str, object]]:
    fixture, attempt, checkpoint, captured = _new_ledger_fixture(test, suffix)
    with ExitStack() as stack:
        if forged_digest_type is not None:
            stack.enter_context(
                patch.object(
                    forged_digest_type,
                    "digest",
                    new_callable=PropertyMock,
                    return_value="e" * 64,
                )
            )
        stack.enter_context(
            patch.object(
                fixture.registry,
                "run",
                side_effect=RuntimeError("strand after durable admission"),
            )
        )
        with test.assertRaisesRegex(ValidationError, "remains unresolved"):
            fixture.run_check(attempt, checkpoint, captured)
    with fixture.database.read() as connection:
        claim = dict(
            connection.execute(
                "SELECT * FROM performance_artifact_run_claims"
            ).fetchone()
        )
    return fixture, claim


class ArtifactRunAdversarialIntegrityTests(unittest.TestCase):
    def assert_integrity_rejects(
        self,
        database: Database,
        *,
        identity: str,
        vocabulary: tuple[str, ...],
    ) -> None:
        report = database.verify_integrity()
        errors = "\n".join(report["errors"]).lower()
        self.assertFalse(report["ok"], report)
        self.assertIn(identity.lower(), errors)
        self.assertTrue(
            any(word in errors for word in vocabulary),
            report["errors"],
        )

    def test_orphan_artifact_run_events_fail_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index, event_type in enumerate(
                (
                    "PerformanceArtifactRunClaimed",
                    "PerformanceArtifactRunObserved",
                )
            ):
                with self.subTest(event_type=event_type):
                    database = Database(Path(directory) / f"{index}.db")
                    database.initialize()
                    self.assertTrue(database.verify_integrity()["ok"])
                    with database.transaction() as connection:
                        event = database.append_event(
                            connection,
                            stream_id=f"system:orphan-artifact-run:{index}",
                            event_type=event_type,
                            payload={},
                        )
                    self.assert_integrity_rejects(
                        database,
                        identity=event["event_id"],
                        vocabulary=("projection", "orphan"),
                    )

    def test_orphan_receipt_projection_rebuilds_only_database_copy(
        self,
    ) -> None:
        fixture, attempt, checkpoint, captured = _new_ledger_fixture(
            self, "orphan-replay"
        )
        run = fixture.run_check(attempt, checkpoint, captured)
        with fixture.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_artifact_run_receipts_no_delete"
            )
            connection.execute(
                """DELETE FROM performance_artifact_run_receipts
                   WHERE claim_id=?""",
                (run["claim_id"],),
            )

        report = ProjectionReplay(fixture.database).check(
            "artifact-run-learner"
        )
        self.assertFalse(report["ok"])
        self.assertTrue(report["rebuild_safe"], report["errors"])
        self.assertEqual(
            report["source_performance_artifact_run_receipt_count"], 0
        )
        self.assertEqual(
            report["reconstructed_performance_artifact_run_receipt_count"], 1
        )

        target = Path(fixture.tempdir.name) / "artifact-rebuilt.db"
        rebuilt_report = ProjectionReplay(fixture.database).rebuild_copy(
            "artifact-run-learner",
            target,
        )
        self.assertTrue(rebuilt_report["ok"], rebuilt_report["errors"])
        with fixture.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_artifact_run_receipts"""
                ).fetchone()["n"],
                0,
            )
        rebuilt = Database(target, read_only=True)
        self.assertTrue(rebuilt.verify_integrity()["ok"])
        with rebuilt.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_artifact_run_receipts"""
                ).fetchone()["n"],
                1,
            )

    def test_typed_claim_digest_forgery_fails_integrity(self) -> None:
        for digest_type, label in (
            (ArtifactRunRequest, "request"),
            (ArtifactRunnerBinding, "binding"),
        ):
            with self.subTest(digest=label):
                fixture, claim = _strand_claim(
                    self,
                    f"forged-{label}",
                    forged_digest_type=digest_type,
                )
                self.assert_integrity_rejects(
                    fixture.database,
                    identity=str(claim["id"]),
                    vocabulary=(label, "digest"),
                )

    def test_forged_check_action_semantics_fail_before_terminal_commit(
        self,
    ) -> None:
        fixture, attempt, checkpoint, captured = _new_ledger_fixture(
            self, "forged-check-action"
        )
        original_terms = ArtifactProcessReceipt.terms

        def forged_terms(receipt: ArtifactProcessReceipt):
            terms = original_terms(receipt)
            terms["result"]["outcome_codes"] = [
                "causal_visibility_invalid",
                "matrix_shape_valid",
            ]
            terms["result"]["passed"] = 1
            terms["result"]["failed"] = 1
            terms["result_digest"] = canonical_digest(terms["result"])
            return terms

        with patch.object(
            ArtifactProcessReceipt,
            "terms",
            forged_terms,
        ), self.assertRaisesRegex(
            ConflictError,
            "mismatched generated check action",
        ):
            fixture.run_check(attempt, checkpoint, captured)
        runs = fixture.ledger.list_artifact_runs(attempt_id=attempt["id"])
        self.assertEqual(runs[0]["status"], "unresolved")
        with fixture.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_artifact_run_receipts"""
                ).fetchone()["n"],
                0,
            )
        self.assertTrue(fixture.database.verify_integrity()["ok"])

    def test_replay_rejects_session_end_between_claim_and_failed_receipt(
        self,
    ) -> None:
        fixture, attempt, checkpoint, captured = _new_ledger_fixture(
            self, "ended-during-run"
        )
        session = fixture.database.get_session(attempt["session_id"])
        ended_at = ledger_fixture.BASE + timedelta(minutes=3)

        def timeout_after_end(request, _material):
            with fixture.database.transaction() as connection:
                fixture.database.append_event(
                    connection,
                    stream_id=f"learner:{session['learner_id']}",
                    event_type="SessionEnded",
                    payload={
                        "session_id": session["id"],
                        "status": "completed",
                        "reason": "adversarial runner race",
                    },
                    metadata={
                        "corpus_release_id": session["corpus_release_id"],
                    },
                    learner_id=session["learner_id"],
                    session_id=session["id"],
                    occurred_at=ended_at,
                )
            return fixture._process_failure(
                request,
                outcome=ArtifactRunOutcome.TIMED_OUT,
                code=ArtifactResultCode.WORKER_TIMEOUT,
                started=True,
            )

        with patch.object(
            fixture.registry,
            "run",
            side_effect=timeout_after_end,
        ):
            run = fixture.run_check(attempt, checkpoint, captured)
        self.assertEqual(run["status"], "timed_out")

        # Reconcile only the ordinary session projection so the immutable
        # interleaving, rather than a stale session row, is what replay rejects.
        with fixture.database.transaction() as connection:
            connection.execute(
                """UPDATE sessions
                   SET status='completed', revision=revision + 1, updated_at=?
                   WHERE id=?""",
                (ended_at.isoformat(), session["id"]),
            )

        with fixture.database.read() as connection, self.assertRaisesRegex(
            ValidationError,
            "crosses a session revision or end boundary",
        ):
            derive_performance_projections(connection)

    def test_list_and_inspect_reject_receipt_row_contradictions(self) -> None:
        for contradiction in ("outcome", "completed_at"):
            with self.subTest(contradiction=contradiction):
                fixture, attempt, checkpoint, captured = _new_ledger_fixture(
                    self, f"receipt-{contradiction}"
                )
                run = fixture.run_check(attempt, checkpoint, captured)
                with fixture.database.transaction() as connection:
                    connection.execute(
                        "DROP TRIGGER performance_artifact_run_receipts_no_update"
                    )
                    if contradiction == "outcome":
                        connection.execute(
                            """UPDATE performance_artifact_run_receipts
                               SET outcome='invalid_artifact'
                               WHERE claim_id=?""",
                            (run["claim_id"],),
                        )
                    else:
                        completed_at = (
                            datetime.fromisoformat(run["completed_at"])
                            + timedelta(seconds=1)
                        ).isoformat()
                        connection.execute(
                            """UPDATE performance_artifact_run_receipts
                               SET completed_at=?
                               WHERE claim_id=?""",
                            (completed_at, run["claim_id"]),
                        )
                for operation in (
                    lambda: fixture.ledger.list_artifact_runs(
                        attempt_id=attempt["id"]
                    ),
                    lambda: fixture.ledger.inspect_artifact_run(
                        run["claim_id"]
                    ),
                ):
                    with self.assertRaisesRegex(
                        ConflictError,
                        "mismatched terminal receipt",
                    ):
                        operation()

    def test_list_and_inspect_reject_forged_check_counters(self) -> None:
        fixture, attempt, checkpoint, captured = _new_ledger_fixture(
            self, "display-check-counters"
        )
        run = fixture.run_check(attempt, checkpoint, captured)
        forged_payload = dict(run["check_action"]["payload"])
        forged_payload.update({"passed": 1, "failed": 1})
        with fixture.database.transaction() as connection:
            connection.execute("DROP TRIGGER performance_actions_no_update")
            connection.execute(
                """UPDATE performance_actions SET payload_json=?
                   WHERE id=?""",
                (
                    json.dumps(
                        forged_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    run["check_action"]["id"],
                ),
            )
        for operation in (
            lambda: fixture.ledger.list_artifact_runs(
                attempt_id=attempt["id"]
            ),
            lambda: fixture.ledger.inspect_artifact_run(run["claim_id"]),
        ):
            with self.assertRaisesRegex(
                ConflictError,
                "mismatched generated check action",
            ):
                operation()

    def test_scoring_rejects_check_counters_forged_after_receipt(self) -> None:
        fixture, attempt, checkpoint, captured = _new_ledger_fixture(
            self, "forged-scoring-source"
        )
        run = fixture.run_check(attempt, checkpoint, captured)
        forged_payload = dict(run["check_action"]["payload"])
        forged_payload.update({"passed": 1, "failed": 1})
        with fixture.database.transaction() as connection:
            connection.execute("DROP TRIGGER performance_actions_no_update")
            connection.execute(
                """UPDATE performance_actions SET payload_json=?
                   WHERE id=?""",
                (
                    json.dumps(
                        forged_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    run["check_action"]["id"],
                ),
            )
        fixture.ledger.record_action(
            attempt["id"],
            ActionKind.SUBMITTED.value,
            {"submission_digest": captured.sha256},
            now=ledger_fixture.BASE + timedelta(minutes=4),
        )
        observation = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_causal_mask",
                    status=EvaluationStatus.VALID,
                    score=1.0,
                    outcome_code="forged_counter_source",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(run["check_action"]["id"],),
                ),
            )
        )
        with self.assertRaisesRegex(
            ValidationError,
            "terminal runner receipt",
        ):
            fixture.ledger.score_attempt(
                attempt["id"],
                fixture._provider_registry(observation),
                fixture.authority.provider_id,
                fixture.authority.provider_version,
                now=ledger_fixture.BASE + timedelta(minutes=5),
            )

    def test_reserved_caller_key_namespaces_fail_at_sql_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index, prefix in enumerate(_RESERVED_CALLER_PREFIXES):
                with self.subTest(prefix=prefix):
                    fixture = _build_fixture(
                        Path(directory) / f"reserved-{index}.db",
                        f"reserved-{index}",
                    )
                    key = prefix + ("a" * 64)

                    def use_reserved_key(row, payload, _metadata):
                        row["idempotency_key"] = key
                        payload["caller_idempotency_key"] = key

                    with self.assertRaisesRegex(
                        sqlite3.IntegrityError,
                        "reserved",
                    ):
                        _insert_claim(fixture, mutate=use_reserved_key)

    def test_claim_event_must_follow_through_sequence_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _build_fixture(
                Path(directory) / "boundary-order.db",
                "boundary-order",
            )
            database = fixture["database"]
            ledger = fixture["ledger"]
            attempt = fixture["attempt"]
            artifact = fixture["artifact"]
            original_append = database.append_event

            def append_then_boundary(connection, **kwargs):
                event = original_append(connection, **kwargs)
                if kwargs["event_type"] == "PerformanceArtifactRunClaimed":
                    attempt_row = connection.execute(
                        "SELECT * FROM performance_attempts WHERE id=?",
                        (attempt["id"],),
                    ).fetchone()
                    ledger._append_action(
                        connection,
                        attempt_row,
                        ActionKind.HINT_REQUESTED,
                        ActionPhase(artifact["phase"]),
                        {"hint_id": "late_boundary", "level": 1},
                        occurred=START + timedelta(minutes=2, seconds=30),
                        idempotency_key=None,
                        command_hash=_command_hash(
                            {"operation": "adversarial_late_boundary"}
                        ),
                    )
                return event

            def claim_later_action(row, payload, _metadata):
                row["through_sequence"] += 1
                payload["through_sequence"] += 1

            with patch.object(
                database,
                "append_event",
                side_effect=append_then_boundary,
            ), self.assertRaisesRegex(
                sqlite3.IntegrityError,
                "claim.*boundary",
            ):
                _insert_claim(fixture, mutate=claim_later_action)


if __name__ == "__main__":
    unittest.main()
