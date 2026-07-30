# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.evidence import (
    ActionKind,
    ActionPhase,
    CriterionScale,
    LearningAction,
    LearningTask,
    RubricCriterion,
    TaskModality,
)
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
    derive_performance_projections,
    require_performance_attempt_trace_consistency,
    require_performance_projection_consistency,
)
from tsq.store import Database

from tests import test_artifact_run_ledger as artifact_run_fixture
from tests import test_artifact_run_schema as artifact_fixture


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2110, 6, 7, 9, 0, tzinfo=timezone.utc)
_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64


class PerformanceTraceScopeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "trace-scope.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)
        self.learner_id = "trace-scope-learner"
        self.engine.create_learner(self.learner_id, "Trace Scope Learner")
        self.session = self.engine.start_session(
            self.learner_id,
            "t_transformers",
            seed=7711,
            now=START,
        )
        with self.database.read() as connection:
            self.corpus_release_id = connection.execute(
                "SELECT value FROM meta WHERE key='active_corpus_release'"
            ).fetchone()["value"]
            source = connection.execute(
                """SELECT source.id, source.content_hash
                   FROM release_sources membership
                   JOIN sources source ON source.id=membership.source_id
                   WHERE membership.release_id=?
                     AND source.id='src_vaswani_attention_2017'""",
                (self.corpus_release_id,),
            ).fetchone()
        self.task = LearningTask(
            id="task_trace_scope_attention",
            version=1,
            family_id="family_trace_scope_attention",
            title="Diagnose a causal attention mask",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_trace_scope",
                    name="Causal mask invariant",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_causal_masking", 1.0),),
                    dependence_group="trace_scope_behavior",
                    evidence_cap=0.8,
                    dependence_cap=0.8,
                ),
            ),
            instructions=(
                "Inspect the pinned causal-mask stimulus, diagnose the "
                "invariant violation, and submit a content-addressed repair."
            ),
            source_manifests=((source["id"], source["content_hash"]),),
            administration_id="admin_trace_scope",
            administration_manifest_digest=_D0,
            stimulus_id="stimulus_trace_scope",
            stimulus_digest=_D1,
        )
        release_time = datetime.now(timezone.utc)
        release = PerformanceTaskRelease(
            title="Attempt-scoped trace test release",
            corpus_release_id=self.corpus_release_id,
            review=TaskReleaseReview(
                reviewer_kind="human",
                reviewer_id="trace_scope_reviewer",
                reviewed_at=release_time.isoformat(),
                independent_of_author=True,
                attestation_digest=_D2,
            ),
            tasks=(("pilot", self.task),),
        )
        self.ledger = PerformanceLedger(self.database)
        self.release_id = self.ledger.publish_release(
            release,
            now=release_time,
        )["release_id"]

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def start_attempt(
        self,
        session_id: str,
        *,
        key: str,
        now: datetime,
    ) -> dict:
        return self.ledger.start_attempt(
            session_id,
            self.task.id,
            task_version=self.task.version,
            task_release_id=self.release_id,
            idempotency_key=key,
            now=now,
        )

    def finish_attempt(
        self,
        attempt_id: str,
        *,
        key_prefix: str,
        now: datetime,
    ) -> dict:
        self.ledger.record_action(
            attempt_id,
            "artifact_checkpoint",
            {"artifact_digest": _D3, "artifact_kind": "patch_digest"},
            idempotency_key=f"{key_prefix}-artifact",
            now=now,
        )
        return self.ledger.record_action(
            attempt_id,
            "submitted",
            {"submission_digest": _D3},
            idempotency_key=f"{key_prefix}-submit",
            now=now + timedelta(seconds=1),
        )

    def count_validation_vm_steps(self, attempt_id: str) -> int:
        """Count deterministic SQLite VM progress callbacks for one validation."""

        callbacks = 0

        def progress() -> int:
            nonlocal callbacks
            callbacks += 1
            return 0

        with self.database.read() as connection:
            connection.set_progress_handler(progress, 10)
            try:
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=attempt_id,
                )
            finally:
                connection.set_progress_handler(None, 0)
        return callbacks * 10

    def scoped_select_plan(self, attempt_id: str) -> list[str]:
        """Capture and explain the exact attempt-scoped event SELECT."""

        statements: list[str] = []
        with self.database.read() as connection:
            connection.set_trace_callback(statements.append)
            try:
                derive_performance_projections(
                    connection,
                    attempt_id=attempt_id,
                    trace_only=True,
                )
            finally:
                connection.set_trace_callback(None)
            scoped = tuple(
                dict.fromkeys(
                    statement
                    for statement in statements
                    if statement.lstrip().startswith(
                        (
                            "WITH scoped_session_ids(session_id)",
                            "WITH scoped_values(value)",
                        )
                    )
                    and (
                        " events AS event " in statement
                        or " JOIN events AS event " in statement
                    )
                )
            )
            self.assertTrue(
                scoped,
                "Attempt derivation did not execute scoped event SELECTs.",
            )
            return [
                row["detail"]
                for statement in scoped
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN " + statement
                ).fetchall()
            ]

    def test_unrelated_same_learner_session_corruption_is_isolated(self) -> None:
        target = self.start_attempt(
            self.session["id"],
            key="scope-target",
            now=START + timedelta(minutes=1),
        )
        peer_session = self.engine.start_session(
            self.learner_id,
            "t_transformers",
            seed=7712,
            now=START + timedelta(days=1),
        )
        peer = self.start_attempt(
            peer_session["id"],
            key="scope-unrelated-peer",
            now=START + timedelta(days=1, minutes=1),
        )
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER performance_actions_no_update")
            connection.execute(
                """UPDATE performance_actions
                   SET elapsed_ms=elapsed_ms + 1
                   WHERE attempt_id=? AND sequence=0""",
                (peer["id"],),
            )

        with self.database.read() as connection:
            derived, _checkpoints = derive_performance_projections(
                connection,
                attempt_id=target["id"],
                trace_only=True,
            )
            self.assertEqual(
                {row["id"] for row in derived["attempts"]},
                {target["id"]},
            )
            require_performance_attempt_trace_consistency(
                connection,
                attempt_id=target["id"],
            )
            with self.assertRaisesRegex(
                ValidationError,
                "immutable event derivation",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=peer["id"],
                )

        self.assertEqual(len(self.ledger.list_actions(target["id"])), 1)

    def test_shared_generic_check_digest_does_not_join_sessions(self) -> None:
        target = self.start_attempt(
            self.session["id"],
            key="scope-shared-digest-target",
            now=START + timedelta(minutes=1),
        )
        peer_session = self.engine.start_session(
            self.learner_id,
            "t_transformers",
            seed=7713,
            now=START + timedelta(days=1),
        )
        peer = self.start_attempt(
            peer_session["id"],
            key="scope-shared-digest-peer",
            now=START + timedelta(days=1, minutes=1),
        )
        check_payload = {
            "check_set_id": "generic_checks_v1",
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "skipped": 0,
            "result_digest": _D3,
        }
        target_check = self.ledger.record_action(
            target["id"],
            "check_run",
            check_payload,
            now=START + timedelta(minutes=2),
        )
        peer_check = self.ledger.record_action(
            peer["id"],
            "check_run",
            check_payload,
            now=START + timedelta(days=1, minutes=2),
        )

        with self.database.read() as connection:
            derived = require_performance_attempt_trace_consistency(
                connection,
                attempt_id=target["id"],
            )
        self.assertIn(
            target_check["id"],
            {row["id"] for row in derived["actions"]},
        )
        self.assertNotIn(
            peer_check["id"],
            {row["id"] for row in derived["actions"]},
        )

    def test_read_context_pins_validation_and_use_snapshot(self) -> None:
        with self.database.read() as reader:
            self.assertTrue(reader.in_transaction)
            before = reader.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]
            with self.database.transaction() as writer:
                self.database.append_event(
                    writer,
                    stream_id="system:read-snapshot-probe",
                    event_type="ReadSnapshotProbe",
                    schema_version=1,
                    payload={},
                    metadata={},
                    correlation_id="read-snapshot-probe",
                    occurred_at=START + timedelta(minutes=1),
                )
            during = reader.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]
            self.assertEqual(during, before)

        with self.database.read() as reader:
            after = reader.execute(
                "SELECT COUNT(*) AS n FROM events"
            ).fetchone()["n"]
        self.assertEqual(after, before + 1)

    def test_same_session_missing_terminal_event_exposes_overlap(self) -> None:
        first = self.start_attempt(
            self.session["id"],
            key="scope-first",
            now=START + timedelta(minutes=1),
        )
        terminal = self.finish_attempt(
            first["id"],
            key_prefix="scope-first",
            now=START + timedelta(minutes=2),
        )
        second = self.start_attempt(
            self.session["id"],
            key="scope-second",
            now=START + timedelta(minutes=3),
        )
        with self.database.transaction() as connection:
            action = connection.execute(
                """SELECT event_id FROM performance_actions WHERE id=?""",
                (terminal["id"],),
            ).fetchone()
            connection.execute("DROP TRIGGER performance_actions_no_delete")
            connection.execute("DROP TRIGGER events_no_delete")
            connection.execute(
                "DELETE FROM performance_actions WHERE id=?",
                (terminal["id"],),
            )
            connection.execute(
                "DELETE FROM events WHERE event_id=?",
                (action["event_id"],),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "overlaps active attempt",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=second["id"],
                )

    def test_target_start_attribution_cannot_escape_attempt_scope(self) -> None:
        attempt = self.start_attempt(
            self.session["id"],
            key="scope-attribution",
            now=START + timedelta(minutes=1),
        )
        with self.database.transaction() as connection:
            row = connection.execute(
                """SELECT event_id FROM performance_attempts WHERE id=?""",
                (attempt["id"],),
            ).fetchone()
            connection.execute("DROP TRIGGER events_no_update")
            connection.execute(
                """UPDATE events SET correlation_id=?
                   WHERE event_id=?""",
                ("malformed-target-attribution", row["event_id"]),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "invalid stream or causal envelope",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=attempt["id"],
                )

    def test_nested_action_attribution_cannot_escape_all_outer_fields(
        self,
    ) -> None:
        attempt = self.start_attempt(
            self.session["id"],
            key="scope-nested-attribution",
            now=START + timedelta(minutes=1),
        )
        forged_action = LearningAction(
            id="pact_nested_attribution_escape",
            trace_id=attempt["id"],
            sequence=1,
            kind=ActionKind.ANSWER_REVISED,
            phase=ActionPhase.UNASSISTED,
            payload={"answer_digest": _D3},
            elapsed_ms=60_000,
        )
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id="system:trace-scope-escape",
                event_type="PerformanceActionRecorded",
                schema_version=1,
                payload={
                    "attempt_id": "pta_escaped_outer",
                    "action": forged_action.terms(),
                },
                metadata={
                    "command_hash": _D0,
                    "action_schema_version": 1,
                    "task_digest": attempt["task_digest"],
                    "task_release_id": attempt["task_release_id"],
                    "corpus_release_id": attempt["corpus_release_id"],
                    "observational_only": True,
                    "shadow_only": True,
                },
                learner_id=None,
                session_id=None,
                correlation_id="escaped-correlation",
                causation_id="escaped-causation",
                occurred_at=START + timedelta(minutes=2),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "action boundary mismatch",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=attempt["id"],
                )

    def test_action_event_id_causation_cannot_escape_attempt_scope(
        self,
    ) -> None:
        attempt = self.start_attempt(
            self.session["id"],
            key="scope-action-event-causation",
            now=START + timedelta(minutes=1),
        )
        with self.database.read() as connection:
            action_event_id = connection.execute(
                """SELECT event_id FROM performance_actions
                   WHERE attempt_id=? AND sequence=0""",
                (attempt["id"],),
            ).fetchone()["event_id"]
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id=f"learner:{self.learner_id}",
                event_type="PerformanceArtifactRunClaimed",
                schema_version=1,
                payload={"attempt_id": "pta_escaped_outer"},
                metadata={},
                learner_id=self.learner_id,
                session_id=None,
                correlation_id="escaped-correlation",
                causation_id=action_event_id,
                occurred_at=START + timedelta(minutes=2),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "artifact-run claim",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=attempt["id"],
                )

    def test_start_event_causal_descendant_cannot_escape_attempt_scope(
        self,
    ) -> None:
        attempt = self.start_attempt(
            self.session["id"],
            key="scope-start-event-descendant",
            now=START + timedelta(minutes=1),
        )
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id=f"learner:{self.learner_id}",
                event_type="PerformanceArtifactRunClaimed",
                schema_version=1,
                payload={"attempt_id": "pta_escaped_outer"},
                metadata={},
                learner_id=self.learner_id,
                session_id=None,
                correlation_id="escaped-correlation",
                causation_id=attempt["event_id"],
                occurred_at=START + timedelta(minutes=2),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "artifact-run claim",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=attempt["id"],
                )

    def test_action_identity_reference_cannot_escape_attempt_scope(
        self,
    ) -> None:
        attempt = self.start_attempt(
            self.session["id"],
            key="scope-action-identity",
            now=START + timedelta(minutes=1),
        )
        with self.database.read() as connection:
            action_id = connection.execute(
                """SELECT id FROM performance_actions
                   WHERE attempt_id=? AND sequence=0""",
                (attempt["id"],),
            ).fetchone()["id"]
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id=f"learner:{self.learner_id}",
                event_type="PerformanceArtifactRunClaimed",
                schema_version=1,
                payload={
                    "attempt_id": "pta_escaped_outer",
                    "artifact_action_id": action_id,
                },
                metadata={},
                learner_id=self.learner_id,
                session_id=None,
                correlation_id="escaped-correlation",
                causation_id="escaped-causation",
                occurred_at=START + timedelta(minutes=2),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "artifact-run claim",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=attempt["id"],
                )

    def test_duplicate_action_identity_cannot_escape_attempt_scope(
        self,
    ) -> None:
        attempt = self.start_attempt(
            self.session["id"],
            key="scope-action-id",
            now=START + timedelta(minutes=1),
        )
        with self.database.read() as connection:
            action_id = connection.execute(
                """SELECT id FROM performance_actions
                   WHERE attempt_id=? AND sequence=0""",
                (attempt["id"],),
            ).fetchone()["id"]
        forged_action = LearningAction(
            id=action_id,
            trace_id="pta_escaped_trace",
            sequence=0,
            kind=ActionKind.STARTED,
            phase=ActionPhase.UNASSISTED,
            payload={},
            elapsed_ms=0,
        )
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id=f"learner:{self.learner_id}",
                event_type="PerformanceActionRecorded",
                schema_version=1,
                payload={
                    "attempt_id": forged_action.trace_id,
                    "action": forged_action.terms(),
                },
                metadata={
                    "command_hash": _D0,
                    "action_schema_version": 1,
                    "task_digest": attempt["task_digest"],
                    "task_release_id": attempt["task_release_id"],
                    "corpus_release_id": attempt["corpus_release_id"],
                    "observational_only": True,
                    "shadow_only": True,
                },
                learner_id=self.learner_id,
                session_id=None,
                correlation_id="escaped-correlation",
                causation_id="escaped-causation",
                occurred_at=START + timedelta(minutes=2),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "action precedes attempt",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=attempt["id"],
                )

    def test_check_action_reference_cannot_escape_attempt_scope(
        self,
    ) -> None:
        attempt = self.start_attempt(
            self.session["id"],
            key="scope-check-action-reference",
            now=START + timedelta(minutes=1),
        )
        with self.database.read() as connection:
            action_id = connection.execute(
                """SELECT id FROM performance_actions
                   WHERE attempt_id=? AND sequence=0""",
                (attempt["id"],),
            ).fetchone()["id"]
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id=f"learner:{self.learner_id}",
                event_type="PerformanceArtifactRunObserved",
                schema_version=1,
                payload={
                    "attempt_id": "pta_escaped_outer",
                    "claim_id": "parc_escaped_outer",
                    "check_action_id": action_id,
                },
                metadata={},
                learner_id=self.learner_id,
                session_id=None,
                correlation_id="escaped-correlation",
                causation_id="escaped-causation",
                occurred_at=START + timedelta(minutes=2),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "artifact-run observation",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=attempt["id"],
                )

    def test_payload_session_start_cannot_hide_behind_null_envelope(self) -> None:
        attempt = self.start_attempt(
            self.session["id"],
            key="scope-hidden-start-target",
            now=START + timedelta(minutes=1),
        )
        hidden_attempt_id = "pta_hidden_payload_session"
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id="system:trace-scope-hidden-start",
                event_type="PerformanceTaskStarted",
                schema_version=1,
                payload={
                    "attempt_id": hidden_attempt_id,
                    "session_id": self.session["id"],
                    "learner_id": self.learner_id,
                    "task_release_id": attempt["task_release_id"],
                    "corpus_release_id": attempt["corpus_release_id"],
                    "task_id": attempt["task_id"],
                    "task_version": attempt["task_version"],
                    "task_digest": attempt["task_digest"],
                    "session_revision": attempt["session_revision"],
                    "learner_revision": attempt["learner_revision"],
                },
                metadata={
                    "command_hash": _D0,
                    "task_schema_version": self.task.schema_version,
                    "shadow_only": True,
                    "projection_applied": False,
                    "certification_applied": False,
                },
                learner_id=self.learner_id,
                session_id=None,
                correlation_id=hidden_attempt_id,
                causation_id=self.session["id"],
                occurred_at=START + timedelta(minutes=2),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "invalid stream or causal envelope",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=attempt["id"],
                )

    def test_unrelated_session_history_has_bounded_vm_step_cost(self) -> None:
        target = self.start_attempt(
            self.session["id"],
            key="scope-scaling-target",
            now=START + timedelta(minutes=1),
        )
        baseline_steps = self.count_validation_vm_steps(target["id"])

        for index in range(24):
            session_time = START + timedelta(days=index + 1)
            session = self.engine.start_session(
                self.learner_id,
                "t_transformers",
                seed=7800 + index,
                now=session_time,
            )
            self.start_attempt(
                session["id"],
                key=f"scope-scaling-peer-{index}",
                now=session_time + timedelta(minutes=1),
            )

        late_session_time = START + timedelta(days=30)
        late_session = self.engine.start_session(
            self.learner_id,
            "t_transformers",
            seed=7900,
            now=late_session_time,
        )
        late_target = self.start_attempt(
            late_session["id"],
            key="scope-scaling-late-target",
            now=late_session_time + timedelta(minutes=1),
        )
        expanded_steps = self.count_validation_vm_steps(target["id"])
        late_target_steps = self.count_validation_vm_steps(late_target["id"])
        self.assertLessEqual(
            expanded_steps,
            baseline_steps + 1_500,
            (
                "Attempt-scoped validation VM work grew with unrelated "
                f"same-learner history: {baseline_steps} -> {expanded_steps}"
            ),
        )
        self.assertLessEqual(
            late_target_steps,
            expanded_steps + 1_500,
            (
                "A target created after unrelated same-learner history did "
                f"unbounded VM work: {expanded_steps} -> {late_target_steps}"
            ),
        )
        plan = self.scoped_select_plan(late_target["id"])
        full_event_scans = [
            detail
            for detail in plan
            if detail.startswith("SCAN event")
        ]
        self.assertEqual(
            full_event_scans,
            [],
            "Attempt-scoped event SELECT performed a scan: " + repr(plan),
        )


class PerformanceArtifactTraceClosureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.fixture = artifact_fixture._build_fixture(
            Path(self.tempdir.name) / "artifact-trace-closure.db",
            "trace-closure",
        )
        self.database = self.fixture["database"]
        self.attempt = self.fixture["attempt"]
        self.artifact = self.fixture["artifact"]
        self.claim = artifact_fixture._insert_claim(self.fixture)
        with self.database.read() as connection:
            require_performance_attempt_trace_consistency(
                connection,
                attempt_id=self.attempt["id"],
            )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def append_escaped_observation(
        self,
        *,
        payload: dict,
        causation_id: str = "escaped-causation",
    ) -> None:
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id=f"learner:{self.attempt['learner_id']}",
                event_type="PerformanceArtifactRunObserved",
                schema_version=1,
                payload=payload,
                metadata={},
                learner_id=self.attempt["learner_id"],
                session_id=None,
                correlation_id="escaped-correlation",
                causation_id=causation_id,
                occurred_at=artifact_fixture.START + timedelta(minutes=4),
            )

    def assert_attempt_rejects_escaped_observation(self) -> None:
        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "artifact-run observation",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=self.attempt["id"],
                )

    def test_claim_event_id_causation_cannot_escape_attempt_scope(
        self,
    ) -> None:
        self.append_escaped_observation(
            payload={
                "attempt_id": "pta_escaped_outer",
                "claim_id": "parc_escaped_outer",
            },
            causation_id=self.claim["event_id"],
        )
        self.assert_attempt_rejects_escaped_observation()

    def test_claim_identity_reference_cannot_escape_attempt_scope(
        self,
    ) -> None:
        self.append_escaped_observation(
            payload={
                "attempt_id": "pta_escaped_outer",
                "claim_id": self.claim["id"],
            },
        )
        self.assert_attempt_rejects_escaped_observation()

    def test_nested_receipt_references_cannot_escape_attempt_scope(
        self,
    ) -> None:
        self.append_escaped_observation(
            payload={
                "attempt_id": "pta_escaped_outer",
                "claim_id": "parc_escaped_outer",
                "receipt": {
                    "attempt_id": self.attempt["id"],
                    "claim_id": self.claim["id"],
                    "artifact_action_id": self.artifact["id"],
                },
            },
        )
        self.assert_attempt_rejects_escaped_observation()


class PerformanceOperationalIdentityClosureTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = artifact_run_fixture.ArtifactRunLedgerTests(
            "test_success_commits_claim_system_action_and_exact_receipts"
        )
        self.harness.setUp()
        (
            self.attempt,
            checkpoint,
            captured,
        ) = self.harness.start_with_artifact(
            artifact_run_fixture.artifact([[True, False], [True, True]]),
            key_suffix="identity-closure",
        )
        self.harness.run_check(
            self.attempt,
            checkpoint,
            captured,
            key="identity-closure-run",
        )
        self.database = self.harness.database
        with self.database.read() as connection:
            self.claim_event = connection.execute(
                """SELECT * FROM events
                   WHERE event_type='PerformanceArtifactRunClaimed'
                     AND correlation_id=?""",
                (self.attempt["id"],),
            ).fetchone()
            self.observation_event = connection.execute(
                """SELECT * FROM events
                   WHERE event_type='PerformanceArtifactRunObserved'
                     AND correlation_id=?""",
                (self.attempt["id"],),
            ).fetchone()
            require_performance_attempt_trace_consistency(
                connection,
                attempt_id=self.attempt["id"],
            )
        self.claim_payload = json.loads(
            self.claim_event["payload_json"]
        )
        self.observation_payload = json.loads(
            self.observation_event["payload_json"]
        )

    def tearDown(self) -> None:
        self.harness.tearDown()

    @staticmethod
    def nested_value(payload: dict, path: tuple[str, ...]):
        value = payload
        for field_name in path:
            value = value[field_name]
        return value

    @staticmethod
    def set_nested_value(
        payload: dict,
        path: tuple[str, ...],
        value,
    ) -> None:
        target = payload
        for field_name in path[:-1]:
            target = target[field_name]
        target[path[-1]] = value

    def append_escaped_event(self, event_type: str, payload: dict) -> None:
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id=f"learner:{self.attempt['learner_id']}",
                event_type=event_type,
                schema_version=1,
                payload=payload,
                metadata={},
                learner_id=self.attempt["learner_id"],
                session_id=None,
                correlation_id="escaped-correlation",
                causation_id="escaped-causation",
                occurred_at=(
                    artifact_run_fixture.BASE + timedelta(minutes=8)
                ),
            )

    def assert_attempt_rejects(self, label: str) -> None:
        with self.database.read() as connection:
            with self.assertRaisesRegex(ValidationError, label):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=self.attempt["id"],
                )

    def append_claim_preserving_only(
        self,
        path: tuple[str, ...],
    ) -> None:
        payload = copy.deepcopy(self.claim_payload)
        preserved = self.nested_value(payload, path)
        payload["claim_id"] = "parc_escaped_identity"
        payload["attempt_id"] = "pta_escaped_identity"
        payload["session_id"] = "ses_escaped_identity"
        payload["artifact_action_id"] = "pact_escaped_identity"
        payload["caller_idempotency_key"] = "escaped-caller-key"
        payload["command_hash"] = "1" * 64
        payload["request_digest"] = "2" * 64
        payload["request"]["run_id"] = "arun_escaped_identity"
        self.set_nested_value(payload, path, preserved)
        self.append_escaped_event(
            "PerformanceArtifactRunClaimed",
            payload,
        )

    def append_observation_preserving_only(
        self,
        path: tuple[str, ...],
    ) -> None:
        payload = copy.deepcopy(self.observation_payload)
        preserved = self.nested_value(payload, path)
        payload["attempt_id"] = "pta_escaped_identity"
        payload["claim_id"] = "parc_escaped_identity"
        payload["check_action_id"] = "pact_escaped_identity"
        payload["receipt_id"] = "parr_escaped_identity"
        payload["receipt_digest"] = "1" * 64
        payload["result_digest"] = "2" * 64
        payload["receipt"]["attempt_id"] = "pta_escaped_identity"
        payload["receipt"]["claim_id"] = "parc_escaped_identity"
        payload["receipt"][
            "artifact_action_id"
        ] = "pact_escaped_identity"
        payload["receipt"]["request_digest"] = "3" * 64
        payload["receipt"]["result_digest"] = "4" * 64
        payload["result"]["request"]["run_id"] = "arun_escaped_identity"
        payload["result"]["request_digest"] = "5" * 64
        self.set_nested_value(payload, path, preserved)
        self.append_escaped_event(
            "PerformanceArtifactRunObserved",
            payload,
        )

    def test_claim_request_run_identity_cannot_escape_scope(self) -> None:
        self.append_claim_preserving_only(("request", "run_id"))
        self.assert_attempt_rejects("artifact-run claim")

    def test_claim_request_digest_cannot_escape_scope(self) -> None:
        self.append_claim_preserving_only(("request_digest",))
        self.assert_attempt_rejects("artifact-run claim")

    def test_claim_command_hash_cannot_escape_scope(self) -> None:
        self.append_claim_preserving_only(("command_hash",))
        self.assert_attempt_rejects("artifact-run claim")

    def test_claim_caller_key_cannot_escape_scope(self) -> None:
        self.append_claim_preserving_only(("caller_idempotency_key",))
        self.assert_attempt_rejects("artifact-run claim")

    def test_receipt_identity_cannot_escape_scope(self) -> None:
        self.append_observation_preserving_only(("receipt_id",))
        self.assert_attempt_rejects("artifact-run observation")

    def test_receipt_digest_cannot_escape_scope(self) -> None:
        self.append_observation_preserving_only(("receipt_digest",))
        self.assert_attempt_rejects("artifact-run observation")

    def test_observed_request_run_identity_cannot_escape_scope(self) -> None:
        self.append_observation_preserving_only(
            ("result", "request", "run_id")
        )
        self.assert_attempt_rejects("artifact-run observation")

    def test_observed_request_digest_cannot_escape_scope(self) -> None:
        self.append_observation_preserving_only(
            ("result", "request_digest")
        )
        self.assert_attempt_rejects("artifact-run observation")

    def test_observed_process_digest_cannot_escape_scope(self) -> None:
        self.append_observation_preserving_only(("result_digest",))
        self.assert_attempt_rejects("artifact-run observation")

    def test_operational_result_digest_cannot_escape_scope(self) -> None:
        self.append_observation_preserving_only(
            ("receipt", "result_digest")
        )
        self.assert_attempt_rejects("artifact-run observation")

    def test_failed_receipt_request_digest_cannot_escape_scope(self) -> None:
        payload = copy.deepcopy(self.observation_payload)
        request_digest = payload["receipt"]["request_digest"]
        payload["attempt_id"] = "pta_escaped_identity"
        payload["claim_id"] = "parc_escaped_identity"
        payload["check_action_id"] = None
        payload["outcome"] = "runner_failed"
        payload["result"] = None
        payload["result_digest"] = None
        payload["receipt_id"] = "parr_escaped_identity"
        payload["receipt_digest"] = "1" * 64
        payload["receipt"]["attempt_id"] = "pta_escaped_identity"
        payload["receipt"]["claim_id"] = "parc_escaped_identity"
        payload["receipt"][
            "artifact_action_id"
        ] = "pact_escaped_identity"
        payload["receipt"]["request_digest"] = request_digest
        payload["receipt"]["result_digest"] = None
        self.append_escaped_event(
            "PerformanceArtifactRunObserved",
            payload,
        )
        self.assert_attempt_rejects("artifact-run observation")

    def test_attempt_guard_compares_claim_projection(self) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_artifact_run_claims_no_update"
            )
            connection.execute(
                """UPDATE performance_artifact_run_claims
                   SET runner_version='tampered'
                   WHERE attempt_id=?""",
                (self.attempt["id"],),
            )
        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "artifact run claims differ",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=self.attempt["id"],
                )
            with self.assertRaisesRegex(
                ValidationError,
                "artifact run claims differ",
            ):
                require_performance_projection_consistency(
                    connection,
                    learner_id=self.attempt["learner_id"],
                    trace_only=True,
                )

    def test_attempt_guard_compares_receipt_projection(self) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_artifact_run_receipts_no_update"
            )
            connection.execute(
                """UPDATE performance_artifact_run_receipts
                   SET outcome='invalid_artifact'
                   WHERE claim_id=?""",
                (self.claim_payload["claim_id"],),
            )
        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "artifact run receipts differ",
            ):
                require_performance_attempt_trace_consistency(
                    connection,
                    attempt_id=self.attempt["id"],
                )
            with self.assertRaisesRegex(
                ValidationError,
                "artifact run receipts differ",
            ):
                require_performance_projection_consistency(
                    connection,
                    learner_id=self.attempt["learner_id"],
                    trace_only=True,
                )

    def test_operational_identity_lookup_plan_is_indexed(self) -> None:
        statements: list[str] = []
        with self.database.read() as connection:
            connection.set_trace_callback(statements.append)
            try:
                derive_performance_projections(
                    connection,
                    attempt_id=self.attempt["id"],
                    trace_only=True,
                )
            finally:
                connection.set_trace_callback(None)
            scoped = tuple(
                dict.fromkeys(
                    statement
                    for statement in statements
                    if statement.lstrip().startswith(
                        (
                            "WITH scoped_session_ids(session_id)",
                            "WITH scoped_values(value)",
                        )
                    )
                    and (
                        " events AS event " in statement
                        or " JOIN events AS event " in statement
                    )
                )
            )
            plans = [
                row["detail"]
                for statement in scoped
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN " + statement
                ).fetchall()
            ]
        self.assertFalse(
            [detail for detail in plans if detail.startswith("SCAN event")],
            plans,
        )
        details = " ".join(plans)
        for index_name in (
            "idx_events_claim_caller_key_stream",
            "idx_events_claim_command_hash_stream",
            "idx_events_claim_request_digest_stream",
            "idx_events_claim_request_run_stream",
            "idx_events_observed_request_digest_stream",
            "idx_events_observed_request_run_stream",
            "idx_events_observed_result_digest_stream",
            "idx_events_receipt_digest_stream",
            "idx_events_receipt_id_stream",
            "idx_events_receipt_request_digest_stream",
            "idx_events_receipt_result_digest_stream",
        ):
            self.assertIn(index_name, details)


if __name__ == "__main__":
    unittest.main()
