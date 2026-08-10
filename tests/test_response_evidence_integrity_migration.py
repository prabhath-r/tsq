# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, ValidationError
from tsq.evidence import (
    ActionPhase,
    CriterionScale,
    EvaluationStatus,
    LearningTask,
    RubricCriterion,
    TaskModality,
    canonical_digest,
)
from tsq.performance import (
    ImportedCriterionResult,
    ImportedEvaluation,
    ScoringProviderRegistry,
    SyntheticDeterministicProvider,
)
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
)
from tsq.performance_reporting import productive_shadow_summary
from tsq.performance_selection import recommend_performance_tasks
from tsq.reconciliation import (
    ReconciliationObservation,
    ReconciliationOutcome,
    ScoringReconciliationReceipt,
    ScoringReconciliationRegistry,
    SyntheticReconciliationAdapter,
)
from tsq.replay import ProjectionReplay
from tsq.store import (
    LEGACY_INVALID_RESPONSE_EVIDENCE_EVENT_TYPE,
    LEGACY_INVALID_RESPONSE_EVIDENCE_POLICY,
    LEGACY_INVALID_RESPONSE_EVIDENCE_REASON,
    SCHEMA_VERSION,
    Database,
    _V0_1_0_QUESTIONS_VALID_STATUS_TRIGGER_SQL,
    _capture_current_schema_contract,
    _expected_v16_schema_contract,
    _expected_v22_schema_contract,
    _legacy_invalid_response_evidence_key,
)

from tests.schema_upgrade_helpers import durable_database_fingerprint
from tests.test_migration_event_lifecycle import (
    build_exact_v16,
    downgrade_to_exact_v16,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
START = datetime(2030, 1, 2, 12, 0, tzinfo=timezone.utc)
_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64


class ResponseEvidenceIntegrityMigrationTests(unittest.TestCase):
    """Preserve the fail-closed meaning of immutable schema-v22 markers."""

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "legacy-evidence.db"
        self.database = Database(self.path)
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)
        self.learner_id = "legacy-invalidated-evidence"
        self.engine.create_learner(self.learner_id)
        self.session = self.engine.start_session(
            self.learner_id,
            "c_attention",
            seed=4101,
            now=START,
        )
        self.presentation = self.engine.next_question(
            self.session["id"],
            now=START + timedelta(minutes=1),
        )
        self.answer_key = "legacy-response-command"
        self.submission = self.engine.submit_answer(
            self.presentation.decision_id,
            self.presentation.question.correct_option.id,
            confidence=0.8,
            response_ms=5000,
            idempotency_key=self.answer_key,
            now=START + timedelta(minutes=2),
        )
        with self.database.read() as connection:
            self.response_attempt = connection.execute(
                "SELECT * FROM attempts WHERE decision_id=?",
                (self.presentation.decision_id,),
            ).fetchone()
            self.active_release_id = connection.execute(
                """SELECT value FROM meta
                   WHERE key='active_corpus_release'"""
            ).fetchone()["value"]
        assert self.response_attempt is not None

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _install_legacy_invalidation(
        self,
        *,
        valid: bool = True,
        database: Database | None = None,
    ) -> dict[str, object]:
        target = database or self.database
        attempt = self.response_attempt
        revoked_at = START + timedelta(minutes=3)
        with target.transaction() as connection:
            revocation_event = target.append_event(
                connection,
                stream_id="corpus:safety",
                event_type="QuestionEmergencyRevoked",
                payload={
                    "question_id": attempt["question_id"],
                    "reason": "Historical independent safety finding.",
                },
                metadata={
                    "schema_version": 22,
                    "active_corpus_release_id": self.active_release_id,
                },
                idempotency_key="legacy-evidence-revocation",
                occurred_at=revoked_at,
            )
            connection.execute(
                """INSERT INTO question_revocations(
                       question_id, reason, revoked_at, event_id
                   ) VALUES (?, ?, ?, ?)""",
                (
                    attempt["question_id"],
                    "Historical independent safety finding.",
                    revoked_at.isoformat(),
                    revocation_event["event_id"],
                ),
            )
            payload = {
                "attempt_id": attempt["id"],
                "response_event_id": attempt["event_id"],
                "learner_id": attempt["learner_id"],
                "question_id": attempt["question_id"],
                "reason": LEGACY_INVALID_RESPONSE_EVIDENCE_REASON,
                "projection_applied": False,
            }
            metadata = {
                "safety_policy": LEGACY_INVALID_RESPONSE_EVIDENCE_POLICY,
                "requires_explicit_rebuild": valid,
            }
            marker = target.append_event(
                connection,
                stream_id=f"learner:{attempt['learner_id']}",
                event_type=LEGACY_INVALID_RESPONSE_EVIDENCE_EVENT_TYPE,
                schema_version=1,
                payload=payload,
                metadata=metadata,
                learner_id=attempt["learner_id"],
                session_id=None,
                idempotency_key=_legacy_invalid_response_evidence_key(
                    attempt["id"]
                ),
                correlation_id=attempt["id"],
                causation_id=attempt["event_id"],
                occurred_at=revoked_at,
            )
        return dict(marker)

    def _downgrade_to_exact_v22(self) -> None:
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER questions_valid_status")
            connection.execute(
                _V0_1_0_QUESTIONS_VALID_STATUS_TRIGGER_SQL
            )
            connection.execute(
                """UPDATE meta SET value='22'
                   WHERE key='schema_version'"""
            )
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            self.assertEqual(
                _capture_current_schema_contract(connection),
                _expected_v22_schema_contract(),
            )
        finally:
            connection.close()

    def _evidence_snapshot(self, database: Database) -> dict[str, object]:
        with database.read() as connection:
            return {
                "attempt": dict(
                    connection.execute(
                        "SELECT * FROM attempts WHERE id=?",
                        (self.response_attempt["id"],),
                    ).fetchone()
                ),
                "learner": dict(
                    connection.execute(
                        "SELECT * FROM learners WHERE id=?",
                        (self.learner_id,),
                    ).fetchone()
                ),
                "events": tuple(
                    tuple(row)
                    for row in connection.execute(
                        """SELECT * FROM events
                           WHERE learner_id=?
                              OR event_type='QuestionEmergencyRevoked'
                           ORDER BY recorded_at, event_id""",
                        (self.learner_id,),
                    ).fetchall()
                ),
                "revocation": tuple(
                    connection.execute(
                        """SELECT * FROM question_revocations
                           WHERE question_id=?""",
                        (self.response_attempt["question_id"],),
                    ).fetchone()
                ),
                "projection_hash": database.learner_projection_hash(
                    self.learner_id,
                    connection,
                ),
            }

    def _upgrade_contaminated_v22(self) -> Database:
        self._install_legacy_invalidation()
        before = self._evidence_snapshot(self.database)
        self._downgrade_to_exact_v22()
        upgraded = Database(self.path)
        upgraded.initialize()
        self.assertEqual(self._evidence_snapshot(upgraded), before)
        return upgraded

    def test_v22_marker_survives_v23_migration_and_reopen_idempotently(
        self,
    ) -> None:
        upgraded = self._upgrade_contaminated_v22()
        before_reopen = self._evidence_snapshot(upgraded)

        with upgraded.read() as connection:
            version = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()["value"]
            marker_count = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type=? AND correlation_id=?""",
                (
                    LEGACY_INVALID_RESPONSE_EVIDENCE_EVENT_TYPE,
                    self.response_attempt["id"],
                ),
            ).fetchone()["n"]
        self.assertEqual(SCHEMA_VERSION, 23)
        self.assertEqual(version, "23")
        self.assertEqual(marker_count, 1)

        upgraded.initialize()
        self.assertEqual(self._evidence_snapshot(upgraded), before_reopen)

    def test_adaptation_reporting_integrity_and_replay_fail_closed(
        self,
    ) -> None:
        upgraded = self._upgrade_contaminated_v22()
        engine = AdaptiveEngine(upgraded)
        expected = "evidence was invalidated after projection"

        clean_learner = "clean-after-v22-upgrade"
        engine.create_learner(clean_learner)
        clean_session = engine.start_session(
            clean_learner,
            "c_attention",
            seed=4200,
            now=START + timedelta(minutes=4),
        )
        self.assertEqual(clean_session["learner_id"], clean_learner)
        self.assertEqual(
            engine.profile(
                clean_learner,
                root_concept_id="c_attention",
                now=START + timedelta(minutes=4),
            )["learner_id"],
            clean_learner,
        )

        operations = (
            lambda: engine.start_session(
                self.learner_id,
                "c_attention",
                seed=4201,
                now=START + timedelta(minutes=4),
            ),
            lambda: engine.next_question(
                self.session["id"],
                now=START + timedelta(minutes=4),
            ),
            lambda: engine.submit_answer(
                self.presentation.decision_id,
                self.presentation.question.correct_option.id,
                confidence=0.8,
                response_ms=5000,
                idempotency_key=self.answer_key,
                now=START + timedelta(minutes=4),
            ),
            lambda: engine.profile(
                self.learner_id,
                root_concept_id="c_attention",
                now=START + timedelta(minutes=4),
            ),
            lambda: engine.session_report(
                self.session["id"],
                now=START + timedelta(minutes=4),
            ),
            lambda: productive_shadow_summary(
                upgraded,
                self.learner_id,
            ),
            lambda: recommend_performance_tasks(
                upgraded,
                self.session["id"],
                now=START + timedelta(minutes=4),
            ),
        )
        for operation in operations:
            with self.assertRaisesRegex(ConflictError, expected):
                operation()

        integrity = upgraded.verify_integrity()
        self.assertFalse(integrity["ok"])
        integrity_message = "projection contains invalidated response evidence"
        self.assertTrue(
            any(integrity_message in error for error in integrity["errors"]),
            integrity["errors"],
        )
        replay = ProjectionReplay(upgraded).check(self.learner_id)
        self.assertFalse(replay["ok"])
        self.assertFalse(replay["rebuild_safe"])
        self.assertTrue(
            any(integrity_message in error for error in replay["errors"]),
            replay["errors"],
        )

    def test_profile_rechecks_integrity_after_projection_reads(self) -> None:
        real_gate = self.database.require_learner_evidence_integrity
        original_get_graph = self.database.get_graph
        state = {"gate_calls": 0, "marker_installed": False}

        def staged_gate(learner_id, connection):
            state["gate_calls"] += 1
            if state["gate_calls"] == 1:
                return None
            return real_gate(learner_id, connection)

        def invalidate_then_get_graph(*args, **kwargs):
            if not state["marker_installed"]:
                self._install_legacy_invalidation(
                    database=Database(self.path)
                )
                state["marker_installed"] = True
            return original_get_graph(*args, **kwargs)

        with (
            patch.object(
                self.database,
                "require_learner_evidence_integrity",
                side_effect=staged_gate,
            ),
            patch.object(
                self.database,
                "get_graph",
                side_effect=invalidate_then_get_graph,
            ),
            self.assertRaisesRegex(
                ConflictError,
                "evidence was invalidated after projection",
            ),
        ):
            self.engine.profile(
                self.learner_id,
                root_concept_id="c_attention",
                now=START + timedelta(minutes=4),
            )

        self.assertTrue(state["marker_installed"])
        self.assertGreaterEqual(state["gate_calls"], 2)

    def test_task_recommendation_rechecks_integrity_before_return(
        self,
    ) -> None:
        self._submitted_performance_attempt()
        real_gate = self.database.require_learner_evidence_integrity
        original_get_graph = self.database.get_graph
        state = {"gate_calls": 0, "marker_installed": False}

        def staged_gate(learner_id, connection):
            state["gate_calls"] += 1
            if state["gate_calls"] == 1:
                return None
            return real_gate(learner_id, connection)

        def invalidate_then_get_graph(*args, **kwargs):
            if not state["marker_installed"]:
                self._install_legacy_invalidation(
                    database=Database(self.path)
                )
                state["marker_installed"] = True
            return original_get_graph(*args, **kwargs)

        with (
            patch.object(
                self.database,
                "require_learner_evidence_integrity",
                side_effect=staged_gate,
            ),
            patch.object(
                self.database,
                "get_graph",
                side_effect=invalidate_then_get_graph,
            ),
            self.assertRaisesRegex(
                ConflictError,
                "evidence was invalidated after projection",
            ),
        ):
            recommend_performance_tasks(
                self.database,
                self.session["id"],
                now=START + timedelta(minutes=8),
            )

        self.assertTrue(state["marker_installed"])
        self.assertGreaterEqual(state["gate_calls"], 2)

    def test_selection_rechecks_integrity_after_scoring(self) -> None:
        selection_session = self.engine.start_session(
            self.learner_id,
            "c_attention",
            seed=4301,
            now=START + timedelta(minutes=4),
        )
        original_score = self.engine.policy._score
        real_gate = self.database.require_learner_evidence_integrity
        state = {
            "gate_calls": 0,
            "marker_installed": False,
            "scored_question_ids": [],
        }

        def staged_gate(learner_id, connection):
            state["gate_calls"] += 1
            if state["gate_calls"] == 1:
                return None
            return real_gate(learner_id, connection)

        def score_with_invalidation(question, **kwargs):
            state["scored_question_ids"].append(question.id)
            if not state["marker_installed"]:
                self._install_legacy_invalidation(
                    database=Database(self.path)
                )
                state["marker_installed"] = True
            return original_score(question, **kwargs)

        with (
            patch.object(
                self.database,
                "require_learner_evidence_integrity",
                side_effect=staged_gate,
            ),
            patch.object(
                self.engine.policy,
                "_score",
                side_effect=score_with_invalidation,
            ),
            self.assertRaisesRegex(
                ConflictError,
                "evidence was invalidated after projection",
            ),
        ):
            self.engine.next_question(
                selection_session["id"],
                now=START + timedelta(minutes=5),
            )

        self.assertTrue(state["marker_installed"])
        self.assertGreaterEqual(state["gate_calls"], 2)
        self.assertTrue(state["scored_question_ids"])
        with self.database.read() as connection:
            decision_count = connection.execute(
                "SELECT COUNT(*) AS n FROM decisions WHERE session_id=?",
                (selection_session["id"],),
            ).fetchone()["n"]
        self.assertEqual(decision_count, 0)

    def test_malformed_v22_marker_aborts_migration_without_writes(self) -> None:
        self._install_legacy_invalidation(valid=False)
        self._downgrade_to_exact_v22()
        before = durable_database_fingerprint(self.path)

        with self.assertRaisesRegex(
            ConflictError,
            "invalid evidence boundary",
        ):
            Database(self.path).initialize()

        self.assertEqual(durable_database_fingerprint(self.path), before)
        connection = sqlite3.connect(self.path)
        try:
            version = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(version, "22")

    def test_pre_marker_response_history_requires_v22_intermediate(self) -> None:
        downgrade_to_exact_v16(self.database)
        with self.database.read() as connection:
            self.assertEqual(
                _capture_current_schema_contract(connection),
                _expected_v16_schema_contract(),
            )
        before = durable_database_fingerprint(self.path)

        with self.assertRaisesRegex(
            ConflictError,
            (
                "Schema v16 contains learner response history from before "
                "the durable evidence-integrity boundary.*schema v22.*"
                "TSQ v0.1.0"
            ),
        ):
            Database(self.path).initialize()

        self.assertEqual(durable_database_fingerprint(self.path), before)
        connection = sqlite3.connect(self.path)
        try:
            version = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertEqual(version, "16")

    def test_pre_marker_database_without_responses_upgrades_directly(
        self,
    ) -> None:
        empty_path = Path(self.tempdir.name) / "empty-v16.db"
        empty_database = build_exact_v16(empty_path)
        with empty_database.read() as connection:
            self.assertIsNone(
                connection.execute("SELECT 1 FROM attempts LIMIT 1").fetchone()
            )

        empty_database.initialize()

        with empty_database.read() as connection:
            version = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()["value"]
        self.assertEqual(version, str(SCHEMA_VERSION))

    def _submitted_performance_attempt(
        self,
    ) -> tuple[
        PerformanceLedger,
        LearningTask,
        dict[str, object],
        dict[str, object],
    ]:
        with self.database.read() as connection:
            source = connection.execute(
                """SELECT source.id, source.content_hash
                   FROM release_sources membership
                   JOIN sources source ON source.id=membership.source_id
                   WHERE membership.release_id=?
                   ORDER BY source.id LIMIT 1""",
                (self.active_release_id,),
            ).fetchone()
        task = LearningTask(
            id="task_legacy_response_integrity",
            version=1,
            family_id="family_legacy_response_integrity",
            title="Inspect one response-integrity invariant",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_legacy_response_integrity",
                    name="Response-integrity invariant",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_attention", 1.0),),
                    dependence_group="legacy_response_integrity",
                    evidence_cap=0.8,
                    dependence_cap=0.8,
                ),
            ),
            instructions=(
                "Inspect the pinned trace and submit a content-addressed "
                "diagnostic artifact."
            ),
            source_manifests=((source["id"], source["content_hash"]),),
            administration_id="legacy_response_integrity_admin",
            administration_manifest_digest=_D0,
            stimulus_id="legacy_response_integrity_stimulus",
            stimulus_digest=_D1,
        )
        ledger = PerformanceLedger(self.database)
        release = ledger.publish_release(
            PerformanceTaskRelease(
                title="Legacy response-integrity fixture",
                corpus_release_id=self.active_release_id,
                review=TaskReleaseReview(
                    reviewer_kind="human",
                    reviewer_id="independent-integrity-reviewer",
                    reviewed_at=(START + timedelta(minutes=4)).isoformat(),
                    independent_of_author=True,
                    attestation_digest=_D2,
                ),
                tasks=(("pilot", task),),
            ),
            now=START + timedelta(minutes=4),
        )
        attempt = ledger.start_attempt(
            self.session["id"],
            task.id,
            task_version=task.version,
            task_release_id=release["release_id"],
            idempotency_key="integrity-task-start",
            now=START + timedelta(minutes=5),
        )
        ledger.record_action(
            attempt["id"],
            "artifact_checkpoint",
            {
                "artifact_digest": _D3,
                "artifact_kind": "diagnostic_digest",
            },
            idempotency_key="integrity-artifact",
            now=START + timedelta(minutes=6),
        )
        submitted = ledger.record_action(
            attempt["id"],
            "submitted",
            {"submission_digest": _D3},
            idempotency_key="integrity-submit",
            now=START + timedelta(minutes=7),
        )
        return ledger, task, attempt, submitted

    def test_marker_arriving_during_provider_call_blocks_result_commit(
        self,
    ) -> None:
        ledger, _task, attempt, submitted = (
            self._submitted_performance_attempt()
        )
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_legacy_response_integrity",
                    status=EvaluationStatus.VALID,
                    score=0.8,
                    outcome_code="provider_race",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                    reliability=0.9,
                ),
            )
        )
        test_case = self
        state = {"provider_called": False, "marker_installed": False}

        class InvalidatingProvider(SyntheticDeterministicProvider):
            def score(self, request):
                state["provider_called"] = True
                test_case._install_legacy_invalidation(
                    database=Database(test_case.path)
                )
                state["marker_installed"] = True
                return super().score(request)

        provider = InvalidatingProvider(
            imported,
            provider_id="synthetic.legacy-response-integrity",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)

        with self.assertRaisesRegex(
            ConflictError,
            "evidence was invalidated after projection",
        ):
            ledger.score_attempt(
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="integrity-provider-race",
                now=START + timedelta(minutes=8),
            )

        self.assertTrue(state["provider_called"])
        self.assertTrue(state["marker_installed"])
        with self.database.read() as connection:
            evaluation_count = connection.execute(
                """SELECT COUNT(*) AS n FROM task_evaluations
                   WHERE attempt_id=?""",
                (attempt["id"],),
            ).fetchone()["n"]
            claim_count = connection.execute(
                """SELECT COUNT(*) AS n FROM performance_scoring_claims
                   WHERE attempt_id=?""",
                (attempt["id"],),
            ).fetchone()["n"]
        self.assertEqual(evaluation_count, 0)
        self.assertEqual(claim_count, 1)

    def test_completed_score_idempotent_replay_is_withheld_after_marker(
        self,
    ) -> None:
        ledger, _task, attempt, submitted = (
            self._submitted_performance_attempt()
        )
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_legacy_response_integrity",
                    status=EvaluationStatus.VALID,
                    score=0.8,
                    outcome_code="completed_before_invalidation",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                    reliability=0.9,
                ),
            )
        )
        provider = SyntheticDeterministicProvider(
            imported,
            provider_id="synthetic.completed-before-invalidation",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        scored = ledger.score_attempt(
            attempt["id"],
            registry,
            provider.provider_id,
            provider.provider_version,
            idempotency_key="completed-score-before-invalidation",
            now=START + timedelta(minutes=8),
        )
        self.assertFalse(scored["idempotent_replay"])
        self._install_legacy_invalidation()

        with self.assertRaisesRegex(
            ConflictError,
            "evidence was invalidated after projection",
        ):
            ledger.score_attempt(
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="completed-score-before-invalidation",
                now=START + timedelta(minutes=8),
            )

    def test_completed_import_idempotent_replay_is_withheld_after_marker(
        self,
    ) -> None:
        ledger, _task, attempt, submitted = (
            self._submitted_performance_attempt()
        )
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_legacy_response_integrity",
                    status=EvaluationStatus.VALID,
                    score=0.8,
                    outcome_code="imported_before_invalidation",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                    reliability=0.9,
                ),
            )
        )
        imported_result = ledger.import_evaluation(
            attempt["id"],
            imported,
            provider_id="import.completed-before-invalidation",
            provider_version="v1",
            idempotency_key="completed-import-before-invalidation",
            now=START + timedelta(minutes=8),
        )
        self.assertFalse(imported_result["idempotent_replay"])
        self._install_legacy_invalidation()

        with self.assertRaisesRegex(
            ConflictError,
            "evidence was invalidated after projection",
        ):
            ledger.import_evaluation(
                attempt["id"],
                imported,
                provider_id="import.completed-before-invalidation",
                provider_version="v1",
                idempotency_key="completed-import-before-invalidation",
                now=START + timedelta(minutes=8),
            )

    def test_marker_arriving_during_reconciliation_blocks_receipt_commit(
        self,
    ) -> None:
        ledger, _task, attempt, submitted = (
            self._submitted_performance_attempt()
        )
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_legacy_response_integrity",
                    status=EvaluationStatus.VALID,
                    score=0.8,
                    outcome_code="unresolved_provider_result",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                    reliability=0.9,
                ),
            )
        )

        class FailingProvider(SyntheticDeterministicProvider):
            def score(self, request):
                raise RuntimeError("admitted callback outcome is unknown")

        provider = FailingProvider(
            imported,
            provider_id="synthetic.unresolved-before-invalidation",
        )
        scoring_registry = ScoringProviderRegistry(allow_synthetic=True)
        scoring_registry.register(provider, provider.authority_binding)
        with self.assertRaisesRegex(ValidationError, "failed safely"):
            ledger.score_attempt(
                attempt["id"],
                scoring_registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="unresolved-score-before-invalidation",
                now=START + timedelta(minutes=8),
            )
        claim = ledger.list_scoring_claims(attempt_id=attempt["id"])[0]
        observed_at = START + timedelta(minutes=9)
        reconciler_id = "synthetic.invalidating-reconciler"
        reconciler_version = "test-v1"
        receipt = ScoringReconciliationReceipt(
            claim_id=claim["id"],
            attempt_id=claim["attempt_id"],
            evaluation_id=claim["evaluation_id"],
            through_sequence=claim["through_sequence"],
            provider_id=claim["provider_id"],
            provider_version=claim["provider_version"],
            reconciler_id=reconciler_id,
            reconciler_version=reconciler_version,
            action_trace_digest=claim["action_trace_digest"],
            command_hash=claim["command_hash"],
            scoring_request_digest=claim["scoring_request_digest"],
            provider_binding_digest=claim["provider_binding_digest"],
            outcome=ReconciliationOutcome.UNKNOWN,
            observed_at=observed_at.isoformat(),
            completed_at=None,
            result_digest=None,
            reason_code="provider_status_unknown",
            provider_operation_digest=claim[
                "provider_operation_digest"
            ],
            provider_receipt_digest=canonical_digest(
                {
                    "claim_id": claim["id"],
                    "outcome": ReconciliationOutcome.UNKNOWN.value,
                    "observed_at": observed_at.isoformat(),
                    "reconciler_version": reconciler_version,
                }
            ),
            attestation_digest=canonical_digest(
                {
                    "provider_operation_digest": claim[
                        "provider_operation_digest"
                    ],
                    "outcome": ReconciliationOutcome.UNKNOWN.value,
                    "reconciler_version": reconciler_version,
                }
            ),
        )
        test_case = self

        class InvalidatingReconciler(SyntheticReconciliationAdapter):
            def lookup(self, request):
                test_case._install_legacy_invalidation(
                    database=Database(test_case.path)
                )
                return super().lookup(request)

        reconciler = InvalidatingReconciler(
            ReconciliationObservation(receipt=receipt),
            reconciler_id=reconciler_id,
            reconciler_version=reconciler_version,
        )
        reconciliation_registry = ScoringReconciliationRegistry(
            allow_synthetic=True
        )
        reconciliation_registry.register(
            reconciler,
            reconciler.authority_binding,
        )

        with self.assertRaisesRegex(
            ConflictError,
            "evidence was invalidated after projection",
        ):
            ledger.reconcile_scoring_claim(
                claim["id"],
                reconciliation_registry,
                reconciler.reconciler_id,
                reconciler.reconciler_version,
                idempotency_key="reconciliation-invalidation-race",
                now=START + timedelta(minutes=10),
            )

        with self.database.read() as connection:
            reconciliation_count = connection.execute(
                """SELECT COUNT(*) AS n
                   FROM performance_scoring_reconciliations
                   WHERE claim_id=?""",
                (claim["id"],),
            ).fetchone()["n"]
            evaluation_count = connection.execute(
                """SELECT COUNT(*) AS n FROM task_evaluations
                   WHERE attempt_id=?""",
                (attempt["id"],),
            ).fetchone()["n"]
        self.assertEqual(reconciliation_count, 0)
        self.assertEqual(evaluation_count, 0)


if __name__ == "__main__":
    unittest.main()
