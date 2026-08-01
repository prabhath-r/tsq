# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import threading
import unittest
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.corpus import read_and_parse
from tsq.cli import main
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, NotFoundError, ValidationError
from tsq.evidence import (
    ActionKind,
    ActionPhase,
    CriterionScale,
    EvaluationStatus,
    LearningTask,
    RubricCriterion,
    ScorerContract,
    ScorerKind,
    TaskModality,
    canonical_digest,
    canonical_json,
)
from tsq.performance import (
    ImportedCriterionResult,
    ImportedEvaluation,
    ProviderAuthorityBinding,
    ScoringProviderRegistry,
    SyntheticDeterministicProvider,
)
from tsq.models import QuestionStatus
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
    derive_performance_projections,
    performance_integrity_errors,
    read_task_release,
    require_performance_projection_consistency,
)
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
    PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
    Database,
    performance_scoring_claim_event_key,
    performance_scoring_reconciliation_event_key,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
TASK_RELEASE_FIXTURE = (
    ROOT / "tests" / "fixtures" / "reviewed_productive_task_release.json"
)
TASK_RELEASE_CORPUS_PLACEHOLDER = "rel_fixture_requires_explicit_pinning"
START = datetime(2110, 6, 7, 9, 0, tzinfo=timezone.utc)
_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64


def declared_task_release_fixture(
    corpus_release_id: str,
) -> PerformanceTaskRelease:
    """Bind the portable fixture definition to one imported corpus release."""

    template = read_task_release(TASK_RELEASE_FIXTURE)
    if template.corpus_release_id != TASK_RELEASE_CORPUS_PLACEHOLDER:
        raise AssertionError(
            "Productive-task fixture must use its explicit corpus placeholder."
        )
    return replace(template, corpus_release_id=corpus_release_id)


class PerformanceLedgerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "performance.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)
        self.engine.create_learner("performance-learner", "Performance Learner")
        self.session = self.engine.start_session(
            "performance-learner", "t_transformers", seed=1402
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
            id="task_attention_mask_debug",
            version=1,
            family_id="family_attention_mask_debug",
            title="Diagnose a causal attention mask",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_mask_invariant",
                    name="Causal mask invariant",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_causal_masking", 1.0),),
                    dependence_group="mask_behavior",
                    evidence_cap=0.8,
                    dependence_cap=0.8,
                ),
            ),
            instructions=(
                "Inspect the pinned causal-mask stimulus, diagnose the invariant "
                "violation, and submit a content-addressed repair artifact."
            ),
            source_manifests=((source["id"], source["content_hash"]),),
            administration_id="admin_digest_only_no_execution",
            administration_manifest_digest=_D0,
            stimulus_id="stimulus_attention_mask_v1",
            stimulus_digest=_D1,
        )
        self.release_time = datetime.now(timezone.utc)
        self.release = PerformanceTaskRelease(
            title="Reviewed transformer performance pilot",
            corpus_release_id=self.corpus_release_id,
            review=TaskReleaseReview(
                reviewer_kind="human",
                reviewer_id="reviewer_fixture_independent",
                reviewed_at=self.release_time.isoformat(),
                independent_of_author=True,
                attestation_digest=_D2,
            ),
            tasks=(("pilot", self.task),),
        )
        self.ledger = PerformanceLedger(self.database)
        self.release_report = self.ledger.publish_release(
            self.release, now=self.release_time
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def projection_snapshot(self) -> tuple[int, int, str]:
        with self.database.read() as connection:
            learner_revision = connection.execute(
                "SELECT revision FROM learners WHERE id='performance-learner'"
            ).fetchone()["revision"]
            session_revision = connection.execute(
                "SELECT revision FROM sessions WHERE id=?",
                (self.session["id"],),
            ).fetchone()["revision"]
            projection_hash = self.database.learner_projection_hash(
                "performance-learner", connection
            )
        return learner_revision, session_revision, projection_hash

    def performance_event_and_projection_counts(self) -> dict[str, int]:
        tables = (
            "events",
            "performance_attempts",
            "performance_actions",
            "performance_artifact_run_claims",
            "performance_artifact_run_receipts",
            "performance_scoring_claims",
            "performance_scoring_reconciliations",
            "task_evaluations",
            "shadow_evidence_bundles",
        )
        with self.database.read() as connection:
            return {
                table: connection.execute(
                    f"SELECT COUNT(*) AS n FROM {table}"
                ).fetchone()["n"]
                for table in tables
            }

    def start(self, *, key: str = "start-performance") -> dict:
        return self.ledger.start_attempt(
            self.session["id"],
            self.task.id,
            task_version=self.task.version,
            task_release_id=self.release_report["release_id"],
            idempotency_key=key,
            now=START + timedelta(minutes=1),
        )

    def submit(self, attempt_id: str) -> dict:
        self.ledger.record_action(
            attempt_id,
            "artifact_checkpoint",
            {"artifact_digest": _D3, "artifact_kind": "patch_digest"},
            idempotency_key="artifact-performance",
            now=START + timedelta(minutes=3),
        )
        return self.ledger.record_action(
            attempt_id,
            "submitted",
            {"submission_digest": _D3},
            idempotency_key="submit-performance",
            now=START + timedelta(minutes=5),
        )

    def failed_scoring_claim(
        self,
        *,
        provider_id: str = "synthetic.reconciliation-source",
        key: str = "score-for-reconciliation",
    ) -> tuple[
        dict,
        dict,
        dict,
        SyntheticDeterministicProvider,
        ImportedEvaluation,
    ]:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.8,
                    outcome_code="recovered_observation",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )

        class FailingProvider(SyntheticDeterministicProvider):
            calls = 0

            def score(self, request):
                self.calls += 1
                raise RuntimeError("admitted callback outcome is unknown")

        provider = FailingProvider(imported, provider_id=provider_id)
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        with self.assertRaisesRegex(ValidationError, "failed safely"):
            self.ledger.score_attempt(
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key=key,
                now=START + timedelta(minutes=6),
            )
        claim = self.ledger.list_scoring_claims(
            attempt_id=attempt["id"]
        )[0]
        return attempt, submitted, claim, provider, imported

    def reconciliation_registry(
        self,
        claim: dict,
        *,
        outcome: ReconciliationOutcome,
        imported: ImportedEvaluation | None = None,
        observed_at: datetime | None = None,
        completed_at: datetime | None = None,
        reconciler_version: str = "test-v1",
        can_prove_absence: bool = False,
    ) -> tuple[ScoringReconciliationRegistry, SyntheticReconciliationAdapter]:
        observed = observed_at or START + timedelta(minutes=7)
        receipt = ScoringReconciliationReceipt(
            claim_id=claim["id"],
            attempt_id=claim["attempt_id"],
            evaluation_id=claim["evaluation_id"],
            through_sequence=claim["through_sequence"],
            provider_id=claim["provider_id"],
            provider_version=claim["provider_version"],
            reconciler_id="synthetic.fixed-reconciler",
            reconciler_version=reconciler_version,
            action_trace_digest=claim["action_trace_digest"],
            command_hash=claim["command_hash"],
            scoring_request_digest=claim["scoring_request_digest"],
            provider_binding_digest=claim["provider_binding_digest"],
            outcome=outcome,
            observed_at=observed.isoformat(),
            completed_at=(
                None if completed_at is None else completed_at.isoformat()
            ),
            result_digest=(
                None if imported is None else imported.digest
            ),
            reason_code={
                ReconciliationOutcome.UNKNOWN: "provider_status_unknown",
                ReconciliationOutcome.DEFINITELY_ABSENT: (
                    "fenced_operation_absent"
                ),
                ReconciliationOutcome.COMPLETED: "provider_result_recovered",
            }[outcome],
            provider_operation_digest=claim[
                "provider_operation_digest"
            ],
            provider_receipt_digest=canonical_digest(
                {
                    "claim_id": claim["id"],
                    "outcome": outcome.value,
                    "observed_at": observed.isoformat(),
                    "reconciler_version": reconciler_version,
                }
            ),
            attestation_digest=canonical_digest(
                {
                    "provider_operation_digest": claim[
                        "provider_operation_digest"
                    ],
                    "outcome": outcome.value,
                    "reconciler_version": reconciler_version,
                }
            ),
        )
        adapter = SyntheticReconciliationAdapter(
            ReconciliationObservation(
                receipt=receipt,
                imported_evaluation=imported,
            ),
            reconciler_version=reconciler_version,
            can_prove_absence=can_prove_absence,
        )
        registry = ScoringReconciliationRegistry(allow_synthetic=True)
        registry.register(adapter, adapter.authority_binding)
        return registry, adapter

    def test_scoring_operations_reject_all_technical_event_namespaces(
        self,
    ) -> None:
        _attempt, _submitted, claim, _provider, _imported = (
            self.failed_scoring_claim()
        )
        reconciliation_registry, adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.UNKNOWN,
        )
        scoring_registry = ScoringProviderRegistry()
        for prefix in (
            PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
            PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
            PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
            PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
        ):
            key = prefix + ("a" * 64)
            with self.subTest(operation="score", prefix=prefix):
                with self.assertRaisesRegex(
                    ValidationError,
                    "reserved technical namespace",
                ):
                    self.ledger.score_attempt(
                        claim["attempt_id"],
                        scoring_registry,
                        "unused.provider",
                        "v1",
                        idempotency_key=key,
                    )
            with self.subTest(operation="reconcile", prefix=prefix):
                with self.assertRaisesRegex(
                    ValidationError,
                    "reserved technical namespace",
                ):
                    self.ledger.reconcile_scoring_claim(
                        claim["id"],
                        reconciliation_registry,
                        adapter.reconciler_id,
                        adapter.reconciler_version,
                        idempotency_key=key,
                    )
        self.assertEqual(adapter.lookup_calls, 0)

    def test_complete_synthetic_flow_is_replayable_shadow_and_projection_neutral(
        self,
    ) -> None:
        before = self.projection_snapshot()
        attempt = self.start()
        replay = self.start()
        self.assertEqual(replay["id"], attempt["id"])
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(self.projection_snapshot(), before)

        with self.assertRaisesRegex(ConflictError, "active performance task"):
            self.engine.next_question(self.session["id"])
        with self.assertRaisesRegex(ConflictError, "active performance task"):
            self.engine.end_session(self.session["id"])

        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=1.0,
                    outcome_code="synthetic_pass",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                    reliability=1.0,
                ),
            )
        )
        provider = SyntheticDeterministicProvider(imported)
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        scored = self.ledger.score_attempt(
            attempt["id"],
            registry,
            provider.provider_id,
            provider.provider_version,
            idempotency_key="score-performance",
            now=START + timedelta(minutes=6),
        )

        self.assertFalse(scored["projection_applied"])
        self.assertFalse(scored["certification_applied"])
        self.assertEqual(
            scored["shadow_evidence"]["total_evidence_weight"], 0.0
        )
        self.assertIn(
            "synthetic_provider_shadow_only",
            {
                reason
                for record in scored["shadow_evidence"]["records"]
                for reason in record["reason_codes"]
            }
            | {
                decision["reason_code"]
                for decision in scored["authority"]["normalized_result"][
                    "decisions"
                ]
            },
        )
        self.assertEqual(self.projection_snapshot(), before)
        report = self.ledger.report(attempt["id"])
        self.assertEqual(report["status"], "submitted")
        self.assertEqual(report["action_count"], 3)
        self.assertEqual(report["evaluation_count"], 1)
        self.assertFalse(report["family_shadow_history"]["mastery_claim"])
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_v2_claim_binds_exact_request_provider_and_operation(self) -> None:
        _attempt, _submitted, claim, provider, _imported = (
            self.failed_scoring_claim()
        )
        self.assertEqual(claim["claim_schema_version"], 2)
        self.assertEqual(claim["status"], "unreconciled")
        self.assertEqual(claim["status_source"], "callback_admission")
        self.assertFalse(claim["terminal"])
        self.assertFalse(claim["automatic_retry_allowed"])
        self.assertEqual(
            claim["provider_binding_digest"],
            provider.authority_binding.digest,
        )
        with self.database.read() as connection:
            event = connection.execute(
                "SELECT * FROM events WHERE event_id=?",
                (claim["event_id"],),
            ).fetchone()
        payload = json.loads(event["payload_json"])
        self.assertEqual(event["schema_version"], 2)
        self.assertEqual(
            payload["provider"]["binding_digest"],
            claim["provider_binding_digest"],
        )
        self.assertEqual(
            payload["scoring_request_digest"],
            claim["scoring_request_digest"],
        )
        self.assertEqual(
            payload["provider_operation_digest"],
            claim["provider_operation_digest"],
        )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_unknown_reconciliation_is_repeatable_but_exact_key_replays(
        self,
    ) -> None:
        _attempt, _submitted, claim, provider, _imported = (
            self.failed_scoring_claim()
        )
        registry_one, adapter_one = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.UNKNOWN,
            reconciler_version="test-v1",
        )
        first = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry_one,
            adapter_one.reconciler_id,
            adapter_one.reconciler_version,
            idempotency_key="reconcile-unknown-one",
            now=START + timedelta(minutes=9),
        )
        replay = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry_one,
            adapter_one.reconciler_id,
            adapter_one.reconciler_version,
            idempotency_key="reconcile-unknown-one",
            now=START + timedelta(minutes=10),
        )
        self.assertEqual(adapter_one.lookup_calls, 1)
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(
            replay["reconciliation_id"], first["reconciliation_id"]
        )
        self.assertEqual(replay["reconciliation_outcome"], "unknown")
        with self.assertRaisesRegex(
            ConflictError, "same reconciliation receipt"
        ):
            self.ledger.reconcile_scoring_claim(
                claim["id"],
                registry_one,
                adapter_one.reconciler_id,
                adapter_one.reconciler_version,
                idempotency_key="different-key-same-receipt",
                now=START + timedelta(minutes=10),
            )
        self.assertEqual(adapter_one.lookup_calls, 2)
        score_registry = ScoringProviderRegistry(allow_synthetic=True)
        score_registry.register(provider, provider.authority_binding)
        with self.assertRaisesRegex(
            ConflictError, "reserved by a scoring reconciliation"
        ):
            self.ledger.score_attempt(
                claim["attempt_id"],
                score_registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="reconcile-unknown-one",
                now=START + timedelta(minutes=10),
            )
        self.assertEqual(provider.calls, 1)

        registry_two, adapter_two = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.UNKNOWN,
            observed_at=START + timedelta(minutes=7, seconds=30),
            reconciler_version="test-v2",
        )
        second = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry_two,
            adapter_two.reconciler_id,
            adapter_two.reconciler_version,
            idempotency_key="reconcile-unknown-two",
            # Deliberately earlier than the first ledger timestamp. Append
            # order, not wall-clock order, defines the latest observation.
            now=START + timedelta(minutes=8),
        )
        inspected = self.ledger.inspect_scoring_claim(claim["id"])
        self.assertEqual(second["status"], "unknown")
        self.assertEqual(inspected["reconciliation_count"], 2)
        self.assertEqual(
            inspected["latest_reconciliation_id"],
            second["reconciliation_id"],
        )
        self.assertFalse(inspected["automatic_retry_allowed"])
        replay_report = ProjectionReplay(self.database).check(
            "performance-learner"
        )
        self.assertTrue(replay_report["ok"], replay_report["errors"])
        self.assertEqual(
            replay_report[
                "reconstructed_performance_scoring_reconciliation_count"
            ],
            2,
        )
        rebuilt_path = Path(self.tempdir.name) / "reconciled-rebuild.db"
        rebuilt_report = ProjectionReplay(self.database).rebuild_copy(
            "performance-learner", rebuilt_path
        )
        self.assertTrue(rebuilt_report["ok"], rebuilt_report["errors"])
        self.assertEqual(
            rebuilt_report[
                "reconstructed_performance_scoring_reconciliation_count"
            ],
            2,
        )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_definite_absence_requires_fenced_authority_and_is_terminal(
        self,
    ) -> None:
        _attempt, _submitted, claim, _provider, _imported = (
            self.failed_scoring_claim()
        )
        weak_registry, weak_adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.DEFINITELY_ABSENT,
            can_prove_absence=False,
        )
        with self.assertRaisesRegex(ValidationError, "failed safely"):
            self.ledger.reconcile_scoring_claim(
                claim["id"],
                weak_registry,
                weak_adapter.reconciler_id,
                weak_adapter.reconciler_version,
                idempotency_key="weak-absence",
                now=START + timedelta(minutes=8),
            )
        strong_registry, strong_adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.DEFINITELY_ABSENT,
            reconciler_version="fenced-v1",
            can_prove_absence=True,
        )
        closed = self.ledger.reconcile_scoring_claim(
            claim["id"],
            strong_registry,
            strong_adapter.reconciler_id,
            strong_adapter.reconciler_version,
            idempotency_key="strong-absence",
            now=START + timedelta(minutes=8),
        )
        self.assertEqual(closed["status"], "definitely_absent")
        self.assertTrue(closed["terminal"])
        self.assertFalse(closed["automatic_retry_allowed"])
        self.assertEqual(strong_adapter.lookup_calls, 1)
        with self.assertRaisesRegex(ConflictError, "terminal reconciliation"):
            self.ledger.reconcile_scoring_claim(
                claim["id"],
                strong_registry,
                strong_adapter.reconciler_id,
                strong_adapter.reconciler_version,
                idempotency_key="different-after-terminal",
                now=START + timedelta(minutes=9),
            )
        self.assertEqual(strong_adapter.lookup_calls, 1)
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM task_evaluations"
                ).fetchone()["n"],
                0,
            )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_completed_reconciliation_recovers_shadow_evidence_after_end(
        self,
    ) -> None:
        attempt, _submitted, claim, _provider, imported = (
            self.failed_scoring_claim()
        )
        self.engine.end_session(
            self.session["id"],
            now=START + timedelta(minutes=7),
        )
        registry, adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.COMPLETED,
            imported=imported,
            observed_at=START + timedelta(minutes=8),
            completed_at=START + timedelta(minutes=7, seconds=30),
        )
        result = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry,
            adapter.reconciler_id,
            adapter.reconciler_version,
            idempotency_key="recover-after-session",
            now=START + timedelta(minutes=9),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["status_source"], "reconciliation")
        self.assertFalse(result["projection_applied"])
        self.assertFalse(result["certification_applied"])
        self.assertTrue(
            result["authority"]["normalized_result"]["shadow_only"]
        )
        self.assertEqual(
            result["authority"]["normalized_result"]["normalization_mode"],
            "direct_import",
        )
        with self.database.read() as connection:
            events = connection.execute(
                """SELECT event_type, session_id, causation_id
                   FROM events
                   WHERE event_type IN (
                       'PerformanceScoringReconciled',
                       'TaskEvaluationRecorded',
                       'ShadowEvidenceReduced'
                   )
                   AND correlation_id=?
                   ORDER BY stream_version""",
                (attempt["id"],),
            ).fetchall()
        self.assertEqual(
            [event["event_type"] for event in events],
            [
                "PerformanceScoringReconciled",
                "TaskEvaluationRecorded",
                "ShadowEvidenceReduced",
            ],
        )
        self.assertTrue(all(event["session_id"] is None for event in events))
        # The exact causation is the immutable reconciliation event, not the
        # ended session or original callback claim.
        with self.database.read() as connection:
            reconciliation_event_id = connection.execute(
                """SELECT event_id
                   FROM performance_scoring_reconciliations
                   WHERE id=?""",
                (result["reconciliation_id"],),
            ).fetchone()["event_id"]
        self.assertEqual(events[1]["causation_id"], reconciliation_event_id)
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_completed_reconciliation_semantically_binds_recovered_result(
        self,
    ) -> None:
        _attempt, _submitted, claim, _provider, imported = (
            self.failed_scoring_claim()
        )
        registry, adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.COMPLETED,
            imported=imported,
            observed_at=START + timedelta(minutes=8),
            completed_at=START + timedelta(minutes=7, seconds=30),
        )
        result = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry,
            adapter.reconciler_id,
            adapter.reconciler_version,
            idempotency_key="semantic-recovery-binding",
            now=START + timedelta(minutes=9),
        )
        with self.database.read() as connection:
            observation = connection.execute(
                """SELECT * FROM performance_scoring_reconciliations
                   WHERE id=?""",
                (result["reconciliation_id"],),
            ).fetchone()
            event = connection.execute(
                "SELECT * FROM events WHERE event_id=?",
                (observation["event_id"],),
            ).fetchone()

        receipt_terms = json.loads(observation["receipt_json"])
        receipt_terms["result_digest"] = "f" * 64
        mutated_receipt = ScoringReconciliationReceipt.from_terms(
            receipt_terms
        )
        payload = json.loads(event["payload_json"])
        payload["receipt"] = mutated_receipt.terms()
        payload["receipt_digest"] = mutated_receipt.digest
        command_hash = canonical_digest(
            {
                "type": "tsq.performance_command",
                "operation": "reconcile_scoring_claim",
                "claim_id": observation["claim_id"],
                "provider_operation_digest": observation[
                    "provider_operation_digest"
                ],
                "reconciler_id": observation["reconciler_id"],
                "reconciler_version": observation["reconciler_version"],
                "reconciliation_binding_digest": observation[
                    "reconciliation_binding_digest"
                ],
                "receipt_digest": mutated_receipt.digest,
                "caller_idempotency_key": observation["idempotency_key"],
            }
        )
        reconciliation_id = "psr_" + command_hash
        payload["reconciliation_id"] = reconciliation_id
        payload["command_hash"] = command_hash
        metadata = json.loads(event["metadata_json"])
        metadata["command_hash"] = command_hash

        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_scoring_reconciliations_no_update"
            )
            connection.execute("DROP TRIGGER events_no_update")
            connection.execute(
                """UPDATE events
                   SET idempotency_key=?, payload_json=?, metadata_json=?
                   WHERE event_id=?""",
                (
                    performance_scoring_reconciliation_event_key(command_hash),
                    canonical_json(payload),
                    canonical_json(metadata),
                    event["event_id"],
                ),
            )
            connection.execute(
                """UPDATE performance_scoring_reconciliations
                   SET id=?, receipt_json=?, receipt_digest=?, command_hash=?
                   WHERE event_id=?""",
                (
                    reconciliation_id,
                    canonical_json(mutated_receipt.terms()),
                    mutated_receipt.digest,
                    command_hash,
                    event["event_id"],
                ),
            )

        with self.database.read() as connection:
            errors = performance_integrity_errors(connection)
            with self.assertRaisesRegex(
                ValidationError,
                "receipt does not match its recovered imported result",
            ):
                derive_performance_projections(connection)
        self.assertTrue(
            any(
                "recovered imported result digest does not match its receipt"
                in error
                for error in errors
            ),
            errors,
        )

    def test_claim_inspection_rejects_completed_projection_without_result(
        self,
    ) -> None:
        _attempt, _submitted, claim, _provider, _imported = (
            self.failed_scoring_claim()
        )
        registry, adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.UNKNOWN,
        )
        result = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry,
            adapter.reconciler_id,
            adapter.reconciler_version,
            idempotency_key="corrupt-completed-read-boundary",
            now=START + timedelta(minutes=8),
        )
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_scoring_reconciliations_no_update"
            )
            connection.execute(
                """UPDATE performance_scoring_reconciliations
                   SET outcome='completed', normalized_result_digest=?
                   WHERE id=?""",
                ("a" * 64, result["reconciliation_id"]),
            )
        with self.assertRaisesRegex(
            ConflictError,
            "without its exact recovered evaluation",
        ):
            self.ledger.inspect_scoring_claim(claim["id"])
        with self.assertRaisesRegex(
            ConflictError,
            "without its exact recovered evaluation",
        ):
            self.ledger.list_scoring_claims(attempt_id=claim["attempt_id"])

    def test_claim_inspection_rejects_orphan_reconciliation_event(
        self,
    ) -> None:
        _attempt, _submitted, claim, _provider, imported = (
            self.failed_scoring_claim()
        )
        registry, adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.COMPLETED,
            imported=imported,
            observed_at=START + timedelta(minutes=8),
            completed_at=START + timedelta(minutes=7, seconds=30),
        )
        result = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry,
            adapter.reconciler_id,
            adapter.reconciler_version,
            idempotency_key="orphan-reconciliation-read-boundary",
            now=START + timedelta(minutes=9),
        )
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_scoring_reconciliations_no_delete"
            )
            connection.execute(
                """DELETE FROM performance_scoring_reconciliations
                   WHERE id=?""",
                (result["reconciliation_id"],),
            )
        for operation in (
            lambda: self.ledger.inspect_scoring_claim(claim["id"]),
            lambda: self.ledger.list_scoring_claims(
                attempt_id=claim["attempt_id"]
            ),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(
                    ConflictError,
                    "reconciliation event .* is missing its projection",
                ):
                    operation()

    def test_integrity_binds_recovery_completion_time_and_event_order(
        self,
    ) -> None:
        attempt, _submitted, claim, _provider, imported = (
            self.failed_scoring_claim()
        )
        registry, adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.COMPLETED,
            imported=imported,
            observed_at=START + timedelta(minutes=8),
            completed_at=START + timedelta(minutes=7, seconds=30),
        )
        result = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry,
            adapter.reconciler_id,
            adapter.reconciler_version,
            idempotency_key="recovery-time-order-binding",
            now=START + timedelta(minutes=9),
        )
        with self.database.read() as connection:
            reconciliation_event = connection.execute(
                """SELECT event.*
                   FROM performance_scoring_reconciliations observation
                   JOIN events event ON event.event_id=observation.event_id
                   WHERE observation.id=?""",
                (result["reconciliation_id"],),
            ).fetchone()
            evaluation_event = connection.execute(
                """SELECT event.*
                   FROM task_evaluations evaluation
                   JOIN events event ON event.event_id=evaluation.event_id
                   WHERE evaluation.id=?""",
                (claim["evaluation_id"],),
            ).fetchone()
            maximum_version = connection.execute(
                """SELECT MAX(stream_version) AS version FROM events
                   WHERE stream_id=?""",
                (reconciliation_event["stream_id"],),
            ).fetchone()["version"]

        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            connection.execute(
                "UPDATE events SET stream_version=? WHERE event_id=?",
                (
                    maximum_version + 100,
                    reconciliation_event["event_id"],
                ),
            )
            connection.execute(
                """UPDATE events
                   SET stream_version=?, occurred_at=?
                   WHERE event_id=?""",
                (
                    reconciliation_event["stream_version"],
                    (START + timedelta(minutes=7, seconds=45)).isoformat(),
                    evaluation_event["event_id"],
                ),
            )
            connection.execute(
                "UPDATE events SET stream_version=? WHERE event_id=?",
                (
                    evaluation_event["stream_version"],
                    reconciliation_event["event_id"],
                ),
            )

        with self.assertRaisesRegex(
            ConflictError,
            "does not bind its recovered evaluation",
        ):
            self.ledger.inspect_scoring_claim(claim["id"])
        with self.database.read() as connection:
            errors = performance_integrity_errors(connection)
        self.assertTrue(
            any(
                "recovered evaluation occurrence does not match its receipt"
                in error
                for error in errors
            ),
            errors,
        )
        self.assertTrue(
            any(
                "recovered evaluation does not follow its reconciliation event"
                in error
                for error in errors
            ),
            errors,
        )
        self.assertEqual(attempt["id"], claim["attempt_id"])

    def test_reconciliation_rejects_preclaim_and_future_timestamps(
        self,
    ) -> None:
        _attempt, _submitted, claim, _provider, _imported = (
            self.failed_scoring_claim()
        )
        preclaim_registry, preclaim_adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.UNKNOWN,
            observed_at=START + timedelta(minutes=5),
        )
        with self.assertRaisesRegex(ValidationError, "predate"):
            self.ledger.reconcile_scoring_claim(
                claim["id"],
                preclaim_registry,
                preclaim_adapter.reconciler_id,
                preclaim_adapter.reconciler_version,
                idempotency_key="preclaim-observation",
                now=START + timedelta(minutes=8),
            )
        future_registry, future_adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.UNKNOWN,
            observed_at=START + timedelta(minutes=10),
            reconciler_version="future-v1",
        )
        with self.assertRaisesRegex(ValidationError, "before its claimed"):
            self.ledger.reconcile_scoring_claim(
                claim["id"],
                future_registry,
                future_adapter.reconciler_id,
                future_adapter.reconciler_version,
                idempotency_key="future-observation",
                now=START + timedelta(minutes=9),
            )
        inspected = self.ledger.inspect_scoring_claim(claim["id"])
        self.assertEqual(inspected["status"], "unreconciled")
        self.assertEqual(inspected["reconciliation_count"], 0)

    def test_reconciliation_obeys_learner_quarantine_before_lookup_and_commit(
        self,
    ) -> None:
        _attempt, _submitted, claim, _provider, _imported = (
            self.failed_scoring_claim()
        )
        registry, adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.UNKNOWN,
        )
        with patch.object(
            self.database,
            "require_learner_evidence_safe",
            side_effect=ConflictError("learner evidence is quarantined"),
        ):
            with self.assertRaisesRegex(ConflictError, "quarantined"):
                self.ledger.reconcile_scoring_claim(
                    claim["id"],
                    registry,
                    adapter.reconciler_id,
                    adapter.reconciler_version,
                    idempotency_key="unsafe-before-lookup",
                    now=START + timedelta(minutes=8),
                )
        self.assertEqual(adapter.lookup_calls, 0)

        with patch.object(
            self.database,
            "require_learner_evidence_safe",
            side_effect=(
                None,
                ConflictError("learner evidence became quarantined"),
            ),
        ):
            with self.assertRaisesRegex(ConflictError, "became quarantined"):
                self.ledger.reconcile_scoring_claim(
                    claim["id"],
                    registry,
                    adapter.reconciler_id,
                    adapter.reconciler_version,
                    idempotency_key="unsafe-during-lookup",
                    now=START + timedelta(minutes=8),
                )
        self.assertEqual(adapter.lookup_calls, 1)
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_scoring_reconciliations
                       WHERE claim_id=?""",
                    (claim["id"],),
                ).fetchone()["n"],
                0,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE event_type='PerformanceScoringReconciled'
                         AND json_extract(payload_json, '$.claim_id')=?""",
                    (claim["id"],),
                ).fetchone()["n"],
                0,
            )

        safe_registry, safe_adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.UNKNOWN,
            reconciler_version="safe-replay-v1",
        )
        recorded = self.ledger.reconcile_scoring_claim(
            claim["id"],
            safe_registry,
            safe_adapter.reconciler_id,
            safe_adapter.reconciler_version,
            idempotency_key="safe-then-quarantined",
            now=START + timedelta(minutes=8),
        )
        self.assertEqual(recorded["status"], "unknown")
        self.assertEqual(safe_adapter.lookup_calls, 1)
        with patch.object(
            self.database,
            "require_learner_evidence_safe",
            side_effect=ConflictError("learner evidence is quarantined"),
        ):
            with self.assertRaisesRegex(ConflictError, "quarantined"):
                self.ledger.reconcile_scoring_claim(
                    claim["id"],
                    safe_registry,
                    safe_adapter.reconciler_id,
                    safe_adapter.reconciler_version,
                    idempotency_key="safe-then-quarantined",
                    now=START + timedelta(minutes=9),
                )
        self.assertEqual(safe_adapter.lookup_calls, 1)
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_reconciliation_crash_before_commit_leaves_no_partial_state(
        self,
    ) -> None:
        _attempt, _submitted, claim, _provider, _imported = (
            self.failed_scoring_claim()
        )
        registry, adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.UNKNOWN,
        )
        append_event = self.database.append_event

        def crash_on_reconciliation(connection, **kwargs):
            if kwargs.get("event_type") == "PerformanceScoringReconciled":
                raise RuntimeError("simulated crash before atomic append")
            return append_event(connection, **kwargs)

        with patch.object(
            self.database,
            "append_event",
            side_effect=crash_on_reconciliation,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                self.ledger.reconcile_scoring_claim(
                    claim["id"],
                    registry,
                    adapter.reconciler_id,
                    adapter.reconciler_version,
                    idempotency_key="crash-before-reconcile-commit",
                    now=START + timedelta(minutes=8),
                )
        self.assertEqual(adapter.lookup_calls, 1)
        inspected = self.ledger.inspect_scoring_claim(claim["id"])
        self.assertEqual(inspected["status"], "unreconciled")
        self.assertEqual(inspected["reconciliation_count"], 0)

        recovered = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry,
            adapter.reconciler_id,
            adapter.reconciler_version,
            idempotency_key="crash-before-reconcile-commit",
            now=START + timedelta(minutes=8),
        )
        self.assertEqual(adapter.lookup_calls, 2)
        self.assertEqual(recovered["status"], "unknown")
        self.assertEqual(recovered["reconciliation_count"], 1)
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_inline_callback_completion_rejects_stale_reconciliation(
        self,
    ) -> None:
        attempt = self.start(key="start-stale-reconciliation-race")
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.75,
                    outcome_code="inline_completion_wins",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )

        class BlockingProvider(SyntheticDeterministicProvider):
            def __init__(self):
                super().__init__(
                    imported,
                    provider_id="synthetic.inline-wins-race",
                )
                self.entered = threading.Event()
                self.release = threading.Event()

            def score(self, request):
                self.entered.set()
                if not self.release.wait(timeout=5):
                    raise RuntimeError("inline race timed out")
                return super().score(request)

        provider = BlockingProvider()
        score_registry = ScoringProviderRegistry(allow_synthetic=True)
        score_registry.register(provider, provider.authority_binding)
        with ThreadPoolExecutor(max_workers=2) as executor:
            scoring = executor.submit(
                self.ledger.score_attempt,
                attempt["id"],
                score_registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="inline-wins-score",
                now=START + timedelta(minutes=6),
            )
            self.assertTrue(provider.entered.wait(timeout=5))
            claim = self.ledger.list_scoring_claims(
                attempt_id=attempt["id"]
            )[0]
            reconciliation_registry, adapter = (
                self.reconciliation_registry(
                    claim,
                    outcome=ReconciliationOutcome.UNKNOWN,
                    observed_at=START + timedelta(minutes=7),
                )
            )
            lookup_entered = threading.Event()
            lookup_release = threading.Event()
            ordinary_lookup = adapter.lookup

            def blocking_lookup(request):
                lookup_entered.set()
                if not lookup_release.wait(timeout=5):
                    raise RuntimeError("stale lookup timed out")
                return ordinary_lookup(request)

            adapter.lookup = blocking_lookup
            reconciling = executor.submit(
                self.ledger.reconcile_scoring_claim,
                claim["id"],
                reconciliation_registry,
                adapter.reconciler_id,
                adapter.reconciler_version,
                idempotency_key="stale-reconciliation",
                now=START + timedelta(minutes=8),
            )
            self.assertTrue(lookup_entered.wait(timeout=5))
            provider.release.set()
            scored = scoring.result(timeout=5)
            lookup_release.set()
            with self.assertRaisesRegex(
                ConflictError, "while reconciliation lookup was in flight"
            ):
                reconciling.result(timeout=5)

        self.assertFalse(scored["idempotent_replay"])
        self.assertEqual(adapter.lookup_calls, 1)
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_scoring_reconciliations
                       WHERE claim_id=?""",
                    (claim["id"],),
                ).fetchone()["n"],
                0,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM task_evaluations WHERE attempt_id=?""",
                    (attempt["id"],),
                ).fetchone()["n"],
                1,
            )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_reconciliation_tampering_is_detected_and_copy_replays_event(
        self,
    ) -> None:
        _attempt, _submitted, claim, _provider, _imported = (
            self.failed_scoring_claim()
        )
        registry, adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.UNKNOWN,
        )
        recorded = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry,
            adapter.reconciler_id,
            adapter.reconciler_version,
            idempotency_key="tamper-reconciliation",
            now=START + timedelta(minutes=8),
        )
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_scoring_reconciliations_no_update"
            )
            connection.execute(
                """UPDATE performance_scoring_reconciliations
                   SET receipt_digest=?
                   WHERE id=?""",
                ("0" * 64, recorded["reconciliation_id"]),
            )
        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "performance scoring reconciliation" in error
                or "performance_scoring_reconciliations" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        replay = ProjectionReplay(self.database).check(
            "performance-learner"
        )
        self.assertFalse(replay["ok"])
        self.assertFalse(replay["performance_projection_matches_replay"])
        target = Path(self.tempdir.name) / "reconciliation-repaired.db"
        rebuilt = ProjectionReplay(self.database).rebuild_copy(
            "performance-learner", target
        )
        self.assertTrue(rebuilt["ok"], rebuilt["errors"])
        repaired = Database(target, read_only=True)
        self.assertTrue(repaired.verify_integrity()["ok"])

    def test_recovered_completion_wins_race_with_original_callback_once(
        self,
    ) -> None:
        attempt = self.start(key="start-reconciliation-race")
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.9,
                    outcome_code="reconciliation_race_completion",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )

        class BlockingProvider(SyntheticDeterministicProvider):
            def __init__(self):
                super().__init__(
                    imported,
                    provider_id="synthetic.reconciliation-race",
                )
                self.entered = threading.Event()
                self.release = threading.Event()

            def score(self, request):
                self.entered.set()
                if not self.release.wait(timeout=5):
                    raise RuntimeError("reconciliation race timed out")
                return super().score(request)

        provider = BlockingProvider()
        score_registry = ScoringProviderRegistry(allow_synthetic=True)
        score_registry.register(provider, provider.authority_binding)
        with ThreadPoolExecutor(max_workers=1) as executor:
            scoring = executor.submit(
                self.ledger.score_attempt,
                attempt["id"],
                score_registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="original-racing-score",
                now=START + timedelta(minutes=6),
            )
            self.assertTrue(provider.entered.wait(timeout=5))
            claim = self.ledger.list_scoring_claims(
                attempt_id=attempt["id"]
            )[0]
            reconciliation_registry, adapter = (
                self.reconciliation_registry(
                    claim,
                    outcome=ReconciliationOutcome.COMPLETED,
                    imported=imported,
                    observed_at=START + timedelta(minutes=7),
                    completed_at=START + timedelta(minutes=7),
                )
            )
            recovered = self.ledger.reconcile_scoring_claim(
                claim["id"],
                reconciliation_registry,
                adapter.reconciler_id,
                adapter.reconciler_version,
                idempotency_key="reconciliation-wins-race",
                now=START + timedelta(minutes=8),
            )
            provider.release.set()
            original_result = scoring.result(timeout=5)

        self.assertEqual(recovered["status"], "completed")
        self.assertTrue(original_result["idempotent_replay"])
        self.assertEqual(
            original_result["evaluation"]["id"],
            recovered["evaluation"]["id"],
        )
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM task_evaluations WHERE attempt_id=?""",
                    (attempt["id"],),
                ).fetchone()["n"],
                1,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_scoring_reconciliations
                       WHERE claim_id=? AND outcome='completed'""",
                    (claim["id"],),
                ).fetchone()["n"],
                1,
            )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_task_start_wins_race_with_inflight_question_selection(self) -> None:
        scoring_started = threading.Event()
        continue_scoring = threading.Event()
        original_score = self.engine.policy._score

        def blocking_score(*args, **kwargs):
            if not scoring_started.is_set():
                scoring_started.set()
                if not continue_scoring.wait(timeout=5):
                    raise RuntimeError("question-selection race probe timed out")
            return original_score(*args, **kwargs)

        self.engine.policy._score = blocking_score
        with ThreadPoolExecutor(max_workers=1) as executor:
            selection = executor.submit(
                self.engine.next_question,
                self.session["id"],
                now=START + timedelta(minutes=2),
            )
            self.assertTrue(scoring_started.wait(timeout=5))
            attempt = self.start(key="start-during-question-scoring")
            continue_scoring.set()
            with self.assertRaisesRegex(
                ConflictError, "active performance task"
            ):
                selection.result(timeout=5)

        with self.database.read() as connection:
            pending_questions = connection.execute(
                """SELECT COUNT(*) AS n FROM decisions
                   WHERE session_id=? AND consumed_at IS NULL
                     AND invalidated_at IS NULL""",
                (self.session["id"],),
            ).fetchone()["n"]
            open_tasks = connection.execute(
                """SELECT COUNT(*) AS n
                   FROM performance_attempts task
                   WHERE task.session_id=?
                     AND NOT EXISTS (
                         SELECT 1 FROM performance_actions terminal
                         WHERE terminal.attempt_id=task.id
                           AND terminal.action_type IN ('submitted', 'abandoned')
                     )""",
                (self.session["id"],),
            ).fetchone()["n"]
        self.assertEqual(pending_questions, 0)
        self.assertEqual(open_tasks, 1)
        self.assertEqual(attempt["session_id"], self.session["id"])
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_scoring_provider_runs_without_holding_database_writer_lock(
        self,
    ) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=1.0,
                    outcome_code="lock_probe_pass",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                    reliability=1.0,
                ),
            )
        )

        class LockProbeProvider(SyntheticDeterministicProvider):
            def score(self, request):
                # A provider callback must not inherit the ledger's immediate
                # writer lock. An empty independent write transaction is a
                # deterministic, fast probe of that boundary.
                with self_database.transaction() as connection:
                    connection.execute("SELECT 1").fetchone()
                return super().score(request)

        self_database = self.database
        provider = LockProbeProvider(
            imported,
            provider_id="synthetic.lock-probe-scorer",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)

        scored = self.ledger.score_attempt(
            attempt["id"],
            registry,
            provider.provider_id,
            provider.provider_version,
            idempotency_key="score-without-writer-lock",
            now=START + timedelta(minutes=6),
        )

        self.assertEqual(scored["evaluation"]["id"].split("_")[0], "teval")
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_reentrant_scoring_claim_fails_closed_before_second_callback(
        self,
    ) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.8,
                    outcome_code="reentrant_probe",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )

        class ReentrantProvider(SyntheticDeterministicProvider):
            calls = 0

            def score(self, request):
                self.calls += 1
                if self.calls == 1:
                    try:
                        self_ledger.score_attempt(
                            attempt["id"],
                            registry,
                            self.provider_id,
                            self.provider_version,
                            idempotency_key="reentrant-score-once",
                            now=START + timedelta(minutes=6),
                        )
                    except ConflictError as exc:
                        nested_conflicts.append(str(exc))
                return super().score(request)

        self_ledger = self.ledger
        nested_conflicts: list[str] = []
        provider = ReentrantProvider(
            imported,
            provider_id="synthetic.reentrant-scorer",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)

        result = self.ledger.score_attempt(
            attempt["id"],
            registry,
            provider.provider_id,
            provider.provider_version,
            idempotency_key="reentrant-score-once",
            now=START + timedelta(minutes=6),
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(nested_conflicts), 1)
        self.assertIn("callback will not be repeated", nested_conflicts[0])
        self.assertFalse(result["idempotent_replay"])
        replay = self.ledger.score_attempt(
            attempt["id"],
            registry,
            provider.provider_id,
            provider.provider_version,
            idempotency_key="reentrant-score-once",
            now=START + timedelta(minutes=6),
        )
        self.assertEqual(replay["evaluation"], result["evaluation"])
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(provider.calls, 1)
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM task_evaluations WHERE attempt_id=?",
                    (attempt["id"],),
                ).fetchone()["n"],
                1,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE event_type='TaskEvaluationRecorded'
                         AND idempotency_key='reentrant-score-once'"""
                ).fetchone()["n"],
                1,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_scoring_claims
                       WHERE idempotency_key='reentrant-score-once'"""
                ).fetchone()["n"],
                1,
            )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_concurrent_same_key_admits_one_provider_callback_across_connections(
        self,
    ) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.9,
                    outcome_code="concurrent_claim_probe",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )

        class BlockingProvider(SyntheticDeterministicProvider):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0
                self.call_lock = threading.Lock()
                self.entered = threading.Event()
                self.release = threading.Event()
                self.duplicate_entered = threading.Event()

            def score(self, request):
                with self.call_lock:
                    self.calls += 1
                    if self.calls > 1:
                        self.duplicate_entered.set()
                self.entered.set()
                if not self.release.wait(timeout=5):
                    raise RuntimeError("threaded scoring probe timed out")
                return super().score(request)

        provider = BlockingProvider(
            imported,
            provider_id="synthetic.concurrent-claim-scorer",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)

        def score_once(
            idempotency_key: str = "concurrent-score-once",
        ) -> dict:
            # Separate service objects force coordination through distinct
            # SQLite connections rather than shared Python state.
            ledger = PerformanceLedger(Database(self.database.path))
            return ledger.score_attempt(
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key=idempotency_key,
                now=START + timedelta(minutes=6),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(score_once)
            self.assertTrue(provider.entered.wait(timeout=5))
            second = executor.submit(score_once)
            try:
                with self.assertRaisesRegex(
                    ConflictError, "callback will not be repeated"
                ):
                    second.result(timeout=5)
            finally:
                provider.release.set()
            first_result = first.result(timeout=5)

        self.assertEqual(provider.calls, 1)
        self.assertFalse(provider.duplicate_entered.is_set())
        replay = score_once()
        self.assertEqual(replay["evaluation"], first_result["evaluation"])
        self.assertTrue(replay["idempotent_replay"])
        with self.assertRaisesRegex(
            ConflictError, "different idempotency key"
        ):
            score_once("concurrent-score-switched-key")
        self.assertEqual(provider.calls, 1)
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_scoring_claims
                       WHERE idempotency_key='concurrent-score-once'"""
                ).fetchone()["n"],
                1,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE idempotency_key='concurrent-score-once'"""
                ).fetchone()["n"],
                1,
            )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_scoring_callback_cannot_commit_after_session_end(self) -> None:
        attempt = self.start(key="start-score-end-race")
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.9,
                    outcome_code="score_end_race",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )

        class BlockingProvider(SyntheticDeterministicProvider):
            def __init__(self):
                super().__init__(
                    imported,
                    provider_id="synthetic.score-end-race",
                )
                self.entered = threading.Event()
                self.release = threading.Event()

            def score(self, request):
                self.entered.set()
                if not self.release.wait(timeout=5):
                    raise RuntimeError("score/end race probe timed out")
                return super().score(request)

        provider = BlockingProvider()
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)

        with ThreadPoolExecutor(max_workers=1) as executor:
            scoring = executor.submit(
                self.ledger.score_attempt,
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="score-end-race",
                now=START + timedelta(minutes=6),
            )
            self.assertTrue(provider.entered.wait(timeout=5))
            self.engine.end_session(
                self.session["id"],
                now=START + timedelta(minutes=7),
            )
            provider.release.set()
            with self.assertRaisesRegex(ConflictError, "not active"):
                scoring.result(timeout=5)

        with self.database.read() as connection:
            evaluation_count = connection.execute(
                """SELECT COUNT(*) AS n FROM task_evaluations
                   WHERE attempt_id=?""",
                (attempt["id"],),
            ).fetchone()["n"]
            unresolved_claims = connection.execute(
                """SELECT COUNT(*) AS n FROM performance_scoring_claims claim
                   WHERE claim.attempt_id=?
                     AND NOT EXISTS (
                         SELECT 1 FROM task_evaluations evaluation
                         WHERE evaluation.id=claim.evaluation_id
                           AND evaluation.attempt_id=claim.attempt_id
                     )""",
                (attempt["id"],),
            ).fetchone()["n"]
            post_end_events = connection.execute(
                """SELECT COUNT(*) AS n FROM events event
                   WHERE event.session_id=?
                     AND event.stream_version > (
                         SELECT ended.stream_version FROM events ended
                         WHERE ended.session_id=?
                           AND ended.event_type='SessionEnded'
                     )""",
                (self.session["id"], self.session["id"]),
            ).fetchone()["n"]
        self.assertEqual(evaluation_count, 0)
        self.assertEqual(unresolved_claims, 1)
        self.assertEqual(post_end_events, 0)
        report = self.database.verify_integrity()
        self.assertTrue(report["ok"], report["errors"])

    def test_new_scoring_and_direct_import_fail_after_session_end(self) -> None:
        attempt = self.start(key="start-score-after-end")
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.8,
                    outcome_code="score_after_end",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )
        provider = SyntheticDeterministicProvider(
            imported,
            provider_id="synthetic.score-after-end",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        self.engine.end_session(
            self.session["id"],
            now=START + timedelta(minutes=6),
        )

        with self.assertRaisesRegex(ConflictError, "not active"):
            self.ledger.score_attempt(
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="score-after-end",
                now=START + timedelta(minutes=7),
            )
        with self.assertRaisesRegex(ConflictError, "not active"):
            self.ledger.import_evaluation(
                attempt["id"],
                imported,
                provider_id="import-after-end",
                provider_version="v1",
                idempotency_key="import-after-end",
                now=START + timedelta(minutes=7),
            )

        with self.database.read() as connection:
            claims = connection.execute(
                """SELECT COUNT(*) AS n FROM performance_scoring_claims
                   WHERE attempt_id=?""",
                (attempt["id"],),
            ).fetchone()["n"]
            evaluations = connection.execute(
                """SELECT COUNT(*) AS n FROM task_evaluations
                   WHERE attempt_id=?""",
                (attempt["id"],),
            ).fetchone()["n"]
        self.assertEqual(claims, 0)
        self.assertEqual(evaluations, 0)
        report = self.database.verify_integrity()
        self.assertTrue(report["ok"], report["errors"])

    def test_failed_provider_claim_is_not_automatically_retried(self) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.INVALID,
                    score=None,
                    outcome_code="provider_failed_before_result",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )

        class FailingProvider(SyntheticDeterministicProvider):
            calls = 0

            def score(self, request):
                self.calls += 1
                raise RuntimeError("deterministic failure probe")

        provider = FailingProvider(
            imported,
            provider_id="synthetic.failing-claim-scorer",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)

        with self.assertRaisesRegex(ValidationError, "failed safely"):
            self.ledger.score_attempt(
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="failed-score-claim",
                now=START + timedelta(minutes=6),
            )
        with self.assertRaisesRegex(
            ConflictError, "callback will not be repeated"
        ):
            self.ledger.score_attempt(
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="failed-score-claim",
                now=START + timedelta(minutes=6),
            )
        unresolved = self.ledger.list_scoring_claims(
            attempt_id=attempt["id"], status="unresolved"
        )
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["status"], "unreconciled")
        self.assertIsNone(unresolved[0]["completed_at"])
        self.assertFalse(unresolved[0]["automatic_retry_allowed"])
        with self.assertRaisesRegex(
            ConflictError, "different idempotency key"
        ):
            self.ledger.score_attempt(
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="failed-score-switched-key",
                now=START + timedelta(minutes=6),
            )
        with self.assertRaisesRegex(
            ConflictError, "reserved by an unfinished performance scoring claim"
        ):
            with self.database.transaction() as connection:
                self.database.append_event(
                    connection,
                    stream_id="system",
                    event_type="UnrelatedProbe",
                    payload={"probe": True},
                    idempotency_key="failed-score-claim",
                    occurred_at=START + timedelta(minutes=7),
                )
        with self.assertRaisesRegex(Exception, "claims are immutable"):
            with self.database.transaction() as connection:
                connection.execute(
                    """UPDATE performance_scoring_claims
                       SET provider_version='changed'
                       WHERE idempotency_key='failed-score-claim'"""
                )
        with self.assertRaisesRegex(Exception, "claims are immutable"):
            with self.database.transaction() as connection:
                connection.execute(
                    """DELETE FROM performance_scoring_claims
                       WHERE idempotency_key='failed-score-claim'"""
                )

        self.assertEqual(provider.calls, 1)
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_scoring_claims
                       WHERE idempotency_key='failed-score-claim'"""
                ).fetchone()["n"],
                1,
            )
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE idempotency_key='failed-score-claim'"""
                ).fetchone()["n"],
                0,
            )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_fresh_scoring_claim_has_one_linked_internal_event(self) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.85,
                    outcome_code="event_backed_claim_probe",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )

        class CountingProvider(SyntheticDeterministicProvider):
            calls = 0

            def score(self, request):
                self.calls += 1
                return super().score(request)

        provider = CountingProvider(
            imported,
            provider_id="synthetic.event-backed-claim-scorer",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)

        scored = self.ledger.score_attempt(
            attempt["id"],
            registry,
            provider.provider_id,
            provider.provider_version,
            idempotency_key="event-backed-claim-score",
            now=START + timedelta(minutes=6),
        )

        with self.database.read() as connection:
            claim = connection.execute(
                """SELECT * FROM performance_scoring_claims
                   WHERE evaluation_id=?""",
                (scored["evaluation"]["id"],),
            ).fetchone()
            self.assertIsNotNone(claim)
            internal_key = performance_scoring_claim_event_key(
                claim["command_hash"]
            )
            events = connection.execute(
                """SELECT * FROM events
                   WHERE event_type='PerformanceScoringClaimed'
                     AND idempotency_key=?""",
                (internal_key,),
            ).fetchall()
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(claim["event_id"], event["event_id"])
            self.assertEqual(event["causation_id"], attempt["id"])
            self.assertEqual(event["correlation_id"], attempt["id"])
            payload = json.loads(event["payload_json"])
            self.assertEqual(payload["claim_id"], claim["id"])
            self.assertEqual(
                payload["caller_idempotency_key"],
                "event-backed-claim-score",
            )
            self.assertEqual(payload["command_hash"], claim["command_hash"])
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM performance_scoring_claims
                       WHERE event_id=?""",
                    (event["event_id"],),
                ).fetchone()["n"],
                1,
            )
        listed = self.ledger.list_scoring_claims(
            attempt_id=attempt["id"], status="completed"
        )
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], claim["id"])
        self.assertEqual(listed[0]["status"], "completed")
        self.assertFalse(listed[0]["automatic_retry_allowed"])
        self.assertTrue(listed[0]["caller_idempotency_key_present"])
        self.assertNotIn("idempotency_key", listed[0])
        self.assertEqual(provider.calls, 1)
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_deleted_completed_claim_cannot_repeat_callback_and_replay_repairs_copy(
        self,
    ) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.9,
                    outcome_code="deleted_completed_claim_probe",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )

        class CountingProvider(SyntheticDeterministicProvider):
            calls = 0

            def score(self, request):
                self.calls += 1
                return super().score(request)

        provider = CountingProvider(
            imported,
            provider_id="synthetic.deleted-completed-claim-scorer",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        completed = self.ledger.score_attempt(
            attempt["id"],
            registry,
            provider.provider_id,
            provider.provider_version,
            idempotency_key="deleted-completed-claim",
            now=START + timedelta(minutes=6),
        )
        with self.database.read() as connection:
            claim = dict(
                connection.execute(
                    """SELECT * FROM performance_scoring_claims
                       WHERE evaluation_id=?""",
                    (completed["evaluation"]["id"],),
                ).fetchone()
            )

        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_scoring_claims_no_delete"
            )
            connection.execute(
                "DELETE FROM performance_scoring_claims WHERE id=?",
                (claim["id"],),
            )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "PerformanceScoringClaimed has no scoring claim projection"
                in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )

        same_key = self.ledger.score_attempt(
            attempt["id"],
            registry,
            provider.provider_id,
            provider.provider_version,
            idempotency_key="deleted-completed-claim",
            now=START + timedelta(minutes=7),
        )
        self.assertTrue(same_key["idempotent_replay"])
        self.assertEqual(same_key["evaluation"], completed["evaluation"])
        for retry_key in ("deleted-completed-switched", None):
            with self.subTest(retry_key=retry_key):
                with self.assertRaisesRegex(
                    ConflictError, "committed in event history"
                ):
                    self.ledger.score_attempt(
                        attempt["id"],
                        registry,
                        provider.provider_id,
                        provider.provider_version,
                        idempotency_key=retry_key,
                        now=START + timedelta(minutes=7),
                    )
        self.assertEqual(provider.calls, 1)

        target = Path(self.tempdir.name) / "deleted-claim-rebuilt.db"
        rebuilt_report = ProjectionReplay(self.database).rebuild_copy(
            "performance-learner", target
        )
        self.assertTrue(rebuilt_report["ok"], rebuilt_report["errors"])
        rebuilt = Database(target, read_only=True)
        with rebuilt.read() as connection:
            restored = connection.execute(
                "SELECT * FROM performance_scoring_claims WHERE id=?",
                (claim["id"],),
            ).fetchone()
            self.assertIsNotNone(restored)
            self.assertEqual(restored["event_id"], claim["event_id"])
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE event_id=?
                         AND event_type='PerformanceScoringClaimed'
                         AND idempotency_key=?""",
                    (
                        restored["event_id"],
                        performance_scoring_claim_event_key(
                            restored["command_hash"]
                        ),
                    ),
                ).fetchone()["n"],
                1,
            )
        self.assertTrue(rebuilt.verify_integrity()["ok"])
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n FROM performance_scoring_claims
                       WHERE id=?""",
                    (claim["id"],),
                ).fetchone()["n"],
                0,
            )

    def test_deleted_unfinished_claim_event_still_blocks_every_retry(self) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.INVALID,
                    score=None,
                    outcome_code="deleted_unfinished_claim_probe",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )

        class FailingProvider(SyntheticDeterministicProvider):
            calls = 0

            def score(self, request):
                self.calls += 1
                raise RuntimeError("deleted unfinished claim provider probe")

        provider = FailingProvider(
            imported,
            provider_id="synthetic.deleted-unfinished-claim-scorer",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        with self.assertRaisesRegex(ValidationError, "failed safely"):
            self.ledger.score_attempt(
                attempt["id"],
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="deleted-unfinished-claim",
                now=START + timedelta(minutes=6),
            )
        with self.database.read() as connection:
            claim = dict(
                connection.execute(
                    """SELECT * FROM performance_scoring_claims
                       WHERE idempotency_key='deleted-unfinished-claim'"""
                ).fetchone()
            )
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_scoring_claims_no_delete"
            )
            connection.execute(
                "DELETE FROM performance_scoring_claims WHERE id=?",
                (claim["id"],),
            )

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "PerformanceScoringClaimed has no scoring claim projection"
                in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        for retry_key in (
            "deleted-unfinished-claim",
            "deleted-unfinished-switched",
            None,
        ):
            with self.subTest(retry_key=retry_key):
                with self.assertRaisesRegex(
                    ConflictError, "committed in event history"
                ):
                    self.ledger.score_attempt(
                        attempt["id"],
                        registry,
                        provider.provider_id,
                        provider.provider_version,
                        idempotency_key=retry_key,
                        now=START + timedelta(minutes=7),
                    )
        self.assertEqual(provider.calls, 1)

    def test_integrity_recomputes_scoring_claim_commitments(self) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.7,
                    outcome_code="claim_integrity_probe",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )
        provider = SyntheticDeterministicProvider(
            imported,
            provider_id="synthetic.claim-integrity-scorer",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        self.ledger.score_attempt(
            attempt["id"],
            registry,
            provider.provider_id,
            provider.provider_version,
            idempotency_key="claim-integrity-score",
            now=START + timedelta(minutes=6),
        )

        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_scoring_claims_no_update"
            )
            connection.execute(
                """UPDATE performance_scoring_claims
                   SET provider_version='forged-version'
                   WHERE idempotency_key='claim-integrity-score'"""
            )

        report = self.database.verify_integrity()
        self.assertFalse(report["ok"])
        self.assertTrue(
            any(
                "performance scoring claim " in error
                and (
                    "command commitment mismatch" in error
                    or "completion event does not match" in error
                )
                for error in report["errors"]
            ),
            report["errors"],
        )

    def test_task_release_and_ledger_rows_are_immutable(self) -> None:
        attempt = self.start()
        with self.assertRaisesRegex(Exception, "immutable"):
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE performance_tasks SET definition_json='{}'"
                )
        with self.assertRaisesRegex(Exception, "immutable"):
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE performance_attempts SET task_digest=? WHERE id=?",
                    (_D0, attempt["id"]),
                )

    def test_quarantined_task_cannot_be_started(self) -> None:
        quarantined = PerformanceTaskRelease(
            title="Quarantined task fixture",
            corpus_release_id=self.corpus_release_id,
            review=self.release.review,
            tasks=(("quarantined", self.task),),
        )
        report = self.ledger.publish_release(
            quarantined, now=START + timedelta(minutes=1)
        )
        with self.assertRaisesRegex(Exception, "No serviceable release"):
            self.ledger.start_attempt(
                self.session["id"],
                self.task.id,
                task_version=self.task.version,
                task_release_id=report["release_id"],
                now=START + timedelta(minutes=2),
            )

    def test_release_rejects_model_review_and_source_manifest_drift(self) -> None:
        with self.assertRaisesRegex(ValidationError, "human review"):
            TaskReleaseReview(
                reviewer_kind="model",
                reviewer_id="model_reviewer",
                reviewed_at=START.isoformat(),
                independent_of_author=True,
                attestation_digest=_D2,
            )
        future_review = TaskReleaseReview(
            reviewer_kind="human",
            reviewer_id="reviewer_future_fixture",
            reviewed_at=(START + timedelta(days=1)).isoformat(),
            independent_of_author=True,
            attestation_digest=_D2,
        )
        with self.assertRaisesRegex(ValidationError, "before its review"):
            self.ledger.publish_release(
                PerformanceTaskRelease(
                    title="Future review fixture",
                    corpus_release_id=self.corpus_release_id,
                    review=future_review,
                    tasks=(("pilot", self.task),),
                ),
                now=START,
            )
        drifted = LearningTask.from_terms(
            {
                **self.task.terms(),
                "source_manifests": [
                    {
                        "source_id": "src_vaswani_attention_2017",
                        "provenance_digest": _D3,
                    }
                ],
            }
        )
        release = PerformanceTaskRelease(
            title="Drifted source fixture",
            corpus_release_id=self.corpus_release_id,
            review=self.release.review,
            tasks=(("pilot", drifted),),
        )
        with self.assertRaisesRegex(ValidationError, "source manifest"):
            self.ledger.publish_release(release, now=START)

    def test_release_validates_objective_membership_and_primary_concept(
        self,
    ) -> None:
        def task_with_objective(task_id: str, objective_id: str) -> LearningTask:
            terms = self.task.terms()
            terms["id"] = task_id
            terms["criteria"][0]["objective_weights"] = [[objective_id, 1.0]]
            return LearningTask.from_terms(terms)

        valid = task_with_objective(
            "task_attention_objective_bound",
            "lo_causal_visibility",
        )
        valid_release = PerformanceTaskRelease(
            title="Objective-bound task fixture",
            corpus_release_id=self.corpus_release_id,
            review=self.release.review,
            tasks=(("pilot", valid),),
        )
        self.assertFalse(
            self.ledger.publish_release(
                valid_release, now=START + timedelta(seconds=1)
            )["idempotent_replay"]
        )

        mismatch = task_with_objective(
            "task_attention_objective_mismatch",
            "lo_attention_value_routing",
        )
        with self.assertRaisesRegex(
            ValidationError, "primary concept.*outside"
        ):
            self.ledger.publish_release(
                PerformanceTaskRelease(
                    title="Objective mismatch fixture",
                    corpus_release_id=self.corpus_release_id,
                    review=self.release.review,
                    tasks=(("pilot", mismatch),),
                ),
                now=START + timedelta(seconds=2),
            )

        absent = task_with_objective(
            "task_attention_objective_absent",
            "objective_not_in_release",
        )
        with self.assertRaisesRegex(ValidationError, "outside its release"):
            self.ledger.publish_release(
                PerformanceTaskRelease(
                    title="Absent objective fixture",
                    corpus_release_id=self.corpus_release_id,
                    review=self.release.review,
                    tasks=(("pilot", absent),),
                ),
                now=START + timedelta(seconds=3),
            )

    def test_release_rejects_misconception_outside_criterion_binding(
        self,
    ) -> None:
        terms = self.task.terms()
        terms["id"] = "task_mask_with_unrelated_misconception"
        terms["criteria"][0]["objective_weights"] = [
            ["lo_causal_visibility", 1.0]
        ]
        terms["criteria"][0]["misconception_ids"] = [
            "m_attention_is_hard_selection"
        ]
        mismatched = LearningTask.from_terms(terms)

        with self.assertRaisesRegex(
            ValidationError, "misconception.*not mapped.*objectives"
        ):
            self.ledger.publish_release(
                PerformanceTaskRelease(
                    title="Misbound misconception fixture",
                    corpus_release_id=self.corpus_release_id,
                    review=self.release.review,
                    tasks=(("pilot", mismatched),),
                ),
                now=START + timedelta(seconds=4),
            )

    def test_release_rejects_quarantine_only_objective_misconception_binding(
        self,
    ) -> None:
        objective_id = "lo_agent_state_reconciliation"
        misconception_id = "m_agent_any_required_condition_suffices"
        parsed = read_and_parse(CORPUS, include_catalog=True)
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT membership.question_id, membership.status,
                          revocation.question_id IS NOT NULL AS revoked
                   FROM release_option_objectives mapping
                   JOIN release_questions membership
                     ON membership.release_id = mapping.release_id
                    AND membership.question_id = mapping.question_id
                   JOIN options option
                     ON option.question_id = mapping.question_id
                    AND option.option_id = mapping.option_id
                   LEFT JOIN question_revocations revocation
                     ON revocation.question_id = mapping.question_id
                   WHERE mapping.release_id = ?
                     AND mapping.objective_id = ?
                     AND option.misconception_id = ?
                   ORDER BY membership.question_id""",
                (
                    self.corpus_release_id,
                    objective_id,
                    misconception_id,
                ),
            ).fetchall()
        self.assertGreater(len(rows), 0)
        mapping_question_ids = {row["question_id"] for row in rows}
        isolated_questions = tuple(
            replace(question, status=QuestionStatus.QUARANTINED)
            if question.id in mapping_question_ids
            else question
            for question in parsed[4]
        )
        isolated_release = self.database.import_corpus(
            parsed[0],
            parsed[1],
            parsed[2],
            parsed[3],
            isolated_questions,
            parsed[5],
            parsed[6],
        )["release_id"]
        with self.database.read() as connection:
            isolated_rows = connection.execute(
                """SELECT membership.question_id, membership.status,
                          revocation.question_id IS NOT NULL AS revoked
                   FROM release_option_objectives mapping
                   JOIN release_questions membership
                     ON membership.release_id = mapping.release_id
                    AND membership.question_id = mapping.question_id
                   JOIN options option
                     ON option.question_id = mapping.question_id
                    AND option.option_id = mapping.option_id
                   LEFT JOIN question_revocations revocation
                     ON revocation.question_id = mapping.question_id
                   WHERE mapping.release_id = ?
                     AND mapping.objective_id = ?
                     AND option.misconception_id = ?
                   ORDER BY membership.question_id""",
                (isolated_release, objective_id, misconception_id),
            ).fetchall()
        self.assertEqual(
            {row["question_id"] for row in isolated_rows},
            mapping_question_ids,
        )
        self.assertTrue(
            all(
                row["status"] == QuestionStatus.QUARANTINED.value
                and not row["revoked"]
                for row in isolated_rows
            )
        )

        terms = self.task.terms()
        terms["id"] = "task_quarantine_only_misconception"
        terms["criteria"][0]["concept_weights"] = [
            ["c_agent_observation_loop", 1.0]
        ]
        terms["criteria"][0]["objective_weights"] = [
            [objective_id, 1.0]
        ]
        terms["criteria"][0]["misconception_ids"] = [misconception_id]
        task = LearningTask.from_terms(terms)

        quarantined = self.ledger.publish_release(
            PerformanceTaskRelease(
                title="Quarantine-only misconception draft fixture",
                corpus_release_id=isolated_release,
                review=self.release.review,
                tasks=(("quarantined", task),),
            ),
            now=START + timedelta(seconds=4),
        )
        self.assertEqual(quarantined["status_counts"]["quarantined"], 1)
        self.assertTrue(self.database.verify_integrity()["ok"])

        with self.assertRaisesRegex(
            ValidationError, "misconception.*not mapped.*objectives"
        ):
            self.ledger.publish_release(
                PerformanceTaskRelease(
                    title="Quarantine-only misconception fixture",
                    corpus_release_id=isolated_release,
                    review=self.release.review,
                    tasks=(("pilot", task),),
                ),
                now=START + timedelta(seconds=5),
            )

    def test_release_rejects_already_revoked_misconception_binding(
        self,
    ) -> None:
        objective_id = "lo_attention_value_routing"
        misconception_id = "m_attention_is_hard_selection"
        terms = self.task.terms()
        terms["id"] = "task_revoked_misconception_binding"
        terms["criteria"][0]["concept_weights"] = [["c_attention", 1.0]]
        terms["criteria"][0]["objective_weights"] = [
            [objective_id, 1.0]
        ]
        terms["criteria"][0]["misconception_ids"] = [misconception_id]
        task = LearningTask.from_terms(terms)
        with self.database.read() as connection:
            supporting_rows = connection.execute(
                """SELECT DISTINCT mapping.question_id
                   FROM release_option_objectives mapping
                   JOIN release_questions membership
                     ON membership.release_id = mapping.release_id
                    AND membership.question_id = mapping.question_id
                   JOIN options option
                     ON option.question_id = mapping.question_id
                    AND option.option_id = mapping.option_id
                   WHERE mapping.release_id = ?
                     AND membership.status IN ('approved', 'calibrated')
                     AND mapping.objective_id = ?
                     AND option.misconception_id = ?
                   ORDER BY mapping.question_id""",
                (
                    self.corpus_release_id,
                    objective_id,
                    misconception_id,
                ),
            ).fetchall()
        self.assertGreater(len(supporting_rows), 0)

        for index, row in enumerate(supporting_rows):
            self.database.revoke_question(
                row["question_id"],
                "Pre-publication productive-task binding revocation fixture.",
                idempotency_key=f"task-prepublish-revocation:{index}",
            )
        with self.assertRaisesRegex(
            ValidationError, "misconception.*not mapped.*objectives"
        ):
            self.ledger.publish_release(
                PerformanceTaskRelease(
                    title="Already-revoked misconception binding fixture",
                    corpus_release_id=self.corpus_release_id,
                    review=self.release.review,
                    tasks=(("pilot", task),),
                ),
                now=START + timedelta(seconds=6),
            )

    def test_revocation_withdraws_new_serviceability_not_history(
        self,
    ) -> None:
        objective_id = "lo_attention_value_routing"
        misconception_id = "m_attention_is_hard_selection"
        terms = self.task.terms()
        terms["id"] = "task_later_revoked_misconception_binding"
        terms["criteria"][0]["concept_weights"] = [["c_attention", 1.0]]
        terms["criteria"][0]["objective_weights"] = [
            [objective_id, 1.0]
        ]
        terms["criteria"][0]["misconception_ids"] = [misconception_id]
        task = LearningTask.from_terms(terms)
        published = self.ledger.publish_release(
            PerformanceTaskRelease(
                title="Later-revoked misconception binding fixture",
                corpus_release_id=self.corpus_release_id,
                review=self.release.review,
                tasks=(("pilot", task),),
            ),
            now=START,
        )
        recommendations_before = recommend_performance_tasks(
            self.database,
            self.session["id"],
            limit=50,
            now=START + timedelta(seconds=1),
        )
        self.assertIn(
            task.id,
            {
                item["task_id"]
                for item in recommendations_before["recommendations"]
            },
        )

        self.engine.create_learner("fresh-after-task-revocation")
        fresh_session = self.engine.start_session(
            "fresh-after-task-revocation",
            "t_transformers",
            seed=1403,
            now=START,
        )
        started = self.ledger.start_attempt(
            self.session["id"],
            task.id,
            task_version=task.version,
            task_release_id=published["release_id"],
            idempotency_key="task-revocation-started-before",
            now=START + timedelta(minutes=1),
        )
        with self.database.read() as connection:
            supporting_rows = connection.execute(
                """SELECT DISTINCT mapping.question_id
                   FROM release_option_objectives mapping
                   JOIN release_questions membership
                     ON membership.release_id = mapping.release_id
                    AND membership.question_id = mapping.question_id
                   JOIN options option
                     ON option.question_id = mapping.question_id
                    AND option.option_id = mapping.option_id
                   WHERE mapping.release_id = ?
                     AND membership.status IN ('approved', 'calibrated')
                     AND mapping.objective_id = ?
                     AND option.misconception_id = ?
                   ORDER BY mapping.question_id""",
                (
                    self.corpus_release_id,
                    objective_id,
                    misconception_id,
                ),
            ).fetchall()
        self.assertGreater(len(supporting_rows), 0)
        for index, row in enumerate(supporting_rows):
            self.database.revoke_question(
                row["question_id"],
                "Post-publication productive-task binding revocation fixture.",
                idempotency_key=f"task-postpublish-revocation:{index}",
            )

        replay = self.ledger.start_attempt(
            self.session["id"],
            task.id,
            task_version=task.version,
            task_release_id=published["release_id"],
            idempotency_key="task-revocation-started-before",
            now=START + timedelta(minutes=2),
        )
        self.assertEqual(replay["id"], started["id"])
        self.assertTrue(replay["idempotent_replay"])
        recommendations_after = recommend_performance_tasks(
            self.database,
            fresh_session["id"],
            limit=50,
            now=START + timedelta(minutes=2),
        )
        self.assertNotIn(
            task.id,
            {
                item["task_id"]
                for item in recommendations_after["recommendations"]
            },
        )
        with self.assertRaisesRegex(
            NotFoundError, "live diagnostic mapping was withdrawn"
        ):
            self.ledger.start_attempt(
                fresh_session["id"],
                task.id,
                task_version=task.version,
                task_release_id=published["release_id"],
                idempotency_key="task-revocation-fresh-start",
                now=START + timedelta(minutes=3),
            )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_direct_imported_deterministic_claim_remains_shadow(self) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.9,
                    outcome_code="claimed_pass",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )
        result = self.ledger.import_evaluation(
            attempt["id"],
            imported,
            provider_id="claimed_deterministic",
            provider_version="v1",
            declared_kind=ScorerKind.DETERMINISTIC,
            now=START + timedelta(minutes=6),
        )
        decision = result["authority"]["normalized_result"]["decisions"][0]
        self.assertEqual(decision["effective_kind"], "imported")
        self.assertEqual(result["shadow_evidence"]["total_evidence_weight"], 0.0)
        self.assertFalse(result["projection_applied"])

    def test_direct_import_rejects_outer_schema_subclass_before_commit(
        self,
    ) -> None:
        class DigestOverride(ImportedEvaluation):
            @property
            def digest(self) -> str:
                return "f" * 64

        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = DigestOverride(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.9,
                    outcome_code="outer_schema_subclass",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )
        before = self.performance_event_and_projection_counts()

        with self.assertRaisesRegex(
            ValidationError,
            "exact ImportedEvaluation",
        ):
            self.ledger.import_evaluation(
                attempt["id"],
                imported,
                provider_id="outer_schema_subclass",
                provider_version="v1",
                now=START + timedelta(minutes=6),
            )

        self.assertEqual(
            self.performance_event_and_projection_counts(),
            before,
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_direct_import_rejects_nested_schema_subclass_before_commit(
        self,
    ) -> None:
        class NestedCriterion(ImportedCriterionResult):
            pass

        attempt = self.start()
        submitted = self.submit(attempt["id"])
        exact_criterion = ImportedCriterionResult(
            criterion_id="criterion_mask_invariant",
            status=EvaluationStatus.VALID,
            score=0.9,
            outcome_code="nested_schema_subclass",
            phase=ActionPhase.UNASSISTED,
            source_action_ids=(submitted["id"],),
        )
        imported = ImportedEvaluation(
            criteria=(exact_criterion,)
        )
        nested_criterion = NestedCriterion(
            criterion_id=exact_criterion.criterion_id,
            status=exact_criterion.status,
            score=exact_criterion.score,
            outcome_code=exact_criterion.outcome_code,
            phase=exact_criterion.phase,
            source_action_ids=exact_criterion.source_action_ids,
            attestation_digest=exact_criterion.attestation_digest,
            misconception_ids=exact_criterion.misconception_ids,
            reliability=exact_criterion.reliability,
        )
        object.__setattr__(imported, "criteria", (nested_criterion,))
        before = self.performance_event_and_projection_counts()

        with self.assertRaisesRegex(
            ValidationError,
            "exact ImportedCriterionResult",
        ):
            self.ledger.import_evaluation(
                attempt["id"],
                imported,
                provider_id="nested_schema_subclass",
                provider_version="v1",
                now=START + timedelta(minutes=6),
            )

        self.assertEqual(
            self.performance_event_and_projection_counts(),
            before,
        )
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_record_action_freezes_stateful_mapping_once_before_commit(
        self,
    ) -> None:
        class StatefulPayload(Mapping[str, object]):
            def __init__(self) -> None:
                self.read_count = 0

            def __getitem__(self, key: str) -> object:
                if key != "reason_code":
                    raise KeyError(key)
                self.read_count += 1
                return (
                    "first_reason"
                    if self.read_count == 1
                    else "changed_reason"
                )

            def __iter__(self):
                return iter(("reason_code",))

            def __len__(self) -> int:
                return 1

        attempt = self.start()
        payload = StatefulPayload()
        before = self.performance_event_and_projection_counts()

        action = self.ledger.record_action(
            attempt["id"],
            ActionKind.ABANDONED.value,
            payload,
            idempotency_key="stateful-action-payload",
            now=START + timedelta(minutes=2),
        )

        self.assertEqual(payload.read_count, 1)
        self.assertEqual(
            action["payload"],
            {"reason_code": "first_reason"},
        )
        after = self.performance_event_and_projection_counts()
        self.assertEqual(after["events"], before["events"] + 1)
        self.assertEqual(
            after["performance_actions"],
            before["performance_actions"] + 1,
        )
        for table in set(before) - {"events", "performance_actions"}:
            self.assertEqual(after[table], before[table], table)
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_integrity_accepts_canonical_request_for_author_ordered_criteria(
        self,
    ) -> None:
        ordered_task = LearningTask(
            id="task_author_ordered_criteria",
            version=1,
            family_id="family_author_ordered_criteria",
            title="Preserve intentional rubric presentation order",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_zeta_diagnosis",
                    name="Diagnose the mask failure",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_causal_masking", 1.0),),
                    dependence_group="ordered_criteria",
                    evidence_cap=0.4,
                    dependence_cap=0.8,
                ),
                RubricCriterion(
                    id="criterion_alpha_repair",
                    name="Repair the mask failure",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_causal_masking", 1.0),),
                    dependence_group="ordered_criteria",
                    evidence_cap=0.4,
                    dependence_cap=0.8,
                ),
            ),
            instructions=(
                "Diagnose the pinned causal-mask failure before describing "
                "the repair."
            ),
            source_manifests=self.task.source_manifests,
            administration_id="admin_author_ordered_criteria",
            administration_manifest_digest=_D0,
            stimulus_id="stimulus_author_ordered_criteria",
            stimulus_digest=_D1,
        )
        release = PerformanceTaskRelease(
            title="Author-ordered rubric fixture",
            corpus_release_id=self.corpus_release_id,
            review=self.release.review,
            tasks=(("pilot", ordered_task),),
        )
        published = self.ledger.publish_release(
            release, now=START + timedelta(seconds=1)
        )
        attempt = self.ledger.start_attempt(
            self.session["id"],
            ordered_task.id,
            task_version=ordered_task.version,
            task_release_id=published["release_id"],
            now=START + timedelta(minutes=1),
        )
        submitted = self.ledger.record_action(
            attempt["id"],
            "submitted",
            {"submission_digest": _D3},
            now=START + timedelta(minutes=2),
        )
        self.ledger.import_evaluation(
            attempt["id"],
            ImportedEvaluation(
                criteria=(
                    ImportedCriterionResult(
                        criterion_id="criterion_zeta_diagnosis",
                        status=EvaluationStatus.VALID,
                        score=0.7,
                        outcome_code="diagnosis_observed",
                        phase=ActionPhase.UNASSISTED,
                        source_action_ids=(submitted["id"],),
                    ),
                    ImportedCriterionResult(
                        criterion_id="criterion_alpha_repair",
                        status=EvaluationStatus.VALID,
                        score=0.6,
                        outcome_code="repair_observed",
                        phase=ActionPhase.UNASSISTED,
                        source_action_ids=(submitted["id"],),
                    ),
                )
            ),
            provider_id="ordered_criteria_import",
            provider_version="v1",
            now=START + timedelta(minutes=3),
        )

        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_registered_scorer_rejects_manually_asserted_check_source(
        self,
    ) -> None:
        authority = ProviderAuthorityBinding(
            provider_id="checks.partial-rubric",
            provider_version="v1",
            declared_kind=ScorerKind.DETERMINISTIC,
            authority_id="authority.partial-rubric",
            authority_manifest_digest=_D2,
            check_set_manifests=(("partial_checks_v1", _D3),),
            verified=True,
        )
        partial_task = LearningTask(
            id="task_partial_registered_scorer",
            version=1,
            family_id="family_partial_registered_scorer",
            title="Score implementation and explanation independently",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_checked_behavior",
                    name="Checked behavior",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_causal_masking", 1.0),),
                    dependence_group="partial_checked_behavior",
                    evidence_cap=0.5,
                    dependence_cap=0.5,
                ),
                RubricCriterion(
                    id="criterion_unreviewed_explanation",
                    name="Explanation quality",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_causal_masking", 1.0),),
                    dependence_group="partial_explanation",
                    evidence_cap=0.5,
                    dependence_cap=0.5,
                ),
            ),
            instructions=(
                "Repair the pinned mask behavior and explain the causal "
                "visibility invariant."
            ),
            source_manifests=self.task.source_manifests,
            administration_id="admin_partial_registered_scorer",
            administration_manifest_digest=_D0,
            stimulus_id="stimulus_partial_registered_scorer",
            stimulus_digest=_D1,
            scorer_contracts=(
                ScorerContract(
                    kind=ScorerKind.DETERMINISTIC,
                    scorer_id=authority.provider_id,
                    scorer_version=authority.provider_version,
                    authority_id=authority.authority_id,
                    authority_manifest_digest=(
                        authority.authority_manifest_digest
                    ),
                    criterion_ids=("criterion_checked_behavior",),
                    evidence_action_kinds=(ActionKind.CHECK_RUN,),
                    check_set_manifests=authority.check_set_manifests,
                ),
            ),
        )
        published = self.ledger.publish_release(
            PerformanceTaskRelease(
                title="Partial registered scorer fixture",
                corpus_release_id=self.corpus_release_id,
                review=self.release.review,
                tasks=(("pilot", partial_task),),
            ),
            now=START + timedelta(seconds=1),
        )
        attempt = self.ledger.start_attempt(
            self.session["id"],
            partial_task.id,
            task_version=partial_task.version,
            task_release_id=published["release_id"],
            now=START + timedelta(minutes=1),
        )
        check = self.ledger.record_action(
            attempt["id"],
            "check_run",
            {
                "check_set_id": "partial_checks_v1",
                "passed": 4,
                "failed": 0,
                "errored": 0,
                "skipped": 0,
                "result_digest": _D3,
            },
            now=START + timedelta(minutes=2),
        )
        self.ledger.record_action(
            attempt["id"],
            "submitted",
            {"submission_digest": _D3},
            now=START + timedelta(minutes=3),
        )
        observation = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_checked_behavior",
                    status=EvaluationStatus.VALID,
                    score=1.0,
                    outcome_code="checks_passed",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(check["id"],),
                ),
            )
        )

        class PartialProvider:
            provider_id = "checks.partial-rubric"
            provider_version = "v1"
            declared_kind = ScorerKind.DETERMINISTIC
            synthetic = False
            calls = 0

            def score(self, request):
                self.calls += 1
                return observation

        registry = ScoringProviderRegistry()
        provider = PartialProvider()
        registry.register(provider, authority)
        with self.assertRaisesRegex(ValidationError, "manually asserted"):
            self.ledger.score_attempt(
                attempt["id"],
                registry,
                authority.provider_id,
                authority.provider_version,
                now=START + timedelta(minutes=4),
            )
        self.assertEqual(provider.calls, 1)
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    """SELECT COUNT(*) AS n FROM task_evaluations
                       WHERE attempt_id=?""",
                    (attempt["id"],),
                ).fetchone()["n"],
                0,
            )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_lifecycle_and_semantic_time_boundaries_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "cannot start before its session"
        ):
            self.ledger.start_attempt(
                self.session["id"],
                self.task.id,
                task_version=self.task.version,
                task_release_id=self.release_report["release_id"],
                now=datetime(2000, 1, 1, tzinfo=timezone.utc),
            )
        with self.assertRaisesRegex(
            ValidationError, "cannot start before its release"
        ):
            self.ledger.start_attempt(
                self.session["id"],
                self.task.id,
                task_version=self.task.version,
                task_release_id=self.release_report["release_id"],
                now=self.release_time - timedelta(microseconds=1),
            )

        attempt = self.start()
        with self.assertRaisesRegex(
            ValidationError, "require a submitted checkpoint"
        ):
            self.ledger.record_action(
                attempt["id"],
                "feedback_shown",
                {"feedback_digest": _D2},
                phase="post_feedback",
                now=START + timedelta(minutes=2),
            )
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.5,
                    outcome_code="out_of_order_fixture",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )
        with self.assertRaisesRegex(
            ValidationError, "cannot occur before.*submitted"
        ):
            self.ledger.import_evaluation(
                attempt["id"],
                imported,
                provider_id="imported_fixture",
                provider_version="v1",
                now=START + timedelta(minutes=4),
            )
        with self.assertRaisesRegex(ValidationError, "failed safely"):
            self.ledger.score_attempt(
                attempt["id"],
                ScoringProviderRegistry(),
                "provider_missing",
                "v1",
                now=START + timedelta(minutes=6),
            )
        self.engine.end_session(
            self.session["id"], now=START + timedelta(minutes=7)
        )
        with self.assertRaisesRegex(ConflictError, "Session .* is completed"):
            self.ledger.record_action(
                attempt["id"],
                "feedback_shown",
                {"feedback_digest": _D2},
                phase="post_feedback",
                now=START + timedelta(minutes=8),
            )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_trace_corruption_is_isolated_to_its_learner(self) -> None:
        primary = self.start(key="start-isolation-primary")
        self.engine.create_learner(
            "performance-isolation-peer",
            "Performance Isolation Peer",
        )
        peer_session = self.engine.start_session(
            "performance-isolation-peer",
            "t_transformers",
            seed=9412,
            now=START,
        )
        peer = self.ledger.start_attempt(
            peer_session["id"],
            self.task.id,
            task_version=self.task.version,
            task_release_id=self.release_report["release_id"],
            idempotency_key="start-isolation-peer",
            now=START + timedelta(minutes=1),
        )
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER performance_actions_no_update")
            connection.execute(
                """UPDATE performance_actions
                   SET elapsed_ms=elapsed_ms + 1
                   WHERE attempt_id=? AND sequence=0""",
                (peer["id"],),
            )

        self.assertEqual(len(self.ledger.list_actions(primary["id"])), 1)
        self.assertEqual(
            self.ledger.report(primary["id"])["id"],
            primary["id"],
        )
        recommendation = recommend_performance_tasks(
            self.database,
            self.session["id"],
            now=START + timedelta(minutes=2),
        )
        self.assertEqual(
            recommendation["learner_id"],
            "performance-learner",
        )
        with self.assertRaisesRegex(
            ValidationError,
            "immutable event derivation",
        ):
            self.ledger.list_actions(peer["id"])

    def test_learner_stream_cannot_hide_mismatched_event_owner(self) -> None:
        attempt = self.start(key="start-stream-owner")
        self.engine.create_learner(
            "performance-stream-intruder",
            "Performance Stream Intruder",
        )
        peer_session = self.engine.start_session(
            "performance-stream-intruder",
            "t_transformers",
            seed=9413,
            now=START,
        )
        peer = self.ledger.start_attempt(
            peer_session["id"],
            self.task.id,
            task_version=self.task.version,
            task_release_id=self.release_report["release_id"],
            idempotency_key="start-stream-intruder-peer",
            now=START + timedelta(minutes=1),
        )
        with self.database.transaction() as connection:
            self.database.append_event(
                connection,
                stream_id="learner:performance-learner",
                event_type="PerformanceActionRecorded",
                schema_version=1,
                payload={},
                metadata={},
                learner_id="performance-stream-intruder",
                session_id=self.session["id"],
                correlation_id="rogue-correlation",
                causation_id="rogue-causation",
                occurred_at=START + timedelta(minutes=1, seconds=30),
            )

        for operation in (
            lambda: self.ledger.report(attempt["id"]),
            lambda: self.ledger.record_action(
                attempt["id"],
                "answer_revised",
                {"answer_digest": _D3},
                idempotency_key="after-stream-intruder",
                now=START + timedelta(minutes=2),
            ),
        ):
            with self.assertRaisesRegex(
                ValidationError,
                "immutable events",
            ):
                operation()
        self.assertEqual(
            self.ledger.report(peer["id"])["learner_id"],
            "performance-stream-intruder",
        )
        self.assertEqual(len(self.ledger.list_actions(peer["id"])), 1)
        self.assertFalse(self.database.verify_integrity()["ok"])

    def test_registered_evaluation_requires_claim_or_legacy_exemption(
        self,
    ) -> None:
        attempt = self.start(key="start-claimless-provider")
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.7,
                    outcome_code="claimless_provider_probe",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )
        provider = SyntheticDeterministicProvider(
            imported,
            provider_id="synthetic.claimless-provider",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        self.ledger.score_attempt(
            attempt["id"],
            registry,
            provider.provider_id,
            provider.provider_version,
            idempotency_key="claimless-provider-score",
            now=START + timedelta(minutes=6),
        )
        with self.database.transaction() as connection:
            claim = connection.execute(
                """SELECT id, event_id FROM performance_scoring_claims
                   WHERE attempt_id=?""",
                (attempt["id"],),
            ).fetchone()
            connection.execute(
                "DROP TRIGGER performance_scoring_claims_no_delete"
            )
            connection.execute("DROP TRIGGER events_no_delete")
            connection.execute(
                "DELETE FROM performance_scoring_claims WHERE id=?",
                (claim["id"],),
            )
            connection.execute(
                "DELETE FROM events WHERE event_id=?",
                (claim["event_id"],),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "exactly one matching scoring claim",
            ):
                require_performance_projection_consistency(
                    connection,
                    learner_id="performance-learner",
                )

    def test_recovered_import_requires_completed_reconciliation(self) -> None:
        attempt, _submitted, claim, _provider, imported = (
            self.failed_scoring_claim(
                provider_id="synthetic.missing-reconciliation",
                key="score-for-missing-reconciliation",
            )
        )
        registry, adapter = self.reconciliation_registry(
            claim,
            outcome=ReconciliationOutcome.COMPLETED,
            imported=imported,
            observed_at=START + timedelta(minutes=8),
            completed_at=START + timedelta(minutes=7, seconds=30),
        )
        result = self.ledger.reconcile_scoring_claim(
            claim["id"],
            registry,
            adapter.reconciler_id,
            adapter.reconciler_version,
            idempotency_key="complete-before-removal",
            now=START + timedelta(minutes=9),
        )
        with self.database.transaction() as connection:
            observation = connection.execute(
                """SELECT id, event_id
                   FROM performance_scoring_reconciliations
                   WHERE id=?""",
                (result["reconciliation_id"],),
            ).fetchone()
            connection.execute(
                "DROP TRIGGER performance_scoring_reconciliations_no_delete"
            )
            connection.execute("DROP TRIGGER events_no_delete")
            connection.execute(
                """DELETE FROM performance_scoring_reconciliations
                   WHERE id=?""",
                (observation["id"],),
            )
            connection.execute(
                "DELETE FROM events WHERE event_id=?",
                (observation["event_id"],),
            )

        with self.database.read() as connection:
            with self.assertRaisesRegex(
                ValidationError,
                "exactly one completed reconciliation",
            ):
                require_performance_projection_consistency(
                    connection,
                    learner_id=attempt["learner_id"],
                )

    def test_integrity_recomputes_shadow_bundle_and_detects_projection_tampering(
        self,
    ) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.6,
                    outcome_code="imported_observation",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )
        self.ledger.import_evaluation(
            attempt["id"],
            imported,
            provider_id="imported_fixture",
            provider_version="v1",
            now=START + timedelta(minutes=6),
        )
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER shadow_evidence_bundles_no_update"
            )
            connection.execute(
                """UPDATE shadow_evidence_bundles
                   SET bundle_json=json_set(bundle_json, '$.evidence_score', 1.0)
                   WHERE attempt_id=?""",
                (attempt["id"],),
            )
        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "deterministic shadow reduction mismatch" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )

    def test_integrity_detects_event_without_shadow_projection(self) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.4,
                    outcome_code="imported_observation",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )
        self.ledger.import_evaluation(
            attempt["id"],
            imported,
            provider_id="imported_fixture",
            provider_version="v1",
            now=START + timedelta(minutes=6),
        )
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER shadow_evidence_bundles_no_delete"
            )
            connection.execute(
                "DELETE FROM shadow_evidence_bundles WHERE attempt_id=?",
                (attempt["id"],),
            )
        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "ShadowEvidenceReduced has no bundle projection" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )

    def test_integrity_recomputes_provider_authority_binding(self) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.4,
                    outcome_code="imported_observation",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )
        result = self.ledger.import_evaluation(
            attempt["id"],
            imported,
            provider_id="imported_fixture",
            provider_version="v1",
            now=START + timedelta(minutes=6),
        )
        evaluation_id = result["evaluation"]["id"]
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER task_evaluations_no_update")
            connection.execute("DROP TRIGGER events_no_update")
            row = connection.execute(
                "SELECT event_id, authority_json FROM task_evaluations WHERE id=?",
                (evaluation_id,),
            ).fetchone()
            authority = json.loads(row["authority_json"])
            authority["normalized_result"]["provider"]["binding_digest"] = _D0
            authority["normalized_result_digest"] = canonical_digest(
                authority["normalized_result"]
            )
            event = connection.execute(
                "SELECT payload_json FROM events WHERE event_id=?",
                (row["event_id"],),
            ).fetchone()
            payload = json.loads(event["payload_json"])
            payload["authority"] = authority
            connection.execute(
                "UPDATE task_evaluations SET authority_json=? WHERE id=?",
                (canonical_json(authority), evaluation_id),
            )
            connection.execute(
                "UPDATE events SET payload_json=? WHERE event_id=?",
                (canonical_json(payload), row["event_id"]),
            )
        with self.database.read() as connection:
            errors = performance_integrity_errors(connection)
        self.assertTrue(
            any(
                "invalid provider binding" in error
                and "binding digest does not match" in error
                for error in errors
            ),
            errors,
        )

    def test_projection_replay_rebuilds_shadow_ledger_from_events(self) -> None:
        attempt = self.start()
        submitted = self.submit(attempt["id"])
        imported = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_mask_invariant",
                    status=EvaluationStatus.VALID,
                    score=0.7,
                    outcome_code="imported_observation",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(submitted["id"],),
                ),
            )
        )
        self.ledger.import_evaluation(
            attempt["id"],
            imported,
            provider_id="imported_fixture",
            provider_version="v1",
            now=START + timedelta(minutes=6),
        )
        clean = ProjectionReplay(self.database).check("performance-learner")
        self.assertTrue(clean["ok"], clean["errors"])
        self.assertTrue(clean["performance_projection_matches_replay"])
        self.assertEqual(clean["reconstructed_performance_attempt_count"], 1)
        self.assertEqual(clean["reconstructed_performance_action_count"], 3)
        self.assertEqual(clean["reconstructed_task_evaluation_count"], 1)
        self.assertEqual(clean["reconstructed_shadow_evidence_bundle_count"], 1)

        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER shadow_evidence_bundles_no_update"
            )
            connection.execute(
                """UPDATE shadow_evidence_bundles
                   SET bundle_json=json_set(bundle_json, '$.evidence_score', 0.99)
                   WHERE attempt_id=?""",
                (attempt["id"],),
            )
        with self.database.read() as connection:
            damaged_bundle = json.loads(
                connection.execute(
                    """SELECT bundle_json FROM shadow_evidence_bundles
                       WHERE attempt_id=?""",
                    (attempt["id"],),
                ).fetchone()["bundle_json"]
            )
        with self.assertRaisesRegex(
            ValidationError,
            "immutable event derivation",
        ):
            self.ledger.report(attempt["id"])
        check = ProjectionReplay(self.database).check("performance-learner")
        self.assertFalse(check["ok"])
        self.assertFalse(check["performance_projection_matches_replay"])

        rebuilt_path = Path(self.tempdir.name) / "performance-rebuilt.db"
        rebuilt = ProjectionReplay(self.database).rebuild_copy(
            "performance-learner", rebuilt_path
        )
        self.assertTrue(rebuilt["ok"], rebuilt["errors"])
        self.assertTrue(rebuilt["source_performance_projection_was_repaired"])
        with self.assertRaisesRegex(
            ValidationError,
            "immutable event derivation",
        ):
            self.ledger.report(attempt["id"])
        rebuilt_database = Database(rebuilt_path, read_only=True)
        self.assertTrue(rebuilt_database.verify_integrity()["ok"])
        rebuilt_report = PerformanceLedger(rebuilt_database).report(attempt["id"])
        self.assertNotEqual(
            rebuilt_report["evaluations"][0]["shadow_evidence"],
            damaged_bundle,
        )

    def test_release_replay_has_stable_shape_and_listing_hides_storage_json(
        self,
    ) -> None:
        replay = self.ledger.publish_release(
            self.release,
            now=self.release_time + timedelta(seconds=1),
        )

        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(
            set(replay),
            {
                "release_id",
                "bundle_hash",
                "bundle_size_bytes",
                "corpus_release_id",
                "release_authority_kind",
                "task_count",
                "status_counts",
                "idempotent_replay",
            },
        )
        self.assertEqual(
            replay["status_counts"],
            self.release_report["status_counts"],
        )
        self.assertEqual(
            replay["release_authority_kind"], "human_review"
        )
        listed = self.ledger.list_releases()
        self.assertNotIn("review_json", listed[0])
        self.assertEqual(
            listed[0]["review"]["reviewer_id"],
            self.release.review.reviewer_id,
        )

    def test_programmatic_release_honors_serialized_size_bound_before_commit(
        self,
    ) -> None:
        release = replace(
            self.release,
            title="Programmatic release size-bound fixture",
        )
        serialized_size = len(
            canonical_json(release.terms()).encode("utf-8")
        )
        with self.database.read() as connection:
            before = {
                table: connection.execute(
                    f"SELECT COUNT(*) AS n FROM {table}"
                ).fetchone()["n"]
                for table in (
                    "performance_tasks",
                    "performance_task_releases",
                    "release_performance_tasks",
                )
            }

        with (
            patch(
                "tsq.performance_ledger.MAX_TASK_RELEASE_BYTES",
                serialized_size - 1,
            ),
            self.assertRaisesRegex(
                ValidationError,
                "Task release exceeds",
            ),
        ):
            self.ledger.publish_release(
                release,
                now=self.release_time + timedelta(seconds=1),
            )

        with self.database.read() as connection:
            after = {
                table: connection.execute(
                    f"SELECT COUNT(*) AS n FROM {table}"
                ).fetchone()["n"]
                for table in before
            }
        self.assertEqual(after, before)
        integrity = self.database.verify_integrity()
        self.assertTrue(integrity["ok"], integrity["errors"])

    def test_cli_imports_inspects_and_recommends_pinned_task_fixture(self) -> None:
        declared_fixture = declared_task_release_fixture(self.corpus_release_id)
        fixture_path = (
            Path(self.tempdir.name) / "declared-productive-task-release.json"
        )
        fixture_path.write_text(
            canonical_json(declared_fixture.terms()) + "\n",
            encoding="utf-8",
        )
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "import",
                    str(fixture_path),
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())
        imported = json.loads(output.getvalue())
        self.assertEqual(imported["status_counts"]["pilot"], 1)
        self.assertEqual(
            imported["corpus_release_id"], self.corpus_release_id
        )

        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "releases",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())
        releases = json.loads(output.getvalue())
        imported_release = next(
            row for row in releases if row["id"] == imported["release_id"]
        )
        self.assertNotIn("review_json", imported_release)
        self.assertEqual(
            imported_release["review"]["reviewer_id"],
            "fixture_independent_reviewer",
        )

        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "show",
                    "task_cli_attention_debug",
                    "--version",
                    "1",
                    "--release",
                    imported["release_id"],
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())
        shown = output.getvalue()
        self.assertIn("lo_causal_visibility", shown)
        self.assertIn(
            "artifact_checkpoint: {artifact_digest:sha256, artifact_kind:id}",
            shown,
        )
        self.assertIn(
            "evidence boundary: shadow only; no mastery or certification update",
            shown,
        )

        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "recommend",
                    "--session",
                    self.session["id"],
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())
        recommendation = json.loads(output.getvalue())
        self.assertIn(
            "task_cli_attention_debug",
            {
                item["task_id"]
                for item in recommendation["recommendations"]
            },
        )
        self.assertTrue(recommendation["selection_boundary"]["read_only"])
        self.assertFalse(
            recommendation["selection_boundary"]["mastery_affected"]
        )

    def test_cli_task_payloads_are_strict_and_utf8_bounded(self) -> None:
        attempt = self.start(key="strict-cli-action")
        duplicate = (
            '{"hint_id":"hint_one","hint_id":"hint_two","level":1}'
        )
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "action",
                    attempt["id"],
                    "hint_requested",
                    "--payload",
                    duplicate,
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("duplicate field 'hint_id'", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())

        invalid_utf8 = Path(self.tempdir.name) / "invalid-payload.json"
        invalid_utf8.write_bytes(b"\xff")
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "action",
                    attempt["id"],
                    "artifact_checkpoint",
                    "--payload-file",
                    str(invalid_utf8),
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertIn("not valid UTF-8", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())
        self.assertEqual(self.ledger.report(attempt["id"])["action_count"], 1)

    def test_cli_checkpoint_file_commits_real_bytes_without_storing_or_scoring(
        self,
    ) -> None:
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "start",
                    self.session["id"],
                    self.task.id,
                    "--version",
                    str(self.task.version),
                    "--release",
                    self.release_report["release_id"],
                    "--idempotency-key",
                    "checkpoint-file-start",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())
        attempt_id = json.loads(output.getvalue())["id"]
        before = self.projection_snapshot()

        artifact = Path(self.tempdir.name) / "private-repair-source.py"
        explanation = Path(self.tempdir.name) / "private-explanation.md"
        submission = Path(self.tempdir.name) / "private-submission.tar"
        artifact_bytes = b"repair sentinel: mask future keys\n"
        explanation_bytes = b"causal visibility follows the key boundary\n"
        submission_bytes = b"sealed productive submission\n"
        artifact.write_bytes(artifact_bytes)
        explanation.write_bytes(explanation_bytes)
        submission.write_bytes(submission_bytes)

        def invoke(*arguments: str) -> tuple[int, str, str]:
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(
                    [
                        "--db",
                        str(self.database.path),
                        "task",
                        "checkpoint-file",
                        attempt_id,
                        *arguments,
                    ]
                )
            return code, stdout.getvalue(), stderr.getvalue()

        code, rendered, failure = invoke(
            str(artifact),
            "--kind",
            "artifact",
            "--artifact-kind",
            "diagnostic_trace_v1",
            "--idempotency-key",
            "checkpoint-file-artifact",
            "--json",
        )
        self.assertEqual(code, 0, failure)
        recorded = json.loads(rendered)
        self.assertEqual(
            recorded["action"]["payload"],
            {
                "artifact_digest": hashlib.sha256(
                    artifact_bytes
                ).hexdigest(),
                "artifact_kind": "diagnostic_trace_v1",
            },
        )
        for boundary in (
            "artifact_content_persisted",
            "artifact_executed",
            "evaluation_created",
            "learner_projection_applied",
            "certification_applied",
        ):
            self.assertFalse(recorded[boundary])

        code, plain, failure = invoke(
            str(artifact),
            "--kind",
            "artifact",
            "--artifact-kind",
            "diagnostic_trace_v1",
            "--idempotency-key",
            "checkpoint-file-artifact",
        )
        self.assertEqual(code, 0, failure)
        self.assertIn("idempotent replay", plain)
        self.assertIn("Artifact content persisted: no", plain)
        self.assertIn("Artifact executed: no", plain)
        self.assertIn("Evaluation created: no", plain)
        self.assertIn("Learner projection applied: no", plain)
        self.assertIn("Certification applied: no", plain)

        artifact.write_bytes(b"different private repair\n")
        code, conflict_output, failure = invoke(
            str(artifact),
            "--kind",
            "artifact",
            "--artifact-kind",
            "diagnostic_trace_v1",
            "--idempotency-key",
            "checkpoint-file-artifact",
        )
        self.assertEqual(code, 2)
        self.assertIn("different command", failure)
        self.assertNotIn(artifact.name, conflict_output + failure)
        self.assertNotIn("different private repair", conflict_output + failure)

        code, rendered, failure = invoke(
            str(explanation),
            "--kind",
            "explanation",
            "--idempotency-key",
            "checkpoint-file-explanation",
            "--json",
        )
        self.assertEqual(code, 0, failure)
        explanation_result = json.loads(rendered)
        self.assertEqual(
            explanation_result["action"]["payload"],
            {
                "explanation_digest": hashlib.sha256(
                    explanation_bytes
                ).hexdigest()
            },
        )

        code, rendered, failure = invoke(
            str(submission),
            "--kind",
            "submission",
            "--idempotency-key",
            "checkpoint-file-submission",
            "--json",
        )
        self.assertEqual(code, 0, failure)
        submission_result = json.loads(rendered)
        self.assertEqual(
            submission_result["action"]["payload"],
            {
                "submission_digest": hashlib.sha256(
                    submission_bytes
                ).hexdigest()
            },
        )

        all_output = plain + json.dumps(recorded) + json.dumps(
            explanation_result
        ) + json.dumps(submission_result)
        sensitive_material = (
            (artifact, artifact_bytes),
            (artifact, b"different private repair\n"),
            (explanation, explanation_bytes),
            (submission, submission_bytes),
        )
        for file_path, content in sensitive_material:
            self.assertNotIn(file_path.name, all_output)
            self.assertNotIn(content.decode("utf-8").strip(), all_output)

        report = self.ledger.report(attempt_id)
        self.assertEqual(report["status"], "submitted")
        self.assertEqual(report["action_count"], 4)
        self.assertEqual(report["evaluation_count"], 0)
        self.assertEqual(self.projection_snapshot(), before)
        with self.database.read() as connection:
            persisted = "\n".join(
                str(value)
                for row in connection.execute(
                    """SELECT payload_json, metadata_json FROM events
                       WHERE correlation_id=?
                       UNION ALL
                       SELECT payload_json, '' FROM performance_actions
                       WHERE attempt_id=?""",
                    (attempt_id, attempt_id),
                ).fetchall()
                for value in row
            )
        for file_path, content in sensitive_material:
            self.assertNotIn(file_path.name, persisted)
            self.assertNotIn(content.decode("utf-8").strip(), persisted)
        self.assertTrue(self.database.verify_integrity()["ok"])
        replay = ProjectionReplay(self.database).check("performance-learner")
        self.assertTrue(replay["ok"], replay["errors"])

    def test_cli_synthetic_score_rejects_invalid_units_without_traceback(
        self,
    ) -> None:
        attempt = self.start(key="invalid-cli-score")
        self.submit(attempt["id"])

        for option, value in (
            ("--score", "2"),
            ("--score", "nan"),
            ("--reliability", "-0.1"),
            ("--reliability", "inf"),
        ):
            arguments = [
                "--db",
                str(self.database.path),
                "task",
                "score",
                attempt["id"],
                "--provider",
                "deterministic-test",
                "--score",
                "0.8",
                option,
                value,
            ]
            output = io.StringIO()
            error = io.StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                exit_code = main(arguments)
            self.assertEqual(exit_code, 2)
            self.assertIn(
                f"{option} must be finite and between 0 and 1",
                error.getvalue(),
            )
            self.assertNotIn("Traceback", error.getvalue())
        self.assertEqual(self.ledger.report(attempt["id"])["evaluation_count"], 0)
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_cli_lists_starts_records_scores_and_reports_tasks(self) -> None:
        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "list",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())
        listed = json.loads(output.getvalue())
        self.assertEqual(listed[0]["task_id"], self.task.id)

        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "start",
                    self.session["id"],
                    self.task.id,
                    "--version",
                    str(self.task.version),
                    "--release",
                    self.release_report["release_id"],
                    "--idempotency-key",
                    "cli-performance-start",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())
        attempt_id = json.loads(output.getvalue())["id"]

        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "action",
                    attempt_id,
                    "submitted",
                    "--payload",
                    json.dumps({"submission_digest": _D3}),
                    "--idempotency-key",
                    "cli-performance-submit",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())

        output = io.StringIO()
        error = io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "score",
                    attempt_id,
                    "--provider",
                    "deterministic-test",
                    "--score",
                    "0.8",
                    "--idempotency-key",
                    "cli-performance-score",
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0, error.getvalue())
        scored = json.loads(output.getvalue())
        self.assertFalse(scored["projection_applied"])
        self.assertEqual(scored["shadow_evidence"]["total_evidence_weight"], 0.0)

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "report",
                    attempt_id,
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "submitted")
        self.assertEqual(report["evaluation_count"], 1)

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "report",
                    attempt_id,
                ]
            )
        self.assertEqual(exit_code, 0)
        rendered = output.getvalue()
        self.assertIn("raw task score 80.0%", rendered)
        self.assertIn("synthetic.fixed-performance-scorer@test-v1", rendered)
        self.assertIn("candidate weight 0.000", rendered)
        self.assertIn("mastery claim: no", rendered)


if __name__ == "__main__":
    unittest.main()
