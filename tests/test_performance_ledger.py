# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.cli import main
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, ValidationError
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
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
    performance_integrity_errors,
    read_task_release,
)
from tsq.replay import ProjectionReplay
from tsq.store import Database, performance_scoring_claim_event_key


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
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
        self.assertEqual(unresolved[0]["status"], "unresolved")
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

    def test_registered_scorer_can_observe_its_authorized_rubric_subset(
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
        result = self.ledger.score_attempt(
            attempt["id"],
            registry,
            authority.provider_id,
            authority.provider_version,
            now=START + timedelta(minutes=4),
        )
        replay = self.ledger.score_attempt(
            attempt["id"],
            registry,
            authority.provider_id,
            authority.provider_version,
            now=START + timedelta(minutes=4),
        )
        self.assertEqual(replay["evaluation"], result["evaluation"])
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(provider.calls, 1)

        records = {
            record["criterion_id"]: record
            for record in result["shadow_evidence"]["records"]
        }
        self.assertGreater(
            records["criterion_checked_behavior"]["effective_weight"], 0.0
        )
        self.assertTrue(
            records["criterion_checked_behavior"]["certification_eligible"]
        )
        self.assertEqual(
            records["criterion_unreviewed_explanation"]["effective_weight"],
            0.0,
        )
        self.assertIn(
            "missing_evaluation",
            records["criterion_unreviewed_explanation"]["reason_codes"],
        )
        normalized = result["authority"]["normalized_result"]
        self.assertEqual(
            normalized["request"]["criterion_ids"],
            ["criterion_checked_behavior"],
        )
        self.assertFalse(result["projection_applied"])
        self.assertFalse(result["certification_applied"])
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
        damaged_source = self.ledger.report(attempt["id"])
        check = ProjectionReplay(self.database).check("performance-learner")
        self.assertFalse(check["ok"])
        self.assertFalse(check["performance_projection_matches_replay"])

        rebuilt_path = Path(self.tempdir.name) / "performance-rebuilt.db"
        rebuilt = ProjectionReplay(self.database).rebuild_copy(
            "performance-learner", rebuilt_path
        )
        self.assertTrue(rebuilt["ok"], rebuilt["errors"])
        self.assertTrue(rebuilt["source_performance_projection_was_repaired"])
        self.assertEqual(self.ledger.report(attempt["id"]), damaged_source)
        rebuilt_database = Database(rebuilt_path, read_only=True)
        self.assertTrue(rebuilt_database.verify_integrity()["ok"])
        rebuilt_report = PerformanceLedger(rebuilt_database).report(attempt["id"])
        self.assertNotEqual(
            rebuilt_report["evaluations"][0]["shadow_evidence"],
            damaged_source["evaluations"][0]["shadow_evidence"],
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
                "task_count",
                "status_counts",
                "idempotent_replay",
            },
        )
        self.assertEqual(
            replay["status_counts"],
            self.release_report["status_counts"],
        )
        listed = self.ledger.list_releases()
        self.assertNotIn("review_json", listed[0])
        self.assertEqual(
            listed[0]["review"]["reviewer_id"],
            self.release.review.reviewer_id,
        )

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
