# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.evidence import (
    ActionPhase,
    CriterionScale,
    EVIDENCE_BUNDLE_SCHEMA_VERSION,
    EvaluationStatus,
    LearningTask,
    RubricCriterion,
    TASK_SCHEMA_VERSION,
    TaskModality,
    canonical_digest,
)
from tsq.errors import ValidationError
from tsq.performance import (
    NORMALIZED_SCORING_RESULT_SCHEMA_VERSION,
    ImportedCriterionResult,
    ImportedEvaluation,
    ScoringProviderRegistry,
    SyntheticDeterministicProvider,
)
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
    performance_projection_snapshot,
)
from tsq.reconciliation import (
    ReconciliationObservation,
    ReconciliationOutcome,
    ScoringReconciliationReceipt,
    ScoringReconciliationRegistry,
    SyntheticReconciliationAdapter,
)
from tsq.replay import ProjectionReplay
from tsq.store import SCHEMA_VERSION, Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
EXPECTED_REPLAY = (
    ROOT / "tests" / "fixtures" / "performance_replay_expected.json"
)
START = datetime(2112, 8, 9, 10, 0, tzinfo=timezone.utc)
DIGEST_A = "4" * 64
DIGEST_B = "5" * 64
DIGEST_C = "6" * 64
RECONCILER_ID = "synthetic.performance-replay-reconciler"
RECONCILER_VERSION = "fixture-v1"


def rehash_event_stream(connection, stream_id: str) -> None:
    """Recompute one test stream after deliberate semantic event tampering."""

    rows = connection.execute(
        """SELECT * FROM events WHERE stream_id=?
           ORDER BY stream_version""",
        (stream_id,),
    ).fetchall()
    previous_hash = None
    for row in rows:
        payload_hash = canonical_digest(
            {
                "event_id": row["event_id"],
                "stream_id": row["stream_id"],
                "stream_version": row["stream_version"],
                "event_type": row["event_type"],
                "schema_version": row["schema_version"],
                "occurred_at": row["occurred_at"],
                "recorded_at": row["recorded_at"],
                "learner_id": row["learner_id"],
                "session_id": row["session_id"],
                "correlation_id": row["correlation_id"],
                "causation_id": row["causation_id"],
                "idempotency_key": row["idempotency_key"],
                "payload": json.loads(row["payload_json"]),
                "metadata": json.loads(row["metadata_json"]),
                "previous_hash": previous_hash,
            }
        )
        connection.execute(
            """UPDATE events SET previous_hash=?, payload_hash=?
               WHERE event_id=?""",
            (previous_hash, payload_hash, row["event_id"]),
        )
        previous_hash = payload_hash
    if rows:
        connection.execute(
            """UPDATE stream_heads
               SET stream_version=?, payload_hash=?, updated_at=?
               WHERE stream_id=?""",
            (
                rows[-1]["stream_version"],
                previous_hash,
                rows[-1]["recorded_at"],
                stream_id,
            ),
        )


class FixedIds:
    def __init__(self) -> None:
        self.counts: dict[str, int] = defaultdict(int)

    def __call__(self, prefix: str) -> str:
        self.counts[prefix] += 1
        return f"{prefix}_performance_replay_{self.counts[prefix]:03d}"


class FixedDateTime(datetime):
    ticks = 0

    @classmethod
    def now(cls, tz: timezone | None = None) -> "FixedDateTime":
        microsecond = cls.ticks
        cls.ticks += 1
        value = cls(
            START.year,
            START.month,
            START.day,
            START.hour,
            START.minute,
            START.second,
            microsecond,
            tzinfo=timezone.utc,
        )
        return value if tz is None else value.astimezone(tz)


def reconciliation_registry(
    claim: dict[str, object],
    *,
    outcome: ReconciliationOutcome,
    observed_at: datetime,
    imported: ImportedEvaluation | None = None,
) -> ScoringReconciliationRegistry:
    result_digest = None if imported is None else imported.digest
    observed_timestamp = observed_at.isoformat()
    provider_receipt_digest = canonical_digest(
        {
            "type": "tsq.performance_replay_provider_receipt",
            "provider_operation_digest": claim["provider_operation_digest"],
            "outcome": outcome.value,
            "observed_at": observed_timestamp,
            "result_digest": result_digest,
        }
    )
    attestation_digest = canonical_digest(
        {
            "type": "tsq.performance_replay_reconciler_attestation",
            "reconciler_id": RECONCILER_ID,
            "reconciler_version": RECONCILER_VERSION,
            "provider_receipt_digest": provider_receipt_digest,
        }
    )
    receipt = ScoringReconciliationReceipt(
        claim_id=claim["id"],
        attempt_id=claim["attempt_id"],
        evaluation_id=claim["evaluation_id"],
        through_sequence=claim["through_sequence"],
        provider_id=claim["provider_id"],
        provider_version=claim["provider_version"],
        reconciler_id=RECONCILER_ID,
        reconciler_version=RECONCILER_VERSION,
        action_trace_digest=claim["action_trace_digest"],
        command_hash=claim["command_hash"],
        scoring_request_digest=claim["scoring_request_digest"],
        provider_binding_digest=claim["provider_binding_digest"],
        outcome=outcome,
        observed_at=observed_timestamp,
        completed_at=(
            observed_timestamp
            if outcome is ReconciliationOutcome.COMPLETED
            else None
        ),
        result_digest=result_digest,
        reason_code={
            ReconciliationOutcome.UNKNOWN: "fixture_callback_still_unknown",
            ReconciliationOutcome.COMPLETED: "fixture_result_recovered",
            ReconciliationOutcome.DEFINITELY_ABSENT: (
                "fixture_callback_definitely_absent"
            ),
        }[outcome],
        provider_operation_digest=claim["provider_operation_digest"],
        provider_receipt_digest=provider_receipt_digest,
        attestation_digest=attestation_digest,
    )
    adapter = SyntheticReconciliationAdapter(
        ReconciliationObservation(
            receipt=receipt,
            imported_evaluation=imported,
        ),
        reconciler_id=RECONCILER_ID,
        reconciler_version=RECONCILER_VERSION,
        can_prove_absence=True,
    )
    registry = ScoringReconciliationRegistry(allow_synthetic=True)
    registry.register(adapter, adapter.authority_binding)
    return registry


def build_performance_database(
    path: Path,
    *,
    with_reconciliation: bool = True,
) -> tuple[Database, str]:
    identifiers = FixedIds()
    FixedDateTime.ticks = 0
    with (
        patch("tsq.store.datetime", FixedDateTime),
        patch("tsq.store.new_id", side_effect=identifiers),
        patch("tsq.engine.new_id", side_effect=identifiers),
        patch("tsq.policy.new_id", side_effect=identifiers),
        patch("tsq.performance_ledger.new_id", side_effect=identifiers),
    ):
        database = Database(path)
        database.initialize()
        database.import_corpus(*read_and_parse(CORPUS, include_catalog=True))
        engine = AdaptiveEngine(database)
        engine.create_learner(
            "performance-replay", "Performance Replay Fixture"
        )
        session = engine.start_session(
            "performance-replay",
            "c_attention",
            mode="learn",
            seed=23,
        )
        with database.read() as connection:
            corpus_release_id = connection.execute(
                "SELECT value FROM meta WHERE key='active_corpus_release'"
            ).fetchone()["value"]
            source = connection.execute(
                """SELECT source.id, source.content_hash
                   FROM release_sources membership
                   JOIN sources source ON source.id=membership.source_id
                   WHERE membership.release_id=?
                     AND source.id='src_vaswani_attention_2017'""",
                (corpus_release_id,),
            ).fetchone()
        task = LearningTask(
            id="task_attention_routing_replay",
            version=1,
            family_id="family_attention_routing_replay",
            title="Trace and validate attention value routing",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_attention_routing",
                    name="Attention value-routing invariant",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_attention", 1.0),),
                    objective_weights=(("lo_attention_value_routing", 1.0),),
                    dependence_group="attention_routing_artifact",
                    evidence_cap=0.75,
                    dependence_cap=0.75,
                ),
            ),
            instructions=(
                "Inspect the pinned routing stimulus, identify the violated "
                "attention invariant, and submit a content-addressed repair."
            ),
            source_manifests=((source["id"], source["content_hash"]),),
            administration_id="admin_replay_digest_only",
            administration_manifest_digest=DIGEST_A,
            stimulus_id="stimulus_attention_routing_replay_v1",
            stimulus_digest=DIGEST_B,
        )
        release = PerformanceTaskRelease(
            title="Performance replay golden release",
            corpus_release_id=corpus_release_id,
            review=TaskReleaseReview(
                reviewer_kind="human",
                reviewer_id="reviewer_performance_replay",
                reviewed_at=(START - timedelta(days=1)).isoformat(),
                independent_of_author=True,
                attestation_digest=DIGEST_C,
            ),
            tasks=(("pilot", task),),
        )
        ledger = PerformanceLedger(database)
        release_report = ledger.publish_release(release, now=START)
        attempt = ledger.start_attempt(
            session["id"],
            task.id,
            task_version=task.version,
            task_release_id=release_report["release_id"],
            idempotency_key="performance-replay-start",
            now=START + timedelta(minutes=1),
        )
        artifact = ledger.record_action(
            attempt["id"],
            "artifact_checkpoint",
            {
                "artifact_digest": DIGEST_A,
                "artifact_kind": "patch_digest",
            },
            idempotency_key="performance-replay-artifact",
            now=START + timedelta(minutes=2),
        )
        checks = ledger.record_action(
            attempt["id"],
            "check_run",
            {
                "check_set_id": "attention_routing_checks_v1",
                "passed": 7,
                "failed": 1,
                "errored": 0,
                "skipped": 0,
                "result_digest": DIGEST_B,
            },
            idempotency_key="performance-replay-checks",
            now=START + timedelta(minutes=3),
        )
        submitted = ledger.record_action(
            attempt["id"],
            "submitted",
            {"submission_digest": DIGEST_A},
            idempotency_key="performance-replay-submit",
            now=START + timedelta(minutes=4),
        )
        recovered_evaluation = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_attention_routing",
                    status=EvaluationStatus.VALID,
                    score=0.75,
                    outcome_code="routing_boundary_partial",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(
                        artifact["id"],
                        checks["id"],
                        submitted["id"],
                    ),
                    reliability=0.9,
                ),
            )
        )

        if not with_reconciliation:
            ledger.import_evaluation(
                attempt["id"],
                recovered_evaluation,
                provider_id="reviewed_import_fixture",
                provider_version="v1",
                idempotency_key="performance-replay-evaluation",
                now=START + timedelta(minutes=5),
            )
        else:

            class StrandedProvider(SyntheticDeterministicProvider):
                def __init__(self, evaluation, **kwargs):
                    super().__init__(evaluation, **kwargs)
                    self.calls = 0

                def score(self, request):
                    self.calls += 1
                    raise RuntimeError(
                        "deterministic stranded callback fixture"
                    )

            provider = StrandedProvider(
                recovered_evaluation,
                provider_id="synthetic.performance-replay-scorer",
                provider_version="fixture-v1",
            )
            scoring_registry = ScoringProviderRegistry(allow_synthetic=True)
            scoring_registry.register(provider, provider.authority_binding)
            try:
                ledger.score_attempt(
                    attempt["id"],
                    scoring_registry,
                    provider.provider_id,
                    provider.provider_version,
                    idempotency_key="performance-replay-scoring-claim",
                    now=START + timedelta(minutes=5),
                )
            except ValidationError as exc:
                if "failed safely" not in str(exc):
                    raise AssertionError(
                        "Stranded scoring fixture failed before its callback."
                    ) from exc
            else:
                raise AssertionError(
                    "Stranded scoring fixture did not fail closed."
                )
            claims = ledger.list_scoring_claims(
                attempt_id=attempt["id"],
                status="unreconciled",
            )
            if len(claims) != 1:
                raise AssertionError(
                    "Stranded scoring fixture has no unique claim."
                )
            claim = claims[0]
            ledger.reconcile_scoring_claim(
                claim["id"],
                reconciliation_registry(
                    claim,
                    outcome=ReconciliationOutcome.UNKNOWN,
                    observed_at=START + timedelta(minutes=6),
                ),
                RECONCILER_ID,
                RECONCILER_VERSION,
                idempotency_key=(
                    "performance-replay-reconciliation-unknown"
                ),
                now=START + timedelta(minutes=6),
            )
            ledger.reconcile_scoring_claim(
                claim["id"],
                reconciliation_registry(
                    claim,
                    outcome=ReconciliationOutcome.COMPLETED,
                    observed_at=START + timedelta(minutes=7),
                    imported=recovered_evaluation,
                ),
                RECONCILER_ID,
                RECONCILER_VERSION,
                idempotency_key=(
                    "performance-replay-reconciliation-completed"
                ),
                now=START + timedelta(minutes=7),
            )
            if provider.calls != 1:
                raise AssertionError(
                    "Reconciliation retried the stranded scoring callback."
                )
    return database, attempt["id"]


def performance_source_snapshot(database: Database) -> dict[str, object]:
    with database.read() as connection:
        events = [
            tuple(row)
            for row in connection.execute(
                """SELECT * FROM events WHERE event_type IN (
                       'PerformanceTaskStarted', 'PerformanceActionRecorded',
                       'PerformanceScoringClaimed',
                       'PerformanceScoringClaimMigrated',
                       'PerformanceScoringReconciled',
                       'PerformanceScoringLegacyExempted',
                       'PerformanceArtifactRunClaimed',
                       'PerformanceArtifactRunObserved',
                       'TaskEvaluationRecorded', 'ShadowEvidenceReduced'
                   ) ORDER BY stream_id, stream_version"""
            )
        ]
        triggers = [
            tuple(row)
            for row in connection.execute(
                """SELECT name, sql FROM sqlite_master
                   WHERE type='trigger' AND tbl_name IN (
                       'performance_attempts', 'performance_actions',
                       'performance_artifact_run_claims',
                       'performance_artifact_run_receipts',
                       'performance_scoring_claims',
                       'performance_scoring_reconciliations',
                       'task_evaluations', 'shadow_evidence_bundles'
                   ) ORDER BY name"""
            )
        ]
        return {
            "projection": performance_projection_snapshot(connection),
            "events": events,
            "triggers": triggers,
        }


def golden_performance_projection(
    report: dict[str, object], database: Database
) -> dict[str, object]:
    with database.read() as connection:
        row = connection.execute(
            """SELECT attempt.task_digest, task.definition_json,
                      evaluation.id AS evaluation_id,
                      evaluation.event_id AS evaluation_event_id,
                      evaluation.evaluation_digest, evaluation.evaluation_json,
                      evaluation.authority_json,
                      bundle.id AS bundle_id,
                      bundle.event_id AS bundle_event_id,
                      bundle.bundle_digest, bundle.bundle_json
               FROM performance_attempts attempt
               JOIN performance_tasks task
                 ON task.task_id=attempt.task_id
                AND task.task_version=attempt.task_version
               JOIN task_evaluations evaluation
                 ON evaluation.attempt_id=attempt.id
               JOIN shadow_evidence_bundles bundle
                 ON bundle.evaluation_id=evaluation.id"""
        ).fetchone()
        scoring_claims = [
            dict(item)
            for item in connection.execute(
                """SELECT id, event_id, claim_schema_version, attempt_id,
                          evaluation_id, through_sequence, provider_id,
                          provider_version, action_trace_digest,
                          scoring_request_digest, provider_binding_digest,
                          provider_operation_digest, command_hash, claimed_at
                   FROM performance_scoring_claims ORDER BY id"""
            )
        ]
        scoring_reconciliations = [
            dict(item)
            for item in connection.execute(
                """SELECT observation.id, observation.event_id,
                          observation.claim_id, observation.attempt_id,
                          observation.evaluation_id, observation.outcome,
                          observation.scoring_request_digest,
                          observation.provider_binding_digest,
                          observation.provider_operation_digest,
                          observation.reconciler_id,
                          observation.reconciler_version,
                          observation.reconciliation_binding_digest,
                          observation.receipt_digest,
                          observation.normalized_result_digest,
                          observation.reconciled_at,
                          observation.command_hash
                   FROM performance_scoring_reconciliations observation
                   JOIN events event
                     ON event.event_id=observation.event_id
                   ORDER BY event.stream_id, event.stream_version,
                            observation.id"""
            )
        ]
        artifact_run_claims = [
            dict(item)
            for item in connection.execute(
                "SELECT * FROM performance_artifact_run_claims ORDER BY id"
            )
        ]
        artifact_run_receipts = [
            dict(item)
            for item in connection.execute(
                "SELECT * FROM performance_artifact_run_receipts ORDER BY id"
            )
        ]
    task = json.loads(row["definition_json"])
    evaluation = json.loads(row["evaluation_json"])
    authority = json.loads(row["authority_json"])
    bundle = json.loads(row["bundle_json"])
    claim_state = PerformanceLedger(database).inspect_scoring_claim(
        scoring_claims[0]["id"]
    )
    return {
        "format_version": report["format_version"],
        "database_schema_version": report["source_schema_version"],
        "task_schema_version": task["schema_version"],
        "evidence_bundle_schema_version": bundle["schema_version"],
        "normalized_scoring_result_schema_version": authority[
            "normalized_result"
        ]["schema_version"],
        "criterion_objective_weights": task["criteria"][0][
            "objective_weights"
        ],
        "evidence_objective_weights": bundle["records"][0][
            "objective_weights"
        ],
        "task_digest": row["task_digest"],
        "action_trace_digest": evaluation["action_trace_digest"],
        "evaluation_digest": row["evaluation_digest"],
        "bundle_digest": row["bundle_digest"],
        "projection_rows": {
            "artifact_run_claims": artifact_run_claims,
            "artifact_run_receipts": artifact_run_receipts,
            "scoring_claims": scoring_claims,
            "scoring_reconciliations": scoring_reconciliations,
            "evaluation": {
                "id": row["evaluation_id"],
                "event_id": row["evaluation_event_id"],
                "digest": row["evaluation_digest"],
            },
            "bundle": {
                "id": row["bundle_id"],
                "event_id": row["bundle_event_id"],
                "digest": row["bundle_digest"],
                "projection_applied": False,
                "certification_applied": False,
            },
        },
        "claim_state": {
            "id": claim_state["id"],
            "status": claim_state["status"],
            "source": claim_state["source"],
            "terminal": claim_state["terminal"],
            "terminal_at": claim_state["terminal_at"],
            "reconciliation_count": claim_state["reconciliation_count"],
            "latest_reconciliation_id": claim_state[
                "latest_reconciliation_id"
            ],
            "latest_reconciliation_outcome": claim_state[
                "latest_reconciliation_outcome"
            ],
            "automatic_retry_allowed": claim_state[
                "automatic_retry_allowed"
            ],
        },
        "performance_event_count": report["performance_event_count"],
        "attempt_count": report["reconstructed_performance_attempt_count"],
        "action_count": report["reconstructed_performance_action_count"],
        "artifact_run_claim_count": report[
            "reconstructed_performance_artifact_run_claim_count"
        ],
        "artifact_run_receipt_count": report[
            "reconstructed_performance_artifact_run_receipt_count"
        ],
        "scoring_claim_count": report[
            "reconstructed_performance_scoring_claim_count"
        ],
        "scoring_reconciliation_count": report[
            "reconstructed_performance_scoring_reconciliation_count"
        ],
        "evaluation_count": report["reconstructed_task_evaluation_count"],
        "bundle_count": report[
            "reconstructed_shadow_evidence_bundle_count"
        ],
        "checkpoints": report["performance_checkpoints"],
    }


class PerformanceProjectionReplayGoldenTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database, self.attempt_id = build_performance_database(
            Path(self.tempdir.name) / "source.db"
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_reconstructs_exact_golden_projection_without_source_writes(
        self,
    ) -> None:
        before = performance_source_snapshot(self.database)

        report = ProjectionReplay(self.database).check("performance-replay")

        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(SCHEMA_VERSION, 22)
        self.assertEqual(TASK_SCHEMA_VERSION, 3)
        self.assertEqual(EVIDENCE_BUNDLE_SCHEMA_VERSION, 2)
        self.assertEqual(NORMALIZED_SCORING_RESULT_SCHEMA_VERSION, 1)
        self.assertTrue(report["performance_projection_matches_replay"])
        self.assertEqual(
            report["source_performance_projection_hash"],
            report["reconstructed_performance_projection_hash"],
        )
        self.assertRegex(
            report["reconstructed_performance_projection_hash"],
            r"^[0-9a-f]{64}$",
        )
        with self.database.read() as connection:
            pinning = connection.execute(
                """SELECT attempt.corpus_release_id AS attempt_release_id,
                          session.corpus_release_id AS session_release_id,
                          release.corpus_release_id AS task_release_id,
                          (
                              SELECT value FROM meta
                              WHERE key='active_corpus_release'
                          ) AS active_release_id
                   FROM performance_attempts attempt
                   JOIN sessions session ON session.id=attempt.session_id
                   JOIN performance_task_releases release
                     ON release.id=attempt.task_release_id
                   WHERE attempt.id=?""",
                (self.attempt_id,),
            ).fetchone()
        self.assertEqual(
            {
                pinning["attempt_release_id"],
                pinning["session_release_id"],
                pinning["task_release_id"],
            },
            {pinning["active_release_id"]},
        )
        self.assertEqual(performance_source_snapshot(self.database), before)
        expected = json.loads(EXPECTED_REPLAY.read_text(encoding="utf-8"))
        actual = golden_performance_projection(report, self.database)
        reconciliation_rows = actual["projection_rows"][
            "scoring_reconciliations"
        ]
        self.assertEqual(
            [row["outcome"] for row in reconciliation_rows],
            ["unknown", "completed"],
        )
        self.assertIsNone(reconciliation_rows[0]["normalized_result_digest"])
        self.assertRegex(
            reconciliation_rows[1]["normalized_result_digest"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(actual["claim_state"]["status"], "completed")
        self.assertEqual(actual["claim_state"]["source"], "reconciliation")
        self.assertEqual(actual["claim_state"]["reconciliation_count"], 2)
        self.assertFalse(
            actual["claim_state"]["automatic_retry_allowed"]
        )
        self.assertEqual(
            actual["projection_rows"]["evaluation"]["id"],
            actual["projection_rows"]["scoring_claims"][0]["evaluation_id"],
        )
        self.assertFalse(
            actual["projection_rows"]["bundle"]["projection_applied"]
        )
        self.assertFalse(
            actual["projection_rows"]["bundle"]["certification_applied"]
        )
        self.assertEqual(
            report["source_performance_artifact_run_claim_count"], 0
        )
        self.assertEqual(
            report["reconstructed_performance_artifact_run_claim_count"], 0
        )
        self.assertEqual(
            report["source_performance_artifact_run_receipt_count"], 0
        )
        self.assertEqual(
            report["reconstructed_performance_artifact_run_receipt_count"], 0
        )
        self.assertEqual(actual, expected)

    def test_corruption_is_detected_and_only_the_copy_is_repaired(self) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER shadow_evidence_bundles_no_update"
            )
            connection.execute(
                """UPDATE shadow_evidence_bundles SET bundle_digest=?
                   WHERE attempt_id=?""",
                ("0" * 64, self.attempt_id),
            )
        corrupted = performance_source_snapshot(self.database)

        report = ProjectionReplay(self.database).check("performance-replay")

        self.assertFalse(report["ok"])
        self.assertFalse(report["performance_projection_matches_replay"])
        self.assertIn(
            "stored performance projection differs from deterministic replay",
            report["errors"],
        )
        self.assertEqual(performance_source_snapshot(self.database), corrupted)

        target = Path(self.tempdir.name) / "performance-rebuilt.db"
        rebuilt_report = ProjectionReplay(self.database).rebuild_copy(
            "performance-replay", target
        )

        self.assertTrue(rebuilt_report["ok"], rebuilt_report["errors"])
        self.assertTrue(
            rebuilt_report["source_performance_projection_was_repaired"]
        )
        self.assertEqual(performance_source_snapshot(self.database), corrupted)
        rebuilt = Database(target, read_only=True)
        self.assertTrue(rebuilt.verify_integrity()["ok"])
        replayed = ProjectionReplay(rebuilt).check("performance-replay")
        self.assertTrue(replayed["ok"], replayed["errors"])
        expected = json.loads(EXPECTED_REPLAY.read_text(encoding="utf-8"))
        self.assertEqual(
            golden_performance_projection(replayed, rebuilt), expected
        )

    def test_semantic_event_corruption_returns_bounded_failure_report(
        self,
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute("DROP TRIGGER events_no_update")
            event = connection.execute(
                """SELECT event_id, stream_id, payload_json
                   FROM events
                   WHERE event_type='PerformanceScoringReconciled'
                     AND json_extract(payload_json, '$.outcome')='completed'
                   ORDER BY stream_version DESC LIMIT 1"""
            ).fetchone()
            payload = json.loads(event["payload_json"])
            payload["unexpected_semantic_field"] = True
            connection.execute(
                "UPDATE events SET payload_json=? WHERE event_id=?",
                (
                    json.dumps(
                        payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    event["event_id"],
                ),
            )
            rehash_event_stream(connection, event["stream_id"])

        report = ProjectionReplay(self.database).check(
            "performance-replay"
        )

        self.assertFalse(report["ok"])
        self.assertFalse(report["rebuild_safe"])
        self.assertFalse(report["performance_projection_matches_replay"])
        self.assertIsInstance(report["performance_replay_error"], str)
        self.assertIn(
            "unexpected_semantic_field",
            report["performance_replay_error"],
        )
        self.assertIsNone(report["performance_event_count"])
        self.assertIsNone(
            report["reconstructed_performance_artifact_run_claim_count"]
        )
        self.assertIsNone(
            report["reconstructed_performance_artifact_run_receipt_count"]
        )
        self.assertIsNone(
            report["reconstructed_performance_scoring_reconciliation_count"]
        )
        self.assertIsNone(
            report["reconstructed_performance_projection_hash"]
        )
        self.assertEqual(report["performance_checkpoints"], [])
        self.assertTrue(
            any(
                error.startswith(
                    "performance projection replay failed closed:"
                )
                for error in report["errors"]
            ),
            report["errors"],
        )
        target = Path(self.tempdir.name) / "semantic-corruption.db"
        with self.assertRaisesRegex(
            ValidationError,
            "Projection replay did not verify",
        ):
            ProjectionReplay(self.database).rebuild_copy(
                "performance-replay",
                target,
            )
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
