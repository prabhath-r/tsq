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
)
from tsq.performance import (
    NORMALIZED_SCORING_RESULT_SCHEMA_VERSION,
    ImportedCriterionResult,
    ImportedEvaluation,
)
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
    performance_projection_snapshot,
)
from tsq.replay import ProjectionReplay
from tsq.store import SCHEMA_VERSION, Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
GOLDEN = ROOT / "tests" / "fixtures" / "performance_replay_baseline_expected.json"
START = datetime(2112, 8, 9, 10, 0, tzinfo=timezone.utc)
DIGEST_A = "4" * 64
DIGEST_B = "5" * 64
DIGEST_C = "6" * 64


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


def build_performance_database(path: Path) -> tuple[Database, str]:
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
        ledger.import_evaluation(
            attempt["id"],
            ImportedEvaluation(
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
            ),
            provider_id="reviewed_import_fixture",
            provider_version="v1",
            idempotency_key="performance-replay-evaluation",
            now=START + timedelta(minutes=5),
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
                       'PerformanceScoringLegacyExempted',
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
                       'performance_scoring_claims',
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
                      evaluation.evaluation_digest, evaluation.evaluation_json,
                      evaluation.authority_json,
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
    task = json.loads(row["definition_json"])
    evaluation = json.loads(row["evaluation_json"])
    authority = json.loads(row["authority_json"])
    bundle = json.loads(row["bundle_json"])
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
        "performance_event_count": report["performance_event_count"],
        "attempt_count": report["reconstructed_performance_attempt_count"],
        "action_count": report["reconstructed_performance_action_count"],
        "scoring_claim_count": report[
            "reconstructed_performance_scoring_claim_count"
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
        self.assertEqual(SCHEMA_VERSION, 17)
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
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(
            golden_performance_projection(report, self.database), expected
        )

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
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(
            golden_performance_projection(replayed, rebuilt), expected
        )


if __name__ == "__main__":
    unittest.main()
