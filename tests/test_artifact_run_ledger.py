# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.artifact_intake import ProductiveArtifactSnapshot
from tsq.artifact_runner import (
    CAUSAL_MASK_CHECK_SET_ID,
    ArtifactProcessReceipt,
    ArtifactResultCode,
    ArtifactRunOutcome,
    ArtifactRunResult,
    SyntheticArtifactRunnerRegistry,
    bundled_synthetic_binding,
)
from tsq.cli import main
from tsq.corpus import read_and_parse
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
)
from tsq.performance import (
    ImportedCriterionResult,
    ImportedEvaluation,
    ProviderAuthorityBinding,
    ScoringProviderRegistry,
)
from tsq.performance_ledger import (
    OperationalArtifactRunReceipt,
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
)
from tsq.store import (
    PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
    Database,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
BASE = datetime(2120, 8, 9, 10, 0, tzinfo=timezone.utc)
_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64


def artifact(mask: list[list[bool]]) -> bytes:
    return json.dumps(
        {"mask": mask, "schema_version": 1},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def snapshot(material: bytes) -> ProductiveArtifactSnapshot:
    return ProductiveArtifactSnapshot(
        content=material,
        sha256=hashlib.sha256(material).hexdigest(),
        size_bytes=len(material),
    )


class ArtifactRunLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "artifact-runs.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)
        self.engine.create_learner("artifact-run-learner", "Artifact Runner")
        self.session = self.engine.start_session(
            "artifact-run-learner", "t_transformers", seed=812
        )
        with self.database.read() as connection:
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
        self.corpus_release_id = corpus_release_id
        self.binding = bundled_synthetic_binding()
        self.registry = SyntheticArtifactRunnerRegistry(allow_synthetic=True)
        self.registry.register(self.binding)
        self.authority = ProviderAuthorityBinding(
            provider_id="checks.causal-mask-matrix",
            provider_version="v1",
            declared_kind=ScorerKind.DETERMINISTIC,
            authority_id="authority.synthetic.causal-mask",
            authority_manifest_digest=_D2,
            check_set_manifests=(
                (
                    self.binding.check_set_id,
                    self.binding.check_set_manifest_digest,
                ),
            ),
            artifact_manifests=(
                (
                    self.binding.artifact_kind,
                    self.binding.artifact_manifest_digest,
                ),
            ),
            verified=True,
        )
        self.contract = ScorerContract(
            kind=ScorerKind.DETERMINISTIC,
            scorer_id=self.authority.provider_id,
            scorer_version=self.authority.provider_version,
            authority_id=self.authority.authority_id,
            authority_manifest_digest=self.authority.authority_manifest_digest,
            criterion_ids=("criterion_causal_mask",),
            evidence_action_kinds=(
                ActionKind.ARTIFACT_CHECKPOINT,
                ActionKind.CHECK_RUN,
            ),
            check_set_manifests=self.authority.check_set_manifests,
            artifact_manifests=self.authority.artifact_manifests,
        )
        self.task = LearningTask(
            id="task_causal_mask_artifact",
            version=1,
            family_id="family_causal_mask_artifact",
            title="Check a causal-mask matrix artifact",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_causal_mask",
                    name="Causal visibility",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_causal_masking", 1.0),),
                    dependence_group="causal_mask_artifact",
                    evidence_cap=0.8,
                    dependence_cap=0.8,
                ),
            ),
            instructions=(
                "Produce a JSON causal-mask matrix and checkpoint it before "
                "running the released deterministic checker."
            ),
            source_manifests=((source["id"], source["content_hash"]),),
            administration_id="admin_causal_mask_artifact_v1",
            administration_manifest_digest=_D0,
            stimulus_id="stimulus_causal_mask_artifact_v1",
            stimulus_digest=_D1,
            scorer_contracts=(self.contract,),
        )
        reviewed = datetime.now(timezone.utc)
        self.release = PerformanceTaskRelease(
            title="Artifact runner test release",
            corpus_release_id=corpus_release_id,
            review=TaskReleaseReview(
                reviewer_kind="human",
                reviewer_id="reviewer_artifact_runner",
                reviewed_at=reviewed.isoformat(),
                independent_of_author=True,
                attestation_digest=_D2,
            ),
            tasks=(("pilot", self.task),),
        )
        self.ledger = PerformanceLedger(self.database)
        self.release_report = self.ledger.publish_release(
            self.release, now=reviewed
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def start_with_artifact(
        self,
        material: bytes,
        *,
        key_suffix: str = "default",
    ) -> tuple[dict, dict, ProductiveArtifactSnapshot]:
        attempt = self.ledger.start_attempt(
            self.session["id"],
            self.task.id,
            task_version=self.task.version,
            task_release_id=self.release_report["release_id"],
            idempotency_key=f"start-{key_suffix}",
            now=BASE + timedelta(minutes=1),
        )
        captured = snapshot(material)
        checkpoint = self.ledger.record_action(
            attempt["id"],
            ActionKind.ARTIFACT_CHECKPOINT.value,
            {
                "artifact_digest": captured.sha256,
                "artifact_kind": self.binding.artifact_kind,
            },
            idempotency_key=f"artifact-{key_suffix}",
            now=BASE + timedelta(minutes=2),
        )
        return attempt, checkpoint, captured

    def run_check(
        self,
        attempt: dict,
        checkpoint: dict,
        captured: ProductiveArtifactSnapshot,
        *,
        key: str | None = "run-default",
    ) -> dict:
        return self.ledger.run_artifact_check(
            attempt["id"],
            captured,
            self.registry,
            self.binding,
            check_set_id=CAUSAL_MASK_CHECK_SET_ID,
            artifact_action_id=checkpoint["id"],
            idempotency_key=key,
            now=BASE + timedelta(minutes=3),
        )

    def test_success_commits_claim_system_action_and_exact_receipts(self) -> None:
        attempt, checkpoint, captured = self.start_with_artifact(
            artifact([[True, False], [True, True]])
        )
        result = self.run_check(attempt, checkpoint, captured)

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["terminal"])
        self.assertFalse(result["retry_allowed"])
        self.assertFalse(result["artifact_content_persisted"])
        self.assertFalse(result["skill_authority"])
        self.assertFalse(result["operating_system_sandboxed"])
        self.assertTrue(result["process_boundary_configured"])
        self.assertTrue(result["process_separated"])
        self.assertTrue(result["worker_process_started"])
        self.assertEqual(
            result["check_action"]["payload"]["result_digest"],
            result["operational_receipt"]["result_digest"],
        )
        self.assertEqual(
            set(result["operational_receipt"]),
            {
                "claim_id",
                "attempt_id",
                "artifact_action_id",
                "artifact_digest",
                "artifact_kind",
                "artifact_manifest_digest",
                "check_set_id",
                "check_set_manifest_digest",
                "runner_id",
                "runner_version",
                "outcome",
                "started_at",
                "completed_at",
                "result_digest",
                "request_digest",
                "binding_digest",
                "schema_version",
            },
        )
        decoded = OperationalArtifactRunReceipt.from_terms(
            result["operational_receipt"]
        )
        self.assertEqual(decoded.digest, result["receipt_digest"])
        with self.database.read() as connection:
            claim_event = connection.execute(
                "SELECT * FROM events WHERE event_id=?",
                (result["event_id"],),
            ).fetchone()
            check_event = connection.execute(
                "SELECT * FROM events WHERE event_id=?",
                (result["check_action"]["event_id"],),
            ).fetchone()
            receipt_event = connection.execute(
                """SELECT event.* FROM performance_artifact_run_receipts receipt
                   JOIN events event ON event.event_id=receipt.event_id
                   WHERE receipt.claim_id=?""",
                (result["claim_id"],),
            ).fetchone()
        self.assertLess(
            claim_event["stream_version"], check_event["stream_version"]
        )
        self.assertLess(
            check_event["stream_version"], receipt_event["stream_version"]
        )
        self.assertIsNone(receipt_event["session_id"])
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_invalid_artifact_is_terminal_and_generates_errored_check(self) -> None:
        attempt, checkpoint, captured = self.start_with_artifact(
            b"{not-json"
        )
        result = self.run_check(attempt, checkpoint, captured)

        self.assertEqual(result["status"], "invalid_artifact")
        self.assertTrue(result["process_boundary_configured"])
        self.assertTrue(result["worker_process_started"])
        self.assertEqual(result["check_action"]["payload"]["errored"], 1)
        self.assertIsNotNone(result["process_receipt"])
        self.assertEqual(
            result["process_receipt"]["result"]["outcome"],
            "invalid_artifact",
        )

    def _process_failure(
        self,
        request,
        *,
        outcome: ArtifactRunOutcome,
        code: ArtifactResultCode,
        started: bool,
    ) -> ArtifactProcessReceipt:
        return ArtifactProcessReceipt(
            request=request,
            binding=self.binding,
            result=ArtifactRunResult(
                checker_id=request.checker_id,
                checker_version=request.checker_version,
                artifact_sha256=request.artifact_sha256,
                outcome=outcome,
                outcome_codes=(code,),
                passed=0,
                failed=0,
                errored=1,
                skipped=0,
            ),
            worker_process_started=started,
        )

    def test_timeout_is_terminal_without_result_or_action(self) -> None:
        attempt, checkpoint, captured = self.start_with_artifact(
            artifact([[True]])
        )

        def timeout(request, _material):
            return self._process_failure(
                request,
                outcome=ArtifactRunOutcome.TIMED_OUT,
                code=ArtifactResultCode.WORKER_TIMEOUT,
                started=True,
            )

        with patch.object(self.registry, "run", side_effect=timeout):
            result = self.run_check(attempt, checkpoint, captured)
        self.assertEqual(result["status"], "timed_out")
        self.assertTrue(result["process_boundary_configured"])
        self.assertIsNone(result["worker_process_started"])
        self.assertIsNone(result["check_action"])
        self.assertIsNone(result["process_receipt"])
        self.assertIsNone(result["operational_receipt"]["result_digest"])
        self.assertEqual(len(self.ledger.list_actions(attempt["id"])), 2)

    def test_worker_and_protocol_failures_map_to_runner_failed(self) -> None:
        for suffix, outcome, code, started in (
            (
                "worker",
                ArtifactRunOutcome.WORKER_FAILED,
                ArtifactResultCode.WORKER_START_FAILED,
                False,
            ),
            (
                "protocol",
                ArtifactRunOutcome.PROTOCOL_ERROR,
                ArtifactResultCode.WORKER_PROTOCOL_INVALID,
                True,
            ),
        ):
            with self.subTest(outcome=outcome.value):
                if suffix != "worker":
                    self.engine.create_learner(
                        f"artifact-run-{suffix}", f"Artifact {suffix}"
                    )
                    self.session = self.engine.start_session(
                        f"artifact-run-{suffix}",
                        "t_transformers",
                        seed=813,
                    )
                attempt, checkpoint, captured = self.start_with_artifact(
                    artifact([[True]]), key_suffix=suffix
                )

                def failure(request, _material):
                    return self._process_failure(
                        request,
                        outcome=outcome,
                        code=code,
                        started=started,
                    )

                with patch.object(
                    self.registry, "run", side_effect=failure
                ):
                    result = self.run_check(
                        attempt,
                        checkpoint,
                        captured,
                        key=f"run-{suffix}",
                    )
                self.assertEqual(result["status"], "runner_failed")
                self.assertIsNone(result["check_action"])
                self.assertIsNone(result["process_receipt"])

    def test_crash_strands_claim_and_varied_key_does_not_retry(self) -> None:
        attempt, checkpoint, captured = self.start_with_artifact(
            artifact([[True]])
        )
        with patch.object(
            self.registry,
            "run",
            side_effect=RuntimeError("simulated process crash"),
        ) as invoke:
            with self.assertRaisesRegex(
                ValidationError, "claim remains unresolved"
            ):
                self.run_check(
                    attempt,
                    checkpoint,
                    captured,
                    key=None,
                )
            replay = self.run_check(
                attempt,
                checkpoint,
                captured,
                key="different-caller-key",
            )
        self.assertEqual(invoke.call_count, 1)
        self.assertEqual(replay["status"], "unresolved")
        self.assertFalse(replay["terminal"])
        self.assertFalse(replay["retry_allowed"])
        self.assertTrue(replay["idempotent_replay"])
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM performance_artifact_run_claims"
                ).fetchone()["n"],
                1,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM performance_artifact_run_receipts"
                ).fetchone()["n"],
                0,
            )

    def test_all_reserved_technical_idempotency_namespaces_are_rejected(
        self,
    ) -> None:
        attempt, checkpoint, captured = self.start_with_artifact(
            artifact([[True]])
        )
        for prefix in (
            PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
            PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
            PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
            PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
        ):
            with self.subTest(prefix=prefix), self.assertRaisesRegex(
                ValidationError, "reserved technical namespace"
            ):
                self.run_check(
                    attempt,
                    checkpoint,
                    captured,
                    key=prefix + ("a" * 64),
                )
        self.assertEqual(self.ledger.list_artifact_runs(), [])

    def test_exact_and_varied_key_replay_survive_terminal_attempt(self) -> None:
        attempt, checkpoint, captured = self.start_with_artifact(
            artifact([[True]])
        )
        first = self.run_check(
            attempt, checkpoint, captured, key="stable-run-key"
        )
        self.ledger.record_action(
            attempt["id"],
            ActionKind.SUBMITTED.value,
            {"submission_digest": captured.sha256},
            idempotency_key="submit-after-run",
            now=BASE + timedelta(minutes=4),
        )
        with patch.object(
            self.registry,
            "run",
            side_effect=AssertionError("runner must not be reinvoked"),
        ) as invoke:
            exact = self.run_check(
                attempt,
                checkpoint,
                captured,
                key="stable-run-key",
            )
            varied = self.run_check(
                attempt,
                checkpoint,
                captured,
                key="new-caller-key",
            )
        invoke.assert_not_called()
        self.assertEqual(exact["claim_id"], first["claim_id"])
        self.assertEqual(varied["claim_id"], first["claim_id"])
        self.assertTrue(exact["idempotent_replay"])
        self.assertTrue(varied["idempotent_replay"])
        self.assertEqual(exact["status"], "completed")

    def test_trace_race_rolls_back_terminal_observation(self) -> None:
        attempt, checkpoint, captured = self.start_with_artifact(
            artifact([[True]])
        )
        original_run = self.registry.run

        def race(request, material):
            self.ledger.record_action(
                attempt["id"],
                ActionKind.HINT_REQUESTED.value,
                {"hint_id": "race_hint", "level": 1},
                idempotency_key="race-action",
                now=BASE + timedelta(minutes=4),
            )
            return original_run(request, material)

        with patch.object(self.registry, "run", side_effect=race):
            with self.assertRaisesRegex(
                ConflictError, "claim remains unresolved"
            ):
                self.run_check(attempt, checkpoint, captured)
        runs = self.ledger.list_artifact_runs(attempt_id=attempt["id"])
        self.assertEqual(runs[0]["status"], "unresolved")
        self.assertEqual(
            [item["action_type"] for item in self.ledger.list_actions(attempt["id"])],
            ["started", "artifact_checkpoint", "hint_requested"],
        )
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM performance_artifact_run_receipts"
                ).fetchone()["n"],
                0,
            )

    def test_process_timestamps_bound_the_actual_registry_call(self) -> None:
        attempt = self.ledger.start_attempt(
            self.session["id"],
            self.task.id,
            task_version=self.task.version,
            task_release_id=self.release_report["release_id"],
            now=datetime.now(timezone.utc),
        )
        captured = snapshot(artifact([[True]]))
        checkpoint = self.ledger.record_action(
            attempt["id"],
            ActionKind.ARTIFACT_CHECKPOINT.value,
            {
                "artifact_digest": captured.sha256,
                "artifact_kind": self.binding.artifact_kind,
            },
            now=datetime.now(timezone.utc),
        )
        original_run = self.registry.run
        observed: dict[str, datetime] = {}

        def timed(request, material):
            observed["entered"] = datetime.now(timezone.utc)
            receipt = original_run(request, material)
            observed["returned"] = datetime.now(timezone.utc)
            return receipt

        with patch.object(self.registry, "run", side_effect=timed):
            result = self.ledger.run_artifact_check(
                attempt["id"],
                captured,
                self.registry,
                self.binding,
                check_set_id=CAUSAL_MASK_CHECK_SET_ID,
                artifact_action_id=checkpoint["id"],
            )
        started = datetime.fromisoformat(
            result["operational_receipt"]["started_at"]
        )
        completed = datetime.fromisoformat(
            result["operational_receipt"]["completed_at"]
        )
        self.assertLessEqual(started, observed["entered"])
        self.assertGreaterEqual(completed, observed["returned"])

    def test_list_and_inspect_reject_orphan_observation_event(self) -> None:
        attempt, checkpoint, captured = self.start_with_artifact(
            artifact([[True]])
        )
        run = self.run_check(attempt, checkpoint, captured)
        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER performance_artifact_run_receipts_no_delete"
            )
            connection.execute(
                """DELETE FROM performance_artifact_run_receipts
                   WHERE claim_id=?""",
                (run["claim_id"],),
            )
        with self.assertRaisesRegex(
            ConflictError, "missing its receipt projection"
        ):
            self.ledger.list_artifact_runs(attempt_id=attempt["id"])
        with self.assertRaisesRegex(
            ConflictError, "missing its receipt projection"
        ):
            self.ledger.inspect_artifact_run(run["claim_id"])

    def test_manifest_and_action_mismatches_fail_before_admission(self) -> None:
        attempt, checkpoint, captured = self.start_with_artifact(
            artifact([[True]]), key_suffix="action-mismatch"
        )
        wrong = snapshot(artifact([[False]]))
        with self.assertRaisesRegex(
            ValidationError, "does not match the captured artifact"
        ):
            self.ledger.run_artifact_check(
                attempt["id"],
                wrong,
                self.registry,
                self.binding,
                check_set_id=CAUSAL_MASK_CHECK_SET_ID,
                artifact_action_id=checkpoint["id"],
                now=BASE + timedelta(minutes=3),
            )
        self.ledger.record_action(
            attempt["id"],
            ActionKind.ABANDONED.value,
            {"reason_code": "mismatch_fixture_complete"},
            now=BASE + timedelta(minutes=4),
        )

        bad_contract = replace(
            self.contract,
            artifact_manifests=((self.binding.artifact_kind, _D1),),
        )
        bad_task = replace(
            self.task,
            id="task_bad_artifact_manifest",
            family_id="family_bad_artifact_manifest",
            scorer_contracts=(bad_contract,),
        )
        published = self.ledger.publish_release(
            PerformanceTaskRelease(
                title="Mismatched artifact manifest",
                corpus_release_id=self.corpus_release_id,
                review=self.release.review,
                tasks=(("pilot", bad_task),),
            ),
            now=BASE,
        )
        attempt = self.ledger.start_attempt(
            self.session["id"],
            bad_task.id,
            task_version=bad_task.version,
            task_release_id=published["release_id"],
            now=BASE + timedelta(minutes=5),
        )
        captured = snapshot(artifact([[True, False], [True, True]]))
        checkpoint = self.ledger.record_action(
            attempt["id"],
            "artifact_checkpoint",
            {
                "artifact_digest": captured.sha256,
                "artifact_kind": self.binding.artifact_kind,
            },
            now=BASE + timedelta(minutes=6),
        )
        with self.assertRaisesRegex(
            ValidationError, "exact released deterministic scorer contract"
        ):
            self.ledger.run_artifact_check(
                attempt["id"],
                captured,
                self.registry,
                self.binding,
                check_set_id=CAUSAL_MASK_CHECK_SET_ID,
                artifact_action_id=checkpoint["id"],
                now=BASE + timedelta(minutes=7),
            )
        self.assertEqual(self.ledger.list_artifact_runs(), [])

    def _provider_registry(
        self,
        observation: ImportedEvaluation,
    ) -> ScoringProviderRegistry:
        class Provider:
            provider_id = self.authority.provider_id
            provider_version = self.authority.provider_version
            declared_kind = ScorerKind.DETERMINISTIC
            synthetic = False

            def score(self, _request):
                return observation

        registry = ScoringProviderRegistry()
        registry.register(Provider(), self.authority)
        return registry

    def test_manual_check_is_rejected_but_system_receipt_is_accepted(self) -> None:
        attempt, _checkpoint, captured = self.start_with_artifact(
            artifact([[True]])
        )
        manual = self.ledger.record_action(
            attempt["id"],
            ActionKind.CHECK_RUN.value,
            {
                "check_set_id": self.binding.check_set_id,
                "passed": 2,
                "failed": 0,
                "errored": 0,
                "skipped": 0,
                "result_digest": _D2,
            },
            now=BASE + timedelta(minutes=3),
        )
        self.ledger.record_action(
            attempt["id"],
            ActionKind.SUBMITTED.value,
            {"submission_digest": captured.sha256},
            now=BASE + timedelta(minutes=4),
        )
        manual_observation = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_causal_mask",
                    status=EvaluationStatus.VALID,
                    score=1.0,
                    outcome_code="manual_claimed_pass",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(manual["id"],),
                ),
            )
        )
        with self.assertRaisesRegex(
            ValidationError, "manually asserted"
        ):
            self.ledger.score_attempt(
                attempt["id"],
                self._provider_registry(manual_observation),
                self.authority.provider_id,
                self.authority.provider_version,
                now=BASE + timedelta(minutes=5),
            )

        self.engine.create_learner("system-receipt-learner", "System Receipt")
        self.session = self.engine.start_session(
            "system-receipt-learner", "t_transformers", seed=814
        )
        attempt, checkpoint, captured = self.start_with_artifact(
            artifact([[True]]), key_suffix="system"
        )
        run = self.run_check(
            attempt, checkpoint, captured, key="system-run"
        )
        self.ledger.record_action(
            attempt["id"],
            ActionKind.SUBMITTED.value,
            {"submission_digest": captured.sha256},
            now=BASE + timedelta(minutes=4),
        )
        system_observation = ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_causal_mask",
                    status=EvaluationStatus.VALID,
                    score=1.0,
                    outcome_code="runner_observed_pass",
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(run["check_action"]["id"],),
                ),
            )
        )
        scored = self.ledger.score_attempt(
            attempt["id"],
            self._provider_registry(system_observation),
            self.authority.provider_id,
            self.authority.provider_version,
            now=BASE + timedelta(minutes=5),
        )
        self.assertGreater(
            scored["shadow_evidence"]["total_evidence_weight"], 0.0
        )

    def test_cli_intake_failure_creates_no_database_and_leaks_no_path(self) -> None:
        database_path = Path(self.tempdir.name) / "must-not-exist.db"
        missing = Path(self.tempdir.name) / "private-missing-artifact.json"
        stderr = io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "--db",
                    str(database_path),
                    "task",
                    "run-check",
                    "attempt_private",
                    str(missing),
                    "--check-set",
                    CAUSAL_MASK_CHECK_SET_ID,
                ]
            )
        self.assertEqual(exit_code, 2)
        self.assertFalse(database_path.exists())
        self.assertNotIn(str(missing), stderr.getvalue())
        self.assertNotIn(missing.name, stderr.getvalue())

    def test_list_and_inspect_are_read_only_and_explicitly_non_authoritative(
        self,
    ) -> None:
        attempt, checkpoint, captured = self.start_with_artifact(
            artifact([[True]])
        )
        run = self.run_check(attempt, checkpoint, captured)
        before = self.database.verify_integrity()
        listed = PerformanceLedger(
            Database(self.database.path, read_only=True)
        ).list_artifact_runs(attempt_id=attempt["id"], status="completed")
        inspected = PerformanceLedger(
            Database(self.database.path, read_only=True)
        ).inspect_artifact_run(run["run_id"])
        after = self.database.verify_integrity()

        self.assertEqual(len(listed), 1)
        self.assertEqual(inspected["claim_id"], run["claim_id"])
        self.assertFalse(inspected["retry_allowed"])
        self.assertFalse(inspected["skill_authority"])
        self.assertTrue(inspected["shadow_only"])
        self.assertTrue(inspected["process_boundary_configured"])
        self.assertTrue(inspected["worker_process_started"])
        self.assertEqual(before, after)

        stdout = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            exit_code = main(
                [
                    "--db",
                    str(self.database.path),
                    "task",
                    "run",
                    run["claim_id"],
                ]
            )
        self.assertEqual(exit_code, 0)
        self.assertIn("Process boundary configured: yes.", stdout.getvalue())
        self.assertIn("Worker process started: yes.", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
