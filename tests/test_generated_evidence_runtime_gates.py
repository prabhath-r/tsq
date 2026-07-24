# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch

from tests import test_generated_activation as generated_activation_tests
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError
from tsq.evidence import (
    ActionPhase,
    CriterionScale,
    EvaluationStatus,
    LearningTask,
    RubricCriterion,
    TaskModality,
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
from tsq.performance_selection import recommend_performance_tasks
from tsq.store import Database


_D0 = "0" * 64
_D1 = "1" * 64
_D2 = "2" * 64
_D3 = "3" * 64


class GeneratedEvidenceRuntimeGateTestCase(unittest.TestCase):
    """Exercise gates that must remain closed after legacy evidence is found."""

    def setUp(self) -> None:
        self.activation_fixture = (
            generated_activation_tests.TypedGeneratedActivationTestCase(
                methodName=(
                    "test_historical_generated_evidence_is_marked_once_without_rewrite"
                )
            )
        )
        self.activation_fixture.setUp()
        self.addCleanup(self.activation_fixture.tearDown)
        self.database = self.activation_fixture.database
        # This helper deliberately recreates behavior accepted by the old trust
        # boundary. Keep the new runtime gate out of that historical setup.
        with patch.object(
            self.database,
            "require_learner_evidence_safe",
            return_value=None,
        ):
            self.contamination = (
                self.activation_fixture.create_historical_contaminated_response()
            )
        self.learner_id = self.contamination["learner_id"]
        self.engine = AdaptiveEngine(self.database)
        self.base_time = datetime.now(timezone.utc) + timedelta(minutes=1)

    def _publish_task(
        self,
    ) -> tuple[PerformanceLedger, LearningTask, dict[str, object], dict[str, object]]:
        with self.database.read() as connection:
            corpus_release_id = connection.execute(
                """SELECT value FROM meta
                   WHERE key='active_corpus_release'"""
            ).fetchone()["value"]
            source = connection.execute(
                """SELECT id, content_hash FROM sources
                   WHERE id='src_tiny'"""
            ).fetchone()
        task = LearningTask(
            id="task_generated_evidence_gate",
            version=1,
            family_id="family_generated_evidence_gate",
            title="Exercise a historical-evidence runtime gate",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_generated_evidence_gate",
                    name="Target invariant",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_target", 1.0),),
                    dependence_group="generated_evidence_gate",
                    evidence_cap=0.8,
                    dependence_cap=0.8,
                ),
            ),
            instructions=(
                "Inspect the pinned target stimulus and submit a content-addressed "
                "diagnostic artifact."
            ),
            source_manifests=((source["id"], source["content_hash"]),),
            administration_id="generated_evidence_gate_admin",
            administration_manifest_digest=_D0,
            stimulus_id="generated_evidence_gate_stimulus",
            stimulus_digest=_D1,
        )
        ledger = PerformanceLedger(self.database)
        release = ledger.publish_release(
            PerformanceTaskRelease(
                title="Generated-evidence gate fixture",
                corpus_release_id=corpus_release_id,
                review=TaskReleaseReview(
                    reviewer_kind="human",
                    reviewer_id="independent_gate_reviewer",
                    reviewed_at=self.base_time.isoformat(),
                    independent_of_author=True,
                    attestation_digest=_D2,
                ),
                tasks=(("pilot", task),),
            ),
            now=self.base_time,
        )
        # Use a clean, empty session for productive probes. The selected-response
        # contamination remains attached to the same learner in the old session.
        with patch.object(
            self.database,
            "require_learner_evidence_safe",
            return_value=None,
        ):
            session = self.engine.start_session(
                self.learner_id,
                "c_target",
                seed=7301,
                now=self.base_time,
            )
        return ledger, task, release, session

    def _submitted_attempt(
        self,
        ledger: PerformanceLedger,
        task: LearningTask,
        release: dict[str, object],
        session: dict[str, object],
        *,
        suffix: str,
        offset: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        with patch.object(
            self.database,
            "require_learner_evidence_safe",
            return_value=None,
        ):
            attempt = ledger.start_attempt(
                str(session["id"]),
                task.id,
                task_version=task.version,
                task_release_id=str(release["release_id"]),
                idempotency_key=f"start-{suffix}",
                now=self.base_time + timedelta(seconds=offset),
            )
            ledger.record_action(
                str(attempt["id"]),
                "artifact_checkpoint",
                {
                    "artifact_digest": _D3,
                    "artifact_kind": "diagnostic_digest",
                },
                idempotency_key=f"artifact-{suffix}",
                now=self.base_time + timedelta(seconds=offset + 1),
            )
            submitted = ledger.record_action(
                str(attempt["id"]),
                "submitted",
                {"submission_digest": _D3},
                idempotency_key=f"submit-{suffix}",
                now=self.base_time + timedelta(seconds=offset + 2),
            )
        return attempt, submitted

    @staticmethod
    def _evaluation(
        source_action_id: str,
        *,
        outcome_code: str,
    ) -> ImportedEvaluation:
        return ImportedEvaluation(
            criteria=(
                ImportedCriterionResult(
                    criterion_id="criterion_generated_evidence_gate",
                    status=EvaluationStatus.VALID,
                    score=0.8,
                    outcome_code=outcome_code,
                    phase=ActionPhase.UNASSISTED,
                    source_action_ids=(source_action_id,),
                    reliability=0.9,
                ),
            )
        )

    def test_score_idempotent_result_is_blocked_after_quarantine(self) -> None:
        ledger, task, release, session = self._publish_task()
        attempt, submitted = self._submitted_attempt(
            ledger,
            task,
            release,
            session,
            suffix="score-prior",
            offset=1,
        )
        imported = self._evaluation(
            str(submitted["id"]),
            outcome_code="score_prior",
        )
        provider = SyntheticDeterministicProvider(
            imported,
            provider_id="synthetic.generated-evidence-prior",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        with patch.object(
            self.database,
            "require_learner_evidence_safe",
            return_value=None,
        ):
            ledger.score_attempt(
                str(attempt["id"]),
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="score-generated-evidence-prior",
                now=self.base_time + timedelta(seconds=4),
            )
        self.database.initialize()

        with self.assertRaisesRegex(
            ConflictError,
            "quarantined generated-question evidence",
        ):
            ledger.score_attempt(
                str(attempt["id"]),
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="score-generated-evidence-prior",
                now=self.base_time + timedelta(seconds=4),
            )

    def test_import_idempotent_result_is_blocked_after_quarantine(self) -> None:
        ledger, task, release, session = self._publish_task()
        attempt, submitted = self._submitted_attempt(
            ledger,
            task,
            release,
            session,
            suffix="import-prior",
            offset=1,
        )
        imported = self._evaluation(
            str(submitted["id"]),
            outcome_code="import_prior",
        )
        with patch.object(
            self.database,
            "require_learner_evidence_safe",
            return_value=None,
        ):
            ledger.import_evaluation(
                str(attempt["id"]),
                imported,
                provider_id="import.generated-evidence-prior",
                provider_version="v1",
                idempotency_key="import-generated-evidence-prior",
                now=self.base_time + timedelta(seconds=4),
            )
        self.database.initialize()

        with self.assertRaisesRegex(
            ConflictError,
            "quarantined generated-question evidence",
        ):
            ledger.import_evaluation(
                str(attempt["id"]),
                imported,
                provider_id="import.generated-evidence-prior",
                provider_version="v1",
                idempotency_key="import-generated-evidence-prior",
                now=self.base_time + timedelta(seconds=4),
            )

    def test_quarantine_during_provider_call_blocks_result_commit(self) -> None:
        ledger, task, release, session = self._publish_task()
        attempt, submitted = self._submitted_attempt(
            ledger,
            task,
            release,
            session,
            suffix="provider-race",
            offset=1,
        )
        imported = self._evaluation(
            str(submitted["id"]),
            outcome_code="provider_race",
        )
        state = {"provider_called": False, "quarantined": False}
        database_path = self.database.path

        class QuarantiningProvider(SyntheticDeterministicProvider):
            def score(self, request):
                state["provider_called"] = True
                Database(database_path).initialize()
                state["quarantined"] = True
                return super().score(request)

        provider = QuarantiningProvider(
            imported,
            provider_id="synthetic.generated-evidence-race",
        )
        registry = ScoringProviderRegistry(allow_synthetic=True)
        registry.register(provider, provider.authority_binding)
        real_gate = self.database.require_learner_evidence_safe

        def staged_gate(learner_id, connection):
            if not state["quarantined"]:
                return None
            return real_gate(learner_id, connection)

        with (
            patch.object(
                self.database,
                "require_learner_evidence_safe",
                side_effect=staged_gate,
            ),
            self.assertRaisesRegex(
                ConflictError,
                "quarantined generated-question evidence",
            ),
        ):
            ledger.score_attempt(
                str(attempt["id"]),
                registry,
                provider.provider_id,
                provider.provider_version,
                idempotency_key="score-generated-evidence-race",
                now=self.base_time + timedelta(seconds=4),
            )

        self.assertTrue(state["provider_called"])
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

    def test_profile_rechecks_safety_after_projection_reads(self) -> None:
        real_gate = self.database.require_learner_evidence_safe
        original_get_graph = self.database.get_graph
        state = {"gate_calls": 0, "quarantined": False}

        def staged_gate(learner_id, connection):
            state["gate_calls"] += 1
            if state["gate_calls"] == 1:
                return None
            return real_gate(learner_id, connection)

        def quarantine_then_get_graph(*args, **kwargs):
            self.database.initialize()
            state["quarantined"] = True
            return original_get_graph(*args, **kwargs)

        with (
            patch.object(
                self.database,
                "require_learner_evidence_safe",
                side_effect=staged_gate,
            ),
            patch.object(
                self.database,
                "get_graph",
                side_effect=quarantine_then_get_graph,
            ),
            self.assertRaisesRegex(
                ConflictError,
                "quarantined generated-question evidence",
            ),
        ):
            self.engine.profile(
                self.learner_id,
                root_concept_id="c_target",
                now=self.base_time,
            )

        self.assertTrue(state["quarantined"])
        self.assertGreaterEqual(state["gate_calls"], 2)

    def test_task_recommendation_rechecks_safety_before_return(self) -> None:
        _, _, _, session = self._publish_task()
        real_gate = self.database.require_learner_evidence_safe
        original_get_graph = self.database.get_graph
        state = {"gate_calls": 0, "quarantined": False}

        def staged_gate(learner_id, connection):
            state["gate_calls"] += 1
            if state["gate_calls"] == 1:
                return None
            return real_gate(learner_id, connection)

        def quarantine_then_get_graph(*args, **kwargs):
            self.database.initialize()
            state["quarantined"] = True
            return original_get_graph(*args, **kwargs)

        with (
            patch.object(
                self.database,
                "require_learner_evidence_safe",
                side_effect=staged_gate,
            ),
            patch.object(
                self.database,
                "get_graph",
                side_effect=quarantine_then_get_graph,
            ),
            self.assertRaisesRegex(
                ConflictError,
                "quarantined generated-question evidence",
            ),
        ):
            recommend_performance_tasks(
                self.database,
                str(session["id"]),
                now=self.base_time + timedelta(seconds=1),
            )

        self.assertTrue(state["quarantined"])
        self.assertGreaterEqual(state["gate_calls"], 2)


if __name__ == "__main__":
    unittest.main()
