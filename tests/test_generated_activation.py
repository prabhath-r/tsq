# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tests.test_store_integrity import tiny_corpus
from tsq.corpus import read_and_parse, validate_bundle
from tsq.engine import AdaptiveEngine
from tsq.errors import ConflictError, ValidationError
from tsq.evidence import (
    CriterionScale,
    LearningTask,
    RubricCriterion,
    TaskModality,
)
from tsq.models import QuestionKind, QuestionStatus
from tsq.performance_ledger import (
    PerformanceLedger,
    PerformanceTaskRelease,
    TaskReleaseReview,
)
from tsq.performance_reporting import productive_shadow_summary
from tsq.provenance import (
    LEGACY_UNATTESTED_COHORT_SHA256,
    legacy_unattested_cohort_digest,
)
from tsq.replay import ProjectionReplay
from tsq.store import (
    HISTORICAL_GENERATED_EVIDENCE_KEY_PREFIX,
    HISTORICAL_GENERATED_EVIDENCE_POLICY,
    LEGACY_UNREVIEWED_GENERATED_REVOCATION_KEY_PREFIX,
    LEGACY_UNREVIEWED_GENERATED_REVOCATION_REASON,
    Database,
    _legacy_question_identity,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
ATTESTATION = "a" * 64


def activation_review() -> dict[str, object]:
    return {
        "reviewer_kind": "human",
        "reviewer_id": "human-reviewer-17",
        "reviewed_at": "2026-07-23T12:00:00+00:00",
        "independent_of_author": True,
        "attestation_digest": ATTESTATION,
    }


def active_generated_provenance() -> dict[str, object]:
    return {
        "generated": True,
        "human_review": True,
        "activation_review": activation_review(),
    }


class RawGeneratedActivationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = json.loads(CORPUS.read_text(encoding="utf-8"))
        self.question = next(
            question
            for question in self.bundle["questions"]
            if question["status"] == "approved"
            and question.get("provenance", {}).get("generated") is not True
        )

    def provenance_codes(self) -> set[str]:
        return {
            issue.code
            for issue in validate_bundle(self.bundle)
            if issue.question_id == self.question["id"]
            and (
                "provenance" in issue.code
                or "review" in issue.code
                or issue.code.endswith("_flag_type")
            )
        }

    def test_quarantined_generated_artifact_needs_no_activation_claim(self) -> None:
        self.question["status"] = "quarantined"
        self.question["provenance"] = {
            "generated": True,
            "human_review": False,
        }

        self.assertEqual(self.provenance_codes(), set())

    def test_exact_packaged_legacy_cohort_retains_compatibility(self) -> None:
        legacy = [
            question
            for question in self.bundle["questions"]
            if "generated" not in question.get("provenance", {})
        ]

        self.assertEqual(len(legacy), 222)
        self.assertFalse(
            any(
                issue.code == "generated_provenance_required"
                for issue in validate_bundle(self.bundle)
            )
        )

    def test_exact_legacy_member_can_be_retired_from_a_later_bundle(self) -> None:
        retired_id = "q_conditional_independence_common_cause_001"
        self.bundle["questions"] = [
            question
            for question in self.bundle["questions"]
            if question["id"] != retired_id
        ]

        self.assertFalse(
            any(
                issue.code == "generated_provenance_required"
                for issue in validate_bundle(self.bundle)
            )
        )

    def test_raw_and_typed_paths_agree_on_packaged_cohort(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        typed_legacy = [
            question
            for question in parsed[4]
            if "generated" not in question.provenance
        ]

        self.assertFalse(validate_bundle(self.bundle))
        self.assertEqual(
            legacy_unattested_cohort_digest(
                _legacy_question_identity(question)
                for question in typed_legacy
            ),
            LEGACY_UNATTESTED_COHORT_SHA256,
        )

    def test_legacy_content_change_invalidates_compact_cohort_commitment(
        self,
    ) -> None:
        self.question["stem"] += " Unattested content mutation."

        self.assertIn(
            "generated_provenance_required",
            self.provenance_codes(),
        )

    def test_legacy_objective_binding_change_invalidates_cohort(self) -> None:
        self.question = next(
            question
            for question in self.bundle["questions"]
            if "generated" not in question.get("provenance", {})
            and question.get("learning_objective_id")
        )
        self.question["learning_objective_id"] += "_mutated"

        self.assertIn(
            "generated_provenance_required",
            self.provenance_codes(),
        )

    def test_legacy_diagnostic_binding_change_invalidates_cohort(self) -> None:
        self.question = next(
            question
            for question in self.bundle["questions"]
            if "generated" not in question.get("provenance", {})
            and any(
                "diagnostic_objective_id" in option
                for option in question["options"]
            )
        )
        option = next(
            option
            for option in self.question["options"]
            if "diagnostic_objective_id" in option
        )
        option["diagnostic_objective_id"] += "_mutated"

        self.assertIn(
            "generated_provenance_required",
            self.provenance_codes(),
        )

    def test_legacy_status_demotion_does_not_change_content_identity(self) -> None:
        self.question["status"] = "quarantined"

        self.assertNotIn(
            "generated_provenance_required",
            self.provenance_codes(),
        )

    def test_new_revision_must_explicitly_declare_generation_provenance(
        self,
    ) -> None:
        revision = deepcopy(self.question)
        revision.update(
            {
                "id": self.question["id"] + "_revision",
                "version": self.question["version"] + 1,
                "status": "quarantined",
                "stem": self.question["stem"] + " New revision.",
                "revision_of": self.question["id"],
            }
        )
        self.bundle["questions"].append(revision)
        self.question = revision

        self.assertIn(
            "generated_provenance_required",
            self.provenance_codes(),
        )

        revision["provenance"] = {
            **revision["provenance"],
            "generated": False,
        }
        self.assertFalse(
            any(
                issue.code == "generated_provenance_required"
                for issue in validate_bundle(self.bundle)
            )
        )

    def test_active_generated_item_rejects_status_only_promotion(self) -> None:
        self.question["provenance"] = {
            "generated": True,
            "human_review": False,
        }

        self.assertEqual(
            self.provenance_codes(),
            {
                "activation_review_required",
                "generated_human_review_required",
            },
        )

    def test_generated_and_human_review_flags_are_exact_booleans(self) -> None:
        self.question["provenance"] = {
            "generated": "true",
            "human_review": "true",
        }

        self.assertEqual(
            self.provenance_codes(),
            {
                "generated_flag_type",
                "human_review_flag_type",
            },
        )

    def test_activation_review_rejects_malformed_claims(self) -> None:
        self.question["provenance"] = active_generated_provenance()
        self.question["provenance"]["activation_review"] = {
            "reviewer_kind": "model",
            "reviewer_id": " ",
            "reviewed_at": "2026-07-23",
            "independent_of_author": "true",
            "attestation_digest": "NOT-A-DIGEST",
        }

        self.assertEqual(
            self.provenance_codes(),
            {
                "activation_review_digest",
                "activation_review_independence",
                "activation_review_reviewer_id",
                "activation_review_reviewer_kind",
                "activation_review_timestamp",
            },
        )

    def test_activation_review_requires_exact_fields(self) -> None:
        self.question["provenance"] = active_generated_provenance()
        review = self.question["provenance"]["activation_review"]
        assert isinstance(review, dict)
        review.pop("reviewer_id")
        review["model_verdict"] = "accept"

        self.assertEqual(
            self.provenance_codes(),
            {
                "activation_review_extra_fields",
                "activation_review_missing_fields",
                "activation_review_reviewer_id",
            },
        )

    def test_valid_human_review_commitment_passes_raw_boundary(self) -> None:
        self.question["provenance"] = active_generated_provenance()

        self.assertEqual(self.provenance_codes(), set())


class TypedGeneratedActivationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "activation.db")
        self.database.initialize()
        self.bundle = tiny_corpus()
        self.expert = self.bundle[-1][0]
        self.generated = replace(
            self.expert,
            id="q_generated",
            family_id="family_generated",
            status=QuestionStatus.QUARANTINED,
            stem=self.expert.stem + " This is the generated legacy variant.",
            kind=QuestionKind.APPLICATION,
            provenance={"generated": True, "human_review": False},
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def serviceable_expert_bank(self):
        variants = (
            ("q_safe_application", QuestionKind.APPLICATION),
            ("q_safe_debugging", QuestionKind.DEBUGGING),
            ("q_safe_transfer", QuestionKind.TRANSFER),
        )
        return (
            self.expert,
            *(
                replace(
                    self.expert,
                    id=question_id,
                    family_id=f"family_{question_id}",
                    stem=self.expert.stem + f" Independent variant {index}.",
                    kind=kind,
                )
                for index, (question_id, kind) in enumerate(variants, start=1)
            ),
        )

    def import_legacy_active_generated_release(self):
        legacy_generated = replace(
            self.generated,
            status=QuestionStatus.APPROVED,
        )
        questions = (*self.serviceable_expert_bank(), legacy_generated)
        # Recreate a release accepted by the historical trust boundary while
        # retaining every other current corpus invariant.
        with patch("tsq.store.question_provenance_issues", return_value=()):
            result = self.database.import_corpus(
                *self.bundle[:-1],
                questions,
            )
        self.assertEqual(result["legacy_generated_revocations"], 0)
        return result, questions

    def create_historical_contaminated_response(self):
        """Exercise the trust boundary that existed before runtime gating."""

        release, _ = self.import_legacy_active_generated_release()
        engine = AdaptiveEngine(self.database)
        presentation = None
        learner_id = ""
        session = None
        submission = None
        pending = None
        with (
            patch(
                "tsq.store.question_runtime_activation_safe",
                return_value=True,
            ),
            patch(
                "tsq.engine.question_runtime_activation_safe",
                return_value=True,
            ),
            patch.object(
                Database,
                "require_learner_evidence_safe",
                return_value=None,
            ),
        ):
            for seed in range(1, 101):
                learner_id = f"historical-contamination-{seed}"
                engine.create_learner(learner_id)
                session = engine.start_session(
                    learner_id,
                    "c_target",
                    seed=seed,
                )
                candidate = engine.next_question(session["id"])
                if candidate.question.id != self.generated.id:
                    continue
                presentation = candidate
                submission = engine.submit_answer(
                    candidate.decision_id,
                    candidate.question.correct_option.id,
                    confidence=0.8,
                    response_ms=0,
                    idempotency_key=(
                        f"historical-generated-answer-{seed}"
                    ),
                )
                pending = engine.next_question(session["id"])
                break

        self.assertIsNotNone(presentation)
        self.assertIsNotNone(session)
        self.assertIsNotNone(submission)
        self.assertIsNotNone(pending)
        assert presentation is not None
        assert session is not None
        assert submission is not None
        assert pending is not None
        self.assertTrue(submission.correct)
        with self.database.read() as connection:
            attempt = connection.execute(
                """SELECT * FROM attempts WHERE decision_id=?""",
                (presentation.decision_id,),
            ).fetchone()
        self.assertIsNotNone(attempt)
        assert attempt is not None
        self.assertEqual(submission.interaction_id, attempt["id"])
        return {
            "release": release,
            "engine": engine,
            "learner_id": learner_id,
            "session": session,
            "presentation": presentation,
            "submission": submission,
            "pending": pending,
            "attempt_id": attempt["id"],
            "response_event_id": attempt["event_id"],
            "answer_idempotency_key": (
                "historical-generated-answer-"
                + learner_id.removeprefix("historical-contamination-")
            ),
        }

    @staticmethod
    def contaminated_evidence_snapshot(
        database: Database,
        learner_id: str,
        attempt_id: str,
    ) -> dict[str, object]:
        with database.read() as connection:
            attempt = connection.execute(
                "SELECT * FROM attempts WHERE id=?",
                (attempt_id,),
            ).fetchone()
            response = connection.execute(
                """SELECT * FROM events
                   WHERE event_id=(
                       SELECT event_id FROM attempts WHERE id=?
                   )""",
                (attempt_id,),
            ).fetchone()
            learner = connection.execute(
                "SELECT * FROM learners WHERE id=?",
                (learner_id,),
            ).fetchone()
            projection_hash = database.learner_projection_hash(
                learner_id,
                connection,
            )
        assert attempt is not None
        assert response is not None
        assert learner is not None
        return {
            "attempt": dict(attempt),
            "response": dict(response),
            "learner": dict(learner),
            "projection_hash": projection_hash,
        }

    def start_performance_attempt_before_quarantine(
        self,
        fixture: dict[str, object],
    ) -> dict[str, object]:
        """Open a released productive task before contamination is classified."""

        engine = fixture["engine"]
        assert isinstance(engine, AdaptiveEngine)
        learner_id = fixture["learner_id"]
        assert isinstance(learner_id, str)
        base = datetime.now(timezone.utc)
        session = engine.start_session(
            learner_id,
            "c_target",
            seed=8101,
            now=base,
        )
        release = fixture["release"]
        assert isinstance(release, dict)
        corpus_release_id = release["release_id"]
        with self.database.read() as connection:
            source = connection.execute(
                """SELECT id, content_hash FROM sources
                   WHERE id='src_tiny'"""
            ).fetchone()
        self.assertIsNotNone(source)
        task = LearningTask(
            id="task_contaminated_runtime_guard",
            version=1,
            family_id="family_contaminated_runtime_guard",
            title="Inspect a target prerequisite invariant",
            modality=TaskModality.DEBUGGING,
            criteria=(
                RubricCriterion(
                    id="criterion_contaminated_runtime_guard",
                    name="Target prerequisite invariant",
                    scale=CriterionScale.CONTINUOUS,
                    concept_weights=(("c_target", 1.0),),
                    dependence_group="contaminated_runtime_guard",
                ),
            ),
            instructions=(
                "Inspect the pinned target-prerequisite trace and record "
                "the observed invariant."
            ),
            source_manifests=((source["id"], source["content_hash"]),),
            administration_id="admin_contaminated_runtime_guard",
            administration_manifest_digest="b" * 64,
            stimulus_id="stimulus_contaminated_runtime_guard",
            stimulus_digest="c" * 64,
        )
        task_release = PerformanceTaskRelease(
            title="Contaminated learner runtime guard",
            corpus_release_id=corpus_release_id,
            review=TaskReleaseReview(
                reviewer_kind="human",
                reviewer_id="independent-contamination-reviewer",
                reviewed_at=base.isoformat(),
                independent_of_author=True,
                attestation_digest="d" * 64,
            ),
            tasks=(("pilot", task),),
        )
        ledger = PerformanceLedger(self.database)
        published = ledger.publish_release(
            task_release,
            now=base + timedelta(seconds=1),
        )
        start_key = "contaminated-performance-start"
        attempt = ledger.start_attempt(
            session["id"],
            task.id,
            task_version=task.version,
            task_release_id=published["release_id"],
            idempotency_key=start_key,
            now=base + timedelta(seconds=2),
        )
        return {
            "ledger": ledger,
            "session": session,
            "task": task,
            "task_release": published,
            "attempt": attempt,
            "start_key": start_key,
            "base": base,
        }

    def test_historical_generated_evidence_is_marked_once_without_rewrite(
        self,
    ) -> None:
        fixture = self.create_historical_contaminated_response()
        before = self.contaminated_evidence_snapshot(
            self.database,
            fixture["learner_id"],
            fixture["attempt_id"],
        )
        reopened = Database(self.database.path)

        reopened.initialize()

        with reopened.read() as connection:
            markers = connection.execute(
                """SELECT * FROM events
                   WHERE event_type='ResponseEvidenceQuarantined'
                     AND correlation_id=?
                   ORDER BY event_id""",
                (fixture["attempt_id"],),
            ).fetchall()
        self.assertEqual(len(markers), 1)
        marker = dict(markers[0])
        self.assertEqual(marker["schema_version"], 1)
        self.assertEqual(
            marker["stream_id"],
            f"learner:{fixture['learner_id']}",
        )
        self.assertEqual(marker["learner_id"], fixture["learner_id"])
        self.assertIsNone(marker["session_id"])
        self.assertEqual(marker["correlation_id"], fixture["attempt_id"])
        self.assertEqual(
            marker["causation_id"],
            fixture["response_event_id"],
        )
        self.assertTrue(
            marker["idempotency_key"].startswith(
                HISTORICAL_GENERATED_EVIDENCE_KEY_PREFIX
            )
        )
        marker_key_suffix = marker["idempotency_key"][
            len(HISTORICAL_GENERATED_EVIDENCE_KEY_PREFIX) :
        ]
        self.assertEqual(len(marker_key_suffix), 64)
        self.assertFalse(set(marker_key_suffix) - set("0123456789abcdef"))
        self.assertEqual(
            json.loads(marker["payload_json"]),
            {
                "attempt_id": fixture["attempt_id"],
                "response_event_id": fixture["response_event_id"],
                "learner_id": fixture["learner_id"],
                "question_id": self.generated.id,
                "reason": LEGACY_UNREVIEWED_GENERATED_REVOCATION_REASON,
                "projection_applied": False,
            },
        )
        self.assertEqual(
            json.loads(marker["metadata_json"]),
            {
                "safety_policy": HISTORICAL_GENERATED_EVIDENCE_POLICY,
                "requires_explicit_rebuild": True,
            },
        )
        self.assertEqual(
            self.contaminated_evidence_snapshot(
                reopened,
                fixture["learner_id"],
                fixture["attempt_id"],
            ),
            before,
        )

        reopened.initialize()

        with reopened.read() as connection:
            repeated = connection.execute(
                """SELECT * FROM events
                   WHERE event_type='ResponseEvidenceQuarantined'
                     AND correlation_id=?""",
                (fixture["attempt_id"],),
            ).fetchall()
        self.assertEqual([dict(row) for row in repeated], [marker])
        self.assertEqual(
            self.contaminated_evidence_snapshot(
                reopened,
                fixture["learner_id"],
                fixture["attempt_id"],
            ),
            before,
        )
        with self.assertRaisesRegex(
            sqlite3.IntegrityError,
            "events are append-only",
        ):
            with reopened.transaction() as connection:
                connection.execute(
                    """UPDATE events SET payload_json=payload_json
                       WHERE event_id=?""",
                    (marker["event_id"],),
                )

    def test_contaminated_learner_runtime_integrity_and_replay_fail_closed(
        self,
    ) -> None:
        fixture = self.create_historical_contaminated_response()
        reopened = Database(self.database.path)
        reopened.initialize()
        engine = AdaptiveEngine(reopened)
        expected = "quarantined generated-question evidence"

        with self.assertRaisesRegex(ConflictError, expected):
            engine.start_session(
                fixture["learner_id"],
                "c_target",
                seed=7001,
            )
        with self.assertRaisesRegex(ConflictError, expected):
            engine.next_question(fixture["session"]["id"])
        with self.assertRaisesRegex(ConflictError, expected):
            engine.submit_answer(
                fixture["pending"].decision_id,
                fixture["pending"].question.correct_option.id,
                confidence=0.7,
                response_ms=0,
                idempotency_key="answer-after-evidence-quarantine",
            )
        with self.assertRaisesRegex(ConflictError, expected):
            engine.profile(
                fixture["learner_id"],
                root_concept_id="c_target",
            )

        integrity = reopened.verify_integrity()
        self.assertFalse(integrity["ok"])
        explicit_error = (
            "projection contains quarantined generated-question evidence"
        )
        self.assertTrue(
            any(explicit_error in error for error in integrity["errors"]),
            integrity["errors"],
        )

        replay = ProjectionReplay(reopened).check(
            fixture["learner_id"],
        )
        self.assertFalse(replay["ok"])
        self.assertFalse(replay["rebuild_safe"])
        self.assertTrue(
            any(explicit_error in error for error in replay["errors"]),
            replay["errors"],
        )

    def test_different_existing_revocation_still_quarantines_evidence(
        self,
    ) -> None:
        fixture = self.create_historical_contaminated_response()
        manual_reason = "Independent emergency review found a content defect."
        self.database.revoke_question(
            self.generated.id,
            manual_reason,
            idempotency_key="manual-generated-revocation",
        )
        reopened = Database(self.database.path)

        reopened.initialize()

        with reopened.read() as connection:
            revocation = connection.execute(
                """SELECT reason FROM question_revocations
                   WHERE question_id=?""",
                (self.generated.id,),
            ).fetchone()
            markers = connection.execute(
                """SELECT * FROM events
                   WHERE event_type='ResponseEvidenceQuarantined'
                     AND correlation_id=?""",
                (fixture["attempt_id"],),
            ).fetchall()
        self.assertEqual(revocation["reason"], manual_reason)
        self.assertEqual(len(markers), 1)
        with self.assertRaisesRegex(
            ConflictError,
            "quarantined generated-question evidence",
        ):
            AdaptiveEngine(reopened).profile(
                fixture["learner_id"],
                root_concept_id="c_target",
            )

    def test_selection_rechecks_evidence_safety_after_scoring(self) -> None:
        fixture = self.create_historical_contaminated_response()
        engine = fixture["engine"]
        assert isinstance(engine, AdaptiveEngine)
        with patch.object(
            Database,
            "require_learner_evidence_safe",
            return_value=None,
        ):
            session = engine.start_session(
                fixture["learner_id"],
                "c_target",
                seed=8201,
            )
        original_score = engine.policy._score
        original_safety_check = (
            self.database.require_learner_evidence_safe
        )
        quarantine_installed = False
        safety_checks = 0
        scored_question_ids: list[str] = []

        def staged_safety_check(learner_id, connection):
            nonlocal safety_checks
            safety_checks += 1
            if safety_checks == 1:
                # Model the clean early snapshot. The scoring hook below
                # installs the quarantine before the final write boundary.
                return None
            return original_safety_check(learner_id, connection)

        def score_with_quarantine(question, **kwargs):
            nonlocal quarantine_installed
            scored_question_ids.append(question.id)
            if not quarantine_installed:
                Database(self.database.path).initialize()
                quarantine_installed = True
            return original_score(question, **kwargs)

        with (
            patch.object(
                self.database,
                "require_learner_evidence_safe",
                side_effect=staged_safety_check,
            ),
            patch.object(
                engine.policy,
                "_score",
                side_effect=score_with_quarantine,
            ),
        ):
            with self.assertRaisesRegex(
                ConflictError,
                "quarantined generated-question evidence",
            ):
                engine.next_question(session["id"])

        self.assertTrue(quarantine_installed)
        self.assertGreaterEqual(safety_checks, 2)
        self.assertTrue(scored_question_ids)
        self.assertNotIn(self.generated.id, scored_question_ids)
        with self.database.read() as connection:
            decisions = connection.execute(
                """SELECT COUNT(*) AS n FROM decisions
                   WHERE session_id=?""",
                (session["id"],),
            ).fetchone()["n"]
        self.assertEqual(decisions, 0)

    def test_reports_and_learning_actions_fail_closed_after_quarantine(
        self,
    ) -> None:
        fixture = self.create_historical_contaminated_response()
        self.assertNotEqual(
            fixture["pending"].question.id,
            self.generated.id,
        )
        reopened = Database(self.database.path)
        reopened.initialize()
        engine = AdaptiveEngine(reopened)
        expected = "quarantined generated-question evidence"
        with reopened.read() as connection:
            actions_before = connection.execute(
                """SELECT COUNT(*) AS n FROM learning_actions
                   WHERE decision_id=?""",
                (fixture["pending"].decision_id,),
            ).fetchone()["n"]

        operations = (
            (
                "session report",
                lambda: engine.session_report(fixture["session"]["id"]),
            ),
            (
                "learning action",
                lambda: engine.record_action(
                    fixture["pending"].decision_id,
                    "started",
                    {},
                    idempotency_key="action-after-evidence-quarantine",
                ),
            ),
        )
        for label, operation in operations:
            with self.subTest(operation=label):
                with self.assertRaisesRegex(ConflictError, expected):
                    operation()

        with reopened.read() as connection:
            actions_after = connection.execute(
                """SELECT COUNT(*) AS n FROM learning_actions
                   WHERE decision_id=?""",
                (fixture["pending"].decision_id,),
            ).fetchone()["n"]
        self.assertEqual(actions_after, actions_before)

    def test_idempotent_contaminated_answer_replay_fails_closed(self) -> None:
        fixture = self.create_historical_contaminated_response()
        reopened = Database(self.database.path)
        reopened.initialize()
        before = self.contaminated_evidence_snapshot(
            reopened,
            fixture["learner_id"],
            fixture["attempt_id"],
        )

        with self.assertRaisesRegex(
            ConflictError,
            "quarantined generated-question evidence",
        ):
            AdaptiveEngine(reopened).submit_answer(
                fixture["presentation"].decision_id,
                fixture["presentation"].question.correct_option.id,
                confidence=0.8,
                response_ms=0,
                idempotency_key=fixture["answer_idempotency_key"],
            )

        self.assertEqual(
            self.contaminated_evidence_snapshot(
                reopened,
                fixture["learner_id"],
                fixture["attempt_id"],
            ),
            before,
        )

    def test_performance_operations_and_reports_fail_closed_after_quarantine(
        self,
    ) -> None:
        with patch.object(
            Database,
            "require_learner_evidence_safe",
            return_value=None,
        ):
            fixture = self.create_historical_contaminated_response()
            performance = self.start_performance_attempt_before_quarantine(
                fixture
            )
        reopened = Database(self.database.path)
        reopened.initialize()
        ledger = PerformanceLedger(reopened)
        attempt = performance["attempt"]
        task = performance["task"]
        task_release = performance["task_release"]
        session = performance["session"]
        base = performance["base"]
        assert isinstance(attempt, dict)
        assert isinstance(task, LearningTask)
        assert isinstance(task_release, dict)
        assert isinstance(session, dict)
        assert isinstance(base, datetime)
        expected = "quarantined generated-question evidence"

        operations = (
            (
                "idempotent task start",
                lambda: ledger.start_attempt(
                    session["id"],
                    task.id,
                    task_version=task.version,
                    task_release_id=task_release["release_id"],
                    idempotency_key=performance["start_key"],
                    now=base + timedelta(seconds=2),
                ),
            ),
            (
                "performance action",
                lambda: ledger.record_action(
                    attempt["id"],
                    "hint_requested",
                    {"hint_id": "blocked_hint", "level": 1},
                    phase="assisted",
                    idempotency_key=(
                        "performance-action-after-quarantine"
                    ),
                    now=base + timedelta(seconds=3),
                ),
            ),
            (
                "attempt report",
                lambda: ledger.report(attempt["id"]),
            ),
            (
                "productive summary",
                lambda: productive_shadow_summary(
                    reopened,
                    fixture["learner_id"],
                ),
            ),
        )
        for label, operation in operations:
            with self.subTest(operation=label):
                with self.assertRaisesRegex(ConflictError, expected):
                    operation()

        with reopened.read() as connection:
            action_count = connection.execute(
                """SELECT COUNT(*) AS n FROM performance_actions
                   WHERE attempt_id=?""",
                (attempt["id"],),
            ).fetchone()["n"]
        self.assertEqual(action_count, 1)

    def test_typed_status_flip_cannot_activate_unreviewed_generation(self) -> None:
        first = self.database.import_corpus(
            *self.bundle[:-1],
            (self.expert, self.generated),
        )
        promoted = replace(
            self.generated,
            status=QuestionStatus.APPROVED,
        )

        with self.assertRaisesRegex(
            ValidationError,
            "generated_human_review_required.*activation_review_required",
        ):
            self.database.import_corpus(
                *self.bundle[:-1],
                (self.expert, promoted),
            )

        with self.database.read() as connection:
            status = connection.execute(
                "SELECT status FROM questions WHERE id='q_generated'"
            ).fetchone()["status"]
            release_count = connection.execute(
                "SELECT COUNT(*) AS n FROM corpus_releases"
            ).fetchone()["n"]
        self.assertEqual(status, "quarantined")
        self.assertEqual(release_count, 1)
        self.assertTrue(first["release_id"])

    def test_typed_fixture_requires_explicit_generation_provenance(self) -> None:
        missing_marker = replace(
            self.expert,
            provenance={"authoring_method": "uncommitted-fixture"},
        )

        with self.assertRaisesRegex(
            ValidationError,
            "generated_provenance_required",
        ):
            self.database.import_corpus(
                *self.bundle[:-1],
                (missing_marker,),
            )

    def test_typed_explicit_nongenerated_question_passes(self) -> None:
        result = self.database.import_corpus(
            *self.bundle[:-1],
            (self.expert,),
        )

        self.assertTrue(result["release_id"])

    def test_typed_packaged_legacy_cohort_retains_compatibility(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)

        result = self.database.import_corpus(*parsed)

        self.assertTrue(result["release_id"])
        self.assertEqual(result["legacy_generated_revocations"], 0)

    def test_typed_exact_legacy_member_can_be_retired_in_a_new_release(
        self,
    ) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        retired_id = "q_conditional_independence_common_cause_001"
        questions = tuple(
            question for question in parsed[4] if question.id != retired_id
        )

        result = self.database.import_corpus(
            *parsed[:4],
            questions,
            *parsed[5:],
        )

        self.assertTrue(result["release_id"])
        with self.database.read() as connection:
            self.assertIsNone(
                connection.execute(
                    """SELECT 1 FROM release_questions
                       WHERE release_id=? AND question_id=?""",
                    (result["release_id"], retired_id),
                ).fetchone()
            )

    def test_typed_legacy_content_change_invalidates_cohort(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        questions = list(parsed[4])
        legacy_index = next(
            index
            for index, question in enumerate(questions)
            if "generated" not in question.provenance
        )
        questions[legacy_index] = replace(
            questions[legacy_index],
            stem=questions[legacy_index].stem + " Unattested content mutation.",
        )

        with self.assertRaisesRegex(
            ValidationError,
            "generated_provenance_required",
        ):
            self.database.import_corpus(
                *parsed[:4],
                questions,
                *parsed[5:],
            )

    def test_typed_legacy_status_demotion_preserves_compatibility(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        questions = list(parsed[4])
        legacy_index = next(
            index
            for index, question in enumerate(questions)
            if "generated" not in question.provenance
        )
        questions[legacy_index] = replace(
            questions[legacy_index],
            status=QuestionStatus.QUARANTINED,
        )

        result = self.database.import_corpus(
            *parsed[:4],
            questions,
            *parsed[5:],
        )

        self.assertTrue(result["release_id"])

    def test_typed_legacy_objective_change_invalidates_cohort(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        questions = list(parsed[4])
        legacy_indexes = [
            index
            for index, question in enumerate(questions)
            if "generated" not in question.provenance
            and question.objective is not None
        ]
        target_index = legacy_indexes[0]
        other_objective = next(
            question.objective
            for index, question in enumerate(questions)
            if index != target_index
            and question.objective is not None
            and question.objective.id != questions[target_index].objective_id
        )
        questions[target_index] = replace(
            questions[target_index],
            objective=other_objective,
        )

        with self.assertRaisesRegex(
            ValidationError,
            "generated_provenance_required",
        ):
            self.database.import_corpus(
                *parsed[:4],
                questions,
                *parsed[5:],
            )

    def test_typed_legacy_diagnostic_change_invalidates_cohort(self) -> None:
        parsed = read_and_parse(CORPUS, include_catalog=True)
        questions = list(parsed[4])
        target_index = next(
            index
            for index, question in enumerate(questions)
            if "generated" not in question.provenance
            and any(
                option.diagnostic_objective_id is not None
                for option in question.options
            )
        )
        target = questions[target_index]
        option_index = next(
            index
            for index, option in enumerate(target.options)
            if option.diagnostic_objective_id is not None
        )
        options = list(target.options)
        options[option_index] = replace(
            options[option_index],
            diagnostic_objective_id=(
                options[option_index].diagnostic_objective_id + "_mutated"
            ),
        )
        questions[target_index] = replace(target, options=tuple(options))

        with self.assertRaisesRegex(
            ValidationError,
            "generated_provenance_required",
        ):
            self.database.import_corpus(
                *parsed[:4],
                questions,
                *parsed[5:],
            )

    def test_fresh_quarantined_generation_adds_no_safety_revocation(self) -> None:
        result = self.database.import_corpus(
            *self.bundle[:-1],
            (self.expert, self.generated),
        )

        with self.database.read() as connection:
            revocations = connection.execute(
                "SELECT COUNT(*) AS n FROM question_revocations"
            ).fetchone()["n"]
            events = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type = 'QuestionEmergencyRevoked'"""
            ).fetchone()["n"]
        self.assertEqual(result["legacy_generated_revocations"], 0)
        self.assertEqual(revocations, 0)
        self.assertEqual(events, 0)

    def test_import_revokes_legacy_active_generated_idempotently(self) -> None:
        legacy, _ = self.import_legacy_active_generated_release()
        safe_questions = self.serviceable_expert_bank()
        first = self.database.import_corpus(
            *self.bundle[:-1],
            (*safe_questions, self.generated),
        )
        repeated = self.database.import_corpus(
            *self.bundle[:-1],
            (*safe_questions, self.generated),
        )

        with self.database.read() as connection:
            revocation = connection.execute(
                """SELECT revocation.*, event.idempotency_key,
                          event.event_type, event.payload_json
                   FROM question_revocations revocation
                   JOIN events event ON event.event_id = revocation.event_id
                   WHERE revocation.question_id = ?""",
                (self.generated.id,),
            ).fetchone()
            safety_events = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type = 'QuestionEmergencyRevoked'"""
            ).fetchone()["n"]
        self.assertNotEqual(legacy["release_id"], first["release_id"])
        self.assertEqual(first["legacy_generated_revocations"], 1)
        self.assertEqual(repeated["legacy_generated_revocations"], 0)
        self.assertEqual(safety_events, 1)
        self.assertEqual(
            revocation["reason"],
            LEGACY_UNREVIEWED_GENERATED_REVOCATION_REASON,
        )
        self.assertTrue(
            revocation["idempotency_key"].startswith(
                LEGACY_UNREVIEWED_GENERATED_REVOCATION_KEY_PREFIX
            )
        )
        self.assertEqual(revocation["event_type"], "QuestionEmergencyRevoked")
        self.assertEqual(
            json.loads(revocation["payload_json"]),
            {
                "question_id": self.generated.id,
                "reason": LEGACY_UNREVIEWED_GENERATED_REVOCATION_REASON,
            },
        )
        report = self.database.verify_integrity()
        self.assertTrue(report["ok"], report["errors"])

    def test_current_schema_reopen_revokes_historical_unreviewed_generation(
        self,
    ) -> None:
        legacy, _ = self.import_legacy_active_generated_release()
        unopened = Database(self.database.path)

        visible = unopened.questions_for_scope(
            {"c_target"},
            release_id=legacy["release_id"],
        )
        self.assertNotIn(
            self.generated.id,
            {question.id for question in visible},
        )
        before = Database(
            self.database.path, read_only=True
        ).verify_integrity()
        self.assertFalse(before["ok"])
        self.assertTrue(
            any(
                "fails the generated-content activation gate" in error
                for error in before["errors"]
            )
        )

        reopened = Database(self.database.path)
        reopened.initialize()

        with reopened.read() as connection:
            revocations = connection.execute(
                """SELECT COUNT(*) AS n FROM question_revocations
                   WHERE question_id=?""",
                (self.generated.id,),
            ).fetchone()["n"]
        self.assertEqual(revocations, 1)
        report = reopened.verify_integrity()
        self.assertTrue(report["ok"], report["errors"])

    def test_pending_historical_generated_question_cannot_be_answered(
        self,
    ) -> None:
        self.import_legacy_active_generated_release()
        engine = AdaptiveEngine(self.database)
        pending = None
        with patch(
            "tsq.store.question_runtime_activation_safe",
            return_value=True,
        ):
            for seed in range(1, 101):
                learner_id = f"unsafe-pending-{seed}"
                engine.create_learner(learner_id)
                session = engine.start_session(
                    learner_id,
                    "c_target",
                    seed=seed,
                )
                presentation = engine.next_question(session["id"])
                if presentation.question.id == self.generated.id:
                    pending = presentation
                    break
        self.assertIsNotNone(pending)
        assert pending is not None

        with self.assertRaisesRegex(
            ConflictError, "human-review commitment"
        ):
            engine.submit_answer(
                pending.decision_id,
                pending.question.correct_option.id,
                idempotency_key="unsafe-generated-answer",
            )
        with self.database.read() as connection:
            attempts = connection.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE decision_id=?",
                (pending.decision_id,),
            ).fetchone()["n"]
        self.assertEqual(attempts, 0)

    def test_legacy_pinned_sessions_cannot_select_or_answer_revoked_item(self) -> None:
        legacy, _ = self.import_legacy_active_generated_release()
        engine = AdaptiveEngine(self.database)
        selection_learner = "legacy-selection"
        engine.create_learner(selection_learner)
        selection_session = engine.start_session(
            selection_learner,
            "c_target",
            seed=101,
        )

        pending = None
        with patch(
            "tsq.store.question_runtime_activation_safe",
            return_value=True,
        ):
            for seed in range(1, 101):
                learner_id = f"legacy-pending-{seed}"
                engine.create_learner(learner_id)
                session = engine.start_session(
                    learner_id,
                    "c_target",
                    seed=seed,
                )
                presentation = engine.next_question(session["id"])
                if presentation.question.id == self.generated.id:
                    pending = presentation
                    break
        self.assertIsNotNone(pending)

        result = self.database.import_corpus(
            *self.bundle[:-1],
            (*self.serviceable_expert_bank(), self.generated),
        )
        self.assertEqual(result["legacy_generated_revocations"], 1)
        self.assertEqual(
            self.database.get_session(selection_session["id"])[
                "corpus_release_id"
            ],
            legacy["release_id"],
        )

        selected_after_revocation = engine.next_question(selection_session["id"])
        self.assertNotEqual(
            selected_after_revocation.question.id,
            self.generated.id,
        )
        assert pending is not None
        with self.assertRaisesRegex(ConflictError, "emergency-revoked"):
            engine.submit_answer(
                pending.decision_id,
                pending.question.correct_option.id,
                idempotency_key="answer-legacy-generated-after-migration",
            )

    def test_reviewed_revision_activates_after_old_id_is_revoked(self) -> None:
        self.import_legacy_active_generated_release()
        safe_questions = self.serviceable_expert_bank()
        quarantined = self.database.import_corpus(
            *self.bundle[:-1],
            (*safe_questions, self.generated),
        )
        reviewed = replace(
            self.generated,
            id="q_generated_reviewed_after_revocation",
            version=2,
            revision_of=self.generated.id,
            status=QuestionStatus.APPROVED,
            stem=self.generated.stem + " Independently reviewed revision.",
            provenance=active_generated_provenance(),
        )

        activated = self.database.import_corpus(
            *self.bundle[:-1],
            (*safe_questions, self.generated, reviewed),
        )

        self.assertNotEqual(quarantined["release_id"], activated["release_id"])
        self.assertEqual(activated["legacy_generated_revocations"], 0)
        active = self.database.questions_for_scope(
            {"c_target"},
            release_id=activated["release_id"],
        )
        active_ids = {question.id for question in active}
        self.assertIn(reviewed.id, active_ids)
        self.assertNotIn(self.generated.id, active_ids)
        with self.database.read() as connection:
            revoked_ids = {
                row["question_id"]
                for row in connection.execute(
                    "SELECT question_id FROM question_revocations"
                ).fetchall()
            }
        self.assertEqual(revoked_ids, {self.generated.id})

    def test_typed_objects_cannot_bypass_provenance_scalar_types(self) -> None:
        invalid_values = (
            ["not", "an", "object"],
            {"generated": "true", "human_review": True},
        )
        for provenance in invalid_values:
            with self.subTest(provenance=provenance):
                invalid = replace(
                    self.expert,
                    provenance=provenance,
                )
                with self.assertRaisesRegex(
                    ValidationError,
                    "provenance",
                ):
                    self.database.import_corpus(
                        *self.bundle[:-1],
                        (invalid,),
                    )

    def test_reviewed_revision_activates_only_in_a_new_release(self) -> None:
        first = self.database.import_corpus(
            *self.bundle[:-1],
            (self.expert, self.generated),
        )
        reviewed = replace(
            self.generated,
            id="q_generated_reviewed",
            version=2,
            revision_of=self.generated.id,
            status=QuestionStatus.APPROVED,
            stem=self.generated.stem + " Independently reviewed revision.",
            provenance=active_generated_provenance(),
        )

        second = self.database.import_corpus(
            *self.bundle[:-1],
            (self.generated, reviewed),
        )

        self.assertNotEqual(first["release_id"], second["release_id"])
        active = self.database.questions_for_scope(
            {"c_target"},
            release_id=second["release_id"],
        )
        self.assertEqual(
            tuple(question.id for question in active),
            ("q_generated_reviewed",),
        )
