# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.engine import AdaptiveEngine
from tsq.learner import (
    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
    LearnerModel,
)
from tsq.models import (
    Concept,
    ConceptEdge,
    ConceptRole,
    ConceptWeight,
    LearningObjective,
    Misconception,
    ObjectiveOperation,
    Option,
    Question,
    QuestionKind,
    QuestionStatus,
    Source,
)
from tsq.store import Database


START = datetime(2104, 4, 5, 9, 0, tzinfo=timezone.utc)
CONCEPT_ID = "c_release_scoped_evidence"
OBJECTIVE_ID = "lo_release_scoped_evidence"
TRANSFER_QUESTION_ID = "q_release_scoped_transfer"


def release_scoped_corpus() -> tuple[
    tuple[Concept, ...],
    tuple[ConceptEdge, ...],
    tuple[Misconception, ...],
    tuple[Source, ...],
    tuple[Question, ...],
]:
    concept = Concept(
        CONCEPT_ID,
        "Release-scoped evidence",
        "Distinguish historical observations from currently accepted evidence.",
    )
    objective = LearningObjective(
        id=OBJECTIVE_ID,
        name="Audit current evidence eligibility",
        description=(
            "Determine whether historical evidence is still accepted by the "
            "active immutable corpus release."
        ),
        primary_concept_id=CONCEPT_ID,
        supporting_concept_ids=(),
        operation=ObjectiveOperation.APPLY,
    )
    misconceptions = (
        Misconception(
            "m_release_history_equals_current",
            CONCEPT_ID,
            "History implies current acceptance",
            "Treats a historical observation as permanently valid current evidence.",
        ),
        Misconception(
            "m_release_membership_ignores_status",
            CONCEPT_ID,
            "Membership ignores eligibility",
            "Assumes release membership alone makes every item certification-eligible.",
        ),
        Misconception(
            "m_release_revocation_is_local",
            CONCEPT_ID,
            "Revocation is session-local",
            "Treats an emergency revocation as applying only to future sessions.",
        ),
    )
    source = Source(
        "src_release_scoped_evidence",
        "Release-scoped evidence test contract",
        "https://example.test/release-scoped-evidence",
        "CC0-1.0",
        {"locator": "Deterministic test fixture, sections 1-3"},
    )
    question_index = 0

    def question(
        question_id: str,
        family_id: str,
        kind: QuestionKind,
        scenario: str,
        difficulty: float,
    ) -> Question:
        nonlocal question_index
        option_rotation = question_index % 4
        question_index += 1
        options = (
            Option(
                "current",
                "Retain the history; include its family while a live family "
                "item remains accepted now.",
                True,
                "Historical storage and current certification are separate claims.",
            ),
            Option(
                "historical",
                "Treat the earlier correct answer as sufficient after a later "
                "release removes the family's live item.",
                False,
                "Historical existence does not imply current acceptance.",
                "m_release_history_equals_current",
                OBJECTIVE_ID,
            ),
            Option(
                "membership",
                "Treat release membership as sufficient when the family's item "
                "status is no longer evidence eligible.",
                False,
                "Only approved or calibrated items are currently eligible.",
                "m_release_membership_ignores_status",
                OBJECTIVE_ID,
            ),
            Option(
                "revocation",
                "Continue counting the family after a global safety event "
                "revokes the family's sole live question.",
                False,
                "Emergency revocation is global rather than session-local.",
                "m_release_revocation_is_local",
                OBJECTIVE_ID,
            ),
        )
        rotated_options = (
            options[option_rotation:] + options[:option_rotation]
        )
        return Question(
            id=question_id,
            version=1,
            family_id=family_id,
            status=QuestionStatus.CALIBRATED,
            stem=(
                f"{scenario} Which conclusion correctly separates immutable "
                "history from current certification evidence?"
            ),
            kind=kind,
            difficulty=difficulty,
            discrimination=1.6,
            guess_rate=0.25,
            slip_rate=0.03,
            concepts=(
                ConceptWeight(
                    CONCEPT_ID,
                    1.0,
                    ConceptRole.PRIMARY,
                ),
            ),
            options=rotated_options,
            source_ids=(source.id,),
            objective=objective,
            provenance={
                "generated": False,
                "authoring_method": "expert-authored-test-fixture",
            },
        )

    questions = (
        question(
            "q_release_scoped_conceptual_a",
            "f_release_scoped_conceptual_a",
            QuestionKind.CONCEPTUAL,
            "An audit reads a correct response from an older release.",
            1.0,
        ),
        question(
            "q_release_scoped_conceptual_b",
            "f_release_scoped_conceptual_b",
            QuestionKind.CONCEPTUAL,
            "A second independent family records the same distinction.",
            1.1,
        ),
        question(
            "q_release_scoped_conceptual_c",
            "f_release_scoped_conceptual_c",
            QuestionKind.CONCEPTUAL,
            "A third independent family records an eligibility decision.",
            1.2,
        ),
        question(
            "q_release_scoped_conceptual_d",
            "f_release_scoped_conceptual_d",
            QuestionKind.CONCEPTUAL,
            "A fourth independent family audits the historical ledger.",
            1.3,
        ),
        question(
            "q_release_scoped_conceptual_e",
            "f_release_scoped_conceptual_e",
            QuestionKind.CONCEPTUAL,
            "A fifth independent family audits release membership.",
            1.4,
        ),
        question(
            "q_release_scoped_conceptual_f",
            "f_release_scoped_conceptual_f",
            QuestionKind.CONCEPTUAL,
            "A sixth independent family audits active item status.",
            1.5,
        ),
        question(
            "q_release_scoped_conceptual_g",
            "f_release_scoped_conceptual_g",
            QuestionKind.CONCEPTUAL,
            "A seventh independent family audits global revocations.",
            1.6,
        ),
        question(
            "q_release_scoped_conceptual_h",
            "f_release_scoped_conceptual_h",
            QuestionKind.CONCEPTUAL,
            "An eighth independent family audits evidence presentation.",
            1.7,
        ),
        question(
            "q_release_scoped_conceptual_i",
            "f_release_scoped_conceptual_i",
            QuestionKind.CONCEPTUAL,
            "A ninth independent family audits retained learner history.",
            1.8,
        ),
        question(
            "q_release_scoped_conceptual_j",
            "f_release_scoped_conceptual_j",
            QuestionKind.CONCEPTUAL,
            "A tenth independent family audits current certification.",
            1.9,
        ),
        question(
            TRANSFER_QUESTION_ID,
            "f_release_scoped_transfer",
            QuestionKind.APPLICATION,
            "A later release or safety action changes which items are accepted.",
            0.95,
        ),
        question(
            "q_release_scoped_application_reserve",
            "f_release_scoped_application_reserve",
            QuestionKind.APPLICATION,
            "An unused transfer family preserves an independent verification path.",
            1.05,
        ),
    )
    return (concept,), (), misconceptions, (source,), questions


class ReleaseScopedEvidenceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.database = Database(
            Path(self.tempdir.name) / "release-scoped-evidence.db"
        )
        self.database.initialize()
        self.corpus = release_scoped_corpus()
        self.release_id = self.database.import_corpus(*self.corpus)[
            "release_id"
        ]
        self.engine = AdaptiveEngine(
            self.database,
            LearnerModel(OBJECTIVE_GAUSSIAN_MODEL_VERSION),
        )
        self.learner_id = "release-scoped-learner"
        self.engine.create_learner(self.learner_id)
        (
            self.profile_now,
            self.withdrawn_question_id,
        ) = self._record_repeated_independent_successes()

    def _record_repeated_independent_successes(
        self,
    ) -> tuple[datetime, str]:
        presented_ids: set[str] = set()
        current = START
        for index in range(10):
            session = self.engine.start_session(
                self.learner_id,
                CONCEPT_ID,
                seed=101 + index,
                now=current,
            )
            presentation = self.engine.next_question(
                session["id"],
                now=current + timedelta(seconds=1),
            )
            presented_ids.add(presentation.question.id)
            self.engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.95,
                response_ms=2_000,
                hint_count=0,
                feedback_shown=False,
                idempotency_key=f"release-scoped-answer:{index}",
                now=current + timedelta(seconds=3),
            )
            self.engine.end_session(
                session["id"],
                status="completed",
                reason="release_scoped_fixture",
                idempotency_key=f"release-scoped-end:{index}",
                now=current + timedelta(seconds=4),
            )
            current += timedelta(minutes=5)
        self.assertEqual(len(presented_ids), 10)
        application_ids = {
            question.id
            for question in self.corpus[4]
            if question.kind is QuestionKind.APPLICATION
        }
        selected_applications = presented_ids & application_ids
        self.assertEqual(len(selected_applications), 1)
        return current, selected_applications.pop()

    def _objective_profile(self) -> dict[str, object]:
        profile = self.engine.profile(
            self.learner_id,
            root_concept_id=CONCEPT_ID,
            now=self.profile_now,
        )
        return next(
            row
            for row in profile["learning_objectives"]
            if row["objective_id"] == OBJECTIVE_ID
        )

    def _currently_certified(
        self,
        release_id: str,
    ) -> set[str]:
        objectives = {
            objective.id: objective
            for objective in self.database.get_learning_objectives(release_id)
        }
        with self.database.read() as connection:
            return self.engine._persistently_verified_objectives(
                connection,
                learner_id=self.learner_id,
                release_id=release_id,
                objective_ids={OBJECTIVE_ID},
                objectives=objectives,
                now=self.profile_now,
            )

    def _historical_counts(self) -> tuple[int, int]:
        with self.database.read() as connection:
            families = connection.execute(
                """SELECT COUNT(*) AS n
                   FROM learner_objective_families
                   WHERE learner_id = ? AND objective_id = ?""",
                (self.learner_id, OBJECTIVE_ID),
            ).fetchone()["n"]
            attempts = connection.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE learner_id = ?",
                (self.learner_id,),
            ).fetchone()["n"]
        return int(families), int(attempts)

    def _assert_current_claim_withdrawn(
        self,
        *,
        before: dict[str, object],
        projection_hash: str,
    ) -> None:
        after = self._objective_profile()
        self.assertEqual(self._historical_counts(), (10, 10))
        self.assertEqual(
            self.database.learner_projection_hash(self.learner_id),
            projection_hash,
        )
        self.assertEqual(after["mastery_probability"], before["mastery_probability"])
        self.assertEqual(after["evidence_mass"], before["evidence_mass"])
        self.assertEqual(after["independent_families"], 9)
        self.assertEqual(after["successful_retrieval_families"], 9)
        self.assertEqual(after["observed_response_families"], 9)
        self.assertEqual(after["operation_kinds"], 1)
        self.assertNotIn(after["state"], {"proficient", "durable"})

    def test_new_active_release_removes_family_only_from_current_claims(
        self,
    ) -> None:
        before = self._objective_profile()
        self.assertEqual(before["independent_families"], 10)
        self.assertEqual(before["observed_response_families"], 10)
        self.assertEqual(before["operation_kinds"], 2)
        self.assertIn(before["state"], {"proficient", "durable"})
        self.assertEqual(
            self._currently_certified(self.release_id),
            {OBJECTIVE_ID},
        )
        projection_hash = self.database.learner_projection_hash(
            self.learner_id
        )
        with self.database.read() as connection:
            historical_session_id = connection.execute(
                """SELECT id FROM sessions WHERE learner_id = ?
                   ORDER BY created_at, id LIMIT 1""",
                (self.learner_id,),
            ).fetchone()["id"]
        historical_contract = self.engine.session_report(
            historical_session_id,
            now=self.profile_now,
        )["selected_response_inference"]
        self.assertEqual(
            historical_contract["corpus_release_id"], self.release_id
        )

        reduced_questions = tuple(
            question
            for question in self.corpus[4]
            if question.id != self.withdrawn_question_id
        )
        new_release_id = self.database.import_corpus(
            self.corpus[0],
            self.corpus[1],
            self.corpus[2],
            self.corpus[3],
            reduced_questions,
        )["release_id"]

        self.assertNotEqual(new_release_id, self.release_id)
        self.assertEqual(self._currently_certified(new_release_id), set())
        with self.database.read() as connection:
            memberships = connection.execute(
                """SELECT release_id FROM release_questions
                   WHERE question_id = ? ORDER BY release_id""",
                (self.withdrawn_question_id,),
            ).fetchall()
        self.assertEqual(
            {row["release_id"] for row in memberships},
            {self.release_id},
        )
        pinned_after = self.engine.session_report(
            historical_session_id,
            now=self.profile_now,
        )["selected_response_inference"]
        current_profile = self.engine.profile(
            self.learner_id,
            root_concept_id=CONCEPT_ID,
            now=self.profile_now,
        )
        current_contract = current_profile["selected_response_inference"]
        self.assertEqual(pinned_after, historical_contract)
        self.assertEqual(current_profile["corpus_release_id"], new_release_id)
        self.assertEqual(
            current_contract["corpus_release_id"], new_release_id
        )
        self.assertEqual(
            current_contract["eligible_question_count"],
            historical_contract["eligible_question_count"] - 1,
        )
        self._assert_current_claim_withdrawn(
            before=before,
            projection_hash=projection_hash,
        )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_inference_contract_counts_mixed_status_and_revocation(self) -> None:
        mixed_questions = tuple(
            replace(
                question,
                status=(
                    QuestionStatus.APPROVED
                    if index == 0
                    else (
                        QuestionStatus.DRAFT
                        if index == 1
                        else QuestionStatus.CALIBRATED
                    )
                ),
            )
            for index, question in enumerate(self.corpus[4])
        )
        mixed_release = self.database.import_corpus(
            self.corpus[0],
            self.corpus[1],
            self.corpus[2],
            self.corpus[3],
            mixed_questions,
        )["release_id"]

        contract = self.engine.profile(
            self.learner_id,
            now=self.profile_now,
        )["selected_response_inference"]
        self.assertEqual(contract["corpus_release_id"], mixed_release)
        self.assertEqual(contract["eligible_question_count"], 11)
        self.assertEqual(contract["eligible_family_count"], 11)
        self.assertEqual(contract["approved_question_count"], 1)
        self.assertEqual(contract["calibrated_question_count"], 10)
        self.assertEqual(contract["calibrated_family_count"], 10)
        self.assertEqual(
            contract["corpus_calibration_status"],
            "partially_calibrated_items",
        )

        revoked_id = mixed_questions[-1].id
        self.database.revoke_question(
            revoked_id,
            "Mixed-status inference-contract revocation fixture.",
            idempotency_key="mixed-status-contract-revocation",
        )
        after = self.engine.profile(
            self.learner_id,
            now=self.profile_now,
        )["selected_response_inference"]
        self.assertEqual(after["eligible_question_count"], 10)
        self.assertEqual(after["eligible_family_count"], 10)
        self.assertEqual(after["approved_question_count"], 1)
        self.assertEqual(after["calibrated_question_count"], 9)
        self.assertEqual(after["calibrated_family_count"], 9)

    def test_profile_gates_objective_misconceptions_on_live_mappings(
        self,
    ) -> None:
        learner_id = "live-diagnostic-mapping-learner"
        misconception_id = "m_release_history_equals_current"
        self.engine.create_learner(learner_id)
        current = self.profile_now + timedelta(days=1)
        live_question_ids: set[str] = set()

        for index in range(2):
            started_at = current + timedelta(minutes=index)
            session = self.engine.start_session(
                learner_id,
                CONCEPT_ID,
                seed=787 + index,
                now=started_at,
            )
            presentation = self.engine.next_question(
                session["id"],
                now=started_at + timedelta(seconds=1),
            )
            selected = next(
                option
                for option in presentation.question.options
                if option.misconception_id == misconception_id
            )
            live_question_ids.add(presentation.question.id)
            self.engine.submit_answer(
                presentation.decision_id,
                selected.id,
                confidence=1.0,
                response_ms=1_000,
                hint_count=0,
                feedback_shown=False,
                idempotency_key=f"live-mapping-wrong:{index}",
                now=started_at + timedelta(seconds=2),
            )
            self.engine.end_session(
                session["id"],
                status="completed",
                reason="live_mapping_fixture",
                idempotency_key=f"live-mapping-end:{index}",
                now=started_at + timedelta(seconds=3),
            )

        self.assertEqual(len(live_question_ids), 2)
        mixed_questions = tuple(
            replace(
                question,
                status=(
                    QuestionStatus.CALIBRATED
                    if question.id in live_question_ids
                    else QuestionStatus.DRAFT
                ),
            )
            for question in self.corpus[4]
        )
        mixed_release = self.database.import_corpus(
            self.corpus[0],
            self.corpus[1],
            self.corpus[2],
            self.corpus[3],
            mixed_questions,
        )["release_id"]
        profile_at = current + timedelta(minutes=3)
        before = self.engine.profile(
            learner_id,
            root_concept_id=CONCEPT_ID,
            now=profile_at,
        )
        before_objective = next(
            row
            for row in before["learning_objectives"]
            if row["objective_id"] == OBJECTIVE_ID
        )
        before_hypothesis = next(
            row
            for row in before["misconception_hypotheses"]
            if row["misconception_id"] == misconception_id
        )
        self.assertGreaterEqual(before_hypothesis["probability"], 0.35)
        self.assertEqual(
            before_objective["active_misconception_probability"],
            before_hypothesis["probability"],
        )
        projection_hash = self.database.learner_projection_hash(learner_id)

        with self.database.read() as connection:
            mapping_rows = connection.execute(
                """SELECT membership.question_id, membership.status
                   FROM release_option_objectives mapping
                   JOIN release_questions membership
                     ON membership.release_id = mapping.release_id
                    AND membership.question_id = mapping.question_id
                   JOIN options option
                     ON option.question_id = mapping.question_id
                    AND option.option_id = mapping.option_id
                   WHERE mapping.release_id = ?
                     AND mapping.objective_id = ?
                     AND option.misconception_id = ?
                   ORDER BY membership.question_id""",
                (mixed_release, OBJECTIVE_ID, misconception_id),
            ).fetchall()
        self.assertEqual(len(mapping_rows), len(mixed_questions))
        self.assertEqual(
            {
                row["question_id"]
                for row in mapping_rows
                if row["status"] == QuestionStatus.CALIBRATED.value
            },
            live_question_ids,
        )
        self.assertEqual(
            sum(
                row["status"] == QuestionStatus.DRAFT.value
                for row in mapping_rows
            ),
            len(mixed_questions) - len(live_question_ids),
        )

        for index, question_id in enumerate(sorted(live_question_ids)):
            self.database.revoke_question(
                question_id,
                "Withdraw the last live diagnostic mapping fixture.",
                idempotency_key=f"live-mapping-revocation:{index}",
            )

        after = self.engine.profile(
            learner_id,
            root_concept_id=CONCEPT_ID,
            now=profile_at,
        )
        after_objective = next(
            row
            for row in after["learning_objectives"]
            if row["objective_id"] == OBJECTIVE_ID
        )
        after_hypothesis = next(
            row
            for row in after["misconception_hypotheses"]
            if row["misconception_id"] == misconception_id
        )
        self.assertEqual(
            after_objective["active_misconception_probability"],
            0.0,
        )
        self.assertEqual(after_hypothesis, before_hypothesis)
        self.assertIn(after_hypothesis, after["active_misconceptions"])
        self.assertEqual(
            self.database.learner_projection_hash(learner_id),
            projection_hash,
        )
        self.assertTrue(self.database.verify_integrity()["ok"])

    def test_emergency_revocation_withdraws_family_from_current_claims(
        self,
    ) -> None:
        before = self._objective_profile()
        self.assertEqual(before["independent_families"], 10)
        self.assertEqual(before["observed_response_families"], 10)
        self.assertEqual(before["operation_kinds"], 2)
        self.assertIn(before["state"], {"proficient", "durable"})
        self.assertEqual(
            self._currently_certified(self.release_id),
            {OBJECTIVE_ID},
        )
        projection_hash = self.database.learner_projection_hash(
            self.learner_id
        )

        self.database.revoke_question(
            self.withdrawn_question_id,
            "The transfer fixture is no longer accepted for evidence.",
            idempotency_key="release-scoped-emergency-revocation",
        )

        self.assertEqual(self._currently_certified(self.release_id), set())
        with self.database.read() as connection:
            membership = connection.execute(
                """SELECT status FROM release_questions
                   WHERE release_id = ? AND question_id = ?""",
                (self.release_id, self.withdrawn_question_id),
            ).fetchone()
            revocation = connection.execute(
                """SELECT reason FROM question_revocations
                   WHERE question_id = ?""",
                (self.withdrawn_question_id,),
            ).fetchone()
        self.assertEqual(
            membership["status"],
            QuestionStatus.CALIBRATED.value,
        )
        self.assertIsNotNone(revocation)
        self._assert_current_claim_withdrawn(
            before=before,
            projection_hash=projection_hash,
        )
        self.assertTrue(self.database.verify_integrity()["ok"])
