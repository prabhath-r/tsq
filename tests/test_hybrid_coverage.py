# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.models import CandidateScore
from tsq.policy import AdaptivePolicy, _HybridCoverage
from tsq.replay import ProjectionReplay
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2145, 5, 4, 9, 0, tzinfo=timezone.utc)


class HybridCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "hybrid.db")
        self.database.initialize()
        self.corpus = read_and_parse(CORPUS, include_catalog=True)
        self.database.import_corpus(*self.corpus)
        self.engine = AdaptiveEngine(self.database)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def coverage_for(
        self,
        learner_id: str,
        question,
        *,
        release_id: str | None = None,
    ) -> _HybridCoverage:
        return self.engine.policy._hybrid_coverage_by_question(
            (question,),
            learner_id=learner_id,
            release_id=release_id or self.database.get_active_release_id(),
            objective_states=self.database.get_objective_states(learner_id),
            concept_states=self.database.get_skill_states(learner_id),
        )[question.id]

    def answer_credibly(self, learner_id: str, *, seed: int):
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id,
            "t_transformers",
            seed=seed,
            explore_related=False,
            now=START,
        )
        presentation = self.engine.next_question(session["id"], now=START)
        self.engine.submit_answer(
            presentation.decision_id,
            presentation.question.correct_option.id,
            confidence=0.95,
            response_ms=7_000,
            hint_count=0,
            feedback_shown=False,
            idempotency_key=f"hybrid-credible:{learner_id}",
            now=START + timedelta(seconds=7),
        )
        return session["corpus_release_id"], presentation.question

    def test_response_quality_discounts_diagnostic_information(self) -> None:
        cases = (
            ("abstain", None, 0.20, 7_000, 0, 0.12),
            ("low-confidence", "correct", 0.20, 7_000, 0, 0.40),
            ("fast", "correct", 0.95, 100, 0, 0.15),
            ("hinted", "correct", 0.95, 7_000, 1, 0.20),
            ("named-wrong", "named-wrong", 0.95, 7_000, 0, 1.00),
        )
        for index, (
            label,
            selection,
            confidence,
            response_ms,
            hints,
            expected_factor,
        ) in enumerate(cases):
            with self.subTest(label=label):
                learner_id = f"hybrid-{label}"
                self.engine.create_learner(learner_id)
                selected_at = START + timedelta(days=index)
                session = self.engine.start_session(
                    learner_id,
                    "t_transformers",
                    seed=700 + index,
                    explore_related=False,
                    now=selected_at,
                )
                presentation = self.engine.next_question(
                    session["id"], now=selected_at
                )
                question = presentation.question
                if selection == "correct":
                    option_id = question.correct_option.id
                elif selection == "named-wrong":
                    option_id = next(
                        option.id
                        for option in question.options
                        if not option.correct and option.misconception_id
                    )
                else:
                    option_id = None
                self.engine.submit_answer(
                    presentation.decision_id,
                    option_id,
                    confidence=confidence,
                    response_ms=response_ms,
                    hint_count=hints,
                    feedback_shown=False,
                    idempotency_key=f"hybrid-response:{label}",
                    now=selected_at + timedelta(milliseconds=response_ms),
                )
                with self.database.read() as connection:
                    decision = connection.execute(
                        "SELECT evidence_weight FROM decisions WHERE id = ?",
                        (presentation.decision_id,),
                    ).fetchone()
                coverage = self.coverage_for(learner_id, question)
                self.assertEqual(coverage.raw_exposures, 1)
                self.assertAlmostEqual(
                    coverage.diagnostic_information,
                    float(decision["evidence_weight"]) * expected_factor,
                    places=12,
                )
                self.assertEqual(
                    coverage.successful_retrieval_families, 0
                )

    def test_credible_success_keeps_retrieval_separate(self) -> None:
        learner_id = "hybrid-credible"
        _, question = self.answer_credibly(learner_id, seed=811)
        coverage = self.coverage_for(learner_id, question)
        self.assertEqual(coverage.raw_exposures, 1)
        self.assertGreater(coverage.diagnostic_information, 0.0)
        self.assertEqual(coverage.successful_retrieval_families, 1)

    def test_removed_family_does_not_count_in_new_release(self) -> None:
        learner_id = "hybrid-release-removal"
        old_release_id, question = self.answer_credibly(
            learner_id, seed=812
        )
        before = self.coverage_for(
            learner_id, question, release_id=old_release_id
        )
        self.assertEqual(before.successful_retrieval_families, 1)
        reduced_questions = tuple(
            candidate
            for candidate in self.corpus[4]
            if not (
                candidate.objective_id == question.objective_id
                and candidate.family_id == question.family_id
            )
        )
        new_release_id = self.database.import_corpus(
            self.corpus[0],
            self.corpus[1],
            self.corpus[2],
            self.corpus[3],
            reduced_questions,
            self.corpus[5],
            self.corpus[6],
        )["release_id"]
        self.assertNotEqual(new_release_id, old_release_id)

        current = self.coverage_for(
            learner_id, question, release_id=new_release_id
        )
        historical = self.coverage_for(
            learner_id, question, release_id=old_release_id
        )
        self.assertEqual(current.raw_exposures, before.raw_exposures)
        self.assertEqual(
            current.diagnostic_information,
            before.diagnostic_information,
        )
        self.assertEqual(current.successful_retrieval_families, 0)
        self.assertEqual(historical.successful_retrieval_families, 1)

    def test_revoked_family_does_not_count_in_pinned_release(self) -> None:
        learner_id = "hybrid-release-revocation"
        release_id, question = self.answer_credibly(learner_id, seed=813)
        before = self.coverage_for(
            learner_id, question, release_id=release_id
        )
        self.assertEqual(before.successful_retrieval_families, 1)
        with self.database.read() as connection:
            matching = connection.execute(
                """SELECT direct.question_id
                   FROM release_question_objectives direct
                   JOIN questions candidate
                     ON candidate.id = direct.question_id
                   JOIN release_questions released
                     ON released.release_id = direct.release_id
                    AND released.question_id = direct.question_id
                   WHERE direct.release_id = ?
                     AND direct.objective_id = ?
                     AND candidate.family_id = ?
                     AND released.status IN ('approved', 'calibrated')
                   ORDER BY direct.question_id""",
                (
                    release_id,
                    question.objective_id,
                    question.family_id,
                ),
            ).fetchall()
        self.assertTrue(matching)
        for index, row in enumerate(matching):
            self.database.revoke_question(
                row["question_id"],
                "Hybrid-coverage revocation fixture.",
                idempotency_key=f"hybrid-revoke:{index}",
            )

        after = self.coverage_for(
            learner_id, question, release_id=release_id
        )
        self.assertEqual(after.raw_exposures, before.raw_exposures)
        self.assertEqual(
            after.diagnostic_information,
            before.diagnostic_information,
        )
        self.assertEqual(after.successful_retrieval_families, 0)

    def test_legacy_concept_family_uses_same_revocation_gate(self) -> None:
        learner_id = "hybrid-concept-revocation"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id,
            "c_data_leakage",
            seed=814,
            now=START,
        )
        presentation = self.engine.next_question(session["id"], now=START)
        question = presentation.question
        self.assertIsNone(question.objective_id)
        self.engine.submit_answer(
            presentation.decision_id,
            question.correct_option.id,
            confidence=0.95,
            response_ms=7_000,
            hint_count=0,
            feedback_shown=False,
            idempotency_key="hybrid-concept-response",
            now=START + timedelta(seconds=7),
        )
        before = self.coverage_for(
            learner_id,
            question,
            release_id=session["corpus_release_id"],
        )
        self.assertEqual(before.successful_retrieval_families, 1)
        with self.database.read() as connection:
            matching = connection.execute(
                """SELECT released.question_id
                   FROM release_questions released
                   JOIN questions candidate
                     ON candidate.id = released.question_id
                   JOIN question_concepts mapping
                     ON mapping.question_id = candidate.id
                    AND mapping.role = 'primary'
                   WHERE released.release_id = ?
                     AND released.status IN ('approved', 'calibrated')
                     AND mapping.concept_id = ?
                     AND candidate.family_id = ?
                   ORDER BY released.question_id""",
                (
                    session["corpus_release_id"],
                    question.primary_concept_id,
                    question.family_id,
                ),
            ).fetchall()
        self.assertTrue(matching)
        for index, row in enumerate(matching):
            self.database.revoke_question(
                row["question_id"],
                "Hybrid concept-family revocation fixture.",
                idempotency_key=f"hybrid-concept-revoke:{index}",
            )
        after = self.coverage_for(
            learner_id,
            question,
            release_id=session["corpus_release_id"],
        )
        self.assertEqual(after.successful_retrieval_families, 0)

    def test_hybrid_frontier_is_burden_bounded_then_evidence_aware(self) -> None:
        fast = SimpleNamespace(id="q_fast", objective_id="lo_fast")
        hinted = SimpleNamespace(id="q_hinted", objective_id="lo_hinted")
        credible = SimpleNamespace(
            id="q_credible", objective_id="lo_credible"
        )
        overburdened = SimpleNamespace(
            id="q_abstain", objective_id="lo_abstain"
        )
        candidates, minimum, bypassed = (
            AdaptivePolicy._fair_coverage_candidates(
                (fast, hinted, credible, overburdened),
                coverage_by_question={
                    "q_fast": _HybridCoverage(2, 0.195, 0),
                    "q_hinted": _HybridCoverage(3, 0.260, 0),
                    "q_credible": _HybridCoverage(2, 1.300, 2),
                    "q_abstain": _HybridCoverage(4, 0.156, 0),
                },
                persistent_gap_objective_ids=set(),
            )
        )
        self.assertEqual(minimum, 2)
        self.assertEqual([question.id for question in candidates], ["q_fast"])
        self.assertEqual(bypassed, set())

    def test_selected_score_and_rationale_bind_all_three_metrics(self) -> None:
        learner_id = "hybrid-persisted"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id,
            "t_transformers",
            seed=917,
            explore_related=False,
            now=START,
        )
        first = self.engine.next_question(session["id"], now=START)
        self.engine.submit_answer(
            first.decision_id,
            first.question.correct_option.id,
            confidence=0.95,
            response_ms=7_000,
            hint_count=1,
            feedback_shown=False,
            idempotency_key="hybrid-persisted-hint",
            now=START + timedelta(seconds=7),
        )
        second = self.engine.next_question(
            session["id"], now=START + timedelta(minutes=1)
        )
        self.assertEqual(
            second.question.objective_id, first.question.objective_id
        )
        state = self.database.get_objective_states(learner_id)[
            first.question.objective_id
        ]
        self.assertEqual(second.score.coverage_raw_exposures, state.exposures)
        self.assertAlmostEqual(
            second.score.coverage_diagnostic_information,
            state.evidence_mass,
            places=12,
        )
        self.assertEqual(
            second.score.coverage_successful_retrieval_families, 0
        )
        with self.database.read() as connection:
            row = connection.execute(
                """SELECT selected_score_json, rationale
                   FROM decisions WHERE id = ?""",
                (second.decision_id,),
            ).fetchone()
        persisted = json.loads(row["selected_score_json"])
        self.assertEqual(
            persisted["coverage_raw_exposures"], state.exposures
        )
        self.assertAlmostEqual(
            persisted["coverage_diagnostic_information"],
            state.evidence_mass,
            places=12,
        )
        self.assertEqual(
            persisted["coverage_successful_retrieval_families"], 0
        )
        self.assertIn(
            f"coverage_raw_exposures={state.exposures}", row["rationale"]
        )
        self.assertIn(
            "coverage_diagnostic_information="
            + format(state.evidence_mass, ".12f"),
            row["rationale"],
        )
        self.assertIn(
            "coverage_successful_retrieval_families=0",
            row["rationale"],
        )
        self.assertTrue(self.database.verify_integrity()["ok"])
        self.assertTrue(ProjectionReplay(self.database).check(learner_id)["ok"])

    def test_legacy_score_defaults_and_malformed_coverage_fail_closed(self) -> None:
        legacy = CandidateScore(
            question_id="q_legacy",
            total=0.5,
            predicted_correct=0.5,
            information_gain=0.5,
            learning_fit=0.5,
            concept_need=0.5,
            misconception_value=0.0,
            prerequisite_value=0.0,
            review_value=0.0,
            novelty=1.0,
            kind_fit=1.0,
        )
        self.assertEqual(legacy.coverage_raw_exposures, 0)
        self.assertEqual(legacy.coverage_diagnostic_information, 0.0)
        self.assertEqual(
            legacy.coverage_successful_retrieval_families, 0
        )
        with self.assertRaisesRegex(ValidationError, "diagnostic information"):
            _HybridCoverage(1, float("nan"), 0)
        with self.assertRaisesRegex(
            ValidationError, "no greater than raw exposures"
        ):
            _HybridCoverage(0, 0.0, 1)


if __name__ == "__main__":
    unittest.main()
