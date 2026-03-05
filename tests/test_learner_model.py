# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import math
import sqlite3
import unittest
from datetime import datetime, timedelta, timezone

from tsq.learner import (
    MAX_POSTERIOR_VARIANCE,
    MIN_POSTERIOR_VARIANCE,
    LearnerModel,
)
from tsq.models import (
    Concept,
    ConceptRole,
    ConceptWeight,
    Option,
    Question,
    QuestionKind,
    QuestionStatus,
    SkillState,
    logit,
)


NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def make_question(
    *,
    question_id: str = "q_model",
    family_id: str = "f_model",
    concepts: tuple[ConceptWeight, ...] | None = None,
    difficulty: float = 0.0,
    discrimination: float = 1.5,
    guess_rate: float = 0.25,
    slip_rate: float = 0.05,
) -> Question:
    return Question(
        id=question_id,
        version=1,
        family_id=family_id,
        status=QuestionStatus.CALIBRATED,
        stem="Which carefully specified model response follows from the evidence in this scenario?",
        kind=QuestionKind.CONCEPTUAL,
        difficulty=difficulty,
        discrimination=discrimination,
        guess_rate=guess_rate,
        slip_rate=slip_rate,
        concepts=concepts
        or (ConceptWeight("c_primary", 1.0, ConceptRole.PRIMARY),),
        options=(
            Option("a", "The calibrated response is justified.", True, "This is the keyed response."),
            Option("b", "The response reverses the evidence.", False, "This reverses the evidence."),
            Option("c", "The response ignores the condition.", False, "This ignores the condition."),
            Option("d", "The response assumes a new premise.", False, "This assumes a new premise."),
        ),
        source_ids=("s_model",),
    )


def model_connection(*concept_ids: str) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE concepts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            domain TEXT NOT NULL,
            prior_mastery REAL NOT NULL
        );
        CREATE TABLE attempts (
            event_id TEXT PRIMARY KEY,
            learner_id TEXT NOT NULL,
            family_id TEXT NOT NULL
        );
        CREATE TABLE misconceptions (
            id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL
        );
        CREATE TABLE skill_states (
            learner_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            mean REAL NOT NULL,
            variance REAL NOT NULL,
            stability_hours REAL NOT NULL,
            exposures INTEGER NOT NULL,
            last_seen_at TEXT,
            next_review_at TEXT,
            evidence_mass REAL NOT NULL,
            as_of_event_id TEXT,
            model_version TEXT NOT NULL,
            PRIMARY KEY (learner_id, concept_id)
        );
        CREATE TABLE learner_skill_families (
            learner_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            family_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            first_unguided_correct_at TEXT NOT NULL,
            last_unguided_correct_at TEXT NOT NULL,
            delayed_unguided_correct_at TEXT,
            PRIMARY KEY (learner_id, concept_id, family_id)
        );
        CREATE TABLE misconception_beliefs (
            learner_id TEXT NOT NULL,
            misconception_id TEXT NOT NULL,
            log_odds REAL NOT NULL,
            evidence_count INTEGER NOT NULL,
            last_seen_at TEXT NOT NULL,
            as_of_event_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            PRIMARY KEY (learner_id, misconception_id)
        );
        """
    )
    for concept_id in concept_ids or ("c_primary",):
        connection.execute(
            "INSERT INTO concepts VALUES (?, ?, ?, ?, ?)",
            (concept_id, concept_id, "A model-test concept.", "test", 0.20),
        )
    return connection


def insert_state(
    connection: sqlite3.Connection,
    *,
    concept_id: str = "c_primary",
    mean: float = 0.0,
    variance: float = 1.0,
    stability_hours: float = 100.0,
    last_seen_at: datetime | None = NOW - timedelta(hours=1),
) -> None:
    connection.execute(
        """INSERT INTO skill_states VALUES (
               'learner', ?, ?, ?, ?, 1, ?, NULL, 0.0, NULL, 'test'
           )""",
        (
            concept_id,
            mean,
            variance,
            stability_hours,
            last_seen_at.isoformat() if last_seen_at else None,
        ),
    )


def project_response(
    connection: sqlite3.Connection,
    question: Question,
    *,
    event_id: str,
    now: datetime = NOW,
    hint_count: int = 0,
    feedback_shown: bool = False,
    correct: bool = True,
    confidence: float | None = 0.8,
    response_ms: int | None = None,
) -> SkillState:
    connection.execute(
        "INSERT INTO attempts VALUES (?, 'learner', ?)",
        (event_id, question.family_id),
    )
    selected = question.correct_option if correct else question.options[1]
    states, _ = LearnerModel().update_from_response(
        connection,
        learner_id="learner",
        question=question,
        selected_option=selected,
        confidence=confidence,
        hint_count=hint_count,
        feedback_shown=feedback_shown,
        evidence_weight_override=1.0,
        event_id=event_id,
        now=now,
        response_ms=response_ms,
    )
    return states[question.primary_concept_id]


class LearnerMathTests(unittest.TestCase):
    def test_repeated_wrong_feedback_cannot_manufacture_mastery(self) -> None:
        question = make_question()
        connection = model_connection("c_primary")
        try:
            state = None
            for index in range(200):
                state = project_response(
                    connection,
                    question,
                    event_id=f"wrong-feedback-{index}",
                    now=NOW + timedelta(minutes=index),
                    correct=False,
                    feedback_shown=True,
                    confidence=0.95,
                    response_ms=900,
                )
            assert state is not None
            self.assertLess(state.expected_competence, 0.30)
            self.assertLess(state.mastery_probability, 0.10)
            self.assertLess(state.mean, logit(0.20))
            self.assertLess(state.evidence_mass, 1.42)
        finally:
            connection.close()

    def test_uncertain_or_instant_correct_does_not_certify_family_retrieval(self) -> None:
        question = make_question()
        scenarios = ((0.10, 900), (0.90, 0))
        for confidence, response_ms in scenarios:
            with self.subTest(confidence=confidence, response_ms=response_ms):
                connection = model_connection("c_primary")
                try:
                    insert_state(connection, stability_hours=100.0)
                    state = project_response(
                        connection,
                        question,
                        event_id=f"uncertain-{confidence}-{response_ms}",
                        confidence=confidence,
                        response_ms=response_ms,
                    )
                    family_count = connection.execute(
                        "SELECT COUNT(*) AS n FROM learner_skill_families"
                    ).fetchone()["n"]
                    self.assertEqual(family_count, 0)
                    self.assertEqual(state.stability_hours, 100.0)
                finally:
                    connection.close()

    def test_mastery_probability_moves_with_mean_and_uncertainty(self) -> None:
        def state(mean: float, variance: float) -> SkillState:
            return SkillState("learner", "concept", mean, variance, 48.0)

        low = state(-1.0, 0.4)
        high = state(1.0, 0.4)
        self.assertLess(low.mastery_probability, high.mastery_probability)

        boundary = logit(0.65)
        confident_above = state(boundary + 0.35, 0.05)
        uncertain_above = state(boundary + 0.35, 2.0)
        self.assertGreater(
            confident_above.mastery_probability,
            uncertain_above.mastery_probability,
        )

        confident_below = state(boundary - 0.35, 0.05)
        uncertain_below = state(boundary - 0.35, 2.0)
        self.assertLess(
            confident_below.mastery_probability,
            uncertain_below.mastery_probability,
        )
        self.assertAlmostEqual(state(boundary, 0.05).mastery_probability, 0.5)
        self.assertAlmostEqual(state(boundary, 3.0).mastery_probability, 0.5)

    def test_inactivity_never_improves_a_below_prior_posterior(self) -> None:
        concept = Concept("c", "Concept", "Description", prior_mastery=0.40)
        state = SkillState(
            "learner",
            "c",
            mean=logit(0.12),
            variance=0.30,
            stability_hours=36.0,
            exposures=3,
            last_seen_at=NOW - timedelta(days=365),
            evidence_mass=2.0,
        )
        projected = LearnerModel().project_state(state, concept, NOW)
        self.assertEqual(projected.mean, state.mean)
        self.assertEqual(projected.variance, state.variance)
        self.assertLessEqual(projected.mastery_probability, state.mastery_probability)

    def test_variance_widening_cannot_manufacture_mastery_or_competence(self) -> None:
        concept = Concept("c", "Concept", "Description", prior_mastery=0.20)
        # A very precise posterior immediately below the mastery boundary is
        # the adversarial case: unconstrained variance inflation can increase
        # its upper tail much faster than its mean decays.
        state = SkillState(
            "learner",
            "c",
            mean=logit(0.65) - 0.05,
            variance=0.001,
            stability_hours=48.0,
            exposures=8,
            last_seen_at=NOW - timedelta(hours=1),
            evidence_mass=7.0,
        )
        projected = LearnerModel().project_state(state, concept, NOW)
        self.assertLess(projected.mean, state.mean)
        self.assertGreaterEqual(projected.variance, state.variance)
        self.assertLessEqual(projected.mastery_probability, state.mastery_probability + 1e-12)
        self.assertLessEqual(projected.expected_competence, state.expected_competence + 1e-12)

        near_zero = SkillState(
            "learner",
            "c",
            mean=-1e-300,
            variance=0.001,
            stability_hours=48.0,
            exposures=2,
            last_seen_at=NOW - timedelta(hours=1),
        )
        projected_near_zero = LearnerModel().project_state(near_zero, concept, NOW)
        self.assertTrue(math.isfinite(projected_near_zero.variance))
        self.assertLessEqual(
            projected_near_zero.expected_competence,
            near_zero.expected_competence,
        )

    def test_review_urgency_is_a_continuous_bounded_ramp(self) -> None:
        due = NOW
        state = SkillState(
            "learner",
            "c",
            mean=0.0,
            variance=1.0,
            stability_hours=48.0,
            next_review_at=due,
        )
        model = LearnerModel()
        self.assertEqual(model.retention_due_value(state, due - timedelta(hours=24)), 0.0)
        self.assertAlmostEqual(
            model.retention_due_value(state, due - timedelta(hours=12)), 0.25
        )
        self.assertAlmostEqual(model.retention_due_value(state, due), 0.5)
        self.assertAlmostEqual(
            model.retention_due_value(state, due + timedelta(hours=12)), 0.75
        )
        self.assertEqual(model.retention_due_value(state, due + timedelta(hours=24)), 1.0)
        self.assertEqual(model.retention_due_value(state, due + timedelta(days=30)), 1.0)

    def test_supporting_and_context_roles_never_receive_scored_evidence(self) -> None:
        question = make_question(
            concepts=(
                ConceptWeight("c_primary", 0.50, ConceptRole.PRIMARY),
                ConceptWeight("c_support", 0.25, ConceptRole.SUPPORTING),
                ConceptWeight("c_context", 0.25, ConceptRole.CONTEXT),
            )
        )
        model = LearnerModel()
        self.assertEqual(model.evidence_weights(question), {"c_primary": 1.0})
        self.assertFalse(ConceptRole.SUPPORTING.carries_scored_evidence)
        self.assertFalse(ConceptRole.CONTEXT.carries_scored_evidence)

        connection = model_connection("c_primary", "c_support", "c_context")
        try:
            states, changes = model.update_from_response(
                connection,
                learner_id="learner",
                question=question,
                selected_option=question.correct_option,
                confidence=0.8,
                hint_count=0,
                feedback_shown=False,
                evidence_weight_override=1.0,
                event_id="event-role",
                now=NOW,
            )
            self.assertEqual(set(states), {"c_primary"})
            self.assertEqual({change["concept_id"] for change in changes}, {"c_primary"})
            persisted = {
                row["concept_id"]
                for row in connection.execute("SELECT concept_id FROM skill_states")
            }
            self.assertEqual(persisted, {"c_primary"})

            connection.execute(
                "INSERT INTO misconceptions VALUES ('m_support', 'c_support')"
            )
            contextual_distractor = Option(
                "b",
                "The response reverses the evidence.",
                False,
                "This reverses the evidence.",
                "m_support",
            )
            question_with_contextual_misconception = Question(
                id="q_contextual_misconception",
                version=question.version,
                family_id="f_contextual_misconception",
                status=question.status,
                stem=question.stem,
                kind=question.kind,
                difficulty=question.difficulty,
                discrimination=question.discrimination,
                guess_rate=question.guess_rate,
                slip_rate=question.slip_rate,
                concepts=question.concepts,
                options=(question.options[0], contextual_distractor, *question.options[2:]),
                source_ids=question.source_ids,
            )
            model.update_from_response(
                connection,
                learner_id="learner",
                question=question_with_contextual_misconception,
                selected_option=contextual_distractor,
                confidence=0.9,
                hint_count=0,
                feedback_shown=False,
                evidence_weight_override=1.0,
                event_id="event-context-belief",
                now=NOW + timedelta(hours=1),
            )
            belief_count = connection.execute(
                "SELECT COUNT(*) AS n FROM misconception_beliefs"
            ).fetchone()["n"]
            self.assertEqual(belief_count, 0)
        finally:
            connection.close()

    def test_one_family_has_bounded_cumulative_influence(self) -> None:
        model = LearnerModel()
        discounts = [model.family_dependence_discount(index) for index in range(100_000)]
        self.assertEqual(discounts[0], 1.0)
        self.assertEqual(discounts[1], 0.25)
        self.assertTrue(all(a > b for a, b in zip(discounts[1:], discounts[2:])))
        self.assertGreater(sum(discounts), 1.40)
        self.assertLess(sum(discounts), 1.412)
        with self.assertRaises(ValueError):
            model.family_dependence_discount(-1)

        connection = model_connection("c_primary")
        try:
            question = make_question()
            after = None
            for index in range(250):
                after = project_response(
                    connection,
                    question,
                    event_id=f"repeat-{index}",
                    now=NOW + timedelta(seconds=index),
                )
            assert after is not None
            self.assertGreater(after.evidence_mass, 1.40)
            self.assertLess(after.evidence_mass, 1.412)
        finally:
            connection.close()

    def test_spacing_is_measured_from_last_successful_retrieval(self) -> None:
        question = make_question()

        recent = model_connection("c_primary")
        try:
            insert_state(recent)
            recent.execute(
                """INSERT INTO learner_skill_families VALUES (
                       'learner', 'c_primary', ?, 'conceptual', ?, ?, NULL
                   )""",
                (
                    question.family_id,
                    (NOW - timedelta(days=10)).isoformat(),
                    (NOW - timedelta(hours=2)).isoformat(),
                ),
            )
            after = project_response(recent, question, event_id="event-recent")
            self.assertEqual(after.stability_hours, 100.0)
            family = recent.execute(
                "SELECT * FROM learner_skill_families"
            ).fetchone()
            self.assertIsNone(family["delayed_unguided_correct_at"])
        finally:
            recent.close()

        spaced = model_connection("c_primary")
        try:
            insert_state(spaced)
            spaced.execute(
                """INSERT INTO learner_skill_families VALUES (
                       'learner', 'c_primary', ?, 'conceptual', ?, ?, NULL
                   )""",
                (
                    question.family_id,
                    (NOW - timedelta(days=10)).isoformat(),
                    (NOW - timedelta(hours=72)).isoformat(),
                ),
            )
            # A later failed attempt belongs to the family, but it is not a
            # successful retrieval and therefore does not reset spacing.
            spaced.execute(
                "INSERT INTO attempts VALUES ('prior-wrong', 'learner', ?)",
                (question.family_id,),
            )
            after = project_response(spaced, question, event_id="event-spaced")
            self.assertGreater(after.stability_hours, 100.0)
            family = spaced.execute(
                "SELECT * FROM learner_skill_families"
            ).fetchone()
            self.assertEqual(family["delayed_unguided_correct_at"], NOW.isoformat())
        finally:
            spaced.close()

    def test_hinted_correct_response_is_weaker_evidence(self) -> None:
        question = make_question()
        unguided_connection = model_connection("c_primary")
        hinted_connection = model_connection("c_primary")
        try:
            insert_state(unguided_connection, mean=-0.5)
            insert_state(hinted_connection, mean=-0.5)
            unguided = project_response(
                unguided_connection,
                question,
                event_id="event-unguided",
                hint_count=0,
            )
            hinted = project_response(
                hinted_connection,
                question,
                event_id="event-hinted",
                hint_count=1,
            )
            self.assertAlmostEqual(unguided.evidence_mass, 1.0)
            self.assertAlmostEqual(hinted.evidence_mass, 0.2)
            self.assertGreater(unguided.mean, hinted.mean)
            self.assertLess(unguided.variance, hinted.variance)
            self.assertGreater(unguided.stability_hours, hinted.stability_hours)
        finally:
            unguided_connection.close()
            hinted_connection.close()

    def test_extreme_valid_items_keep_probabilities_and_variances_finite(self) -> None:
        model = LearnerModel()
        cases = (
            ("easy", -4.0, 3.0, 6.0, 1e-12, True),
            ("hard", 4.0, 3.0, -6.0, MAX_POSTERIOR_VARIANCE, False),
        )
        for name, difficulty, discrimination, mean, variance, correct in cases:
            with self.subTest(name=name):
                question = make_question(
                    question_id=f"q_{name}",
                    family_id=f"f_{name}",
                    difficulty=difficulty,
                    discrimination=discrimination,
                    guess_rate=0.25,
                    slip_rate=0.25,
                )
                state = SkillState(
                    "learner",
                    "c_primary",
                    mean=mean,
                    variance=variance,
                    stability_hours=12.0,
                )
                predicted = model.predict_correct(question, {"c_primary": state})
                information = model.expected_information_gain(
                    question, {"c_primary": state}
                )
                self.assertTrue(math.isfinite(predicted))
                self.assertGreater(predicted, 0.0)
                self.assertLess(predicted, 1.0)
                self.assertTrue(math.isfinite(information))
                self.assertGreaterEqual(information, 0.0)

                connection = model_connection("c_primary")
                try:
                    insert_state(
                        connection,
                        mean=mean,
                        variance=variance,
                        stability_hours=12.0,
                        last_seen_at=None,
                    )
                    after = project_response(
                        connection,
                        question,
                        event_id=f"event-{name}",
                        correct=correct,
                        feedback_shown=True,
                    )
                    self.assertTrue(math.isfinite(after.mean))
                    self.assertTrue(math.isfinite(after.variance))
                    self.assertGreaterEqual(after.variance, MIN_POSTERIOR_VARIANCE)
                    self.assertLessEqual(after.variance, MAX_POSTERIOR_VARIANCE)
                    self.assertTrue(math.isfinite(after.mastery_probability))
                    self.assertGreaterEqual(after.mastery_probability, 0.0)
                    self.assertLessEqual(after.mastery_probability, 1.0)
                finally:
                    connection.close()


if __name__ == "__main__":
    unittest.main()
