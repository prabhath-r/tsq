# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.learner import MODEL_VERSION
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
NOW = datetime(2100, 8, 1, 12, 0, tzinfo=timezone.utc)


class ObjectiveEvidenceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "objective-v6.db")
        self.database.initialize()
        parsed = read_and_parse(CORPUS, include_catalog=True)
        self.questions = parsed[4]
        self.release_id = self.database.import_corpus(*parsed)["release_id"]
        self.engine = AdaptiveEngine(self.database)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _apply(
        self,
        learner_id: str,
        question,
        *,
        correct: bool,
        selected: bool = True,
    ) -> None:
        with self.database.transaction() as connection:
            event = self.database.append_event(
                connection,
                stream_id=f"learner:{learner_id}",
                event_type="ObjectiveV6EvidenceTested",
                payload={"question_id": question.id, "correct": correct},
                learner_id=learner_id,
                occurred_at=NOW,
            )
            option = None
            if selected:
                option = (
                    question.correct_option
                    if correct
                    else next(item for item in question.options if not item.correct)
                )
            self.engine.learner_model.update_from_response(
                connection,
                learner_id=learner_id,
                question=question,
                selected_option=option,
                confidence=0.9,
                hint_count=0,
                feedback_shown=False,
                evidence_weight_override=1.0,
                event_id=event["event_id"],
                now=NOW,
                response_ms=1200,
                prior_family_attempts_override=0,
            )

    def test_static_distinct_family_evidence_commutes_through_storage(self) -> None:
        by_objective: dict[str, list] = {}
        for question in self.questions:
            if question.objective_id is not None:
                by_objective.setdefault(question.objective_id, []).append(question)
        objective_id, questions = next(
            (objective_id, items[:3])
            for objective_id, items in sorted(by_objective.items())
            if len({item.family_id for item in items}) >= 3
        )
        questions = list({item.family_id: item for item in questions}.values())[:3]
        outcomes = {questions[0].id: True, questions[1].id: False, questions[2].id: True}
        for learner_id, ordered in (
            ("forward", questions),
            ("reverse", list(reversed(questions))),
        ):
            self.engine.create_learner(learner_id)
            for question in ordered:
                self._apply(
                    learner_id,
                    question,
                    correct=outcomes[question.id],
                )

        forward = self.database.get_objective_states("forward")[objective_id]
        reverse = self.database.get_objective_states("reverse")[objective_id]
        self.assertEqual(forward.model_version, MODEL_VERSION)
        self.assertIsNotNone(forward.posterior)
        self.assertEqual(len(forward.posterior.pending_observations), 3)
        self.assertAlmostEqual(forward.mean, reverse.mean, places=14)
        self.assertAlmostEqual(forward.variance, reverse.variance, places=14)
        self.assertAlmostEqual(
            forward.mastery_probability,
            reverse.mastery_probability,
            places=14,
        )
        self.assertAlmostEqual(
            forward.expected_competence,
            reverse.expected_competence,
            places=14,
        )
        with self.database.read() as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) AS n FROM objective_grid_states"
                ).fetchone()["n"],
                2,
            )

    def test_abstention_is_weaker_than_a_named_wrong_response(self) -> None:
        question = next(
            item
            for item in self.questions
            if item.objective_id is not None and item.misconception_ids
        )
        for learner_id in ("named-wrong", "abstained"):
            self.engine.create_learner(learner_id)
        self._apply("named-wrong", question, correct=False, selected=True)
        self._apply("abstained", question, correct=False, selected=False)

        named = self.database.get_objective_states("named-wrong")[
            question.objective_id
        ]
        abstained = self.database.get_objective_states("abstained")[
            question.objective_id
        ]
        self.assertGreater(named.evidence_mass, abstained.evidence_mass)
        self.assertLess(named.expected_competence, abstained.expected_competence)
        self.assertLess(named.mastery_probability, abstained.mastery_probability)
        self.assertLess(named.variance, abstained.variance)
        self.assertTrue(
            self.database.get_misconception_beliefs("named-wrong")
        )
        self.assertEqual(
            self.database.get_misconception_beliefs("abstained"), {}
        )


if __name__ == "__main__":
    unittest.main()
