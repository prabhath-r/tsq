# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.learner import CONCEPT_MODEL_VERSION, MODEL_VERSION, LearnerModel
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2100, 3, 1, 9, 0, tzinfo=timezone.utc)


class SessionReportingTests(unittest.TestCase):
    def test_child_topic_question_is_inside_requested_parent_topic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "report.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(database)
            engine.create_learner("report-learner")
            session = engine.start_session(
                "report-learner",
                "t_large_language_models",
                seed=0,
            )

            presentation = engine.next_question(session["id"], now=START)
            self.assertEqual(
                database.question_topics(
                    presentation.question.id,
                    session["corpus_release_id"],
                )[0]["id"],
                "t_transformers",
            )
            engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.90,
                response_ms=4_000,
                hint_count=0,
                feedback_shown=True,
                now=START + timedelta(seconds=4),
            )

            report = engine.session_report(
                session["id"], now=START + timedelta(seconds=4)
            )
            self.assertEqual(report["topic"]["id"], "t_large_language_models")
            self.assertEqual(report["outside_requested_topic_questions"], 0)

    def test_wrong_family_is_reported_as_observed_not_successful(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "report.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(database)
            learner_id = "negative-family-evidence"
            engine.create_learner(learner_id)
            session = engine.start_session(
                learner_id,
                "t_transformers",
                seed=0,
                now=START,
            )
            presentation = engine.next_question(session["id"], now=START)
            wrong = next(
                option
                for option in presentation.question.options
                if not option.correct
            )
            engine.submit_answer(
                presentation.decision_id,
                wrong.id,
                confidence=0.90,
                response_ms=4_000,
                hint_count=0,
                feedback_shown=True,
                now=START + timedelta(seconds=4),
            )

            report = engine.session_report(
                session["id"], now=START + timedelta(seconds=4)
            )
            objective_id = presentation.question.objective_id
            objective = next(
                row
                for row in report["objective_performance"]
                if row["objective_id"] == objective_id
            )
            self.assertEqual(objective["session"]["observed_families"], 1)
            self.assertEqual(
                objective["session"]["correct_response_families"], 0
            )
            self.assertEqual(
                objective["session"]["successful_retrieval_families"], 0
            )
            self.assertEqual(
                objective["current_projection"]["independent_families"], 0
            )
            self.assertEqual(
                objective["current_projection"][
                    "successful_retrieval_families"
                ],
                0,
            )

            profile = engine.profile(learner_id, now=START + timedelta(seconds=4))
            objective_profile = next(
                row
                for row in profile["learning_objectives"]
                if row["objective_id"] == objective_id
            )
            self.assertEqual(
                objective_profile["observed_response_families"], 1
            )
            self.assertEqual(
                objective_profile["successful_retrieval_families"], 0
            )

    def test_exact_projection_and_missing_timing_are_reported_truthfully(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "report.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(database)
            learner_id = "exact-reporting"
            engine.create_learner(learner_id)
            session = engine.start_session(
                learner_id,
                "t_transformers",
                seed=3,
                now=START,
            )
            presentation = engine.next_question(session["id"], now=START)
            self.assertIsNotNone(presentation.question.objective_id)
            engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.90,
                response_ms=None,
                hint_count=0,
                feedback_shown=False,
                now=START + timedelta(seconds=4),
            )

            report = engine.session_report(
                session["id"], now=START + timedelta(seconds=4)
            )
            objective = next(
                row
                for row in report["objective_performance"]
                if row["objective_id"] == presentation.question.objective_id
            )
            self.assertEqual(objective["session"]["correct"], 1)
            self.assertEqual(objective["session"]["missing_response_time"], 1)
            self.assertEqual(objective["session"]["uncertain_responses"], 1)
            self.assertEqual(
                objective["session"]["successful_retrieval_families"], 0
            )
            projection = objective["current_projection"]
            self.assertEqual(projection["inference_model_version"], MODEL_VERSION)
            self.assertEqual(projection["posterior_representation"], "exact_grid")
            self.assertEqual(len(projection["posterior_digest"]), 64)
            self.assertGreater(
                projection["mastery_probability_error_bound"], 0.0
            )
            self.assertLessEqual(
                projection["mastery_probability"],
                projection["estimated_mastery_probability"],
            )
            self.assertEqual(report["response_time"]["missing_values"], 1)

            profile = engine.profile(
                learner_id, now=START + timedelta(seconds=4)
            )
            objective_profile = next(
                row
                for row in profile["learning_objectives"]
                if row["objective_id"] == presentation.question.objective_id
            )
            self.assertEqual(
                objective_profile["inference_model_version"], MODEL_VERSION
            )
            self.assertEqual(
                objective_profile["posterior_representation"], "exact_grid"
            )
            concept_profile = next(
                row
                for row in profile["skills"]
                if row["concept_id"]
                == presentation.question.primary_concept_id
            )
            self.assertEqual(
                concept_profile["projection_kind"],
                "derived_objective_readiness_floor",
            )

    def test_v7_missing_confidence_is_counted_as_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "report.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(database)
            learner_id = "missing-confidence-reporting"
            engine.create_learner(learner_id)
            session = engine.start_session(
                learner_id,
                "t_transformers",
                seed=5,
                now=START,
            )
            presentation = engine.next_question(session["id"], now=START)
            engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=None,
                response_ms=4_000,
                hint_count=0,
                now=START + timedelta(seconds=4),
            )

            report = engine.session_report(
                session["id"], now=START + timedelta(seconds=4)
            )
            objective = next(
                row
                for row in report["objective_performance"]
                if row["objective_id"] == presentation.question.objective_id
            )
            concept = next(
                row
                for row in report["concept_performance"]
                if row["concept_id"]
                == presentation.question.primary_concept_id
            )
            self.assertEqual(objective["session"]["uncertain_responses"], 1)
            self.assertEqual(concept["session"]["uncertain_responses"], 1)
            self.assertEqual(
                objective["session"]["successful_retrieval_families"], 0
            )

    def test_selection_window_report_uses_exact_integer_milliseconds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "report.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(database)
            engine.create_learner("exact-window-reporting")
            session = engine.start_session(
                "exact-window-reporting",
                "t_transformers",
                seed=7,
                now=START,
            )
            presentation = engine.next_question(session["id"], now=START)
            answered_at = START + timedelta(milliseconds=1_001)
            engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.90,
                response_ms=1_001,
                hint_count=0,
                now=answered_at,
            )

            report = engine.session_report(
                session["id"], now=answered_at
            )

            self.assertEqual(
                report["response_time"][
                    "selection_window_inconsistencies"
                ],
                0,
            )

    def test_legacy_response_window_is_not_retroactively_judged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "report.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(
                database,
                LearnerModel(CONCEPT_MODEL_VERSION),
            )
            engine.create_learner("legacy-window-reporting")
            session = engine.start_session(
                "legacy-window-reporting",
                "c_clustering",
                seed=11,
                now=START,
            )
            presentation = engine.next_question(session["id"], now=START)
            answered_at = START + timedelta(milliseconds=1)
            engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=None,
                response_ms=4_000,
                hint_count=0,
                now=answered_at,
            )

            report = engine.session_report(
                session["id"], now=answered_at
            )

            self.assertEqual(
                report["response_time"][
                    "selection_window_inconsistencies"
                ],
                0,
            )
            self.assertIn(
                "legacy telemetry is not retrospectively judged",
                report["response_time"]["evidence_contract"],
            )


if __name__ == "__main__":
    unittest.main()
