# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from math import comb
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.learner import CONCEPT_MODEL_VERSION, MODEL_VERSION, LearnerModel
from tsq.response_patterns import (
    POSITION_ANALYSIS_WINDOW,
    analyze_position_observations,
)
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
START = datetime(2100, 3, 1, 9, 0, tzinfo=timezone.utc)


def exact_binomial_upper_tail(
    probability: Fraction,
    trials: int,
    observed: int,
) -> Fraction:
    return sum(
        (
            Fraction(comb(trials, successes))
            * probability**successes
            * (1 - probability) ** (trials - successes)
        )
        for successes in range(observed, trials + 1)
    )


class ResponsePositionAnalysisTests(unittest.TestCase):
    def test_small_stream_matches_unbounded_exact_binomial_reference(self) -> None:
        observations = tuple((position, 4) for position in range(4)) * 6

        result = analyze_position_observations(observations)

        window = result["window"]
        self.assertEqual(window["total_non_abstained_observations"], 24)
        self.assertEqual(window["analyzed_non_abstained_observations"], 24)
        self.assertEqual(window["truncated_non_abstained_observations"], 0)
        for test in result["inference"]["position_tests"]:
            expected = exact_binomial_upper_tail(Fraction(1, 4), 24, 6)
            self.assertEqual(
                test["raw_upper_tail_probability"]["exact"],
                f"{expected.numerator}/{expected.denominator}",
            )
            self.assertEqual(
                test["calculation"],
                "exact_binomial_equal_probability",
            )

    def test_large_stream_analyzes_only_recent_bounded_window(self) -> None:
        earlier = tuple((index % 4, 4) for index in range(9_744))
        observations = earlier + ((0, 4),) * POSITION_ANALYSIS_WINDOW

        result = analyze_position_observations(observations)

        window = result["window"]
        self.assertEqual(window["total_non_abstained_observations"], 10_000)
        self.assertEqual(
            window["analyzed_non_abstained_observations"],
            POSITION_ANALYSIS_WINDOW,
        )
        self.assertEqual(
            window["truncated_non_abstained_observations"],
            10_000 - POSITION_ANALYSIS_WINDOW,
        )
        dominant = result["inference"]["dominant_position"]
        self.assertEqual(dominant["display_position"], 1)
        self.assertEqual(
            dominant["selected_count"],
            POSITION_ANALYSIS_WINDOW,
        )
        self.assertEqual(
            dominant["calculation"],
            "exact_binomial_equal_probability",
        )
        self.assertEqual(
            dominant["raw_upper_tail_probability"]["exact"],
            f"1/{4 ** POSITION_ANALYSIS_WINDOW}",
        )

    def test_mixed_option_counts_use_bounded_exact_dynamic_program(self) -> None:
        recent = tuple(
            (index % option_count, option_count)
            for index, option_count in enumerate(
                (3, 4) * (POSITION_ANALYSIS_WINDOW // 2)
            )
        )
        observations = ((0, 3),) * 50 + recent

        result = analyze_position_observations(observations)

        self.assertEqual(
            result["window"]["analyzed_non_abstained_observations"],
            POSITION_ANALYSIS_WINDOW,
        )
        self.assertEqual(
            result["window"]["truncated_non_abstained_observations"],
            50,
        )
        self.assertTrue(
            all(
                test["calculation"]
                == "exact_poisson_binomial_dynamic_program"
                for test in result["inference"]["position_tests"]
            )
        )


def run_position_pattern(
    directory: str,
    *,
    learner_id: str,
    count: int,
    selector,
) -> tuple[Database, AdaptiveEngine, dict, datetime, set[str]]:
    database = Database(Path(directory) / "report.db")
    database.initialize()
    database.import_corpus(
        *read_and_parse(CORPUS, include_catalog=True)
    )
    engine = AdaptiveEngine(database)
    engine.create_learner(learner_id)
    session = engine.start_session(
        learner_id,
        # This integration helper deliberately needs up to 24 observations in
        # one session.  Use the broad released curriculum so the position-habit
        # test does not depend on one narrower topic's remediation capacity.
        "t_machine_learning",
        seed=71,
        now=START,
    )
    current = START
    first_displayed_ids: set[str] = set()
    for index in range(count):
        presentation = engine.next_question(session["id"], now=current)
        first_displayed_ids.add(presentation.ordered_options[0].id)
        selected = selector(presentation, index)
        answered_at = current + timedelta(seconds=2)
        engine.submit_answer(
            presentation.decision_id,
            selected,
            confidence=0.75 if selected is not None else 0.20,
            response_ms=1_200,
            hint_count=0,
            feedback_shown=True,
            idempotency_key=f"{learner_id}:{index}",
            now=answered_at,
        )
        current = answered_at + timedelta(minutes=5)
    return database, engine, session, current, first_displayed_ids


class SessionReportingTests(unittest.TestCase):
    def test_position_shadow_detects_true_first_displayed_habit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _database, engine, session, current, first_ids = run_position_pattern(
                directory,
                learner_id="first-position-habit",
                count=16,
                selector=lambda presentation, _index: (
                    presentation.ordered_options[0].id
                ),
            )
            report = engine.session_report(session["id"], now=current)

        shadow = report["response_position_shadow"]
        self.assertGreaterEqual(len(first_ids), 3)
        self.assertTrue(shadow["observational_only"])
        self.assertFalse(shadow["affects_mastery"])
        self.assertFalse(shadow["affects_certification"])
        self.assertFalse(shadow["affects_selection"])
        self.assertTrue(shadow["evidence"]["boundary_valid"])
        self.assertEqual(
            shadow["evidence"]["non_abstained_observations"], 16
        )
        self.assertEqual(
            shadow["evidence"]["analyzed_non_abstained_observations"], 16
        )
        self.assertEqual(
            shadow["evidence"]["truncated_non_abstained_observations"], 0
        )
        self.assertEqual(
            shadow["test_contract"][
                "maximum_recent_non_abstained_observations"
            ],
            POSITION_ANALYSIS_WINDOW,
        )
        inference = shadow["inference"]
        self.assertEqual(
            inference["status"], "position_concentration_signal"
        )
        dominant = inference["dominant_position"]
        self.assertEqual(dominant["display_position"], 1)
        self.assertEqual(dominant["selected_count"], 16)
        self.assertEqual(
            dominant["expected_count"]["exact"], "4/1"
        )
        self.assertEqual(
            dominant["raw_upper_tail_probability"]["exact"],
            "1/4294967296",
        )
        self.assertEqual(
            dominant["bonferroni_adjusted_probability"]["exact"],
            "1/1073741824",
        )

    def test_position_shadow_does_not_flag_content_based_correct_answers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _database, engine, session, current, _first_ids = run_position_pattern(
                directory,
                learner_id="content-based-answers",
                count=24,
                selector=lambda presentation, _index: (
                    presentation.question.correct_option.id
                ),
            )
            report = engine.session_report(session["id"], now=current)

        shadow = report["response_position_shadow"]
        self.assertTrue(shadow["evidence"]["boundary_valid"])
        self.assertEqual(
            shadow["inference"]["status"], "no_signal"
        )
        observed_positions = {
            row["display_position"]
            for row in shadow["inference"]["position_tests"]
            if row["selected_count"]
        }
        self.assertEqual(observed_positions, {1, 2, 3, 4})
        self.assertFalse(
            any(
                row["familywise_signal"]
                for row in shadow["inference"]["position_tests"]
            )
        )

    def test_position_shadow_is_inconclusive_for_small_abstaining_sample(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _database, engine, session, current, _first_ids = run_position_pattern(
                directory,
                learner_id="small-position-sample",
                count=8,
                selector=lambda presentation, index: (
                    presentation.ordered_options[0].id
                    if index < 4
                    else None
                ),
            )
            report = engine.session_report(session["id"], now=current)

        shadow = report["response_position_shadow"]
        self.assertTrue(shadow["evidence"]["boundary_valid"])
        self.assertEqual(shadow["evidence"]["answered_observations"], 8)
        self.assertEqual(shadow["evidence"]["non_abstained_observations"], 4)
        self.assertEqual(shadow["evidence"]["abstentions_excluded"], 4)
        self.assertEqual(shadow["inference"]["status"], "inconclusive")
        self.assertEqual(
            shadow["inference"]["reason"],
            "insufficient_non_abstained_observations",
        )
        self.assertEqual(shadow["inference"]["position_tests"], [])

    def test_position_shadow_fails_closed_on_tampered_order_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database, engine, session, current, _first_ids = run_position_pattern(
                directory,
                learner_id="tampered-position-boundary",
                count=12,
                selector=lambda presentation, _index: (
                    presentation.ordered_options[0].id
                ),
            )
            before_hash = database.learner_projection_hash(
                "tampered-position-boundary"
            )
            with database.transaction() as connection:
                connection.execute(
                    """UPDATE decisions SET option_order_json = '{'
                       WHERE id = (
                           SELECT decision_id FROM attempts
                           WHERE session_id = ?
                           ORDER BY answered_at, id LIMIT 1
                       )""",
                    (session["id"],),
                )
            report = engine.session_report(session["id"], now=current)
            after_hash = database.learner_projection_hash(
                "tampered-position-boundary"
            )

        shadow = report["response_position_shadow"]
        self.assertEqual(before_hash, after_hash)
        self.assertFalse(shadow["evidence"]["boundary_valid"])
        self.assertTrue(shadow["evidence"]["boundary_errors"])
        self.assertEqual(shadow["inference"]["status"], "unavailable")
        self.assertEqual(
            shadow["inference"]["reason"],
            "immutable_evidence_boundary_invalid",
        )
        self.assertEqual(shadow["inference"]["position_tests"], [])

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
            catalog = database.get_catalog(session["corpus_release_id"])
            descendants = {"t_large_language_models"}
            while True:
                discovered = {
                    topic["id"]
                    for topic in catalog["topics"]
                    if topic["parent_id"] in descendants
                }
                if discovered.issubset(descendants):
                    break
                descendants.update(discovered)
            question_topic_ids = {
                topic["id"]
                for topic in database.question_topics(
                    presentation.question.id,
                    session["corpus_release_id"],
                )
            }
            self.assertTrue(
                question_topic_ids & descendants,
                (presentation.question.id, question_topic_ids),
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

    def test_abstention_is_not_reported_as_selected_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "report.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(database)
            learner_id = "abstention-reporting"
            engine.create_learner(learner_id)
            session = engine.start_session(
                learner_id,
                "c_agent_tool_use",
                seed=0,
                now=START,
            )
            presentation = engine.next_question(session["id"], now=START)
            engine.submit_answer(
                presentation.decision_id,
                None,
                confidence=0.20,
                response_ms=4_000,
                hint_count=0,
                now=START + timedelta(seconds=4),
            )

            report = engine.session_report(
                session["id"], now=START + timedelta(seconds=4)
            )

        self.assertEqual(report["questions_answered"], 1)
        self.assertEqual(report["correct"], 0)
        self.assertEqual(report["accuracy"], 0.0)
        self.assertEqual(report["abstained"], 1)
        self.assertEqual(report["selected_answers"], 0)
        self.assertEqual(report["selected_incorrect"], 0)
        self.assertIsNone(report["selected_accuracy"])
        self.assertIn(
            "including abstentions",
            report["response_count_definitions"]["accuracy"],
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
        for observed in (objective, concept):
            session_counts = observed["session"]
            # The compatibility count remains one non-correct submission.
            self.assertEqual(session_counts["incorrect"], 1)
            self.assertEqual(session_counts["abstained"], 1)
            self.assertEqual(session_counts["selected_answers"], 0)
            self.assertEqual(session_counts["selected_incorrect"], 0)
            self.assertIsNone(session_counts["selected_accuracy"])
            self.assertEqual(session_counts["verification_failures"], 0)
            self.assertEqual(
                session_counts["verification_inconclusive"], 0
            )
            self.assertNotIn(
                "incorrect_responses", observed["attention_reasons"]
            )
            self.assertNotIn(
                "failed_independent_verification",
                observed["attention_reasons"],
            )
            self.assertIn(
                "uncertain_or_noncredible_evidence",
                observed["attention_reasons"],
            )

    def test_verification_reports_credible_failure_or_inconclusive_evidence(
        self,
    ) -> None:
        cases = (
            ("abstained", None, 0.20, 0, 1, False),
            ("low-credibility", "wrong", 0.20, 0, 1, False),
            ("credible-failure", "wrong", 0.90, 1, 0, True),
        )
        for (
            label,
            selection,
            confidence,
            expected_failures,
            expected_inconclusive,
            expects_failed_reason,
        ) in cases:
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory() as directory,
            ):
                database = Database(Path(directory) / "report.db")
                database.initialize()
                database.import_corpus(
                    *read_and_parse(CORPUS, include_catalog=True)
                )
                engine = AdaptiveEngine(database)
                learner_id = f"verification-{label}"
                engine.create_learner(learner_id)
                session = engine.start_session(
                    learner_id,
                    "c_agent_tool_use",
                    seed=313,
                    now=START,
                )
                first = engine.next_question(session["id"], now=START)
                engine.submit_answer(
                    first.decision_id,
                    first.question.correct_option.id,
                    confidence=0.95,
                    response_ms=0,
                    hint_count=0,
                    now=START + timedelta(seconds=1),
                )
                verification = engine.next_question(
                    session["id"], now=START + timedelta(seconds=2)
                )
                self.assertEqual(
                    verification.pedagogical_role, "verification"
                )
                self.assertEqual(
                    verification.question.objective_id,
                    first.question.objective_id,
                )
                selected_option_id = None
                if selection == "wrong":
                    selected_option_id = next(
                        option.id
                        for option in verification.question.options
                        if not option.correct
                    )
                engine.submit_answer(
                    verification.decision_id,
                    selected_option_id,
                    confidence=confidence,
                    response_ms=900,
                    hint_count=0,
                    now=START + timedelta(seconds=3),
                )

                report = engine.session_report(
                    session["id"], now=START + timedelta(seconds=3)
                )

                objective = next(
                    row
                    for row in report["objective_performance"]
                    if row["objective_id"] == first.question.objective_id
                )
                concept = next(
                    row
                    for row in report["concept_performance"]
                    if row["concept_id"]
                    == first.question.primary_concept_id
                )
                for observed in (objective, concept):
                    session_counts = observed["session"]
                    self.assertEqual(
                        session_counts["verification_failures"],
                        expected_failures,
                    )
                    self.assertEqual(
                        session_counts["verification_inconclusive"],
                        expected_inconclusive,
                    )
                    self.assertEqual(
                        "failed_independent_verification"
                        in observed["attention_reasons"],
                        expects_failed_reason,
                    )
                    self.assertEqual(
                        "inconclusive_independent_verification"
                        in observed["attention_reasons"],
                        bool(expected_inconclusive),
                    )
                if label == "abstained":
                    self.assertEqual(report["accuracy"], 0.5)
                    self.assertEqual(report["selected_answers"], 1)
                    self.assertEqual(report["selected_incorrect"], 0)
                    self.assertEqual(report["selected_accuracy"], 1.0)
                    for observed in (objective, concept):
                        self.assertNotIn(
                            "incorrect_responses",
                            observed["attention_reasons"],
                        )
                elif label == "low-credibility":
                    self.assertEqual(report["selected_answers"], 2)
                    self.assertEqual(report["selected_incorrect"], 1)
                    self.assertEqual(report["selected_accuracy"], 0.5)

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
            inference = report["selected_response_inference"]
            self.assertEqual(
                inference["claim_scope"],
                "provisional_selected_response_inference",
            )
            self.assertEqual(
                inference["model_validation_status"],
                "not_empirically_validated",
            )
            self.assertEqual(
                inference["corpus_calibration_status"],
                "no_calibrated_items",
            )
            self.assertEqual(
                inference["contract_version"],
                "selected-response-inference-v1",
            )
            self.assertEqual(
                inference["corpus_release_id"],
                report["corpus_release_id"],
            )
            self.assertEqual(
                inference["count_scope"], "entire_corpus_release"
            )
            self.assertGreater(inference["eligible_question_count"], 0)
            self.assertEqual(
                inference["eligible_question_count"],
                inference["approved_question_count"]
                + inference["calibrated_question_count"],
            )
            self.assertEqual(inference["calibrated_question_count"], 0)
            self.assertIn(
                "not a proven universal error envelope",
                inference["numerical_guard_scope"],
            )
            self.assertIn(
                "legacy gaussian_moments",
                inference["numerical_guard_scope"],
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
            self.assertEqual(
                profile["selected_response_inference"], inference
            )
            self.assertEqual(
                profile["corpus_release_id"], report["corpus_release_id"]
            )
            self.assertIn(
                "does not cover item calibration",
                profile["projection_definitions"][
                    "mastery_probability_error_bound"
                ],
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
            self.assertEqual(
                objective_profile["state_qualification"],
                "provisional_selected_response_state",
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
            self.assertEqual(
                concept_profile["state_qualification"],
                "provisional_selected_response_state",
            )
            floor_objective_profile = next(
                row
                for row in profile["learning_objectives"]
                if row["objective_id"]
                == concept_profile["objective_floor_source_id"]
            )
            self.assertEqual(
                concept_profile["independent_families"],
                floor_objective_profile["independent_families"],
            )
            self.assertEqual(
                concept_profile["observed_response_families"],
                floor_objective_profile["observed_response_families"],
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

    def test_derived_concept_reports_its_exact_floor_objective_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "report.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(database)
            learner_id = "derived-floor-evidence"
            engine.create_learner(learner_id)
            session = engine.start_session(
                learner_id,
                "c_causal_masking",
                seed=1,
                now=START,
            )
            presentation = engine.next_question(session["id"], now=START)
            self.assertEqual(
                presentation.question.objective_id,
                "lo_causal_visibility",
            )
            engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.95,
                response_ms=4_000,
                hint_count=0,
                feedback_shown=False,
                now=START + timedelta(seconds=4),
            )
            profile = engine.profile(
                learner_id,
                root_concept_id="c_causal_masking",
                now=START + timedelta(seconds=4),
            )

        concept = next(
            row
            for row in profile["skills"]
            if row["concept_id"] == "c_causal_masking"
        )
        objective = next(
            row
            for row in profile["learning_objectives"]
            if row["objective_id"] == "lo_causal_visibility"
        )
        self.assertEqual(
            concept["objective_floor_source_id"],
            objective["objective_id"],
        )
        self.assertEqual(concept["independent_families"], 1)
        self.assertEqual(concept["observed_response_families"], 1)
        self.assertEqual(
            concept["independent_families"],
            objective["independent_families"],
        )
        self.assertEqual(
            concept["observed_response_families"],
            objective["observed_response_families"],
        )
        self.assertEqual(concept["state"], objective["state"])

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
