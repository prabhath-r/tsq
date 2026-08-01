# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsq.corpus import load_bundle, parse_bundle, parse_catalog, read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
START = datetime(2100, 4, 1, 9, 0, tzinfo=timezone.utc)


def evidence_anchor(question) -> str:
    return (
        question.objective.primary_concept_id
        if question.objective is not None
        else question.primary_concept_id
    )


def focused_wrong_option(question, focus_objective_id: str | None):
    wrong = [option for option in question.options if not option.correct]
    if focus_objective_id is not None:
        matching = [
            option
            for option in wrong
            if (option.diagnostic_objective_id or question.objective_id)
            == focus_objective_id
        ]
        if matching:
            return matching[0]
    return wrong[0]


class TargetFirstRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "routing.db")
        self.database.initialize()
        self.database.import_corpus(
            *read_and_parse(CORPUS, include_catalog=True)
        )
        self.engine = AdaptiveEngine(self.database)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_concept_root_probes_requested_target_before_prerequisites(self) -> None:
        for seed in range(5):
            with self.subTest(seed=seed):
                learner_id = f"target-first-{seed}"
                self.engine.create_learner(learner_id)
                session = self.engine.start_session(
                    learner_id,
                    "c_causal_masking",
                    seed=seed,
                )
                presentation = self.engine.next_question(
                    session["id"], now=START
                )
                self.assertEqual(
                    evidence_anchor(presentation.question),
                    "c_causal_masking",
                )

    def test_topic_main_path_cannot_starve_an_unassessed_objective(self) -> None:
        self.engine.create_learner("objective-fairness")
        session = self.engine.start_session(
            "objective-fairness",
            "t_large_language_models",
            seed=9,
            explore_related=False,
        )
        owned = self.database.topic_owned_concepts(
            "t_large_language_models",
            session["corpus_release_id"],
            include_descendants=True,
        )
        expected = {
            objective.id
            for objective in self.database.get_learning_objectives(
                session["corpus_release_id"]
            )
            if objective.primary_concept_id in owned
        }
        self.assertTrue(expected)

        seen: set[str] = set()
        current = START
        for index in range(30):
            presentation = self.engine.next_question(
                session["id"], now=current
            )
            self.assertIn("fair_coverage_target_exposures=", presentation.rationale)
            if presentation.question.objective_id is not None:
                seen.add(presentation.question.objective_id)
            current += timedelta(seconds=4)
            self.engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.95,
                response_ms=4_000,
                hint_count=0,
                feedback_shown=True,
                idempotency_key=f"objective-fairness-{index}",
                now=current,
            )
            if seen == expected:
                break
            current += timedelta(minutes=5)

        self.assertEqual(seen, expected)
        self.assertLessEqual(index + 1, 30)

    def test_credible_target_failure_can_unlock_a_direct_prerequisite(self) -> None:
        self.engine.create_learner("target-descent")
        session = self.engine.start_session(
            "target-descent",
            "c_causal_masking",
            seed=0,
        )
        first = self.engine.next_question(session["id"], now=START)
        self.assertEqual(evidence_anchor(first.question), "c_causal_masking")
        first_wrong = focused_wrong_option(
            first.question, first.question.objective_id
        )
        first_result = self.engine.submit_answer(
            first.decision_id,
            first_wrong.id,
            confidence=0.95,
            response_ms=4_000,
            hint_count=0,
            feedback_shown=True,
            now=START + timedelta(seconds=4),
        )
        self.assertEqual(first_result.next_phase.value, "remediate")
        self.assertEqual(first_result.focus_concept_id, "c_causal_masking")

        second_at = START + timedelta(minutes=5)
        second = self.engine.next_question(session["id"], now=second_at)
        self.assertEqual(evidence_anchor(second.question), "c_causal_masking")
        second_wrong = focused_wrong_option(
            second.question, first_result.focus_objective_id
        )
        second_result = self.engine.submit_answer(
            second.decision_id,
            second_wrong.id,
            confidence=0.95,
            response_ms=4_000,
            hint_count=0,
            feedback_shown=True,
            now=second_at + timedelta(seconds=4),
        )

        objectives = {
            objective.id: objective
            for objective in self.database.get_learning_objectives(
                session["corpus_release_id"]
            )
        }
        parent = objectives[first_result.focus_objective_id]
        direct_prerequisites = {
            edge.source_id for edge in parent.prerequisites
        }
        self.assertEqual(
            second_result.transition_reason,
            "descend_to_evidence_boundary",
        )
        self.assertIn(second_result.focus_objective_id, direct_prerequisites)
        selected = objectives[second_result.focus_objective_id]
        self.assertEqual(
            second_result.focus_concept_id, selected.primary_concept_id
        )

    def test_legacy_objective_release_retains_broad_concept_fallback(self) -> None:
        raw = load_bundle(CORPUS)
        raw["schema_version"] = 2
        raw.pop("objective_edges")
        parsed = parse_bundle(raw)
        catalog = parse_catalog(raw, parsed[0], parsed[4])
        legacy = Database(Path(self.tempdir.name) / "legacy-routing.db")
        legacy.initialize()
        release_id = legacy.import_corpus(*parsed, *catalog)["release_id"]
        self.assertIsNone(legacy.get_objective_graph(release_id)[0])
        engine = AdaptiveEngine(legacy)
        engine.create_learner("legacy-target-descent")
        session = engine.start_session(
            "legacy-target-descent", "c_causal_masking", seed=0
        )
        first = engine.next_question(session["id"], now=START)
        first_result = engine.submit_answer(
            first.decision_id,
            focused_wrong_option(first.question, first.question.objective_id).id,
            confidence=0.95,
            response_ms=4_000,
            hint_count=0,
            feedback_shown=True,
            now=START + timedelta(seconds=4),
        )
        second_at = START + timedelta(minutes=5)
        second = engine.next_question(session["id"], now=second_at)
        second_result = engine.submit_answer(
            second.decision_id,
            focused_wrong_option(
                second.question, first_result.focus_objective_id
            ).id,
            confidence=0.95,
            response_ms=4_000,
            hint_count=0,
            feedback_shown=True,
            now=second_at + timedelta(seconds=4),
        )

        direct_concepts = {
            concept_id
            for concept_id, _ in legacy.get_graph(
                release_id
            ).direct_prerequisites("c_causal_masking")
        }
        self.assertEqual(
            second_result.transition_reason,
            "descend_to_evidence_boundary",
        )
        self.assertIn(second_result.focus_concept_id, direct_concepts)
        self.assertIn("focus_concept_id", second_result.boundary_decision)
        self.assertNotIn("focus_objective_id", second_result.boundary_decision)

    def test_declared_empty_objective_graph_disables_broad_fallback(self) -> None:
        raw = load_bundle(CORPUS)
        raw["objective_edges"] = []
        parsed = parse_bundle(raw)
        catalog = parse_catalog(raw, parsed[0], parsed[4])
        empty = Database(Path(self.tempdir.name) / "empty-routing.db")
        empty.initialize()
        release_id = empty.import_corpus(*parsed, *catalog)["release_id"]
        self.assertEqual(empty.get_objective_graph(release_id), (1, ()))
        engine = AdaptiveEngine(empty)
        engine.create_learner("empty-target-descent")
        session = engine.start_session(
            "empty-target-descent", "c_causal_masking", seed=0
        )
        first = engine.next_question(session["id"], now=START)
        first_result = engine.submit_answer(
            first.decision_id,
            focused_wrong_option(first.question, first.question.objective_id).id,
            confidence=0.95,
            response_ms=4_000,
            hint_count=0,
            feedback_shown=True,
            now=START + timedelta(seconds=4),
        )
        second_at = START + timedelta(minutes=5)
        second = engine.next_question(session["id"], now=second_at)
        second_result = engine.submit_answer(
            second.decision_id,
            focused_wrong_option(
                second.question, first_result.focus_objective_id
            ).id,
            confidence=0.95,
            response_ms=4_000,
            hint_count=0,
            feedback_shown=True,
            now=second_at + timedelta(seconds=4),
        )

        self.assertEqual(second_result.transition_reason, "bounded_failure_exit")
        self.assertIsNone(second_result.focus_concept_id)
        self.assertIsNone(second_result.focus_objective_id)
        self.assertIsNone(second_result.boundary_decision)

    def test_declared_prerequisite_must_have_fresh_independent_capacity(self) -> None:
        self.engine.create_learner("unserviceable-objective-boundary")
        session = self.engine.start_session(
            "unserviceable-objective-boundary", "c_causal_masking", seed=0
        )
        first = self.engine.next_question(session["id"], now=START)
        first_result = self.engine.submit_answer(
            first.decision_id,
            focused_wrong_option(first.question, first.question.objective_id).id,
            confidence=0.95,
            response_ms=4_000,
            hint_count=0,
            feedback_shown=True,
            now=START + timedelta(seconds=4),
        )
        second_at = START + timedelta(minutes=5)
        second = self.engine.next_question(session["id"], now=second_at)
        with patch.object(
            self.engine,
            "_fresh_objective_focus_capacity",
            return_value={},
        ):
            second_result = self.engine.submit_answer(
                second.decision_id,
                focused_wrong_option(
                    second.question, first_result.focus_objective_id
                ).id,
                confidence=0.95,
                response_ms=4_000,
                hint_count=0,
                feedback_shown=True,
                now=second_at + timedelta(seconds=4),
            )

        self.assertEqual(
            second_result.transition_reason,
            "no_serviceable_prerequisite_boundary",
        )
        self.assertIsNone(second_result.focus_concept_id)
        self.assertIsNone(second_result.focus_objective_id)


if __name__ == "__main__":
    unittest.main()
