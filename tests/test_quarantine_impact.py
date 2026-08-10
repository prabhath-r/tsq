# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

from tsq.capacity import (
    DEFAULT_STATE_LIMIT,
    CapacityTarget,
    analyze_sustained_capacity,
    concept_target,
)
from tsq.evidence import canonical_digest
from tsq.graph import KnowledgeGraph
from tsq.models import (
    Concept,
    ConceptEdge,
    ConceptWeight,
    LearningObjective,
    Misconception,
    ObjectiveOperation,
    Option,
    Question,
    QuestionKind,
    QuestionStatus,
    RelationType,
)
from tsq.quarantine_impact import (
    MAX_CANDIDATE_LIMIT,
    MAX_COMBINATION_SIZE,
    MAX_EVALUATION_LIMIT,
    MAX_RESULT_LIMIT,
    REPORT_SCHEMA,
    analyze_quarantine_capacity_impact,
)


def make_question(
    question_id: str,
    family_id: str,
    concept_id: str,
    status: QuestionStatus,
) -> Question:
    return Question(
        id=question_id,
        version=1,
        family_id=family_id,
        status=status,
        stem=(
            "Which carefully specified response follows from the stated "
            "conditions?"
        ),
        kind=QuestionKind.APPLICATION,
        difficulty=0.0,
        discrimination=1.0,
        guess_rate=0.25,
        slip_rate=0.05,
        concepts=(ConceptWeight(concept_id, 1.0),),
        options=(
            Option("a", "The supported conclusion.", True, "Correct."),
            Option("b", "A reversed conclusion.", False, "It reverses the facts."),
            Option("c", "An omitted condition.", False, "It drops a condition."),
            Option("d", "An added premise.", False, "It adds an unsupported premise."),
        ),
        source_ids=("src_fixture",),
        provenance={
            "generated": True,
            "batch_id": "batch_quarantine_impact_test",
            "review_status": "synthetic_fixture_only",
            "human_review": False,
            "human_review_status": "required",
            "activation": "forbidden",
        },
    )


def support_questions(
    concept_id: str,
    count: int,
) -> tuple[Question, ...]:
    return tuple(
        make_question(
            f"q_{concept_id}_support_{index}",
            f"f_{concept_id}_support_{index}",
            concept_id,
            QuestionStatus.APPROVED,
        )
        for index in range(1, count + 1)
    )


def graph_for(*concept_ids: str) -> KnowledgeGraph:
    return KnowledgeGraph(
        tuple(
            Concept(concept_id, concept_id.upper(), "Fixture concept.")
            for concept_id in concept_ids
        ),
        (),
    )


class QuarantineCapacityImpactTestCase(unittest.TestCase):
    def test_source_question_is_frozen_and_status_is_not_mutated(self) -> None:
        candidate = make_question(
            "q_candidate",
            "f_candidate",
            "c",
            QuestionStatus.QUARANTINED,
        )
        questions = (*support_questions("c", 2), candidate)
        before = tuple(
            (question.id, question.status, dict(question.provenance))
            for question in questions
        )

        with self.assertRaises(FrozenInstanceError):
            setattr(candidate, "status", QuestionStatus.APPROVED)

        report = analyze_quarantine_capacity_impact(
            questions,
            graph_for("c"),
            (),
            concept_target("c", target_main_count=1),
        )

        self.assertEqual(
            tuple(
                (question.id, question.status, dict(question.provenance))
                for question in questions
            ),
            before,
        )
        self.assertIs(candidate.status, QuestionStatus.QUARANTINED)
        self.assertEqual(
            report["candidates"][0]["original_question_status"],
            QuestionStatus.QUARANTINED.value,
        )
        self.assertFalse(
            report["candidates"][0][
                "original_question_eligible_for_adaptation"
            ]
        )
        self.assertEqual(
            report["candidates"][0]["counterfactual_status"],
            QuestionStatus.APPROVED.value,
        )
        self.assertFalse(
            report["boundary"]["source_corpus_mutated_by_analyzer"]
        )
        self.assertFalse(
            report["boundary"]["activation_performed_by_analyzer"]
        )

    def test_exact_one_candidate_closes_configured_target(self) -> None:
        candidate = make_question(
            "q_candidate",
            "f_candidate",
            "c",
            QuestionStatus.QUARANTINED,
        )

        report = analyze_quarantine_capacity_impact(
            (*support_questions("c", 2), candidate),
            graph_for("c"),
            (),
            concept_target("c", target_main_count=1),
            maximum_combination_size=3,
        )

        self.assertEqual(report["schema"], REPORT_SCHEMA)
        self.assertEqual(report["baseline"]["status"], "blocked")
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["evaluated_combination_count"], 1)
        self.assertEqual(report["largest_evaluated_combination_size"], 1)
        self.assertEqual(report["minimal_closing_combination_size"], 1)
        self.assertEqual(report["closing_combination_count_at_minimum"], 1)
        self.assertEqual(
            report["closing_combinations"][0]["question_ids"],
            ["q_candidate"],
        )
        impact = report["closing_combinations"][0]["impact"]
        self.assertTrue(impact["closes_configured_target"])
        self.assertEqual(impact["capacity"]["status"], "adequate")
        self.assertEqual(impact["delta"]["order_robust_main_capacity"], 1)
        self.assertEqual(
            impact["resolved_thin_owned_concept_ids"],
            ["c"],
        )

    def test_two_concepts_require_exact_two_candidate_minimal_pair(self) -> None:
        candidates = (
            make_question(
                "q_c1_candidate",
                "f_c1_candidate",
                "c1",
                QuestionStatus.QUARANTINED,
            ),
            make_question(
                "q_c2_candidate",
                "f_c2_candidate",
                "c2",
                QuestionStatus.QUARANTINED,
            ),
        )
        questions = (
            *support_questions("c1", 4),
            *support_questions("c2", 4),
            *candidates,
        )
        target = CapacityTarget("both", "topic", ("c1", "c2"))

        report = analyze_quarantine_capacity_impact(
            questions,
            graph_for("c1", "c2"),
            (),
            target,
            maximum_combination_size=2,
        )

        self.assertEqual(report["baseline"]["status"], "thin")
        self.assertEqual(
            report["baseline"]["thin_owned_concept_ids"],
            ["c1", "c2"],
        )
        self.assertEqual(report["evaluated_combination_count"], 3)
        self.assertEqual(report["minimal_closing_combination_size"], 2)
        self.assertEqual(report["closing_combination_count_at_minimum"], 1)
        closing = report["closing_combinations"][0]
        self.assertEqual(
            closing["question_ids"],
            ["q_c1_candidate", "q_c2_candidate"],
        )
        self.assertEqual(
            closing["impact"]["resolved_thin_owned_concept_ids"],
            ["c1", "c2"],
        )
        self.assertEqual(
            closing["impact"]["capacity"]["owned_concept_order_robust_floor"],
            3,
        )
        singleton_rows = [
            row
            for row in report["best_available_combinations"]
            if len(row["question_ids"]) == 1
        ]
        self.assertEqual(len(singleton_rows), 2)
        self.assertTrue(
            all(
                not row["impact"]["closes_configured_target"]
                for row in singleton_rows
            )
        )

    def test_candidate_limit_refuses_to_truncate(self) -> None:
        candidates = (
            make_question(
                "q_candidate_a",
                "f_candidate_a",
                "c",
                QuestionStatus.QUARANTINED,
            ),
            make_question(
                "q_candidate_b",
                "f_candidate_b",
                "c",
                QuestionStatus.QUARANTINED,
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "refuses to truncate 2 candidates at limit 1",
        ):
            analyze_quarantine_capacity_impact(
                (*support_questions("c", 2), *candidates),
                graph_for("c"),
                (),
                concept_target("c", target_main_count=2),
                candidate_limit=1,
            )

        self.assertTrue(
            all(
                question.status is QuestionStatus.QUARANTINED
                for question in candidates
            )
        )

    def test_explicit_candidate_filters_are_exact_and_auditable(self) -> None:
        first = make_question(
            "q_candidate_a",
            "f_candidate_a",
            "c",
            QuestionStatus.QUARANTINED,
        )
        second = replace(
            make_question(
                "q_candidate_b",
                "f_candidate_b",
                "c",
                QuestionStatus.QUARANTINED,
            ),
            provenance={
                **first.provenance,
                "batch_id": "batch_second",
            },
        )
        third = replace(
            make_question(
                "q_candidate_c",
                "f_candidate_c",
                "c",
                QuestionStatus.QUARANTINED,
            ),
            provenance={
                **first.provenance,
                "batch_id": "batch_second",
            },
        )
        questions = (*support_questions("c", 2), first, second, third)
        target = concept_target("c", target_main_count=2)

        by_question = analyze_quarantine_capacity_impact(
            questions,
            graph_for("c"),
            (),
            target,
            candidate_question_ids=(third.id, first.id),
        )
        by_batch = analyze_quarantine_capacity_impact(
            questions,
            graph_for("c"),
            (),
            target,
            candidate_batch_ids=("batch_second",),
        )

        self.assertEqual(
            by_question["candidate_scope"],
            "explicit-filter-within-target-owned-primary-concepts",
        )
        self.assertEqual(
            by_question["eligible_candidate_count_before_filter"], 3
        )
        self.assertEqual(
            by_question["candidate_filter"],
            {
                "mode": "question_ids",
                "values": [first.id, third.id],
            },
        )
        self.assertEqual(
            [row["question_id"] for row in by_question["candidates"]],
            [first.id, third.id],
        )
        self.assertEqual(
            by_batch["candidate_filter"],
            {"mode": "batch_ids", "values": ["batch_second"]},
        )
        self.assertEqual(
            [row["question_id"] for row in by_batch["candidates"]],
            [second.id, third.id],
        )
        self.assertEqual(by_question["baseline"], by_batch["baseline"])
        self.assertNotEqual(
            by_question["analysis_input_manifest_digest"],
            by_batch["analysis_input_manifest_digest"],
        )

    def test_candidate_filters_reject_ambiguous_or_ineligible_scope(
        self,
    ) -> None:
        candidate = make_question(
            "q_candidate",
            "f_candidate",
            "c",
            QuestionStatus.QUARANTINED,
        )
        questions = (*support_questions("c", 2), candidate)
        graph = graph_for("c")
        target = concept_target("c", target_main_count=1)

        invalid_cases = (
            (
                {"candidate_question_ids": candidate.id},
                "not a scalar string",
            ),
            (
                {"candidate_question_ids": 42},
                "must be an iterable",
            ),
            (
                {"candidate_question_ids": ()},
                "must not be empty",
            ),
            (
                {"candidate_question_ids": ("q_missing",)},
                "not quarantined target-owned candidates",
            ),
            (
                {"candidate_batch_ids": ("batch_missing",)},
                "no quarantined target-owned candidates",
            ),
            (
                {"candidate_question_ids": (candidate.id, candidate.id)},
                "contains duplicate values",
            ),
            (
                {
                    "candidate_question_ids": (candidate.id,),
                    "candidate_batch_ids": (
                        "batch_quarantine_impact_test",
                    ),
                },
                "mutually exclusive",
            ),
        )
        for kwargs, message in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    analyze_quarantine_capacity_impact(
                        questions,
                        graph,
                        (),
                        target,
                        **kwargs,
                    )

    def test_repeated_family_subset_is_skipped(self) -> None:
        candidates = (
            make_question(
                "q_same_family_a",
                "f_shared_candidate",
                "c",
                QuestionStatus.QUARANTINED,
            ),
            make_question(
                "q_same_family_b",
                "f_shared_candidate",
                "c",
                QuestionStatus.QUARANTINED,
            ),
        )

        report = analyze_quarantine_capacity_impact(
            (*support_questions("c", 2), *candidates),
            graph_for("c"),
            (),
            concept_target("c", target_main_count=2),
            maximum_combination_size=2,
        )

        self.assertIsNone(report["minimal_closing_combination_size"])
        self.assertEqual(report["largest_enumerated_combination_size"], 2)
        self.assertEqual(report["largest_evaluated_combination_size"], 1)
        self.assertEqual(report["evaluated_combination_count"], 2)
        self.assertEqual(report["repeated_family_subsets_skipped"], 1)
        self.assertEqual(report["closing_combinations"], [])
        self.assertTrue(
            all(
                len(set(row["family_ids"])) == len(row["family_ids"])
                for row in report["best_available_combinations"]
            )
        )

    def test_public_bounds_fail_closed_and_upper_bounds_are_inclusive(self) -> None:
        candidate = make_question(
            "q_candidate",
            "f_candidate",
            "c",
            QuestionStatus.QUARANTINED,
        )
        questions = (*support_questions("c", 2), candidate)
        graph = graph_for("c")
        target = concept_target("c", target_main_count=1)
        invalid_cases = (
            (
                {"maximum_combination_size": 0},
                "maximum_combination_size must be between",
            ),
            (
                {"maximum_combination_size": MAX_COMBINATION_SIZE + 1},
                "maximum_combination_size must be between",
            ),
            (
                {"maximum_combination_size": True},
                "maximum_combination_size must be between",
            ),
            (
                {"candidate_limit": 0},
                "candidate_limit must be between",
            ),
            (
                {"candidate_limit": MAX_CANDIDATE_LIMIT + 1},
                "candidate_limit must be between",
            ),
            (
                {"candidate_limit": True},
                "candidate_limit must be between",
            ),
            (
                {"result_limit": 0},
                "result_limit must be between",
            ),
            (
                {"result_limit": MAX_RESULT_LIMIT + 1},
                "result_limit must be between",
            ),
            (
                {"result_limit": True},
                "result_limit must be between",
            ),
            (
                {"state_limit": 0},
                "state_limit must be between",
            ),
            (
                {"state_limit": True},
                "state_limit must be between",
            ),
            (
                {"state_limit": DEFAULT_STATE_LIMIT + 1},
                "state_limit must be between",
            ),
            (
                {"evaluation_limit": 0},
                "evaluation_limit must be between",
            ),
            (
                {"evaluation_limit": MAX_EVALUATION_LIMIT + 1},
                "evaluation_limit must be between",
            ),
            (
                {"evaluation_limit": True},
                "evaluation_limit must be between",
            ),
        )
        for kwargs, message in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    analyze_quarantine_capacity_impact(
                        questions,
                        graph,
                        (),
                        target,
                        **kwargs,
                    )

        report = analyze_quarantine_capacity_impact(
            questions,
            graph,
            (),
            target,
            maximum_combination_size=MAX_COMBINATION_SIZE,
            candidate_limit=MAX_CANDIDATE_LIMIT,
            result_limit=MAX_RESULT_LIMIT,
        )
        self.assertEqual(
            report["maximum_combination_size"],
            MAX_COMBINATION_SIZE,
        )
        self.assertEqual(report["candidate_limit"], MAX_CANDIDATE_LIMIT)

    def test_input_commitment_changes_with_every_material_input_class(
        self,
    ) -> None:
        objective = LearningObjective(
            "lo_a",
            "Objective A",
            "A bounded fixture objective.",
            "c1",
            (),
            ObjectiveOperation.APPLY,
        )
        alternate_objective = LearningObjective(
            "lo_b",
            "Objective B",
            "A distinct bounded fixture objective.",
            "c1",
            (),
            ObjectiveOperation.DIAGNOSE,
        )
        candidate = replace(
            make_question(
                "q_candidate",
                "f_candidate",
                "c1",
                QuestionStatus.QUARANTINED,
            ),
            objective=objective,
        )
        active = support_questions("c1", 2)
        target = concept_target("c1", target_main_count=1)
        concepts = (
            Concept("c0", "C0", "Fixture prerequisite."),
            Concept("c1", "C1", "Fixture target."),
        )
        graph = KnowledgeGraph(
            concepts,
            (
                ConceptEdge(
                    "c0",
                    "c1",
                    RelationType.PREREQUISITE,
                    1.0,
                ),
            ),
        )
        misconceptions = (
            Misconception("m", "c1", "M", "Fixture misconception."),
        )

        def run(
            questions: tuple[Question, ...] = (*active, candidate),
            selected_graph: KnowledgeGraph = graph,
            selected_misconceptions: tuple[Misconception, ...] = (
                misconceptions
            ),
        ) -> dict[str, object]:
            return analyze_quarantine_capacity_impact(
                questions,
                selected_graph,
                selected_misconceptions,
                target,
            )

        baseline = run()
        changed_inputs = (
            run(
                (
                    *active,
                    replace(candidate, stem=candidate.stem + " Changed."),
                )
            ),
            run(
                (
                    *active,
                    replace(candidate, objective=alternate_objective),
                )
            ),
            run(
                (
                    *active,
                    replace(
                        candidate,
                        options=(
                            candidate.options[0],
                            replace(
                                candidate.options[1],
                                diagnostic_objective_id=objective.id,
                            ),
                            *candidate.options[2:],
                        ),
                    ),
                )
            ),
            run(
                selected_misconceptions=(
                    Misconception(
                        "m",
                        "c0",
                        "M",
                        "Fixture misconception.",
                    ),
                )
            ),
            run(
                selected_graph=KnowledgeGraph(
                    concepts,
                    (
                        ConceptEdge(
                            "c0",
                            "c1",
                            RelationType.REQUIRES,
                            1.0,
                        ),
                    ),
                )
            ),
            run(
                (
                    replace(
                        active[0],
                        status=QuestionStatus.CALIBRATED,
                    ),
                    active[1],
                    candidate,
                )
            ),
        )

        for changed in changed_inputs:
            with self.subTest(
                digest=changed["analysis_input_manifest_digest"]
            ):
                self.assertNotEqual(
                    changed["analysis_input_manifest_digest"],
                    baseline["analysis_input_manifest_digest"],
                )
                self.assertNotEqual(
                    changed["report_digest"],
                    baseline["report_digest"],
                )
        for changed in changed_inputs[:3]:
            self.assertNotEqual(
                changed["candidate_set_digest"],
                baseline["candidate_set_digest"],
            )

    def test_declared_target_owned_scope_excludes_off_owned_objective_reserve(
        self,
    ) -> None:
        objective = LearningObjective(
            "lo_shared",
            "Shared objective",
            "The target can use a supporting-concept reserve.",
            "c1",
            ("c2",),
            ObjectiveOperation.APPLY,
        )
        owned = tuple(
            replace(question, objective=objective)
            for question in support_questions("c1", 2)
        )
        off_owned = replace(
            make_question(
                "q_off_owned",
                "f_off_owned",
                "c2",
                QuestionStatus.QUARANTINED,
            ),
            objective=objective,
        )
        questions = (*owned, off_owned)
        graph = graph_for("c1", "c2")
        target = concept_target("c1", target_main_count=1)

        report = analyze_quarantine_capacity_impact(
            questions,
            graph,
            (),
            target,
        )
        promoted = analyze_sustained_capacity(
            (
                *owned,
                replace(off_owned, status=QuestionStatus.APPROVED),
            ),
            graph,
            (),
            (target,),
        ).targets[0]

        self.assertEqual(report["candidate_count"], 0)
        self.assertEqual(
            report["candidate_scope"],
            "target-owned-primary-concepts-only",
        )
        self.assertFalse(
            report["boundary"][
                "release_wide_objective_reserve_candidates_ranked"
            ]
        )
        self.assertEqual(report["baseline"]["status"], "blocked")
        self.assertEqual(promoted.status, "adequate")

    def test_baseline_and_empty_candidate_search_outcomes_are_explicit(
        self,
    ) -> None:
        adequate = analyze_quarantine_capacity_impact(
            support_questions("c", 3),
            graph_for("c"),
            (),
            concept_target("c", target_main_count=1),
        )
        thin = analyze_quarantine_capacity_impact(
            support_questions("c", 2),
            graph_for("c"),
            (),
            concept_target("c", target_main_count=1),
        )

        self.assertEqual(
            adequate["search_outcome"], "live_target_already_met"
        )
        self.assertEqual(adequate["minimal_closing_combination_size"], 0)
        self.assertEqual(adequate["closing_combinations"][0]["question_ids"], [])
        self.assertEqual(adequate["evaluated_combination_count"], 0)
        self.assertEqual(adequate["largest_evaluated_combination_size"], 0)
        self.assertEqual(
            thin["search_outcome"],
            "no_target_owned_quarantined_candidates",
        )
        self.assertTrue(
            thin[
                "closure_absence_is_exact_within_declared_candidate_space"
            ]
        )

    def test_evaluation_budget_refuses_before_subset_analysis(self) -> None:
        candidates = tuple(
            make_question(
                f"q_candidate_{index}",
                f"f_candidate_{index}",
                "c",
                QuestionStatus.QUARANTINED,
            )
            for index in range(4)
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires 10 admissible candidate-subset evaluations.*"
            "No heuristic or partial result was used",
        ):
            analyze_quarantine_capacity_impact(
                (*support_questions("c", 2), *candidates),
                graph_for("c"),
                (),
                concept_target("c", target_main_count=4),
                maximum_combination_size=2,
                evaluation_limit=9,
            )

    def test_ambiguous_global_identities_fail_before_counterfactuals(
        self,
    ) -> None:
        candidate = make_question(
            "q_duplicate",
            "f_visible",
            "c1",
            QuestionStatus.QUARANTINED,
        )
        off_owned_duplicate = make_question(
            "q_duplicate",
            "f_hidden",
            "c2",
            QuestionStatus.QUARANTINED,
        )
        questions = (
            *support_questions("c1", 2),
            candidate,
            off_owned_duplicate,
        )

        with self.assertRaisesRegex(
            ValueError, "globally unique question IDs"
        ):
            analyze_quarantine_capacity_impact(
                questions,
                graph_for("c1", "c2"),
                (),
                concept_target("c1", target_main_count=1),
            )
        with self.assertRaisesRegex(
            ValueError, "globally unique misconception IDs"
        ):
            analyze_quarantine_capacity_impact(
                support_questions("c1", 2),
                graph_for("c1", "c2"),
                (
                    Misconception("m", "c1", "M1", "First."),
                    Misconception("m", "c2", "M2", "Second."),
                ),
                concept_target("c1", target_main_count=1),
            )

        objective_a = LearningObjective(
            "lo_same",
            "First definition",
            "First.",
            "c1",
            (),
            ObjectiveOperation.APPLY,
        )
        objective_b = LearningObjective(
            "lo_same",
            "Second definition",
            "Second.",
            "c1",
            (),
            ObjectiveOperation.DIAGNOSE,
        )
        conflicting = (
            replace(
                make_question(
                    "q_objective_a",
                    "f_objective_a",
                    "c1",
                    QuestionStatus.APPROVED,
                ),
                objective=objective_a,
            ),
            replace(
                make_question(
                    "q_objective_b",
                    "f_objective_b",
                    "c1",
                    QuestionStatus.APPROVED,
                ),
                objective=objective_b,
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "conflicting definitions.*lo_same"
        ):
            analyze_quarantine_capacity_impact(
                conflicting,
                graph_for("c1"),
                (),
                concept_target("c1", target_main_count=1),
            )

    def test_report_and_candidate_digests_are_deterministic(self) -> None:
        candidates = (
            make_question(
                "q_c1_candidate",
                "f_c1_candidate",
                "c1",
                QuestionStatus.QUARANTINED,
            ),
            make_question(
                "q_c2_candidate",
                "f_c2_candidate",
                "c2",
                QuestionStatus.QUARANTINED,
            ),
        )
        questions = (
            *support_questions("c1", 4),
            *support_questions("c2", 4),
            *candidates,
        )
        target = CapacityTarget("both", "topic", ("c1", "c2"))

        forward = analyze_quarantine_capacity_impact(
            questions,
            graph_for("c1", "c2"),
            (),
            target,
            maximum_combination_size=2,
        )
        reverse = analyze_quarantine_capacity_impact(
            reversed(questions),
            graph_for("c2", "c1"),
            (),
            target,
            maximum_combination_size=2,
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(
            forward["candidate_set_digest"],
            reverse["candidate_set_digest"],
        )
        self.assertEqual(forward["report_digest"], reverse["report_digest"])
        unsigned_report = dict(forward)
        digest = unsigned_report.pop("report_digest")
        self.assertEqual(digest, canonical_digest(unsigned_report))
        self.assertRegex(digest, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
