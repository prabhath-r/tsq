# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import tsq.capacity as capacity_module
from tsq.corpus import corpus_source_digest, load_bundle
from tsq.capacity import (
    CapacityAnalysisLimitError,
    CapacityTarget,
    analyze_sustained_capacity,
    concept_target,
    topic_target,
)
from tsq.cli import build_parser, main
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
    Topic,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


def make_question(
    family_id: str,
    primary_concept_id: str,
    misconception_ids: tuple[str, ...],
    *,
    question_id: str | None = None,
    kind: QuestionKind = QuestionKind.TRANSFER,
    status: QuestionStatus = QuestionStatus.APPROVED,
) -> Question:
    if not misconception_ids:
        raise ValueError("Synthetic questions require a distractor misconception.")
    expanded = tuple(
        misconception_ids[index % len(misconception_ids)] for index in range(3)
    )
    return Question(
        id=question_id or f"q_{family_id}",
        version=1,
        family_id=family_id,
        status=status,
        stem="A sufficiently precise synthetic capacity question for deterministic analysis.",
        kind=kind,
        difficulty=0.0,
        discrimination=1.0,
        guess_rate=0.25,
        slip_rate=0.05,
        concepts=(ConceptWeight(primary_concept_id, 1.0),),
        options=(
            Option("correct", "The supported conclusion.", True, "This is correct."),
            *(
                Option(
                    f"wrong_{index}",
                    f"Named misconception response {index}.",
                    False,
                    "This response instantiates the named misconception.",
                    misconception_id,
                )
                for index, misconception_id in enumerate(expanded, start=1)
            ),
        ),
        source_ids=("source",),
    )


def make_objective_question(
    family_id: str,
    primary_concept_id: str,
    misconception_id: str,
    objective: LearningObjective,
    *,
    diagnostic_objective_id: str | None = None,
) -> Question:
    return Question(
        id=f"q_{family_id}",
        version=1,
        family_id=family_id,
        status=QuestionStatus.APPROVED,
        stem="A precise synthetic objective question for deterministic capacity analysis.",
        kind=QuestionKind.TRANSFER,
        difficulty=0.0,
        discrimination=1.0,
        guess_rate=0.25,
        slip_rate=0.05,
        concepts=(ConceptWeight(primary_concept_id, 1.0),),
        options=(
            Option("correct", "The supported conclusion.", True, "Correct."),
            *(
                Option(
                    f"wrong_{index}",
                    f"Named misconception response {index}.",
                    False,
                    "This response instantiates the named misconception.",
                    misconception_id,
                    diagnostic_objective_id or objective.id,
                )
                for index in range(1, 4)
            ),
        ),
        source_ids=("source",),
        objective=objective,
    )


def interchangeable_bank(
    concept_id: str, prefix: str, count: int
) -> tuple[list[Question], list[Misconception]]:
    misconceptions = [
        Misconception(f"{prefix}_m{index}", concept_id, f"M{index}", "Description")
        for index in range(1, 4)
    ]
    questions = [
        make_question(
            f"{prefix}_f{index}",
            concept_id,
            tuple(item.id for item in misconceptions),
        )
        for index in range(1, count + 1)
    ]
    return questions, misconceptions


class SustainedCapacityTestCase(unittest.TestCase):
    def test_overlapping_quota_packing_matches_binary_brute_force(self) -> None:
        triangle = (
            (0b011, 1),
            (0b110, 1),
            (0b101, 1),
        )
        self.assertEqual(
            capacity_module._maximum_quota_packing(0b111, triangle),
            1,
        )

        rng = random.Random(47_117)
        for case in range(120):
            variable_count = rng.randint(1, 8)
            safe_mask = (1 << variable_count) - 1
            constraints: list[tuple[int, int]] = []
            for _ in range(rng.randint(0, 10)):
                block = sum(
                    1 << index
                    for index in range(variable_count)
                    if rng.random() < 0.55
                )
                if not block:
                    block = 1 << rng.randrange(variable_count)
                constraints.append(
                    (block, rng.randint(0, block.bit_count()))
                )
            expected = max(
                subset.bit_count()
                for subset in range(safe_mask + 1)
                if subset & ~safe_mask == 0
                and all(
                    (subset & block).bit_count() <= quota
                    for block, quota in constraints
                )
            )
            with self.subTest(case=case):
                self.assertEqual(
                    capacity_module._maximum_quota_packing(
                        safe_mask, constraints
                    ),
                    expected,
                )

    def assert_matches_brute_force(
        self,
        questions: list[Question],
        graph: KnowledgeGraph,
        misconceptions: list[Misconception],
        target: CapacityTarget,
    ) -> None:
        optimized_component_bounds = capacity_module._component_bounds
        checked_components = 0

        def checked_component_bounds(component, **kwargs):
            nonlocal checked_components
            checked_components += 1
            actual = optimized_component_bounds(component, **kwargs)
            all_mask = kwargs["all_mask"]
            family_index = kwargs["family_index"]
            compiled_by_family = kwargs["compiled_by_family"]
            question_safe = kwargs["question_safe"]
            global_bits = tuple(
                1 << family_index[family_id] for family_id in component
            )

            @lru_cache(maxsize=None)
            def safe_indices(consumed: int) -> tuple[int, ...]:
                removed = sum(
                    bit
                    for index, bit in enumerate(global_bits)
                    if consumed & (1 << index)
                )
                remaining = all_mask & ~removed
                return tuple(
                    index
                    for index, family_id in enumerate(component)
                    if not consumed & (1 << index)
                    and any(
                        question_safe(question, remaining)
                        for question in compiled_by_family[family_id]
                    )
                )

            @lru_cache(maxsize=None)
            def brute_low(consumed: int) -> tuple[int, tuple[str, ...]]:
                safe = safe_indices(consumed)
                if not safe:
                    return 0, ()
                return min(
                    (
                        1 + child_count,
                        (component[index],) + child_sequence,
                    )
                    for index in safe
                    for child_count, child_sequence in (
                        brute_low(consumed | (1 << index)),
                    )
                )

            @lru_cache(maxsize=None)
            def brute_high(consumed: int) -> tuple[int, tuple[str, ...]]:
                safe = safe_indices(consumed)
                if not safe:
                    return 0, ()
                candidates = [
                    (
                        1 + child_count,
                        (component[index],) + child_sequence,
                    )
                    for index in safe
                    for child_count, child_sequence in (
                        brute_high(consumed | (1 << index)),
                    )
                ]
                return min(
                    candidates,
                    key=lambda candidate: (-candidate[0], candidate[1]),
                )

            expected_low, expected_low_sequence = brute_low(0)
            expected_high, expected_high_sequence = brute_high(0)
            self.assertEqual(
                (
                    actual.low,
                    actual.high,
                    actual.low_sequence,
                    actual.high_sequence,
                ),
                (
                    expected_low,
                    expected_high,
                    expected_low_sequence,
                    expected_high_sequence,
                ),
            )
            return actual

        with patch.object(
            capacity_module,
            "_component_bounds",
            side_effect=checked_component_bounds,
        ):
            analyze_sustained_capacity(
                questions,
                graph,
                misconceptions,
                [target],
            )
        self.assertGreater(checked_components, 0)

    def test_certificates_match_exhaustive_random_small_banks(self) -> None:
        concept = Concept("c", "Concept", "Description")
        graph = KnowledgeGraph([concept], [])
        rng = random.Random(803_021)
        for case in range(80):
            misconception_count = rng.randint(2, 4)
            misconceptions = [
                Misconception(
                    f"m_{case}_{index}",
                    "c",
                    f"Gap {index}",
                    "Description",
                )
                for index in range(misconception_count)
            ]
            misconception_ids = tuple(
                misconception.id for misconception in misconceptions
            )
            questions: list[Question] = []
            for family_index in range(rng.randint(3, 7)):
                family_id = f"f_{case}_{family_index}"
                variant_count = 2 if rng.random() < 0.35 else 1
                for variant in range(variant_count):
                    selected = tuple(
                        misconception_id
                        for misconception_id in misconception_ids
                        if rng.random() < 0.55
                    )
                    if not selected:
                        selected = (
                            misconception_ids[
                                (family_index + variant) % misconception_count
                            ],
                        )
                    questions.append(
                        make_question(
                            family_id,
                            "c",
                            selected,
                            question_id=f"q_{family_id}_{variant}",
                        )
                    )
            with self.subTest(case=case):
                self.assert_matches_brute_force(
                    questions,
                    graph,
                    misconceptions,
                    concept_target("c"),
                )

    def test_certificates_match_brute_force_with_external_reserves(self) -> None:
        root = Concept("root", "Root", "Description")
        support = Concept("support", "Support", "Description")
        graph = KnowledgeGraph(
            [root, support],
            [ConceptEdge("support", "root", RelationType.PREREQUISITE)],
        )
        misconceptions = [
            Misconception("shared", "root", "Shared gap", "Description"),
            Misconception("other", "root", "Other gap", "Description"),
            Misconception("support_gap", "support", "Support gap", "Description"),
        ]
        questions = [
            make_question("main_a", "root", ("shared", "other")),
            make_question(
                "main_b",
                "root",
                ("shared",),
                question_id="q_main_b_variant_a",
            ),
            make_question(
                "main_b",
                "root",
                ("other",),
                question_id="q_main_b_variant_b",
            ),
            make_question("main_c", "root", ("other",)),
            make_question("support_a", "support", ("shared", "support_gap")),
            make_question("support_b", "support", ("shared", "support_gap")),
        ]

        self.assert_matches_brute_force(
            questions,
            graph,
            misconceptions,
            concept_target("root"),
        )

    def test_certificates_match_brute_force_with_exact_automorphisms(self) -> None:
        concept = Concept("c", "Concept", "Description")
        misconceptions = [
            Misconception(name, "c", name, "Description")
            for name in ("m1", "m2", "m3")
        ]
        questions = [
            make_question(family_id, "c", ("m1", "m2"))
            for family_id in ("symmetric_a", "symmetric_b", "symmetric_c")
        ]
        questions.extend(
            (
                make_question("overlap_a", "c", ("m2", "m3")),
                make_question("overlap_b", "c", ("m1", "m3")),
                make_question(
                    "variant",
                    "c",
                    ("m1",),
                    question_id="q_variant_a",
                ),
                make_question(
                    "variant",
                    "c",
                    ("m3",),
                    question_id="q_variant_b",
                ),
            )
        )

        self.assert_matches_brute_force(
            questions,
            KnowledgeGraph([concept], []),
            misconceptions,
            concept_target("c"),
        )

    def test_objective_reserves_may_use_a_supporting_primary_concept(self) -> None:
        root = Concept("root", "Root", "Description")
        support = Concept("support", "Support", "Description")
        misconception = Misconception("m", "root", "Named gap", "Description")
        objective = LearningObjective(
            "lo",
            "Trace the operation",
            "Trace the operation across its supported representations.",
            "root",
            ("support",),
            ObjectiveOperation.TRACE,
        )
        main = make_objective_question("main", "root", "m", objective)
        reserve_a = make_objective_question("reserve_a", "support", "m", objective)
        reserve_b = make_objective_question("reserve_b", "support", "m", objective)
        graph = KnowledgeGraph([root, support], [])

        blocked = analyze_sustained_capacity(
            [main, reserve_a], graph, [misconception], [concept_target("root")]
        ).targets[0]
        unlocked = analyze_sustained_capacity(
            [main, reserve_a, reserve_b],
            graph,
            [misconception],
            [concept_target("root")],
        ).targets[0]

        self.assertEqual(blocked.initial_safe_family_ids, ())
        self.assertEqual(unlocked.initial_safe_family_ids, ("main",))
        self.assertEqual(unlocked.achievable_main_capacity, 1)
        self.assertNotIn(
            "reserve_a", unlocked.maximum_capacity.consumed_family_ids
        )

    def test_cross_diagnosis_does_not_substitute_for_direct_objective_service(self) -> None:
        root = Concept("root", "Root", "Description")
        other = Concept("other", "Other", "Description")
        misconception = Misconception("m", "root", "Named gap", "Description")
        target_objective = LearningObjective(
            "lo_target",
            "Trace the target",
            "Trace the target operation.",
            "root",
            (),
            ObjectiveOperation.TRACE,
        )
        other_objective = LearningObjective(
            "lo_other",
            "Trace another target",
            "Trace another target operation.",
            "other",
            (),
            ObjectiveOperation.TRACE,
        )
        main = make_objective_question(
            "main", "root", "m", target_objective
        )
        cross_diagnostics = [
            make_objective_question(
                family_id,
                "other",
                "m",
                other_objective,
                diagnostic_objective_id=target_objective.id,
            )
            for family_id in ("cross_a", "cross_b")
        ]
        graph = KnowledgeGraph([root, other], [])

        blocked = analyze_sustained_capacity(
            [main, *cross_diagnostics],
            graph,
            [misconception],
            [concept_target("root")],
        ).targets[0]
        unlocked = analyze_sustained_capacity(
            [
                main,
                *cross_diagnostics,
                make_objective_question(
                    "direct_a", "root", "m", target_objective
                ),
                make_objective_question(
                    "direct_b", "root", "m", target_objective
                ),
            ],
            graph,
            [misconception],
            [concept_target("root")],
        ).targets[0]

        self.assertEqual(blocked.initial_safe_family_ids, ())
        self.assertIn("main", unlocked.initial_safe_family_ids)

    def test_objective_blockers_identify_the_exact_evidence_path(self) -> None:
        concept = Concept("c", "Concept", "Description")
        misconception = Misconception("m", "c", "Named gap", "Description")
        objective = LearningObjective(
            "lo",
            "Distinguish cases",
            "Distinguish the relevant cases.",
            "c",
            (),
            ObjectiveOperation.DISTINGUISH,
        )
        questions = [
            make_objective_question(family_id, "c", "m", objective)
            for family_id in ("a", "b")
        ]

        result = analyze_sustained_capacity(
            questions,
            KnowledgeGraph([concept], []),
            [misconception],
            [concept_target("c")],
        ).targets[0]

        self.assertEqual(result.achievable_main_capacity, 0)
        self.assertTrue(result.maximum_capacity.blockers)
        self.assertEqual(
            {blocker.objective_id for blocker in result.maximum_capacity.blockers},
            {"lo"},
        )

    def test_initial_serviceability_does_not_imply_sustained_capacity(self) -> None:
        concept = Concept("c", "Concept", "Description")
        questions, misconceptions = interchangeable_bank("c", "x", 3)

        result = analyze_sustained_capacity(
            questions,
            KnowledgeGraph([concept], []),
            misconceptions,
            [concept_target("c")],
        ).targets[0]

        self.assertEqual(len(result.initial_safe_family_ids), 3)
        self.assertEqual(result.order_robust_main_capacity, 1)
        self.assertEqual(result.achievable_main_capacity, 1)
        self.assertEqual(len(result.maximum_capacity.terminal_main_family_ids), 2)
        self.assertEqual(result.status, "thin")

    def test_five_interchangeable_families_sustain_three(self) -> None:
        concept = Concept("c", "Concept", "Description")
        questions, misconceptions = interchangeable_bank("c", "x", 5)

        result = analyze_sustained_capacity(
            questions,
            KnowledgeGraph([concept], []),
            misconceptions,
            [concept_target("c")],
        ).targets[0]

        self.assertEqual(result.order_robust_main_capacity, 3)
        self.assertEqual(result.achievable_main_capacity, 3)
        self.assertEqual(
            result.maximum_capacity.consumed_family_ids,
            ("x_f1", "x_f2", "x_f3"),
        )
        self.assertEqual(result.status, "adequate")

    def test_repair_and_verification_must_be_distinct_families(self) -> None:
        concept = Concept("c", "Concept", "Description")
        two_questions, misconceptions = interchangeable_bank("c", "x", 2)
        three_questions, _ = interchangeable_bank("c", "x", 3)
        graph = KnowledgeGraph([concept], [])

        blocked = analyze_sustained_capacity(
            two_questions, graph, misconceptions, [concept_target("c")]
        ).targets[0]
        unlocked = analyze_sustained_capacity(
            three_questions, graph, misconceptions, [concept_target("c")]
        ).targets[0]

        self.assertEqual(blocked.initial_safe_family_ids, ())
        self.assertEqual(blocked.achievable_main_capacity, 0)
        self.assertTrue(
            all(
                blocker.reason == "repair_verification_not_independent"
                for blocker in blocked.maximum_capacity.blockers
            )
        )
        self.assertEqual(unlocked.achievable_main_capacity, 1)

    def test_order_robust_and_achievable_bounds_can_differ(self) -> None:
        concept = Concept("c", "Concept", "Description")
        misconceptions = [
            Misconception(name, "c", name, "Description")
            for name in ("m_a", "m_c", "m_d")
        ]
        questions = [
            make_question("a", "c", ("m_a",)),
            make_question("b", "c", ("m_a", "m_c", "m_d")),
            make_question("c", "c", ("m_c",)),
            make_question("d", "c", ("m_d",)),
        ]

        result = analyze_sustained_capacity(
            questions,
            KnowledgeGraph([concept], []),
            misconceptions,
            [concept_target("c", target_main_count=2)],
        ).targets[0]

        self.assertEqual(result.order_robust_main_capacity, 1)
        self.assertEqual(result.achievable_main_capacity, 2)
        self.assertEqual(result.earliest_exhaustion.consumed_family_ids, ("b",))
        self.assertEqual(result.maximum_capacity.consumed_family_ids, ("a", "c"))
        self.assertEqual(result.status, "order_sensitive")

    def test_cross_concept_pair_must_be_inside_learning_scope(self) -> None:
        root = Concept("root", "Root", "Description")
        prerequisite = Concept("pre", "Prerequisite", "Description")
        unrelated = Concept("other", "Other", "Description")
        graph = KnowledgeGraph(
            [root, prerequisite, unrelated],
            [ConceptEdge("pre", "root", RelationType.PREREQUISITE)],
        )
        misconceptions = [
            Misconception("root_1", "root", "R1", "Description"),
            Misconception("root_2", "root", "R2", "Description"),
            Misconception("pre_gap", "pre", "P", "Description"),
        ]
        root_questions = [
            make_question("trigger", "root", ("pre_gap", "root_1", "root_2")),
            make_question("root_a", "root", ("root_1", "root_2")),
            make_question("root_b", "root", ("root_1", "root_2")),
        ]
        one_scoped = [make_question("pre_a", "pre", ("pre_gap",))]
        out_of_scope = [
            make_question("other_a", "other", ("pre_gap",)),
            make_question("other_b", "other", ("pre_gap",)),
        ]

        blocked = analyze_sustained_capacity(
            [*root_questions, *one_scoped, *out_of_scope],
            graph,
            misconceptions,
            [concept_target("root")],
        ).targets[0]
        unlocked = analyze_sustained_capacity(
            [
                *root_questions,
                *one_scoped,
                make_question("pre_b", "pre", ("pre_gap",)),
                *out_of_scope,
            ],
            graph,
            misconceptions,
            [concept_target("root")],
        ).targets[0]

        self.assertNotIn("trigger", blocked.initial_safe_family_ids)
        self.assertIn("trigger", unlocked.initial_safe_family_ids)
        self.assertNotIn("other_a", unlocked.maximum_capacity.consumed_family_ids)

    def test_family_variants_only_use_target_owned_questions(self) -> None:
        root = Concept("root", "Root", "Description")
        prerequisite = Concept("pre", "Prerequisite", "Description")
        graph = KnowledgeGraph(
            [root, prerequisite],
            [ConceptEdge("pre", "root", RelationType.PREREQUISITE)],
        )
        misconceptions = [
            Misconception("common", "root", "Common", "Description"),
            Misconception("unique", "root", "Unique", "Description"),
            Misconception("pre_common", "pre", "Pre", "Description"),
        ]
        base = [
            make_question("shared", "root", ("unique",), question_id="owned_bad"),
            make_question(
                "shared", "pre", ("pre_common",), question_id="support_good"
            ),
            make_question("root_a", "root", ("common",)),
            make_question("root_b", "root", ("common",)),
            make_question("pre_a", "pre", ("pre_common",)),
            make_question("pre_b", "pre", ("pre_common",)),
        ]

        blocked = analyze_sustained_capacity(
            base, graph, misconceptions, [concept_target("root")]
        ).targets[0]
        unlocked = analyze_sustained_capacity(
            [
                *base,
                make_question(
                    "shared", "root", ("common",), question_id="owned_good"
                ),
            ],
            graph,
            misconceptions,
            [concept_target("root")],
        ).targets[0]

        self.assertEqual(blocked.eligible_family_count, 3)
        self.assertNotIn("shared", blocked.initial_safe_family_ids)
        self.assertIn("shared", unlocked.initial_safe_family_ids)

    def test_independent_components_sum_and_reports_are_deterministic_json(self) -> None:
        concepts = [
            Concept("c1", "One", "Description"),
            Concept("c2", "Two", "Description"),
        ]
        first_questions, first_misconceptions = interchangeable_bank("c1", "a", 3)
        second_questions, second_misconceptions = interchangeable_bank("c2", "b", 3)
        target = CapacityTarget("both", "topic", ("c1", "c2"), 2)
        graph = KnowledgeGraph(concepts, [])

        forward = analyze_sustained_capacity(
            [*first_questions, *second_questions],
            graph,
            [*first_misconceptions, *second_misconceptions],
            [target],
        )
        reverse = analyze_sustained_capacity(
            [*reversed(second_questions), *reversed(first_questions)],
            graph,
            [*reversed(second_misconceptions), *reversed(first_misconceptions)],
            [target],
        )

        result = forward.targets[0]
        self.assertEqual(result.component_count, 2)
        self.assertEqual(result.largest_component, 3)
        self.assertEqual(result.order_robust_main_capacity, 2)
        self.assertEqual(result.achievable_main_capacity, 2)
        self.assertEqual(forward.to_dict(), reverse.to_dict())
        json.dumps(forward.to_dict(), allow_nan=False)

    def test_healthy_aggregate_is_blocked_by_an_uncovered_owned_concept(self) -> None:
        concepts = [
            Concept("covered", "Covered", "Description"),
            Concept("missing", "Missing", "Description"),
        ]
        questions, misconceptions = interchangeable_bank("covered", "rich", 7)
        target = CapacityTarget("topic", "topic", ("covered", "missing"))

        result = analyze_sustained_capacity(
            questions,
            KnowledgeGraph(concepts, []),
            misconceptions,
            [target],
        ).targets[0]
        terms = result.to_dict()

        self.assertEqual(result.aggregate_status, "healthy")
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.owned_concept_order_robust_floor, 0)
        self.assertEqual(result.owned_concept_achievable_floor, 0)
        self.assertEqual(result.missing_owned_concept_ids, ("missing",))
        self.assertEqual(result.thin_owned_concept_ids, ())
        self.assertEqual(
            [row["concept_id"] for row in terms["owned_concepts"]],
            ["covered", "missing"],
        )
        self.assertEqual(terms["missing_owned_concept_ids"], ["missing"])

    def test_healthy_aggregate_is_downgraded_by_a_thin_owned_concept(self) -> None:
        concepts = [
            Concept("deep", "Deep", "Description"),
            Concept("thin", "Thin", "Description"),
        ]
        deep_questions, deep_misconceptions = interchangeable_bank(
            "deep", "deep", 9
        )
        thin_questions, thin_misconceptions = interchangeable_bank(
            "thin", "thin", 3
        )

        result = analyze_sustained_capacity(
            [*deep_questions, *thin_questions],
            KnowledgeGraph(concepts, []),
            [*deep_misconceptions, *thin_misconceptions],
            [CapacityTarget("topic", "topic", ("deep", "thin"))],
        ).targets[0]

        self.assertEqual(result.aggregate_status, "healthy")
        self.assertEqual(result.status, "thin")
        self.assertEqual(result.missing_owned_concept_ids, ())
        self.assertEqual(result.thin_owned_concept_ids, ("thin",))
        self.assertEqual(result.owned_concept_order_robust_floor, 1)

    def test_non_live_and_unavailable_families_do_not_supply_capacity(self) -> None:
        concept = Concept("c", "Concept", "Description")
        questions, misconceptions = interchangeable_bank("c", "x", 5)
        questions[3] = make_question(
            "x_f4", "c", tuple(item.id for item in misconceptions), status=QuestionStatus.DRAFT
        )

        result = analyze_sustained_capacity(
            questions,
            KnowledgeGraph([concept], []),
            misconceptions,
            [concept_target("c")],
            unavailable_family_ids={"x_f5"},
        ).targets[0]

        self.assertEqual(result.eligible_family_count, 3)
        self.assertEqual(result.achievable_main_capacity, 1)

    def test_exact_state_limit_fails_closed(self) -> None:
        concept = Concept("c", "Concept", "Description")
        questions, misconceptions = interchangeable_bank("c", "x", 5)

        with self.assertRaises(CapacityAnalysisLimitError) as raised:
            analyze_sustained_capacity(
                questions,
                KnowledgeGraph([concept], []),
                misconceptions,
                [concept_target("c")],
                state_limit=1,
            )

        self.assertEqual(raised.exception.target_id, "c")
        self.assertEqual(raised.exception.state_limit, 1)

    def test_topic_helper_includes_descendant_concepts(self) -> None:
        topics = [
            Topic("parent", "domain", "Parent", "Description", ("c1",)),
            Topic(
                "child",
                "domain",
                "Child",
                "Description",
                ("c2",),
                parent_id="parent",
            ),
        ]

        target = topic_target("parent", topics)

        self.assertEqual(target.owned_concept_ids, ("c1", "c2"))


class CapacityCliTestCase(unittest.TestCase):
    @staticmethod
    def run_cli(arguments: list[str]) -> tuple[int, str, str]:
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(arguments)
        return code, output.getvalue(), errors.getvalue()

    def test_focused_json_output_is_exact_and_machine_readable(self) -> None:
        code, output, errors = self.run_cli(
            [
                "capacity",
                str(CORPUS),
                "--topic",
                "t_transformers",
                "--json",
            ]
        )

        payload = json.loads(output)
        self.assertEqual(code, 0, errors)
        self.assertTrue(payload["exact"])
        self.assertEqual(payload["algorithm"], "sustained-serviceability-v2")
        self.assertEqual(payload["summary"]["target_count"], 1)
        result = payload["targets"][0]
        self.assertEqual(result["target_id"], "t_transformers")
        self.assertLessEqual(
            result["order_robust_main_capacity"],
            result["achievable_main_capacity"],
        )

    def test_quarantined_transformer_candidates_do_not_inflate_live_capacity(
        self,
    ) -> None:
        bundle = load_bundle(CORPUS)
        topic = next(
            item for item in bundle["topics"] if item["id"] == "t_transformers"
        )
        owned_concepts = set(topic["concept_ids"])
        active_questions = [
            question
            for question in bundle["questions"]
            if question["status"] == "approved"
            and any(
                concept["concept_id"] in owned_concepts
                and concept["role"] == "primary"
                for concept in question["concepts"]
            )
        ]
        quarantined_bridge_ids = {
            "q_attention_runtime_workspace_boundary_001",
            "q_causal_cross_attention_mask_scope_002",
            "q_transformer_unexpected_cross_token_path_001",
        }

        self.assertEqual(len(active_questions), 48)
        self.assertEqual(
            len({question["family_id"] for question in active_questions}),
            48,
        )
        self.assertTrue(
            quarantined_bridge_ids.isdisjoint(
                {question["id"] for question in active_questions}
            )
        )

        code, output, errors = self.run_cli(
            [
                "capacity",
                str(CORPUS),
                "--topic",
                "t_transformers",
                "--json",
            ]
        )
        payload = json.loads(output)
        self.assertEqual(code, 0, errors)
        [result] = payload["targets"]
        self.assertEqual(result["order_robust_main_capacity"], 26)
        self.assertEqual(result["achievable_main_capacity"], 26)

    def test_default_scope_analyzes_all_topics(self) -> None:
        code, output, errors = self.run_cli(
            ["capacity", str(CORPUS), "--json"]
        )

        payload = json.loads(output)
        bundle = load_bundle(CORPUS)
        self.assertEqual(code, 0, errors)
        self.assertEqual(payload["summary"]["target_count"], len(bundle["topics"]))
        self.assertTrue(
            all(row["target_type"] == "topic" for row in payload["targets"])
        )
        for row in payload["targets"]:
            robust_values = [
                child["order_robust_main_capacity"]
                for child in row["owned_concepts"]
            ]
            achievable_values = [
                child["achievable_main_capacity"]
                for child in row["owned_concepts"]
            ]
            self.assertEqual(
                row["owned_concept_order_robust_floor"], min(robust_values)
            )
            self.assertEqual(
                row["owned_concept_achievable_floor"], min(achievable_values)
            )
            child_deficits = (
                row["missing_owned_concept_ids"]
                + row["thin_owned_concept_ids"]
                + row["order_sensitive_owned_concept_ids"]
            )
            if child_deficits:
                self.assertNotEqual(row["status"], "healthy")

    def test_strict_fails_for_blocked_target_but_regular_mode_is_informational(
        self,
    ) -> None:
        bundle = load_bundle(CORPUS)
        for question in bundle["questions"]:
            question["status"] = "retired"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retired.json"
            path.write_text(json.dumps(bundle), encoding="utf-8")
            regular, output, errors = self.run_cli(
                ["capacity", str(path), "--concept", "c_causal_inference", "--json"]
            )
            strict, strict_output, strict_errors = self.run_cli(
                [
                    "capacity",
                    str(path),
                    "--concept",
                    "c_causal_inference",
                    "--strict",
                    "--json",
                ]
            )

        self.assertEqual(regular, 0, errors)
        self.assertEqual(json.loads(output)["targets"][0]["status"], "blocked")
        self.assertEqual(strict, 2, strict_errors)
        self.assertEqual(
            json.loads(strict_output)["summary"]["strict_failure_count"], 1
        )

    def test_explicit_main_horizon_exposes_topic_capacity_debt(self) -> None:
        code, output, errors = self.run_cli(
            [
                "capacity",
                str(CORPUS),
                "--topic",
                "t_retrieval_augmented_generation",
                "--target-main-count",
                "10",
                "--strict",
                "--json",
            ]
        )

        payload = json.loads(output)
        self.assertEqual(code, 2, errors)
        self.assertEqual(payload["summary"]["requested_main_capacity"], 10)
        self.assertEqual(
            payload["summary"]["strict_failure_targets"],
            ["t_retrieval_augmented_generation"],
        )
        [result] = payload["targets"]
        self.assertEqual(result["target_main_count"], 10)
        # Generated questions without human review are quarantined and must not
        # inflate the live serviceability horizon.
        self.assertEqual(result["order_robust_main_capacity"], 4)
        self.assertEqual(result["achievable_main_capacity"], 4)
        self.assertEqual(result["status"], "thin")

    def test_state_limit_failure_never_emits_a_partial_report(self) -> None:
        code, output, errors = self.run_cli(
            [
                "capacity",
                str(CORPUS),
                "--topic",
                "t_transformers",
                "--state-limit",
                "1",
                "--json",
            ]
        )

        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("analysis is incomplete", errors)
        self.assertIn("no heuristic result", errors)

    def test_quarantine_impact_finds_exact_agent_bridge_without_activation(
        self,
    ) -> None:
        before = corpus_source_digest(CORPUS)
        code, output, errors = self.run_cli(
            [
                "capacity",
                str(CORPUS),
                "--topic",
                "t_llm_agents",
                "--quarantine-impact",
                "--json",
            ]
        )

        payload = json.loads(output)
        self.assertEqual(code, 0, errors)
        self.assertEqual(
            corpus_source_digest(CORPUS), before
        )
        self.assertTrue(
            payload["exact_within_declared_search_space"]
        )
        self.assertTrue(payload["non_activating"])
        self.assertEqual(payload["corpus_sha256"], before)
        self.assertEqual(payload["baseline"]["status"], "thin")
        self.assertEqual(payload["minimal_closing_combination_size"], 1)
        self.assertEqual(
            {
                tuple(row["question_ids"])
                for row in payload["closing_combinations"]
            },
            {
                ("q_agent_dry_run_effect_boundary_001",),
                ("q_agent_revision_ordering_001",),
                ("q_agent_saga_compensation_001",),
                ("q_agent_snapshot_observation_completeness_001",),
            },
        )
        self.assertFalse(
            payload["boundary"]["activation_performed_by_analyzer"]
        )
        self.assertFalse(
            payload["boundary"]["source_corpus_mutated_by_analyzer"]
        )
        self.assertTrue(
            payload["boundary"]["human_review_required_before_activation"]
        )
        for candidate in payload["candidates"]:
            self.assertEqual(
                candidate["original_question_status"], "quarantined"
            )
            self.assertFalse(
                candidate[
                    "original_question_eligible_for_adaptation"
                ]
            )
            self.assertRegex(
                candidate["question_input_digest"], r"^[0-9a-f]{64}$"
            )
            self.assertNotIn("stem", candidate)
            self.assertNotIn("options", candidate)

    def test_quarantine_impact_finds_rag_pair_and_reports_truncation(self) -> None:
        code, output, errors = self.run_cli(
            [
                "capacity",
                str(CORPUS),
                "--topic",
                "t_retrieval_augmented_generation",
                "--quarantine-impact",
                "--result-limit",
                "1",
                "--json",
            ]
        )

        payload = json.loads(output)
        self.assertEqual(code, 0, errors)
        self.assertEqual(payload["candidate_count"], 8)
        self.assertEqual(payload["evaluated_combination_count"], 36)
        self.assertEqual(payload["baseline"]["status"], "thin")
        self.assertEqual(payload["minimal_closing_combination_size"], 2)
        self.assertEqual(
            payload["closing_combination_count_at_minimum"], 16
        )
        self.assertEqual(len(payload["closing_combinations"]), 1)
        self.assertTrue(payload["closing_combinations_truncated"])
        self.assertEqual(
            payload["search_outcome"],
            "minimal_closing_combination_found",
        )

    def test_quarantine_impact_bounded_nonclosure_is_not_impossibility(
        self,
    ) -> None:
        code, output, errors = self.run_cli(
            [
                "capacity",
                str(CORPUS),
                "--topic",
                "t_llm_agents",
                "--target-main-count",
                "10",
                "--quarantine-impact",
                "--maximum-combination-size",
                "3",
                "--json",
            ]
        )

        payload = json.loads(output)
        self.assertEqual(code, 0, errors)
        self.assertEqual(payload["candidate_count"], 10)
        self.assertEqual(
            payload["preflight_admissible_combination_count"], 175
        )
        self.assertEqual(payload["evaluated_combination_count"], 175)
        self.assertIsNone(payload["minimal_closing_combination_size"])
        self.assertFalse(
            payload["admissible_candidate_search_space_fully_exhausted"]
        )
        self.assertFalse(
            payload[
                "closure_absence_is_exact_within_declared_candidate_space"
            ]
        )
        self.assertEqual(
            payload["search_outcome"],
            "no_closing_combination_within_bound",
        )

    def test_quarantine_impact_requires_one_scope_and_is_not_a_strict_gate(
        self,
    ) -> None:
        for arguments, expected in (
            (
                [
                    "capacity",
                    str(CORPUS),
                    "--quarantine-impact",
                    "--json",
                ],
                "requires exactly one",
            ),
            (
                [
                    "capacity",
                    str(CORPUS),
                    "--topic",
                    "t_llm_agents",
                    "--quarantine-impact",
                    "--strict",
                    "--json",
                ],
                "not a live corpus gate",
            ),
            (
                [
                    "capacity",
                    str(CORPUS),
                    "--candidate-limit",
                    "3",
                    "--json",
                ],
                "require --quarantine-impact",
            ),
            (
                [
                    "capacity",
                    str(CORPUS),
                    "--topic",
                    "t_llm_agents",
                    "--quarantine-impact",
                    "--candidate-limit",
                    "0",
                    "--json",
                ],
                "candidate_limit must be between",
            ),
        ):
            with self.subTest(arguments=arguments):
                code, output, errors = self.run_cli(arguments)
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertIn(expected, errors)

    def test_quarantine_impact_candidate_limit_fails_instead_of_truncating(
        self,
    ) -> None:
        code, output, errors = self.run_cli(
            [
                "capacity",
                str(CORPUS),
                "--topic",
                "t_llm_agents",
                "--quarantine-impact",
                "--candidate-limit",
                "9",
                "--json",
            ]
        )

        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn("refuses to truncate 10 candidates", errors)

    def test_quarantine_impact_evaluation_budget_emits_no_partial_report(
        self,
    ) -> None:
        code, output, errors = self.run_cli(
            [
                "capacity",
                str(CORPUS),
                "--topic",
                "t_llm_agents",
                "--target-main-count",
                "10",
                "--quarantine-impact",
                "--maximum-combination-size",
                "3",
                "--evaluation-limit",
                "174",
                "--json",
            ]
        )

        self.assertEqual(code, 2)
        self.assertEqual(output, "")
        self.assertIn(
            "requires 175 admissible candidate-subset evaluations",
            errors,
        )
        self.assertIn("No heuristic or partial result was used", errors)

    def test_capacity_selectors_are_mutually_exclusive(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "capacity",
                    str(CORPUS),
                    "--concept",
                    "c_attention",
                    "--topic",
                    "t_transformers",
                ]
            )


if __name__ == "__main__":
    unittest.main()
