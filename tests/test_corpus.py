# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tsq.corpus import (
    load_bundle,
    parse_bundle,
    parse_catalog,
    read_and_parse,
    validate_bundle,
)
from tsq.errors import ConflictError, ValidationError
from tsq.graph import KnowledgeGraph
from tsq.models import (
    Concept,
    ConceptEdge,
    ConceptRole,
    ConceptWeight,
    Option,
    QuestionStatus,
    RelationType,
)
from tsq.quality import audit_corpus, validate_question
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"


def declare_test_fixture_generation_provenance(
    bundle: dict[str, object],
) -> None:
    """Move deliberately mutated raw fixtures outside the legacy exception."""

    questions = bundle.get("questions")
    if not isinstance(questions, list):
        return
    for question in questions:
        if not isinstance(question, dict):
            continue
        provenance = question.get("provenance")
        if isinstance(provenance, dict) and "generated" not in provenance:
            provenance["generated"] = False


class CorpusTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.concepts,
            cls.edges,
            cls.misconceptions,
            cls.sources,
            cls.questions,
            cls.domains,
            cls.topics,
        ) = read_and_parse(CORPUS, include_catalog=True)

    def test_curriculum_catalog_is_clear_complete_and_canonical(self) -> None:
        self.assertEqual([domain.name for domain in self.domains], ["Artificial Intelligence"])
        topic_by_id = {topic.id: topic for topic in self.topics}
        self.assertEqual(
            topic_by_id["t_large_language_models"].parent_id,
            None,
        )
        self.assertEqual(
            {
                topic.name
                for topic in self.topics
                if topic.parent_id == "t_large_language_models"
            },
            {
                "Language Modeling",
                "Transformers",
                "Retrieval-Augmented Generation",
                "LLM Agents",
            },
        )
        owners = [
            concept_id for topic in self.topics for concept_id in topic.concept_ids
        ]
        self.assertCountEqual(owners, [concept.id for concept in self.concepts])
        self.assertEqual(len(owners), len(set(owners)))
        self.assertNotIn("c_ai_learning_systems", owners)

    def test_catalog_rejects_ambiguous_ownership_and_asymmetric_relations(self) -> None:
        bundle = json.loads(CORPUS.read_text(encoding="utf-8"))
        concepts, _, _, _, questions = parse_bundle(bundle)
        bundle["topics"][0]["concept_ids"].append(
            bundle["topics"][1]["concept_ids"][0]
        )
        related = bundle["topics"][0]["related_topic_ids"][0]
        counterpart = next(topic for topic in bundle["topics"] if topic["id"] == related)
        counterpart["related_topic_ids"].remove(bundle["topics"][0]["id"])

        with self.assertRaises(ValidationError) as raised:
            parse_catalog(bundle, concepts, questions)

        codes = {issue.code for issue in raised.exception.issues}
        self.assertIn("multiple_topic_owners", codes)
        self.assertIn("asymmetric_related_topic", codes)

    def test_new_llm_objectives_have_independent_misconception_families(self) -> None:
        new_objectives = {
            "c_autoregressive_language_modeling",
            "c_attention_scaling",
            "c_causal_masking",
            "c_rag_grounding",
            "c_rag_retrieval_quality",
            "c_agent_tool_use",
            "c_agent_observation_loop",
        }
        for concept_id in new_objectives:
            items = [
                question
                for question in self.questions
                if question.primary_concept_id == concept_id
            ]
            self.assertGreaterEqual(len(items), 3, concept_id)
            self.assertEqual(len({item.family_id for item in items}), len(items), concept_id)
            misconception_sets = [item.misconception_ids for item in items]
            self.assertTrue(all(len(values) == 3 for values in misconception_sets))

    def test_objective_diagnoses_have_three_direct_evidence_families(self) -> None:
        diagnosed_pairs: set[tuple[str, str]] = set()
        direct_families: dict[tuple[str, str], set[str]] = {}
        for question in self.questions:
            if (
                question.objective_id is None
                or not question.status.eligible_for_adaptation
            ):
                continue
            for option in question.options:
                if option.correct or option.misconception_id is None:
                    continue
                diagnostic_objective_id = (
                    option.diagnostic_objective_id or question.objective_id
                )
                pair = (diagnostic_objective_id, option.misconception_id)
                diagnosed_pairs.add(pair)
                if question.objective_id == diagnostic_objective_id:
                    direct_families.setdefault(pair, set()).add(question.family_id)

        self.assertTrue(diagnosed_pairs)
        for pair in sorted(diagnosed_pairs):
            self.assertGreaterEqual(
                len(direct_families.get(pair, set())),
                3,
                f"{pair[0]}/{pair[1]}",
            )

    def test_seed_corpus_has_no_blocking_deterministic_quality_issues(self) -> None:
        errors = [issue for issue in audit_corpus(self.questions) if issue.severity == "error"]
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(self.questions), 20)
        self.assertTrue(all(len(question.misconception_ids) == 3 for question in self.questions))

    def test_correct_answer_source_positions_have_no_material_skew(self) -> None:
        counts = [0, 0, 0, 0]
        for question in self.questions:
            index = next(index for index, option in enumerate(question.options) if option.correct)
            counts[index] += 1
        # Runtime presentation shuffles option order, so forcing an exact
        # source-file tie would make every reviewed corpus addition require a
        # semantically irrelevant rewrite. Retain a tight deterministic guard
        # against material authoring-position skew instead.
        self.assertLessEqual(
            max(counts) - min(counts),
            max(4, round(len(self.questions) * 0.02)),
        )

    def test_default_topic_has_deep_independent_focus_repair_paths(self) -> None:
        assessable = {
            "c_clustering",
            "c_dimensionality_reduction",
            "c_discriminative_models",
            "c_empirical_risk_minimization",
            "c_feature_scaling",
            "c_linear_models",
            "c_probability_reasoning",
            "c_reinforcement_learning",
        }
        live = [
            question
            for question in self.questions
            if question.status.eligible_for_adaptation
            and question.primary_concept_id in assessable
        ]
        families_by_concept = {
            concept_id: {
                question.family_id
                for question in live
                if question.primary_concept_id == concept_id
            }
            for concept_id in assessable
        }
        self.assertTrue(
            all(len(families) >= 8 for families in families_by_concept.values()),
            families_by_concept,
        )

        missing_repair_paths = []
        for question in live:
            for misconception_id in question.misconception_ids:
                independent = [
                    candidate
                    for candidate in live
                    if candidate.family_id != question.family_id
                    and misconception_id in candidate.misconception_ids
                ]
                if not independent:
                    missing_repair_paths.append((question.id, misconception_id))
        self.assertEqual(missing_repair_paths, [])

    def test_answer_length_leak_is_rejected(self) -> None:
        question = self.questions[0]
        correct = question.correct_option
        inflated = replace(
            correct,
            text=correct.text + " with a conspicuously exhaustive qualification" * 10,
        )
        modified = replace(
            question,
            options=tuple(inflated if option.id == correct.id else option for option in question.options),
        )
        issues = validate_question(modified)
        self.assertTrue(any(issue.code == "answer_length_leak" for issue in issues))

    def test_guess_rate_cannot_fall_below_forced_choice_chance(self) -> None:
        question = replace(self.questions[0], guess_rate=0.20)

        issues = validate_question(question)

        self.assertIn("guess_below_forced_choice_chance", {issue.code for issue in issues})

    def test_concept_roles_are_typed_and_reject_unknown_values(self) -> None:
        mapping = ConceptWeight("c_test", 1.0, "primary")
        self.assertIs(mapping.role, ConceptRole.PRIMARY)
        with self.assertRaises(ValueError):
            ConceptWeight("c_test", 1.0, "invented-role")

    def test_raw_bundle_validation_aggregates_type_range_and_finite_issues(self) -> None:
        bundle = json.loads(CORPUS.read_text(encoding="utf-8"))
        for question in bundle["questions"]:
            question["guess_rate"] = 0.25
        bundle["schema_version"] = "1"
        bundle["concepts"][0]["prior_mastery"] = float("nan")
        bundle["edges"][0]["weight"] = 0.0
        question = bundle["questions"][0]
        question["version"] = True
        question["difficulty"] = "hard"
        question["discrimination"] = float("inf")
        question["guess_rate"] = 0.20
        question["slip_rate"] = 0.30
        question["concepts"][0]["weight"] = False
        question["concepts"][0]["role"] = "invented-role"
        question["options"][0]["correct"] = "true"
        question["source_ids"][0] = 42
        question["provenance"] = []
        question["tags"] = [None]

        issues = validate_bundle(bundle)
        codes = {issue.code for issue in issues}

        self.assertGreaterEqual(len(issues), 13)
        self.assertTrue(
            {
                "schema_version_type",
                "non_finite",
                "out_of_range",
                "field_type",
                "invalid_concept_role",
                "guess_below_forced_choice_chance",
            }.issubset(codes)
        )
        self.assertTrue(all(issue.path for issue in issues))
        with self.assertRaises(ValidationError) as raised:
            parse_bundle(bundle)
        self.assertGreaterEqual(len(raised.exception.issues), 13)

    def test_raw_bundle_validation_rejects_non_object_and_missing_top_level_fields(self) -> None:
        self.assertEqual(validate_bundle([])[0].code, "bundle_type")

        issues = validate_bundle({})

        self.assertEqual(len(issues), 7)
        self.assertEqual({issue.code for issue in issues}, {"missing_field"})
        self.assertEqual(
            {issue.path for issue in issues},
            {
                "schema_version",
                "title",
                "concepts",
                "edges",
                "misconceptions",
                "sources",
                "questions",
            },
        )

    def test_json_loader_rejects_duplicate_keys_and_nonstandard_numbers(self) -> None:
        invalid_documents = (
            '{"schema_version":1,"title":"first","title":"second"}',
            '{"schema_version":1,"title":"invalid","concepts":[NaN]}',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for document in invalid_documents:
                with self.subTest(document=document):
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaises(ValidationError):
                        load_bundle(path)

    def test_semantic_bundle_validation_aggregates_reference_issues(self) -> None:
        bundle = json.loads(CORPUS.read_text(encoding="utf-8"))
        declare_test_fixture_generation_provenance(bundle)
        for question in bundle["questions"]:
            question["guess_rate"] = 0.25
        original_concept_id = bundle["concepts"][1]["id"]
        bundle["concepts"][1]["id"] = bundle["concepts"][0]["id"]
        bundle["questions"][1]["id"] = bundle["questions"][0]["id"]
        bundle["questions"][0]["options"][1]["misconception_id"] = "m_missing"
        bundle["questions"][0]["source_ids"] = ["s_missing"]

        with self.assertRaises(ValidationError) as raised:
            parse_bundle(bundle)

        codes = {issue.code for issue in raised.exception.issues}
        self.assertIn("duplicate_id", codes)
        self.assertTrue(
            any(
                issue.code == "unknown_concept_reference"
                and original_concept_id in issue.message
                for issue in raised.exception.issues
            )
        )
        self.assertIn("unknown_misconception_reference", codes)
        self.assertIn("unknown_source_reference", codes)

    def test_distractor_misconception_owner_must_be_mapped_on_item(self) -> None:
        bundle = json.loads(CORPUS.read_text(encoding="utf-8"))
        declare_test_fixture_generation_provenance(bundle)
        question = bundle["questions"][0]
        mapped = {mapping["concept_id"] for mapping in question["concepts"]}
        foreign_misconception = next(
            misconception
            for misconception in bundle["misconceptions"]
            if misconception["concept_id"] not in mapped
        )
        distractor = next(option for option in question["options"] if not option["correct"])
        distractor["misconception_id"] = foreign_misconception["id"]

        with self.assertRaises(ValidationError) as raised:
            parse_bundle(bundle)

        matching = [
            issue
            for issue in raised.exception.issues
            if issue.code == "unmapped_misconception_owner"
            and issue.question_id == question["id"]
        ]
        self.assertEqual(len(matching), 1)
        self.assertIn(foreign_misconception["concept_id"], matching[0].message)

    def test_revision_graph_preserves_identity_and_acyclic_order(self) -> None:
        baseline = json.loads(CORPUS.read_text(encoding="utf-8"))
        declare_test_fixture_generation_provenance(baseline)
        cases = (
            ("self", "revision_self_reference"),
            ("cycle", "revision_cycle"),
            ("version", "revision_version_order"),
            ("family", "revision_family_mismatch"),
        )
        for case, expected_code in cases:
            with self.subTest(case=case):
                bundle = copy.deepcopy(baseline)
                parent = bundle["questions"][0]
                child = copy.deepcopy(parent)
                child["id"] = f"q_revision_test_{case}"
                child["version"] = parent.get("version", 1) + 1
                child["stem"] += f" This is revised wording for the {case} invariant."
                child["revision_of"] = parent["id"]
                bundle["questions"].append(child)

                if case == "self":
                    child["revision_of"] = child["id"]
                elif case == "cycle":
                    parent["revision_of"] = child["id"]
                elif case == "version":
                    child["version"] = parent.get("version", 1)
                elif case == "family":
                    child["family_id"] += "_different"

                with self.assertRaises(ValidationError) as raised:
                    parse_bundle(bundle)
                self.assertIn(
                    expected_code,
                    {issue.code for issue in raised.exception.issues},
                )

    def test_corpus_audit_detects_option_only_answer_keys(self) -> None:
        base = self.questions[0]
        options = (
            Option(
                "a",
                "Choose the rigorously supported contextual conclusion here",
                True,
                "This is the supported conclusion under the stated evidence.",
            ),
            Option(
                "b",
                "Always choose shortcut alpha",
                False,
                "This shortcut ignores evidence needed for the conclusion.",
                "m_alpha",
            ),
            Option(
                "c",
                "Never inspect relevant evidence",
                False,
                "This rule discards evidence that changes the conclusion.",
                "m_beta",
            ),
            Option(
                "d",
                "Only trust surface wording",
                False,
                "This response mistakes surface wording for substantive evidence.",
                "m_gamma",
            ),
        )
        questions = [
            replace(
                base,
                id=f"q_metric_{index}",
                family_id=f"f_metric_{index}",
                stem=(
                    "A reviewer must decide which conclusion is justified by the available "
                    f"evidence in independent scenario {index}; which response is best?"
                ),
                options=options,
            )
            for index in range(12)
        ]

        issues = audit_corpus(questions)
        by_code = {issue.code: issue for issue in issues}

        self.assertEqual(by_code["longest_option_key_leak"].severity, "error")
        self.assertEqual(by_code["absolute_qualifier_key_leak"].severity, "error")

    def test_corpus_audit_detects_primary_mapping_coverage_gaps(self) -> None:
        issues = audit_corpus([self.questions[0]])
        codes = {issue.code for issue in issues}

        self.assertIn("missing_primary_mapping_coverage", codes)
        self.assertIn("insufficient_primary_family_coverage", codes)

    def test_corpus_audit_detects_contextually_unserviceable_families(self) -> None:
        bias_families = sorted(
            {
                question.family_id
                for question in self.questions
                if question.primary_concept_id == "c_bias_variance"
                and question.status.eligible_for_adaptation
            }
        )
        self.assertGreaterEqual(len(bias_families), 3)
        retained = set(bias_families[:2])
        without_third_safe_family = [
            question
            for question in self.questions
            if question.primary_concept_id != "c_bias_variance"
            or question.family_id in retained
        ]

        issues = audit_corpus(
            without_third_safe_family,
            knowledge_graph=KnowledgeGraph(self.concepts, self.edges),
            misconceptions=self.misconceptions,
        )

        contextual = [
            issue
            for issue in issues
            if issue.code == "insufficient_contextual_family_coverage"
            and "c_bias_variance" in issue.message
        ]
        self.assertEqual(len(contextual), 1)

    def test_seed_has_three_serviceable_families_for_every_primary_root(self) -> None:
        issues = audit_corpus(
            self.questions,
            knowledge_graph=KnowledgeGraph(self.concepts, self.edges),
            misconceptions=self.misconceptions,
        )

        contextual = [
            issue
            for issue in issues
            if issue.code == "insufficient_contextual_family_coverage"
        ]
        self.assertEqual(contextual, [])

    def test_ineligible_items_do_not_satisfy_live_bank_coverage(self) -> None:
        retired = replace(self.questions[0], status=QuestionStatus.RETIRED)

        issues = audit_corpus(
            [retired], expected_primary_concept_ids={retired.primary_concept_id}
        )

        self.assertIn(
            "missing_primary_mapping_coverage", {issue.code for issue in issues}
        )

    def test_strict_prerequisite_cycle_is_rejected(self) -> None:
        concepts = [
            Concept("a", "A", "Concept A"),
            Concept("b", "B", "Concept B"),
        ]
        edges = [
            ConceptEdge("a", "b", RelationType.PREREQUISITE),
            ConceptEdge("b", "a", RelationType.PREREQUISITE),
        ]
        with self.assertRaises(ValidationError):
            KnowledgeGraph(concepts, edges)

    def test_part_of_cycle_is_rejected(self) -> None:
        concepts = [
            Concept("a", "A", "Concept A"),
            Concept("b", "B", "Concept B"),
        ]
        edges = [
            ConceptEdge("a", "b", RelationType.PART_OF),
            ConceptEdge("b", "a", RelationType.PART_OF),
        ]

        with self.assertRaises(ValidationError):
            KnowledgeGraph(concepts, edges)

    def test_learning_scope_closes_over_parts_of_prerequisite_containers(self) -> None:
        concepts = [
            Concept("foundation", "Foundation", "A topic container."),
            Concept("foundation_part", "Foundation part", "An assessable part."),
            Concept("target", "Target", "The requested target."),
        ]
        edges = [
            ConceptEdge("foundation_part", "foundation", RelationType.PART_OF),
            ConceptEdge("foundation", "target", RelationType.PREREQUISITE),
        ]
        graph = KnowledgeGraph(concepts, edges)

        self.assertEqual(
            graph.learning_scope("target"),
            {"target", "foundation", "foundation_part"},
        )

    def test_published_question_content_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "immutability.db")
            database.initialize()
            database.import_corpus(
                self.concepts, self.edges, self.misconceptions, self.sources, self.questions
            )
            question = next(
                item
                for item in self.questions
                if "generated" in item.provenance
            )
            first = question.options[0]
            changed_option = replace(first, rationale=first.rationale + " Mutated in place.")
            changed_question = replace(
                question,
                options=tuple(
                    changed_option if option.id == first.id else option for option in question.options
                ),
            )
            with self.assertRaises(ConflictError):
                database.import_corpus(
                    self.concepts,
                    self.edges,
                    self.misconceptions,
                    self.sources,
                    [
                        *(
                            item
                            for item in self.questions
                            if item.id != changed_question.id
                        ),
                        changed_question,
                    ],
                )

    def test_reimporting_active_identical_release_is_event_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "idempotent-import.db")
            database.initialize()
            first = database.import_corpus(
                self.concepts,
                self.edges,
                self.misconceptions,
                self.sources,
                self.questions,
                self.domains,
                self.topics,
            )
            with database.read() as connection:
                before = "\n".join(connection.iterdump())
                first_events = connection.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE event_type='CorpusImported'"""
                ).fetchone()["n"]

            repeated = database.import_corpus(
                self.concepts,
                self.edges,
                self.misconceptions,
                self.sources,
                self.questions,
                self.domains,
                self.topics,
            )

            with database.read() as connection:
                after = "\n".join(connection.iterdump())
                repeated_events = connection.execute(
                    """SELECT COUNT(*) AS n FROM events
                       WHERE event_type='CorpusImported'"""
                ).fetchone()["n"]
                release_count = connection.execute(
                    "SELECT COUNT(*) AS n FROM corpus_releases"
                ).fetchone()["n"]
            self.assertEqual(repeated["release_id"], first["release_id"])
            self.assertEqual(first_events, 1)
            self.assertEqual(repeated_events, first_events)
            self.assertEqual(release_count, 1)
            self.assertEqual(after, before)
            self.assertTrue(database.verify_integrity()["ok"])


if __name__ == "__main__":
    unittest.main()
