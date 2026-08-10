# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tsq.corpus import (
    corpus_source_digest,
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
    Question,
    QuestionStatus,
    RelationType,
)
from tsq.quality import audit_corpus, validate_question
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"

CLEAN_PROVENANCE_REVISIONS = {
    "q_transformer_token_mixing_001": "q_transformer_token_mixing_002",
    "q_causal_mask_matrix_001": "q_causal_mask_matrix_002",
    "q_transformer_position_signal_ablation_001": (
        "q_transformer_position_signal_ablation_002"
    ),
    "q_causal_mask_softmax_normalization_001": (
        "q_causal_mask_softmax_normalization_002"
    ),
    "q_causal_full_incremental_equivalence_001": (
        "q_causal_full_incremental_equivalence_002"
    ),
    "q_transformer_token_intervention_trace_001": (
        "q_transformer_token_intervention_trace_002"
    ),
    "q_causal_cross_attention_mask_scope_002": (
        "q_causal_cross_attention_mask_scope_003"
    ),
    "q_attention_runtime_workspace_boundary_001": (
        "q_attention_runtime_workspace_boundary_002"
    ),
    "q_transformer_unexpected_cross_token_path_001": (
        "q_transformer_unexpected_cross_token_path_002"
    ),
    "q_transformer_kv_cache_alignment_001": (
        "q_transformer_kv_cache_alignment_002"
    ),
    "q_transformer_kv_cache_eviction_equivalence_001": (
        "q_transformer_kv_cache_eviction_equivalence_002"
    ),
    "q_attention_duplicate_value_identifiability_002": (
        "q_attention_duplicate_value_identifiability_003"
    ),
    "q_attention_value_gradient_routing_001": (
        "q_attention_value_gradient_routing_002"
    ),
    "q_agent_delegated_capability_envelope_001": (
        "q_agent_delegated_capability_envelope_002"
    ),
    "q_agent_approval_argument_binding_001": (
        "q_agent_approval_argument_binding_002"
    ),
    "q_agent_saga_compensation_001": "q_agent_saga_compensation_002",
    "q_agent_snapshot_observation_completeness_001": (
        "q_agent_snapshot_observation_completeness_002"
    ),
    "q_agent_catalog_expansion_boundary_001": (
        "q_agent_catalog_expansion_boundary_002"
    ),
    "q_agent_granted_subset_match_001": (
        "q_agent_granted_subset_match_002"
    ),
    "q_agent_context_caveat_expiry_001": (
        "q_agent_context_caveat_expiry_002"
    ),
    "q_agent_dry_run_effect_boundary_001": (
        "q_agent_dry_run_effect_boundary_002"
    ),
    "q_agent_completion_predicate_counterfactual_001": (
        "q_agent_completion_predicate_counterfactual_002"
    ),
    "q_rag_causal_bridge_attribution_001": (
        "q_rag_causal_bridge_attribution_002"
    ),
    "q_rag_derived_rate_entailment_001": (
        "q_rag_derived_rate_entailment_002"
    ),
    "q_rag_candidate_reranker_funnel_001": (
        "q_rag_candidate_reranker_funnel_002"
    ),
    "q_rag_temporal_scope_counterfactual_001": (
        "q_rag_temporal_scope_counterfactual_002"
    ),
    "q_rag_claim_citation_alignment_revision_001": (
        "q_rag_claim_citation_alignment_revision_002"
    ),
    "q_rag_conjunctive_facet_coverage_001": (
        "q_rag_conjunctive_facet_coverage_002"
    ),
}

SOURCE_SCOPE_CORRECTION_REVISIONS = {
    "q_agent_delegated_capability_envelope_001": (
        "q_agent_delegated_capability_envelope_002"
    ),
    "q_agent_approval_argument_binding_001": (
        "q_agent_approval_argument_binding_002"
    ),
    "q_agent_saga_compensation_001": "q_agent_saga_compensation_002",
    "q_agent_snapshot_observation_completeness_001": (
        "q_agent_snapshot_observation_completeness_002"
    ),
    "q_rag_temporal_scope_counterfactual_001": (
        "q_rag_temporal_scope_counterfactual_002"
    ),
}

OBSOLETE_ACTIVATION_GATE_SUFFIX = (
    "; human review and activation of this generated item remain required."
)


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
        bundle = load_bundle(CORPUS)
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

    def test_llm_objective_dependent_practice_is_explicitly_labeled(
        self,
    ) -> None:
        new_objectives = {
            "c_autoregressive_language_modeling",
            "c_attention_scaling",
            "c_causal_masking",
            "c_rag_grounding",
            "c_rag_retrieval_quality",
            "c_agent_tool_use",
            "c_agent_observation_loop",
        }
        questions_by_id = {question.id: question for question in self.questions}
        for concept_id in new_objectives:
            items = [
                question
                for question in self.questions
                if question.primary_concept_id == concept_id
            ]
            root_items = [
                question
                for question in items
                if question.revision_of is None
                and question.status.eligible_for_adaptation
            ]
            self.assertGreaterEqual(len(root_items), 3, concept_id)
            roots_by_family: dict[str, list[Question]] = {}
            for item in root_items:
                roots_by_family.setdefault(
                    item.published_family_id or item.family_id,
                    [],
                ).append(item)
            for family_id, family_items in roots_by_family.items():
                if len(family_items) == 1:
                    continue
                self.assertTrue(
                    all(
                        item.status.value == "approved"
                        and item.status.eligible_for_adaptation
                        for item in family_items
                    ),
                    f"{concept_id}:{family_id}",
                )
                self.assertTrue(
                    all(
                        item.provenance.get("human_review") is False
                        for item in family_items
                        if item.provenance.get("generated") is True
                    ),
                    f"{concept_id}:{family_id}",
                )
                self.assertGreaterEqual(
                    sum(
                        bool(item.provenance.get("independence_note"))
                        for item in family_items
                    ),
                    len(family_items) - 1,
                    f"{concept_id}:{family_id}",
                )
            for item in items:
                if item.revision_of is None:
                    continue
                parent = questions_by_id[item.revision_of]
                self.assertEqual(
                    item.published_family_id,
                    parent.published_family_id,
                )
                self.assertGreater(item.version, parent.version)
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

    def test_curriculum_has_no_blocking_deterministic_quality_issues(self) -> None:
        errors = [issue for issue in audit_corpus(self.questions) if issue.severity == "error"]
        self.assertEqual(errors, [])
        self.assertGreaterEqual(len(self.questions), 20)
        self.assertTrue(all(len(question.misconception_ids) == 3 for question in self.questions))

    def test_current_curriculum_is_approved_or_explicitly_superseded(
        self,
    ) -> None:
        status_counts = Counter(
            question.status.value for question in self.questions
        )
        self.assertEqual(set(status_counts), {"approved", "retired"})
        self.assertEqual(len(self.questions), 532)
        self.assertEqual(status_counts["approved"], 491)
        self.assertEqual(status_counts["retired"], 41)

        retired_ids = {
            question.id
            for question in self.questions
            if question.status is QuestionStatus.RETIRED
        }
        children_by_parent: dict[str, list[Question]] = {}
        for question in self.questions:
            if question.revision_of is not None:
                children_by_parent.setdefault(question.revision_of, []).append(
                    question
                )

        def has_approved_descendant(question_id: str) -> bool:
            children = children_by_parent.get(question_id, [])
            return any(
                child.status is QuestionStatus.APPROVED
                or has_approved_descendant(child.id)
                for child in children
            )

        self.assertTrue(
            all(has_approved_descendant(question_id) for question_id in retired_ids)
        )

    def test_clean_provenance_revisions_preserve_substantive_payload(self) -> None:
        bundle = load_bundle(CORPUS)
        by_id = {question["id"]: question for question in bundle["questions"]}
        identity_fields = {"id", "version", "revision_of", "status", "provenance"}
        changed_provenance_fields = {
            "activation",
            "human_review_status",
            "review_status",
            "review_statement",
            "revision_review_batch_id",
            "revision_reviewed_on",
        }

        self.assertEqual(len(CLEAN_PROVENANCE_REVISIONS), 28)
        self.assertEqual(len(SOURCE_SCOPE_CORRECTION_REVISIONS), 5)
        self.assertEqual(
            SOURCE_SCOPE_CORRECTION_REVISIONS,
            {
                parent_id: CLEAN_PROVENANCE_REVISIONS[parent_id]
                for parent_id in SOURCE_SCOPE_CORRECTION_REVISIONS
            },
        )
        for parent_id, child_id in CLEAN_PROVENANCE_REVISIONS.items():
            with self.subTest(parent_id=parent_id, child_id=child_id):
                parent = by_id[parent_id]
                child = by_id[child_id]
                self.assertEqual(parent["status"], "retired")
                self.assertEqual(child["status"], "approved")
                self.assertEqual(child["revision_of"], parent_id)
                self.assertEqual(child["version"], parent["version"] + 1)
                self.assertEqual(
                    {
                        key: value
                        for key, value in child.items()
                        if key not in identity_fields
                    },
                    {
                        key: value
                        for key, value in parent.items()
                        if key not in identity_fields
                    },
                )

                parent_provenance = parent["provenance"]
                child_provenance = child["provenance"]
                self.assertIs(child_provenance["generated"], True)
                self.assertIs(child_provenance["human_review"], False)
                self.assertEqual(
                    child_provenance["review_status"],
                    "independent_model_review_passed",
                )
                self.assertNotIn("activation", child_provenance)
                self.assertNotIn("human_review_status", child_provenance)
                self.assertIn("answer-redacted", child_provenance["review_statement"])
                self.assertIn("keyed answer", child_provenance["review_statement"])
                for key, value in parent_provenance.items():
                    if (
                        key == "source_scope"
                        and parent_id in SOURCE_SCOPE_CORRECTION_REVISIONS
                    ):
                        self.assertTrue(
                            value.endswith(OBSOLETE_ACTIVATION_GATE_SUFFIX)
                        )
                        self.assertEqual(
                            child_provenance[key],
                            value.removesuffix(
                                OBSOLETE_ACTIVATION_GATE_SUFFIX
                            )
                            + ".",
                        )
                        continue
                    if key not in changed_provenance_fields:
                        self.assertEqual(child_provenance.get(key), value, key)

    def test_active_provenance_free_text_has_no_obsolete_activation_gate(
        self,
    ) -> None:
        free_text_fields = {
            "derivation",
            "independence_note",
            "review_statement",
            "revision_reason",
            "source_scope",
        }
        obsolete_markers = {
            "activation of this generated item remain required",
            "human review and activation of this generated item remain required",
            "human review remains required",
            "manual only after human review and new immutable release",
            "pending quarantined",
            "required before activation",
        }
        for question in self.questions:
            if question.status is not QuestionStatus.APPROVED:
                continue
            for field in free_text_fields:
                value = question.provenance.get(field)
                if not isinstance(value, str):
                    continue
                normalized = value.casefold().replace("_", " ").replace("-", " ")
                for marker in obsolete_markers:
                    self.assertNotIn(
                        marker,
                        normalized,
                        f"{question.id}:provenance.{field}",
                    )

    def test_legacy_review_promotions_have_accepted_independent_review(self) -> None:
        promoted_ids = {
            "q_transformer_token_mixing_002",
            "q_causal_mask_matrix_002",
            "q_transformer_position_signal_ablation_002",
            "q_causal_mask_softmax_normalization_002",
            "q_causal_full_incremental_equivalence_002",
            "q_transformer_token_intervention_trace_002",
        }
        promoted = {
            question.id: question
            for question in self.questions
            if question.id in promoted_ids
        }
        self.assertEqual(set(promoted), promoted_ids)
        for question_id, question in promoted.items():
            with self.subTest(question_id=question_id):
                self.assertTrue(question.status.eligible_for_adaptation)
                self.assertIs(question.provenance.get("generated"), True)
                self.assertIs(question.provenance.get("human_review"), False)
                self.assertEqual(
                    question.provenance.get("review_status"),
                    "independent_model_review_passed",
                )
                self.assertTrue(question.provenance.get("source_scope"))
                self.assertTrue(question.provenance.get("independence_note"))
                self.assertIn(
                    "answer-redacted",
                    question.provenance.get("review_statement", ""),
                )
                self.assertIn(
                    "keyed answer",
                    question.provenance.get("review_statement", ""),
                )
                self.assertNotIn("activation", question.provenance)
                self.assertNotIn("human_review_status", question.provenance)

    def test_active_review_provenance_has_no_pending_status(self) -> None:
        for question in self.questions:
            if not question.status.eligible_for_adaptation:
                continue
            review_status = str(question.provenance.get("review_status", "")).lower()
            self.assertNotIn("pending", review_status, question.id)
            self.assertNotIn("quarantin", review_status, question.id)

    def test_expansion_provenance_has_no_manual_activation_condition(self) -> None:
        expansion = [
            question
            for question in self.questions
            if "20260809" in str(question.provenance.get("batch_id", ""))
            and question.provenance.get("method") == "ai_assisted_source_scoped"
        ]
        self.assertEqual(len(expansion), 192)
        for question in expansion:
            self.assertNotIn("activation", question.provenance, question.id)
            self.assertNotIn("human_review_status", question.provenance, question.id)
            self.assertEqual(
                question.provenance.get("review_status"),
                "independent_model_review_passed",
                question.id,
            )

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
        bundle = load_bundle(CORPUS)
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

    def test_sharded_directory_and_manifest_are_one_deterministic_corpus(self) -> None:
        directory_bundle = load_bundle(CORPUS)
        manifest_bundle = load_bundle(CORPUS / "manifest.json")

        self.assertEqual(directory_bundle, manifest_bundle)
        self.assertEqual(
            corpus_source_digest(CORPUS),
            corpus_source_digest(CORPUS / "manifest.json"),
        )
        self.assertEqual(len(directory_bundle["topics"]), 16)
        self.assertGreaterEqual(len(directory_bundle["questions"]), 480)

        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.json"
            encoded = json.dumps(
                directory_bundle,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            legacy.write_bytes(encoded)
            self.assertEqual(
                corpus_source_digest(legacy),
                hashlib.sha256(encoded).hexdigest(),
            )
            self.assertEqual(load_bundle(legacy), directory_bundle)

    def test_sharding_preserves_the_immutable_release_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            sharded = Database(Path(directory) / "sharded.db")
            sharded.initialize()
            sharded_release = sharded.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )["release_id"]

            bundle = load_bundle(CORPUS)
            legacy_path = Path(directory) / "legacy.json"
            legacy_path.write_text(
                json.dumps(bundle, sort_keys=True),
                encoding="utf-8",
            )
            legacy = Database(Path(directory) / "legacy.db")
            legacy.initialize()
            legacy_release = legacy.import_corpus(
                *read_and_parse(legacy_path, include_catalog=True)
            )["release_id"]

        self.assertEqual(sharded_release, legacy_release)

    def test_sharded_loader_rejects_path_and_inventory_ambiguity(self) -> None:
        cases = (
            ("traversal", "../outside.json", "canonical relative"),
            ("absolute", "/tmp/outside.json", "canonical relative"),
            (
                "noncanonical",
                "topics/../topics/machine_learning.json",
                "canonical relative",
            ),
        )
        for case, replacement, expected in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                copied = Path(directory) / "corpus"
                shutil.copytree(CORPUS, copied)
                manifest_path = copied / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["topic_files"][0]["path"] = replacement
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                with self.assertRaisesRegex(ValidationError, expected):
                    load_bundle(copied)

        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "corpus"
            shutil.copytree(CORPUS, copied)
            (copied / "topics" / "unlisted.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValidationError, "unlisted"):
                load_bundle(copied)

    def test_sharded_loader_rejects_duplicate_ids_paths_and_wrong_order(self) -> None:
        mutations = (
            (
                "topic id",
                lambda manifest: manifest["topic_files"][1].update(
                    topic_id=manifest["topic_files"][0]["topic_id"]
                ),
                "duplicate topic IDs",
            ),
            (
                "path",
                lambda manifest: manifest["topic_files"][1].update(
                    path=manifest["topic_files"][0]["path"]
                ),
                "duplicate or case-colliding paths",
            ),
            (
                "order",
                lambda manifest: manifest["topic_files"].__setitem__(
                    slice(0, 2), reversed(manifest["topic_files"][:2])
                ),
                "canonical catalog order",
            ),
        )
        for case, mutate, expected in mutations:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                copied = Path(directory) / "corpus"
                shutil.copytree(CORPUS, copied)
                manifest_path = copied / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                with self.assertRaisesRegex(ValidationError, expected):
                    load_bundle(copied)

    def test_sharded_format_requires_the_current_corpus_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "corpus"
            shutil.copytree(CORPUS, copied)
            manifest_path = copied / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 2
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(
                ValidationError,
                "format_version 1 requires corpus schema_version 3",
            ):
                load_bundle(copied)

    def test_sharded_loader_rejects_topic_content_in_the_wrong_file(self) -> None:
        mutations = (
            (
                "concept inventory",
                "machine_learning_foundations.json",
                lambda shard: shard["concepts"].pop(),
                "concept ownership",
            ),
            (
                "objective owner",
                "transformers.json",
                lambda shard: shard["learning_objectives"][0].update(
                    primary_concept_id="c_agent_tool_use"
                ),
                "not topic",
            ),
            (
                "misconception owner",
                "machine_learning_foundations.json",
                lambda shard: shard["misconceptions"][0].update(
                    concept_id="c_transformers"
                ),
                "not topic",
            ),
            (
                "question owner",
                "machine_learning_foundations.json",
                lambda shard: next(
                    mapping
                    for mapping in shard["questions"][0]["concepts"]
                    if mapping["role"] == "primary"
                ).update(concept_id="c_transformers"),
                "not owned by topic",
            ),
        )
        for case, filename, mutate, expected in mutations:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                copied = Path(directory) / "corpus"
                shutil.copytree(CORPUS, copied)
                shard_path = copied / "topics" / filename
                shard = json.loads(shard_path.read_text(encoding="utf-8"))
                mutate(shard)
                shard_path.write_text(json.dumps(shard), encoding="utf-8")

                with self.assertRaisesRegex(ValidationError, expected):
                    load_bundle(copied)

    def test_semantic_bundle_validation_aggregates_reference_issues(self) -> None:
        bundle = load_bundle(CORPUS)
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
        bundle = load_bundle(CORPUS)
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
        baseline = load_bundle(CORPUS)
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
