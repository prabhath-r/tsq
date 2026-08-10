# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from collections import Counter
import unittest
from pathlib import Path

from tsq.corpus import corpus_source_digest, read_and_parse


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CORPUS = ROOT / "corpus"
PACKAGED_CORPUS = ROOT / "src" / "tsq" / "data" / "curriculum"

NEW_AGENT_INTRO_QUESTIONS = {
    "q_agent_catalog_expansion_boundary_002",
    "q_agent_granted_subset_match_002",
    "q_agent_context_caveat_expiry_002",
    "q_agent_dry_run_effect_boundary_002",
    "q_agent_completion_predicate_counterfactual_002",
}
NEW_RAG_INTRO_QUESTIONS = {
    "q_rag_claim_citation_alignment_revision_002",
    "q_rag_conjunctive_facet_coverage_002",
}
NEW_TRANSFORMER_INTRO_QUESTIONS = {
    "q_attention_runtime_workspace_boundary_002",
    "q_transformer_unexpected_cross_token_path_002",
}
NEW_TRANSFORMER_CAPACITY_QUESTIONS = {
    "q_transformer_kv_cache_alignment_002",
    "q_transformer_kv_cache_eviction_equivalence_002",
    "q_attention_duplicate_value_identifiability_003",
    "q_attention_value_gradient_routing_002",
}
AGENT_INTRO_BATCH_ID = "batch_agent_intro_bridges_20260724_a"
RAG_INTRO_BATCH_ID = "batch_rag_intro_bridges_20260724_a"
TRANSFORMER_INTRO_BATCH_ID = "batch_transformer_intro_bridges_20260724_a"
TRANSFORMER_CAPACITY_BATCH_ID = (
    "batch_transformer_capacity_repairs_20260725_a"
)
CURRENT_CANDIDATE_BATCH_COUNTS = {
    "batch_agent_authorization_boundaries_20260809_a": 12,
    "batch_agent_authorization_controls_20260809_b": 12,
    "batch_agent_reconciliation_diagnostics_20260809_b": 12,
    "batch_agent_reconciliation_state_20260809_a": 12,
    "batch_language_modeling_diagnostics_20260809_b": 12,
    "batch_language_modeling_foundations_20260809_a": 12,
    "batch_rag_entailment_coverage_20260809_d": 12,
    "batch_rag_evidence_reconciliation_20260809_b": 12,
    "batch_rag_grounding_claim_checks_20260809_a": 12,
    "batch_rag_retrieval_pipeline_20260809_c": 12,
    "batch_transformer_causality_scaling_20260809_e": 12,
    "batch_transformer_composition_causality_20260809_d": 12,
    "batch_transformer_order_resources_20260809_b": 12,
    "batch_transformer_resources_paths_20260809_c": 12,
    "batch_transformer_routing_order_20260809_a": 12,
    "batch_transformer_scaling_cache_20260809_f": 12,
}
REVIEWED_RELEASE_BATCH_COUNTS = {
    "batch_corpus_release_20260809_a": 13,
    "batch_transformer_serviceability_20260809_a": 10,
}
PUBLIC_IDENTITY_PROVENANCE_FIELDS = frozenset(
    {"provider", "model", "generator", "provider_name", "model_name"}
)


class CorpusMisconceptionRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        canonical = read_and_parse(CANONICAL_CORPUS)
        packaged = read_and_parse(PACKAGED_CORPUS)
        cls.canonical_misconceptions = {
            item.id: item for item in canonical[2]
        }
        cls.canonical_sources = {
            item.id: item for item in canonical[3]
        }
        cls.canonical_questions = {
            item.id: item for item in canonical[4]
        }
        cls.packaged_questions = {
            item.id: item for item in packaged[4]
        }

    def test_corpus_copies_remain_byte_identical(self) -> None:
        self.assertEqual(
            corpus_source_digest(CANONICAL_CORPUS),
            corpus_source_digest(PACKAGED_CORPUS),
        )

    def test_transformer_intervention_routes_mask_claim_to_visibility(self) -> None:
        question = self.canonical_questions[
            "q_transformer_token_intervention_trace_002"
        ]
        distractor = next(option for option in question.options if option.id == "d")

        self.assertEqual(question.status.value, "approved")
        self.assertTrue(question.status.eligible_for_adaptation)
        self.assertIn("mask that blocks that direction", distractor.text)
        self.assertEqual(
            distractor.misconception_id,
            "m_transformers_are_inherently_bidirectional",
        )
        self.assertEqual(
            distractor.diagnostic_objective_id,
            "lo_causal_visibility",
        )

    def test_causal_cross_attention_revision_supersedes_its_parent(
        self,
    ) -> None:
        expected_routes = {
            "m_mask_only_inference",
            "m_mask_hides_current_token",
            "m_causal_mask_truncates_encoder_memory",
        }
        for questions in (
            self.canonical_questions,
            self.packaged_questions,
        ):
            original = questions[
                "q_causal_cross_attention_mask_scope_001"
            ]
            parent = questions[
                "q_causal_cross_attention_mask_scope_002"
            ]
            revision = questions[
                "q_causal_cross_attention_mask_scope_003"
            ]
            self.assertEqual(revision.revision_of, parent.id)
            self.assertGreater(revision.version, parent.version)
            self.assertEqual(parent.revision_of, original.id)
            self.assertGreater(parent.version, original.version)
            self.assertEqual(original.status.value, "retired")
            self.assertEqual(parent.status.value, "retired")
            self.assertEqual(revision.family_id, parent.family_id)
            self.assertEqual(
                revision.family_id,
                "f_causal_cross_attention_mask_scope",
            )
            self.assertEqual(
                revision.objective_id,
                "lo_causal_visibility",
            )
            self.assertEqual(revision.status.value, "approved")
            self.assertTrue(revision.status.eligible_for_adaptation)
            self.assertEqual(
                {
                    option.misconception_id
                    for option in revision.options
                    if not option.correct
                },
                expected_routes,
            )
            self.assertTrue(
                all(
                    option.diagnostic_objective_id
                    == "lo_causal_visibility"
                    for option in revision.options
                    if not option.correct
                )
            )
            correct = revision.correct_option
            self.assertIn("{t0, t1}", correct.text)
            self.assertIn("{s0, s1, s2}", correct.text)
            cross_scope = next(
                option
                for option in revision.options
                if option.misconception_id
                == "m_causal_mask_truncates_encoder_memory"
            )
            self.assertIn("{s0, s1}", cross_scope.text)
            misconception = self.canonical_misconceptions[
                "m_causal_mask_truncates_encoder_memory"
            ]
            self.assertEqual(
                misconception.concept_id,
                "c_causal_masking",
            )
            self.assertIn(
                "encoder cross-attention",
                misconception.description,
            )
            self.assertEqual(
                revision.source_ids,
                (
                    "src_vaswani_attention_2017",
                    "src_expert_synthesis_2026",
                ),
            )
            self.assertIs(revision.provenance["generated"], True)
            self.assertIs(revision.provenance["human_review"], False)
            self.assertEqual(
                revision.provenance["batch_id"],
                "batch_causal_revision_20260724_a",
            )
            self.assertIn(
                original.id,
                revision.provenance["independence_note"],
            )

    def test_agent_capability_availability_has_exact_named_route(self) -> None:
        question = self.canonical_questions[
            "q_agent_approval_argument_binding_002"
        ]
        distractor = next(option for option in question.options if option.id == "d")
        misconception = self.canonical_misconceptions[
            "m_agent_tool_availability_is_authorization"
        ]

        self.assertEqual(question.status.value, "approved")
        self.assertTrue(question.status.eligible_for_adaptation)
        self.assertEqual(
            distractor.misconception_id,
            misconception.id,
        )
        self.assertEqual(
            distractor.diagnostic_objective_id,
            "lo_agent_tool_authorization",
        )
        self.assertEqual(misconception.concept_id, "c_agent_tool_use")
        self.assertIn("permission to execute", misconception.description)


    def test_agent_intro_questions_are_approved_and_exactly_routed(
        self,
    ) -> None:
        for questions in (
            self.canonical_questions,
            self.packaged_questions,
        ):
            batch = {
                question.id: question
                for question in questions.values()
                if question.provenance.get("batch_id") == AGENT_INTRO_BATCH_ID
                and question.status.eligible_for_adaptation
            }
            self.assertEqual(set(batch), NEW_AGENT_INTRO_QUESTIONS)
            self.assertEqual(
                len({question.family_id for question in batch.values()}),
                5,
            )
            self.assertTrue(
                all(
                    question.status.value == "approved"
                    and question.status.eligible_for_adaptation
                    and question.difficulty < -0.3
                    and question.provenance.get("generated") is True
                    and question.provenance.get("human_review") is False
                    and bool(question.provenance.get("source_scope"))
                    for question in batch.values()
                )
            )
            authorization = [
                question
                for question in batch.values()
                if question.objective_id == "lo_agent_tool_authorization"
            ]
            reconciliation = [
                question
                for question in batch.values()
                if question.objective_id == "lo_agent_state_reconciliation"
            ]
            self.assertEqual(len(authorization), 3)
            self.assertEqual(len(reconciliation), 2)
            self.assertTrue(
                all(
                    "src_rfc9396_2023" not in question.source_ids
                    and any(
                        option.misconception_id
                        == "m_agent_tool_availability_is_authorization"
                        and option.diagnostic_objective_id
                        == "lo_agent_tool_authorization"
                        for option in question.options
                    )
                    for question in authorization
                )
            )
            subset = batch["q_agent_granted_subset_match_002"]
            self.assertEqual(
                next(
                    option for option in subset.options if option.id == "a"
                ).misconception_id,
                "m_agent_requested_authority_is_granted",
            )
            self.assertEqual(
                next(
                    option for option in subset.options if option.id == "c"
                ).misconception_id,
                "m_agent_tool_availability_is_authorization",
            )
            self.assertEqual(
                self.canonical_misconceptions[
                    "m_agent_requested_authority_is_granted"
                ].concept_id,
                "c_agent_tool_use",
            )
            completion = batch[
                "q_agent_completion_predicate_counterfactual_002"
            ]
            self.assertEqual(
                next(
                    option for option in completion.options if option.id == "c"
                ).misconception_id,
                "m_agent_any_required_condition_suffices",
            )
            self.assertEqual(
                next(
                    option for option in completion.options if option.id == "d"
                ).misconception_id,
                "m_agent_observation_equals_success",
            )
            self.assertEqual(
                self.canonical_misconceptions[
                    "m_agent_any_required_condition_suffices"
                ].concept_id,
                "c_agent_observation_loop",
            )


    def test_public_question_provenance_omits_model_identity(self) -> None:
        for questions in (
            self.canonical_questions,
            self.packaged_questions,
        ):
            self.assertTrue(
                all(
                    PUBLIC_IDENTITY_PROVENANCE_FIELDS.isdisjoint(
                        question.provenance
                    )
                    for question in questions.values()
                )
            )

    def test_rag_intro_questions_are_approved_and_independently_routed(
        self,
    ) -> None:
        expected_routes = {
            "q_rag_claim_citation_alignment_revision_002": {
                "m_rag_citation_proves_entailment",
                "m_rag_context_guarantees_use",
                "m_rag_retrieval_updates_weights",
            },
            "q_rag_conjunctive_facet_coverage_002": {
                "m_rag_more_context_monotonic",
                "m_rag_recall_alone_sufficient",
                "m_rag_top_score_is_truth",
            },
        }
        for questions in (
            self.canonical_questions,
            self.packaged_questions,
        ):
            batch = {
                question.id: question
                for question in questions.values()
                if question.provenance.get("batch_id") == RAG_INTRO_BATCH_ID
                and question.status.eligible_for_adaptation
            }
            self.assertEqual(set(batch), NEW_RAG_INTRO_QUESTIONS)
            self.assertEqual(
                len({question.family_id for question in batch.values()}),
                2,
            )
            self.assertTrue(
                all(
                    question.status.value == "approved"
                    and question.status.eligible_for_adaptation
                    and question.difficulty < -0.5
                    and question.provenance.get("generated") is True
                    and question.provenance.get("human_review") is False
                    and bool(question.provenance.get("source_scope"))
                    for question in batch.values()
                )
            )
            for question_id, routes in expected_routes.items():
                question = batch[question_id]
                self.assertEqual(
                    {
                        option.misconception_id
                        for option in question.options
                        if not option.correct
                    },
                    routes,
                )
                self.assertTrue(
                    all(
                        option.diagnostic_objective_id
                        == question.objective_id
                        for option in question.options
                        if not option.correct
                    )
                )
            grounding = batch[
                "q_rag_claim_citation_alignment_revision_002"
            ]
            retrieval = batch["q_rag_conjunctive_facet_coverage_002"]
            self.assertEqual(
                grounding.objective_id,
                "lo_rag_claim_grounding",
            )
            self.assertEqual(
                retrieval.objective_id,
                "lo_rag_retrieval_evidence_quality",
            )
            self.assertEqual(grounding.kind.value, "application")
            self.assertEqual(retrieval.kind.value, "diagnostic")
            self.assertIn("src_alce_2023", grounding.source_ids)
            self.assertNotIn("src_ircot_2023", retrieval.source_ids)
            self.assertNotIn("src_mrag_2025", retrieval.source_ids)

    def test_transformer_intro_questions_are_approved_and_exactly_routed(
        self,
    ) -> None:
        expected_routes = {
            "q_attention_runtime_workspace_boundary_002": {
                "m_full_attention_is_linear_in_length": (
                    "lo_attention_resource_scaling"
                ),
                "m_parameter_count_scales_with_tokens": (
                    "lo_attention_resource_scaling"
                ),
                "m_quadratic_attention_only_during_training": (
                    "lo_attention_resource_scaling"
                ),
            },
            "q_transformer_unexpected_cross_token_path_002": {
                "m_feedforward_layers_mix_token_positions": (
                    "lo_transformer_information_paths"
                ),
                "m_residuals_make_sublayer_order_irrelevant": (
                    "lo_transformer_sublayer_composition"
                ),
                "m_transformers_are_inherently_bidirectional": (
                    "lo_causal_visibility"
                ),
            },
        }
        for questions in (
            self.canonical_questions,
            self.packaged_questions,
        ):
            batch = {
                question.id: question
                for question in questions.values()
                if question.provenance.get("batch_id")
                == TRANSFORMER_INTRO_BATCH_ID
                and question.status.eligible_for_adaptation
            }
            self.assertEqual(set(batch), NEW_TRANSFORMER_INTRO_QUESTIONS)
            self.assertEqual(
                len({question.family_id for question in batch.values()}),
                2,
            )
            self.assertEqual(
                {
                    question_id: question.family_id
                    for question_id, question in batch.items()
                },
                {
                    "q_attention_runtime_workspace_boundary_002": (
                        "f_transformer_sequence_shape_audit"
                    ),
                    "q_transformer_unexpected_cross_token_path_002": (
                        "f_transformer_removed_attention_audit"
                    ),
                },
            )
            self.assertEqual(
                {question.kind.value for question in batch.values()},
                {"diagnostic"},
            )
            self.assertEqual(
                {question.objective_id for question in batch.values()},
                {
                    "lo_attention_resource_scaling",
                    "lo_transformer_information_paths",
                },
            )
            resource = batch[
                "q_attention_runtime_workspace_boundary_002"
            ]
            path = batch[
                "q_transformer_unexpected_cross_token_path_002"
            ]
            self.assertEqual(
                resource.source_ids,
                (
                    "src_vaswani_attention_2017",
                    "src_goodfellow_dl_2016",
                    "src_expert_synthesis_2026",
                ),
            )
            self.assertEqual(
                path.source_ids,
                (
                    "src_vaswani_attention_2017",
                    "src_expert_synthesis_2026",
                ),
            )
            self.assertIn(
                "q_transformer_sequence_shape_audit_001",
                resource.provenance["independence_note"],
            )
            self.assertIn(
                "q_transformer_token_mixing_001",
                path.provenance["independence_note"],
            )
            self.assertTrue(
                all(
                    question.status.value == "approved"
                    and question.status.eligible_for_adaptation
                    and question.difficulty < -0.5
                    and question.provenance.get("generated") is True
                    and question.provenance.get("human_review") is False
                    and bool(question.provenance.get("source_scope"))
                    and bool(question.provenance.get("independence_note"))
                    and question.provenance.get("psychometrics")
                    == "uncalibrated_author_prior"
                    for question in batch.values()
                )
            )
            for question_id, routes in expected_routes.items():
                observed = {
                    option.misconception_id: option.diagnostic_objective_id
                    for option in batch[question_id].options
                    if not option.correct
                }
                self.assertEqual(observed, routes)

    def test_transformer_capacity_questions_are_distinct_and_approved(
        self,
    ) -> None:
        expected_routes = {
            "q_transformer_kv_cache_alignment_002": {
                "m_decode_cache_retains_queries": (
                    "lo_incremental_kv_cache"
                ),
                "m_multiquery_shares_queries": (
                    "lo_incremental_kv_cache"
                ),
                "m_kv_sharing_collapses_time": (
                    "lo_incremental_kv_cache"
                ),
            },
            "q_transformer_kv_cache_eviction_equivalence_002": {
                "m_multiquery_shares_queries": (
                    "lo_incremental_kv_cache"
                ),
                "m_kv_sharing_collapses_time": (
                    "lo_incremental_kv_cache"
                ),
                "m_decode_cache_retains_queries": (
                    "lo_incremental_kv_cache"
                ),
            },
            "q_attention_duplicate_value_identifiability_003": {
                "m_attention_values_determine_weights": (
                    "lo_attention_value_routing"
                ),
                "m_attention_keys_are_output_payloads": (
                    "lo_attention_value_routing"
                ),
                "m_attention_is_hard_selection": (
                    "lo_attention_value_routing"
                ),
            },
            "q_attention_value_gradient_routing_002": {
                "m_attention_values_determine_weights": (
                    "lo_attention_value_routing"
                ),
                "m_attention_keys_are_output_payloads": (
                    "lo_attention_value_routing"
                ),
                "m_attention_is_hard_selection": (
                    "lo_attention_value_routing"
                ),
            },
        }
        for questions in (
            self.canonical_questions,
            self.packaged_questions,
        ):
            batch = {
                question.id: question
                for question in questions.values()
                if question.provenance.get("batch_id")
                == TRANSFORMER_CAPACITY_BATCH_ID
                and question.status.eligible_for_adaptation
            }
            self.assertEqual(
                set(batch),
                NEW_TRANSFORMER_CAPACITY_QUESTIONS,
            )
            self.assertEqual(
                len({question.family_id for question in batch.values()}),
                4,
            )
            self.assertEqual(
                {
                    question.objective_id for question in batch.values()
                },
                {
                    "lo_incremental_kv_cache",
                    "lo_attention_value_routing",
                },
            )
            self.assertTrue(
                all(
                    question.status.value == "approved"
                    and question.status.eligible_for_adaptation
                    and question.provenance.get("generated") is True
                    and question.provenance.get("human_review") is False
                    and bool(question.provenance.get("source_scope"))
                    and bool(question.provenance.get("independence_note"))
                    for question in batch.values()
                )
            )
            for question_id, routes in expected_routes.items():
                observed = {
                    option.misconception_id: option.diagnostic_objective_id
                    for option in batch[question_id].options
                    if not option.correct
                }
                self.assertEqual(observed, routes)

            alignment = batch[
                "q_transformer_kv_cache_alignment_002"
            ]
            eviction = batch[
                "q_transformer_kv_cache_eviction_equivalence_002"
            ]
            self.assertEqual(
                alignment.source_ids,
                (
                    "src_vaswani_attention_2017",
                    "src_shazeer_fast_decoding_2019",
                    "src_expert_synthesis_2026",
                ),
            )
            self.assertEqual(eviction.source_ids, alignment.source_ids)
            self.assertIn(
                "same temporal ordering",
                alignment.correct_option.text,
            )
            self.assertIn(
                "changing the visible context",
                eviction.correct_option.text,
            )

            parent = questions[
                "q_attention_duplicate_value_identifiability_001"
            ]
            provenance_parent = questions[
                "q_attention_duplicate_value_identifiability_002"
            ]
            revision = batch[
                "q_attention_duplicate_value_identifiability_003"
            ]
            self.assertEqual(revision.revision_of, provenance_parent.id)
            self.assertEqual(revision.version, 3)
            self.assertEqual(provenance_parent.revision_of, parent.id)
            self.assertGreater(revision.version, provenance_parent.version)
            self.assertEqual(provenance_parent.status.value, "retired")
            self.assertEqual(revision.family_id, parent.family_id)
            self.assertNotIn("0.0", revision.stem)
            self.assertIn("[0.8, 0.1, 0.1]", revision.stem)
            self.assertIn("[0.1, 0.8, 0.1]", revision.stem)
            self.assertIn(
                "0.9u + 0.1w",
                revision.correct_option.text,
            )
            self.assertIn(
                "src_brunner_identifiability_2020",
                revision.source_ids,
            )

            gradient = batch[
                "q_attention_value_gradient_routing_002"
            ]
            self.assertIn(
                "0.6g1 + 0.2g2",
                gradient.correct_option.text,
            )
            self.assertIn(
                "0.4g1 + 0.8g2",
                gradient.correct_option.text,
            )
            self.assertIn(
                "A-transpose",
                gradient.correct_option.text,
            )

        source = self.canonical_sources[
            "src_brunner_identifiability_2020"
        ]
        self.assertEqual(source.title, "On Identifiability in Transformers")
        self.assertEqual(
            source.uri,
            "https://arxiv.org/abs/1908.04211",
        )


    def test_new_curriculum_batches_are_reviewed_balanced_and_routed(
        self,
    ) -> None:
        expected_batches = set(CURRENT_CANDIDATE_BATCH_COUNTS)
        for questions in (
            self.canonical_questions,
            self.packaged_questions,
        ):
            batches = {
                batch_id: [
                    question
                    for question in questions.values()
                    if question.provenance.get("batch_id") == batch_id
                ]
                for batch_id in expected_batches
            }
            observed_current_batches = {
                str(question.provenance.get("batch_id"))
                for question in questions.values()
                if "20260809"
                in str(question.provenance.get("batch_id", ""))
            }
            self.assertEqual(
                observed_current_batches,
                expected_batches | set(REVIEWED_RELEASE_BATCH_COUNTS),
            )

            for batch_id, batch in batches.items():
                with self.subTest(batch_id=batch_id):
                    self.assertEqual(
                        len(batch),
                        CURRENT_CANDIDATE_BATCH_COUNTS[batch_id],
                    )
                    correct_positions = Counter(
                        next(
                            index
                            for index, option in enumerate(question.options)
                            if option.correct
                        )
                        for question in batch
                    )
                    self.assertEqual(
                        correct_positions,
                        Counter({0: 3, 1: 3, 2: 3, 3: 3}),
                    )
                    self.assertLessEqual(
                        min(question.difficulty for question in batch),
                        -0.3,
                    )
                    self.assertGreaterEqual(
                        max(question.difficulty for question in batch),
                        0.75,
                    )
                    for question in batch:
                        self.assertIn(
                            question.status.value,
                            {"approved", "retired"},
                        )
                        if question.status.value == "retired":
                            replacements = [
                                candidate
                                for candidate in questions.values()
                                if candidate.revision_of == question.id
                                and candidate.status.eligible_for_adaptation
                            ]
                            self.assertEqual(len(replacements), 1)
                        else:
                            self.assertTrue(
                                question.status.eligible_for_adaptation
                            )
                        self.assertIs(
                            question.provenance.get("generated"),
                            True,
                        )
                        self.assertTrue(
                            PUBLIC_IDENTITY_PROVENANCE_FIELDS.isdisjoint(
                                question.provenance
                            ),
                        )
                        self.assertIs(
                            question.provenance.get("human_review"),
                            False,
                        )
                        self.assertEqual(
                            question.provenance.get("psychometrics"),
                            "uncalibrated_author_prior",
                        )
                        self.assertTrue(
                            question.provenance.get("source_scope")
                        )
                        self.assertTrue(
                            question.provenance.get("independence_note")
                        )
                        self.assertEqual(len(question.options), 4)
                        routes = [
                            option.misconception_id
                            for option in question.options
                            if not option.correct
                        ]
                        self.assertEqual(len(routes), 3)
                        self.assertNotIn(None, routes)
                        self.assertEqual(len(set(routes)), 3)

            for batch_id, expected_count in REVIEWED_RELEASE_BATCH_COUNTS.items():
                with self.subTest(batch_id=batch_id):
                    release_batch = [
                        question
                        for question in questions.values()
                        if question.provenance.get("batch_id") == batch_id
                    ]
                    self.assertEqual(len(release_batch), expected_count)
                    self.assertTrue(
                        all(
                            question.status.value == "approved"
                            and question.status.eligible_for_adaptation
                            and question.provenance.get("review_status")
                            == "independent_model_review_passed"
                            for question in release_batch
                        )
                    )


if __name__ == "__main__":
    unittest.main()
