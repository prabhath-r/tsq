# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from tsq.corpus import corpus_source_digest, read_and_parse


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CORPUS = ROOT / "corpus"
PACKAGED_CORPUS = ROOT / "src" / "tsq" / "data" / "curriculum"

NEW_TRANSFORMER_QUESTIONS = {
    "q_attention_duplicate_value_identifiability_001",
    "q_causal_mask_softmax_normalization_001",
    "q_causal_full_incremental_equivalence_001",
    "q_causal_cross_attention_mask_scope_001",
    "q_causal_cross_attention_mask_scope_002",
    "q_transformer_token_intervention_trace_001",
}
NEW_AGENT_INTRO_QUESTIONS = {
    "q_agent_catalog_expansion_boundary_001",
    "q_agent_granted_subset_match_001",
    "q_agent_context_caveat_expiry_001",
    "q_agent_dry_run_effect_boundary_001",
    "q_agent_completion_predicate_counterfactual_001",
}
NEW_RAG_INTRO_QUESTIONS = {
    "q_rag_claim_citation_alignment_revision_001",
    "q_rag_conjunctive_facet_coverage_001",
}
NEW_TRANSFORMER_INTRO_QUESTIONS = {
    "q_attention_runtime_workspace_boundary_001",
    "q_transformer_unexpected_cross_token_path_001",
}
NEW_TRANSFORMER_CAPACITY_QUESTIONS = {
    "q_transformer_kv_cache_alignment_001",
    "q_transformer_kv_cache_eviction_equivalence_001",
    "q_attention_duplicate_value_identifiability_002",
    "q_attention_value_gradient_routing_001",
}
GENERATED_BATCH_ID = "batch_rag_agent_headroom_20260723_c"
AGENT_INTRO_BATCH_ID = "batch_agent_intro_bridges_20260724_a"
RAG_INTRO_BATCH_ID = "batch_rag_intro_bridges_20260724_a"
TRANSFORMER_INTRO_BATCH_ID = "batch_transformer_intro_bridges_20260724_a"
TRANSFORMER_CAPACITY_BATCH_ID = (
    "batch_transformer_capacity_repairs_20260725_a"
)
LEGACY_GENERATED_MIGRATION_COUNT = 39
TOTAL_UNREVIEWED_GENERATED_COUNT = 66


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
            "q_transformer_token_intervention_trace_001"
        ]
        distractor = next(option for option in question.options if option.id == "d")

        self.assertEqual(question.status.value, "quarantined")
        self.assertFalse(question.status.eligible_for_adaptation)
        self.assertIn("mask that blocks that direction", distractor.text)
        self.assertEqual(
            distractor.misconception_id,
            "m_transformers_are_inherently_bidirectional",
        )
        self.assertEqual(
            distractor.diagnostic_objective_id,
            "lo_causal_visibility",
        )

    def test_causal_cross_attention_revision_is_quarantined_same_family(
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
            parent = questions[
                "q_causal_cross_attention_mask_scope_001"
            ]
            revision = questions[
                "q_causal_cross_attention_mask_scope_002"
            ]
            self.assertEqual(revision.revision_of, parent.id)
            self.assertGreater(revision.version, parent.version)
            self.assertEqual(revision.family_id, parent.family_id)
            self.assertEqual(
                revision.family_id,
                "f_causal_cross_attention_mask_scope",
            )
            self.assertEqual(
                revision.objective_id,
                "lo_causal_visibility",
            )
            self.assertEqual(revision.status.value, "quarantined")
            self.assertFalse(revision.status.eligible_for_adaptation)
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
            self.assertEqual(
                revision.provenance["review_status"],
                "internal_ai_critique_pending_quarantined",
            )
            self.assertEqual(
                revision.provenance["human_review_status"],
                "required_before_activation",
            )
            self.assertIn(
                parent.id,
                revision.provenance["independence_note"],
            )

    def test_agent_capability_availability_has_exact_named_route(self) -> None:
        question = self.canonical_questions[
            "q_agent_approval_argument_binding_001"
        ]
        distractor = next(option for option in question.options if option.id == "d")
        misconception = self.canonical_misconceptions[
            "m_agent_tool_availability_is_authorization"
        ]

        self.assertEqual(question.status.value, "quarantined")
        self.assertFalse(question.status.eligible_for_adaptation)
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

    def test_approval_fix_preserves_generated_review_boundary(self) -> None:
        question = self.canonical_questions[
            "q_agent_approval_argument_binding_001"
        ]

        self.assertIs(question.provenance["generated"], True)
        self.assertIs(question.provenance["human_review"], False)
        self.assertEqual(
            question.provenance["review_status"],
            "final_independent_ai_review_passed_quarantined",
        )
        self.assertEqual(
            question.provenance["activation"],
            "manual_only_after_human_review_and_new_immutable_release",
        )

    def test_agent_intro_candidates_are_quarantined_and_exactly_routed(
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
            }
            self.assertEqual(set(batch), NEW_AGENT_INTRO_QUESTIONS)
            self.assertEqual(
                len({question.family_id for question in batch.values()}),
                5,
            )
            self.assertTrue(
                all(
                    question.status.value == "quarantined"
                    and not question.status.eligible_for_adaptation
                    and question.difficulty < -0.3
                    and question.provenance.get("generated") is True
                    and question.provenance.get("human_review") is False
                    and question.provenance.get("human_review_status")
                    == "required_before_activation"
                    and question.provenance.get("review_status")
                    == "internal_ai_critique_pending_quarantined"
                    and bool(question.provenance.get("source_scope"))
                    and question.provenance.get("activation")
                    == "manual_only_after_human_review_and_new_immutable_release"
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
            subset = batch["q_agent_granted_subset_match_001"]
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
                "q_agent_completion_predicate_counterfactual_001"
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

    def test_target_status_boundaries_are_unchanged_in_both_copies(self) -> None:
        for questions in (
            self.canonical_questions,
            self.packaged_questions,
        ):
            self.assertTrue(
                all(
                    questions[question_id].status.value == "quarantined"
                    and not questions[
                        question_id
                    ].status.eligible_for_adaptation
                    and questions[question_id].provenance.get("generated") is True
                    and questions[question_id].provenance.get("provider")
                    == "openai_codex"
                    and questions[question_id].provenance.get("human_review")
                    is False
                    and questions[question_id].provenance.get("activation")
                    == (
                        "manual_only_after_human_review_and_new_immutable_release"
                    )
                    for question_id in NEW_TRANSFORMER_QUESTIONS
                )
            )
            generated = [
                question
                for question in questions.values()
                if question.provenance.get("batch_id") == GENERATED_BATCH_ID
            ]
            self.assertEqual(len(generated), 8)
            self.assertTrue(
                all(
                    question.status.value == "quarantined"
                    and not question.status.eligible_for_adaptation
                    for question in generated
                )
            )

    def test_rag_intro_candidates_are_quarantined_and_independently_routed(
        self,
    ) -> None:
        expected_routes = {
            "q_rag_claim_citation_alignment_revision_001": {
                "m_rag_citation_proves_entailment",
                "m_rag_context_guarantees_use",
                "m_rag_retrieval_updates_weights",
            },
            "q_rag_conjunctive_facet_coverage_001": {
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
            }
            self.assertEqual(set(batch), NEW_RAG_INTRO_QUESTIONS)
            self.assertEqual(
                len({question.family_id for question in batch.values()}),
                2,
            )
            self.assertTrue(
                all(
                    question.status.value == "quarantined"
                    and not question.status.eligible_for_adaptation
                    and question.difficulty < -0.5
                    and question.provenance.get("generated") is True
                    and question.provenance.get("human_review") is False
                    and question.provenance.get("human_review_status")
                    == "required_before_activation"
                    and question.provenance.get("review_status")
                    == "internal_ai_critique_pending_quarantined"
                    and bool(question.provenance.get("source_scope"))
                    and question.provenance.get("activation")
                    == "manual_only_after_human_review_and_new_immutable_release"
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
                "q_rag_claim_citation_alignment_revision_001"
            ]
            retrieval = batch["q_rag_conjunctive_facet_coverage_001"]
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

    def test_transformer_intro_candidates_are_quarantined_and_exactly_routed(
        self,
    ) -> None:
        expected_routes = {
            "q_attention_runtime_workspace_boundary_001": {
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
            "q_transformer_unexpected_cross_token_path_001": {
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
                    "q_attention_runtime_workspace_boundary_001": (
                        "f_transformer_sequence_shape_audit"
                    ),
                    "q_transformer_unexpected_cross_token_path_001": (
                        "f_transformer_axis_mixing"
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
                "q_attention_runtime_workspace_boundary_001"
            ]
            path = batch[
                "q_transformer_unexpected_cross_token_path_001"
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
                    question.status.value == "quarantined"
                    and not question.status.eligible_for_adaptation
                    and question.difficulty < -0.5
                    and question.provenance.get("generated") is True
                    and question.provenance.get("human_review") is False
                    and question.provenance.get("human_review_status")
                    == "required_before_activation"
                    and question.provenance.get("review_status")
                    == "internal_ai_critique_pending_quarantined"
                    and bool(question.provenance.get("source_scope"))
                    and bool(question.provenance.get("independence_note"))
                    and question.provenance.get("psychometrics")
                    == "uncalibrated_author_prior"
                    and question.provenance.get("activation")
                    == (
                        "manual_only_after_human_review_and_new_immutable_release"
                    )
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

    def test_transformer_capacity_candidates_are_distinct_and_quarantined(
        self,
    ) -> None:
        expected_routes = {
            "q_transformer_kv_cache_alignment_001": {
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
            "q_transformer_kv_cache_eviction_equivalence_001": {
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
            "q_attention_duplicate_value_identifiability_002": {
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
            "q_attention_value_gradient_routing_001": {
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
                    question.status.value == "quarantined"
                    and not question.status.eligible_for_adaptation
                    and question.provenance.get("generated") is True
                    and question.provenance.get("human_review") is False
                    and question.provenance.get("human_review_status")
                    == "required_before_activation"
                    and question.provenance.get("review_status")
                    == "internal_ai_critique_pending_quarantined"
                    and question.provenance.get("activation")
                    == (
                        "manual_only_after_human_review_and_new_immutable_release"
                    )
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
                "q_transformer_kv_cache_alignment_001"
            ]
            eviction = batch[
                "q_transformer_kv_cache_eviction_equivalence_001"
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
            revision = batch[
                "q_attention_duplicate_value_identifiability_002"
            ]
            self.assertEqual(revision.revision_of, parent.id)
            self.assertEqual(revision.version, 2)
            self.assertGreater(revision.version, parent.version)
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
                "q_attention_value_gradient_routing_001"
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

    def test_generated_without_human_review_is_never_adaptation_eligible(
        self,
    ) -> None:
        for questions in (
            self.canonical_questions,
            self.packaged_questions,
        ):
            unreviewed_generated = [
                question
                for question in questions.values()
                if question.provenance.get("generated") is True
                and question.provenance.get("human_review") is False
            ]
            legacy_migration = [
                question
                for question in unreviewed_generated
                if question.provenance.get("batch_id") != GENERATED_BATCH_ID
                and question.id not in NEW_TRANSFORMER_QUESTIONS
                and question.id not in NEW_AGENT_INTRO_QUESTIONS
                and question.id not in NEW_RAG_INTRO_QUESTIONS
                and question.id not in NEW_TRANSFORMER_INTRO_QUESTIONS
                and question.id not in NEW_TRANSFORMER_CAPACITY_QUESTIONS
            ]

            self.assertEqual(
                len(unreviewed_generated),
                TOTAL_UNREVIEWED_GENERATED_COUNT,
            )
            self.assertEqual(
                len(legacy_migration),
                LEGACY_GENERATED_MIGRATION_COUNT,
            )
            self.assertTrue(
                all(
                    question.status.value == "quarantined"
                    and not question.status.eligible_for_adaptation
                    for question in unreviewed_generated
                )
            )


if __name__ == "__main__":
    unittest.main()
