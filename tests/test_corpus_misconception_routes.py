# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest
from pathlib import Path

from tsq.corpus import read_and_parse


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CORPUS = ROOT / "corpus" / "ai_curriculum.json"
PACKAGED_CORPUS = ROOT / "src" / "tsq" / "data" / "ai_curriculum.json"

NEW_TRANSFORMER_QUESTIONS = {
    "q_attention_duplicate_value_identifiability_001",
    "q_causal_mask_softmax_normalization_001",
    "q_causal_full_incremental_equivalence_001",
    "q_causal_cross_attention_mask_scope_001",
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
GENERATED_BATCH_ID = "batch_rag_agent_headroom_20260723_c"
AGENT_INTRO_BATCH_ID = "batch_agent_intro_bridges_20260724_a"
RAG_INTRO_BATCH_ID = "batch_rag_intro_bridges_20260724_a"
TRANSFORMER_INTRO_BATCH_ID = "batch_transformer_intro_bridges_20260724_a"
LEGACY_GENERATED_MIGRATION_COUNT = 39
TOTAL_UNREVIEWED_GENERATED_COUNT = 61


class CorpusMisconceptionRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        canonical = read_and_parse(CANONICAL_CORPUS)
        packaged = read_and_parse(PACKAGED_CORPUS)
        cls.canonical_misconceptions = {
            item.id: item for item in canonical[2]
        }
        cls.canonical_questions = {
            item.id: item for item in canonical[4]
        }
        cls.packaged_questions = {
            item.id: item for item in packaged[4]
        }

    def test_corpus_copies_remain_byte_identical(self) -> None:
        self.assertEqual(
            CANONICAL_CORPUS.read_bytes(),
            PACKAGED_CORPUS.read_bytes(),
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
