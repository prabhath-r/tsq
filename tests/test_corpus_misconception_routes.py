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
GENERATED_BATCH_ID = "batch_rag_agent_headroom_20260723_c"
LEGACY_GENERATED_MIGRATION_COUNT = 39
TOTAL_UNREVIEWED_GENERATED_COUNT = 52


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
