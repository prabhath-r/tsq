# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.family_independence_lab import (
    CANDIDATE_REPAIR_BATCH_ID,
    DEFAULT_CORPUS,
    LAB_VERSION,
    QuestionFeatures,
    build_report,
    canonical_hash,
    main,
    normalized_tokens,
    pair_evidence,
)
from tsq.corpus import corpus_source_digest, read_and_parse


class FamilyIndependenceLabTests(unittest.TestCase):
    def test_transparent_pair_signal_is_only_a_review_nomination(self) -> None:
        shared_stem = normalized_tokens(
            "A decoder cache retains keys and values for every prefix token."
        )
        shared_solution = normalized_tokens(
            "Keep the sequence axis; sharing key heads does not collapse time."
        )
        left = QuestionFeatures(
            question=SimpleNamespace(
                id="q_left",
                family_id="f_left",
                objective_id="lo_cache",
                misconception_ids={"m_time", "m_queries"},
            ),
            stem_tokens=shared_stem,
            solution_tokens=shared_solution,
        )
        right = QuestionFeatures(
            question=SimpleNamespace(
                id="q_right",
                family_id="f_right",
                objective_id="lo_cache",
                misconception_ids={"m_queries", "m_time"},
            ),
            stem_tokens=shared_stem,
            solution_tokens=shared_solution,
        )

        result = pair_evidence(left, right)

        self.assertTrue(result["signal_threshold_passed"])
        self.assertEqual(
            result["candidate_status"],
            "requires_independent_semantic_review",
        )
        self.assertFalse(result["semantic_dependence_established"])
        self.assertEqual(
            result["stem_overlap"]["shared_tokens"],
            sorted(shared_stem),
        )

        different_objective = QuestionFeatures(
            question=SimpleNamespace(
                id="q_other",
                family_id="f_other",
                objective_id="lo_other",
                misconception_ids={"m_queries", "m_time"},
            ),
            stem_tokens=shared_stem,
            solution_tokens=shared_solution,
        )
        self.assertFalse(
            pair_evidence(left, different_objective)[
                "signal_threshold_passed"
            ]
        )

    def test_declared_clusters_are_capacity_critical_and_fail_visible(
        self,
    ) -> None:
        report = build_report()
        declared = {
            row["cluster_id"]: row
            for row in report["declared_review_clusters"]
        }

        parsed_questions = read_and_parse(DEFAULT_CORPUS)[4]
        active = [
            question
            for question in parsed_questions
            if question.status.eligible_for_adaptation
        ]
        self.assertEqual(
            report["corpus"]["active_question_count"],
            len(active),
        )
        self.assertEqual(
            report["corpus"]["active_family_count"],
            len({question.family_id for question in active}),
        )
        self.assertEqual(
            set(declared),
            {
                "attention_value_routing_trio",
                "causal_training_visibility_triad",
                "multiquery_kv_cache_trio",
            },
        )
        expected = {
            "attention_value_routing_trio": {
                "q_attention_value_role_ablation_001",
                "q_attention_value_projection_counterfactual_001",
                "q_attention_value_perturbation_001",
            },
            "multiquery_kv_cache_trio": {
                "q_transformer_multiquery_cache_axes_001",
                "q_transformer_multiquery_cache_inventory_001",
                "q_transformer_multiquery_state_transition_001",
            },
            "causal_training_visibility_triad": {
                "q_causal_mask_training_leak_001",
                "q_causal_mask_parallelism_001",
                "q_causal_mask_batch_matrix_001",
            },
        }
        nominated_pairs = {
            frozenset(
                (row["left_question_id"], row["right_question_id"])
            )
            for row in report["duplicate_pair_candidates"]
        }
        for cluster_id in (
            "attention_value_routing_trio",
            "multiquery_kv_cache_trio",
        ):
            question_ids = expected[cluster_id]
            cluster = declared[cluster_id]
            self.assertEqual(set(cluster["question_ids"]), question_ids)
            self.assertTrue(cluster["automatic_signal_connected"])
            self.assertFalse(cluster["semantic_dependence_established"])
            self.assertEqual(
                len(
                    [
                        pair
                        for pair in nominated_pairs
                        if pair <= question_ids
                    ]
                ),
                3,
            )
            stress = cluster["capacity_stress_test"]
            self.assertTrue(
                stress["impact"][
                    "capacity_critical_if_dependence_confirmed"
                ]
            )
            self.assertEqual(
                stress["baseline"]["order_robust_main_capacity"], 1
            )
            self.assertEqual(
                stress["collapsed"]["order_robust_main_capacity"], 0
            )
            self.assertEqual(
                stress["collapsed"]["aggregate_status"], "blocked"
            )

        causal = declared["causal_training_visibility_triad"]
        causal_ids = expected["causal_training_visibility_triad"]
        self.assertEqual(set(causal["question_ids"]), causal_ids)
        self.assertFalse(causal["automatic_signal_connected"])
        self.assertFalse(causal["semantic_dependence_established"])
        self.assertEqual(
            len(
                [
                    pair
                    for pair in nominated_pairs
                    if pair <= causal_ids
                ]
            ),
            1,
        )
        causal_stress = causal["capacity_stress_test"]
        self.assertEqual(
            causal_stress["baseline"]["order_robust_main_capacity"], 2
        )
        self.assertEqual(
            causal_stress["baseline"]["achievable_main_capacity"], 2
        )
        self.assertEqual(
            causal_stress["collapsed"]["order_robust_main_capacity"], 0
        )
        self.assertEqual(
            causal_stress["collapsed"]["achievable_main_capacity"], 0
        )
        self.assertEqual(
            causal_stress["collapsed"]["aggregate_status"], "blocked"
        )
        self.assertEqual(
            set(
                causal_stress[
                    "counterfactually_unavailable_family_ids"
                ]
            ),
            {
                "f_causal_mask_parallelism",
                "f_causal_mask_training_leak",
            },
        )

    def test_quarantined_repairs_restore_declared_collapse_in_memory_only(
        self,
    ) -> None:
        before = corpus_source_digest(DEFAULT_CORPUS)
        report = build_report()
        after = corpus_source_digest(DEFAULT_CORPUS)
        batch = report["quarantined_candidate_counterfactual"]

        self.assertEqual(LAB_VERSION, "family-independence-falsification-v4")
        self.assertEqual(report["lab_version"], LAB_VERSION)
        self.assertEqual(before, after)
        self.assertEqual(batch["batch_id"], CANDIDATE_REPAIR_BATCH_ID)
        self.assertEqual(
            batch["status"],
            "counterfactual_only_human_review_required",
        )
        self.assertFalse(batch["source_corpus_mutated"])
        self.assertFalse(batch["semantic_independence_established"])
        self.assertTrue(batch["manual_activation_required"])
        self.assertEqual(
            set(batch["candidate_question_ids"]),
            {
                "q_transformer_kv_cache_alignment_001",
                "q_transformer_kv_cache_eviction_equivalence_001",
                "q_attention_duplicate_value_identifiability_002",
                "q_attention_value_gradient_routing_001",
            },
        )

        parsed = {
            question.id: question
            for question in read_and_parse(DEFAULT_CORPUS)[4]
        }
        for row in batch["source_candidate_state"]:
            question = parsed[row["question_id"]]
            self.assertEqual(question.status.value, "quarantined")
            self.assertFalse(question.status.eligible_for_adaptation)
            self.assertTrue(question.provenance["generated"])
            self.assertFalse(question.provenance["human_review"])
            self.assertEqual(
                question.provenance["activation"],
                "manual_only_after_human_review_and_new_immutable_release",
            )
            self.assertTrue(row["quarantine_invariants_validated"])
            self.assertEqual(row["source_status"], "quarantined")
            self.assertFalse(row["source_eligible_for_adaptation"])
            self.assertTrue(row["generated"])
            self.assertFalse(row["human_review"])

        comparisons = {
            row["declared_cluster_id"]: row
            for row in batch["cluster_comparisons"]
        }
        self.assertEqual(
            set(comparisons),
            {
                "attention_value_routing_trio",
                "multiquery_kv_cache_trio",
            },
        )
        for comparison in comparisons.values():
            baseline = comparison["baseline"]
            collapsed = comparison["declared_cluster_collapsed"]
            expanded = comparison[
                "with_candidates_counterfactually_eligible"
            ]
            repaired = comparison[
                "collapsed_with_candidates_counterfactually_eligible"
            ]
            self.assertLess(
                collapsed["order_robust_main_capacity"],
                baseline["order_robust_main_capacity"],
            )
            self.assertLess(
                collapsed["achievable_main_capacity"],
                baseline["achievable_main_capacity"],
            )
            self.assertGreaterEqual(
                expanded["order_robust_main_capacity"],
                baseline["order_robust_main_capacity"],
            )
            self.assertGreaterEqual(
                repaired["order_robust_main_capacity"],
                baseline["order_robust_main_capacity"],
            )
            self.assertGreaterEqual(
                repaired["achievable_main_capacity"],
                baseline["achievable_main_capacity"],
            )
            self.assertTrue(
                comparison["restoration_check"][
                    "restores_baseline_if_candidate_families_survive_review"
                ]
            )
            self.assertFalse(
                comparison["semantic_independence_established"]
            )
            self.assertTrue(
                comparison["human_review_required_before_activation"]
            )

    def test_causal_reserve_power_set_is_exact_quarantined_and_read_only(
        self,
    ) -> None:
        before = corpus_source_digest(DEFAULT_CORPUS)
        report = build_report()
        after = corpus_source_digest(DEFAULT_CORPUS)
        reserve = report["causal_reserve_counterfactual"]

        candidate_ids = {
            "q_causal_mask_matrix_001",
            "q_causal_mask_softmax_normalization_001",
            "q_causal_full_incremental_equivalence_001",
            "q_causal_cross_attention_mask_scope_002",
        }
        route_candidate_ids = {
            "q_causal_mask_matrix_001",
            "q_causal_mask_softmax_normalization_001",
            "q_causal_full_incremental_equivalence_001",
        }
        historical_cohort_ids = {
            *candidate_ids,
            "q_causal_cross_attention_mask_scope_001",
        }
        dependent_variants = {
            "q_causal_mask_first_row_001": (
                "f_causal_mask_softmax_normalization",
                "batch_transformer_composition_causality_20260809_d",
            ),
            "q_causal_mask_probability_row_001": (
                "f_causal_mask_softmax_normalization",
                "batch_transformer_composition_causality_20260809_d",
            ),
            "q_causal_mask_post_softmax_bug_001": (
                "f_causal_mask_softmax_normalization",
                "batch_transformer_composition_causality_20260809_d",
            ),
            "q_causal_full_incremental_warmup_001": (
                "f_causal_full_incremental_equivalence",
                "batch_transformer_causality_scaling_20260809_e",
            ),
            "q_causal_full_incremental_intermediate_row_001": (
                "f_causal_full_incremental_equivalence",
                "batch_transformer_causality_scaling_20260809_e",
            ),
            "q_causal_full_incremental_mismatch_001": (
                "f_causal_full_incremental_equivalence",
                "batch_transformer_causality_scaling_20260809_e",
            ),
        }
        current_family_member_ids = {
            *historical_cohort_ids,
            *dependent_variants,
        }
        self.assertEqual(before, after)
        self.assertTrue(report["corpus"]["source_bytes_unchanged"])
        self.assertFalse(reserve["source_corpus_mutated"])
        self.assertFalse(reserve["semantic_independence_established"])
        self.assertTrue(
            reserve["human_review_required_before_activation"]
        )
        self.assertTrue(reserve["manual_activation_required"])
        self.assertEqual(
            set(reserve["candidate_question_ids"]), candidate_ids
        )
        self.assertEqual(reserve["candidate_family_count"], 4)
        self.assertEqual(reserve["historical_cohort_count"], 5)
        self.assertEqual(
            set(reserve["historical_cohort_question_ids"]),
            historical_cohort_ids,
        )
        self.assertEqual(reserve["dependent_variant_count"], 6)
        self.assertEqual(
            set(reserve["dependent_variant_question_ids"]),
            set(dependent_variants),
        )
        self.assertEqual(reserve["current_family_member_count"], 11)
        self.assertEqual(
            set(reserve["current_family_member_question_ids"]),
            current_family_member_ids,
        )
        self.assertEqual(reserve["source_family_member_count"], 11)
        self.assertEqual(
            set(reserve["source_family_member_question_ids"]),
            current_family_member_ids,
        )
        self.assertTrue(
            reserve["dependent_variants_excluded_from_counterfactual"]
        )
        self.assertFalse(
            reserve["dependent_variants_capacity_credit_granted"]
        )
        self.assertTrue(candidate_ids.isdisjoint(dependent_variants))
        declared_historical_ids = {
            question_id
            for row in reserve["candidate_declarations"]
            for question_id in row["historical_family_question_ids"]
        }
        declared_dependent_ids = {
            question_id
            for row in reserve["candidate_declarations"]
            for question_id in row["dependent_variant_question_ids"]
        }
        self.assertEqual(declared_historical_ids, historical_cohort_ids)
        self.assertEqual(declared_dependent_ids, set(dependent_variants))
        cross = next(
            row
            for row in reserve["candidate_declarations"]
            if row["family_id"]
            == "f_causal_cross_attention_mask_scope"
        )
        self.assertEqual(
            cross["family_question_ids"],
            [
                "q_causal_cross_attention_mask_scope_001",
                "q_causal_cross_attention_mask_scope_002",
            ],
        )
        self.assertEqual(cross["same_family_member_count"], 2)

        dependent_rows = {
            row["question_id"]: row
            for row in reserve["dependent_variant_declarations"]
        }
        self.assertEqual(set(dependent_rows), set(dependent_variants))
        for question_id, (family_id, batch_id) in dependent_variants.items():
            row = dependent_rows[question_id]
            self.assertEqual(row["family_id"], family_id)
            self.assertEqual(row["batch_id"], batch_id)
            self.assertEqual(row["objective_id"], "lo_causal_visibility")
            self.assertEqual(row["source_status"], "quarantined")
            self.assertFalse(row["source_eligible_for_adaptation"])
            self.assertTrue(row["generated"])
            self.assertFalse(row["human_review"])
            self.assertEqual(
                row["human_review_status"],
                "required_before_activation",
            )
            self.assertEqual(
                row["activation"],
                "manual_only_after_human_review_and_new_immutable_release",
            )
            self.assertEqual(
                row["review_status"],
                "candidate_pending_independent_review",
            )
            self.assertEqual(
                row["psychometrics"],
                "uncalibrated_author_prior",
            )
            self.assertFalse(row["counterfactual_representative"])
            self.assertFalse(row["included_in_subset_analysis"])
            self.assertFalse(row["capacity_credit_granted"])
            self.assertTrue(row["quarantine_invariants_validated"])

        source = {
            row["question_id"]: row
            for row in reserve["source_candidate_state"]
        }
        self.assertEqual(
            reserve["provenance_gap_question_ids"],
            ["q_causal_mask_matrix_001"],
        )
        self.assertFalse(reserve["all_source_provenance_complete"])
        for row in source.values():
            self.assertEqual(row["source_status"], "quarantined")
            self.assertFalse(row["source_eligible_for_adaptation"])
            self.assertTrue(
                row["activation_ceiling_preserved_by_source_status"]
            )
        for question_id in historical_cohort_ids:
            self.assertEqual(
                source[question_id]["cohort_role"],
                "historical_cohort_member",
            )
        for question_id, (_, batch_id) in dependent_variants.items():
            row = source[question_id]
            self.assertEqual(row["cohort_role"], "declared_dependent_variant")
            self.assertEqual(row["expected_batch_id"], batch_id)
            self.assertEqual(row["source_batch_id"], batch_id)
            self.assertFalse(row["counterfactual_representative"])
            self.assertFalse(row["included_in_subset_analysis"])
            self.assertFalse(row["capacity_credit_granted"])
        self.assertEqual(
            source["q_causal_mask_matrix_001"]["provenance_gaps"],
            [
                "activation_is_manual_only",
                "generated_is_explicit_true",
                "human_review_is_explicit_false",
            ],
        )
        for question_id, row in source.items():
            if question_id == "q_causal_mask_matrix_001":
                continue
            self.assertTrue(
                row["provenance_complete_for_counterfactual_review"]
            )
            self.assertEqual(row["provenance_gaps"], [])

        self.assertEqual(reserve["subset_count"], 16)
        self.assertEqual(
            reserve["subset_kind_counts"],
            {
                "none": 1,
                "single": 4,
                "pair": 6,
                "triple": 4,
                "all": 1,
            },
        )
        subsets = {
            frozenset(row["candidate_question_ids"]): row
            for row in reserve["subset_analysis"]
        }
        self.assertEqual(len(subsets), 16)
        self.assertIn(frozenset(), subsets)
        self.assertIn(frozenset(candidate_ids), subsets)
        self.assertTrue(
            all(
                set(row["candidate_question_ids"]).isdisjoint(
                    dependent_variants
                )
                for row in reserve["subset_analysis"]
            )
        )

        scenarios = {
            "declared_families": (4, 2),
            "batch_training_pair_collapsed": (3, 1),
            "training_visibility_triad_collapsed": (2, 0),
        }
        for subset_ids, row in subsets.items():
            concept_increment = len(subset_ids)
            route_increment = len(
                subset_ids & route_candidate_ids
            )
            for scenario_id, (
                concept_baseline,
                route_baseline,
            ) in scenarios.items():
                snapshot = row["collapse_scenarios"][scenario_id]
                concept = snapshot["whole_concept"]
                route = snapshot["exact_route"]
                expected_concept = concept_baseline + concept_increment
                expected_route = route_baseline + route_increment
                self.assertEqual(
                    concept["order_robust_main_capacity"],
                    expected_concept,
                )
                self.assertEqual(
                    concept["achievable_main_capacity"],
                    expected_concept,
                )
                self.assertEqual(
                    route["order_robust_main_capacity"],
                    expected_route,
                )
                self.assertEqual(
                    route["achievable_main_capacity"],
                    expected_route,
                )

        all_candidates = subsets[frozenset(candidate_ids)]
        triad = all_candidates["collapse_scenarios"][
            "training_visibility_triad_collapsed"
        ]
        self.assertEqual(
            triad["whole_concept"]["order_robust_main_capacity"], 6
        )
        self.assertEqual(
            triad["exact_route"]["order_robust_main_capacity"], 3
        )

    def test_report_is_deterministic_versioned_and_read_only(self) -> None:
        before = corpus_source_digest(DEFAULT_CORPUS)
        first = build_report()
        second = build_report()
        after = corpus_source_digest(DEFAULT_CORPUS)

        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(first["corpus"]["sha256"], before)
        self.assertEqual(
            first["artifact_sha256"],
            canonical_hash(
                {
                    key: value
                    for key, value in first.items()
                    if key != "artifact_sha256"
                }
            ),
        )
        self.assertEqual(
            set(first["production_versions"]),
            {
                "capacity_algorithm",
                "learner_model",
                "selection_boundary",
                "selection_policy",
            },
        )
        self.assertIn(
            "does not prove",
            first["analysis_contract"]["lexical_signal_semantics"],
        )

    def test_fail_on_critical_exit_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="tsq-family-independence-test-"
        ) as raw:
            output = Path(raw) / "report.json"
            status = main(
                [
                    "--output",
                    str(output),
                    "--fail-on-critical",
                ]
            )

            self.assertEqual(status, 3)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
