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
from tsq.corpus import read_and_parse


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

    def test_known_trios_are_signal_connected_and_capacity_critical(self) -> None:
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
        }
        nominated_pairs = {
            frozenset(
                (row["left_question_id"], row["right_question_id"])
            )
            for row in report["duplicate_pair_candidates"]
        }
        for cluster_id, question_ids in expected.items():
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

    def test_quarantined_repairs_restore_declared_collapse_in_memory_only(
        self,
    ) -> None:
        before = DEFAULT_CORPUS.read_bytes()
        report = build_report()
        after = DEFAULT_CORPUS.read_bytes()
        batch = report["quarantined_candidate_counterfactual"]

        self.assertEqual(LAB_VERSION, "family-independence-falsification-v2")
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

    def test_report_is_deterministic_versioned_and_read_only(self) -> None:
        before = hashlib.sha256(DEFAULT_CORPUS.read_bytes()).hexdigest()
        first = build_report()
        second = build_report()
        after = hashlib.sha256(DEFAULT_CORPUS.read_bytes()).hexdigest()

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
