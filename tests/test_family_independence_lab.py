# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.family_independence_lab import (
    ANSWER_REDACTED_REVIEW_VERSION,
    DEFAULT_CORPUS,
    LAB_VERSION,
    REPORT_CONTRACT_VERSION,
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
                published_family_id="f_left",
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
                published_family_id="f_right",
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
                published_family_id="f_other",
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

    def test_declared_clusters_record_exact_resolved_operations(
        self,
    ) -> None:
        report = build_report()
        declared = {
            row["cluster_id"]: row
            for row in report["declared_review_clusters"]
        }
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
            "causal_training_visibility_triad": {
                "q_causal_mask_training_leak_001",
                "q_causal_mask_parallelism_001",
                "q_causal_mask_batch_matrix_001",
            },
            "multiquery_kv_cache_trio": {
                "q_transformer_multiquery_cache_axes_001",
                "q_transformer_multiquery_cache_inventory_001",
                "q_transformer_multiquery_state_transition_001",
            },
        }
        for cluster_id, question_ids in expected.items():
            self.assertEqual(
                set(declared[cluster_id]["question_ids"]),
                question_ids,
            )

        for cluster_id in (
            "attention_value_routing_trio",
            "causal_training_visibility_triad",
        ):
            cluster = declared[cluster_id]
            self.assertTrue(cluster["semantic_dependence_established"])
            self.assertEqual(
                cluster["review_status"],
                "reviewed_equivalent_in_family_manifest",
            )
            stress = cluster["capacity_stress_test"]
            self.assertEqual(
                stress["impact"]["assessment"],
                "already_collapsed_by_semantic_review",
            )
            self.assertEqual(
                stress["baseline"]["order_robust_main_capacity"],
                stress["collapsed"]["order_robust_main_capacity"],
            )

        distinct = declared["multiquery_kv_cache_trio"]
        self.assertFalse(distinct["semantic_dependence_established"])
        self.assertTrue(distinct["semantic_independence_established"])
        self.assertEqual(
            distinct["review_status"],
            "reviewed_distinct_after_answer_redacted_review",
        )
        self.assertEqual(distinct["review_input"], "stem_and_option_text_only")
        self.assertEqual(
            {
                family_id
                for operation in distinct["solution_operations"]
                for family_id in operation["evidence_family_ids"]
            },
            set(distinct["family_ids"]),
        )
        self.assertTrue(distinct["automatic_signal_connected"])
        stress = distinct["capacity_stress_test"]
        self.assertTrue(
            stress["impact"]["capacity_critical_if_dependence_confirmed"]
        )
        self.assertGreater(
            stress["baseline"]["order_robust_main_capacity"],
            stress["collapsed"]["order_robust_main_capacity"],
        )
        self.assertGreater(
            stress["baseline"]["achievable_main_capacity"],
            stress["collapsed"]["achievable_main_capacity"],
        )

    def test_all_signal_clusters_have_fail_closed_adjudications(self) -> None:
        report = build_report()
        reviewed = {
            row["cluster_id"]: row
            for row in report["reviewed_signal_clusters"]
        }

        self.assertEqual(
            set(reviewed),
            {
                "attention_equivariance_matrix_equivalence",
                "attention_equivariance_operation_partition",
                "attention_resource_pair_count_equivalence",
                "attention_scaling_operation_partition",
                "attention_temperature_vs_variance",
                "causal_visibility_operation_partition",
                "kv_cache_operation_partition",
                "normalization_residual_operation_partition",
            },
        )
        self.assertEqual(report["unresolved_signal_clusters"], [])
        self.assertEqual(
            report["findings"]["unresolved_signal_cluster_count"],
            0,
        )
        self.assertTrue(report["findings"]["required_signal_reviews_present"])
        self.assertEqual(
            len(report["signal_nominated_clusters"]),
            len(reviewed),
        )

        equivalent = {
            "attention_equivariance_matrix_equivalence",
            "attention_resource_pair_count_equivalence",
        }
        for cluster_id, cluster in reviewed.items():
            self.assertEqual(
                cluster["review_version"],
                ANSWER_REDACTED_REVIEW_VERSION,
            )
            self.assertEqual(
                cluster["review_input"],
                "stem_and_option_text_only",
            )
            operation_families = [
                family_id
                for operation in cluster["solution_operations"]
                for family_id in operation["evidence_family_ids"]
            ]
            self.assertEqual(
                sorted(operation_families),
                sorted(cluster["family_ids"]),
            )
            self.assertEqual(
                len(operation_families),
                len(set(operation_families)),
            )
            if cluster_id in equivalent:
                self.assertTrue(cluster["semantic_dependence_established"])
                self.assertFalse(cluster["semantic_independence_established"])
                self.assertEqual(
                    cluster["review_status"],
                    "reviewed_equivalent_in_family_manifest",
                )
                self.assertEqual(len(cluster["family_ids"]), 1)
                self.assertEqual(
                    cluster["capacity_stress_test"]["impact"]["assessment"],
                    "already_collapsed_by_semantic_review",
                )
            else:
                self.assertFalse(cluster["semantic_dependence_established"])
                self.assertTrue(cluster["semantic_independence_established"])
                self.assertEqual(
                    cluster["review_status"],
                    "reviewed_distinct_after_answer_redacted_review",
                )
                self.assertGreater(len(cluster["family_ids"]), 1)
                self.assertEqual(
                    len(cluster["automatic_signal_component_question_ids"]),
                    1,
                )

    def test_family_manifest_is_exact_explicit_and_fail_closed(self) -> None:
        report = build_report()
        audit = report["family_manifest_audit"]
        assignments = audit["member_assignments"]

        self.assertTrue(audit["fail_closed"])
        self.assertTrue(audit["all_alias_assignments_exact"])
        self.assertTrue(audit["all_alias_users_explicit"])
        self.assertTrue(audit["all_large_cohorts_exactly_reviewed"])
        self.assertGreater(audit["explicit_alias_family_count"], 0)
        self.assertEqual(
            audit["explicit_alias_member_count"],
            len(assignments),
        )
        self.assertEqual(
            len({row["question_id"] for row in assignments}),
            len(assignments),
        )
        questions = {
            question.id: question
            for question in read_and_parse(DEFAULT_CORPUS)[4]
        }
        for row in assignments:
            self.assertTrue(row["exact_assignment_valid"])
            self.assertEqual(
                row["published_family_id"],
                row["manifest_published_family_id"],
            )
            self.assertEqual(
                row["evidence_family_id"],
                row["manifest_evidence_family_id"],
            )
            if row["status"] == "approved":
                continue

            # A manifested alias remains necessary for immutable historical
            # identity after revision.  It may be retired only when one direct
            # approved child preserves both its published and canonical family.
            self.assertEqual(row["status"], "retired")
            children = [
                question
                for question in questions.values()
                if question.revision_of == row["question_id"]
                and question.status.value == "approved"
                and question.published_family_id
                == row["published_family_id"]
                and question.family_id == row["evidence_family_id"]
            ]
            self.assertEqual(
                len(children),
                1,
                f"Retired alias {row['question_id']} lacks one exact active child",
            )

        cohorts = audit["reviewed_large_cohorts"]
        self.assertEqual(
            audit["reviewed_large_cohort_count"],
            len(cohorts),
        )
        self.assertTrue(
            all(
                row["question_count"]
                == len(row["question_ids"])
                > audit["large_family_threshold"]
                and row["exact_reviewed_cohort"]
                for row in cohorts
            )
        )

    def test_serviceability_audit_matches_strict_corpus_quality(self) -> None:
        report = build_report()
        audit = report["serviceability_audit"]

        self.assertEqual(
            audit["question_count"],
            report["corpus"]["question_count"],
        )
        self.assertGreaterEqual(audit["question_count"], 493)
        self.assertEqual(audit["error_count"], 0)
        self.assertEqual(audit["warning_count"], 0)
        self.assertTrue(audit["strict_pass"])
        self.assertEqual(audit["error_codes"], [])
        self.assertEqual(audit["warning_codes"], [])
        self.assertEqual(audit["error_identifiers"], [])
        self.assertEqual(audit["warning_identifiers"], [])
        self.assertEqual(audit["serviceability_issue_count"], 0)
        self.assertEqual(audit["serviceability_issue_codes"], [])
        self.assertEqual(audit["serviceability_issue_identifiers"], [])
        self.assertEqual(audit["serviceability_issues"], [])
        self.assertTrue(
            report["findings"]["strict_corpus_audit_passed"]
        )
        self.assertEqual(
            report["findings"]["in_memory_status_substitution_count"],
            0,
        )

    def test_report_is_deterministic_versioned_and_read_only(self) -> None:
        before = corpus_source_digest(DEFAULT_CORPUS)
        first = build_report()
        second = build_report()
        after = corpus_source_digest(DEFAULT_CORPUS)

        self.assertEqual(LAB_VERSION, "family-independence-falsification-v6")
        self.assertEqual(
            REPORT_CONTRACT_VERSION,
            "family-independence-report-v3",
        )
        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(first["lab_version"], LAB_VERSION)
        self.assertEqual(
            first["report_contract_version"],
            REPORT_CONTRACT_VERSION,
        )
        self.assertEqual(first["corpus"]["sha256"], before)
        self.assertTrue(first["corpus"]["source_bytes_unchanged"])
        self.assertEqual(
            first["corpus"]["status_counts"].get("approved", 0),
            491,
        )
        self.assertEqual(
            first["corpus"]["status_counts"].get("retired", 0),
            41,
        )
        self.assertEqual(
            set(first["corpus"]["status_counts"]),
            {"approved", "retired"},
        )
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
        self.assertEqual(
            first["analysis_contract"]["content_activation_semantics"],
            (
                "none; the lab analyzes only currently adaptation-eligible "
                "questions and never substitutes a status"
            ),
        )

    def test_fail_on_critical_accepts_resolved_clusters(self) -> None:
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

            self.assertEqual(status, 0)
            self.assertTrue(output.is_file())

    def test_fail_on_serviceability_accepts_the_reviewed_corpus(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="tsq-family-serviceability-test-"
        ) as raw:
            output = Path(raw) / "report.json"
            status = main(
                [
                    "--output",
                    str(output),
                    "--fail-on-serviceability",
                ]
            )

            self.assertEqual(status, 0)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
