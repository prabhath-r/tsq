# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
LAB_PATH = ROOT / "experiments" / "cold_start_lab.py"
SPEC = importlib.util.spec_from_file_location("cold_start_lab", LAB_PATH)
assert SPEC is not None and SPEC.loader is not None
COLD_START_LAB = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COLD_START_LAB
SPEC.loader.exec_module(COLD_START_LAB)


class ColdStartLabTests(unittest.TestCase):
    def test_objective_depths_follow_declared_prerequisites(self) -> None:
        bundle = {
            "learning_objectives": [
                {"id": "root"},
                {"id": "middle"},
                {"id": "leaf"},
            ],
            "objective_edges": [
                {"source": "root", "target": "middle"},
                {"source": "middle", "target": "leaf"},
            ],
        }
        self.assertEqual(
            COLD_START_LAB._objective_depths(bundle),
            {"root": 0, "middle": 1, "leaf": 2},
        )

    def test_scope_excursion_reused_as_main_is_not_hidden(self) -> None:
        coverage = SimpleNamespace(
            observed_outside_scope_concepts=("c_out",),
            observed_outside_scope_objectives=("lo_out",),
            observed_outside_scope_questions=("q_out",),
            observed_outside_scope_families=("fam_out",),
            observed_outside_scope_evidence_families=(
                "objective:lo_out|family:fam_out",
            ),
            observed_outside_scope_objective_families=(
                "objective:lo_out|family:fam_out",
            ),
        )
        common = {
            "evidence_anchor_concept_id": "c_out",
            "learning_objective_id": "lo_out",
            "question_id": "q_out",
            "family_id": "fam_out",
        }
        exploration = SimpleNamespace(
            pedagogical_role="exploration_probe",
            **common,
        )
        main = SimpleNamespace(pedagogical_role="main", **common)

        deliberate_only = COLD_START_LAB._unlabeled_scope_excursions(
            SimpleNamespace(steps=(exploration,), coverage=coverage)
        )
        self.assertTrue(
            all(not identifiers for identifiers in deliberate_only.values())
        )

        reused_as_main = COLD_START_LAB._unlabeled_scope_excursions(
            SimpleNamespace(
                steps=(exploration, main),
                coverage=coverage,
            )
        )
        self.assertTrue(
            all(identifiers for identifiers in reused_as_main.values())
        )

    def test_small_agents_audit_is_deterministic_and_integral(self) -> None:
        artifact = COLD_START_LAB.run_cold_start_audit(
            corpus=ROOT / "corpus" / "ai_curriculum.json",
            topics=("t_llm_agents",),
            seeds=(0,),
            max_steps=4,
            replicate=True,
        )

        self.assertEqual(artifact["status"], "ok", artifact)
        self.assertTrue(artifact["replication_checked"])
        self.assertTrue(artifact["deterministic_replication"])
        self.assertEqual(len(artifact["runs"]), 3)
        self.assertEqual(
            {run["profile"] for run in artifact["runs"]},
            {"credible-correct", "credible-wrong", "abstaining"},
        )
        self.assertTrue(
            all(artifact["hard_invariants"].values()),
            artifact["hard_invariants"],
        )
        self.assertEqual(
            artifact["primary_signature"],
            artifact["replication_signature"],
        )
        self.assertIn("policy", artifact["backend_versions"])
        self.assertIn("learner_model", artifact["backend_versions"])
        self.assertIn(
            "simulation_feedback_protocol",
            artifact["backend_versions"],
        )
        for run in artifact["runs"]:
            self.assertEqual(run["attempted"], len(run["steps"]))
            self.assertEqual(
                run["abstained"],
                sum(
                    step["selected_option_id"] is None
                    for step in run["steps"]
                ),
            )
            self.assertEqual(
                run["deliberate_exploration_questions"],
                sum(
                    step["pedagogical_role"] == "exploration_probe"
                    for step in run["steps"]
                ),
            )
            self.assertEqual(
                run["predicted_success"]["learn_phase_count"],
                sum(
                    step["phase_before"] == "learn"
                    for step in run["steps"]
                ),
            )
            self.assertEqual(
                len(run["behavior_signature"]),
                64,
            )
            self.assertEqual(
                len(run["semantic_projection_signature"]["sha256"]),
                64,
            )
            self.assertEqual(
                run["candidate_inventory"][
                    "complete_ranked_inventory_steps"
                ],
                sum(
                    step["candidate_inventory"][
                        "ranked_inventory_complete"
                    ]
                    for step in run["steps"]
                ),
            )
            self.assertTrue(
                run["steps"][0]["candidate_inventory"][
                    "ranked_inventory_complete"
                ]
            )
            for step in run["steps"]:
                inventory = step["candidate_inventory"]
                self.assertEqual(
                    inventory["stored_ranked_prefix_count"],
                    min(
                        COLD_START_LAB.CANDIDATE_PREFIX_LIMIT,
                        inventory["eligible_scored_candidate_count"],
                    ),
                )
                if inventory["ranked_inventory_complete"]:
                    self.assertTrue(
                        inventory[
                            "rank_and_quantized_coverage_digest_verified"
                        ]
                    )
                    self.assertEqual(
                        inventory["unobserved_candidate_count"], 0
                    )
                    self.assertEqual(
                        inventory["complete_inventory_difficulty"],
                        inventory["ranked_prefix_difficulty"],
                    )
                else:
                    self.assertIsNone(
                        inventory[
                            "rank_and_quantized_coverage_digest_verified"
                        ]
                    )
                    self.assertGreater(
                        inventory["unobserved_candidate_count"], 0
                    )
                    self.assertIsNone(
                        inventory["complete_inventory_difficulty"]
                    )
                self.assertGreaterEqual(
                    inventory["ranked_prefix_difficulty"][
                        "selected_minus_minimum"
                    ],
                    0.0,
                )
                self.assertLessEqual(
                    inventory["selected_rank"],
                    inventory["sampling_frontier_count"],
                )
                self.assertEqual(
                    inventory["ranked_candidates"][
                        inventory["selected_rank"] - 1
                    ]["question_id"],
                    step["question_id"],
                )
        abstaining = next(
            run
            for run in artifact["runs"]
            if run["profile"] == "abstaining"
        )
        self.assertEqual(abstaining["abstained"], abstaining["attempted"])
        self.assertTrue(
            all(
                step["selected_option_id"] is None
                for step in abstaining["steps"]
            )
        )

    def test_unreplicated_audit_is_not_labeled_deterministic(self) -> None:
        artifact = COLD_START_LAB.run_cold_start_audit(
            corpus=ROOT / "corpus" / "ai_curriculum.json",
            topics=("t_llm_agents",),
            seeds=(0,),
            max_steps=1,
            replicate=False,
        )

        self.assertEqual(artifact["status"], "ok", artifact)
        self.assertFalse(artifact["replication_checked"])
        self.assertIsNone(artifact["deterministic_replication"])
        self.assertNotIn(
            "deterministic_replication",
            artifact["hard_invariants"],
        )

    def test_broad_llm_candidate_trace_labels_truncated_prefix(self) -> None:
        artifact = COLD_START_LAB.run_cold_start_audit(
            corpus=ROOT / "corpus" / "ai_curriculum.json",
            topics=("t_large_language_models",),
            seeds=(0,),
            max_steps=1,
            replicate=True,
        )

        self.assertTrue(artifact["deterministic_replication"])
        for run in artifact["runs"]:
            inventory = run["steps"][0]["candidate_inventory"]
            self.assertGreater(
                inventory["eligible_scored_candidate_count"],
                COLD_START_LAB.CANDIDATE_PREFIX_LIMIT,
            )
            self.assertEqual(
                inventory["stored_ranked_prefix_count"],
                COLD_START_LAB.CANDIDATE_PREFIX_LIMIT,
            )
            self.assertFalse(inventory["ranked_inventory_complete"])
            self.assertIsNone(
                inventory[
                    "rank_and_quantized_coverage_digest_verified"
                ]
            )
            self.assertIsNone(
                inventory["complete_inventory_difficulty"]
            )
            self.assertIsNotNone(
                inventory["ranked_prefix_difficulty"]
            )
            self.assertEqual(
                inventory["unobserved_candidate_count"],
                inventory["eligible_scored_candidate_count"]
                - COLD_START_LAB.CANDIDATE_PREFIX_LIMIT,
            )

    def test_candidate_trace_tampering_fails_closed(self) -> None:
        def score(total: float) -> dict[str, float | int]:
            return {
                "total": total,
                "predicted_correct": 0.4,
                "information_gain": 0.1,
                "learning_fit": 0.2,
                "concept_need": 0.3,
                "misconception_value": 0.0,
                "prerequisite_value": 0.0,
                "review_value": 0.0,
                "novelty": 1.0,
                "kind_fit": 1.0,
                "continuity": 0.5,
                "boundary_fit": 0.25,
                "coverage_raw_exposures": 0,
                "coverage_diagnostic_information": 0.0,
                "coverage_successful_retrieval_families": 0,
            }

        first = score(1.0)
        second = score(0.5)
        when = datetime(2110, 1, 1, tzinfo=timezone.utc)
        step = SimpleNamespace(
            question_id="q1",
            family_id="family-q1",
            question_kind="conceptual",
            learning_objective_id=None,
            surface_primary_concept_id="concept",
            phase_before=SimpleNamespace(value="learn"),
            pedagogical_role="main",
            selected_at=when,
            answered_at=when,
            predicted_correct=0.4,
        )
        metadata = {
            question_id: {
                "family_id": f"family-{question_id}",
                "learning_objective_id": None,
                "primary_concept_id": "concept",
                "question_kind": "conceptual",
                "difficulty": difficulty,
                "corpus_release_id": "release",
                "global_status": "approved",
                "release_status": "approved",
                "evidence_weight": 1.0,
                "revoked": False,
                "runtime_activation_safe": True,
            }
            for question_id, difficulty in (("q1", -1.0), ("q2", 0.0))
        }
        top_candidates = [
            {"question_id": "q1", **first},
            {"question_id": "q2", **second},
        ]
        base_row = {
            "attempted_question_id": "q1",
            "selected_question_id": "q1",
            "selected_family_id": "family-q1",
            "selected_question_kind": "conceptual",
            "selected_objective_id": None,
            "corpus_release_id": "release",
            "selected_question_status": "approved",
            "selected_evidence_weight": 1.0,
            "phase": "learn",
            "pedagogical_role": "main",
            "selected_at": when.isoformat(),
            "answered_at": when.isoformat(),
            "candidate_count": 2,
            "candidate_digest": COLD_START_LAB._ranked_candidate_digest(
                (("q1", first), ("q2", second))
            ),
            "top_candidates_json": json.dumps(top_candidates),
            "selected_score_json": json.dumps(first),
        }
        valid_inventory = (
            COLD_START_LAB._candidate_inventory_from_row(
                row=base_row,
                step=step,
                question_metadata=metadata,
                trace_label="valid",
            )
        )
        self.assertTrue(
            valid_inventory[
                "rank_and_quantized_coverage_digest_verified"
            ]
        )

        def assert_invalid(
            row: dict[str, object],
            *,
            committed_step: object = step,
            known_metadata: dict[str, dict[str, object]] = metadata,
        ) -> None:
            with self.assertRaises(
                COLD_START_LAB.ColdStartInvariantError
            ):
                COLD_START_LAB._candidate_inventory_from_row(
                    row=row,
                    step=committed_step,
                    question_metadata=known_metadata,
                    trace_label="tampered",
                )

        misaligned = dict(base_row)
        misaligned["phase"] = "verify"
        assert_invalid(misaligned)

        metadata_mismatch = dict(base_row)
        metadata_mismatch["selected_family_id"] = "family-other"
        assert_invalid(metadata_mismatch)

        conflicting_metadata = json.loads(json.dumps(metadata))
        conflicting_metadata["q1"]["family_id"] = "family-other"
        assert_invalid(
            base_row,
            known_metadata=conflicting_metadata,
        )

        unsafe_metadata = json.loads(json.dumps(metadata))
        unsafe_metadata["q2"]["global_status"] = "quarantined"
        unsafe_metadata["q2"]["release_status"] = "quarantined"
        unsafe_metadata["q2"]["evidence_weight"] = 0.0
        assert_invalid(
            base_row,
            known_metadata=unsafe_metadata,
        )

        revoked_metadata = json.loads(json.dumps(metadata))
        revoked_metadata["q2"]["revoked"] = True
        assert_invalid(
            base_row,
            known_metadata=revoked_metadata,
        )

        unsafe_generated_metadata = json.loads(json.dumps(metadata))
        unsafe_generated_metadata["q2"][
            "runtime_activation_safe"
        ] = False
        assert_invalid(
            base_row,
            known_metadata=unsafe_generated_metadata,
        )

        bad_digest = dict(base_row)
        bad_digest["candidate_digest"] = "0" * 64
        assert_invalid(bad_digest)

        rank_disorder = dict(base_row)
        rank_disorder["top_candidates_json"] = json.dumps(
            list(reversed(top_candidates))
        )
        assert_invalid(rank_disorder)

        duplicate_candidate = dict(base_row)
        duplicate_candidate["top_candidates_json"] = json.dumps(
            [top_candidates[0], top_candidates[0]]
        )
        assert_invalid(duplicate_candidate)

        unknown_candidate = dict(base_row)
        unknown_candidates = json.loads(
            unknown_candidate["top_candidates_json"]
        )
        unknown_candidates[1]["question_id"] = "q-not-in-corpus"
        unknown_candidate["top_candidates_json"] = json.dumps(
            unknown_candidates
        )
        assert_invalid(unknown_candidate)

        prefix_mismatch = dict(base_row)
        prefix_mismatch["candidate_count"] = 3
        assert_invalid(prefix_mismatch)

        selected_score_mismatch = dict(base_row)
        mismatched_score = dict(first)
        mismatched_score["information_gain"] = 0.11
        selected_score_mismatch["selected_score_json"] = json.dumps(
            mismatched_score
        )
        assert_invalid(selected_score_mismatch)

        component_tamper = dict(base_row)
        altered_candidates = json.loads(
            component_tamper["top_candidates_json"]
        )
        altered_candidates[1]["information_gain"] = 0.2
        component_tamper["top_candidates_json"] = json.dumps(
            altered_candidates
        )
        altered_inventory = (
            COLD_START_LAB._candidate_inventory_from_row(
                row=component_tamper,
                step=step,
                question_metadata=metadata,
                trace_label="component boundary",
            )
        )
        self.assertTrue(
            altered_inventory[
                "rank_and_quantized_coverage_digest_verified"
            ]
        )
        self.assertNotEqual(
            altered_inventory["stored_ranked_prefix_sha256"],
            valid_inventory["stored_ranked_prefix_sha256"],
        )

        quantized_float_boundary = dict(base_row)
        quantized_candidates = json.loads(
            quantized_float_boundary["top_candidates_json"]
        )
        quantized_candidates[1]["total"] += 1e-10
        quantized_float_boundary["top_candidates_json"] = json.dumps(
            quantized_candidates
        )
        quantized_inventory = (
            COLD_START_LAB._candidate_inventory_from_row(
                row=quantized_float_boundary,
                step=step,
                question_metadata=metadata,
                trace_label="quantized float boundary",
            )
        )
        self.assertTrue(
            quantized_inventory[
                "rank_and_quantized_coverage_digest_verified"
            ]
        )
        self.assertNotEqual(
            quantized_inventory["stored_ranked_prefix_sha256"],
            valid_inventory["stored_ranked_prefix_sha256"],
        )

        negative_coverage = dict(base_row)
        invalid_candidates = json.loads(
            negative_coverage["top_candidates_json"]
        )
        invalid_candidates[1][
            "coverage_diagnostic_information"
        ] = -0.1
        negative_coverage["top_candidates_json"] = json.dumps(
            invalid_candidates
        )
        assert_invalid(negative_coverage)

        impossible_successful_families = dict(base_row)
        invalid_candidates = json.loads(
            impossible_successful_families["top_candidates_json"]
        )
        invalid_candidates[1]["coverage_raw_exposures"] = 0
        invalid_candidates[1][
            "coverage_successful_retrieval_families"
        ] = 1
        impossible_successful_families["top_candidates_json"] = json.dumps(
            invalid_candidates
        )
        assert_invalid(impossible_successful_families)

        six_scores = [score(float(6 - index)) for index in range(6)]
        six_metadata = {
            f"q{index + 1}": {
                "family_id": f"family-q{index + 1}",
                "learning_objective_id": None,
                "primary_concept_id": "concept",
                "question_kind": "conceptual",
                "difficulty": float(index),
                "corpus_release_id": "release",
                "global_status": "approved",
                "release_status": "approved",
                "evidence_weight": 1.0,
                "revoked": False,
                "runtime_activation_safe": True,
            }
            for index in range(6)
        }
        six_candidates = [
            {
                "question_id": f"q{index + 1}",
                **six_scores[index],
            }
            for index in range(6)
        ]
        outside_frontier = {
            **base_row,
            "attempted_question_id": "q6",
            "selected_question_id": "q6",
            "selected_family_id": "family-q6",
            "candidate_count": 6,
            "candidate_digest": COLD_START_LAB._ranked_candidate_digest(
                tuple(
                    (f"q{index + 1}", six_scores[index])
                    for index in range(6)
                )
            ),
            "top_candidates_json": json.dumps(six_candidates),
            "selected_score_json": json.dumps(six_scores[-1]),
        }
        outside_step = SimpleNamespace(
            **{
                **vars(step),
                "question_id": "q6",
                "family_id": "family-q6",
                "predicted_correct": 0.4,
            }
        )
        assert_invalid(
            outside_frontier,
            committed_step=outside_step,
            known_metadata=six_metadata,
        )


if __name__ == "__main__":
    unittest.main()
