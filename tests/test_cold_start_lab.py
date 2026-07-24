# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import importlib.util
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
