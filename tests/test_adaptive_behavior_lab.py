# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from experiments.adaptive_behavior_lab import (
    aggregate_audit_gaps,
    capacity_and_demand_snapshot,
    compact_session_report,
    planned_check_status,
    projection_summary,
    run_delayed_family_retrieval_check,
    run_misconception_recovery_check,
    serialize_trace,
)
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.simulation import BehavioralSimulator, SyntheticLearner
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2102, 4, 5, 9, 0, tzinfo=timezone.utc)


class AdaptiveBehaviorLabTests(unittest.TestCase):
    def test_named_misconception_can_be_induced_and_recovered(self) -> None:
        for seed in (3, 79, 97):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as directory:
                result = run_misconception_recovery_check(
                    database_path=Path(directory) / "recovery.db",
                    corpus_path=CORPUS,
                    seed=seed,
                )

                path = result["probability_path"]
                self.assertFalse(result["failures"], result["failures"])
                self.assertIn(result["status"], {"passed", "partial"})
                self.assertEqual(
                    result["ok"], result["status"] == "passed"
                )
                if result["gap_records"]:
                    self.assertEqual(result["status"], "partial")
                    self.assertFalse(result["ok"])
                else:
                    self.assertEqual(
                        result["completed_attempts"],
                        result["planned_attempts"],
                    )
                self.assertGreaterEqual(path["after_induction"], 0.35)
                self.assertGreater(
                    path["after_induction"], path["after_first_recovery"]
                )
                self.assertGreater(path["after_first_recovery"], path["final"])
                self.assertGreaterEqual(
                    len(path["after_recovery_sessions"]), 2
                )
                self.assertLess(path["final"], 0.35)
                self.assertTrue(result["active_after_induction"])
                self.assertFalse(result["active_after_recovery"])
                self.assertTrue(result["integrity"]["ok"])

    def test_delayed_family_probe_revisits_and_certifies_due_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_delayed_family_retrieval_check(
                database_path=Path(directory) / "delayed.db",
                corpus_path=CORPUS,
                seed=71,
            )

        self.assertTrue(result["ok"], result["failures"])
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["gap_records"])
        self.assertGreaterEqual(result["session_count"], 2)
        self.assertTrue(result["repeated_families"])
        self.assertTrue(
            set(result["repeated_families"]).issubset(
                result["delayed_certified_families"]
            )
        )
        self.assertTrue(result["integrity"]["ok"])

    def test_planned_check_status_and_gap_aggregation_are_fail_closed(self) -> None:
        ordinary_gap = {
            "step": 2,
            "phase": "learn",
            "category": "corpus_gap",
            "message": "ordinary gap",
        }
        special_gap = {
            "segment": "recovery_1",
            "step": 3,
            "phase": "learn",
            "category": "corpus_gap",
            "message": "special gap",
        }

        self.assertEqual(planned_check_status([], []), "passed")
        self.assertEqual(
            planned_check_status([], [special_gap]), "partial"
        )
        self.assertEqual(
            planned_check_status(["integrity failed"], [special_gap]),
            "failed",
        )
        self.assertEqual(
            aggregate_audit_gaps(
                ({"id": "ordinary", "gaps": [ordinary_gap]},),
                (
                    {
                        "id": "misconception_recovery",
                        "gap_records": [special_gap],
                    },
                ),
            ),
            [
                {"scenario": "ordinary", **ordinary_gap},
                {
                    "scenario": "misconception_recovery",
                    **special_gap,
                },
            ],
        )

    def test_objective_projection_and_session_details_are_not_collapsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "lab.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            engine = AdaptiveEngine(database)
            report = BehavioralSimulator(engine).run(
                SyntheticLearner(
                    "lab-objectives",
                    default_objective_ability=0.90,
                    forced_correctness=True,
                    confidence_override=0.95,
                    base_response_ms=4_000,
                    seed=51,
                ),
                learner_id="lab-objectives",
                root_concept_id="t_transformers",
                policy_seed=51,
                max_steps=3,
                start_at=START,
            )
            with database.read() as connection:
                session_id = connection.execute(
                    "SELECT id FROM sessions WHERE learner_id='lab-objectives'"
                ).fetchone()["id"]
                objective_evidence = connection.execute(
                    """SELECT COALESCE(SUM(evidence_mass), 0.0) AS mass
                       FROM objective_states WHERE learner_id='lab-objectives'"""
                ).fetchone()["mass"]
                objective_families = connection.execute(
                    """SELECT COUNT(*) AS n FROM learner_objective_families
                       WHERE learner_id='lab-objectives'"""
                ).fetchone()["n"]
            session = database.get_session(session_id)
            persisted = engine.session_report(session_id, now=report.ended_at)
            projection = projection_summary(
                engine.profile(
                    "lab-objectives",
                    root_concept_id="t_transformers",
                    now=report.ended_at,
                ),
                database=database,
                learner_id="lab-objectives",
            )
            compact = compact_session_report(persisted)
            trace, violations = serialize_trace(
                report, persisted, database, session
            )
            capacity = capacity_and_demand_snapshot(database, session)

            self.assertEqual(projection["legacy_evidence_mass"], 0.0)
            self.assertEqual(
                projection["objective_evidence_mass"], objective_evidence
            )
            self.assertEqual(
                projection["objective_independent_families"],
                objective_families,
            )
            self.assertGreater(projection["persisted_objective_count"], 0)
            self.assertEqual(len(projection["learning_objectives"]), 8)
            self.assertTrue(
                projection["database_projection_commitment"][
                    "matches_latest_event"
                ]
            )
            self.assertIn("objective_changes", compact)
            self.assertIn("objective_performance", compact)
            self.assertIn("family_evidence_definitions", compact)
            self.assertTrue(compact["objective_changes"])
            self.assertTrue(compact["objective_performance"])
            self.assertFalse(violations)
            self.assertTrue(trace)
            self.assertTrue(
                all(
                    step["question"]["surface_primary_concept_id"]
                    and step["question"]["evidence_anchor_concept_id"]
                    for step in trace
                )
            )
            self.assertEqual(len(capacity["owned_objectives"]), 8)
            self.assertIn("remaining_owned_evidence_families", capacity)


if __name__ == "__main__":
    unittest.main()
