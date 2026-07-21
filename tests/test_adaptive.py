# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tsq.adaptive import (
    BOUNDARY_ALGORITHM_VERSION,
    RecursiveEvidenceBoundary,
)
from tsq.graph import KnowledgeGraph
from tsq.learner import LearnerModel
from tsq.models import Concept, ConceptEdge, RelationType, SkillState


NOW = datetime(2100, 7, 21, 12, 0, tzinfo=timezone.utc)


def state(
    concept_id: str,
    mean: float,
    *,
    variance: float = 0.4,
    evidence_mass: float = 4.0,
) -> SkillState:
    return SkillState(
        learner_id="learner",
        concept_id=concept_id,
        mean=mean,
        variance=variance,
        stability_hours=48.0,
        exposures=3,
        evidence_mass=evidence_mass,
    )


class RecursiveEvidenceBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        concepts = (
            Concept("foundation", "Foundation", "Foundational objective."),
            Concept("middle", "Middle", "Intermediate objective."),
            Concept("alternative", "Alternative", "Parallel prerequisite."),
            Concept("advanced", "Advanced", "Advanced objective."),
        )
        edges = (
            ConceptEdge(
                "foundation", "middle", RelationType.PREREQUISITE, 1.0
            ),
            ConceptEdge("middle", "advanced", RelationType.PREREQUISITE, 1.0),
            ConceptEdge(
                "alternative", "advanced", RelationType.PREREQUISITE, 0.8
            ),
        )
        self.graph = KnowledgeGraph(concepts, edges)
        self.planner = RecursiveEvidenceBoundary(LearnerModel())

    def test_recursive_weakness_limits_downstream_readiness(self) -> None:
        weak_foundation = {
            "foundation": state("foundation", -3.0),
            "middle": state("middle", 3.0),
            "alternative": state("alternative", 0.8),
            "advanced": state("advanced", 2.0),
        }
        weak = self.planner.readiness_map(
            learner_id="learner",
            graph=self.graph,
            stored_states=weak_foundation,
            now=NOW,
        )
        strong_foundation = dict(weak_foundation)
        strong_foundation["foundation"] = state("foundation", 3.0)
        strong = self.planner.readiness_map(
            learner_id="learner",
            graph=self.graph,
            stored_states=strong_foundation,
            now=NOW,
        )

        self.assertGreater(weak["middle"].intrinsic_readiness, 0.90)
        self.assertLess(weak["middle"].effective_readiness, 0.50)
        self.assertEqual(weak["middle"].bottleneck_concept_id, "foundation")
        self.assertEqual(weak["advanced"].bottleneck_concept_id, "foundation")
        self.assertGreater(
            strong["advanced"].effective_readiness,
            weak["advanced"].effective_readiness,
        )

    def test_boundary_descends_one_edge_toward_recursive_bottleneck(self) -> None:
        states = {
            "foundation": state("foundation", -3.0),
            "middle": state("middle", 3.0),
            "alternative": state("alternative", 0.8),
            "advanced": state("advanced", 2.0),
        }

        decision = self.planner.choose_direct_boundary(
            learner_id="learner",
            focus_concept_id="advanced",
            graph=self.graph,
            stored_states=states,
            now=NOW,
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.algorithm_version, BOUNDARY_ALGORITHM_VERSION)
        self.assertEqual(decision.selected_concept_id, "middle")
        self.assertEqual(
            decision.selected.recursive_bottleneck_concept_id, "foundation"
        )
        self.assertEqual(
            [candidate.concept_id for candidate in decision.candidates],
            ["middle", "alternative"],
        )
        self.assertGreater(
            decision.candidates[0].score, decision.candidates[1].score
        )

    def test_recent_independent_failures_break_an_equal_boundary(self) -> None:
        concepts = (
            Concept("left", "Left", "Left prerequisite."),
            Concept("right", "Right", "Right prerequisite."),
            Concept("target", "Target", "Target objective."),
        )
        graph = KnowledgeGraph(
            concepts,
            (
                ConceptEdge("left", "target", RelationType.REQUIRES, 1.0),
                ConceptEdge("right", "target", RelationType.REQUIRES, 1.0),
            ),
        )
        states = {
            "left": state("left", 0.0),
            "right": state("right", 0.0),
        }

        decision = self.planner.choose_direct_boundary(
            learner_id="learner",
            focus_concept_id="target",
            graph=graph,
            stored_states=states,
            now=NOW,
            recent_performance={"left": (3, 0), "right": (3, 3)},
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.selected_concept_id, "right")
        self.assertEqual(decision.selected.recent_failure_rate, 1.0)

    def test_graph_root_has_no_deeper_boundary(self) -> None:
        decision = self.planner.choose_direct_boundary(
            learner_id="learner",
            focus_concept_id="foundation",
            graph=self.graph,
            stored_states={},
            now=NOW,
        )

        self.assertIsNone(decision)

    def test_freshly_verified_prerequisite_can_be_excluded(self) -> None:
        states = {
            "foundation": state("foundation", -3.0),
            "middle": state("middle", 3.0),
            "alternative": state("alternative", 0.8),
        }

        decision = self.planner.choose_direct_boundary(
            learner_id="learner",
            focus_concept_id="advanced",
            graph=self.graph,
            stored_states=states,
            now=NOW,
            excluded_concept_ids={"middle"},
        )

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.selected_concept_id, "alternative")
        self.assertEqual(len(decision.candidates), 1)


if __name__ == "__main__":
    unittest.main()
