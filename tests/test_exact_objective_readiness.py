# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from tsq.adaptive import RecursiveEvidenceBoundary
from tsq.graph import KnowledgeGraph
from tsq.learner import (
    MODEL_VERSION,
    LearnerModel,
    ObjectiveReadinessFloor,
)
from tsq.models import (
    Concept,
    ConceptEdge,
    LearningObjective,
    ObjectiveOperation,
    ObjectiveState,
    RelationType,
    SkillState,
)
from tsq.objective_posterior import (
    LikelihoodObservation,
    ObjectivePosterior,
)


NOW = datetime(2100, 7, 23, 12, 0, tzinfo=timezone.utc)


def objective(
    objective_id: str,
    concept_id: str,
) -> LearningObjective:
    return LearningObjective(
        id=objective_id,
        name=f"Objective {objective_id}",
        description=f"Assess {objective_id}.",
        primary_concept_id=concept_id,
        supporting_concept_ids=(),
        operation=ObjectiveOperation.PREDICT,
        prior_mastery=0.20,
    )


def exact_state(
    objective_id: str,
    outcomes: tuple[bool, ...],
    *,
    difficulty: float,
    discrimination: float,
) -> ObjectiveState:
    posterior = ObjectivePosterior.from_prior(0.20).with_observations(
        LikelihoodObservation(
            observation_id=f"{objective_id}_observation_{index}",
            family_id=f"{objective_id}_family_{index}",
            difficulty=difficulty,
            discrimination=discrimination,
            guess_rate=0.25,
            slip_rate=0.07,
            option_count=4,
            correct=correct,
            evidence_power=0.65,
        )
        for index, correct in enumerate(outcomes)
    )
    metrics = posterior.metrics()
    return ObjectiveState(
        learner_id="learner",
        objective_id=objective_id,
        mean=metrics.mean,
        variance=metrics.variance,
        stability_hours=48.0,
        exposures=len(outcomes),
        last_seen_at=NOW,
        evidence_mass=metrics.evidence_mass,
        posterior=posterior,
        model_version=MODEL_VERSION,
    )


def skill_state(
    concept_id: str,
    *,
    mean: float = 0.0,
    variance: float = 0.4,
) -> SkillState:
    return SkillState(
        learner_id="learner",
        concept_id=concept_id,
        mean=mean,
        variance=variance,
        stability_hours=48.0,
        exposures=3,
        evidence_mass=4.0,
    )


class ExactObjectiveReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = LearnerModel()
        self.planner = RecursiveEvidenceBoundary(self.model)

    def test_non_gaussian_objective_metrics_reach_readiness_exactly(self) -> None:
        concept = Concept("foundation", "Foundation", "A foundation.")
        learning_objective = objective("lo_foundation", concept.id)
        projected = exact_state(
            learning_objective.id,
            (False, True, True, True, True, True, True, True),
            difficulty=0.8,
            discrimination=2.2,
        )
        floor_projection = (
            self.model.concept_projection_with_objective_floor(
                learner_id="learner",
                concepts={concept.id: concept},
                stored_states={},
                objectives=(learning_objective,),
                stored_objective_states={},
                projected_objective_states={
                    learning_objective.id: projected
                },
                now=NOW,
            )
        )
        moment_state = floor_projection.states[concept.id]
        exact_floor = floor_projection.exact_floors[concept.id]
        graph = KnowledgeGraph((concept,), ())
        readiness = self.planner.readiness_map(
            learner_id="learner",
            graph=graph,
            stored_states=floor_projection.states,
            intrinsic_overrides=floor_projection.exact_floors,
            now=NOW,
        )[concept.id]

        exact_intrinsic = (
            0.55 * projected.mastery_probability
            + 0.45 * projected.expected_competence
        )
        moment_intrinsic = (
            0.55 * moment_state.mastery_probability
            + 0.45 * moment_state.expected_competence
        )
        self.assertAlmostEqual(
            projected.mastery_probability,
            0.7440885359648799,
            places=12,
        )
        self.assertAlmostEqual(
            projected.expected_competence,
            0.6991041295141082,
            places=12,
        )
        self.assertGreater(exact_intrinsic - moment_intrinsic, 0.08)
        self.assertEqual(
            exact_floor.source_objective_id,
            learning_objective.id,
        )
        self.assertAlmostEqual(
            readiness.mastery_probability,
            projected.mastery_probability,
            places=12,
        )
        self.assertAlmostEqual(
            readiness.expected_competence,
            projected.expected_competence,
            places=12,
        )
        self.assertAlmostEqual(
            readiness.intrinsic_readiness,
            exact_intrinsic,
            places=12,
        )
        self.assertEqual(
            readiness.objective_floor_source_id,
            learning_objective.id,
        )

    def test_weakest_objective_is_chosen_by_exact_not_gaussian_order(self) -> None:
        concept = Concept("shared", "Shared", "A shared broad concept.")
        exact_weaker = objective("lo_exact_weaker", concept.id)
        gaussian_weaker = objective("lo_gaussian_weaker", concept.id)
        exact_weaker_state = exact_state(
            exact_weaker.id,
            (True,) * 10,
            difficulty=-0.5,
            discrimination=3.0,
        )
        gaussian_weaker_state = exact_state(
            gaussian_weaker.id,
            (False,) * 3 + (True,) * 9,
            difficulty=0.8,
            discrimination=3.0,
        )

        def exact_readiness(state: ObjectiveState) -> float:
            return (
                0.55 * state.mastery_probability
                + 0.45 * state.expected_competence
            )

        def gaussian_readiness(state: ObjectiveState) -> float:
            moments = SkillState(
                learner_id=state.learner_id,
                concept_id=concept.id,
                mean=state.mean,
                variance=state.variance,
                stability_hours=state.stability_hours,
            )
            return (
                0.55 * moments.mastery_probability
                + 0.45 * moments.expected_competence
            )

        self.assertLess(
            exact_readiness(exact_weaker_state),
            exact_readiness(gaussian_weaker_state),
        )
        self.assertGreater(
            gaussian_readiness(exact_weaker_state),
            gaussian_readiness(gaussian_weaker_state),
        )
        floor_projection = (
            self.model.concept_projection_with_objective_floor(
                learner_id="learner",
                concepts={concept.id: concept},
                stored_states={},
                objectives=(exact_weaker, gaussian_weaker),
                stored_objective_states={},
                projected_objective_states={
                    exact_weaker.id: exact_weaker_state,
                    gaussian_weaker.id: gaussian_weaker_state,
                },
                now=NOW,
            )
        )

        floor = floor_projection.exact_floors[concept.id]
        state = floor_projection.states[concept.id]
        self.assertEqual(floor.source_objective_id, exact_weaker.id)
        self.assertAlmostEqual(
            floor.mastery_probability,
            exact_weaker_state.mastery_probability,
            places=12,
        )
        self.assertAlmostEqual(state.mean, exact_weaker_state.mean, places=12)
        self.assertAlmostEqual(
            state.variance,
            exact_weaker_state.variance,
            places=12,
        )

    def test_exact_override_propagates_recursively_through_graph(self) -> None:
        concepts = (
            Concept("foundation", "Foundation", "A foundation."),
            Concept("middle", "Middle", "An intermediate concept."),
            Concept("advanced", "Advanced", "An advanced concept."),
        )
        graph = KnowledgeGraph(
            concepts,
            (
                ConceptEdge(
                    "foundation",
                    "middle",
                    RelationType.PREREQUISITE,
                    1.0,
                ),
                ConceptEdge(
                    "middle",
                    "advanced",
                    RelationType.PREREQUISITE,
                    1.0,
                ),
            ),
        )
        learning_objective = objective("lo_foundation", "foundation")
        projected = exact_state(
            learning_objective.id,
            (False, True, True, True, True, True, True, True),
            difficulty=0.8,
            discrimination=2.2,
        )
        floor_projection = (
            self.model.concept_projection_with_objective_floor(
                learner_id="learner",
                concepts=graph.concepts,
                stored_states={
                    "middle": skill_state("middle", mean=3.0),
                    "advanced": skill_state("advanced", mean=3.0),
                },
                objectives=(learning_objective,),
                stored_objective_states={},
                projected_objective_states={
                    learning_objective.id: projected
                },
                now=NOW,
            )
        )
        readiness = self.planner.readiness_map(
            learner_id="learner",
            graph=graph,
            stored_states=floor_projection.states,
            intrinsic_overrides=floor_projection.exact_floors,
            now=NOW,
        )

        foundation = readiness["foundation"]
        middle = readiness["middle"]
        advanced = readiness["advanced"]
        self.assertAlmostEqual(
            middle.prerequisite_support,
            foundation.effective_readiness,
            places=12,
        )
        self.assertAlmostEqual(
            middle.effective_readiness,
            middle.intrinsic_readiness
            * (0.40 + 0.60 * foundation.effective_readiness),
            places=12,
        )
        self.assertAlmostEqual(
            advanced.prerequisite_support,
            middle.effective_readiness,
            places=12,
        )
        self.assertEqual(middle.bottleneck_concept_id, "foundation")
        self.assertEqual(advanced.bottleneck_concept_id, "foundation")

    def test_unknown_exact_override_fails_closed(self) -> None:
        concept = Concept("known", "Known", "A known concept.")
        graph = KnowledgeGraph((concept,), ())

        with self.assertRaisesRegex(
            ValueError,
            "Unknown objective readiness overrides: ghost",
        ):
            self.planner.readiness_map(
                learner_id="learner",
                graph=graph,
                stored_states={"known": skill_state("known")},
                intrinsic_overrides={
                    "ghost": ObjectiveReadinessFloor(
                        source_objective_id="lo_ghost",
                        mastery_probability=0.20,
                        expected_competence=0.30,
                    )
                },
                now=NOW,
            )

    def test_direct_boundary_forwards_exact_overrides(self) -> None:
        concepts = (
            Concept("z_weak", "Weak", "The weaker prerequisite."),
            Concept("a_strong", "Strong", "The stronger prerequisite."),
            Concept("target", "Target", "The dependent concept."),
        )
        graph = KnowledgeGraph(
            concepts,
            (
                ConceptEdge(
                    "z_weak",
                    "target",
                    RelationType.PREREQUISITE,
                    1.0,
                ),
                ConceptEdge(
                    "a_strong",
                    "target",
                    RelationType.PREREQUISITE,
                    1.0,
                ),
            ),
        )
        states = {
            "z_weak": skill_state("z_weak"),
            "a_strong": skill_state("a_strong"),
        }
        tied = self.planner.choose_direct_boundary(
            learner_id="learner",
            focus_concept_id="target",
            graph=graph,
            stored_states=states,
            now=NOW,
        )
        overridden = self.planner.choose_direct_boundary(
            learner_id="learner",
            focus_concept_id="target",
            graph=graph,
            stored_states=states,
            intrinsic_overrides={
                "z_weak": ObjectiveReadinessFloor(
                    source_objective_id="lo_z_weak",
                    mastery_probability=0.05,
                    expected_competence=0.15,
                ),
                "a_strong": ObjectiveReadinessFloor(
                    source_objective_id="lo_a_strong",
                    mastery_probability=0.90,
                    expected_competence=0.85,
                ),
            },
            now=NOW,
        )

        self.assertIsNotNone(tied)
        self.assertIsNotNone(overridden)
        assert tied is not None
        assert overridden is not None
        self.assertEqual(tied.selected_concept_id, "a_strong")
        self.assertEqual(overridden.selected_concept_id, "z_weak")
        self.assertGreater(
            overridden.selected.score,
            next(
                candidate.score
                for candidate in overridden.candidates
                if candidate.concept_id == "a_strong"
            ),
        )


if __name__ == "__main__":
    unittest.main()
