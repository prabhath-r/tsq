# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import tsq.objective_posterior as posterior_module
from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.learner import MODEL_VERSION, LearnerModel
from tsq.models import SessionPhase
from tsq.objective_posterior import LikelihoodObservation, ObjectivePosterior
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
NOW = datetime(2103, 4, 5, 12, 0, tzinfo=timezone.utc)


class ObjectivePosteriorCacheTestCase(unittest.TestCase):
    @staticmethod
    def _posterior() -> ObjectivePosterior:
        return ObjectivePosterior.from_prior(0.20).with_observation(
            LikelihoodObservation(
                observation_id="cache_observation",
                family_id="cache_family",
                difficulty=0.4,
                discrimination=1.7,
                guess_rate=0.25,
                slip_rate=0.08,
                option_count=4,
                correct=False,
                evidence_power=0.65,
            )
        )

    @staticmethod
    def _legacy_metrics(
        log_density: tuple[float, ...],
        *,
        evidence_mass: float,
        acquisition_mass: float,
    ):
        """Reproduce the pre-cache independent-integration sequence."""

        mean = posterior_module._integrate_function(
            log_density, posterior_module.THETA_GRID
        )
        second_moment = posterior_module._integrate_function(
            log_density,
            (
                theta * theta
                for theta in posterior_module.THETA_GRID
            ),
        )
        variance = max(0.0, second_moment - mean * mean)
        threshold = posterior_module._logit(
            posterior_module.DEFAULT_MASTERY_THRESHOLD
        )
        mastery = posterior_module._probability_above_theta(
            log_density, threshold
        )
        coarse_mastery = (
            posterior_module._probability_above_theta_on_stride(
                log_density,
                threshold,
                stride=2,
            )
        )
        mastery_error_bound = min(
            1.0,
            max(
                posterior_module.MASTERY_PROBABILITY_ERROR_BOUND,
                abs(mastery - coarse_mastery),
            ),
        )
        competence = posterior_module._integrate_function(
            log_density,
            (
                posterior_module._sigmoid(theta)
                for theta in posterior_module.THETA_GRID
            ),
        )
        edge = posterior_module._edge_mass(log_density)
        return posterior_module.PosteriorMetrics(
            mean=mean,
            variance=variance,
            mastery_probability=mastery,
            expected_competence=competence,
            edge_mass=edge,
            evidence_mass=evidence_mass,
            acquisition_mass=acquisition_mass,
            mastery_probability_error_bound=mastery_error_bound,
        )

    def test_cached_derivatives_preserve_canonical_identity(self) -> None:
        posterior = self._posterior()
        encoded_before = posterior.encode()
        digest_before = hashlib.sha256(encoded_before).hexdigest()

        with mock.patch.object(
            posterior_module,
            "_apply_observations",
            wraps=posterior_module._apply_observations,
        ) as apply_observations:
            first_density = posterior.log_density
            second_density = posterior.log_density
        self.assertIs(first_density, second_density)
        apply_observations.assert_not_called()

        with mock.patch.object(
            posterior_module,
            "_metrics_for_density",
            wraps=posterior_module._metrics_for_density,
        ) as metrics_for_density:
            first_metrics = posterior.metrics()
            second_metrics = posterior.metrics()
        self.assertIs(first_metrics, second_metrics)
        metrics_for_density.assert_called_once()

        self.assertEqual(posterior.encode(), encoded_before)
        self.assertEqual(posterior.digest, digest_before)
        self.assertNotIn("_log_density_cache", repr(posterior))
        self.assertNotIn("_metrics_cache", repr(posterior))

        decoded = ObjectivePosterior.decode(
            encoded_before, expected_digest=digest_before
        )
        self.assertEqual(decoded, posterior)
        self.assertEqual(hash(decoded), hash(posterior))
        self.assertEqual(decoded.encode(), encoded_before)
        self.assertEqual(decoded.metrics(), first_metrics)

    def test_expected_information_reuses_current_metrics(self) -> None:
        posterior = self._posterior()
        posterior.metrics()

        with mock.patch.object(
            posterior_module,
            "_metrics_for_density",
            wraps=posterior_module._metrics_for_density,
        ) as metrics_for_density:
            posterior.expected_information(
                difficulty=0.7,
                discrimination=1.8,
                guess_rate=0.25,
                slip_rate=0.08,
                option_count=4,
                evidence_power=0.75,
            )

        # Only the two hypothetical outcome densities require new metrics; the
        # current posterior's variance comes from its immutable cache.
        self.assertEqual(metrics_for_density.call_count, 2)

    def test_expected_information_is_bit_exact_to_legacy_grid_sequence(
        self,
    ) -> None:
        posterior = self._posterior()
        parameters = {
            "difficulty": 0.7,
            "discrimination": 1.8,
            "guess_rate": 0.25,
            "slip_rate": 0.08,
            "option_count": 4,
            "evidence_power": 0.75,
        }
        current = posterior.log_density
        predicted_correct = posterior_module._integrate_function(
            current,
            (
                posterior_module._response_probability(
                    theta,
                    difficulty=parameters["difficulty"],
                    discrimination=parameters["discrimination"],
                    guess_rate=parameters["guess_rate"],
                    slip_rate=parameters["slip_rate"],
                    option_count=parameters["option_count"],
                )
                for theta in posterior_module.THETA_GRID
            ),
        )
        factors = tuple(
            LikelihoodObservation(
                observation_id=(
                    "expected_information_correct"
                    if correct
                    else "expected_information_incorrect"
                ),
                family_id="expected_information_family",
                correct=correct,
                **parameters,
            )
            for correct in (True, False)
        )
        densities = tuple(
            posterior_module._apply_observations(current, (factor,))
            for factor in factors
        )
        metrics = tuple(
            self._legacy_metrics(
                density,
                evidence_mass=(
                    posterior.evidence_mass + parameters["evidence_power"]
                ),
                acquisition_mass=posterior.acquisition_mass,
            )
            for density in densities
        )
        for density, expected_metrics in zip(
            densities, metrics, strict=True
        ):
            self.assertEqual(
                posterior_module._metrics_for_density(
                    density,
                    evidence_mass=(
                        posterior.evidence_mass
                        + parameters["evidence_power"]
                    ),
                    acquisition_mass=posterior.acquisition_mass,
                ),
                expected_metrics,
            )
        divergences = tuple(
            posterior_module._kl_divergence(density, current)
            for density in densities
        )
        expected_information = (
            predicted_correct * divergences[0]
            + (1.0 - predicted_correct) * divergences[1]
        )
        current_variance = posterior.metrics().variance
        expected = posterior_module.ExpectedInformation(
            predicted_correct=max(0.0, min(1.0, predicted_correct)),
            expected_information_nats=max(0.0, expected_information),
            expected_variance=max(
                0.0,
                predicted_correct * metrics[0].variance
                + (1.0 - predicted_correct) * metrics[1].variance,
            ),
            variance_reduction=(
                current_variance
                - (
                    predicted_correct * metrics[0].variance
                    + (1.0 - predicted_correct) * metrics[1].variance
                )
            ),
            correct_mastery_probability=metrics[0].mastery_probability,
            incorrect_mastery_probability=metrics[1].mastery_probability,
        )

        self.assertEqual(posterior.expected_information(**parameters), expected)
        self.assertEqual(
            posterior.expected_information_nats(**parameters),
            expected.expected_information_nats,
        )


class PolicyObjectiveProjectionCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Database(
            Path(self.temporary_directory.name) / "projection-cache.db"
        )
        self.database.initialize()
        parsed = read_and_parse(CORPUS, include_catalog=True)
        self.questions = parsed[4]
        self.release_id = self.database.import_corpus(*parsed)["release_id"]
        self.model = LearnerModel(MODEL_VERSION)
        self.engine = AdaptiveEngine(self.database, self.model)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_cached_and_uncached_score_paths_are_exactly_equivalent(self) -> None:
        question = next(
            item for item in self.questions if item.objective is not None
        )
        objective = question.objective
        assert objective is not None
        learner_id = "cache-score-equivalence"
        graph = self.database.get_graph(self.release_id)
        stored_states = {
            mapping.concept_id: self.model.initial_state(
                learner_id, graph.concepts[mapping.concept_id]
            )
            for mapping in question.concepts
        }
        objective_state = self.model.initial_objective_state(
            learner_id, objective
        )
        projected = self.model.project_objective_state(
            objective_state, objective, NOW
        )
        session = {
            "learner_id": learner_id,
            "topic_id": None,
            "focus_concept_id": None,
            "focus_misconception_id": None,
            "focus_objective_id": None,
        }
        common = {
            "session": session,
            "phase": SessionPhase.LEARN,
            "prerequisite_distances": {question.primary_concept_id: 0},
            "concepts": graph.concepts,
            "stored_states": stored_states,
            "objective_states": {objective.id: objective_state},
            "beliefs": {},
            "exposure": {"questions": {}, "families": {}},
            "recent_families": [],
            "last_primary_concept": None,
            "topic_by_concept": {},
            "base_scope": {question.primary_concept_id},
            "connected_pairs": set(),
            "readiness": {
                question.primary_concept_id: SimpleNamespace(
                    bottleneck_concept_id=None,
                    prerequisite_support=1.0,
                )
            },
            "now": NOW,
        }

        uncached = self.engine.policy._score(question, **common)
        cached = self.engine.policy._score(
            question,
            **common,
            projected_objective_states={objective.id: projected},
        )

        self.assertEqual(cached, uncached)

    def test_batched_cold_states_are_exact_and_request_immutable(self) -> None:
        objectives = self.database.get_learning_objectives(self.release_id)
        individual = {
            objective.id: self.model.initial_objective_state(
                "batch-cold-state", objective
            )
            for objective in objectives
        }

        with mock.patch.object(
            posterior_module.ObjectivePosterior,
            "from_prior",
            wraps=posterior_module.ObjectivePosterior.from_prior,
        ) as from_prior:
            batched = self.model.initial_objective_states(
                "batch-cold-state", objectives
            )

        self.assertEqual(dict(batched), individual)
        self.assertEqual(
            from_prior.call_count,
            len({objective.prior_mastery for objective in objectives}),
        )
        with self.assertRaises(TypeError):
            batched[objectives[0].id] = individual[objectives[0].id]

    def test_full_selection_trace_matches_uncached_reference_path(self) -> None:
        sessions = []
        for learner_id in ("trace-cached", "trace-uncached"):
            self.engine.create_learner(learner_id)
            sessions.append(
                self.engine.start_session(
                    learner_id,
                    "t_transformers",
                    mode="learn",
                    explore_related=False,
                    seed=4709,
                    now=NOW,
                )
            )

        cached = self.engine.next_question(sessions[0]["id"], now=NOW)
        original_floor = (
            self.model.concept_projection_with_objective_floor
        )
        original_score = self.engine.policy._score

        def uncached_floor(*args, **kwargs):
            kwargs.pop("projected_objective_states", None)
            return original_floor(*args, **kwargs)

        def uncached_score(question, **kwargs):
            kwargs.pop("projected_objective_states", None)
            return original_score(question, **kwargs)

        with (
            mock.patch.object(
                self.model,
                "concept_projection_with_objective_floor",
                side_effect=uncached_floor,
            ),
            mock.patch.object(
                self.engine.policy,
                "_score",
                side_effect=uncached_score,
            ),
        ):
            uncached = self.engine.next_question(sessions[1]["id"], now=NOW)

        self.assertEqual(cached.question.id, uncached.question.id)
        self.assertEqual(cached.option_order, uncached.option_order)
        fields = (
            "question_id",
            "candidate_count",
            "candidate_digest",
            "top_candidates_json",
            "selected_score_json",
            "propensity",
            "option_order_json",
            "rationale",
        )
        with self.database.read() as connection:
            snapshots = []
            for decision_id in (cached.decision_id, uncached.decision_id):
                row = connection.execute(
                    f"""SELECT {", ".join(fields)}
                        FROM decisions WHERE id = ?""",
                    (decision_id,),
                ).fetchone()
                snapshots.append(tuple(row[field] for field in fields))
        self.assertEqual(snapshots[0], snapshots[1])

    def test_one_selection_projects_each_release_objective_once(self) -> None:
        learner_id = "cache-selection-count"
        self.engine.create_learner(learner_id)
        session = self.engine.start_session(
            learner_id,
            "t_transformers",
            mode="learn",
            explore_related=False,
            seed=1729,
            now=NOW,
        )
        objective_ids = {
            objective.id
            for objective in self.database.get_learning_objectives(
                self.release_id
            )
        }
        original = self.model.project_objective_state

        with mock.patch.object(
            self.model, "project_objective_state", wraps=original
        ) as project_objective_state:
            presentation = self.engine.next_question(session["id"], now=NOW)

        self.assertIsNotNone(presentation.question.objective_id)
        calls_by_objective = Counter(
            call.args[1].id for call in project_objective_state.call_args_list
        )
        self.assertEqual(set(calls_by_objective), objective_ids)
        self.assertEqual(
            calls_by_objective,
            Counter({objective_id: 1 for objective_id in objective_ids}),
        )


if __name__ == "__main__":
    unittest.main()
