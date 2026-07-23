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
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
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

    def test_one_selection_projects_each_release_objective_at_most_twice(self) -> None:
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
        self.assertTrue(
            all(count == 2 for count in calls_by_objective.values()),
            calls_by_objective,
        )


if __name__ == "__main__":
    unittest.main()
