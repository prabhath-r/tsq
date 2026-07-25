# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.errors import ValidationError
from tsq.models import CandidateScore
from tsq.policy import AdaptivePolicy
from tsq.policy_shadow import (
    GREEDY_POLICY_DEFINITION,
    GREEDY_POLICY_DEFINITION_DIGEST,
    GREEDY_POLICY_VERSION,
    POLICY_SHADOW_CONTRACT_VERSION,
    build_policy_shadow_evaluation,
    derive_policy_shadow_projections,
    policy_shadow_integrity_errors,
    policy_shadow_logging_probabilities,
    policy_shadow_projection_snapshot,
    rebuild_policy_shadow_projections,
)
from tsq.replay import ProjectionReplay
from tsq.store import Database

from tests.test_scoring_claim_history_upgrade import rehash_event_streams


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"


def score(question_id: str, total: float) -> CandidateScore:
    return CandidateScore(
        question_id=question_id,
        total=total,
        predicted_correct=0.55,
        information_gain=0.20,
        learning_fit=0.75,
        concept_need=0.60,
        misconception_value=0.25,
        prerequisite_value=0.20,
        review_value=0.10,
        novelty=1.0,
        kind_fit=0.90,
        continuity=0.80,
        boundary_fit=0.70,
        coverage_raw_exposures=2,
        coverage_diagnostic_information=0.625,
        coverage_successful_retrieval_families=1,
    )


class PolicyShadowContractTests(unittest.TestCase):
    def test_live_sampler_matches_the_frozen_v17_arithmetic(self) -> None:
        scores = tuple(
            score(question_id, total)
            for question_id, total in (
                ("q_a", 1.25),
                ("q_b", 1.05),
                ("q_c", 0.95),
                ("q_d", 0.70),
                ("q_e", 0.50),
            )
        )
        peak = max(item.total for item in scores)
        weights = [
            math.exp((item.total - peak) / 0.10)
            for item in scores
        ]
        expected_probabilities = tuple(
            weight / sum(weights) for weight in weights
        )
        distribution = AdaptivePolicy._top_k_distribution(scores)
        self.assertEqual(
            tuple(probability for _item, probability in distribution),
            expected_probabilities,
        )

        for seed in (0, 17, 9_999):
            for step in (0, 1, 23):
                threshold = random.Random(f"policy:{seed}:{step}").random()
                cumulative = 0.0
                expected_index = len(scores) - 1
                for index, probability in enumerate(
                    expected_probabilities
                ):
                    cumulative += probability
                    if threshold <= cumulative:
                        expected_index = index
                        break
                selected, propensity = AdaptivePolicy._sample_distribution(
                    distribution,
                    seed=seed,
                    step=step,
                )
                self.assertEqual(
                    selected.question_id,
                    scores[expected_index].question_id,
                )
                self.assertEqual(
                    propensity,
                    expected_probabilities[expected_index],
                )

    def test_definition_and_evaluation_are_canonical_and_content_addressed(
        self,
    ) -> None:
        definition_json = json.dumps(
            GREEDY_POLICY_DEFINITION,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self.assertEqual(
            GREEDY_POLICY_DEFINITION_DIGEST,
            hashlib.sha256(definition_json.encode("utf-8")).hexdigest(),
        )
        scores = tuple(
            score(question_id, total)
            for question_id, total in (
                ("q_a", 1.20),
                ("q_b", 0.80),
                ("q_c", 0.70),
                ("q_d", 0.60),
                ("q_e", 0.50),
            )
        )
        probabilities = policy_shadow_logging_probabilities(scores)
        frontier = tuple(zip(scores, probabilities, strict=True))
        arguments = {
            "decision_id": "dec_one",
            "logging_policy_version": "recursive-evidence-graph-v18",
            "learner_model_version": "irt-grid-window-v8",
            "corpus_release_id": "rel_one",
            "candidate_count": 8,
            "candidate_digest": "a" * 64,
            "frontier": frontier,
            "live_question_id": "q_c",
            "evaluated_at": "2026-07-25T09:30:00+00:00",
        }

        first = build_policy_shadow_evaluation(**arguments)
        second = build_policy_shadow_evaluation(**arguments)

        self.assertEqual(first, second)
        self.assertEqual(
            first.payload["challenger_policy_version"],
            GREEDY_POLICY_VERSION,
        )
        self.assertEqual(
            first.payload["challenger_definition_digest"],
            GREEDY_POLICY_DEFINITION_DIGEST,
        )
        self.assertEqual(first.payload["challenger_question_id"], "q_a")
        self.assertFalse(first.payload["agreement"])
        self.assertRegex(first.evaluation_id, r"^pse_[0-9a-f]{24}$")
        self.assertEqual(
            set(first.payload["frontier"][0]),
            {"question_id", "logging_probability", *scores[0].terms()},
        )
        self.assertEqual(
            first.metadata,
            {
                "shadow_contract_version": POLICY_SHADOW_CONTRACT_VERSION,
                "shadow_only": True,
                "selection_applied": False,
                "mastery_applied": False,
            },
        )
        self.assertTrue(first.payload["shadow_only"])
        self.assertFalse(first.payload["selection_applied"])
        self.assertFalse(first.payload["mastery_applied"])

    def test_greedy_tie_break_and_logging_probabilities_are_recomputed(
        self,
    ) -> None:
        scores = (
            score("q_a", 0.8),
            score("q_b", 0.8),
            score("q_c", 0.7),
        )
        probabilities = policy_shadow_logging_probabilities(scores)
        built = build_policy_shadow_evaluation(
            decision_id="dec_tie",
            logging_policy_version="recursive-evidence-graph-v18",
            learner_model_version="irt-grid-window-v8",
            corpus_release_id="rel_tie",
            candidate_count=3,
            candidate_digest="b" * 64,
            frontier=tuple(zip(scores, probabilities, strict=True)),
            live_question_id="q_b",
            evaluated_at="2026-07-25T10:00:00+00:00",
        )
        self.assertEqual(built.payload["challenger_question_id"], "q_a")
        self.assertAlmostEqual(sum(probabilities), 1.0)

        wrong = list(zip(scores, probabilities, strict=True))
        wrong[0] = (wrong[0][0], wrong[0][1] + 1e-12)
        with self.assertRaisesRegex(
            ValidationError, "does not match the frozen softmax"
        ):
            build_policy_shadow_evaluation(
                decision_id="dec_tie",
                logging_policy_version="recursive-evidence-graph-v18",
                learner_model_version="irt-grid-window-v8",
                corpus_release_id="rel_tie",
                candidate_count=3,
                candidate_digest="b" * 64,
                frontier=wrong,
                live_question_id="q_b",
                evaluated_at="2026-07-25T10:00:00+00:00",
            )

    def test_logging_probability_underflow_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ValidationError, "lost full support through numerical underflow"
        ):
            policy_shadow_logging_probabilities(
                (
                    score("q_peak", 1_000.0),
                    score("q_underflow", -1_000.0),
                )
            )

    def test_builder_rejects_incomplete_or_unsafe_score_frontiers(self) -> None:
        scores = (
            score("q_a", 0.8),
            score("q_b", 0.7),
        )
        probabilities = policy_shadow_logging_probabilities(scores)
        with self.assertRaisesRegex(ValidationError, "exactly the safe top-k"):
            build_policy_shadow_evaluation(
                decision_id="dec_short",
                logging_policy_version="recursive-evidence-graph-v18",
                learner_model_version="irt-grid-window-v8",
                corpus_release_id="rel_short",
                candidate_count=5,
                candidate_digest="c" * 64,
                frontier=tuple(zip(scores, probabilities, strict=True)),
                live_question_id="q_a",
                evaluated_at="2026-07-25T10:30:00+00:00",
            )

        unsafe_scores = (
            replace(score("q_a", 0.8), predicted_correct=1.2),
            score("q_b", 0.7),
        )
        unsafe_probabilities = policy_shadow_logging_probabilities(
            unsafe_scores
        )
        with self.assertRaisesRegex(
            ValidationError, "predicted_correct must be between"
        ):
            build_policy_shadow_evaluation(
                decision_id="dec_unsafe",
                logging_policy_version="recursive-evidence-graph-v18",
                learner_model_version="irt-grid-window-v8",
                corpus_release_id="rel_unsafe",
                candidate_count=2,
                candidate_digest="d" * 64,
                frontier=tuple(
                    zip(unsafe_scores, unsafe_probabilities, strict=True)
                ),
                live_question_id="q_a",
                evaluated_at="2026-07-25T10:30:00+00:00",
            )


class PolicyShadowProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.tempdir.name) / "shadow.db")
        self.database.initialize()
        parsed = read_and_parse(CORPUS, include_catalog=True)
        self.database.import_corpus(*parsed)
        self.engine = AdaptiveEngine(self.database)
        self.engine.create_learner("shadow-learner")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _rewrite_event_payload(self, event_type, mutate) -> None:
        with self.database.transaction() as connection:
            guard = connection.execute(
                """SELECT sql FROM sqlite_master
                   WHERE type='trigger' AND name='events_no_update'"""
            ).fetchone()
            self.assertIsNotNone(guard)
            self.assertIsNotNone(guard["sql"])
            connection.execute("DROP TRIGGER events_no_update")
            event = connection.execute(
                """SELECT * FROM events
                   WHERE learner_id='shadow-learner' AND event_type=?
                   ORDER BY stream_version LIMIT 1""",
                (event_type,),
            ).fetchone()
            self.assertIsNotNone(event)
            payload = json.loads(event["payload_json"])
            mutate(connection, payload)
            connection.execute(
                """UPDATE events SET payload_json=? WHERE event_id=?""",
                (
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    event["event_id"],
                ),
            )
            rehash_event_streams(connection)
            connection.execute(guard["sql"])

    def _start_shadow_decision(self):
        session = self.engine.start_session(
            "shadow-learner",
            "t_transformers",
            seed=573,
        )
        return self.engine.next_question(session["id"])

    def test_event_derivation_snapshot_rebuild_and_integrity(self) -> None:
        presentation = self._start_shadow_decision()
        with self.database.transaction() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id=?",
                (presentation.decision_id,),
            ).fetchone()
            selection = connection.execute(
                """SELECT * FROM events
                   WHERE event_type='QuestionSelected'
                     AND json_extract(payload_json, '$.decision_id')=?
                   ORDER BY stream_version""",
                (presentation.decision_id,),
            ).fetchone()
            existing = connection.execute(
                """SELECT * FROM events
                   WHERE event_type='PolicyShadowEvaluated'
                     AND json_extract(payload_json, '$.decision_id')=?""",
                (presentation.decision_id,),
            ).fetchone()
            self.assertIsNotNone(existing)
            self.assertEqual(existing["causation_id"], selection["event_id"])
            self.assertEqual(existing["correlation_id"], decision["id"])
            self.assertEqual(
                existing["stream_version"],
                selection["stream_version"] + 1,
            )

        with self.database.read() as connection:
            stored = policy_shadow_projection_snapshot(connection)
            derived, checkpoints = derive_policy_shadow_projections(connection)
            errors = policy_shadow_integrity_errors(connection)
        self.assertEqual(stored, derived)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(errors, [])
        self.assertFalse(checkpoints[0]["agreement"] is None)

        with self.database.transaction() as connection:
            rebuilt = rebuild_policy_shadow_projections(connection)
        self.assertEqual(rebuilt["snapshot"], derived)
        self.assertEqual(rebuilt["evaluation_count"], 1)

        with self.database.transaction() as connection:
            connection.execute(
                "DROP TRIGGER policy_shadow_evaluations_no_update"
            )
            connection.execute(
                """UPDATE policy_shadow_evaluations
                   SET output_digest=?""",
                ("f" * 64,),
            )
        with self.database.read() as connection:
            errors = policy_shadow_integrity_errors(connection)
        self.assertEqual(len(errors), 1)
        self.assertIn("projection differs", errors[0])

    def test_rehashed_propensity_tamper_is_blocking(self) -> None:
        presentation = self._start_shadow_decision()

        def mutate(connection, payload) -> None:
            altered = payload["propensity"] / 2.0
            payload["propensity"] = altered
            connection.execute(
                """UPDATE decisions SET propensity=? WHERE id=?""",
                (altered, presentation.decision_id),
            )

        self._rewrite_event_payload("QuestionSelected", mutate)

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "propensity differs from the immutable safe frontier"
                in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        replay = ProjectionReplay(self.database).check("shadow-learner")
        self.assertFalse(replay["ok"])
        self.assertTrue(
            any(
                "propensity differs from the immutable safe frontier"
                in error
                for error in replay["blocking_source_integrity_errors"]
            ),
            replay["blocking_source_integrity_errors"],
        )

    def test_rehashed_selected_score_tamper_is_blocking(self) -> None:
        presentation = self._start_shadow_decision()

        def mutate(connection, payload) -> None:
            score_terms = payload["score"]
            score_terms["predicted_correct"] = (
                0.99
                if score_terms["predicted_correct"] < 0.99
                else 0.01
            )
            connection.execute(
                """UPDATE decisions SET selected_score_json=? WHERE id=?""",
                (
                    json.dumps(score_terms, sort_keys=True),
                    presentation.decision_id,
                ),
            )

        self._rewrite_event_payload("QuestionSelected", mutate)

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "selected score differs from the immutable safe frontier"
                in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )

    def test_huge_shadow_number_is_reported_as_blocking_corruption(
        self,
    ) -> None:
        self._start_shadow_decision()

        def mutate(_connection, payload) -> None:
            payload["frontier"][0]["total"] = 10**400

        self._rewrite_event_payload("PolicyShadowEvaluated", mutate)

        integrity = self.database.verify_integrity()
        self.assertFalse(integrity["ok"])
        self.assertTrue(
            any(
                "must be a finite number" in error
                for error in integrity["errors"]
            ),
            integrity["errors"],
        )
        replay = ProjectionReplay(self.database).check("shadow-learner")
        self.assertFalse(replay["ok"])
        self.assertTrue(
            any(
                "must be a finite number" in error
                for error in replay["blocking_source_integrity_errors"]
            ),
            replay["blocking_source_integrity_errors"],
        )
        target = Path(self.tempdir.name) / "corrupt-shadow-rebuild.db"
        with self.assertRaisesRegex(
            ValidationError, "no rebuilt copy was published"
        ):
            ProjectionReplay(self.database).rebuild_copy(
                "shadow-learner",
                target,
            )
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
