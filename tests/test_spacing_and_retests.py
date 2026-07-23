# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tsq.corpus import read_and_parse
from tsq.engine import AdaptiveEngine
from tsq.learner import (
    FAMILY_RETEST_RENEWAL_HEAD,
    MODEL_VERSION,
    OBJECTIVE_GRID_V6_MODEL_VERSION,
    FamilyResponseRecord,
    LearnerModel,
)
from tsq.models import (
    LearningObjective,
    ObjectiveOperation,
    ObjectiveState,
)
from tsq.objective_posterior import (
    LikelihoodObservation,
    ObjectivePosterior,
)
from tsq.replay import ProjectionReplay
from tsq.store import Database


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "ai_curriculum.json"
START = datetime(2110, 1, 2, 9, 0, tzinfo=timezone.utc)


class SpacingAwareFamilyPowerTestCase(unittest.TestCase):
    def test_immediate_repeats_retain_the_square_summable_tail(self) -> None:
        model = LearnerModel()
        history: list[FamilyResponseRecord] = []
        expected = (1.0, 0.25, 0.0625, 0.25 / 9.0)
        for index, expected_power in enumerate(expected):
            occurred_at = START + timedelta(minutes=index)
            result = model.spacing_aware_family_evidence_power(
                prior_records=history,
                occurred_at=occurred_at,
                credible=True,
            )
            self.assertAlmostEqual(result.power, expected_power)
            self.assertEqual(result.renewal_power, 0.0)
            self.assertIsNone(result.renewal_index)
            history.append(
                FamilyResponseRecord(
                    occurred_at=occurred_at,
                    credible=True,
                )
            )

    def test_only_credible_genuinely_spaced_retests_renew(self) -> None:
        model = LearnerModel()
        first = FamilyResponseRecord(START, credible=True)

        too_soon = model.spacing_aware_family_evidence_power(
            prior_records=(first,),
            occurred_at=START + timedelta(days=6, hours=23),
            credible=True,
        )
        self.assertEqual(too_soon.power, 0.25)

        noncredible = model.spacing_aware_family_evidence_power(
            prior_records=(first,),
            occurred_at=START + timedelta(days=8),
            credible=False,
        )
        self.assertEqual(noncredible.power, 0.25)

        renewed = model.spacing_aware_family_evidence_power(
            prior_records=(first,),
            occurred_at=START + timedelta(days=8),
            credible=True,
        )
        self.assertEqual(renewed.base_power, 0.25)
        self.assertEqual(
            renewed.renewal_power, FAMILY_RETEST_RENEWAL_HEAD
        )
        self.assertEqual(renewed.power, 0.75)
        self.assertEqual(renewed.renewal_index, 1)

    def test_credibility_is_signed_and_every_exposure_resets_spacing(self) -> None:
        model = LearnerModel()
        # There is deliberately no correctness input: a thoughtful wrong
        # response is credible negative evidence under the same bounded power.
        self.assertTrue(
            model.credible_family_retest(
                selected_option_id="named-misconception",
                confidence=0.9,
                response_ms=1200,
                hint_count=0,
            )
        )

        history = (
            FamilyResponseRecord(START, credible=True),
            FamilyResponseRecord(
                START + timedelta(days=6),
                credible=False,
            ),
        )
        blocked = model.spacing_aware_family_evidence_power(
            prior_records=history,
            occurred_at=START + timedelta(days=8),
            credible=True,
        )
        self.assertEqual(blocked.renewal_power, 0.0)

        renewed = model.spacing_aware_family_evidence_power(
            prior_records=history,
            occurred_at=START + timedelta(days=14),
            credible=True,
        )
        self.assertEqual(
            renewed.renewal_power, FAMILY_RETEST_RENEWAL_HEAD
        )

    def test_naive_timestamps_fail_and_out_of_order_time_cannot_renew(self) -> None:
        model = LearnerModel()
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            model.spacing_aware_family_evidence_power(
                prior_records=(),
                occurred_at=datetime(2110, 1, 2, 9, 0),
                credible=True,
            )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            model.spacing_aware_family_evidence_power(
                prior_records=(
                    FamilyResponseRecord(
                        datetime(2110, 1, 2, 9, 0),
                        credible=True,
                    ),
                ),
                occurred_at=START,
                credible=True,
            )

        out_of_order = model.spacing_aware_family_evidence_power(
            prior_records=(
                FamilyResponseRecord(
                    START + timedelta(days=8),
                    credible=True,
                ),
            ),
            occurred_at=START,
            credible=True,
        )
        self.assertEqual(out_of_order.power, 0.25)
        self.assertEqual(out_of_order.renewal_power, 0.0)

    def test_lifetime_family_power_stays_below_the_analytical_cap(self) -> None:
        model = LearnerModel()
        history: list[FamilyResponseRecord] = []
        total = 0.0
        for index in range(128):
            occurred_at = START + timedelta(days=8 * index)
            result = model.spacing_aware_family_evidence_power(
                prior_records=history,
                occurred_at=occurred_at,
                credible=True,
            )
            total += result.power
            self.assertLess(total, model.family_evidence_power_bound())
            history.append(
                FamilyResponseRecord(
                    occurred_at=occurred_at,
                    credible=True,
                )
            )

        self.assertGreater(total, 2.40)
        self.assertLess(model.family_evidence_power_bound(), 2.412)

    def test_v6_history_anchors_spacing_without_spending_v7_renewals(self) -> None:
        model = LearnerModel()
        result = model.spacing_aware_family_evidence_power(
            prior_records=(
                FamilyResponseRecord(
                    START,
                    credible=True,
                    renewal_eligible=False,
                ),
            ),
            occurred_at=START + timedelta(days=8),
            credible=True,
        )
        self.assertEqual(result.power, 0.75)
        self.assertEqual(result.renewal_index, 1)

    def test_four_session_eighteen_step_equivalent_reverses_but_does_not_dominate(
        self,
    ) -> None:
        """One target probe per 18-step spaced session; 17 fillers are unrelated."""

        model = LearnerModel()
        posterior = ObjectivePosterior.from_prior(0.20)
        initial_mastery = posterior.metrics().mastery_probability
        history: list[FamilyResponseRecord] = []
        target_steps = 0
        total_steps = 0
        trough_mastery = initial_mastery
        target_family_ids: list[str] = []

        for session_index in range(4):
            # The other seventeen steps belong to other objectives and must not
            # be manufactured into evidence for this target latent.
            total_steps += 17
            occurred_at = (
                START
                + timedelta(days=14 * session_index)
                + timedelta(minutes=17)
            )
            power = model.spacing_aware_family_evidence_power(
                prior_records=history,
                occurred_at=occurred_at,
                credible=True,
            )
            posterior = posterior.with_observation(
                LikelihoodObservation(
                    observation_id=f"session-{session_index + 1}",
                    family_id="family_matched_retest",
                    difficulty=0.5,
                    discrimination=1.5,
                    guess_rate=0.25,
                    slip_rate=0.08,
                    option_count=4,
                    correct=session_index > 0,
                    evidence_power=power.power,
                )
            )
            target_family_ids.append("family_matched_retest")
            history.append(
                FamilyResponseRecord(
                    occurred_at=occurred_at,
                    credible=True,
                )
            )
            target_steps += 1
            total_steps += 1
            if session_index == 0:
                trough_mastery = posterior.metrics().mastery_probability

        final = posterior.metrics()
        self.assertEqual(total_steps, 4 * 18)
        self.assertEqual(target_steps, 4)
        self.assertLess(trough_mastery, initial_mastery)
        self.assertGreater(final.mastery_probability, initial_mastery)
        self.assertLess(final.mastery_probability, 0.15)
        self.assertLess(
            posterior.evidence_mass, model.family_evidence_power_bound()
        )
        self.assertEqual(set(target_family_ids), {"family_matched_retest"})
        state = ObjectiveState(
            learner_id="bounded-recovery",
            objective_id="objective-bounded-recovery",
            mean=final.mean,
            variance=final.variance,
            stability_hours=48.0,
            exposures=target_steps,
            evidence_mass=final.evidence_mass,
            posterior=posterior,
            model_version=MODEL_VERSION,
        )
        self.assertEqual(
            model.mastery_label(
                state,
                independent_families=1,
                delayed_retrievals=1,
                operation_kinds=1,
            ),
            "fragile",
        )


class SpacingModelBoundaryTests(unittest.TestCase):
    def test_planned_family_power_anticipates_a_spaced_renewal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "planned-renewal.db")
            database.initialize()
            parsed = read_and_parse(CORPUS, include_catalog=True)
            release_id = database.import_corpus(*parsed)["release_id"]
            learner_id = "planned-renewal"
            engine = AdaptiveEngine(database)
            engine.create_learner(learner_id)
            session = engine.start_session(
                learner_id,
                "c_bias_variance",
                seed=811,
                now=START,
            )
            presentation = engine.next_question(session["id"], now=START)
            engine.submit_answer(
                presentation.decision_id,
                presentation.question.correct_option.id,
                confidence=0.9,
                response_ms=1200,
                now=START + timedelta(minutes=1),
            )

            model = LearnerModel()
            with database.read() as connection:
                immediate = model.potential_family_evidence_power(
                    connection,
                    learner_id=learner_id,
                    family_id=presentation.question.family_id,
                    now=START + timedelta(minutes=2),
                )
                spaced = model.potential_family_evidence_power(
                    connection,
                    learner_id=learner_id,
                    family_id=presentation.question.family_id,
                    now=START + timedelta(days=8),
                )
                repeated = model.potential_family_evidence_power(
                    connection,
                    learner_id=learner_id,
                    family_id=presentation.question.family_id,
                    now=START + timedelta(days=8),
                )

            self.assertEqual(immediate.power, 0.25)
            self.assertEqual(immediate.renewal_power, 0.0)
            self.assertEqual(spaced.power, 0.75)
            self.assertEqual(spaced.renewal_power, 0.5)
            self.assertEqual(spaced.renewal_index, 1)
            self.assertEqual(repeated, spaced)

            question = next(
                item for item in parsed[4] if item.objective is not None
            )
            graph = database.get_graph(release_id)
            states = {
                mapping.concept_id: model.initial_state(
                    learner_id, graph.concepts[mapping.concept_id]
                )
                for mapping in question.concepts
            }
            immediate_information = model.expected_information_gain(
                question,
                states,
                objective_state=model.initial_objective_state(
                    learner_id, question.objective
                ),
                evidence_power_override=(
                    question.status.evidence_weight * immediate.power
                ),
            )
            spaced_information = model.expected_information_gain(
                question,
                states,
                objective_state=model.initial_objective_state(
                    learner_id, question.objective
                ),
                evidence_power_override=(
                    question.status.evidence_weight * spaced.power
                ),
            )
            self.assertGreater(spaced_information, immediate_information)

            with self.assertRaisesRegex(ValueError, "spacing-aware model"):
                with database.read() as connection:
                    LearnerModel(
                        OBJECTIVE_GRID_V6_MODEL_VERSION
                    ).potential_family_evidence_power(
                        connection,
                        learner_id=learner_id,
                        family_id=presentation.question.family_id,
                        now=START + timedelta(days=8),
                    )

    def test_spaced_repeat_renews_without_new_family_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "family-certificate.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            learner_id = "v7-family-certificate"
            engine = AdaptiveEngine(database)
            engine.create_learner(learner_id)
            families: list[str] = []
            repeated_family: str | None = None
            repeated_change: dict[str, object] | None = None

            for index in range(10):
                now = START + timedelta(days=8 * index)
                session = engine.start_session(
                    learner_id,
                    "c_bias_variance",
                    seed=100 + index,
                    now=now,
                )
                presentation = engine.next_question(
                    session["id"], now=now
                )
                result = engine.submit_answer(
                    presentation.decision_id,
                    presentation.question.correct_option.id,
                    confidence=0.9,
                    response_ms=1200,
                    now=now + timedelta(minutes=1),
                )
                family_id = presentation.question.family_id
                if family_id in families:
                    repeated_family = family_id
                    repeated_change = next(
                        change
                        for change in result.state_changes
                        if change.get("concept_id")
                        == presentation.question.primary_concept_id
                    )
                    break
                families.append(family_id)

            self.assertIsNotNone(repeated_family, families)
            self.assertIsNotNone(repeated_change)
            self.assertIs(repeated_change["family_retest_renewed"], True)
            self.assertIsNotNone(
                repeated_change["family_retest_renewal_index"]
            )
            with database.read() as connection:
                certificate_count = connection.execute(
                    """SELECT COUNT(*) AS n
                       FROM learner_skill_families
                       WHERE learner_id = ? AND concept_id = ?""",
                    (
                        learner_id,
                        presentation.question.primary_concept_id,
                    ),
                ).fetchone()["n"]
                repeated_row = connection.execute(
                    """SELECT delayed_unguided_correct_at
                       FROM learner_skill_families
                       WHERE learner_id = ? AND concept_id = ?
                         AND family_id = ?""",
                    (
                        learner_id,
                        presentation.question.primary_concept_id,
                        repeated_family,
                    ),
                ).fetchone()
            self.assertEqual(certificate_count, len(set(families)))
            self.assertIsNotNone(repeated_row)
            self.assertIsNotNone(
                repeated_row["delayed_unguided_correct_at"]
            )
            replay = ProjectionReplay(database).check(learner_id)
            self.assertTrue(replay["ok"], replay["errors"])

    def test_v6_exact_posterior_carries_forward_and_missing_child_fails(self) -> None:
        objective = LearningObjective(
            id="lo_v7_carry",
            name="V7 carry",
            description="Exact v6 state is carried into v7.",
            primary_concept_id="c_v7_carry",
            supporting_concept_ids=(),
            operation=ObjectiveOperation.APPLY,
            prior_mastery=0.20,
        )
        posterior = ObjectivePosterior.from_prior(
            objective.prior_mastery
        ).with_observation(
            LikelihoodObservation(
                observation_id="v6-observation",
                family_id="family-v6",
                difficulty=0.5,
                discrimination=1.5,
                guess_rate=0.25,
                slip_rate=0.08,
                option_count=4,
                correct=False,
                evidence_power=1.0,
            )
        )
        metrics = posterior.metrics()
        v6_state = ObjectiveState(
            learner_id="v6-carry",
            objective_id=objective.id,
            mean=metrics.mean,
            variance=metrics.variance,
            stability_hours=48.0,
            exposures=1,
            evidence_mass=metrics.evidence_mass,
            posterior=posterior,
            model_version=OBJECTIVE_GRID_V6_MODEL_VERSION,
        )

        carried = LearnerModel(MODEL_VERSION)._migrate_objective_posterior(
            v6_state, objective
        )
        self.assertIs(carried, posterior)
        self.assertEqual(carried.digest, posterior.digest)

        missing = ObjectiveState(
            learner_id="v6-missing",
            objective_id=objective.id,
            mean=metrics.mean,
            variance=metrics.variance,
            stability_hours=48.0,
            exposures=1,
            evidence_mass=metrics.evidence_mass,
            model_version=OBJECTIVE_GRID_V6_MODEL_VERSION,
        )
        with self.assertRaisesRegex(
            ValueError, "missing its posterior projection"
        ):
            LearnerModel(MODEL_VERSION)._migrate_objective_posterior(
                missing, objective
            )
        with self.assertRaisesRegex(ValueError, "incompatible model version"):
            LearnerModel(
                OBJECTIVE_GRID_V6_MODEL_VERSION
            )._migrate_objective_posterior(
                ObjectiveState(
                    learner_id="v7-downgrade",
                    objective_id=objective.id,
                    mean=metrics.mean,
                    variance=metrics.variance,
                    stability_hours=48.0,
                    exposures=1,
                    evidence_mass=metrics.evidence_mass,
                    posterior=posterior,
                    model_version=MODEL_VERSION,
                ),
                objective,
            )

    def test_mixed_v6_v7_exact_history_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "mixed-v6-v7.db")
            database.initialize()
            database.import_corpus(
                *read_and_parse(CORPUS, include_catalog=True)
            )
            learner_id = "mixed-v6-v7"

            engine_v6 = AdaptiveEngine(
                database,
                LearnerModel(OBJECTIVE_GRID_V6_MODEL_VERSION),
            )
            engine_v6.create_learner(learner_id)
            session = engine_v6.start_session(
                learner_id,
                "t_transformers",
                seed=19,
                now=START,
            )
            first = engine_v6.next_question(
                session["id"], now=START
            )
            wrong = next(
                option
                for option in first.question.options
                if not option.correct
            )
            engine_v6.submit_answer(
                first.decision_id,
                wrong.id,
                confidence=0.9,
                response_ms=1200,
                now=START + timedelta(minutes=1),
            )

            objective_id = first.question.objective_id
            self.assertIsNotNone(objective_id)
            v6_state = database.get_objective_states(learner_id)[
                objective_id
            ]
            self.assertEqual(
                v6_state.model_version, OBJECTIVE_GRID_V6_MODEL_VERSION
            )
            self.assertIsNotNone(v6_state.posterior)
            self.assertEqual(
                LearnerModel(MODEL_VERSION)
                ._migrate_objective_posterior(
                    v6_state, first.question.objective
                )
                .digest,
                v6_state.posterior.digest,
            )

            engine_v7 = AdaptiveEngine(database)
            second_at = START + timedelta(days=8)
            second = engine_v7.next_question(
                session["id"], now=second_at
            )
            engine_v7.submit_answer(
                second.decision_id,
                second.question.correct_option.id,
                confidence=0.9,
                response_ms=1200,
                now=second_at + timedelta(minutes=1),
            )

            with database.read() as connection:
                projections = connection.execute(
                    """SELECT schema_version, metadata_json
                       FROM events
                       WHERE learner_id = ?
                         AND event_type = 'LearnerProjectionAdvanced'
                       ORDER BY stream_version""",
                    (learner_id,),
                ).fetchall()
            self.assertEqual(
                [row["schema_version"] for row in projections], [4, 4]
            )
            self.assertEqual(
                [
                    json.loads(row["metadata_json"])[
                        "learner_model_version"
                    ]
                    for row in projections
                ],
                [OBJECTIVE_GRID_V6_MODEL_VERSION, MODEL_VERSION],
            )
            replay = ProjectionReplay(database).check(learner_id)
            self.assertTrue(replay["ok"], replay["errors"])
            self.assertTrue(replay["source_projection_matches_replay"])
            self.assertTrue(replay["commitment_matches_replay"])


if __name__ == "__main__":
    unittest.main()
