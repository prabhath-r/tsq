# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
import unittest

from tsq.objective_posterior import (
    DEFAULT_MASTERY_THRESHOLD,
    DEFAULT_PRIOR_VARIANCE,
    GRID_LOWER,
    GRID_SIZE,
    GRID_STEP,
    GRID_UPPER,
    MASTERY_PROBABILITY_ERROR_BOUND,
    OBJECTIVE_POSTERIOR_GRID_ID,
    THETA_GRID,
    LikelihoodObservation,
    ObjectivePosterior,
    ObjectivePosteriorError,
    decode_objective_posterior,
    posterior_digest,
)


def observation(
    observation_id: str,
    *,
    family_id: str | None = None,
    difficulty: float = 0.5,
    discrimination: float = 1.6,
    correct: bool = True,
    power: float = 0.65,
    option_count: int = 4,
) -> LikelihoodObservation:
    return LikelihoodObservation(
        observation_id=observation_id,
        family_id=family_id or f"family_{observation_id}",
        difficulty=difficulty,
        discrimination=discrimination,
        guess_rate=0.25,
        slip_rate=0.08,
        option_count=option_count,
        correct=correct,
        evidence_power=power,
    )


def _fine_reference(
    observations: tuple[LikelihoodObservation, ...],
    *,
    predictive: LikelihoodObservation,
) -> dict[str, float]:
    steps_per_unit = 256
    step = 1.0 / steps_per_unit
    size = int((GRID_UPPER - GRID_LOWER) * steps_per_unit) + 1
    xs = [GRID_LOWER + index / steps_per_unit for index in range(size)]
    prior_mean = math.log(0.20 / 0.80)

    def sigmoid(value: float) -> float:
        if value >= 0.0:
            term = math.exp(-value)
            return 1.0 / (1.0 + term)
        term = math.exp(value)
        return term / (1.0 + term)

    def probability(item: LikelihoodObservation, theta: float) -> float:
        logistic = sigmoid(
            item.discrimination * (theta - item.difficulty)
        )
        modeled = item.guess_rate + (
            1.0 - item.guess_rate - item.slip_rate
        ) * logistic
        return 0.03 * (1.0 / item.option_count) + 0.97 * modeled

    logs: list[float] = []
    for theta in xs:
        terms = []
        for item in observations:
            chance = probability(item, theta)
            likelihood = chance if item.correct else 1.0 - chance
            terms.append(item.evidence_power * math.log(likelihood))
        logs.append(
            -0.5 * (theta - prior_mean) ** 2 / DEFAULT_PRIOR_VARIANCE
            + math.fsum(terms)
        )
    peak = max(logs)
    density = [math.exp(value - peak) for value in logs]
    intervals = [
        0.5 * step * (density[index] + density[index + 1])
        for index in range(size - 1)
    ]
    normalizer = math.fsum(intervals)

    def expectation(values: list[float]) -> float:
        return math.fsum(
            0.5
            * step
            * (
                values[index] * density[index]
                + values[index + 1] * density[index + 1]
            )
            for index in range(size - 1)
        ) / normalizer

    boundary = math.log(
        DEFAULT_MASTERY_THRESHOLD / (1.0 - DEFAULT_MASTERY_THRESHOLD)
    )
    boundary_index = int((boundary - GRID_LOWER) // step)
    below = math.fsum(intervals[:boundary_index])
    left = xs[boundary_index]
    fraction = (boundary - left) / step
    at_boundary = density[boundary_index] + fraction * (
        density[boundary_index + 1] - density[boundary_index]
    )
    below += 0.5 * (boundary - left) * (
        density[boundary_index] + at_boundary
    )
    return {
        "mean": expectation(xs),
        "mastery_probability": 1.0 - below / normalizer,
        "expected_competence": expectation([sigmoid(theta) for theta in xs]),
        "predictive": expectation(
            [probability(predictive, theta) for theta in xs]
        ),
    }


def _normalized_density(log_values: list[float]) -> tuple[float, ...]:
    peak = max(log_values)
    scaled = [math.exp(value - peak) for value in log_values]
    normalizer = math.fsum(
        value
        * GRID_STEP
        * (0.5 if index in {0, GRID_SIZE - 1} else 1.0)
        for index, value in enumerate(scaled)
    )
    log_normalizer = peak + math.log(normalizer)
    return tuple(value - log_normalizer for value in log_values)


class ObjectivePosteriorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prior = ObjectivePosterior.from_prior(0.20)
        self.responses = (
            observation("easy_correct", difficulty=-0.4, discrimination=1.3),
            observation("mid_wrong", difficulty=0.4, discrimination=1.6, correct=False),
            observation("hard_correct", difficulty=1.4, discrimination=2.0),
            observation("mid_correct", difficulty=0.7, discrimination=1.8),
        )

    def test_fixed_grid_covers_current_v5_mean_and_variance_envelope(self) -> None:
        self.assertEqual(THETA_GRID[0], GRID_LOWER)
        self.assertEqual(THETA_GRID[-1], GRID_UPPER)
        self.assertEqual(len(THETA_GRID), GRID_SIZE)
        self.assertTrue(
            all(
                THETA_GRID[index + 1] - THETA_GRID[index] == GRID_STEP
                for index in range(GRID_SIZE - 1)
            )
        )
        for mean in (-6.0, 6.0):
            migrated = ObjectivePosterior.from_gaussian(
                prior_mastery=0.20,
                mean=mean,
                variance=4.0,
            )
            self.assertLess(migrated.metrics().edge_mass, 1e-8)
        with self.assertRaisesRegex(ObjectivePosteriorError, "cannot faithfully"):
            ObjectivePosterior.from_gaussian(
                prior_mastery=0.20,
                mean=0.01,
                variance=1e-6,
            )
        with self.assertRaisesRegex(ObjectivePosteriorError, "cannot faithfully"):
            ObjectivePosterior.from_prior(0.20, prior_variance=1e-6)

    def test_static_likelihoods_are_exactly_permutation_invariant(self) -> None:
        baseline = self.prior.with_observations(self.responses)
        for permutation in itertools.permutations(self.responses):
            candidate = self.prior.with_observations(permutation)
            self.assertEqual(candidate, baseline)
            self.assertEqual(candidate.log_density, baseline.log_density)
            self.assertEqual(candidate.encode(), baseline.encode())
            self.assertEqual(candidate.digest, baseline.digest)

    def test_sequential_updates_equal_one_batch_without_transitions(self) -> None:
        sequential = self.prior
        for item in self.responses:
            sequential = sequential.with_observation(item)
        batch = self.prior.with_observations(self.responses)
        self.assertEqual(sequential, batch)
        self.assertEqual(sequential.metrics(), batch.metrics())
        self.assertEqual(sequential.evidence_mass, 2.6)

    def test_repeated_family_powers_remain_caller_assigned(self) -> None:
        first = observation(
            "family_first", family_id="shared_family", power=1.0
        )
        repeated = observation(
            "family_repeated", family_id="shared_family", power=0.25
        )
        posterior = self.prior.with_observations((repeated, first))
        by_id = {
            item.observation_id: item.evidence_power
            for item in posterior.pending_observations
        }
        self.assertEqual(
            by_id, {"family_first": 1.0, "family_repeated": 0.25}
        )
        self.assertEqual(posterior.evidence_mass, 1.25)
        self.assertEqual(
            posterior,
            self.prior.with_observations((first, repeated)),
        )

    def test_correct_and_wrong_likelihoods_move_both_published_metrics(self) -> None:
        prior = self.prior.metrics()
        correct = self.prior.with_observation(
            observation("surprising_correct", difficulty=1.8, discrimination=2.0)
        ).metrics()
        wrong = self.prior.with_observation(
            observation(
                "credible_wrong",
                difficulty=-0.2,
                discrimination=1.7,
                correct=False,
            )
        ).metrics()
        self.assertGreater(correct.mean, prior.mean)
        self.assertGreater(correct.mastery_probability, prior.mastery_probability)
        self.assertGreater(correct.expected_competence, prior.expected_competence)
        self.assertLess(wrong.mean, prior.mean)
        self.assertLess(wrong.mastery_probability, prior.mastery_probability)
        self.assertLess(wrong.expected_competence, prior.expected_competence)

    def test_full_density_retains_hard_correct_skew_hidden_by_gaussian(self) -> None:
        hard = tuple(
            observation(
                f"hard_{index}",
                difficulty=difficulty,
                discrimination=discrimination,
            )
            for index, (difficulty, discrimination) in enumerate(
                ((1.05, 1.7), (1.30, 1.8), (1.00, 1.75), (0.85, 1.75), (0.70, 1.75))
            )
        )
        posterior = self.prior.with_observations(hard)
        metrics = posterior.metrics()
        boundary = math.log(0.65 / 0.35)
        gaussian_tail = 0.5 * math.erfc(
            (boundary - metrics.mean) / math.sqrt(2.0 * metrics.variance)
        )
        self.assertGreater(metrics.mastery_probability, gaussian_tail + 0.09)

    def test_negative_evidence_is_frozen_during_inactivity(self) -> None:
        negative = self.prior.with_observations(
            tuple(
                observation(
                    f"wrong_{index}",
                    difficulty=0.0,
                    discrimination=1.6,
                    correct=False,
                )
                for index in range(3)
            )
        )
        before = negative.metrics()
        retained = negative.apply_retention(
            elapsed_hours=24.0 * 365.0,
            stability_hours=48.0,
        )
        self.assertEqual(retained.log_density, negative.log_density)
        self.assertEqual(retained.metrics(), before)
        self.assertEqual(retained.pending_observations, ())
        self.assertEqual(retained.evidence_mass, negative.evidence_mass)

    def test_retention_is_a_survival_mixture_and_never_improves_metrics(self) -> None:
        positive = self.prior.with_observations(
            tuple(
                observation(
                    f"correct_{index}",
                    difficulty=difficulty,
                    discrimination=1.7,
                )
                for index, difficulty in enumerate((-0.5, 0.0, 0.5, 1.0, 1.5))
            )
        )
        before = positive.metrics()
        prior_metrics = self.prior.metrics()
        retained = positive.apply_retention(
            elapsed_hours=48.0,
            stability_hours=48.0,
        )
        after = retained.metrics()
        survival = 0.5
        self.assertAlmostEqual(
            after.expected_competence,
            survival * before.expected_competence
            + (1.0 - survival) * prior_metrics.expected_competence,
            places=12,
        )
        self.assertAlmostEqual(
            after.mastery_probability,
            survival * before.mastery_probability
            + (1.0 - survival) * prior_metrics.mastery_probability,
            places=12,
        )
        self.assertLess(after.mastery_probability, before.mastery_probability)
        self.assertLess(after.expected_competence, before.expected_competence)
        self.assertEqual(retained.evidence_mass, positive.evidence_mass)

        two_steps = positive.apply_retention(
            elapsed_hours=24.0,
            stability_hours=48.0,
        ).apply_retention(
            elapsed_hours=24.0,
            stability_hours=48.0,
        )
        self.assertAlmostEqual(
            retained.mastery_probability,
            two_steps.mastery_probability,
            places=12,
        )
        self.assertAlmostEqual(
            retained.expected_competence,
            two_steps.expected_competence,
            places=12,
        )

    def test_retention_keeps_distinct_epochs_chronological(self) -> None:
        chronological = tuple(
            observation(
                f"chronological_{index}",
                difficulty=difficulty,
                discrimination=discrimination,
            )
            for index, (difficulty, discrimination) in enumerate(
                ((-0.5, 1.3), (0.0, 1.5), (1.0, 1.8), (1.8, 2.1))
            )
        )
        first = chronological[:2]
        second = chronological[2:]
        same_epoch = self.prior.with_observations((*first, *second))
        separated = self.prior.with_observations(first).apply_retention(
            elapsed_hours=168.0,
            stability_hours=48.0,
        ).with_observations(second)
        reverse = self.prior.with_observations(second).apply_retention(
            elapsed_hours=168.0,
            stability_hours=48.0,
        ).with_observations(first)
        self.assertNotAlmostEqual(
            separated.mastery_probability,
            same_epoch.mastery_probability,
            places=5,
        )
        self.assertNotAlmostEqual(
            separated.mastery_probability,
            reverse.mastery_probability,
            places=5,
        )

    def test_correct_feedback_is_uncertain_monotone_transport_not_evidence(self) -> None:
        answered = self.prior.with_observation(
            observation("answer", difficulty=0.8, discrimination=1.7)
        )
        before = answered.metrics()
        acquired = answered.apply_correct_feedback(0.65)
        after = acquired.metrics()
        self.assertGreater(after.mean, before.mean)
        self.assertGreaterEqual(
            after.mastery_probability, before.mastery_probability
        )
        self.assertGreater(
            after.expected_competence, before.expected_competence
        )
        self.assertEqual(after.evidence_mass, before.evidence_mass)
        self.assertEqual(after.acquisition_mass, 0.65)
        self.assertEqual(acquired.pending_observations, ())
        self.assertEqual(answered.apply_correct_feedback(0.0), answered)

    def test_tail_competence_and_predictive_integrals_converge_to_fine_grid(self) -> None:
        hard = tuple(
            observation(
                f"hard_{index}",
                difficulty=difficulty,
                discrimination=discrimination,
            )
            for index, (difficulty, discrimination) in enumerate(
                ((1.05, 1.7), (1.30, 1.8), (1.00, 1.75), (0.85, 1.75), (0.70, 1.75))
            )
        )
        predictive = observation(
            "predictive",
            difficulty=0.9,
            discrimination=1.7,
        )
        posterior = self.prior.with_observations(hard)
        reference = _fine_reference(hard, predictive=predictive)
        metrics = posterior.metrics()
        prediction = posterior.predict_correct(
            difficulty=predictive.difficulty,
            discrimination=predictive.discrimination,
            guess_rate=predictive.guess_rate,
            slip_rate=predictive.slip_rate,
        )
        self.assertLess(abs(metrics.mean - reference["mean"]), 5e-5)
        self.assertLess(
            abs(
                metrics.mastery_probability
                - reference["mastery_probability"]
            ),
            MASTERY_PROBABILITY_ERROR_BOUND,
        )
        self.assertLessEqual(
            abs(
                metrics.mastery_probability
                - reference["mastery_probability"]
            ),
            metrics.mastery_probability_error_bound,
        )
        self.assertLess(
            abs(metrics.expected_competence - reference["expected_competence"]),
            5e-5,
        )
        self.assertLess(abs(prediction - reference["predictive"]), 5e-5)
        self.assertEqual(
            metrics.mastery_probability_error_bound,
            MASTERY_PROBABILITY_ERROR_BOUND,
        )
        self.assertAlmostEqual(
            metrics.conservative_mastery_probability,
            metrics.mastery_probability - MASTERY_PROBABILITY_ERROR_BOUND,
        )
        self.assertEqual(
            posterior.conservative_mastery_probability,
            metrics.conservative_mastery_probability,
        )

    def test_option_count_changes_lapse_probability_and_is_committed(self) -> None:
        binary = observation("binary", option_count=2)
        # The authored guess parameter remains the same; only the explicit
        # lapse/random-response component changes with the response format.
        prediction_two = self.prior.predict_correct(
            difficulty=0.5,
            discrimination=1.6,
            guess_rate=0.25,
            slip_rate=0.08,
            option_count=2,
        )
        prediction_eight = self.prior.predict_correct(
            difficulty=0.5,
            discrimination=1.6,
            guess_rate=0.25,
            slip_rate=0.08,
            option_count=8,
        )
        self.assertGreater(prediction_two, prediction_eight)
        posterior = self.prior.with_observation(binary)
        decoded = decode_objective_posterior(posterior.encode())
        self.assertEqual(decoded.pending_observations[0].option_count, 2)
        with self.assertRaises(ObjectivePosteriorError):
            self.prior.predict_correct(
                difficulty=0.5,
                discrimination=1.6,
                guess_rate=0.25,
                slip_rate=0.08,
                option_count=1,
            )

    def test_expected_information_integrates_both_hypothetical_outcomes(self) -> None:
        information = self.prior.expected_information(
            difficulty=0.7,
            discrimination=1.8,
            guess_rate=0.25,
            slip_rate=0.08,
            option_count=4,
            evidence_power=1.0,
        )
        prediction = self.prior.predict_correct(
            difficulty=0.7,
            discrimination=1.8,
            guess_rate=0.25,
            slip_rate=0.08,
            option_count=4,
        )
        self.assertAlmostEqual(information.predicted_correct, prediction, places=12)
        self.assertGreater(information.expected_information_nats, 0.0)
        self.assertGreater(information.variance_reduction, 0.0)
        self.assertGreater(
            information.correct_mastery_probability,
            information.incorrect_mastery_probability,
        )

        neutral = self.prior.expected_information(
            difficulty=0.7,
            discrimination=1.8,
            guess_rate=0.25,
            slip_rate=0.08,
            evidence_power=0.0,
        )
        self.assertAlmostEqual(neutral.expected_information_nats, 0.0, places=14)
        self.assertAlmostEqual(neutral.variance_reduction, 0.0, places=12)

    def test_serialization_roundtrip_digest_and_canonical_form(self) -> None:
        posterior = self.prior.with_observations(self.responses)
        encoded = posterior.encode()
        digest = hashlib.sha256(encoded).hexdigest()
        self.assertEqual(posterior.digest, digest)
        self.assertEqual(posterior_digest(encoded), digest)
        decoded = decode_objective_posterior(
            encoded, expected_digest=digest
        )
        self.assertEqual(decoded, posterior)
        self.assertEqual(decoded.encode(), encoded)
        self.assertEqual(decoded.digest, digest)

        with self.assertRaisesRegex(ObjectivePosteriorError, "digest mismatch"):
            decode_objective_posterior(encoded, expected_digest="0" * 64)
        with self.assertRaisesRegex(ObjectivePosteriorError, "canonical form"):
            decode_objective_posterior(b" " + encoded)

    def test_decode_rejects_unknown_duplicate_nonfinite_and_corrupt_state(self) -> None:
        payload = json.loads(self.prior.encode())

        future = dict(payload)
        future["grid_id"] = OBJECTIVE_POSTERIOR_GRID_ID + "-future"
        with self.assertRaisesRegex(ObjectivePosteriorError, "Unsupported posterior grid"):
            decode_objective_posterior(
                json.dumps(
                    future, sort_keys=True, separators=(",", ":")
                ).encode()
            )

        duplicate = self.prior.encode().replace(
            b'{"acquisition_mass":0.0,',
            b'{"acquisition_mass":0.0,"acquisition_mass":0.0,',
            1,
        )
        with self.assertRaisesRegex(ObjectivePosteriorError, "repeats JSON key"):
            decode_objective_posterior(duplicate)

        nonfinite = self.prior.encode().replace(
            b'"acquisition_mass":0.0', b'"acquisition_mass":NaN', 1
        )
        with self.assertRaisesRegex(ObjectivePosteriorError, "non-finite JSON"):
            decode_objective_posterior(nonfinite)

        corrupt = dict(payload)
        corrupt_density = list(corrupt["anchor_log_density"])
        corrupt_density[GRID_SIZE // 2] += 0.1
        corrupt["anchor_log_density"] = corrupt_density
        encoded_corrupt = json.dumps(
            corrupt, sort_keys=True, separators=(",", ":")
        ).encode()
        with self.assertRaisesRegex(ObjectivePosteriorError, "not normalized"):
            decode_objective_posterior(encoded_corrupt)

    def test_edge_concentration_and_invalid_values_fail_closed(self) -> None:
        unsafe = _normalized_density(
            [-0.5 * (theta - 17.9) ** 2 / 0.01 for theta in THETA_GRID]
        )
        with self.assertRaisesRegex(ObjectivePosteriorError, "unsafe mass"):
            ObjectivePosterior(
                prior_mastery=0.20,
                prior_variance=DEFAULT_PRIOR_VARIANCE,
                anchor_log_density=unsafe,
            )

        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(value=bad):
                with self.assertRaises(ObjectivePosteriorError):
                    ObjectivePosterior.from_prior(bad)
        with self.assertRaises(ObjectivePosteriorError):
            ObjectivePosterior.from_prior(10**400)
        with self.assertRaises(ObjectivePosteriorError):
            observation("overflow", difficulty=10**400)
        with self.assertRaises(ObjectivePosteriorError):
            observation("bad_power", power=1.01)
        with self.assertRaises(ObjectivePosteriorError):
            self.prior.with_observations((object(),))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ObjectivePosteriorError, "repeat"):
            self.prior.with_observations(
                (observation("duplicate"), observation("duplicate"))
            )

    def test_realistic_scalar_projection_runtime_is_bounded(self) -> None:
        factors = tuple(
            observation(
                f"runtime_{index}",
                difficulty=0.2 + 0.15 * index,
                discrimination=1.5 + 0.03 * index,
                correct=index % 5 != 0,
            )
            for index in range(8)
        )
        started = time.perf_counter()
        for _ in range(20):
            projected = self.prior.with_observations(factors)
            projected.metrics()
            projected.predict_correct(
                difficulty=0.9,
                discrimination=1.7,
                guess_rate=0.25,
                slip_rate=0.08,
            )
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
