# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import itertools
import math
import unittest

from experiments.order_invariant_evidence_lab import (
    APPROVED_EVIDENCE_WEIGHT,
    GRID_SPECS,
    INITIAL_VARIANCE,
    ResponseFactor,
    batch_fisher_laplace,
    family_budget,
    gaussian_summary,
    grid_posterior,
    retention_project,
    symmetric_family_exponents,
    two_epoch_projection,
)
from tsq.models import logit


def factor(
    item_id: str,
    family_id: str,
    difficulty: float,
    discrimination: float,
    *,
    correct: bool = True,
    quality: float = APPROVED_EVIDENCE_WEIGHT,
) -> ResponseFactor:
    return ResponseFactor(
        item_id=item_id,
        family_id=family_id,
        difficulty=difficulty,
        discrimination=discrimination,
        guess_rate=0.25,
        slip_rate=0.07,
        correct=correct,
        evidence_quality=quality,
        feedback_quality=quality,
    )


class OrderInvariantEvidenceLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prior_mean = logit(0.20)
        self.factors = (
            factor("q_easy", "family_a", -0.5, 1.2),
            factor("q_mid", "family_b", 0.6, 1.7),
            factor("q_hard", "family_c", 1.8, 2.1),
            factor("q_repeat_a", "family_repeat", 0.3, 1.5, quality=0.30),
            factor("q_repeat_b", "family_repeat", 1.0, 1.9, quality=0.80),
        )

    def test_grid_and_batch_fisher_are_exactly_permutation_invariant(self) -> None:
        baseline_grid = grid_posterior(
            self.factors,
            prior_mean=self.prior_mean,
            prior_variance=INITIAL_VARIANCE,
            spec=GRID_SPECS[0],
        )
        baseline_laplace = batch_fisher_laplace(
            self.factors,
            prior_mean=self.prior_mean,
            prior_variance=INITIAL_VARIANCE,
        )
        for order in itertools.permutations(self.factors):
            self.assertEqual(
                grid_posterior(
                    order,
                    prior_mean=self.prior_mean,
                    prior_variance=INITIAL_VARIANCE,
                    spec=GRID_SPECS[0],
                ),
                baseline_grid,
            )
            self.assertEqual(
                batch_fisher_laplace(
                    order,
                    prior_mean=self.prior_mean,
                    prior_variance=INITIAL_VARIANCE,
                ),
                baseline_laplace,
            )

    def test_grid_tail_converges_without_boundary_mass(self) -> None:
        reference = grid_posterior(
            self.factors,
            prior_mean=self.prior_mean,
            prior_variance=INITIAL_VARIANCE,
            spec=GRID_SPECS[1],
        )
        fine = grid_posterior(
            self.factors,
            prior_mean=self.prior_mean,
            prior_variance=INITIAL_VARIANCE,
            spec=GRID_SPECS[2],
        )
        self.assertLess(
            abs(reference.mastery_probability - fine.mastery_probability),
            5e-5,
        )
        self.assertLess(abs(reference.mean - fine.mean), 5e-5)
        self.assertLess(reference.edge_mass, 1e-12)
        self.assertLess(fine.edge_mass, 1e-12)

    def test_wrong_only_batch_cannot_raise_competence_or_mastery(self) -> None:
        wrong = tuple(
            factor(
                item.item_id,
                item.family_id,
                item.difficulty,
                item.discrimination,
                correct=False,
                quality=item.evidence_quality,
            )
            for item in self.factors
        )
        posterior = grid_posterior(
            wrong,
            prior_mean=self.prior_mean,
            prior_variance=INITIAL_VARIANCE,
            spec=GRID_SPECS[1],
        )
        prior = gaussian_summary(self.prior_mean, INITIAL_VARIANCE)
        self.assertLess(posterior.mastery_probability, prior.mastery_probability)
        self.assertLess(posterior.expected_competence, prior.expected_competence)

    def test_simultaneous_family_budget_is_symmetric_and_bounded(self) -> None:
        repeated = tuple(
            factor(
                f"q_{index:04d}",
                "one_family",
                0.2 + index / 10_000.0,
                1.5,
            )
            for index in range(1000)
        )
        forward = symmetric_family_exponents(repeated)
        reverse = symmetric_family_exponents(tuple(reversed(repeated)))
        self.assertEqual(forward, reverse)
        self.assertAlmostEqual(
            math.fsum(forward.values()),
            APPROVED_EVIDENCE_WEIGHT * family_budget(0, len(repeated)),
            places=12,
        )
        infinite_limit = APPROVED_EVIDENCE_WEIGHT * (
            1.0 + 0.25 * math.pi**2 / 6.0
        )
        self.assertLess(math.fsum(forward.values()), infinite_limit)

        informative = factor(
            "q_informative", "mixed_family", 0.5, 1.5, quality=0.80
        )
        zero_quality = factor(
            "q_zero", "mixed_family", 1.0, 1.7, quality=0.0
        )
        alone = symmetric_family_exponents((informative,))
        with_zero = symmetric_family_exponents((informative, zero_quality))
        self.assertEqual(
            with_zero[informative.item_id], alone[informative.item_id]
        )
        self.assertEqual(with_zero[zero_quality.item_id], 0.0)

    def test_retention_between_epochs_preserves_chronological_order(self) -> None:
        easy = self.factors[:2]
        hard = self.factors[2:]
        easy_then_hard = two_epoch_projection(
            easy,
            hard,
            prior_mean=self.prior_mean,
            prior_variance=INITIAL_VARIANCE,
            elapsed_hours=168.0,
            spec=GRID_SPECS[1],
        )
        hard_then_easy = two_epoch_projection(
            hard,
            easy,
            prior_mean=self.prior_mean,
            prior_variance=INITIAL_VARIANCE,
            elapsed_hours=168.0,
            spec=GRID_SPECS[1],
        )
        self.assertGreater(
            abs(
                easy_then_hard.mastery_probability
                - hard_then_easy.mastery_probability
            ),
            1e-4,
        )

        first = grid_posterior(
            easy,
            prior_mean=self.prior_mean,
            prior_variance=INITIAL_VARIANCE,
            spec=GRID_SPECS[1],
        )
        retained_mean, retained_variance = retention_project(
            first,
            prior_mean=self.prior_mean,
            stability_hours=48.0,
            elapsed_hours=168.0,
        )
        retained = gaussian_summary(retained_mean, retained_variance)
        self.assertLessEqual(
            retained.mastery_probability,
            first.gaussian_mastery_probability + 1e-12,
        )


if __name__ == "__main__":
    unittest.main()
