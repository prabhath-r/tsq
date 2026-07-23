#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0

"""Compare TSQ's sequential Gaussian update with exchangeable evidence batches.

This is a pure, disposable research laboratory.  It does not open a TSQ
database, mutate learner projections, or define a production learner model.
Its purpose is to make three mathematical questions inspectable before a
versioned migration is attempted:

* how much the current assumed-density update depends on the order of static
  response likelihoods;
* whether a bounded log-density grid converges and preserves permutation
  invariance for one-dimensional objective evidence; and
* what information a Gaussian batch-Laplace summary loses when a run of hard
  correct responses creates a strongly skewed posterior.

The grid implements a generalized posterior.  Evidence discounts are power
likelihood exponents, and simultaneous observations from one correlated family
share the same square-summable family budget without depending on arrival
order.  Retention is demonstrated only between distinct timestamp epochs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from tsq.learner import (  # noqa: E402
    MAX_POSTERIOR_VARIANCE,
    MIN_POSTERIOR_VARIANCE,
    SESSION_LAPSE_RATE,
)
from tsq.models import MASTERY_THRESHOLD, SkillState, logit, sigmoid  # noqa: E402


LAB_VERSION = "order-invariant-evidence-lab-v1"
DEFAULT_CORPUS = PROJECT_ROOT / "corpus" / "ai_curriculum.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "results" / "order_invariant_evidence_lab.json"
)
INITIAL_VARIANCE = 2.25
INITIAL_STABILITY_HOURS = 48.0
APPROVED_EVIDENCE_WEIGHT = 0.65


class LabInvariantError(RuntimeError):
    """Raised when the prototype contradicts a required safety property."""


@dataclass(frozen=True, slots=True)
class ResponseFactor:
    """Immutable scalar response-likelihood input for one objective."""

    item_id: str
    family_id: str
    difficulty: float
    discrimination: float
    guess_rate: float
    slip_rate: float
    correct: bool
    evidence_quality: float = APPROVED_EVIDENCE_WEIGHT
    feedback_quality: float = APPROVED_EVIDENCE_WEIGHT

    def __post_init__(self) -> None:
        if not self.item_id or not self.family_id:
            raise ValueError("Evidence factors require stable item and family IDs.")
        for field_name in (
            "difficulty",
            "discrimination",
            "guess_rate",
            "slip_rate",
            "evidence_quality",
            "feedback_quality",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
        if self.discrimination <= 0.0:
            raise ValueError("Discrimination must be positive.")
        if not 0.0 <= self.guess_rate < 1.0:
            raise ValueError("Guess rate must be in [0, 1).")
        if not 0.0 <= self.slip_rate < 1.0:
            raise ValueError("Slip rate must be in [0, 1).")
        if self.guess_rate + self.slip_rate >= 1.0:
            raise ValueError("Guess and slip rates must sum to less than one.")
        if not 0.0 <= self.evidence_quality <= 1.0:
            raise ValueError("Evidence quality must be in [0, 1].")
        if not 0.0 <= self.feedback_quality <= 1.0:
            raise ValueError("Feedback quality must be in [0, 1].")


@dataclass(frozen=True, slots=True)
class GaussianSummary:
    mean: float
    variance: float
    mastery_probability: float
    expected_competence: float


@dataclass(frozen=True, slots=True)
class GridSummary:
    mean: float
    variance: float
    mastery_probability: float
    expected_competence: float
    gaussian_mastery_probability: float
    edge_mass: float
    log_normalizer: float


@dataclass(frozen=True, slots=True)
class GridSpec:
    name: str
    lower: float
    upper: float
    step: float

    @property
    def points(self) -> int:
        return int(round((self.upper - self.lower) / self.step)) + 1


GRID_SPECS = (
    GridSpec("coarse", -10.0, 10.0, 0.02),
    GridSpec("reference", -12.0, 12.0, 0.01),
    GridSpec("fine", -14.0, 14.0, 0.005),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LabInvariantError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def family_dependence_discount(prior_attempts: int) -> float:
    if type(prior_attempts) is not int or prior_attempts < 0:
        raise ValueError("Prior family attempts must be a non-negative integer.")
    if prior_attempts == 0:
        return 1.0
    return 0.25 / float(prior_attempts**2)


def family_budget(prior_attempts: int, simultaneous_count: int) -> float:
    """Return the order-free family influence available to one timestamp batch."""

    if type(simultaneous_count) is not int or simultaneous_count <= 0:
        raise ValueError("Simultaneous family count must be a positive integer.")
    return math.fsum(
        family_dependence_discount(prior_attempts + offset)
        for offset in range(simultaneous_count)
    )


def _canonical_factors(factors: Iterable[ResponseFactor]) -> tuple[ResponseFactor, ...]:
    ordered = tuple(
        sorted(
            factors,
            key=lambda factor: (
                factor.family_id,
                factor.item_id,
                factor.difficulty,
                factor.discrimination,
                factor.guess_rate,
                factor.slip_rate,
                factor.correct,
                factor.evidence_quality,
                factor.feedback_quality,
            ),
        )
    )
    ids = [factor.item_id for factor in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("One evidence epoch cannot repeat an item ID.")
    if not ordered:
        raise ValueError("An evidence epoch must contain at least one response.")
    return ordered


def symmetric_family_exponents(
    factors: Iterable[ResponseFactor],
    prior_family_attempts: Mapping[str, int] | None = None,
) -> dict[str, float]:
    """Allocate each family's bounded batch budget without an arrival-order rank.

    For family ``f`` with ``k`` historical attempts and ``n`` simultaneous
    responses, the group receives at most
    ``max(quality) * sum(d(k+r), r=0..n-1)`` evidence-equivalents. Raw quality
    weights are scaled proportionally only when they exceed that cap.
    Identical-quality repetitions therefore reproduce the old cumulative tail,
    heterogeneous observations are handled symmetrically, and adding a
    zero-quality observation cannot dilute evidence already in the batch.
    """

    ordered = _canonical_factors(factors)
    prior_counts = dict(prior_family_attempts or {})
    grouped: dict[str, list[ResponseFactor]] = {}
    for factor in ordered:
        grouped.setdefault(factor.family_id, []).append(factor)
    result: dict[str, float] = {}
    for family_id in sorted(grouped):
        group = grouped[family_id]
        prior_count = prior_counts.get(family_id, 0)
        if type(prior_count) is not int or prior_count < 0:
            raise ValueError("Historical family counts must be non-negative integers.")
        budget = family_budget(prior_count, len(group))
        total_quality = math.fsum(
            factor.evidence_quality for factor in group
        )
        maximum_quality = max(factor.evidence_quality for factor in group)
        cap = budget * maximum_quality
        scale = (
            0.0
            if total_quality == 0.0
            else min(1.0, cap / total_quality)
        )
        for factor in group:
            result[factor.item_id] = scale * factor.evidence_quality
    return result


def response_probability(factor: ResponseFactor, theta: float) -> tuple[float, float]:
    logistic = sigmoid(
        factor.discrimination * (theta - factor.difficulty)
    )
    modeled = factor.guess_rate + (
        1.0 - factor.guess_rate - factor.slip_rate
    ) * logistic
    probability = (
        SESSION_LAPSE_RATE * 0.25
        + (1.0 - SESSION_LAPSE_RATE) * modeled
    )
    derivative = (
        (1.0 - SESSION_LAPSE_RATE)
        * (1.0 - factor.guess_rate - factor.slip_rate)
        * factor.discrimination
        * logistic
        * (1.0 - logistic)
    )
    return probability, derivative


def log_response_likelihood(factor: ResponseFactor, theta: float) -> float:
    probability, _ = response_probability(factor, theta)
    likelihood = probability if factor.correct else 1.0 - probability
    return math.log(max(1e-300, likelihood))


def gaussian_summary(mean: float, variance: float) -> GaussianSummary:
    state = SkillState("lab", "objective", mean, variance, INITIAL_STABILITY_HOURS)
    return GaussianSummary(
        mean=state.mean,
        variance=state.variance,
        mastery_probability=state.mastery_probability,
        expected_competence=state.expected_competence,
    )


def sequential_adf(
    factors: Sequence[ResponseFactor],
    *,
    prior_mean: float,
    prior_variance: float,
    feedback_shown: bool,
    prior_family_attempts: Mapping[str, int] | None = None,
) -> GaussianSummary:
    """Reproduce the current scalar v5 update, including order-ranked families."""

    mean = prior_mean
    variance = prior_variance
    family_counts = dict(prior_family_attempts or {})
    for factor in factors:
        before_mean = mean
        before_variance = variance
        count = family_counts.get(factor.family_id, 0)
        dependence = family_dependence_discount(count)
        probability, derivative = response_probability(factor, mean)
        y = 1.0 if factor.correct else 0.0
        denominator = max(1e-6, probability * (1.0 - probability))
        score = (y - probability) * derivative / denominator
        fisher = derivative * derivative / denominator
        evidence_weight = factor.evidence_quality * dependence
        variance = max(
            MIN_POSTERIOR_VARIANCE,
            1.0 / (1.0 / variance + evidence_weight * fisher),
        )
        mean += variance * evidence_weight * score
        feedback_weight = factor.feedback_quality * dependence
        if feedback_shown and factor.correct:
            mean += (
                0.025
                * feedback_weight
                * (1.0 - sigmoid(mean))
            )
        variance = min(
            MAX_POSTERIOR_VARIANCE,
            variance + (0.02 * feedback_weight if feedback_shown else 0.0),
        )
        if not factor.correct:
            # Reproduce v5's two published no-gain caps as well as its
            # precision cap.  The all-correct permutation measurements do not
            # exercise this branch, but keeping the comparator faithful makes
            # counterfactual lab extensions safe.
            variance = min(before_variance, variance)
            mastery_boundary = logit(MASTERY_THRESHOLD)
            mastery_mean_cap = mastery_boundary + math.sqrt(
                variance / before_variance
            ) * (before_mean - mastery_boundary)
            competence_mean_cap = math.sqrt(
                (1.0 + math.pi * variance / 8.0)
                / (1.0 + math.pi * before_variance / 8.0)
            ) * before_mean
            mean = min(
                mean,
                before_mean,
                mastery_mean_cap,
                competence_mean_cap,
            )
        mean = max(-6.0, min(6.0, mean))
        family_counts[factor.family_id] = count + 1
    return gaussian_summary(mean, variance)


def _linear_partial_integral(
    left_x: float,
    right_x: float,
    left_y: float,
    right_y: float,
    boundary: float,
) -> float:
    width = boundary - left_x
    full_width = right_x - left_x
    boundary_y = left_y + (right_y - left_y) * width / full_width
    return 0.5 * width * (left_y + boundary_y)


def grid_posterior(
    factors: Iterable[ResponseFactor],
    *,
    prior_mean: float,
    prior_variance: float,
    spec: GridSpec,
    prior_family_attempts: Mapping[str, int] | None = None,
) -> GridSummary:
    """Moment and tail summaries of a canonical fixed-grid posterior."""

    ordered = _canonical_factors(factors)
    exponents = symmetric_family_exponents(ordered, prior_family_attempts)
    xs = [spec.lower + index * spec.step for index in range(spec.points)]
    log_densities: list[float] = []
    prior_constant = -0.5 * math.log(2.0 * math.pi * prior_variance)
    for theta in xs:
        terms = [
            exponents[factor.item_id]
            * log_response_likelihood(factor, theta)
            for factor in ordered
        ]
        log_densities.append(
            prior_constant
            - 0.5 * (theta - prior_mean) ** 2 / prior_variance
            + math.fsum(terms)
        )
    peak = max(log_densities)
    density = [math.exp(value - peak) for value in log_densities]
    trapezoids = [
        0.5 * spec.step * (density[index] + density[index + 1])
        for index in range(len(density) - 1)
    ]
    normalizer = math.fsum(trapezoids)
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise LabInvariantError("Grid posterior has no finite normalizer.")

    first_moment = math.fsum(
        0.5
        * spec.step
        * (
            xs[index] * density[index]
            + xs[index + 1] * density[index + 1]
        )
        for index in range(len(density) - 1)
    ) / normalizer
    second_moment = math.fsum(
        0.5
        * spec.step
        * (
            xs[index] ** 2 * density[index]
            + xs[index + 1] ** 2 * density[index + 1]
        )
        for index in range(len(density) - 1)
    ) / normalizer
    variance = max(MIN_POSTERIOR_VARIANCE, second_moment - first_moment**2)
    expected_competence = math.fsum(
        0.5
        * spec.step
        * (
            sigmoid(xs[index]) * density[index]
            + sigmoid(xs[index + 1]) * density[index + 1]
        )
        for index in range(len(density) - 1)
    ) / normalizer

    boundary = logit(MASTERY_THRESHOLD)
    if boundary <= spec.lower:
        below = 0.0
    elif boundary >= spec.upper:
        below = normalizer
    else:
        boundary_index = int((boundary - spec.lower) // spec.step)
        boundary_index = min(boundary_index, len(xs) - 2)
        below = math.fsum(trapezoids[:boundary_index])
        below += _linear_partial_integral(
            xs[boundary_index],
            xs[boundary_index + 1],
            density[boundary_index],
            density[boundary_index + 1],
            boundary,
        )
    mastery_probability = max(0.0, min(1.0, 1.0 - below / normalizer))
    gaussian = gaussian_summary(first_moment, variance)
    edge_mass = (
        0.5
        * spec.step
        * (density[0] + density[-1])
        / normalizer
    )
    return GridSummary(
        mean=first_moment,
        variance=variance,
        mastery_probability=mastery_probability,
        expected_competence=expected_competence,
        gaussian_mastery_probability=gaussian.mastery_probability,
        edge_mass=edge_mass,
        log_normalizer=math.log(normalizer) + peak,
    )


def batch_fisher_laplace(
    factors: Iterable[ResponseFactor],
    *,
    prior_mean: float,
    prior_variance: float,
    prior_family_attempts: Mapping[str, int] | None = None,
) -> tuple[GaussianSummary, int]:
    """Canonical scalar batch Fisher solve used as a Gaussian comparator."""

    ordered = _canonical_factors(factors)
    exponents = symmetric_family_exponents(ordered, prior_family_attempts)

    def objective(theta: float) -> float:
        return (
            -0.5 * (theta - prior_mean) ** 2 / prior_variance
            + math.fsum(
                exponents[factor.item_id]
                * log_response_likelihood(factor, theta)
                for factor in ordered
            )
        )

    mean = prior_mean
    iterations = 0
    for iterations in range(1, 129):
        gradients: list[float] = []
        informations: list[float] = []
        for factor in ordered:
            probability, derivative = response_probability(factor, mean)
            y = 1.0 if factor.correct else 0.0
            denominator = max(1e-12, probability * (1.0 - probability))
            exponent = exponents[factor.item_id]
            gradients.append(
                exponent * (y - probability) * derivative / denominator
            )
            informations.append(
                exponent * derivative * derivative / denominator
            )
        gradient = -(mean - prior_mean) / prior_variance + math.fsum(gradients)
        precision = 1.0 / prior_variance + math.fsum(informations)
        if not math.isfinite(precision) or precision <= 0.0:
            raise LabInvariantError("Batch Fisher precision is not positive and finite.")
        step = gradient / precision
        old_objective = objective(mean)
        scale = 1.0
        while scale >= 2.0**-30 and objective(mean + scale * step) < old_objective:
            scale *= 0.5
        if scale < 2.0**-30:
            raise LabInvariantError("Batch Fisher line search failed to improve.")
        updated = mean + scale * step
        if abs(updated - mean) <= 1e-12 * max(1.0, abs(mean)):
            mean = updated
            break
        mean = updated
    else:
        raise LabInvariantError("Batch Fisher solve did not converge.")

    informations = []
    for factor in ordered:
        probability, derivative = response_probability(factor, mean)
        denominator = max(1e-12, probability * (1.0 - probability))
        informations.append(
            exponents[factor.item_id]
            * derivative
            * derivative
            / denominator
        )
    variance = 1.0 / (1.0 / prior_variance + math.fsum(informations))
    return gaussian_summary(mean, variance), iterations


def retention_project(
    summary: GridSummary,
    *,
    prior_mean: float,
    stability_hours: float,
    elapsed_hours: float,
) -> tuple[float, float]:
    """Collapse one epoch, then apply the production retention shape."""

    if elapsed_hours < 0.0:
        raise ValueError("Elapsed retention time cannot be negative.")
    if summary.mean <= prior_mean:
        return summary.mean, summary.variance
    retention = 2.0 ** (-elapsed_hours / max(12.0, stability_hours))
    mean = prior_mean + retention * (summary.mean - prior_mean)
    variance = min(
        MAX_POSTERIOR_VARIANCE,
        summary.variance + (1.0 - retention) * 0.45,
    )
    # Match the production guard: forgetting cannot increase a below-boundary
    # Gaussian tail merely by widening its variance.
    mastery_boundary = logit(MASTERY_THRESHOLD)
    if summary.mean < mastery_boundary and mean < mastery_boundary:
        old_gap = mastery_boundary - summary.mean
        new_gap = mastery_boundary - mean
        variance = min(
            variance,
            summary.variance * (new_gap / old_gap) ** 2,
        )
    if summary.mean < -1e-12 and mean < 0.0:
        scale_term = math.pi / 8.0
        ratio = mean / summary.mean
        if ratio < 1e150:
            variance = min(
                variance,
                (
                    ratio**2 * (1.0 + scale_term * summary.variance) - 1.0
                )
                / scale_term,
            )
    return mean, max(summary.variance, variance)


def two_epoch_projection(
    first: Sequence[ResponseFactor],
    second: Sequence[ResponseFactor],
    *,
    prior_mean: float,
    prior_variance: float,
    elapsed_hours: float,
    spec: GridSpec,
) -> GridSummary:
    first_summary = grid_posterior(
        first,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        spec=spec,
    )
    retained_mean, retained_variance = retention_project(
        first_summary,
        prior_mean=prior_mean,
        stability_hours=INITIAL_STABILITY_HOURS,
        elapsed_hours=elapsed_hours,
    )
    return grid_posterior(
        second,
        prior_mean=retained_mean,
        prior_variance=retained_variance,
        spec=spec,
    )


def _load_objective_factors(
    corpus_path: Path,
) -> tuple[dict[str, float], dict[str, tuple[ResponseFactor, ...]]]:
    bundle = json.loads(corpus_path.read_text(encoding="utf-8"))
    priors = {
        objective["id"]: float(objective["prior_mastery"])
        for objective in bundle.get("learning_objectives", [])
    }
    groups: dict[str, list[ResponseFactor]] = {objective_id: [] for objective_id in priors}
    for question in bundle["questions"]:
        objective_id = question.get("learning_objective_id")
        if objective_id is None:
            continue
        status = question["status"]
        if status not in {"approved", "calibrated"}:
            continue
        evidence_weight = 1.0 if status == "calibrated" else APPROVED_EVIDENCE_WEIGHT
        groups[objective_id].append(
            ResponseFactor(
                item_id=question["id"],
                family_id=question["family_id"],
                difficulty=float(question["difficulty"]),
                discrimination=float(question["discrimination"]),
                guess_rate=float(question["guess_rate"]),
                slip_rate=float(question["slip_rate"]),
                correct=True,
                evidence_quality=evidence_weight,
                feedback_quality=evidence_weight,
            )
        )
    return priors, {
        objective_id: _canonical_factors(factors)
        for objective_id, factors in groups.items()
        if factors
    }


def _permutation_extremes(
    factors: Sequence[ResponseFactor],
    *,
    prior_mean: float,
    prior_variance: float,
    feedback_shown: bool,
) -> dict[str, Any]:
    values: list[tuple[float, GaussianSummary, tuple[str, ...]]] = []
    for order in itertools.permutations(factors):
        summary = sequential_adf(
            order,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            feedback_shown=feedback_shown,
        )
        values.append(
            (
                summary.mastery_probability,
                summary,
                tuple(factor.item_id for factor in order),
            )
        )
    minimum = min(values, key=lambda item: (item[0], item[2]))
    maximum = max(values, key=lambda item: (item[0], item[2]))
    return {
        "permutations": len(values),
        "minimum": {
            "mastery_probability": minimum[1].mastery_probability,
            "expected_competence": minimum[1].expected_competence,
            "mean": minimum[1].mean,
            "variance": minimum[1].variance,
            "order": minimum[2],
        },
        "maximum": {
            "mastery_probability": maximum[1].mastery_probability,
            "expected_competence": maximum[1].expected_competence,
            "mean": maximum[1].mean,
            "variance": maximum[1].variance,
            "order": maximum[2],
        },
        "mastery_range": maximum[0] - minimum[0],
        "crosses_proficient_threshold": (
            minimum[0] < 0.75 <= maximum[0]
        ),
    }


def _grid_as_dict(summary: GridSummary) -> dict[str, float]:
    return {
        "mean": summary.mean,
        "variance": summary.variance,
        "mastery_probability": summary.mastery_probability,
        "expected_competence": summary.expected_competence,
        "gaussian_mastery_probability": summary.gaussian_mastery_probability,
        "edge_mass": summary.edge_mass,
        "log_normalizer": summary.log_normalizer,
    }


def _benchmark_grid(
    factors: Sequence[ResponseFactor],
    *,
    prior_mean: float,
    prior_variance: float,
    spec: GridSpec,
    repetitions: int = 31,
) -> dict[str, float | int]:
    durations: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter()
        grid_posterior(
            factors,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            spec=spec,
        )
        durations.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(durations)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "factors": len(factors),
        "grid_points": spec.points,
        "repetitions": repetitions,
        "median_ms": statistics.median(durations),
        "p95_ms": ordered[p95_index],
    }


def build_report(corpus_path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    priors, objectives = _load_objective_factors(corpus_path)
    if not objectives:
        raise LabInvariantError("The corpus has no objective-linked active questions.")
    objective_reports: dict[str, Any] = {}
    permutation_digests: list[str] = []
    wrong_deltas: list[float] = []

    for objective_id, factors in sorted(objectives.items()):
        prior_mean = logit(priors[objective_id])
        static_extremes = _permutation_extremes(
            factors,
            prior_mean=prior_mean,
            prior_variance=INITIAL_VARIANCE,
            feedback_shown=False,
        )
        feedback_extremes = _permutation_extremes(
            factors,
            prior_mean=prior_mean,
            prior_variance=INITIAL_VARIANCE,
            feedback_shown=True,
        )
        grids = {
            spec.name: grid_posterior(
                factors,
                prior_mean=prior_mean,
                prior_variance=INITIAL_VARIANCE,
                spec=spec,
            )
            for spec in GRID_SPECS
        }
        reference = grids["reference"]
        fine = grids["fine"]
        laplace, laplace_iterations = batch_fisher_laplace(
            factors,
            prior_mean=prior_mean,
            prior_variance=INITIAL_VARIANCE,
        )

        reversed_reference = grid_posterior(
            tuple(reversed(factors)),
            prior_mean=prior_mean,
            prior_variance=INITIAL_VARIANCE,
            spec=GRID_SPECS[1],
        )
        reference_payload = _grid_as_dict(reference)
        require(
            reference_payload == _grid_as_dict(reversed_reference),
            f"{objective_id} grid posterior changed under permutation.",
        )
        permutation_digests.append(canonical_digest(reference_payload))

        wrong_factors = tuple(
            ResponseFactor(
                item_id=factor.item_id,
                family_id=factor.family_id,
                difficulty=factor.difficulty,
                discrimination=factor.discrimination,
                guess_rate=factor.guess_rate,
                slip_rate=factor.slip_rate,
                correct=False,
                evidence_quality=factor.evidence_quality,
                feedback_quality=factor.feedback_quality,
            )
            for factor in factors
        )
        wrong = grid_posterior(
            wrong_factors,
            prior_mean=prior_mean,
            prior_variance=INITIAL_VARIANCE,
            spec=GRID_SPECS[1],
        )
        prior = gaussian_summary(prior_mean, INITIAL_VARIANCE)
        wrong_delta = wrong.mastery_probability - prior.mastery_probability
        wrong_deltas.append(wrong_delta)
        require(
            wrong_delta <= 1e-10,
            f"{objective_id} wrong-only batch raised mastery probability.",
        )
        require(
            wrong.expected_competence <= prior.expected_competence + 1e-10,
            f"{objective_id} wrong-only batch raised expected competence.",
        )

        objective_reports[objective_id] = {
            "items": len(factors),
            "current_static_adf": static_extremes,
            "current_feedback_adf": feedback_extremes,
            "grid": {
                name: _grid_as_dict(summary)
                for name, summary in grids.items()
            },
            "reference_fine_absolute_delta": {
                "mean": abs(reference.mean - fine.mean),
                "variance": abs(reference.variance - fine.variance),
                "mastery_probability": abs(
                    reference.mastery_probability - fine.mastery_probability
                ),
                "expected_competence": abs(
                    reference.expected_competence - fine.expected_competence
                ),
            },
            "batch_fisher_laplace": {
                "mean": laplace.mean,
                "variance": laplace.variance,
                "mastery_probability": laplace.mastery_probability,
                "expected_competence": laplace.expected_competence,
                "iterations": laplace_iterations,
            },
            "wrong_only": {
                "mastery_probability": wrong.mastery_probability,
                "expected_competence": wrong.expected_competence,
                "mastery_delta_from_prior": wrong_delta,
            },
            "finite_bank_thresholds": {
                "grid_proficient": reference.mastery_probability >= 0.75,
                "grid_durable": reference.mastery_probability >= 0.90,
                "laplace_proficient": laplace.mastery_probability >= 0.75,
                "laplace_durable": laplace.mastery_probability >= 0.90,
            },
        }

    threshold_flips = sorted(
        objective_id
        for objective_id, report in objective_reports.items()
        if report["current_feedback_adf"]["crosses_proficient_threshold"]
    )
    require(threshold_flips, "Current ADF did not reproduce a proficiency threshold flip.")

    max_convergence_delta = max(
        report["reference_fine_absolute_delta"]["mastery_probability"]
        for report in objective_reports.values()
    )
    max_reference_edge_mass = max(
        report["grid"]["reference"]["edge_mass"]
        for report in objective_reports.values()
    )
    require(
        max_convergence_delta < 5e-5,
        "Reference and fine grids do not agree on mastery tail probability.",
    )
    require(
        max_reference_edge_mass < 1e-12,
        "Reference grid leaves material density at a boundary.",
    )

    repeated = tuple(
        ResponseFactor(
            item_id=f"repeat_{index:04d}",
            family_id="one_family",
            difficulty=0.5 + 0.001 * index,
            discrimination=1.5,
            guess_rate=0.25,
            slip_rate=0.07,
            correct=True,
        )
        for index in range(1000)
    )
    repeated_exponents = symmetric_family_exponents(repeated)
    reverse_exponents = symmetric_family_exponents(tuple(reversed(repeated)))
    require(
        repeated_exponents == reverse_exponents,
        "Simultaneous family allocation depends on arrival order.",
    )
    total_repeated_weight = math.fsum(repeated_exponents.values())
    theoretical_family_limit = APPROVED_EVIDENCE_WEIGHT * (
        1.0 + 0.25 * math.pi**2 / 6.0
    )
    require(
        total_repeated_weight < theoretical_family_limit,
        "Finite repeated-family weight exceeded its analytic infinite limit.",
    )

    # The same factors cease to commute intentionally when assigned to distinct
    # time epochs because retention acts between the two likelihood products.
    timing_objective = max(objectives, key=lambda key: len(objectives[key]))
    timing_factors = sorted(
        objectives[timing_objective], key=lambda factor: factor.difficulty
    )
    split = len(timing_factors) // 2
    easy = tuple(timing_factors[:split])
    hard = tuple(timing_factors[split:])
    timing_prior = logit(priors[timing_objective])
    easy_then_hard = two_epoch_projection(
        easy,
        hard,
        prior_mean=timing_prior,
        prior_variance=INITIAL_VARIANCE,
        elapsed_hours=24.0 * 7.0,
        spec=GRID_SPECS[1],
    )
    hard_then_easy = two_epoch_projection(
        hard,
        easy,
        prior_mean=timing_prior,
        prior_variance=INITIAL_VARIANCE,
        elapsed_hours=24.0 * 7.0,
        spec=GRID_SPECS[1],
    )
    temporal_delta = abs(
        easy_then_hard.mastery_probability
        - hard_then_easy.mastery_probability
    )
    require(
        temporal_delta > 1e-4,
        "Separated epochs unexpectedly hid retention order.",
    )

    benchmark_objective = max(objectives, key=lambda key: len(objectives[key]))
    benchmark = _benchmark_grid(
        objectives[benchmark_objective],
        prior_mean=logit(priors[benchmark_objective]),
        prior_variance=INITIAL_VARIANCE,
        spec=GRID_SPECS[1],
    )

    deterministic = {
        "lab_version": LAB_VERSION,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "grid_specs": [
            {
                "name": spec.name,
                "lower": spec.lower,
                "upper": spec.upper,
                "step": spec.step,
                "points": spec.points,
            }
            for spec in GRID_SPECS
        ],
        "objectives": objective_reports,
        "findings": {
            "feedback_adf_proficiency_threshold_flips": threshold_flips,
            "maximum_reference_fine_tail_delta": max_convergence_delta,
            "maximum_reference_edge_mass": max_reference_edge_mass,
            "wrong_only_maximum_mastery_delta": max(wrong_deltas),
            "permutation_commitment": canonical_digest(permutation_digests),
            "repeated_family": {
                "simultaneous_responses": len(repeated),
                "total_effective_weight": total_repeated_weight,
                "infinite_effective_weight_limit": theoretical_family_limit,
                "order_invariant": repeated_exponents == reverse_exponents,
            },
            "spaced_epochs": {
                "objective_id": timing_objective,
                "elapsed_hours": 168.0,
                "easy_then_hard_mastery": easy_then_hard.mastery_probability,
                "hard_then_easy_mastery": hard_then_easy.mastery_probability,
                "absolute_delta": temporal_delta,
                "retention_order_preserved": True,
            },
            "recommendation": {
                "certification": (
                    "Use the converged posterior P(theta > mastery threshold), "
                    "or its validated numerical lower envelope, plus independent-"
                    "family/operation/delayed-retrieval gates. Do not certify from "
                    "a Gaussian tail reconstructed only from mean and variance."
                ),
                "routing": (
                    "Use item-specific posterior predictive probability for "
                    "challenge and information scoring; expected sigmoid(theta) "
                    "is only a fallback scalar."
                ),
                "retention": (
                    "Apply retention and acquisition between chronological epochs; "
                    "do not include them in the exchangeable likelihood product."
                ),
            },
        },
    }
    return {
        **deterministic,
        "artifact_sha256": canonical_digest(deterministic),
        "runtime_benchmark": {
            "objective_id": benchmark_objective,
            **benchmark,
            "note": "Wall-clock measurements are excluded from artifact_sha256.",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout", action="store_true")
    arguments = parser.parse_args(argv)
    report = build_report(arguments.corpus)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.stdout:
        sys.stdout.write(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
        receipt = {
            "artifact_sha256": report["artifact_sha256"],
            "objectives": len(report["objectives"]),
            "threshold_flips": report["findings"][
                "feedback_adf_proficiency_threshold_flips"
            ],
            "maximum_reference_fine_tail_delta": report["findings"][
                "maximum_reference_fine_tail_delta"
            ],
            "runtime_benchmark": report["runtime_benchmark"],
            "output": str(arguments.output),
        }
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
