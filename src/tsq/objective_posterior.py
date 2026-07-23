# SPDX-License-Identifier: MPL-2.0

"""Portable one-dimensional posterior for fine learning objectives.

The exact-grid v6/v7 online learners use this versionable projection primitive
for named learning objectives. It retains a full scalar density on a fixed
grid, rather than collapsing every response to a Gaussian. Static response
likelihoods within one epoch are recomputed in canonical order, so their result
is independent of arrival order. Retention and feedback are explicit Markov
transitions that close an epoch; they therefore remain chronological and cannot
be mistaken for scored evidence.

The fixed grid covers ``[-18, 18]`` in exact binary steps of ``1/32``.  It is six
standard deviations beyond the current v5 mean/variance boundary (mean +/- 6,
variance <= 4).  Every public construction and transition fails closed when
normalization, finite-number, or edge-mass checks indicate that this bounded
representation is no longer trustworthy.

State uses canonical UTF-8 JSON rather than an implementation-specific pickle,
compressed stream, or native-endian float array.  It is intentionally verbose
(about 23 KiB for an empty epoch) but inspectable, byte-stable across supported
Python platforms, and suitable for content hashing.  A future compact codec
must receive a new explicit codec and schema identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ObjectivePosteriorIdentity:
    """Immutable on-disk identity for one posterior representation."""

    schema_version: int
    algorithm: str
    grid_id: str
    codec: str


OBJECTIVE_POSTERIOR_V1_IDENTITY = ObjectivePosteriorIdentity(
    schema_version=1,
    algorithm="objective-density-grid-v1",
    grid_id="theta[-18,18]/32-v1",
    codec="canonical-json-utf8-v1",
)
SUPPORTED_OBJECTIVE_POSTERIOR_IDENTITIES = frozenset(
    {OBJECTIVE_POSTERIOR_V1_IDENTITY}
)

# Compatibility names remain available to readers and writers, but they are
# explicitly bound to v1 rather than acting as mutable "current" identities.
OBJECTIVE_POSTERIOR_SCHEMA_VERSION = (
    OBJECTIVE_POSTERIOR_V1_IDENTITY.schema_version
)
OBJECTIVE_POSTERIOR_ALGORITHM = OBJECTIVE_POSTERIOR_V1_IDENTITY.algorithm
OBJECTIVE_POSTERIOR_GRID_ID = OBJECTIVE_POSTERIOR_V1_IDENTITY.grid_id
OBJECTIVE_POSTERIOR_CODEC = OBJECTIVE_POSTERIOR_V1_IDENTITY.codec

GRID_LOWER = -18.0
GRID_UPPER = 18.0
GRID_STEPS_PER_UNIT = 32
GRID_STEP = 1.0 / GRID_STEPS_PER_UNIT
GRID_INTERVALS = int((GRID_UPPER - GRID_LOWER) * GRID_STEPS_PER_UNIT)
GRID_SIZE = GRID_INTERVALS + 1
THETA_GRID = tuple(
    GRID_LOWER + index / GRID_STEPS_PER_UNIT
    for index in range(GRID_SIZE)
)

# These bounds deliberately mirror the existing model's representable Gaussian
# envelope without importing that reducer or coupling its model version here.
V5_MEAN_ABS_BOUND = 6.0
V5_MIN_VARIANCE = 1e-6
V5_MAX_VARIANCE = 4.0
DEFAULT_PRIOR_VARIANCE = 2.25
DEFAULT_LAPSE_RATE = 0.03
DEFAULT_OPTION_COUNT = 4
MIN_OPTION_COUNT = 2
MAX_OPTION_COUNT = 64
DEFAULT_MASTERY_THRESHOLD = 0.65
MIN_RETENTION_HALF_LIFE_HOURS = 12.0

# Eight nodes span one quarter of a theta unit at each boundary.  The maximum
# permits the current v5 worst-case Gaussian envelope but rejects a posterior
# that is beginning to rely materially on a truncated tail.
EDGE_BAND_NODES = 8
MAX_EDGE_MASS = 1e-8
NORMALIZATION_TOLERANCE = 5e-12
METRIC_TOLERANCE = 2e-12

# Across the active objective bank, this 1/32 grid differs from an independent
# 1/256 integration by less than 3e-5 in P(theta > logit(.65)).  The larger
# declared bound is intentionally available to policy so threshold gates can
# use ``max(0, probability - bound)`` rather than exploit numerical noise.
MASTERY_PROBABILITY_ERROR_BOUND = 5e-5
GAUSSIAN_MEAN_REPRESENTATION_TOLERANCE = GRID_STEP / 64.0
GAUSSIAN_VARIANCE_RELATIVE_TOLERANCE = 0.05
GAUSSIAN_VARIANCE_ABSOLUTE_TOLERANCE = 1e-8

# A correct-feedback transition has approximately the same small theta-scale as
# v5's hand-designed transition, but is represented as uncertain upward movement
# rather than a deterministic mean increment.
CORRECT_FEEDBACK_THETA_RATE = 0.025
MAX_ACQUISITION_STRENGTH = 1.0


class ObjectivePosteriorError(ValueError):
    """Raised when an objective posterior cannot be trusted or represented."""


def _is_finite_number(value: object) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _require_finite(value: object, label: str) -> float:
    if not _is_finite_number(value):
        raise ObjectivePosteriorError(f"{label} must be a finite number.")
    result = float(value)
    # JSON distinguishes -0.0 from 0.0 textually even though the model does
    # not.  Collapse signed zero so semantically equal states hash identically.
    return 0.0 if result == 0.0 else result


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    maximum = max(left, right)
    return maximum + math.log(
        math.exp(left - maximum) + math.exp(right - maximum)
    )


_LOG_QUADRATURE_WEIGHTS = tuple(
    math.log(GRID_STEP * (0.5 if index in {0, GRID_SIZE - 1} else 1.0))
    for index in range(GRID_SIZE)
)


def _log_integral(log_density: tuple[float, ...]) -> float:
    terms = tuple(
        value + _LOG_QUADRATURE_WEIGHTS[index]
        for index, value in enumerate(log_density)
    )
    maximum = max(terms)
    return maximum + math.log(
        math.fsum(math.exp(term - maximum) for term in terms)
    )


def _normalize_log_density(values: Iterable[float]) -> tuple[float, ...]:
    log_density = tuple(float(value) for value in values)
    if len(log_density) != GRID_SIZE:
        raise ObjectivePosteriorError(
            f"A posterior density must contain exactly {GRID_SIZE} grid values."
        )
    if any(not math.isfinite(value) for value in log_density):
        raise ObjectivePosteriorError("Posterior log density must be entirely finite.")
    normalizer = _log_integral(log_density)
    if not math.isfinite(normalizer):
        raise ObjectivePosteriorError("Posterior density has no finite normalizer.")
    normalized = tuple(value - normalizer for value in log_density)
    if any(not math.isfinite(value) for value in normalized):
        raise ObjectivePosteriorError("Normalized posterior density is not finite.")
    return normalized


def _scaled_density(log_density: tuple[float, ...]) -> tuple[tuple[float, ...], float]:
    peak = max(log_density)
    density = tuple(math.exp(value - peak) for value in log_density)
    normalizer = math.fsum(
        density[index]
        * GRID_STEP
        * (0.5 if index in {0, GRID_SIZE - 1} else 1.0)
        for index in range(GRID_SIZE)
    )
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ObjectivePosteriorError("Posterior metric integral is not finite.")
    return density, normalizer


def _edge_mass(log_density: tuple[float, ...]) -> float:
    density, normalizer = _scaled_density(log_density)
    edge_indexes = tuple(range(EDGE_BAND_NODES)) + tuple(
        range(GRID_SIZE - EDGE_BAND_NODES, GRID_SIZE)
    )
    return math.fsum(
        density[index]
        * GRID_STEP
        * (0.5 if index in {0, GRID_SIZE - 1} else 1.0)
        for index in edge_indexes
    ) / normalizer


def _validate_log_density(log_density: object, label: str) -> tuple[float, ...]:
    if type(log_density) is not tuple:
        raise ObjectivePosteriorError(f"{label} must be an immutable tuple.")
    if len(log_density) != GRID_SIZE:
        raise ObjectivePosteriorError(
            f"{label} must contain exactly {GRID_SIZE} grid values."
        )
    values: list[float] = []
    for index, value in enumerate(log_density):
        values.append(_require_finite(value, f"{label}[{index}]"))
    result = tuple(values)
    normalization_error = abs(_log_integral(result))
    if normalization_error > NORMALIZATION_TOLERANCE:
        raise ObjectivePosteriorError(
            f"{label} is not normalized (log error {normalization_error:.3g})."
        )
    edge = _edge_mass(result)
    if not math.isfinite(edge) or edge > MAX_EDGE_MASS:
        raise ObjectivePosteriorError(
            f"{label} places unsafe mass at the grid edge ({edge:.6g})."
        )
    return result


def _gaussian_log_density(mean: float, variance: float) -> tuple[float, ...]:
    constant = -0.5 * math.log(2.0 * math.pi * variance)
    return _normalize_log_density(
        constant - 0.5 * (theta - mean) ** 2 / variance
        for theta in THETA_GRID
    )


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)


def _validate_item_parameters(
    *,
    difficulty: object,
    discrimination: object,
    guess_rate: object,
    slip_rate: object,
    option_count: object,
) -> tuple[float, float, float, float, int]:
    difficulty_value = _require_finite(difficulty, "Difficulty")
    discrimination_value = _require_finite(discrimination, "Discrimination")
    guess_value = _require_finite(guess_rate, "Guess rate")
    slip_value = _require_finite(slip_rate, "Slip rate")
    if not -6.0 <= difficulty_value <= 6.0:
        raise ObjectivePosteriorError("Difficulty must be between -6 and 6.")
    if not 0.0 < discrimination_value <= 4.0:
        raise ObjectivePosteriorError(
            "Discrimination must be greater than zero and at most four."
        )
    if not 0.0 <= guess_value < 1.0:
        raise ObjectivePosteriorError("Guess rate must be in [0, 1).")
    if not 0.0 <= slip_value < 1.0:
        raise ObjectivePosteriorError("Slip rate must be in [0, 1).")
    if guess_value + slip_value >= 1.0:
        raise ObjectivePosteriorError(
            "Guess and slip rates must sum to less than one."
        )
    if (
        type(option_count) is not int
        or not MIN_OPTION_COUNT <= option_count <= MAX_OPTION_COUNT
    ):
        raise ObjectivePosteriorError(
            f"Option count must be an integer in [{MIN_OPTION_COUNT}, {MAX_OPTION_COUNT}]."
        )
    return (
        difficulty_value,
        discrimination_value,
        guess_value,
        slip_value,
        option_count,
    )


def _response_probability(
    theta: float,
    *,
    difficulty: float,
    discrimination: float,
    guess_rate: float,
    slip_rate: float,
    option_count: int,
) -> float:
    logistic = _sigmoid(discrimination * (theta - difficulty))
    modeled = guess_rate + (1.0 - guess_rate - slip_rate) * logistic
    return (
        DEFAULT_LAPSE_RATE * (1.0 / option_count)
        + (1.0 - DEFAULT_LAPSE_RATE) * modeled
    )


@dataclass(frozen=True, slots=True)
class LikelihoodObservation:
    """One immutable, already-discounted response-likelihood factor.

    ``evidence_power`` must include release trust, hint, confidence, latency,
    and family-dependence discounts selected by the versioned caller.  Keeping
    that policy outside this scalar primitive makes the posterior reducer
    replayable without silently redefining evidence independence.
    """

    observation_id: str
    family_id: str
    difficulty: float
    discrimination: float
    guess_rate: float
    slip_rate: float
    option_count: int
    correct: bool
    evidence_power: float

    def __post_init__(self) -> None:
        if type(self.observation_id) is not str or not self.observation_id.strip():
            raise ObjectivePosteriorError(
                "Likelihood observations require a nonblank stable ID."
            )
        if type(self.family_id) is not str or not self.family_id.strip():
            raise ObjectivePosteriorError(
                "Likelihood observations require a nonblank family ID."
            )
        if type(self.correct) is not bool:
            raise ObjectivePosteriorError("Observation correctness must be boolean.")
        difficulty, discrimination, guess_rate, slip_rate, option_count = (
            _validate_item_parameters(
                difficulty=self.difficulty,
                discrimination=self.discrimination,
                guess_rate=self.guess_rate,
                slip_rate=self.slip_rate,
                option_count=self.option_count,
            )
        )
        evidence_power = _require_finite(
            self.evidence_power, "Evidence power"
        )
        if not 0.0 <= evidence_power <= 1.0:
            raise ObjectivePosteriorError("Evidence power must be in [0, 1].")
        object.__setattr__(self, "difficulty", difficulty)
        object.__setattr__(self, "discrimination", discrimination)
        object.__setattr__(self, "guess_rate", guess_rate)
        object.__setattr__(self, "slip_rate", slip_rate)
        object.__setattr__(self, "option_count", option_count)
        object.__setattr__(self, "evidence_power", evidence_power)

    @property
    def canonical_key(self) -> tuple[object, ...]:
        return (
            self.family_id,
            self.observation_id,
            self.difficulty,
            self.discrimination,
            self.guess_rate,
            self.slip_rate,
            self.option_count,
            self.correct,
            self.evidence_power,
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "family_id": self.family_id,
            "difficulty": self.difficulty,
            "discrimination": self.discrimination,
            "guess_rate": self.guess_rate,
            "slip_rate": self.slip_rate,
            "option_count": self.option_count,
            "correct": self.correct,
            "evidence_power": self.evidence_power,
        }


@dataclass(frozen=True, slots=True)
class PosteriorMetrics:
    mean: float
    variance: float
    mastery_probability: float
    expected_competence: float
    edge_mass: float
    evidence_mass: float
    acquisition_mass: float
    mastery_probability_error_bound: float

    @property
    def conservative_mastery_probability(self) -> float:
        return max(
            0.0,
            self.mastery_probability - self.mastery_probability_error_bound,
        )


@dataclass(frozen=True, slots=True)
class ExpectedInformation:
    """Exact fixed-grid information from both hypothetical item outcomes."""

    predicted_correct: float
    expected_information_nats: float
    expected_variance: float
    variance_reduction: float
    correct_mastery_probability: float
    incorrect_mastery_probability: float


def _canonical_observations(
    observations: Iterable[LikelihoodObservation],
) -> tuple[LikelihoodObservation, ...]:
    supplied = tuple(observations)
    if any(not isinstance(item, LikelihoodObservation) for item in supplied):
        raise ObjectivePosteriorError(
            "Every pending likelihood factor must be a LikelihoodObservation."
        )
    result = tuple(sorted(supplied, key=lambda item: item.canonical_key))
    ids = [item.observation_id for item in result]
    if len(ids) != len(set(ids)):
        raise ObjectivePosteriorError(
            "An evidence epoch cannot repeat an observation ID."
        )
    return result


def _apply_observations(
    anchor: tuple[float, ...],
    observations: tuple[LikelihoodObservation, ...],
) -> tuple[float, ...]:
    if not observations:
        return anchor
    updated: list[float] = []
    for index, theta in enumerate(THETA_GRID):
        likelihood_terms: list[float] = []
        for observation in observations:
            probability = _response_probability(
                theta,
                difficulty=observation.difficulty,
                discrimination=observation.discrimination,
                guess_rate=observation.guess_rate,
                slip_rate=observation.slip_rate,
                option_count=observation.option_count,
            )
            likelihood = probability if observation.correct else 1.0 - probability
            if not 0.0 < likelihood < 1.0 or not math.isfinite(likelihood):
                raise ObjectivePosteriorError(
                    "A response likelihood left the finite open unit interval."
                )
            likelihood_terms.append(
                observation.evidence_power * math.log(likelihood)
            )
        updated.append(anchor[index] + math.fsum(likelihood_terms))
    return _normalize_log_density(updated)


def _integrate_function(
    log_density: tuple[float, ...], values: Iterable[float]
) -> float:
    function_values = tuple(float(value) for value in values)
    if len(function_values) != GRID_SIZE or any(
        not math.isfinite(value) for value in function_values
    ):
        raise ObjectivePosteriorError(
            "A posterior integrand must be finite on the complete grid."
        )
    density, normalizer = _scaled_density(log_density)
    numerator = math.fsum(
        0.5
        * GRID_STEP
        * (
            function_values[index] * density[index]
            + function_values[index + 1] * density[index + 1]
        )
        for index in range(GRID_INTERVALS)
    )
    result = numerator / normalizer
    if not math.isfinite(result):
        raise ObjectivePosteriorError("A posterior expectation is not finite.")
    return result


def _kl_divergence(
    posterior: tuple[float, ...], prior: tuple[float, ...]
) -> float:
    result = _integrate_function(
        posterior,
        (
            posterior[index] - prior[index]
            for index in range(GRID_SIZE)
        ),
    )
    if result < -METRIC_TOLERANCE:
        raise ObjectivePosteriorError(
            "Posterior KL divergence became materially negative."
        )
    return max(0.0, result)


def _probability_above_theta_on_stride(
    log_density: tuple[float, ...], threshold: float, *, stride: int
) -> float:
    if not math.isfinite(threshold):
        raise ObjectivePosteriorError("A posterior threshold must be finite.")
    if type(stride) is not int or stride <= 0 or GRID_INTERVALS % stride != 0:
        raise ObjectivePosteriorError(
            "Tail-integration stride must divide the fixed grid exactly."
        )
    if threshold <= GRID_LOWER:
        return 1.0
    if threshold >= GRID_UPPER:
        return 0.0
    indexes = tuple(range(0, GRID_SIZE, stride))
    peak = max(log_density[index] for index in indexes)
    density = tuple(math.exp(log_density[index] - peak) for index in indexes)
    step = GRID_STEP * stride
    normalizer = math.fsum(
        density[index]
        * step
        * (0.5 if index in {0, len(indexes) - 1} else 1.0)
        for index in range(len(indexes))
    )
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise ObjectivePosteriorError("Posterior tail integral is not finite.")
    interval = int((threshold - GRID_LOWER) // step)
    interval = max(0, min(len(indexes) - 2, interval))
    below = math.fsum(
        0.5 * step * (density[index] + density[index + 1])
        for index in range(interval)
    )
    left = THETA_GRID[indexes[interval]]
    fraction = (threshold - left) / step
    threshold_density = density[interval] + fraction * (
        density[interval + 1] - density[interval]
    )
    partial_width = threshold - left
    below += 0.5 * partial_width * (
        density[interval] + threshold_density
    )
    result = 1.0 - below / normalizer
    if not math.isfinite(result):
        raise ObjectivePosteriorError("Posterior tail probability is not finite.")
    return max(0.0, min(1.0, result))


def _probability_above_theta(
    log_density: tuple[float, ...], threshold: float
) -> float:
    return _probability_above_theta_on_stride(
        log_density, threshold, stride=1
    )


def _mixture_log_density(
    left: tuple[float, ...],
    right: tuple[float, ...],
    left_weight: float,
) -> tuple[float, ...]:
    if not 0.0 <= left_weight <= 1.0:
        raise ObjectivePosteriorError("Mixture weight must be in [0, 1].")
    if left_weight == 1.0:
        return left
    if left_weight == 0.0:
        return right
    log_left_weight = math.log(left_weight)
    log_right_weight = math.log1p(-left_weight)
    return _normalize_log_density(
        _logaddexp(
            left[index] + log_left_weight,
            right[index] + log_right_weight,
        )
        for index in range(GRID_SIZE)
    )


def _validated_gaussian_log_density(
    mean: float,
    variance: float,
    *,
    label: str,
) -> tuple[float, ...]:
    """Build a Gaussian only when the fixed grid represents its moments."""

    log_density = _gaussian_log_density(mean, variance)
    _validate_log_density(log_density, f"{label} log density")
    represented_mean = _integrate_function(log_density, THETA_GRID)
    represented_second_moment = _integrate_function(
        log_density, (theta * theta for theta in THETA_GRID)
    )
    represented_variance = max(
        0.0, represented_second_moment - represented_mean * represented_mean
    )
    variance_tolerance = max(
        GAUSSIAN_VARIANCE_ABSOLUTE_TOLERANCE,
        GAUSSIAN_VARIANCE_RELATIVE_TOLERANCE * variance,
    )
    if (
        abs(represented_mean - mean)
        > GAUSSIAN_MEAN_REPRESENTATION_TOLERANCE
        or abs(represented_variance - variance) > variance_tolerance
    ):
        raise ObjectivePosteriorError(
            f"The fixed grid cannot faithfully represent {label}."
        )
    return log_density


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ObjectivePosteriorError(
                f"Encoded posterior repeats JSON key {key!r}."
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ObjectivePosteriorError(
        f"Encoded posterior contains non-finite JSON constant {value}."
    )


@dataclass(frozen=True, slots=True)
class ObjectivePosterior:
    """Full scalar posterior plus one open exchangeable likelihood epoch."""

    prior_mastery: float
    prior_variance: float
    anchor_log_density: tuple[float, ...]
    pending_observations: tuple[LikelihoodObservation, ...] = ()
    committed_evidence_mass: float = 0.0
    acquisition_mass: float = 0.0
    _log_density_cache: tuple[float, ...] = field(
        init=False, repr=False, compare=False
    )
    _metrics_cache: PosteriorMetrics | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        prior_mastery = _require_finite(self.prior_mastery, "Prior mastery")
        prior_variance = _require_finite(self.prior_variance, "Prior variance")
        if not 0.0 < prior_mastery < 1.0:
            raise ObjectivePosteriorError("Prior mastery must be in (0, 1).")
        if not V5_MIN_VARIANCE <= prior_variance <= V5_MAX_VARIANCE:
            raise ObjectivePosteriorError(
                f"Prior variance must be in [{V5_MIN_VARIANCE}, {V5_MAX_VARIANCE}]."
            )
        _validated_gaussian_log_density(
            _logit(prior_mastery),
            prior_variance,
            label="the declared prior Gaussian",
        )
        anchor = _validate_log_density(
            self.anchor_log_density, "Posterior anchor log density"
        )
        if type(self.pending_observations) is not tuple:
            raise ObjectivePosteriorError(
                "Pending observations must be an immutable tuple."
            )
        pending = _canonical_observations(self.pending_observations)
        if pending != self.pending_observations:
            raise ObjectivePosteriorError(
                "Pending observations must use canonical order."
            )
        committed = _require_finite(
            self.committed_evidence_mass, "Committed evidence mass"
        )
        acquisition = _require_finite(self.acquisition_mass, "Acquisition mass")
        if committed < 0.0:
            raise ObjectivePosteriorError(
                "Committed evidence mass cannot be negative."
            )
        if acquisition < 0.0:
            raise ObjectivePosteriorError("Acquisition mass cannot be negative.")
        current = _validate_log_density(
            _apply_observations(anchor, pending),
            "Current posterior log density",
        )
        object.__setattr__(self, "prior_mastery", prior_mastery)
        object.__setattr__(self, "prior_variance", prior_variance)
        object.__setattr__(self, "anchor_log_density", anchor)
        object.__setattr__(self, "committed_evidence_mass", committed)
        object.__setattr__(self, "acquisition_mass", acquisition)
        # These caches contain only deterministic derivatives of the immutable
        # canonical fields.  They are deliberately absent from equality, repr,
        # and the wire codec, so caching cannot change persisted bytes or hashes.
        object.__setattr__(self, "_log_density_cache", current)
        object.__setattr__(self, "_metrics_cache", None)

    @classmethod
    def from_prior(
        cls,
        prior_mastery: float,
        *,
        prior_variance: float = DEFAULT_PRIOR_VARIANCE,
    ) -> ObjectivePosterior:
        prior = _require_finite(prior_mastery, "Prior mastery")
        variance = _require_finite(prior_variance, "Prior variance")
        if not 0.0 < prior < 1.0:
            raise ObjectivePosteriorError("Prior mastery must be in (0, 1).")
        if not V5_MIN_VARIANCE <= variance <= V5_MAX_VARIANCE:
            raise ObjectivePosteriorError(
                f"Prior variance must be in [{V5_MIN_VARIANCE}, {V5_MAX_VARIANCE}]."
            )
        return cls(
            prior_mastery=prior,
            prior_variance=variance,
            anchor_log_density=_validated_gaussian_log_density(
                _logit(prior), variance, label="the prior Gaussian"
            ),
        )

    @classmethod
    def from_gaussian(
        cls,
        *,
        prior_mastery: float,
        mean: float,
        variance: float,
        prior_variance: float = DEFAULT_PRIOR_VARIANCE,
        committed_evidence_mass: float = 0.0,
        acquisition_mass: float = 0.0,
    ) -> ObjectivePosterior:
        mean_value = _require_finite(mean, "Gaussian mean")
        variance_value = _require_finite(variance, "Gaussian variance")
        if abs(mean_value) > V5_MEAN_ABS_BOUND:
            raise ObjectivePosteriorError(
                f"Gaussian mean must be within +/-{V5_MEAN_ABS_BOUND}."
            )
        if not V5_MIN_VARIANCE <= variance_value <= V5_MAX_VARIANCE:
            raise ObjectivePosteriorError(
                f"Gaussian variance must be in [{V5_MIN_VARIANCE}, {V5_MAX_VARIANCE}]."
            )
        return cls(
            prior_mastery=prior_mastery,
            prior_variance=prior_variance,
            anchor_log_density=_validated_gaussian_log_density(
                mean_value, variance_value, label="this Gaussian state"
            ),
            committed_evidence_mass=committed_evidence_mass,
            acquisition_mass=acquisition_mass,
        )

    @property
    def log_density(self) -> tuple[float, ...]:
        return self._log_density_cache

    @property
    def evidence_mass(self) -> float:
        return self.committed_evidence_mass + math.fsum(
            observation.evidence_power
            for observation in self.pending_observations
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.encode()).hexdigest()

    def with_observations(
        self, observations: Iterable[LikelihoodObservation]
    ) -> ObjectivePosterior:
        additions = tuple(observations)
        if not additions:
            return self
        combined = _canonical_observations(
            (*self.pending_observations, *additions)
        )
        return ObjectivePosterior(
            prior_mastery=self.prior_mastery,
            prior_variance=self.prior_variance,
            anchor_log_density=self.anchor_log_density,
            pending_observations=combined,
            committed_evidence_mass=self.committed_evidence_mass,
            acquisition_mass=self.acquisition_mass,
        )

    def with_observation(
        self, observation: LikelihoodObservation
    ) -> ObjectivePosterior:
        return self.with_observations((observation,))

    def _materialized(
        self,
        log_density: tuple[float, ...],
        *,
        acquisition_mass: float | None = None,
    ) -> ObjectivePosterior:
        return ObjectivePosterior(
            prior_mastery=self.prior_mastery,
            prior_variance=self.prior_variance,
            anchor_log_density=_normalize_log_density(log_density),
            pending_observations=(),
            committed_evidence_mass=self.evidence_mass,
            acquisition_mass=(
                self.acquisition_mass
                if acquisition_mass is None
                else acquisition_mass
            ),
        )

    def apply_retention(
        self, *, elapsed_hours: float, stability_hours: float
    ) -> ObjectivePosterior:
        elapsed = _require_finite(elapsed_hours, "Elapsed retention hours")
        stability = _require_finite(stability_hours, "Retention stability hours")
        if elapsed < 0.0:
            raise ObjectivePosteriorError(
                "Elapsed retention hours cannot be negative."
            )
        if stability <= 0.0:
            raise ObjectivePosteriorError(
                "Retention stability hours must be positive."
            )
        if elapsed == 0.0:
            return self

        current = self.log_density
        current_metrics = self.metrics()
        prior_density = _gaussian_log_density(
            _logit(self.prior_mastery), self.prior_variance
        )
        prior_metrics = _metrics_for_density(
            prior_density,
            evidence_mass=self.evidence_mass,
            acquisition_mass=self.acquisition_mass,
        )

        # Population priors are cold-start beliefs, not positive evidence.  A
        # below-prior posterior is therefore frozen, exactly as the v5 safety
        # rule requires, while the epoch is still closed chronologically.
        if current_metrics.mean <= prior_metrics.mean:
            return self._materialized(current)

        survival = 2.0 ** (
            -elapsed / max(MIN_RETENTION_HALF_LIFE_HOURS, stability)
        )
        candidate = _mixture_log_density(current, prior_density, survival)
        candidate_metrics = _metrics_for_density(
            candidate,
            evidence_mass=self.evidence_mass,
            acquisition_mass=self.acquisition_mass,
        )
        if (
            candidate_metrics.mastery_probability
            > current_metrics.mastery_probability + METRIC_TOLERANCE
            or candidate_metrics.expected_competence
            > current_metrics.expected_competence + METRIC_TOLERANCE
        ):
            # A skewed posterior can disagree with its mean about which side is
            # stronger.  Inactivity is never allowed to exploit that disagreement
            # to improve a published competence metric.
            return self._materialized(current)
        return self._materialized(candidate)

    def apply_correct_feedback(self, strength: float) -> ObjectivePosterior:
        """Apply uncertain upward Markov transport without adding evidence mass."""

        strength_value = _require_finite(strength, "Acquisition strength")
        if not 0.0 <= strength_value <= MAX_ACQUISITION_STRENGTH:
            raise ObjectivePosteriorError(
                f"Acquisition strength must be in [0, {MAX_ACQUISITION_STRENGTH}]."
            )
        if strength_value == 0.0:
            return self

        current = self.log_density
        log_masses = tuple(
            current[index] + _LOG_QUADRATURE_WEIGHTS[index]
            for index in range(GRID_SIZE)
        )
        transported = [-math.inf] * GRID_SIZE
        for index, theta in enumerate(THETA_GRID):
            if index == GRID_SIZE - 1:
                move_probability = 0.0
            else:
                rate = (
                    CORRECT_FEEDBACK_THETA_RATE
                    * strength_value
                    * (1.0 - _sigmoid(theta))
                    / GRID_STEP
                )
                move_probability = -math.expm1(-rate)
                move_probability = max(
                    0.0, min(1.0 - 1e-15, move_probability)
                )
            stay_probability = 1.0 - move_probability
            if stay_probability > 0.0:
                transported[index] = _logaddexp(
                    transported[index],
                    log_masses[index] + math.log(stay_probability),
                )
            if move_probability > 0.0:
                transported[index + 1] = _logaddexp(
                    transported[index + 1],
                    log_masses[index] + math.log(move_probability),
                )
        if any(value == -math.inf or not math.isfinite(value) for value in transported):
            raise ObjectivePosteriorError(
                "Correct-feedback transport produced an unrepresentable density."
            )
        log_density = _normalize_log_density(
            transported[index] - _LOG_QUADRATURE_WEIGHTS[index]
            for index in range(GRID_SIZE)
        )
        before = self.metrics()
        after = _metrics_for_density(
            log_density,
            evidence_mass=self.evidence_mass,
            acquisition_mass=self.acquisition_mass + strength_value,
        )
        if (
            after.mean + METRIC_TOLERANCE < before.mean
            or after.mastery_probability + METRIC_TOLERANCE
            < before.mastery_probability
            or after.expected_competence + METRIC_TOLERANCE
            < before.expected_competence
        ):
            raise ObjectivePosteriorError(
                "Correct-feedback transport violated monotone acquisition."
            )
        return self._materialized(
            log_density,
            acquisition_mass=self.acquisition_mass + strength_value,
        )

    def probability_above_theta(self, threshold: float) -> float:
        return _probability_above_theta(
            self.log_density,
            _require_finite(threshold, "Theta threshold"),
        )

    def probability_above_competence(self, threshold: float) -> float:
        threshold_value = _require_finite(
            threshold, "Competence threshold"
        )
        if not 0.0 < threshold_value < 1.0:
            raise ObjectivePosteriorError(
                "Competence threshold must be in (0, 1)."
            )
        return self.probability_above_theta(_logit(threshold_value))

    @property
    def mastery_probability(self) -> float:
        return self.probability_above_competence(DEFAULT_MASTERY_THRESHOLD)

    @property
    def conservative_mastery_probability(self) -> float:
        return self.metrics().conservative_mastery_probability

    @property
    def expected_competence(self) -> float:
        return _integrate_function(
            self.log_density, (_sigmoid(theta) for theta in THETA_GRID)
        )

    def predict_correct(
        self,
        *,
        difficulty: float,
        discrimination: float,
        guess_rate: float,
        slip_rate: float,
        option_count: int = DEFAULT_OPTION_COUNT,
    ) -> float:
        (
            difficulty_value,
            discrimination_value,
            guess_value,
            slip_value,
            option_count_value,
        ) = _validate_item_parameters(
            difficulty=difficulty,
            discrimination=discrimination,
            guess_rate=guess_rate,
            slip_rate=slip_rate,
            option_count=option_count,
        )
        result = _integrate_function(
            self.log_density,
            (
                _response_probability(
                    theta,
                    difficulty=difficulty_value,
                    discrimination=discrimination_value,
                    guess_rate=guess_value,
                    slip_rate=slip_value,
                    option_count=option_count_value,
                )
                for theta in THETA_GRID
            ),
        )
        return max(0.0, min(1.0, result))

    def expected_information(
        self,
        *,
        difficulty: float,
        discrimination: float,
        guess_rate: float,
        slip_rate: float,
        evidence_power: float,
        option_count: int = DEFAULT_OPTION_COUNT,
    ) -> ExpectedInformation:
        """Integrate both hypothetical response posteriors on the full grid.

        Outcome probabilities use the untempered predictive model.  The two
        posterior branches use the caller-declared evidence power, matching the
        generalized likelihood update that would be applied after observation.
        The result is therefore exact for this fixed-grid representation and
        does not use a local Fisher or Gaussian approximation.
        """

        (
            difficulty_value,
            discrimination_value,
            guess_value,
            slip_value,
            option_count_value,
        ) = _validate_item_parameters(
            difficulty=difficulty,
            discrimination=discrimination,
            guess_rate=guess_rate,
            slip_rate=slip_rate,
            option_count=option_count,
        )
        power = _require_finite(evidence_power, "Evidence power")
        if not 0.0 <= power <= 1.0:
            raise ObjectivePosteriorError("Evidence power must be in [0, 1].")
        current = self.log_density
        predicted_correct = _integrate_function(
            current,
            (
                _response_probability(
                    theta,
                    difficulty=difficulty_value,
                    discrimination=discrimination_value,
                    guess_rate=guess_value,
                    slip_rate=slip_value,
                    option_count=option_count_value,
                )
                for theta in THETA_GRID
            ),
        )
        correct_factor = LikelihoodObservation(
            observation_id="expected_information_correct",
            family_id="expected_information_family",
            difficulty=difficulty_value,
            discrimination=discrimination_value,
            guess_rate=guess_value,
            slip_rate=slip_value,
            option_count=option_count_value,
            correct=True,
            evidence_power=power,
        )
        incorrect_factor = LikelihoodObservation(
            observation_id="expected_information_incorrect",
            family_id="expected_information_family",
            difficulty=difficulty_value,
            discrimination=discrimination_value,
            guess_rate=guess_value,
            slip_rate=slip_value,
            option_count=option_count_value,
            correct=False,
            evidence_power=power,
        )
        correct_density = _apply_observations(current, (correct_factor,))
        incorrect_density = _apply_observations(current, (incorrect_factor,))
        correct_metrics = _metrics_for_density(
            correct_density,
            evidence_mass=self.evidence_mass + power,
            acquisition_mass=self.acquisition_mass,
        )
        incorrect_metrics = _metrics_for_density(
            incorrect_density,
            evidence_mass=self.evidence_mass + power,
            acquisition_mass=self.acquisition_mass,
        )
        correct_kl = _kl_divergence(correct_density, current)
        incorrect_kl = _kl_divergence(incorrect_density, current)
        expected_information = (
            predicted_correct * correct_kl
            + (1.0 - predicted_correct) * incorrect_kl
        )
        current_variance = self.metrics().variance
        expected_variance = (
            predicted_correct * correct_metrics.variance
            + (1.0 - predicted_correct) * incorrect_metrics.variance
        )
        variance_reduction = current_variance - expected_variance
        if expected_information < -METRIC_TOLERANCE:
            raise ObjectivePosteriorError(
                "Expected information became materially negative."
            )
        return ExpectedInformation(
            predicted_correct=max(0.0, min(1.0, predicted_correct)),
            expected_information_nats=max(0.0, expected_information),
            expected_variance=max(0.0, expected_variance),
            # Fractional generalized likelihoods can make variance reduction
            # extremely slightly negative through quadrature. Preserve that
            # diagnostic rather than silently claiming information gain.
            variance_reduction=variance_reduction,
            correct_mastery_probability=correct_metrics.mastery_probability,
            incorrect_mastery_probability=(
                incorrect_metrics.mastery_probability
            ),
        )

    def metrics(self) -> PosteriorMetrics:
        cached = self._metrics_cache
        if cached is None:
            cached = _metrics_for_density(
                self.log_density,
                evidence_mass=self.evidence_mass,
                acquisition_mass=self.acquisition_mass,
            )
            # A concurrent duplicate computation would produce the same frozen
            # value; publishing either result preserves immutable semantics.
            object.__setattr__(self, "_metrics_cache", cached)
        return cached

    def encode(self) -> bytes:
        identity = OBJECTIVE_POSTERIOR_V1_IDENTITY
        payload = {
            "schema_version": identity.schema_version,
            "algorithm": identity.algorithm,
            "codec": identity.codec,
            "grid_id": identity.grid_id,
            "prior_mastery": self.prior_mastery,
            "prior_variance": self.prior_variance,
            "anchor_log_density": list(self.anchor_log_density),
            "pending_observations": [
                observation.as_payload()
                for observation in self.pending_observations
            ],
            "committed_evidence_mass": self.committed_evidence_mass,
            "acquisition_mass": self.acquisition_mass,
        }
        try:
            return json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ObjectivePosteriorError(
                "Posterior state cannot be encoded canonically."
            ) from exc

    @classmethod
    def decode(
        cls, encoded: bytes, *, expected_digest: str | None = None
    ) -> ObjectivePosterior:
        if type(encoded) is not bytes:
            raise ObjectivePosteriorError("Encoded posterior must be bytes.")
        if expected_digest is not None:
            if (
                type(expected_digest) is not str
                or len(expected_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected_digest
                )
            ):
                raise ObjectivePosteriorError(
                    "Expected posterior digest must be lowercase SHA-256."
                )
            actual_digest = hashlib.sha256(encoded).hexdigest()
            if actual_digest != expected_digest:
                raise ObjectivePosteriorError("Posterior digest mismatch.")
        try:
            decoded = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_object,
                parse_constant=_reject_json_constant,
            )
        except ObjectivePosteriorError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ObjectivePosteriorError(
                "Encoded posterior is not strict UTF-8 JSON."
            ) from exc
        if type(decoded) is not dict:
            raise ObjectivePosteriorError("Encoded posterior must be a JSON object.")
        required = {
            "schema_version",
            "algorithm",
            "codec",
            "grid_id",
            "prior_mastery",
            "prior_variance",
            "anchor_log_density",
            "pending_observations",
            "committed_evidence_mass",
            "acquisition_mass",
        }
        if set(decoded) != required:
            raise ObjectivePosteriorError(
                "Encoded posterior fields do not match schema version one."
            )
        identity = OBJECTIVE_POSTERIOR_V1_IDENTITY
        if decoded["schema_version"] != identity.schema_version:
            raise ObjectivePosteriorError("Unsupported posterior schema version.")
        if decoded["algorithm"] != identity.algorithm:
            raise ObjectivePosteriorError("Unsupported posterior algorithm.")
        if decoded["codec"] != identity.codec:
            raise ObjectivePosteriorError("Unsupported posterior codec.")
        if decoded["grid_id"] != identity.grid_id:
            raise ObjectivePosteriorError("Unsupported posterior grid.")
        raw_density = decoded["anchor_log_density"]
        if type(raw_density) is not list:
            raise ObjectivePosteriorError(
                "Encoded anchor log density must be an array."
            )
        raw_observations = decoded["pending_observations"]
        if type(raw_observations) is not list:
            raise ObjectivePosteriorError(
                "Encoded pending observations must be an array."
            )
        observations: list[LikelihoodObservation] = []
        observation_fields = {
            "observation_id",
            "family_id",
            "difficulty",
            "discrimination",
            "guess_rate",
            "slip_rate",
            "option_count",
            "correct",
            "evidence_power",
        }
        for index, raw in enumerate(raw_observations):
            if type(raw) is not dict or set(raw) != observation_fields:
                raise ObjectivePosteriorError(
                    f"Encoded observation {index} has invalid fields."
                )
            observations.append(LikelihoodObservation(**raw))
        posterior = cls(
            prior_mastery=decoded["prior_mastery"],
            prior_variance=decoded["prior_variance"],
            anchor_log_density=tuple(raw_density),
            pending_observations=tuple(observations),
            committed_evidence_mass=decoded["committed_evidence_mass"],
            acquisition_mass=decoded["acquisition_mass"],
        )
        if posterior.encode() != encoded:
            raise ObjectivePosteriorError(
                "Encoded posterior is valid but not in canonical form."
            )
        return posterior


def _metrics_for_density(
    log_density: tuple[float, ...],
    *,
    evidence_mass: float,
    acquisition_mass: float,
) -> PosteriorMetrics:
    mean = _integrate_function(log_density, THETA_GRID)
    second_moment = _integrate_function(
        log_density, (theta * theta for theta in THETA_GRID)
    )
    variance = max(0.0, second_moment - mean * mean)
    mastery = _probability_above_theta(
        log_density, _logit(DEFAULT_MASTERY_THRESHOLD)
    )
    coarse_mastery = _probability_above_theta_on_stride(
        log_density,
        _logit(DEFAULT_MASTERY_THRESHOLD),
        stride=2,
    )
    mastery_error_bound = min(
        1.0,
        max(
            MASTERY_PROBABILITY_ERROR_BOUND,
            abs(mastery - coarse_mastery),
        ),
    )
    competence = _integrate_function(
        log_density, (_sigmoid(theta) for theta in THETA_GRID)
    )
    edge = _edge_mass(log_density)
    for label, value in (
        ("mean", mean),
        ("variance", variance),
        ("mastery probability", mastery),
        ("expected competence", competence),
        ("edge mass", edge),
    ):
        if not math.isfinite(value):
            raise ObjectivePosteriorError(f"Posterior {label} is not finite.")
    return PosteriorMetrics(
        mean=mean,
        variance=variance,
        mastery_probability=mastery,
        expected_competence=competence,
        edge_mass=edge,
        evidence_mass=evidence_mass,
        acquisition_mass=acquisition_mass,
        mastery_probability_error_bound=mastery_error_bound,
    )


def posterior_digest(encoded: bytes) -> str:
    if type(encoded) is not bytes:
        raise ObjectivePosteriorError("Encoded posterior must be bytes.")
    return hashlib.sha256(encoded).hexdigest()


def decode_objective_posterior(
    encoded: bytes, *, expected_digest: str | None = None
) -> ObjectivePosterior:
    return ObjectivePosterior.decode(encoded, expected_digest=expected_digest)


__all__ = [
    "CORRECT_FEEDBACK_THETA_RATE",
    "DEFAULT_LAPSE_RATE",
    "DEFAULT_MASTERY_THRESHOLD",
    "DEFAULT_OPTION_COUNT",
    "DEFAULT_PRIOR_VARIANCE",
    "EDGE_BAND_NODES",
    "ExpectedInformation",
    "GAUSSIAN_MEAN_REPRESENTATION_TOLERANCE",
    "GAUSSIAN_VARIANCE_ABSOLUTE_TOLERANCE",
    "GAUSSIAN_VARIANCE_RELATIVE_TOLERANCE",
    "GRID_INTERVALS",
    "GRID_LOWER",
    "GRID_SIZE",
    "GRID_STEP",
    "GRID_UPPER",
    "LikelihoodObservation",
    "MAX_ACQUISITION_STRENGTH",
    "MAX_EDGE_MASS",
    "MAX_OPTION_COUNT",
    "MASTERY_PROBABILITY_ERROR_BOUND",
    "MIN_OPTION_COUNT",
    "MIN_RETENTION_HALF_LIFE_HOURS",
    "OBJECTIVE_POSTERIOR_ALGORITHM",
    "OBJECTIVE_POSTERIOR_CODEC",
    "OBJECTIVE_POSTERIOR_GRID_ID",
    "OBJECTIVE_POSTERIOR_SCHEMA_VERSION",
    "OBJECTIVE_POSTERIOR_V1_IDENTITY",
    "ObjectivePosterior",
    "ObjectivePosteriorIdentity",
    "ObjectivePosteriorError",
    "PosteriorMetrics",
    "THETA_GRID",
    "SUPPORTED_OBJECTIVE_POSTERIOR_IDENTITIES",
    "V5_MAX_VARIANCE",
    "V5_MEAN_ABS_BOUND",
    "V5_MIN_VARIANCE",
    "decode_objective_posterior",
    "posterior_digest",
]
