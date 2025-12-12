# SPDX-License-Identifier: MPL-2.0

"""Deterministic, end-to-end behavioral simulations for the adaptive engine.

The simulator deliberately calls :class:`~tsq.engine.AdaptiveEngine` rather
than duplicating selection or learner-update logic.  A synthetic learner only
decides which presented option to choose.  Consequently, corpus gaps and policy
constraints remain visible as blockers instead of being papered over by a
simulation-only fallback.

The harness is intended for isolated audit databases.  Its clock is explicit so
that a run can be replayed byte-for-byte at the behavioral level even though the
engine's internal entity identifiers are random.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .engine import AdaptiveEngine, MAX_REMEDIATION_DEPTH
from .errors import ConflictError, ExhaustedError, ValidationError
from .models import Option, Presentation, Question, SessionPhase, logit, sigmoid


DEFAULT_SIMULATION_START = datetime(2100, 1, 1, 9, 0, tzinfo=timezone.utc)
_MAIN_PHASES = frozenset(
    {SessionPhase.LEARN, SessionPhase.DIAGNOSE, SessionPhase.REVIEW}
)


def _stable_uniform(material: str) -> float:
    """Return a stable pseudo-random variate without process hash randomness."""

    # Keep the arithmetic separate from the bit shift; fifty-three bits map
    # exactly onto the useful precision of a Python float.
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big") >> 11
    return integer / float(2**53)


def _clamp_probability(value: float) -> float:
    return min(1.0 - 1e-9, max(1e-9, value))


def _validate_probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite probability.")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValidationError(f"{name} must be between 0 and 1.")


@dataclass(slots=True)
class SimulationClock:
    """A small controllable UTC clock used for selection and answer timestamps."""

    current: datetime = DEFAULT_SIMULATION_START

    def __post_init__(self) -> None:
        if self.current.tzinfo is None or self.current.utcoffset() is None:
            raise ValidationError("SimulationClock.current must be timezone-aware.")
        self.current = self.current.astimezone(timezone.utc)

    def advance(self, duration: timedelta | float) -> datetime:
        if isinstance(duration, timedelta):
            delta = duration
        elif (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
        ):
            raise ValidationError("Clock advancement must be a finite duration.")
        else:
            delta = timedelta(seconds=duration)
        if delta.total_seconds() < 0:
            raise ValidationError("A simulation clock cannot move backwards.")
        self.current += delta
        return self.current


@dataclass(frozen=True, slots=True)
class SyntheticAnswer:
    selected_option_id: str
    correct: bool
    ground_truth_probability: float
    confidence: float
    response_ms: int


@dataclass(frozen=True, slots=True)
class SyntheticLearner:
    """Ground-truth response model, separate from the engine's inferred state.

    Abilities and misconception strengths use the intuitive ``[0, 1]`` scale.
    Abilities are combined on a log-odds scale, then passed through the item's
    difficulty, discrimination, guessing, and slipping parameters.  The profile
    slip and guess probabilities model person-level lapses and lucky recovery.

    All draws are keyed by the question and encounter number.  Adding logging or
    changing option display order therefore cannot silently perturb later answers.
    """

    name: str
    concept_abilities: Mapping[str, float] = field(default_factory=dict)
    misconception_strengths: Mapping[str, float] = field(default_factory=dict)
    default_ability: float = 0.50
    slip_probability: float = 0.04
    guess_probability: float = 0.02
    seed: int = 0
    base_response_ms: int = 4_000

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValidationError("A synthetic learner needs a non-empty name.")
        _validate_probability("default_ability", self.default_ability)
        _validate_probability("slip_probability", self.slip_probability)
        _validate_probability("guess_probability", self.guess_probability)
        for concept_id, value in self.concept_abilities.items():
            if not isinstance(concept_id, str) or not concept_id:
                raise ValidationError("Synthetic ability keys must be concept IDs.")
            _validate_probability(f"ability[{concept_id}]", value)
        for misconception_id, value in self.misconception_strengths.items():
            if not isinstance(misconception_id, str) or not misconception_id:
                raise ValidationError(
                    "Synthetic misconception keys must be misconception IDs."
                )
            _validate_probability(f"misconception[{misconception_id}]", value)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValidationError("Synthetic learner seed must be an integer.")
        if (
            isinstance(self.base_response_ms, bool)
            or not isinstance(self.base_response_ms, int)
            or self.base_response_ms <= 0
        ):
            raise ValidationError("base_response_ms must be a positive integer.")

        # Detach the profile from caller-owned mutable dictionaries.  Sorted
        # insertion order also makes diagnostic serialization stable.
        object.__setattr__(
            self,
            "concept_abilities",
            MappingProxyType(dict(sorted(self.concept_abilities.items()))),
        )
        object.__setattr__(
            self,
            "misconception_strengths",
            MappingProxyType(dict(sorted(self.misconception_strengths.items()))),
        )

    def probability_correct(self, question: Question) -> float:
        scored_mappings = tuple(
            mapping
            for mapping in question.concepts
            if mapping.role.carries_scored_evidence
        )
        total_weight = sum(abs(mapping.weight) for mapping in scored_mappings)
        if total_weight <= 0:
            raise ValidationError(f"Question {question.id} has no positive skill weight.")
        latent_ability = sum(
            abs(mapping.weight)
            * logit(
                _clamp_probability(
                    self.concept_abilities.get(mapping.concept_id, self.default_ability)
                )
            )
            for mapping in scored_mappings
        ) / total_weight
        item_success = sigmoid(
            question.discrimination * (latent_ability - question.difficulty)
        )
        item_success = question.guess_rate + (
            1.0 - question.guess_rate - question.slip_rate
        ) * item_success

        relevant_misconceptions = [
            self.misconception_strengths.get(option.misconception_id, 0.0)
            for option in question.options
            if not option.correct and option.misconception_id
        ]
        misconception_pressure = max(relevant_misconceptions, default=0.0)
        item_success *= 1.0 - 0.35 * misconception_pressure

        # A lapse turns an otherwise correct response into an error, while a
        # lucky guess can recover an otherwise incorrect response.
        person_success = (1.0 - self.slip_probability) * item_success
        person_success += self.guess_probability * (1.0 - item_success)
        return min(1.0, max(0.0, person_success))

    def answer(
        self,
        presentation: Presentation,
        *,
        simulation_seed: int,
        trial_index: int,
        encounter: int,
    ) -> SyntheticAnswer:
        question = presentation.question
        probability = self.probability_correct(question)
        key = (
            f"{self.seed}|{simulation_seed}|{trial_index}|{question.id}|"
            f"{encounter}"
        )
        correct = _stable_uniform(f"{key}|outcome") < probability
        if correct:
            selected = question.correct_option
        else:
            selected = self._incorrect_option(question, f"{key}|distractor")

        # Confidence is intentionally not a proxy for observed correctness.  A
        # low-ability learner may be confidently wrong, which exercises the
        # engine's confidence-sensitive evidence path.
        confidence = min(0.99, max(0.01, max(probability, 1.0 - probability)))
        latency_jitter = 0.75 + 0.5 * _stable_uniform(f"{key}|latency")
        difficulty_factor = 1.0 + 0.12 * abs(question.difficulty)
        response_ms = max(
            250,
            int(round(self.base_response_ms * latency_jitter * difficulty_factor)),
        )
        return SyntheticAnswer(
            selected_option_id=selected.id,
            correct=correct,
            ground_truth_probability=probability,
            confidence=confidence,
            response_ms=response_ms,
        )

    def _incorrect_option(self, question: Question, key: str) -> Option:
        distractors = sorted(
            (option for option in question.options if not option.correct),
            key=lambda option: option.id,
        )
        if not distractors:
            raise ValidationError(f"Question {question.id} has no distractor option.")
        weights = [
            1.0
            + 6.0
            * self.misconception_strengths.get(option.misconception_id or "", 0.0)
            for option in distractors
        ]
        threshold = _stable_uniform(key) * sum(weights)
        cumulative = 0.0
        for option, weight in zip(distractors, weights, strict=True):
            cumulative += weight
            if threshold < cumulative:
                return option
        return distractors[-1]
