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
from .inference import ResponseClass, classify_response_for_model
from .models import Option, Presentation, Question, SessionPhase, logit, sigmoid


DEFAULT_SIMULATION_START = datetime(2100, 1, 1, 9, 0, tzinfo=timezone.utc)
SIMULATION_FEEDBACK_PROTOCOL_VERSION = (
    "response-then-observational-feedback-v1"
)
_MAIN_PHASES = frozenset(
    {SessionPhase.LEARN, SessionPhase.DIAGNOSE, SessionPhase.REVIEW}
)
SYNTHETIC_RESPONSE_MODELS = frozenset(
    {
        "four_parameter_logistic",
        "discontinuous_threshold",
        "ability_only",
    }
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


def evidence_anchor_concept_id(question: Question) -> str:
    """Return the concept whose latent actually receives this response.

    Objective-aware questions may deliberately use a broad or contrasting
    surface primary concept.  Production scores those questions against the
    release-pinned objective's canonical owner, while legacy questions retain
    their authored primary concept.
    """

    if question.objective is not None:
        return question.objective.primary_concept_id
    return question.primary_concept_id


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
    selected_option_id: str | None
    correct: bool
    ground_truth_probability: float
    confidence: float | None
    response_ms: int
    hint_count: int


@dataclass(frozen=True, slots=True)
class SyntheticLearner:
    """Ground-truth response model, separate from the engine's inferred state.

    Abilities and misconception strengths use the intuitive ``[0, 1]`` scale.
    Objective-aware items use one objective ability, falling back to that
    objective's canonical owner concept; legacy items retain the weighted
    concept behavior.  The default response model then applies item difficulty,
    discrimination, guessing, and slipping.  Explicit misspecified generators
    are available so evaluation is not circularly limited to the engine's own
    smooth item-response assumptions.  Profile slip and guess probabilities
    model person-level lapses and lucky recovery.

    All draws are keyed by the question and encounter number.  Adding logging or
    changing option display order therefore cannot silently perturb later answers.
    """

    name: str
    concept_abilities: Mapping[str, float] = field(default_factory=dict)
    objective_abilities: Mapping[str, float] = field(default_factory=dict)
    misconception_strengths: Mapping[str, float] = field(default_factory=dict)
    default_ability: float = 0.50
    default_objective_ability: float | None = None
    slip_probability: float = 0.04
    guess_probability: float = 0.02
    response_model: str = "four_parameter_logistic"
    seed: int = 0
    base_response_ms: int = 4_000
    abstain_probability: float = 0.0
    confidence_override: float | None = None
    forced_correctness: bool | None = None
    hint_count: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValidationError("A synthetic learner needs a non-empty name.")
        _validate_probability("default_ability", self.default_ability)
        if self.default_objective_ability is not None:
            _validate_probability(
                "default_objective_ability", self.default_objective_ability
            )
        _validate_probability("slip_probability", self.slip_probability)
        _validate_probability("guess_probability", self.guess_probability)
        _validate_probability("abstain_probability", self.abstain_probability)
        for concept_id, value in self.concept_abilities.items():
            if not isinstance(concept_id, str) or not concept_id:
                raise ValidationError("Synthetic ability keys must be concept IDs.")
            _validate_probability(f"ability[{concept_id}]", value)
        for objective_id, value in self.objective_abilities.items():
            if not isinstance(objective_id, str) or not objective_id:
                raise ValidationError(
                    "Synthetic objective ability keys must be objective IDs."
                )
            _validate_probability(f"objective_ability[{objective_id}]", value)
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
        if self.confidence_override is not None:
            _validate_probability("confidence_override", self.confidence_override)
        if self.forced_correctness is not None and type(self.forced_correctness) is not bool:
            raise ValidationError("forced_correctness must be true, false, or null.")
        if type(self.hint_count) is not int or self.hint_count < 0:
            raise ValidationError("hint_count must be a non-negative integer.")
        if self.response_model not in SYNTHETIC_RESPONSE_MODELS:
            supported = ", ".join(sorted(SYNTHETIC_RESPONSE_MODELS))
            raise ValidationError(
                f"Unknown synthetic response model {self.response_model!r}; "
                f"expected one of: {supported}."
            )

        # Detach the profile from caller-owned mutable dictionaries.  Sorted
        # insertion order also makes diagnostic serialization stable.
        object.__setattr__(
            self,
            "concept_abilities",
            MappingProxyType(dict(sorted(self.concept_abilities.items()))),
        )
        object.__setattr__(
            self,
            "objective_abilities",
            MappingProxyType(dict(sorted(self.objective_abilities.items()))),
        )
        object.__setattr__(
            self,
            "misconception_strengths",
            MappingProxyType(dict(sorted(self.misconception_strengths.items()))),
        )

    def _ability_for_question(self, question: Question) -> float:
        if question.objective is not None:
            objective_id = question.objective.id
            if objective_id in self.objective_abilities:
                return self.objective_abilities[objective_id]
            if self.default_objective_ability is not None:
                return self.default_objective_ability
            return self.concept_abilities.get(
                question.objective.primary_concept_id, self.default_ability
            )

        scored_mappings = tuple(
            mapping
            for mapping in question.concepts
            if mapping.role.carries_scored_evidence
        )
        total_weight = sum(abs(mapping.weight) for mapping in scored_mappings)
        if total_weight <= 0:
            raise ValidationError(f"Question {question.id} has no positive skill weight.")
        latent_log_odds = sum(
            abs(mapping.weight)
            * logit(
                _clamp_probability(
                    self.concept_abilities.get(mapping.concept_id, self.default_ability)
                )
            )
            for mapping in scored_mappings
        ) / total_weight
        return sigmoid(latent_log_odds)

    def probability_correct(self, question: Question) -> float:
        ability = _clamp_probability(self._ability_for_question(question))
        latent_ability = logit(ability)
        if self.response_model == "four_parameter_logistic":
            item_success = sigmoid(
                question.discrimination * (latent_ability - question.difficulty)
            )
            item_success = question.guess_rate + (
                1.0 - question.guess_rate - question.slip_rate
            ) * item_success
        elif self.response_model == "discontinuous_threshold":
            # Deliberately violates the engine's smooth item-response
            # assumption.  It is useful for detecting an evaluator that merely
            # confirms the same model it uses to generate answers.
            item_success = 0.92 if latent_ability >= question.difficulty else 0.08
        else:
            # A difficulty-blind learner is another explicit misspecification:
            # performance follows the latent ability but not authored item b/a.
            item_success = ability

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
        abstained = _stable_uniform(f"{key}|abstain") < self.abstain_probability
        sampled_correct = _stable_uniform(f"{key}|outcome") < probability
        correct = (
            self.forced_correctness
            if self.forced_correctness is not None
            else sampled_correct
        )
        correct = bool(correct and not abstained)
        if abstained:
            selected = None
        elif correct:
            selected = question.correct_option
        else:
            selected = self._incorrect_option(question, f"{key}|distractor")

        # Confidence is intentionally not a proxy for observed correctness.  A
        # low-ability learner may be confidently wrong, which exercises the
        # engine's confidence-sensitive evidence path.
        confidence = (
            self.confidence_override
            if self.confidence_override is not None
            else min(0.99, max(0.01, max(probability, 1.0 - probability)))
        )
        latency_jitter = 0.75 + 0.5 * _stable_uniform(f"{key}|latency")
        difficulty_factor = 1.0 + 0.12 * abs(question.difficulty)
        response_ms = max(
            1,
            int(round(self.base_response_ms * latency_jitter * difficulty_factor)),
        )
        return SyntheticAnswer(
            selected_option_id=selected.id if selected is not None else None,
            correct=correct,
            ground_truth_probability=probability,
            confidence=confidence,
            response_ms=response_ms,
            hint_count=self.hint_count,
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


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_accuracy: float


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    count: int
    brier_score: float | None
    log_loss: float | None
    expected_calibration_error: float | None
    bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True, slots=True)
class SimulationStep:
    index: int
    phase_before: SessionPhase
    phase_after: SessionPhase
    question_id: str
    family_id: str
    surface_primary_concept_id: str
    evidence_anchor_concept_id: str
    # Backwards-compatible alias for the authored surface primary.
    primary_concept_id: str
    learning_objective_id: str | None
    question_kind: str
    pedagogical_role: str
    topic_ids: tuple[str, ...]
    continuity: float
    predicted_correct: float
    ground_truth_probability: float
    actual_correct: bool
    selected_option_id: str | None
    focus_concept_before: str | None
    focus_concept_after: str | None
    focus_objective_before: str | None
    focus_objective_after: str | None
    focus_misconception_before: str | None
    focus_misconception_after: str | None
    exact_repeat: bool
    family_repeat: bool
    response_ms: int
    confidence: float | None
    hint_count: int
    selected_at: datetime
    answered_at: datetime


@dataclass(frozen=True, slots=True)
class SimulationGap:
    step_index: int
    phase: SessionPhase
    focus_concept_id: str | None
    focus_objective_id: str | None
    focus_misconception_id: str | None
    category: str
    message: str


@dataclass(frozen=True, slots=True)
class FocusEpisode:
    start_step: int
    end_step: int
    trigger_question_id: str
    trigger_family_id: str
    initial_focus_concept_id: str | None
    initial_focus_objective_id: str | None
    initial_focus_misconception_id: str | None
    focus_path: tuple[str, ...]
    objective_focus_path: tuple[str, ...]
    question_ids: tuple[str, ...]
    family_ids: tuple[str, ...]
    outcome: str
    exact_repeat_count: int
    family_repeat_count: int

    @property
    def length(self) -> int:
        """Number of remediation/verification items, excluding the trigger."""

        return len(self.question_ids)


@dataclass(frozen=True, slots=True)
class CoverageMetrics:
    scope_concepts: int
    scope_objectives: int
    eligible_concepts: int
    eligible_objectives: int
    eligible_questions: int
    eligible_families: int
    eligible_evidence_families: int
    eligible_objective_families: int
    observed_concepts: tuple[str, ...]
    observed_surface_concepts: tuple[str, ...]
    observed_objectives: tuple[str, ...]
    observed_questions: tuple[str, ...]
    observed_families: tuple[str, ...]
    observed_evidence_families: tuple[str, ...]
    observed_objective_families: tuple[str, ...]
    observed_outside_scope_concepts: tuple[str, ...]
    observed_outside_scope_objectives: tuple[str, ...]
    observed_outside_scope_questions: tuple[str, ...]
    observed_outside_scope_families: tuple[str, ...]
    observed_outside_scope_evidence_families: tuple[str, ...]
    observed_outside_scope_objective_families: tuple[str, ...]

    @property
    def concept_fraction(self) -> float:
        return len(self.observed_concepts) / max(1, self.eligible_concepts)

    @property
    def question_fraction(self) -> float:
        return len(self.observed_questions) / max(1, self.eligible_questions)

    @property
    def family_fraction(self) -> float:
        return len(self.observed_families) / max(1, self.eligible_families)

    @property
    def evidence_family_fraction(self) -> float:
        return len(self.observed_evidence_families) / max(
            1, self.eligible_evidence_families
        )

    @property
    def objective_fraction(self) -> float:
        return len(self.observed_objectives) / max(1, self.eligible_objectives)

    @property
    def objective_family_fraction(self) -> float:
        return len(self.observed_objective_families) / max(
            1, self.eligible_objective_families
        )


@dataclass(frozen=True, slots=True)
class SimulationReport:
    profile_name: str
    generator_model: str
    learner_id: str
    root_concept_id: str
    mode: str
    policy_seed: int
    trial_index: int
    started_at: datetime
    ended_at: datetime
    steps: tuple[SimulationStep, ...]
    gaps: tuple[SimulationGap, ...]
    focus_episodes: tuple[FocusEpisode, ...]
    coverage: CoverageMetrics
    calibration: CalibrationMetrics
    phase_counts: Mapping[str, int]
    phase_transitions: Mapping[str, int]
    exact_repeat_count: int
    family_repeat_count: int
    remediation_exact_repeat_count: int
    remediation_family_repeat_count: int
    idempotent_retries_verified: int = 0

    @property
    def attempted(self) -> int:
        return len(self.steps)

    @property
    def correct(self) -> int:
        return sum(step.actual_correct for step in self.steps)

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.attempted)

    @property
    def has_blockers(self) -> bool:
        return bool(self.gaps)

    def behavior_signature(self) -> str:
        """Canonical replay signature excluding random database entity IDs."""

        payload = {
            "profile": self.profile_name,
            "generator_model": self.generator_model,
            "root": self.root_concept_id,
            "mode": self.mode,
            "policy_seed": self.policy_seed,
            "trial_index": self.trial_index,
            "steps": [
                {
                    "index": step.index,
                    "phase_before": step.phase_before.value,
                    "phase_after": step.phase_after.value,
                    "question_id": step.question_id,
                    "family_id": step.family_id,
                    "surface_primary_concept_id": (
                        step.surface_primary_concept_id
                    ),
                    "evidence_anchor_concept_id": (
                        step.evidence_anchor_concept_id
                    ),
                    "learning_objective_id": step.learning_objective_id,
                    "pedagogical_role": step.pedagogical_role,
                    "topic_ids": step.topic_ids,
                    "continuity": round(step.continuity, 12),
                    "predicted_correct": round(step.predicted_correct, 12),
                    "ground_truth_probability": round(
                        step.ground_truth_probability, 12
                    ),
                    "actual_correct": step.actual_correct,
                    "selected_option_id": step.selected_option_id,
                    "confidence": (
                        round(step.confidence, 12)
                        if step.confidence is not None
                        else None
                    ),
                    "hint_count": step.hint_count,
                    "focus_before": [
                        step.focus_concept_before,
                        step.focus_objective_before,
                        step.focus_misconception_before,
                    ],
                    "focus_after": [
                        step.focus_concept_after,
                        step.focus_objective_after,
                        step.focus_misconception_after,
                    ],
                    "selected_at": step.selected_at.isoformat(),
                    "answered_at": step.answered_at.isoformat(),
                }
                for step in self.steps
            ],
            "gaps": [
                {
                    "step": gap.step_index,
                    "phase": gap.phase.value,
                    "focus": [
                        gap.focus_concept_id,
                        gap.focus_objective_id,
                        gap.focus_misconception_id,
                    ],
                    "category": gap.category,
                    "message": gap.message,
                }
                for gap in self.gaps
            ],
            "episodes": [
                {
                    "start": episode.start_step,
                    "end": episode.end_step,
                    "trigger": [
                        episode.trigger_question_id,
                        episode.trigger_family_id,
                    ],
                    "focus_path": episode.focus_path,
                    "objective_focus_path": episode.objective_focus_path,
                    "questions": episode.question_ids,
                    "families": episode.family_ids,
                    "outcome": episode.outcome,
                    "exact_repeats": episode.exact_repeat_count,
                    "family_repeats": episode.family_repeat_count,
                }
                for episode in self.focus_episodes
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def summary(self) -> dict[str, Any]:
        """A compact JSON-safe audit summary suitable for CI artifacts."""

        return {
            "profile": self.profile_name,
            "generator_model": self.generator_model,
            "feedback_protocol": SIMULATION_FEEDBACK_PROTOCOL_VERSION,
            "learner_id": self.learner_id,
            "root_concept_id": self.root_concept_id,
            "mode": self.mode,
            "policy_seed": self.policy_seed,
            "attempted": self.attempted,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "coverage": {
                "scope_concepts": self.coverage.scope_concepts,
                "scope_objectives": self.coverage.scope_objectives,
                "eligible_concepts": self.coverage.eligible_concepts,
                "eligible_objectives": self.coverage.eligible_objectives,
                "eligible_questions": self.coverage.eligible_questions,
                "eligible_families": self.coverage.eligible_families,
                "eligible_evidence_families": (
                    self.coverage.eligible_evidence_families
                ),
                "eligible_objective_families": (
                    self.coverage.eligible_objective_families
                ),
                "observed_concepts": len(self.coverage.observed_concepts),
                "observed_surface_concepts": len(
                    self.coverage.observed_surface_concepts
                ),
                "observed_objectives": len(self.coverage.observed_objectives),
                "observed_questions": len(self.coverage.observed_questions),
                "observed_families": len(self.coverage.observed_families),
                "observed_evidence_families": len(
                    self.coverage.observed_evidence_families
                ),
                "observed_objective_families": len(
                    self.coverage.observed_objective_families
                ),
                "observed_outside_scope_concepts": len(
                    self.coverage.observed_outside_scope_concepts
                ),
                "observed_outside_scope_objectives": len(
                    self.coverage.observed_outside_scope_objectives
                ),
                "observed_outside_scope_questions": len(
                    self.coverage.observed_outside_scope_questions
                ),
                "observed_outside_scope_families": len(
                    self.coverage.observed_outside_scope_families
                ),
                "observed_outside_scope_evidence_families": len(
                    self.coverage.observed_outside_scope_evidence_families
                ),
                "observed_outside_scope_objective_families": len(
                    self.coverage.observed_outside_scope_objective_families
                ),
                "concept_fraction": self.coverage.concept_fraction,
                "question_fraction": self.coverage.question_fraction,
                "family_fraction": self.coverage.family_fraction,
                "evidence_family_fraction": (
                    self.coverage.evidence_family_fraction
                ),
                "objective_fraction": self.coverage.objective_fraction,
                "objective_family_fraction": (
                    self.coverage.objective_family_fraction
                ),
                "denominator_contract": (
                    "Questions are in scope by the objective's canonical owner "
                    "when objective-aware, otherwise by the authored primary. "
                    "Evidence-family counts use objective/family or legacy "
                    "concept/family pairs."
                ),
            },
            "focus_episodes": len(self.focus_episodes),
            "focus_outcomes": dict(
                Counter(episode.outcome for episode in self.focus_episodes)
            ),
            "exact_repeats": self.exact_repeat_count,
            "family_repeats": self.family_repeat_count,
            "remediation_exact_repeats": self.remediation_exact_repeat_count,
            "remediation_family_repeats": self.remediation_family_repeat_count,
            "answer_patterns": {
                "correct": self.correct,
                "incorrect": self.attempted - self.correct,
                "abstained": sum(
                    step.selected_option_id is None for step in self.steps
                ),
                "low_confidence": sum(
                    step.confidence is not None and step.confidence < 0.50
                    for step in self.steps
                ),
                "missing_confidence": sum(
                    step.confidence is None for step in self.steps
                ),
                "fast_under_250ms": sum(
                    step.response_ms < 250 for step in self.steps
                ),
                "slow_at_least_500ms": sum(
                    step.response_ms >= 500 for step in self.steps
                ),
                "hinted": sum(step.hint_count > 0 for step in self.steps),
            },
            "selection_patterns": {
                "roles": dict(Counter(step.pedagogical_role for step in self.steps)),
                "cross_topic_questions": sum(
                    len(step.topic_ids) > 1 for step in self.steps
                ),
                "average_continuity": (
                    sum(step.continuity for step in self.steps) / self.attempted
                    if self.steps
                    else None
                ),
            },
            "phase_counts": dict(self.phase_counts),
            "phase_transitions": dict(self.phase_transitions),
            "idempotent_retries_verified": self.idempotent_retries_verified,
            "calibration": {
                "count": self.calibration.count,
                "brier_score": self.calibration.brier_score,
                "log_loss": self.calibration.log_loss,
                "expected_calibration_error": self.calibration.expected_calibration_error,
                "interpretation": (
                    "Predictive fit against a declared synthetic generator; "
                    "this is not empirical human calibration."
                ),
            },
            "blockers": [
                {
                    "step": gap.step_index,
                    "phase": gap.phase.value,
                    "focus_concept_id": gap.focus_concept_id,
                    "focus_objective_id": gap.focus_objective_id,
                    "focus_misconception_id": gap.focus_misconception_id,
                    "category": gap.category,
                    "message": gap.message,
                }
                for gap in self.gaps
            ],
            "behavior_signature": self.behavior_signature(),
        }


@dataclass(frozen=True, slots=True)
class CohortReport:
    profile_name: str
    trials: tuple[SimulationReport, ...]
    calibration: CalibrationMetrics

    @property
    def attempted(self) -> int:
        return sum(report.attempted for report in self.trials)

    @property
    def correct(self) -> int:
        return sum(report.correct for report in self.trials)

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.attempted)

    @property
    def blocker_count(self) -> int:
        return sum(len(report.gaps) for report in self.trials)

    def summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "trials": len(self.trials),
            "attempted": self.attempted,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "blockers": self.blocker_count,
            "brier_score": self.calibration.brier_score,
            "log_loss": self.calibration.log_loss,
            "expected_calibration_error": self.calibration.expected_calibration_error,
        }


@dataclass(slots=True)
class _EpisodeBuilder:
    start_step: int
    trigger_question_id: str
    trigger_family_id: str
    initial_focus_concept_id: str | None
    initial_focus_objective_id: str | None
    initial_focus_misconception_id: str | None
    focus_path: list[str] = field(default_factory=list)
    objective_focus_path: list[str] = field(default_factory=list)
    question_ids: list[str] = field(default_factory=list)
    family_ids: list[str] = field(default_factory=list)
    exact_repeat_count: int = 0
    family_repeat_count: int = 0

    def observe(
        self,
        *,
        question_id: str,
        family_id: str,
        focus_concept_id: str | None,
        focus_objective_id: str | None,
    ) -> None:
        if focus_concept_id and (
            not self.focus_path or self.focus_path[-1] != focus_concept_id
        ):
            self.focus_path.append(focus_concept_id)
        if focus_objective_id and (
            not self.objective_focus_path
            or self.objective_focus_path[-1] != focus_objective_id
        ):
            self.objective_focus_path.append(focus_objective_id)
        if question_id == self.trigger_question_id or question_id in self.question_ids:
            self.exact_repeat_count += 1
        if family_id == self.trigger_family_id or family_id in self.family_ids:
            self.family_repeat_count += 1
        self.question_ids.append(question_id)
        self.family_ids.append(family_id)

    def finish(self, *, end_step: int, outcome: str) -> FocusEpisode:
        return FocusEpisode(
            start_step=self.start_step,
            end_step=end_step,
            trigger_question_id=self.trigger_question_id,
            trigger_family_id=self.trigger_family_id,
            initial_focus_concept_id=self.initial_focus_concept_id,
            initial_focus_objective_id=self.initial_focus_objective_id,
            initial_focus_misconception_id=self.initial_focus_misconception_id,
            focus_path=tuple(self.focus_path),
            objective_focus_path=tuple(self.objective_focus_path),
            question_ids=tuple(self.question_ids),
            family_ids=tuple(self.family_ids),
            outcome=outcome,
            exact_repeat_count=self.exact_repeat_count,
            family_repeat_count=self.family_repeat_count,
        )


@dataclass(frozen=True, slots=True)
class _CoverageDenominator:
    scope_concepts: int
    scope_objectives: int
    eligible_concepts: int
    eligible_objectives: int
    eligible_questions: int
    eligible_families: int
    eligible_evidence_families: int
    eligible_objective_families: int
    eligible_concept_ids: frozenset[str]
    eligible_objective_ids: frozenset[str]
    eligible_question_ids: frozenset[str]
    eligible_family_ids: frozenset[str]
    eligible_evidence_family_ids: frozenset[str]
    eligible_objective_family_ids: frozenset[str]


class BehavioralSimulator:
    """Drive real adaptive sessions and retain behaviorally relevant traces."""

    def __init__(self, engine: AdaptiveEngine):
        self.engine = engine

    def _immutable_response_class(self, interaction_id: str) -> ResponseClass:
        """Classify one durable answer under its event-declared model contract."""

        with self.engine.database.read() as connection:
            row = connection.execute(
                """SELECT attempt.is_correct, attempt.selected_option_id,
                          attempt.confidence, attempt.response_ms,
                          attempt.hint_count, event.event_type,
                          event.metadata_json,
                          selected_option.misconception_id
                              AS selected_misconception_id
                   FROM attempts attempt
                   JOIN events event ON event.event_id = attempt.event_id
                   LEFT JOIN options selected_option
                     ON selected_option.question_id = attempt.question_id
                    AND selected_option.option_id =
                        attempt.selected_option_id
                   WHERE attempt.id = ?""",
                (interaction_id,),
            ).fetchone()
        if row is None or row["event_type"] != "ResponseSubmitted":
            raise ConflictError(
                "A simulated answer is missing its immutable response event."
            )
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, ValueError) as exc:
            raise ConflictError(
                "A simulated answer has invalid immutable response metadata."
            ) from exc
        if type(metadata) is not dict:
            raise ConflictError(
                "A simulated answer has invalid immutable response metadata."
            )
        return classify_response_for_model(
            model_version=metadata.get("learner_model_version"),
            correct=bool(row["is_correct"]),
            selected_option_id=row["selected_option_id"],
            selected_misconception_id=row[
                "selected_misconception_id"
            ],
            confidence=(
                float(row["confidence"])
                if row["confidence"] is not None
                else None
            ),
            response_ms=(
                int(row["response_ms"])
                if row["response_ms"] is not None
                else None
            ),
            hint_count=int(row["hint_count"]),
        )

    def run(
        self,
        profile: SyntheticLearner,
        *,
        learner_id: str,
        root_concept_id: str,
        policy_seed: int,
        max_steps: int = 40,
        mode: str = "learn",
        start_at: datetime = DEFAULT_SIMULATION_START,
        inter_item_delay: timedelta = timedelta(minutes=5),
        trial_index: int = 0,
        require_fresh_learner: bool = True,
        verify_idempotency: bool = False,
    ) -> SimulationReport:
        if not learner_id:
            raise ValidationError("learner_id cannot be empty.")
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
            raise ValidationError("policy_seed must be an integer.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValidationError("max_steps must be a positive integer.")
        if inter_item_delay.total_seconds() < 0:
            raise ValidationError("inter_item_delay cannot be negative.")
        if type(verify_idempotency) is not bool:
            raise ValidationError("verify_idempotency must be true or false.")

        self.engine.create_learner(learner_id, f"simulation:{profile.name}")
        if require_fresh_learner:
            with self.engine.database.read() as connection:
                attempts = connection.execute(
                    "SELECT COUNT(*) AS n FROM attempts WHERE learner_id = ?",
                    (learner_id,),
                ).fetchone()["n"]
            if attempts:
                raise ConflictError(
                    f"Synthetic learner {learner_id} already has evidence; use a fresh ID "
                    "or explicitly set require_fresh_learner=False."
                )

        clock = SimulationClock(start_at)
        started_at = clock.current
        session = self.engine.start_session(
            learner_id,
            root_concept_id,
            mode=mode,
            seed=policy_seed,
            now=clock.current,
        )
        coverage_denominator = self._coverage_denominator(session)

        steps: list[SimulationStep] = []
        gaps: list[SimulationGap] = []
        episodes: list[FocusEpisode] = []
        active_episode: _EpisodeBuilder | None = None
        encounters: Counter[str] = Counter()
        seen_questions: set[str] = set()
        seen_families: set[str] = set()
        phase_counts: Counter[str] = Counter()
        transitions: Counter[str] = Counter()
        idempotent_retries_verified = 0

        for step_index in range(max_steps):
            current_session = self.engine.database.get_session(session["id"])
            try:
                presentation = self.engine.next_question(
                    session["id"], now=clock.current
                )
            except ExhaustedError as error:
                phase = SessionPhase(current_session["phase"])
                message = str(error)
                gaps.append(
                    SimulationGap(
                        step_index=step_index,
                        phase=phase,
                        focus_concept_id=current_session["focus_concept_id"],
                        focus_objective_id=current_session[
                            "focus_objective_id"
                        ],
                        focus_misconception_id=current_session[
                            "focus_misconception_id"
                        ],
                        category=(
                            "corpus_gap"
                            if message.lower().startswith("corpus gap:")
                            else "exhausted"
                        ),
                        message=message,
                    )
                )
                if active_episode is not None:
                    episodes.append(
                        active_episode.finish(end_step=step_index - 1, outcome="gap")
                    )
                    active_episode = None
                break

            question = presentation.question
            encounters[question.id] += 1
            answer = profile.answer(
                presentation,
                simulation_seed=policy_seed,
                trial_index=trial_index,
                encounter=encounters[question.id],
            )
            phase_before = presentation.phase
            focus_concept_before = current_session["focus_concept_id"]
            focus_objective_before = current_session[
                "focus_objective_id"
            ]
            focus_misconception_before = current_session["focus_misconception_id"]
            selected_at = clock.current
            clock.advance(timedelta(milliseconds=answer.response_ms))

            idempotency_key = (
                f"simulation-answer:{learner_id}:{policy_seed}:"
                f"{trial_index}:{step_index}"
            )
            result = self.engine.submit_answer(
                presentation.decision_id,
                answer.selected_option_id,
                confidence=answer.confidence,
                response_ms=answer.response_ms,
                hint_count=answer.hint_count,
                # Match the production CLI boundary: the response is committed
                # before feedback reaches the output boundary.  Observational
                # feedback telemetry is appended separately below and must not
                # silently become acquisition evidence.
                feedback_shown=False,
                idempotency_key=idempotency_key,
                now=clock.current,
            )
            if verify_idempotency:
                with self.engine.database.read() as connection:
                    before_retry = {
                        "events": connection.execute(
                            "SELECT COUNT(*) AS n FROM events WHERE learner_id = ?",
                            (learner_id,),
                        ).fetchone()["n"],
                        "attempts": connection.execute(
                            "SELECT COUNT(*) AS n FROM attempts WHERE learner_id = ?",
                            (learner_id,),
                        ).fetchone()["n"],
                        "revision": connection.execute(
                            "SELECT revision FROM learners WHERE id = ?",
                            (learner_id,),
                        ).fetchone()["revision"],
                        "projection_hash": (
                            self.engine.database.learner_projection_hash(
                                learner_id, connection
                            )
                        ),
                    }
                retried = self.engine.submit_answer(
                    presentation.decision_id,
                    answer.selected_option_id,
                    confidence=answer.confidence,
                    response_ms=answer.response_ms,
                    hint_count=answer.hint_count,
                    feedback_shown=False,
                    idempotency_key=idempotency_key,
                    now=clock.current,
                )
                with self.engine.database.read() as connection:
                    after_retry = {
                        "events": connection.execute(
                            "SELECT COUNT(*) AS n FROM events WHERE learner_id = ?",
                            (learner_id,),
                        ).fetchone()["n"],
                        "attempts": connection.execute(
                            "SELECT COUNT(*) AS n FROM attempts WHERE learner_id = ?",
                            (learner_id,),
                        ).fetchone()["n"],
                        "revision": connection.execute(
                            "SELECT revision FROM learners WHERE id = ?",
                            (learner_id,),
                        ).fetchone()["revision"],
                        "projection_hash": (
                            self.engine.database.learner_projection_hash(
                                learner_id, connection
                            )
                        ),
                    }
                if (
                    not retried.idempotent_replay
                    or retried.interaction_id != result.interaction_id
                    or after_retry != before_retry
                ):
                    raise ConflictError(
                        "An idempotent simulation retry changed durable learner state."
                    )
                idempotent_retries_verified += 1
            feedback_material = json.dumps(
                {
                    "decision_id": presentation.decision_id,
                    "selected_option_id": (
                        result.selected_option.id
                        if result.selected_option is not None
                        else None
                    ),
                    "correct_option_id": result.correct_option.id,
                    "selected_rationale": (
                        result.selected_option.rationale
                        if result.selected_option is not None
                        else None
                    ),
                    "correct_rationale": result.correct_option.rationale,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            feedback_digest = hashlib.sha256(feedback_material).hexdigest()
            feedback_key_digest = hashlib.sha256(
                (
                    f"{presentation.decision_id}:{feedback_digest}"
                ).encode("utf-8")
            ).hexdigest()
            feedback_action = self.engine.record_action(
                presentation.decision_id,
                "feedback_shown",
                {"feedback_digest": feedback_digest},
                stage="post_feedback",
                idempotency_key=(
                    f"simulation-feedback:{feedback_key_digest}"
                ),
                now=clock.current,
            )
            if verify_idempotency:
                feedback_retry = self.engine.record_action(
                    presentation.decision_id,
                    "feedback_shown",
                    {"feedback_digest": feedback_digest},
                    stage="post_feedback",
                    idempotency_key=(
                        f"simulation-feedback:{feedback_key_digest}"
                    ),
                    now=clock.current,
                )
                if (
                    not feedback_retry["idempotent_replay"]
                    or feedback_retry["id"] != feedback_action["id"]
                ):
                    raise ConflictError(
                        "An idempotent feedback retry changed durable action state."
                    )
            if result.correct != answer.correct:
                raise ConflictError(
                    "Synthetic outcome disagreed with the corpus answer key for "
                    f"{question.id}; the corpus or simulator is inconsistent."
                )
            response_class = self._immutable_response_class(
                result.interaction_id
            )
            answered_at = clock.current

            exact_repeat = question.id in seen_questions
            family_repeat = question.family_id in seen_families
            phase_counts[phase_before.value] += 1
            transitions[f"{phase_before.value}->{result.next_phase.value}"] += 1

            if phase_before in {SessionPhase.REMEDIATE, SessionPhase.VERIFY}:
                # This can occur if a simulation starts from an already focused
                # learner with require_fresh_learner=False.
                if active_episode is None:
                    active_episode = _EpisodeBuilder(
                        start_step=step_index,
                        trigger_question_id="",
                        trigger_family_id="",
                        initial_focus_concept_id=focus_concept_before,
                        initial_focus_objective_id=focus_objective_before,
                        initial_focus_misconception_id=focus_misconception_before,
                    )
                active_episode.observe(
                    question_id=question.id,
                    family_id=question.family_id,
                    focus_concept_id=focus_concept_before,
                    focus_objective_id=focus_objective_before,
                )

            step = SimulationStep(
                index=step_index,
                phase_before=phase_before,
                phase_after=result.next_phase,
                question_id=question.id,
                family_id=question.family_id,
                surface_primary_concept_id=question.primary_concept_id,
                evidence_anchor_concept_id=evidence_anchor_concept_id(question),
                primary_concept_id=question.primary_concept_id,
                learning_objective_id=question.objective_id,
                question_kind=question.kind.value,
                pedagogical_role=presentation.pedagogical_role,
                topic_ids=tuple(
                    item["id"]
                    for item in self.engine.database.question_topics(
                        question.id, session["corpus_release_id"]
                    )
                ),
                continuity=presentation.score.continuity,
                predicted_correct=presentation.score.predicted_correct,
                ground_truth_probability=answer.ground_truth_probability,
                actual_correct=answer.correct,
                selected_option_id=answer.selected_option_id,
                focus_concept_before=focus_concept_before,
                focus_concept_after=result.focus_concept_id,
                focus_objective_before=focus_objective_before,
                focus_objective_after=result.focus_objective_id,
                focus_misconception_before=focus_misconception_before,
                focus_misconception_after=result.focus_misconception_id,
                exact_repeat=exact_repeat,
                family_repeat=family_repeat,
                response_ms=answer.response_ms,
                confidence=answer.confidence,
                hint_count=answer.hint_count,
                selected_at=selected_at,
                answered_at=answered_at,
            )
            steps.append(step)

            if (
                phase_before in _MAIN_PHASES
                and result.next_phase == SessionPhase.REMEDIATE
            ):
                active_episode = _EpisodeBuilder(
                    start_step=step_index + 1,
                    trigger_question_id=question.id,
                    trigger_family_id=question.family_id,
                    initial_focus_concept_id=result.focus_concept_id,
                    initial_focus_objective_id=result.focus_objective_id,
                    initial_focus_misconception_id=result.focus_misconception_id,
                    focus_path=(
                        [result.focus_concept_id] if result.focus_concept_id else []
                    ),
                    objective_focus_path=(
                        [result.focus_objective_id]
                        if result.focus_objective_id
                        else []
                    ),
                )
            elif (
                phase_before in {SessionPhase.REMEDIATE, SessionPhase.VERIFY}
                and result.next_phase in _MAIN_PHASES
                and active_episode is not None
            ):
                if (
                    phase_before == SessionPhase.VERIFY
                    and response_class.certifies_retrieval
                ):
                    outcome = "resolved"
                elif result.correct:
                    outcome = "bounded_uncertainty_exit"
                else:
                    outcome = "bounded_failure_exit"
                episodes.append(active_episode.finish(end_step=step_index, outcome=outcome))
                active_episode = None

            seen_questions.add(question.id)
            seen_families.add(question.family_id)
            clock.advance(inter_item_delay)

        if active_episode is not None:
            episodes.append(
                active_episode.finish(
                    end_step=len(steps) - 1,
                    outcome="step_limit",
                )
            )

        calibration = calibration_metrics(steps)
        observed_evidence_families = {
            (
                f"objective:{step.learning_objective_id}|family:{step.family_id}"
                if step.learning_objective_id is not None
                else (
                    f"concept:{step.evidence_anchor_concept_id}|"
                    f"family:{step.family_id}"
                )
            )
            for step in steps
        }
        observed_objective_families = {
            f"objective:{step.learning_objective_id}|family:{step.family_id}"
            for step in steps
            if step.learning_objective_id is not None
        }
        observed_concepts = {
            step.evidence_anchor_concept_id for step in steps
        }
        observed_objectives = {
            step.learning_objective_id
            for step in steps
            if step.learning_objective_id is not None
        }
        observed_questions = {step.question_id for step in steps}
        observed_families = {step.family_id for step in steps}
        coverage = CoverageMetrics(
            scope_concepts=coverage_denominator.scope_concepts,
            scope_objectives=coverage_denominator.scope_objectives,
            eligible_concepts=coverage_denominator.eligible_concepts,
            eligible_objectives=coverage_denominator.eligible_objectives,
            eligible_questions=coverage_denominator.eligible_questions,
            eligible_families=coverage_denominator.eligible_families,
            eligible_evidence_families=(
                coverage_denominator.eligible_evidence_families
            ),
            eligible_objective_families=(
                coverage_denominator.eligible_objective_families
            ),
            observed_concepts=tuple(
                sorted(
                    observed_concepts
                    & coverage_denominator.eligible_concept_ids
                )
            ),
            observed_surface_concepts=tuple(
                sorted({step.surface_primary_concept_id for step in steps})
            ),
            observed_objectives=tuple(
                sorted(
                    observed_objectives
                    & coverage_denominator.eligible_objective_ids
                )
            ),
            observed_questions=tuple(
                sorted(
                    observed_questions
                    & coverage_denominator.eligible_question_ids
                )
            ),
            observed_families=tuple(
                sorted(
                    observed_families
                    & coverage_denominator.eligible_family_ids
                )
            ),
            observed_evidence_families=tuple(
                sorted(
                    observed_evidence_families
                    & coverage_denominator.eligible_evidence_family_ids
                )
            ),
            observed_objective_families=tuple(
                sorted(
                    observed_objective_families
                    & coverage_denominator.eligible_objective_family_ids
                )
            ),
            observed_outside_scope_concepts=tuple(
                sorted(
                    observed_concepts
                    - coverage_denominator.eligible_concept_ids
                )
            ),
            observed_outside_scope_objectives=tuple(
                sorted(
                    observed_objectives
                    - coverage_denominator.eligible_objective_ids
                )
            ),
            observed_outside_scope_questions=tuple(
                sorted(
                    observed_questions
                    - coverage_denominator.eligible_question_ids
                )
            ),
            observed_outside_scope_families=tuple(
                sorted(
                    observed_families
                    - coverage_denominator.eligible_family_ids
                )
            ),
            observed_outside_scope_evidence_families=tuple(
                sorted(
                    observed_evidence_families
                    - coverage_denominator.eligible_evidence_family_ids
                )
            ),
            observed_outside_scope_objective_families=tuple(
                sorted(
                    observed_objective_families
                    - coverage_denominator.eligible_objective_family_ids
                )
            ),
        )
        return SimulationReport(
            profile_name=profile.name,
            generator_model=getattr(
                profile,
                "response_model",
                f"pattern:{getattr(profile, 'rule', 'custom')}",
            ),
            learner_id=learner_id,
            root_concept_id=root_concept_id,
            mode=mode,
            policy_seed=policy_seed,
            trial_index=trial_index,
            started_at=started_at,
            ended_at=clock.current,
            steps=tuple(steps),
            gaps=tuple(gaps),
            focus_episodes=tuple(episodes),
            coverage=coverage,
            calibration=calibration,
            phase_counts=dict(sorted(phase_counts.items())),
            phase_transitions=dict(sorted(transitions.items())),
            exact_repeat_count=sum(step.exact_repeat for step in steps),
            family_repeat_count=sum(step.family_repeat for step in steps),
            remediation_exact_repeat_count=sum(
                episode.exact_repeat_count for episode in episodes
            ),
            remediation_family_repeat_count=sum(
                episode.family_repeat_count for episode in episodes
            ),
            idempotent_retries_verified=idempotent_retries_verified,
        )

    def evaluate(
        self,
        profile: SyntheticLearner,
        *,
        learner_id_prefix: str,
        root_concept_id: str,
        policy_seeds: Iterable[int],
        max_steps: int = 30,
        mode: str = "learn",
        start_at: datetime = DEFAULT_SIMULATION_START,
        inter_item_delay: timedelta = timedelta(minutes=5),
    ) -> CohortReport:
        """Run independent learners over a paired, caller-supplied seed set."""

        seeds = tuple(policy_seeds)
        if not seeds:
            raise ValidationError("At least one policy seed is required.")
        reports = tuple(
            self.run(
                profile,
                learner_id=f"{learner_id_prefix}-{trial_index}",
                root_concept_id=root_concept_id,
                policy_seed=seed,
                max_steps=max_steps,
                mode=mode,
                start_at=start_at + timedelta(days=trial_index),
                inter_item_delay=inter_item_delay,
                trial_index=trial_index,
            )
            for trial_index, seed in enumerate(seeds)
        )
        return CohortReport(
            profile_name=profile.name,
            trials=reports,
            calibration=calibration_metrics(
                step for report in reports for step in report.steps
            ),
        )

    def _coverage_denominator(
        self, session: Mapping[str, Any]
    ) -> _CoverageDenominator:
        graph = self.engine.database.get_graph(session["corpus_release_id"])
        scope = (
            self.engine.database.topic_scope(
                session["topic_id"], session["corpus_release_id"]
            )
            if session.get("topic_id")
            else graph.learning_scope(session["root_concept_id"])
        )
        release_id = session["corpus_release_id"]
        objectives = self.engine.database.get_learning_objectives(release_id)
        scope_objectives = {
            objective.id
            for objective in objectives
            if objective.primary_concept_id in scope
        }
        with self.engine.database.read() as connection:
            rows = connection.execute(
                """SELECT q.id AS question_id, q.family_id,
                          surface.concept_id AS surface_concept_id,
                          direct.objective_id,
                          objective.primary_concept_id
                              AS objective_owner_concept_id
                   FROM release_questions membership
                   JOIN questions q ON q.id = membership.question_id
                   JOIN question_concepts surface
                     ON surface.question_id = q.id AND surface.role = 'primary'
                   LEFT JOIN release_question_objectives direct
                     ON direct.release_id = membership.release_id
                    AND direct.question_id = q.id
                   LEFT JOIN learning_objectives objective
                     ON objective.id = direct.objective_id
                   WHERE membership.release_id = ?
                     AND membership.status IN ('approved', 'calibrated')
                     AND NOT EXISTS (
                         SELECT 1
                         FROM question_revocations revoked
                         WHERE revoked.question_id = q.id
                     )
                   ORDER BY q.id""",
                (release_id,),
            ).fetchall()

        eligible = []
        for row in rows:
            anchor = (
                row["objective_owner_concept_id"]
                if row["objective_id"] is not None
                else row["surface_concept_id"]
            )
            if anchor in scope:
                eligible.append((row, anchor))
        evidence_families = {
            (
                f"objective:{row['objective_id']}|family:{row['family_id']}"
                if row["objective_id"] is not None
                else f"concept:{anchor}|family:{row['family_id']}"
            )
            for row, anchor in eligible
        }
        objective_families = {
            f"objective:{row['objective_id']}|family:{row['family_id']}"
            for row, _anchor in eligible
            if row["objective_id"] is not None
        }
        eligible_concept_ids = frozenset(anchor for _row, anchor in eligible)
        eligible_objective_ids = frozenset(
            row["objective_id"]
            for row, _anchor in eligible
            if row["objective_id"] is not None
        )
        eligible_question_ids = frozenset(
            row["question_id"] for row, _anchor in eligible
        )
        eligible_family_ids = frozenset(
            row["family_id"] for row, _anchor in eligible
        )
        return _CoverageDenominator(
            scope_concepts=len(scope),
            scope_objectives=len(scope_objectives),
            eligible_concepts=len(eligible_concept_ids),
            eligible_objectives=len(eligible_objective_ids),
            eligible_questions=len(eligible_question_ids),
            eligible_families=len(eligible_family_ids),
            eligible_evidence_families=len(evidence_families),
            eligible_objective_families=len(objective_families),
            eligible_concept_ids=eligible_concept_ids,
            eligible_objective_ids=eligible_objective_ids,
            eligible_question_ids=eligible_question_ids,
            eligible_family_ids=eligible_family_ids,
            eligible_evidence_family_ids=frozenset(evidence_families),
            eligible_objective_family_ids=frozenset(objective_families),
        )


def calibration_metrics(
    steps: Iterable[SimulationStep], *, bin_count: int = 10
) -> CalibrationMetrics:
    """Calculate proper scoring rules and equal-width reliability bins."""

    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count <= 0:
        raise ValidationError("bin_count must be a positive integer.")
    observations_list: list[tuple[float, int]] = []
    for step in steps:
        if not math.isfinite(step.predicted_correct) or not (
            0.0 <= step.predicted_correct <= 1.0
        ):
            raise ValidationError(
                f"Policy emitted an invalid correctness probability for {step.question_id}."
            )
        observations_list.append(
            (step.predicted_correct, int(step.actual_correct))
        )
    observations = tuple(observations_list)
    if not observations:
        return CalibrationMetrics(0, None, None, None, ())

    brier = sum((prediction - actual) ** 2 for prediction, actual in observations)
    brier /= len(observations)
    log_loss = -sum(
        actual * math.log(_clamp_probability(prediction))
        + (1 - actual) * math.log(_clamp_probability(1.0 - prediction))
        for prediction, actual in observations
    ) / len(observations)

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bin_count)]
    for prediction, actual in observations:
        bucket_index = min(bin_count - 1, int(prediction * bin_count))
        buckets[bucket_index].append((prediction, actual))
    bins: list[CalibrationBin] = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_prediction = sum(value[0] for value in bucket) / len(bucket)
        observed_accuracy = sum(value[1] for value in bucket) / len(bucket)
        ece += len(bucket) / len(observations) * abs(
            mean_prediction - observed_accuracy
        )
        bins.append(
            CalibrationBin(
                lower=index / bin_count,
                upper=(index + 1) / bin_count,
                count=len(bucket),
                mean_predicted=mean_prediction,
                observed_accuracy=observed_accuracy,
            )
        )
    return CalibrationMetrics(
        count=len(observations),
        brier_score=brier,
        log_loss=log_loss,
        expected_calibration_error=ece,
        bins=tuple(bins),
    )


def assert_behavioral_invariants(report: SimulationReport) -> None:
    """Raise a useful audit error without substituting easier questions.

    Corpus exhaustion is surfaced first because it often explains a truncated
    episode.  Repetition and tunnel failures are then checked independently.
    """

    if report.gaps:
        details = "; ".join(
            f"step {gap.step_index} ({gap.phase.value}): {gap.message}"
            for gap in report.gaps
        )
        raise AssertionError(f"Behavioral simulation encountered blockers: {details}")
    if report.remediation_exact_repeat_count:
        raise AssertionError(
            "A remediation episode reused an exact question, including its trigger."
        )
    if report.remediation_family_repeat_count:
        raise AssertionError(
            "A remediation episode reused an item family, including its trigger."
        )
    overlong_failure_runs: list[FocusEpisode] = []
    for episode in report.focus_episodes:
        consecutive_failures = 0
        longest_failure_run = 0
        for step in report.steps[episode.start_step : episode.end_step + 1]:
            if (
                step.phase_before in {SessionPhase.REMEDIATE, SessionPhase.VERIFY}
                and not step.actual_correct
            ):
                consecutive_failures += 1
                longest_failure_run = max(
                    longest_failure_run, consecutive_failures
                )
            else:
                consecutive_failures = 0
        if longest_failure_run > MAX_REMEDIATION_DEPTH:
            overlong_failure_runs.append(episode)
    if overlong_failure_runs:
        raise AssertionError(
            "A remediation episode exceeded the consecutive-failure tunnel bound."
        )
