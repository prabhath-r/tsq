# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from math import erfc, exp, isfinite, log, pi, sqrt
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def logit(probability: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, probability))
    return log(p / (1.0 - p))


MASTERY_THRESHOLD = 0.65


class RelationType(StrEnum):
    PREREQUISITE = "prerequisite"
    REQUIRES = "requires"
    PART_OF = "part_of"
    CONTRASTS_WITH = "contrasts_with"
    HELPS_WITH = "helps_with"
    RELATED_TO = "related_to"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"

    @property
    def is_strict_prerequisite(self) -> bool:
        return self in {RelationType.PREREQUISITE, RelationType.REQUIRES}


class QuestionStatus(StrEnum):
    DRAFT = "draft"
    QUARANTINED = "quarantined"
    PILOT = "pilot"
    APPROVED = "approved"
    CALIBRATED = "calibrated"
    RETIRED = "retired"

    @property
    def eligible_for_adaptation(self) -> bool:
        return self in {QuestionStatus.APPROVED, QuestionStatus.CALIBRATED}

    @property
    def evidence_weight(self) -> float:
        return {
            QuestionStatus.DRAFT: 0.0,
            QuestionStatus.QUARANTINED: 0.0,
            QuestionStatus.PILOT: 0.2,
            QuestionStatus.APPROVED: 0.65,
            QuestionStatus.CALIBRATED: 1.0,
            QuestionStatus.RETIRED: 0.0,
        }[self]


class QuestionKind(StrEnum):
    DIAGNOSTIC = "diagnostic"
    CONCEPTUAL = "conceptual"
    APPLICATION = "application"
    DEBUGGING = "debugging"
    COUNTERFACTUAL = "counterfactual"
    TRANSFER = "transfer"
    PREREQUISITE_PROBE = "prerequisite_probe"
    CALCULATION = "calculation"
    COMPARISON = "comparison"


class ConceptRole(StrEnum):
    """A reviewed item's relationship to a concept.

    Roles remain deliberately descriptive in this kernel: the learner model may
    decide which roles carry scored evidence, but corpus authors cannot invent
    one-off strings that silently acquire scoring semantics.
    """

    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUPPORTING = "supporting"
    PREREQUISITE = "prerequisite"
    CONTEXT = "context"
    CONTRAST = "contrast"
    TRANSFER = "transfer"

    @property
    def carries_scored_evidence(self) -> bool:
        """Whether this mapping is allowed to change a learner posterior.

        This is intentionally a property of the role rather than a scattered
        learner-model convention.  Supporting and context mappings describe an
        item for retrieval and authoring, but observing the response is not
        evidence about those concepts.
        """
        return self in {
            ConceptRole.PRIMARY,
            ConceptRole.SECONDARY,
            ConceptRole.PREREQUISITE,
            ConceptRole.CONTRAST,
            ConceptRole.TRANSFER,
        }


class SessionPhase(StrEnum):
    DIAGNOSE = "diagnose"
    LEARN = "learn"
    REMEDIATE = "remediate"
    VERIFY = "verify"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class Concept:
    id: str
    name: str
    description: str
    domain: str = "ai"
    prior_mastery: float = 0.20


@dataclass(frozen=True, slots=True)
class ConceptEdge:
    source_id: str
    target_id: str
    relation: RelationType
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class Misconception:
    id: str
    concept_id: str
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    title: str
    uri: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConceptWeight:
    concept_id: str
    weight: float
    role: ConceptRole = ConceptRole.PRIMARY

    def __post_init__(self) -> None:
        if not isinstance(self.role, ConceptRole):
            object.__setattr__(self, "role", ConceptRole(self.role))


@dataclass(frozen=True, slots=True)
class Option:
    id: str
    text: str
    correct: bool
    rationale: str
    misconception_id: str | None = None


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    version: int
    family_id: str
    status: QuestionStatus
    stem: str
    kind: QuestionKind
    difficulty: float
    discrimination: float
    guess_rate: float
    slip_rate: float
    concepts: tuple[ConceptWeight, ...]
    options: tuple[Option, ...]
    source_ids: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    revision_of: str | None = None

    @property
    def correct_option(self) -> Option:
        return next(option for option in self.options if option.correct)

    @property
    def primary_concept_id(self) -> str:
        primary = [c for c in self.concepts if c.role is ConceptRole.PRIMARY]
        return (primary[0] if primary else max(self.concepts, key=lambda c: c.weight)).concept_id

    @property
    def misconception_ids(self) -> set[str]:
        return {o.misconception_id for o in self.options if o.misconception_id}


@dataclass(slots=True)
class SkillState:
    learner_id: str
    concept_id: str
    mean: float
    variance: float
    stability_hours: float
    exposures: int = 0
    last_seen_at: datetime | None = None
    next_review_at: datetime | None = None
    evidence_mass: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.mean):
            raise ValueError("Skill-state mean must be finite.")
        if not isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError("Skill-state variance must be finite and positive.")
        if not isfinite(self.stability_hours) or self.stability_hours <= 0.0:
            raise ValueError("Skill-state stability must be finite and positive.")
        if (
            not isinstance(self.exposures, int)
            or isinstance(self.exposures, bool)
            or self.exposures < 0
        ):
            raise ValueError("Skill-state exposures must be a non-negative integer.")
        if not isfinite(self.evidence_mass) or self.evidence_mass < 0.0:
            raise ValueError("Skill-state evidence mass must be finite and non-negative.")

    @property
    def expected_competence(self) -> float:
        """Approximate posterior mean on the logistic competence scale."""
        logistic_normal_scale = sqrt(1.0 + pi * self.variance / 8.0)
        return sigmoid(self.mean / logistic_normal_scale)

    def probability_above(self, threshold: float) -> float:
        """Posterior probability that latent competence exceeds a threshold."""
        boundary = logit(threshold)
        z = (boundary - self.mean) / sqrt(2.0 * self.variance)
        return min(1.0, max(0.0, 0.5 * erfc(z)))

    @property
    def mastery_probability(self) -> float:
        """Calibrated certification belief, including posterior uncertainty."""
        return self.probability_above(MASTERY_THRESHOLD)


@dataclass(slots=True)
class MisconceptionBelief:
    learner_id: str
    misconception_id: str
    log_odds: float
    evidence_count: int = 0
    last_seen_at: datetime | None = None

    @property
    def probability(self) -> float:
        return sigmoid(self.log_odds)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    question_id: str
    total: float
    predicted_correct: float
    information_gain: float
    learning_fit: float
    concept_need: float
    misconception_value: float
    prerequisite_value: float
    review_value: float
    novelty: float
    kind_fit: float

    def terms(self) -> dict[str, float]:
        return {
            "total": self.total,
            "predicted_correct": self.predicted_correct,
            "information_gain": self.information_gain,
            "learning_fit": self.learning_fit,
            "concept_need": self.concept_need,
            "misconception_value": self.misconception_value,
            "prerequisite_value": self.prerequisite_value,
            "review_value": self.review_value,
            "novelty": self.novelty,
            "kind_fit": self.kind_fit,
        }


@dataclass(frozen=True, slots=True)
class Presentation:
    decision_id: str
    session_id: str
    question: Question
    option_order: tuple[str, ...]
    phase: SessionPhase
    score: CandidateScore
    propensity: float
    rationale: str

    @property
    def ordered_options(self) -> tuple[Option, ...]:
        by_id = {option.id: option for option in self.question.options}
        return tuple(by_id[option_id] for option_id in self.option_order)


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    interaction_id: str
    correct: bool
    selected_option: Option | None
    correct_option: Option
    next_phase: SessionPhase
    focus_concept_id: str | None
    focus_misconception_id: str | None
    state_changes: tuple[dict[str, Any], ...]
    idempotent_replay: bool = False


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    question_id: str | None = None
    path: str | None = None
