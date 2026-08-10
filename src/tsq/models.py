# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from math import erfc, exp, isfinite, log, pi, sqrt
from typing import Any

from .families import family_assignment
from .objective_posterior import ObjectivePosterior


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

# Public command bounds keep adversarial integers inside both the pedagogical
# meaning of an MCQ attempt and SQLite's fixed-width INTEGER representation.
# Long-running project telemetry uses semantic checkpoints, not one enormous
# multiple-choice response duration.
MAX_RESPONSE_MS = 7 * 24 * 60 * 60 * 1000
MAX_HINT_COUNT = 10_000
MAX_REMEDIATION_DEPTH = 3


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
            QuestionStatus.PILOT: 0.2,
            QuestionStatus.APPROVED: 0.65,
            QuestionStatus.CALIBRATED: 1.0,
            QuestionStatus.RETIRED: 0.0,
        }[self]


def question_status_from_storage(value: object) -> QuestionStatus:
    """Decode a persisted question status from any supported database.

    Releases written before schema 23 may contain the retired curriculum-only
    ``quarantined`` label.  It had the same zero-evidence behavior as ``draft``
    and is decoded that way for pinned-session and replay compatibility.  New
    corpus input and new database rows cannot use the legacy label.
    """

    if value == "quarantined":
        return QuestionStatus.DRAFT
    return QuestionStatus(value)


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


class ObjectiveOperation(StrEnum):
    """The observable reasoning operation named by a learning objective.

    These labels describe what an item is intended to elicit.  They do not
    upgrade selected-response evidence into productive performance evidence;
    that distinction is carried separately by ``evidence_type``.
    """

    DISTINGUISH = "distinguish"
    EXPLAIN = "explain"
    PREDICT = "predict"
    TRACE = "trace"
    DIAGNOSE = "diagnose"
    APPLY = "apply"


@dataclass(frozen=True, slots=True)
class ObjectiveEdge:
    """A reviewed prerequisite claim between two learning objectives.

    ``source_id`` is the prerequisite and ``target_id`` is the dependent
    objective.  Objective edges are release data rather than part of either
    objective's global identity: curriculum authors can revise the dependency
    claim only by publishing a new immutable release.
    """

    id: str
    source_id: str
    target_id: str
    relation: RelationType
    weight: float
    rationale: str

    def __post_init__(self) -> None:
        for field_name in ("id", "source_id", "target_id", "rationale"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Objective-edge {field_name} must be a non-blank string."
                )
        if not isinstance(self.relation, RelationType):
            object.__setattr__(self, "relation", RelationType(self.relation))
        if not self.relation.is_strict_prerequisite:
            raise ValueError(
                "Objective edges currently support only prerequisite relations."
            )
        if self.source_id == self.target_id:
            raise ValueError("A learning objective cannot require itself.")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not isfinite(float(self.weight))
            or not 0.0 < float(self.weight) <= 1.0
        ):
            raise ValueError(
                "Objective-edge weight must be finite and in the interval (0, 1]."
            )


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
class Domain:
    """A learner-facing subject area in one immutable corpus release."""

    id: str
    name: str
    description: str
    sort_order: int = 0

    def __post_init__(self) -> None:
        for field_name in ("id", "name", "description"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Domain {field_name} must be a non-blank string.")
        if (
            not isinstance(self.sort_order, int)
            or isinstance(self.sort_order, bool)
            or self.sort_order < 0
        ):
            raise ValueError("Domain sort_order must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class Topic:
    """A release-pinned curriculum node owning one or more graph concepts.

    Topics are deliberately separate from concepts.  A topic is a navigational
    curriculum bucket that may contain child topics, while concepts remain the
    stable units against which learner evidence is recorded.
    """

    id: str
    domain_id: str
    name: str
    description: str
    concept_ids: tuple[str, ...] = ()
    parent_id: str | None = None
    related_topic_ids: tuple[str, ...] = ()
    sort_order: int = 0

    def __post_init__(self) -> None:
        for field_name in ("id", "domain_id", "name", "description"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Topic {field_name} must be a non-blank string.")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, str) or not self.parent_id.strip()
        ):
            raise ValueError("Topic parent_id must be null or a non-blank string.")
        for field_name in ("concept_ids", "related_topic_ids"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(
                    f"Topic {field_name} must be a tuple of non-blank strings."
                )
        if (
            not isinstance(self.sort_order, int)
            or isinstance(self.sort_order, bool)
            or self.sort_order < 0
        ):
            raise ValueError("Topic sort_order must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class Concept:
    id: str
    name: str
    description: str
    domain: str = "ai"
    prior_mastery: float = 0.20


@dataclass(frozen=True, slots=True)
class LearningObjective:
    """A release-pinned, assessable claim or operation within graph concepts.

    Concepts remain the reusable nodes in the prerequisite graph.  Objectives
    are deliberately finer: they state the boundary that one response can
    actually provide evidence about and may span several closely related graph
    concepts.  The first production slice is selected-response only, so the
    evidence type is explicit and cannot be mistaken for implementation,
    explanation, or design performance.
    """

    id: str
    name: str
    description: str
    primary_concept_id: str
    supporting_concept_ids: tuple[str, ...]
    operation: ObjectiveOperation
    evidence_type: str = "selected_response"
    prior_mastery: float = 0.20
    prerequisites: tuple[ObjectiveEdge, ...] = ()
    objective_graph_version: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("id", "name", "description", "evidence_type"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Learning-objective {field_name} must be a non-blank string."
                )
        if not isinstance(self.operation, ObjectiveOperation):
            object.__setattr__(
                self, "operation", ObjectiveOperation(self.operation)
            )
        if (
            not isinstance(self.primary_concept_id, str)
            or not self.primary_concept_id.strip()
        ):
            raise ValueError(
                "Learning objectives require a non-blank primary concept ID."
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.supporting_concept_ids
        ):
            raise ValueError(
                "Learning-objective supporting concept IDs must be non-blank strings."
            )
        if (
            self.primary_concept_id in self.supporting_concept_ids
            or len(set(self.supporting_concept_ids))
            != len(self.supporting_concept_ids)
        ):
            raise ValueError("Learning-objective concept IDs must be unique.")
        if self.evidence_type != "selected_response":
            raise ValueError(
                "This corpus version supports only selected_response objectives."
            )
        if (
            isinstance(self.prior_mastery, bool)
            or not isinstance(self.prior_mastery, (int, float))
            or not isfinite(float(self.prior_mastery))
            or not 0.0 < float(self.prior_mastery) < 1.0
        ):
            raise ValueError(
                "Learning-objective prior_mastery must be finite and between zero and one."
            )
        if self.objective_graph_version not in {None, 1}:
            raise ValueError("Unsupported learning-objective graph version.")
        if self.prerequisites and self.objective_graph_version is None:
            raise ValueError(
                "Learning-objective prerequisites require a declared graph version."
            )
        if any(edge.target_id != self.id for edge in self.prerequisites):
            raise ValueError(
                "Learning-objective prerequisite edges must target their containing objective."
            )
        if len({edge.id for edge in self.prerequisites}) != len(self.prerequisites):
            raise ValueError("Learning-objective prerequisite edge IDs must be unique.")

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return (self.primary_concept_id, *self.supporting_concept_ids)


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
    diagnostic_objective_id: str | None = None


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
    objective: LearningObjective | None = None
    published_family_id: str | None = None

    def __post_init__(self) -> None:
        assignment = family_assignment(
            self.id,
            self.published_family_id or self.family_id,
        )
        object.__setattr__(
            self,
            "published_family_id",
            assignment.published_family_id,
        )
        object.__setattr__(self, "family_id", assignment.evidence_family_id)

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

    @property
    def objective_id(self) -> str | None:
        return self.objective.id if self.objective is not None else None


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
        """Model-implied certification belief, pending empirical calibration."""
        return self.probability_above(MASTERY_THRESHOLD)


@dataclass(slots=True)
class ObjectiveState:
    """Uncertain learner state for one fine-grained learning objective."""

    learner_id: str
    objective_id: str
    mean: float
    variance: float
    stability_hours: float
    exposures: int = 0
    last_seen_at: datetime | None = None
    next_review_at: datetime | None = None
    evidence_mass: float = 0.0
    posterior: ObjectivePosterior | None = field(default=None, repr=False)
    model_version: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.mean):
            raise ValueError("Objective-state mean must be finite.")
        if not isfinite(self.variance) or self.variance <= 0.0:
            raise ValueError("Objective-state variance must be finite and positive.")
        if not isfinite(self.stability_hours) or self.stability_hours <= 0.0:
            raise ValueError("Objective-state stability must be finite and positive.")
        if (
            not isinstance(self.exposures, int)
            or isinstance(self.exposures, bool)
            or self.exposures < 0
        ):
            raise ValueError("Objective-state exposures must be a non-negative integer.")
        if not isfinite(self.evidence_mass) or self.evidence_mass < 0.0:
            raise ValueError(
                "Objective-state evidence mass must be finite and non-negative."
            )
        if self.posterior is not None:
            if not isinstance(self.posterior, ObjectivePosterior):
                raise ValueError(
                    "Objective-state posterior must be an ObjectivePosterior."
                )
            metrics = self.posterior.metrics()
            for label, stored, derived in (
                ("mean", self.mean, metrics.mean),
                ("variance", self.variance, metrics.variance),
                ("evidence mass", self.evidence_mass, metrics.evidence_mass),
            ):
                if abs(stored - derived) > 1e-10 * max(
                    1.0, abs(stored), abs(derived)
                ):
                    raise ValueError(
                        f"Objective-state {label} does not match its exact posterior."
                    )
        if self.model_version is not None and (
            not isinstance(self.model_version, str)
            or not self.model_version.strip()
        ):
            raise ValueError(
                "Objective-state model version must be null or a non-blank string."
            )

    @property
    def expected_competence(self) -> float:
        if self.posterior is not None:
            return self.posterior.expected_competence
        logistic_normal_scale = sqrt(1.0 + pi * self.variance / 8.0)
        return sigmoid(self.mean / logistic_normal_scale)

    def probability_above(self, threshold: float) -> float:
        if self.posterior is not None:
            return self.posterior.probability_above_competence(threshold)
        boundary = logit(threshold)
        z = (boundary - self.mean) / sqrt(2.0 * self.variance)
        return min(1.0, max(0.0, 0.5 * erfc(z)))

    @property
    def estimated_mastery_probability(self) -> float:
        """Represented posterior tail before the numerical safety envelope."""

        return self.probability_above(MASTERY_THRESHOLD)

    @property
    def mastery_probability_error_bound(self) -> float:
        if self.posterior is None:
            return 0.0
        return self.posterior.metrics().mastery_probability_error_bound

    @property
    def mastery_probability(self) -> float:
        if self.posterior is not None:
            return self.posterior.conservative_mastery_probability
        return self.estimated_mastery_probability

    @property
    def acquisition_mass(self) -> float:
        return 0.0 if self.posterior is None else self.posterior.acquisition_mass


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
    continuity: float = 0.0
    boundary_fit: float = 0.0
    # Hybrid-coverage values describe the selected target before presentation.
    # Raw exposure is a burden guard, diagnostic information is cumulative
    # quality-discounted evidence in the target projection, and successful
    # retrieval families are learner-model-verified internal selected-response
    # families. The defaults keep historical selected-score JSON
    # reconstructible; none of these fields is external skill certification.
    coverage_raw_exposures: int = 0
    coverage_diagnostic_information: float = 0.0
    coverage_successful_retrieval_families: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.coverage_raw_exposures) is not int
            or self.coverage_raw_exposures < 0
        ):
            raise ValueError(
                "coverage_raw_exposures must be a non-negative integer."
            )
        if (
            isinstance(self.coverage_diagnostic_information, bool)
            or not isinstance(
                self.coverage_diagnostic_information, (int, float)
            )
            or not isfinite(float(self.coverage_diagnostic_information))
            or self.coverage_diagnostic_information < 0.0
        ):
            raise ValueError(
                "coverage_diagnostic_information must be finite and "
                "non-negative."
            )
        if (
            type(self.coverage_successful_retrieval_families) is not int
            or self.coverage_successful_retrieval_families < 0
            or self.coverage_successful_retrieval_families
            > self.coverage_raw_exposures
        ):
            raise ValueError(
                "coverage_successful_retrieval_families must be a "
                "non-negative integer no greater than raw exposures."
            )

    def terms(self) -> dict[str, float | int]:
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
            "continuity": self.continuity,
            "boundary_fit": self.boundary_fit,
            "coverage_raw_exposures": self.coverage_raw_exposures,
            "coverage_diagnostic_information": (
                self.coverage_diagnostic_information
            ),
            "coverage_successful_retrieval_families": (
                self.coverage_successful_retrieval_families
            ),
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
    pedagogical_role: str = "main"

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
    focus_objective_id: str | None
    state_changes: tuple[dict[str, Any], ...]
    transition_reason: str = "legacy_transition"
    boundary_decision: dict[str, Any] | None = None
    idempotent_replay: bool = False


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    question_id: str | None = None
    path: str | None = None
