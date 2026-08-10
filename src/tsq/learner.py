# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite, pi, sqrt
from types import MappingProxyType
from typing import Iterable, Mapping

from .families import (
    canonical_family_label,
    evidence_family_id,
    register_family_sql_functions,
)
from .inference import (
    LEGACY_MISCONCEPTION_ALGORITHM,
    MIN_CREDIBLE_CONFIDENCE,
    MIN_CREDIBLE_RESPONSE_MS,
    MIN_NAMED_ERROR_CONFIDENCE,
    MISCONCEPTION_ALGORITHM_METADATA_KEY,
    MISCONCEPTION_ALGORITHM_VERSION,
    ResponseClass,
    SUPPORTED_MISCONCEPTION_ALGORITHMS,
    classify_response_for_model,
)
from .models import (
    MASTERY_THRESHOLD,
    Concept,
    ConceptRole,
    LearningObjective,
    ObjectiveState,
    Option,
    Question,
    SkillState,
    logit,
    sigmoid,
)
from .objective_posterior import (
    LikelihoodObservation,
    ObjectivePosterior,
    PosteriorMetrics,
)
from .store import Database, to_timestamp
from .versions import (
    CANONICAL_FAMILY_MODEL_VERSIONS,
    CANONICAL_FAMILY_V9_MODEL_VERSION,
    CONCEPT_MODEL_VERSION,
    DEFAULT_LEARNER_MODEL_VERSION,
    LEGACY_MODEL_VERSION,
    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
    OBJECTIVE_GRID_MODEL_VERSIONS,
    OBJECTIVE_GRID_V6_MODEL_VERSION,
    OBJECTIVE_GRID_V7_MODEL_VERSION,
    OBJECTIVE_GRID_V8_MODEL_VERSION,
    OBJECTIVE_MODEL_VERSIONS,
    SPACING_AWARE_FAMILY_MODEL_VERSIONS,
    SUPPORTED_MODEL_VERSIONS,
)


# Compatibility alias for callers selecting the current default. Historical
# dispatch below always uses explicit constants or frozen capability sets.
MODEL_VERSION = DEFAULT_LEARNER_MODEL_VERSION
INITIAL_VARIANCE = 2.25
INITIAL_STABILITY_HOURS = 48.0
MIN_POSTERIOR_VARIANCE = 1e-6
MAX_POSTERIOR_VARIANCE = 4.0
SESSION_LAPSE_RATE = 0.03
MISSING_CONFIDENCE_DISCOUNT = 0.85
MISSING_RESPONSE_TIME_DISCOUNT = 0.70
ABSTENTION_EVIDENCE_DISCOUNT = 0.30

MAX_MISCONCEPTION_FAMILY_CONTRIBUTION = 1.50
MIN_CREDIBLE_MISCONCEPTION_RESPONSE_MS = MIN_CREDIBLE_RESPONSE_MS
MIN_CREDIBLE_CORRECT_CONFIDENCE = MIN_CREDIBLE_CONFIDENCE
MIN_CREDIBLE_NAMED_ERROR_CONFIDENCE = MIN_NAMED_ERROR_CONFIDENCE

# Spacing-aware models preserve v6's square-summable lifetime repeat tail and add a
# geometrically bounded renewal budget.  A renewal requires a fully observable,
# unguided response after a real time gap.  Missing or weak behavioral metadata
# can still contribute discounted ordinary evidence, but can never claim the
# renewal bonus.
MIN_CREDIBLE_RETEST_RESPONSE_MS = 500
MIN_FAMILY_RETEST_SPACING = timedelta(days=7)
FAMILY_RETEST_RENEWAL_HEAD = 0.50
FAMILY_RETEST_RENEWAL_DECAY = 0.50


@dataclass(frozen=True, slots=True)
class FamilyResponseRecord:
    """Immutable response facts needed to reconstruct renewal windows."""

    occurred_at: datetime
    credible: bool
    renewal_eligible: bool = True


@dataclass(frozen=True, slots=True)
class FamilyEvidencePower:
    """One response's bounded same-family likelihood exponent."""

    power: float
    base_power: float
    renewal_power: float
    renewal_index: int | None


@dataclass(frozen=True, slots=True)
class ObjectiveReadinessFloor:
    """Exact fine-objective metrics used by one broad graph node."""

    source_objective_id: str
    mastery_probability: float
    expected_competence: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_objective_id, str)
            or not self.source_objective_id.strip()
        ):
            raise ValueError("Objective readiness floors require a source ID.")
        for label, value in (
            ("mastery_probability", self.mastery_probability),
            ("expected_competence", self.expected_competence),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(
                    f"Objective readiness floor {label} must be a probability."
                )


@dataclass(frozen=True, slots=True)
class ConceptFloorProjection:
    """Non-persisted broad states plus exact objective readiness overrides."""

    states: Mapping[str, SkillState]
    exact_floors: Mapping[str, ObjectiveReadinessFloor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))
        object.__setattr__(
            self,
            "exact_floors",
            MappingProxyType(dict(self.exact_floors)),
        )


class LearnerModel:
    """Interpretable online projection over proficiency and misconceptions.

    Broad legacy concepts retain diagonal Gaussian latent traits. Fine learning
    objectives use a durable fixed-grid posterior under the current model, so
    item prediction and information value integrate uncertainty instead of
    evaluating only a point estimate. Feedback remains a separate, weak learning
    transition so one response is not counted twice as evidence and acquisition.
    """

    def __init__(self, model_version: str = MODEL_VERSION):
        if model_version not in SUPPORTED_MODEL_VERSIONS:
            raise ValueError(f"Unsupported learner model version: {model_version}")
        self.model_version = model_version

    def initial_state(self, learner_id: str, concept: Concept) -> SkillState:
        return SkillState(
            learner_id=learner_id,
            concept_id=concept.id,
            mean=logit(concept.prior_mastery),
            variance=INITIAL_VARIANCE,
            stability_hours=INITIAL_STABILITY_HOURS,
        )

    def initial_objective_state(
        self, learner_id: str, objective: LearningObjective
    ) -> ObjectiveState:
        if self.model_version in OBJECTIVE_GRID_MODEL_VERSIONS:
            posterior = ObjectivePosterior.from_prior(objective.prior_mastery)
            metrics = posterior.metrics()
            return ObjectiveState(
                learner_id=learner_id,
                objective_id=objective.id,
                mean=metrics.mean,
                variance=metrics.variance,
                stability_hours=INITIAL_STABILITY_HOURS,
                evidence_mass=metrics.evidence_mass,
                posterior=posterior,
                model_version=self.model_version,
            )
        return ObjectiveState(
            learner_id=learner_id,
            objective_id=objective.id,
            mean=logit(objective.prior_mastery),
            variance=INITIAL_VARIANCE,
            stability_hours=INITIAL_STABILITY_HOURS,
            model_version=self.model_version,
        )

    def initial_objective_states(
        self,
        learner_id: str,
        objectives: Iterable[LearningObjective],
    ) -> Mapping[str, ObjectiveState]:
        """Materialize immutable cold states once per distinct release prior.

        A release commonly assigns the same reviewed prior to several fine
        objectives. Exact-grid construction and validation are deterministic
        functions of that prior, and the posterior is immutable, so sharing one
        request-local posterior derivative across those objective states
        preserves every encoded byte and metric while avoiding duplicate grid
        integrations. The returned mapping cannot be mutated by callers.
        """

        objective_list = tuple(objectives)
        objective_ids = [objective.id for objective in objective_list]
        if len(objective_ids) != len(set(objective_ids)):
            raise ValueError("Initial objective-state IDs must be unique.")
        if self.model_version not in OBJECTIVE_GRID_MODEL_VERSIONS:
            return MappingProxyType(
                {
                    objective.id: self.initial_objective_state(
                        learner_id, objective
                    )
                    for objective in objective_list
                }
            )

        prior_derivatives: dict[
            float, tuple[ObjectivePosterior, PosteriorMetrics]
        ] = {}
        result: dict[str, ObjectiveState] = {}
        for objective in objective_list:
            derivative = prior_derivatives.get(objective.prior_mastery)
            if derivative is None:
                posterior = ObjectivePosterior.from_prior(
                    objective.prior_mastery
                )
                derivative = (posterior, posterior.metrics())
                prior_derivatives[objective.prior_mastery] = derivative
            posterior, metrics = derivative
            result[objective.id] = ObjectiveState(
                learner_id=learner_id,
                objective_id=objective.id,
                mean=metrics.mean,
                variance=metrics.variance,
                stability_hours=INITIAL_STABILITY_HOURS,
                evidence_mass=metrics.evidence_mass,
                posterior=posterior,
                model_version=self.model_version,
            )
        return MappingProxyType(result)

    def _migrate_objective_posterior(
        self, state: ObjectiveState, objective: LearningObjective
    ) -> ObjectivePosterior:
        """Return exact state, or conservatively lift one v5 Gaussian once.

        A row already labelled v6 or later must have its durable grid child.
        Missing exact state is corruption, not a reason to silently rebuild a
        different posterior from redundant moments. v7 may carry a validated
        v6 exact posterior forward byte-for-byte before applying its first new
        event. Older objective-aware v5 rows may be converted lazily; the next
        response persists the conversion atomically.
        """

        if state.objective_id != objective.id:
            raise ValueError(
                "Objective state does not match the requested learning objective."
            )
        if state.posterior is not None:
            compatible_exact_versions: set[str | None]
            if self.model_version == CANONICAL_FAMILY_V9_MODEL_VERSION:
                compatible_exact_versions = {
                    None,
                    OBJECTIVE_GRID_V6_MODEL_VERSION,
                    OBJECTIVE_GRID_V7_MODEL_VERSION,
                    OBJECTIVE_GRID_V8_MODEL_VERSION,
                    CANONICAL_FAMILY_V9_MODEL_VERSION,
                }
            elif self.model_version == OBJECTIVE_GRID_V8_MODEL_VERSION:
                compatible_exact_versions = {
                    None,
                    OBJECTIVE_GRID_V6_MODEL_VERSION,
                    OBJECTIVE_GRID_V7_MODEL_VERSION,
                    OBJECTIVE_GRID_V8_MODEL_VERSION,
                }
            elif self.model_version == OBJECTIVE_GRID_V7_MODEL_VERSION:
                compatible_exact_versions = {
                    None,
                    OBJECTIVE_GRID_V6_MODEL_VERSION,
                    OBJECTIVE_GRID_V7_MODEL_VERSION,
                }
            elif self.model_version == OBJECTIVE_GRID_V6_MODEL_VERSION:
                compatible_exact_versions = {
                    None,
                    OBJECTIVE_GRID_V6_MODEL_VERSION,
                }
            else:
                compatible_exact_versions = set()
            if state.model_version not in compatible_exact_versions:
                raise ValueError(
                    "An exact objective posterior has an incompatible model version."
                )
            if abs(state.posterior.prior_mastery - objective.prior_mastery) > 1e-12:
                raise ValueError(
                    "Exact objective posterior does not match the release prior."
                )
            return state.posterior
        if state.model_version in OBJECTIVE_GRID_MODEL_VERSIONS:
            raise ValueError(
                "An exact-grid objective state is missing its posterior projection."
            )
        if state.model_version not in {
            None,
            OBJECTIVE_GAUSSIAN_MODEL_VERSION,
        }:
            raise ValueError(
                "Only objective-aware v5 states can be lifted into an exact grid."
            )
        return ObjectivePosterior.from_gaussian(
            prior_mastery=objective.prior_mastery,
            mean=state.mean,
            variance=state.variance,
            committed_evidence_mass=state.evidence_mass,
        )

    def project_objective_state(
        self,
        state: ObjectiveState,
        objective: LearningObjective,
        now: datetime,
    ) -> ObjectiveState:
        """Project one objective using its event-declared model semantics."""

        if self.model_version in OBJECTIVE_GRID_MODEL_VERSIONS:
            posterior = self._migrate_objective_posterior(state, objective)
            if state.last_seen_at is not None:
                elapsed_hours = max(
                    0.0,
                    (now - state.last_seen_at).total_seconds() / 3600.0,
                )
                posterior = posterior.apply_retention(
                    elapsed_hours=elapsed_hours,
                    stability_hours=state.stability_hours,
                )
            metrics = posterior.metrics()
            return ObjectiveState(
                learner_id=state.learner_id,
                objective_id=state.objective_id,
                mean=metrics.mean,
                variance=metrics.variance,
                stability_hours=state.stability_hours,
                exposures=state.exposures,
                last_seen_at=state.last_seen_at,
                next_review_at=state.next_review_at,
                evidence_mass=metrics.evidence_mass,
                posterior=posterior,
                model_version=self.model_version,
            )

        if (
            state.posterior is not None
            or state.model_version in OBJECTIVE_GRID_MODEL_VERSIONS
        ):
            raise ValueError("An exact-grid objective state cannot be downgraded.")

        proxy = SkillState(
            learner_id=state.learner_id,
            concept_id=state.objective_id,
            mean=state.mean,
            variance=state.variance,
            stability_hours=state.stability_hours,
            exposures=state.exposures,
            last_seen_at=state.last_seen_at,
            next_review_at=state.next_review_at,
            evidence_mass=state.evidence_mass,
        )
        projected = self.project_state(
            proxy,
            Concept(
                id=objective.id,
                name=objective.name,
                description=objective.description,
                domain="learning_objective",
                prior_mastery=objective.prior_mastery,
            ),
            now,
        )
        return ObjectiveState(
            learner_id=projected.learner_id,
            objective_id=projected.concept_id,
            mean=projected.mean,
            variance=projected.variance,
            stability_hours=projected.stability_hours,
            exposures=projected.exposures,
            last_seen_at=projected.last_seen_at,
            next_review_at=projected.next_review_at,
            evidence_mass=projected.evidence_mass,
            model_version=self.model_version,
        )

    def concept_projection_with_objective_floor(
        self,
        *,
        learner_id: str,
        concepts: Mapping[str, Concept],
        stored_states: Mapping[str, SkillState],
        objectives: Iterable[LearningObjective],
        stored_objective_states: Mapping[str, ObjectiveState],
        now: datetime,
        projected_objective_states: Mapping[str, ObjectiveState] | None = None,
    ) -> ConceptFloorProjection:
        """Derive conservative broad states without losing exact-grid metrics.

        Objective responses must not be double-counted into the persisted broad
        concept posterior.  Graph routing still needs a meaningful intrinsic
        readiness value, however.  For every canonical objective owner this
        method derives a non-persisted concept state from its weakest current
        objective.  Unassessed objectives therefore keep the broad node
        uncertain, while progress across *all* objectives can raise graph
        readiness without fabricating a second evidence observation.
        """

        result = dict(stored_states)
        exact_floors: dict[str, ObjectiveReadinessFloor] = {}
        objectives_by_concept: dict[str, list[LearningObjective]] = {}
        for objective in objectives:
            if objective.primary_concept_id in concepts:
                objectives_by_concept.setdefault(
                    objective.primary_concept_id, []
                ).append(objective)
        for concept_id, owned_objectives in objectives_by_concept.items():
            projected = []
            for objective in owned_objectives:
                if projected_objective_states is not None:
                    projected_state = projected_objective_states.get(
                        objective.id
                    )
                    if projected_state is None:
                        raise ValueError(
                            "Objective projection cache is incomplete for "
                            f"{objective.id}."
                        )
                    if (
                        projected_state.learner_id != learner_id
                        or projected_state.objective_id != objective.id
                    ):
                        raise ValueError(
                            "Objective projection cache contains a mismatched "
                            f"state for {objective.id}."
                        )
                    projected.append(projected_state)
                else:
                    state = stored_objective_states.get(objective.id)
                    if state is None:
                        state = self.initial_objective_state(
                            learner_id, objective
                        )
                    projected.append(
                        self.project_objective_state(state, objective, now)
                    )
            weakest = min(
                projected,
                key=lambda state: (
                    0.55 * state.mastery_probability
                    + 0.45 * state.expected_competence,
                    state.objective_id,
                ),
            )
            all_seen = all(state.last_seen_at is not None for state in projected)
            review_times = [
                state.next_review_at
                for state in projected
                if state.next_review_at is not None
            ]
            result[concept_id] = SkillState(
                learner_id=learner_id,
                concept_id=concept_id,
                mean=weakest.mean,
                variance=weakest.variance,
                stability_hours=min(
                    state.stability_hours for state in projected
                ),
                exposures=min(state.exposures for state in projected),
                # Every objective was already projected to this common clock.
                # Anchoring the derived state here prevents a second retention
                # decay when the graph planner projects it immediately.
                last_seen_at=now if any(
                    state.last_seen_at is not None for state in projected
                ) else None,
                next_review_at=(
                    min(review_times)
                    if all_seen and len(review_times) == len(projected)
                    else None
                ),
                evidence_mass=min(
                    state.evidence_mass for state in projected
                ),
            )
            exact_floors[concept_id] = ObjectiveReadinessFloor(
                source_objective_id=weakest.objective_id,
                mastery_probability=weakest.mastery_probability,
                expected_competence=weakest.expected_competence,
            )
        return ConceptFloorProjection(result, exact_floors)

    def concept_states_with_objective_floor(
        self,
        *,
        learner_id: str,
        concepts: Mapping[str, Concept],
        stored_states: Mapping[str, SkillState],
        objectives: Iterable[LearningObjective],
        stored_objective_states: Mapping[str, ObjectiveState],
        now: datetime,
        projected_objective_states: Mapping[str, ObjectiveState] | None = None,
    ) -> dict[str, SkillState]:
        """Compatibility view of objective-floored Gaussian state moments."""

        projection = self.concept_projection_with_objective_floor(
            learner_id=learner_id,
            concepts=concepts,
            stored_states=stored_states,
            objectives=objectives,
            stored_objective_states=stored_objective_states,
            now=now,
            projected_objective_states=projected_objective_states,
        )
        return dict(projection.states)

    @staticmethod
    def evidence_weights(question: Question) -> dict[str, float]:
        """Return normalized weights only for explicitly scored concept roles.

        Supporting/context mappings affect authoring and candidate constraints but
        cannot silently become learner evidence. At least the primary mapping is
        always scored by corpus validation.
        """
        eligible = [
            mapping
            for mapping in question.concepts
            if mapping.role.carries_scored_evidence
        ]
        total = sum(mapping.weight for mapping in eligible)
        if total <= 0.0:
            return {}
        return {mapping.concept_id: mapping.weight / total for mapping in eligible}

    def project_state(self, state: SkillState, concept: Concept, now: datetime) -> SkillState:
        if not state.last_seen_at:
            return SkillState(
                learner_id=state.learner_id,
                concept_id=state.concept_id,
                mean=state.mean,
                variance=state.variance,
                stability_hours=state.stability_hours,
                exposures=state.exposures,
                last_seen_at=state.last_seen_at,
                next_review_at=state.next_review_at,
                evidence_mass=state.evidence_mass,
            )
        elapsed_hours = max(0.0, (now - state.last_seen_at).total_seconds() / 3600.0)
        retention = 2.0 ** (-elapsed_hours / max(12.0, state.stability_hours))
        prior_mean = logit(concept.prior_mastery)
        projected_mean = prior_mean + retention * (state.mean - prior_mean)
        # Population priors are suitable cold-start beliefs, not a source of
        # positive evidence.  Once direct evidence places a learner below the
        # prior, inactivity must not raise either the latent estimate or the
        # probability of exceeding the mastery boundary through variance
        # inflation.  We conservatively preserve that posterior until another
        # observation arrives.
        if state.mean <= prior_mean:
            mean = state.mean
            variance = state.variance
        else:
            mean = projected_mean
            variance = min(
                MAX_POSTERIOR_VARIANCE,
                state.variance + (1.0 - retention) * 0.45,
            )
            # Widening a precise, below-threshold Gaussian can paradoxically
            # increase its upper-tail certification probability even while its
            # mean decays.  Cap the widening at the largest variance that keeps
            # both reported competence measures non-improving.  The caps are
            # closed-form consequences of the normal-tail and logistic-normal
            # approximations used by SkillState.
            mastery_boundary = logit(MASTERY_THRESHOLD)
            if state.mean < mastery_boundary and mean < mastery_boundary:
                old_gap = mastery_boundary - state.mean
                new_gap = mastery_boundary - mean
                mastery_variance_cap = state.variance * (new_gap / old_gap) ** 2
                variance = min(variance, mastery_variance_cap)
            if state.mean < -1e-12 and mean < 0.0:
                logistic_scale_term = pi / 8.0
                mean_ratio = mean / state.mean
                # A state arbitrarily close to zero can make the ratio exceed
                # floating-point range. In that case the old expected
                # competence is already arbitrarily close to 0.5 and any
                # finite negative projection is necessarily non-improving.
                if mean_ratio < 1e150:
                    competence_variance_cap = (
                        mean_ratio**2
                        * (1.0 + logistic_scale_term * state.variance)
                        - 1.0
                    ) / logistic_scale_term
                    variance = min(variance, competence_variance_cap)
            variance = max(state.variance, variance)
        return SkillState(
            learner_id=state.learner_id,
            concept_id=state.concept_id,
            mean=mean,
            variance=variance,
            stability_hours=state.stability_hours,
            exposures=state.exposures,
            last_seen_at=state.last_seen_at,
            next_review_at=state.next_review_at,
            evidence_mass=state.evidence_mass,
        )

    def states_for_question(
        self,
        learner_id: str,
        question: Question,
        concepts: Mapping[str, Concept],
        stored_states: Mapping[str, SkillState],
        now: datetime,
    ) -> dict[str, SkillState]:
        states: dict[str, SkillState] = {}
        for mapping in question.concepts:
            concept = concepts[mapping.concept_id]
            state = stored_states.get(mapping.concept_id) or self.initial_state(learner_id, concept)
            states[mapping.concept_id] = self.project_state(state, concept, now)
        return states

    def predict_correct(
        self,
        question: Question,
        states: Mapping[str, SkillState],
        objective_state: ObjectiveState | None = None,
    ) -> float:
        if question.objective_id is not None:
            if objective_state is None:
                raise ValueError(
                    "Objective-aware questions require their release-pinned "
                    "objective state."
                )
            if self.model_version in OBJECTIVE_GRID_MODEL_VERSIONS:
                if question.objective is None:
                    raise ValueError(
                        "Objective-aware questions require objective metadata."
                    )
                posterior = self._migrate_objective_posterior(
                    objective_state, question.objective
                )
                return posterior.predict_correct(
                    difficulty=question.difficulty,
                    discrimination=question.discrimination,
                    guess_rate=question.guess_rate,
                    slip_rate=question.slip_rate,
                    option_count=len(question.options),
                )
            # The reviewed objective is the response's one scored latent
            # dimension. Concept mappings still constrain retrieval and graph
            # readiness, but a selected response cannot identify which broad
            # prerequisite produced an error.
            theta = objective_state.mean
        else:
            weights = self.evidence_weights(question)
            theta = sum(
                weight * states[concept_id].mean
                for concept_id, weight in weights.items()
            )
        logistic = sigmoid(question.discrimination * (theta - question.difficulty))
        modeled = question.guess_rate + (1.0 - question.guess_rate - question.slip_rate) * logistic
        # A lapse mixture prevents one surprising response from destroying a strong posterior.
        random_click = 1.0 / max(2, len(question.options))
        return SESSION_LAPSE_RATE * random_click + (1.0 - SESSION_LAPSE_RATE) * modeled

    def expected_information_gain(
        self,
        question: Question,
        states: Mapping[str, SkillState],
        objective_state: ObjectiveState | None = None,
        *,
        evidence_power_override: float | None = None,
    ) -> float:
        if question.objective_id is not None:
            if objective_state is None:
                raise ValueError(
                    "Objective-aware questions require their release-pinned "
                    "objective state."
                )
            if self.model_version in OBJECTIVE_GRID_MODEL_VERSIONS:
                if question.objective is None:
                    raise ValueError(
                        "Objective-aware questions require objective metadata."
                    )
                posterior = self._migrate_objective_posterior(
                    objective_state, question.objective
                )
                evidence_power = (
                    question.status.evidence_weight
                    if evidence_power_override is None
                    else evidence_power_override
                )
                if (
                    isinstance(evidence_power, bool)
                    or not isinstance(evidence_power, (int, float))
                    or not 0.0 <= float(evidence_power) <= 1.0
                ):
                    raise ValueError(
                        "Expected-information evidence power must be in [0, 1]."
                    )
                return posterior.expected_information_nats(
                    difficulty=question.difficulty,
                    discrimination=question.discrimination,
                    guess_rate=question.guess_rate,
                    slip_rate=question.slip_rate,
                    option_count=len(question.options),
                    evidence_power=float(evidence_power),
                )
            theta = objective_state.mean
        else:
            weights = self.evidence_weights(question)
            theta = sum(
                weight * states[concept_id].mean
                for concept_id, weight in weights.items()
            )
        logistic = sigmoid(question.discrimination * (theta - question.difficulty))
        p = self.predict_correct(
            question, states, objective_state=objective_state
        )
        dp_dtheta = (
            (1.0 - SESSION_LAPSE_RATE)
            * (1.0 - question.guess_rate - question.slip_rate)
            * question.discrimination
            * logistic
            * (1.0 - logistic)
        )
        denominator = max(1e-6, p * (1.0 - p))
        legacy_evidence_power = 1.0
        if evidence_power_override is not None:
            if (
                isinstance(evidence_power_override, bool)
                or not isinstance(evidence_power_override, (int, float))
                or not 0.0 <= float(evidence_power_override) <= 1.0
            ):
                raise ValueError(
                    "Expected-information evidence power must be in [0, 1]."
                )
            legacy_evidence_power = float(evidence_power_override)
        if question.objective_id is not None:
            fisher = dp_dtheta**2 / denominator
            posterior_variance = 1.0 / (
                1.0 / objective_state.variance
                + legacy_evidence_power * fisher
            )
            return max(
                0.0, objective_state.variance - posterior_variance
            )

        gain = 0.0
        for concept_id, weight in weights.items():
            state = states[concept_id]
            fisher = (weight * dp_dtheta) ** 2 / denominator
            posterior_variance = 1.0 / (
                1.0 / state.variance + legacy_evidence_power * fisher
            )
            gain += weight * max(0.0, state.variance - posterior_variance)
        return gain

    @staticmethod
    def family_dependence_discount(prior_family_attempts: int) -> float:
        """Discount repeated observations from a correlated item family.

        ``prior_family_attempts`` excludes the response currently being
        projected.  The first response receives full weight; later responses
        follow a square-summable tail.  Consequently, infinitely many repeats
        of one family have less than 1.412 response-equivalents of influence,
        while genuinely different families can continue adding evidence.
        """
        if prior_family_attempts < 0:
            raise ValueError("Prior family-attempt count cannot be negative.")
        if prior_family_attempts == 0:
            return 1.0
        return 0.25 / float(prior_family_attempts**2)

    @staticmethod
    def family_evidence_power_bound() -> float:
        """Strict upper bound on one spacing-aware family's lifetime evidence power.

        The unchanged base series is

        ``1 + 0.25 * sum(n^-2, n>=1) = 1 + pi^2/24``.

        Credible spaced renewals add at most the geometric series

        ``0.5 * sum(0.5^r, r>=0) = 1``.

        The returned value is the analytical supremum; every finite response
        history is strictly below it.
        """

        return (
            1.0
            + pi * pi / 24.0
            + FAMILY_RETEST_RENEWAL_HEAD
            / (1.0 - FAMILY_RETEST_RENEWAL_DECAY)
        )

    @staticmethod
    def credible_family_retest(
        *,
        selected_option_id: str | None,
        confidence: float | None,
        response_ms: int | None,
        hint_count: int,
    ) -> bool:
        """Whether immutable response metadata can support a renewal bonus.

        Credibility is intentionally outcome-symmetric. A deliberate, unguided
        wrong answer is still fresh signed diagnostic evidence; correctness
        determines the likelihood direction, not whether the observation is
        allowed to renew.
        """

        return bool(
            selected_option_id is not None
            and hint_count == 0
            and confidence is not None
            and confidence >= MIN_CREDIBLE_CORRECT_CONFIDENCE
            and response_ms is not None
            and response_ms >= MIN_CREDIBLE_RETEST_RESPONSE_MS
        )

    @classmethod
    def spacing_aware_family_evidence_power(
        cls,
        *,
        prior_records: Iterable[FamilyResponseRecord],
        occurred_at: datetime,
        credible: bool,
    ) -> FamilyEvidencePower:
        """Return spacing-aware evidence power from immutable family history.

        Stream order, rather than timestamps, defines response order.
        Timestamps only decide whether a credible response earns a renewal.
        Every exposure, including a noncredible one, advances the spacing
        anchor; this prevents rapid low-quality exposures from being hidden
        between two nominally spaced observations. A naive timestamp fails
        closed. A duplicate or out-of-order timestamp remains an ordinary
        correlated repeat and cannot move the spacing anchor backwards.
        """

        records = tuple(prior_records)
        for index, record in enumerate((*records, FamilyResponseRecord(
            occurred_at=occurred_at,
            credible=credible,
            renewal_eligible=True,
        ))):
            if not isinstance(record, FamilyResponseRecord):
                raise ValueError(
                    "Family response history must contain FamilyResponseRecord values."
                )
            timestamp = record.occurred_at
            if (
                not isinstance(timestamp, datetime)
                or timestamp.tzinfo is None
                or timestamp.utcoffset() is None
            ):
                raise ValueError(
                    f"Family response timestamp {index} must be timezone-aware."
                )
            if type(record.credible) is not bool:
                raise ValueError(
                    f"Family response credibility {index} must be boolean."
                )
            if type(record.renewal_eligible) is not bool:
                raise ValueError(
                    f"Family response renewal eligibility {index} must be boolean."
                )

        renewal_count = 0
        latest_seen_at: datetime | None = None
        current_renewal_power = 0.0
        current_renewal_index: int | None = None
        all_records = (
            *records,
            FamilyResponseRecord(
                occurred_at=occurred_at,
                credible=credible,
                renewal_eligible=True,
            ),
        )
        for index, record in enumerate(all_records):
            timestamp = record.occurred_at
            is_current = index == len(records)
            genuinely_spaced = bool(
                latest_seen_at is not None
                and timestamp > latest_seen_at
                and timestamp - latest_seen_at
                >= MIN_FAMILY_RETEST_SPACING
            )
            if (
                index > 0
                and record.credible
                and record.renewal_eligible
                and genuinely_spaced
            ):
                bonus = (
                    FAMILY_RETEST_RENEWAL_HEAD
                    * FAMILY_RETEST_RENEWAL_DECAY**renewal_count
                )
                renewal_count += 1
                if is_current:
                    current_renewal_power = bonus
                    current_renewal_index = renewal_count
            if latest_seen_at is None or timestamp > latest_seen_at:
                latest_seen_at = timestamp

        base_power = cls.family_dependence_discount(len(records))
        power = base_power + current_renewal_power
        if not 0.0 <= power <= 1.0:
            raise ValueError("Spacing-aware family evidence power left [0, 1].")
        return FamilyEvidencePower(
            power=power,
            base_power=base_power,
            renewal_power=current_renewal_power,
            renewal_index=current_renewal_index,
        )

    @classmethod
    def _family_response_records(
        cls,
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        family_id: str,
        event_id: str,
        now: datetime,
        allow_missing_event: bool = False,
        match_equivalent_families: bool = True,
    ) -> tuple[FamilyResponseRecord, ...] | None:
        """Reconstruct prior family facts through the current event boundary."""

        boundary = connection.execute(
            """SELECT stream_id, stream_version, learner_id, occurred_at
               FROM events
               WHERE event_id = ? AND event_type = 'ResponseSubmitted'""",
            (event_id,),
        ).fetchone()
        expected_stream_id = f"learner:{learner_id}"
        if boundary is None and allow_missing_event:
            # Conservative compatibility path for isolated reducer tests and
            # callers that supply an explicit historical count without an
            # event store. No history means no renewal can be claimed.
            return None
        if (
            boundary is None
            or boundary["learner_id"] != learner_id
            or boundary["stream_id"] != expected_stream_id
        ):
            raise ValueError(
                "Spacing-aware family evidence requires the current learner's "
                "ResponseSubmitted event."
            )
        try:
            boundary_at = datetime.fromisoformat(boundary["occurred_at"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "The current family response event has an invalid timestamp."
            ) from exc
        if (
            boundary_at.tzinfo is None
            or boundary_at.utcoffset() is None
            or now.tzinfo is None
            or now.utcoffset() is None
        ):
            raise ValueError(
                "Spacing-aware family evidence requires timezone-aware timestamps."
            )
        if boundary_at != now:
            raise ValueError(
                "The learner update timestamp does not match its immutable "
                "response event."
            )
        family_predicate = (
            "tsq_canonical_family(attempt.family_id) = ?"
            if match_equivalent_families
            else "attempt.family_id = ?"
        )
        rows = connection.execute(
            f"""SELECT event.occurred_at, event.metadata_json,
                      attempt.selected_option_id,
                      attempt.confidence, attempt.response_ms,
                      attempt.hint_count
               FROM attempts attempt
               JOIN events event ON event.event_id = attempt.event_id
               WHERE attempt.learner_id = ?
                 AND {family_predicate}
                 AND event.stream_id = ?
                 AND event.event_type = 'ResponseSubmitted'
                 AND event.stream_version < ?
               ORDER BY event.stream_version""",
            (
                learner_id,
                family_id,
                expected_stream_id,
                boundary["stream_version"],
            ),
        ).fetchall()
        result: list[FamilyResponseRecord] = []
        for row in rows:
            try:
                occurred_at = datetime.fromisoformat(row["occurred_at"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Family response history contains an invalid timestamp."
                ) from exc
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Family response history contains invalid event metadata."
                ) from exc
            if type(metadata) is not dict:
                raise ValueError(
                    "Family response history metadata must be an object."
                )
            event_model_version = metadata.get("learner_model_version")
            if event_model_version not in SUPPORTED_MODEL_VERSIONS:
                raise ValueError(
                    "Family response history names an unsupported learner model."
                )
            result.append(
                FamilyResponseRecord(
                    occurred_at=occurred_at,
                    credible=cls.credible_family_retest(
                        selected_option_id=row["selected_option_id"],
                        confidence=row["confidence"],
                        response_ms=row["response_ms"],
                        hint_count=row["hint_count"],
                    ),
                    renewal_eligible=(
                        event_model_version
                        in SPACING_AWARE_FAMILY_MODEL_VERSIONS
                    ),
                )
            )
        return tuple(result)

    def potential_family_evidence_powers(
        self,
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        family_ids: Iterable[str],
        now: datetime,
    ) -> dict[str, FamilyEvidencePower]:
        """Return credible-current spacing-aware power for each family.

        Selection knows immutable history and the spacing clock, but cannot
        know whether the learner's future response will be observable and
        unguided. This reports the power the reducer would grant *if* that
        response is credible. The reducer still classifies actual metadata
        independently and fails closed.

        Historical v6 responses anchor spacing without spending a renewal
        budget, exactly as they do during projection replay.
        """

        if self.model_version not in SPACING_AWARE_FAMILY_MODEL_VERSIONS:
            raise ValueError(
                "Potential family power requires a spacing-aware model."
            )
        if (
            not isinstance(now, datetime)
            or now.tzinfo is None
            or now.utcoffset() is None
        ):
            raise ValueError(
                "Potential family evidence requires a timezone-aware timestamp."
            )
        requested = tuple(family_ids)
        if any(
            type(family_id) is not str or not family_id.strip()
            for family_id in requested
        ):
            raise ValueError("Potential family IDs must be non-blank strings.")
        canonical_families = (
            self.model_version in CANONICAL_FAMILY_MODEL_VERSIONS
        )
        normalized = tuple(
            sorted(
                {
                    canonical_family_label(family_id)
                    if canonical_families
                    else family_id
                    for family_id in requested
                }
            )
        )
        if not normalized:
            return {}

        placeholders = ",".join("?" for _ in normalized)
        expected_stream_id = f"learner:{learner_id}"
        register_family_sql_functions(connection)
        family_expression = (
            "tsq_canonical_family(attempt.family_id)"
            if canonical_families
            else "attempt.family_id"
        )
        rows = connection.execute(
            f"""SELECT {family_expression} AS family_id,
                       attempt.selected_option_id,
                       attempt.confidence, attempt.response_ms,
                       attempt.hint_count, event.occurred_at,
                       event.metadata_json, event.stream_id,
                       event.learner_id
                FROM attempts attempt
                JOIN events event ON event.event_id = attempt.event_id
                WHERE attempt.learner_id = ?
                  AND {family_expression} IN ({placeholders})
                  AND event.event_type = 'ResponseSubmitted'
                ORDER BY event.stream_version""",
            (learner_id, *normalized),
        ).fetchall()
        records: dict[str, list[FamilyResponseRecord]] = {
            family_id: [] for family_id in normalized
        }
        for row in rows:
            if (
                row["stream_id"] != expected_stream_id
                or row["learner_id"] != learner_id
                or row["family_id"] not in records
            ):
                raise ValueError(
                    "Potential family history crossed its learner stream."
                )
            try:
                occurred_at = datetime.fromisoformat(row["occurred_at"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Potential family history contains an invalid timestamp."
                ) from exc
            try:
                metadata = json.loads(row["metadata_json"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Potential family history contains invalid event metadata."
                ) from exc
            if type(metadata) is not dict:
                raise ValueError(
                    "Potential family history metadata must be an object."
                )
            event_model_version = metadata.get("learner_model_version")
            if event_model_version not in SUPPORTED_MODEL_VERSIONS:
                raise ValueError(
                    "Potential family history names an unsupported learner model."
                )
            records[row["family_id"]].append(
                FamilyResponseRecord(
                    occurred_at=occurred_at,
                    credible=self.credible_family_retest(
                        selected_option_id=row["selected_option_id"],
                        confidence=row["confidence"],
                        response_ms=row["response_ms"],
                        hint_count=row["hint_count"],
                    ),
                    renewal_eligible=(
                        event_model_version
                        in SPACING_AWARE_FAMILY_MODEL_VERSIONS
                    ),
                )
            )

        return {
            family_id: self.spacing_aware_family_evidence_power(
                prior_records=family_records,
                occurred_at=now,
                credible=True,
            )
            for family_id, family_records in records.items()
        }

    def potential_family_evidence_power(
        self,
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        family_id: str,
        now: datetime,
    ) -> FamilyEvidencePower:
        """Return one family's credible-current spacing-aware power."""

        return self.potential_family_evidence_powers(
            connection,
            learner_id=learner_id,
            family_ids=(family_id,),
            now=now,
        )[family_id]

    @staticmethod
    def required_family_spacing(state: SkillState) -> timedelta:
        """Minimum interval before one family can be independent evidence again."""
        return timedelta(
            hours=max(
                24.0,
                min(24.0 * 30.0, state.stability_hours * 0.50),
            )
        )

    def response_family_id(self, question: Question) -> str:
        """Return the family label committed by this immutable model version.

        Models through v8 wrote the item's published family label.  v9 is the
        first reducer that treats reviewed aliases as one evidence family and
        therefore writes the canonical label.  Keeping this decision on the
        model boundary lets replay reconstruct either contract exactly.
        """

        if self.model_version in CANONICAL_FAMILY_MODEL_VERSIONS:
            return question.family_id
        if question.published_family_id is None:
            raise ValueError("Question lacks its published family identity.")
        return question.published_family_id

    def update_from_response(
        self,
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        question: Question,
        selected_option: Option | None,
        confidence: float | None,
        hint_count: int,
        feedback_shown: bool,
        evidence_weight_override: float | None,
        event_id: str,
        now: datetime,
        response_ms: int | None = None,
        prior_family_attempts_override: int | None = None,
        family_id_override: str | None = None,
        misconception_algorithm: str = LEGACY_MISCONCEPTION_ALGORITHM,
    ) -> tuple[dict[str, SkillState], list[dict[str, object]]]:
        if misconception_algorithm not in SUPPORTED_MISCONCEPTION_ALGORITHMS:
            raise ValueError(
                "Unsupported misconception inference algorithm: "
                f"{misconception_algorithm}"
            )
        register_family_sql_functions(connection)
        response_family_id = self.response_family_id(question)
        match_equivalent_families = (
            self.model_version in CANONICAL_FAMILY_MODEL_VERSIONS
        )
        if family_id_override is not None:
            if (
                type(family_id_override) is not str
                or not family_id_override.strip()
                or evidence_family_id(question.id, family_id_override)
                != question.family_id
            ):
                raise ValueError(
                    "Family override must be the question's published or "
                    "reviewed evidence family."
                )
            expected_family_id = self.response_family_id(question)
            if self.model_version in CANONICAL_FAMILY_MODEL_VERSIONS:
                response_family_id = canonical_family_label(
                    family_id_override
                )
            else:
                response_family_id = family_id_override
            if response_family_id != expected_family_id:
                raise ValueError(
                    "Family override does not match the event model's family "
                    "identity contract."
                )
        projection_family_predicate = (
            "tsq_canonical_family(family_id) = ?"
            if match_equivalent_families
            else "family_id = ?"
        )
        concept_rows = connection.execute(
            "SELECT * FROM concepts WHERE id IN ({})".format(
                ",".join("?" for _ in question.concepts)
            ),
            tuple(mapping.concept_id for mapping in question.concepts),
        ).fetchall()
        concepts = {
            row["id"]: Concept(
                row["id"], row["name"], row["description"], row["domain"], row["prior_mastery"]
            )
            for row in concept_rows
        }
        stored_rows = connection.execute(
            "SELECT * FROM skill_states WHERE learner_id = ? AND concept_id IN ({})".format(
                ",".join("?" for _ in question.concepts)
            ),
            (learner_id, *(mapping.concept_id for mapping in question.concepts)),
        ).fetchall()
        stored = {
            row["concept_id"]: SkillState(
                learner_id=row["learner_id"],
                concept_id=row["concept_id"],
                mean=row["mean"],
                variance=row["variance"],
                stability_hours=row["stability_hours"],
                exposures=row["exposures"],
                last_seen_at=datetime.fromisoformat(row["last_seen_at"]) if row["last_seen_at"] else None,
                next_review_at=datetime.fromisoformat(row["next_review_at"]) if row["next_review_at"] else None,
                evidence_mass=row["evidence_mass"],
            )
            for row in stored_rows
        }
        projected = self.states_for_question(learner_id, question, concepts, stored, now)
        projected_objective: ObjectiveState | None = None
        if (
            self.model_version in OBJECTIVE_MODEL_VERSIONS
            and question.objective is not None
        ):
            objective_state = Database.load_objective_state(
                connection, learner_id, question.objective.id
            )
            if objective_state is None:
                objective_state = self.initial_objective_state(
                    learner_id, question.objective
                )
            projected_objective = self.project_objective_state(
                objective_state, question.objective, now
            )
        p = self.predict_correct(
            question, projected, objective_state=projected_objective
        )
        is_correct = bool(selected_option and selected_option.correct)
        y = 1.0 if is_correct else 0.0
        weights = self.evidence_weights(question)
        if projected_objective is not None:
            theta = projected_objective.mean
        else:
            theta = sum(
                weight * projected[concept_id].mean
                for concept_id, weight in weights.items()
            )
        logistic = sigmoid(question.discrimination * (theta - question.difficulty))
        dp_dtheta = (
            (1.0 - SESSION_LAPSE_RATE)
            * (1.0 - question.guess_rate - question.slip_rate)
            * question.discrimination
            * logistic
            * (1.0 - logistic)
        )
        score_theta = (y - p) * dp_dtheta / max(1e-6, p * (1.0 - p))
        if prior_family_attempts_override is not None:
            if (
                type(prior_family_attempts_override) is not int
                or prior_family_attempts_override < 0
            ):
                raise ValueError(
                    "Prior family-attempt override must be a non-negative integer."
                )
            prior_family_attempts = prior_family_attempts_override
        else:
            attempt_family_predicate = (
                "tsq_canonical_family(family_id) = ?"
                if match_equivalent_families
                else "family_id = ?"
            )
            prior_family_attempts = connection.execute(
                f"""SELECT COUNT(*) AS n FROM attempts
                    WHERE learner_id = ? AND {attempt_family_predicate}
                      AND event_id <> ?""",
                (learner_id, response_family_id, event_id),
            ).fetchone()["n"]

        if self.model_version in SPACING_AWARE_FAMILY_MODEL_VERSIONS:
            prior_family_records = self._family_response_records(
                connection,
                learner_id=learner_id,
                family_id=response_family_id,
                event_id=event_id,
                now=now,
                allow_missing_event=(
                    prior_family_attempts_override is not None
                ),
                match_equivalent_families=match_equivalent_families,
            )
            if prior_family_records is None:
                base_power = self.family_dependence_discount(
                    prior_family_attempts
                )
                family_power = FamilyEvidencePower(
                    power=base_power,
                    base_power=base_power,
                    renewal_power=0.0,
                    renewal_index=None,
                )
            else:
                if (
                    prior_family_attempts_override is not None
                    and prior_family_attempts != len(prior_family_records)
                ):
                    raise ValueError(
                        "Replay family-attempt count does not match immutable "
                        "spacing-aware family history."
                    )
                prior_family_attempts = len(prior_family_records)
                family_power = self.spacing_aware_family_evidence_power(
                    prior_records=prior_family_records,
                    occurred_at=now,
                    credible=self.credible_family_retest(
                        selected_option_id=(
                            selected_option.id
                            if selected_option is not None
                            else None
                        ),
                        confidence=confidence,
                        response_ms=response_ms,
                        hint_count=hint_count,
                    ),
                )
        else:
            base_power = self.family_dependence_discount(
                prior_family_attempts
            )
            family_power = FamilyEvidencePower(
                power=base_power,
                base_power=base_power,
                renewal_power=0.0,
                renewal_index=None,
            )
        dependence_discount = family_power.power
        if self.model_version in OBJECTIVE_GRID_MODEL_VERSIONS:
            confidence_discount = (
                MISSING_CONFIDENCE_DISCOUNT
                if confidence is None
                else (
                    1.0
                    if confidence >= 0.50
                    else max(0.10, confidence / 0.50)
                )
            )
            response_discount = (
                MISSING_RESPONSE_TIME_DISCOUNT
                if response_ms is None
                else (
                    1.0
                    if response_ms >= 500
                    else (0.55 if response_ms >= 250 else 0.15)
                )
            )
            selection_discount = (
                ABSTENTION_EVIDENCE_DISCOUNT
                if selected_option is None
                else 1.0
            )
        else:
            # Historical event replay must retain the v3-v5 evidence contract.
            confidence_discount = (
                1.0
                if confidence is None or confidence >= 0.50
                else max(0.10, confidence / 0.50)
            )
            response_discount = (
                1.0
                if response_ms is None or response_ms >= 500
                else (0.55 if response_ms >= 250 else 0.15)
            )
            selection_discount = 1.0
        base_evidence_weight = (
            question.status.evidence_weight
            if evidence_weight_override is None
            else evidence_weight_override
        )
        hint_discount = 1.0 if hint_count == 0 else 0.20
        evidence_weight = (
            base_evidence_weight
            * hint_discount
            * dependence_discount
            * confidence_discount
            * response_discount
            * selection_discount
        )
        # Feedback is a possible acquisition transition, not proof of knowing.
        # It is bounded by item trust and the same-family tail, and hints reduce
        # the incremental value because much of the explanation was pre-exposed.
        feedback_weight = (
            base_evidence_weight
            * dependence_discount
            * (1.0 if hint_count == 0 else 0.35)
        )
        response_class = classify_response_for_model(
            model_version=self.model_version,
            correct=is_correct,
            selected_option_id=(
                selected_option.id if selected_option is not None else None
            ),
            selected_misconception_id=(
                selected_option.misconception_id
                if selected_option is not None
                else None
            ),
            confidence=confidence,
            response_ms=response_ms,
            hint_count=hint_count,
        )
        certifying_retrieval = response_class.certifies_retrieval
        primary_mapping = next(
            mapping for mapping in question.concepts if mapping.role is ConceptRole.PRIMARY
        )
        if projected_objective is not None:
            prior_primary_family = connection.execute(
                f"""SELECT MIN(first_unguided_correct_at)
                                AS first_unguided_correct_at,
                           MAX(last_unguided_correct_at)
                                AS last_unguided_correct_at,
                           MIN(delayed_unguided_correct_at)
                                AS delayed_unguided_correct_at
                    FROM learner_objective_families
                    WHERE learner_id = ? AND objective_id = ?
                      AND {projection_family_predicate}""",
                (learner_id, question.objective_id, response_family_id),
            ).fetchone()
        else:
            prior_primary_family = connection.execute(
                f"""SELECT MIN(first_unguided_correct_at)
                                AS first_unguided_correct_at,
                           MAX(last_unguided_correct_at)
                                AS last_unguided_correct_at,
                           MIN(delayed_unguided_correct_at)
                                AS delayed_unguided_correct_at
                    FROM learner_skill_families
                    WHERE learner_id = ? AND concept_id = ?
                      AND {projection_family_predicate}""",
                (
                    learner_id,
                    primary_mapping.concept_id,
                    response_family_id,
                ),
            ).fetchone()
        independent_retrieval = (
            prior_primary_family is None
            or prior_primary_family["last_unguided_correct_at"] is None
        )
        if not independent_retrieval:
            last_at = datetime.fromisoformat(prior_primary_family["last_unguided_correct_at"])
            primary_state = (
                projected_objective
                if projected_objective is not None
                else projected[primary_mapping.concept_id]
            )
            required_spacing = self.required_family_spacing(primary_state)
            independent_retrieval = (now - last_at) >= required_spacing
        new_states: dict[str, SkillState] = {}
        changes: list[dict[str, object]] = []

        for mapping in question.concepts:
            if mapping.concept_id not in weights:
                continue
            if projected_objective is not None:
                # The fine objective replaces—not duplicates—every broad
                # concept mapping as this response's scored latent dimension.
                # In particular, a wrong answer cannot identify whether a
                # prerequisite, contrast, or transfer concept caused it.
                continue
            effective_weight = weights[mapping.concept_id]
            before = projected[mapping.concept_id]
            fisher = (effective_weight * dp_dtheta) ** 2 / max(1e-6, p * (1.0 - p))
            new_variance = max(
                MIN_POSTERIOR_VARIANCE,
                1.0 / (1.0 / before.variance + evidence_weight * fisher),
            )
            evidence_delta = new_variance * evidence_weight * effective_weight * score_theta
            post_evidence_mean = before.mean + evidence_delta

            learning_delta = 0.0
            if feedback_shown and (
                is_correct or self.model_version == LEGACY_MODEL_VERSION
            ):
                transition_rate = 0.025 if is_correct else 0.085
                learning_delta = (
                    transition_rate
                    * feedback_weight
                    * effective_weight
                    * (1.0 - sigmoid(post_evidence_mean))
                )
            new_mean = max(-6.0, min(6.0, post_evidence_mean + learning_delta))
            # Learning after feedback is uncertain; widen instead of treating it as scored evidence.
            new_variance = min(
                MAX_POSTERIOR_VARIANCE,
                new_variance
                + (0.02 * feedback_weight if feedback_shown else 0.0),
            )
            if not is_correct and self.model_version != LEGACY_MODEL_VERSION:
                # Feedback proves exposure, not acquisition. A wrong retrieval
                # must not improve either published competence measure merely
                # because its explanation was displayed. Returning at most the
                # response's precision gain preserves uncertainty without
                # turning feedback into positive evidence.
                new_variance = min(before.variance, new_variance)
                mastery_boundary = logit(MASTERY_THRESHOLD)
                mastery_mean_cap = mastery_boundary + sqrt(
                    new_variance / before.variance
                ) * (before.mean - mastery_boundary)
                competence_mean_cap = sqrt(
                    (1.0 + pi * new_variance / 8.0)
                    / (1.0 + pi * before.variance / 8.0)
                ) * before.mean
                new_mean = min(
                    new_mean,
                    before.mean,
                    mastery_mean_cap,
                    competence_mean_cap,
                )
                new_mean = max(-6.0, min(6.0, new_mean))

            overdue_factor = 0.0
            if before.last_seen_at:
                elapsed = max(0.0, (now - before.last_seen_at).total_seconds() / 3600.0)
                overdue_factor = min(1.0, elapsed / max(12.0, before.stability_hours))
            if (
                mapping.role is ConceptRole.PRIMARY
                and certifying_retrieval
                and independent_retrieval
            ):
                stability = min(24.0 * 365.0, before.stability_hours * (1.30 + 0.25 * overdue_factor))
            elif mapping.role is not ConceptRole.PRIMARY or is_correct:
                stability = before.stability_hours
            else:
                stability = max(12.0, before.stability_hours * 0.78)
            # Stability is a half-life. Review before a full half-life has elapsed,
            # when roughly 65% of the learned-above-prior signal remains.
            next_review = now + timedelta(hours=stability * 0.62)
            after = SkillState(
                learner_id=learner_id,
                concept_id=mapping.concept_id,
                mean=new_mean,
                variance=new_variance,
                stability_hours=stability,
                exposures=before.exposures + 1,
                last_seen_at=now,
                next_review_at=next_review,
                evidence_mass=before.evidence_mass + evidence_weight * effective_weight,
            )
            new_states[mapping.concept_id] = after
            connection.execute(
                """INSERT INTO skill_states(
                       learner_id, concept_id, mean, variance, stability_hours,
                       exposures, last_seen_at, next_review_at, evidence_mass,
                       as_of_event_id, model_version
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(learner_id, concept_id) DO UPDATE SET
                       mean=excluded.mean, variance=excluded.variance,
                       stability_hours=excluded.stability_hours,
                       exposures=excluded.exposures, last_seen_at=excluded.last_seen_at,
                       next_review_at=excluded.next_review_at,
                       evidence_mass=excluded.evidence_mass,
                       as_of_event_id=excluded.as_of_event_id,
                       model_version=excluded.model_version""",
                (
                    learner_id,
                    mapping.concept_id,
                    after.mean,
                    after.variance,
                    after.stability_hours,
                    after.exposures,
                    to_timestamp(after.last_seen_at),
                    to_timestamp(after.next_review_at),
                    after.evidence_mass,
                    event_id,
                    self.model_version,
                ),
            )
            if certifying_retrieval and mapping.role is ConceptRole.PRIMARY:
                family_evidence = connection.execute(
                    f"""SELECT MIN(first_unguided_correct_at)
                                    AS first_unguided_correct_at,
                               MAX(last_unguided_correct_at)
                                    AS last_unguided_correct_at,
                               MIN(delayed_unguided_correct_at)
                                    AS delayed_unguided_correct_at
                        FROM learner_skill_families
                        WHERE learner_id = ? AND concept_id = ?
                          AND {projection_family_predicate}""",
                    (learner_id, mapping.concept_id, response_family_id),
                ).fetchone()
                delayed_at = None
                first_at = to_timestamp(now)
                if (
                    family_evidence is not None
                    and family_evidence["last_unguided_correct_at"] is not None
                ):
                    first_at = family_evidence[
                        "first_unguided_correct_at"
                    ]
                    delayed_at = family_evidence["delayed_unguided_correct_at"]
                    last_at = datetime.fromisoformat(
                        family_evidence["last_unguided_correct_at"]
                    )
                    if (
                        not delayed_at
                        and independent_retrieval
                        and (now - last_at) >= timedelta(hours=24)
                    ):
                        delayed_at = to_timestamp(now)
                connection.execute(
                    """INSERT INTO learner_skill_families(
                           learner_id, concept_id, family_id, kind,
                           first_unguided_correct_at, last_unguided_correct_at,
                           delayed_unguided_correct_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(learner_id, concept_id, family_id) DO UPDATE SET
                           kind=excluded.kind,
                           last_unguided_correct_at=excluded.last_unguided_correct_at,
                           delayed_unguided_correct_at=COALESCE(
                               learner_skill_families.delayed_unguided_correct_at,
                               excluded.delayed_unguided_correct_at
                           )""",
                    (
                        learner_id,
                        mapping.concept_id,
                        response_family_id,
                        question.kind.value,
                        first_at,
                        to_timestamp(now),
                        delayed_at,
                    ),
                )
            concept_change: dict[str, object] = {
                "concept_id": mapping.concept_id,
                "prior_mastery": round(before.mastery_probability, 5),
                "posterior_mastery": round(after.mastery_probability, 5),
                "prior_expected_competence": round(before.expected_competence, 5),
                "posterior_expected_competence": round(after.expected_competence, 5),
                "prior_variance": round(before.variance, 5),
                "posterior_variance": round(after.variance, 5),
                "evidence_delta": round(evidence_delta, 5),
                "feedback_transition": round(learning_delta, 5),
                "effective_evidence_weight": round(evidence_weight, 5),
                "retrieval_certified": certifying_retrieval,
                "stability_hours": round(stability, 2),
            }
            if self.model_version in SPACING_AWARE_FAMILY_MODEL_VERSIONS:
                concept_change.update(
                    {
                        "family_evidence_power": round(
                            family_power.power, 8
                        ),
                        "family_retest_renewed": (
                            family_power.renewal_index is not None
                        ),
                        "family_retest_renewal_index": (
                            family_power.renewal_index
                        ),
                    }
                )
            changes.append(concept_change)

        if projected_objective is not None:
            # Objective-aware items are calibrated against one objective
            # latent, so its derivative and evidence mass use unit weight even
            # when the question's graph mappings distribute retrieval weight.
            effective_weight = 1.0
            before = projected_objective
            after_posterior: ObjectivePosterior | None = None
            if self.model_version in OBJECTIVE_GRID_MODEL_VERSIONS:
                if question.objective is None:
                    raise ValueError(
                        "Objective-aware evidence requires objective metadata."
                    )
                before_posterior = self._migrate_objective_posterior(
                    before, question.objective
                )
                evidence_posterior = before_posterior
                if evidence_weight > 0.0:
                    evidence_posterior = before_posterior.with_observation(
                        LikelihoodObservation(
                            observation_id=event_id,
                            family_id=response_family_id,
                            difficulty=question.difficulty,
                            discrimination=question.discrimination,
                            guess_rate=question.guess_rate,
                            slip_rate=question.slip_rate,
                            option_count=len(question.options),
                            correct=is_correct,
                            evidence_power=evidence_weight,
                        )
                    )
                evidence_metrics = evidence_posterior.metrics()
                evidence_delta = evidence_metrics.mean - before.mean
                after_posterior = evidence_posterior
                if feedback_shown and is_correct and feedback_weight > 0.0:
                    after_posterior = after_posterior.apply_correct_feedback(
                        feedback_weight
                    )
                after_metrics = after_posterior.metrics()
                learning_delta = after_metrics.mean - evidence_metrics.mean
                new_mean = after_metrics.mean
                new_variance = after_metrics.variance
            else:
                fisher = (effective_weight * dp_dtheta) ** 2 / max(
                    1e-6, p * (1.0 - p)
                )
                new_variance = max(
                    MIN_POSTERIOR_VARIANCE,
                    1.0 / (1.0 / before.variance + evidence_weight * fisher),
                )
                evidence_delta = (
                    new_variance
                    * evidence_weight
                    * effective_weight
                    * score_theta
                )
                post_evidence_mean = before.mean + evidence_delta
                learning_delta = 0.0
                if feedback_shown and is_correct:
                    learning_delta = (
                        0.025
                        * feedback_weight
                        * effective_weight
                        * (1.0 - sigmoid(post_evidence_mean))
                    )
                new_mean = max(
                    -6.0, min(6.0, post_evidence_mean + learning_delta)
                )
                new_variance = min(
                    MAX_POSTERIOR_VARIANCE,
                    new_variance
                    + (0.02 * feedback_weight if feedback_shown else 0.0),
                )
                if not is_correct:
                    new_variance = min(before.variance, new_variance)
                    mastery_boundary = logit(MASTERY_THRESHOLD)
                    mastery_mean_cap = mastery_boundary + sqrt(
                        new_variance / before.variance
                    ) * (before.mean - mastery_boundary)
                    competence_mean_cap = sqrt(
                        (1.0 + pi * new_variance / 8.0)
                        / (1.0 + pi * before.variance / 8.0)
                    ) * before.mean
                    new_mean = min(
                        new_mean,
                        before.mean,
                        mastery_mean_cap,
                        competence_mean_cap,
                    )
                    new_mean = max(-6.0, min(6.0, new_mean))

            overdue_factor = 0.0
            if before.last_seen_at:
                elapsed = max(
                    0.0,
                    (now - before.last_seen_at).total_seconds() / 3600.0,
                )
                overdue_factor = min(
                    1.0, elapsed / max(12.0, before.stability_hours)
                )
            if certifying_retrieval and independent_retrieval:
                stability = min(
                    24.0 * 365.0,
                    before.stability_hours
                    * (1.30 + 0.25 * overdue_factor),
                )
            elif is_correct:
                stability = before.stability_hours
            else:
                stability = max(12.0, before.stability_hours * 0.78)
            next_review = now + timedelta(hours=stability * 0.62)
            after_objective = ObjectiveState(
                learner_id=learner_id,
                objective_id=question.objective_id,
                mean=new_mean,
                variance=new_variance,
                stability_hours=stability,
                exposures=before.exposures + 1,
                last_seen_at=now,
                next_review_at=next_review,
                evidence_mass=(
                    after_posterior.evidence_mass
                    if after_posterior is not None
                    else before.evidence_mass
                    + evidence_weight * effective_weight
                ),
                posterior=after_posterior,
                model_version=self.model_version,
            )
            connection.execute(
                """INSERT INTO objective_states(
                       learner_id, objective_id, mean, variance,
                       stability_hours, exposures, last_seen_at,
                       next_review_at, evidence_mass, as_of_event_id,
                       model_version
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(learner_id, objective_id) DO UPDATE SET
                       mean=excluded.mean, variance=excluded.variance,
                       stability_hours=excluded.stability_hours,
                       exposures=excluded.exposures,
                       last_seen_at=excluded.last_seen_at,
                       next_review_at=excluded.next_review_at,
                       evidence_mass=excluded.evidence_mass,
                       as_of_event_id=excluded.as_of_event_id,
                       model_version=excluded.model_version""",
                (
                    learner_id,
                    question.objective_id,
                    after_objective.mean,
                    after_objective.variance,
                    after_objective.stability_hours,
                    after_objective.exposures,
                    to_timestamp(after_objective.last_seen_at),
                    to_timestamp(after_objective.next_review_at),
                    after_objective.evidence_mass,
                    event_id,
                    self.model_version,
                ),
            )
            if after_posterior is not None:
                Database.upsert_objective_grid_state(
                    connection,
                    learner_id=learner_id,
                    objective_id=question.objective_id,
                    posterior=after_posterior,
                    as_of_event_id=event_id,
                    model_version=self.model_version,
                )
            if certifying_retrieval:
                family_evidence = connection.execute(
                    f"""SELECT MIN(first_unguided_correct_at)
                                    AS first_unguided_correct_at,
                               MAX(last_unguided_correct_at)
                                    AS last_unguided_correct_at,
                               MIN(delayed_unguided_correct_at)
                                    AS delayed_unguided_correct_at
                        FROM learner_objective_families
                        WHERE learner_id = ? AND objective_id = ?
                          AND {projection_family_predicate}""",
                    (learner_id, question.objective_id, response_family_id),
                ).fetchone()
                delayed_at = None
                first_at = to_timestamp(now)
                if (
                    family_evidence is not None
                    and family_evidence["last_unguided_correct_at"] is not None
                ):
                    first_at = family_evidence[
                        "first_unguided_correct_at"
                    ]
                    delayed_at = family_evidence[
                        "delayed_unguided_correct_at"
                    ]
                    last_at = datetime.fromisoformat(
                        family_evidence["last_unguided_correct_at"]
                    )
                    if (
                        not delayed_at
                        and independent_retrieval
                        and (now - last_at) >= timedelta(hours=24)
                    ):
                        delayed_at = to_timestamp(now)
                connection.execute(
                    """INSERT INTO learner_objective_families(
                           learner_id, objective_id, family_id, kind,
                           first_unguided_correct_at,
                           last_unguided_correct_at,
                           delayed_unguided_correct_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(learner_id, objective_id, family_id)
                       DO UPDATE SET
                           kind=excluded.kind,
                           last_unguided_correct_at=
                               excluded.last_unguided_correct_at,
                           delayed_unguided_correct_at=COALESCE(
                               learner_objective_families.delayed_unguided_correct_at,
                               excluded.delayed_unguided_correct_at
                           )""",
                    (
                        learner_id,
                        question.objective_id,
                        response_family_id,
                        question.kind.value,
                        first_at,
                        to_timestamp(now),
                        delayed_at,
                    ),
                )
            objective_change: dict[str, object] = {
                "objective_id": question.objective_id,
                "prior_mastery": round(before.mastery_probability, 5),
                "posterior_mastery": round(
                    after_objective.mastery_probability, 5
                ),
                "prior_expected_competence": round(
                    before.expected_competence, 5
                ),
                "posterior_expected_competence": round(
                    after_objective.expected_competence, 5
                ),
                "prior_variance": round(before.variance, 5),
                "posterior_variance": round(
                    after_objective.variance, 5
                ),
                "evidence_delta": round(evidence_delta, 5),
                "feedback_transition": round(learning_delta, 5),
                "effective_evidence_weight": round(evidence_weight, 5),
                "retrieval_certified": certifying_retrieval,
                "stability_hours": round(stability, 2),
            }
            if self.model_version in SPACING_AWARE_FAMILY_MODEL_VERSIONS:
                objective_change.update(
                    {
                        "family_evidence_power": round(
                            family_power.power, 8
                        ),
                        "family_retest_renewed": (
                            family_power.renewal_index is not None
                        ),
                        "family_retest_renewal_index": (
                            family_power.renewal_index
                        ),
                    }
                )
            if after_posterior is not None:
                objective_change.update(
                    {
                        "posterior_digest": after_posterior.digest,
                        "mastery_probability_error_bound": round(
                            after_objective.mastery_probability_error_bound, 8
                        ),
                        "acquisition_mass": round(
                            after_objective.acquisition_mass, 5
                        ),
                    }
                )
            changes.append(objective_change)

        self._update_misconceptions(
            connection,
            learner_id=learner_id,
            question=question,
            selected_option=selected_option,
            event_id=event_id,
            now=now,
            evidence_weight=evidence_weight,
            confidence=confidence,
            misconception_algorithm=misconception_algorithm,
        )
        return new_states, changes

    def _update_misconceptions(
        self,
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        question: Question,
        selected_option: Option | None,
        event_id: str,
        now: datetime,
        evidence_weight: float,
        confidence: float | None,
        misconception_algorithm: str,
    ) -> None:
        referenced_ids = sorted(question.misconception_ids)
        scored_concept_ids = sorted(self.evidence_weights(question))
        if not referenced_ids or not scored_concept_ids:
            return
        misconception_placeholders = ",".join("?" for _ in referenced_ids)
        concept_placeholders = ",".join("?" for _ in scored_concept_ids)
        eligible_rows = connection.execute(
            f"""SELECT id FROM misconceptions
                WHERE id IN ({misconception_placeholders})
                  AND concept_id IN ({concept_placeholders})""",
            (*referenced_ids, *scored_concept_ids),
        ).fetchall()
        misconception_ids = sorted(row["id"] for row in eligible_rows)
        if not misconception_ids:
            return
        if misconception_algorithm == LEGACY_MISCONCEPTION_ALGORITHM:
            self._update_misconceptions_legacy(
                connection,
                learner_id=learner_id,
                question=question,
                selected_option=selected_option,
                event_id=event_id,
                now=now,
                evidence_weight=evidence_weight,
                confidence=confidence,
                misconception_ids=misconception_ids,
            )
            return
        if misconception_algorithm != MISCONCEPTION_ALGORITHM_VERSION:
            raise ValueError(
                "Unsupported misconception inference algorithm: "
                f"{misconception_algorithm}"
            )
        current_event = connection.execute(
            "SELECT metadata_json FROM events WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if current_event is None:
            raise ValueError(
                "Versioned misconception inference requires its "
                "ResponseSubmitted event."
            )
        (
            current_event_algorithm,
            _,
            _,
        ) = self._misconception_event_metadata(
            current_event["metadata_json"], event_id=event_id
        )
        if current_event_algorithm != misconception_algorithm:
            raise ValueError(
                "The requested misconception algorithm does not match the "
                "current response event marker."
            )

        # Rebuild only the hypotheses touched by this question from immutable
        # response/attempt history.  The source database may contain later
        # attempts during projection replay, so the reconstruction is bounded
        # by the current response's learner-stream version rather than by wall
        # time or ``event_id <> current``.
        for misconception_id in misconception_ids:
            reconstruction = self._reconstruct_misconception_belief(
                connection,
                learner_id=learner_id,
                misconception_id=misconception_id,
                through_event_id=event_id,
            )
            if not reconstruction["current_observation_applied"]:
                continue
            connection.execute(
                """INSERT INTO misconception_beliefs(
                       learner_id, misconception_id, log_odds, evidence_count,
                       last_seen_at, as_of_event_id, model_version
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(learner_id, misconception_id) DO UPDATE SET
                       log_odds=excluded.log_odds,
                       evidence_count=excluded.evidence_count,
                       last_seen_at=excluded.last_seen_at,
                       as_of_event_id=excluded.as_of_event_id,
                       model_version=excluded.model_version""",
                (
                    learner_id,
                    misconception_id,
                    reconstruction["log_odds"],
                    reconstruction["evidence_count"],
                    to_timestamp(now),
                    event_id,
                    self.model_version,
                ),
            )

    def _update_misconceptions_legacy(
        self,
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        question: Question,
        selected_option: Option | None,
        event_id: str,
        now: datetime,
        evidence_weight: float,
        confidence: float | None,
        misconception_ids: list[str],
    ) -> None:
        """Apply the exact pre-marker additive contract.

        This path intentionally remains incremental.  Projection replay invokes
        it only for historical response events whose metadata has no explicit
        misconception algorithm marker.
        """
        placeholders = ",".join("?" for _ in misconception_ids)
        rows = connection.execute(
            f"""SELECT * FROM misconception_beliefs
                WHERE learner_id = ? AND misconception_id IN ({placeholders})""",
            (learner_id, *misconception_ids),
        ).fetchall()
        beliefs = {row["misconception_id"]: row for row in rows}
        selected_misconception = selected_option.misconception_id if selected_option else None
        correct = bool(selected_option and selected_option.correct)
        for misconception_id in misconception_ids:
            row = beliefs.get(misconception_id)
            prior = row["log_odds"] if row else logit(0.10)
            evidence_count = row["evidence_count"] if row else 0
            delta = 0.0
            if selected_misconception == misconception_id:
                confidence_factor = 1.10 if confidence is not None and confidence >= 0.80 else 1.0
                delta = (
                    evidence_weight
                    * confidence_factor
                    * min(1.6, 0.75 + 0.35 * question.discrimination)
                )
            elif correct:
                delta = -evidence_weight * 0.30 * question.discrimination
            if delta == 0.0:
                continue
            posterior = max(-6.0, min(6.0, prior + delta))
            connection.execute(
                """INSERT INTO misconception_beliefs(
                       learner_id, misconception_id, log_odds, evidence_count,
                       last_seen_at, as_of_event_id, model_version
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(learner_id, misconception_id) DO UPDATE SET
                       log_odds=excluded.log_odds,
                       evidence_count=excluded.evidence_count,
                       last_seen_at=excluded.last_seen_at,
                       as_of_event_id=excluded.as_of_event_id,
                       model_version=excluded.model_version""",
                (
                    learner_id,
                    misconception_id,
                    posterior,
                    evidence_count + 1,
                    to_timestamp(now),
                    event_id,
                    self.model_version,
                ),
            )

    @staticmethod
    def misconception_family_contribution(
        *,
        model_version: str,
        selected_misconception_id: str | None,
        misconception_id: str,
        correct: bool,
        confidence: float | None,
        response_ms: int | None,
        hint_count: int,
        base_evidence_weight: float,
        discrimination: float,
    ) -> float | None:
        """Return one credible signed family observation, or ``None``.

        A family contribution is deliberately independent of mastery's
        square-summable repeat discount.  Credible later evidence *replaces*
        this value during reconstruction; it is never added as another
        same-family vote.  Hinted, implausibly fast, missing-confidence, and
        low-confidence responses cannot erase an earlier strong observation.
        """

        if base_evidence_weight <= 0.0:
            return None
        response_class = classify_response_for_model(
            model_version=model_version,
            correct=correct,
            # This reducer is called only for a misconception referenced by the
            # presented item. A correct response necessarily selected the keyed
            # option; a named error carries the selected option's hypothesis.
            selected_option_id=(
                "selected"
                if correct or selected_misconception_id is not None
                else None
            ),
            selected_misconception_id=selected_misconception_id,
            confidence=confidence,
            response_ms=response_ms,
            hint_count=hint_count,
        )
        if response_class is ResponseClass.CREDIBLE_SUCCESS:
            # A credible keyed distinction is direct counter-evidence for each
            # named distractor on the item.  Keep it comparable to (but no
            # stronger than) a confident named-error observation so a family
            # can genuinely reverse in either direction.
            contribution = -base_evidence_weight * 0.75 * discrimination
        elif (
            response_class is ResponseClass.CREDIBLE_NAMED_ERROR
            and selected_misconception_id == misconception_id
        ):
            contribution = (
                base_evidence_weight
                * min(1.6, 0.75 + 0.35 * discrimination)
            )
        else:
            return None
        return max(
            -MAX_MISCONCEPTION_FAMILY_CONTRIBUTION,
            min(MAX_MISCONCEPTION_FAMILY_CONTRIBUTION, contribution),
        )

    @staticmethod
    def _misconception_event_metadata(
        raw: str, *, event_id: str
    ) -> tuple[str, str, float]:
        try:
            metadata = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Response event {event_id} has invalid misconception metadata."
            ) from exc
        if type(metadata) is not dict:
            raise ValueError(
                f"Response event {event_id} metadata must be an object."
            )
        marker = metadata.get(MISCONCEPTION_ALGORITHM_METADATA_KEY)
        algorithm = (
            LEGACY_MISCONCEPTION_ALGORITHM if marker is None else marker
        )
        if algorithm not in SUPPORTED_MISCONCEPTION_ALGORITHMS:
            raise ValueError(
                f"Response event {event_id} uses unsupported misconception "
                f"algorithm {algorithm!r}."
            )
        model_version = metadata.get("learner_model_version")
        if model_version not in SUPPORTED_MODEL_VERSIONS:
            raise ValueError(
                f"Response event {event_id} uses unsupported learner model "
                f"{model_version!r}."
            )
        base_evidence_weight = metadata.get("evidence_weight")
        if (
            isinstance(base_evidence_weight, bool)
            or not isinstance(base_evidence_weight, (int, float))
            or not 0.0 <= float(base_evidence_weight) <= 1.0
        ):
            raise ValueError(
                f"Response event {event_id} has invalid evidence weight."
            )
        return algorithm, model_version, float(base_evidence_weight)

    @classmethod
    def _legacy_misconception_evidence_weight(
        cls,
        *,
        model_version: str,
        base_evidence_weight: float,
        prior_family_attempts: int,
        confidence: float | None,
        response_ms: int | None,
        hint_count: int,
        selected_option_id: str | None,
        dependence_discount_override: float | None = None,
    ) -> float:
        """Reproduce the historical evidence weight for deterministic migration."""

        if dependence_discount_override is None:
            dependence_discount = cls.family_dependence_discount(
                prior_family_attempts
            )
        elif (
            isinstance(dependence_discount_override, bool)
            or not isinstance(dependence_discount_override, (int, float))
            or not 0.0 <= float(dependence_discount_override) <= 1.0
        ):
            raise ValueError(
                "Legacy misconception dependence override must be in [0, 1]."
            )
        else:
            dependence_discount = float(dependence_discount_override)
        if model_version in OBJECTIVE_GRID_MODEL_VERSIONS:
            confidence_discount = (
                MISSING_CONFIDENCE_DISCOUNT
                if confidence is None
                else (
                    1.0
                    if confidence >= 0.50
                    else max(0.10, confidence / 0.50)
                )
            )
            response_discount = (
                MISSING_RESPONSE_TIME_DISCOUNT
                if response_ms is None
                else (
                    1.0
                    if response_ms >= 500
                    else (0.55 if response_ms >= 250 else 0.15)
                )
            )
            selection_discount = (
                ABSTENTION_EVIDENCE_DISCOUNT
                if selected_option_id is None
                else 1.0
            )
        else:
            confidence_discount = (
                1.0
                if confidence is None or confidence >= 0.50
                else max(0.10, confidence / 0.50)
            )
            response_discount = (
                1.0
                if response_ms is None or response_ms >= 500
                else (0.55 if response_ms >= 250 else 0.15)
            )
            selection_discount = 1.0
        hint_discount = 1.0 if hint_count == 0 else 0.20
        return (
            base_evidence_weight
            * hint_discount
            * dependence_discount
            * confidence_discount
            * response_discount
            * selection_discount
        )

    def _reconstruct_misconception_belief(
        self,
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        misconception_id: str,
        through_event_id: str,
    ) -> dict[str, object]:
        """Reconstruct one hybrid legacy/current misconception projection.

        Historical unmarked events are replayed with their exact additive,
        clipped log-odds behavior.  On the first credible marked observation
        for a family, that family's *effective* legacy contribution is removed
        and replaced by one bounded signed contribution.  Later credible
        observations replace that signed value.  Noncredible observations do
        nothing.  Thus a family's infinite marked repeat sequence has at most
        ``MAX_MISCONCEPTION_FAMILY_CONTRIBUTION`` absolute influence.
        """

        boundary = connection.execute(
            """SELECT stream_id, stream_version, learner_id
               FROM events WHERE event_id=? AND event_type='ResponseSubmitted'""",
            (through_event_id,),
        ).fetchone()
        if (
            boundary is None
            or boundary["learner_id"] != learner_id
            or boundary["stream_id"] != f"learner:{learner_id}"
        ):
            raise ValueError(
                "Misconception reconstruction requires the current learner's "
                "ResponseSubmitted event."
            )
        target = connection.execute(
            "SELECT concept_id FROM misconceptions WHERE id=?",
            (misconception_id,),
        ).fetchone()
        if target is None:
            raise ValueError(
                f"Unknown misconception during reconstruction: {misconception_id}"
            )
        rows = connection.execute(
            """SELECT event.event_id, event.stream_version, event.occurred_at,
                      event.metadata_json,
                      attempt.family_id, attempt.selected_option_id,
                      attempt.is_correct, attempt.confidence,
                      attempt.response_ms, attempt.hint_count,
                      question.discrimination,
                      selected.misconception_id AS selected_misconception_id,
                      EXISTS(
                          SELECT 1 FROM options referenced
                          WHERE referenced.question_id=attempt.question_id
                            AND referenced.misconception_id=?
                      ) AS references_misconception,
                      EXISTS(
                          SELECT 1 FROM question_concepts mapping
                          WHERE mapping.question_id=attempt.question_id
                            AND mapping.concept_id=?
                            AND mapping.role IN (
                                'primary', 'secondary', 'prerequisite',
                                'contrast', 'transfer'
                            )
                      ) AS scores_misconception
               FROM attempts attempt
               JOIN events event ON event.event_id=attempt.event_id
               JOIN questions question ON question.id=attempt.question_id
               LEFT JOIN options selected
                 ON selected.question_id=attempt.question_id
                AND selected.option_id=attempt.selected_option_id
               WHERE attempt.learner_id=?
                 AND event.stream_id=?
                 AND event.event_type='ResponseSubmitted'
                 AND event.stream_version <= ?
               ORDER BY event.stream_version""",
            (
                misconception_id,
                target["concept_id"],
                learner_id,
                boundary["stream_id"],
                boundary["stream_version"],
            ),
        ).fetchall()
        if not rows or rows[-1]["event_id"] != through_event_id:
            raise ValueError(
                "Misconception reconstruction cannot find the current attempt "
                "inside its event-order boundary."
            )

        log_odds_value = logit(0.10)
        evidence_count = 0
        family_attempts: dict[str, int] = {}
        family_records: dict[str, list[FamilyResponseRecord]] = {}
        legacy_family_effects: dict[str, float] = {}
        legacy_family_counts: dict[str, int] = {}
        latest_family_contributions: dict[str, float] = {}
        latest_family_units: dict[str, int] = {}
        explicit_algorithm_seen = False
        current_observation_applied = False

        for row in rows:
            (
                algorithm,
                event_model_version,
                base_evidence_weight,
            ) = self._misconception_event_metadata(
                row["metadata_json"], event_id=row["event_id"]
            )
            if algorithm == MISCONCEPTION_ALGORITHM_VERSION:
                explicit_algorithm_seen = True
            elif explicit_algorithm_seen:
                raise ValueError(
                    "A legacy misconception event cannot follow an explicitly "
                    "versioned misconception event in one learner stream."
                )
            raw_family_id = row["family_id"]
            family_id = (
                canonical_family_label(raw_family_id)
                if event_model_version
                in CANONICAL_FAMILY_MODEL_VERSIONS
                else raw_family_id
            )
            if event_model_version in CANONICAL_FAMILY_MODEL_VERSIONS:
                # A v9 observation is the explicit boundary at which every
                # earlier published alias becomes one reducer key.  Merge the
                # already-applied bookkeeping without changing its accumulated
                # value; the current credible observation below can then
                # replace the whole equivalent-family contribution once.
                known_family_ids = (
                    set(family_attempts)
                    | set(family_records)
                    | set(legacy_family_effects)
                    | set(legacy_family_counts)
                    | set(latest_family_contributions)
                    | set(latest_family_units)
                )
                aliases = {
                    prior_family_id
                    for prior_family_id in known_family_ids
                    if prior_family_id != family_id
                    and canonical_family_label(prior_family_id)
                    == family_id
                }
                for alias in aliases:
                    family_attempts[family_id] = (
                        family_attempts.get(family_id, 0)
                        + family_attempts.pop(alias, 0)
                    )
                    family_records.setdefault(family_id, []).extend(
                        family_records.pop(alias, ())
                    )
                    legacy_family_effects[family_id] = (
                        legacy_family_effects.get(family_id, 0.0)
                        + legacy_family_effects.pop(alias, 0.0)
                    )
                    legacy_family_counts[family_id] = (
                        legacy_family_counts.get(family_id, 0)
                        + legacy_family_counts.pop(alias, 0)
                    )
                    latest_family_contributions[family_id] = (
                        latest_family_contributions.get(family_id, 0.0)
                        + latest_family_contributions.pop(alias, 0.0)
                    )
                    latest_family_units[family_id] = (
                        latest_family_units.get(family_id, 0)
                        + latest_family_units.pop(alias, 0)
                    )
                if family_id in family_records:
                    family_records[family_id].sort(
                        key=lambda record: record.occurred_at
                    )
            prior_family_attempts = family_attempts.get(family_id, 0)
            try:
                occurred_at = datetime.fromisoformat(row["occurred_at"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "Misconception reconstruction encountered an invalid "
                    "response timestamp."
                ) from exc
            current_family_record = FamilyResponseRecord(
                occurred_at=occurred_at,
                credible=self.credible_family_retest(
                    selected_option_id=row["selected_option_id"],
                    confidence=row["confidence"],
                    response_ms=row["response_ms"],
                    hint_count=row["hint_count"],
                ),
                renewal_eligible=(
                    event_model_version
                    in SPACING_AWARE_FAMILY_MODEL_VERSIONS
                ),
            )
            dependence_discount_override = None
            if event_model_version in SPACING_AWARE_FAMILY_MODEL_VERSIONS:
                dependence_discount_override = (
                    self.spacing_aware_family_evidence_power(
                        prior_records=family_records.get(family_id, ()),
                        occurred_at=occurred_at,
                        credible=current_family_record.credible,
                    ).power
                )
            relevant = bool(
                row["references_misconception"]
                and row["scores_misconception"]
            )
            applied = False

            if relevant and algorithm == LEGACY_MISCONCEPTION_ALGORITHM:
                legacy_weight = self._legacy_misconception_evidence_weight(
                    model_version=event_model_version,
                    base_evidence_weight=base_evidence_weight,
                    prior_family_attempts=prior_family_attempts,
                    confidence=row["confidence"],
                    response_ms=row["response_ms"],
                    hint_count=row["hint_count"],
                    selected_option_id=row["selected_option_id"],
                    dependence_discount_override=(
                        dependence_discount_override
                    ),
                )
                delta = 0.0
                if row["selected_misconception_id"] == misconception_id:
                    confidence_factor = (
                        1.10
                        if row["confidence"] is not None
                        and row["confidence"] >= 0.80
                        else 1.0
                    )
                    delta = (
                        legacy_weight
                        * confidence_factor
                        * min(
                            1.6,
                            0.75 + 0.35 * row["discrimination"],
                        )
                    )
                elif bool(row["is_correct"]):
                    delta = (
                        -legacy_weight
                        * 0.30
                        * row["discrimination"]
                    )
                if delta != 0.0:
                    before = log_odds_value
                    log_odds_value = max(
                        -6.0, min(6.0, log_odds_value + delta)
                    )
                    legacy_family_effects[family_id] = (
                        legacy_family_effects.get(family_id, 0.0)
                        + log_odds_value
                        - before
                    )
                    legacy_family_counts[family_id] = (
                        legacy_family_counts.get(family_id, 0) + 1
                    )
                    evidence_count += 1
                    applied = True

            elif relevant:
                contribution = self.misconception_family_contribution(
                    model_version=event_model_version,
                    selected_misconception_id=row[
                        "selected_misconception_id"
                    ],
                    misconception_id=misconception_id,
                    correct=bool(row["is_correct"]),
                    confidence=row["confidence"],
                    response_ms=row["response_ms"],
                    hint_count=row["hint_count"],
                    base_evidence_weight=base_evidence_weight,
                    discrimination=row["discrimination"],
                )
                if contribution is not None:
                    # Replace all effective votes for this semantic family.
                    # For v9 that includes every reviewed alias accumulated by
                    # older event versions; for earlier versions the key stays
                    # the exact published family label.
                    log_odds_value -= legacy_family_effects.get(
                        family_id, 0.0
                    )
                    log_odds_value -= latest_family_contributions.get(
                        family_id, 0.0
                    )
                    evidence_count -= legacy_family_counts.get(family_id, 0)
                    evidence_count -= latest_family_units.get(family_id, 0)
                    log_odds_value += contribution
                    evidence_count += 1
                    legacy_family_effects[family_id] = 0.0
                    legacy_family_counts[family_id] = 0
                    latest_family_contributions[family_id] = contribution
                    latest_family_units[family_id] = 1
                    applied = True

            family_attempts[family_id] = prior_family_attempts + 1
            family_records.setdefault(family_id, []).append(
                current_family_record
            )
            if row["event_id"] == through_event_id:
                current_observation_applied = applied

        return {
            "log_odds": max(-6.0, min(6.0, log_odds_value)),
            "evidence_count": evidence_count,
            "current_observation_applied": current_observation_applied,
        }

    @staticmethod
    def mastery_label(
        state: SkillState,
        independent_families: int,
        delayed_retrievals: int = 0,
        operation_kinds: int = 0,
        active_misconception_probability: float = 0.0,
        prerequisites_ready: bool = True,
        *,
        mastery_probability_override: float | None = None,
    ) -> str:
        if mastery_probability_override is not None and (
            isinstance(mastery_probability_override, bool)
            or not isinstance(mastery_probability_override, (int, float))
            or not isfinite(float(mastery_probability_override))
            or not 0.0 <= float(mastery_probability_override) <= 1.0
        ):
            raise ValueError(
                "mastery_probability_override must be a finite probability."
            )
        probability = (
            state.mastery_probability
            if mastery_probability_override is None
            else float(mastery_probability_override)
        )
        if state.exposures == 0:
            return "unassessed"
        if active_misconception_probability >= 0.35:
            return "fragile" if probability < 0.75 else "emerging"
        if (
            probability >= 0.90
            and independent_families >= 3
            and delayed_retrievals >= 1
            and operation_kinds >= 2
            and prerequisites_ready
        ):
            return "durable"
        if (
            probability >= 0.75
            and independent_families >= 2
            and operation_kinds >= 2
            and prerequisites_ready
        ):
            return "proficient"
        if probability >= 0.40:
            return "emerging"
        return "fragile"

    @staticmethod
    def retention_due_value(state: SkillState, now: datetime) -> float:
        if not state.next_review_at:
            return 0.0
        offset_hours = (now - state.next_review_at).total_seconds() / 3600.0
        window = max(12.0, state.stability_hours * 0.50)
        # Ramp from zero one window before the due time, through 0.5 at due,
        # to one after one overdue window.
        return max(0.0, min(1.0, 0.5 + 0.5 * offset_hours / window))
