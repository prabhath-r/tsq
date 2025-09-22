# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from math import pi
from typing import Mapping

from .models import (
    MASTERY_THRESHOLD,
    Concept,
    ConceptRole,
    Option,
    Question,
    SkillState,
    logit,
    sigmoid,
)
from .store import to_timestamp


MODEL_VERSION = "irt-gaussian-retention-v3"
INITIAL_VARIANCE = 2.25
INITIAL_STABILITY_HOURS = 48.0
MIN_POSTERIOR_VARIANCE = 1e-6
MAX_POSTERIOR_VARIANCE = 4.0
SESSION_LAPSE_RATE = 0.03


class LearnerModel:
    """Interpretable online projection over proficiency and misconceptions.

    Competence is a diagonal Gaussian latent trait per concept. Responses are
    observed through a 4-parameter logistic item model; elapsed time projects
    competence toward its hierarchical prior and widens uncertainty. Feedback is
    a separate, small learning transition so one response is not counted twice as
    both evidence and acquisition.
    """

    def initial_state(self, learner_id: str, concept: Concept) -> SkillState:
        return SkillState(
            learner_id=learner_id,
            concept_id=concept.id,
            mean=logit(concept.prior_mastery),
            variance=INITIAL_VARIANCE,
            stability_hours=INITIAL_STABILITY_HOURS,
        )

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
    ) -> float:
        weights = self.evidence_weights(question)
        theta = sum(weight * states[concept_id].mean for concept_id, weight in weights.items())
        logistic = sigmoid(question.discrimination * (theta - question.difficulty))
        modeled = question.guess_rate + (1.0 - question.guess_rate - question.slip_rate) * logistic
        # A lapse mixture prevents one surprising response from destroying a strong posterior.
        random_click = 1.0 / max(2, len(question.options))
        return SESSION_LAPSE_RATE * random_click + (1.0 - SESSION_LAPSE_RATE) * modeled

    def expected_information_gain(
        self,
        question: Question,
        states: Mapping[str, SkillState],
    ) -> float:
        weights = self.evidence_weights(question)
        theta = sum(weight * states[concept_id].mean for concept_id, weight in weights.items())
        logistic = sigmoid(question.discrimination * (theta - question.difficulty))
        p = self.predict_correct(question, states)
        dp_dtheta = (
            (1.0 - SESSION_LAPSE_RATE)
            * (1.0 - question.guess_rate - question.slip_rate)
            * question.discrimination
            * logistic
            * (1.0 - logistic)
        )
        denominator = max(1e-6, p * (1.0 - p))
        gain = 0.0
        for concept_id, weight in weights.items():
            state = states[concept_id]
            fisher = (weight * dp_dtheta) ** 2 / denominator
            posterior_variance = 1.0 / (1.0 / state.variance + fisher)
            gain += weight * max(0.0, state.variance - posterior_variance)
        return gain
