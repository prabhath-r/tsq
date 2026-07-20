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
    ) -> tuple[dict[str, SkillState], list[dict[str, object]]]:
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
        p = self.predict_correct(question, projected)
        is_correct = bool(selected_option and selected_option.correct)
        y = 1.0 if is_correct else 0.0
        weights = self.evidence_weights(question)
        theta = sum(weight * projected[concept_id].mean for concept_id, weight in weights.items())
        logistic = sigmoid(question.discrimination * (theta - question.difficulty))
        dp_dtheta = (
            (1.0 - SESSION_LAPSE_RATE)
            * (1.0 - question.guess_rate - question.slip_rate)
            * question.discrimination
            * logistic
            * (1.0 - logistic)
        )
        score_theta = (y - p) * dp_dtheta / max(1e-6, p * (1.0 - p))
        if prior_family_attempts_override is None:
            prior_family_attempts = connection.execute(
                """SELECT COUNT(*) AS n FROM attempts
                   WHERE learner_id = ? AND family_id = ? AND event_id <> ?""",
                (learner_id, question.family_id, event_id),
            ).fetchone()["n"]
        else:
            if (
                type(prior_family_attempts_override) is not int
                or prior_family_attempts_override < 0
            ):
                raise ValueError(
                    "Prior family-attempt override must be a non-negative integer."
                )
            prior_family_attempts = prior_family_attempts_override
        dependence_discount = self.family_dependence_discount(prior_family_attempts)
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
        )
        # Feedback is a possible acquisition transition, not proof of knowing.
        # It is bounded by item trust and the same-family tail, and hints reduce
        # the incremental value because much of the explanation was pre-exposed.
        feedback_weight = (
            base_evidence_weight
            * dependence_discount
            * (1.0 if hint_count == 0 else 0.35)
        )
        certifying_retrieval = (
            is_correct
            and hint_count == 0
            and (confidence is None or confidence >= 0.50)
            and (response_ms is None or response_ms >= 250)
        )
        primary_mapping = next(
            mapping for mapping in question.concepts if mapping.role is ConceptRole.PRIMARY
        )
        prior_primary_family = connection.execute(
            """SELECT * FROM learner_skill_families
               WHERE learner_id = ? AND concept_id = ? AND family_id = ?""",
            (learner_id, primary_mapping.concept_id, question.family_id),
        ).fetchone()
        independent_retrieval = prior_primary_family is None
        if prior_primary_family:
            last_at = datetime.fromisoformat(prior_primary_family["last_unguided_correct_at"])
            primary_state = projected[primary_mapping.concept_id]
            required_spacing = timedelta(
                hours=max(24.0, min(24.0 * 30.0, primary_state.stability_hours * 0.50))
            )
            independent_retrieval = (now - last_at) >= required_spacing
        new_states: dict[str, SkillState] = {}
        changes: list[dict[str, object]] = []

        for mapping in question.concepts:
            if mapping.concept_id not in weights:
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
            if feedback_shown:
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
                    MODEL_VERSION,
                ),
            )
            if certifying_retrieval and mapping.role is ConceptRole.PRIMARY:
                family_evidence = connection.execute(
                    """SELECT * FROM learner_skill_families
                       WHERE learner_id = ? AND concept_id = ? AND family_id = ?""",
                    (learner_id, mapping.concept_id, question.family_id),
                ).fetchone()
                delayed_at = None
                if family_evidence:
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
                        question.family_id,
                        question.kind.value,
                        to_timestamp(now),
                        to_timestamp(now),
                        delayed_at,
                    ),
                )
            changes.append(
                {
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
            )

        self._update_misconceptions(
            connection,
            learner_id=learner_id,
            question=question,
            selected_option=selected_option,
            event_id=event_id,
            now=now,
            evidence_weight=evidence_weight,
            confidence=confidence,
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
                    MODEL_VERSION,
                ),
            )

    @staticmethod
    def mastery_label(
        state: SkillState,
        independent_families: int,
        delayed_retrievals: int = 0,
        operation_kinds: int = 0,
        active_misconception_probability: float = 0.0,
        prerequisites_ready: bool = True,
    ) -> str:
        probability = state.mastery_probability
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
