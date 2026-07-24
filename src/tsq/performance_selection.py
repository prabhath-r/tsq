# SPDX-License-Identifier: MPL-2.0

"""Explainable selection of released productive-skill probes.

The selected-response policy and the productive-task ledger deliberately have
different evidence boundaries.  This module connects them in one safe
direction only: an uncertain selected-response projection may recommend a
released implementation, debugging, explanation, design, or other productive
probe.  Productive observations remain shadow-only and never flow back into
mastery or certification through this policy.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from math import exp, isfinite, sqrt
from typing import Any

from .errors import ConflictError, NotFoundError, ValidationError
from .evidence import LearningTask, canonical_digest
from .learner import LearnerModel
from .store import Database


PRODUCTIVE_PROBE_POLICY_VERSION = "productive-probe-policy-v1"
MAX_RECOMMENDATION_LIMIT = 50


def _now(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if not isinstance(resolved, datetime):
        raise ValidationError("now must be a datetime.")
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValidationError("now must be timezone-aware.")
    return resolved.astimezone(timezone.utc)


def _probability(log_odds: float) -> float:
    if type(log_odds) not in {int, float} or not isfinite(log_odds):
        raise ValidationError(
            "Stored misconception log odds must be a finite number."
        )
    if log_odds >= 0.0:
        return 1.0 / (1.0 + exp(-log_odds))
    exponential = exp(log_odds)
    return exponential / (1.0 + exponential)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(
                f"Stored performance task contains duplicate field {key!r}."
            )
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValidationError(
        f"Stored performance task contains invalid number {value!r}."
    )


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise ValidationError(
            "Stored performance task contains a non-finite number."
        )
    return parsed


def _decode_task_definition(raw: object, label: str) -> LearningTask:
    if type(raw) is not str:
        raise ValidationError(f"{label} definition must be strict JSON text.")
    try:
        terms = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
            parse_float=_finite_json_float,
        )
    except ValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} definition is invalid JSON: {exc}") from exc
    if type(terms) is not dict:
        raise ValidationError(f"{label} definition must be a JSON object.")
    try:
        return LearningTask.from_terms(terms)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} definition is invalid: {exc}") from exc


def _criterion_weight_total(task: LearningTask) -> float:
    total = sum(criterion.score_weight for criterion in task.criteria)
    if not isfinite(total) or total <= 0.0:
        raise ValidationError(
            f"Performance task {task.id}@{task.version} has invalid rubric weights."
        )
    return total


def _task_objective_weights(task: LearningTask) -> dict[str, float]:
    """Return each objective's absolute share of the complete task rubric."""

    weights: Counter[str] = Counter()
    total = _criterion_weight_total(task)
    for criterion in task.criteria:
        if not criterion.objective_weights:
            continue
        for objective_id, weight in criterion.objective_weights:
            weights[objective_id] += criterion.score_weight * weight
    return {
        objective_id: weight / total
        for objective_id, weight in sorted(weights.items())
    }


def _task_concept_weights(task: LearningTask) -> dict[str, float]:
    weights: Counter[str] = Counter()
    total = _criterion_weight_total(task)
    for criterion in task.criteria:
        for concept_id, weight in criterion.concept_weights:
            weights[concept_id] += criterion.score_weight * weight
    return {
        concept_id: weight / total
        for concept_id, weight in sorted(weights.items())
    }


def _task_unmapped_concept_weights(task: LearningTask) -> dict[str, float]:
    """Return absolute concept shares for criteria lacking objective bindings."""

    weights: Counter[str] = Counter()
    total = _criterion_weight_total(task)
    for criterion in task.criteria:
        if criterion.objective_weights:
            continue
        for concept_id, weight in criterion.concept_weights:
            weights[concept_id] += criterion.score_weight * weight
    return {
        concept_id: weight / total
        for concept_id, weight in sorted(weights.items())
    }


def _task_misconception_weights(task: LearningTask) -> dict[str, float]:
    """Bound a misconception signal by its actual rubric share."""

    weights: Counter[str] = Counter()
    total = _criterion_weight_total(task)
    for criterion in task.criteria:
        if not criterion.misconception_ids:
            continue
        share = criterion.score_weight / len(criterion.misconception_ids)
        for misconception_id in criterion.misconception_ids:
            weights[misconception_id] += share
    return {
        misconception_id: weight / total
        for misconception_id, weight in sorted(weights.items())
    }


def _weighted_sum(
    weights: dict[str, float],
    values: dict[str, float],
    *,
    label: str,
) -> float:
    missing = set(weights) - set(values)
    if missing:
        raise ValidationError(
            f"{label} is missing released state for: "
            + ", ".join(sorted(missing))
            + "."
        )
    result = sum(weight * values[key] for key, weight in weights.items())
    if not isfinite(result):
        raise ValidationError(f"{label} produced a non-finite score.")
    return result


def _selected_response_components(
    state: Any,
    *,
    prior_mastery: float,
    current_time: datetime,
    label: str,
) -> tuple[float, float, float, float]:
    if state is None:
        return prior_mastery, 1.0, 1.0, 0.0
    try:
        mastery = float(state.mastery_probability)
        variance = float(state.variance)
        evidence_mass = float(state.evidence_mass)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} projection is malformed.") from exc
    if (
        not isfinite(mastery)
        or not 0.0 <= mastery <= 1.0
        or not isfinite(variance)
        or variance <= 0.0
        or not isfinite(evidence_mass)
        or evidence_mass < 0.0
    ):
        raise ValidationError(f"{label} projection is outside its valid range.")
    next_review_at = state.next_review_at
    if next_review_at is not None and (
        not isinstance(next_review_at, datetime)
        or next_review_at.tzinfo is None
        or next_review_at.utcoffset() is None
    ):
        raise ValidationError(f"{label} next-review time must be timezone-aware.")
    uncertainty = min(1.0, sqrt(variance) / 2.0)
    scarcity = 1.0 / (1.0 + evidence_mass)
    due = float(
        next_review_at is not None
        and next_review_at.astimezone(timezone.utc) <= current_time
    )
    return mastery, uncertainty, scarcity, due


def recommend_performance_tasks(
    database: Database,
    session_id: str,
    *,
    limit: int = 5,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Rank exact released tasks as optional diagnostic probes.

    Family novelty is a hard constraint whenever at least one fresh candidate
    exists.  Every score component is returned, and ties are resolved by
    immutable task identity.  The function is read-only and cannot start a task.
    """

    if type(session_id) is not str or not session_id.strip():
        raise ValidationError("session_id must be a non-blank string.")
    if type(limit) is not int or not 1 <= limit <= MAX_RECOMMENDATION_LIMIT:
        raise ValidationError(
            f"limit must be an integer from 1 through {MAX_RECOMMENDATION_LIMIT}."
        )
    current_time = _now(now)
    projection_model = LearnerModel()
    try:
        session = database.get_session(session_id)
        database.validate_session_focus(session)
    except NotFoundError:
        raise
    except ValidationError:
        raise
    if session["status"] != "active":
        raise ConflictError(
            f"Session {session_id} is {session['status']}; productive probes are "
            "recommended only inside active sessions."
        )
    with database.read() as connection:
        database.require_learner_evidence_safe(
            session["learner_id"],
            connection,
        )

    try:
        graph = database.get_graph(session["corpus_release_id"])
        if session["topic_id"]:
            scope = database.topic_owned_concepts(
                session["topic_id"],
                session["corpus_release_id"],
                include_descendants=True,
            )
        else:
            scope = graph.learning_scope(session["root_concept_id"])
        objectives = {
            objective.id: objective
            for objective in database.get_learning_objectives(
                session["corpus_release_id"]
            )
        }
        objective_states = database.get_objective_states(
            session["learner_id"]
        )
        concept_states = database.get_skill_states(session["learner_id"])
    except (NotFoundError, ValidationError):
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            "Selected-response projection cannot be loaded safely."
        ) from exc
    focus_objective_id = session["focus_objective_id"]
    focus_objective = objectives.get(focus_objective_id)
    if focus_objective_id is not None and focus_objective is None:
        raise ValidationError(
            f"Session focus objective {focus_objective_id} is outside its release."
        )
    if focus_objective is not None:
        scope.update(focus_objective.concept_ids)
    unknown_scope = scope - set(graph.concepts)
    if unknown_scope:
        raise ValidationError(
            "Session scope contains concepts outside its pinned release: "
            + ", ".join(sorted(unknown_scope))
            + "."
        )
    scoped_objective_ids = {
        objective_id
        for objective_id, objective in objectives.items()
        if objective.primary_concept_id in scope
    }

    objective_need: dict[str, float] = {}
    objective_uncertainty: dict[str, float] = {}
    objective_scarcity: dict[str, float] = {}
    objective_due: dict[str, float] = {}
    for objective_id, objective in objectives.items():
        state = objective_states.get(objective_id)
        if state is not None:
            try:
                state = projection_model.project_objective_state(
                    state,
                    objective,
                    current_time,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(
                    f"Objective {objective_id} cannot be projected safely "
                    f"with learner model {projection_model.model_version}."
                ) from exc
        mastery, uncertainty, scarcity, due = _selected_response_components(
            state,
            prior_mastery=objective.prior_mastery,
            current_time=current_time,
            label=f"Objective {objective_id}",
        )
        objective_uncertainty[objective_id] = uncertainty
        objective_scarcity[objective_id] = scarcity
        objective_due[objective_id] = due
        objective_need[objective_id] = (
            0.50 * (1.0 - mastery)
            + 0.30 * uncertainty
            + 0.15 * scarcity
            + 0.05 * due
        )

    concept_need: dict[str, float] = {}
    concept_uncertainty: dict[str, float] = {}
    concept_scarcity: dict[str, float] = {}
    concept_due: dict[str, float] = {}
    for concept_id in scope:
        state = concept_states.get(concept_id)
        if state is not None:
            try:
                state = projection_model.project_state(
                    state,
                    graph.concepts[concept_id],
                    current_time,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(
                    f"Concept {concept_id} cannot be projected safely "
                    f"with learner model {projection_model.model_version}."
                ) from exc
        mastery, uncertainty, scarcity, due = _selected_response_components(
            state,
            prior_mastery=graph.concepts[concept_id].prior_mastery,
            current_time=current_time,
            label=f"Concept {concept_id}",
        )
        concept_uncertainty[concept_id] = uncertainty
        concept_scarcity[concept_id] = scarcity
        concept_due[concept_id] = due
        concept_need[concept_id] = (
            0.50 * (1.0 - mastery)
            + 0.30 * uncertainty
            + 0.15 * scarcity
            + 0.05 * due
        )

    with database.read() as connection:
        database.require_learner_evidence_safe(
            session["learner_id"],
            connection,
        )
        pending_question = connection.execute(
            """SELECT decision.id FROM decisions decision
               WHERE decision.session_id=?
                 AND decision.consumed_at IS NULL
                 AND decision.invalidated_at IS NULL
               ORDER BY decision.created_at, decision.id LIMIT 1""",
            (session_id,),
        ).fetchone()
        active_attempt = connection.execute(
            """SELECT attempt.id FROM performance_attempts attempt
               WHERE attempt.session_id=?
                 AND NOT EXISTS (
                     SELECT 1 FROM performance_actions terminal
                     WHERE terminal.attempt_id=attempt.id
                       AND terminal.action_type IN ('submitted', 'abandoned')
                 )
               ORDER BY attempt.started_at, attempt.id LIMIT 1""",
            (session_id,),
        ).fetchone()
        candidate_rows = connection.execute(
            """SELECT member.release_id, member.status,
                      task.task_id, task.task_version, task.task_digest,
                      member.task_digest AS membership_task_digest,
                      task.definition_json, task_release.created_at
               FROM release_performance_tasks member
               JOIN performance_task_releases task_release
                 ON task_release.id=member.release_id
               JOIN performance_tasks task
                 ON task.task_id=member.task_id
                AND task.task_version=member.task_version
               WHERE task_release.corpus_release_id=?
                 AND member.status IN ('pilot', 'approved')
               ORDER BY task.task_id, task.task_version, member.release_id""",
            (session["corpus_release_id"],),
        ).fetchall()
        history_rows = connection.execute(
            """SELECT attempt.id, attempt.task_id, attempt.task_version,
                      attempt.task_digest AS attempt_task_digest,
                      attempt.started_at, task.task_digest AS stored_task_digest,
                      task.definition_json
               FROM performance_attempts attempt
               JOIN performance_tasks task
                 ON task.task_id=attempt.task_id
                AND task.task_version=attempt.task_version
               WHERE attempt.learner_id=?
               ORDER BY attempt.started_at, attempt.id""",
            (session["learner_id"],),
        ).fetchall()
        belief_rows = connection.execute(
            """SELECT belief.misconception_id, belief.log_odds,
                      belief.evidence_count
               FROM misconception_beliefs belief
               JOIN release_misconceptions membership
                 ON membership.misconception_id=belief.misconception_id
                AND membership.release_id=?
               WHERE belief.learner_id=?""",
            (session["corpus_release_id"], session["learner_id"]),
        ).fetchall()

    family_attempts: Counter[str] = Counter()
    modality_attempts: Counter[str] = Counter()
    for row in history_rows:
        historical_task = _decode_task_definition(
            row["definition_json"],
            f"Stored performance task {row['task_id']}@{row['task_version']}",
        )
        if (
            historical_task.id != row["task_id"]
            or historical_task.version != row["task_version"]
            or historical_task.digest != row["stored_task_digest"]
            or historical_task.digest != row["attempt_task_digest"]
        ):
            raise ValidationError(
                f"Stored performance history {row['id']} has a task identity mismatch."
            )
        family_attempts[historical_task.family_id] += 1
        modality_attempts[historical_task.modality.value] += 1

    beliefs: dict[str, float] = {}
    for row in belief_rows:
        evidence_count = row["evidence_count"]
        if type(evidence_count) is not int or evidence_count < 0:
            raise ValidationError(
                f"Misconception {row['misconception_id']} has an invalid evidence count."
            )
        if evidence_count > 0:
            beliefs[row["misconception_id"]] = _probability(
                row["log_odds"]
            )

    # The same immutable task may appear in more than one release.  Recommend
    # one exact membership, preferring approved over pilot and then the stable
    # release ID, so a caller can start it without an ambiguous lookup.
    memberships: dict[tuple[str, int, str], tuple[Any, LearningTask]] = {}
    for row in candidate_rows:
        task = _decode_task_definition(
            row["definition_json"],
            f"Stored performance task {row['task_id']}@{row['task_version']}",
        )
        if (
            task.id != row["task_id"]
            or task.version != row["task_version"]
            or task.digest != row["task_digest"]
            or task.digest != row["membership_task_digest"]
        ):
            raise ValidationError(
                f"Stored performance task {row['task_id']}@{row['task_version']} "
                "has an identity or membership digest mismatch."
            )
        task_concept_ids = set(task.concept_ids)
        task_objective_ids = set(task.objective_ids)
        unknown_objectives = task_objective_ids - set(objectives)
        if unknown_objectives:
            raise ValidationError(
                f"Stored performance task {task.id}@{task.version} references "
                "objectives outside its release: "
                + ", ".join(sorted(unknown_objectives))
                + "."
            )
        if not task_concept_ids or not task_concept_ids.issubset(scope):
            continue
        if not task_objective_ids.issubset(scoped_objective_ids):
            continue
        key = (task.id, task.version, task.digest)
        prior = memberships.get(key)
        rank = (row["status"] != "approved", row["release_id"])
        if prior is None or rank < (
            prior[0]["status"] != "approved",
            prior[0]["release_id"],
        ):
            memberships[key] = (row, task)

    scored: list[dict[str, Any]] = []
    focus_objective_id = session["focus_objective_id"]
    focus_concept_id = session["focus_concept_id"]
    focus_misconception_id = session["focus_misconception_id"]
    for row, task in memberships.values():
        task_objectives = _task_objective_weights(task)
        task_concepts = _task_concept_weights(task)
        unmapped_concepts = _task_unmapped_concept_weights(task)
        task_misconceptions = _task_misconception_weights(task)
        binding_specificity = sum(task_objectives.values())
        need = _weighted_sum(
            task_objectives,
            objective_need,
            label=f"Task {task.id}@{task.version} objective need",
        ) + _weighted_sum(
            unmapped_concepts,
            concept_need,
            label=f"Task {task.id}@{task.version} concept need",
        )
        uncertainty = _weighted_sum(
            task_objectives,
            objective_uncertainty,
            label=f"Task {task.id}@{task.version} objective uncertainty",
        ) + _weighted_sum(
            unmapped_concepts,
            concept_uncertainty,
            label=f"Task {task.id}@{task.version} concept uncertainty",
        )
        scarcity = _weighted_sum(
            task_objectives,
            objective_scarcity,
            label=f"Task {task.id}@{task.version} objective scarcity",
        ) + _weighted_sum(
            unmapped_concepts,
            concept_scarcity,
            label=f"Task {task.id}@{task.version} concept scarcity",
        )
        due = _weighted_sum(
            task_objectives,
            objective_due,
            label=f"Task {task.id}@{task.version} objective due state",
        ) + _weighted_sum(
            unmapped_concepts,
            concept_due,
            label=f"Task {task.id}@{task.version} concept due state",
        )
        focus_alignment = max(
            task_objectives.get(focus_objective_id, 0.0),
            0.65 * task_concepts.get(focus_concept_id, 0.0),
            0.90 * task_misconceptions.get(focus_misconception_id, 0.0),
        )
        misconception_signal = sum(
            weight * beliefs.get(misconception_id, 0.0)
            for misconception_id, weight in task_misconceptions.items()
        )
        if not isfinite(misconception_signal):
            raise ValidationError(
                f"Task {task.id}@{task.version} misconception belief produced "
                "a non-finite score."
            )
        family_count = family_attempts[task.family_id]
        family_novelty = 1.0 / (1.0 + family_count)
        modality_novelty = 1.0 / (1.0 + modality_attempts[task.modality.value])
        release_quality = 1.0 if row["status"] == "approved" else 0.6
        score = (
            0.34 * need
            + 0.24 * focus_alignment
            + 0.14 * misconception_signal
            + 0.10 * binding_specificity
            + 0.08 * family_novelty
            + 0.05 * modality_novelty
            + 0.05 * release_quality
        )
        if not isfinite(score):
            raise ValidationError(
                f"Task {task.id}@{task.version} produced a non-finite score."
            )
        reasons: list[str] = []
        if task_objectives.get(focus_objective_id, 0.0) > 0.0:
            reasons.append("matches_active_objective")
        elif task_concepts.get(focus_concept_id, 0.0) > 0.0:
            reasons.append("matches_active_concept")
        if task_misconceptions.get(focus_misconception_id, 0.0) > 0.0:
            reasons.append("tests_active_misconception")
        if due > 0.0:
            reasons.append("contains_due_selected_response_binding")
        if binding_specificity > 0.0:
            reasons.append("release_pinned_objective_binding")
        else:
            reasons.append("concept_only_binding")
        reasons.append("fresh_family" if family_count == 0 else "repeated_family")
        scored.append(
            {
                "task_id": task.id,
                "task_version": task.version,
                "task_digest": task.digest,
                "task_release_id": row["release_id"],
                "status": row["status"],
                "title": task.title,
                "modality": task.modality.value,
                "family_id": task.family_id,
                "objective_weights": task_objectives,
                "concept_weights": task_concepts,
                "misconception_ids": list(task.misconception_ids),
                "misconception_weights": task_misconceptions,
                "prior_family_attempts": family_count,
                "score": score,
                "components": {
                    "selected_response_probe_need": need,
                    "selected_response_uncertainty": uncertainty,
                    "selected_response_evidence_scarcity": scarcity,
                    "due_selected_response_binding_share": due,
                    "focus_alignment": focus_alignment,
                    "selected_response_misconception_signal": misconception_signal,
                    "objective_binding_specificity": binding_specificity,
                    "family_novelty": family_novelty,
                    "modality_novelty": modality_novelty,
                    "release_quality": release_quality,
                },
                "reasons": reasons,
            }
        )

    scored.sort(
        key=lambda item: (
            -item["score"],
            item["task_digest"],
            item["task_release_id"],
        )
    )
    eligible_candidate_count = len(scored)
    candidate_digest = canonical_digest(
        {
            "policy_version": PRODUCTIVE_PROBE_POLICY_VERSION,
            "learner_model_version": projection_model.model_version,
            "projection_time": current_time.isoformat(),
            "corpus_release_id": session["corpus_release_id"],
            "focus": {
                "objective_id": focus_objective_id,
                "concept_id": focus_concept_id,
                "misconception_id": focus_misconception_id,
            },
            "scope_concept_ids": sorted(scope),
            "ranked_candidates": scored,
        }
    )
    fresh_available = any(item["prior_family_attempts"] == 0 for item in scored)
    if fresh_available:
        scored = [item for item in scored if item["prior_family_attempts"] == 0]
    family_representatives: list[dict[str, Any]] = []
    represented_families: set[str] = set()
    for item in scored:
        if item["family_id"] in represented_families:
            continue
        represented_families.add(item["family_id"])
        family_representatives.append(item)
    recommendations = family_representatives[:limit]
    blockers: list[dict[str, str]] = []
    if pending_question is not None:
        blockers.append(
            {
                "code": "pending_question",
                "id": pending_question["id"],
                "resolution": "answer or invalidate the pending question",
            }
        )
    if active_attempt is not None:
        blockers.append(
            {
                "code": "active_performance_attempt",
                "id": active_attempt["id"],
                "resolution": "submit or abandon the active performance task",
            }
        )
    report = {
        "policy_version": PRODUCTIVE_PROBE_POLICY_VERSION,
        "learner_model_version": projection_model.model_version,
        "projection_time": current_time.isoformat(),
        "session_id": session_id,
        "learner_id": session["learner_id"],
        "corpus_release_id": session["corpus_release_id"],
        "focus": {
            "objective_id": focus_objective_id,
            "concept_id": focus_concept_id,
            "misconception_id": focus_misconception_id,
        },
        "scope_concept_ids": sorted(scope),
        "eligible_candidate_count": eligible_candidate_count,
        "ranked_family_count": len(family_representatives),
        "candidate_digest": candidate_digest,
        "fresh_family_constraint_applied": fresh_available,
        "recommendations": recommendations,
        "selection_boundary": {
            "read_only": True,
            "task_started": False,
            "startable_now": not blockers,
            "start_blockers": blockers,
            "productive_evidence_applied": False,
            "mastery_affected": False,
            "certification_affected": False,
            "interpretation": (
                "Selected-response uncertainty routes an optional productive "
                "probe. Productive observations remain shadow-only and cannot "
                "change mastery, certification, or the MCQ policy."
            ),
        },
    }
    with database.read() as connection:
        database.require_learner_evidence_safe(
            session["learner_id"],
            connection,
        )
    return report


__all__ = [
    "MAX_RECOMMENDATION_LIMIT",
    "PRODUCTIVE_PROBE_POLICY_VERSION",
    "recommend_performance_tasks",
]
