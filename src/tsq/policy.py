# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
from datetime import datetime, timezone
from statistics import median
from typing import Iterable

from .adaptive import RecursiveEvidenceBoundary
from .capacity import VERIFICATION_KINDS
from .errors import ExhaustedError, ValidationError
from .learner import MODEL_VERSION, LearnerModel
from .models import CandidateScore, Presentation, Question, SessionPhase
from .store import Database, new_id


POLICY_VERSION = "recursive-evidence-graph-v6"
MAX_REMEDIATION_DEPTH = 3


class _RetrySelection(Exception):
    pass


_PHASE_WEIGHTS: dict[SessionPhase, dict[str, float]] = {
    SessionPhase.DIAGNOSE: {
        "information_gain": 0.28,
        "learning_fit": 0.08,
        "concept_need": 0.18,
        "misconception_value": 0.14,
        "prerequisite_value": 0.10,
        "review_value": 0.03,
        "novelty": 0.08,
        "kind_fit": 0.03,
        "boundary_fit": 0.08,
    },
    SessionPhase.LEARN: {
        "information_gain": 0.15,
        "learning_fit": 0.22,
        "concept_need": 0.19,
        "misconception_value": 0.12,
        "prerequisite_value": 0.06,
        "review_value": 0.05,
        "novelty": 0.08,
        "kind_fit": 0.03,
        "boundary_fit": 0.10,
    },
    SessionPhase.REMEDIATE: {
        "information_gain": 0.16,
        "learning_fit": 0.16,
        "concept_need": 0.15,
        "misconception_value": 0.27,
        "prerequisite_value": 0.06,
        "review_value": 0.00,
        "novelty": 0.08,
        "kind_fit": 0.05,
        "boundary_fit": 0.07,
    },
    SessionPhase.VERIFY: {
        "information_gain": 0.12,
        "learning_fit": 0.14,
        "concept_need": 0.14,
        "misconception_value": 0.12,
        "prerequisite_value": 0.05,
        "review_value": 0.00,
        "novelty": 0.13,
        "kind_fit": 0.25,
        "boundary_fit": 0.05,
    },
    SessionPhase.REVIEW: {
        "information_gain": 0.12,
        "learning_fit": 0.16,
        "concept_need": 0.10,
        "misconception_value": 0.08,
        "prerequisite_value": 0.02,
        "review_value": 0.32,
        "novelty": 0.09,
        "kind_fit": 0.06,
        "boundary_fit": 0.05,
    },
}


_TARGET_SUCCESS = {
    SessionPhase.DIAGNOSE: 0.52,
    SessionPhase.LEARN: 0.68,
    SessionPhase.REMEDIATE: 0.72,
    SessionPhase.VERIFY: 0.64,
    SessionPhase.REVIEW: 0.74,
}


_KIND_FIT = {
    SessionPhase.DIAGNOSE: {
        "diagnostic": 1.0,
        "prerequisite_probe": 0.95,
        "comparison": 0.75,
    },
    SessionPhase.LEARN: {
        "conceptual": 0.90,
        "application": 1.0,
        "calculation": 0.85,
        "debugging": 0.85,
    },
    SessionPhase.REMEDIATE: {
        "prerequisite_probe": 1.0,
        "debugging": 0.95,
        "counterfactual": 0.90,
        "comparison": 0.90,
    },
    SessionPhase.VERIFY: {
        "transfer": 1.0,
        "application": 0.95,
        "counterfactual": 0.95,
        "debugging": 0.85,
    },
    SessionPhase.REVIEW: {
        "transfer": 1.0,
        "application": 0.95,
        "conceptual": 0.75,
    },
}


class AdaptivePolicy:
    """Candidate retrieval, constrained scoring, and randomized top-k choice."""

    def __init__(self, database: Database, learner_model: LearnerModel | None = None):
        self.database = database
        self.learner_model = learner_model or LearnerModel()
        self.boundary_planner = RecursiveEvidenceBoundary(self.learner_model)

    def choose(self, session_id: str, *, now: datetime | None = None) -> Presentation:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValidationError("now must be timezone-aware.")
        now = now.astimezone(timezone.utc)
        for _ in range(4):
            try:
                return self._choose_once(session_id, now=now)
            except _RetrySelection:
                continue
        raise ExhaustedError("Selection state changed repeatedly; retry the request.")

    def _choose_once(self, session_id: str, *, now: datetime) -> Presentation:
        session = self.database.get_session(session_id)
        if session["status"] != "active":
            raise ExhaustedError(f"Session {session_id} is {session['status']}.")
        # A choice is valid only for the learner projection against which it was
        # selected.  Another active session may advance that projection before
        # this session answers, so reconcile the pending choice under the write
        # lock before either returning it or doing expensive candidate scoring.
        current_pending = False
        with self.database.transaction() as connection:
            current_session = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            learner_row = connection.execute(
                "SELECT revision FROM learners WHERE id = ?", (session["learner_id"],)
            ).fetchone()
            if (
                not current_session
                or current_session["status"] != "active"
                or current_session["revision"] != session["revision"]
                or not learner_row
            ):
                raise _RetrySelection()
            learner_revision = learner_row["revision"]
            pending_row = connection.execute(
                """SELECT decision.*, revocation.revoked_at AS emergency_revoked_at
                   FROM decisions decision
                   LEFT JOIN question_revocations revocation
                     ON revocation.question_id = decision.question_id
                   WHERE decision.session_id = ?
                     AND decision.consumed_at IS NULL
                     AND decision.invalidated_at IS NULL
                   ORDER BY decision.created_at DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            if (
                pending_row
                and pending_row["emergency_revoked_at"] is None
                and pending_row["learner_revision"] == learner_revision
            ):
                current_pending = True
            elif pending_row:
                reason = (
                    "question_emergency_revoked"
                    if pending_row["emergency_revoked_at"] is not None
                    else "learner_projection_advanced"
                )
                invalidated = connection.execute(
                    """UPDATE decisions
                       SET invalidated_at = ?, invalidation_reason = ?
                       WHERE id = ? AND consumed_at IS NULL
                         AND invalidated_at IS NULL""",
                    (now.isoformat(), reason, pending_row["id"]),
                )
                if invalidated.rowcount != 1:
                    raise _RetrySelection()
                self.database.append_event(
                    connection,
                    stream_id=f"learner:{session['learner_id']}",
                    event_type="DecisionInvalidated",
                    payload={
                        "decision_id": pending_row["id"],
                        "reason": reason,
                        "selection_learner_revision": pending_row["learner_revision"],
                        "current_learner_revision": learner_revision,
                    },
                    metadata={
                        "policy_version": pending_row["policy_version"],
                        "learner_model_version": MODEL_VERSION,
                        "corpus_release_id": pending_row["corpus_release_id"],
                    },
                    learner_id=session["learner_id"],
                    session_id=session_id,
                    causation_id=pending_row["id"],
                    occurred_at=now,
                )
        if current_pending:
            pending = self.database.pending_presentation(session_id)
            if pending:
                return pending
            raise _RetrySelection()
        release_id = session["corpus_release_id"]
        phase = SessionPhase(session["phase"])
        graph = self.database.get_graph(release_id)
        topic_id = session.get("topic_id")
        if topic_id:
            base_scope = self.database.topic_scope(topic_id, release_id)
            owned_targets = self.database.topic_owned_concepts(
                topic_id, release_id, include_descendants=True
            )
        else:
            base_scope = graph.learning_scope(session["root_concept_id"])
            owned_targets = {session["root_concept_id"]}
        scope = set(base_scope)
        # A failed deliberate exploration can focus remediation outside the
        # requested topic. The focused tunnel must remain serviceable even
        # though ordinary main questions return to the requested curriculum.
        if session["focus_concept_id"]:
            scope.update(graph.learning_scope(session["focus_concept_id"]))

        recent_performance = self.database.session_recent_performance(
            session_id, limit=3
        )
        exploration_topic_ids: tuple[str, ...] = ()
        exploring = False
        if self._should_explore(session, phase, recent_performance):
            catalog = self.database.get_catalog(release_id)
            topic = next(
                (item for item in catalog["topics"] if item["id"] == topic_id),
                None,
            )
            if topic is not None:
                exploration_topic_ids = tuple(topic["related_topic_ids"])
                exploration_scope: set[str] = set()
                for related_topic_id in exploration_topic_ids:
                    exploration_scope.update(
                        self.database.topic_scope(related_topic_id, release_id)
                    )
                exploration_scope -= base_scope
                if exploration_scope:
                    scope = exploration_scope
                    exploring = True
        concepts = graph.concepts
        stored_states = self.database.get_skill_states(session["learner_id"])
        target_ids = [session["focus_concept_id"]] if session["focus_concept_id"] else list(scope)
        target_means = []
        for concept_id in target_ids:
            if not concept_id or concept_id not in concepts:
                continue
            concept = concepts[concept_id]
            state = stored_states.get(concept_id) or self.learner_model.initial_state(
                session["learner_id"], concept
            )
            target_means.append(self.learner_model.project_state(state, concept, now).mean)
        target_difficulty = median(target_means) if target_means else 0.0
        questions = self.database.questions_for_scope(
            scope,
            learner_id=session["learner_id"],
            focus_concept_id=session["focus_concept_id"],
            focus_misconception_id=session["focus_misconception_id"],
            release_id=release_id,
            target_difficulty=target_difficulty,
            limit=600,
        )
        if not questions:
            target = topic_id or session["root_concept_id"]
            raise ExhaustedError(f"Corpus gap: no approved questions cover {target}.")

        beliefs = self.database.get_misconception_beliefs(session["learner_id"])
        exposure = self.database.get_exposure_summary(
            session["learner_id"],
            question_ids={question.id for question in questions},
            family_ids={question.family_id for question in questions},
        )
        reuse_eligible: list[Question] = []
        for question in questions:
            family_exposure = exposure["families"].get(question.family_id)
            if family_exposure is None:
                reuse_eligible.append(question)
                continue
            try:
                family_last_at = datetime.fromisoformat(family_exposure["last_at"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Family {question.family_id} has an invalid exposure "
                    "timestamp."
                ) from exc
            if family_last_at.tzinfo is None or family_last_at.utcoffset() is None:
                raise ValidationError(
                    f"Family {question.family_id} has a timezone-naive exposure "
                    "timestamp."
                )
            concept = concepts[question.primary_concept_id]
            state = stored_states.get(question.primary_concept_id)
            if state is None:
                state = self.learner_model.initial_state(
                    session["learner_id"], concept
                )
            projected = self.learner_model.project_state(state, concept, now)
            if now - family_last_at < self.learner_model.required_family_spacing(
                projected
            ):
                continue
            if phase == SessionPhase.REVIEW and (
                projected.next_review_at is None
                or projected.next_review_at > now
            ):
                continue
            reuse_eligible.append(question)
        questions = reuse_eligible
        if not questions:
            raise ExhaustedError(
                "No safely independent question family is due yet for this learner."
            )
        recent = list(session["recent_families"])
        focus_concept = session["focus_concept_id"]
        focus_misconception = session["focus_misconception_id"]

        session_exposure = self.database.session_exposure_summary(session_id)
        independent_pool = [
            question
            for question in questions
            if question.id not in session_exposure["questions"]
            and question.family_id not in session_exposure["families"]
            and question.family_id not in recent[-4:]
        ]
        if not independent_pool:
            raise ExhaustedError(
                "Corpus gap: no unseen independent item family remains in this session."
            )
        candidate_pool = independent_pool

        pedagogical_role = "exploration_probe" if exploring else "main"
        focus_valid = False
        eligible = candidate_pool
        if phase == SessionPhase.REMEDIATE and (focus_concept or focus_misconception):
            verification_families = {
                question.family_id
                for question in independent_pool
                if focus_concept
                and question.primary_concept_id == focus_concept
                and question.kind in VERIFICATION_KINDS
            }
            if focus_misconception:
                probes = [
                    question
                    for question in independent_pool
                    if focus_misconception in question.misconception_ids
                ]
            else:
                probes = [
                    question
                    for question in independent_pool
                    if focus_concept
                    and question.primary_concept_id == focus_concept
                ]
            # Do not consume the last independent transfer family as the repair
            # probe; verification must remain possible before serving anything.
            focused = [
                question
                for question in probes
                if verification_families - {question.family_id}
            ]
            if not focused:
                raise ExhaustedError(
                    "Corpus gap: no remediation-plus-verification pair exists for "
                    f"{focus_misconception or focus_concept}. Run `tsq coverage --enqueue`."
                )
            eligible = focused
            pedagogical_role = "remediation_probe"
            focus_valid = True
        elif phase == SessionPhase.VERIFY and focus_concept:
            focused = [
                question
                for question in independent_pool
                if question.primary_concept_id == focus_concept
                and question.kind in VERIFICATION_KINDS
            ]
            if not focused:
                raise ExhaustedError(
                    "Corpus gap: no independent verification item exists for "
                    f"{focus_concept}. Run `tsq coverage --enqueue`."
                )
            eligible = focused
            pedagogical_role = "verification"
            focus_valid = True
        elif phase in {
            SessionPhase.LEARN,
            SessionPhase.DIAGNOSE,
            SessionPhase.REVIEW,
        }:
            families_by_concept: dict[str, set[str]] = {}
            families_by_misconception: dict[str, set[str]] = {}
            verification_by_concept: dict[str, set[str]] = {}
            for question in independent_pool:
                families_by_concept.setdefault(
                    question.primary_concept_id, set()
                ).add(question.family_id)
                if question.kind in VERIFICATION_KINDS:
                    verification_by_concept.setdefault(
                        question.primary_concept_id, set()
                    ).add(question.family_id)
                for misconception_id in question.misconception_ids:
                    families_by_misconception.setdefault(
                        misconception_id, set()
                    ).add(question.family_id)
            misconception_owners = {
                item.id: item.concept_id
                for item in self.database.get_misconceptions(
                    {
                        misconception_id
                        for question in independent_pool
                        for misconception_id in question.misconception_ids
                    },
                    release_id=release_id
                )
            }

            def serviceable(question: Question) -> bool:
                trigger_family = question.family_id
                primary = question.primary_concept_id
                # A skipped/uncategorized error still needs one repair and one
                # independent transfer family after the trigger.
                primary_repairs = (
                    families_by_concept.get(primary, set()) - {trigger_family}
                )
                primary_verifications = (
                    verification_by_concept.get(primary, set()) - {trigger_family}
                )
                if not any(
                    primary_verifications - {repair_family}
                    for repair_family in primary_repairs
                ):
                    return False
                for misconception_id in question.misconception_ids:
                    owner = misconception_owners.get(misconception_id, primary)
                    repair_families = (
                        families_by_misconception.get(misconception_id, set())
                        - {trigger_family}
                    )
                    verification_families = (
                        verification_by_concept.get(owner, set())
                        - {trigger_family}
                    )
                    if not any(
                        verification_families - {repair_family}
                        for repair_family in repair_families
                    ):
                        return False
                return True

            safe = [question for question in independent_pool if serviceable(question)]
            if not safe:
                raise ExhaustedError(
                    "Corpus gap: no safely serviceable main question remains while preserving "
                    "independent repair and verification families."
                )
            eligible = safe
            if topic_id and not exploring:
                direct_topic_items = [
                    question
                    for question in eligible
                    if question.primary_concept_id in owned_targets
                ]
                if not direct_topic_items:
                    raise ExhaustedError(
                        "Corpus gap: no safely serviceable question remains for "
                        f"curriculum topic {topic_id}."
                    )
                # Prerequisites remain in the modeled scope and can become a
                # focused remediation target after evidence warrants descent.
                # They are not silently substituted for the topic the learner
                # explicitly selected.
                eligible = direct_topic_items

        # Across sessions, prefer genuinely independent evidence before reusing
        # a family for the same primary concept.  Review is intentionally
        # exempt: spaced retrieval sometimes needs the previously learned
        # family, and its due-date signal should remain authoritative.
        if phase != SessionPhase.REVIEW:
            eligible = self._least_exposed_families_by_primary(eligible, exposure)

        prerequisite_distances: dict[str, int] = {}
        for target_id in owned_targets:
            for concept_id, distance in graph.learning_distances_to(
                target_id
            ).items():
                prior = prerequisite_distances.get(concept_id)
                if prior is None or distance < prior:
                    prerequisite_distances[concept_id] = distance

        topic_by_concept: dict[str, str] = {}
        if topic_id:
            for catalog_topic in self.database.get_catalog(release_id)["topics"]:
                for concept in catalog_topic["concepts"]:
                    topic_by_concept[concept["id"]] = catalog_topic["id"]
        last_primary_concept = (
            recent_performance[0]["primary_concept_id"]
            if recent_performance
            else None
        )
        connected_pairs = {
            frozenset((edge.source_id, edge.target_id)) for edge in graph.edges
        }
        readiness = self.boundary_planner.readiness_map(
            learner_id=session["learner_id"],
            graph=graph,
            stored_states=stored_states,
            now=now,
            concept_ids={question.primary_concept_id for question in eligible},
        )

        scores = [
            self._score(
                question,
                session=session,
                phase=phase,
                prerequisite_distances=prerequisite_distances,
                concepts=concepts,
                stored_states=stored_states,
                beliefs=beliefs,
                exposure=exposure,
                recent_families=recent,
                last_primary_concept=last_primary_concept,
                topic_by_concept=topic_by_concept,
                base_scope=base_scope,
                connected_pairs=connected_pairs,
                readiness=readiness,
                now=now,
            )
            for question in eligible
        ]
        scores.sort(key=lambda score: (-score.total, score.question_id))
        if not scores:
            raise ExhaustedError("Corpus gap: the policy found no eligible question.")

        top_k = scores[: min(5, len(scores))]
        chosen_score, propensity = self._sample_top_k(
            top_k, seed=session["rng_seed"], step=session["step"]
        )
        question_by_id = {question.id: question for question in eligible}
        chosen = question_by_id[chosen_score.question_id]
        option_ids = [option.id for option in chosen.options]
        order_rng = random.Random(f"options:{session['rng_seed']}:{session['step']}:{chosen.id}")
        order_rng.shuffle(option_ids)

        digest_material = "|".join(f"{score.question_id}:{score.total:.8f}" for score in scores)
        candidate_digest = hashlib.sha256(digest_material.encode("utf-8")).hexdigest()
        rationale = self._rationale(
            chosen,
            chosen_score,
            phase,
            focus_concept,
            focus_misconception,
            exploration_topic_ids=exploration_topic_ids if exploring else (),
        )
        decision_id = new_id("dec")

        with self.database.transaction() as connection:
            current = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            current_learner = connection.execute(
                "SELECT revision FROM learners WHERE id = ?", (session["learner_id"],)
            ).fetchone()
            release_question = connection.execute(
                """SELECT rq.*, q.version, q.content_hash FROM release_questions rq
                   JOIN questions q ON q.id = rq.question_id
                   WHERE rq.release_id = ? AND rq.question_id = ?""",
                (release_id, chosen.id),
            ).fetchone()
            if (
                not current
                or current["status"] != "active"
                or current["revision"] != session["revision"]
                or current["phase"] != session["phase"]
                or not current_learner
                or current_learner["revision"] != learner_revision
                or current["corpus_release_id"] != release_id
                or not release_question
                or release_question["status"]
                not in {"approved", "calibrated"}
                or connection.execute(
                    "SELECT 1 FROM question_revocations WHERE question_id = ?",
                    (chosen.id,),
                ).fetchone()
            ):
                raise _RetrySelection()
            existing = connection.execute(
                """SELECT id FROM decisions WHERE session_id = ?
                   AND consumed_at IS NULL AND invalidated_at IS NULL""",
                (session_id,),
            ).fetchone()
            if existing:
                # Another caller won the race; return the durable pending choice.
                pass
            else:
                connection.execute(
                    """INSERT INTO decisions(
                           id, session_id, learner_id, question_id,
                           question_version, question_content_hash, question_status,
                           evidence_weight, corpus_release_id, session_revision,
                           learner_revision, phase, focus_concept_id,
                           focus_misconception_id, pedagogical_role, focus_valid,
                           policy_version, candidate_count, candidate_digest,
                           top_candidates_json, selected_score_json, propensity,
                           option_order_json, rationale, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        decision_id,
                        session_id,
                        session["learner_id"],
                        chosen.id,
                        release_question["version"],
                        release_question["content_hash"],
                        release_question["status"],
                        release_question["evidence_weight"],
                        release_id,
                        session["revision"],
                        learner_revision,
                        phase.value,
                        focus_concept,
                        focus_misconception,
                        pedagogical_role,
                        int(focus_valid),
                        POLICY_VERSION,
                        len(scores),
                        candidate_digest,
                        json.dumps(
                            [
                                {"question_id": score.question_id, **score.terms()}
                                for score in scores[:10]
                            ],
                            sort_keys=True,
                        ),
                        json.dumps(chosen_score.terms(), sort_keys=True),
                        propensity,
                        json.dumps(option_ids),
                        rationale,
                        now.isoformat(),
                    ),
                )
                self.database.append_event(
                    connection,
                    stream_id=f"learner:{session['learner_id']}",
                    event_type="QuestionSelected",
                    payload={
                        "decision_id": decision_id,
                        "question_id": chosen.id,
                        "phase": phase.value,
                        "candidate_count": len(scores),
                        "candidate_digest": candidate_digest,
                        "propensity": propensity,
                        "score": chosen_score.terms(),
                        "option_order": option_ids,
                        "question_version": release_question["version"],
                        "question_content_hash": release_question["content_hash"],
                        "question_status": release_question["status"],
                        "evidence_weight": release_question["evidence_weight"],
                        "corpus_release_id": release_id,
                        "session_revision": session["revision"],
                        "learner_revision": learner_revision,
                        "focus_concept_id": focus_concept,
                        "focus_misconception_id": focus_misconception,
                        "pedagogical_role": pedagogical_role,
                        "focus_valid": focus_valid,
                    },
                    metadata={
                        "policy_version": POLICY_VERSION,
                        "learner_model_version": MODEL_VERSION,
                        "corpus_release_id": release_id,
                    },
                    learner_id=session["learner_id"],
                    session_id=session_id,
                )
                updated = connection.execute(
                    """UPDATE sessions SET step = step + 1, revision = revision + 1,
                           updated_at = ? WHERE id = ? AND revision = ? AND phase = ?
                           AND status = 'active'""",
                    (now.isoformat(), session_id, session["revision"], session["phase"]),
                )
                if updated.rowcount != 1:
                    raise _RetrySelection()
        durable = self.database.pending_presentation(session_id)
        if not durable:
            raise ExhaustedError("Selection was not persisted.")
        return durable

    @staticmethod
    def _should_explore(
        session: dict,
        phase: SessionPhase,
        recent_performance: list[dict],
    ) -> bool:
        """Gate bounded, explicit probes outside a learner's requested topic."""
        if (
            not session.get("topic_id")
            or session.get("exploration_mode") != "adaptive"
            or phase not in {
                SessionPhase.LEARN,
                SessionPhase.DIAGNOSE,
                SessionPhase.REVIEW,
            }
            or session.get("focus_concept_id")
            or session.get("focus_misconception_id")
            or len(recent_performance) < 3
            or session["step"] < 3
            or (session["step"] - 3) % 5 != 0
        ):
            return False
        return all(
            attempt["correct"]
            and attempt["pedagogical_role"] in {"main", "exploration_probe"}
            and attempt["hint_count"] == 0
            and (
                attempt["confidence"] is None
                or attempt["confidence"] >= 0.65
            )
            and (
                attempt["response_ms"] is None
                or attempt["response_ms"] >= 250
            )
            for attempt in recent_performance
        )

    @staticmethod
    def _least_exposed_families_by_primary(
        questions: Iterable[Question], exposure: dict
    ) -> list[Question]:
        candidates = list(questions)
        least_by_concept: dict[str, int] = {}
        for question in candidates:
            family_count = exposure["families"].get(
                question.family_id, {}
            ).get("count", 0)
            previous = least_by_concept.get(question.primary_concept_id)
            if previous is None or family_count < previous:
                least_by_concept[question.primary_concept_id] = family_count
        return [
            question
            for question in candidates
            if exposure["families"].get(question.family_id, {}).get("count", 0)
            == least_by_concept[question.primary_concept_id]
        ]

    def _score(
        self,
        question: Question,
        *,
        session: dict,
        phase: SessionPhase,
        prerequisite_distances,
        concepts,
        stored_states,
        beliefs,
        exposure,
        recent_families: list[str],
        last_primary_concept: str | None,
        topic_by_concept: dict[str, str],
        base_scope: set[str],
        connected_pairs: set[frozenset[str]],
        readiness,
        now: datetime,
    ) -> CandidateScore:
        states = self.learner_model.states_for_question(
            session["learner_id"], question, concepts, stored_states, now
        )
        predicted = self.learner_model.predict_correct(question, states)
        raw_ig = self.learner_model.expected_information_gain(question, states)
        evidence_weights = self.learner_model.evidence_weights(question)
        information_gain = 1.0 - math.exp(-2.5 * raw_ig)
        target = _TARGET_SUCCESS[phase]
        learning_fit = math.exp(-((predicted - target) / 0.24) ** 2)
        concept_need = sum(
            weight * (1.0 - states[concept_id].mastery_probability)
            for concept_id, weight in evidence_weights.items()
        )

        misconception_value = 0.0
        if session["focus_misconception_id"] in question.misconception_ids:
            misconception_value = 1.0
        elif question.misconception_ids:
            misconception_value = max(
                (beliefs[mid].probability if mid in beliefs else 0.10)
                for mid in question.misconception_ids
            )

        distance = prerequisite_distances.get(question.primary_concept_id)
        if distance is None or distance == 0:
            prerequisite_value = 0.0
        else:
            prerequisite_value = min(1.0, concept_need * (0.55 + 0.45 / distance))

        review_value = sum(
            weight * self.learner_model.retention_due_value(states[concept_id], now)
            for concept_id, weight in evidence_weights.items()
        )
        q_exposure = exposure["questions"].get(question.id, 0)
        family_exposure = exposure["families"].get(question.family_id, {}).get("count", 0)
        novelty = 1.0 / (1.0 + 0.55 * q_exposure + 0.25 * family_exposure)
        if question.family_id in recent_families[-4:]:
            novelty *= 0.08

        kind_fit = _KIND_FIT[phase].get(question.kind.value, 0.55)
        primary = question.primary_concept_id
        primary_readiness = readiness[primary]
        if primary_readiness.bottleneck_concept_id is None:
            boundary_fit = 1.0
        else:
            target_support = {
                SessionPhase.DIAGNOSE: 0.55,
                SessionPhase.LEARN: 0.68,
                SessionPhase.REMEDIATE: 0.76,
                SessionPhase.VERIFY: 0.74,
                SessionPhase.REVIEW: 0.82,
            }[phase]
            # Readiness is a floor, not a narrow target: strong prerequisite
            # evidence must never make an otherwise useful advanced item less
            # eligible. Unsupported items are reduced smoothly instead.
            boundary_fit = min(
                1.0, primary_readiness.prerequisite_support / target_support
            )
        if last_primary_concept is None:
            continuity = 0.50
        elif primary == last_primary_concept:
            continuity = 1.0
        elif frozenset((primary, last_primary_concept)) in connected_pairs:
            continuity = 0.85
        elif (
            topic_by_concept.get(primary) is not None
            and topic_by_concept.get(primary)
            == topic_by_concept.get(last_primary_concept)
        ):
            continuity = 0.70
        elif primary in base_scope:
            continuity = 0.40
        else:
            continuity = 0.15
        values = {
            "information_gain": information_gain,
            "learning_fit": learning_fit,
            "concept_need": concept_need,
            "misconception_value": misconception_value,
            "prerequisite_value": prerequisite_value,
            "review_value": review_value,
            "novelty": novelty,
            "kind_fit": kind_fit,
            "boundary_fit": boundary_fit,
        }
        total = sum(_PHASE_WEIGHTS[phase][name] * value for name, value in values.items())
        if session.get("topic_id"):
            total += {
                SessionPhase.DIAGNOSE: 0.05,
                SessionPhase.LEARN: 0.10,
                SessionPhase.REMEDIATE: 0.12,
                SessionPhase.VERIFY: 0.12,
                SessionPhase.REVIEW: 0.08,
            }[phase] * continuity
        if session["focus_concept_id"] and any(
            mapping.concept_id == session["focus_concept_id"] for mapping in question.concepts
        ):
            total += 0.12
        if question.status.value == "calibrated":
            total += 0.03
        return CandidateScore(
            question_id=question.id,
            total=total,
            predicted_correct=predicted,
            information_gain=information_gain,
            learning_fit=learning_fit,
            concept_need=concept_need,
            misconception_value=misconception_value,
            prerequisite_value=prerequisite_value,
            review_value=review_value,
            novelty=novelty,
            kind_fit=kind_fit,
            continuity=continuity,
            boundary_fit=boundary_fit,
        )

    @staticmethod
    def _sample_top_k(
        scores: Iterable[CandidateScore], *, seed: int, step: int
    ) -> tuple[CandidateScore, float]:
        candidates = list(scores)
        temperature = 0.10
        peak = max(score.total for score in candidates)
        weights = [math.exp((score.total - peak) / temperature) for score in candidates]
        total_weight = sum(weights)
        probabilities = [weight / total_weight for weight in weights]
        rng = random.Random(f"policy:{seed}:{step}")
        threshold = rng.random()
        cumulative = 0.0
        for score, probability in zip(candidates, probabilities, strict=True):
            cumulative += probability
            if threshold <= cumulative:
                return score, probability
        return candidates[-1], probabilities[-1]

    @staticmethod
    def _rationale(
        question: Question,
        score: CandidateScore,
        phase: SessionPhase,
        focus_concept: str | None,
        focus_misconception: str | None,
        *,
        exploration_topic_ids: tuple[str, ...] = (),
    ) -> str:
        reasons = [
            f"phase={phase.value}",
            f"predicted_success={score.predicted_correct:.2f}",
            f"information={score.information_gain:.2f}",
            f"need={score.concept_need:.2f}",
        ]
        if focus_misconception and focus_misconception in question.misconception_ids:
            reasons.append(f"discriminates_misconception={focus_misconception}")
        elif focus_concept and question.primary_concept_id == focus_concept:
            reasons.append(f"tests_focus={focus_concept}")
        if score.review_value > 0.2:
            reasons.append(f"review_due={score.review_value:.2f}")
        if score.boundary_fit < 0.70:
            reasons.append(f"boundary_fit={score.boundary_fit:.2f}")
        if exploration_topic_ids:
            reasons.append(
                "deliberate_related_topic_probe=" + ",".join(exploration_topic_ids)
            )
        elif score.continuity >= 0.70:
            reasons.append(f"continuity={score.continuity:.2f}")
        return "; ".join(reasons)
