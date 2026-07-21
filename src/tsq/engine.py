# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from math import exp, isfinite, log
from statistics import mean, median
from typing import Any

from .adaptive import BOUNDARY_ALGORITHM_VERSION, RecursiveEvidenceBoundary
from .errors import ConflictError, ExhaustedError, NotFoundError, ValidationError
from .learner import LearnerModel, MODEL_VERSION
from .models import (
    Option,
    Presentation,
    QuestionStatus,
    SessionPhase,
    SubmissionResult,
)
from .policy import AdaptivePolicy, MAX_REMEDIATION_DEPTH, POLICY_VERSION
from .store import Database, new_id


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _option_payload(option: Option | None) -> dict[str, Any] | None:
    if option is None:
        return None
    return {
        "id": option.id,
        "text": option.text,
        "correct": option.correct,
        "rationale": option.rationale,
        "misconception_id": option.misconception_id,
    }


def _option_from_payload(payload: dict[str, Any] | None) -> Option | None:
    if payload is None:
        return None
    return Option(
        id=payload["id"],
        text=payload["text"],
        correct=bool(payload["correct"]),
        rationale=payload["rationale"],
        misconception_id=payload.get("misconception_id"),
    )


def _main_phase(mode: str) -> SessionPhase:
    return {
        "diagnose": SessionPhase.DIAGNOSE,
        "review": SessionPhase.REVIEW,
    }.get(mode, SessionPhase.LEARN)


class AdaptiveEngine:
    """Application service coordinating sessions, policy, evidence, and projections."""

    def __init__(self, database: Database):
        self.database = database
        self.learner_model = LearnerModel()
        self.boundary_planner = RecursiveEvidenceBoundary(self.learner_model)
        self.policy = AdaptivePolicy(database, self.learner_model)

    def create_learner(self, learner_id: str, display_name: str | None = None) -> dict[str, Any]:
        return self.database.ensure_learner(learner_id, display_name)

    def start_session(
        self,
        learner_id: str,
        root_concept_id: str | None = None,
        *,
        topic_id: str | None = None,
        explore_related: bool = True,
        mode: str = "learn",
        seed: int | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"learn", "diagnose", "review"}:
            raise ValidationError("Mode must be learn, diagnose, or review.")
        resolved_topic_id: str | None = None
        resolved_root = root_concept_id
        topic_reference = topic_id
        if topic_reference is None and root_concept_id is not None:
            try:
                topic = self.database.resolve_topic(root_concept_id)
            except NotFoundError:
                topic = None
            if topic is not None:
                resolved_topic_id = topic["id"]
                resolved_root = None
        elif topic_reference is not None:
            topic = self.database.resolve_topic(topic_reference)
            resolved_topic_id = topic["id"]
            resolved_root = None
        return self.database.create_session(
            learner_id,
            resolved_root,
            topic_id=resolved_topic_id,
            exploration_mode=(
                "adaptive" if resolved_topic_id and explore_related else "off"
            ),
            mode=mode,
            seed=seed,
            idempotency_key=idempotency_key,
        )

    def next_question(self, session_id: str, *, now: datetime | None = None) -> Presentation:
        try:
            return self.policy.choose(session_id, now=now)
        except ExhaustedError as exc:
            if str(exc).startswith("Corpus gap:"):
                event_time = now or datetime.now(timezone.utc)
                if event_time.tzinfo is not None and event_time.utcoffset() is not None:
                    self._record_corpus_gap(
                        session_id,
                        message=str(exc),
                        now=event_time.astimezone(timezone.utc),
                    )
            raise

    def _record_corpus_gap(
        self, session_id: str, *, message: str, now: datetime
    ) -> None:
        """Persist an actionable, deduplicated authoring demand for a live gap."""
        # Importing here keeps the online engine independent of authoring worker
        # adapters while reusing their exact persisted blueprint contract.
        from .authoring import GenerationBlueprint, PROMPT_VERSION

        session = self.database.get_session(session_id)
        graph = self.database.get_graph(session["corpus_release_id"])
        focus_concept_id = session["focus_concept_id"]
        if focus_concept_id is None:
            containers = {
                edge.target_id
                for edge in graph.edges
                if edge.relation.value == "part_of"
            }
            if session["topic_id"]:
                # The topic session root is a stable replay anchor.  An
                # authoring demand must remain attached to an objective owned
                # by the selected curriculum bucket, not drift to an external
                # prerequisite merely because that prerequisite is in scope.
                assessable = self.database.topic_owned_concepts(
                    session["topic_id"],
                    session["corpus_release_id"],
                    include_descendants=True,
                ) - containers
            else:
                assessable = (
                    graph.learning_scope(session["root_concept_id"]) - containers
                )
            states = self.database.get_skill_states(session["learner_id"])

            def need_key(concept_id: str) -> tuple[float, str]:
                state = states.get(concept_id)
                if state is not None:
                    return state.mean, concept_id
                prior = graph.concepts[concept_id].prior_mastery
                return log(prior / (1.0 - prior)), concept_id

            focus_concept_id = min(
                assessable or {session["root_concept_id"]}, key=need_key
            )
        concept = graph.concepts.get(focus_concept_id)
        if concept is None:
            return
        with self.database.transaction() as connection:
            current = connection.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if (
                not current
                or current["status"] != "active"
                or current["revision"] != session["revision"]
            ):
                return
            demand_key = _canonical_hash(
                {
                    "session_id": session_id,
                    "session_revision": current["revision"],
                    "phase": current["phase"],
                    "focus_concept_id": focus_concept_id,
                    "focus_misconception_id": current["focus_misconception_id"],
                    "corpus_release_id": current["corpus_release_id"],
                    "message": message,
                }
            )
            idempotency_key = f"corpus-gap:{demand_key}"
            if connection.execute(
                "SELECT 1 FROM events WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone():
                return

            source_ids = tuple(
                row["source_id"]
                for row in connection.execute(
                    """SELECT DISTINCT qs.source_id
                       FROM question_sources qs
                       JOIN question_concepts qc ON qc.question_id = qs.question_id
                       JOIN release_questions rq ON rq.question_id = qs.question_id
                       WHERE qc.concept_id = ? AND rq.release_id = ?
                       ORDER BY qs.source_id LIMIT 8""",
                    (focus_concept_id, current["corpus_release_id"]),
                )
            )
            if not source_ids:
                source_ids = tuple(
                    row["source_id"]
                    for row in connection.execute(
                        """SELECT source_id FROM release_sources
                           WHERE release_id = ? ORDER BY source_id LIMIT 8""",
                        (current["corpus_release_id"],),
                    )
                )
            focus_misconception_id = current["focus_misconception_id"]
            if focus_misconception_id:
                misconception_ids = (focus_misconception_id,)
            else:
                misconception_ids = tuple(
                    row["misconception_id"]
                    for row in connection.execute(
                        """SELECT membership.misconception_id
                           FROM release_misconceptions membership
                           JOIN misconceptions misconception
                             ON misconception.id = membership.misconception_id
                           WHERE membership.release_id = ?
                             AND misconception.concept_id = ?
                           ORDER BY membership.misconception_id LIMIT 3""",
                        (current["corpus_release_id"], focus_concept_id),
                    )
                )
            state = connection.execute(
                """SELECT mean FROM skill_states
                   WHERE learner_id = ? AND concept_id = ?""",
                (current["learner_id"], focus_concept_id),
            ).fetchone()
            if state:
                target_difficulty = max(-2.5, min(2.5, float(state["mean"])))
            else:
                prior = max(0.02, min(0.98, concept.prior_mastery))
                target_difficulty = max(-2.5, min(2.5, log(prior / (1.0 - prior))))

            if current["phase"] == SessionPhase.VERIFY.value:
                requested_kinds = ("transfer",)
            elif current["phase"] == SessionPhase.REMEDIATE.value:
                # A remediation gap requires both a misconception-sensitive
                # repair route and a separately authored transfer check.
                requested_kinds = ("debugging", "transfer")
            else:
                requested_kinds = ("application",)
            blueprint_payloads: list[dict[str, Any]] = []
            for kind in requested_kinds:
                blueprint_payloads.append(
                    asdict(
                        GenerationBlueprint(
                            concept_id=focus_concept_id,
                            concept_name=concept.name,
                            kind=kind,
                            target_difficulty=target_difficulty,
                            misconception_ids=misconception_ids,
                            source_ids=source_ids,
                            family_constraint=(
                                "Create a new independent family for this observed live "
                                "corpus gap; do not paraphrase any presented family."
                            ),
                        )
                    )
                )

            existing_jobs = {
                json.dumps(
                    json.loads(row["blueprint_json"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ): row["id"]
                for row in connection.execute(
                    """SELECT id, blueprint_json FROM generation_jobs
                       WHERE status IN ('planned', 'running')"""
                )
            }
            job_ids: list[str] = []
            for blueprint_payload in blueprint_payloads:
                blueprint_json = json.dumps(
                    blueprint_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                job_id = existing_jobs.get(blueprint_json)
                if job_id is None:
                    job_id = new_id("gen")
                    connection.execute(
                        """INSERT INTO generation_jobs(
                               id, blueprint_json, status, prompt_version,
                               created_at, updated_at
                           ) VALUES (?, ?, 'planned', ?, ?, ?)""",
                        (
                            job_id,
                            blueprint_json,
                            PROMPT_VERSION,
                            now.isoformat(),
                            now.isoformat(),
                        ),
                    )
                    existing_jobs[blueprint_json] = job_id
                job_ids.append(job_id)
            self.database.append_event(
                connection,
                stream_id=f"learner:{current['learner_id']}",
                event_type="CorpusGapDetected",
                payload={
                    "session_revision": current["revision"],
                    "phase": current["phase"],
                    "focus_concept_id": focus_concept_id,
                    "focus_misconception_id": focus_misconception_id,
                    "corpus_release_id": current["corpus_release_id"],
                    "message": message,
                    "job_id": job_ids[0],
                    "job_ids": job_ids,
                },
                metadata={
                    "policy_version": POLICY_VERSION,
                    "learner_model_version": MODEL_VERSION,
                    "corpus_release_id": current["corpus_release_id"],
                },
                learner_id=current["learner_id"],
                session_id=session_id,
                idempotency_key=idempotency_key,
                occurred_at=now,
            )

    def end_session(
        self,
        session_id: str,
        *,
        completed: bool | None = None,
        status: str | None = None,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if completed is not None and not isinstance(completed, bool):
            raise ValidationError("completed must be a boolean when provided.")
        resolved_status = status or (
            "completed" if completed is None or completed else "abandoned"
        )
        if status is not None and completed is not None and (
            (status == "completed") != completed
        ):
            raise ValidationError("completed and status specify different outcomes.")
        return self.database.end_session(
            session_id,
            status=resolved_status,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    def submit_answer(
        self,
        decision_id: str,
        selected_option_id: str | None,
        *,
        confidence: float | None = None,
        response_ms: int | None = None,
        hint_count: int = 0,
        feedback_shown: bool = True,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> SubmissionResult:
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValidationError("decision_id must be a non-blank string.")
        if selected_option_id is not None and (
            not isinstance(selected_option_id, str) or not selected_option_id.strip()
        ):
            raise ValidationError("selected_option_id must be a non-blank string or null.")
        if confidence is not None and (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not isfinite(float(confidence))
            or not (0.0 <= confidence <= 1.0)
        ):
            raise ValidationError("Confidence must be a finite number between 0 and 1.")
        if confidence is not None:
            # SQLite's REAL affinity persists accepted integer endpoints as
            # floats.  Canonicalize at the command boundary so the immutable
            # event, command hash, learner update, and stored attempt all use
            # the same numeric representation.
            confidence = float(confidence)
        if response_ms is not None and (
            not isinstance(response_ms, int) or isinstance(response_ms, bool) or response_ms < 0
        ):
            raise ValidationError("response_ms must be a non-negative integer.")
        if not isinstance(hint_count, int) or isinstance(hint_count, bool) or hint_count < 0:
            raise ValidationError("hint_count must be a non-negative integer.")
        if not isinstance(feedback_shown, bool):
            raise ValidationError("feedback_shown must be true or false.")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key) > 200
        ):
            raise ValidationError(
                "idempotency_key must be a non-blank string of at most 200 characters."
            )
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValidationError("now must be timezone-aware.")
        now = now.astimezone(timezone.utc)

        with self.database.transaction() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
            if not decision:
                raise NotFoundError(f"Unknown decision: {decision_id}")
            session = self.database.get_session(decision["session_id"], connection)
            question = self.database.get_question(decision["question_id"], connection)
            with_hash = connection.execute(
                "SELECT version, content_hash FROM questions WHERE id = ?",
                (decision["question_id"],),
            ).fetchone()
            if (
                not with_hash
                or with_hash["version"] != decision["question_version"]
                or with_hash["content_hash"] != decision["question_content_hash"]
            ):
                raise ConflictError(
                    "The selected question no longer matches its pinned content; repair the corpus."
                )
            options = {option.id: option for option in question.options}
            if selected_option_id is not None and selected_option_id not in options:
                raise ValidationError("Selected option was not part of the presented question.")
            selected_option = options.get(selected_option_id) if selected_option_id else None
            correct = bool(selected_option and selected_option.correct)
            presented_order = json.loads(decision["option_order_json"])
            command_payload = {
                "decision_id": decision_id,
                "question_id": decision["question_id"],
                "question_version": decision["question_version"],
                "selected_option_id": selected_option_id,
                "is_correct": correct,
                "confidence": confidence,
                "response_ms": response_ms,
                "hint_count": hint_count,
                "feedback_shown": feedback_shown,
                "presented_order": presented_order,
            }
            command_hash = _canonical_hash(command_payload)

            if idempotency_key:
                prior_event = connection.execute(
                    "SELECT * FROM events WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if prior_event:
                    if (
                        prior_event["event_type"] != "ResponseSubmitted"
                        or prior_event["learner_id"] != session["learner_id"]
                        or prior_event["session_id"] != session["id"]
                        or json.loads(prior_event["payload_json"]) != command_payload
                    ):
                        raise ConflictError(
                            "Idempotency key was reused with different answer inputs."
                        )
                    return self._result_for_attempt(
                        connection,
                        prior_event["event_id"],
                        command_hash=command_hash,
                        idempotent=True,
                    )

            if decision["consumed_at"]:
                raise ConflictError("This question decision has already been answered.")
            if decision["invalidated_at"]:
                raise ConflictError(
                    "This question decision was invalidated after the learner model changed; "
                    "request a new question."
                )
            revocation = connection.execute(
                "SELECT reason FROM question_revocations WHERE question_id = ?",
                (decision["question_id"],),
            ).fetchone()
            if revocation:
                raise ConflictError(
                    "This question was emergency-revoked and cannot contribute learner "
                    f"evidence: {revocation['reason']}"
                )
            if session["status"] != "active":
                raise ConflictError("Session is not active.")
            if (
                session["phase"] != decision["phase"]
                or session["focus_concept_id"] != decision["focus_concept_id"]
                or session["focus_misconception_id"] != decision["focus_misconception_id"]
                or session["corpus_release_id"] != decision["corpus_release_id"]
                or session["revision"] != decision["session_revision"] + 1
            ):
                raise ConflictError(
                    "Session state changed after this question was selected; request a new question."
                )
            learner = connection.execute(
                "SELECT revision FROM learners WHERE id = ?", (session["learner_id"],)
            ).fetchone()
            if not learner:
                raise ConflictError("Session learner no longer exists.")
            if learner["revision"] != decision["learner_revision"]:
                raise ConflictError(
                    "The learner model changed after this question was selected; "
                    "request a new question."
                )
            selected_at = datetime.fromisoformat(decision["created_at"])
            if now < selected_at:
                raise ValidationError("An answer cannot occur before its question was selected.")
            last_answer = connection.execute(
                "SELECT MAX(answered_at) AS answered_at FROM attempts WHERE learner_id = ?",
                (session["learner_id"],),
            ).fetchone()["answered_at"]
            if last_answer and now < datetime.fromisoformat(last_answer):
                raise ValidationError(
                    "Out-of-order answer time would invalidate the online learner projection."
                )
            event = self.database.append_event(
                connection,
                stream_id=f"learner:{session['learner_id']}",
                event_type="ResponseSubmitted",
                payload=command_payload,
                metadata={
                    "policy_version": decision["policy_version"],
                    "learner_model_version": MODEL_VERSION,
                    "corpus_release_id": decision["corpus_release_id"],
                    "question_content_hash": decision["question_content_hash"],
                    "question_status": decision["question_status"],
                    "evidence_weight": decision["evidence_weight"],
                    "selection_learner_revision": decision["learner_revision"],
                    "application_learner_revision": learner["revision"],
                },
                learner_id=session["learner_id"],
                session_id=session["id"],
                idempotency_key=idempotency_key,
                causation_id=decision_id,
                occurred_at=now,
            )
            interaction_id = new_id("att")
            connection.execute(
                """INSERT INTO attempts(
                       id, event_id, decision_id, session_id, learner_id,
                       question_id, question_version, family_id,
                       presented_order_json, selected_option_id, is_correct,
                       confidence, response_ms, hint_count, feedback_shown, answered_at,
                       command_hash, outcome_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
                (
                    interaction_id,
                    event["event_id"],
                    decision_id,
                    session["id"],
                    session["learner_id"],
                    question.id,
                    question.version,
                    question.family_id,
                    decision["option_order_json"],
                    selected_option_id,
                    int(correct),
                    confidence,
                    response_ms,
                    hint_count,
                    int(feedback_shown),
                    now.isoformat(),
                    command_hash,
                ),
            )

            _, changes = self.learner_model.update_from_response(
                connection,
                learner_id=session["learner_id"],
                question=question,
                selected_option=selected_option,
                confidence=confidence,
                hint_count=hint_count,
                feedback_shown=feedback_shown,
                evidence_weight_override=decision["evidence_weight"],
                event_id=event["event_id"],
                now=now,
                response_ms=response_ms,
            )
            transition = self._transition(
                connection,
                session=session,
                question=question,
                selected_option=selected_option,
                decision=dict(decision),
                hint_count=hint_count,
                confidence=confidence,
                response_ms=response_ms,
                now=now,
            )
            recent_families = (session["recent_families"] + [question.family_id])[-6:]
            updated_session = connection.execute(
                """UPDATE sessions SET phase = ?, focus_concept_id = ?,
                       focus_misconception_id = ?, remediation_depth = ?,
                       remediation_path_json = ?, recent_families_json = ?,
                       revision = revision + 1, updated_at = ?
                   WHERE id = ? AND revision = ? AND status = 'active'""",
                (
                    transition["phase"].value,
                    transition["focus_concept_id"],
                    transition["focus_misconception_id"],
                    transition["remediation_depth"],
                    json.dumps(
                        transition["remediation_path"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    json.dumps(recent_families),
                    now.isoformat(),
                    session["id"],
                    session["revision"],
                ),
            )
            if updated_session.rowcount != 1:
                raise ConflictError("Session changed while applying this response.")
            updated_learner = connection.execute(
                "UPDATE learners SET revision = revision + 1 WHERE id = ? AND revision = ?",
                (session["learner_id"], learner["revision"]),
            )
            if updated_learner.rowcount != 1:
                raise ConflictError("Learner projection changed while applying this response.")
            consumed = connection.execute(
                """UPDATE decisions SET consumed_at = ?
                   WHERE id = ? AND consumed_at IS NULL
                     AND invalidated_at IS NULL""",
                (now.isoformat(), decision_id),
            )
            if consumed.rowcount != 1:
                raise ConflictError("This question decision was answered concurrently.")
            connection.execute(
                """UPDATE item_stats SET exposures = exposures + 1,
                       correct_count = correct_count + ?,
                       total_response_ms = total_response_ms + ?
                   WHERE question_id = ?""",
                (int(correct), response_ms or 0, question.id),
            )
            projection_hash = self.database.learner_projection_hash(
                session["learner_id"], connection
            )
            projection_event = self.database.append_event(
                connection,
                stream_id=f"learner:{session['learner_id']}",
                event_type="LearnerProjectionAdvanced",
                schema_version=2,
                payload={
                    "response_event_id": event["event_id"],
                    "state_changes": changes,
                    "phase": transition["phase"].value,
                    "focus_concept_id": transition["focus_concept_id"],
                    "focus_misconception_id": transition["focus_misconception_id"],
                    "remediation_depth": transition["remediation_depth"],
                    "remediation_path": transition["remediation_path"],
                    "transition_reason": transition["transition_reason"],
                    "boundary_decision": transition["boundary_decision"],
                    "corpus_release_id": decision["corpus_release_id"],
                    "learner_revision": learner["revision"] + 1,
                    "projection_hash": projection_hash,
                },
                metadata={
                    "learner_model_version": MODEL_VERSION,
                    "corpus_release_id": decision["corpus_release_id"],
                    "evidence_weight": decision["evidence_weight"],
                },
                learner_id=session["learner_id"],
                session_id=session["id"],
                causation_id=event["event_id"],
                occurred_at=now,
            )
            transition_changed = (
                transition["phase"].value != session["phase"]
                or transition["focus_concept_id"] != session["focus_concept_id"]
                or transition["focus_misconception_id"]
                != session["focus_misconception_id"]
                or transition["remediation_depth"] != session["remediation_depth"]
                or transition["remediation_path"] != session["remediation_path"]
            )
            if transition_changed:
                self.database.append_event(
                    connection,
                    stream_id=f"learner:{session['learner_id']}",
                    event_type="RemediationTransitioned",
                    schema_version=2,
                    payload={
                        "from_phase": session["phase"],
                        "to_phase": transition["phase"].value,
                        "focus_concept_id": transition["focus_concept_id"],
                        "focus_misconception_id": transition["focus_misconception_id"],
                        "remediation_depth": transition["remediation_depth"],
                        "remediation_path": transition["remediation_path"],
                        "transition_reason": transition["transition_reason"],
                        "boundary_decision": transition["boundary_decision"],
                        "pedagogical_role": decision["pedagogical_role"],
                        "focus_valid": bool(decision["focus_valid"]),
                        "unguided": hint_count == 0,
                    },
                    metadata={
                        "policy_version": POLICY_VERSION,
                        "corpus_release_id": decision["corpus_release_id"],
                    },
                    learner_id=session["learner_id"],
                    session_id=session["id"],
                    causation_id=projection_event["event_id"],
                    occurred_at=now,
                )
            result = SubmissionResult(
                interaction_id=interaction_id,
                correct=correct,
                selected_option=selected_option,
                correct_option=question.correct_option,
                next_phase=transition["phase"],
                focus_concept_id=transition["focus_concept_id"],
                focus_misconception_id=transition["focus_misconception_id"],
                state_changes=tuple(changes),
                transition_reason=transition["transition_reason"],
                boundary_decision=transition["boundary_decision"],
            )
            outcome = self._outcome_payload(result)
            stored_outcome = connection.execute(
                """UPDATE attempts SET outcome_json = ?
                   WHERE id = ? AND outcome_json IS NULL""",
                (json.dumps(outcome, sort_keys=True, separators=(",", ":")), interaction_id),
            )
            if stored_outcome.rowcount != 1:
                raise ConflictError("Attempt outcome could not be finalized atomically.")
            return result

    def _transition(
        self,
        connection: sqlite3.Connection,
        *,
        session: dict[str, Any],
        question,
        selected_option: Option | None,
        decision: dict[str, Any],
        hint_count: int,
        confidence: float | None,
        response_ms: int | None,
        now: datetime,
    ) -> dict[str, Any]:
        current = SessionPhase(session["phase"])
        correct = bool(selected_option and selected_option.correct)
        remediation_path = [dict(frame) for frame in session["remediation_path"]]
        focus_snapshot_matches = (
            decision["phase"] == session["phase"]
            and decision["focus_concept_id"] == session["focus_concept_id"]
            and decision["focus_misconception_id"] == session["focus_misconception_id"]
        )
        credible_retrieval = (
            hint_count == 0
            and (confidence is None or confidence >= 0.50)
            and (response_ms is None or response_ms >= 250)
        )
        valid_unguided_focus_evidence = (
            bool(decision["focus_valid"])
            and focus_snapshot_matches
            and credible_retrieval
            and (
                (current == SessionPhase.REMEDIATE and decision["pedagogical_role"] == "remediation_probe")
                or (current == SessionPhase.VERIFY and decision["pedagogical_role"] == "verification")
            )
        )
        if correct:
            if current in {
                SessionPhase.LEARN,
                SessionPhase.DIAGNOSE,
                SessionPhase.REVIEW,
            } and not credible_retrieval:
                # A guessed, hinted, low-confidence, or implausibly fast success
                # is useful evidence but not certification. Route to an
                # independent transfer check instead of silently moving on.
                return {
                    "phase": SessionPhase.VERIFY,
                    "focus_concept_id": question.primary_concept_id,
                    "focus_misconception_id": None,
                    "remediation_depth": 1,
                    "remediation_path": remediation_path,
                    "transition_reason": "noncredible_success_requires_verification",
                    "boundary_decision": None,
                }
            if current == SessionPhase.REMEDIATE and valid_unguided_focus_evidence:
                return {
                    "phase": SessionPhase.VERIFY,
                    "focus_concept_id": session["focus_concept_id"] or question.primary_concept_id,
                    "focus_misconception_id": session["focus_misconception_id"],
                    "remediation_depth": session["remediation_depth"],
                    "remediation_path": remediation_path,
                    "transition_reason": "focused_repair_requires_independent_verification",
                    "boundary_decision": None,
                }
            if current == SessionPhase.VERIFY and valid_unguided_focus_evidence:
                if remediation_path:
                    parent = remediation_path.pop()
                    parent_capacity = self._fresh_focus_capacity(
                        connection,
                        session_id=session["id"],
                        release_id=decision["corpus_release_id"],
                        concept_ids={parent["concept_id"]},
                    ).get(parent["concept_id"], {})
                    if not parent_capacity.get("verification_families"):
                        # The prerequisite gain remains in the projection, but
                        # a same-session transfer claim would be unverifiable.
                        # Defer the parent instead of selecting into a known
                        # corpus dead end.
                        return {
                            "phase": _main_phase(session["mode"]),
                            "focus_concept_id": None,
                            "focus_misconception_id": None,
                            "remediation_depth": 0,
                            "remediation_path": [],
                            "transition_reason": "prerequisite_verified_parent_deferred",
                            "boundary_decision": None,
                        }
                    # Repairing and independently verifying the prerequisite is
                    # itself the instructional intervention. Recheck transfer at
                    # the unresolved parent directly; routing through another
                    # parent repair first would consume two more families and can
                    # strand an otherwise serviceable four-family objective.
                    return {
                        "phase": SessionPhase.VERIFY,
                        "focus_concept_id": parent["concept_id"],
                        "focus_misconception_id": parent.get("misconception_id"),
                        "remediation_depth": max(
                            1, session["remediation_depth"] - 1
                        ),
                        "remediation_path": remediation_path,
                        "transition_reason": "prerequisite_verified_resume_parent",
                        "boundary_decision": None,
                    }
                focus_misconception = session["focus_misconception_id"]
                residual_hypothesis = False
                if focus_misconception:
                    belief = connection.execute(
                        """SELECT log_odds FROM misconception_beliefs
                           WHERE learner_id = ? AND misconception_id = ?""",
                        (session["learner_id"], focus_misconception),
                    ).fetchone()
                    residual_hypothesis = bool(
                        belief
                        and 1.0 / (1.0 + exp(-belief["log_odds"])) >= 0.35
                    )
                # One credible independent verification closes the bounded
                # teaching episode. A still-active posterior is deliberately
                # retained for later need/review scoring instead of opening an
                # unbounded repair/verify loop in the same session.
                return {
                    "phase": _main_phase(session["mode"]),
                    "focus_concept_id": None,
                    "focus_misconception_id": None,
                    "remediation_depth": 0,
                    "remediation_path": [],
                    "transition_reason": (
                        "independent_verification_residual_hypothesis"
                        if residual_hypothesis
                        else "independent_verification_completed"
                    ),
                    "boundary_decision": None,
                }
            if current == SessionPhase.VERIFY:
                # The independent check was completed, but a hinted,
                # low-confidence, implausibly fast, or stale-focus success still
                # cannot certify retrieval. Do not consume an unbounded number
                # of independent families waiting for response behavior to
                # change. Return to the main phase with the unresolved
                # projection intact so later need/review scoring can revisit it.
                return {
                    "phase": _main_phase(session["mode"]),
                    "focus_concept_id": None,
                    "focus_misconception_id": None,
                    "remediation_depth": 0,
                    "remediation_path": [],
                    "transition_reason": "noncredible_verification_bounded_exit",
                    "boundary_decision": None,
                }
            if current in {
                SessionPhase.LEARN,
                SessionPhase.DIAGNOSE,
                SessionPhase.REVIEW,
            }:
                return {
                    "phase": current,
                    "focus_concept_id": None,
                    "focus_misconception_id": None,
                    "remediation_depth": 0,
                    "remediation_path": [],
                    "transition_reason": "credible_main_success",
                    "boundary_decision": None,
                }
            # A hinted or stale-focus repair success can teach, but cannot
            # certify that the active gap has been repaired. Permit one more
            # independent repair probe, then leave the bounded tunnel.
            next_depth = session["remediation_depth"]
            if current == SessionPhase.REMEDIATE:
                next_depth += 1
                if next_depth >= MAX_REMEDIATION_DEPTH:
                    return {
                        "phase": _main_phase(session["mode"]),
                        "focus_concept_id": None,
                        "focus_misconception_id": None,
                        "remediation_depth": 0,
                        "remediation_path": [],
                        "transition_reason": "noncredible_repair_bounded_exit",
                        "boundary_decision": None,
                    }
            return {
                "phase": current,
                "focus_concept_id": session["focus_concept_id"],
                "focus_misconception_id": session["focus_misconception_id"],
                "remediation_depth": next_depth,
                "remediation_path": remediation_path,
                "transition_reason": "noncredible_repair_requires_another_probe",
                "boundary_decision": None,
            }

        next_depth = (
            session["remediation_depth"] + 1
            if current in {SessionPhase.REMEDIATE, SessionPhase.VERIFY}
            else 1
        )
        focus_concept = session["focus_concept_id"] or question.primary_concept_id
        focus_misconception = session["focus_misconception_id"]
        if current not in {SessionPhase.REMEDIATE, SessionPhase.VERIFY}:
            focus_misconception = (
                selected_option.misconception_id
                if selected_option
                and selected_option.misconception_id
                and (confidence is None or confidence >= 0.35)
                and (response_ms is None or response_ms >= 250)
                else None
            )
            if focus_misconception:
                owner = connection.execute(
                    "SELECT concept_id FROM misconceptions WHERE id = ?",
                    (focus_misconception,),
                ).fetchone()
                mapped_concepts = {
                    mapping.concept_id for mapping in question.concepts
                }
                if owner and owner["concept_id"] in mapped_concepts:
                    focus_concept = owner["concept_id"]
        elif not focus_misconception and selected_option:
            focus_misconception = selected_option.misconception_id
        descended_to_prerequisite = False
        boundary_decision_payload: dict[str, Any] | None = None
        verified_prerequisites: set[str] = set()
        unserviceable_prerequisites: set[str] = set()
        if next_depth == 2 and focus_concept:
            graph = self.database.get_graph(decision["corpus_release_id"])
            direct_prerequisite_ids = {
                concept_id
                for concept_id, _ in graph.direct_prerequisites(focus_concept)
            }
            if direct_prerequisite_ids:
                placeholders = ",".join("?" for _ in direct_prerequisite_ids)
                verified_rows = connection.execute(
                    f"""SELECT mapping.concept_id,
                               COUNT(DISTINCT CASE
                                   WHEN choice.pedagogical_role='remediation_probe'
                                    AND attempt.is_correct=1
                                    AND attempt.hint_count=0
                                    AND (attempt.confidence IS NULL
                                         OR attempt.confidence >= 0.50)
                                    AND (attempt.response_ms IS NULL
                                         OR attempt.response_ms >= 250)
                                   THEN attempt.family_id END) AS repair_families,
                               COUNT(DISTINCT CASE
                                   WHEN choice.pedagogical_role='verification'
                                    AND attempt.is_correct=1
                                    AND attempt.hint_count=0
                                    AND (attempt.confidence IS NULL
                                         OR attempt.confidence >= 0.50)
                                    AND (attempt.response_ms IS NULL
                                         OR attempt.response_ms >= 250)
                                   THEN attempt.family_id END) AS verification_families
                        FROM attempts attempt
                        JOIN decisions choice ON choice.id=attempt.decision_id
                        JOIN question_concepts mapping
                          ON mapping.question_id=attempt.question_id
                         AND mapping.role='primary'
                        WHERE attempt.session_id=?
                          AND mapping.concept_id IN ({placeholders})
                        GROUP BY mapping.concept_id""",
                    (session["id"], *sorted(direct_prerequisite_ids)),
                ).fetchall()
                verified_prerequisites = {
                    row["concept_id"]
                    for row in verified_rows
                    if int(row["repair_families"]) >= 1
                    and int(row["verification_families"]) >= 1
                }
                focus_capacity = self._fresh_focus_capacity(
                    connection,
                    session_id=session["id"],
                    release_id=decision["corpus_release_id"],
                    concept_ids=direct_prerequisite_ids,
                )
                for concept_id in direct_prerequisite_ids:
                    capacity = focus_capacity.get(concept_id, {})
                    families = set(capacity.get("families", set()))
                    verification_families = set(
                        capacity.get("verification_families", set())
                    )
                    if not any(
                        verification_families - {repair_family}
                        for repair_family in families
                    ):
                        unserviceable_prerequisites.add(concept_id)
            performance_rows = connection.execute(
                """SELECT mapping.concept_id, COUNT(*) AS attempted,
                          SUM(CASE WHEN attempt.is_correct = 0 THEN 1 ELSE 0 END)
                              AS incorrect
                   FROM attempts attempt
                   JOIN question_concepts mapping
                     ON mapping.question_id = attempt.question_id
                    AND mapping.role = 'primary'
                   WHERE attempt.session_id = ?
                   GROUP BY mapping.concept_id""",
                (session["id"],),
            ).fetchall()
            performance = {
                row["concept_id"]: (
                    int(row["attempted"]),
                    int(row["incorrect"]),
                )
                for row in performance_rows
            }
            boundary = self.boundary_planner.choose_direct_boundary(
                learner_id=session["learner_id"],
                focus_concept_id=focus_concept,
                graph=graph,
                stored_states=self.database.get_skill_states(
                    session["learner_id"], connection
                ),
                now=now,
                recent_performance=performance,
                excluded_concept_ids=(
                    verified_prerequisites | unserviceable_prerequisites
                ),
            )
            if boundary is not None:
                prerequisite = boundary.selected_concept_id
                boundary_decision_payload = boundary.terms()
                if prerequisite != focus_concept:
                    remediation_path.append(
                        {
                            "concept_id": focus_concept,
                            "misconception_id": focus_misconception,
                        }
                    )
                    focus_concept = prerequisite
                    descended_to_prerequisite = True
                # The original misconception is evidence about its owning
                # concept, not automatically about the prerequisite we step to.
                focus_misconception = None
        if next_depth >= MAX_REMEDIATION_DEPTH or (
            next_depth == 2 and not descended_to_prerequisite
        ):
            # Avoid an infinite failure tunnel. The unresolved gap remains in the
            # learner projection and will re-enter via future need/review scores.
            # A second cross-family failure with no deeper prerequisite to test
            # is already sufficient to stop drilling the same hypothesis.
            return {
                "phase": _main_phase(session["mode"]),
                "focus_concept_id": None,
                "focus_misconception_id": None,
                "remediation_depth": 0,
                "remediation_path": [],
                "transition_reason": (
                    "no_serviceable_prerequisite_boundary"
                    if unserviceable_prerequisites
                    else (
                        "verified_prerequisite_not_reopened"
                        if verified_prerequisites
                        else "bounded_failure_exit"
                    )
                ),
                "boundary_decision": boundary_decision_payload,
            }
        return {
            "phase": SessionPhase.REMEDIATE,
            "focus_concept_id": focus_concept,
            "focus_misconception_id": focus_misconception,
            "remediation_depth": next_depth,
            "remediation_path": remediation_path,
            "transition_reason": (
                "descend_to_evidence_boundary"
                if descended_to_prerequisite
                else "incorrect_answer_focus"
            ),
            "boundary_decision": boundary_decision_payload,
        }

    def _fresh_focus_capacity(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        release_id: str,
        concept_ids: set[str],
    ) -> dict[str, dict[str, set[str]]]:
        """Return unseen family capacity for bounded repair and verification.

        Selection never repeats a question or independence family within a
        session. The recursive planner must therefore prove that the boundary
        it chooses still has both a repair probe and a distinct transfer check;
        otherwise the next call would enter a corpus gap by construction.
        """

        if not concept_ids:
            return {}
        placeholders = ",".join("?" for _ in concept_ids)
        rows = connection.execute(
            f"""SELECT mapping.concept_id, question.family_id, question.kind
                FROM question_concepts mapping
                JOIN questions question ON question.id = mapping.question_id
                JOIN release_questions release_question
                  ON release_question.question_id = question.id
                 AND release_question.release_id = ?
                WHERE mapping.role = 'primary'
                  AND mapping.concept_id IN ({placeholders})
                  AND release_question.status IN (?, ?)
                  AND NOT EXISTS (
                      SELECT 1 FROM question_revocations revoked
                      WHERE revoked.question_id = question.id
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM decisions seen
                      JOIN questions seen_question
                        ON seen_question.id = seen.question_id
                      WHERE seen.session_id = ?
                        AND (
                            seen.question_id = question.id
                            OR seen_question.family_id = question.family_id
                        )
                  )""",
            (
                release_id,
                *sorted(concept_ids),
                QuestionStatus.APPROVED.value,
                QuestionStatus.CALIBRATED.value,
                session_id,
            ),
        ).fetchall()
        result: dict[str, dict[str, set[str]]] = {
            concept_id: {"families": set(), "verification_families": set()}
            for concept_id in concept_ids
        }
        verification_kinds = {
            "application",
            "calculation",
            "comparison",
            "counterfactual",
            "debugging",
            "transfer",
        }
        for row in rows:
            result[row["concept_id"]]["families"].add(row["family_id"])
            if row["kind"] in verification_kinds:
                result[row["concept_id"]]["verification_families"].add(
                    row["family_id"]
                )
        return result

    def profile(
        self, learner_id: str, *, root_concept_id: str | None = None, now: datetime | None = None
    ) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValidationError("now must be timezone-aware.")
        now = now.astimezone(timezone.utc)
        with self.database.read() as connection:
            if not connection.execute(
                "SELECT 1 FROM learners WHERE id = ?", (learner_id,)
            ).fetchone():
                raise NotFoundError(f"Unknown learner: {learner_id}")
        graph = self.database.get_graph()
        resolved_target: dict[str, Any] | None = None
        if root_concept_id:
            try:
                topic = self.database.resolve_topic(root_concept_id)
            except NotFoundError:
                topic = None
            if topic is not None:
                scope = self.database.topic_scope(
                    topic["id"], topic["release_id"]
                )
                resolved_target = {
                    "type": "topic",
                    "id": topic["id"],
                    "name": topic["name"],
                }
            else:
                scope = graph.learning_scope(root_concept_id)
                resolved_target = {
                    "type": "concept",
                    "id": root_concept_id,
                    "name": graph.concepts[root_concept_id].name,
                }
        else:
            scope = set(graph.concepts)
        stored = self.database.get_skill_states(learner_id)
        readiness = self.boundary_planner.readiness_map(
            learner_id=learner_id,
            graph=graph,
            stored_states=stored,
            now=now,
            concept_ids=scope,
        )
        evidence_summaries = self.database.independent_evidence_summaries(learner_id, scope)
        beliefs = self.database.get_misconception_beliefs(learner_id)
        active_release = self.database.get_active_release_id()
        definitions = {
            m.id: m
            for m in self.database.get_misconceptions(
                set(beliefs), release_id=active_release
            )
        }
        misconception_by_concept: dict[str, float] = {}
        for misconception_id, belief in beliefs.items():
            definition = definitions.get(misconception_id)
            if definition:
                misconception_by_concept[definition.concept_id] = max(
                    misconception_by_concept.get(definition.concept_id, 0.0),
                    belief.probability,
                )
        projected_states = {}
        for concept_id in scope:
            concept = graph.concepts[concept_id]
            state = stored.get(concept_id) or self.learner_model.initial_state(
                learner_id, concept
            )
            projected_states[concept_id] = self.learner_model.project_state(
                state, concept, now
            )
        skills = []
        for concept_id in sorted(scope, key=lambda cid: graph.concepts[cid].name):
            concept = graph.concepts[concept_id]
            projected = projected_states[concept_id]
            boundary = readiness[concept_id]
            evidence = evidence_summaries[concept_id]
            family_count = evidence["families"]
            delayed_retrievals = evidence["delayed"]
            operation_kinds = evidence["operation_kinds"]
            direct_prerequisites = [
                prerequisite_id
                for prerequisite_id, _ in graph.direct_prerequisites(concept_id)
            ]
            prerequisites_ready = all(
                prerequisite_id in projected_states
                and projected_states[prerequisite_id].exposures > 0
                and projected_states[prerequisite_id].mastery_probability >= 0.40
                for prerequisite_id in direct_prerequisites
            )
            skills.append(
                {
                    "concept_id": concept_id,
                    "name": concept.name,
                    "mastery": projected.mastery_probability,
                    "expected_competence": projected.expected_competence,
                    "uncertainty": projected.variance**0.5,
                    "stability_hours": projected.stability_hours,
                    "evidence_mass": projected.evidence_mass,
                    "independent_families": family_count,
                    "delayed_retrievals": delayed_retrievals,
                    "operation_kinds": operation_kinds,
                    "prerequisites_ready": prerequisites_ready,
                    "intrinsic_readiness": boundary.intrinsic_readiness,
                    "prerequisite_support": boundary.prerequisite_support,
                    "effective_readiness": boundary.effective_readiness,
                    "bottleneck_concept_id": boundary.bottleneck_concept_id,
                    "bottleneck_name": (
                        graph.concepts[boundary.bottleneck_concept_id].name
                        if boundary.bottleneck_concept_id
                        else None
                    ),
                    "state": self.learner_model.mastery_label(
                        projected,
                        family_count,
                        delayed_retrievals,
                        operation_kinds,
                        misconception_by_concept.get(concept_id, 0.0),
                        prerequisites_ready,
                    ),
                    "next_review_at": projected.next_review_at.isoformat() if projected.next_review_at else None,
                }
            )
        misconceptions = [
            {
                "misconception_id": misconception_id,
                "name": definitions[misconception_id].name if misconception_id in definitions else misconception_id,
                "probability": belief.probability,
                "evidence_count": belief.evidence_count,
            }
            for misconception_id, belief in beliefs.items()
            if belief.probability >= 0.12 and misconception_id in definitions
        ]
        misconceptions.sort(key=lambda item: (-item["probability"], item["misconception_id"]))
        return {
            "learner_id": learner_id,
            "target": resolved_target,
            "boundary_algorithm_version": BOUNDARY_ALGORITHM_VERSION,
            "skills": skills,
            "active_misconceptions": misconceptions,
        }

    def session_report(
        self, session_id: str, *, now: datetime | None = None
    ) -> dict[str, Any]:
        """Summarize one session without collapsing learner uncertainty."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValidationError("now must be timezone-aware.")
        now = now.astimezone(timezone.utc)
        session = self.database.get_session(session_id)
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT decision.id AS decision_id, decision.question_id,
                          decision.phase, decision.pedagogical_role,
                          decision.focus_concept_id AS focus_concept_before,
                          decision.focus_misconception_id
                              AS focus_misconception_before,
                          decision.selected_score_json, decision.created_at,
                          question.family_id, question.difficulty, question.kind,
                          primary_mapping.concept_id AS primary_concept_id,
                          attempt.id AS attempt_id, attempt.is_correct,
                          attempt.selected_option_id, attempt.confidence,
                          attempt.response_ms, attempt.hint_count,
                          attempt.answered_at, attempt.outcome_json,
                          selected_option.misconception_id
                              AS selected_misconception_id,
                          (SELECT COUNT(*) FROM release_question_topics topic_map
                           WHERE topic_map.release_id = decision.corpus_release_id
                             AND topic_map.question_id = decision.question_id
                          ) AS topic_count
                   FROM decisions decision
                   JOIN questions question ON question.id = decision.question_id
                   JOIN question_concepts primary_mapping
                     ON primary_mapping.question_id = decision.question_id
                    AND primary_mapping.role = 'primary'
                   LEFT JOIN attempts attempt
                     ON attempt.decision_id = decision.id
                   LEFT JOIN options selected_option
                     ON selected_option.question_id = attempt.question_id
                    AND selected_option.option_id = attempt.selected_option_id
                   WHERE decision.session_id = ?
                     AND decision.invalidated_at IS NULL
                   ORDER BY decision.created_at, decision.id""",
                (session_id,),
            ).fetchall()
            topic_counts = connection.execute(
                """SELECT topic.topic_id, topic.name, COUNT(*) AS n
                   FROM decisions decision
                   JOIN release_question_topics mapping
                     ON mapping.release_id = decision.corpus_release_id
                    AND mapping.question_id = decision.question_id
                    AND mapping.relation = 'primary'
                   JOIN release_topics topic
                     ON topic.release_id = mapping.release_id
                    AND topic.topic_id = mapping.topic_id
                   WHERE decision.session_id = ?
                     AND decision.invalidated_at IS NULL
                     AND decision.consumed_at IS NOT NULL
                   GROUP BY topic.topic_id, topic.name
                   ORDER BY n DESC, topic.name""",
                (session_id,),
            ).fetchall()

        answered = [row for row in rows if row["attempt_id"] is not None]
        difficulties = [float(row["difficulty"]) for row in answered]
        response_times = [
            int(row["response_ms"])
            for row in answered
            if row["response_ms"] is not None
        ]
        predicted = []
        continuity = []
        for row in answered:
            score = json.loads(row["selected_score_json"])
            if isinstance(score.get("predicted_correct"), (int, float)):
                predicted.append(float(score["predicted_correct"]))
            if isinstance(score.get("continuity"), (int, float)):
                continuity.append(float(score["continuity"]))

        state_by_concept: dict[str, dict[str, float]] = {}
        total_evidence_delta = 0.0
        for row in answered:
            if not row["outcome_json"]:
                continue
            outcome = json.loads(row["outcome_json"])
            for change in outcome.get("state_changes", []):
                concept_id = change.get("concept_id")
                if not isinstance(concept_id, str):
                    continue
                evidence_delta = float(change.get("evidence_delta", 0.0))
                total_evidence_delta += evidence_delta
                summary = state_by_concept.setdefault(
                    concept_id,
                    {
                        "prior_mastery": float(
                            change.get("prior_mastery", 0.0)
                        ),
                        "posterior_mastery": float(
                            change.get("posterior_mastery", 0.0)
                        ),
                        "evidence_delta": 0.0,
                    },
                )
                summary["posterior_mastery"] = float(
                    change.get(
                        "posterior_mastery", summary["posterior_mastery"]
                    )
                )
                summary["evidence_delta"] += evidence_delta

        started_at = datetime.fromisoformat(session["created_at"])
        ended_at = (
            datetime.fromisoformat(session["updated_at"])
            if session["status"] != "active"
            else now
        )
        wall_seconds = max(0.0, (ended_at - started_at).total_seconds())
        correct_count = sum(bool(row["is_correct"]) for row in answered)
        abstained = sum(row["selected_option_id"] is None for row in answered)
        exploration_rows = [
            row
            for row in answered
            if row["pedagogical_role"] == "exploration_probe"
        ]
        topic = (
            self.database.resolve_topic(
                session["topic_id"], session["corpus_release_id"]
            )
            if session.get("topic_id")
            else None
        )
        difficulty_bands = Counter(
            "introductory"
            if value < -0.5
            else "advanced"
            if value > 0.75
            else "intermediate"
            for value in difficulties
        )
        concept_changes = [
            {
                "concept_id": concept_id,
                **values,
                "mastery_change": values["posterior_mastery"]
                - values["prior_mastery"],
            }
            for concept_id, values in state_by_concept.items()
        ]
        concept_changes.sort(
            key=lambda item: (-abs(item["evidence_delta"]), item["concept_id"])
        )

        graph = self.database.get_graph(session["corpus_release_id"])
        catalog = self.database.get_catalog(session["corpus_release_id"])
        topic_by_concept = {
            concept["id"]: {"id": topic_row["id"], "name": topic_row["name"]}
            for topic_row in catalog["topics"]
            for concept in topic_row["concepts"]
        }
        beliefs = self.database.get_misconception_beliefs(session["learner_id"])
        selected_misconception_ids = {
            row["selected_misconception_id"]
            for row in answered
            if row["selected_misconception_id"]
        }
        definition_ids = set(beliefs) | selected_misconception_ids
        definitions = {
            item.id: item
            for item in self.database.get_misconceptions(
                definition_ids,
                release_id=session["corpus_release_id"],
            )
        }

        adaptive_path: list[dict[str, Any]] = []
        boundary_concepts: set[str] = set()
        for index, row in enumerate(answered, start=1):
            outcome = json.loads(row["outcome_json"]) if row["outcome_json"] else {}
            after_focus = outcome.get("focus_concept_id")
            boundary_decision = outcome.get("boundary_decision")
            if isinstance(boundary_decision, dict):
                selected_boundary = boundary_decision.get("selected_concept_id")
                if isinstance(selected_boundary, str):
                    boundary_concepts.add(selected_boundary)
            transition_reason = outcome.get(
                "transition_reason", "legacy_transition"
            )
            selected_misconception_id = row["selected_misconception_id"]
            adaptive_path.append(
                {
                    "step": index,
                    "question_id": row["question_id"],
                    "family_id": row["family_id"],
                    "pedagogical_role": row["pedagogical_role"],
                    "difficulty": float(row["difficulty"]),
                    "from_phase": row["phase"],
                    "to_phase": outcome.get("next_phase", row["phase"]),
                    "correct": bool(row["is_correct"]),
                    "primary_concept_id": row["primary_concept_id"],
                    "primary_concept_name": graph.concepts[
                        row["primary_concept_id"]
                    ].name,
                    "focus_before": row["focus_concept_before"],
                    "focus_before_name": (
                        graph.concepts[row["focus_concept_before"]].name
                        if row["focus_concept_before"]
                        else None
                    ),
                    "focus_after": after_focus,
                    "focus_after_name": (
                        graph.concepts[after_focus].name
                        if isinstance(after_focus, str) and after_focus in graph.concepts
                        else None
                    ),
                    "selected_misconception_id": selected_misconception_id,
                    "selected_misconception_name": (
                        definitions[selected_misconception_id].name
                        if selected_misconception_id in definitions
                        else None
                    ),
                    "transition_reason": transition_reason,
                    "boundary_decision": boundary_decision,
                }
            )
        transition_counts = Counter(
            step["transition_reason"] for step in adaptive_path
        )
        current_focused_run = 0
        maximum_focused_run = 0
        for step in adaptive_path:
            if step["from_phase"] in {
                SessionPhase.REMEDIATE.value,
                SessionPhase.VERIFY.value,
            }:
                current_focused_run += 1
                maximum_focused_run = max(
                    maximum_focused_run, current_focused_run
                )
            else:
                current_focused_run = 0
        adaptive_routing = {
            "algorithm_version": BOUNDARY_ALGORITHM_VERSION,
            "transition_counts": dict(transition_counts),
            "prerequisite_descents": transition_counts[
                "descend_to_evidence_boundary"
            ],
            "prerequisite_resumptions": transition_counts[
                "prerequisite_verified_resume_parent"
            ],
            "prevented_reopenings": transition_counts[
                "verified_prerequisite_not_reopened"
            ],
            "capacity_exits": (
                transition_counts["no_serviceable_prerequisite_boundary"]
                + transition_counts["prerequisite_verified_parent_deferred"]
            ),
            "bounded_exits": sum(
                count
                for reason, count in transition_counts.items()
                if "bounded" in reason
                or reason
                in {
                    "verified_prerequisite_not_reopened",
                    "no_serviceable_prerequisite_boundary",
                    "prerequisite_verified_parent_deferred",
                }
            ),
            "maximum_consecutive_focused_questions": maximum_focused_run,
        }

        seen_concepts = {
            row["primary_concept_id"] for row in answered
        } | boundary_concepts
        stored_states = self.database.get_skill_states(session["learner_id"])
        current_readiness = self.boundary_planner.readiness_map(
            learner_id=session["learner_id"],
            graph=graph,
            stored_states=stored_states,
            now=now,
            concept_ids=seen_concepts,
        ) if seen_concepts else {}
        evidence = self.database.independent_evidence_summaries(
            session["learner_id"], seen_concepts
        )
        session_by_concept: dict[str, dict[str, Any]] = {}
        for row in answered:
            concept_id = row["primary_concept_id"]
            summary = session_by_concept.setdefault(
                concept_id,
                {
                    "attempted": 0,
                    "correct": 0,
                    "abstained": 0,
                    "uncertain_responses": 0,
                    "remediation_questions": 0,
                    "verification_failures": 0,
                    "difficulties": [],
                    "misconception_signals": Counter(),
                },
            )
            summary["attempted"] += 1
            summary["correct"] += int(bool(row["is_correct"]))
            summary["abstained"] += int(row["selected_option_id"] is None)
            summary["uncertain_responses"] += int(
                row["selected_option_id"] is None
                or row["hint_count"] > 0
                or (
                    row["confidence"] is not None
                    and row["confidence"] < 0.50
                )
                or (
                    row["response_ms"] is not None
                    and row["response_ms"] < 250
                )
            )
            summary["remediation_questions"] += int(
                row["pedagogical_role"]
                in {"remediation_probe", "verification"}
            )
            summary["verification_failures"] += int(
                row["pedagogical_role"] == "verification"
                and not bool(row["is_correct"])
            )
            summary["difficulties"].append(float(row["difficulty"]))
            if row["selected_misconception_id"]:
                summary["misconception_signals"][
                    row["selected_misconception_id"]
                ] += 1

        concept_performance: list[dict[str, Any]] = []
        for concept_id in sorted(
            seen_concepts, key=lambda item: graph.concepts[item].name
        ):
            observed = session_by_concept.get(
                concept_id,
                {
                    "attempted": 0,
                    "correct": 0,
                    "abstained": 0,
                    "uncertain_responses": 0,
                    "remediation_questions": 0,
                    "verification_failures": 0,
                    "difficulties": [],
                    "misconception_signals": Counter(),
                },
            )
            readiness = current_readiness[concept_id]
            active_hypotheses = []
            for misconception_id, definition in definitions.items():
                if definition.concept_id != concept_id:
                    continue
                belief = beliefs.get(misconception_id)
                signals = int(
                    observed["misconception_signals"].get(
                        misconception_id, 0
                    )
                )
                if signals == 0 and (
                    belief is None or belief.probability < 0.12
                ):
                    continue
                active_hypotheses.append(
                    {
                        "misconception_id": misconception_id,
                        "name": definition.name,
                        "session_signals": signals,
                        "posterior_probability": (
                            belief.probability if belief else None
                        ),
                        "lifetime_evidence_count": (
                            belief.evidence_count if belief else 0
                        ),
                    }
                )
            active_hypotheses.sort(
                key=lambda item: (
                    -(item["posterior_probability"] or 0.0),
                    -item["session_signals"],
                    item["misconception_id"],
                )
            )
            attention_reasons: list[str] = []
            incorrect = observed["attempted"] - observed["correct"]
            if incorrect:
                attention_reasons.append("incorrect_responses")
            if observed["uncertain_responses"]:
                attention_reasons.append("uncertain_or_noncredible_evidence")
            if observed["verification_failures"]:
                attention_reasons.append("failed_independent_verification")
            if concept_id in boundary_concepts:
                attention_reasons.append("selected_prerequisite_boundary")
            if any(
                (item["posterior_probability"] or 0.0) >= 0.35
                for item in active_hypotheses
            ):
                attention_reasons.append("active_misconception_hypothesis")
            if readiness.evidence_mass > 0 and readiness.mastery_probability < 0.50:
                attention_reasons.append("low_current_mastery_probability")
            difficulties_for_concept = observed["difficulties"]
            family_count = evidence[concept_id]["families"]
            concept_performance.append(
                {
                    "concept_id": concept_id,
                    "name": graph.concepts[concept_id].name,
                    "topic": topic_by_concept.get(concept_id),
                    "session": {
                        "attempted": observed["attempted"],
                        "correct": observed["correct"],
                        "incorrect": incorrect,
                        "abstained": observed["abstained"],
                        "uncertain_responses": observed[
                            "uncertain_responses"
                        ],
                        "remediation_questions": observed[
                            "remediation_questions"
                        ],
                        "verification_failures": observed[
                            "verification_failures"
                        ],
                        "average_difficulty": (
                            mean(difficulties_for_concept)
                            if difficulties_for_concept
                            else None
                        ),
                    },
                    "current_projection": {
                        "mastery_probability": readiness.mastery_probability,
                        "expected_competence": readiness.expected_competence,
                        "uncertainty": readiness.uncertainty,
                        "evidence_mass": readiness.evidence_mass,
                        "independent_families": family_count,
                        "evidence_diversity": (
                            "none"
                            if family_count == 0
                            else "single_family"
                            if family_count == 1
                            else "developing"
                            if family_count == 2
                            else "diverse"
                        ),
                        "prerequisite_support": readiness.prerequisite_support,
                        "effective_readiness": readiness.effective_readiness,
                        "bottleneck_concept_id": (
                            readiness.bottleneck_concept_id
                        ),
                    },
                    "misconception_hypotheses": active_hypotheses,
                    "attention_reasons": attention_reasons,
                }
            )
        diagnostic_findings = [
            row for row in concept_performance if row["attention_reasons"]
        ]
        diagnostic_findings.sort(
            key=lambda row: (
                -len(row["attention_reasons"]),
                -row["session"]["incorrect"],
                row["current_projection"]["effective_readiness"],
                row["name"],
            )
        )
        return {
            "session_id": session_id,
            "learner_id": session["learner_id"],
            "status": session["status"],
            "mode": session["mode"],
            "topic": (
                {"id": topic["id"], "name": topic["name"]} if topic else None
            ),
            "root_concept_id": session["root_concept_id"],
            "corpus_release_id": session["corpus_release_id"],
            "questions_presented": len(rows),
            "questions_answered": len(answered),
            "correct": correct_count,
            "accuracy": correct_count / len(answered) if answered else None,
            "abstained": abstained,
            "unique_families": len({row["family_id"] for row in answered}),
            "unique_concepts": len(
                {row["primary_concept_id"] for row in answered}
            ),
            "phase_counts": dict(Counter(row["phase"] for row in answered)),
            "response_time": {
                "active_seconds": sum(response_times) / 1000.0,
                "average_seconds": (
                    mean(response_times) / 1000.0 if response_times else None
                ),
                "median_seconds": (
                    median(response_times) / 1000.0 if response_times else None
                ),
                "wall_seconds": wall_seconds,
            },
            "difficulty": {
                "average": mean(difficulties) if difficulties else None,
                "minimum": min(difficulties) if difficulties else None,
                "maximum": max(difficulties) if difficulties else None,
                "bands": dict(difficulty_bands),
                "scale": "expert-prior IRT b; not yet empirically calibrated",
            },
            "average_predicted_success": (
                mean(predicted) if predicted else None
            ),
            "continuity": {
                "average_score": mean(continuity) if continuity else None,
                "strong_follow_up_rate": (
                    sum(value >= 0.70 for value in continuity) / len(continuity)
                    if continuity
                    else None
                ),
            },
            "exploration": {
                "questions": len(exploration_rows),
                "correct": sum(bool(row["is_correct"]) for row in exploration_rows),
            },
            "remediation_questions": sum(
                row["pedagogical_role"] in {"remediation_probe", "verification"}
                for row in answered
            ),
            "cross_topic_questions": sum(
                int(row["topic_count"] or 0) > 1 for row in answered
            ),
            "topic_distribution": [dict(row) for row in topic_counts],
            "evidence_delta": total_evidence_delta,
            "concept_changes": concept_changes,
            "boundary_algorithm_version": BOUNDARY_ALGORITHM_VERSION,
            "adaptive_routing": adaptive_routing,
            "adaptive_path": adaptive_path,
            "concept_performance": concept_performance,
            "diagnostic_findings": diagnostic_findings,
            "diagnostic_contract": (
                "Findings expose observed evidence and posterior hypotheses; "
                "they are not causal or clinical diagnoses."
            ),
        }

    def trace(self, session_id: str) -> list[dict[str, Any]]:
        self.database.get_session(session_id)
        return self.database.recent_decisions(session_id, limit=200)

    @staticmethod
    def _outcome_payload(result: SubmissionResult) -> dict[str, Any]:
        return {
            "interaction_id": result.interaction_id,
            "correct": result.correct,
            "selected_option": _option_payload(result.selected_option),
            "correct_option": _option_payload(result.correct_option),
            "next_phase": result.next_phase.value,
            "focus_concept_id": result.focus_concept_id,
            "focus_misconception_id": result.focus_misconception_id,
            "state_changes": list(result.state_changes),
            "transition_reason": result.transition_reason,
            "boundary_decision": result.boundary_decision,
        }

    def _result_for_attempt(
        self,
        connection: sqlite3.Connection,
        event_id: str,
        *,
        command_hash: str,
        idempotent: bool,
    ) -> SubmissionResult:
        attempt = connection.execute("SELECT * FROM attempts WHERE event_id = ?", (event_id,)).fetchone()
        if not attempt:
            raise ConflictError("Idempotent event exists without its projection; database needs repair.")
        if attempt["command_hash"] != command_hash:
            raise ConflictError("Idempotency key was reused with different answer inputs.")
        if not attempt["outcome_json"]:
            raise ConflictError(
                "Idempotent response exists without its finalized outcome; database needs repair."
            )
        outcome = json.loads(attempt["outcome_json"])
        selected = _option_from_payload(outcome["selected_option"])
        correct_option = _option_from_payload(outcome["correct_option"])
        if correct_option is None:
            raise ConflictError("Stored attempt outcome has no correct option; database needs repair.")
        return SubmissionResult(
            interaction_id=outcome["interaction_id"],
            correct=bool(outcome["correct"]),
            selected_option=selected,
            correct_option=correct_option,
            next_phase=SessionPhase(outcome["next_phase"]),
            focus_concept_id=outcome["focus_concept_id"],
            focus_misconception_id=outcome["focus_misconception_id"],
            state_changes=tuple(outcome["state_changes"]),
            transition_reason=outcome.get(
                "transition_reason", "legacy_transition"
            ),
            boundary_decision=outcome.get("boundary_decision"),
            idempotent_replay=idempotent,
        )
