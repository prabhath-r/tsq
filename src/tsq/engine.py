# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from math import exp, isfinite, log
from typing import Any

from .errors import ConflictError, ExhaustedError, NotFoundError, ValidationError
from .learner import LearnerModel, MODEL_VERSION
from .models import Option, Presentation, SessionPhase, SubmissionResult
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
        self.policy = AdaptivePolicy(database, self.learner_model)

    def create_learner(self, learner_id: str, display_name: str | None = None) -> dict[str, Any]:
        return self.database.ensure_learner(learner_id, display_name)

    def start_session(
        self,
        learner_id: str,
        root_concept_id: str,
        *,
        mode: str = "learn",
        seed: int | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"learn", "diagnose", "review"}:
            raise ValidationError("Mode must be learn, diagnose, or review.")
        return self.database.create_session(
            learner_id,
            root_concept_id,
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
            assessable = graph.learning_scope(session["root_concept_id"]) - containers
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
        completed: bool = True,
        reason: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return self.database.end_session(
            session_id,
            status="completed" if completed else "abandoned",
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
                payload={
                    "response_event_id": event["event_id"],
                    "state_changes": changes,
                    "phase": transition["phase"].value,
                    "focus_concept_id": transition["focus_concept_id"],
                    "focus_misconception_id": transition["focus_misconception_id"],
                    "remediation_depth": transition["remediation_depth"],
                    "remediation_path": transition["remediation_path"],
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
                    payload={
                        "from_phase": session["phase"],
                        "to_phase": transition["phase"].value,
                        "focus_concept_id": transition["focus_concept_id"],
                        "focus_misconception_id": transition["focus_misconception_id"],
                        "remediation_depth": transition["remediation_depth"],
                        "remediation_path": transition["remediation_path"],
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
                }
            if current == SessionPhase.REMEDIATE and valid_unguided_focus_evidence:
                return {
                    "phase": SessionPhase.VERIFY,
                    "focus_concept_id": session["focus_concept_id"] or question.primary_concept_id,
                    "focus_misconception_id": session["focus_misconception_id"],
                    "remediation_depth": session["remediation_depth"],
                    "remediation_path": remediation_path,
                }
            if current == SessionPhase.VERIFY and valid_unguided_focus_evidence:
                focus_misconception = session["focus_misconception_id"]
                if focus_misconception:
                    belief = connection.execute(
                        """SELECT log_odds FROM misconception_beliefs
                           WHERE learner_id = ? AND misconception_id = ?""",
                        (session["learner_id"], focus_misconception),
                    ).fetchone()
                    if belief and 1.0 / (1.0 + exp(-belief["log_odds"])) >= 0.35:
                        return {
                            "phase": SessionPhase.REMEDIATE,
                            "focus_concept_id": session["focus_concept_id"],
                            "focus_misconception_id": focus_misconception,
                            "remediation_depth": session["remediation_depth"],
                            "remediation_path": remediation_path,
                        }
                if remediation_path:
                    parent = remediation_path.pop()
                    return {
                        "phase": SessionPhase.REMEDIATE,
                        "focus_concept_id": parent["concept_id"],
                        "focus_misconception_id": parent.get("misconception_id"),
                        "remediation_depth": max(
                            1, session["remediation_depth"] - 1
                        ),
                        "remediation_path": remediation_path,
                    }
                return {
                    "phase": _main_phase(session["mode"]),
                    "focus_concept_id": None,
                    "focus_misconception_id": None,
                    "remediation_depth": 0,
                    "remediation_path": [],
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
                    }
            return {
                "phase": current,
                "focus_concept_id": session["focus_concept_id"],
                "focus_misconception_id": session["focus_misconception_id"],
                "remediation_depth": next_depth,
                "remediation_path": remediation_path,
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
        if next_depth == 2 and focus_concept:
            graph = self.database.get_graph(decision["corpus_release_id"])
            prerequisites = graph.direct_prerequisites(focus_concept)
            if prerequisites:
                rows = connection.execute(
                    "SELECT concept_id, mean FROM skill_states WHERE learner_id = ? AND concept_id IN ({})".format(
                        ",".join("?" for _ in prerequisites)
                    ),
                    (session["learner_id"], *(concept_id for concept_id, _ in prerequisites)),
                ).fetchall()
                means = {row["concept_id"]: row["mean"] for row in rows}
                prerequisite = min(
                    prerequisites,
                    key=lambda pair: means.get(pair[0], -1.386),
                )[0]
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
            }
        return {
            "phase": SessionPhase.REMEDIATE,
            "focus_concept_id": focus_concept,
            "focus_misconception_id": focus_misconception,
            "remediation_depth": next_depth,
            "remediation_path": remediation_path,
        }

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
        scope = graph.learning_scope(root_concept_id) if root_concept_id else set(graph.concepts)
        stored = self.database.get_skill_states(learner_id)
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
        return {"learner_id": learner_id, "skills": skills, "active_misconceptions": misconceptions}

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
            idempotent_replay=idempotent,
        )
