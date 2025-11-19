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
