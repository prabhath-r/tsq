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
from .capacity import VERIFICATION_KINDS
from .evidence import ActionKind, ActionPhase, LearningAction, summarize_actions
from .errors import ConflictError, ExhaustedError, NotFoundError, ValidationError
from .inference import (
    LEGACY_MISCONCEPTION_ALGORITHM,
    MISCONCEPTION_ALGORITHM_METADATA_KEY,
    MISCONCEPTION_ALGORITHM_VERSION,
    ResponseClass,
    SUPPORTED_MISCONCEPTION_ALGORITHMS,
    classify_response_for_model,
    credible_response_sql,
    response_window,
)
from .learner import (
    INITIAL_STABILITY_HOURS,
    LearnerModel,
)
from .models import (
    MAX_HINT_COUNT,
    MAX_RESPONSE_MS,
    Option,
    Presentation,
    QuestionStatus,
    SessionPhase,
    SubmissionResult,
)
from .policy import (
    MAX_REMEDIATION_DEPTH,
    POLICY_VERSION,
    AdaptivePolicy,
    _selection_version_boundary,
)
from .performance_reporting import productive_shadow_summary
from .response_patterns import display_position_shadow
from .store import Database, new_id, question_runtime_activation_safe
from .versions import (
    AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS,
    COMPLETE_TRANSITION_OUTCOME_MODEL_VERSIONS,
    OBJECTIVE_MODEL_VERSIONS,
    projection_format_for,
)


OBJECTIVE_PREREQUISITE_MASTERY_FLOOR = 0.40
MISCONCEPTION_MONITORING_THRESHOLD = 0.12
MISCONCEPTION_ACTIVE_THRESHOLD = 0.35
_RESPONSE_MODEL_SQL = (
    "json_extract(response_event.metadata_json, '$.learner_model_version')"
)
_RESPONSE_CREDIBILITY_SQL = credible_response_sql(
    model_expression=_RESPONSE_MODEL_SQL,
)
_CREDIBLE_ROUTING_ATTEMPT_SQL = (
    f"AND ({_RESPONSE_CREDIBILITY_SQL})"
)


def _certificate_observability_sql() -> str:
    """Return the event-versioned immutable telemetry contract for certificates.

    v5 and earlier permitted omitted confidence and timing. v6 made timing
    mandatory while retaining optional confidence. Spacing-aware v7+ models
    share the fully fail-closed response classifier and require both fields. The
    response event, rather than the currently running engine, owns that
    interpretation so a mixed-version session remains historically honest.
    """

    return f" AND ({_RESPONSE_CREDIBILITY_SQL})"


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_artifact_reference(
    artifact: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate a content-addressed artifact without accepting its contents."""
    if artifact is None:
        return None
    if type(artifact) is not dict:
        raise ValidationError("artifact must be an object or null.")
    required = {"sha256", "size_bytes", "media_type"}
    if set(artifact) != required:
        missing = sorted(required - set(artifact))
        unknown = sorted(set(artifact) - required)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unexpected " + ", ".join(unknown))
        raise ValidationError("artifact has " + "; ".join(details) + ".")
    digest = artifact["sha256"]
    if not (
        type(digest) is str
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        raise ValidationError("artifact sha256 must be a lowercase SHA-256 digest.")
    size_bytes = artifact["size_bytes"]
    if (
        type(size_bytes) is not int
        or size_bytes < 0
        or size_bytes > 1_073_741_824
    ):
        raise ValidationError(
            "artifact size_bytes must be an integer between 0 and 1073741824."
        )
    media_type = artifact["media_type"]
    if (
        type(media_type) is not str
        or not media_type.strip()
        or media_type != media_type.strip()
        or len(media_type) > 127
        or any(character.isspace() for character in media_type)
        or "/" not in media_type
    ):
        raise ValidationError(
            "artifact media_type must be a compact MIME type of at most 127 characters."
        )
    return {
        "sha256": digest,
        "size_bytes": size_bytes,
        "media_type": media_type,
    }


def _validated_learning_action(
    *,
    action_id: str,
    decision_id: str,
    sequence: int,
    action_type: str,
    stage: str,
    payload: dict[str, Any],
    elapsed_ms: int,
) -> LearningAction:
    """Build the shared strict action model and translate its errors."""
    if not isinstance(action_type, str):
        raise ValidationError("action_type must name an allowed action kind.")
    try:
        kind = ActionKind(action_type)
    except ValueError as exc:
        choices = ", ".join(kind.value for kind in ActionKind)
        raise ValidationError(
            f"Unknown action_type {action_type!r}; expected one of: {choices}."
        ) from exc
    canonical_stage = "unassisted" if stage == "pre_response" else stage
    try:
        phase = ActionPhase(canonical_stage)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "stage must be unassisted, assisted, or post_feedback."
        ) from exc
    if type(payload) is not dict:
        raise ValidationError("action payload must be an object.")
    try:
        validated = LearningAction(
            id=action_id,
            trace_id=decision_id,
            sequence=sequence,
            kind=kind,
            phase=phase,
            payload=payload,
            elapsed_ms=elapsed_ms,
        )
        encoded_payload = json.dumps(
            validated.terms()["payload"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, ValueError) as exc:
        raise ValidationError(f"Invalid {kind.value} action: {exc}") from exc
    if len(encoded_payload) > 16_384:
        raise ValidationError(
            f"Invalid {kind.value} action: canonical payload exceeds 16384 bytes."
        )
    return validated


def _option_payload(option: Option | None) -> dict[str, Any] | None:
    if option is None:
        return None
    payload = {
        "id": option.id,
        "text": option.text,
        "correct": option.correct,
        "rationale": option.rationale,
        "misconception_id": option.misconception_id,
    }
    if option.diagnostic_objective_id is not None:
        payload["diagnostic_objective_id"] = (
            option.diagnostic_objective_id
        )
    return payload


def _option_from_payload(payload: dict[str, Any] | None) -> Option | None:
    if payload is None:
        return None
    return Option(
        id=payload["id"],
        text=payload["text"],
        correct=bool(payload["correct"]),
        rationale=payload["rationale"],
        misconception_id=payload.get("misconception_id"),
        diagnostic_objective_id=payload.get("diagnostic_objective_id"),
    )


def _main_phase(mode: str) -> SessionPhase:
    return {
        "diagnose": SessionPhase.DIAGNOSE,
        "review": SessionPhase.REVIEW,
    }.get(mode, SessionPhase.LEARN)


def _family_diversity_label(family_count: int) -> str:
    """Render a compact label without obscuring what the count represents."""
    if family_count <= 0:
        return "none"
    if family_count == 1:
        return "single_family"
    if family_count == 2:
        return "developing"
    return "diverse"


class AdaptiveEngine:
    """Application service coordinating sessions, policy, evidence, and projections."""

    def __init__(
        self,
        database: Database,
        learner_model: LearnerModel | None = None,
        *,
        misconception_algorithm: str = MISCONCEPTION_ALGORITHM_VERSION,
    ):
        if misconception_algorithm not in SUPPORTED_MISCONCEPTION_ALGORITHMS:
            raise ValueError(
                "Unsupported misconception inference algorithm: "
                f"{misconception_algorithm}"
            )
        self.database = database
        self.learner_model = learner_model or LearnerModel()
        self.misconception_algorithm = misconception_algorithm
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
        now: datetime | None = None,
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
            now=now,
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
        release_objectives = self.database.get_learning_objectives(
            session["corpus_release_id"]
        )
        objective_by_id = {
            objective.id: objective for objective in release_objectives
        }
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
        focus_objective = objective_by_id.get(session["focus_objective_id"])
        if session["focus_objective_id"] is not None and focus_objective is None:
            raise ValidationError(
                "Live corpus gap references an objective outside its pinned release."
            )
        if focus_objective is None:
            objective_candidates = [
                objective
                for objective in release_objectives
                if focus_concept_id in objective.concept_ids
            ]
            canonically_owned = [
                objective
                for objective in objective_candidates
                if objective.primary_concept_id == focus_concept_id
            ]
            if canonically_owned:
                objective_candidates = canonically_owned
            if objective_candidates:
                objective_states = self.database.get_objective_states(
                    session["learner_id"]
                )

                def objective_need(objective):
                    state = objective_states.get(
                        objective.id
                    ) or self.learner_model.initial_objective_state(
                        session["learner_id"], objective
                    )
                    projected = self.learner_model.project_objective_state(
                        state, objective, now
                    )
                    return (
                        0.55 * projected.mastery_probability
                        + 0.45 * projected.expected_competence,
                        objective.id,
                    )

                focus_objective = min(
                    objective_candidates, key=objective_need
                )
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
            self.database.validate_session_focus(
                current, connection=connection
            )
            demand_key = _canonical_hash(
                {
                    "session_id": session_id,
                    "session_revision": current["revision"],
                    "phase": current["phase"],
                    "focus_concept_id": focus_concept_id,
                    "focus_misconception_id": current["focus_misconception_id"],
                    "focus_objective_id": (
                        focus_objective.id if focus_objective else None
                    ),
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

            if focus_objective is not None:
                source_ids = tuple(
                    row["source_id"]
                    for row in connection.execute(
                        """SELECT DISTINCT source.source_id
                           FROM release_question_objectives direct
                           JOIN question_sources source
                             ON source.question_id = direct.question_id
                           JOIN release_questions membership
                             ON membership.release_id = direct.release_id
                            AND membership.question_id = direct.question_id
                           WHERE direct.release_id = ?
                             AND direct.objective_id = ?
                             AND membership.status
                                 IN ('approved', 'calibrated')
                             AND NOT EXISTS (
                                 SELECT 1
                                 FROM question_revocations revoked
                                 WHERE revoked.question_id = direct.question_id
                             )
                           ORDER BY source.source_id LIMIT 8""",
                        (
                            current["corpus_release_id"],
                            focus_objective.id,
                        ),
                    )
                )
                if not source_ids:
                    source_ids = tuple(
                        row["source_id"]
                        for row in connection.execute(
                            """SELECT DISTINCT source.source_id
                               FROM question_sources source
                               JOIN question_concepts mapping
                                 ON mapping.question_id = source.question_id
                               JOIN release_questions membership
                                 ON membership.question_id = source.question_id
                               WHERE membership.release_id = ?
                                 AND mapping.concept_id = ?
                                 AND mapping.role = 'primary'
                                 AND membership.status
                                     IN ('approved', 'calibrated')
                                 AND NOT EXISTS (
                                     SELECT 1
                                     FROM question_revocations revoked
                                     WHERE revoked.question_id =
                                           source.question_id
                                 )
                               ORDER BY source.source_id LIMIT 8""",
                            (
                                current["corpus_release_id"],
                                focus_objective.primary_concept_id,
                            ),
                        )
                    )
            else:
                source_ids = tuple(
                    row["source_id"]
                    for row in connection.execute(
                        """SELECT DISTINCT qs.source_id
                           FROM question_sources qs
                           JOIN question_concepts qc
                             ON qc.question_id = qs.question_id
                           JOIN release_questions rq
                             ON rq.question_id = qs.question_id
                           WHERE qc.concept_id = ? AND rq.release_id = ?
                             AND rq.status IN ('approved', 'calibrated')
                             AND NOT EXISTS (
                                 SELECT 1
                                 FROM question_revocations revoked
                                 WHERE revoked.question_id = qs.question_id
                             )
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
            elif focus_objective is not None:
                misconception_ids = tuple(
                    row["misconception_id"]
                    for row in connection.execute(
                        """SELECT DISTINCT option.misconception_id
                           FROM release_option_objectives diagnostic
                           JOIN release_questions membership
                             ON membership.release_id =
                                diagnostic.release_id
                            AND membership.question_id =
                                diagnostic.question_id
                           JOIN options option
                             ON option.question_id = diagnostic.question_id
                            AND option.option_id = diagnostic.option_id
                           WHERE diagnostic.release_id = ?
                             AND diagnostic.objective_id = ?
                             AND membership.status
                                 IN ('approved', 'calibrated')
                             AND NOT EXISTS (
                                 SELECT 1
                                 FROM question_revocations revoked
                                 WHERE revoked.question_id =
                                       diagnostic.question_id
                             )
                             AND option.misconception_id IS NOT NULL
                           ORDER BY option.misconception_id LIMIT 3""",
                        (
                            current["corpus_release_id"],
                            focus_objective.id,
                        ),
                    )
                )
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
                (
                    """SELECT mean FROM objective_states
                       WHERE learner_id = ? AND objective_id = ?"""
                    if focus_objective is not None
                    else """SELECT mean FROM skill_states
                       WHERE learner_id = ? AND concept_id = ?"""
                ),
                (
                    current["learner_id"],
                    (
                        focus_objective.id
                        if focus_objective is not None
                        else focus_concept_id
                    ),
                ),
            ).fetchone()
            if state:
                target_difficulty = max(-2.5, min(2.5, float(state["mean"])))
            else:
                prior = max(
                    0.02,
                    min(
                        0.98,
                        (
                            focus_objective.prior_mastery
                            if focus_objective is not None
                            else concept.prior_mastery
                        ),
                    ),
                )
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
                            corpus_release_id=current["corpus_release_id"],
                            learning_objective_id=(
                                focus_objective.id
                                if focus_objective is not None
                                else None
                            ),
                            learning_objective_name=(
                                focus_objective.name
                                if focus_objective is not None
                                else None
                            ),
                            learning_objective_description=(
                                focus_objective.description
                                if focus_objective is not None
                                else None
                            ),
                            learning_objective_operation=(
                                focus_objective.operation.value
                                if focus_objective is not None
                                else None
                            ),
                            learning_objective_evidence_type=(
                                focus_objective.evidence_type
                                if focus_objective is not None
                                else None
                            ),
                            target_misconception_id=(
                                focus_misconception_id
                                if focus_objective is not None
                                else None
                            ),
                            coverage_goal="live_corpus_gap",
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
                    "focus_objective_id": (
                        focus_objective.id if focus_objective else None
                    ),
                    "corpus_release_id": current["corpus_release_id"],
                    "message": message,
                    "job_id": job_ids[0],
                    "job_ids": job_ids,
                },
                metadata={
                    "policy_version": POLICY_VERSION,
                    "learner_model_version": self.learner_model.model_version,
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
        now: datetime | None = None,
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
            now=now,
        )

    @staticmethod
    def _action_projection(
        row: sqlite3.Row, *, idempotent_replay: bool = False
    ) -> dict[str, Any]:
        artifact = (
            {
                "sha256": row["artifact_sha256"],
                "size_bytes": row["artifact_size_bytes"],
                "media_type": row["artifact_media_type"],
            }
            if row["artifact_sha256"] is not None
            else None
        )
        selected_at = datetime.fromisoformat(row["decision_created_at"])
        occurred_at = datetime.fromisoformat(row["occurred_at"])
        return {
            "id": row["id"],
            "event_id": row["event_id"],
            "decision_id": row["decision_id"],
            "session_id": row["session_id"],
            "learner_id": row["learner_id"],
            "sequence": row["sequence"],
            "stage": row["stage"],
            "action_type": row["action_type"],
            "payload": json.loads(row["payload_json"]),
            "artifact": artifact,
            "elapsed_ms": max(
                0, int((occurred_at - selected_at).total_seconds() * 1000)
            ),
            "occurred_at": row["occurred_at"],
            "recorded_at": row["recorded_at"],
            "idempotent_replay": idempotent_replay,
        }

    @staticmethod
    def _action_row_for_event(
        connection: sqlite3.Connection, event_id: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            """SELECT action.*,
                      decision.created_at AS decision_created_at,
                      artifact.sha256 AS artifact_sha256,
                      artifact.size_bytes AS artifact_size_bytes,
                      artifact.media_type AS artifact_media_type
               FROM learning_actions action
               JOIN decisions decision ON decision.id = action.decision_id
               LEFT JOIN learning_artifacts artifact
                 ON artifact.id = action.artifact_id
               WHERE action.event_id = ?""",
            (event_id,),
        ).fetchone()

    def record_action(
        self,
        decision_id: str,
        action_type: str,
        payload: dict[str, Any],
        *,
        stage: str = "unassisted",
        artifact: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Append validated behavior telemetry without treating it as mastery.

        Only content digests, stable identifiers, and bounded counters enter the
        ledger.  This command never advances learner/session revisions or skill
        projections; rubric evaluation is an explicit later boundary.  An
        ``abandoned`` checkpoint atomically invalidates its pending decision so
        the session can continue with a fresh question.
        """
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValidationError("decision_id must be a non-blank string.")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key) > 200
        ):
            raise ValidationError(
                "idempotency_key must be a non-blank string of at most 200 characters."
            )
        artifact_reference = _canonical_artifact_reference(artifact)
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
            self.database.require_learner_evidence_safe(
                session["learner_id"],
                connection,
            )
            self.database.validate_release_focus_tuple(
                decision["corpus_release_id"],
                decision["focus_concept_id"],
                decision["focus_misconception_id"],
                decision["focus_objective_id"],
                connection=connection,
                label=f"decision {decision_id} focus",
            )
            selected_at = datetime.fromisoformat(decision["created_at"])

            # Validate the closed payload before looking for an idempotent
            # projection.  The placeholder identity never reaches storage.
            preliminary = _validated_learning_action(
                action_id="validation",
                decision_id=decision_id,
                sequence=0,
                action_type=action_type,
                stage=stage,
                payload=payload,
                elapsed_ms=max(
                    0, int((now - selected_at).total_seconds() * 1000)
                ),
            )
            action_type = preliminary.kind.value
            stage = preliminary.phase.value
            canonical_payload = preliminary.terms()["payload"]
            if (
                preliminary.kind is ActionKind.FEEDBACK_SHOWN
                and preliminary.phase is not ActionPhase.POST_FEEDBACK
            ):
                raise ValidationError(
                    "feedback_shown actions must use the post_feedback stage."
                )
            command_payload = {
                "decision_id": decision_id,
                "stage": stage,
                "action_type": action_type,
                "payload": canonical_payload,
                "artifact": artifact_reference,
            }
            command_hash = _canonical_hash(command_payload)

            if idempotency_key:
                prior_event = connection.execute(
                    "SELECT * FROM events WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if prior_event:
                    prior_payload = json.loads(prior_event["payload_json"])
                    comparable = {
                        field: prior_payload.get(field)
                        for field in (
                            "decision_id",
                            "stage",
                            "action_type",
                            "payload",
                            "artifact",
                        )
                    }
                    if (
                        prior_event["event_type"] != "LearnerActionRecorded"
                        or prior_event["schema_version"] != 1
                        or prior_event["stream_id"]
                        != f"learner:{session['learner_id']}"
                        or prior_event["learner_id"] != session["learner_id"]
                        or prior_event["session_id"] != session["id"]
                        or prior_event["causation_id"] != decision_id
                        or comparable != command_payload
                    ):
                        raise ConflictError(
                            "Idempotency key was reused with different action inputs."
                        )
                    projected = self._action_row_for_event(
                        connection, prior_event["event_id"]
                    )
                    if projected is None:
                        raise ConflictError(
                            "Idempotent action event has no projection; database needs repair."
                        )
                    if projected["command_hash"] != command_hash:
                        raise ConflictError(
                            "Idempotency key was reused with different action inputs."
                        )
                    return self._action_projection(
                        projected, idempotent_replay=True
                    )

            if decision["invalidated_at"]:
                raise ConflictError(
                    "This decision was invalidated or became stale; request a new question."
                )
            revocation = connection.execute(
                "SELECT reason FROM question_revocations WHERE question_id = ?",
                (decision["question_id"],),
            ).fetchone()
            if revocation:
                raise ConflictError(
                    "This question was emergency-revoked; no new actions may be attached."
                )
            if session["status"] != "active":
                raise ConflictError("Session is not active.")
            if now < selected_at:
                raise ValidationError(
                    "A learning action cannot occur before its question was selected."
                )

            if preliminary.phase is ActionPhase.POST_FEEDBACK:
                self.database.validate_session_focus(
                    session, connection=connection
                )
                attempt = connection.execute(
                    "SELECT answered_at FROM attempts WHERE decision_id = ?",
                    (decision_id,),
                ).fetchone()
                if decision["consumed_at"] is None or attempt is None:
                    raise ConflictError(
                        "A post-feedback action requires an answered decision."
                    )
                if now < datetime.fromisoformat(attempt["answered_at"]):
                    raise ValidationError(
                        "A post-feedback action cannot occur before the answer."
                    )
            else:
                if decision["consumed_at"]:
                    raise ConflictError(
                        "This decision has already been answered; pre-response actions "
                        "require a pending decision."
                    )
                learner = connection.execute(
                    "SELECT revision FROM learners WHERE id = ?",
                    (session["learner_id"],),
                ).fetchone()
                if learner is None:
                    raise ConflictError("Session learner no longer exists.")
                if (
                    session["phase"] != decision["phase"]
                    or session["focus_concept_id"] != decision["focus_concept_id"]
                    or session["focus_misconception_id"]
                    != decision["focus_misconception_id"]
                    or session["focus_objective_id"]
                    != decision["focus_objective_id"]
                    or session["corpus_release_id"]
                    != decision["corpus_release_id"]
                    or session["revision"] != decision["session_revision"] + 1
                    or learner["revision"] != decision["learner_revision"]
                ):
                    raise ConflictError(
                        "This decision is stale; request a new question before recording actions."
                    )
                self.database.validate_session_focus(
                    session, connection=connection
                )

            prior_rows = connection.execute(
                """SELECT sequence, occurred_at, action_type, stage, payload_json
                   FROM learning_actions WHERE decision_id = ?
                   ORDER BY sequence""",
                (decision_id,),
            ).fetchall()
            prior = prior_rows[-1] if prior_rows else None
            if (
                preliminary.kind is ActionKind.HINT_REQUESTED
                and sum(
                    row["action_type"] == ActionKind.HINT_REQUESTED.value
                    for row in prior_rows
                )
                >= MAX_HINT_COUNT
            ):
                raise ValidationError(
                    f"A decision may record at most {MAX_HINT_COUNT} hint requests."
                )
            if prior and now < datetime.fromisoformat(prior["occurred_at"]):
                raise ValidationError(
                    "Learning action times must be monotonic for a decision."
                )
            sequence = prior["sequence"] + 1 if prior else 1
            action_id = new_id("act")
            elapsed_ms = max(
                0, int((now - selected_at).total_seconds() * 1000)
            )
            validated = _validated_learning_action(
                action_id=action_id,
                decision_id=decision_id,
                sequence=sequence,
                action_type=action_type,
                stage=stage,
                payload=canonical_payload,
                elapsed_ms=elapsed_ms,
            )
            canonical_payload = validated.terms()["payload"]
            try:
                prior_actions = tuple(
                    LearningAction(
                        id=f"stored_{row['sequence']}",
                        trace_id=decision_id,
                        sequence=row["sequence"],
                        kind=ActionKind(row["action_type"]),
                        phase=ActionPhase(row["stage"]),
                        payload=json.loads(row["payload_json"]),
                        elapsed_ms=max(
                            0,
                            int(
                                (
                                    datetime.fromisoformat(row["occurred_at"])
                                    - selected_at
                                ).total_seconds()
                                * 1000
                            ),
                        ),
                    )
                    for row in prior_rows
                )
                summarize_actions((*prior_actions, validated))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValidationError(
                    f"Invalid learner-action lifecycle: {exc}"
                ) from exc

            if artifact_reference is not None:
                payload_digests = {
                    value
                    for field, value in canonical_payload.items()
                    if field.endswith("_digest") and isinstance(value, str)
                }
                if not payload_digests or artifact_reference["sha256"] not in payload_digests:
                    raise ValidationError(
                        "artifact sha256 must match a digest declared by the action payload."
                    )

            artifact_id = None
            if artifact_reference is not None:
                existing_artifact = connection.execute(
                    "SELECT * FROM learning_artifacts WHERE sha256 = ?",
                    (artifact_reference["sha256"],),
                ).fetchone()
                if existing_artifact:
                    if (
                        existing_artifact["size_bytes"]
                        != artifact_reference["size_bytes"]
                        or existing_artifact["media_type"]
                        != artifact_reference["media_type"]
                    ):
                        raise ConflictError(
                            "This artifact digest was already declared with different metadata."
                        )
                    artifact_id = existing_artifact["id"]
                else:
                    artifact_id = f"art_{artifact_reference['sha256']}"
                    connection.execute(
                        """INSERT INTO learning_artifacts(
                               id, sha256, size_bytes, media_type, created_at
                           ) VALUES (?, ?, ?, ?, ?)""",
                        (
                            artifact_id,
                            artifact_reference["sha256"],
                            artifact_reference["size_bytes"],
                            artifact_reference["media_type"],
                            now.isoformat(),
                        ),
                    )

            event_payload = {
                "action_id": action_id,
                "decision_id": decision_id,
                "sequence": sequence,
                "stage": stage,
                "action_type": action_type,
                "payload": canonical_payload,
                "artifact": artifact_reference,
            }
            event = self.database.append_event(
                connection,
                stream_id=f"learner:{session['learner_id']}",
                event_type="LearnerActionRecorded",
                schema_version=1,
                payload=event_payload,
                metadata={
                    "action_schema_version": 1,
                    "observational_only": True,
                    "corpus_release_id": decision["corpus_release_id"],
                },
                learner_id=session["learner_id"],
                session_id=session["id"],
                idempotency_key=idempotency_key,
                causation_id=decision_id,
                occurred_at=now,
            )
            connection.execute(
                """INSERT INTO learning_actions(
                       id, event_id, decision_id, session_id, learner_id,
                       sequence, stage, action_type, payload_json, artifact_id,
                       occurred_at, recorded_at, command_hash
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    action_id,
                    event["event_id"],
                    decision_id,
                    session["id"],
                    session["learner_id"],
                    sequence,
                    stage,
                    action_type,
                    json.dumps(
                        canonical_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    artifact_id,
                    now.isoformat(),
                    event["recorded_at"],
                    command_hash,
                ),
            )
            projected = self._action_row_for_event(connection, event["event_id"])
            if projected is None:
                raise ConflictError("Learning action projection was not persisted.")
            if preliminary.kind is ActionKind.ABANDONED:
                invalidated = connection.execute(
                    """UPDATE decisions
                       SET invalidated_at = ?, invalidation_reason = ?
                       WHERE id = ? AND consumed_at IS NULL
                         AND invalidated_at IS NULL""",
                    (now.isoformat(), "learner_abandoned_trace", decision_id),
                )
                if invalidated.rowcount != 1:
                    raise ConflictError(
                        "The abandoned decision changed before it could be invalidated."
                    )
                current_learner = connection.execute(
                    "SELECT revision FROM learners WHERE id = ?",
                    (session["learner_id"],),
                ).fetchone()
                if current_learner is None:
                    raise ConflictError("Session learner no longer exists.")
                self.database.append_event(
                    connection,
                    stream_id=f"learner:{session['learner_id']}",
                    event_type="DecisionInvalidated",
                    schema_version=1,
                    payload={
                        "decision_id": decision_id,
                        "reason": "learner_abandoned_trace",
                        "selection_learner_revision": decision["learner_revision"],
                        "current_learner_revision": current_learner["revision"],
                    },
                    metadata={
                        "policy_version": decision["policy_version"],
                        "learner_model_version": self.learner_model.model_version,
                        "corpus_release_id": decision["corpus_release_id"],
                    },
                    learner_id=session["learner_id"],
                    session_id=session["id"],
                    causation_id=decision_id,
                    occurred_at=now,
                )
            return self._action_projection(projected)

    def list_actions(self, decision_id: str) -> list[dict[str, Any]]:
        """Return the immutable semantic trace for one selected question."""
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValidationError("decision_id must be a non-blank string.")
        with self.database.read() as connection:
            if not connection.execute(
                "SELECT 1 FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone():
                raise NotFoundError(f"Unknown decision: {decision_id}")
            rows = connection.execute(
                """SELECT action.*,
                          decision.created_at AS decision_created_at,
                          artifact.sha256 AS artifact_sha256,
                          artifact.size_bytes AS artifact_size_bytes,
                          artifact.media_type AS artifact_media_type
                   FROM learning_actions action
                   JOIN decisions decision ON decision.id = action.decision_id
                   LEFT JOIN learning_artifacts artifact
                     ON artifact.id = action.artifact_id
                   WHERE action.decision_id = ?
                   ORDER BY action.sequence""",
                (decision_id,),
            ).fetchall()
        return [self._action_projection(row) for row in rows]

    def _invalidate_pending_decision_for_version_change(
        self,
        decision_id: str,
        *,
        idempotency_key: str | None,
        now: datetime,
    ) -> str | None:
        """Close a promise selected by a different model or adaptive policy.

        The invalidation is committed in its own transaction so callers can be
        told to request a replacement without rolling back the durable safety
        boundary.  Already-consumed decisions remain available for exact
        idempotent replay across a process upgrade.
        """

        def stale_pending(
            connection: sqlite3.Connection,
        ) -> tuple[sqlite3.Row, str] | None:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
            if (
                decision is None
                or decision["consumed_at"] is not None
                or decision["invalidated_at"] is not None
            ):
                return None
            self.database.require_learner_evidence_safe(
                decision["learner_id"],
                connection,
            )
            # Let the ordinary command boundary diagnose any reused key.  In a
            # healthy database a pending decision cannot already have a durable
            # ResponseSubmitted event, so no valid replay is skipped here.
            if idempotency_key and connection.execute(
                "SELECT 1 FROM events WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone():
                return None
            (
                selection_model_version,
                selection_policy_version,
                _,
            ) = _selection_version_boundary(
                connection,
                decision_id=decision_id,
                session_id=decision["session_id"],
            )
            if selection_policy_version != decision["policy_version"]:
                raise ValidationError(
                    f"Pending decision {decision_id} policy does not match "
                    "its QuestionSelected boundary."
                )
            if selection_model_version != self.learner_model.model_version:
                invalidation_reason = "learner_model_changed"
            elif selection_policy_version != POLICY_VERSION:
                invalidation_reason = "policy_changed"
            else:
                return None
            selected_at = datetime.fromisoformat(decision["created_at"])
            if now < selected_at:
                raise ValidationError(
                    "An answer cannot occur before its question was selected."
                )
            return decision, invalidation_reason

        # The overwhelmingly common same-model path needs only an immutable
        # event lookup; do not take a second write lock for every answer.
        with self.database.read() as connection:
            if stale_pending(connection) is None:
                return None

        # Recheck under the write lock before projecting the invalidation.  A
        # concurrent answer or idempotent replay may have closed the decision
        # after the read-only inspection.
        with self.database.transaction() as connection:
            stale = stale_pending(connection)
            if stale is None:
                return None
            decision, invalidation_reason = stale
            learner = connection.execute(
                "SELECT revision FROM learners WHERE id = ?",
                (decision["learner_id"],),
            ).fetchone()
            if learner is None:
                raise ConflictError("Decision learner no longer exists.")
            if learner["revision"] < decision["learner_revision"]:
                raise ConflictError(
                    "Pending decision is ahead of the learner projection."
                )
            invalidated = connection.execute(
                """UPDATE decisions
                   SET invalidated_at = ?, invalidation_reason = ?
                   WHERE id = ? AND consumed_at IS NULL
                     AND invalidated_at IS NULL""",
                (now.isoformat(), invalidation_reason, decision_id),
            )
            if invalidated.rowcount != 1:
                return None
            self.database.append_event(
                connection,
                stream_id=f"learner:{decision['learner_id']}",
                event_type="DecisionInvalidated",
                schema_version=1,
                payload={
                    "decision_id": decision_id,
                    "reason": invalidation_reason,
                    "selection_learner_revision": decision[
                        "learner_revision"
                    ],
                    "current_learner_revision": learner["revision"],
                },
                metadata={
                    "policy_version": POLICY_VERSION,
                    "learner_model_version": self.learner_model.model_version,
                    "corpus_release_id": decision["corpus_release_id"],
                },
                learner_id=decision["learner_id"],
                session_id=decision["session_id"],
                causation_id=decision_id,
                occurred_at=now,
            )
            return invalidation_reason

    def submit_answer(
        self,
        decision_id: str,
        selected_option_id: str | None,
        *,
        confidence: float | None = None,
        response_ms: int | None = None,
        hint_count: int = 0,
        feedback_shown: bool = False,
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
            if confidence == 0.0:
                # SQLite normalizes REAL negative zero on round-trip. Keep the
                # immutable event, command hash, and projection canonical too.
                confidence = 0.0
        if response_ms is not None and (
            type(response_ms) is not int
            or not 0 <= response_ms <= MAX_RESPONSE_MS
        ):
            raise ValidationError(
                f"response_ms must be an integer between 0 and {MAX_RESPONSE_MS}."
            )
        if type(hint_count) is not int or not 0 <= hint_count <= MAX_HINT_COUNT:
            raise ValidationError(
                f"hint_count must be an integer between 0 and {MAX_HINT_COUNT}."
            )
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

        invalidation_reason = self._invalidate_pending_decision_for_version_change(
            decision_id,
            idempotency_key=idempotency_key,
            now=now,
        )
        if invalidation_reason is not None:
            boundary = (
                "learner model"
                if invalidation_reason == "learner_model_changed"
                else "adaptive policy"
            )
            raise ConflictError(
                f"The {boundary} changed after this question was selected; "
                "the stale decision was invalidated. Request a new question."
            )

        with self.database.transaction() as connection:
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
            if not decision:
                raise NotFoundError(f"Unknown decision: {decision_id}")
            (
                selection_model_version,
                selection_policy_version,
                selection_occurred_at,
            ) = _selection_version_boundary(
                connection,
                decision_id=decision_id,
                session_id=decision["session_id"],
            )
            session = self.database.get_session(decision["session_id"], connection)
            # A prior idempotent response still exposes learner-facing
            # adaptive output. Quarantine therefore precedes both replay and
            # every new projection write.
            self.database.require_learner_evidence_safe(
                session["learner_id"],
                connection,
            )
            self.database.validate_release_focus_tuple(
                decision["corpus_release_id"],
                decision["focus_concept_id"],
                decision["focus_misconception_id"],
                decision["focus_objective_id"],
                connection=connection,
                label=f"decision {decision_id} focus",
            )
            question = self.database.get_question(
                decision["question_id"],
                connection,
                release_id=decision["corpus_release_id"],
            )
            if question.objective_id != decision["question_objective_id"]:
                raise ConflictError(
                    "The selected question no longer matches its pinned learning objective."
                )
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
            action_trace = connection.execute(
                """SELECT COUNT(*) AS action_count,
                          SUM(CASE WHEN action_type='hint_requested' THEN 1 ELSE 0 END)
                              AS hint_count,
                          SUM(CASE WHEN action_type='abandoned' THEN 1 ELSE 0 END)
                              AS abandoned_count,
                          MAX(occurred_at) AS latest_occurred_at
                   FROM learning_actions
                   WHERE decision_id = ?
                     AND stage IN ('unassisted', 'assisted')""",
                (decision_id,),
            ).fetchone()
            if action_trace["latest_occurred_at"] and now < datetime.fromisoformat(
                action_trace["latest_occurred_at"]
            ):
                raise ValidationError(
                    "An answer cannot occur before its latest recorded learning action."
                )
            if int(action_trace["abandoned_count"] or 0):
                raise ConflictError(
                    "This learning trace was abandoned; request a new question."
                )
            # Instrumented hint events are authoritative lower bounds.  Legacy
            # and external clients may still report additional uninstrumented
            # hints, so never reduce a caller's conservative count.
            traced_hint_count = int(action_trace["hint_count"] or 0)
            if traced_hint_count > MAX_HINT_COUNT:
                raise ConflictError(
                    "The stored hint trace exceeds the supported bound; "
                    "verify and repair the database before answering."
                )
            hint_count = max(hint_count, traced_hint_count)
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

            if (
                selection_model_version != self.learner_model.model_version
                or selection_policy_version != decision["policy_version"]
                or selection_policy_version != POLICY_VERSION
            ):
                raise ConflictError(
                    "The selected question no longer matches the active model "
                    "and policy boundary; request a new question."
                )
            if (
                question.objective_id is not None
                and self.learner_model.model_version
                not in OBJECTIVE_MODEL_VERSIONS
            ):
                raise ValidationError(
                    f"Learner model {self.learner_model.model_version} cannot "
                    "apply objective-aware evidence; use the current learner model."
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
            if not question_runtime_activation_safe(
                question,
                status=decision["question_status"],
            ):
                raise ConflictError(
                    "This generated question lacks the immutable independent "
                    "human-review commitment required for activation and "
                    "cannot contribute learner evidence."
                )
            if session["status"] != "active":
                raise ConflictError("Session is not active.")
            if (
                session["phase"] != decision["phase"]
                or session["focus_concept_id"] != decision["focus_concept_id"]
                or session["focus_misconception_id"] != decision["focus_misconception_id"]
                or session["focus_objective_id"] != decision["focus_objective_id"]
                or session["corpus_release_id"] != decision["corpus_release_id"]
                or session["revision"] != decision["session_revision"] + 1
            ):
                raise ConflictError(
                    "Session state changed after this question was selected; request a new question."
                )
            self.database.validate_session_focus(
                session, connection=connection
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
            if (
                self.learner_model.model_version
                in AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS
            ):
                try:
                    authoritative_window = response_window(
                        selected_at=selection_occurred_at,
                        answered_at=now,
                        response_ms=response_ms,
                    )
                except ValueError as exc:
                    raise ValidationError(str(exc)) from exc
                if not authoritative_window.consistent:
                    raise ValidationError(
                        "response_ms cannot exceed the authoritative time between "
                        "question selection and answer submission."
                    )
            last_answer = connection.execute(
                "SELECT MAX(answered_at) AS answered_at FROM attempts WHERE learner_id = ?",
                (session["learner_id"],),
            ).fetchone()["answered_at"]
            if last_answer and now < datetime.fromisoformat(last_answer):
                raise ValidationError(
                    "Out-of-order answer time would invalidate the online learner projection."
                )
            if (
                self.misconception_algorithm
                == LEGACY_MISCONCEPTION_ALGORITHM
            ):
                explicitly_versioned = connection.execute(
                    """SELECT 1 FROM events
                       WHERE stream_id=?
                         AND event_type='ResponseSubmitted'
                         AND schema_version >= 2
                       LIMIT 1""",
                    (f"learner:{session['learner_id']}",),
                ).fetchone()
                if explicitly_versioned is not None:
                    raise ConflictError(
                        "Misconception inference cannot regress to legacy "
                        "semantics after an explicitly versioned response."
                    )
            response_metadata = {
                "policy_version": decision["policy_version"],
                "learner_model_version": self.learner_model.model_version,
                "corpus_release_id": decision["corpus_release_id"],
                "question_content_hash": decision["question_content_hash"],
                "question_status": decision["question_status"],
                "evidence_weight": decision["evidence_weight"],
                "selection_learner_revision": decision["learner_revision"],
                "application_learner_revision": learner["revision"],
            }
            response_schema_version = 1
            if (
                self.misconception_algorithm
                != LEGACY_MISCONCEPTION_ALGORITHM
            ):
                response_schema_version = 2
                response_metadata[MISCONCEPTION_ALGORITHM_METADATA_KEY] = (
                    self.misconception_algorithm
                )
            event = self.database.append_event(
                connection,
                stream_id=f"learner:{session['learner_id']}",
                event_type="ResponseSubmitted",
                schema_version=response_schema_version,
                payload=command_payload,
                metadata=response_metadata,
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
                misconception_algorithm=self.misconception_algorithm,
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
                       focus_misconception_id = ?, focus_objective_id = ?,
                       remediation_depth = ?,
                       remediation_path_json = ?, recent_families_json = ?,
                       revision = revision + 1, updated_at = ?
                   WHERE id = ? AND revision = ? AND status = 'active'""",
                (
                    transition["phase"].value,
                    transition["focus_concept_id"],
                    transition["focus_misconception_id"],
                    transition["focus_objective_id"],
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
            prior_objective_projection = connection.execute(
                """SELECT 1 FROM objective_states WHERE learner_id = ?
                   UNION ALL
                   SELECT 1 FROM learner_objective_families WHERE learner_id = ?
                   LIMIT 1""",
                (session["learner_id"], session["learner_id"]),
            ).fetchone()
            objective_aware = bool(
                question.objective_id
                or session.get("focus_objective_id")
                or transition.get("focus_objective_id")
                or prior_objective_projection
            )
            if (
                objective_aware
                and self.learner_model.model_version
                not in OBJECTIVE_MODEL_VERSIONS
            ):
                raise ValidationError(
                    f"Learner model {self.learner_model.model_version} cannot "
                    "advance an objective-aware projection."
                )
            try:
                projection_format = projection_format_for(
                    self.learner_model.model_version,
                    objective_aware=objective_aware,
                )
            except ValueError as exc:
                raise ValidationError(str(exc)) from exc
            projection_hash_version = projection_format.hash_version
            projection_schema_version = (
                projection_format.event_schema_version
            )
            projection_hash = self.database.learner_projection_hash(
                session["learner_id"],
                connection,
                hash_version=projection_hash_version,
            )
            projection_payload = {
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
            }
            if objective_aware:
                projection_payload.update(
                    {
                        "question_objective_id": question.objective_id,
                        "focus_objective_id": transition[
                            "focus_objective_id"
                        ],
                        "projection_hash_version": projection_hash_version,
                    }
                )
            projection_metadata = {
                "learner_model_version": self.learner_model.model_version,
                "corpus_release_id": decision["corpus_release_id"],
                "evidence_weight": decision["evidence_weight"],
            }
            if (
                self.misconception_algorithm
                != LEGACY_MISCONCEPTION_ALGORITHM
            ):
                projection_metadata[MISCONCEPTION_ALGORITHM_METADATA_KEY] = (
                    self.misconception_algorithm
                )
            projection_event = self.database.append_event(
                connection,
                stream_id=f"learner:{session['learner_id']}",
                event_type="LearnerProjectionAdvanced",
                schema_version=projection_schema_version,
                payload=projection_payload,
                metadata=projection_metadata,
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
                or transition["focus_objective_id"]
                != session["focus_objective_id"]
                or transition["remediation_depth"] != session["remediation_depth"]
                or transition["remediation_path"] != session["remediation_path"]
            )
            if transition_changed:
                transition_payload = {
                    "from_phase": session["phase"],
                    "to_phase": transition["phase"].value,
                    "focus_concept_id": transition["focus_concept_id"],
                    "focus_misconception_id": transition[
                        "focus_misconception_id"
                    ],
                    "remediation_depth": transition["remediation_depth"],
                    "remediation_path": transition["remediation_path"],
                    "transition_reason": transition["transition_reason"],
                    "boundary_decision": transition["boundary_decision"],
                    "pedagogical_role": decision["pedagogical_role"],
                    "focus_valid": bool(decision["focus_valid"]),
                    "unguided": hint_count == 0,
                }
                if objective_aware:
                    transition_payload["focus_objective_id"] = transition[
                        "focus_objective_id"
                    ]
                self.database.append_event(
                    connection,
                    stream_id=f"learner:{session['learner_id']}",
                    event_type="RemediationTransitioned",
                    schema_version=3 if objective_aware else 2,
                    payload=transition_payload,
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
                focus_objective_id=transition["focus_objective_id"],
                state_changes=tuple(changes),
                transition_reason=transition["transition_reason"],
                boundary_decision=transition["boundary_decision"],
            )
            outcome = self._outcome_payload(result)
            if (
                self.learner_model.model_version
                in COMPLETE_TRANSITION_OUTCOME_MODEL_VERSIONS
            ):
                # v8 closes the transition commitment: every duplicated field
                # in the projection event is also pinned in the immutable
                # attempt outcome used by exact idempotent retries.
                outcome.update(
                    {
                        "remediation_depth": transition[
                            "remediation_depth"
                        ],
                        "remediation_path": transition[
                            "remediation_path"
                        ],
                    }
                )
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
        response_class = classify_response_for_model(
            model_version=self.learner_model.model_version,
            correct=correct,
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
        focus_snapshot_matches = (
            decision["phase"] == session["phase"]
            and decision["focus_concept_id"] == session["focus_concept_id"]
            and decision["focus_misconception_id"] == session["focus_misconception_id"]
            and decision["focus_objective_id"] == session["focus_objective_id"]
        )
        direct_objective_anchor = (
            question.objective.primary_concept_id
            if question.objective is not None
            else question.primary_concept_id
        )
        valid_unguided_focus_evidence = (
            bool(decision["focus_valid"])
            and focus_snapshot_matches
            and response_class.certifies_retrieval
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
            } and not response_class.certifies_retrieval:
                # A guessed, hinted, low-confidence, or implausibly fast success
                # is useful evidence but not certification. Route to an
                # independent transfer check instead of silently moving on.
                return {
                    "phase": SessionPhase.VERIFY,
                    "focus_concept_id": direct_objective_anchor,
                    "focus_misconception_id": None,
                    "focus_objective_id": question.objective_id,
                    "remediation_depth": 1,
                    "remediation_path": remediation_path,
                    "transition_reason": "noncredible_success_requires_verification",
                    "boundary_decision": None,
                }
            if current == SessionPhase.REMEDIATE and valid_unguided_focus_evidence:
                if remediation_path and session["focus_objective_id"] is not None:
                    objectives = {
                        objective.id: objective
                        for objective in self.database.get_learning_objectives(
                            decision["corpus_release_id"]
                        )
                    }
                    persistently_verified = (
                        self._persistently_verified_objectives(
                            connection,
                            learner_id=session["learner_id"],
                            release_id=decision["corpus_release_id"],
                            objective_ids={session["focus_objective_id"]},
                            objectives=objectives,
                            now=now,
                        )
                    )
                    if session["focus_objective_id"] in persistently_verified:
                        return self._resume_verified_prerequisite(
                            connection,
                            session=session,
                            decision=decision,
                            remediation_path=remediation_path,
                            now=now,
                            transition_reason=(
                                "persistent_prerequisite_verification_resume_parent"
                            ),
                            deferred_reason=(
                                "persistent_prerequisite_verified_parent_deferred"
                            ),
                        )
                return {
                    "phase": SessionPhase.VERIFY,
                    "focus_concept_id": session["focus_concept_id"] or question.primary_concept_id,
                    "focus_misconception_id": session["focus_misconception_id"],
                    "focus_objective_id": session["focus_objective_id"],
                    "remediation_depth": session["remediation_depth"],
                    "remediation_path": remediation_path,
                    "transition_reason": "focused_repair_requires_independent_verification",
                    "boundary_decision": None,
                }
            if current == SessionPhase.VERIFY and valid_unguided_focus_evidence:
                if remediation_path:
                    return self._resume_verified_prerequisite(
                        connection,
                        session=session,
                        decision=decision,
                        remediation_path=remediation_path,
                        now=now,
                        transition_reason=(
                            "prerequisite_verified_resume_parent"
                        ),
                        deferred_reason=(
                            "prerequisite_verified_parent_deferred"
                        ),
                    )
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
                        and 1.0 / (1.0 + exp(-belief["log_odds"]))
                        >= MISCONCEPTION_ACTIVE_THRESHOLD
                    )
                # One credible independent verification closes the bounded
                # teaching episode. A still-active posterior is deliberately
                # retained for later need/review scoring instead of opening an
                # unbounded repair/verify loop in the same session.
                return {
                    "phase": _main_phase(session["mode"]),
                    "focus_concept_id": None,
                    "focus_misconception_id": None,
                    "focus_objective_id": None,
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
                    "focus_objective_id": None,
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
                    "focus_objective_id": None,
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
                        "focus_objective_id": None,
                        "remediation_depth": 0,
                        "remediation_path": [],
                        "transition_reason": "noncredible_repair_bounded_exit",
                        "boundary_decision": None,
                    }
            return {
                "phase": current,
                "focus_concept_id": session["focus_concept_id"],
                "focus_misconception_id": session["focus_misconception_id"],
                "focus_objective_id": session["focus_objective_id"],
                "remediation_depth": next_depth,
                "remediation_path": remediation_path,
                "transition_reason": "noncredible_repair_requires_another_probe",
                "boundary_decision": None,
            }

        if response_class is ResponseClass.UNCERTAIN_OR_ABSTAINED:
            if current in {
                SessionPhase.LEARN,
                SessionPhase.DIAGNOSE,
                SessionPhase.REVIEW,
            }:
                # An abstention or weakly observed error says that another
                # measurement is needed; it does not identify a misconception
                # or justify descending the prerequisite graph.
                return {
                    "phase": SessionPhase.VERIFY,
                    "focus_concept_id": direct_objective_anchor,
                    "focus_misconception_id": None,
                    "focus_objective_id": question.objective_id,
                    "remediation_depth": 1,
                    "remediation_path": remediation_path,
                    "transition_reason": (
                        "uncertain_main_requires_independent_diagnostic"
                    ),
                    "boundary_decision": None,
                }
            if current == SessionPhase.VERIFY:
                if self._fresh_focus_has_repair_and_verification(
                    connection,
                    session=session,
                    decision=decision,
                    remediation_path=remediation_path,
                    now=now,
                ):
                    return {
                        "phase": SessionPhase.REMEDIATE,
                        "focus_concept_id": session["focus_concept_id"],
                        "focus_misconception_id": session[
                            "focus_misconception_id"
                        ],
                        "focus_objective_id": session["focus_objective_id"],
                        "remediation_depth": min(
                            MAX_REMEDIATION_DEPTH - 1,
                            max(1, session["remediation_depth"] + 1),
                        ),
                        "remediation_path": remediation_path,
                        "transition_reason": (
                            "repeated_uncertainty_requires_bounded_remediation"
                        ),
                        "boundary_decision": None,
                    }
                return {
                    "phase": _main_phase(session["mode"]),
                    "focus_concept_id": None,
                    "focus_misconception_id": None,
                    "focus_objective_id": None,
                    "remediation_depth": 0,
                    "remediation_path": [],
                    "transition_reason": "repeated_uncertainty_bounded_exit",
                    "boundary_decision": None,
                }
            # A diagnostic and one bounded remediation probe have both remained
            # inconclusive. Preserve their uncertain projection, but do not turn
            # missing evidence into a prerequisite failure or an unbounded loop.
            return {
                "phase": _main_phase(session["mode"]),
                "focus_concept_id": None,
                "focus_misconception_id": None,
                "focus_objective_id": None,
                "remediation_depth": 0,
                "remediation_path": [],
                "transition_reason": "uncertain_remediation_bounded_exit",
                "boundary_decision": None,
            }

        next_depth = (
            session["remediation_depth"] + 1
            if current in {SessionPhase.REMEDIATE, SessionPhase.VERIFY}
            else 1
        )
        focus_concept = (
            session["focus_concept_id"] or direct_objective_anchor
        )
        focus_misconception = session["focus_misconception_id"]
        focus_objective = session["focus_objective_id"] or question.objective_id
        cross_objective_diagnostic = False
        credible_generic_error = (
            current
            not in {SessionPhase.REMEDIATE, SessionPhase.VERIFY}
            and response_class is ResponseClass.CREDIBLE_GENERIC_ERROR
        )
        if current not in {SessionPhase.REMEDIATE, SessionPhase.VERIFY}:
            diagnostic_objective = (
                selected_option.diagnostic_objective_id
                if selected_option
                and selected_option.diagnostic_objective_id
                and response_class.supports_named_misconception
                else question.objective_id
            )
            cross_objective_diagnostic = bool(
                response_class.supports_named_misconception
                and selected_option is not None
                and question.objective_id is not None
                and diagnostic_objective is not None
                and diagnostic_objective != question.objective_id
            )
            if cross_objective_diagnostic:
                # The option diagnoses a more precise repair target than the
                # objective measured by the trigger. Preserve that measured
                # objective as an explicit obligation: after independently
                # repairing and verifying the diagnostic objective, the normal
                # resume machinery must independently recheck transfer at the
                # parent before returning to the main path. The diagnostic
                # misconception belongs to the child and must not leak into
                # the parent frame.
                remediation_path.append(
                    {
                        "concept_id": direct_objective_anchor,
                        "objective_id": question.objective_id,
                        "misconception_id": None,
                    }
                )
            focus_objective = diagnostic_objective
            if diagnostic_objective is not None:
                anchor = connection.execute(
                    """SELECT objective.primary_concept_id
                       FROM release_learning_objectives release_objective
                       JOIN learning_objectives objective
                         ON objective.id = release_objective.objective_id
                       WHERE release_objective.release_id = ?
                         AND release_objective.objective_id = ?""",
                    (decision["corpus_release_id"], diagnostic_objective),
                ).fetchone()
                if anchor is None:
                    raise ValidationError(
                        f"Diagnostic learning objective {diagnostic_objective} "
                        "is outside the pinned corpus release."
                    )
                focus_concept = anchor["primary_concept_id"]
            focus_misconception = (
                selected_option.misconception_id
                if selected_option
                and selected_option.misconception_id
                and response_class.supports_named_misconception
                else None
            )
            if focus_misconception and diagnostic_objective is None:
                owner = connection.execute(
                    "SELECT concept_id FROM misconceptions WHERE id = ?",
                    (focus_misconception,),
                ).fetchone()
                mapped_concepts = {
                    mapping.concept_id for mapping in question.concepts
                }
                if owner and owner["concept_id"] in mapped_concepts:
                    focus_concept = owner["concept_id"]
        elif (
            not focus_misconception
            and selected_option
            and selected_option.misconception_id
            and response_class.supports_named_misconception
        ):
            # A bounded repair episode keeps its original objective. A wrong
            # distractor can refine the named hypothesis only when that option
            # diagnoses the same objective; cross-objective errors are retained
            # in the misconception posterior for later planning without
            # silently retargeting the active tunnel.
            selected_target = (
                selected_option.diagnostic_objective_id
                or question.objective_id
            )
            if focus_objective is None or selected_target == focus_objective:
                focus_misconception = selected_option.misconception_id
        descended_to_prerequisite = False
        boundary_decision_payload: dict[str, Any] | None = None
        verified_prerequisites: set[str] = set()
        unserviceable_prerequisites: set[str] = set()
        prerequisite_objectives: dict[str, str] = {}
        if 2 <= next_depth <= MAX_REMEDIATION_DEPTH and focus_concept:
            if focus_objective is not None:
                declared_boundary = self._declared_objective_boundary(
                    connection,
                    session_id=session["id"],
                    learner_id=session["learner_id"],
                    release_id=decision["corpus_release_id"],
                    focus_objective_id=focus_objective,
                    now=now,
                )
                if declared_boundary is not None:
                    selected_objective_id = declared_boundary[
                        "selected_objective_id"
                    ]
                    if selected_objective_id is None:
                        return {
                            "phase": _main_phase(session["mode"]),
                            "focus_concept_id": None,
                            "focus_misconception_id": None,
                            "focus_objective_id": None,
                            "remediation_depth": 0,
                            "remediation_path": [],
                            "transition_reason": (
                                "no_serviceable_prerequisite_boundary"
                                if declared_boundary["unserviceable"]
                                else (
                                    "verified_prerequisite_not_reopened"
                                    if declared_boundary["verified"]
                                    else "bounded_failure_exit"
                                )
                            ),
                            "boundary_decision": None,
                        }
                    parent_focus = {
                        "concept_id": focus_concept,
                        "objective_id": focus_objective,
                        "misconception_id": focus_misconception,
                    }
                    remediation_path.append(parent_focus)
                    return {
                        "phase": SessionPhase.REMEDIATE,
                        "focus_concept_id": declared_boundary[
                            "selected_concept_id"
                        ],
                        "focus_misconception_id": None,
                        "focus_objective_id": selected_objective_id,
                        "remediation_depth": next_depth,
                        "remediation_path": remediation_path,
                        "transition_reason": "descend_to_evidence_boundary",
                        "boundary_decision": declared_boundary[
                            "boundary_decision"
                        ],
                    }
            if next_depth > 2:
                # Recursive descent beyond the first hop is safe only when the
                # pinned release declares the exact objective edge. Legacy
                # concept inference remains deliberately one-hop bounded.
                return {
                    "phase": _main_phase(session["mode"]),
                    "focus_concept_id": None,
                    "focus_misconception_id": None,
                    "focus_objective_id": None,
                    "remediation_depth": 0,
                    "remediation_path": [],
                    "transition_reason": "bounded_failure_exit",
                    "boundary_decision": None,
                }
            graph = self.database.get_graph(decision["corpus_release_id"])
            direct_prerequisite_ids = {
                concept_id
                for concept_id, _ in graph.direct_prerequisites(focus_concept)
            }
            boundary_states = self.database.get_skill_states(
                session["learner_id"], connection
            )
            boundary_objective_states = self.database.get_objective_states(
                session["learner_id"], connection
            )
            release_objectives = self.database.get_learning_objectives(
                decision["corpus_release_id"]
            )
            boundary_projection = (
                self.learner_model.concept_projection_with_objective_floor(
                    learner_id=session["learner_id"],
                    concepts=graph.concepts,
                    stored_states=boundary_states,
                    objectives=release_objectives,
                    stored_objective_states=boundary_objective_states,
                    now=now,
                )
            )
            boundary_states = boundary_projection.states
            if direct_prerequisite_ids:
                objectives_by_owner: dict[str, list[Any]] = {}
                objective_enabled_concepts: set[str] = set()
                ranked_objectives_by_concept: dict[str, list[str]] = {}
                for objective in release_objectives:
                    objectives_by_owner.setdefault(
                        objective.primary_concept_id, []
                    ).append(objective)
                    objective_enabled_concepts.update(objective.concept_ids)
                for concept_id in direct_prerequisite_ids:
                    owned_objectives = objectives_by_owner.get(concept_id, [])
                    if not owned_objectives:
                        if concept_id in objective_enabled_concepts:
                            # A broad fallback here could mix questions from
                            # objectives owned elsewhere. Require a canonical
                            # owned objective before claiming a fine boundary.
                            unserviceable_prerequisites.add(concept_id)
                        continue
                    projected_candidates: list[tuple[float, str]] = []
                    for objective in owned_objectives:
                        objective_state = boundary_objective_states.get(
                            objective.id
                        ) or self.learner_model.initial_objective_state(
                            session["learner_id"], objective
                        )
                        projected_objective = (
                            self.learner_model.project_objective_state(
                                objective_state, objective, now
                            )
                        )
                        projected_candidates.append(
                            (
                                0.55
                                * projected_objective.mastery_probability
                                + 0.45
                                * projected_objective.expected_competence,
                                objective.id,
                            )
                        )
                    projected_candidates.sort()
                    ranked_objectives_by_concept[concept_id] = [
                        objective_id
                        for _, objective_id in projected_candidates
                    ]

                certificate_observability = _certificate_observability_sql()
                verified_objective_focuses = (
                    self._same_session_verified_objective_focuses(
                        connection,
                        session_id=session["id"],
                        release_id=decision["corpus_release_id"],
                        objective_ids={
                            objective_id
                            for objective_ids in (
                                ranked_objectives_by_concept.values()
                            )
                            for objective_id in objective_ids
                        },
                    )
                )
                for concept_id, ranked_objective_ids in (
                    ranked_objectives_by_concept.items()
                ):
                    unresolved = [
                        objective_id
                        for objective_id in ranked_objective_ids
                        if (objective_id, None)
                        not in verified_objective_focuses
                    ]
                    if unresolved:
                        prerequisite_objectives[concept_id] = unresolved[0]
                    else:
                        verified_prerequisites.add(concept_id)

                legacy_concept_ids = (
                    direct_prerequisite_ids
                    - set(ranked_objectives_by_concept)
                    - unserviceable_prerequisites
                )
                if legacy_concept_ids:
                    legacy_placeholders = ",".join(
                        "?" for _ in legacy_concept_ids
                    )
                    verified_legacy_rows = connection.execute(
                        f"""SELECT mapping.concept_id,
                                   COUNT(DISTINCT CASE
                                       WHEN choice.pedagogical_role='remediation_probe'
                                        AND attempt.is_correct=1
                                        AND attempt.hint_count=0
                                        {certificate_observability}
                                       THEN attempt.family_id END)
                                           AS repair_families,
                                   COUNT(DISTINCT CASE
                                       WHEN choice.pedagogical_role='verification'
                                        AND attempt.is_correct=1
                                        AND attempt.hint_count=0
                                        {certificate_observability}
                                       THEN attempt.family_id END)
                                           AS verification_families
                            FROM attempts attempt
                            JOIN decisions choice
                              ON choice.id=attempt.decision_id
                            JOIN events response_event
                              ON response_event.event_id=attempt.event_id
                            JOIN release_questions released
                              ON released.release_id=choice.corpus_release_id
                             AND released.question_id=attempt.question_id
                            JOIN question_concepts mapping
                              ON mapping.question_id=attempt.question_id
                             AND mapping.role='primary'
                            WHERE attempt.session_id=?
                              AND choice.corpus_release_id=?
                              AND released.status IN (?, ?)
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM question_revocations revoked
                                  WHERE revoked.question_id=attempt.question_id
                              )
                              AND choice.focus_valid=1
                              AND choice.focus_objective_id IS NULL
                              AND choice.question_objective_id IS NULL
                              AND mapping.concept_id IN ({legacy_placeholders})
                            GROUP BY mapping.concept_id""",
                        (
                            session["id"],
                            decision["corpus_release_id"],
                            QuestionStatus.APPROVED.value,
                            QuestionStatus.CALIBRATED.value,
                            *sorted(legacy_concept_ids),
                        ),
                    ).fetchall()
                    verified_prerequisites.update(
                        row["concept_id"]
                        for row in verified_legacy_rows
                        if int(row["repair_families"]) >= 1
                        and int(row["verification_families"]) >= 1
                    )
                objective_focuses = {
                    (objective_id, None)
                    for objective_id in prerequisite_objectives.values()
                }
                objective_capacity = self._fresh_objective_focus_capacity(
                    connection,
                    session_id=session["id"],
                    learner_id=session["learner_id"],
                    release_id=decision["corpus_release_id"],
                    focuses=objective_focuses,
                    now=now,
                )
                for concept_id in direct_prerequisite_ids:
                    objective_id = prerequisite_objectives.get(concept_id)
                    if objective_id is not None:
                        objective_focus = (objective_id, None)
                        capacity = objective_capacity.get(
                            objective_focus, {}
                        )
                    elif concept_id in unserviceable_prerequisites:
                        continue
                    else:
                        capacity = self._fresh_focus_capacity(
                            connection,
                            session_id=session["id"],
                            learner_id=session["learner_id"],
                            release_id=decision["corpus_release_id"],
                            concept_ids={concept_id},
                            now=now,
                        ).get(concept_id, {})
                    families = set(capacity.get("families", set()))
                    verification_families = set(
                        capacity.get("verification_families", set())
                    )
                    if not any(
                        verification_families - {repair_family}
                        for repair_family in families
                    ):
                        unserviceable_prerequisites.add(concept_id)
            credible_attempt_clause = _CREDIBLE_ROUTING_ATTEMPT_SQL
            performance_rows = connection.execute(
                f"""SELECT COALESCE(objective.primary_concept_id,
                                    mapping.concept_id) AS concept_id,
                          COUNT(*) AS attempted,
                          SUM(CASE WHEN attempt.is_correct = 0 THEN 1 ELSE 0 END)
                              AS incorrect
                   FROM attempts attempt
                   JOIN question_concepts mapping
                     ON mapping.question_id = attempt.question_id
                    AND mapping.role = 'primary'
                   JOIN decisions choice ON choice.id = attempt.decision_id
                   JOIN events response_event
                     ON response_event.event_id=attempt.event_id
                   LEFT JOIN learning_objectives objective
                     ON objective.id = choice.question_objective_id
                   WHERE attempt.session_id = ?
                   {credible_attempt_clause}
                   GROUP BY COALESCE(objective.primary_concept_id,
                                     mapping.concept_id)""",
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
                stored_states=boundary_states,
                now=now,
                recent_performance=performance,
                excluded_concept_ids=(
                    verified_prerequisites | unserviceable_prerequisites
                ),
                intrinsic_overrides=boundary_projection.exact_floors,
            )
            if boundary is not None:
                prerequisite = boundary.selected_concept_id
                boundary_decision_payload = boundary.terms()
                if prerequisite != focus_concept:
                    parent_focus = {
                        "concept_id": focus_concept,
                        "misconception_id": focus_misconception,
                    }
                    if focus_objective is not None:
                        parent_focus["objective_id"] = focus_objective
                    remediation_path.append(parent_focus)
                    focus_concept = prerequisite
                    focus_objective = prerequisite_objectives.get(
                        prerequisite
                    )
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
                "focus_objective_id": None,
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
            "focus_objective_id": focus_objective,
            "remediation_depth": next_depth,
            "remediation_path": remediation_path,
            "transition_reason": (
                "descend_to_evidence_boundary"
                if descended_to_prerequisite
                else (
                    "cross_objective_diagnostic_focus"
                    if cross_objective_diagnostic
                    else (
                        "credible_generic_error_focus"
                        if credible_generic_error
                        else "incorrect_answer_focus"
                    )
                )
            ),
            "boundary_decision": boundary_decision_payload,
        }

    def _fresh_focus_has_repair_and_verification(
        self,
        connection: sqlite3.Connection,
        *,
        session: dict[str, Any],
        decision: sqlite3.Row | dict[str, Any],
        remediation_path: list[dict[str, Any]],
        now: datetime,
    ) -> bool:
        """Prove an uncertainty escalation can finish without a corpus gap."""

        focus_objective_id = session["focus_objective_id"]
        if focus_objective_id is not None:
            focus = (
                focus_objective_id,
                session["focus_misconception_id"],
            )
            capacity = self._fresh_objective_focus_capacity(
                connection,
                session_id=session["id"],
                learner_id=session["learner_id"],
                release_id=decision["corpus_release_id"],
                focuses={focus},
                now=now,
            ).get(focus, {})
        else:
            focus_concept_id = session["focus_concept_id"]
            if focus_concept_id is None:
                return False
            capacity = self._fresh_focus_capacity(
                connection,
                session_id=session["id"],
                learner_id=session["learner_id"],
                release_id=decision["corpus_release_id"],
                concept_ids={focus_concept_id},
                now=now,
            ).get(focus_concept_id, {})
        repair_families = set(capacity.get("families", set()))
        verification_families = set(
            capacity.get("verification_families", set())
        )
        if not remediation_path:
            return any(
                verification_families - {repair_family}
                for repair_family in repair_families
            )

        parent = remediation_path[-1]
        parent_objective_id = parent.get("objective_id")
        if parent_objective_id is not None:
            parent_focus = (
                parent_objective_id,
                parent.get("misconception_id"),
            )
            parent_capacity = self._fresh_objective_focus_capacity(
                connection,
                session_id=session["id"],
                learner_id=session["learner_id"],
                release_id=decision["corpus_release_id"],
                focuses={parent_focus},
                now=now,
            ).get(parent_focus, {})
        else:
            parent_concept_id = parent["concept_id"]
            parent_capacity = self._fresh_focus_capacity(
                connection,
                session_id=session["id"],
                learner_id=session["learner_id"],
                release_id=decision["corpus_release_id"],
                concept_ids={parent_concept_id},
                now=now,
            ).get(parent_concept_id, {})
        parent_verifications = set(
            parent_capacity.get("verification_families", set())
        )
        return any(
            parent_verifications
            - {repair_family, verification_family}
            for repair_family in repair_families
            for verification_family in (
                verification_families - {repair_family}
            )
        )

    def _resume_verified_prerequisite(
        self,
        connection: sqlite3.Connection,
        *,
        session: dict[str, Any],
        decision: dict[str, Any],
        remediation_path: list[dict[str, Any]],
        now: datetime,
        transition_reason: str,
        deferred_reason: str,
    ) -> dict[str, Any]:
        """Pop one exact remediation frame after prerequisite verification."""

        if not remediation_path:
            raise ValidationError(
                "Cannot resume a verified prerequisite without a parent frame."
            )
        parent = remediation_path.pop()
        parent_objective_id = parent.get("objective_id")
        if parent_objective_id is not None:
            parent_focus = (
                parent_objective_id,
                parent.get("misconception_id"),
            )
            parent_capacity = self._fresh_objective_focus_capacity(
                connection,
                session_id=session["id"],
                learner_id=session["learner_id"],
                release_id=decision["corpus_release_id"],
                focuses={parent_focus},
                now=now,
            ).get(parent_focus, {})
        else:
            parent_capacity = self._fresh_focus_capacity(
                connection,
                session_id=session["id"],
                learner_id=session["learner_id"],
                release_id=decision["corpus_release_id"],
                concept_ids={parent["concept_id"]},
                now=now,
            ).get(parent["concept_id"], {})
        if not parent_capacity.get("verification_families"):
            # The prerequisite gain remains in the projection, but a
            # same-session transfer claim would be unverifiable. Defer the
            # parent instead of selecting into a known corpus dead end.
            return {
                "phase": _main_phase(session["mode"]),
                "focus_concept_id": None,
                "focus_misconception_id": None,
                "focus_objective_id": None,
                "remediation_depth": 0,
                "remediation_path": [],
                "transition_reason": deferred_reason,
                "boundary_decision": None,
            }
        # The independently verified prerequisite is the intervention. Recheck
        # transfer at its unresolved parent directly; another parent repair
        # would consume two more families and can strand a serviceable target.
        return {
            "phase": SessionPhase.VERIFY,
            "focus_concept_id": parent["concept_id"],
            "focus_misconception_id": parent.get("misconception_id"),
            "focus_objective_id": parent_objective_id,
            "remediation_depth": max(
                1, session["remediation_depth"] - 1
            ),
            "remediation_path": remediation_path,
            "transition_reason": transition_reason,
            "boundary_decision": None,
        }

    def _persistently_verified_objectives(
        self,
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        release_id: str,
        objective_ids: set[str],
        objectives: dict[str, Any],
        now: datetime,
    ) -> set[str]:
        """Return prerequisites supported by durable independent retrieval.

        A certificate needs two distinct successful-retrieval families that
        remain present and eligible in the pinned release, with at least one
        transfer-capable family. The conservative mastery projection must also
        stay above the reporting readiness floor, so evidence persists across
        sessions but is not immune to modeled forgetting.
        """

        if not objective_ids:
            return set()
        if not objective_ids <= set(objectives):
            raise ValidationError(
                "Objective verification references an objective outside the "
                "pinned release."
            )
        placeholders = ",".join("?" for _ in objective_ids)
        rows = connection.execute(
            f"""SELECT evidence.objective_id, evidence.family_id, evidence.kind
                FROM learner_objective_families evidence
                WHERE evidence.learner_id = ?
                  AND evidence.objective_id IN ({placeholders})
                  AND EXISTS (
                      SELECT 1
                      FROM release_question_objectives direct
                      JOIN questions question
                        ON question.id = direct.question_id
                      JOIN release_questions release_question
                        ON release_question.release_id = direct.release_id
                       AND release_question.question_id = direct.question_id
                      WHERE direct.release_id = ?
                        AND direct.objective_id = evidence.objective_id
                        AND question.family_id = evidence.family_id
                        AND release_question.status IN (?, ?)
                        AND NOT EXISTS (
                            SELECT 1 FROM question_revocations revoked
                            WHERE revoked.question_id = question.id
                        )
                  )""",
            (
                learner_id,
                *sorted(objective_ids),
                release_id,
                QuestionStatus.APPROVED.value,
                QuestionStatus.CALIBRATED.value,
            ),
        ).fetchall()
        families: dict[str, set[str]] = {
            objective_id: set() for objective_id in objective_ids
        }
        verification_families: dict[str, set[str]] = {
            objective_id: set() for objective_id in objective_ids
        }
        verification_kinds = {kind.value for kind in VERIFICATION_KINDS}
        for row in rows:
            objective_id = row["objective_id"]
            families[objective_id].add(row["family_id"])
            if row["kind"] in verification_kinds:
                verification_families[objective_id].add(row["family_id"])

        states = self.database.get_objective_states(learner_id, connection)
        verified: set[str] = set()
        for objective_id in sorted(objective_ids):
            has_independent_transfer = any(
                families[objective_id] - {verification_family}
                for verification_family in verification_families[objective_id]
            )
            if not has_independent_transfer:
                continue
            objective = objectives[objective_id]
            state = states.get(
                objective_id
            ) or self.learner_model.initial_objective_state(
                learner_id, objective
            )
            projected = self.learner_model.project_objective_state(
                state, objective, now
            )
            if (
                projected.mastery_probability
                >= OBJECTIVE_PREREQUISITE_MASTERY_FLOOR
            ):
                verified.add(objective_id)
        return verified

    def _same_session_verified_objective_focuses(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        release_id: str,
        objective_ids: set[str] | None = None,
    ) -> set[tuple[str, str | None]]:
        """Return objective focuses certified by independent retrieval families.

        Certificates are derived from the immutable response event's learner
        model version, rather than the model currently running the engine.
        This keeps mixed-version sessions historically honest.  A repair and a
        transfer check must both be credible, focus-valid, directly mapped to
        the objective, and come from distinct independence families.
        """

        if objective_ids is not None and not objective_ids:
            return set()
        objective_clause = ""
        parameters: list[Any] = [
            session_id,
            release_id,
            QuestionStatus.APPROVED.value,
            QuestionStatus.CALIBRATED.value,
        ]
        if objective_ids is not None:
            placeholders = ",".join("?" for _ in objective_ids)
            objective_clause = (
                f"AND choice.focus_objective_id IN ({placeholders})"
            )
            parameters.extend(sorted(objective_ids))
        certificate_observability = _certificate_observability_sql()
        rows = connection.execute(
            f"""SELECT choice.focus_objective_id,
                       choice.focus_misconception_id,
                       choice.pedagogical_role,
                       attempt.family_id
                FROM attempts attempt
                JOIN decisions choice ON choice.id=attempt.decision_id
                JOIN events response_event
                  ON response_event.event_id=attempt.event_id
                JOIN release_questions released
                  ON released.release_id=choice.corpus_release_id
                 AND released.question_id=attempt.question_id
                WHERE attempt.session_id=?
                  AND choice.corpus_release_id=?
                  AND released.status IN (?, ?)
                  AND NOT EXISTS (
                      SELECT 1
                      FROM question_revocations revoked
                      WHERE revoked.question_id=attempt.question_id
                  )
                  AND choice.focus_valid=1
                  AND choice.focus_objective_id IS NOT NULL
                  AND choice.question_objective_id
                      = choice.focus_objective_id
                  AND choice.pedagogical_role IN (
                      'remediation_probe', 'verification'
                  )
                  AND attempt.is_correct=1
                  AND attempt.hint_count=0
                  {certificate_observability}
                  {objective_clause}
                GROUP BY choice.focus_objective_id,
                         choice.focus_misconception_id,
                         choice.pedagogical_role,
                         attempt.family_id""",
            parameters,
        ).fetchall()
        repair_families: dict[tuple[str, str | None], set[str]] = {}
        verification_families: dict[
            tuple[str, str | None], set[str]
        ] = {}
        for row in rows:
            focus = (
                row["focus_objective_id"],
                row["focus_misconception_id"],
            )
            target = (
                repair_families
                if row["pedagogical_role"] == "remediation_probe"
                else verification_families
            )
            target.setdefault(focus, set()).add(row["family_id"])
        return {
            focus
            for focus, repairs in repair_families.items()
            if any(
                verification_family != repair_family
                for repair_family in repairs
                for verification_family in verification_families.get(
                    focus, set()
                )
            )
        }

    def _declared_objective_boundary(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        learner_id: str,
        release_id: str,
        focus_objective_id: str,
        now: datetime,
    ) -> dict[str, Any] | None:
        """Choose one direct prerequisite from a declared objective graph.

        ``None`` means the pinned release predates objective graphs.  A mapping
        with no selected objective means the graph is declared but every
        direct prerequisite is absent, already verified, or currently lacks an
        independent repair/verification pair.  That distinction prevents an
        intentionally empty graph from falling through to broad concept
        heuristics.
        """

        graph_version, edges = self.database.get_objective_graph(release_id)
        if graph_version is None:
            return None
        if graph_version != 1:
            raise ValidationError(
                f"Unsupported objective graph version {graph_version}."
            )
        direct_edges = sorted(
            (
                edge
                for edge in edges
                if edge.target_id == focus_objective_id
                and edge.relation.is_strict_prerequisite
            ),
            key=lambda edge: edge.id,
        )
        direct_ids = {edge.source_id for edge in direct_edges}
        if len(direct_ids) != len(direct_edges):
            raise ValidationError(
                f"Objective {focus_objective_id} has ambiguous direct "
                "prerequisite fanout."
            )
        if not direct_edges:
            return {
                "selected_objective_id": None,
                "selected_concept_id": None,
                "boundary_decision": None,
                "verified": set(),
                "unserviceable": set(),
            }

        objectives = {
            objective.id: objective
            for objective in self.database.get_learning_objectives(release_id)
        }
        if focus_objective_id not in objectives or not direct_ids <= set(objectives):
            raise ValidationError(
                "Objective prerequisite graph references an objective outside "
                "the pinned release."
            )

        verified = self._persistently_verified_objectives(
            connection,
            learner_id=learner_id,
            release_id=release_id,
            objective_ids=direct_ids,
            objectives=objectives,
            now=now,
        )
        verified.update(
            objective_id
            for objective_id, _ in (
                self._same_session_verified_objective_focuses(
                    connection,
                    session_id=session_id,
                    release_id=release_id,
                    objective_ids=direct_ids,
                )
            )
        )
        unresolved = direct_ids - verified
        if not unresolved:
            return {
                "selected_objective_id": None,
                "selected_concept_id": None,
                "boundary_decision": None,
                "verified": verified,
                "unserviceable": set(),
            }

        capacity = self._fresh_objective_focus_capacity(
            connection,
            session_id=session_id,
            learner_id=learner_id,
            release_id=release_id,
            focuses={(objective_id, None) for objective_id in unresolved},
            now=now,
        )
        serviceable: set[str] = set()
        for objective_id in unresolved:
            objective_capacity = capacity.get((objective_id, None), {})
            families = set(objective_capacity.get("families", set()))
            verification_families = set(
                objective_capacity.get("verification_families", set())
            )
            if any(
                verification_families - {repair_family}
                for repair_family in families
            ):
                serviceable.add(objective_id)
        unserviceable = unresolved - serviceable
        if not serviceable:
            return {
                "selected_objective_id": None,
                "selected_concept_id": None,
                "boundary_decision": None,
                "verified": verified,
                "unserviceable": unserviceable,
            }

        placeholders = ",".join("?" for _ in serviceable)
        credible_attempt_clause = _CREDIBLE_ROUTING_ATTEMPT_SQL
        performance_rows = connection.execute(
            f"""SELECT choice.question_objective_id AS objective_id,
                       COUNT(*) AS attempted,
                       SUM(CASE WHEN attempt.is_correct=0 THEN 1 ELSE 0 END)
                           AS incorrect
                FROM attempts attempt
                JOIN decisions choice ON choice.id=attempt.decision_id
                JOIN events response_event
                  ON response_event.event_id=attempt.event_id
                WHERE attempt.session_id=?
                  AND choice.question_objective_id IN ({placeholders})
                  {credible_attempt_clause}
                GROUP BY choice.question_objective_id""",
            (session_id, *sorted(serviceable)),
        ).fetchall()
        performance = {
            row["objective_id"]: (int(row["attempted"]), int(row["incorrect"]))
            for row in performance_rows
        }
        states = self.database.get_objective_states(learner_id, connection)
        candidates: list[dict[str, Any]] = []
        edge_by_source = {edge.source_id: edge for edge in direct_edges}
        for objective_id in sorted(serviceable):
            objective = objectives[objective_id]
            state = states.get(
                objective_id
            ) or self.learner_model.initial_objective_state(
                learner_id, objective
            )
            projected = self.learner_model.project_objective_state(
                state, objective, now
            )
            readiness = (
                0.55 * projected.mastery_probability
                + 0.45 * projected.expected_competence
            )
            need = 1.0 - readiness
            uncertainty_value = 1.0 - exp(-(projected.variance**0.5))
            evidence_gap = 1.0 / (1.0 + projected.evidence_mass)
            attempted, incorrect = performance.get(objective_id, (0, 0))
            failure_rate = incorrect / attempted if attempted else 0.0
            edge = edge_by_source[objective_id]
            score = (
                0.34 * need
                + 0.18 * (1.0 - projected.mastery_probability)
                + 0.14 * uncertainty_value
                + 0.12 * evidence_gap
                + 0.12 * failure_rate
                + 0.10 * edge.weight
            )
            candidates.append(
                {
                    "edge_id": edge.id,
                    "objective_id": objective_id,
                    "concept_id": objective.primary_concept_id,
                    "relation": edge.relation.value,
                    "rationale": edge.rationale,
                    "edge_weight": edge.weight,
                    "score": score,
                    "need": need,
                    "mastery_probability": projected.mastery_probability,
                    "expected_competence": projected.expected_competence,
                    "uncertainty_value": uncertainty_value,
                    "evidence_gap": evidence_gap,
                    "recent_failure_rate": failure_rate,
                }
            )
        candidates.sort(key=lambda item: (-item["score"], item["objective_id"]))
        selected = candidates[0]
        return {
            "selected_objective_id": selected["objective_id"],
            "selected_concept_id": selected["concept_id"],
            "boundary_decision": {
                "focus_objective_id": focus_objective_id,
                "selected_objective_id": selected["objective_id"],
                "algorithm_version": BOUNDARY_ALGORITHM_VERSION,
                "selected": selected,
                "candidates": candidates,
            },
            "verified": verified,
            "unserviceable": unserviceable,
        }

    def _fresh_focus_capacity(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        learner_id: str,
        release_id: str,
        concept_ids: set[str],
        now: datetime,
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
                LEFT JOIN release_question_objectives objective_mapping
                  ON objective_mapping.release_id = release_question.release_id
                 AND objective_mapping.question_id = question.id
                WHERE mapping.role = 'primary'
                  AND mapping.concept_id IN ({placeholders})
                  AND objective_mapping.objective_id IS NULL
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
        family_ids = {row["family_id"] for row in rows}
        last_presented = self._family_last_presented(
            connection, learner_id=learner_id, family_ids=family_ids
        )
        skill_states = self.database.get_skill_states(
            learner_id, connection
        )
        verification_kinds = {kind.value for kind in VERIFICATION_KINDS}
        for row in rows:
            state = skill_states.get(row["concept_id"])
            stability_hours = (
                state.stability_hours
                if state is not None
                else INITIAL_STABILITY_HOURS
            )
            family_last_at = last_presented.get(row["family_id"])
            if family_last_at is not None and (
                now - family_last_at
            ).total_seconds() / 3600.0 < max(
                24.0, min(24.0 * 30.0, stability_hours * 0.50)
            ):
                continue
            result[row["concept_id"]]["families"].add(row["family_id"])
            if row["kind"] in verification_kinds:
                result[row["concept_id"]]["verification_families"].add(
                    row["family_id"]
                )
        return result

    def _fresh_objective_focus_capacity(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        learner_id: str,
        release_id: str,
        focuses: set[tuple[str, str | None]],
        now: datetime,
    ) -> dict[tuple[str, str | None], dict[str, set[str]]]:
        """Return exact unseen capacity for objective/hypothesis focuses.

        A question contributes only when it directly assesses the focused
        objective. If a named misconception is active, the same option must be
        release-mapped back to that objective. This mirrors focused policy
        eligibility without relying on a bounded candidate query.
        """

        if not focuses:
            return {}
        objective_ids = sorted({objective_id for objective_id, _ in focuses})
        placeholders = ",".join("?" for _ in objective_ids)
        rows = connection.execute(
            f"""SELECT direct.objective_id, question.id AS question_id,
                       question.family_id, question.kind,
                       option.misconception_id,
                       diagnostic.objective_id AS diagnostic_objective_id
                FROM release_question_objectives direct
                JOIN questions question ON question.id = direct.question_id
                JOIN release_questions release_question
                  ON release_question.release_id = direct.release_id
                 AND release_question.question_id = direct.question_id
                LEFT JOIN options option
                  ON option.question_id = question.id
                 AND option.misconception_id IS NOT NULL
                LEFT JOIN release_option_objectives diagnostic
                  ON diagnostic.release_id = direct.release_id
                 AND diagnostic.question_id = option.question_id
                 AND diagnostic.option_id = option.option_id
                WHERE direct.release_id = ?
                  AND direct.objective_id IN ({placeholders})
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
                *objective_ids,
                QuestionStatus.APPROVED.value,
                QuestionStatus.CALIBRATED.value,
                session_id,
            ),
        ).fetchall()
        questions: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        for row in rows:
            key = (row["objective_id"], row["question_id"])
            summary = questions.setdefault(
                key,
                {
                    "family_id": row["family_id"],
                    "kind": row["kind"],
                    "diagnoses": set(),
                },
            )
            if (
                row["misconception_id"] is not None
                and row["diagnostic_objective_id"] is not None
            ):
                summary["diagnoses"].add(
                    (
                        row["diagnostic_objective_id"],
                        row["misconception_id"],
                    )
                )
        result = {
            focus: {"families": set(), "verification_families": set()}
            for focus in focuses
        }
        family_ids = {
            summary["family_id"] for summary in questions.values()
        }
        last_presented = self._family_last_presented(
            connection, learner_id=learner_id, family_ids=family_ids
        )
        objective_states = self.database.get_objective_states(
            learner_id, connection
        )
        verification_kinds = {kind.value for kind in VERIFICATION_KINDS}
        for (objective_id, _question_id), summary in questions.items():
            for focus in focuses:
                focus_objective_id, misconception_id = focus
                if focus_objective_id != objective_id:
                    continue
                if (
                    misconception_id is not None
                    and (objective_id, misconception_id)
                    not in summary["diagnoses"]
                ):
                    continue
                family_id = summary["family_id"]
                state = objective_states.get(objective_id)
                stability_hours = (
                    state.stability_hours
                    if state is not None
                    else INITIAL_STABILITY_HOURS
                )
                family_last_at = last_presented.get(family_id)
                if family_last_at is not None and (
                    now - family_last_at
                ).total_seconds() / 3600.0 < max(
                    24.0,
                    min(
                        24.0 * 30.0,
                        stability_hours * 0.50,
                    ),
                ):
                    continue
                result[focus]["families"].add(family_id)
                if summary["kind"] in verification_kinds:
                    result[focus]["verification_families"].add(
                        family_id
                    )
        return result

    @staticmethod
    def _family_last_presented(
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        family_ids: set[str],
    ) -> dict[str, datetime]:
        if not family_ids:
            return {}
        placeholders = ",".join("?" for _ in family_ids)
        rows = connection.execute(
            f"""SELECT question.family_id,
                       MAX(decision.created_at) AS last_at
                FROM decisions decision
                JOIN questions question ON question.id = decision.question_id
                WHERE decision.learner_id = ?
                  AND question.family_id IN ({placeholders})
                GROUP BY question.family_id""",
            (learner_id, *sorted(family_ids)),
        ).fetchall()
        result: dict[str, datetime] = {}
        for row in rows:
            try:
                parsed = datetime.fromisoformat(row["last_at"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Family {row['family_id']} has an invalid exposure timestamp."
                ) from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValidationError(
                    f"Family {row['family_id']} has a timezone-naive exposure timestamp."
                )
            result[row["family_id"]] = parsed.astimezone(timezone.utc)
        return result

    def _selected_response_inference_contract(
        self,
        release_id: str,
    ) -> dict[str, Any]:
        """Describe what the current selected-response numbers can support."""

        with self.database.read() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS active_questions,
                          COUNT(DISTINCT question.family_id) AS active_families,
                          SUM(CASE WHEN released.status = ? THEN 1 ELSE 0 END)
                              AS approved_questions,
                          SUM(CASE WHEN released.status = ? THEN 1 ELSE 0 END)
                              AS calibrated_questions,
                          COUNT(DISTINCT CASE
                              WHEN released.status = ? THEN question.family_id
                              ELSE NULL END) AS calibrated_families
                   FROM release_questions released
                   JOIN questions question
                     ON question.id = released.question_id
                   WHERE released.release_id = ?
                     AND released.status IN (?, ?)
                     AND NOT EXISTS (
                         SELECT 1
                         FROM question_revocations revoked
                         WHERE revoked.question_id = released.question_id
                     )""",
                (
                    QuestionStatus.APPROVED.value,
                    QuestionStatus.CALIBRATED.value,
                    QuestionStatus.CALIBRATED.value,
                    release_id,
                    QuestionStatus.APPROVED.value,
                    QuestionStatus.CALIBRATED.value,
                ),
            ).fetchone()
        active_questions = int(row["active_questions"] or 0)
        approved_questions = int(row["approved_questions"] or 0)
        calibrated_questions = int(row["calibrated_questions"] or 0)
        if active_questions == 0:
            corpus_calibration_status = "no_active_items"
        elif calibrated_questions == 0:
            corpus_calibration_status = "no_calibrated_items"
        elif calibrated_questions < active_questions:
            corpus_calibration_status = "partially_calibrated_items"
        else:
            corpus_calibration_status = "all_items_marked_calibrated"
        return {
            "contract_version": "selected-response-inference-v1",
            "corpus_release_id": release_id,
            "count_scope": "entire_corpus_release",
            "claim_scope": "provisional_selected_response_inference",
            "model_validation_status": "not_empirically_validated",
            "corpus_calibration_status": corpus_calibration_status,
            "eligible_question_count": active_questions,
            "eligible_family_count": int(row["active_families"] or 0),
            "approved_question_count": approved_questions,
            "calibrated_question_count": calibrated_questions,
            "calibrated_family_count": int(
                row["calibrated_families"] or 0
            ),
            "eligibility_definition": (
                "Release members currently marked approved or calibrated and "
                "not globally revoked."
            ),
            "calibrated_family_definition": (
                "Distinct eligible families containing at least one item whose "
                "release status is calibrated."
            ),
            "item_parameter_basis": (
                "Approved items use authored priors. A corpus-declared "
                "calibrated status is reported separately but does not by "
                "itself validate the inference model or target population."
            ),
            "state_label_qualification": (
                "Labels such as proficient or durable are provisional "
                "selected-response states, not validated real-world skill "
                "certificates."
            ),
            "numerical_guard_scope": (
                "For exact-grid posteriors, mastery_probability_error_bound is "
                "a conservative numerical approximation guard, not a proven "
                "universal error envelope. It excludes item-parameter "
                "uncertainty, model misspecification, semantic family dependence, "
                "population calibration, and transfer to productive skill. A "
                "zero value on a legacy gaussian_moments projection is not a "
                "validated numerical or model-error bound."
            ),
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
            self.database.require_learner_evidence_safe(
                learner_id,
                connection,
            )
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
        active_release = self.database.get_active_release_id()
        release_objectives = self.database.get_learning_objectives(active_release)
        objectives = [
            objective
            for objective in release_objectives
            if set(objective.concept_ids) & scope
        ]
        persisted_concept_states = self.database.get_skill_states(learner_id)
        stored_objectives = self.database.get_objective_states(learner_id)
        floor_projection = (
            self.learner_model.concept_projection_with_objective_floor(
                learner_id=learner_id,
                concepts=graph.concepts,
                stored_states=persisted_concept_states,
                objectives=release_objectives,
                stored_objective_states=stored_objectives,
                now=now,
            )
        )
        stored = floor_projection.states
        all_readiness = self.boundary_planner.readiness_map(
            learner_id=learner_id,
            graph=graph,
            stored_states=stored,
            now=now,
            concept_ids=set(graph.concepts),
            intrinsic_overrides=floor_projection.exact_floors,
        )
        readiness = {
            concept_id: all_readiness[concept_id] for concept_id in scope
        }
        evidence_summaries = self.database.independent_evidence_summaries(
            learner_id,
            set(graph.concepts),
            release_id=active_release,
        )
        objective_ids_by_primary_concept: dict[str, list[str]] = {}
        for objective in release_objectives:
            objective_ids_by_primary_concept.setdefault(
                objective.primary_concept_id, []
            ).append(objective.id)
        release_objective_ids = {
            objective.id for objective in release_objectives
        }
        objective_evidence = {
            objective_id: {
                "families": 0,
                "delayed": 0,
                "operation_kinds": 0,
            }
            for objective_id in release_objective_ids
        }
        observed_objective_families = {
            objective_id: 0 for objective_id in release_objective_ids
        }
        if release_objective_ids:
            with self.database.read() as connection:
                objective_evidence_rows = connection.execute(
                    """SELECT evidence.objective_id, COUNT(*) AS families,
                              COUNT(DISTINCT evidence.kind) AS operation_kinds,
                              SUM(CASE WHEN evidence.delayed_unguided_correct_at
                                                  IS NOT NULL
                                       THEN 1 ELSE 0 END) AS delayed
                       FROM learner_objective_families evidence
                       WHERE evidence.learner_id = ?
                         AND EXISTS (
                             SELECT 1
                             FROM release_question_objectives direct
                             JOIN questions question
                               ON question.id = direct.question_id
                             JOIN release_questions released
                               ON released.release_id = direct.release_id
                              AND released.question_id = direct.question_id
                             WHERE direct.release_id = ?
                               AND direct.objective_id = evidence.objective_id
                               AND question.family_id = evidence.family_id
                               AND released.status IN (?, ?)
                               AND NOT EXISTS (
                                   SELECT 1
                                   FROM question_revocations revoked
                                   WHERE revoked.question_id = question.id
                               )
                         )
                       GROUP BY evidence.objective_id""",
                    (
                        learner_id,
                        active_release,
                        QuestionStatus.APPROVED.value,
                        QuestionStatus.CALIBRATED.value,
                    ),
                ).fetchall()
                observed_objective_rows = connection.execute(
                    """SELECT decision.question_objective_id AS objective_id,
                              COUNT(DISTINCT attempt.family_id) AS families
                       FROM attempts attempt
                       JOIN decisions decision
                         ON decision.id = attempt.decision_id
                       JOIN release_question_objectives direct
                         ON direct.release_id = ?
                        AND direct.question_id = attempt.question_id
                        AND direct.objective_id
                            = decision.question_objective_id
                       JOIN release_questions released
                         ON released.release_id = direct.release_id
                        AND released.question_id = direct.question_id
                       WHERE attempt.learner_id = ?
                         AND decision.question_objective_id IS NOT NULL
                         AND released.status IN (?, ?)
                         AND NOT EXISTS (
                             SELECT 1
                             FROM question_revocations revoked
                             WHERE revoked.question_id = attempt.question_id
                         )
                       GROUP BY decision.question_objective_id""",
                    (
                        active_release,
                        learner_id,
                        QuestionStatus.APPROVED.value,
                        QuestionStatus.CALIBRATED.value,
                    ),
                ).fetchall()
            for row in objective_evidence_rows:
                if row["objective_id"] in objective_evidence:
                    objective_evidence[row["objective_id"]] = {
                        "families": int(row["families"] or 0),
                        "delayed": int(row["delayed"] or 0),
                        "operation_kinds": int(row["operation_kinds"] or 0),
                    }
            for row in observed_objective_rows:
                if row["objective_id"] in observed_objective_families:
                    observed_objective_families[row["objective_id"]] = int(
                        row["families"] or 0
                    )
        observed_concept_families = {concept_id: 0 for concept_id in scope}
        if scope:
            placeholders = ",".join("?" for _ in scope)
            with self.database.read() as connection:
                observed_rows = connection.execute(
                    f"""SELECT mapping.concept_id,
                               COUNT(DISTINCT attempt.family_id) AS families
                        FROM attempts attempt
                        JOIN question_concepts mapping
                          ON mapping.question_id = attempt.question_id
                         AND mapping.role = 'primary'
                        JOIN release_questions released
                          ON released.release_id = ?
                         AND released.question_id = attempt.question_id
                        WHERE attempt.learner_id = ?
                          AND mapping.concept_id IN ({placeholders})
                          AND released.status IN (?, ?)
                          AND NOT EXISTS (
                              SELECT 1
                              FROM question_revocations revoked
                              WHERE revoked.question_id = attempt.question_id
                          )
                        GROUP BY mapping.concept_id""",
                    (
                        active_release,
                        learner_id,
                        *sorted(scope),
                        QuestionStatus.APPROVED.value,
                        QuestionStatus.CALIBRATED.value,
                    ),
                ).fetchall()
            for row in observed_rows:
                observed_concept_families[row["concept_id"]] = int(
                    row["families"] or 0
                )
        beliefs = self.database.get_misconception_beliefs(learner_id)
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
        for concept_id in graph.concepts:
            concept = graph.concepts[concept_id]
            state = stored.get(concept_id) or self.learner_model.initial_state(
                learner_id, concept
            )
            projected_states[concept_id] = self.learner_model.project_state(
                state, concept, now
            )

        def evidence_for_concept(concept_id: str) -> dict[str, int]:
            """Match a displayed broad readiness floor to its evidence ledger."""

            floor_objective_id = all_readiness[
                concept_id
            ].objective_floor_source_id
            if floor_objective_id is not None:
                return objective_evidence[floor_objective_id]
            return evidence_summaries[concept_id]

        def observed_families_for_concept(concept_id: str) -> int:
            floor_objective_id = all_readiness[
                concept_id
            ].objective_floor_source_id
            if floor_objective_id is not None:
                return observed_objective_families[floor_objective_id]
            return observed_concept_families.get(concept_id, 0)

        skills = []
        for concept_id in sorted(scope, key=lambda cid: graph.concepts[cid].name):
            concept = graph.concepts[concept_id]
            projected = projected_states[concept_id]
            boundary = readiness[concept_id]
            evidence = evidence_for_concept(concept_id)
            family_count = evidence["families"]
            delayed_retrievals = evidence["delayed"]
            operation_kinds = evidence["operation_kinds"]
            direct_prerequisites = [
                prerequisite_id
                for prerequisite_id, _ in graph.direct_prerequisites(concept_id)
            ]
            prerequisites_ready = all(
                prerequisite_id in all_readiness
                and all_readiness[prerequisite_id].exposures > 0
                and all_readiness[prerequisite_id].mastery_probability >= 0.40
                and evidence_for_concept(prerequisite_id)["families"] >= 1
                for prerequisite_id in direct_prerequisites
            )
            derived_objective_ids = sorted(
                objective_ids_by_primary_concept.get(concept_id, [])
            )
            skills.append(
                {
                    "concept_id": concept_id,
                    "name": concept.name,
                    "mastery": boundary.mastery_probability,
                    "expected_competence": boundary.expected_competence,
                    "uncertainty": boundary.uncertainty,
                    "stability_hours": projected.stability_hours,
                    "evidence_mass": projected.evidence_mass,
                    "projection_kind": (
                        "derived_objective_readiness_floor"
                        if derived_objective_ids
                        else "concept_posterior"
                    ),
                    "derived_from_objective_ids": derived_objective_ids,
                    "objective_floor_source_id": (
                        boundary.objective_floor_source_id
                    ),
                    "independent_families": family_count,
                    "successful_retrieval_families": family_count,
                    "observed_response_families": (
                        observed_families_for_concept(concept_id)
                    ),
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
                        mastery_probability_override=(
                            boundary.mastery_probability
                        ),
                    ),
                    "state_qualification": (
                        "provisional_selected_response_state"
                    ),
                    "next_review_at": projected.next_review_at.isoformat() if projected.next_review_at else None,
                }
            )
        misconception_hypotheses = [
            {
                "misconception_id": misconception_id,
                "name": definitions[misconception_id].name if misconception_id in definitions else misconception_id,
                "probability": belief.probability,
                "evidence_count": belief.evidence_count,
                "status": (
                    "active"
                    if belief.probability
                    >= MISCONCEPTION_ACTIVE_THRESHOLD
                    else "monitoring"
                ),
            }
            for misconception_id, belief in beliefs.items()
            if belief.probability >= MISCONCEPTION_MONITORING_THRESHOLD
            and misconception_id in definitions
        ]
        misconception_hypotheses.sort(
            key=lambda item: (-item["probability"], item["misconception_id"])
        )
        active_misconceptions = [
            item
            for item in misconception_hypotheses
            if item["status"] == "active"
        ]
        profile = {
            "learner_id": learner_id,
            "corpus_release_id": active_release,
            "target": resolved_target,
            "boundary_algorithm_version": BOUNDARY_ALGORITHM_VERSION,
            "skills": skills,
            "misconception_thresholds": {
                "monitoring": MISCONCEPTION_MONITORING_THRESHOLD,
                "active_routing": MISCONCEPTION_ACTIVE_THRESHOLD,
            },
            "misconception_hypotheses": misconception_hypotheses,
            "active_misconceptions": active_misconceptions,
            "selected_response_inference": (
                self._selected_response_inference_contract(active_release)
            ),
        }
        if objectives:
            objective_ids = {objective.id for objective in objectives}
            diagnostic_misconceptions: dict[str, set[str]] = {
                objective_id: set() for objective_id in objective_ids
            }
            with self.database.read() as connection:
                diagnostic_rows = connection.execute(
                    """SELECT DISTINCT mapping.objective_id,
                                      option.misconception_id
                       FROM release_option_objectives mapping
                       JOIN release_questions membership
                         ON membership.release_id = mapping.release_id
                        AND membership.question_id = mapping.question_id
                       JOIN options option
                         ON option.question_id = mapping.question_id
                        AND option.option_id = mapping.option_id
                       WHERE mapping.release_id = ?
                         AND membership.status IN (?, ?)
                         AND option.is_correct = 0
                         AND NOT EXISTS (
                             SELECT 1
                             FROM question_revocations revoked
                             WHERE revoked.question_id = mapping.question_id
                         )
                         AND option.misconception_id IS NOT NULL""",
                    (
                        active_release,
                        QuestionStatus.APPROVED.value,
                        QuestionStatus.CALIBRATED.value,
                    ),
                ).fetchall()
            for row in diagnostic_rows:
                if row["objective_id"] in diagnostic_misconceptions:
                    diagnostic_misconceptions[row["objective_id"]].add(
                        row["misconception_id"]
                    )

            projected_objectives = {}
            for release_objective in release_objectives:
                release_state = stored_objectives.get(
                    release_objective.id
                ) or self.learner_model.initial_objective_state(
                    learner_id, release_objective
                )
                projected_objectives[release_objective.id] = (
                    self.learner_model.project_objective_state(
                        release_state, release_objective, now
                    )
                )
            objective_rows = []
            for objective in objectives:
                projected = projected_objectives[objective.id]
                evidence = objective_evidence[objective.id]
                if objective.objective_graph_version is not None:
                    prerequisite_objective_ids = [
                        edge.source_id for edge in objective.prerequisites
                    ]
                    objective_prerequisites_ready = all(
                        projected_objectives[prerequisite_id].exposures > 0
                        and projected_objectives[
                            prerequisite_id
                        ].mastery_probability
                        >= 0.40
                        and objective_evidence[prerequisite_id]["families"] >= 1
                        for prerequisite_id in prerequisite_objective_ids
                    )
                    prerequisite_support = min(
                        (
                            1.0
                            - edge.weight
                            * (
                                1.0
                                - (
                                    0.55
                                    * projected_objectives[
                                        edge.source_id
                                    ].mastery_probability
                                    + 0.45
                                    * projected_objectives[
                                        edge.source_id
                                    ].expected_competence
                                )
                            )
                            for edge in objective.prerequisites
                        ),
                        default=1.0,
                    )
                    prerequisite_mode = "objective"
                else:
                    prerequisite_objective_ids = []
                    concept_prerequisites = [
                        prerequisite_id
                        for prerequisite_id, _ in graph.direct_prerequisites(
                            objective.primary_concept_id
                        )
                    ]
                    objective_prerequisites_ready = all(
                        all_readiness[prerequisite_id].exposures > 0
                        and all_readiness[
                            prerequisite_id
                        ].mastery_probability
                        >= 0.40
                        and evidence_summaries[prerequisite_id]["families"] >= 1
                        for prerequisite_id in concept_prerequisites
                    )
                    prerequisite_support = all_readiness[
                        objective.primary_concept_id
                    ].prerequisite_support
                    prerequisite_mode = "legacy_concept"
                active_probability = max(
                    (
                        beliefs[misconception_id].probability
                        for misconception_id in diagnostic_misconceptions[
                            objective.id
                        ]
                        if misconception_id in beliefs
                    ),
                    default=0.0,
                )
                objective_rows.append(
                    {
                        "objective_id": objective.id,
                        "name": objective.name,
                        "description": objective.description,
                        "operation": objective.operation.value,
                        "evidence_type": objective.evidence_type,
                        "primary_concept_id": objective.primary_concept_id,
                        "supporting_concept_ids": list(
                            objective.supporting_concept_ids
                        ),
                        "mastery": projected.mastery_probability,
                        "mastery_probability": projected.mastery_probability,
                        "estimated_mastery_probability": (
                            projected.estimated_mastery_probability
                        ),
                        "mastery_probability_error_bound": (
                            projected.mastery_probability_error_bound
                        ),
                        "expected_competence": projected.expected_competence,
                        "uncertainty": projected.variance**0.5,
                        "stability_hours": projected.stability_hours,
                        "evidence_mass": projected.evidence_mass,
                        "acquisition_mass": projected.acquisition_mass,
                        "inference_model_version": projected.model_version,
                        "posterior_representation": (
                            "exact_grid"
                            if projected.posterior is not None
                            else "gaussian_moments"
                        ),
                        "posterior_digest": (
                            projected.posterior.digest
                            if projected.posterior is not None
                            else None
                        ),
                        "independent_families": evidence["families"],
                        "successful_retrieval_families": evidence[
                            "families"
                        ],
                        "observed_response_families": (
                            observed_objective_families[objective.id]
                        ),
                        "delayed_retrievals": evidence["delayed"],
                        "operation_kinds": evidence["operation_kinds"],
                        "active_misconception_probability": active_probability,
                        "prerequisite_mode": prerequisite_mode,
                        "prerequisite_objective_ids": prerequisite_objective_ids,
                        "prerequisites_ready": objective_prerequisites_ready,
                        "prerequisite_support": prerequisite_support,
                        "state": self.learner_model.mastery_label(
                            projected,
                            evidence["families"],
                            evidence["delayed"],
                            evidence["operation_kinds"],
                            active_probability,
                            objective_prerequisites_ready,
                        ),
                        "state_qualification": (
                            "provisional_selected_response_state"
                        ),
                        "next_review_at": (
                            projected.next_review_at.isoformat()
                            if projected.next_review_at
                            else None
                        ),
                    }
                )
            profile["learning_objectives"] = objective_rows
            profile["objective_evidence_scope"] = (
                "Selected-response evidence for each named objective; it is not "
                "evidence of implementation, explanation, design, or project skill."
            )
        profile["family_evidence_definitions"] = {
            "observed_response_families": (
                "Distinct independence families attempted, including incorrect, "
                "abstained, hinted, fast, and low-confidence responses."
            ),
            "successful_retrieval_families": (
                "Distinct families with at least one correct, unhinted, "
                "credible retrieval; this is an internal selected-response "
                "verification gate, not real-world skill certification."
            ),
        }
        profile["projection_definitions"] = {
            "concept_posterior": (
                "A directly maintained broad-concept projection for legacy "
                "concept-scored evidence."
            ),
            "derived_objective_readiness_floor": (
                "A non-persisted routing floor derived from the weakest named "
                "objective owned by the concept; it is not a second posterior "
                "or independent evidence."
            ),
            "mastery_probability": (
                "The conservative probability used for decisions; exact-grid "
                "estimates subtract their numerical error bound."
            ),
            "mastery_probability_error_bound": (
                "A numerical approximation guard only; it does not cover "
                "item calibration, model validity, family dependence, or "
                "productive-skill transfer."
            ),
            "prerequisites_ready": (
                "Every direct prerequisite has at least one certified independent "
                "family and a current retention-adjusted mastery probability of "
                "at least 0.40."
            ),
        }
        profile["productive_skill_shadow"] = productive_shadow_summary(
            self.database,
            learner_id,
            concept_ids=scope,
        )
        with self.database.read() as connection:
            self.database.require_learner_evidence_safe(
                learner_id,
                connection,
            )
        return profile

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
            self.database.require_learner_evidence_safe(
                session["learner_id"],
                connection,
            )
            rows = connection.execute(
                """SELECT decision.id AS decision_id, decision.question_id,
                          decision.question_objective_id,
                          decision.phase, decision.pedagogical_role,
                          decision.focus_concept_id AS focus_concept_before,
                          decision.focus_misconception_id
                              AS focus_misconception_before,
                          decision.focus_objective_id AS focus_objective_before,
                          decision.selected_score_json, decision.created_at,
                          selection_event.occurred_at
                              AS selection_occurred_at,
                          question.family_id, question.difficulty, question.kind,
                          primary_mapping.concept_id AS primary_concept_id,
                          attempt.id AS attempt_id, attempt.is_correct,
                          attempt.selected_option_id, attempt.confidence,
                          attempt.response_ms, attempt.hint_count,
                          attempt.answered_at, attempt.outcome_json,
                          json_extract(
                              response_event.metadata_json,
                              '$.learner_model_version'
                          ) AS response_model_version,
                          selected_option.misconception_id
                              AS selected_misconception_id,
                          selected_diagnostic.objective_id
                              AS selected_diagnostic_objective_id,
                          (SELECT topic_map.topic_id
                           FROM release_question_topics topic_map
                           WHERE topic_map.release_id = decision.corpus_release_id
                             AND topic_map.question_id = decision.question_id
                             AND topic_map.relation = 'primary'
                           ORDER BY topic_map.topic_id LIMIT 1
                          ) AS primary_topic_id,
                          (SELECT COUNT(*) FROM release_question_topics topic_map
                           WHERE topic_map.release_id = decision.corpus_release_id
                             AND topic_map.question_id = decision.question_id
                          ) AS topic_count
                   FROM decisions decision
                   JOIN questions question ON question.id = decision.question_id
                   JOIN question_concepts primary_mapping
                     ON primary_mapping.question_id = decision.question_id
                    AND primary_mapping.role = 'primary'
                   LEFT JOIN events selection_event
                     ON selection_event.event_type = 'QuestionSelected'
                    AND selection_event.session_id = decision.session_id
                    AND json_extract(
                        selection_event.payload_json, '$.decision_id'
                    ) = decision.id
                   LEFT JOIN attempts attempt
                     ON attempt.decision_id = decision.id
                   LEFT JOIN events response_event
                     ON response_event.event_id = attempt.event_id
                    AND response_event.event_type = 'ResponseSubmitted'
                   LEFT JOIN options selected_option
                     ON selected_option.question_id = attempt.question_id
                    AND selected_option.option_id = attempt.selected_option_id
                   LEFT JOIN release_option_objectives selected_diagnostic
                     ON selected_diagnostic.release_id = decision.corpus_release_id
                    AND selected_diagnostic.question_id = attempt.question_id
                    AND selected_diagnostic.option_id = attempt.selected_option_id
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
            action_rows = connection.execute(
                """SELECT action.id, action.decision_id, action.sequence, action.stage,
                          action.action_type, action.payload_json,
                          action.artifact_id, action.occurred_at
                   FROM learning_actions action
                   WHERE action.session_id = ?
                   ORDER BY action.occurred_at, action.decision_id, action.sequence""",
                (session_id,),
            ).fetchall()
            position_rows = connection.execute(
                """SELECT decision.id AS decision_id,
                          decision.question_id,
                          decision.option_order_json
                              AS decision_option_order_json,
                          attempt.selected_option_id,
                          attempt.presented_order_json
                              AS attempt_presented_order_json,
                          (
                              SELECT json_group_array(option.option_id)
                              FROM options option
                              WHERE option.question_id = decision.question_id
                          ) AS question_option_ids_json,
                          response.event_id AS response_event_id,
                          response.stream_id AS response_stream_id,
                          response.stream_version AS response_stream_version,
                          response.event_type AS response_event_type,
                          response.schema_version AS response_schema_version,
                          response.occurred_at AS response_occurred_at,
                          response.recorded_at AS response_recorded_at,
                          response.learner_id AS response_learner_id,
                          response.session_id AS response_session_id,
                          response.correlation_id AS response_correlation_id,
                          response.causation_id AS response_causation_id,
                          response.idempotency_key AS response_idempotency_key,
                          response.payload_json AS response_payload_json,
                          response.metadata_json AS response_metadata_json,
                          response.previous_hash AS response_previous_hash,
                          response.payload_hash AS response_payload_hash
                   FROM attempts attempt
                   JOIN decisions decision ON decision.id = attempt.decision_id
                   LEFT JOIN events response
                     ON response.event_id = attempt.event_id
                   WHERE attempt.session_id = ?
                     AND decision.invalidated_at IS NULL
                   ORDER BY attempt.answered_at, attempt.id""",
                (session_id,),
            ).fetchall()
            position_selection_event_rows = connection.execute(
                """SELECT event.event_id AS selection_event_id,
                          event.stream_id AS selection_stream_id,
                          event.stream_version AS selection_stream_version,
                          event.event_type AS selection_event_type,
                          event.schema_version AS selection_schema_version,
                          event.occurred_at AS selection_occurred_at,
                          event.recorded_at AS selection_recorded_at,
                          event.learner_id AS selection_learner_id,
                          event.session_id AS selection_session_id,
                          event.correlation_id AS selection_correlation_id,
                          event.causation_id AS selection_causation_id,
                          event.idempotency_key AS selection_idempotency_key,
                          event.payload_json AS selection_payload_json,
                          event.metadata_json AS selection_metadata_json,
                          event.previous_hash AS selection_previous_hash,
                          event.payload_hash AS selection_payload_hash
                   FROM events event
                   WHERE event.session_id = ?
                     AND event.event_type = 'QuestionSelected'
                   ORDER BY event.stream_version, event.event_id""",
                (session_id,),
            ).fetchall()

        answered = [row for row in rows if row["attempt_id"] is not None]
        response_position_shadow = display_position_shadow(
            (dict(row) for row in position_rows),
            (dict(row) for row in position_selection_event_rows),
        )
        difficulties = [float(row["difficulty"]) for row in answered]
        response_times = [
            int(row["response_ms"])
            for row in answered
            if row["response_ms"] is not None
        ]
        timing_inconsistencies = 0
        for row in answered:
            if (
                row["response_ms"] is None
                or row["response_model_version"]
                not in AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS
            ):
                continue
            try:
                selected_at = datetime.fromisoformat(
                    row["selection_occurred_at"]
                )
                answered_at = datetime.fromisoformat(row["answered_at"])
                authoritative_window = response_window(
                    selected_at=selected_at,
                    answered_at=answered_at,
                    response_ms=int(row["response_ms"]),
                )
            except (TypeError, ValueError, OverflowError):
                timing_inconsistencies += 1
                continue
            if not authoritative_window.consistent:
                timing_inconsistencies += 1
        predicted = []
        continuity = []
        for row in answered:
            score = json.loads(row["selected_score_json"])
            if isinstance(score.get("predicted_correct"), (int, float)):
                predicted.append(float(score["predicted_correct"]))
            if isinstance(score.get("continuity"), (int, float)):
                continuity.append(float(score["continuity"]))

        outcome_by_attempt: dict[str, dict[str, Any]] = {}
        retrieval_certified_by_attempt: dict[str, bool] = {}
        response_class_by_attempt: dict[str, ResponseClass] = {}
        for row in answered:
            response_class_by_attempt[row["attempt_id"]] = (
                classify_response_for_model(
                    model_version=row["response_model_version"],
                    correct=bool(row["is_correct"]),
                    selected_option_id=row["selected_option_id"],
                    selected_misconception_id=(
                        row["selected_misconception_id"]
                    ),
                    confidence=(
                        float(row["confidence"])
                        if row["confidence"] is not None
                        else None
                    ),
                    response_ms=(
                        int(row["response_ms"])
                        if row["response_ms"] is not None
                        else None
                    ),
                    hint_count=int(row["hint_count"]),
                )
            )
            if not row["outcome_json"]:
                outcome_by_attempt[row["attempt_id"]] = {}
                retrieval_certified_by_attempt[row["attempt_id"]] = False
                continue
            outcome = json.loads(row["outcome_json"])
            if type(outcome) is not dict:
                raise ValidationError(
                    f"Attempt {row['attempt_id']} has a non-object outcome."
                )
            outcome_by_attempt[row["attempt_id"]] = outcome
            retrieval_certified_by_attempt[row["attempt_id"]] = any(
                change.get("retrieval_certified") is True
                for change in outcome.get("state_changes", [])
                if type(change) is dict
            )

        def credible_verification_failure(row: Any) -> bool:
            response_class = response_class_by_attempt[row["attempt_id"]]
            return bool(
                row["pedagogical_role"] == "verification"
                and response_class.supports_failure_localization
            )

        def inconclusive_verification(row: Any) -> bool:
            if row["pedagogical_role"] != "verification":
                return False
            response_class = response_class_by_attempt[row["attempt_id"]]
            return not (
                response_class.certifies_retrieval
                or response_class.supports_failure_localization
            )

        state_by_concept: dict[str, dict[str, float]] = {}
        state_by_objective: dict[str, dict[str, float]] = {}
        total_evidence_delta = 0.0
        for row in answered:
            if not row["outcome_json"]:
                continue
            outcome = outcome_by_attempt[row["attempt_id"]]
            for change in outcome.get("state_changes", []):
                evidence_delta = float(change.get("evidence_delta", 0.0))
                total_evidence_delta += evidence_delta
                concept_id = change.get("concept_id")
                objective_id = change.get("objective_id")
                if isinstance(concept_id, str):
                    state_key = concept_id
                    state_summaries = state_by_concept
                elif isinstance(objective_id, str):
                    state_key = objective_id
                    state_summaries = state_by_objective
                else:
                    continue
                summary = state_summaries.setdefault(
                    state_key,
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
        selected_answers = len(answered) - abstained
        selected_incorrect = selected_answers - correct_count
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
        requested_topic_ids: set[str] = set()
        if topic is not None:
            requested_topic_ids.add(topic["id"])
            catalog_topics = self.database.get_catalog(
                session["corpus_release_id"]
            )["topics"]
            changed = True
            while changed:
                changed = False
                for candidate in catalog_topics:
                    if (
                        candidate["parent_id"] in requested_topic_ids
                        and candidate["id"] not in requested_topic_ids
                    ):
                        requested_topic_ids.add(candidate["id"])
                        changed = True
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
        objective_changes = [
            {
                "objective_id": objective_id,
                **values,
                "mastery_change": values["posterior_mastery"]
                - values["prior_mastery"],
            }
            for objective_id, values in state_by_objective.items()
        ]
        objective_changes.sort(
            key=lambda item: (
                -abs(item["evidence_delta"]),
                item["objective_id"],
            )
        )

        graph = self.database.get_graph(session["corpus_release_id"])
        release_objectives = self.database.get_learning_objectives(
            session["corpus_release_id"]
        )
        objective_by_id = {
            objective.id: objective for objective in release_objectives
        }
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
            after_focus_objective = outcome.get("focus_objective_id")
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
                    "question_objective_id": row["question_objective_id"],
                    "question_objective_name": (
                        objective_by_id[row["question_objective_id"]].name
                        if row["question_objective_id"] in objective_by_id
                        else None
                    ),
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
                    "focus_objective_before": row[
                        "focus_objective_before"
                    ],
                    "focus_objective_before_name": (
                        objective_by_id[row["focus_objective_before"]].name
                        if row["focus_objective_before"] in objective_by_id
                        else None
                    ),
                    "focus_objective_after": after_focus_objective,
                    "focus_objective_after_name": (
                        objective_by_id[after_focus_objective].name
                        if after_focus_objective in objective_by_id
                        else None
                    ),
                    "selected_misconception_id": selected_misconception_id,
                    "selected_misconception_name": (
                        definitions[selected_misconception_id].name
                        if selected_misconception_id in definitions
                        else None
                    ),
                    "selected_diagnostic_objective_id": row[
                        "selected_diagnostic_objective_id"
                    ],
                    "selected_diagnostic_objective_name": (
                        objective_by_id[
                            row["selected_diagnostic_objective_id"]
                        ].name
                        if row["selected_diagnostic_objective_id"]
                        in objective_by_id
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
        stored_objective_states = self.database.get_objective_states(
            session["learner_id"]
        )
        floor_projection = (
            self.learner_model.concept_projection_with_objective_floor(
                learner_id=session["learner_id"],
                concepts=graph.concepts,
                stored_states=stored_states,
                objectives=release_objectives,
                stored_objective_states=stored_objective_states,
                now=now,
            )
        )
        stored_states = floor_projection.states
        current_readiness = self.boundary_planner.readiness_map(
            learner_id=session["learner_id"],
            graph=graph,
            stored_states=stored_states,
            now=now,
            concept_ids=seen_concepts,
            intrinsic_overrides=floor_projection.exact_floors,
        ) if seen_concepts else {}
        evidence = self.database.independent_evidence_summaries(
            session["learner_id"],
            seen_concepts,
            release_id=session["corpus_release_id"],
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
                    "selected_answers": 0,
                    "selected_incorrect": 0,
                    "uncertain_responses": 0,
                    "remediation_questions": 0,
                    "verification_failures": 0,
                    "verification_inconclusive": 0,
                    "missing_response_time": 0,
                    "difficulties": [],
                    "misconception_signals": Counter(),
                    "observed_families": set(),
                    "correct_response_families": set(),
                    "successful_retrieval_families": set(),
                },
            )
            summary["attempted"] += 1
            summary["correct"] += int(bool(row["is_correct"]))
            summary["abstained"] += int(row["selected_option_id"] is None)
            summary["selected_answers"] += int(
                row["selected_option_id"] is not None
            )
            summary["selected_incorrect"] += int(
                row["selected_option_id"] is not None
                and not bool(row["is_correct"])
            )
            summary["uncertain_responses"] += int(
                response_class_by_attempt[row["attempt_id"]]
                in {
                    ResponseClass.NONCREDIBLE_SUCCESS,
                    ResponseClass.UNCERTAIN_OR_ABSTAINED,
                }
            )
            summary["remediation_questions"] += int(
                row["pedagogical_role"]
                in {"remediation_probe", "verification"}
            )
            summary["verification_failures"] += int(
                credible_verification_failure(row)
            )
            summary["verification_inconclusive"] += int(
                inconclusive_verification(row)
            )
            summary["missing_response_time"] += int(
                row["response_ms"] is None
            )
            summary["difficulties"].append(float(row["difficulty"]))
            summary["observed_families"].add(row["family_id"])
            if bool(row["is_correct"]):
                summary["correct_response_families"].add(
                    row["family_id"]
                )
                if retrieval_certified_by_attempt[row["attempt_id"]]:
                    summary["successful_retrieval_families"].add(
                        row["family_id"]
                    )
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
                    "selected_answers": 0,
                    "selected_incorrect": 0,
                    "uncertain_responses": 0,
                    "remediation_questions": 0,
                    "verification_failures": 0,
                    "verification_inconclusive": 0,
                    "missing_response_time": 0,
                    "difficulties": [],
                    "misconception_signals": Counter(),
                    "observed_families": set(),
                    "correct_response_families": set(),
                    "successful_retrieval_families": set(),
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
                    belief is None
                    or belief.probability
                    < MISCONCEPTION_MONITORING_THRESHOLD
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
            if observed["selected_incorrect"]:
                attention_reasons.append("incorrect_responses")
            if observed["uncertain_responses"]:
                attention_reasons.append("uncertain_or_noncredible_evidence")
            if observed["verification_failures"]:
                attention_reasons.append("failed_independent_verification")
            if observed["verification_inconclusive"]:
                attention_reasons.append(
                    "inconclusive_independent_verification"
                )
            if concept_id in boundary_concepts:
                attention_reasons.append("selected_prerequisite_boundary")
            if any(
                (item["posterior_probability"] or 0.0)
                >= MISCONCEPTION_ACTIVE_THRESHOLD
                for item in active_hypotheses
            ):
                attention_reasons.append("active_misconception_hypothesis")
            if readiness.evidence_mass > 0 and readiness.mastery_probability < 0.50:
                attention_reasons.append("low_current_mastery_probability")
            difficulties_for_concept = observed["difficulties"]
            family_count = evidence[concept_id]["families"]
            derived_objective_ids = sorted(
                objective.id
                for objective in release_objectives
                if objective.primary_concept_id == concept_id
            )
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
                        "selected_answers": observed["selected_answers"],
                        "selected_incorrect": observed[
                            "selected_incorrect"
                        ],
                        "selected_accuracy": (
                            observed["correct"]
                            / observed["selected_answers"]
                            if observed["selected_answers"]
                            else None
                        ),
                        "uncertain_responses": observed[
                            "uncertain_responses"
                        ],
                        "remediation_questions": observed[
                            "remediation_questions"
                        ],
                        "verification_failures": observed[
                            "verification_failures"
                        ],
                        "verification_inconclusive": observed[
                            "verification_inconclusive"
                        ],
                        "missing_response_time": observed[
                            "missing_response_time"
                        ],
                        "average_difficulty": (
                            mean(difficulties_for_concept)
                            if difficulties_for_concept
                            else None
                        ),
                        "observed_families": len(
                            observed["observed_families"]
                        ),
                        "correct_response_families": len(
                            observed["correct_response_families"]
                        ),
                        "successful_retrieval_families": len(
                            observed["successful_retrieval_families"]
                        ),
                    },
                    "current_projection": {
                        "mastery_probability": readiness.mastery_probability,
                        "expected_competence": readiness.expected_competence,
                        "uncertainty": readiness.uncertainty,
                        "evidence_mass": readiness.evidence_mass,
                        "projection_kind": (
                            "derived_objective_readiness_floor"
                            if derived_objective_ids
                            else "concept_posterior"
                        ),
                        "derived_from_objective_ids": derived_objective_ids,
                        "objective_floor_source_id": (
                            readiness.objective_floor_source_id
                        ),
                        "independent_families": family_count,
                        "successful_retrieval_families": family_count,
                        "successful_retrieval_diversity": (
                            _family_diversity_label(family_count)
                        ),
                        "evidence_diversity": _family_diversity_label(
                            family_count
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

        seen_objective_ids = {
            row["question_objective_id"]
            for row in answered
            if row["question_objective_id"] in objective_by_id
        }
        objective_family_counts = {
            objective_id: 0 for objective_id in seen_objective_ids
        }
        if seen_objective_ids:
            placeholders = ",".join("?" for _ in seen_objective_ids)
            with self.database.read() as connection:
                family_rows = connection.execute(
                    f"""SELECT evidence.objective_id, COUNT(*) AS families
                         FROM learner_objective_families evidence
                         WHERE evidence.learner_id = ?
                           AND evidence.objective_id IN ({placeholders})
                           AND EXISTS (
                               SELECT 1
                               FROM release_question_objectives direct
                               JOIN questions question
                                 ON question.id = direct.question_id
                               JOIN release_questions released
                                 ON released.release_id = direct.release_id
                                AND released.question_id = direct.question_id
                               WHERE direct.release_id = ?
                                 AND direct.objective_id
                                     = evidence.objective_id
                                 AND question.family_id = evidence.family_id
                                 AND released.status IN (?, ?)
                                 AND NOT EXISTS (
                                     SELECT 1
                                     FROM question_revocations revoked
                                     WHERE revoked.question_id = question.id
                                 )
                           )
                         GROUP BY evidence.objective_id""",
                    (
                        session["learner_id"],
                        *sorted(seen_objective_ids),
                        session["corpus_release_id"],
                        QuestionStatus.APPROVED.value,
                        QuestionStatus.CALIBRATED.value,
                    ),
                ).fetchall()
            for row in family_rows:
                objective_family_counts[row["objective_id"]] = int(
                    row["families"] or 0
                )

        session_by_objective: dict[str, dict[str, Any]] = {}
        for row in answered:
            objective_id = row["question_objective_id"]
            if objective_id not in seen_objective_ids:
                continue
            observed = session_by_objective.setdefault(
                objective_id,
                {
                    "attempted": 0,
                    "correct": 0,
                    "abstained": 0,
                    "selected_answers": 0,
                    "selected_incorrect": 0,
                    "uncertain_responses": 0,
                    "remediation_questions": 0,
                    "verification_failures": 0,
                    "verification_inconclusive": 0,
                    "missing_response_time": 0,
                    "difficulties": [],
                    "observed_families": set(),
                    "correct_response_families": set(),
                    "successful_retrieval_families": set(),
                },
            )
            observed["attempted"] += 1
            observed["correct"] += int(bool(row["is_correct"]))
            observed["abstained"] += int(
                row["selected_option_id"] is None
            )
            observed["selected_answers"] += int(
                row["selected_option_id"] is not None
            )
            observed["selected_incorrect"] += int(
                row["selected_option_id"] is not None
                and not bool(row["is_correct"])
            )
            observed["uncertain_responses"] += int(
                response_class_by_attempt[row["attempt_id"]]
                in {
                    ResponseClass.NONCREDIBLE_SUCCESS,
                    ResponseClass.UNCERTAIN_OR_ABSTAINED,
                }
            )
            observed["remediation_questions"] += int(
                row["pedagogical_role"]
                in {"remediation_probe", "verification"}
            )
            observed["verification_failures"] += int(
                credible_verification_failure(row)
            )
            observed["verification_inconclusive"] += int(
                inconclusive_verification(row)
            )
            observed["missing_response_time"] += int(
                row["response_ms"] is None
            )
            observed["difficulties"].append(float(row["difficulty"]))
            observed["observed_families"].add(row["family_id"])
            if bool(row["is_correct"]):
                observed["correct_response_families"].add(
                    row["family_id"]
                )
                if retrieval_certified_by_attempt[row["attempt_id"]]:
                    observed["successful_retrieval_families"].add(
                        row["family_id"]
                    )

        objective_performance: list[dict[str, Any]] = []
        for objective_id in sorted(
            seen_objective_ids,
            key=lambda item: objective_by_id[item].name,
        ):
            objective = objective_by_id[objective_id]
            observed = session_by_objective[objective_id]
            state = stored_objective_states.get(
                objective_id
            ) or self.learner_model.initial_objective_state(
                session["learner_id"], objective
            )
            projected = self.learner_model.project_objective_state(
                state, objective, now
            )
            family_count = objective_family_counts[objective_id]
            incorrect = observed["attempted"] - observed["correct"]
            attention_reasons: list[str] = []
            if observed["selected_incorrect"]:
                attention_reasons.append("incorrect_responses")
            if observed["uncertain_responses"]:
                attention_reasons.append(
                    "uncertain_or_noncredible_evidence"
                )
            if observed["verification_failures"]:
                attention_reasons.append(
                    "failed_independent_verification"
                )
            if observed["verification_inconclusive"]:
                attention_reasons.append(
                    "inconclusive_independent_verification"
                )
            if (
                projected.evidence_mass > 0
                and projected.mastery_probability < 0.50
            ):
                attention_reasons.append(
                    "low_current_mastery_probability"
                )
            objective_performance.append(
                {
                    "objective_id": objective_id,
                    "name": objective.name,
                    "description": objective.description,
                    "operation": objective.operation.value,
                    "evidence_type": objective.evidence_type,
                    "primary_concept_id": objective.primary_concept_id,
                    "supporting_concept_ids": list(
                        objective.supporting_concept_ids
                    ),
                    "session": {
                        "attempted": observed["attempted"],
                        "correct": observed["correct"],
                        "incorrect": incorrect,
                        "abstained": observed["abstained"],
                        "selected_answers": observed["selected_answers"],
                        "selected_incorrect": observed[
                            "selected_incorrect"
                        ],
                        "selected_accuracy": (
                            observed["correct"]
                            / observed["selected_answers"]
                            if observed["selected_answers"]
                            else None
                        ),
                        "uncertain_responses": observed[
                            "uncertain_responses"
                        ],
                        "remediation_questions": observed[
                            "remediation_questions"
                        ],
                        "verification_failures": observed[
                            "verification_failures"
                        ],
                        "verification_inconclusive": observed[
                            "verification_inconclusive"
                        ],
                        "missing_response_time": observed[
                            "missing_response_time"
                        ],
                        "average_difficulty": (
                            mean(observed["difficulties"])
                            if observed["difficulties"]
                            else None
                        ),
                        "observed_families": len(
                            observed["observed_families"]
                        ),
                        "correct_response_families": len(
                            observed["correct_response_families"]
                        ),
                        "successful_retrieval_families": len(
                            observed["successful_retrieval_families"]
                        ),
                    },
                    "current_projection": {
                        "mastery_probability": (
                            projected.mastery_probability
                        ),
                        "estimated_mastery_probability": (
                            projected.estimated_mastery_probability
                        ),
                        "mastery_probability_error_bound": (
                            projected.mastery_probability_error_bound
                        ),
                        "expected_competence": (
                            projected.expected_competence
                        ),
                        "uncertainty": projected.variance**0.5,
                        "evidence_mass": projected.evidence_mass,
                        "acquisition_mass": projected.acquisition_mass,
                        "inference_model_version": projected.model_version,
                        "posterior_representation": (
                            "exact_grid"
                            if projected.posterior is not None
                            else "gaussian_moments"
                        ),
                        "posterior_digest": (
                            projected.posterior.digest
                            if projected.posterior is not None
                            else None
                        ),
                        "independent_families": family_count,
                        "successful_retrieval_families": family_count,
                        "successful_retrieval_diversity": (
                            _family_diversity_label(family_count)
                        ),
                        "evidence_diversity": _family_diversity_label(
                            family_count
                        ),
                        "next_review_at": (
                            projected.next_review_at.isoformat()
                            if projected.next_review_at
                            else None
                        ),
                    },
                    "attention_reasons": attention_reasons,
                }
            )
        diagnostic_findings = [
            row for row in concept_performance if row["attention_reasons"]
        ]
        diagnostic_findings.sort(
            key=lambda row: (
                -len(row["attention_reasons"]),
                -row["session"]["selected_incorrect"],
                row["current_projection"]["effective_readiness"],
                row["name"],
            )
        )
        action_type_counts = Counter(row["action_type"] for row in action_rows)
        action_stage_counts = Counter(row["stage"] for row in action_rows)
        phase_rank = {"unassisted": 0, "assisted": 1, "post_feedback": 2}
        phase_for_rank = {rank: phase for phase, rank in phase_rank.items()}
        current_rank_by_decision: dict[str, int] = {}
        effective_stage_by_action: dict[str, str] = {}
        phase_corrections = 0
        for row in action_rows:
            current_rank = current_rank_by_decision.get(row["decision_id"], 0)
            effective_rank = max(current_rank, phase_rank[row["stage"]])
            effective_stage = phase_for_rank[effective_rank]
            effective_stage_by_action[row["id"]] = effective_stage
            phase_corrections += int(effective_stage != row["stage"])
            if row["action_type"] == ActionKind.HINT_REQUESTED.value:
                effective_rank = max(effective_rank, 1)
            elif row["action_type"] == ActionKind.FEEDBACK_SHOWN.value:
                effective_rank = 2
            current_rank_by_decision[row["decision_id"]] = effective_rank
        effective_stage_counts = Counter(effective_stage_by_action.values())
        check_progression = []
        for row in action_rows:
            if row["action_type"] != ActionKind.CHECK_RUN.value:
                continue
            payload = json.loads(row["payload_json"])
            attempted = payload["passed"] + payload["failed"] + payload["errored"]
            check_progression.append(
                {
                    "decision_id": row["decision_id"],
                    "sequence": row["sequence"],
                    "check_set_id": payload["check_set_id"],
                    "passed": payload["passed"],
                    "failed": payload["failed"],
                    "errored": payload["errored"],
                    "skipped": payload["skipped"],
                    "pass_rate": (
                        payload["passed"] / attempted if attempted else None
                    ),
                    "stage": effective_stage_by_action[row["id"]],
                }
            )
        report = {
            "session_id": session_id,
            "learner_id": session["learner_id"],
            "status": session["status"],
            "mode": session["mode"],
            "topic": (
                {"id": topic["id"], "name": topic["name"]} if topic else None
            ),
            "root_concept_id": session["root_concept_id"],
            "corpus_release_id": session["corpus_release_id"],
            "selected_response_inference": (
                self._selected_response_inference_contract(
                    session["corpus_release_id"]
                )
            ),
            "questions_presented": len(rows),
            "questions_answered": len(answered),
            "correct": correct_count,
            "accuracy": correct_count / len(answered) if answered else None,
            "abstained": abstained,
            "selected_answers": selected_answers,
            "selected_incorrect": selected_incorrect,
            "selected_accuracy": (
                correct_count / selected_answers
                if selected_answers
                else None
            ),
            "response_count_definitions": {
                "questions_answered": (
                    "Submitted responses, including explicit abstentions."
                ),
                "accuracy": (
                    "Compatibility metric: correct responses divided by all "
                    "submitted responses, including abstentions."
                ),
                "incorrect": (
                    "Compatibility field in concept and objective session "
                    "summaries: submitted responses minus correct responses, "
                    "including abstentions."
                ),
                "selected_answers": (
                    "Submitted responses with an option selected."
                ),
                "selected_incorrect": (
                    "Selected answers that were not correct; abstentions are "
                    "excluded."
                ),
                "selected_accuracy": (
                    "Correct selected answers divided by selected answers; "
                    "null when no option was selected."
                ),
            },
            "verification_evidence_definitions": {
                "verification_failures": (
                    "Independent verification responses that credibly support "
                    "failure localization under their immutable response-model "
                    "contract."
                ),
                "verification_inconclusive": (
                    "Independent verification responses that neither certify "
                    "retrieval nor credibly localize failure, including "
                    "abstentions and low-credibility responses."
                ),
            },
            "unique_families": len({row["family_id"] for row in answered}),
            "unique_concepts": len(
                {row["primary_concept_id"] for row in answered}
            ),
            "unique_objectives": len(seen_objective_ids),
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
                "submitted_values": len(response_times),
                "missing_values": len(answered) - len(response_times),
                "selection_window_inconsistencies": timing_inconsistencies,
                "active_exceeds_session_wall": (
                    sum(response_times) / 1000.0 > wall_seconds
                ),
                "evidence_contract": (
                    "Submitted active-time values are checked against the "
                    "authoritative selection-event window only for learner-model "
                    "versions that define that contract; legacy telemetry is not "
                    "retrospectively judged."
                ),
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
            "cross_topic_definition": (
                "Questions mapped to more than one curriculum topic; this is "
                "different from an adaptive excursion outside the requested topic."
            ),
            "outside_requested_topic_questions": sum(
                bool(requested_topic_ids)
                and row["primary_topic_id"] not in requested_topic_ids
                for row in answered
            ),
            "requested_topic_scope_definition": (
                "The requested curriculum topic and all of its descendants; "
                "a question is outside only when its primary topic is outside "
                "that hierarchy."
            ),
            "topic_distribution": [dict(row) for row in topic_counts],
            "evidence_delta": total_evidence_delta,
            "concept_changes": concept_changes,
            "objective_changes": objective_changes,
            "objective_performance": objective_performance,
            "objective_evidence_scope": (
                "Selected-response evidence is attributed only to each "
                "release-pinned objective; it does not establish productive "
                "implementation, explanation, design, or project skill."
            ),
            "family_evidence_definitions": {
                "observed_families": (
                    "Distinct independence families attempted in this session, "
                    "regardless of outcome or credibility."
                ),
                "correct_response_families": (
                    "Distinct families answered correctly in this session, "
                    "including assisted or noncredible responses."
                ),
                "successful_retrieval_families": (
                    "Distinct families whose immutable answer outcome records a "
                    "learner-model-verified retrieval under that answer's model "
                    "version; lifetime projection counts use the same internal "
                    "selected-response evidence."
                ),
            },
            "projection_definitions": {
                "concept_posterior": (
                    "A directly maintained broad-concept projection for legacy "
                    "concept-scored evidence."
                ),
                "derived_objective_readiness_floor": (
                    "A non-persisted routing floor derived from the weakest named "
                    "objective owned by the concept; it is not a second posterior "
                    "or independent evidence."
                ),
                "mastery_probability": (
                    "The conservative probability used for decisions; exact-grid "
                    "estimates subtract their numerical error bound."
                ),
                "mastery_probability_error_bound": (
                    "A numerical approximation guard only; it does not cover "
                    "item calibration, model validity, family dependence, or "
                    "productive-skill transfer."
                ),
            },
            "boundary_algorithm_version": BOUNDARY_ALGORITHM_VERSION,
            "adaptive_routing": adaptive_routing,
            "adaptive_path": adaptive_path,
            "concept_performance": concept_performance,
            "diagnostic_findings": diagnostic_findings,
            "behavior_trace": {
                "observational_only": True,
                "actions": len(action_rows),
                "decisions_observed": len(
                    {row["decision_id"] for row in action_rows}
                ),
                "by_type": dict(action_type_counts),
                "by_stage": dict(action_stage_counts),
                "effective_by_stage": dict(effective_stage_counts),
                "phase_corrections": phase_corrections,
                "hint_requests": action_type_counts[
                    ActionKind.HINT_REQUESTED.value
                ],
                "submitted_hint_count": sum(
                    int(row["hint_count"]) for row in answered
                ),
                "hinted_responses": sum(
                    int(row["hint_count"]) > 0 for row in answered
                ),
                "answer_revisions": action_type_counts[
                    ActionKind.ANSWER_REVISED.value
                ],
                "tool_uses": action_type_counts[ActionKind.TOOL_USED.value],
                "check_runs": action_type_counts[ActionKind.CHECK_RUN.value],
                "artifact_references": sum(
                    row["artifact_id"] is not None for row in action_rows
                ),
                "check_progression": check_progression,
                "evidence_contract": (
                    "Telemetry describes behavior but does not update competence "
                    "without a release-pinned rubric and independently valid evaluation."
                ),
            },
            "response_position_shadow": response_position_shadow,
            "productive_skill_shadow": productive_shadow_summary(
                self.database,
                session["learner_id"],
                session_id=session_id,
            ),
            "diagnostic_contract": (
                "Findings expose observed evidence and posterior hypotheses; "
                "they are not causal or clinical diagnoses."
            ),
        }
        with self.database.read() as connection:
            self.database.require_learner_evidence_safe(
                session["learner_id"],
                connection,
            )
        return report

    def trace(self, session_id: str) -> list[dict[str, Any]]:
        self.database.get_session(session_id)
        return self.database.recent_decisions(session_id, limit=200)

    @staticmethod
    def _outcome_payload(result: SubmissionResult) -> dict[str, Any]:
        payload = {
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
        if result.focus_objective_id is not None or any(
            "objective_id" in change for change in result.state_changes
        ):
            payload["focus_objective_id"] = result.focus_objective_id
        return payload

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
            focus_objective_id=outcome.get("focus_objective_id"),
            state_changes=tuple(outcome["state_changes"]),
            transition_reason=outcome.get(
                "transition_reason", "legacy_transition"
            ),
            boundary_decision=outcome.get("boundary_decision"),
            idempotent_replay=idempotent,
        )
