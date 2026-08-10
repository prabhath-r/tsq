# SPDX-License-Identifier: MPL-2.0

"""Conservative learner-facing summaries of productive-task observations.

The performance ledger intentionally remains shadow-only.  This module makes
its timing, semantic actions, and rubric observations visible without joining
them to selected-response mastery, certification, or routing projections.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Iterable

from .errors import ValidationError
from .evidence import (
    ActionKind,
    LearningAction,
    LearningTask,
    TaskEvaluation,
    action_trace_digest,
    canonical_digest,
    canonical_json,
    reduce_evidence,
)
from .performance import NormalizedScoringResult, ScoringProtocolError
from .performance_ledger import require_performance_projection_consistency
from .store import Database


PRODUCTIVE_SHADOW_REPORT_VERSION = 1
RECENT_ATTEMPT_LIMIT = 20


def _decode_canonical_object(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is not valid JSON: {exc}") from exc
    if type(value) is not dict:
        raise ValidationError(f"{label} must be a JSON object.")
    try:
        encoded = canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} is not canonical JSON: {exc}") from exc
    if encoded != raw:
        raise ValidationError(
            f"{label} is not in the canonical stored representation."
        )
    return value


def _decode_task(
    raw: str,
    task_id: str,
    task_version: int,
    *committed_digests: object,
) -> LearningTask:
    try:
        task = LearningTask.from_terms(
            _decode_canonical_object(
                raw, f"Performance task {task_id} v{task_version}"
            )
        )
    except ValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Performance task {task_id} v{task_version} cannot be reported "
            f"safely: {exc}"
        ) from exc
    if (task.id, task.version) != (task_id, task_version):
        raise ValidationError(
            f"Performance task {task_id} v{task_version} has a mismatched "
            "stored identity; reporting fails closed."
        )
    if any(digest != task.digest for digest in committed_digests):
        raise ValidationError(
            f"Performance task {task_id} v{task_version} does not match every "
            "stored task, release, and attempt digest commitment; reporting "
            "fails closed."
        )
    return task


def _decode_evaluation(raw: str, evaluation_id: str) -> TaskEvaluation:
    try:
        evaluation = TaskEvaluation.from_terms(
            _decode_canonical_object(
                raw, f"Task evaluation {evaluation_id}"
            )
        )
    except ValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"Task evaluation {evaluation_id} cannot be reported safely: {exc}"
        ) from exc
    if evaluation.id != evaluation_id:
        raise ValidationError(
            f"Task evaluation {evaluation_id} has a mismatched stored identity; "
            "reporting fails closed."
        )
    return evaluation


def _decode_action(row: Any) -> LearningAction:
    label = f"Performance action {row['id']}"
    try:
        action = LearningAction.from_terms(
            {
                "id": row["id"],
                "trace_id": row["attempt_id"],
                "sequence": row["sequence"],
                "kind": row["action_type"],
                "phase": row["phase"],
                "payload": _decode_canonical_object(
                    row["payload_json"], f"{label} payload"
                ),
                "elapsed_ms": row["elapsed_ms"],
                "schema_version": 1,
            }
        )
    except ValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{label} cannot be reported safely: {exc}"
        ) from exc
    return action


def _timestamp(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise ValidationError(f"{label} must be an ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} is not a valid timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{label} must include a timezone offset.")
    return parsed.astimezone(timezone.utc)


def _validate_projection_event(
    event: Any | None,
    *,
    event_id: str,
    event_type: str,
    learner_id: str,
    session_id: str,
    occurred_at: str,
    payload: dict[str, Any],
) -> None:
    label = f"{event_type} event {event_id}"
    if event is None:
        raise ValidationError(
            f"{label} is missing; reporting fails closed."
        )
    if (
        event["event_type"] != event_type
        or event["schema_version"] != 1
        or event["learner_id"] != learner_id
        or event["session_id"] != session_id
        or event["occurred_at"] != occurred_at
    ):
        raise ValidationError(
            f"{label} does not match its projection envelope; reporting "
            "fails closed."
        )
    stored_payload = _decode_canonical_object(
        event["payload_json"], f"{label} payload"
    )
    stored_metadata = _decode_canonical_object(
        event["metadata_json"], f"{label} metadata"
    )
    if stored_payload != payload:
        raise ValidationError(
            f"{label} payload does not match its relational projection; "
            "reporting fails closed."
        )
    envelope = {
        "event_id": event["event_id"],
        "stream_id": event["stream_id"],
        "stream_version": event["stream_version"],
        "event_type": event["event_type"],
        "schema_version": event["schema_version"],
        "occurred_at": event["occurred_at"],
        "recorded_at": event["recorded_at"],
        "learner_id": event["learner_id"],
        "session_id": event["session_id"],
        "correlation_id": event["correlation_id"],
        "causation_id": event["causation_id"],
        "idempotency_key": event["idempotency_key"],
        "payload": stored_payload,
        "metadata": stored_metadata,
        "previous_hash": event["previous_hash"],
    }
    expected_hash = hashlib.sha256(
        canonical_json(envelope).encode("utf-8")
    ).hexdigest()
    if event["payload_hash"] != expected_hash:
        raise ValidationError(
            f"{label} hash does not match its immutable envelope; reporting "
            "fails closed."
        )


def _decode_authority(
    raw: str,
    evaluation: TaskEvaluation,
    task: LearningTask,
) -> dict[str, Any]:
    authority = _decode_canonical_object(
        raw, f"Task evaluation {evaluation.id} authority"
    )
    if set(authority) != {
        "normalized_result",
        "normalized_result_digest",
    }:
        raise ValidationError(
            f"Task evaluation {evaluation.id} authority has incompatible "
            "fields; reporting fails closed."
        )
    normalized = authority["normalized_result"]
    if type(normalized) is not dict:
        raise ValidationError(
            f"Task evaluation {evaluation.id} normalized authority must be "
            "an object; reporting fails closed."
        )
    if authority["normalized_result_digest"] != canonical_digest(normalized):
        raise ValidationError(
            f"Task evaluation {evaluation.id} normalized authority digest "
            "does not match; reporting fails closed."
        )
    if normalized.get("evaluation") != evaluation.terms():
        raise ValidationError(
            f"Task evaluation {evaluation.id} authority does not commit to "
            "the reported evaluation; reporting fails closed."
        )
    try:
        normalized_result = NormalizedScoringResult.from_terms(normalized)
    except (TypeError, ValueError, ScoringProtocolError) as exc:
        raise ValidationError(
            f"Task evaluation {evaluation.id} normalized authority is invalid; "
            f"reporting fails closed: {exc}"
        ) from exc
    contract = normalized_result.request.scorer_contract
    if contract is not None and all(
        canonical_json(contract.terms())
        != canonical_json(candidate.terms())
        for candidate in task.scorer_contracts
    ):
        raise ValidationError(
            f"Task evaluation {evaluation.id} scorer contract is not present "
            "in its released task; reporting fails closed."
        )
    return authority


def _empty_summary(*, scope: str) -> dict[str, Any]:
    return {
        "report_version": PRODUCTIVE_SHADOW_REPORT_VERSION,
        "scope": scope,
        "observational_only": True,
        "shadow_only": True,
        "attempt_count": 0,
        "distinct_task_count": 0,
        "observed_task_families": 0,
        "attempt_statuses": {},
        "modalities": {},
        "observed_elapsed_seconds": 0.0,
        "behavior": {
            "actions": 0,
            "by_type": {},
            "by_phase": {},
            "hint_requests": 0,
            "answer_revisions": 0,
            "artifact_checkpoints": 0,
            "explanation_checkpoints": 0,
            "check_runs": 0,
            "tool_uses": 0,
        },
        "rubric_observations": {
            "evaluations": 0,
            "evaluated_attempts": 0,
            "criteria_observed": 0,
            "by_status": {},
            "by_scorer_kind": {},
            "valid_score_average": None,
            "misconception_signals": {},
            "by_objective": {},
        },
        "recent_attempts": [],
        "recent_attempt_limit": RECENT_ATTEMPT_LIMIT,
        "recent_attempts_truncated": False,
        "scope_binding": {
            "concept_ids": [],
            "objective_ids": [],
            "objective_bindings": [],
            "objective_binding_available": False,
            "contract": (
                "Objective IDs are reported only from explicit release-pinned "
                "rubric objective weights. Concept mappings are never used to "
                "infer objective attribution. Topic-scoped profiles match task "
                "concept mappings against that topic's concept learning scope, "
                "including its prerequisites."
            ),
        },
        "evidence_boundary": {
            "mastery_claim": False,
            "certification_claim": False,
            "selected_response_projection_affected": False,
            "objective_projection_affected": False,
            "declared_objective_bindings_applied": False,
            "adaptive_routing_affected": False,
            "stored_projection_applications": 0,
            "stored_certification_applications": 0,
            "interpretation": (
                "These are diagnostic observations from released productive "
                "tasks. They do not update selected-response mastery, establish "
                "productive skill, certify competence, or affect routing."
            ),
        },
        "timing_contract": (
            "Observed elapsed time is the sum of the last server-derived action "
            "timestamp in each included attempt. It excludes idle time after the "
            "last recorded action and is not an estimate of focused work."
        ),
    }


def productive_shadow_summary(
    database: Database,
    learner_id: str,
    *,
    session_id: str | None = None,
    concept_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a bounded shadow summary for a learner or one of their sessions.

    ``concept_ids`` filters tasks by any rubric concept mapping.  It is used by
    topic-scoped learner profiles so unrelated task observations do not leak
    into the requested scope.
    """

    requested_concepts = (
        None if concept_ids is None else frozenset(concept_ids)
    )
    scope_label = "session" if session_id is not None else "learner"
    summary = _empty_summary(scope=scope_label)
    summary["scope_binding"]["concept_ids"] = (
        [] if requested_concepts is None else sorted(requested_concepts)
    )

    attempt_parameters: list[object] = [learner_id]
    attempt_filter = "attempt.learner_id=?"
    if session_id is not None:
        attempt_filter += " AND attempt.session_id=?"
        attempt_parameters.append(session_id)

    with database.read() as connection:
        database.require_learner_evidence_integrity(
            learner_id,
            connection,
        )
        attempt_rows = connection.execute(
            f"""SELECT attempt.*, task.definition_json,
                       task.task_digest AS definition_task_digest,
                       membership.task_digest AS release_task_digest,
                       task_release.corpus_release_id
                           AS release_corpus_release_id
                FROM performance_attempts attempt
                JOIN performance_tasks task
                  ON task.task_id=attempt.task_id
                 AND task.task_version=attempt.task_version
                LEFT JOIN release_performance_tasks membership
                  ON membership.release_id=attempt.task_release_id
                 AND membership.task_id=attempt.task_id
                 AND membership.task_version=attempt.task_version
                LEFT JOIN performance_task_releases task_release
                  ON task_release.id=attempt.task_release_id
                WHERE {attempt_filter}
                ORDER BY attempt.started_at DESC, attempt.id DESC""",
            tuple(attempt_parameters),
        ).fetchall()
        action_rows = connection.execute(
            f"""SELECT action.*
                FROM performance_actions action
                JOIN performance_attempts attempt
                  ON attempt.id=action.attempt_id
                WHERE {attempt_filter}
                ORDER BY action.attempt_id, action.sequence""",
            tuple(attempt_parameters),
        ).fetchall()
        evaluation_rows = connection.execute(
            f"""SELECT evaluation.*, bundle.id AS bundle_id,
                       bundle.event_id AS bundle_event_id,
                       bundle.attempt_id AS bundle_attempt_id,
                       bundle.bundle_digest, bundle.bundle_json,
                       bundle.projection_applied,
                       bundle.certification_applied,
                       bundle.recorded_at AS bundle_recorded_at
                FROM task_evaluations evaluation
                JOIN performance_attempts attempt
                  ON attempt.id=evaluation.attempt_id
                LEFT JOIN shadow_evidence_bundles bundle
                  ON bundle.evaluation_id=evaluation.id
                WHERE {attempt_filter}
                ORDER BY evaluation.recorded_at, evaluation.id""",
            tuple(attempt_parameters),
        ).fetchall()
        bundle_rows = connection.execute(
            f"""SELECT bundle.id, bundle.evaluation_id, bundle.attempt_id,
                       evaluation.id AS matched_evaluation_id,
                       evaluation.attempt_id AS evaluation_attempt_id
                FROM shadow_evidence_bundles bundle
                JOIN performance_attempts attempt
                  ON attempt.id=bundle.attempt_id
                LEFT JOIN task_evaluations evaluation
                  ON evaluation.id=bundle.evaluation_id
                WHERE {attempt_filter}
                ORDER BY bundle.id""",
            tuple(attempt_parameters),
        ).fetchall()
        event_filter = "learner_id=?"
        event_parameters: list[object] = [learner_id]
        if session_id is not None:
            event_filter += " AND session_id=?"
            event_parameters.append(session_id)
        event_rows = connection.execute(
            f"""SELECT * FROM events
                WHERE {event_filter}
                  AND event_type IN (
                      'PerformanceTaskStarted',
                      'PerformanceActionRecorded',
                      'TaskEvaluationRecorded',
                      'ShadowEvidenceReduced'
                  )""",
            tuple(event_parameters),
        ).fetchall()
    events_by_id = {row["event_id"]: row for row in event_rows}

    tasks_by_attempt: dict[str, LearningTask] = {}
    included_attempt_rows = []
    for row in attempt_rows:
        task = _decode_task(
            row["definition_json"],
            row["task_id"],
            int(row["task_version"]),
            row["definition_task_digest"],
            row["release_task_digest"],
            row["task_digest"],
        )
        if row["release_corpus_release_id"] != row["corpus_release_id"]:
            raise ValidationError(
                f"Performance attempt {row['id']} crosses its task-release "
                "corpus boundary; reporting fails closed."
            )
        attempt_event = events_by_id.get(row["event_id"])
        if (
            attempt_event is not None
            and attempt_event["recorded_at"] != row["recorded_at"]
        ):
            raise ValidationError(
                f"Performance attempt {row['id']} recording time does not "
                "match its event; reporting fails closed."
            )
        _validate_projection_event(
            attempt_event,
            event_id=row["event_id"],
            event_type="PerformanceTaskStarted",
            learner_id=row["learner_id"],
            session_id=row["session_id"],
            occurred_at=row["started_at"],
            payload={
                "attempt_id": row["id"],
                "session_id": row["session_id"],
                "learner_id": row["learner_id"],
                "task_release_id": row["task_release_id"],
                "corpus_release_id": row["corpus_release_id"],
                "task_id": row["task_id"],
                "task_version": row["task_version"],
                "task_digest": row["task_digest"],
                "session_revision": row["session_revision"],
                "learner_revision": row["learner_revision"],
            },
        )
        task_concepts = {
            concept_id
            for criterion in task.criteria
            for concept_id, _ in criterion.concept_weights
        }
        if (
            requested_concepts is not None
            and not task_concepts.intersection(requested_concepts)
        ):
            continue
        tasks_by_attempt[row["id"]] = task
        included_attempt_rows.append(row)

    if not included_attempt_rows:
        with database.read() as connection:
            database.require_learner_evidence_integrity(
                learner_id,
                connection,
            )
            require_performance_projection_consistency(
                connection,
                learner_id=learner_id,
            )
        return summary

    included_attempt_ids = set(tasks_by_attempt)
    for row in bundle_rows:
        if row["attempt_id"] not in included_attempt_ids:
            continue
        if (
            row["matched_evaluation_id"] is None
            or row["evaluation_attempt_id"] != row["attempt_id"]
        ):
            raise ValidationError(
                f"Shadow evidence bundle {row['id']} is orphaned or crosses "
                "an attempt boundary; reporting fails closed."
            )

    actions_by_attempt: dict[str, list[LearningAction]] = defaultdict(list)
    action_rows_by_id: dict[str, Any] = {}
    attempt_rows_by_id = {row["id"]: row for row in included_attempt_rows}
    action_types: Counter[str] = Counter()
    action_phases: Counter[str] = Counter()
    for row in action_rows:
        if row["attempt_id"] not in included_attempt_ids:
            continue
        action = _decode_action(row)
        task = tasks_by_attempt[row["attempt_id"]]
        if action.kind not in task.allowed_action_kinds:
            raise ValidationError(
                f"Performance action {action.id} is not allowed by task "
                f"{task.id}; reporting fails closed."
            )
        attempt_row = attempt_rows_by_id[row["attempt_id"]]
        action_event = events_by_id.get(row["event_id"])
        if (
            action_event is not None
            and action_event["recorded_at"] != row["recorded_at"]
        ):
            raise ValidationError(
                f"Performance action {action.id} recording time does not "
                "match its event; reporting fails closed."
            )
        _validate_projection_event(
            action_event,
            event_id=row["event_id"],
            event_type="PerformanceActionRecorded",
            learner_id=attempt_row["learner_id"],
            session_id=attempt_row["session_id"],
            occurred_at=row["occurred_at"],
            payload={
                "attempt_id": row["attempt_id"],
                "action": action.terms(),
            },
        )
        actions_by_attempt[row["attempt_id"]].append(action)
        action_rows_by_id[action.id] = row
        action_types[action.kind.value] += 1
        action_phases[action.phase.value] += 1

    for attempt_id in tasks_by_attempt:
        actions = actions_by_attempt[attempt_id]
        if not actions or actions[0].kind is not ActionKind.STARTED:
            raise ValidationError(
                f"Performance attempt {attempt_id} lacks its initial started "
                "action; reporting fails closed."
            )
        if [action.sequence for action in actions] != list(
            range(len(actions))
        ):
            raise ValidationError(
                f"Performance attempt {attempt_id} has a noncanonical action "
                "sequence; reporting fails closed."
            )
        try:
            action_trace_digest(actions)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Performance attempt {attempt_id} has an invalid action "
                f"trace: {exc}"
            ) from exc
        started_at = _timestamp(
            attempt_rows_by_id[attempt_id]["started_at"],
            f"Performance attempt {attempt_id} start",
        )
        for action in actions:
            occurred_at = _timestamp(
                action_rows_by_id[action.id]["occurred_at"],
                f"Performance action {action.id} occurrence",
            )
            expected_elapsed_ms = int(
                (occurred_at - started_at).total_seconds() * 1000
            )
            if (
                expected_elapsed_ms < 0
                or action.elapsed_ms != expected_elapsed_ms
            ):
                raise ValidationError(
                    f"Performance action {action.id} does not match its "
                    "server-derived elapsed time; reporting fails closed."
                )

    evaluations_by_attempt: dict[str, list[TaskEvaluation]] = defaultdict(list)
    criterion_statuses: Counter[str] = Counter()
    scorer_kinds: Counter[str] = Counter()
    misconception_signals: Counter[str] = Counter()
    valid_scores: list[float] = []
    objective_observations: dict[str, dict[str, Any]] = {}
    projection_applications = 0
    certification_applications = 0
    for row in evaluation_rows:
        if row["attempt_id"] not in included_attempt_ids:
            continue
        if (
            row["bundle_id"] is None
            or row["bundle_attempt_id"] != row["attempt_id"]
        ):
            raise ValidationError(
                f"Task evaluation {row['id']} lacks an attempt-matched shadow "
                "bundle; reporting fails closed."
            )
        evaluation = _decode_evaluation(
            row["evaluation_json"], row["id"]
        )
        task = tasks_by_attempt[row["attempt_id"]]
        if (
            row["evaluation_digest"] != evaluation.digest
            or evaluation.trace_id != row["attempt_id"]
            or evaluation.task_id != task.id
            or evaluation.task_version != task.version
            or evaluation.task_digest != task.digest
        ):
            raise ValidationError(
                f"Task evaluation {row['id']} does not match its attempt and "
                "task commitments; reporting fails closed."
            )
        actions = actions_by_attempt[row["attempt_id"]]
        through_sequence = row["through_sequence"]
        if (
            type(through_sequence) is not int
            or through_sequence < 0
        ):
            raise ValidationError(
                f"Task evaluation {row['id']} has an invalid action boundary; "
                "reporting fails closed."
            )
        boundary_actions = tuple(
            action
            for action in actions
            if action.sequence <= through_sequence
        )
        if (
            not boundary_actions
            or boundary_actions[-1].sequence != through_sequence
            or boundary_actions[-1].kind is not ActionKind.SUBMITTED
        ):
            raise ValidationError(
                f"Task evaluation {row['id']} does not terminate at its "
                "submitted action boundary; reporting fails closed."
            )
        try:
            committed_trace_digest = action_trace_digest(boundary_actions)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Task evaluation {row['id']} has an invalid committed action "
                f"trace: {exc}"
            ) from exc
        if evaluation.action_trace_digest != committed_trace_digest:
            raise ValidationError(
                f"Task evaluation {row['id']} action-trace digest does not "
                "match its stored boundary; reporting fails closed."
            )
        try:
            expected_bundle = reduce_evidence(
                task, evaluation, boundary_actions
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Task evaluation {row['id']} cannot be reduced safely: {exc}"
            ) from exc
        stored_bundle = _decode_canonical_object(
            row["bundle_json"],
            f"Shadow evidence bundle {row['bundle_id']}",
        )
        if (
            stored_bundle != expected_bundle.terms()
            or row["bundle_digest"] != expected_bundle.digest
        ):
            raise ValidationError(
                f"Shadow evidence bundle {row['bundle_id']} does not match "
                "its deterministic reduction; reporting fails closed."
            )
        authority = _decode_authority(
            row["authority_json"], evaluation, task
        )
        attempt_row = attempt_rows_by_id[row["attempt_id"]]
        evaluation_event = events_by_id.get(row["event_id"])
        if (
            evaluation_event is not None
            and evaluation_event["recorded_at"] != row["recorded_at"]
        ):
            raise ValidationError(
                f"Task evaluation {row['id']} recording time does not match "
                "its event; reporting fails closed."
            )
        _validate_projection_event(
            evaluation_event,
            event_id=row["event_id"],
            event_type="TaskEvaluationRecorded",
            learner_id=attempt_row["learner_id"],
            session_id=attempt_row["session_id"],
            occurred_at=(
                evaluation_event["occurred_at"]
                if evaluation_event is not None
                else ""
            ),
            payload={
                "attempt_id": row["attempt_id"],
                "through_sequence": through_sequence,
                "evaluation_digest": evaluation.digest,
                "evaluation": evaluation.terms(),
                "authority": authority,
            },
        )
        bundle_event = events_by_id.get(row["bundle_event_id"])
        if (
            bundle_event is not None
            and bundle_event["recorded_at"] != row["bundle_recorded_at"]
        ):
            raise ValidationError(
                f"Shadow evidence bundle {row['bundle_id']} recording time "
                "does not match its event; reporting fails closed."
            )
        _validate_projection_event(
            bundle_event,
            event_id=row["bundle_event_id"],
            event_type="ShadowEvidenceReduced",
            learner_id=attempt_row["learner_id"],
            session_id=attempt_row["session_id"],
            occurred_at=(
                bundle_event["occurred_at"]
                if bundle_event is not None
                else ""
            ),
            payload={
                "bundle_id": row["bundle_id"],
                "evaluation_id": evaluation.id,
                "attempt_id": row["attempt_id"],
                "bundle_digest": expected_bundle.digest,
                "bundle": expected_bundle.terms(),
                "projection_applied": False,
                "certification_applied": False,
            },
        )
        evaluations_by_attempt[row["attempt_id"]].append(evaluation)
        if type(row["projection_applied"]) is not int or type(
            row["certification_applied"]
        ) is not int:
            raise ValidationError(
                f"Shadow evidence bundle {row['bundle_id']} has invalid "
                "application flags; reporting fails closed."
            )
        projection_applications += row["projection_applied"]
        certification_applications += row["certification_applied"]
        criteria_by_id = {
            criterion.id: criterion for criterion in task.criteria
        }
        for criterion in evaluation.criteria:
            criterion_statuses[criterion.status.value] += 1
            scorer_kinds[criterion.scorer_kind.value] += 1
            misconception_signals.update(criterion.misconception_ids)
            if criterion.score is not None:
                valid_scores.append(criterion.score)
            for objective_id, _ in criteria_by_id[
                criterion.criterion_id
            ].objective_weights:
                objective_summary = objective_observations.setdefault(
                    objective_id,
                    {
                        "criteria_observed": 0,
                        "by_status": Counter(),
                        "valid_scores": [],
                    },
                )
                objective_summary["criteria_observed"] += 1
                objective_summary["by_status"][
                    criterion.status.value
                ] += 1
                if criterion.score is not None:
                    objective_summary["valid_scores"].append(
                        criterion.score
                    )

    if projection_applications or certification_applications:
        raise ValidationError(
            "Productive-task shadow evidence was marked as applied; reporting "
            "fails closed because this schema cannot update mastery or certification."
        )

    attempt_statuses: Counter[str] = Counter()
    modalities: Counter[str] = Counter()
    elapsed_ms_total = 0
    details: list[dict[str, Any]] = []
    for row in included_attempt_rows:
        attempt_id = row["id"]
        task = tasks_by_attempt[attempt_id]
        actions = actions_by_attempt[attempt_id]
        evaluations = evaluations_by_attempt[attempt_id]
        terminal_types = {
            action.kind.value
            for action in actions
            if action.kind in {ActionKind.SUBMITTED, ActionKind.ABANDONED}
        }
        if "submitted" in terminal_types:
            status = "submitted"
        elif "abandoned" in terminal_types:
            status = "abandoned"
        else:
            status = "active"
        attempt_statuses[status] += 1
        modalities[task.modality.value] += 1
        elapsed_ms = max(
            (
                int(action.elapsed_ms)
                for action in actions
                if action.elapsed_ms is not None
            ),
            default=0,
        )
        elapsed_ms_total += elapsed_ms
        attempt_status_counts = Counter(
            criterion.status.value
            for evaluation in evaluations
            for criterion in evaluation.criteria
        )
        attempt_scores = [
            criterion.score
            for evaluation in evaluations
            for criterion in evaluation.criteria
            if criterion.score is not None
        ]
        attempt_misconceptions = Counter(
            misconception_id
            for evaluation in evaluations
            for criterion in evaluation.criteria
            for misconception_id in criterion.misconception_ids
        )
        task_concepts = sorted(
            {
                concept_id
                for criterion in task.criteria
                for concept_id, _ in criterion.concept_weights
            }
        )
        task_objectives = sorted(
            {
                objective_id
                for criterion in task.criteria
                for objective_id, _ in criterion.objective_weights
            }
        )
        rubric_objective_bindings = [
            {
                "criterion_id": criterion.id,
                "objective_weights": [
                    {
                        "objective_id": objective_id,
                        "weight": weight,
                    }
                    for objective_id, weight in criterion.objective_weights
                ],
            }
            for criterion in task.criteria
            if criterion.objective_weights
        ]
        details.append(
            {
                "attempt_id": attempt_id,
                "session_id": row["session_id"],
                "started_at": row["started_at"],
                "status": status,
                "task": {
                    "id": task.id,
                    "version": task.version,
                    "title": task.title,
                    "modality": task.modality.value,
                    "family_id": task.family_id,
                    "concept_ids": task_concepts,
                    "objective_ids": task_objectives,
                    "rubric_objective_bindings": rubric_objective_bindings,
                    "objective_binding_available": bool(task_objectives),
                    "criterion_count": len(task.criteria),
                },
                "observed_elapsed_seconds": elapsed_ms / 1000.0,
                "actions": len(actions),
                "action_types": dict(
                    sorted(
                        Counter(
                            action.kind.value for action in actions
                        ).items()
                    )
                ),
                "evaluations": len(evaluations),
                "criteria_observed": sum(attempt_status_counts.values()),
                "criterion_statuses": dict(sorted(attempt_status_counts.items())),
                "valid_score_average": (
                    mean(attempt_scores) if attempt_scores else None
                ),
                "misconception_signals": dict(
                    sorted(attempt_misconceptions.items())
                ),
                "mastery_claim": False,
                "certification_claim": False,
            }
        )

    summary.update(
        {
            "attempt_count": len(included_attempt_rows),
            "distinct_task_count": len(
                {
                    (row["task_id"], int(row["task_version"]))
                    for row in included_attempt_rows
                }
            ),
            "observed_task_families": len(
                {task.family_id for task in tasks_by_attempt.values()}
            ),
            "attempt_statuses": dict(sorted(attempt_statuses.items())),
            "modalities": dict(sorted(modalities.items())),
            "observed_elapsed_seconds": elapsed_ms_total / 1000.0,
            "behavior": {
                "actions": sum(action_types.values()),
                "by_type": dict(sorted(action_types.items())),
                "by_phase": dict(sorted(action_phases.items())),
                "hint_requests": action_types["hint_requested"],
                "answer_revisions": action_types["answer_revised"],
                "artifact_checkpoints": action_types[
                    "artifact_checkpoint"
                ],
                "explanation_checkpoints": action_types[
                    "explanation_checkpoint"
                ],
                "check_runs": action_types["check_run"],
                "tool_uses": action_types["tool_used"],
            },
            "rubric_observations": {
                "evaluations": sum(
                    len(values) for values in evaluations_by_attempt.values()
                ),
                "evaluated_attempts": sum(
                    bool(values) for values in evaluations_by_attempt.values()
                ),
                "criteria_observed": sum(criterion_statuses.values()),
                "by_status": dict(sorted(criterion_statuses.items())),
                "by_scorer_kind": dict(sorted(scorer_kinds.items())),
                "valid_score_average": (
                    mean(valid_scores) if valid_scores else None
                ),
                "misconception_signals": dict(
                    sorted(misconception_signals.items())
                ),
                "by_objective": {
                    objective_id: {
                        "criteria_observed": values[
                            "criteria_observed"
                        ],
                        "by_status": dict(
                            sorted(values["by_status"].items())
                        ),
                        "valid_score_average": (
                            mean(values["valid_scores"])
                            if values["valid_scores"]
                            else None
                        ),
                    }
                    for objective_id, values in sorted(
                        objective_observations.items()
                    )
                },
            },
            "recent_attempts": details[:RECENT_ATTEMPT_LIMIT],
            "recent_attempts_truncated": (
                len(details) > RECENT_ATTEMPT_LIMIT
            ),
        }
    )
    explicit_objective_ids = sorted(
        {
            objective_id
            for task in tasks_by_attempt.values()
            for criterion in task.criteria
            for objective_id, _ in criterion.objective_weights
        }
    )
    summary["scope_binding"]["objective_ids"] = explicit_objective_ids
    summary["scope_binding"]["objective_bindings"] = [
        {
            "task_id": task.id,
            "task_version": task.version,
            "criterion_id": criterion.id,
            "objective_weights": [
                {
                    "objective_id": objective_id,
                    "weight": weight,
                }
                for objective_id, weight in criterion.objective_weights
            ],
        }
        for task in sorted(
            {
                (task.id, task.version): task
                for task in tasks_by_attempt.values()
            }.values(),
            key=lambda item: (item.id, item.version),
        )
        for criterion in task.criteria
        if criterion.objective_weights
    ]
    summary["scope_binding"]["objective_binding_available"] = bool(
        explicit_objective_ids
    )
    with database.read() as connection:
        database.require_learner_evidence_integrity(
            learner_id,
            connection,
        )
        require_performance_projection_consistency(
            connection,
            learner_id=learner_id,
        )
    return summary
