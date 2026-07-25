# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any

from .evidence import (
    ActionKind,
    ActionPhase,
    LearningAction,
    canonical_json,
    summarize_actions,
)
from .event_contracts import (
    PROJECTION_FIELDS_V1 as _PROJECTION_FIELDS_V1,
    PROJECTION_FIELDS_V2 as _PROJECTION_FIELDS_V2,
    PROJECTION_FIELDS_V3 as _PROJECTION_FIELDS_V3,
    PROJECTION_FIELDS_V4 as _PROJECTION_FIELDS_V4,
    PROJECTION_METADATA_FIELDS as _PROJECTION_METADATA_FIELDS,
    PROJECTION_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM as _PROJECTION_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM,
    QUESTION_SELECTED_BASE_FIELDS,
    QUESTION_SELECTED_METADATA_FIELDS,
    QUESTION_SELECTED_OBJECTIVE_FIELDS,
    RESPONSE_FIELDS as _RESPONSE_FIELDS,
    RESPONSE_METADATA_FIELDS as _RESPONSE_METADATA_FIELDS,
    RESPONSE_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM as _RESPONSE_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM,
    same_json_value,
)
from .errors import ConflictError, NotFoundError, TSQError, ValidationError
from .inference import (
    LEGACY_MISCONCEPTION_ALGORITHM,
    MISCONCEPTION_ALGORITHM_METADATA_KEY,
    MISCONCEPTION_ALGORITHM_VERSION,
    response_window,
)
from .learner import (
    LearnerModel,
)
from .models import MAX_REMEDIATION_DEPTH, QuestionStatus, SessionPhase
from .performance_ledger import (
    performance_projection_snapshot,
    rebuild_performance_projections,
)
from .policy_shadow import (
    policy_shadow_projection_snapshot,
    rebuild_policy_shadow_projections,
)
from .store import SCHEMA_VERSION, Database, question_content_hash
from .versions import (
    AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS,
    BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION,
    COMPLETE_TRANSITION_OUTCOME_MODEL_VERSIONS,
    DEFAULT_LEARNER_MODEL_VERSION,
    PROJECTION_HASH_VERSION_BY_EVENT_SCHEMA,
    PROJECTION_MODEL_VERSIONS_BY_EVENT_SCHEMA,
    SUPPORTED_MODEL_VERSIONS,
    question_selected_schema_for,
)


REPLAY_FORMAT_VERSION = 1
RESPONSE_EVENT_SCHEMA_VERSIONS = frozenset({1, 2})
CURRENT_RESPONSE_EVENT_SCHEMA_VERSION = 2
PROJECTION_EVENT_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4})
ACTION_EVENT_SCHEMA_VERSION = 1

_ACTION_FIELDS = frozenset(
    {
        "action_id",
        "decision_id",
        "sequence",
        "stage",
        "action_type",
        "payload",
        "artifact",
    }
)
_ACTION_METADATA_FIELDS = frozenset(
    {"action_schema_version", "observational_only", "corpus_release_id"}
)
_ARTIFACT_FIELDS = frozenset({"sha256", "size_bytes", "media_type"})
_ACTION_ARTIFACT_DIGEST_FIELDS = {
    ActionKind.ANSWER_REVISED: "answer_digest",
    ActionKind.ARTIFACT_CHECKPOINT: "artifact_digest",
    ActionKind.EXPLANATION_CHECKPOINT: "explanation_digest",
    ActionKind.CHECK_RUN: "result_digest",
    ActionKind.SUBMITTED: "submission_digest",
    ActionKind.FEEDBACK_SHOWN: "feedback_digest",
}
_ARTIFACT_PROJECTION_COLUMNS = (
    "id",
    "sha256",
    "size_bytes",
    "media_type",
    "created_at",
)
_ACTION_PROJECTION_COLUMNS = (
    "id",
    "event_id",
    "decision_id",
    "session_id",
    "learner_id",
    "sequence",
    "stage",
    "action_type",
    "payload_json",
    "artifact_id",
    "occurred_at",
    "recorded_at",
    "command_hash",
)

_BOUNDARY_DECISION_FIELDS = frozenset(
    {
        "focus_concept_id",
        "selected_concept_id",
        "algorithm_version",
        "selected",
        "candidates",
    }
)
_BOUNDARY_CANDIDATE_FIELDS = frozenset(
    {
        "concept_id",
        "edge_weight",
        "score",
        "need",
        "uncertainty_value",
        "evidence_gap",
        "recent_failure_rate",
        "prerequisite_support",
        "effective_readiness",
        "recursive_bottleneck_concept_id",
    }
)
_OBJECTIVE_BOUNDARY_DECISION_FIELDS = frozenset(
    {
        "focus_objective_id",
        "selected_objective_id",
        "algorithm_version",
        "selected",
        "candidates",
    }
)
_OBJECTIVE_BOUNDARY_CANDIDATE_FIELDS = frozenset(
    {
        "edge_id",
        "objective_id",
        "concept_id",
        "relation",
        "rationale",
        "edge_weight",
        "score",
        "need",
        "mastery_probability",
        "expected_competence",
        "uncertainty_value",
        "evidence_gap",
        "recent_failure_rate",
    }
)


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value}")
    return parsed


def _strict_object(raw: str, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_reject_duplicate_object,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} contains invalid JSON: {exc}") from exc
    if type(value) is not dict:
        raise ValidationError(f"{label} must be a JSON object.")
    return value


def _strict_array(raw: str, label: str) -> list[Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw,
            parse_constant=reject_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_reject_duplicate_object,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} contains invalid JSON: {exc}") from exc
    if type(value) is not list:
        raise ValidationError(f"{label} must be a JSON array.")
    return value


def _require_exact_fields(value: dict[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise ValidationError(f"{label} has incompatible fields ({'; '.join(details)}).")


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} must be an integer at least {minimum}.")
    return value


def _require_optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float} or not isfinite(value):
        raise ValidationError(f"{label} must be a finite JSON number or null.")
    return float(value)


def _require_sha256(value: Any, label: str) -> str:
    if not (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_nonblank_string(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError(f"{label} must be a non-blank string.")
    return value


def _require_aware_timestamp(value: Any, label: str) -> datetime:
    if type(value) is not str:
        raise ValidationError(f"{label} must be a timezone-aware ISO timestamp.")
    try:
        timestamp = datetime.fromisoformat(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(
            f"{label} must be a timezone-aware ISO timestamp."
        ) from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValidationError(f"{label} must be timezone-aware.")
    return timestamp


@dataclass(frozen=True, slots=True)
class _SelectionBoundary:
    event: sqlite3.Row
    model_version: str
    selected_at: datetime


def _index_question_selected_events(
    connection: sqlite3.Connection,
) -> dict[str, list[tuple[sqlite3.Row, dict[str, Any]]]]:
    """Index immutable selections once so replay stays linear in event count."""

    indexed: dict[str, list[tuple[sqlite3.Row, dict[str, Any]]]] = {}
    for event in connection.execute(
        """SELECT * FROM events
           WHERE event_type='QuestionSelected'
           ORDER BY event_id"""
    ).fetchall():
        payload = _strict_object(
            event["payload_json"],
            f"QuestionSelected event {event['event_id']} payload",
        )
        decision_id = _require_nonblank_string(
            payload.get("decision_id"),
            f"QuestionSelected event {event['event_id']} decision_id",
        )
        indexed.setdefault(decision_id, []).append((event, payload))
    return indexed


def _validate_question_selected_event(
    *,
    decision: sqlite3.Row,
    follower: sqlite3.Row,
    label: str,
    index: dict[str, list[tuple[sqlite3.Row, dict[str, Any]]]],
    cache: dict[str, _SelectionBoundary] | None = None,
) -> _SelectionBoundary:
    """Resolve one immutable selection boundary and bind it to a later event."""

    decision_id = _require_nonblank_string(
        decision["id"], f"{label} decision ID"
    )
    boundary = cache.get(decision_id) if cache is not None else None
    if boundary is None:
        matching = index.get(decision_id, [])
        if len(matching) != 1:
            raise ValidationError(
                f"{label} has no unique QuestionSelected event anchor."
            )

        selected_event, payload = matching[0]
        selection_label = (
            f"QuestionSelected event {selected_event['event_id']}"
        )
        metadata = _strict_object(
            selected_event["metadata_json"], f"{selection_label} metadata"
        )
        _require_exact_fields(
            metadata,
            QUESTION_SELECTED_METADATA_FIELDS,
            f"{selection_label} metadata",
        )
        model_version = metadata["learner_model_version"]
        if (
            type(model_version) is not str
            or model_version not in SUPPORTED_MODEL_VERSIONS
        ):
            raise ValidationError(
                f"{selection_label} uses unsupported learner model "
                f"{model_version!r}; this binary supports "
                f"{sorted(SUPPORTED_MODEL_VERSIONS)!r}."
            )
        policy_version = _require_nonblank_string(
            metadata["policy_version"], f"{selection_label} policy_version"
        )
        corpus_release_id = _require_nonblank_string(
            metadata["corpus_release_id"],
            f"{selection_label} corpus_release_id",
        )
        if (
            policy_version != decision["policy_version"]
            or corpus_release_id != decision["corpus_release_id"]
        ):
            raise ValidationError(
                f"{selection_label} metadata does not match its decision."
            )

        objective_aware = bool(
            decision["question_objective_id"] is not None
            or decision["focus_objective_id"] is not None
        )
        try:
            expected_schema = question_selected_schema_for(
                model_version, objective_aware=objective_aware
            )
        except ValueError as exc:
            raise ValidationError(
                f"{selection_label} has no supported schema boundary."
            ) from exc
        if selected_event["schema_version"] != expected_schema:
            raise ValidationError(
                f"{selection_label} schema does not match learner model "
                f"{model_version}."
            )

        _require_exact_fields(
            payload,
            (
                QUESTION_SELECTED_OBJECTIVE_FIELDS
                if objective_aware
                else QUESTION_SELECTED_BASE_FIELDS
            ),
            f"{selection_label} payload",
        )
        expected_payload = {
            "decision_id": decision_id,
            "question_id": decision["question_id"],
            "phase": decision["phase"],
            "candidate_count": decision["candidate_count"],
            "candidate_digest": decision["candidate_digest"],
            "propensity": decision["propensity"],
            "score": _strict_object(
                decision["selected_score_json"],
                f"Decision {decision_id} selected score",
            ),
            "option_order": _strict_array(
                decision["option_order_json"],
                f"Decision {decision_id} option order",
            ),
            "question_version": decision["question_version"],
            "question_content_hash": decision["question_content_hash"],
            "question_status": decision["question_status"],
            "evidence_weight": decision["evidence_weight"],
            "corpus_release_id": decision["corpus_release_id"],
            "session_revision": decision["session_revision"],
            "learner_revision": decision["learner_revision"],
            "focus_concept_id": decision["focus_concept_id"],
            "focus_misconception_id": decision[
                "focus_misconception_id"
            ],
            "pedagogical_role": decision["pedagogical_role"],
            "focus_valid": bool(decision["focus_valid"]),
        }
        if objective_aware:
            expected_payload.update(
                {
                    "question_objective_id": decision[
                        "question_objective_id"
                    ],
                    "focus_objective_id": decision["focus_objective_id"],
                }
            )
        if not same_json_value(payload, expected_payload):
            raise ValidationError(
                f"{selection_label} payload does not match its decision."
            )

        selected_at = _require_aware_timestamp(
            selected_event["occurred_at"],
            f"{selection_label} occurrence time",
        )
        if (
            expected_schema == BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION
            and (
                selected_event["occurred_at"] != decision["created_at"]
                or _require_aware_timestamp(
                    decision["created_at"],
                    f"Decision {decision_id} creation time",
                )
                != selected_at
            )
        ):
            raise ValidationError(
                f"{selection_label} occurrence time does not match its "
                "decision clock."
            )
        boundary = _SelectionBoundary(
            event=selected_event,
            model_version=model_version,
            selected_at=selected_at,
        )
        if cache is not None:
            cache[decision_id] = boundary

    selected_event = boundary.event
    expected_stream_id = f"learner:{decision['learner_id']}"
    if (
        selected_event["learner_id"] != decision["learner_id"]
        or selected_event["session_id"] != decision["session_id"]
        or selected_event["stream_id"] != expected_stream_id
        or follower["learner_id"] != decision["learner_id"]
        or follower["session_id"] != decision["session_id"]
        or follower["stream_id"] != expected_stream_id
    ):
        raise ValidationError(
            f"{label} does not match its QuestionSelected event envelope."
        )
    selection_version = _require_int(
        selected_event["stream_version"],
        f"QuestionSelected event {selected_event['event_id']} stream version",
        minimum=1,
    )
    follower_version = _require_int(
        follower["stream_version"], f"{label} stream version", minimum=1
    )
    if selection_version >= follower_version:
        raise ValidationError(
            f"{label} does not follow its QuestionSelected event anchor."
        )
    return boundary


def _validate_artifact_reference(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object or null.")
    _require_exact_fields(value, _ARTIFACT_FIELDS, label)
    digest = _require_sha256(value["sha256"], f"{label} sha256")
    size_bytes = _require_int(value["size_bytes"], f"{label} size_bytes")
    if size_bytes > 1_073_741_824:
        raise ValidationError(
            f"{label} size_bytes must be at most 1073741824."
        )
    media_type = value["media_type"]
    if not (
        type(media_type) is str
        and media_type == media_type.strip()
        and 1 <= len(media_type) <= 127
        and "/" in media_type
        and not any(character.isspace() for character in media_type)
    ):
        raise ValidationError(f"{label} media_type is not a compact MIME type.")
    return {
        "sha256": digest,
        "size_bytes": size_bytes,
        "media_type": media_type,
    }


def _projection_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _command_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_boundary_candidate(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object.")
    _require_exact_fields(value, _BOUNDARY_CANDIDATE_FIELDS, label)
    _require_nonblank_string(value["concept_id"], f"{label} concept_id")
    _require_nonblank_string(
        value["recursive_bottleneck_concept_id"],
        f"{label} recursive_bottleneck_concept_id",
    )
    for field in _BOUNDARY_CANDIDATE_FIELDS - {
        "concept_id",
        "recursive_bottleneck_concept_id",
    }:
        number = _require_optional_number(value[field], f"{label} {field}")
        if number is None or not 0.0 <= number <= 1.0:
            raise ValidationError(f"{label} {field} must be between zero and one.")
    return value


def _validate_boundary_decision(value: Any, label: str) -> None:
    if value is None:
        return
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object or null.")
    if "focus_objective_id" in value:
        _require_exact_fields(value, _OBJECTIVE_BOUNDARY_DECISION_FIELDS, label)
        focus_objective = _require_nonblank_string(
            value["focus_objective_id"], f"{label} focus_objective_id"
        )
        selected_objective = _require_nonblank_string(
            value["selected_objective_id"], f"{label} selected_objective_id"
        )
        _require_nonblank_string(
            value["algorithm_version"], f"{label} algorithm_version"
        )
        if focus_objective == selected_objective:
            raise ValidationError(f"{label} cannot select its own focus objective.")
        candidates = value["candidates"]
        if type(candidates) is not list or not candidates:
            raise ValidationError(f"{label} candidates must be a non-empty array.")
        candidate_ids: list[str] = []
        for index, candidate in enumerate(candidates):
            candidate_label = f"{label} candidate {index}"
            if type(candidate) is not dict:
                raise ValidationError(f"{candidate_label} must be an object.")
            _require_exact_fields(
                candidate, _OBJECTIVE_BOUNDARY_CANDIDATE_FIELDS, candidate_label
            )
            for field in (
                "edge_id",
                "objective_id",
                "concept_id",
                "rationale",
            ):
                _require_nonblank_string(
                    candidate[field], f"{candidate_label} {field}"
                )
            relation = _require_nonblank_string(
                candidate["relation"], f"{candidate_label} relation"
            )
            if relation not in {"prerequisite", "requires"}:
                raise ValidationError(
                    f"{candidate_label} relation must be a prerequisite relation."
                )
            for field in _OBJECTIVE_BOUNDARY_CANDIDATE_FIELDS - {
                "edge_id",
                "objective_id",
                "concept_id",
                "relation",
                "rationale",
            }:
                number = _require_optional_number(
                    candidate[field], f"{candidate_label} {field}"
                )
                if number is None or not 0.0 <= number <= 1.0:
                    raise ValidationError(
                        f"{candidate_label} {field} must be between zero and one."
                    )
            candidate_ids.append(candidate["objective_id"])
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValidationError(
                f"{label} candidate objective IDs must be unique."
            )
        if selected_objective not in candidate_ids:
            raise ValidationError(
                f"{label} selected objective is absent from candidates."
            )
        selected = value["selected"]
        if type(selected) is not dict or selected.get("objective_id") != selected_objective:
            raise ValidationError(
                f"{label} selected candidate does not match selected_objective_id."
            )
        if selected not in candidates:
            raise ValidationError(
                f"{label} selected candidate is absent from candidates."
            )
        return
    _require_exact_fields(value, _BOUNDARY_DECISION_FIELDS, label)
    focus = _require_nonblank_string(
        value["focus_concept_id"], f"{label} focus_concept_id"
    )
    selected_id = _require_nonblank_string(
        value["selected_concept_id"], f"{label} selected_concept_id"
    )
    _require_nonblank_string(
        value["algorithm_version"], f"{label} algorithm_version"
    )
    if focus == selected_id:
        raise ValidationError(f"{label} cannot select its own focus concept.")
    candidates = value["candidates"]
    if type(candidates) is not list or not candidates:
        raise ValidationError(f"{label} candidates must be a non-empty array.")
    validated = [
        _validate_boundary_candidate(candidate, f"{label} candidate {index}")
        for index, candidate in enumerate(candidates)
    ]
    candidate_ids = [candidate["concept_id"] for candidate in validated]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValidationError(f"{label} candidate concept IDs must be unique.")
    if selected_id not in candidate_ids:
        raise ValidationError(f"{label} selected concept is absent from candidates.")
    selected = _validate_boundary_candidate(value["selected"], f"{label} selected")
    matching = next(
        candidate for candidate in validated if candidate["concept_id"] == selected_id
    )
    if selected != matching:
        raise ValidationError(f"{label} selected candidate snapshot does not match.")


class ProjectionReplay:
    """Rebuild learner-model projections on disposable SQLite copies.

    The source database is opened read-only and is never initialized or
    migrated in place. Schema migration, projection deletion, replay, and
    verification happen only on a SQLite backup.
    """

    def __init__(self, database: Database):
        self.database = database

    def _validate_source(self) -> int:
        path = self.database.path
        connection = Database(path, read_only=True).connect()
        try:
            has_meta = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
            ).fetchone()
            if not has_meta:
                raise ValidationError("Database has no TSQ schema metadata.")
            row = connection.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                raise ValidationError("Database has no TSQ schema version.")
            try:
                version = int(row["value"])
            except (TypeError, ValueError) as exc:
                raise ValidationError("Database schema version is not an integer.") from exc
            if version > SCHEMA_VERSION:
                raise ConflictError(
                    f"Database schema is {version}; replay supports at most {SCHEMA_VERSION}."
                )
            return version
        finally:
            connection.close()

    def _backup_to(self, destination: Path) -> int:
        source_version = self._validate_source()
        source = Database(self.database.path, read_only=True).connect()
        destination_connection = sqlite3.connect(destination, timeout=20.0)
        try:
            source.backup(destination_connection)
        except sqlite3.Error as exc:
            raise ValidationError(f"Could not create a consistent database copy: {exc}") from exc
        finally:
            destination_connection.close()
            source.close()
        return source_version

    @staticmethod
    def _projection_commitment(
        connection: sqlite3.Connection, learner_id: str
    ) -> tuple[str | None, int | None]:
        row = connection.execute(
            """SELECT schema_version, payload_json FROM events
               WHERE stream_id=? AND event_type='LearnerProjectionAdvanced'
               ORDER BY stream_version DESC LIMIT 1""",
            (f"learner:{learner_id}",),
        ).fetchone()
        if row is None:
            return None, None
        payload = _strict_object(row["payload_json"], "Latest projection event payload")
        commitment = _require_sha256(
            payload.get("projection_hash"), "Latest projection event commitment"
        )
        hash_version = payload.get("projection_hash_version", 1)
        if type(hash_version) is not int or hash_version not in {1, 2, 3}:
            raise ValidationError(
                "Latest projection event has an invalid projection hash version."
            )
        required_hash_version = PROJECTION_HASH_VERSION_BY_EVENT_SCHEMA.get(
            row["schema_version"]
        )
        if (
            required_hash_version is not None
            and hash_version != required_hash_version
        ):
            raise ValidationError(
                "Latest projection event schema/hash version mismatch."
            )
        return commitment, hash_version

    @staticmethod
    def _recoverable_projection_errors(
        errors: list[str], learner_id: str
    ) -> tuple[list[str], list[str]]:
        recoverable_message = f"learner {learner_id}: projection hash mismatch"
        projection_hash_failure = (
            f"learner {learner_id}: projection cannot be hashed ("
        )
        objective_projection_prefix = f"objective state {learner_id}/"

        def action_projection_error(error: str) -> bool:
            return (
                error.startswith("learning action ")
                or error.startswith("learning artifact ")
                or "LearnerActionRecorded has no action projection" in error
                or "traced hints" in error
            )

        def performance_projection_error(error: str) -> bool:
            return (
                error.startswith("performance attempt ")
                or error.startswith("performance action ")
                or error.startswith("performance scoring claim ")
                or error.startswith("task evaluation ")
                or error.startswith("shadow evidence bundle ")
                or "PerformanceTaskStarted has no attempt projection" in error
                or "PerformanceActionRecorded has no action projection" in error
                or "PerformanceScoringClaimed has no scoring claim projection"
                in error
                or "PerformanceScoringClaimMigrated has no scoring claim projection"
                in error
                or "TaskEvaluationRecorded has no evaluation projection" in error
                or "ShadowEvidenceReduced has no bundle projection" in error
            )

        recoverable = [
            error
            for error in errors
            if (
                error == recoverable_message
                or error.startswith(projection_hash_failure)
                or error.startswith(objective_projection_prefix)
                or action_projection_error(error)
                or performance_projection_error(error)
                or error.startswith("policy shadow evaluation ")
            )
        ]
        blocking = [error for error in errors if error not in recoverable]
        return recoverable, blocking

    @staticmethod
    def _action_projection_snapshot(
        connection: sqlite3.Connection,
    ) -> dict[str, list[dict[str, Any]]]:
        artifact_rows = connection.execute(
            """SELECT id, sha256, size_bytes, media_type, created_at
               FROM learning_artifacts ORDER BY id"""
        ).fetchall()
        action_rows = connection.execute(
            """SELECT id, event_id, decision_id, session_id, learner_id,
                      sequence, stage, action_type, payload_json, artifact_id,
                      occurred_at, recorded_at, command_hash
               FROM learning_actions ORDER BY decision_id, sequence, id"""
        ).fetchall()
        return {
            "artifacts": [
                {column: row[column] for column in _ARTIFACT_PROJECTION_COLUMNS}
                for row in artifact_rows
            ],
            "actions": [
                {column: row[column] for column in _ACTION_PROJECTION_COLUMNS}
                for row in action_rows
            ],
        }

    def _derive_action_projections(
        self, connection: sqlite3.Connection
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Derive action and artifact rows exclusively from immutable events."""

        events = list(
            connection.execute(
                """SELECT * FROM events
                   WHERE event_type='LearnerActionRecorded'
                   ORDER BY event_id"""
            ).fetchall()
        )
        for event in events:
            _require_nonblank_string(event["event_id"], "Action event ID")
            _require_aware_timestamp(
                event["recorded_at"], f"Action event {event['event_id']} recording time"
            )
        events.sort(
            key=lambda event: (
                datetime.fromisoformat(event["recorded_at"]),
                event["event_id"],
            )
        )
        actions: list[dict[str, Any]] = []
        checkpoints: list[dict[str, Any]] = []
        artifact_rows: dict[str, dict[str, Any]] = {}
        artifact_metadata: dict[str, tuple[int, str]] = {}
        artifact_first_recording: dict[str, tuple[datetime, str]] = {}
        action_ids: set[str] = set()

        attempts = {
            row["decision_id"]: row
            for row in connection.execute(
                """SELECT attempt.decision_id, attempt.answered_at,
                          response.stream_id AS response_stream_id,
                          response.stream_version AS response_stream_version
                   FROM attempts attempt
                   JOIN events response ON response.event_id=attempt.event_id"""
            ).fetchall()
        }
        session_bounds: dict[str, dict[str, list[sqlite3.Row]]] = {}
        for boundary in connection.execute(
            """SELECT * FROM events
               WHERE event_type IN ('SessionStarted', 'SessionEnded')
               ORDER BY stream_id, stream_version"""
        ).fetchall():
            session_id = boundary["session_id"]
            if type(session_id) is not str or not session_id:
                continue
            kind = "started" if boundary["event_type"] == "SessionStarted" else "ended"
            session_bounds.setdefault(
                session_id, {"started": [], "ended": []}
            )[kind].append(boundary)
        revocations = {
            row["question_id"]: row
            for row in connection.execute(
                """SELECT revocation.question_id, revocation.revoked_at,
                          event.event_id AS revocation_event_id,
                          event.occurred_at AS revocation_occurred_at,
                          event.recorded_at AS revocation_recorded_at
                   FROM question_revocations revocation
                   JOIN events event ON event.event_id=revocation.event_id"""
            ).fetchall()
        }
        selection_event_index = _index_question_selected_events(connection)
        selection_event_cache: dict[str, _SelectionBoundary] = {}
        invalidation_events: dict[str, list[sqlite3.Row]] = {}
        for invalidation in connection.execute(
            """SELECT * FROM events
               WHERE event_type='DecisionInvalidated'
               ORDER BY stream_id, stream_version"""
        ).fetchall():
            invalidation_payload = _strict_object(
                invalidation["payload_json"],
                f"DecisionInvalidated event {invalidation['event_id']} payload",
            )
            invalidated_decision_id = invalidation_payload.get("decision_id")
            if type(invalidated_decision_id) is str and invalidated_decision_id:
                invalidation_events.setdefault(
                    invalidated_decision_id, []
                ).append(invalidation)

        for event in events:
            label = f"Action event {event['event_id']}"
            _require_int(
                event["stream_version"], f"{label} stream version", minimum=1
            )
            if event["schema_version"] != ACTION_EVENT_SCHEMA_VERSION:
                raise ValidationError(
                    f"{label} uses unsupported schema version "
                    f"{event['schema_version']}; replay supports exactly version "
                    f"{ACTION_EVENT_SCHEMA_VERSION} for LearnerActionRecorded."
                )
            payload = _strict_object(event["payload_json"], f"{label} payload")
            metadata = _strict_object(event["metadata_json"], f"{label} metadata")
            _require_exact_fields(payload, _ACTION_FIELDS, f"{label} payload")
            _require_exact_fields(
                metadata, _ACTION_METADATA_FIELDS, f"{label} metadata"
            )
            if type(metadata["action_schema_version"]) is not int or metadata[
                "action_schema_version"
            ] != ACTION_EVENT_SCHEMA_VERSION:
                raise ValidationError(f"{label} has an unsupported action schema.")
            if metadata["observational_only"] is not True:
                raise ValidationError(
                    f"{label} must be explicitly marked observational_only."
                )

            decision_id = _require_nonblank_string(
                payload["decision_id"], f"{label} decision_id"
            )
            decision = connection.execute(
                """SELECT decision.*,
                          session.learner_id AS session_learner_id
                   FROM decisions decision
                   JOIN sessions session ON session.id=decision.session_id
                   WHERE decision.id=?""",
                (decision_id,),
            ).fetchone()
            if decision is None:
                raise ValidationError(f"{label} cites an unknown decision.")
            if (
                event["causation_id"] != decision_id
                or event["learner_id"] != decision["learner_id"]
                or event["learner_id"] != decision["session_learner_id"]
                or event["session_id"] != decision["session_id"]
                or event["stream_id"] != f"learner:{decision['learner_id']}"
            ):
                raise ValidationError(f"{label} does not match its decision envelope.")
            if metadata["corpus_release_id"] != decision["corpus_release_id"]:
                raise ValidationError(f"{label} has a corpus-release mismatch.")
            selection_boundary = _validate_question_selected_event(
                decision=decision,
                follower=event,
                label=label,
                index=selection_event_index,
                cache=selection_event_cache,
            )
            selected_event = selection_boundary.event
            decision_invalidations = invalidation_events.get(decision_id, [])
            if decision["invalidated_at"] is not None:
                if len(decision_invalidations) != 1:
                    raise ValidationError(
                        f"{label} has no unique DecisionInvalidated event boundary."
                    )
                invalidation = decision_invalidations[0]
                if (
                    invalidation["schema_version"] != 1
                    or invalidation["stream_id"] != event["stream_id"]
                    or invalidation["learner_id"] != event["learner_id"]
                    or invalidation["session_id"] != event["session_id"]
                    or invalidation["causation_id"] != decision_id
                    or invalidation["occurred_at"] != decision["invalidated_at"]
                ):
                    raise ValidationError(
                        f"{label} has an inconsistent DecisionInvalidated boundary."
                    )
                if event["stream_version"] >= invalidation["stream_version"]:
                    raise ValidationError(
                        f"{label} was appended at or after decision invalidation."
                    )
            elif decision_invalidations:
                raise ValidationError(
                    f"{label} has an unprojected DecisionInvalidated boundary."
                )
            bounds = session_bounds.get(decision["session_id"])
            if bounds is None or len(bounds["started"]) != 1:
                raise ValidationError(
                    f"{label} has no unique SessionStarted event boundary."
                )
            if len(bounds["ended"]) > 1:
                raise ValidationError(
                    f"{label} has multiple SessionEnded event boundaries."
                )
            started = bounds["started"][0]
            ended = bounds["ended"][0] if bounds["ended"] else None
            start_payload = _strict_object(
                started["payload_json"],
                f"SessionStarted event {started['event_id']} payload",
            )
            if (
                start_payload.get("session_id") != decision["session_id"]
                or started["stream_id"] != event["stream_id"]
                or started["learner_id"] != event["learner_id"]
                or started["stream_version"] >= event["stream_version"]
            ):
                raise ValidationError(
                    f"{label} falls outside its session-active event interval."
                )
            if ended is not None:
                end_payload = _strict_object(
                    ended["payload_json"],
                    f"SessionEnded event {ended['event_id']} payload",
                )
                if (
                    end_payload.get("session_id") != decision["session_id"]
                    or ended["stream_id"] != event["stream_id"]
                    or ended["learner_id"] != event["learner_id"]
                    or ended["stream_version"] <= event["stream_version"]
                ):
                    raise ValidationError(
                        f"{label} falls outside its session-active event interval."
                    )

            action_id = _require_nonblank_string(
                payload["action_id"], f"{label} action_id"
            )
            if action_id in action_ids:
                raise ValidationError(f"{label} repeats action ID {action_id!r}.")
            action_ids.add(action_id)
            sequence = _require_int(payload["sequence"], f"{label} sequence", minimum=1)
            try:
                kind = ActionKind(payload["action_type"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{label} has an unknown action type.") from exc
            try:
                phase = ActionPhase(payload["stage"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{label} has an unknown action stage.") from exc
            action_payload = payload["payload"]
            if type(action_payload) is not dict:
                raise ValidationError(f"{label} semantic payload must be an object.")
            selected_at = _require_aware_timestamp(
                decision["created_at"], f"{label} decision timestamp"
            )
            occurred_at = _require_aware_timestamp(
                event["occurred_at"], f"{label} occurrence time"
            )
            recorded_at = _require_aware_timestamp(
                event["recorded_at"], f"{label} recording time"
            )
            revocation = revocations.get(decision["question_id"])
            if revocation is not None:
                if revocation["revoked_at"] != revocation["revocation_occurred_at"]:
                    raise ValidationError(
                        f"{label} has an inconsistent emergency-revocation boundary."
                    )
                revoked_recorded_at = _require_aware_timestamp(
                    revocation["revocation_recorded_at"],
                    f"Revocation event {revocation['revocation_event_id']} recording time",
                )
                if recorded_at >= revoked_recorded_at:
                    raise ValidationError(
                        f"{label} was recorded at or after emergency revocation."
                    )
            if occurred_at < selected_at:
                raise ValidationError(f"{label} occurred before question selection.")
            elapsed_ms = int((occurred_at - selected_at).total_seconds() * 1000)
            try:
                typed_action = LearningAction(
                    id=action_id,
                    trace_id=decision_id,
                    sequence=sequence,
                    kind=kind,
                    phase=phase,
                    payload=action_payload,
                    elapsed_ms=elapsed_ms,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(f"{label} has invalid semantic content: {exc}") from exc
            canonical_payload = typed_action.terms()["payload"]
            payload_json = canonical_json(canonical_payload)
            if len(payload_json.encode("utf-8")) > 16_384:
                raise ValidationError(f"{label} semantic payload exceeds 16384 bytes.")

            artifact = _validate_artifact_reference(
                payload["artifact"], f"{label} artifact"
            )
            artifact_id: str | None = None
            if artifact is not None:
                digest_field = _ACTION_ARTIFACT_DIGEST_FIELDS.get(kind)
                if (
                    digest_field is None
                    or canonical_payload.get(digest_field) != artifact["sha256"]
                ):
                    raise ValidationError(
                        f"{label} artifact does not match its semantic payload digest."
                    )
                artifact_id = f"art_{artifact['sha256']}"
                declared_metadata = (artifact["size_bytes"], artifact["media_type"])
                prior_metadata = artifact_metadata.get(artifact["sha256"])
                if prior_metadata is not None and prior_metadata != declared_metadata:
                    raise ValidationError(
                        f"{label} redeclares an artifact digest with different metadata."
                    )
                artifact_metadata[artifact["sha256"]] = declared_metadata
                if artifact_id not in artifact_rows:
                    artifact_rows[artifact_id] = {
                        "id": artifact_id,
                        "sha256": artifact["sha256"],
                        "size_bytes": artifact["size_bytes"],
                        "media_type": artifact["media_type"],
                        "created_at": event["occurred_at"],
                    }
                    artifact_first_recording[artifact["sha256"]] = (
                        recorded_at,
                        event["occurred_at"],
                    )
                else:
                    first_recorded_at, first_occurred_at = artifact_first_recording[
                        artifact["sha256"]
                    ]
                    if (
                        recorded_at == first_recorded_at
                        and event["occurred_at"] != first_occurred_at
                    ):
                        raise ValidationError(
                            f"{label} makes artifact creation order ambiguous; equal "
                            "recording times cite different occurrence times."
                        )

            attempt = attempts.get(decision_id)
            if phase is ActionPhase.POST_FEEDBACK:
                if attempt is None or decision["consumed_at"] is None:
                    raise ValidationError(
                        f"{label} is post-feedback but its decision has no answer."
                    )
                answered_at = _require_aware_timestamp(
                    attempt["answered_at"], f"{label} answer time"
                )
                if occurred_at < answered_at:
                    raise ValidationError(f"{label} precedes its linked answer.")
                if (
                    event["stream_id"] != attempt["response_stream_id"]
                    or event["stream_version"] <= attempt["response_stream_version"]
                ):
                    raise ValidationError(
                        f"{label} does not follow its response in the event stream."
                    )
            elif attempt is not None:
                answered_at = _require_aware_timestamp(
                    attempt["answered_at"], f"{label} answer time"
                )
                if occurred_at > answered_at:
                    raise ValidationError(f"{label} follows its linked answer.")
                if (
                    event["stream_id"] != attempt["response_stream_id"]
                    or event["stream_version"] >= attempt["response_stream_version"]
                ):
                    raise ValidationError(
                        f"{label} does not precede its response in the event stream."
                    )
            if phase is not ActionPhase.POST_FEEDBACK and connection.execute(
                """SELECT 1 FROM events
                   WHERE stream_id=?
                     AND event_type='LearnerProjectionAdvanced'
                     AND stream_version>? AND stream_version<?
                   LIMIT 1""",
                (
                    event["stream_id"],
                    selected_event["stream_version"],
                    event["stream_version"],
                ),
            ).fetchone():
                raise ValidationError(
                    f"{label} records pre-response work for a stale decision."
                )

            command = {
                "decision_id": decision_id,
                "stage": phase.value,
                "action_type": kind.value,
                "payload": canonical_payload,
                "artifact": artifact,
            }
            action_row = {
                "id": action_id,
                "event_id": event["event_id"],
                "decision_id": decision_id,
                "session_id": decision["session_id"],
                "learner_id": decision["learner_id"],
                "sequence": sequence,
                "stage": phase.value,
                "action_type": kind.value,
                "payload_json": payload_json,
                "artifact_id": artifact_id,
                "occurred_at": event["occurred_at"],
                "recorded_at": event["recorded_at"],
                "command_hash": _command_digest(command),
                "_stream_version": event["stream_version"],
                "_typed_action": typed_action,
            }
            actions.append(action_row)
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "action_id": action_id,
                    "decision_id": decision_id,
                    "sequence": sequence,
                    "artifact_id": artifact_id,
                }
            )

        by_decision: dict[str, list[dict[str, Any]]] = {}
        for action in actions:
            by_decision.setdefault(action["decision_id"], []).append(action)
        for decision_id, trace in by_decision.items():
            trace.sort(key=lambda item: (item["sequence"], item["event_id"]))
            for expected_sequence, action in enumerate(trace, start=1):
                if action["sequence"] != expected_sequence:
                    raise ValidationError(
                        f"Decision {decision_id} action sequence is not contiguous at "
                        f"{expected_sequence}."
                    )
                if expected_sequence > 1:
                    prior = trace[expected_sequence - 2]
                    if action["_stream_version"] <= prior["_stream_version"]:
                        raise ValidationError(
                            f"Decision {decision_id} action event order is not monotonic."
                        )
                    if _require_aware_timestamp(
                        action["occurred_at"], "Action occurrence time"
                    ) < _require_aware_timestamp(
                        prior["occurred_at"], "Prior action occurrence time"
                    ):
                        raise ValidationError(
                            f"Decision {decision_id} action times are not monotonic."
                        )
            try:
                summarize_actions(action["_typed_action"] for action in trace)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError(
                    f"Decision {decision_id} has an invalid action lifecycle: {exc}"
                ) from exc

        actions.sort(key=lambda item: (item["decision_id"], item["sequence"], item["id"]))
        for action in actions:
            del action["_stream_version"]
            del action["_typed_action"]
        checkpoints.sort(
            key=lambda item: (item["decision_id"], item["sequence"], item["action_id"])
        )
        artifacts = sorted(artifact_rows.values(), key=lambda item: item["id"])
        return actions, artifacts, checkpoints

    def _rebuild_action_projections(
        self, work_database: Database
    ) -> dict[str, Any]:
        with work_database.transaction() as connection:
            actions, artifacts, checkpoints = self._derive_action_projections(connection)
            connection.executescript(
                """
                DROP TRIGGER IF EXISTS learning_artifacts_no_update;
                DROP TRIGGER IF EXISTS learning_artifacts_no_delete;
                DROP TRIGGER IF EXISTS learning_actions_validate_insert;
                DROP TRIGGER IF EXISTS learning_actions_no_update;
                DROP TRIGGER IF EXISTS learning_actions_no_delete;
                DELETE FROM learning_actions;
                DELETE FROM learning_artifacts;
                """
            )
            connection.executemany(
                """INSERT INTO learning_artifacts(
                       id, sha256, size_bytes, media_type, created_at
                   ) VALUES (:id, :sha256, :size_bytes, :media_type, :created_at)""",
                artifacts,
            )
            connection.executemany(
                """INSERT INTO learning_actions(
                       id, event_id, decision_id, session_id, learner_id,
                       sequence, stage, action_type, payload_json, artifact_id,
                       occurred_at, recorded_at, command_hash
                   ) VALUES (
                       :id, :event_id, :decision_id, :session_id, :learner_id,
                       :sequence, :stage, :action_type, :payload_json, :artifact_id,
                       :occurred_at, :recorded_at, :command_hash
                   )""",
                actions,
            )
            work_database._install_v8_learning_action_triggers(connection)
            snapshot = self._action_projection_snapshot(connection)
        return {
            "action_count": len(actions),
            "artifact_count": len(artifacts),
            "checkpoints": checkpoints,
            "snapshot": snapshot,
            "projection_hash": _projection_digest(snapshot),
        }

    def _pair_events(
        self, connection: sqlite3.Connection, learner_id: str
    ) -> list[tuple[sqlite3.Row, sqlite3.Row]]:
        stream_id = f"learner:{learner_id}"
        events = connection.execute(
            "SELECT * FROM events WHERE stream_id=? ORDER BY stream_version",
            (stream_id,),
        ).fetchall()
        pairs: list[tuple[sqlite3.Row, sqlite3.Row]] = []
        pending: sqlite3.Row | None = None
        for event in events:
            if event["event_type"] == "ResponseSubmitted":
                if pending is not None:
                    raise ValidationError(
                        f"Response event {pending['event_id']} has no immediately following "
                        "projection event."
                    )
                pending = event
                continue
            if event["event_type"] == "LearnerProjectionAdvanced":
                if pending is None:
                    raise ValidationError(
                        f"Projection event {event['event_id']} has no preceding response event."
                    )
                pairs.append((pending, event))
                pending = None
                continue
            if pending is not None:
                raise ValidationError(
                    f"Response event {pending['event_id']} is separated from its projection "
                    f"by {event['event_type']}."
                )
        if pending is not None:
            raise ValidationError(
                f"Response event {pending['event_id']} has no projection event."
            )
        return pairs

    def _validate_pair_envelope(
        self,
        learner_id: str,
        response: sqlite3.Row,
        projection: sqlite3.Row,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        if response["schema_version"] not in RESPONSE_EVENT_SCHEMA_VERSIONS:
            supported = ", ".join(
                str(version)
                for version in sorted(RESPONSE_EVENT_SCHEMA_VERSIONS)
            )
            raise ValidationError(
                f"Event {response['event_id']} uses unsupported schema version "
                f"{response['schema_version']}; replay supports versions "
                f"{supported} for ResponseSubmitted."
            )
        if projection["schema_version"] not in PROJECTION_EVENT_SCHEMA_VERSIONS:
            supported = ", ".join(
                str(version) for version in sorted(PROJECTION_EVENT_SCHEMA_VERSIONS)
            )
            raise ValidationError(
                f"Event {projection['event_id']} uses unsupported schema version "
                f"{projection['schema_version']}; replay supports versions {supported} "
                "for LearnerProjectionAdvanced."
            )
        for event in (response, projection):
            if event["learner_id"] != learner_id:
                raise ValidationError(f"Event {event['event_id']} has a learner mismatch.")
        if projection["causation_id"] != response["event_id"]:
            raise ValidationError(
                f"Projection event {projection['event_id']} does not cite response "
                f"{response['event_id']} as its cause."
            )
        if response["session_id"] != projection["session_id"]:
            raise ValidationError(
                f"Response {response['event_id']} and projection {projection['event_id']} "
                "have different sessions."
            )
        if response["occurred_at"] != projection["occurred_at"]:
            raise ValidationError(
                f"Response {response['event_id']} and projection {projection['event_id']} "
                "have different occurrence times."
            )
        response_payload = _strict_object(
            response["payload_json"], f"Response event {response['event_id']} payload"
        )
        response_metadata = _strict_object(
            response["metadata_json"], f"Response event {response['event_id']} metadata"
        )
        projection_payload = _strict_object(
            projection["payload_json"], f"Projection event {projection['event_id']} payload"
        )
        projection_metadata = _strict_object(
            projection["metadata_json"], f"Projection event {projection['event_id']} metadata"
        )
        _require_exact_fields(
            response_payload, _RESPONSE_FIELDS, f"Response event {response['event_id']} payload"
        )
        _require_exact_fields(
            response_metadata,
            (
                _RESPONSE_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM
                if response["schema_version"]
                == CURRENT_RESPONSE_EVENT_SCHEMA_VERSION
                else _RESPONSE_METADATA_FIELDS
            ),
            f"Response event {response['event_id']} metadata",
        )
        _require_exact_fields(
            projection_payload,
            (
                _PROJECTION_FIELDS_V4
                if projection["schema_version"] == 4
                else (
                    _PROJECTION_FIELDS_V3
                    if projection["schema_version"] == 3
                    else (
                        _PROJECTION_FIELDS_V2
                        if projection["schema_version"] == 2
                        else _PROJECTION_FIELDS_V1
                    )
                )
            ),
            f"Projection event {projection['event_id']} payload",
        )
        _require_exact_fields(
            projection_metadata,
            (
                _PROJECTION_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM
                if response["schema_version"]
                == CURRENT_RESPONSE_EVENT_SCHEMA_VERSION
                else _PROJECTION_METADATA_FIELDS
            ),
            f"Projection event {projection['event_id']} metadata",
        )
        response_misconception_algorithm = response_metadata.get(
            MISCONCEPTION_ALGORITHM_METADATA_KEY
        )
        projection_misconception_algorithm = projection_metadata.get(
            MISCONCEPTION_ALGORITHM_METADATA_KEY
        )
        if (
            response_misconception_algorithm
            != projection_misconception_algorithm
        ):
            raise ValidationError(
                f"Response {response['event_id']} and projection "
                f"{projection['event_id']} name different misconception "
                "algorithms."
            )
        if (
            response["schema_version"]
            == CURRENT_RESPONSE_EVENT_SCHEMA_VERSION
            and response_misconception_algorithm
            != MISCONCEPTION_ALGORITHM_VERSION
        ):
            raise ValidationError(
                f"Response event {response['event_id']} uses unsupported "
                "misconception algorithm "
                f"{response_misconception_algorithm!r}."
            )
        if (
            type(projection_payload["response_event_id"]) is not str
            or projection_payload["response_event_id"]
            != response["event_id"]
        ):
            raise ValidationError(
                f"Projection event {projection['event_id']} names a different response."
            )
        try:
            SessionPhase(projection_payload["phase"])
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Projection event {projection['event_id']} has an invalid phase."
            ) from exc
        remediation_depth = _require_int(
            projection_payload["remediation_depth"],
            f"Projection event {projection['event_id']} remediation depth",
        )
        if remediation_depth > MAX_REMEDIATION_DEPTH:
            raise ValidationError(
                f"Projection event {projection['event_id']} remediation depth "
                f"must be at most {MAX_REMEDIATION_DEPTH}."
            )
        remediation_path = projection_payload["remediation_path"]
        if type(remediation_path) is not list or len(
            remediation_path
        ) > MAX_REMEDIATION_DEPTH:
            raise ValidationError(
                f"Projection event {projection['event_id']} has an invalid "
                "remediation path."
            )
        for frame_index, frame in enumerate(remediation_path):
            frame_label = (
                f"Projection event {projection['event_id']} remediation path "
                f"frame {frame_index}"
            )
            if type(frame) is not dict:
                raise ValidationError(f"{frame_label} must be an object.")
            expected_frame_fields = {
                "concept_id",
                "misconception_id",
            }
            if "objective_id" in frame:
                expected_frame_fields.add("objective_id")
            _require_exact_fields(frame, expected_frame_fields, frame_label)
            _require_nonblank_string(
                frame["concept_id"], f"{frame_label} concept_id"
            )
            for field in ("misconception_id", "objective_id"):
                if field in frame and frame[field] is not None:
                    _require_nonblank_string(
                        frame[field], f"{frame_label} {field}"
                    )
        state_changes = projection_payload["state_changes"]
        if type(state_changes) is not list or any(
            type(change) is not dict for change in state_changes
        ):
            raise ValidationError(
                f"Projection event {projection['event_id']} state changes must "
                "be an array of objects."
            )
        _require_int(
            projection_payload["learner_revision"],
            f"Projection event {projection['event_id']} learner revision",
            minimum=1,
        )
        if projection["schema_version"] >= 2:
            reason = projection_payload["transition_reason"]
            if type(reason) is not str or not reason.strip():
                raise ValidationError(
                    f"Projection event {projection['event_id']} has an invalid "
                    "transition reason."
                )
            _validate_boundary_decision(
                projection_payload["boundary_decision"],
                f"Projection event {projection['event_id']} boundary decision",
            )
        if projection["schema_version"] >= 3:
            required_hash_version = (
                3 if projection["schema_version"] == 4 else 2
            )
            if (
                projection_payload["projection_hash_version"]
                != required_hash_version
            ):
                raise ValidationError(
                    f"Projection event {projection['event_id']} must use "
                    f"projection hash version {required_hash_version}."
                )
            for field in ("question_objective_id", "focus_objective_id"):
                value = projection_payload[field]
                if value is not None and (
                    type(value) is not str or not value.strip()
                ):
                    raise ValidationError(
                        f"Projection event {projection['event_id']} has an invalid {field}."
                    )
        pair_model_versions: set[str] = set()
        for label, metadata in (
            (f"Response event {response['event_id']}", response_metadata),
            (f"Projection event {projection['event_id']}", projection_metadata),
        ):
            model_version = metadata["learner_model_version"]
            if model_version not in SUPPORTED_MODEL_VERSIONS:
                raise ValidationError(
                    f"{label} uses unsupported learner model {model_version!r}; "
                    f"this binary supports {sorted(SUPPORTED_MODEL_VERSIONS)!r}."
                )
            pair_model_versions.add(model_version)
        if len(pair_model_versions) != 1:
            raise ValidationError(
                f"Response {response['event_id']} and projection "
                f"{projection['event_id']} name different learner models."
            )
        pair_model_version = next(iter(pair_model_versions))
        required_model_versions = (
            PROJECTION_MODEL_VERSIONS_BY_EVENT_SCHEMA.get(
                projection["schema_version"]
            )
        )
        if (
            required_model_versions is not None
            and pair_model_version not in required_model_versions
        ):
            raise ValidationError(
                f"Projection event {projection['event_id']} schema "
                f"{projection['schema_version']} requires learner model "
                f"in {sorted(required_model_versions)!r}."
            )
        return (
            response_payload,
            response_metadata,
            projection_payload,
            projection_metadata,
        )

    def _rebuild_projection(
        self, work_database: Database, learner_id: str
    ) -> dict[str, Any]:
        checkpoints: list[dict[str, Any]] = []
        replay_errors: list[str] = []
        family_attempts: dict[str, int] = {}
        explicit_misconception_algorithm_seen = False
        with work_database.transaction() as connection:
            learner = connection.execute(
                "SELECT * FROM learners WHERE id=?", (learner_id,)
            ).fetchone()
            if learner is None:
                raise NotFoundError(f"Unknown learner: {learner_id}")
            pairs = self._pair_events(connection, learner_id)
            selection_event_index = _index_question_selected_events(connection)
            selection_event_cache: dict[str, _SelectionBoundary] = {}
            connection.execute("DELETE FROM skill_states WHERE learner_id=?", (learner_id,))
            connection.execute(
                "DELETE FROM objective_grid_states WHERE learner_id=?",
                (learner_id,),
            )
            connection.execute(
                "DELETE FROM objective_states WHERE learner_id=?", (learner_id,)
            )
            connection.execute(
                "DELETE FROM misconception_beliefs WHERE learner_id=?", (learner_id,)
            )
            connection.execute(
                "DELETE FROM learner_skill_families WHERE learner_id=?", (learner_id,)
            )
            connection.execute(
                "DELETE FROM learner_objective_families WHERE learner_id=?",
                (learner_id,),
            )
            connection.execute(
                "UPDATE learners SET revision=0 WHERE id=?", (learner_id,)
            )

            for expected_revision, (response, projection) in enumerate(pairs, start=1):
                (
                    response_payload,
                    response_metadata,
                    projection_payload,
                    projection_metadata,
                ) = self._validate_pair_envelope(
                    learner_id, response, projection
                )
                misconception_algorithm = response_metadata.get(
                    MISCONCEPTION_ALGORITHM_METADATA_KEY
                )
                if misconception_algorithm is None:
                    if explicit_misconception_algorithm_seen:
                        raise ValidationError(
                            f"Response event {response['event_id']} omits the "
                            "misconception algorithm after an explicitly "
                            "versioned response."
                        )
                    misconception_algorithm = LEGACY_MISCONCEPTION_ALGORITHM
                else:
                    explicit_misconception_algorithm_seen = True
                label = f"Response event {response['event_id']}"
                decision_id = response_payload["decision_id"]
                if type(decision_id) is not str or response["causation_id"] != decision_id:
                    raise ValidationError(f"{label} has an invalid decision cause.")
                decision = connection.execute(
                    "SELECT * FROM decisions WHERE id=?", (decision_id,)
                ).fetchone()
                attempt = connection.execute(
                    "SELECT * FROM attempts WHERE event_id=?", (response["event_id"],)
                ).fetchone()
                if decision is None or attempt is None:
                    raise ValidationError(f"{label} lacks its decision or attempt projection.")
                if decision["learner_id"] != learner_id or attempt["learner_id"] != learner_id:
                    raise ValidationError(f"{label} has inconsistent learner projections.")
                if decision["session_id"] != response["session_id"] or attempt[
                    "session_id"
                ] != response["session_id"]:
                    raise ValidationError(f"{label} has inconsistent session projections.")
                if attempt["decision_id"] != decision_id:
                    raise ValidationError(f"{label} attempt references a different decision.")
                selection_boundary = _validate_question_selected_event(
                    decision=decision,
                    follower=response,
                    label=label,
                    index=selection_event_index,
                    cache=selection_event_cache,
                )
                if (
                    response_metadata["learner_model_version"]
                    != selection_boundary.model_version
                ):
                    raise ValidationError(
                        f"{label} learner model does not match its "
                        "QuestionSelected event."
                    )
                question_id = response_payload["question_id"]
                if type(question_id) is not str or not question_id:
                    raise ValidationError(f"{label} has an invalid question ID.")
                if decision["question_id"] != question_id or attempt["question_id"] != question_id:
                    raise ValidationError(f"{label} has inconsistent question projections.")
                release_id = decision["corpus_release_id"]
                boundary = projection_payload.get("boundary_decision")
                if isinstance(boundary, dict) and "focus_objective_id" in boundary:
                    if boundary["focus_objective_id"] != decision["focus_objective_id"]:
                        raise ValidationError(
                            f"{label} objective boundary does not start at the "
                            "decision focus."
                        )
                    if (
                        boundary["selected_objective_id"]
                        != projection_payload["focus_objective_id"]
                    ):
                        raise ValidationError(
                            f"{label} objective boundary does not match the "
                            "projected focus."
                        )
                    graph = connection.execute(
                        """SELECT graph_version FROM release_objective_graphs
                           WHERE release_id=?""",
                        (release_id,),
                    ).fetchone()
                    if graph is None or graph["graph_version"] != 1:
                        raise ValidationError(
                            f"{label} objective boundary has no declared pinned graph."
                        )
                    for candidate in boundary["candidates"]:
                        edge = connection.execute(
                            """SELECT edge_id, source_objective_id,
                                      target_objective_id, relation, weight,
                                      rationale, objective.primary_concept_id
                               FROM release_objective_edges edge
                               JOIN learning_objectives objective
                                 ON objective.id=edge.source_objective_id
                               WHERE edge.release_id=? AND edge.edge_id=?""",
                            (release_id, candidate["edge_id"]),
                        ).fetchone()
                        if edge is None or any(
                            (
                                edge["source_objective_id"]
                                != candidate["objective_id"],
                                edge["target_objective_id"]
                                != boundary["focus_objective_id"],
                                edge["relation"] != candidate["relation"],
                                edge["weight"] != candidate["edge_weight"],
                                edge["rationale"] != candidate["rationale"],
                                edge["primary_concept_id"]
                                != candidate["concept_id"],
                            )
                        ):
                            raise ValidationError(
                                f"{label} objective boundary edge snapshot mismatch."
                            )
                question = work_database.get_question(
                    question_id,
                    connection,
                    release_id=release_id,
                )
                if question.objective_id != decision["question_objective_id"]:
                    raise ValidationError(
                        f"{label} learning-objective snapshot mismatch."
                    )
                if projection["schema_version"] >= 3:
                    if (
                        projection_payload["question_objective_id"]
                        != question.objective_id
                    ):
                        raise ValidationError(
                            f"{label} projection objective mismatch."
                        )
                elif question.objective_id is not None:
                    raise ValidationError(
                        f"{label} uses an objective-aware release with a legacy "
                        "projection event schema."
                    )
                response_question_version = _require_int(
                    response_payload["question_version"],
                    f"{label} question_version",
                    minimum=1,
                )
                if (
                    response_question_version != question.version
                    or decision["question_version"] != question.version
                    or attempt["question_version"] != question.version
                ):
                    raise ValidationError(f"{label} question version does not match the registry.")
                content_hash = question_content_hash(question)
                if decision["question_content_hash"] != content_hash or response_metadata[
                    "question_content_hash"
                ] != content_hash:
                    raise ValidationError(f"{label} question content hash mismatch.")
                if response_metadata["question_status"] != decision["question_status"]:
                    raise ValidationError(f"{label} question status snapshot mismatch.")
                try:
                    pinned_status = QuestionStatus(decision["question_status"])
                except ValueError as exc:
                    raise ValidationError(f"{label} has an invalid pinned question status.") from exc
                question = replace(question, status=pinned_status)
                if attempt["family_id"] != question.family_id:
                    raise ValidationError(f"{label} question family mismatch.")

                presented_order = response_payload["presented_order"]
                if type(presented_order) is not list or any(
                    type(option_id) is not str for option_id in presented_order
                ):
                    raise ValidationError(f"{label} presented order must be an array of IDs.")
                if _strict_array(
                    attempt["presented_order_json"], f"Attempt {attempt['id']} presented order"
                ) != presented_order:
                    raise ValidationError(f"{label} presented order mismatch.")
                selected_option_id = response_payload["selected_option_id"]
                if selected_option_id is not None and type(selected_option_id) is not str:
                    raise ValidationError(f"{label} selected option must be a string or null.")
                options = {option.id: option for option in question.options}
                if selected_option_id is not None and selected_option_id not in options:
                    raise ValidationError(f"{label} selected option is not in the question.")
                selected_option = options.get(selected_option_id)
                is_correct = bool(selected_option and selected_option.correct)
                if type(response_payload["is_correct"]) is not bool or response_payload[
                    "is_correct"
                ] != is_correct:
                    raise ValidationError(f"{label} correctness does not match the answer key.")
                if attempt["selected_option_id"] != selected_option_id or bool(
                    attempt["is_correct"]
                ) != is_correct:
                    raise ValidationError(f"{label} attempt answer mismatch.")

                confidence = _require_optional_number(
                    response_payload["confidence"], f"{label} confidence"
                )
                if confidence is not None and not 0.0 <= confidence <= 1.0:
                    raise ValidationError(f"{label} confidence is outside zero to one.")
                response_ms_value = response_payload["response_ms"]
                response_ms = (
                    None
                    if response_ms_value is None
                    else _require_int(response_ms_value, f"{label} response_ms")
                )
                hint_count = _require_int(
                    response_payload["hint_count"], f"{label} hint_count"
                )
                feedback_shown = response_payload["feedback_shown"]
                if type(feedback_shown) is not bool:
                    raise ValidationError(f"{label} feedback_shown must be boolean.")
                if (
                    attempt["confidence"] != confidence
                    or attempt["response_ms"] != response_ms
                    or attempt["hint_count"] != hint_count
                    or bool(attempt["feedback_shown"]) != feedback_shown
                    or attempt["answered_at"] != response["occurred_at"]
                ):
                    raise ValidationError(f"{label} attempt evidence fields mismatch.")
                try:
                    occurred_at = datetime.fromisoformat(response["occurred_at"])
                except (TypeError, ValueError) as exc:
                    raise ValidationError(f"{label} has an invalid occurrence time.") from exc
                if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
                    raise ValidationError(f"{label} occurrence time is timezone-naive.")
                if (
                    response_metadata["learner_model_version"]
                    in AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS
                ):
                    try:
                        authoritative_window = response_window(
                            selected_at=selection_boundary.selected_at,
                            answered_at=occurred_at,
                            response_ms=response_ms,
                        )
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValidationError(
                            f"{label} has an invalid authoritative response "
                            f"window: {exc}"
                        ) from exc
                    if not authoritative_window.consistent:
                        raise ValidationError(
                            f"{label} response_ms exceeds its authoritative "
                            "selection-to-answer window."
                        )

                evidence_weight = _require_optional_number(
                    response_metadata["evidence_weight"], f"{label} evidence weight"
                )
                if evidence_weight is None or not 0.0 <= evidence_weight <= 1.0:
                    raise ValidationError(f"{label} evidence weight is outside zero to one.")
                if (
                    evidence_weight != decision["evidence_weight"]
                    or projection_metadata["evidence_weight"] != evidence_weight
                ):
                    raise ValidationError(f"{label} evidence-weight snapshot mismatch.")
                if any(
                    value != release_id
                    for value in (
                        response_metadata["corpus_release_id"],
                        projection_payload["corpus_release_id"],
                        projection_metadata["corpus_release_id"],
                    )
                ):
                    raise ValidationError(f"{label} corpus-release snapshot mismatch.")
                if (
                    response_metadata["selection_learner_revision"]
                    != decision["learner_revision"]
                    or response_metadata["application_learner_revision"]
                    != expected_revision - 1
                    or decision["learner_revision"] != expected_revision - 1
                    or projection_payload["learner_revision"] != expected_revision
                ):
                    raise ValidationError(f"{label} learner revision sequence mismatch.")

                attempt_outcome = (
                    _strict_object(
                        attempt["outcome_json"],
                        f"Attempt {attempt['id']} outcome",
                    )
                    if attempt["outcome_json"] is not None
                    else None
                )
                if (
                    response_metadata["learner_model_version"]
                    in COMPLETE_TRANSITION_OUTCOME_MODEL_VERSIONS
                    and attempt_outcome is None
                ):
                    raise ValidationError(
                        f"{label} lacks its complete transition outcome."
                    )
                if attempt_outcome is not None:
                    expected_projection_values = {
                        "response_event_id": response["event_id"],
                        "state_changes": attempt_outcome.get("state_changes"),
                        "phase": attempt_outcome.get("next_phase"),
                        "focus_concept_id": attempt_outcome.get(
                            "focus_concept_id"
                        ),
                        "focus_misconception_id": attempt_outcome.get(
                            "focus_misconception_id"
                        ),
                        "corpus_release_id": release_id,
                        "learner_revision": expected_revision,
                    }
                    if projection["schema_version"] >= 2:
                        expected_projection_values.update(
                            {
                                "transition_reason": attempt_outcome.get(
                                    "transition_reason"
                                ),
                                "boundary_decision": attempt_outcome.get(
                                    "boundary_decision"
                                ),
                            }
                        )
                    if projection["schema_version"] >= 3:
                        expected_projection_values.update(
                            {
                                "question_objective_id": decision[
                                    "question_objective_id"
                                ],
                                "focus_objective_id": attempt_outcome.get(
                                    "focus_objective_id"
                                ),
                            }
                        )
                    if (
                        response_metadata["learner_model_version"]
                        in COMPLETE_TRANSITION_OUTCOME_MODEL_VERSIONS
                    ):
                        for field in (
                            "remediation_depth",
                            "remediation_path",
                        ):
                            if field not in attempt_outcome:
                                raise ValidationError(
                                    f"{label} complete transition outcome is "
                                    f"missing {field}."
                                )
                            expected_projection_values[field] = (
                                attempt_outcome[field]
                            )
                    for field, expected_value in (
                        expected_projection_values.items()
                    ):
                        if not same_json_value(
                            projection_payload.get(field), expected_value
                        ):
                            raise ValidationError(
                                f"{label} projection {field} does not match "
                                "its finalized attempt outcome."
                            )

                prior_family_attempts = family_attempts.get(question.family_id, 0)
                event_model = LearnerModel(
                    response_metadata["learner_model_version"]
                )
                _, changes = event_model.update_from_response(
                    connection,
                    learner_id=learner_id,
                    question=question,
                    selected_option=selected_option,
                    confidence=confidence,
                    hint_count=hint_count,
                    feedback_shown=feedback_shown,
                    evidence_weight_override=evidence_weight,
                    event_id=response["event_id"],
                    now=occurred_at,
                    response_ms=response_ms,
                    prior_family_attempts_override=prior_family_attempts,
                    misconception_algorithm=misconception_algorithm,
                )
                family_attempts[question.family_id] = prior_family_attempts + 1
                connection.execute(
                    "UPDATE learners SET revision=? WHERE id=?",
                    (expected_revision, learner_id),
                )
                hash_version = (
                    projection_payload["projection_hash_version"]
                    if projection["schema_version"] >= 3
                    else 1
                )
                actual_hash = work_database.learner_projection_hash(
                    learner_id,
                    connection,
                    hash_version=hash_version,
                )
                expected_hash = _require_sha256(
                    projection_payload["projection_hash"],
                    f"Projection event {projection['event_id']} hash",
                )
                state_changes_match = changes == projection_payload["state_changes"]
                hash_matches = actual_hash == expected_hash
                if not state_changes_match:
                    replay_errors.append(
                        f"projection revision {expected_revision}: state changes mismatch"
                    )
                if not hash_matches:
                    replay_errors.append(
                        f"projection revision {expected_revision}: hash mismatch"
                    )
                checkpoints.append(
                    {
                        "revision": expected_revision,
                        "response_event_id": response["event_id"],
                        "projection_event_id": projection["event_id"],
                        "question_id": question.id,
                        "family_id": question.family_id,
                        "state_changes": changes,
                        "state_changes_match": state_changes_match,
                        "expected_projection_hash": expected_hash,
                        "actual_projection_hash": actual_hash,
                        "hash_matches": hash_matches,
                    }
                )

            final_hash_version = (
                _strict_object(
                    pairs[-1][1]["payload_json"],
                    f"Projection event {pairs[-1][1]['event_id']} payload",
                )["projection_hash_version"]
                if pairs and pairs[-1][1]["schema_version"] >= 3
                else 1
            )
            reconstructed_hash = work_database.learner_projection_hash(
                learner_id,
                connection,
                hash_version=final_hash_version,
            )
        return {
            "checkpoints": checkpoints,
            "replay_errors": replay_errors,
            "reconstructed_projection_hash": reconstructed_hash,
            "learner_model_version": (
                _strict_object(
                    pairs[-1][0]["metadata_json"],
                    f"Response event {pairs[-1][0]['event_id']} metadata",
                )["learner_model_version"]
                if pairs
                else DEFAULT_LEARNER_MODEL_VERSION
            ),
        }

    def _run_on_copy(self, copy_path: Path, learner_id: str) -> dict[str, Any]:
        source_schema_version = self._backup_to(copy_path)
        work_database = Database(copy_path)
        missing_source_schema_guards: tuple[str, ...] = ()
        if source_schema_version < SCHEMA_VERSION:
            work_database.initialize()
        else:
            # The source is never normalized.  A current-version replay copy
            # may be missing a canonical immutability trigger precisely because
            # its mutable projection was corrupted.  Restore only absent guards
            # after exact structural validation, on this isolated copy alone.
            missing_source_schema_guards = (
                work_database._restore_missing_schema_guards_for_replay_copy()
            )
        with work_database.read() as connection:
            if not connection.execute(
                "SELECT 1 FROM learners WHERE id=?", (learner_id,)
            ).fetchone():
                raise NotFoundError(f"Unknown learner: {learner_id}")
            (
                committed_projection_hash,
                committed_hash_version,
            ) = self._projection_commitment(
                connection, learner_id
            )
            source_projection_hash_error: str | None = None
            try:
                source_projection_hash = work_database.learner_projection_hash(
                    learner_id,
                    connection,
                    hash_version=committed_hash_version,
                )
            except (
                ValidationError,
                sqlite3.DatabaseError,
                TypeError,
                ValueError,
            ) as exc:
                # A damaged mutable projection is precisely what replay can
                # diagnose and repair from immutable events.  Preserve the
                # failure in the report while allowing reconstruction to run.
                source_projection_hash = None
                source_projection_hash_error = str(exc)
            source_action_snapshot = self._action_projection_snapshot(connection)
            source_action_projection_hash = _projection_digest(
                source_action_snapshot
            )
            source_performance_snapshot = performance_projection_snapshot(connection)
            source_performance_projection_hash = _projection_digest(
                source_performance_snapshot
            )
            source_policy_shadow_snapshot = policy_shadow_projection_snapshot(
                connection
            )
            source_policy_shadow_projection_hash = _projection_digest(
                source_policy_shadow_snapshot
            )
            source_policy_shadow_event_count = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE event_type='PolicyShadowEvaluated'"""
            ).fetchone()["n"]
        source_integrity = work_database.verify_integrity()
        recoverable, blocking = self._recoverable_projection_errors(
            source_integrity["errors"], learner_id
        )
        source_schema_guard_errors = [
            "source database is missing canonical schema trigger " + trigger_name
            for trigger_name in missing_source_schema_guards
        ]
        recoverable = [*source_schema_guard_errors, *recoverable]
        replay = self._rebuild_projection(work_database, learner_id)
        action_replay = self._rebuild_action_projections(work_database)
        with work_database.transaction() as connection:
            performance_replay = rebuild_performance_projections(connection)
        policy_shadow_replay_error: str | None = None
        try:
            with work_database.transaction() as connection:
                policy_shadow_replay = rebuild_policy_shadow_projections(
                    connection
                )
        except (
            sqlite3.DatabaseError,
            TypeError,
            ValueError,
            ValidationError,
            OverflowError,
        ) as exc:
            # Semantic event corruption is not a repairable projection
            # mismatch.  Keep the check report bounded and explicit while
            # withholding a reconstructed projection.
            policy_shadow_replay_error = str(exc)
            policy_shadow_replay = {
                "snapshot": None,
                "checkpoints": [],
                "evaluation_count": None,
                "projection_hash": None,
            }
        rebuilt_integrity = work_database.verify_integrity()
        source_matches = (
            source_projection_hash == replay["reconstructed_projection_hash"]
        )
        action_source_matches = (
            source_action_snapshot == action_replay["snapshot"]
        )
        performance_source_matches = (
            source_performance_snapshot == performance_replay["snapshot"]
        )
        policy_shadow_source_matches = (
            policy_shadow_replay_error is None
            and source_policy_shadow_snapshot
            == policy_shadow_replay["snapshot"]
        )
        commitment_matches = (
            committed_projection_hash is None
            or committed_projection_hash == replay["reconstructed_projection_hash"]
        )
        errors = [
            *blocking,
            *source_schema_guard_errors,
            *replay["replay_errors"],
        ]
        if not source_matches:
            errors.append("stored learner projection differs from deterministic replay")
        if source_projection_hash_error is not None:
            errors.append(
                "stored learner projection cannot be hashed: "
                + source_projection_hash_error
            )
        if not commitment_matches:
            errors.append("latest projection commitment differs from deterministic replay")
        if not action_source_matches:
            errors.append(
                "stored learning-action projection differs from deterministic replay"
            )
        if not performance_source_matches:
            errors.append(
                "stored performance projection differs from deterministic replay"
            )
        if (
            policy_shadow_replay_error is None
            and not policy_shadow_source_matches
        ):
            errors.append(
                "stored policy-shadow projection differs from deterministic replay"
            )
        if policy_shadow_replay_error is not None:
            errors.append(
                "policy-shadow projection replay failed closed: "
                + policy_shadow_replay_error
            )
        if not rebuilt_integrity["ok"]:
            errors.extend(
                f"rebuilt copy integrity: {error}" for error in rebuilt_integrity["errors"]
            )
        rebuild_safe = (
            not blocking
            and not replay["replay_errors"]
            and policy_shadow_replay_error is None
            and commitment_matches
            and rebuilt_integrity["ok"]
        )
        return {
            "format_version": REPLAY_FORMAT_VERSION,
            "learner_id": learner_id,
            "learner_model_version": replay["learner_model_version"],
            "source_schema_version": source_schema_version,
            "replay_schema_version": SCHEMA_VERSION,
            "missing_source_schema_guards": list(
                missing_source_schema_guards
            ),
            "response_count": len(replay["checkpoints"]),
            "source_projection_hash": source_projection_hash,
            "source_projection_hash_error": source_projection_hash_error,
            "committed_projection_hash": committed_projection_hash,
            "reconstructed_projection_hash": replay[
                "reconstructed_projection_hash"
            ],
            "source_projection_matches_replay": source_matches,
            "commitment_matches_replay": commitment_matches,
            "action_event_count": action_replay["action_count"],
            "artifact_count": action_replay["artifact_count"],
            "source_action_count": len(source_action_snapshot["actions"]),
            "reconstructed_action_count": action_replay["action_count"],
            "source_artifact_count": len(source_action_snapshot["artifacts"]),
            "reconstructed_artifact_count": action_replay["artifact_count"],
            "source_action_projection_hash": source_action_projection_hash,
            "reconstructed_action_projection_hash": action_replay[
                "projection_hash"
            ],
            "action_projection_matches_replay": action_source_matches,
            "action_checkpoints": action_replay["checkpoints"],
            "performance_event_count": len(performance_replay["checkpoints"]),
            "source_performance_attempt_count": len(
                source_performance_snapshot["attempts"]
            ),
            "reconstructed_performance_attempt_count": performance_replay[
                "attempt_count"
            ],
            "source_performance_action_count": len(
                source_performance_snapshot["actions"]
            ),
            "reconstructed_performance_action_count": performance_replay[
                "action_count"
            ],
            "source_performance_scoring_claim_count": len(
                source_performance_snapshot["scoring_claims"]
            ),
            "reconstructed_performance_scoring_claim_count": performance_replay[
                "scoring_claim_count"
            ],
            "source_task_evaluation_count": len(
                source_performance_snapshot["evaluations"]
            ),
            "reconstructed_task_evaluation_count": performance_replay[
                "evaluation_count"
            ],
            "source_shadow_evidence_bundle_count": len(
                source_performance_snapshot["bundles"]
            ),
            "reconstructed_shadow_evidence_bundle_count": performance_replay[
                "bundle_count"
            ],
            "source_performance_projection_hash": (
                source_performance_projection_hash
            ),
            "reconstructed_performance_projection_hash": performance_replay[
                "projection_hash"
            ],
            "performance_projection_matches_replay": performance_source_matches,
            "performance_checkpoints": performance_replay["checkpoints"],
            "policy_shadow_event_count": source_policy_shadow_event_count,
            "source_policy_shadow_evaluation_count": len(
                source_policy_shadow_snapshot["evaluations"]
            ),
            "reconstructed_policy_shadow_evaluation_count": (
                policy_shadow_replay["evaluation_count"]
            ),
            "source_policy_shadow_projection_hash": (
                source_policy_shadow_projection_hash
            ),
            "reconstructed_policy_shadow_projection_hash": (
                policy_shadow_replay["projection_hash"]
            ),
            "policy_shadow_projection_matches_replay": (
                policy_shadow_source_matches
            ),
            "policy_shadow_checkpoints": policy_shadow_replay[
                "checkpoints"
            ],
            "policy_shadow_replay_error": policy_shadow_replay_error,
            "recoverable_source_integrity_errors": recoverable,
            "blocking_source_integrity_errors": blocking,
            "checkpoints": replay["checkpoints"],
            "rebuilt_integrity": rebuilt_integrity,
            "rebuild_safe": rebuild_safe,
            "errors": errors,
            "ok": not errors,
        }

    def check(self, learner_id: str) -> dict[str, Any]:
        if type(learner_id) is not str or not learner_id.strip():
            raise ValidationError("Learner ID must be a non-empty string.")
        with tempfile.TemporaryDirectory(prefix="tsq-replay-check-") as directory:
            copy_path = Path(directory) / "replay.db"
            report = self._run_on_copy(copy_path, learner_id)
        report["mode"] = "check"
        report["source_database"] = str(self.database.path)
        return report

    @staticmethod
    def _checkpoint_and_close(copy_path: Path) -> None:
        connection = sqlite3.connect(copy_path, timeout=20.0)
        try:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()
        with copy_path.open("rb") as handle:
            os.fsync(handle.fileno())

    def rebuild_copy(self, learner_id: str, target: str | Path) -> dict[str, Any]:
        if type(learner_id) is not str or not learner_id.strip():
            raise ValidationError("Learner ID must be a non-empty string.")
        target_path = Path(target)
        if target_path.resolve() == self.database.path.resolve():
            raise ConflictError("Replay target must not be the source database.")
        if target_path.exists():
            raise ConflictError(f"Replay target already exists: {target_path}")
        parent = target_path.parent
        if not parent.exists() or not parent.is_dir():
            raise ValidationError(f"Replay target directory does not exist: {parent}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target_path.name}.replay-", suffix=".tmp", dir=parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            report = self._run_on_copy(temporary_path, learner_id)
            if not report["rebuild_safe"]:
                raise ValidationError(
                    "Projection replay did not verify; no rebuilt copy was published: "
                    + "; ".join(report["errors"][:5])
                )
            self._checkpoint_and_close(temporary_path)
            try:
                os.link(temporary_path, target_path)
            except FileExistsError as exc:
                raise ConflictError(f"Replay target already exists: {target_path}") from exc
            temporary_path.unlink()
        finally:
            temporary_path.unlink(missing_ok=True)
            Path(f"{temporary_path}-wal").unlink(missing_ok=True)
            Path(f"{temporary_path}-shm").unlink(missing_ok=True)
        report["mode"] = "rebuild-copy"
        report["source_database"] = str(self.database.path)
        report["rebuilt_database"] = str(target_path)
        report["source_projection_was_repaired"] = not report[
            "source_projection_matches_replay"
        ]
        report["source_action_projection_was_repaired"] = not report[
            "action_projection_matches_replay"
        ]
        report["source_performance_projection_was_repaired"] = not report[
            "performance_projection_matches_replay"
        ]
        report["source_policy_shadow_projection_was_repaired"] = not report[
            "policy_shadow_projection_matches_replay"
        ]
        report["source_discrepancies"] = list(report["errors"])
        report["errors"] = []
        report["ok"] = True
        return report


def replay_or_error(operation: Any) -> dict[str, Any]:
    """Normalize unexpected SQLite failures at the CLI boundary."""

    try:
        return operation()
    except TSQError:
        raise
    except sqlite3.Error as exc:
        raise ValidationError(f"Projection replay failed safely: {exc}") from exc
