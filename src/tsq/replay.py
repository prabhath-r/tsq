# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import replace
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .errors import ConflictError, NotFoundError, TSQError, ValidationError
from .learner import MODEL_VERSION, LearnerModel
from .models import QuestionStatus
from .store import SCHEMA_VERSION, Database, question_content_hash


REPLAY_FORMAT_VERSION = 1
RESPONSE_EVENT_SCHEMA_VERSION = 1
PROJECTION_EVENT_SCHEMA_VERSIONS = frozenset({1, 2})

_RESPONSE_FIELDS = frozenset(
    {
        "decision_id",
        "question_id",
        "question_version",
        "selected_option_id",
        "is_correct",
        "confidence",
        "response_ms",
        "hint_count",
        "feedback_shown",
        "presented_order",
    }
)
_RESPONSE_METADATA_FIELDS = frozenset(
    {
        "policy_version",
        "learner_model_version",
        "corpus_release_id",
        "question_content_hash",
        "question_status",
        "evidence_weight",
        "selection_learner_revision",
        "application_learner_revision",
    }
)
_PROJECTION_FIELDS_V1 = frozenset(
    {
        "response_event_id",
        "state_changes",
        "phase",
        "focus_concept_id",
        "focus_misconception_id",
        "remediation_depth",
        "remediation_path",
        "corpus_release_id",
        "learner_revision",
        "projection_hash",
    }
)
_PROJECTION_FIELDS_V2 = _PROJECTION_FIELDS_V1 | frozenset(
    {"transition_reason", "boundary_decision"}
)
_PROJECTION_METADATA_FIELDS = frozenset(
    {"learner_model_version", "corpus_release_id", "evidence_weight"}
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


def _strict_object(raw: str, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(raw, parse_constant=reject_constant)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} contains invalid JSON: {exc}") from exc
    if type(value) is not dict:
        raise ValidationError(f"{label} must be a JSON object.")
    return value


def _strict_array(raw: str, label: str) -> list[Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(raw, parse_constant=reject_constant)
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
        self.learner_model = LearnerModel()

    def _validate_source(self) -> int:
        path = self.database.path
        if not path.exists() or not path.is_file():
            raise NotFoundError(f"Database does not exist: {path}")
        uri = f"file:{quote(str(path.resolve()))}?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=20.0)
        except sqlite3.Error as exc:
            raise ValidationError(f"Could not open database read-only: {exc}") from exc
        connection.row_factory = sqlite3.Row
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
        uri = f"file:{quote(str(self.database.path.resolve()))}?mode=ro"
        source = sqlite3.connect(uri, uri=True, timeout=20.0)
        destination_connection = sqlite3.connect(destination, timeout=20.0)
        try:
            source.execute("PRAGMA query_only = ON")
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
    ) -> str | None:
        row = connection.execute(
            """SELECT payload_json FROM events
               WHERE stream_id=? AND event_type='LearnerProjectionAdvanced'
               ORDER BY stream_version DESC LIMIT 1""",
            (f"learner:{learner_id}",),
        ).fetchone()
        if row is None:
            return None
        payload = _strict_object(row["payload_json"], "Latest projection event payload")
        return _require_sha256(
            payload.get("projection_hash"), "Latest projection event commitment"
        )

    @staticmethod
    def _recoverable_projection_errors(
        errors: list[str], learner_id: str
    ) -> tuple[list[str], list[str]]:
        recoverable_message = f"learner {learner_id}: projection hash mismatch"
        recoverable = [error for error in errors if error == recoverable_message]
        blocking = [error for error in errors if error != recoverable_message]
        return recoverable, blocking

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
        if response["schema_version"] != RESPONSE_EVENT_SCHEMA_VERSION:
            raise ValidationError(
                f"Event {response['event_id']} uses unsupported schema version "
                f"{response['schema_version']}; replay supports exactly version "
                f"{RESPONSE_EVENT_SCHEMA_VERSION} for ResponseSubmitted."
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
            _RESPONSE_METADATA_FIELDS,
            f"Response event {response['event_id']} metadata",
        )
        _require_exact_fields(
            projection_payload,
            (
                _PROJECTION_FIELDS_V2
                if projection["schema_version"] == 2
                else _PROJECTION_FIELDS_V1
            ),
            f"Projection event {projection['event_id']} payload",
        )
        _require_exact_fields(
            projection_metadata,
            _PROJECTION_METADATA_FIELDS,
            f"Projection event {projection['event_id']} metadata",
        )
        if projection_payload["response_event_id"] != response["event_id"]:
            raise ValidationError(
                f"Projection event {projection['event_id']} names a different response."
            )
        if projection["schema_version"] == 2:
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
        for label, metadata in (
            (f"Response event {response['event_id']}", response_metadata),
            (f"Projection event {projection['event_id']}", projection_metadata),
        ):
            if metadata["learner_model_version"] != MODEL_VERSION:
                raise ValidationError(
                    f"{label} uses learner model {metadata['learner_model_version']!r}; "
                    f"this binary replays exactly {MODEL_VERSION!r}."
                )
        return response_payload, response_metadata, projection_payload, projection_metadata

    def _rebuild_projection(
        self, work_database: Database, learner_id: str
    ) -> dict[str, Any]:
        checkpoints: list[dict[str, Any]] = []
        replay_errors: list[str] = []
        family_attempts: dict[str, int] = {}
        with work_database.transaction() as connection:
            learner = connection.execute(
                "SELECT * FROM learners WHERE id=?", (learner_id,)
            ).fetchone()
            if learner is None:
                raise NotFoundError(f"Unknown learner: {learner_id}")
            pairs = self._pair_events(connection, learner_id)
            connection.execute("DELETE FROM skill_states WHERE learner_id=?", (learner_id,))
            connection.execute(
                "DELETE FROM misconception_beliefs WHERE learner_id=?", (learner_id,)
            )
            connection.execute(
                "DELETE FROM learner_skill_families WHERE learner_id=?", (learner_id,)
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
                question_id = response_payload["question_id"]
                if type(question_id) is not str or not question_id:
                    raise ValidationError(f"{label} has an invalid question ID.")
                if decision["question_id"] != question_id or attempt["question_id"] != question_id:
                    raise ValidationError(f"{label} has inconsistent question projections.")
                question = work_database.get_question(question_id, connection)
                if (
                    response_payload["question_version"] != question.version
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
                release_id = decision["corpus_release_id"]
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

                prior_family_attempts = family_attempts.get(question.family_id, 0)
                _, changes = self.learner_model.update_from_response(
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
                )
                family_attempts[question.family_id] = prior_family_attempts + 1
                connection.execute(
                    "UPDATE learners SET revision=? WHERE id=?",
                    (expected_revision, learner_id),
                )
                actual_hash = work_database.learner_projection_hash(
                    learner_id, connection
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

            reconstructed_hash = work_database.learner_projection_hash(
                learner_id, connection
            )
        return {
            "checkpoints": checkpoints,
            "replay_errors": replay_errors,
            "reconstructed_projection_hash": reconstructed_hash,
        }

    def _run_on_copy(self, copy_path: Path, learner_id: str) -> dict[str, Any]:
        source_schema_version = self._backup_to(copy_path)
        work_database = Database(copy_path)
        work_database.initialize()
        with work_database.read() as connection:
            if not connection.execute(
                "SELECT 1 FROM learners WHERE id=?", (learner_id,)
            ).fetchone():
                raise NotFoundError(f"Unknown learner: {learner_id}")
            source_projection_hash = work_database.learner_projection_hash(
                learner_id, connection
            )
            committed_projection_hash = self._projection_commitment(
                connection, learner_id
            )
        source_integrity = work_database.verify_integrity()
        recoverable, blocking = self._recoverable_projection_errors(
            source_integrity["errors"], learner_id
        )
        replay = self._rebuild_projection(work_database, learner_id)
        rebuilt_integrity = work_database.verify_integrity()
        source_matches = (
            source_projection_hash == replay["reconstructed_projection_hash"]
        )
        commitment_matches = (
            committed_projection_hash is None
            or committed_projection_hash == replay["reconstructed_projection_hash"]
        )
        errors = [*blocking, *replay["replay_errors"]]
        if not source_matches:
            errors.append("stored learner projection differs from deterministic replay")
        if not commitment_matches:
            errors.append("latest projection commitment differs from deterministic replay")
        if not rebuilt_integrity["ok"]:
            errors.extend(
                f"rebuilt copy integrity: {error}" for error in rebuilt_integrity["errors"]
            )
        rebuild_safe = (
            not blocking
            and not replay["replay_errors"]
            and commitment_matches
            and rebuilt_integrity["ok"]
        )
        return {
            "format_version": REPLAY_FORMAT_VERSION,
            "learner_id": learner_id,
            "learner_model_version": MODEL_VERSION,
            "source_schema_version": source_schema_version,
            "replay_schema_version": SCHEMA_VERSION,
            "response_count": len(replay["checkpoints"]),
            "source_projection_hash": source_projection_hash,
            "committed_projection_hash": committed_projection_hash,
            "reconstructed_projection_hash": replay[
                "reconstructed_projection_hash"
            ],
            "source_projection_matches_replay": source_matches,
            "commitment_matches_replay": commitment_matches,
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
