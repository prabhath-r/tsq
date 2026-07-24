# SPDX-License-Identifier: MPL-2.0

"""Immutable, replayable shadow ledger for productive-skill tasks.

This module operationalizes the pure contracts in :mod:`tsq.evidence` without
promoting rubric observations into learner mastery.  Task releases are pinned
to an immutable curriculum release; attempts, semantic actions, evaluations,
and reduced bundles are committed to the learner event stream.  The ledger
stores digests and closed semantic payloads only.  It does not execute learner
artifacts, code, commands, tests, or model calls.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .errors import ConflictError, NotFoundError, ValidationError
from .evidence import (
    ActionKind,
    ActionPhase,
    LearningAction,
    LearningTask,
    ScorerContract,
    ScorerKind,
    TaskEvaluation,
    action_trace_digest,
    canonical_digest,
    canonical_json,
    reduce_evidence,
    summarize_actions,
)
from .performance import (
    ImportedEvaluation,
    NORMALIZED_SCORING_RESULT_SCHEMA_VERSION,
    NormalizationMode,
    NormalizedScoringResult,
    ProviderAuthorityBinding,
    RegisteredProvider,
    ScoringProviderRegistry,
    ScoringRequest,
    normalize_imported_evaluation,
)
from .store import (
    PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
    Database,
    from_timestamp,
    new_id,
    performance_scoring_claim_event_key,
    performance_scoring_claim_payload,
    to_timestamp,
)


TASK_RELEASE_SCHEMA_VERSION = 1
PERFORMANCE_EVENT_SCHEMA_VERSION = 1
TASK_STATUSES = frozenset({"quarantined", "pilot", "approved"})
SERVICEABLE_TASK_STATUSES = frozenset({"pilot", "approved"})
MAX_TASK_RELEASE_BYTES = 16 * 1024 * 1024

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_FIELDS = frozenset(
    {
        "reviewer_kind",
        "reviewer_id",
        "reviewed_at",
        "independent_of_author",
        "attestation_digest",
    }
)
_BUNDLE_FIELDS = frozenset(
    {"schema_version", "title", "corpus_release_id", "review", "tasks"}
)
_TASK_ENTRY_FIELDS = frozenset({"status", "task"})


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(
                f"Performance-task release contains duplicate field {key!r}."
            )
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValidationError(
        f"Performance-task release contains invalid number {value!r}."
    )


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise ValidationError(
            "Performance-task JSON contains a non-finite number."
        )
    return parsed


def _require_exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual == expected:
        return
    details: list[str] = []
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    raise ValidationError(f"{label} has " + "; ".join(details) + ".")


def _require_id(value: object, label: str) -> str:
    if type(value) is not str or not _ID_PATTERN.fullmatch(value):
        raise ValidationError(
            f"{label} must match {_ID_PATTERN.pattern!r}."
        )
    return value


def _require_digest(value: object, label: str) -> str:
    if type(value) is not str or not _DIGEST_PATTERN.fullmatch(value):
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_text(value: object, label: str, maximum: int) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValidationError(f"{label} must be a trimmed, non-blank string.")
    if len(value) > maximum:
        raise ValidationError(f"{label} must contain at most {maximum} characters.")
    return value


def _aware_timestamp(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise ValidationError(f"{label} must be an ISO-8601 timestamp string.")
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} is not a valid timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{label} must include a timezone offset.")
    return parsed.astimezone(timezone.utc)


def _now(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValidationError("now must be timezone-aware.")
    return resolved.astimezone(timezone.utc)


def _json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_json_float,
        )
    except ValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is not strict JSON: {exc}") from exc
    if type(value) is not dict:
        raise ValidationError(f"{label} must be a JSON object.")
    return value


@dataclass(frozen=True, slots=True)
class TaskReleaseReview:
    """Review commitment required before a task release can be sealed."""

    reviewer_kind: str
    reviewer_id: str
    reviewed_at: str
    independent_of_author: bool
    attestation_digest: str

    def __post_init__(self) -> None:
        if self.reviewer_kind != "human":
            raise ValidationError(
                "Task releases require a human review commitment; generated or "
                "model-only review cannot activate tasks."
            )
        _require_id(self.reviewer_id, "review.reviewer_id")
        _aware_timestamp(self.reviewed_at, "review.reviewed_at")
        if self.independent_of_author is not True:
            raise ValidationError(
                "Task release review must be independent of the task author."
            )
        _require_digest(
            self.attestation_digest, "review.attestation_digest"
        )

    def terms(self) -> dict[str, Any]:
        return {
            "reviewer_kind": self.reviewer_kind,
            "reviewer_id": self.reviewer_id,
            "reviewed_at": self.reviewed_at,
            "independent_of_author": self.independent_of_author,
            "attestation_digest": self.attestation_digest,
        }

    @classmethod
    def from_terms(cls, value: object) -> "TaskReleaseReview":
        if type(value) is not dict:
            raise ValidationError("review must be an object.")
        _require_exact_fields(value, _REVIEW_FIELDS, "review")
        return cls(
            reviewer_kind=value["reviewer_kind"],
            reviewer_id=value["reviewer_id"],
            reviewed_at=value["reviewed_at"],
            independent_of_author=value["independent_of_author"],
            attestation_digest=value["attestation_digest"],
        )


@dataclass(frozen=True, slots=True)
class PerformanceTaskRelease:
    """Canonical bundle of immutable productive-skill task definitions."""

    title: str
    corpus_release_id: str
    review: TaskReleaseReview
    tasks: tuple[tuple[str, LearningTask], ...]
    schema_version: int = TASK_RELEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.title, "title", 256)
        _require_id(self.corpus_release_id, "corpus_release_id")
        if not isinstance(self.review, TaskReleaseReview):
            raise ValidationError("review must be a TaskReleaseReview.")
        if type(self.tasks) is not tuple or not self.tasks:
            raise ValidationError("tasks must be a non-empty tuple.")
        normalized: list[tuple[str, LearningTask]] = []
        keys: set[tuple[str, int]] = set()
        for status, task in self.tasks:
            if status not in TASK_STATUSES:
                raise ValidationError(
                    "Task status must be quarantined, pilot, or approved."
                )
            if not isinstance(task, LearningTask):
                raise ValidationError("tasks must contain LearningTask values.")
            key = (task.id, task.version)
            if key in keys:
                raise ValidationError(
                    f"Task release repeats {task.id} version {task.version}."
                )
            keys.add(key)
            normalized.append((status, task))
        object.__setattr__(
            self,
            "tasks",
            tuple(sorted(normalized, key=lambda item: (item[1].id, item[1].version))),
        )
        if self.schema_version != TASK_RELEASE_SCHEMA_VERSION:
            raise ValidationError(
                f"schema_version must be {TASK_RELEASE_SCHEMA_VERSION}."
            )

    def terms(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "corpus_release_id": self.corpus_release_id,
            "review": self.review.terms(),
            "tasks": [
                {"status": status, "task": task.terms()}
                for status, task in self.tasks
            ],
        }

    @property
    def bundle_hash(self) -> str:
        return canonical_digest(self.terms())

    @property
    def release_id(self) -> str:
        return "ptrel_" + self.bundle_hash[:24]

    @classmethod
    def from_terms(cls, value: object) -> "PerformanceTaskRelease":
        if type(value) is not dict:
            raise ValidationError("Performance-task release must be an object.")
        _require_exact_fields(value, _BUNDLE_FIELDS, "Performance-task release")
        raw_tasks = value["tasks"]
        if type(raw_tasks) is not list:
            raise ValidationError("tasks must be an array.")
        tasks: list[tuple[str, LearningTask]] = []
        for index, entry in enumerate(raw_tasks):
            if type(entry) is not dict:
                raise ValidationError(f"tasks[{index}] must be an object.")
            _require_exact_fields(entry, _TASK_ENTRY_FIELDS, f"tasks[{index}]")
            try:
                task = LearningTask.from_terms(entry["task"])
            except ValueError as exc:
                raise ValidationError(
                    f"tasks[{index}].task is invalid: {exc}"
                ) from exc
            tasks.append((entry["status"], task))
        return cls(
            title=value["title"],
            corpus_release_id=value["corpus_release_id"],
            review=TaskReleaseReview.from_terms(value["review"]),
            tasks=tuple(tasks),
            schema_version=value["schema_version"],
        )


def read_task_release(path: str | Path) -> PerformanceTaskRelease:
    """Read one bounded UTF-8 task release with strict JSON semantics."""

    resolved = Path(path)
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ValidationError(f"Could not inspect task release {resolved}: {exc}") from exc
    if size > MAX_TASK_RELEASE_BYTES:
        raise ValidationError(
            f"Task release exceeds the {MAX_TASK_RELEASE_BYTES}-byte limit."
        )
    try:
        raw = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValidationError(f"Could not read task release {resolved}: {exc}") from exc
    return PerformanceTaskRelease.from_terms(
        _json_object(raw, "Performance-task release")
    )


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _command_hash(value: Mapping[str, Any]) -> str:
    return canonical_digest({"type": "tsq.performance_command", **value})


def _event_metadata(event: sqlite3.Row) -> dict[str, Any]:
    return _json_object(event["metadata_json"], f"Event {event['event_id']} metadata")


class PerformanceLedger:
    """Operational service for immutable, shadow-only performance evidence."""

    def __init__(self, database: Database):
        self.database = database

    def publish_release(
        self,
        bundle: PerformanceTaskRelease,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if not isinstance(bundle, PerformanceTaskRelease):
            raise ValidationError("bundle must be a PerformanceTaskRelease.")
        published_at = _now(now)
        reviewed_at = _aware_timestamp(
            bundle.review.reviewed_at, "review.reviewed_at"
        )
        if reviewed_at > published_at:
            raise ValidationError(
                "Task release cannot be published before its review."
            )
        timestamp = to_timestamp(published_at)
        bundle_json = canonical_json(bundle.terms())
        review_json = canonical_json(bundle.review.terms())
        bundle_size_bytes = len(bundle_json.encode("utf-8"))
        status_counts = {
            status: sum(item_status == status for item_status, _ in bundle.tasks)
            for status in sorted(TASK_STATUSES)
        }
        with self.database.transaction() as connection:
            corpus_release = connection.execute(
                "SELECT * FROM corpus_releases WHERE id=?",
                (bundle.corpus_release_id,),
            ).fetchone()
            if corpus_release is None:
                raise NotFoundError(
                    f"Corpus release {bundle.corpus_release_id} does not exist."
                )
            if corpus_release["sealed_at"] is None:
                raise ConflictError(
                    "Performance tasks can reference only a sealed corpus release."
                )
            corpus_sealed_at = _aware_timestamp(
                corpus_release["sealed_at"], "Corpus release seal boundary"
            )
            if published_at < corpus_sealed_at:
                raise ValidationError(
                    "Performance-task release cannot precede its corpus release."
                )
            prior = connection.execute(
                "SELECT * FROM performance_task_releases WHERE bundle_hash=?",
                (bundle.bundle_hash,),
            ).fetchone()
            if prior is not None:
                if prior["id"] != bundle.release_id:
                    raise ConflictError("Task release hash resolved to a conflicting ID.")
                return {
                    "release_id": prior["id"],
                    "bundle_hash": prior["bundle_hash"],
                    "bundle_size_bytes": bundle_size_bytes,
                    "corpus_release_id": prior["corpus_release_id"],
                    "task_count": len(bundle.tasks),
                    "status_counts": status_counts,
                    "idempotent_replay": True,
                }

            release_sources = {
                row["source_id"]: row["content_hash"]
                for row in connection.execute(
                    """SELECT membership.source_id, source.content_hash
                       FROM release_sources membership
                       JOIN sources source ON source.id=membership.source_id
                       WHERE membership.release_id=?""",
                    (bundle.corpus_release_id,),
                )
            }
            release_concepts = {
                row["concept_id"]
                for row in connection.execute(
                    "SELECT concept_id FROM release_concepts WHERE release_id=?",
                    (bundle.corpus_release_id,),
                )
            }
            release_objectives = {
                row["objective_id"]: row["primary_concept_id"]
                for row in connection.execute(
                    """SELECT membership.objective_id,
                              objective.primary_concept_id
                       FROM release_learning_objectives membership
                       JOIN learning_objectives objective
                         ON objective.id=membership.objective_id
                       WHERE membership.release_id=?""",
                    (bundle.corpus_release_id,),
                )
            }
            release_misconceptions = {
                row["misconception_id"]
                for row in connection.execute(
                    """SELECT misconception_id FROM release_misconceptions
                       WHERE release_id=?""",
                    (bundle.corpus_release_id,),
                )
            }
            release_misconception_concepts = {
                row["misconception_id"]: row["concept_id"]
                for row in connection.execute(
                    """SELECT membership.misconception_id,
                              misconception.concept_id
                       FROM release_misconceptions membership
                       JOIN misconceptions misconception
                         ON misconception.id=membership.misconception_id
                       WHERE membership.release_id=?""",
                    (bundle.corpus_release_id,),
                )
            }
            release_misconception_objectives: dict[str, set[str]] = {}
            for row in connection.execute(
                """SELECT DISTINCT option.misconception_id,
                                  mapping.objective_id
                   FROM release_option_objectives mapping
                   JOIN options option
                     ON option.question_id=mapping.question_id
                    AND option.option_id=mapping.option_id
                   WHERE mapping.release_id=?
                     AND option.misconception_id IS NOT NULL""",
                (bundle.corpus_release_id,),
            ):
                release_misconception_objectives.setdefault(
                    row["misconception_id"], set()
                ).add(row["objective_id"])
            for _, task in bundle.tasks:
                for source_id, provenance_digest in task.source_manifests:
                    if source_id not in release_sources:
                        raise ValidationError(
                            f"Task {task.id} cites source {source_id} outside its "
                            "pinned corpus release."
                        )
                    if release_sources[source_id] != provenance_digest:
                        raise ValidationError(
                            f"Task {task.id} source manifest for {source_id} does "
                            "not match the immutable source definition."
                        )
                unknown_concepts = set(task.concept_ids) - release_concepts
                if unknown_concepts:
                    raise ValidationError(
                        f"Task {task.id} references concepts outside its release: "
                        + ", ".join(sorted(unknown_concepts))
                    )
                for criterion in task.criteria:
                    for objective_id in criterion.objective_ids:
                        primary_concept_id = release_objectives.get(
                            objective_id
                        )
                        if primary_concept_id is None:
                            raise ValidationError(
                                f"Task {task.id} criterion {criterion.id} "
                                f"references objective {objective_id} outside "
                                "its release."
                            )
                        if primary_concept_id not in criterion.concept_ids:
                            raise ValidationError(
                                f"Task {task.id} criterion {criterion.id} "
                                f"objective {objective_id} has primary concept "
                                f"{primary_concept_id} outside that criterion's "
                                "concept mapping."
                            )
                unknown_misconceptions = (
                    set(task.misconception_ids) - release_misconceptions
                )
                if unknown_misconceptions:
                    raise ValidationError(
                        f"Task {task.id} references misconceptions outside its release: "
                        + ", ".join(sorted(unknown_misconceptions))
                    )
                for criterion in task.criteria:
                    criterion_objectives = set(criterion.objective_ids)
                    criterion_concepts = set(criterion.concept_ids)
                    for misconception_id in criterion.misconception_ids:
                        if criterion_objectives:
                            mapped_objectives = (
                                release_misconception_objectives.get(
                                    misconception_id, set()
                                )
                            )
                            if not criterion_objectives & mapped_objectives:
                                raise ValidationError(
                                    f"Task {task.id} criterion {criterion.id} "
                                    f"misconception {misconception_id} is not "
                                    "mapped to any of that criterion's "
                                    "objectives in the pinned release."
                                )
                        elif release_misconception_concepts.get(
                            misconception_id
                        ) not in criterion_concepts:
                            raise ValidationError(
                                f"Task {task.id} criterion {criterion.id} "
                                f"misconception {misconception_id} is outside "
                                "that criterion's concept mapping."
                            )
                definition_json = canonical_json(task.terms())
                existing = connection.execute(
                    """SELECT task_digest, definition_json
                       FROM performance_tasks
                       WHERE task_id=? AND task_version=?""",
                    (task.id, task.version),
                ).fetchone()
                if existing is not None and (
                    existing["task_digest"] != task.digest
                    or existing["definition_json"] != definition_json
                ):
                    raise ConflictError(
                        f"Performance task {task.id} version {task.version} is immutable; "
                        "publish a new version."
                    )
                connection.execute(
                    """INSERT OR IGNORE INTO performance_tasks(
                           task_id, task_version, task_digest,
                           definition_json, imported_at
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (task.id, task.version, task.digest, definition_json, timestamp),
                )

            connection.execute(
                """INSERT INTO performance_task_releases(
                       id, corpus_release_id, bundle_hash, title,
                       review_json, created_at, sealed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    bundle.release_id,
                    bundle.corpus_release_id,
                    bundle.bundle_hash,
                    bundle.title,
                    review_json,
                    timestamp,
                    timestamp,
                ),
            )
            connection.executemany(
                """INSERT INTO release_performance_tasks(
                       release_id, task_id, task_version, task_digest, status
                   ) VALUES (?, ?, ?, ?, ?)""",
                (
                    (
                        bundle.release_id,
                        task.id,
                        task.version,
                        task.digest,
                        status,
                    )
                    for status, task in bundle.tasks
                ),
            )
        return {
            "release_id": bundle.release_id,
            "bundle_hash": bundle.bundle_hash,
            "bundle_size_bytes": bundle_size_bytes,
            "corpus_release_id": bundle.corpus_release_id,
            "task_count": len(bundle.tasks),
            "status_counts": status_counts,
            "idempotent_replay": False,
        }

    def import_release(
        self, path: str | Path, *, now: datetime | None = None
    ) -> dict[str, Any]:
        return self.publish_release(read_task_release(path), now=now)

    def list_releases(self) -> list[dict[str, Any]]:
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT task_release.*,
                          COUNT(member.task_id) AS task_count,
                          SUM(member.status='approved') AS approved_count,
                          SUM(member.status='pilot') AS pilot_count,
                          SUM(member.status='quarantined') AS quarantined_count
                   FROM performance_task_releases task_release
                   LEFT JOIN release_performance_tasks member
                     ON member.release_id=task_release.id
                   GROUP BY task_release.id
                   ORDER BY task_release.created_at DESC, task_release.id"""
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = _row_dict(row)
            review_json = item.pop("review_json")
            item["review"] = _json_object(review_json, "Stored review")
            result.append(item)
        return result

    def list_tasks(
        self,
        *,
        release_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        if release_id is not None:
            _require_id(release_id, "release_id")
        if status is not None and status not in TASK_STATUSES:
            raise ValidationError("status must be quarantined, pilot, or approved.")
        clauses: list[str] = []
        parameters: list[Any] = []
        if release_id is not None:
            clauses.append("member.release_id=?")
            parameters.append(release_id)
        if status is not None:
            clauses.append("member.status=?")
            parameters.append(status)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT member.release_id, member.status,
                          task.task_id, task.task_version, task.task_digest,
                          json_extract(task.definition_json, '$.title') AS title,
                          json_extract(task.definition_json, '$.modality') AS modality,
                          task_release.corpus_release_id,
                          task_release.created_at
                   FROM release_performance_tasks member
                   JOIN performance_tasks task
                     ON task.task_id=member.task_id
                    AND task.task_version=member.task_version
                   JOIN performance_task_releases task_release
                     ON task_release.id=member.release_id"""
                + where
                + " ORDER BY task.task_id, task.task_version, member.release_id",
                tuple(parameters),
            ).fetchall()
        return [_row_dict(row) for row in rows]

    def show_task(
        self,
        task_id: str,
        *,
        task_version: int | None = None,
        release_id: str | None = None,
    ) -> dict[str, Any]:
        _require_id(task_id, "task_id")
        if type(task_version) not in {int, type(None)} or (
            task_version is not None and task_version < 1
        ):
            raise ValidationError("task_version must be a positive integer.")
        clauses = ["task.task_id=?"]
        parameters: list[Any] = [task_id]
        if task_version is not None:
            clauses.append("task.task_version=?")
            parameters.append(task_version)
        if release_id is not None:
            _require_id(release_id, "release_id")
            clauses.append("member.release_id=?")
            parameters.append(release_id)
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT task.*, member.release_id, member.status,
                          task_release.corpus_release_id
                   FROM performance_tasks task
                   JOIN release_performance_tasks member
                     ON member.task_id=task.task_id
                    AND member.task_version=task.task_version
                   JOIN performance_task_releases task_release
                     ON task_release.id=member.release_id
                   WHERE """
                + " AND ".join(clauses)
                + " ORDER BY task.task_version DESC, member.release_id",
                tuple(parameters),
            ).fetchall()
        if not rows:
            raise NotFoundError(f"Performance task {task_id} was not found.")
        if len(rows) != 1:
            raise ConflictError(
                "Task reference is ambiguous; specify both --version and --release."
            )
        row = rows[0]
        definition = _json_object(row["definition_json"], "Stored task definition")
        try:
            task = LearningTask.from_terms(definition)
        except ValueError as exc:
            raise ValidationError(f"Stored task definition is invalid: {exc}") from exc
        if task.digest != row["task_digest"]:
            raise ValidationError("Stored task digest does not match its definition.")
        return {
            "task": task.terms(),
            "task_digest": task.digest,
            "status": row["status"],
            "release_id": row["release_id"],
            "corpus_release_id": row["corpus_release_id"],
            "imported_at": row["imported_at"],
        }

    @staticmethod
    def _prior_command(
        connection: sqlite3.Connection,
        idempotency_key: str | None,
        event_type: str,
        command_hash: str,
    ) -> sqlite3.Row | None:
        if idempotency_key is None:
            return None
        _require_text(idempotency_key, "idempotency_key", 256)
        event = connection.execute(
            "SELECT * FROM events WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        if event is None:
            return None
        metadata = _event_metadata(event)
        if (
            event["event_type"] != event_type
            or metadata.get("command_hash") != command_hash
        ):
            raise ConflictError(
                "Idempotency key was already used for a different command."
            )
        return event

    def _prior_scoring_claim(
        self,
        connection: sqlite3.Connection,
        *,
        idempotency_key: str | None,
        command_hash: str,
    ) -> dict[str, Any] | None:
        """Resolve an immutable callback admission without invoking a provider."""

        key_claim = None
        if idempotency_key is not None:
            key_claim = connection.execute(
                """SELECT * FROM performance_scoring_claims
                   WHERE idempotency_key=?""",
                (idempotency_key,),
            ).fetchone()
            if (
                key_claim is not None
                and key_claim["command_hash"] != command_hash
            ):
                raise ConflictError(
                    "Idempotency key was already reserved for a different "
                    "scoring command."
                )
        command_claim = connection.execute(
            """SELECT * FROM performance_scoring_claims
               WHERE command_hash=?""",
            (command_hash,),
        ).fetchone()
        if command_claim is None:
            claim_event = connection.execute(
                "SELECT event_id FROM events WHERE idempotency_key=?",
                (performance_scoring_claim_event_key(command_hash),),
            ).fetchone()
            if claim_event is not None:
                raise ConflictError(
                    "Provider scoring for this logical command is committed "
                    "in event history but its claim projection is missing; "
                    "the provider callback will not be repeated. Verify "
                    "integrity and rebuild projections on a database copy."
                )
            return None
        claim_event = connection.execute(
            "SELECT event_id FROM events WHERE event_id=? AND idempotency_key=?",
            (
                command_claim["event_id"],
                performance_scoring_claim_event_key(command_hash),
            ),
        ).fetchone()
        if claim_event is None:
            raise ConflictError(
                "Provider scoring claim is missing its committed admission "
                "event; the provider callback will not be repeated. Verify "
                "integrity before reconciliation."
            )
        if command_claim["idempotency_key"] != idempotency_key:
            completion = connection.execute(
                "SELECT 1 FROM task_evaluations WHERE id=?",
                (command_claim["evaluation_id"],),
            ).fetchone()
            state = "completed" if completion is not None else "may be in progress"
            raise ConflictError(
                "This logical scoring command "
                f"{state} under a different idempotency key; the provider "
                "callback will not be repeated."
            )
        evaluation = connection.execute(
            "SELECT id FROM task_evaluations WHERE id=?",
            (command_claim["evaluation_id"],),
        ).fetchone()
        if evaluation is not None:
            return self._evaluation_report(
                connection, evaluation["id"], True
            )
        raise ConflictError(
            "Provider scoring for this logical command is already in progress "
            "or its callback was interrupted; the provider callback will not "
            "be repeated. Retry after the in-flight call completes; a "
            "persistent claim requires explicit provider reconciliation."
        )

    @staticmethod
    def _attempt_status(
        connection: sqlite3.Connection, attempt_id: str
    ) -> str:
        terminal = connection.execute(
            """SELECT action_type FROM performance_actions
               WHERE attempt_id=? AND action_type IN ('submitted', 'abandoned')
               ORDER BY sequence LIMIT 1""",
            (attempt_id,),
        ).fetchone()
        return terminal["action_type"] if terminal else "active"

    def start_attempt(
        self,
        session_id: str,
        task_id: str,
        *,
        task_version: int | None = None,
        task_release_id: str | None = None,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        _require_id(session_id, "session_id")
        _require_id(task_id, "task_id")
        if task_version is not None and (
            type(task_version) is not int or task_version < 1
        ):
            raise ValidationError("task_version must be a positive integer.")
        if task_release_id is not None:
            _require_id(task_release_id, "task_release_id")
        occurred = _now(now)
        command = {
            "operation": "start_attempt",
            "session_id": session_id,
            "task_id": task_id,
            "task_version": task_version,
            "task_release_id": task_release_id,
        }
        command_hash = _command_hash(command)
        with self.database.transaction() as connection:
            session = connection.execute(
                """SELECT session.*, learner.revision AS learner_revision
                   FROM sessions session
                   JOIN learners learner ON learner.id=session.learner_id
                   WHERE session.id=?""",
                (session_id,),
            ).fetchone()
            if session is None:
                raise NotFoundError(f"Session {session_id} does not exist.")
            self.database.require_learner_evidence_safe(
                session["learner_id"],
                connection,
            )
            prior = self._prior_command(
                connection,
                idempotency_key,
                "PerformanceTaskStarted",
                command_hash,
            )
            if prior is not None:
                payload = _json_object(prior["payload_json"], "Prior start event")
                return self._attempt_report(connection, payload["attempt_id"], True)
            if session["status"] != "active":
                raise ConflictError(f"Session {session_id} is {session['status']}.")
            session_boundaries = connection.execute(
                """SELECT event_type, occurred_at
                   FROM events
                   WHERE session_id=?
                     AND event_type IN ('SessionStarted', 'SessionEnded')
                   ORDER BY stream_version""",
                (session_id,),
            ).fetchall()
            start_boundaries = [
                row for row in session_boundaries
                if row["event_type"] == "SessionStarted"
            ]
            end_boundaries = [
                row for row in session_boundaries
                if row["event_type"] == "SessionEnded"
            ]
            if len(start_boundaries) != 1 or end_boundaries:
                raise ValidationError(
                    "Session event history does not contain one active start boundary."
                )
            session_started_at = _aware_timestamp(
                start_boundaries[0]["occurred_at"],
                f"Session {session_id} start boundary",
            )
            if occurred < session_started_at:
                raise ValidationError(
                    "Performance task cannot start before its session."
                )
            if connection.execute(
                """SELECT 1 FROM decisions
                   WHERE session_id=? AND consumed_at IS NULL
                     AND invalidated_at IS NULL LIMIT 1""",
                (session_id,),
            ).fetchone() is not None:
                raise ConflictError(
                    "Answer or invalidate the pending question before starting "
                    "a performance task."
                )
            open_attempt = connection.execute(
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
            if open_attempt is not None:
                raise ConflictError(
                    f"Session already has active performance task {open_attempt['id']}."
                )
            clauses = [
                "member.task_id=?",
                "task_release.corpus_release_id=?",
                "member.status IN ('pilot', 'approved')",
            ]
            parameters: list[Any] = [task_id, session["corpus_release_id"]]
            if task_version is not None:
                clauses.append("member.task_version=?")
                parameters.append(task_version)
            if task_release_id is not None:
                clauses.append("member.release_id=?")
                parameters.append(task_release_id)
            candidates = connection.execute(
                """SELECT member.*, task.definition_json,
                          task_release.corpus_release_id,
                          task_release.created_at AS task_release_created_at
                   FROM release_performance_tasks member
                   JOIN performance_task_releases task_release
                     ON task_release.id=member.release_id
                   JOIN performance_tasks task
                     ON task.task_id=member.task_id
                    AND task.task_version=member.task_version
                   WHERE """
                + " AND ".join(clauses)
                + " ORDER BY member.task_version DESC, member.release_id",
                tuple(parameters),
            ).fetchall()
            if not candidates:
                raise NotFoundError(
                    f"No serviceable release contains task {task_id} for this session."
                )
            if len(candidates) != 1:
                raise ConflictError(
                    "Task reference is ambiguous; specify task version and release."
                )
            member = candidates[0]
            task_release_created_at = _aware_timestamp(
                member["task_release_created_at"],
                "Performance-task release publication",
            )
            if occurred < task_release_created_at:
                raise ValidationError(
                    "Performance task cannot start before its release."
                )
            try:
                task = LearningTask.from_terms(
                    _json_object(
                        member["definition_json"], "Stored task definition"
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Stored task definition is invalid: {exc}"
                ) from exc
            if ActionKind.STARTED not in task.allowed_action_kinds:
                raise ValidationError(
                    f"Task {task.id} does not allow its required started action."
                )
            attempt_id = new_id("pta")
            payload = {
                "attempt_id": attempt_id,
                "session_id": session_id,
                "learner_id": session["learner_id"],
                "task_release_id": member["release_id"],
                "corpus_release_id": session["corpus_release_id"],
                "task_id": member["task_id"],
                "task_version": member["task_version"],
                "task_digest": member["task_digest"],
                "session_revision": session["revision"],
                "learner_revision": session["learner_revision"],
            }
            metadata = {
                "command_hash": command_hash,
                "task_schema_version": _json_object(
                    member["definition_json"], "Stored task definition"
                )["schema_version"],
                "shadow_only": True,
                "projection_applied": False,
                "certification_applied": False,
            }
            event = self.database.append_event(
                connection,
                stream_id=f"learner:{session['learner_id']}",
                event_type="PerformanceTaskStarted",
                schema_version=PERFORMANCE_EVENT_SCHEMA_VERSION,
                payload=payload,
                metadata=metadata,
                learner_id=session["learner_id"],
                session_id=session_id,
                idempotency_key=idempotency_key,
                correlation_id=attempt_id,
                causation_id=session_id,
                occurred_at=occurred,
            )
            connection.execute(
                """INSERT INTO performance_attempts(
                       id, event_id, task_release_id, corpus_release_id,
                       task_id, task_version, task_digest, session_id, learner_id,
                       session_revision, learner_revision, started_at,
                       recorded_at, command_hash
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    attempt_id,
                    event["event_id"],
                    member["release_id"],
                    session["corpus_release_id"],
                    member["task_id"],
                    member["task_version"],
                    member["task_digest"],
                    session_id,
                    session["learner_id"],
                    session["revision"],
                    session["learner_revision"],
                    to_timestamp(occurred),
                    event["recorded_at"],
                    command_hash,
                ),
            )
            self._append_action(
                connection,
                connection.execute(
                    "SELECT * FROM performance_attempts WHERE id=?", (attempt_id,)
                ).fetchone(),
                ActionKind.STARTED,
                ActionPhase.UNASSISTED,
                {},
                occurred=occurred,
                idempotency_key=None,
                command_hash=_command_hash(
                    {"operation": "automatic_start", "attempt_id": attempt_id}
                ),
            )
            return self._attempt_report(connection, attempt_id, False)

    def _append_action(
        self,
        connection: sqlite3.Connection,
        attempt: sqlite3.Row,
        kind: ActionKind,
        phase: ActionPhase,
        payload: Mapping[str, Any],
        *,
        occurred: datetime,
        idempotency_key: str | None,
        command_hash: str,
    ) -> LearningAction:
        previous = connection.execute(
            """SELECT MAX(sequence) AS sequence FROM performance_actions
               WHERE attempt_id=?""",
            (attempt["id"],),
        ).fetchone()
        sequence = 0 if previous["sequence"] is None else previous["sequence"] + 1
        started = from_timestamp(attempt["started_at"])
        if started is None or occurred < started:
            raise ValidationError("Performance action cannot precede task start.")
        elapsed_ms = int((occurred - started).total_seconds() * 1000)
        existing_actions = self._typed_actions(connection, attempt["id"])
        if (
            phase is ActionPhase.POST_FEEDBACK
            and not any(
                item.kind is ActionKind.SUBMITTED for item in existing_actions
            )
        ):
            raise ValidationError(
                "Post-feedback actions require a submitted checkpoint."
            )
        try:
            action = LearningAction(
                id=new_id("pact"),
                trace_id=attempt["id"],
                sequence=sequence,
                kind=kind,
                phase=phase,
                payload=dict(payload),
                elapsed_ms=elapsed_ms,
            )
        except ValueError as exc:
            raise ValidationError(f"Invalid performance action: {exc}") from exc
        try:
            summarize_actions((*existing_actions, action))
        except ValueError as exc:
            raise ValidationError(
                f"Performance action violates trace lifecycle: {exc}"
            ) from exc
        event = self.database.append_event(
            connection,
            stream_id=f"learner:{attempt['learner_id']}",
            event_type="PerformanceActionRecorded",
            schema_version=PERFORMANCE_EVENT_SCHEMA_VERSION,
            payload={"attempt_id": attempt["id"], "action": action.terms()},
            metadata={
                "command_hash": command_hash,
                "action_schema_version": action.schema_version,
                "task_digest": attempt["task_digest"],
                "task_release_id": attempt["task_release_id"],
                "corpus_release_id": attempt["corpus_release_id"],
                "observational_only": True,
                "shadow_only": True,
            },
            learner_id=attempt["learner_id"],
            session_id=attempt["session_id"],
            idempotency_key=idempotency_key,
            correlation_id=attempt["id"],
            causation_id=attempt["id"],
            occurred_at=occurred,
        )
        terms = action.terms()
        connection.execute(
            """INSERT INTO performance_actions(
                   id, event_id, attempt_id, sequence, phase, action_type,
                   payload_json, elapsed_ms, occurred_at, recorded_at, command_hash
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                action.id,
                event["event_id"],
                attempt["id"],
                action.sequence,
                action.phase.value,
                action.kind.value,
                canonical_json(terms["payload"]),
                action.elapsed_ms,
                to_timestamp(occurred),
                event["recorded_at"],
                command_hash,
            ),
        )
        return action

    def record_action(
        self,
        attempt_id: str,
        action_type: str,
        payload: Mapping[str, Any],
        *,
        phase: str = "unassisted",
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        _require_id(attempt_id, "attempt_id")
        try:
            kind = ActionKind(action_type)
            typed_phase = ActionPhase(phase)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Unknown performance action type or phase.") from exc
        if kind is ActionKind.STARTED:
            raise ValidationError("started is recorded automatically.")
        if not isinstance(payload, Mapping):
            raise ValidationError("payload must be an object.")
        try:
            command_hash = _command_hash(
                {
                    "operation": "record_action",
                    "attempt_id": attempt_id,
                    "action_type": kind.value,
                    "phase": typed_phase.value,
                    "payload": dict(payload),
                }
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Performance action payload is not finite JSON: {exc}"
            ) from exc
        occurred = _now(now)
        with self.database.transaction() as connection:
            attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise NotFoundError(
                    f"Performance attempt {attempt_id} does not exist."
                )
            self.database.require_learner_evidence_safe(
                attempt["learner_id"],
                connection,
            )
            prior = self._prior_command(
                connection,
                idempotency_key,
                "PerformanceActionRecorded",
                command_hash,
            )
            if prior is not None:
                row = connection.execute(
                    "SELECT * FROM performance_actions WHERE event_id=?",
                    (prior["event_id"],),
                ).fetchone()
                if row is None:
                    raise ValidationError("Prior action event lacks its projection.")
                return {**self._action_view(row), "idempotent_replay": True}
            session_status = connection.execute(
                "SELECT status FROM sessions WHERE id=?",
                (attempt["session_id"],),
            ).fetchone()
            if session_status is None:
                raise ValidationError("Performance attempt has no session.")
            if session_status["status"] != "active":
                raise ConflictError(
                    f"Session {attempt['session_id']} is "
                    f"{session_status['status']}."
                )
            task_row = connection.execute(
                """SELECT definition_json FROM performance_tasks
                   WHERE task_id=? AND task_version=?""",
                (attempt["task_id"], attempt["task_version"]),
            ).fetchone()
            if task_row is None:
                raise ValidationError("Performance attempt has no task definition.")
            try:
                task = LearningTask.from_terms(
                    _json_object(task_row["definition_json"], "Stored task definition")
                )
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"Stored task definition is invalid: {exc}") from exc
            if kind not in task.allowed_action_kinds:
                raise ValidationError(
                    f"Task {task.id} does not allow {kind.value} actions."
                )
            action = self._append_action(
                connection,
                attempt,
                kind,
                typed_phase,
                payload,
                occurred=occurred,
                idempotency_key=idempotency_key,
                command_hash=command_hash,
            )
            row = connection.execute(
                "SELECT * FROM performance_actions WHERE id=?", (action.id,)
            ).fetchone()
            return {**self._action_view(row), "idempotent_replay": False}

    @staticmethod
    def _action_view(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "event_id": row["event_id"],
            "attempt_id": row["attempt_id"],
            "sequence": row["sequence"],
            "phase": row["phase"],
            "action_type": row["action_type"],
            "payload": _json_object(row["payload_json"], "Stored action payload"),
            "elapsed_ms": row["elapsed_ms"],
            "occurred_at": row["occurred_at"],
            "recorded_at": row["recorded_at"],
        }

    def list_actions(self, attempt_id: str) -> list[dict[str, Any]]:
        _require_id(attempt_id, "attempt_id")
        with self.database.read() as connection:
            if connection.execute(
                "SELECT 1 FROM performance_attempts WHERE id=?", (attempt_id,)
            ).fetchone() is None:
                raise NotFoundError(f"Performance attempt {attempt_id} does not exist.")
            rows = connection.execute(
                """SELECT * FROM performance_actions
                   WHERE attempt_id=? ORDER BY sequence""",
                (attempt_id,),
            ).fetchall()
        return [self._action_view(row) for row in rows]

    def list_scoring_claims(
        self,
        *,
        attempt_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List durable callback admissions without reconciling any claim."""

        if attempt_id is not None:
            _require_id(attempt_id, "attempt_id")
        if status not in {None, "completed", "unresolved"}:
            raise ValidationError(
                "Scoring claim status must be completed or unresolved."
            )
        clauses: list[str] = []
        parameters: list[Any] = []
        if attempt_id is not None:
            clauses.append("claim.attempt_id=?")
            parameters.append(attempt_id)
        if status == "completed":
            clauses.append("evaluation.id IS NOT NULL")
        elif status == "unresolved":
            clauses.append("evaluation.id IS NULL")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.database.read() as connection:
            if attempt_id is not None and connection.execute(
                "SELECT 1 FROM performance_attempts WHERE id=?",
                (attempt_id,),
            ).fetchone() is None:
                raise NotFoundError(
                    f"Performance attempt {attempt_id} does not exist."
                )
            orphan_parameters: tuple[Any, ...] = (
                (attempt_id,)
                if attempt_id is not None
                else ()
            )
            orphan_attempt_clause = (
                " AND json_extract(event.payload_json, '$.attempt_id')=?"
                if attempt_id is not None
                else ""
            )
            orphan = connection.execute(
                """SELECT event.event_id
                   FROM events event
                   WHERE event.event_type IN (
                       'PerformanceScoringClaimed',
                       'PerformanceScoringClaimMigrated'
                   )
                     AND NOT EXISTS (
                         SELECT 1 FROM performance_scoring_claims claim
                         WHERE claim.event_id=event.event_id
                     )"""
                + orphan_attempt_clause
                + " ORDER BY event.event_id LIMIT 1",
                orphan_parameters,
            ).fetchone()
            if orphan is not None:
                raise ConflictError(
                    f"Scoring admission event {orphan['event_id']} is missing "
                    "its claim projection; run integrity verification and "
                    "rebuild on a database copy."
                )
            rows = connection.execute(
                """SELECT claim.*, claim_event.event_type,
                          claim_event.metadata_json,
                          evaluation.recorded_at AS completed_at
                   FROM performance_scoring_claims claim
                   LEFT JOIN events claim_event
                     ON claim_event.event_id=claim.event_id
                   LEFT JOIN task_evaluations evaluation
                     ON evaluation.id=claim.evaluation_id"""
                + where
                + " ORDER BY claim.claimed_at, claim.id",
                tuple(parameters),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            if row["event_type"] is None or row["metadata_json"] is None:
                raise ConflictError(
                    f"Scoring claim {row['id']} is missing its admission event; "
                    "run integrity verification."
                )
            metadata = _json_object(
                row["metadata_json"], "Stored scoring claim event metadata"
            )
            completed = row["completed_at"] is not None
            result.append(
                {
                    "id": row["id"],
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "admission_mode": metadata.get("admission_mode"),
                    "attempt_id": row["attempt_id"],
                    "evaluation_id": row["evaluation_id"],
                    "through_sequence": row["through_sequence"],
                    "provider_id": row["provider_id"],
                    "provider_version": row["provider_version"],
                    "action_trace_digest": row["action_trace_digest"],
                    "command_hash": row["command_hash"],
                    "claimed_at": row["claimed_at"],
                    "status": "completed" if completed else "unresolved",
                    "completed_at": row["completed_at"],
                    "caller_idempotency_key_present": (
                        row["idempotency_key"] is not None
                    ),
                    "automatic_retry_allowed": False,
                }
            )
        return result

    @staticmethod
    def _typed_actions(
        connection: sqlite3.Connection,
        attempt_id: str,
        *,
        through_sequence: int | None = None,
    ) -> tuple[LearningAction, ...]:
        clause = " AND sequence<=?" if through_sequence is not None else ""
        parameters: tuple[Any, ...] = (
            (attempt_id, through_sequence)
            if through_sequence is not None
            else (attempt_id,)
        )
        rows = connection.execute(
            """SELECT * FROM performance_actions WHERE attempt_id=?"""
            + clause
            + " ORDER BY sequence",
            parameters,
        ).fetchall()
        actions: list[LearningAction] = []
        for row in rows:
            try:
                actions.append(
                    LearningAction.from_terms(
                        {
                            "id": row["id"],
                            "trace_id": row["attempt_id"],
                            "sequence": row["sequence"],
                            "kind": row["action_type"],
                            "phase": row["phase"],
                            "payload": _json_object(
                                row["payload_json"], "Stored action payload"
                            ),
                            "elapsed_ms": row["elapsed_ms"],
                            "schema_version": 1,
                        }
                    )
                )
            except ValueError as exc:
                raise ValidationError(f"Stored performance action is invalid: {exc}") from exc
        return tuple(actions)

    @staticmethod
    def _task_for_attempt(
        connection: sqlite3.Connection, attempt: sqlite3.Row
    ) -> LearningTask:
        row = connection.execute(
            """SELECT * FROM performance_tasks
               WHERE task_id=? AND task_version=?""",
            (attempt["task_id"], attempt["task_version"]),
        ).fetchone()
        if row is None:
            raise ValidationError("Performance attempt has no task definition.")
        try:
            task = LearningTask.from_terms(
                _json_object(row["definition_json"], "Stored task definition")
            )
        except ValueError as exc:
            raise ValidationError(f"Stored task definition is invalid: {exc}") from exc
        if task.digest != attempt["task_digest"]:
            raise ValidationError("Performance attempt task digest mismatch.")
        return task

    @staticmethod
    def _submission_boundary(
        connection: sqlite3.Connection, attempt_id: str
    ) -> int:
        rows = connection.execute(
            """SELECT sequence FROM performance_actions
               WHERE attempt_id=? AND action_type='submitted'
               ORDER BY sequence""",
            (attempt_id,),
        ).fetchall()
        if len(rows) != 1:
            raise ConflictError(
                "Performance attempt must have exactly one submitted checkpoint before scoring."
            )
        return rows[0]["sequence"]

    def _require_active_session_interval(
        self,
        connection: sqlite3.Connection,
        attempt: sqlite3.Row,
        occurred: datetime,
    ) -> None:
        """Keep every new scoring event inside its immutable session interval."""

        session = connection.execute(
            "SELECT status FROM sessions WHERE id=?",
            (attempt["session_id"],),
        ).fetchone()
        if session is None:
            raise ValidationError("Performance attempt has no session.")
        self.database.require_learner_evidence_safe(
            attempt["learner_id"],
            connection,
        )
        boundaries = connection.execute(
            """SELECT event_type, occurred_at
               FROM events
               WHERE session_id=?
                 AND event_type IN ('SessionStarted', 'SessionEnded')
               ORDER BY stream_version""",
            (attempt["session_id"],),
        ).fetchall()
        starts = [row for row in boundaries if row["event_type"] == "SessionStarted"]
        ends = [row for row in boundaries if row["event_type"] == "SessionEnded"]
        if len(starts) != 1 or ends or session["status"] != "active":
            raise ConflictError(
                f"Session {attempt['session_id']} is not active for new task scoring."
            )
        started_at = _aware_timestamp(
            starts[0]["occurred_at"],
            f"Session {attempt['session_id']} start boundary",
        )
        if occurred < started_at:
            raise ValidationError(
                "Task scoring cannot occur before its session start boundary."
            )

    @staticmethod
    def _validate_authority_binding(
        task: LearningTask, result: NormalizedScoringResult
    ) -> None:
        contracts = {contract.key: contract for contract in task.scorer_contracts}
        for decision in result.decisions:
            if decision.effective_kind not in {
                ScorerKind.DETERMINISTIC,
                ScorerKind.HUMAN,
            }:
                continue
            key = (
                decision.effective_kind,
                result.provider.provider_id,
                result.provider.provider_version,
            )
            contract = contracts.get(key)
            if contract is None:
                raise ValidationError(
                    f"Provider lacks a task scorer contract for {decision.criterion_id}."
                )
            if (
                contract.authority_id != result.provider.authority_id
                or contract.authority_manifest_digest
                != result.provider.authority_manifest_digest
                or contract.check_set_manifests
                != result.provider.check_set_manifests
                or contract.artifact_manifests
                != result.provider.artifact_manifests
                or decision.criterion_id not in contract.criterion_ids
            ):
                raise ValidationError(
                    f"Provider authority does not match the released scorer contract "
                    f"for {decision.criterion_id}."
                )

    def _record_result(
        self,
        connection: sqlite3.Connection,
        attempt: sqlite3.Row,
        task: LearningTask,
        actions: tuple[LearningAction, ...],
        through_sequence: int,
        result: NormalizedScoringResult,
        *,
        idempotency_key: str | None,
        occurred: datetime,
        command_hash: str,
        claim_event_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_active_session_interval(
            connection,
            attempt,
            occurred,
        )
        evaluation = result.evaluation
        if (
            evaluation.trace_id != attempt["id"]
            or evaluation.task_id != task.id
            or evaluation.task_version != task.version
            or evaluation.task_digest != task.digest
            or evaluation.action_trace_digest != action_trace_digest(actions)
        ):
            raise ValidationError(
                "Scoring result is not pinned to the attempt task and submitted trace."
            )
        submission = connection.execute(
            """SELECT occurred_at FROM performance_actions
               WHERE attempt_id=? AND sequence=? AND action_type='submitted'""",
            (attempt["id"], through_sequence),
        ).fetchone()
        if submission is None:
            raise ConflictError(
                "Scoring requires its exact submitted trace boundary."
            )
        submitted_at = _aware_timestamp(
            submission["occurred_at"], "Submitted performance checkpoint"
        )
        if occurred < submitted_at:
            raise ValidationError(
                "Task evaluation cannot occur before its submitted checkpoint."
            )
        self._validate_authority_binding(task, result)
        try:
            bundle = reduce_evidence(task, evaluation, actions)
        except ValueError as exc:
            raise ValidationError(f"Evidence reduction failed closed: {exc}") from exc
        # Persist the complete normalized authority envelope.  Keeping only
        # TaskEvaluation would discard whether the score came through a direct
        # import, which task scorer contract was requested, and which provider
        # manifest the registry actually bound.
        authority = {
            "normalized_result": result.terms(),
            "normalized_result_digest": result.digest,
        }
        event = self.database.append_event(
            connection,
            stream_id=f"learner:{attempt['learner_id']}",
            event_type="TaskEvaluationRecorded",
            schema_version=PERFORMANCE_EVENT_SCHEMA_VERSION,
            payload={
                "attempt_id": attempt["id"],
                "through_sequence": through_sequence,
                "evaluation_digest": evaluation.digest,
                "evaluation": evaluation.terms(),
                "authority": authority,
            },
            metadata={
                "command_hash": command_hash,
                "task_release_id": attempt["task_release_id"],
                "corpus_release_id": attempt["corpus_release_id"],
                "shadow_only": True,
                "projection_applied": False,
                "certification_applied": False,
            },
            learner_id=attempt["learner_id"],
            session_id=attempt["session_id"],
            idempotency_key=idempotency_key,
            correlation_id=attempt["id"],
            causation_id=claim_event_id or attempt["id"],
            occurred_at=occurred,
        )
        connection.execute(
            """INSERT INTO task_evaluations(
                   id, event_id, attempt_id, through_sequence,
                   evaluation_digest, evaluation_json, authority_json,
                   recorded_at, command_hash
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                evaluation.id,
                event["event_id"],
                attempt["id"],
                through_sequence,
                evaluation.digest,
                canonical_json(evaluation.terms()),
                canonical_json(authority),
                event["recorded_at"],
                command_hash,
            ),
        )
        bundle_terms = bundle.terms()
        bundle_digest = canonical_digest(bundle_terms)
        bundle_id = "seb_" + bundle_digest[:24]
        bundle_event = self.database.append_event(
            connection,
            stream_id=f"learner:{attempt['learner_id']}",
            event_type="ShadowEvidenceReduced",
            schema_version=PERFORMANCE_EVENT_SCHEMA_VERSION,
            payload={
                "bundle_id": bundle_id,
                "evaluation_id": evaluation.id,
                "attempt_id": attempt["id"],
                "bundle_digest": bundle_digest,
                "bundle": bundle_terms,
                "projection_applied": False,
                "certification_applied": False,
            },
            metadata={
                "reducer": "deterministic-evidence-v2",
                "task_release_id": attempt["task_release_id"],
                "corpus_release_id": attempt["corpus_release_id"],
                "shadow_only": True,
            },
            learner_id=attempt["learner_id"],
            session_id=attempt["session_id"],
            correlation_id=attempt["id"],
            causation_id=evaluation.id,
            occurred_at=occurred,
        )
        connection.execute(
            """INSERT INTO shadow_evidence_bundles(
                   id, event_id, evaluation_id, attempt_id,
                   bundle_digest, bundle_json, projection_applied,
                   certification_applied, recorded_at
               ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?)""",
            (
                bundle_id,
                bundle_event["event_id"],
                evaluation.id,
                attempt["id"],
                bundle_digest,
                canonical_json(bundle_terms),
                bundle_event["recorded_at"],
            ),
        )
        return {
            "attempt_id": attempt["id"],
            "evaluation": evaluation.terms(),
            "evaluation_digest": evaluation.digest,
            "authority": authority,
            "shadow_evidence": bundle_terms,
            "bundle_digest": bundle_digest,
            "projection_applied": False,
            "certification_applied": False,
            "idempotent_replay": False,
        }

    def score_attempt(
        self,
        attempt_id: str,
        registry: ScoringProviderRegistry,
        provider_id: str,
        provider_version: str,
        *,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        _require_id(attempt_id, "attempt_id")
        if (
            idempotency_key is not None
            and idempotency_key.startswith(
                PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX
            )
        ):
            raise ValidationError(
                "Scoring idempotency keys cannot use TSQ's reserved "
                "callback-admission namespace."
            )
        if not isinstance(registry, ScoringProviderRegistry):
            raise ValidationError("registry must be a ScoringProviderRegistry.")
        if self.database.read_only:
            # Do not invoke an adapter when the command can never be committed.
            # Providers are currently content-free importers, but this boundary
            # also keeps a future remote adapter from performing needless work.
            raise ConflictError(
                "A read-only database cannot start a write transaction."
            )
        occurred = _now(now)
        # Assemble a complete immutable scoring request without holding
        # SQLite's writer lock. Provider latency and failure must not block
        # unrelated learners. The exact submitted boundary is re-read and
        # compared inside the eventual write transaction before any result is
        # accepted.
        with self.database.read() as connection:
            attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise NotFoundError(f"Performance attempt {attempt_id} does not exist.")
            self.database.require_learner_evidence_safe(
                attempt["learner_id"],
                connection,
            )
            through_sequence = self._submission_boundary(connection, attempt_id)
            task = self._task_for_attempt(connection, attempt)
            actions = self._typed_actions(
                connection, attempt_id, through_sequence=through_sequence
            )
            request_material = {
                "operation": "score_attempt",
                "attempt_id": attempt_id,
                "through_sequence": through_sequence,
                "provider_id": provider_id,
                "provider_version": provider_version,
                "action_trace_digest": action_trace_digest(actions),
            }
            command_hash = _command_hash(request_material)
            prior = self._prior_command(
                connection,
                idempotency_key,
                "TaskEvaluationRecorded",
                command_hash,
            )
            if prior is not None:
                payload = _json_object(prior["payload_json"], "Prior evaluation event")
                return self._evaluation_report(connection, payload["evaluation"]["id"], True)
            claimed = self._prior_scoring_claim(
                connection,
                idempotency_key=idempotency_key,
                command_hash=command_hash,
            )
            if claimed is not None:
                return claimed
            try:
                provider = registry.inspect(provider_id, provider_version)
            except (LookupError, RuntimeError, ValueError) as exc:
                raise ValidationError(f"Task scoring failed safely: {exc}") from exc
            scorer_contract = next(
                (
                    contract
                    for contract in task.scorer_contracts
                    if contract.key
                    == (
                        provider.declared_kind,
                        provider.provider_id,
                        provider.provider_version,
                    )
                ),
                None,
            )
            request = ScoringRequest(
                evaluation_id=new_id("teval"),
                trace_id=attempt_id,
                task_id=task.id,
                task_version=task.version,
                task_digest=task.digest,
                action_trace_digest=action_trace_digest(actions),
                criterion_ids=(
                    scorer_contract.criterion_ids
                    if scorer_contract is not None
                    else tuple(criterion.id for criterion in task.criteria)
                ),
                scorer_contract=scorer_contract,
            )

        # Reserve the logical scoring command before crossing the provider
        # boundary.  The transaction is deliberately short and is committed
        # before provider execution.  Claims are unique by both command hash
        # and caller key (when supplied), are never stolen, and never expire:
        # the earlier callback may still be running, or may have run before a
        # process crash, so invoking it again would violate at-most-once
        # admission.
        with self.database.transaction() as connection:
            current_attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?",
                (attempt_id,),
            ).fetchone()
            if current_attempt is None:
                raise ConflictError(
                    "Performance attempt disappeared before scoring admission."
                )
            self.database.require_learner_evidence_safe(
                current_attempt["learner_id"],
                connection,
            )
            prior = self._prior_command(
                connection,
                idempotency_key,
                "TaskEvaluationRecorded",
                command_hash,
            )
            if prior is not None:
                payload = _json_object(
                    prior["payload_json"], "Prior evaluation event"
                )
                return self._evaluation_report(
                    connection, payload["evaluation"]["id"], True
                )
            claimed = self._prior_scoring_claim(
                connection,
                idempotency_key=idempotency_key,
                command_hash=command_hash,
            )
            if claimed is not None:
                return claimed
            current_through_sequence = self._submission_boundary(
                connection, attempt_id
            )
            current_actions = self._typed_actions(
                connection,
                attempt_id,
                through_sequence=current_through_sequence,
            )
            current_action_trace_digest = action_trace_digest(
                current_actions
            )
            current_command_hash = _command_hash(
                {
                    "operation": "score_attempt",
                    "attempt_id": attempt_id,
                    "through_sequence": current_through_sequence,
                    "provider_id": provider_id,
                    "provider_version": provider_version,
                    "action_trace_digest": current_action_trace_digest,
                }
            )
            if (
                current_command_hash != command_hash
                or current_through_sequence != through_sequence
                or current_attempt["task_id"] != attempt["task_id"]
                or current_attempt["task_version"] != attempt["task_version"]
                or current_attempt["task_digest"] != attempt["task_digest"]
            ):
                raise ConflictError(
                    "Performance attempt changed before scoring admission; "
                    "retry against the current submitted trace."
                )
            self._require_active_session_interval(
                connection,
                current_attempt,
                occurred,
            )
            claimed_at = to_timestamp(occurred)
            claim_id = "psc_" + command_hash
            claim_event = self.database.append_event(
                connection,
                stream_id=f"learner:{current_attempt['learner_id']}",
                event_type="PerformanceScoringClaimed",
                schema_version=PERFORMANCE_EVENT_SCHEMA_VERSION,
                payload=performance_scoring_claim_payload(
                    claim_id=claim_id,
                    caller_idempotency_key=idempotency_key,
                    attempt_id=attempt_id,
                    evaluation_id=request.evaluation_id,
                    through_sequence=through_sequence,
                    provider_id=provider_id,
                    provider_version=provider_version,
                    action_trace_digest_value=current_action_trace_digest,
                    command_hash=command_hash,
                    claimed_at=claimed_at,
                ),
                metadata={
                    "claim_schema_version": 1,
                    "admission_mode": "pre_callback",
                    "source_schema_version": None,
                    "shadow_only": True,
                },
                learner_id=current_attempt["learner_id"],
                session_id=current_attempt["session_id"],
                idempotency_key=performance_scoring_claim_event_key(
                    command_hash
                ),
                correlation_id=attempt_id,
                causation_id=attempt_id,
                occurred_at=occurred,
            )
            connection.execute(
                """INSERT INTO performance_scoring_claims(
                       id, event_id, idempotency_key, attempt_id, evaluation_id,
                       through_sequence, provider_id, provider_version,
                       action_trace_digest, command_hash, claimed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    claim_id,
                    claim_event["event_id"],
                    idempotency_key,
                    attempt_id,
                    request.evaluation_id,
                    through_sequence,
                    provider_id,
                    provider_version,
                    current_action_trace_digest,
                    command_hash,
                    claimed_at,
                ),
            )
            claim_event_id = claim_event["event_id"]

        try:
            result = registry.score(provider_id, provider_version, request)
        except (LookupError, RuntimeError, ValueError) as exc:
            raise ValidationError(f"Task scoring failed safely: {exc}") from exc

        with self.database.transaction() as connection:
            current_attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if current_attempt is None:
                raise ConflictError(
                    "Performance attempt disappeared while it was being scored."
                )
            self.database.require_learner_evidence_safe(
                current_attempt["learner_id"],
                connection,
            )
            current_through_sequence = self._submission_boundary(
                connection, attempt_id
            )
            current_task = self._task_for_attempt(connection, current_attempt)
            current_actions = self._typed_actions(
                connection,
                attempt_id,
                through_sequence=current_through_sequence,
            )
            current_command_hash = _command_hash(
                {
                    "operation": "score_attempt",
                    "attempt_id": attempt_id,
                    "through_sequence": current_through_sequence,
                    "provider_id": provider_id,
                    "provider_version": provider_version,
                    "action_trace_digest": action_trace_digest(
                        current_actions
                    ),
                }
            )
            prior = self._prior_command(
                connection,
                idempotency_key,
                "TaskEvaluationRecorded",
                current_command_hash,
            )
            if prior is not None:
                payload = _json_object(
                    prior["payload_json"], "Prior evaluation event"
                )
                return self._evaluation_report(
                    connection, payload["evaluation"]["id"], True
                )
            if (
                current_command_hash != command_hash
                or current_through_sequence != through_sequence
                or current_attempt["task_id"] != attempt["task_id"]
                or current_attempt["task_version"] != attempt["task_version"]
                or current_attempt["task_digest"] != attempt["task_digest"]
                or current_task.digest != task.digest
            ):
                raise ConflictError(
                    "Performance attempt changed while it was being scored; "
                    "retry against the current submitted trace."
                )
            return self._record_result(
                connection,
                current_attempt,
                current_task,
                current_actions,
                current_through_sequence,
                result,
                idempotency_key=idempotency_key,
                occurred=occurred,
                command_hash=current_command_hash,
                claim_event_id=claim_event_id,
            )

    def import_evaluation(
        self,
        attempt_id: str,
        imported: ImportedEvaluation,
        *,
        provider_id: str,
        provider_version: str,
        declared_kind: ScorerKind = ScorerKind.IMPORTED,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        _require_id(attempt_id, "attempt_id")
        if not isinstance(imported, ImportedEvaluation):
            raise ValidationError("imported must be an ImportedEvaluation.")
        if not isinstance(declared_kind, ScorerKind):
            raise ValidationError("declared_kind must be a ScorerKind.")
        occurred = _now(now)
        with self.database.transaction() as connection:
            attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise NotFoundError(f"Performance attempt {attempt_id} does not exist.")
            self.database.require_learner_evidence_safe(
                attempt["learner_id"],
                connection,
            )
            through_sequence = self._submission_boundary(connection, attempt_id)
            task = self._task_for_attempt(connection, attempt)
            actions = self._typed_actions(
                connection, attempt_id, through_sequence=through_sequence
            )
            command_hash = _command_hash(
                {
                    "operation": "import_evaluation",
                    "attempt_id": attempt_id,
                    "through_sequence": through_sequence,
                    "provider_id": provider_id,
                    "provider_version": provider_version,
                    "declared_kind": declared_kind.value,
                    "imported_digest": imported.digest,
                }
            )
            prior = self._prior_command(
                connection,
                idempotency_key,
                "TaskEvaluationRecorded",
                command_hash,
            )
            if prior is not None:
                payload = _json_object(prior["payload_json"], "Prior evaluation event")
                return self._evaluation_report(connection, payload["evaluation"]["id"], True)
            request = ScoringRequest(
                evaluation_id=new_id("teval"),
                trace_id=attempt_id,
                task_id=task.id,
                task_version=task.version,
                task_digest=task.digest,
                action_trace_digest=action_trace_digest(actions),
                criterion_ids=tuple(criterion.id for criterion in task.criteria),
            )
            try:
                result = normalize_imported_evaluation(
                    request,
                    imported,
                    provider_id=provider_id,
                    provider_version=provider_version,
                    declared_kind=declared_kind,
                )
            except ValueError as exc:
                raise ValidationError(f"Imported evaluation failed safely: {exc}") from exc
            return self._record_result(
                connection,
                attempt,
                task,
                actions,
                through_sequence,
                result,
                idempotency_key=idempotency_key,
                occurred=occurred,
                command_hash=command_hash,
            )

    @staticmethod
    def _evaluation_report(
        connection: sqlite3.Connection, evaluation_id: str, replay: bool
    ) -> dict[str, Any]:
        row = connection.execute(
            """SELECT evaluation.*, bundle.bundle_digest, bundle.bundle_json,
                      bundle.projection_applied, bundle.certification_applied
               FROM task_evaluations evaluation
               JOIN shadow_evidence_bundles bundle
                 ON bundle.evaluation_id=evaluation.id
               WHERE evaluation.id=?""",
            (evaluation_id,),
        ).fetchone()
        if row is None:
            raise ValidationError("Evaluation event lacks its shadow projection.")
        return {
            "attempt_id": row["attempt_id"],
            "evaluation": _json_object(row["evaluation_json"], "Stored evaluation"),
            "evaluation_digest": row["evaluation_digest"],
            "authority": _json_object(row["authority_json"], "Stored authority"),
            "shadow_evidence": _json_object(row["bundle_json"], "Stored evidence"),
            "bundle_digest": row["bundle_digest"],
            "projection_applied": bool(row["projection_applied"]),
            "certification_applied": bool(row["certification_applied"]),
            "idempotent_replay": replay,
        }

    @classmethod
    def _attempt_report(
        cls, connection: sqlite3.Connection, attempt_id: str, replay: bool
    ) -> dict[str, Any]:
        attempt = connection.execute(
            "SELECT * FROM performance_attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        if attempt is None:
            raise NotFoundError(f"Performance attempt {attempt_id} does not exist.")
        definition = connection.execute(
            """SELECT definition_json FROM performance_tasks
               WHERE task_id=? AND task_version=?""",
            (attempt["task_id"], attempt["task_version"]),
        ).fetchone()
        counts = connection.execute(
            """SELECT COUNT(*) AS action_count,
                      SUM(action_type='hint_requested') AS hint_count,
                      SUM(action_type='check_run') AS check_count,
                      MAX(elapsed_ms) AS elapsed_ms
               FROM performance_actions WHERE attempt_id=?""",
            (attempt_id,),
        ).fetchone()
        evaluation_count = connection.execute(
            "SELECT COUNT(*) AS n FROM task_evaluations WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()["n"]
        return {
            **_row_dict(attempt),
            "status": cls._attempt_status(connection, attempt_id),
            "task": _json_object(definition["definition_json"], "Stored task definition"),
            "action_count": counts["action_count"],
            "hint_count": counts["hint_count"] or 0,
            "check_count": counts["check_count"] or 0,
            "elapsed_ms": counts["elapsed_ms"],
            "evaluation_count": evaluation_count,
            "shadow_only": True,
            "idempotent_replay": replay,
        }

    def report(self, attempt_id: str) -> dict[str, Any]:
        _require_id(attempt_id, "attempt_id")
        with self.database.read() as connection:
            owner = connection.execute(
                """SELECT learner_id FROM performance_attempts WHERE id=?""",
                (attempt_id,),
            ).fetchone()
            if owner is None:
                raise NotFoundError(
                    f"Performance attempt {attempt_id} does not exist."
                )
            self.database.require_learner_evidence_safe(
                owner["learner_id"],
                connection,
            )
            report = self._attempt_report(connection, attempt_id, False)
            evaluations = connection.execute(
                """SELECT evaluation.id FROM task_evaluations evaluation
                   WHERE evaluation.attempt_id=?
                   ORDER BY evaluation.recorded_at, evaluation.id""",
                (attempt_id,),
            ).fetchall()
            report["evaluations"] = [
                self._evaluation_report(connection, row["id"], False)
                for row in evaluations
            ]
            report["actions"] = [
                self._action_view(row)
                for row in connection.execute(
                    """SELECT * FROM performance_actions
                       WHERE attempt_id=? ORDER BY sequence""",
                    (attempt_id,),
                )
            ]
            family = report["task"]["family_id"]
            independent = connection.execute(
                """SELECT COUNT(DISTINCT attempt.task_id) AS task_count,
                          COUNT(DISTINCT attempt.id) AS attempt_count
                   FROM performance_attempts attempt
                   JOIN performance_tasks task
                     ON task.task_id=attempt.task_id
                    AND task.task_version=attempt.task_version
                   WHERE attempt.learner_id=?
                     AND json_extract(task.definition_json, '$.family_id')=?""",
                (report["learner_id"], family),
            ).fetchone()
            report["family_shadow_history"] = {
                "family_id": family,
                "distinct_task_count": independent["task_count"],
                "attempt_count": independent["attempt_count"],
                "mastery_claim": False,
                "certification_claim": False,
            }
        with self.database.read() as connection:
            self.database.require_learner_evidence_safe(
                report["learner_id"],
                connection,
            )
        return report


def performance_integrity_errors(
    connection: sqlite3.Connection,
) -> list[str]:
    """Recompute every shadow-ledger commitment from immutable definitions/events.

    The function is intentionally read-only and exception-safe for
    :meth:`tsq.store.Database.verify_integrity`.  It validates task releases,
    event/projection bijections, trace lifecycle, scorer authority envelopes,
    and deterministic evidence reduction.  It never applies the evidence.
    """

    errors: list[str] = []

    def decode(raw: str, label: str) -> dict[str, Any] | None:
        try:
            return _json_object(raw, label)
        except (TypeError, ValueError, ValidationError) as exc:
            errors.append(f"{label}: {exc}")
            return None

    def exact(
        value: Mapping[str, Any] | None,
        fields: set[str],
        label: str,
    ) -> bool:
        if value is None:
            return False
        if set(value) == fields:
            return True
        errors.append(
            f"{label}: expected fields {', '.join(sorted(fields))}; "
            f"found {', '.join(sorted(value))}"
        )
        return False

    def authority_decision(
        declared_kind: ScorerKind,
        *,
        verified: bool,
        synthetic: bool,
        direct_import: bool,
        attestation_digest: str | None,
    ) -> tuple[ScorerKind, str]:
        if synthetic:
            return (ScorerKind.IMPORTED, "synthetic_provider_shadow_only")
        if direct_import:
            if declared_kind is ScorerKind.MODEL:
                return (ScorerKind.MODEL, "model_score_shadow_only")
            if declared_kind is ScorerKind.HUMAN:
                return (ScorerKind.IMPORTED, "unverified_human_shadow_only")
            if declared_kind is ScorerKind.DETERMINISTIC:
                return (
                    ScorerKind.IMPORTED,
                    "direct_import_cannot_claim_authority",
                )
            return (ScorerKind.IMPORTED, "imported_score_unadjudicated")
        if declared_kind is ScorerKind.MODEL:
            return (ScorerKind.MODEL, "model_score_shadow_only")
        if declared_kind is ScorerKind.IMPORTED:
            return (ScorerKind.IMPORTED, "imported_score_unadjudicated")
        if not verified:
            return (
                ScorerKind.IMPORTED,
                (
                    "unverified_human_shadow_only"
                    if declared_kind is ScorerKind.HUMAN
                    else "unverified_deterministic_shadow_only"
                ),
            )
        if declared_kind is ScorerKind.HUMAN:
            if attestation_digest is None:
                return (
                    ScorerKind.IMPORTED,
                    "missing_verified_human_attestation",
                )
            return (ScorerKind.HUMAN, "verified_human_authority")
        return (
            ScorerKind.DETERMINISTIC,
            "verified_deterministic_authority",
        )

    def validate_authority(
        authority: Mapping[str, Any],
        evaluation: TaskEvaluation,
        task: LearningTask,
        label: str,
    ) -> None:
        exact(
            authority,
            {"normalized_result", "normalized_result_digest"},
            f"{label} authority",
        )
        normalized = authority.get("normalized_result")
        if type(normalized) is not dict:
            errors.append(f"{label}: normalized authority result is not an object")
            return
        exact(
            normalized,
            {
                "evaluation",
                "request",
                "provider",
                "decisions",
                "normalization_mode",
                "shadow_only",
                "schema_version",
            },
            f"{label} normalized result",
        )
        if (
            normalized.get("schema_version")
            != NORMALIZED_SCORING_RESULT_SCHEMA_VERSION
        ):
            errors.append(f"{label}: unsupported normalized result schema")
        if authority.get("normalized_result_digest") != canonical_digest(
            normalized
        ):
            errors.append(f"{label}: normalized authority digest mismatch")
        if canonical_json(normalized.get("evaluation")) != canonical_json(
            evaluation.terms()
        ):
            errors.append(f"{label}: authority discarded or changed evaluation")

        request_terms = normalized.get("request")
        provider_terms = normalized.get("provider")
        decisions = normalized.get("decisions")
        if type(request_terms) is not dict:
            errors.append(f"{label}: scoring request is malformed")
            return
        exact(
            request_terms,
            {
                "evaluation_id",
                "trace_id",
                "task_id",
                "task_version",
                "task_digest",
                "action_trace_digest",
                "criterion_ids",
                "scorer_contract",
                "scorer_contract_digest",
            },
            f"{label} scoring request",
        )
        contract_terms = request_terms.get("scorer_contract")
        contract: ScorerContract | None = None
        if contract_terms is not None:
            try:
                contract = ScorerContract.from_terms(contract_terms)
            except (TypeError, ValueError) as exc:
                errors.append(f"{label}: invalid scorer contract ({exc})")
            else:
                if request_terms.get(
                    "scorer_contract_digest"
                ) != canonical_digest(contract.terms()):
                    errors.append(f"{label}: scorer contract digest mismatch")
                if all(
                    canonical_json(contract.terms())
                    != canonical_json(candidate.terms())
                    for candidate in task.scorer_contracts
                ):
                    errors.append(
                        f"{label}: scorer contract is absent from task release"
                    )
        elif request_terms.get("scorer_contract_digest") is not None:
            errors.append(
                f"{label}: null scorer contract has a non-null digest"
            )
        criterion_ids = request_terms.get("criterion_ids")
        try:
            request = ScoringRequest(
                evaluation_id=request_terms.get("evaluation_id"),
                trace_id=request_terms.get("trace_id"),
                task_id=request_terms.get("task_id"),
                task_version=request_terms.get("task_version"),
                task_digest=request_terms.get("task_digest"),
                action_trace_digest=request_terms.get(
                    "action_trace_digest"
                ),
                criterion_ids=(
                    tuple(criterion_ids)
                    if type(criterion_ids) is list
                    else criterion_ids
                ),
                scorer_contract=contract,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"{label}: invalid scoring request ({exc})")
            request = None
        if request is not None and canonical_json(
            request.terms()
        ) != canonical_json(request_terms):
            errors.append(f"{label}: scoring request is not canonical")
        expected_request_criterion_ids = (
            contract.criterion_ids
            if contract is not None
            else tuple(sorted(criterion.id for criterion in task.criteria))
        )
        if request is not None and (
            request.evaluation_id != evaluation.id
            or request.trace_id != evaluation.trace_id
            or request.task_id != task.id
            or request.task_version != task.version
            or request.task_digest != task.digest
            or request.action_trace_digest
            != evaluation.action_trace_digest
            or request.criterion_ids
            != expected_request_criterion_ids
            or {
                criterion.criterion_id
                for criterion in evaluation.criteria
            }
            != set(request.criterion_ids)
        ):
            errors.append(f"{label}: scoring request boundary mismatch")

        if type(provider_terms) is not dict:
            errors.append(f"{label}: provider authority is malformed")
            return
        exact(
            provider_terms,
            {
                "provider_id",
                "provider_version",
                "declared_kind",
                "authority_id",
                "authority_manifest_digest",
                "binding_digest",
                "check_set_manifests",
                "artifact_manifests",
                "verified",
                "synthetic",
                "shadow_only",
            },
            f"{label} provider",
        )
        try:
            provider = RegisteredProvider.from_terms(provider_terms)
        except (TypeError, ValueError) as exc:
            errors.append(f"{label}: invalid provider binding ({exc})")
            return
        binding = ProviderAuthorityBinding(
            provider_id=provider.provider_id,
            provider_version=provider.provider_version,
            declared_kind=provider.declared_kind,
            authority_id=provider.authority_id,
            authority_manifest_digest=provider.authority_manifest_digest,
            check_set_manifests=provider.check_set_manifests,
            artifact_manifests=provider.artifact_manifests,
            verified=provider.verified,
        )
        declared_kind = provider.declared_kind
        synthetic = provider.synthetic
        provider_shadow_only = provider.shadow_only

        try:
            mode = NormalizationMode(normalized.get("normalization_mode"))
        except (TypeError, ValueError):
            errors.append(f"{label}: unknown normalization mode")
            return
        if mode is NormalizationMode.DIRECT_IMPORT:
            expected_authority_id = "authority.direct-import-shadow"
            expected_manifest = canonical_digest(
                {
                    "authority_id": expected_authority_id,
                    "rule": (
                        "direct imports cannot confer deterministic or human "
                        "authority"
                    ),
                    "schema_version": binding.schema_version,
                }
            )
            if (
                binding.authority_id != expected_authority_id
                or binding.authority_manifest_digest != expected_manifest
                or binding.verified
                or synthetic
                or contract is not None
            ):
                errors.append(
                    f"{label}: direct-import authority boundary mismatch"
                )

        if contract is not None and (
            contract.key
            != (
                declared_kind,
                binding.provider_id,
                binding.provider_version,
            )
            or contract.authority_id != binding.authority_id
            or contract.authority_manifest_digest
            != binding.authority_manifest_digest
            or contract.check_set_manifests != binding.check_set_manifests
            or contract.artifact_manifests != binding.artifact_manifests
        ):
            errors.append(f"{label}: provider does not match scorer contract")
        if (
            binding.verified
            and not provider_shadow_only
            and contract is None
        ):
            errors.append(
                f"{label}: verified provider lacks task scorer contract"
            )

        if type(decisions) is not list:
            errors.append(f"{label}: authority decisions are malformed")
            return
        decisions_by_id: dict[str, Mapping[str, Any]] = {}
        for index, decision in enumerate(decisions):
            decision_label = f"{label} authority decision {index}"
            if type(decision) is not dict:
                errors.append(f"{decision_label}: expected an object")
                continue
            exact(
                decision,
                {
                    "criterion_id",
                    "declared_kind",
                    "effective_kind",
                    "reason_code",
                    "shadow_only",
                },
                decision_label,
            )
            criterion_id = decision.get("criterion_id")
            if type(criterion_id) is not str:
                errors.append(f"{decision_label}: invalid criterion identity")
                continue
            if criterion_id in decisions_by_id:
                errors.append(
                    f"{label}: authority decision criterion is duplicated"
                )
            decisions_by_id[criterion_id] = decision
        evaluation_ids = {
            criterion.criterion_id for criterion in evaluation.criteria
        }
        if set(decisions_by_id) != evaluation_ids:
            errors.append(
                f"{label}: authority decisions do not exactly cover criteria"
            )
        shadow_decisions: list[bool] = []
        for criterion in evaluation.criteria:
            decision = decisions_by_id.get(criterion.criterion_id)
            if decision is None:
                continue
            expected_kind, expected_reason = authority_decision(
                declared_kind,
                verified=binding.verified,
                synthetic=synthetic,
                direct_import=mode is NormalizationMode.DIRECT_IMPORT,
                attestation_digest=criterion.attestation_digest,
            )
            expected_shadow = expected_kind in {
                ScorerKind.MODEL,
                ScorerKind.IMPORTED,
            }
            shadow_decisions.append(expected_shadow)
            if (
                decision.get("declared_kind") != declared_kind.value
                or decision.get("effective_kind") != expected_kind.value
                or decision.get("reason_code") != expected_reason
                or decision.get("shadow_only") is not expected_shadow
                or criterion.scorer_kind is not expected_kind
                or criterion.scorer_id != binding.provider_id
                or criterion.scorer_version != binding.provider_version
            ):
                errors.append(
                    f"{label}: authority decision does not match criterion "
                    f"{criterion.criterion_id}"
                )
            if (
                expected_kind
                in {ScorerKind.DETERMINISTIC, ScorerKind.HUMAN}
                and (
                    contract is None
                    or criterion.criterion_id not in contract.criterion_ids
                )
            ):
                errors.append(
                    f"{label}: trusted criterion lacks released authority"
                )
        if (
            type(normalized.get("shadow_only")) is not bool
            or normalized.get("shadow_only") is not all(shadow_decisions)
        ):
            errors.append(f"{label}: normalized shadow-only decision mismatch")

    task_rows = connection.execute(
        "SELECT * FROM performance_tasks ORDER BY task_id, task_version"
    ).fetchall()
    tasks: dict[tuple[str, int], LearningTask] = {}
    for row in task_rows:
        label = f"performance task {row['task_id']}@{row['task_version']}"
        terms = decode(row["definition_json"], f"{label} definition")
        if terms is None:
            continue
        try:
            task = LearningTask.from_terms(terms)
        except (TypeError, ValueError) as exc:
            errors.append(f"{label}: invalid definition ({exc})")
            continue
        if (
            task.id != row["task_id"]
            or task.version != row["task_version"]
            or task.digest != row["task_digest"]
            or canonical_json(task.terms()) != row["definition_json"]
        ):
            errors.append(f"{label}: identity or digest mismatch")
        try:
            _aware_timestamp(row["imported_at"], f"{label} imported_at")
        except ValidationError as exc:
            errors.append(str(exc))
        tasks[(row["task_id"], row["task_version"])] = task

    release_rows = connection.execute(
        "SELECT * FROM performance_task_releases ORDER BY id"
    ).fetchall()
    release_ids = {row["id"] for row in release_rows}
    for row in release_rows:
        label = f"performance task release {row['id']}"
        review_terms = decode(row["review_json"], f"{label} review")
        if review_terms is None:
            continue
        try:
            review = TaskReleaseReview.from_terms(review_terms)
        except (TypeError, ValueError, ValidationError) as exc:
            errors.append(f"{label}: invalid review ({exc})")
            continue
        members = connection.execute(
            """SELECT * FROM release_performance_tasks
               WHERE release_id=? ORDER BY task_id, task_version""",
            (row["id"],),
        ).fetchall()
        source_rows = {
            source["source_id"]: source["content_hash"]
            for source in connection.execute(
                """SELECT membership.source_id, source.content_hash
                   FROM release_sources membership
                   JOIN sources source ON source.id=membership.source_id
                   WHERE membership.release_id=?""",
                (row["corpus_release_id"],),
            )
        }
        concept_ids = {
            item["concept_id"]
            for item in connection.execute(
                """SELECT concept_id FROM release_concepts
                   WHERE release_id=?""",
                (row["corpus_release_id"],),
            )
        }
        objective_primary_concepts = {
            item["objective_id"]: item["primary_concept_id"]
            for item in connection.execute(
                """SELECT membership.objective_id,
                          objective.primary_concept_id
                   FROM release_learning_objectives membership
                   JOIN learning_objectives objective
                     ON objective.id=membership.objective_id
                   WHERE membership.release_id=?""",
                (row["corpus_release_id"],),
            )
        }
        misconception_ids = {
            item["misconception_id"]
            for item in connection.execute(
                """SELECT misconception_id FROM release_misconceptions
                   WHERE release_id=?""",
                (row["corpus_release_id"],),
            )
        }
        misconception_concepts = {
            item["misconception_id"]: item["concept_id"]
            for item in connection.execute(
                """SELECT membership.misconception_id,
                          misconception.concept_id
                   FROM release_misconceptions membership
                   JOIN misconceptions misconception
                     ON misconception.id=membership.misconception_id
                   WHERE membership.release_id=?""",
                (row["corpus_release_id"],),
            )
        }
        misconception_objectives: dict[str, set[str]] = {}
        for item in connection.execute(
            """SELECT DISTINCT option.misconception_id,
                              mapping.objective_id
               FROM release_option_objectives mapping
               JOIN options option
                 ON option.question_id=mapping.question_id
                AND option.option_id=mapping.option_id
               WHERE mapping.release_id=?
                 AND option.misconception_id IS NOT NULL""",
            (row["corpus_release_id"],),
        ):
            misconception_objectives.setdefault(
                item["misconception_id"], set()
            ).add(item["objective_id"])
        definitions: list[tuple[str, LearningTask]] = []
        for member in members:
            task = tasks.get((member["task_id"], member["task_version"]))
            if task is None:
                errors.append(f"{label}: member has no valid task definition")
                continue
            if member["task_digest"] != task.digest:
                errors.append(
                    f"{label}: member {task.id}@{task.version} digest mismatch"
                )
            definitions.append((member["status"], task))
            for source_id, digest in task.source_manifests:
                if source_rows.get(source_id) != digest:
                    errors.append(
                        f"{label}: task {task.id} source {source_id} is not "
                        "pinned to the corpus release"
                    )
            if not set(task.concept_ids) <= concept_ids:
                errors.append(f"{label}: task {task.id} has out-of-release concepts")
            for criterion in task.criteria:
                for objective_id in criterion.objective_ids:
                    primary_concept_id = objective_primary_concepts.get(
                        objective_id
                    )
                    if primary_concept_id is None:
                        errors.append(
                            f"{label}: task {task.id} criterion "
                            f"{criterion.id} has out-of-release objective "
                            f"{objective_id}"
                        )
                    elif primary_concept_id not in criterion.concept_ids:
                        errors.append(
                            f"{label}: task {task.id} criterion "
                            f"{criterion.id} objective {objective_id} primary "
                            "concept is outside its criterion concept mapping"
                        )
            if not set(task.misconception_ids) <= misconception_ids:
                errors.append(
                    f"{label}: task {task.id} has out-of-release misconceptions"
                )
            for criterion in task.criteria:
                criterion_objectives = set(criterion.objective_ids)
                criterion_concepts = set(criterion.concept_ids)
                for misconception_id in criterion.misconception_ids:
                    if criterion_objectives and not (
                        criterion_objectives
                        & misconception_objectives.get(
                            misconception_id, set()
                        )
                    ):
                        errors.append(
                            f"{label}: task {task.id} criterion "
                            f"{criterion.id} misconception "
                            f"{misconception_id} is not mapped to any of its "
                            "objectives"
                        )
                    elif (
                        not criterion_objectives
                        and misconception_concepts.get(misconception_id)
                        not in criterion_concepts
                    ):
                        errors.append(
                            f"{label}: task {task.id} criterion "
                            f"{criterion.id} misconception "
                            f"{misconception_id} is outside its concept mapping"
                        )
        try:
            reconstructed = PerformanceTaskRelease(
                title=row["title"],
                corpus_release_id=row["corpus_release_id"],
                review=review,
                tasks=tuple(definitions),
            )
            if (
                reconstructed.release_id != row["id"]
                or reconstructed.bundle_hash != row["bundle_hash"]
            ):
                errors.append(f"{label}: bundle commitment mismatch")
        except (TypeError, ValueError, ValidationError) as exc:
            errors.append(f"{label}: cannot reconstruct release ({exc})")
        corpus = connection.execute(
            "SELECT sealed_at FROM corpus_releases WHERE id=?",
            (row["corpus_release_id"],),
        ).fetchone()
        if corpus is None or corpus["sealed_at"] is None:
            errors.append(f"{label}: pinned corpus release is absent or unsealed")
        elif row["created_at"] is not None:
            try:
                if _aware_timestamp(
                    row["created_at"], f"{label} created_at"
                ) < _aware_timestamp(
                    corpus["sealed_at"], f"{label} corpus seal"
                ):
                    errors.append(
                        f"{label}: publication precedes corpus release"
                    )
            except ValidationError as exc:
                errors.append(str(exc))
        for field in ("created_at", "sealed_at"):
            try:
                _aware_timestamp(row[field], f"{label} {field}")
            except ValidationError as exc:
                errors.append(str(exc))
        try:
            reviewed_at = _aware_timestamp(
                review.reviewed_at, f"{label} reviewed_at"
            )
            created_at = _aware_timestamp(
                row["created_at"], f"{label} created_at"
            )
            if reviewed_at > created_at:
                errors.append(f"{label}: publication precedes review")
        except ValidationError as exc:
            errors.append(str(exc))
        if row["sealed_at"] != row["created_at"]:
            errors.append(f"{label}: seal boundary differs from publication")

    performance_events = connection.execute(
        """SELECT * FROM events WHERE event_type IN (
               'PerformanceTaskStarted', 'PerformanceActionRecorded',
               'PerformanceScoringClaimed',
               'PerformanceScoringClaimMigrated',
               'PerformanceScoringLegacyExempted',
               'TaskEvaluationRecorded', 'ShadowEvidenceReduced'
           ) ORDER BY stream_id, stream_version"""
    ).fetchall()
    events_by_type: dict[str, dict[str, sqlite3.Row]] = {
        event_type: {}
        for event_type in (
            "PerformanceTaskStarted",
            "PerformanceActionRecorded",
            "PerformanceScoringClaimed",
            "PerformanceScoringClaimMigrated",
            "PerformanceScoringLegacyExempted",
            "TaskEvaluationRecorded",
            "ShadowEvidenceReduced",
        )
    }
    for event in performance_events:
        events_by_type[event["event_type"]][event["event_id"]] = event
    session_boundaries: dict[str, dict[str, list[sqlite3.Row]]] = {}
    for event in connection.execute(
        """SELECT * FROM events
           WHERE event_type IN ('SessionStarted', 'SessionEnded')
           ORDER BY stream_id, stream_version"""
    ).fetchall():
        session_id = event["session_id"]
        if type(session_id) is not str:
            continue
        session_boundaries.setdefault(
            session_id, {"started": [], "ended": []}
        )[
            "started" if event["event_type"] == "SessionStarted" else "ended"
        ].append(event)

    attempt_rows = connection.execute(
        "SELECT * FROM performance_attempts ORDER BY id"
    ).fetchall()
    attempts = {row["id"]: row for row in attempt_rows}
    projected_start_events: set[str] = set()
    attempt_tasks: dict[str, LearningTask] = {}
    for row in attempt_rows:
        label = f"performance attempt {row['id']}"
        event = events_by_type["PerformanceTaskStarted"].get(row["event_id"])
        if event is None:
            errors.append(f"{label}: missing PerformanceTaskStarted event")
            continue
        projected_start_events.add(event["event_id"])
        payload = decode(event["payload_json"], f"{label} event payload")
        metadata = decode(event["metadata_json"], f"{label} event metadata")
        exact(
            payload,
            {
                "attempt_id",
                "session_id",
                "learner_id",
                "task_release_id",
                "corpus_release_id",
                "task_id",
                "task_version",
                "task_digest",
                "session_revision",
                "learner_revision",
            },
            f"{label} event payload",
        )
        exact(
            metadata,
            {
                "command_hash",
                "task_schema_version",
                "shadow_only",
                "projection_applied",
                "certification_applied",
            },
            f"{label} event metadata",
        )
        expected_payload = {
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
        }
        if payload is not None and canonical_json(payload) != canonical_json(expected_payload):
            errors.append(f"{label}: event payload mismatch")
        if metadata is not None and (
            metadata.get("command_hash") != row["command_hash"]
            or metadata.get("shadow_only") is not True
            or metadata.get("projection_applied") is not False
            or metadata.get("certification_applied") is not False
        ):
            errors.append(f"{label}: unsafe or mismatched event metadata")
        if (
            event["schema_version"] != PERFORMANCE_EVENT_SCHEMA_VERSION
            or event["stream_id"] != f"learner:{row['learner_id']}"
            or event["learner_id"] != row["learner_id"]
            or event["session_id"] != row["session_id"]
            or event["correlation_id"] != row["id"]
            or event["causation_id"] != row["session_id"]
            or event["occurred_at"] != row["started_at"]
            or event["recorded_at"] != row["recorded_at"]
        ):
            errors.append(f"{label}: event envelope mismatch")
        boundaries = session_boundaries.get(
            row["session_id"], {"started": [], "ended": []}
        )
        started_boundaries = boundaries["started"]
        ended_boundaries = boundaries["ended"]
        if len(started_boundaries) != 1:
            errors.append(
                f"{label}: expected one SessionStarted boundary, "
                f"found {len(started_boundaries)}"
            )
        else:
            session_start = started_boundaries[0]
            if (
                session_start["stream_id"] != event["stream_id"]
                or session_start["learner_id"] != row["learner_id"]
                or session_start["session_id"] != row["session_id"]
                or session_start["stream_version"] >= event["stream_version"]
            ):
                errors.append(
                    f"{label}: start falls outside its session-active interval"
                )
            try:
                if _aware_timestamp(
                    event["occurred_at"], f"{label} event occurrence"
                ) < _aware_timestamp(
                    session_start["occurred_at"],
                    f"{label} session start occurrence",
                ):
                    errors.append(
                        f"{label}: occurrence precedes its session start"
                    )
            except ValidationError as exc:
                errors.append(str(exc))
        if len(ended_boundaries) > 1:
            errors.append(f"{label}: multiple SessionEnded boundaries exist")
        elif ended_boundaries and (
            ended_boundaries[0]["stream_id"] != event["stream_id"]
            or ended_boundaries[0]["learner_id"] != row["learner_id"]
            or ended_boundaries[0]["stream_version"] <= event["stream_version"]
        ):
            errors.append(
                f"{label}: start falls outside its session-active interval"
            )
        session_row = connection.execute(
            """SELECT learner_id, corpus_release_id FROM sessions WHERE id=?""",
            (row["session_id"],),
        ).fetchone()
        if (
            session_row is None
            or session_row["learner_id"] != row["learner_id"]
            or session_row["corpus_release_id"] != row["corpus_release_id"]
        ):
            errors.append(f"{label}: session identity/release mismatch")
        task = tasks.get((row["task_id"], row["task_version"]))
        if task is None or task.digest != row["task_digest"]:
            errors.append(f"{label}: task definition/digest mismatch")
        else:
            attempt_tasks[row["id"]] = task
            if metadata is not None and metadata.get("task_schema_version") != task.schema_version:
                errors.append(f"{label}: task schema metadata mismatch")
        membership = connection.execute(
            """SELECT member.task_digest, member.status,
                      release.corpus_release_id, release.created_at
               FROM release_performance_tasks member
               JOIN performance_task_releases release ON release.id=member.release_id
               WHERE member.release_id=? AND member.task_id=?
                 AND member.task_version=?""",
            (row["task_release_id"], row["task_id"], row["task_version"]),
        ).fetchone()
        if (
            membership is None
            or membership["task_digest"] != row["task_digest"]
            or membership["status"] not in SERVICEABLE_TASK_STATUSES
            or membership["corpus_release_id"] != row["corpus_release_id"]
            or row["task_release_id"] not in release_ids
        ):
            errors.append(f"{label}: invalid task-release boundary")
        elif event is not None:
            try:
                if _aware_timestamp(
                    event["occurred_at"], f"{label} event occurrence"
                ) < _aware_timestamp(
                    membership["created_at"], f"{label} release publication"
                ):
                    errors.append(
                        f"{label}: occurrence precedes task release"
                    )
            except ValidationError as exc:
                errors.append(str(exc))
        try:
            _aware_timestamp(row["started_at"], f"{label} started_at")
            _aware_timestamp(row["recorded_at"], f"{label} recorded_at")
        except ValidationError as exc:
            errors.append(str(exc))

    for event_id in events_by_type["PerformanceTaskStarted"]:
        if event_id not in projected_start_events:
            errors.append(
                f"event {event_id}: PerformanceTaskStarted has no attempt projection"
            )

    action_rows = connection.execute(
        "SELECT * FROM performance_actions ORDER BY attempt_id, sequence, id"
    ).fetchall()
    actions_by_attempt: dict[str, list[LearningAction]] = {}
    action_events_by_attempt_sequence: dict[
        tuple[str, int], sqlite3.Row
    ] = {}
    prior_action_event_version: dict[str, int] = {}
    prior_action_occurrence: dict[str, datetime] = {}
    projected_action_events: set[str] = set()
    for row in action_rows:
        label = f"performance action {row['id']}"
        attempt = attempts.get(row["attempt_id"])
        if attempt is None:
            errors.append(f"{label}: missing attempt")
            continue
        event = events_by_type["PerformanceActionRecorded"].get(row["event_id"])
        if event is None:
            errors.append(f"{label}: missing PerformanceActionRecorded event")
            continue
        projected_action_events.add(event["event_id"])
        event_payload = decode(event["payload_json"], f"{label} event payload")
        event_metadata = decode(event["metadata_json"], f"{label} event metadata")
        exact(event_payload, {"attempt_id", "action"}, f"{label} event payload")
        exact(
            event_metadata,
            {
                "command_hash",
                "action_schema_version",
                "task_digest",
                "task_release_id",
                "corpus_release_id",
                "observational_only",
                "shadow_only",
            },
            f"{label} event metadata",
        )
        action_terms = event_payload.get("action") if event_payload else None
        try:
            action = LearningAction.from_terms(action_terms)
        except (TypeError, ValueError) as exc:
            errors.append(f"{label}: invalid event action ({exc})")
            continue
        expected_terms = {
            "id": row["id"],
            "trace_id": row["attempt_id"],
            "sequence": row["sequence"],
            "kind": row["action_type"],
            "phase": row["phase"],
            "payload": decode(row["payload_json"], f"{label} payload"),
            "elapsed_ms": row["elapsed_ms"],
            "schema_version": action.schema_version,
        }
        if canonical_json(action.terms()) != canonical_json(expected_terms):
            errors.append(f"{label}: event/projection mismatch")
        if event_payload is not None and event_payload.get("attempt_id") != row["attempt_id"]:
            errors.append(f"{label}: event attempt mismatch")
        if event_metadata is not None and (
            event_metadata.get("command_hash") != row["command_hash"]
            or event_metadata.get("action_schema_version") != action.schema_version
            or event_metadata.get("task_digest") != attempt["task_digest"]
            or event_metadata.get("task_release_id") != attempt["task_release_id"]
            or event_metadata.get("corpus_release_id") != attempt["corpus_release_id"]
            or event_metadata.get("observational_only") is not True
            or event_metadata.get("shadow_only") is not True
        ):
            errors.append(f"{label}: event metadata mismatch")
        start_event = events_by_type["PerformanceTaskStarted"].get(attempt["event_id"])
        if (
            event["schema_version"] != PERFORMANCE_EVENT_SCHEMA_VERSION
            or event["stream_id"] != f"learner:{attempt['learner_id']}"
            or event["learner_id"] != attempt["learner_id"]
            or event["session_id"] != attempt["session_id"]
            or event["correlation_id"] != attempt["id"]
            or event["causation_id"] != attempt["id"]
            or event["occurred_at"] != row["occurred_at"]
            or event["recorded_at"] != row["recorded_at"]
            or start_event is None
            or start_event["stream_id"] != event["stream_id"]
            or start_event["stream_version"] >= event["stream_version"]
        ):
            errors.append(f"{label}: event envelope/boundary mismatch")
        prior_version = prior_action_event_version.get(row["attempt_id"])
        if (
            prior_version is not None
            and event["stream_version"] <= prior_version
        ):
            errors.append(f"{label}: event order does not follow action sequence")
        prior_action_event_version[row["attempt_id"]] = event["stream_version"]
        action_events_by_attempt_sequence[
            (row["attempt_id"], row["sequence"])
        ] = event
        try:
            occurred_at = _aware_timestamp(
                row["occurred_at"], f"{label} occurred_at"
            )
            started_at = _aware_timestamp(
                attempt["started_at"], f"{label} attempt start"
            )
            _aware_timestamp(row["recorded_at"], f"{label} recorded_at")
            if occurred_at < started_at:
                errors.append(f"{label}: occurred before task start")
            prior_occurrence = prior_action_occurrence.get(row["attempt_id"])
            if prior_occurrence is not None and occurred_at < prior_occurrence:
                errors.append(f"{label}: occurrence time is not monotonic")
            prior_action_occurrence[row["attempt_id"]] = occurred_at
            expected_elapsed = int(
                (occurred_at - started_at).total_seconds() * 1000
            )
            if row["elapsed_ms"] != expected_elapsed:
                errors.append(f"{label}: elapsed time does not match occurrence")
        except ValidationError as exc:
            errors.append(str(exc))
        boundaries = session_boundaries.get(
            attempt["session_id"], {"started": [], "ended": []}
        )
        if len(boundaries["started"]) == 1 and (
            boundaries["started"][0]["stream_id"] != event["stream_id"]
            or boundaries["started"][0]["stream_version"]
            >= event["stream_version"]
        ):
            errors.append(
                f"{label}: action falls outside its session-active interval"
            )
        if len(boundaries["ended"]) == 1 and (
            boundaries["ended"][0]["stream_id"] != event["stream_id"]
            or boundaries["ended"][0]["stream_version"]
            <= event["stream_version"]
        ):
            errors.append(
                f"{label}: action falls outside its session-active interval"
            )
        actions_by_attempt.setdefault(row["attempt_id"], []).append(action)

    for event_id in events_by_type["PerformanceActionRecorded"]:
        if event_id not in projected_action_events:
            errors.append(
                f"event {event_id}: PerformanceActionRecorded has no action projection"
            )
    for attempt_id in attempts:
        actions = actions_by_attempt.get(attempt_id, [])
        try:
            summary = summarize_actions(actions)
            if not actions or actions[0].kind is not ActionKind.STARTED:
                errors.append(f"performance attempt {attempt_id}: trace has no start")
            if summary.trace_id != attempt_id:
                errors.append(f"performance attempt {attempt_id}: trace identity mismatch")
        except (TypeError, ValueError) as exc:
            errors.append(f"performance attempt {attempt_id}: invalid trace ({exc})")

    evaluation_rows = connection.execute(
        "SELECT * FROM task_evaluations ORDER BY attempt_id, recorded_at, id"
    ).fetchall()
    evaluation_rows_by_id = {row["id"]: row for row in evaluation_rows}
    evaluations: dict[str, TaskEvaluation] = {}
    evaluation_modes: dict[str, NormalizationMode] = {}
    projected_evaluation_events: set[str] = set()
    for row in evaluation_rows:
        label = f"task evaluation {row['id']}"
        attempt = attempts.get(row["attempt_id"])
        task = attempt_tasks.get(row["attempt_id"])
        if attempt is None or task is None:
            errors.append(f"{label}: missing valid attempt/task")
            continue
        event = events_by_type["TaskEvaluationRecorded"].get(row["event_id"])
        if event is None:
            errors.append(f"{label}: missing TaskEvaluationRecorded event")
            continue
        projected_evaluation_events.add(event["event_id"])
        event_payload = decode(event["payload_json"], f"{label} event payload")
        event_metadata = decode(event["metadata_json"], f"{label} event metadata")
        exact(
            event_payload,
            {"attempt_id", "through_sequence", "evaluation_digest", "evaluation", "authority"},
            f"{label} event payload",
        )
        exact(
            event_metadata,
            {
                "command_hash",
                "task_release_id",
                "corpus_release_id",
                "shadow_only",
                "projection_applied",
                "certification_applied",
            },
            f"{label} event metadata",
        )
        terms = decode(row["evaluation_json"], f"{label} projection")
        authority = decode(row["authority_json"], f"{label} authority")
        try:
            evaluation = TaskEvaluation.from_terms(terms)
        except (TypeError, ValueError) as exc:
            errors.append(f"{label}: invalid evaluation ({exc})")
            continue
        if (
            evaluation.id != row["id"]
            or evaluation.digest != row["evaluation_digest"]
            or evaluation.trace_id != row["attempt_id"]
            or evaluation.task_id != task.id
            or evaluation.task_version != task.version
            or evaluation.task_digest != task.digest
            or canonical_json(evaluation.terms()) != row["evaluation_json"]
        ):
            errors.append(f"{label}: evaluation identity/digest mismatch")
        trace = tuple(
            action
            for action in actions_by_attempt.get(row["attempt_id"], [])
            if action.sequence <= row["through_sequence"]
        )
        submissions = [
            action
            for action in trace
            if action.kind is ActionKind.SUBMITTED
            and action.sequence == row["through_sequence"]
        ]
        if len(submissions) != 1 or evaluation.action_trace_digest != action_trace_digest(trace):
            errors.append(f"{label}: submitted trace boundary/digest mismatch")
        submission_event = action_events_by_attempt_sequence.get(
            (row["attempt_id"], row["through_sequence"])
        )
        if (
            submission_event is None
            or submission_event["stream_id"] != event["stream_id"]
            or submission_event["stream_version"] >= event["stream_version"]
        ):
            errors.append(
                f"{label}: event does not follow its submitted trace boundary"
            )
        try:
            evaluation_occurred_at = _aware_timestamp(
                event["occurred_at"], f"{label} event occurrence"
            )
            if submission_event is not None and evaluation_occurred_at < (
                _aware_timestamp(
                    submission_event["occurred_at"],
                    f"{label} submission occurrence",
                )
            ):
                errors.append(
                    f"{label}: occurrence precedes its submitted checkpoint"
                )
            _aware_timestamp(row["recorded_at"], f"{label} recorded_at")
        except ValidationError as exc:
            errors.append(str(exc))
        if event_payload is not None and (
            event_payload.get("attempt_id") != row["attempt_id"]
            or event_payload.get("through_sequence") != row["through_sequence"]
            or event_payload.get("evaluation_digest") != row["evaluation_digest"]
            or canonical_json(event_payload.get("evaluation")) != canonical_json(terms)
            or canonical_json(event_payload.get("authority")) != canonical_json(authority)
        ):
            errors.append(f"{label}: event payload mismatch")
        if event_metadata is not None and (
            event_metadata.get("command_hash") != row["command_hash"]
            or event_metadata.get("task_release_id") != attempt["task_release_id"]
            or event_metadata.get("corpus_release_id") != attempt["corpus_release_id"]
            or event_metadata.get("shadow_only") is not True
            or event_metadata.get("projection_applied") is not False
            or event_metadata.get("certification_applied") is not False
        ):
            errors.append(f"{label}: unsafe or mismatched event metadata")
        scoring_causation = connection.execute(
            """SELECT claim.event_id, claim_event.event_type
               FROM performance_scoring_claims claim
               JOIN events claim_event ON claim_event.event_id=claim.event_id
               WHERE claim.evaluation_id=?""",
            (row["id"],),
        ).fetchone()
        expected_causation_id = (
            scoring_causation["event_id"]
            if scoring_causation is not None
            and scoring_causation["event_type"]
            == "PerformanceScoringClaimed"
            else attempt["id"]
        )
        if (
            event["schema_version"] != PERFORMANCE_EVENT_SCHEMA_VERSION
            or event["stream_id"] != f"learner:{attempt['learner_id']}"
            or event["learner_id"] != attempt["learner_id"]
            or event["session_id"] != attempt["session_id"]
            or event["correlation_id"] != attempt["id"]
            or event["causation_id"] != expected_causation_id
            or event["recorded_at"] != row["recorded_at"]
        ):
            errors.append(f"{label}: event envelope mismatch")
        if authority is not None:
            validate_authority(authority, evaluation, task, label)
            normalized = authority.get("normalized_result")
            if type(normalized) is dict:
                try:
                    evaluation_modes[row["id"]] = NormalizationMode(
                        normalized.get("normalization_mode")
                    )
                except (TypeError, ValueError):
                    pass
        evaluations[row["id"]] = evaluation

    for event_id in events_by_type["TaskEvaluationRecorded"]:
        if event_id not in projected_evaluation_events:
            errors.append(
                f"event {event_id}: TaskEvaluationRecorded has no evaluation projection"
            )

    scoring_claim_rows = connection.execute(
        """SELECT * FROM performance_scoring_claims
           ORDER BY id"""
    ).fetchall()
    projected_claim_events: set[str] = set()
    claim_rows_by_evaluation: dict[str, list[sqlite3.Row]] = {}
    for row in scoring_claim_rows:
        label = f"performance scoring claim {row['id']}"
        attempt = attempts.get(row["attempt_id"])
        claim_rows_by_evaluation.setdefault(row["evaluation_id"], []).append(row)
        expected_claim_event_key: str | None = None
        try:
            _require_id(row["id"], "claim_id")
            _require_id(row["event_id"], "claim_event_id")
            if row["idempotency_key"] is not None:
                _require_text(
                    row["idempotency_key"], "idempotency_key", 256
                )
            _require_id(row["evaluation_id"], "evaluation_id")
            _require_id(row["provider_id"], "provider_id")
            _require_id(row["provider_version"], "provider_version")
            _require_digest(row["action_trace_digest"], "action_trace_digest")
            _require_digest(row["command_hash"], "command_hash")
            expected_claim_event_key = performance_scoring_claim_event_key(
                row["command_hash"]
            )
            _aware_timestamp(row["claimed_at"], f"{label} claimed_at")
        except ValidationError as exc:
            errors.append(f"{label}: {exc}")
        if attempt is None:
            errors.append(f"{label}: missing performance attempt")
            continue
        claim_event = (
            events_by_type["PerformanceScoringClaimed"].get(row["event_id"])
            or events_by_type["PerformanceScoringClaimMigrated"].get(
                row["event_id"]
            )
        )
        if claim_event is None:
            errors.append(f"{label}: missing scoring admission event")
        else:
            projected_claim_events.add(claim_event["event_id"])
            claim_payload = decode(
                claim_event["payload_json"], f"{label} admission payload"
            )
            claim_metadata = decode(
                claim_event["metadata_json"], f"{label} admission metadata"
            )
            exact(
                claim_payload,
                {
                    "claim_id",
                    "caller_idempotency_key",
                    "attempt_id",
                    "evaluation_id",
                    "through_sequence",
                    "provider_id",
                    "provider_version",
                    "action_trace_digest",
                    "command_hash",
                    "claimed_at",
                },
                f"{label} admission payload",
            )
            exact(
                claim_metadata,
                {
                    "claim_schema_version",
                    "admission_mode",
                    "source_schema_version",
                    "shadow_only",
                },
                f"{label} admission metadata",
            )
            expected_payload = performance_scoring_claim_payload(
                claim_id=row["id"],
                caller_idempotency_key=row["idempotency_key"],
                attempt_id=row["attempt_id"],
                evaluation_id=row["evaluation_id"],
                through_sequence=row["through_sequence"],
                provider_id=row["provider_id"],
                provider_version=row["provider_version"],
                action_trace_digest_value=row["action_trace_digest"],
                command_hash=row["command_hash"],
                claimed_at=row["claimed_at"],
            )
            if claim_payload is not None and canonical_json(
                claim_payload
            ) != canonical_json(expected_payload):
                errors.append(f"{label}: admission event payload mismatch")
            expected_mode = (
                "pre_callback"
                if claim_event["event_type"] == "PerformanceScoringClaimed"
                else "legacy_projection_migration"
            )
            expected_source_version = (
                None
                if expected_mode == "pre_callback"
                else 15
            )
            expected_session_ids = (
                {attempt["session_id"]}
                if expected_mode == "pre_callback"
                else {None, attempt["session_id"]}
            )
            if claim_metadata is not None and (
                claim_metadata.get("claim_schema_version") != 1
                or claim_metadata.get("admission_mode") != expected_mode
                or claim_metadata.get("source_schema_version")
                != expected_source_version
                or claim_metadata.get("shadow_only") is not True
            ):
                errors.append(f"{label}: admission event metadata mismatch")
            if (
                claim_event["schema_version"]
                != PERFORMANCE_EVENT_SCHEMA_VERSION
                or claim_event["stream_id"]
                != f"learner:{attempt['learner_id']}"
                or claim_event["learner_id"] != attempt["learner_id"]
                or claim_event["session_id"] not in expected_session_ids
                or claim_event["correlation_id"] != attempt["id"]
                or claim_event["causation_id"] != attempt["id"]
                or expected_claim_event_key is None
                or claim_event["idempotency_key"] != expected_claim_event_key
                or (
                    expected_mode == "pre_callback"
                    and claim_event["occurred_at"] != row["claimed_at"]
                )
            ):
                errors.append(f"{label}: admission event envelope mismatch")
        trace = tuple(
            action
            for action in actions_by_attempt.get(row["attempt_id"], [])
            if action.sequence <= row["through_sequence"]
        )
        submissions = [
            action
            for action in trace
            if action.kind is ActionKind.SUBMITTED
            and action.sequence == row["through_sequence"]
        ]
        try:
            trace_digest = action_trace_digest(trace)
        except (TypeError, ValueError) as exc:
            errors.append(f"{label}: cannot reconstruct submitted trace ({exc})")
            continue
        expected_command_hash = _command_hash(
            {
                "operation": "score_attempt",
                "attempt_id": row["attempt_id"],
                "through_sequence": row["through_sequence"],
                "provider_id": row["provider_id"],
                "provider_version": row["provider_version"],
                "action_trace_digest": trace_digest,
            }
        )
        if (
            len(submissions) != 1
            or row["action_trace_digest"] != trace_digest
            or row["command_hash"] != expected_command_hash
            or row["id"] != "psc_" + expected_command_hash
        ):
            errors.append(f"{label}: submitted trace or command commitment mismatch")
        submission_event = action_events_by_attempt_sequence.get(
            (row["attempt_id"], row["through_sequence"])
        )
        if (
            claim_event is not None
            and submission_event is not None
            and claim_event["event_type"] == "PerformanceScoringClaimed"
            and claim_event["stream_version"]
            <= submission_event["stream_version"]
        ):
            errors.append(
                f"{label}: admission does not follow submitted trace boundary"
            )
        evaluation_row = evaluation_rows_by_id.get(row["evaluation_id"])
        event = (
            events_by_type["TaskEvaluationRecorded"].get(
                evaluation_row["event_id"]
            )
            if evaluation_row is not None
            else None
        )
        if event is None:
            # An uncompleted claim is a valid fail-closed state.  It records
            # that a callback may have run, without fabricating an evaluation.
            continue
        payload = decode(event["payload_json"], f"{label} event payload")
        metadata = decode(event["metadata_json"], f"{label} event metadata")
        evaluation_terms = (
            payload.get("evaluation") if payload is not None else None
        )
        authority_terms = (
            payload.get("authority") if payload is not None else None
        )
        normalized_terms = (
            authority_terms.get("normalized_result")
            if type(authority_terms) is dict
            else None
        )
        provider_terms = (
            normalized_terms.get("provider")
            if type(normalized_terms) is dict
            else None
        )
        if (
            event["event_type"] != "TaskEvaluationRecorded"
            or event["idempotency_key"] != row["idempotency_key"]
            or metadata is None
            or metadata.get("command_hash") != row["command_hash"]
            or payload is None
            or payload.get("attempt_id") != row["attempt_id"]
            or payload.get("through_sequence") != row["through_sequence"]
            or type(evaluation_terms) is not dict
            or evaluation_terms.get("id") != row["evaluation_id"]
            or type(normalized_terms) is not dict
            or normalized_terms.get("normalization_mode")
            != NormalizationMode.REGISTERED_PROVIDER.value
            or type(provider_terms) is not dict
            or provider_terms.get("provider_id") != row["provider_id"]
            or provider_terms.get("provider_version") != row["provider_version"]
            or (
                claim_event is not None
                and claim_event["event_type"] == "PerformanceScoringClaimed"
                and (
                    event["causation_id"] != claim_event["event_id"]
                    or event["stream_version"]
                    <= claim_event["stream_version"]
                )
            )
        ):
            errors.append(f"{label}: completion event does not match its claim")

    for event_type in (
        "PerformanceScoringClaimed",
        "PerformanceScoringClaimMigrated",
    ):
        for event_id in events_by_type[event_type]:
            if event_id not in projected_claim_events:
                errors.append(
                    f"event {event_id}: {event_type} has no scoring claim projection"
                )

    legacy_exemptions: dict[str, sqlite3.Row] = {}
    for event in events_by_type["PerformanceScoringLegacyExempted"].values():
        label = f"legacy scoring exemption {event['event_id']}"
        payload = decode(event["payload_json"], f"{label} payload")
        metadata = decode(event["metadata_json"], f"{label} metadata")
        exact(
            payload,
            {"evaluation_id", "attempt_id", "command_hash", "reason"},
            f"{label} payload",
        )
        exact(
            metadata,
            {"migration_from_schema_version", "shadow_only"},
            f"{label} metadata",
        )
        if payload is None:
            continue
        evaluation_id = payload.get("evaluation_id")
        evaluation_row = evaluation_rows_by_id.get(evaluation_id)
        attempt = attempts.get(payload.get("attempt_id"))
        evaluation_event = (
            events_by_type["TaskEvaluationRecorded"].get(
                evaluation_row["event_id"]
            )
            if evaluation_row is not None
            else None
        )
        if evaluation_id in legacy_exemptions:
            errors.append(f"{label}: duplicate legacy evaluation exemption")
        else:
            legacy_exemptions[evaluation_id] = event
        if (
            evaluation_row is None
            or attempt is None
            or evaluation_event is None
            or evaluation_row["attempt_id"] != payload.get("attempt_id")
            or evaluation_row["command_hash"] != payload.get("command_hash")
            or evaluation_modes.get(evaluation_id)
            is not NormalizationMode.REGISTERED_PROVIDER
            or claim_rows_by_evaluation.get(evaluation_id)
            or payload.get("reason")
            != "schema_v14_predates_callback_claims"
            or metadata is None
            or metadata.get("migration_from_schema_version") != 14
            or metadata.get("shadow_only") is not True
            or event["schema_version"] != PERFORMANCE_EVENT_SCHEMA_VERSION
            or event["stream_id"] != f"learner:{attempt['learner_id']}"
            or event["learner_id"] != attempt["learner_id"]
            or event["session_id"] not in {None, attempt["session_id"]}
            or event["correlation_id"] != attempt["id"]
            or event["causation_id"] != evaluation_id
            or event["idempotency_key"]
            != "performance-score-legacy:v1:" + str(evaluation_id)
            or event["stream_version"] <= evaluation_event["stream_version"]
        ):
            errors.append(f"{label}: invalid schema-v14 exception boundary")

    for evaluation_id, mode in evaluation_modes.items():
        claims = claim_rows_by_evaluation.get(evaluation_id, [])
        exemption = legacy_exemptions.get(evaluation_id)
        if mode is NormalizationMode.REGISTERED_PROVIDER:
            if len(claims) + (1 if exemption is not None else 0) != 1:
                errors.append(
                    f"task evaluation {evaluation_id}: registered provider "
                    "does not have exactly one claim or schema-v14 exemption"
                )
        elif claims or exemption is not None:
            errors.append(
                f"task evaluation {evaluation_id}: direct import cannot have "
                "a provider-callback claim or legacy exemption"
            )

    bundle_rows = connection.execute(
        "SELECT * FROM shadow_evidence_bundles ORDER BY attempt_id, id"
    ).fetchall()
    projected_bundle_events: set[str] = set()
    for row in bundle_rows:
        label = f"shadow evidence bundle {row['id']}"
        evaluation = evaluations.get(row["evaluation_id"])
        attempt = attempts.get(row["attempt_id"])
        task = attempt_tasks.get(row["attempt_id"])
        evaluation_row = evaluation_rows_by_id.get(row["evaluation_id"])
        if evaluation is None or attempt is None or task is None or evaluation_row is None:
            errors.append(f"{label}: missing evaluation/attempt/task")
            continue
        event = events_by_type["ShadowEvidenceReduced"].get(row["event_id"])
        if event is None:
            errors.append(f"{label}: missing ShadowEvidenceReduced event")
            continue
        projected_bundle_events.add(event["event_id"])
        terms = decode(row["bundle_json"], f"{label} projection")
        event_payload = decode(event["payload_json"], f"{label} event payload")
        event_metadata = decode(event["metadata_json"], f"{label} event metadata")
        exact(
            event_payload,
            {
                "bundle_id",
                "evaluation_id",
                "attempt_id",
                "bundle_digest",
                "bundle",
                "projection_applied",
                "certification_applied",
            },
            f"{label} event payload",
        )
        exact(
            event_metadata,
            {"reducer", "task_release_id", "corpus_release_id", "shadow_only"},
            f"{label} event metadata",
        )
        trace = tuple(
            action
            for action in actions_by_attempt.get(row["attempt_id"], [])
            if action.sequence <= evaluation_row["through_sequence"]
        )
        try:
            expected_bundle = reduce_evidence(task, evaluation, trace).terms()
        except (TypeError, ValueError) as exc:
            errors.append(f"{label}: cannot recompute evidence ({exc})")
            expected_bundle = None
        if (
            terms is None
            or expected_bundle is None
            or canonical_json(terms) != canonical_json(expected_bundle)
            or row["bundle_digest"] != canonical_digest(expected_bundle)
            or row["id"] != "seb_" + canonical_digest(expected_bundle)[:24]
            or row["projection_applied"] != 0
            or row["certification_applied"] != 0
        ):
            errors.append(f"{label}: deterministic shadow reduction mismatch")
        if event_payload is not None and (
            event_payload.get("bundle_id") != row["id"]
            or event_payload.get("evaluation_id") != row["evaluation_id"]
            or event_payload.get("attempt_id") != row["attempt_id"]
            or event_payload.get("bundle_digest") != row["bundle_digest"]
            or canonical_json(event_payload.get("bundle")) != canonical_json(terms)
            or event_payload.get("projection_applied") is not False
            or event_payload.get("certification_applied") is not False
        ):
            errors.append(f"{label}: event payload mismatch")
        if event_metadata is not None and (
            event_metadata.get("reducer") != "deterministic-evidence-v2"
            or event_metadata.get("task_release_id") != attempt["task_release_id"]
            or event_metadata.get("corpus_release_id") != attempt["corpus_release_id"]
            or event_metadata.get("shadow_only") is not True
        ):
            errors.append(f"{label}: event metadata mismatch")
        if (
            event["schema_version"] != PERFORMANCE_EVENT_SCHEMA_VERSION
            or event["stream_id"] != f"learner:{attempt['learner_id']}"
            or event["learner_id"] != attempt["learner_id"]
            or event["session_id"] != attempt["session_id"]
            or event["correlation_id"] != attempt["id"]
            or event["causation_id"] != row["evaluation_id"]
            or event["recorded_at"] != row["recorded_at"]
        ):
            errors.append(f"{label}: event envelope mismatch")
        evaluation_event = events_by_type["TaskEvaluationRecorded"].get(
            evaluation_row["event_id"]
        )
        if (
            evaluation_event is None
            or evaluation_event["stream_id"] != event["stream_id"]
            or evaluation_event["stream_version"] >= event["stream_version"]
        ):
            errors.append(f"{label}: event does not follow its evaluation")
        try:
            bundle_occurred_at = _aware_timestamp(
                event["occurred_at"], f"{label} event occurrence"
            )
            if evaluation_event is not None and bundle_occurred_at < (
                _aware_timestamp(
                    evaluation_event["occurred_at"],
                    f"{label} evaluation occurrence",
                )
            ):
                errors.append(f"{label}: occurrence precedes its evaluation")
            _aware_timestamp(row["recorded_at"], f"{label} recorded_at")
        except ValidationError as exc:
            errors.append(str(exc))

    for event_id in events_by_type["ShadowEvidenceReduced"]:
        if event_id not in projected_bundle_events:
            errors.append(
                f"event {event_id}: ShadowEvidenceReduced has no bundle projection"
            )
    return errors


_PERFORMANCE_PROJECTION_TABLES = (
    "performance_attempts",
    "performance_actions",
    "performance_scoring_claims",
    "task_evaluations",
    "shadow_evidence_bundles",
)


def performance_projection_snapshot(
    connection: sqlite3.Connection,
) -> dict[str, list[dict[str, Any]]]:
    """Return the exact mutable shadow projections in stable row order."""

    queries = {
        "attempts": "SELECT * FROM performance_attempts ORDER BY id",
        "actions": (
            "SELECT * FROM performance_actions "
            "ORDER BY attempt_id, sequence, id"
        ),
        "scoring_claims": (
            "SELECT * FROM performance_scoring_claims ORDER BY id"
        ),
        "evaluations": (
            "SELECT * FROM task_evaluations "
            "ORDER BY attempt_id, recorded_at, id"
        ),
        "bundles": (
            "SELECT * FROM shadow_evidence_bundles "
            "ORDER BY attempt_id, evaluation_id, id"
        ),
    }
    snapshot: dict[str, list[dict[str, Any]]] = {}
    for name, query in queries.items():
        if name == "scoring_claims" and connection.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type='table' AND name='performance_scoring_claims'"""
        ).fetchone() is None:
            # Exact v14 migration fixtures predate callback admission.  An
            # absent table and the newly installed empty projection represent
            # the same historical state; no row is synthesized.
            snapshot[name] = []
            continue
        snapshot[name] = [
            _row_dict(row) for row in connection.execute(query).fetchall()
        ]
    return snapshot


def derive_performance_projections(
    connection: sqlite3.Connection,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Derive shadow projection rows exclusively from immutable events."""

    attempts: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    scoring_claims: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    events = connection.execute(
        """SELECT * FROM events WHERE event_type IN (
               'PerformanceTaskStarted', 'PerformanceActionRecorded',
               'PerformanceScoringClaimed',
               'PerformanceScoringClaimMigrated',
               'PerformanceScoringLegacyExempted',
               'TaskEvaluationRecorded', 'ShadowEvidenceReduced'
           ) ORDER BY stream_id, stream_version"""
    ).fetchall()
    for event in events:
        if event["schema_version"] != PERFORMANCE_EVENT_SCHEMA_VERSION:
            raise ValidationError(
                f"Performance event {event['event_id']} uses unsupported schema "
                f"{event['schema_version']}."
            )
        payload = _json_object(
            event["payload_json"], f"Performance event {event['event_id']} payload"
        )
        metadata = _json_object(
            event["metadata_json"],
            f"Performance event {event['event_id']} metadata",
        )
        event_type = event["event_type"]
        if event_type == "PerformanceTaskStarted":
            _require_exact_fields(
                payload,
                frozenset(
                    {
                        "attempt_id",
                        "session_id",
                        "learner_id",
                        "task_release_id",
                        "corpus_release_id",
                        "task_id",
                        "task_version",
                        "task_digest",
                        "session_revision",
                        "learner_revision",
                    }
                ),
                f"Performance event {event['event_id']} payload",
            )
            _require_exact_fields(
                metadata,
                frozenset(
                    {
                        "command_hash",
                        "task_schema_version",
                        "shadow_only",
                        "projection_applied",
                        "certification_applied",
                    }
                ),
                f"Performance event {event['event_id']} metadata",
            )
            if (
                metadata["shadow_only"] is not True
                or metadata["projection_applied"] is not False
                or metadata["certification_applied"] is not False
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} is not shadow-only."
                )
            attempts.append(
                {
                    "id": payload["attempt_id"],
                    "event_id": event["event_id"],
                    "task_release_id": payload["task_release_id"],
                    "corpus_release_id": payload["corpus_release_id"],
                    "task_id": payload["task_id"],
                    "task_version": payload["task_version"],
                    "task_digest": payload["task_digest"],
                    "session_id": payload["session_id"],
                    "learner_id": payload["learner_id"],
                    "session_revision": payload["session_revision"],
                    "learner_revision": payload["learner_revision"],
                    "started_at": event["occurred_at"],
                    "recorded_at": event["recorded_at"],
                    "command_hash": metadata["command_hash"],
                }
            )
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event_type,
                    "attempt_id": payload["attempt_id"],
                }
            )
        elif event_type == "PerformanceActionRecorded":
            _require_exact_fields(
                payload,
                frozenset({"attempt_id", "action"}),
                f"Performance event {event['event_id']} payload",
            )
            _require_exact_fields(
                metadata,
                frozenset(
                    {
                        "command_hash",
                        "action_schema_version",
                        "task_digest",
                        "task_release_id",
                        "corpus_release_id",
                        "observational_only",
                        "shadow_only",
                    }
                ),
                f"Performance event {event['event_id']} metadata",
            )
            try:
                action = LearningAction.from_terms(payload["action"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Performance event {event['event_id']} action is invalid: {exc}"
                ) from exc
            if (
                action.trace_id != payload["attempt_id"]
                or metadata["action_schema_version"] != action.schema_version
                or metadata["observational_only"] is not True
                or metadata["shadow_only"] is not True
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} action boundary mismatch."
                )
            actions.append(
                {
                    "id": action.id,
                    "event_id": event["event_id"],
                    "attempt_id": action.trace_id,
                    "sequence": action.sequence,
                    "phase": action.phase.value,
                    "action_type": action.kind.value,
                    "payload_json": canonical_json(action.terms()["payload"]),
                    "elapsed_ms": action.elapsed_ms,
                    "occurred_at": event["occurred_at"],
                    "recorded_at": event["recorded_at"],
                    "command_hash": metadata["command_hash"],
                }
            )
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event_type,
                    "attempt_id": action.trace_id,
                    "action_id": action.id,
                    "sequence": action.sequence,
                }
            )
        elif event_type in {
            "PerformanceScoringClaimed",
            "PerformanceScoringClaimMigrated",
        }:
            _require_exact_fields(
                payload,
                frozenset(
                    {
                        "claim_id",
                        "caller_idempotency_key",
                        "attempt_id",
                        "evaluation_id",
                        "through_sequence",
                        "provider_id",
                        "provider_version",
                        "action_trace_digest",
                        "command_hash",
                        "claimed_at",
                    }
                ),
                f"Performance event {event['event_id']} payload",
            )
            _require_exact_fields(
                metadata,
                frozenset(
                    {
                        "claim_schema_version",
                        "admission_mode",
                        "source_schema_version",
                        "shadow_only",
                    }
                ),
                f"Performance event {event['event_id']} metadata",
            )
            expected_mode = (
                "pre_callback"
                if event_type == "PerformanceScoringClaimed"
                else "legacy_projection_migration"
            )
            if (
                metadata["claim_schema_version"] != 1
                or metadata["admission_mode"] != expected_mode
                or metadata["source_schema_version"]
                != (None if expected_mode == "pre_callback" else 15)
                or metadata["shadow_only"] is not True
                or event["idempotency_key"]
                != performance_scoring_claim_event_key(
                    payload["command_hash"]
                )
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} scoring claim "
                    "boundary mismatch."
                )
            scoring_claims.append(
                {
                    "id": payload["claim_id"],
                    "event_id": event["event_id"],
                    "idempotency_key": payload["caller_idempotency_key"],
                    "attempt_id": payload["attempt_id"],
                    "evaluation_id": payload["evaluation_id"],
                    "through_sequence": payload["through_sequence"],
                    "provider_id": payload["provider_id"],
                    "provider_version": payload["provider_version"],
                    "action_trace_digest": payload["action_trace_digest"],
                    "command_hash": payload["command_hash"],
                    "claimed_at": payload["claimed_at"],
                }
            )
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event_type,
                    "attempt_id": payload["attempt_id"],
                    "claim_id": payload["claim_id"],
                    "evaluation_id": payload["evaluation_id"],
                }
            )
        elif event_type == "PerformanceScoringLegacyExempted":
            _require_exact_fields(
                payload,
                frozenset(
                    {"evaluation_id", "attempt_id", "command_hash", "reason"}
                ),
                f"Performance event {event['event_id']} payload",
            )
            _require_exact_fields(
                metadata,
                frozenset(
                    {"migration_from_schema_version", "shadow_only"}
                ),
                f"Performance event {event['event_id']} metadata",
            )
            if (
                payload["reason"]
                != "schema_v14_predates_callback_claims"
                or metadata["migration_from_schema_version"] != 14
                or metadata["shadow_only"] is not True
                or event["idempotency_key"]
                != "performance-score-legacy:v1:" + payload["evaluation_id"]
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} legacy scoring "
                    "exception mismatch."
                )
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event_type,
                    "attempt_id": payload["attempt_id"],
                    "evaluation_id": payload["evaluation_id"],
                }
            )
        elif event_type == "TaskEvaluationRecorded":
            _require_exact_fields(
                payload,
                frozenset(
                    {
                        "attempt_id",
                        "through_sequence",
                        "evaluation_digest",
                        "evaluation",
                        "authority",
                    }
                ),
                f"Performance event {event['event_id']} payload",
            )
            _require_exact_fields(
                metadata,
                frozenset(
                    {
                        "command_hash",
                        "task_release_id",
                        "corpus_release_id",
                        "shadow_only",
                        "projection_applied",
                        "certification_applied",
                    }
                ),
                f"Performance event {event['event_id']} metadata",
            )
            try:
                evaluation = TaskEvaluation.from_terms(payload["evaluation"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Performance event {event['event_id']} evaluation is invalid: {exc}"
                ) from exc
            if (
                evaluation.trace_id != payload["attempt_id"]
                or evaluation.digest != payload["evaluation_digest"]
                or type(payload["through_sequence"]) is not int
                or payload["through_sequence"] < 0
                or type(payload["authority"]) is not dict
                or metadata["shadow_only"] is not True
                or metadata["projection_applied"] is not False
                or metadata["certification_applied"] is not False
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} evaluation boundary mismatch."
                )
            evaluations.append(
                {
                    "id": evaluation.id,
                    "event_id": event["event_id"],
                    "attempt_id": evaluation.trace_id,
                    "through_sequence": payload["through_sequence"],
                    "evaluation_digest": evaluation.digest,
                    "evaluation_json": canonical_json(evaluation.terms()),
                    "authority_json": canonical_json(payload["authority"]),
                    "recorded_at": event["recorded_at"],
                    "command_hash": metadata["command_hash"],
                }
            )
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event_type,
                    "attempt_id": evaluation.trace_id,
                    "evaluation_id": evaluation.id,
                }
            )
        else:
            _require_exact_fields(
                payload,
                frozenset(
                    {
                        "bundle_id",
                        "evaluation_id",
                        "attempt_id",
                        "bundle_digest",
                        "bundle",
                        "projection_applied",
                        "certification_applied",
                    }
                ),
                f"Performance event {event['event_id']} payload",
            )
            _require_exact_fields(
                metadata,
                frozenset(
                    {"reducer", "task_release_id", "corpus_release_id", "shadow_only"}
                ),
                f"Performance event {event['event_id']} metadata",
            )
            if (
                type(payload["bundle"]) is not dict
                or canonical_digest(payload["bundle"]) != payload["bundle_digest"]
                or payload["projection_applied"] is not False
                or payload["certification_applied"] is not False
                or metadata["shadow_only"] is not True
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} shadow bundle mismatch."
                )
            bundles.append(
                {
                    "id": payload["bundle_id"],
                    "event_id": event["event_id"],
                    "evaluation_id": payload["evaluation_id"],
                    "attempt_id": payload["attempt_id"],
                    "bundle_digest": payload["bundle_digest"],
                    "bundle_json": canonical_json(payload["bundle"]),
                    "projection_applied": 0,
                    "certification_applied": 0,
                    "recorded_at": event["recorded_at"],
                }
            )
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event_type,
                    "attempt_id": payload["attempt_id"],
                    "evaluation_id": payload["evaluation_id"],
                    "bundle_id": payload["bundle_id"],
                }
            )
    snapshot = {
        "attempts": sorted(attempts, key=lambda item: item["id"]),
        "actions": sorted(
            actions,
            key=lambda item: (item["attempt_id"], item["sequence"], item["id"]),
        ),
        "scoring_claims": sorted(
            scoring_claims,
            key=lambda item: item["id"],
        ),
        "evaluations": sorted(
            evaluations,
            key=lambda item: (item["attempt_id"], item["recorded_at"], item["id"]),
        ),
        "bundles": sorted(
            bundles,
            key=lambda item: (
                item["attempt_id"],
                item["evaluation_id"],
                item["id"],
            ),
        ),
    }
    return snapshot, checkpoints


def rebuild_performance_projections(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    """Replace mutable shadow projections from events on a disposable copy."""

    snapshot, checkpoints = derive_performance_projections(connection)
    trigger_rows = connection.execute(
        """SELECT name, sql FROM sqlite_master
           WHERE type='trigger' AND tbl_name IN (
               'performance_attempts', 'performance_actions',
               'performance_scoring_claims', 'task_evaluations',
               'shadow_evidence_bundles'
           ) ORDER BY name"""
    ).fetchall()
    for trigger in trigger_rows:
        escaped = trigger["name"].replace('"', '""')
        connection.execute(f'DROP TRIGGER "{escaped}"')
    connection.execute("DELETE FROM shadow_evidence_bundles")
    connection.execute("DELETE FROM task_evaluations")
    connection.execute("DELETE FROM performance_scoring_claims")
    connection.execute("DELETE FROM performance_actions")
    connection.execute("DELETE FROM performance_attempts")
    connection.executemany(
        """INSERT INTO performance_attempts(
               id, event_id, task_release_id, corpus_release_id, task_id,
               task_version, task_digest, session_id, learner_id,
               session_revision, learner_revision, started_at, recorded_at,
               command_hash
           ) VALUES (
               :id, :event_id, :task_release_id, :corpus_release_id, :task_id,
               :task_version, :task_digest, :session_id, :learner_id,
               :session_revision, :learner_revision, :started_at, :recorded_at,
               :command_hash
           )""",
        snapshot["attempts"],
    )
    connection.executemany(
        """INSERT INTO performance_actions(
               id, event_id, attempt_id, sequence, phase, action_type,
               payload_json, elapsed_ms, occurred_at, recorded_at, command_hash
           ) VALUES (
               :id, :event_id, :attempt_id, :sequence, :phase, :action_type,
               :payload_json, :elapsed_ms, :occurred_at, :recorded_at, :command_hash
           )""",
        snapshot["actions"],
    )
    connection.executemany(
        """INSERT INTO performance_scoring_claims(
               id, event_id, idempotency_key, attempt_id, evaluation_id,
               through_sequence, provider_id, provider_version,
               action_trace_digest, command_hash, claimed_at
           ) VALUES (
               :id, :event_id, :idempotency_key, :attempt_id, :evaluation_id,
               :through_sequence, :provider_id, :provider_version,
               :action_trace_digest, :command_hash, :claimed_at
           )""",
        snapshot["scoring_claims"],
    )
    connection.executemany(
        """INSERT INTO task_evaluations(
               id, event_id, attempt_id, through_sequence, evaluation_digest,
               evaluation_json, authority_json, recorded_at, command_hash
           ) VALUES (
               :id, :event_id, :attempt_id, :through_sequence, :evaluation_digest,
               :evaluation_json, :authority_json, :recorded_at, :command_hash
           )""",
        snapshot["evaluations"],
    )
    connection.executemany(
        """INSERT INTO shadow_evidence_bundles(
               id, event_id, evaluation_id, attempt_id, bundle_digest,
               bundle_json, projection_applied, certification_applied, recorded_at
           ) VALUES (
               :id, :event_id, :evaluation_id, :attempt_id, :bundle_digest,
               :bundle_json, :projection_applied, :certification_applied, :recorded_at
           )""",
        snapshot["bundles"],
    )
    for trigger in trigger_rows:
        if trigger["sql"] is None:
            raise ValidationError(
                f"Performance projection trigger {trigger['name']} has no SQL."
            )
        connection.execute(trigger["sql"])
    return {
        "snapshot": performance_projection_snapshot(connection),
        "checkpoints": checkpoints,
        "attempt_count": len(snapshot["attempts"]),
        "action_count": len(snapshot["actions"]),
        "scoring_claim_count": len(snapshot["scoring_claims"]),
        "evaluation_count": len(snapshot["evaluations"]),
        "bundle_count": len(snapshot["bundles"]),
        "projection_hash": canonical_digest(snapshot),
    }


__all__ = [
    "MAX_TASK_RELEASE_BYTES",
    "PERFORMANCE_EVENT_SCHEMA_VERSION",
    "SERVICEABLE_TASK_STATUSES",
    "TASK_RELEASE_SCHEMA_VERSION",
    "TASK_STATUSES",
    "PerformanceLedger",
    "PerformanceTaskRelease",
    "TaskReleaseReview",
    "derive_performance_projections",
    "performance_integrity_errors",
    "performance_projection_snapshot",
    "read_task_release",
    "rebuild_performance_projections",
]
