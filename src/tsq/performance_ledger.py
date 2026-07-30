# SPDX-License-Identifier: MPL-2.0

"""Immutable, replayable shadow ledger for productive-skill tasks.

This module operationalizes the pure contracts in :mod:`tsq.evidence` without
promoting rubric observations into learner mastery.  Task releases are pinned
to an immutable curriculum release; attempts, semantic actions, evaluations,
and reduced bundles are committed to the learner event stream.  The only
artifact checking admitted here sends one inert, bounded data snapshot to a
fixed bundled checker in a separate process.  It never executes learner code,
commands, plugins, tests, or model calls, and it remains explicitly
non-authoritative.
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

from .artifact_intake import ProductiveArtifactSnapshot
from .artifact_runner import (
    ArtifactProcessReceipt,
    ArtifactRunOutcome,
    ArtifactRunRequest,
    ArtifactRunnerBinding,
    ArtifactRunnerProtocolError,
    SyntheticArtifactRunnerRegistry,
    build_artifact_run_request,
    bundled_synthetic_binding,
)
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
    ImportedCriterionResult,
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
from .performance_boundaries import (
    missing_objective_misconception_bindings,
    release_misconception_objectives,
)
from .reconciliation import (
    ReconciliationOutcome,
    RegisteredReconciler,
    ScoringReconciliationRegistry,
    ScoringReconciliationReceipt,
    ScoringReconciliationRequest,
    provider_scoring_operation_digest,
)
from .store import (
    PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
    Database,
    from_timestamp,
    new_id,
    performance_artifact_run_claim_event_key,
    performance_artifact_run_claim_payload,
    performance_artifact_run_observed_payload,
    performance_artifact_run_receipt_event_key,
    performance_scoring_claim_event_key,
    performance_scoring_claim_payload,
    performance_scoring_claim_v2_payload,
    performance_scoring_reconciliation_event_key,
    performance_scoring_reconciliation_payload,
    to_timestamp,
)


TASK_RELEASE_SCHEMA_VERSION = 1
SYNTHETIC_TASK_LAB_RELEASE_SCHEMA_VERSION = 2
PERFORMANCE_EVENT_SCHEMA_VERSION = 1
TASK_STATUSES = frozenset({"quarantined", "pilot", "approved"})
SERVICEABLE_TASK_STATUSES = frozenset({"pilot", "approved"})
MAX_TASK_RELEASE_BYTES = 16 * 1024 * 1024
PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION = 1
_PERFORMANCE_TECHNICAL_EVENT_KEY_PREFIXES = (
    PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
    PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
)
_PERFORMANCE_TRACE_EVENT_TYPES = (
    "PerformanceTaskStarted",
    "PerformanceActionRecorded",
    "PerformanceArtifactRunClaimed",
    "PerformanceArtifactRunObserved",
)
_PERFORMANCE_TRACE_EVENT_TYPES_SQL = ", ".join(
    f"'{event_type}'" for event_type in _PERFORMANCE_TRACE_EVENT_TYPES
)

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
_SYNTHETIC_LAB_DECLARATION_FIELDS = frozenset(
    {
        "declaration_kind",
        "producer_id",
        "producer_version",
        "declared_at",
        "manifest_digest",
        "human_reviewed",
        "activation_authority",
    }
)
_BUNDLE_FIELDS = frozenset(
    {"schema_version", "title", "corpus_release_id", "review", "tasks"}
)
_TASK_ENTRY_FIELDS = frozenset({"status", "task"})
_OPERATIONAL_ARTIFACT_RUN_RECEIPT_FIELDS = frozenset(
    {
        "claim_id",
        "attempt_id",
        "artifact_action_id",
        "artifact_digest",
        "artifact_kind",
        "artifact_manifest_digest",
        "check_set_id",
        "check_set_manifest_digest",
        "runner_id",
        "runner_version",
        "outcome",
        "started_at",
        "completed_at",
        "result_digest",
        "request_digest",
        "binding_digest",
        "schema_version",
    }
)


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
class OperationalArtifactRunReceipt:
    """Ledger receipt binding one terminal process observation to its claim."""

    claim_id: str
    attempt_id: str
    artifact_action_id: str
    artifact_digest: str
    artifact_kind: str
    artifact_manifest_digest: str
    check_set_id: str
    check_set_manifest_digest: str
    runner_id: str
    runner_version: str
    outcome: str
    started_at: str
    completed_at: str
    result_digest: str | None
    request_digest: str
    binding_digest: str
    schema_version: int = PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "claim_id",
            "attempt_id",
            "artifact_action_id",
            "artifact_kind",
            "check_set_id",
            "runner_id",
            "runner_version",
        ):
            _require_id(
                getattr(self, field_name),
                f"Artifact run receipt {field_name}",
            )
        for field_name in (
            "artifact_digest",
            "artifact_manifest_digest",
            "check_set_manifest_digest",
            "request_digest",
            "binding_digest",
        ):
            _require_digest(
                getattr(self, field_name),
                f"Artifact run receipt {field_name}",
            )
        if self.outcome not in {
            ArtifactRunOutcome.COMPLETED.value,
            ArtifactRunOutcome.INVALID_ARTIFACT.value,
            "runner_failed",
            ArtifactRunOutcome.TIMED_OUT.value,
        }:
            raise ValidationError("Artifact run receipt has an unknown outcome.")
        started = _aware_timestamp(
            self.started_at, "Artifact run receipt started_at"
        )
        completed = _aware_timestamp(
            self.completed_at, "Artifact run receipt completed_at"
        )
        if completed < started:
            raise ValidationError(
                "Artifact run receipt cannot complete before it starts."
            )
        if self.outcome in {
            ArtifactRunOutcome.COMPLETED.value,
            ArtifactRunOutcome.INVALID_ARTIFACT.value,
        }:
            _require_digest(
                self.result_digest,
                "Artifact run receipt result_digest",
            )
        elif self.result_digest is not None:
            raise ValidationError(
                "Failed artifact runs cannot carry a result digest."
            )
        if self.schema_version != PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION:
            raise ValidationError(
                "Artifact run receipt has an unsupported schema_version."
            )

    @property
    def digest(self) -> str:
        return canonical_digest(self.terms())

    def terms(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "attempt_id": self.attempt_id,
            "artifact_action_id": self.artifact_action_id,
            "artifact_digest": self.artifact_digest,
            "artifact_kind": self.artifact_kind,
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "check_set_id": self.check_set_id,
            "check_set_manifest_digest": self.check_set_manifest_digest,
            "runner_id": self.runner_id,
            "runner_version": self.runner_version,
            "outcome": self.outcome,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result_digest": self.result_digest,
            "request_digest": self.request_digest,
            "binding_digest": self.binding_digest,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_terms(cls, value: object) -> "OperationalArtifactRunReceipt":
        if type(value) is not dict:
            raise ValidationError("Artifact run receipt must be an object.")
        _require_exact_fields(
            value,
            _OPERATIONAL_ARTIFACT_RUN_RECEIPT_FIELDS,
            "Artifact run receipt",
        )
        receipt = cls(
            claim_id=value["claim_id"],
            attempt_id=value["attempt_id"],
            artifact_action_id=value["artifact_action_id"],
            artifact_digest=value["artifact_digest"],
            artifact_kind=value["artifact_kind"],
            artifact_manifest_digest=value["artifact_manifest_digest"],
            check_set_id=value["check_set_id"],
            check_set_manifest_digest=value["check_set_manifest_digest"],
            runner_id=value["runner_id"],
            runner_version=value["runner_version"],
            outcome=value["outcome"],
            started_at=value["started_at"],
            completed_at=value["completed_at"],
            result_digest=value["result_digest"],
            request_digest=value["request_digest"],
            binding_digest=value["binding_digest"],
            schema_version=value["schema_version"],
        )
        if canonical_json(receipt.terms()) != canonical_json(value):
            raise ValidationError("Artifact run receipt is not canonical.")
        return receipt


@dataclass(frozen=True, slots=True)
class TaskReleaseReview:
    """Independent human commitment required for a serviceable v1 release."""

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
class SyntheticTaskLabDeclaration:
    """Non-authoritative provenance for quarantined laboratory fixtures.

    This is deliberately not a review.  It cannot attest independence,
    authorize activation, or make a task serviceable.  The distinct v2 release
    schema prevents a synthetic declaration from being interpreted as the
    historical v1 human-review commitment.
    """

    producer_id: str
    producer_version: str
    declared_at: str
    manifest_digest: str

    def __post_init__(self) -> None:
        _require_id(self.producer_id, "declaration.producer_id")
        _require_id(self.producer_version, "declaration.producer_version")
        _aware_timestamp(self.declared_at, "declaration.declared_at")
        _require_digest(
            self.manifest_digest, "declaration.manifest_digest"
        )

    def terms(self) -> dict[str, Any]:
        return {
            "declaration_kind": "synthetic_lab",
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "declared_at": self.declared_at,
            "manifest_digest": self.manifest_digest,
            "human_reviewed": False,
            "activation_authority": False,
        }

    @classmethod
    def from_terms(cls, value: object) -> "SyntheticTaskLabDeclaration":
        if type(value) is not dict:
            raise ValidationError(
                "Synthetic laboratory declaration must be an object."
            )
        _require_exact_fields(
            value,
            _SYNTHETIC_LAB_DECLARATION_FIELDS,
            "Synthetic laboratory declaration",
        )
        if value["declaration_kind"] != "synthetic_lab":
            raise ValidationError(
                "Synthetic laboratory declaration kind must be synthetic_lab."
            )
        if value["human_reviewed"] is not False:
            raise ValidationError(
                "Synthetic laboratory content cannot claim human review."
            )
        if value["activation_authority"] is not False:
            raise ValidationError(
                "Synthetic laboratory content cannot claim activation authority."
            )
        declaration = cls(
            producer_id=value["producer_id"],
            producer_version=value["producer_version"],
            declared_at=value["declared_at"],
            manifest_digest=value["manifest_digest"],
        )
        if canonical_json(declaration.terms()) != canonical_json(value):
            raise ValidationError(
                "Synthetic laboratory declaration is not canonical."
            )
        return declaration


def _release_authority_from_terms(
    value: object,
) -> TaskReleaseReview | SyntheticTaskLabDeclaration:
    if type(value) is not dict:
        raise ValidationError("Release authority must be an object.")
    if "reviewer_kind" in value:
        return TaskReleaseReview.from_terms(value)
    if "declaration_kind" in value:
        return SyntheticTaskLabDeclaration.from_terms(value)
    raise ValidationError(
        "Release authority is neither a human review nor a synthetic "
        "laboratory declaration."
    )


@dataclass(frozen=True, slots=True)
class PerformanceTaskRelease:
    """Canonical bundle of immutable productive-skill task definitions."""

    title: str
    corpus_release_id: str
    review: TaskReleaseReview | SyntheticTaskLabDeclaration
    tasks: tuple[tuple[str, LearningTask], ...]
    schema_version: int = TASK_RELEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.title, "title", 256)
        _require_id(self.corpus_release_id, "corpus_release_id")
        if type(self.review) not in {
            TaskReleaseReview,
            SyntheticTaskLabDeclaration,
        }:
            raise ValidationError(
                "review must be a TaskReleaseReview or "
                "SyntheticTaskLabDeclaration."
            )
        if type(self.tasks) is not tuple or not self.tasks:
            raise ValidationError("tasks must be a non-empty tuple.")
        normalized: list[tuple[str, LearningTask]] = []
        keys: set[tuple[str, int]] = set()
        for status, task in self.tasks:
            if status not in TASK_STATUSES:
                raise ValidationError(
                    "Task status must be quarantined, pilot, or approved."
                )
            if type(task) is not LearningTask:
                raise ValidationError(
                    "tasks must contain exact LearningTask values."
                )
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
        expected_schema_version = (
            TASK_RELEASE_SCHEMA_VERSION
            if type(self.review) is TaskReleaseReview
            else SYNTHETIC_TASK_LAB_RELEASE_SCHEMA_VERSION
        )
        if (
            type(self.schema_version) is not int
            or self.schema_version != expected_schema_version
        ):
            raise ValidationError(
                f"schema_version must be {expected_schema_version} for this "
                "release authority."
            )
        if type(self.review) is SyntheticTaskLabDeclaration:
            non_quarantined = [
                task.id
                for status, task in self.tasks
                if status != "quarantined"
            ]
            if non_quarantined:
                raise ValidationError(
                    "Synthetic laboratory releases may contain only "
                    "quarantined tasks: "
                    + ", ".join(sorted(non_quarantined))
                    + "."
                )
            unsafe_tasks: list[str] = []
            for _status, task in self.tasks:
                if (
                    task.evidence_cap != 0.0
                    or task.scorer_contracts
                    or any(
                        criterion.evidence_cap != 0.0
                        or criterion.dependence_cap != 0.0
                        or criterion.assisted_evidence_factor != 0.0
                        or criterion.certification_eligible is not False
                        for criterion in task.criteria
                    )
                ):
                    unsafe_tasks.append(task.id)
            if unsafe_tasks:
                raise ValidationError(
                    "Synthetic laboratory tasks must have zero evidence and "
                    "dependence caps, no scorer contracts, zero assisted "
                    "evidence, and no certification eligibility: "
                    + ", ".join(sorted(unsafe_tasks))
                    + "."
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
        schema_version = value["schema_version"]
        if (
            type(schema_version) is int
            and schema_version == TASK_RELEASE_SCHEMA_VERSION
        ):
            authority: TaskReleaseReview | SyntheticTaskLabDeclaration = (
                TaskReleaseReview.from_terms(value["review"])
            )
        elif (
            type(schema_version) is int
            and schema_version
            == SYNTHETIC_TASK_LAB_RELEASE_SCHEMA_VERSION
        ):
            authority = SyntheticTaskLabDeclaration.from_terms(
                value["review"]
            )
        else:
            raise ValidationError(
                "Performance-task release schema_version must be "
                f"{TASK_RELEASE_SCHEMA_VERSION} or "
                f"{SYNTHETIC_TASK_LAB_RELEASE_SCHEMA_VERSION}."
            )
        release = cls(
            title=value["title"],
            corpus_release_id=value["corpus_release_id"],
            review=authority,
            tasks=tuple(tasks),
            schema_version=schema_version,
        )
        if canonical_json(release.terms()) != canonical_json(value):
            raise ValidationError(
                "Performance-task release is not canonical."
            )
        return release


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


def _validate_release_corpus_bindings(
    connection: sqlite3.Connection,
    bundle: PerformanceTaskRelease,
    *,
    require_current_live_bindings: bool,
) -> None:
    """Validate every task binding against one immutable corpus release."""

    corpus_release = connection.execute(
        "SELECT sealed_at FROM corpus_releases WHERE id=?",
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
    all_misconception_objectives = release_misconception_objectives(
        connection,
        bundle.corpus_release_id,
        accepted_only=False,
        exclude_revoked=False,
    )
    live_misconception_objectives = release_misconception_objectives(
        connection,
        bundle.corpus_release_id,
        accepted_only=True,
        exclude_revoked=True,
    )
    accepted_misconception_objectives = release_misconception_objectives(
        connection,
        bundle.corpus_release_id,
        accepted_only=True,
        exclude_revoked=False,
    )
    for task_status, task in bundle.tasks:
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
                primary_concept_id = release_objectives.get(objective_id)
                if primary_concept_id is None:
                    raise ValidationError(
                        f"Task {task.id} criterion {criterion.id} references "
                        f"objective {objective_id} outside its release."
                    )
                if primary_concept_id not in criterion.concept_ids:
                    raise ValidationError(
                        f"Task {task.id} criterion {criterion.id} objective "
                        f"{objective_id} has primary concept "
                        f"{primary_concept_id} outside that criterion's "
                        "concept mapping."
                    )
        unknown_misconceptions = (
            set(task.misconception_ids) - release_misconceptions
        )
        if unknown_misconceptions:
            raise ValidationError(
                f"Task {task.id} references misconceptions outside its "
                "release: "
                + ", ".join(sorted(unknown_misconceptions))
            )
        task_misconception_objectives = (
            (
                live_misconception_objectives
                if require_current_live_bindings
                else accepted_misconception_objectives
            )
            if task_status in SERVICEABLE_TASK_STATUSES
            else all_misconception_objectives
        )
        for criterion in task.criteria:
            criterion_concepts = set(criterion.concept_ids)
            for misconception_id in criterion.misconception_ids:
                if (
                    not criterion.objective_ids
                    and release_misconception_concepts.get(
                        misconception_id
                    )
                    not in criterion_concepts
                ):
                    raise ValidationError(
                        f"Task {task.id} criterion {criterion.id} "
                        f"misconception {misconception_id} is outside that "
                        "criterion's concept mapping."
                    )
        missing_bindings = missing_objective_misconception_bindings(
            task,
            task_misconception_objectives,
        )
        if missing_bindings:
            criterion_id, misconception_id = missing_bindings[0]
            raise ValidationError(
                f"Task {task.id} criterion {criterion_id} misconception "
                f"{misconception_id} is not mapped to any of that "
                "criterion's objectives in the pinned release."
            )


def load_stored_task_release(
    connection: sqlite3.Connection,
    release_id: str,
) -> tuple[PerformanceTaskRelease, datetime]:
    """Strictly reconstruct one stored release and its publication time."""

    _require_id(release_id, "release_id")
    row = connection.execute(
        "SELECT * FROM performance_task_releases WHERE id=?",
        (release_id,),
    ).fetchone()
    if row is None:
        raise NotFoundError(
            f"Performance task release {release_id} does not exist."
        )
    review = _release_authority_from_terms(
        _json_object(row["review_json"], f"Release {release_id} authority")
    )
    member_rows = connection.execute(
        """SELECT member.*, task.task_digest AS stored_task_digest,
                  task.definition_json, task.imported_at
           FROM release_performance_tasks member
           LEFT JOIN performance_tasks task
             ON task.task_id=member.task_id
            AND task.task_version=member.task_version
           WHERE member.release_id=?
           ORDER BY member.task_id, member.task_version""",
        (release_id,),
    ).fetchall()
    created_at = _aware_timestamp(
        row["created_at"], f"Release {release_id} created_at"
    )
    sealed_at = _aware_timestamp(
        row["sealed_at"], f"Release {release_id} sealed_at"
    )
    if row["sealed_at"] != row["created_at"] or sealed_at != created_at:
        raise ValidationError(
            f"Release {release_id} seal boundary differs from publication."
        )
    authority_at = _aware_timestamp(
        (
            review.reviewed_at
            if type(review) is TaskReleaseReview
            else review.declared_at
        ),
        f"Release {release_id} authority timestamp",
    )
    if authority_at > created_at:
        raise ValidationError(
            f"Release {release_id} publication precedes its authority."
        )
    corpus_release = connection.execute(
        "SELECT sealed_at FROM corpus_releases WHERE id=?",
        (row["corpus_release_id"],),
    ).fetchone()
    if corpus_release is None:
        raise NotFoundError(
            f"Corpus release {row['corpus_release_id']} does not exist."
        )
    corpus_sealed_at = _aware_timestamp(
        corpus_release["sealed_at"],
        f"Corpus release {row['corpus_release_id']} sealed_at",
    )
    if created_at < corpus_sealed_at:
        raise ValidationError(
            f"Release {release_id} publication precedes its corpus release."
        )
    definitions: list[tuple[str, LearningTask]] = []
    for member in member_rows:
        if member["definition_json"] is None:
            raise ValidationError(
                f"Release {release_id} member {member['task_id']}@"
                f"{member['task_version']} has no task definition."
            )
        try:
            task = LearningTask.from_terms(
                _json_object(
                    member["definition_json"],
                    (
                        f"Release {release_id} task "
                        f"{member['task_id']}@{member['task_version']}"
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Release {release_id} contains an invalid task: {exc}"
            ) from exc
        if (
            task.id != member["task_id"]
            or task.version != member["task_version"]
            or task.digest != member["task_digest"]
            or task.digest != member["stored_task_digest"]
            or canonical_json(task.terms()) != member["definition_json"]
        ):
            raise ValidationError(
                f"Release {release_id} task {member['task_id']}@"
                f"{member['task_version']} has an identity, definition, or "
                "membership digest mismatch."
            )
        imported_at = _aware_timestamp(
            member["imported_at"],
            (
                f"Release {release_id} task "
                f"{member['task_id']}@{member['task_version']} imported_at"
            ),
        )
        if imported_at > created_at:
            raise ValidationError(
                f"Release {release_id} contains a task imported after "
                "publication."
            )
        definitions.append((member["status"], task))
    schema_version = (
        TASK_RELEASE_SCHEMA_VERSION
        if type(review) is TaskReleaseReview
        else SYNTHETIC_TASK_LAB_RELEASE_SCHEMA_VERSION
    )
    try:
        bundle = PerformanceTaskRelease(
            title=row["title"],
            corpus_release_id=row["corpus_release_id"],
            review=review,
            tasks=tuple(definitions),
            schema_version=schema_version,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValidationError(
            f"Release {release_id} cannot be reconstructed: {exc}"
        ) from exc
    if (
        bundle.release_id != row["id"]
        or bundle.bundle_hash != row["bundle_hash"]
    ):
        raise ValidationError(
            f"Release {release_id} bundle commitment mismatch."
        )
    _validate_release_corpus_bindings(
        connection,
        bundle,
        require_current_live_bindings=False,
    )
    return bundle, created_at


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _command_hash(value: Mapping[str, Any]) -> str:
    return canonical_digest({"type": "tsq.performance_command", **value})


def _event_metadata(event: sqlite3.Row) -> dict[str, Any]:
    return _json_object(event["metadata_json"], f"Event {event['event_id']} metadata")


def _authority_free_imported_evaluation(
    evaluation: TaskEvaluation,
) -> ImportedEvaluation:
    """Recover the exact authority-free observation from a normalized record."""

    return ImportedEvaluation(
        criteria=tuple(
            ImportedCriterionResult(
                criterion_id=criterion.criterion_id,
                status=criterion.status,
                score=criterion.score,
                outcome_code=criterion.outcome_code,
                phase=criterion.phase,
                source_action_ids=criterion.source_action_ids,
                attestation_digest=criterion.attestation_digest,
                misconception_ids=criterion.misconception_ids,
                reliability=criterion.reliability,
            )
            for criterion in evaluation.criteria
        )
    )


def _normalize_recovered_imported_evaluation(
    scoring_request: ScoringRequest,
    imported: ImportedEvaluation,
    provider: RegisteredProvider,
    provider_operation_digest: str,
) -> NormalizedScoringResult:
    """Apply the closed, shadow-only normalization used for recovered results."""

    direct_request = ScoringRequest(
        evaluation_id=scoring_request.evaluation_id,
        trace_id=scoring_request.trace_id,
        task_id=scoring_request.task_id,
        task_version=scoring_request.task_version,
        task_digest=scoring_request.task_digest,
        action_trace_digest=scoring_request.action_trace_digest,
        criterion_ids=scoring_request.criterion_ids,
        scorer_contract=None,
    )
    direct_provider_id = provider.provider_id
    if direct_provider_id.startswith("synthetic."):
        direct_provider_id = "recovered." + provider_operation_digest[:24]
    return normalize_imported_evaluation(
        direct_request,
        imported,
        provider_id=direct_provider_id,
        provider_version=provider.provider_version,
        declared_kind=provider.declared_kind,
    )


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
        """Publish a human-reviewed v1 release.

        Synthetic laboratory fixtures have a separate explicit entry point and
        can never pass through this activation-capable authoring command.
        """

        if type(bundle) is not PerformanceTaskRelease:
            raise ValidationError(
                "bundle must be an exact PerformanceTaskRelease."
            )
        if type(bundle.review) is not TaskReleaseReview:
            raise ValidationError(
                "Synthetic laboratory releases require "
                "publish_synthetic_lab_release and remain quarantined."
            )
        return self._publish_release(bundle, now=now)

    def publish_synthetic_lab_release(
        self,
        bundle: PerformanceTaskRelease,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Seal one non-authoritative, quarantine-only laboratory release."""

        if type(bundle) is not PerformanceTaskRelease:
            raise ValidationError(
                "bundle must be an exact PerformanceTaskRelease."
            )
        if type(bundle.review) is not SyntheticTaskLabDeclaration:
            raise ValidationError(
                "Synthetic laboratory publication requires a "
                "SyntheticTaskLabDeclaration."
            )
        return self._publish_release(bundle, now=now)

    def _publish_release(
        self,
        bundle: PerformanceTaskRelease,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if type(bundle) is not PerformanceTaskRelease:
            raise ValidationError(
                "bundle must be an exact PerformanceTaskRelease."
            )
        try:
            bundle = PerformanceTaskRelease.from_terms(bundle.terms())
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValidationError(
                f"Task release is not canonical: {exc}"
            ) from exc
        published_at = _now(now)
        authority_at_value = (
            bundle.review.reviewed_at
            if type(bundle.review) is TaskReleaseReview
            else bundle.review.declared_at
        )
        authority_at_label = (
            "review.reviewed_at"
            if type(bundle.review) is TaskReleaseReview
            else "declaration.declared_at"
        )
        authority_at = _aware_timestamp(
            authority_at_value, authority_at_label
        )
        if authority_at > published_at:
            raise ValidationError(
                "Task release cannot be published before its review or "
                "synthetic declaration."
            )
        timestamp = to_timestamp(published_at)
        bundle_json = canonical_json(bundle.terms())
        review_json = canonical_json(bundle.review.terms())
        bundle_size_bytes = len(bundle_json.encode("utf-8"))
        if bundle_size_bytes > MAX_TASK_RELEASE_BYTES:
            raise ValidationError(
                f"Task release exceeds the {MAX_TASK_RELEASE_BYTES}-byte limit."
            )
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
                stored_bundle, stored_created_at = load_stored_task_release(
                    connection,
                    prior["id"],
                )
                if stored_created_at > published_at:
                    raise ValidationError(
                        "Task release cannot be replayed before its stored "
                        "publication time."
                    )
                if (
                    canonical_json(stored_bundle.terms())
                    != canonical_json(bundle.terms())
                ):
                    raise ConflictError(
                        "Stored task release does not match the requested "
                        "immutable bundle."
                    )
                return {
                    "release_id": prior["id"],
                    "bundle_hash": prior["bundle_hash"],
                    "bundle_size_bytes": bundle_size_bytes,
                    "corpus_release_id": prior["corpus_release_id"],
                    "release_authority_kind": (
                        "human_review"
                        if type(bundle.review) is TaskReleaseReview
                        else "synthetic_lab"
                    ),
                    "task_count": len(bundle.tasks),
                    "status_counts": status_counts,
                    "idempotent_replay": True,
                }

            _validate_release_corpus_bindings(
                connection,
                bundle,
                require_current_live_bindings=True,
            )

            for _task_status, task in bundle.tasks:
                definition_json = canonical_json(task.terms())
                existing = connection.execute(
                    """SELECT task_digest, definition_json, imported_at
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
                if existing is not None and (
                    _aware_timestamp(
                        existing["imported_at"],
                        f"Performance task {task.id}@{task.version} imported_at",
                    )
                    > published_at
                ):
                    raise ValidationError(
                        f"Performance task {task.id}@{task.version} cannot be "
                        "reused by a release that predates its immutable import."
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
            "release_authority_kind": (
                "human_review"
                if type(bundle.review) is TaskReleaseReview
                else "synthetic_lab"
            ),
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
                bundle, _created_at = load_stored_task_release(
                    connection,
                    row["id"],
                )
                item = _row_dict(row)
                item.pop("review_json")
                item["review"] = bundle.review.terms()
                item["release_authority_kind"] = (
                    "human_review"
                    if type(bundle.review) is TaskReleaseReview
                    else "synthetic_lab"
                )
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
                          COALESCE(
                              json_extract(
                                  task_release.review_json, '$.reviewer_kind'
                              ),
                              json_extract(
                                  task_release.review_json,
                                  '$.declaration_kind'
                              )
                          ) AS release_authority_kind,
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
            authorities: dict[str, str] = {}
            for task_release_id in {
                row["release_id"] for row in rows
            }:
                bundle, _created_at = load_stored_task_release(
                    connection,
                    task_release_id,
                )
                authorities[task_release_id] = (
                    "human_review"
                    if type(bundle.review) is TaskReleaseReview
                    else "synthetic_lab"
                )
            result = []
            for row in rows:
                item = _row_dict(row)
                item["release_authority_kind"] = authorities[
                    row["release_id"]
                ]
                result.append(item)
        return result

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
                          task_release.corpus_release_id,
                          COALESCE(
                              json_extract(
                                  task_release.review_json, '$.reviewer_kind'
                              ),
                              json_extract(
                                  task_release.review_json,
                                  '$.declaration_kind'
                              )
                          ) AS release_authority_kind
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
        with self.database.read() as connection:
            bundle, _created_at = load_stored_task_release(
                connection,
                row["release_id"],
            )
        release_authority_kind = (
            "human_review"
            if type(bundle.review) is TaskReleaseReview
            else "synthetic_lab"
        )
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
            "release_authority_kind": release_authority_kind,
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
            if connection.execute(
                """SELECT 1 FROM performance_scoring_reconciliations
                   WHERE idempotency_key=?""",
                (idempotency_key,),
            ).fetchone() is not None:
                raise ConflictError(
                    "Idempotency key is already reserved by a scoring "
                    "reconciliation and cannot admit a provider callback."
                )
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
                (
                    "json_extract(task_release.review_json, "
                    "'$.reviewer_kind')='human'"
                ),
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
            stored_release, task_release_created_at = (
                load_stored_task_release(
                    connection,
                    member["release_id"],
                )
            )
            if type(stored_release.review) is not TaskReleaseReview:
                raise NotFoundError(
                    f"No serviceable release contains task {task_id} for "
                    "this session."
                )
            stored_members = {
                (task.id, task.version): (status, task.digest)
                for status, task in stored_release.tasks
            }
            stored_member = stored_members.get(
                (member["task_id"], member["task_version"])
            )
            if (
                stored_member is None
                or stored_member[0] not in SERVICEABLE_TASK_STATUSES
                or stored_member[1] != member["task_digest"]
            ):
                raise NotFoundError(
                    f"No serviceable release contains task {task_id} for "
                    "this session."
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
            live_misconception_objectives = (
                release_misconception_objectives(
                    connection,
                    session["corpus_release_id"],
                    accepted_only=True,
                    exclude_revoked=True,
                )
            )
            missing_bindings = missing_objective_misconception_bindings(
                task,
                live_misconception_objectives,
            )
            if missing_bindings:
                raise NotFoundError(
                    f"No currently serviceable release contains task "
                    f"{task_id}; its live diagnostic mapping was withdrawn."
                )
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
        projection_validated: bool = False,
    ) -> LearningAction:
        if not projection_validated:
            self._task_for_attempt(connection, attempt)
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
            frozen_payload = _json_object(
                canonical_json(dict(payload)),
                "Performance action payload",
            )
            command_hash = _command_hash(
                {
                    "operation": "record_action",
                    "attempt_id": attempt_id,
                    "action_type": kind.value,
                    "phase": typed_phase.value,
                    "payload": frozen_payload,
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
            self._task_for_attempt(connection, attempt)
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
                frozen_payload,
                occurred=occurred,
                idempotency_key=idempotency_key,
                command_hash=command_hash,
                projection_validated=True,
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
            attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise NotFoundError(f"Performance attempt {attempt_id} does not exist.")
            self._task_for_attempt(connection, attempt)
            rows = connection.execute(
                """SELECT * FROM performance_actions
                   WHERE attempt_id=? ORDER BY sequence""",
                (attempt_id,),
            ).fetchall()
        return [self._action_view(row) for row in rows]

    @staticmethod
    def _artifact_runner_contract(
        task: LearningTask,
        binding: ArtifactRunnerBinding,
    ) -> ScorerContract:
        required_actions = (
            ActionKind.ARTIFACT_CHECKPOINT,
            ActionKind.CHECK_RUN,
        )
        if any(kind not in task.allowed_action_kinds for kind in required_actions):
            raise ValidationError(
                "The released task does not allow the artifact/check action pair."
            )
        contracts = [
            contract
            for contract in task.scorer_contracts
            if (
                contract.kind is ScorerKind.DETERMINISTIC
                and contract.evidence_action_kinds == required_actions
                and contract.artifact_manifests
                == (
                    (
                        binding.artifact_kind,
                        binding.artifact_manifest_digest,
                    ),
                )
                and contract.check_set_manifests
                == (
                    (
                        binding.check_set_id,
                        binding.check_set_manifest_digest,
                    ),
                )
                and contract.requires_attestation is False
            )
        ]
        if len(contracts) != 1:
            raise ValidationError(
                "Artifact checking requires one exact released deterministic "
                "scorer contract pairing the bundled artifact and check-set "
                "manifests with artifact_checkpoint and check_run actions."
            )
        return contracts[0]

    @staticmethod
    def _resolve_artifact_action(
        connection: sqlite3.Connection,
        attempt_id: str,
        snapshot: ProductiveArtifactSnapshot,
        binding: ArtifactRunnerBinding,
        artifact_action_id: str | None,
    ) -> sqlite3.Row:
        if artifact_action_id is not None:
            _require_id(artifact_action_id, "artifact_action_id")
            row = connection.execute(
                """SELECT * FROM performance_actions
                   WHERE id=? AND attempt_id=?
                     AND action_type='artifact_checkpoint'""",
                (artifact_action_id, attempt_id),
            ).fetchone()
            if row is None:
                raise NotFoundError(
                    "The requested artifact checkpoint does not exist on "
                    "this performance attempt."
                )
            payload = _json_object(
                row["payload_json"], "Stored artifact checkpoint"
            )
            if (
                payload.get("artifact_digest") != snapshot.sha256
                or payload.get("artifact_kind") != binding.artifact_kind
            ):
                raise ValidationError(
                    "The requested artifact checkpoint does not match the "
                    "captured artifact and exact runner binding."
                )
            return row
        rows = connection.execute(
            """SELECT * FROM performance_actions
               WHERE attempt_id=? AND action_type='artifact_checkpoint'
                 AND json_extract(
                     payload_json, '$.artifact_digest'
                 )=?
                 AND json_extract(
                     payload_json, '$.artifact_kind'
                 )=?
               ORDER BY sequence, id""",
            (attempt_id, snapshot.sha256, binding.artifact_kind),
        ).fetchall()
        if not rows:
            raise NotFoundError(
                "No artifact checkpoint matches the captured artifact and "
                "exact runner binding."
            )
        if len(rows) != 1:
            raise ConflictError(
                "Multiple artifact checkpoints match this snapshot; specify "
                "--artifact-action explicitly."
            )
        return rows[0]

    @staticmethod
    def _artifact_run_projection_guard(
        connection: sqlite3.Connection,
        *,
        attempt_id: str | None = None,
        claim_id: str | None = None,
    ) -> None:
        clauses: list[str] = []
        parameters: list[Any] = []
        if attempt_id is not None:
            clauses.append(
                "json_extract(event.payload_json, '$.attempt_id')=?"
            )
            parameters.append(attempt_id)
        if claim_id is not None:
            clauses.append(
                "json_extract(event.payload_json, '$.claim_id')=?"
            )
            parameters.append(claim_id)
        scope = " AND " + " AND ".join(clauses) if clauses else ""
        orphan_claim = connection.execute(
            """SELECT event.event_id FROM events event
               WHERE event.event_type='PerformanceArtifactRunClaimed'
                 AND NOT EXISTS (
                     SELECT 1 FROM performance_artifact_run_claims claim
                     WHERE claim.event_id=event.event_id
                 )"""
            + scope
            + " ORDER BY event.stream_id, event.stream_version LIMIT 1",
            tuple(parameters),
        ).fetchone()
        if orphan_claim is not None:
            raise ConflictError(
                "Artifact-run admission event "
                f"{orphan_claim['event_id']} is missing its claim projection; "
                "the runner will not be invoked. Verify integrity and rebuild "
                "projections on a database copy."
            )
        orphan_receipt = connection.execute(
            """SELECT event.event_id FROM events event
               WHERE event.event_type='PerformanceArtifactRunObserved'
                 AND NOT EXISTS (
                     SELECT 1 FROM performance_artifact_run_receipts receipt
                     WHERE receipt.event_id=event.event_id
                 )"""
            + scope
            + " ORDER BY event.stream_id, event.stream_version LIMIT 1",
            tuple(parameters),
        ).fetchone()
        if orphan_receipt is not None:
            raise ConflictError(
                "Artifact-run observation event "
                f"{orphan_receipt['event_id']} is missing its receipt "
                "projection. Verify integrity and rebuild projections on a "
                "database copy."
            )

    @classmethod
    def _artifact_run_view(
        cls,
        connection: sqlite3.Connection,
        claim: sqlite3.Row,
        *,
        idempotent_replay: bool = False,
    ) -> dict[str, Any]:
        claim_event = connection.execute(
            "SELECT * FROM events WHERE event_id=?",
            (claim["event_id"],),
        ).fetchone()
        if claim_event is None:
            raise ConflictError(
                f"Artifact-run claim {claim['id']} is missing its admission "
                "event; the operation will not be repeated."
            )
        if (
            claim_event["event_type"] != "PerformanceArtifactRunClaimed"
            or claim_event["idempotency_key"]
            != performance_artifact_run_claim_event_key(
                claim["command_hash"]
            )
        ):
            raise ConflictError(
                f"Artifact-run claim {claim['id']} has an invalid admission "
                "event; the operation will not be repeated."
            )
        metadata = _json_object(
            claim_event["metadata_json"],
            "Stored artifact-run claim metadata",
        )
        try:
            request = ArtifactRunRequest.from_terms(
                _json_object(
                    claim["request_json"], "Stored artifact-run request"
                )
            )
            binding = ArtifactRunnerBinding.from_terms(
                _json_object(
                    claim["binding_json"], "Stored artifact-run binding"
                )
            )
        except (ArtifactRunnerProtocolError, TypeError, ValueError) as exc:
            raise ConflictError(
                f"Artifact-run claim {claim['id']} has invalid request or "
                "binding terms; the operation will not be repeated."
            ) from exc
        if (
            request.digest != claim["request_digest"]
            or binding.digest != claim["binding_digest"]
            or request.runner_binding_digest != binding.digest
            or request.artifact_sha256 != claim["artifact_digest"]
            or request.artifact_kind != claim["artifact_kind"]
            or request.artifact_manifest_digest
            != claim["artifact_manifest_digest"]
            or request.check_set_id != claim["check_set_id"]
            or request.check_set_manifest_digest
            != claim["check_set_manifest_digest"]
            or binding.runner_id != claim["runner_id"]
            or binding.runner_version != claim["runner_version"]
        ):
            raise ConflictError(
                f"Artifact-run claim {claim['id']} has mismatched immutable "
                "terms; the operation will not be repeated."
            )
        receipt_row = connection.execute(
            """SELECT * FROM performance_artifact_run_receipts
               WHERE claim_id=?""",
            (claim["id"],),
        ).fetchone()
        process_terms: dict[str, Any] | None = None
        receipt_terms: dict[str, Any] | None = None
        receipt_digest: str | None = None
        check_action: dict[str, Any] | None = None
        outcome: str | None = None
        started_at: str | None = None
        completed_at: str | None = None
        worker_process_started: bool | None = None
        terminal = receipt_row is not None
        if receipt_row is not None:
            receipt_event = connection.execute(
                "SELECT * FROM events WHERE event_id=?",
                (receipt_row["event_id"],),
            ).fetchone()
            if (
                receipt_event is None
                or receipt_event["event_type"]
                != "PerformanceArtifactRunObserved"
                or receipt_event["idempotency_key"]
                != performance_artifact_run_receipt_event_key(
                    receipt_row["receipt_digest"]
                )
            ):
                raise ConflictError(
                    f"Artifact-run claim {claim['id']} has an invalid terminal "
                    "event; verify integrity."
                )
            try:
                operational = OperationalArtifactRunReceipt.from_terms(
                    _json_object(
                        receipt_row["receipt_json"],
                        "Stored operational artifact-run receipt",
                    )
                )
            except (TypeError, ValueError, ValidationError) as exc:
                raise ConflictError(
                    f"Artifact-run claim {claim['id']} has an invalid terminal "
                    "receipt; verify integrity."
                ) from exc
            try:
                parsed_started = _aware_timestamp(
                    receipt_row["started_at"],
                    "Stored artifact-run started_at",
                )
                parsed_completed = _aware_timestamp(
                    receipt_row["completed_at"],
                    "Stored artifact-run completed_at",
                )
                expected_operational = OperationalArtifactRunReceipt(
                    claim_id=claim["id"],
                    attempt_id=claim["attempt_id"],
                    artifact_action_id=claim["artifact_action_id"],
                    artifact_digest=claim["artifact_digest"],
                    artifact_kind=claim["artifact_kind"],
                    artifact_manifest_digest=claim[
                        "artifact_manifest_digest"
                    ],
                    check_set_id=claim["check_set_id"],
                    check_set_manifest_digest=claim[
                        "check_set_manifest_digest"
                    ],
                    runner_id=claim["runner_id"],
                    runner_version=claim["runner_version"],
                    outcome=receipt_row["outcome"],
                    started_at=receipt_row["started_at"],
                    completed_at=receipt_row["completed_at"],
                    result_digest=receipt_row["result_digest"],
                    request_digest=request.digest,
                    binding_digest=binding.digest,
                )
            except ValidationError as exc:
                raise ConflictError(
                    f"Artifact-run claim {claim['id']} has invalid terminal "
                    "projection fields; verify integrity."
                ) from exc
            if (
                to_timestamp(parsed_started) != receipt_row["started_at"]
                or to_timestamp(parsed_completed)
                != receipt_row["completed_at"]
                or operational != expected_operational
                or operational.digest != receipt_row["receipt_digest"]
                or canonical_json(operational.terms())
                != receipt_row["receipt_json"]
            ):
                raise ConflictError(
                    f"Artifact-run claim {claim['id']} has a mismatched "
                    "terminal receipt; verify integrity."
                )
            outcome = receipt_row["outcome"]
            started_at = receipt_row["started_at"]
            completed_at = receipt_row["completed_at"]
            receipt_terms = expected_operational.terms()
            receipt_digest = expected_operational.digest
            successful = outcome in {
                ArtifactRunOutcome.COMPLETED.value,
                ArtifactRunOutcome.INVALID_ARTIFACT.value,
            }
            if successful:
                if (
                    receipt_row["result_json"] is None
                    or receipt_row["result_digest"] is None
                    or receipt_row["check_action_id"] is None
                ):
                    raise ConflictError(
                        f"Artifact-run claim {claim['id']} has an incomplete "
                        "successful observation; verify integrity."
                    )
                try:
                    process = ArtifactProcessReceipt.from_terms(
                        _json_object(
                            receipt_row["result_json"],
                            "Stored artifact process receipt",
                        )
                    )
                except (ArtifactRunnerProtocolError, TypeError, ValueError) as exc:
                    raise ConflictError(
                        f"Artifact-run claim {claim['id']} has an invalid "
                        "process receipt; verify integrity."
                    ) from exc
                if (
                    process.digest != receipt_row["result_digest"]
                    or process.request != request
                    or process.binding != binding
                    or process.result.outcome.value != outcome
                    or expected_operational.result_digest != process.digest
                    or canonical_json(process.terms())
                    != receipt_row["result_json"]
                ):
                    raise ConflictError(
                        f"Artifact-run claim {claim['id']} has a mismatched "
                        "process receipt; verify integrity."
                    )
                process_terms = process.terms()
                worker_process_started = process.worker_process_started
                action = connection.execute(
                    "SELECT * FROM performance_actions WHERE id=?",
                    (receipt_row["check_action_id"],),
                ).fetchone()
                artifact = connection.execute(
                    "SELECT * FROM performance_actions WHERE id=?",
                    (claim["artifact_action_id"],),
                ).fetchone()
                if action is None or artifact is None:
                    raise ConflictError(
                        f"Artifact-run claim {claim['id']} is missing its "
                        "artifact/check action pair; verify integrity."
                    )
                try:
                    check_payload = _json_object(
                        action["payload_json"],
                        "Stored generated artifact check action",
                    )
                    _require_exact_fields(
                        check_payload,
                        frozenset(
                            {
                                "check_set_id",
                                "passed",
                                "failed",
                                "errored",
                                "skipped",
                                "result_digest",
                            }
                        ),
                        "Stored generated artifact check action",
                    )
                except ValidationError as exc:
                    raise ConflictError(
                        f"Artifact-run claim {claim['id']} has an invalid "
                        "generated check action; verify integrity."
                    ) from exc
                result = process.result
                expected_check_command = _command_hash(
                    {
                        "operation": "record_artifact_run_check",
                        "claim_id": claim["id"],
                        "result_digest": process.digest,
                    }
                )
                if (
                    artifact["attempt_id"] != claim["attempt_id"]
                    or artifact["action_type"]
                    != ActionKind.ARTIFACT_CHECKPOINT.value
                    or action["attempt_id"] != claim["attempt_id"]
                    or action["sequence"] != claim["through_sequence"] + 1
                    or action["phase"] != artifact["phase"]
                    or action["action_type"] != ActionKind.CHECK_RUN.value
                    or action["occurred_at"] != receipt_row["completed_at"]
                    or action["command_hash"] != expected_check_command
                    or check_payload["check_set_id"]
                    != claim["check_set_id"]
                    or check_payload["passed"] != result.passed
                    or check_payload["failed"] != result.failed
                    or check_payload["errored"] != result.errored
                    or check_payload["skipped"] != result.skipped
                    or check_payload["result_digest"] != process.digest
                ):
                    raise ConflictError(
                        f"Artifact-run claim {claim['id']} has a mismatched "
                        "generated check action; verify integrity."
                    )
                check_action = cls._action_view(action)
            elif (
                outcome
                not in {
                    "runner_failed",
                    ArtifactRunOutcome.TIMED_OUT.value,
                }
                or receipt_row["result_json"] is not None
                or receipt_row["result_digest"] is not None
                or receipt_row["check_action_id"] is not None
                or expected_operational.result_digest is not None
            ):
                raise ConflictError(
                    f"Artifact-run claim {claim['id']} has a mismatched failed "
                    "observation; verify integrity."
                )
        return {
            "id": claim["id"],
            "claim_id": claim["id"],
            "event_id": claim["event_id"],
            "run_id": request.run_id,
            "attempt_id": claim["attempt_id"],
            "session_id": claim["session_id"],
            "session_revision": claim["session_revision"],
            "artifact_action_id": claim["artifact_action_id"],
            "through_sequence": claim["through_sequence"],
            "task_release_id": claim["task_release_id"],
            "task_id": claim["task_id"],
            "task_version": claim["task_version"],
            "task_digest": claim["task_digest"],
            "artifact_digest": claim["artifact_digest"],
            "artifact_kind": claim["artifact_kind"],
            "artifact_manifest_digest": claim[
                "artifact_manifest_digest"
            ],
            "check_set_id": claim["check_set_id"],
            "check_set_manifest_digest": claim[
                "check_set_manifest_digest"
            ],
            "runner_id": claim["runner_id"],
            "runner_version": claim["runner_version"],
            "request": request.terms(),
            "request_digest": request.digest,
            "binding": binding.terms(),
            "binding_digest": binding.digest,
            "command_hash": claim["command_hash"],
            "claimed_at": claim["claimed_at"],
            "status": outcome or "unresolved",
            "outcome": outcome,
            "terminal": terminal,
            "started_at": started_at,
            "completed_at": completed_at,
            "check_action": check_action,
            "process_receipt": process_terms,
            "operational_receipt": receipt_terms,
            "receipt_digest": receipt_digest,
            "caller_idempotency_key_present": (
                claim["idempotency_key"] is not None
            ),
            "admission_mode": metadata.get("admission_mode"),
            "retry_allowed": False,
            "automatic_retry_allowed": False,
            "artifact_content_persisted": False,
            "artifact_executed": False,
            "evaluation_created": False,
            "learner_projection_applied": False,
            "mastery_applied": False,
            "certification_applied": False,
            "skill_authority": False,
            "shadow_only": True,
            "process_boundary_configured": binding.process_separated,
            "process_separated": binding.process_separated,
            "worker_process_started": worker_process_started,
            "operating_system_sandboxed": False,
            "filesystem_isolation_enforced": False,
            "network_isolation_enforced": False,
            "idempotent_replay": idempotent_replay,
        }

    @classmethod
    def _prior_artifact_run_claim(
        cls,
        connection: sqlite3.Connection,
        *,
        idempotency_key: str | None,
        command_hash: str,
    ) -> dict[str, Any] | None:
        key_claim = None
        if idempotency_key is not None:
            key_claim = connection.execute(
                """SELECT * FROM performance_artifact_run_claims
                   WHERE idempotency_key=?""",
                (idempotency_key,),
            ).fetchone()
            if (
                key_claim is not None
                and key_claim["command_hash"] != command_hash
            ):
                raise ConflictError(
                    "Idempotency key was already reserved for a different "
                    "artifact-run command."
                )
        claim = connection.execute(
            """SELECT * FROM performance_artifact_run_claims
               WHERE command_hash=?""",
            (command_hash,),
        ).fetchone()
        if claim is None:
            orphan = connection.execute(
                "SELECT event_id FROM events WHERE idempotency_key=?",
                (performance_artifact_run_claim_event_key(command_hash),),
            ).fetchone()
            if orphan is not None:
                raise ConflictError(
                    "Artifact-run admission is committed in event history but "
                    "its claim projection is missing; the runner will not be "
                    "invoked. Verify integrity and rebuild projections on a "
                    "database copy."
                )
            if key_claim is not None:
                raise ConflictError(
                    "Artifact-run idempotency state is inconsistent; the "
                    "runner will not be invoked."
                )
            return None
        return cls._artifact_run_view(
            connection,
            claim,
            idempotent_replay=True,
        )

    @staticmethod
    def _validate_artifact_run_caller_key(
        connection: sqlite3.Connection,
        idempotency_key: str | None,
    ) -> None:
        if idempotency_key is None:
            return
        _require_text(idempotency_key, "idempotency_key", 256)
        if idempotency_key.startswith(
            _PERFORMANCE_TECHNICAL_EVENT_KEY_PREFIXES
        ):
            raise ValidationError(
                "Artifact-run idempotency keys cannot use a TSQ reserved "
                "technical namespace."
            )
        collision = connection.execute(
            """SELECT 1 FROM events event
               WHERE event.idempotency_key=?
                  OR (
                      event.event_type IN (
                          'PerformanceScoringClaimed',
                          'PerformanceScoringClaimMigrated',
                          'PerformanceScoringReconciled'
                      )
                      AND json_extract(
                          event.payload_json, '$.caller_idempotency_key'
                      )=?
                  )
               LIMIT 1""",
            (idempotency_key, idempotency_key),
        ).fetchone()
        if collision is not None:
            raise ConflictError(
                "Idempotency key is already reserved outside this "
                "artifact-run command."
            )

    @staticmethod
    def _validate_artifact_run_technical_event_key(
        connection: sqlite3.Connection,
        event_key: str,
    ) -> None:
        """Keep technical event keys disjoint from historical caller keys."""

        collision = connection.execute(
            """SELECT 'scoring claim' AS source
               FROM performance_scoring_claims
               WHERE idempotency_key=?
               UNION ALL
               SELECT 'scoring reconciliation' AS source
               FROM performance_scoring_reconciliations
               WHERE idempotency_key=?
               LIMIT 1""",
            (event_key, event_key),
        ).fetchone()
        if collision is not None:
            raise ConflictError(
                "Artifact-run technical event key collides with a historical "
                f"{collision['source']} caller idempotency key."
            )

    def list_artifact_runs(
        self,
        *,
        attempt_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List durable artifact-run admissions without invoking a runner."""

        if attempt_id is not None:
            _require_id(attempt_id, "attempt_id")
        allowed_statuses = {
            None,
            "unresolved",
            ArtifactRunOutcome.COMPLETED.value,
            ArtifactRunOutcome.INVALID_ARTIFACT.value,
            "runner_failed",
            ArtifactRunOutcome.TIMED_OUT.value,
        }
        if status not in allowed_statuses:
            raise ValidationError(
                "Artifact-run status must be unresolved, completed, "
                "invalid_artifact, runner_failed, or timed_out."
            )
        with self.database.read() as connection:
            validated_learners: set[str] = set()
            if attempt_id is not None:
                attempt = connection.execute(
                    "SELECT * FROM performance_attempts WHERE id=?",
                    (attempt_id,),
                ).fetchone()
                if attempt is None:
                    raise NotFoundError(
                        f"Performance attempt {attempt_id} does not exist."
                    )
            self._artifact_run_projection_guard(
                connection, attempt_id=attempt_id
            )
            if attempt_id is not None:
                self._task_for_attempt(connection, attempt)
                validated_learners.add(attempt["learner_id"])
            rows = connection.execute(
                """SELECT * FROM performance_artifact_run_claims"""
                + (
                    " WHERE attempt_id=?"
                    if attempt_id is not None
                    else ""
                )
                + " ORDER BY claimed_at, id",
                (() if attempt_id is None else (attempt_id,)),
            ).fetchall()
            for row_attempt_id in sorted(
                {row["attempt_id"] for row in rows}
            ):
                attempt = connection.execute(
                    "SELECT * FROM performance_attempts WHERE id=?",
                    (row_attempt_id,),
                ).fetchone()
                if attempt is None:
                    raise ValidationError(
                        f"Artifact run has no performance attempt "
                        f"{row_attempt_id}."
                    )
                learner_id = attempt["learner_id"]
                if learner_id not in validated_learners:
                    require_performance_projection_consistency(
                        connection,
                        learner_id=learner_id,
                        trace_only=True,
                    )
                    validated_learners.add(learner_id)
                self._task_for_attempt(
                    connection,
                    attempt,
                    trace_validated=True,
                )
            result = [
                self._artifact_run_view(connection, row) for row in rows
            ]
        if status is None:
            return result
        return [item for item in result if item["status"] == status]

    def inspect_artifact_run(self, run_id: str) -> dict[str, Any]:
        """Inspect one claim, logical run ID, or receipt without executing it."""

        _require_id(run_id, "run_id")
        with self.database.read() as connection:
            rows = connection.execute(
                """SELECT DISTINCT claim.*
                   FROM performance_artifact_run_claims claim
                   LEFT JOIN performance_artifact_run_receipts receipt
                     ON receipt.claim_id=claim.id
                   WHERE claim.id=?
                      OR json_extract(claim.request_json, '$.run_id')=?
                      OR receipt.id=?""",
                (run_id, run_id, run_id),
            ).fetchall()
            if not rows:
                raise NotFoundError(
                    f"Performance artifact run {run_id} does not exist."
                )
            if len(rows) != 1:
                raise ConflictError(
                    "Artifact-run reference is ambiguous; inspect by claim ID."
                )
            self._artifact_run_projection_guard(
                connection, claim_id=rows[0]["id"]
            )
            attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?",
                (rows[0]["attempt_id"],),
            ).fetchone()
            if attempt is None:
                raise ValidationError(
                    "Artifact run has no performance attempt."
                )
            self._task_for_attempt(connection, attempt)
            return self._artifact_run_view(connection, rows[0])

    def run_artifact_check(
        self,
        attempt_id: str,
        snapshot: ProductiveArtifactSnapshot,
        registry: SyntheticArtifactRunnerRegistry,
        binding: ArtifactRunnerBinding,
        *,
        check_set_id: str,
        artifact_action_id: str | None = None,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Admit, run, and terminally observe one bundled synthetic checker."""

        _require_id(attempt_id, "attempt_id")
        _require_id(check_set_id, "check_set_id")
        if artifact_action_id is not None:
            _require_id(artifact_action_id, "artifact_action_id")
        if type(snapshot) is not ProductiveArtifactSnapshot:
            raise ValidationError(
                "snapshot must be a ProductiveArtifactSnapshot captured by "
                "TSQ intake."
            )
        if type(registry) is not SyntheticArtifactRunnerRegistry:
            raise ValidationError(
                "registry must be the exact synthetic artifact-runner registry."
            )
        if type(binding) is not ArtifactRunnerBinding:
            raise ValidationError(
                "binding must be an exact ArtifactRunnerBinding."
            )
        try:
            expected_binding = bundled_synthetic_binding()
            registered_binding = registry.inspect(
                binding.checker_id, binding.checker_version
            )
        except (ArtifactRunnerProtocolError, LookupError, ValueError) as exc:
            raise ValidationError(
                f"Artifact checking failed safely before admission: {exc}"
            ) from exc
        if (
            binding != expected_binding
            or registered_binding != expected_binding
            or check_set_id != binding.check_set_id
        ):
            raise ValidationError(
                "Artifact checking requires the exact registered bundled "
                "binding and check set."
            )
        if self.database.read_only:
            raise ConflictError(
                "A read-only database cannot admit an artifact run."
            )
        claimed = _now(now)
        with self.database.transaction() as connection:
            attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?",
                (attempt_id,),
            ).fetchone()
            if attempt is None:
                raise NotFoundError(
                    f"Performance attempt {attempt_id} does not exist."
                )
            self.database.require_learner_evidence_safe(
                attempt["learner_id"], connection
            )
            session = connection.execute(
                "SELECT * FROM sessions WHERE id=?",
                (attempt["session_id"],),
            ).fetchone()
            if session is None:
                raise ValidationError("Performance attempt has no session.")
            self._validate_artifact_run_caller_key(
                connection, idempotency_key
            )
            task = self._task_for_attempt(connection, attempt)
            self._artifact_runner_contract(task, binding)
            artifact = self._resolve_artifact_action(
                connection,
                attempt_id,
                snapshot,
                binding,
                artifact_action_id,
            )
            logical_command = {
                "operation": "run_artifact_check",
                "attempt_id": attempt_id,
                "artifact_action_id": artifact["id"],
                "artifact_digest": snapshot.sha256,
                "artifact_size_bytes": snapshot.size_bytes,
                "artifact_kind": binding.artifact_kind,
                "artifact_manifest_digest": (
                    binding.artifact_manifest_digest
                ),
                "check_set_id": binding.check_set_id,
                "check_set_manifest_digest": (
                    binding.check_set_manifest_digest
                ),
                "checker_id": binding.checker_id.value,
                "checker_version": binding.checker_version,
                "runner_id": binding.runner_id,
                "runner_version": binding.runner_version,
                "binding_digest": binding.digest,
            }
            command_hash = _command_hash(logical_command)
            self._validate_artifact_run_technical_event_key(
                connection,
                performance_artifact_run_claim_event_key(command_hash),
            )
            prior = self._prior_artifact_run_claim(
                connection,
                idempotency_key=idempotency_key,
                command_hash=command_hash,
            )
            if prior is not None:
                return prior
            if session["status"] != "active":
                raise ConflictError(
                    f"Session {attempt['session_id']} is {session['status']}."
                )
            if self._attempt_status(connection, attempt_id) != "active":
                raise ConflictError(
                    "Artifact checks must be admitted before the attempt's "
                    "terminal checkpoint."
                )
            artifact_occurred = _aware_timestamp(
                artifact["occurred_at"], "Artifact checkpoint occurrence"
            )
            if claimed < artifact_occurred:
                raise ValidationError(
                    "Artifact-run admission cannot precede its artifact checkpoint."
                )
            try:
                request = build_artifact_run_request(
                    "arun_" + command_hash[:24],
                    snapshot.content,
                    binding,
                )
            except (ArtifactRunnerProtocolError, TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Artifact checking failed safely before admission: {exc}"
                ) from exc
            boundary = connection.execute(
                """SELECT MAX(sequence) AS sequence
                   FROM performance_actions WHERE attempt_id=?""",
                (attempt_id,),
            ).fetchone()
            if boundary["sequence"] is None:
                raise ValidationError(
                    "Performance attempt has no action-trace head."
                )
            through_sequence = boundary["sequence"]
            claim_id = new_id("parc")
            claimed_at = to_timestamp(claimed)
            claim_payload = performance_artifact_run_claim_payload(
                claim_id=claim_id,
                caller_idempotency_key=idempotency_key,
                attempt_id=attempt_id,
                session_id=attempt["session_id"],
                session_revision=session["revision"],
                artifact_action_id=artifact["id"],
                through_sequence=through_sequence,
                task_release_id=attempt["task_release_id"],
                task_id=attempt["task_id"],
                task_version=attempt["task_version"],
                task_digest=attempt["task_digest"],
                artifact_digest=snapshot.sha256,
                artifact_kind=binding.artifact_kind,
                artifact_manifest_digest=(
                    binding.artifact_manifest_digest
                ),
                check_set_id=binding.check_set_id,
                check_set_manifest_digest=(
                    binding.check_set_manifest_digest
                ),
                runner_id=binding.runner_id,
                runner_version=binding.runner_version,
                request=request.terms(),
                request_digest=request.digest,
                binding=binding.terms(),
                binding_digest=binding.digest,
                command_hash=command_hash,
                claimed_at=claimed_at,
            )
            claim_event = self.database.append_event(
                connection,
                stream_id=f"learner:{attempt['learner_id']}",
                event_type="PerformanceArtifactRunClaimed",
                schema_version=PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION,
                payload=claim_payload,
                metadata={
                    "artifact_run_schema_version": (
                        PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION
                    ),
                    "admission_mode": "pre_runner",
                    "automatic_retry_allowed": False,
                    "shadow_only": True,
                },
                learner_id=attempt["learner_id"],
                session_id=attempt["session_id"],
                idempotency_key=performance_artifact_run_claim_event_key(
                    command_hash
                ),
                correlation_id=attempt_id,
                causation_id=artifact["event_id"],
                occurred_at=claimed,
            )
            connection.execute(
                """INSERT INTO performance_artifact_run_claims(
                       id, event_id, idempotency_key, attempt_id, session_id,
                       session_revision, artifact_action_id, through_sequence,
                       task_release_id, task_id, task_version, task_digest,
                       artifact_digest, artifact_kind,
                       artifact_manifest_digest, check_set_id,
                       check_set_manifest_digest, runner_id, runner_version,
                       request_json, request_digest, binding_json,
                       binding_digest, command_hash, claimed_at
                   ) VALUES (
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?, ?
                   )""",
                (
                    claim_id,
                    claim_event["event_id"],
                    idempotency_key,
                    attempt_id,
                    attempt["session_id"],
                    session["revision"],
                    artifact["id"],
                    through_sequence,
                    attempt["task_release_id"],
                    attempt["task_id"],
                    attempt["task_version"],
                    attempt["task_digest"],
                    snapshot.sha256,
                    binding.artifact_kind,
                    binding.artifact_manifest_digest,
                    binding.check_set_id,
                    binding.check_set_manifest_digest,
                    binding.runner_id,
                    binding.runner_version,
                    canonical_json(request.terms()),
                    request.digest,
                    canonical_json(binding.terms()),
                    binding.digest,
                    command_hash,
                    claimed_at,
                ),
            )

        started = max(claimed, datetime.now(timezone.utc))
        try:
            process_receipt = registry.run(request, snapshot.content)
        except Exception as exc:
            raise ValidationError(
                "Artifact runner failed after durable admission; the claim "
                "remains unresolved and the operation will not be retried "
                "automatically."
            ) from exc
        if (
            type(process_receipt) is not ArtifactProcessReceipt
            or process_receipt.request != request
            or process_receipt.binding != binding
            or process_receipt.result.artifact_sha256 != snapshot.sha256
            or process_receipt.digest
            != canonical_digest(process_receipt.terms())
        ):
            raise ValidationError(
                "Artifact runner returned an observation that does not match "
                "its durable claim; the claim remains unresolved."
            )
        pure_outcome = process_receipt.result.outcome
        if pure_outcome in {
            ArtifactRunOutcome.COMPLETED,
            ArtifactRunOutcome.INVALID_ARTIFACT,
        }:
            outcome = pure_outcome.value
            process_terms = process_receipt.terms()
            result_digest = process_receipt.digest
        elif pure_outcome is ArtifactRunOutcome.TIMED_OUT:
            outcome = ArtifactRunOutcome.TIMED_OUT.value
            process_terms = None
            result_digest = None
        else:
            outcome = "runner_failed"
            process_terms = None
            result_digest = None
        completed = max(started, datetime.now(timezone.utc))
        started_at = to_timestamp(started)
        completed_at = to_timestamp(completed)

        with self.database.transaction() as connection:
            claim = connection.execute(
                """SELECT * FROM performance_artifact_run_claims WHERE id=?""",
                (claim_id,),
            ).fetchone()
            if claim is None:
                raise ConflictError(
                    "Artifact-run claim disappeared before finalization."
                )
            current_attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?",
                (attempt_id,),
            ).fetchone()
            if current_attempt is None:
                raise ConflictError(
                    "Performance attempt disappeared before artifact-run "
                    "finalization."
                )
            self._task_for_attempt(connection, current_attempt)
            prior_receipt = connection.execute(
                """SELECT 1 FROM performance_artifact_run_receipts
                   WHERE claim_id=?""",
                (claim_id,),
            ).fetchone()
            if prior_receipt is not None:
                return self._artifact_run_view(
                    connection, claim, idempotent_replay=True
                )
            session = connection.execute(
                "SELECT * FROM sessions WHERE id=?",
                (claim["session_id"],),
            ).fetchone()
            current_head = connection.execute(
                """SELECT MAX(sequence) AS sequence
                   FROM performance_actions WHERE attempt_id=?""",
                (attempt_id,),
            ).fetchone()["sequence"]
            current_artifact = connection.execute(
                "SELECT * FROM performance_actions WHERE id=?",
                (claim["artifact_action_id"],),
            ).fetchone()
            if (
                current_attempt is None
                or session is None
                or session["status"] != "active"
                or session["revision"] != claim["session_revision"]
                or current_attempt["task_release_id"]
                != claim["task_release_id"]
                or current_attempt["task_id"] != claim["task_id"]
                or current_attempt["task_version"] != claim["task_version"]
                or current_attempt["task_digest"] != claim["task_digest"]
                or current_head != claim["through_sequence"]
                or current_artifact is None
            ):
                raise ConflictError(
                    "Artifact-run trace, session, task, or release changed "
                    "after admission; the observation was rejected and the "
                    "claim remains unresolved."
                )
            self.database.require_learner_evidence_safe(
                current_attempt["learner_id"], connection
            )
            check_action: LearningAction | None = None
            if process_terms is not None:
                result = process_receipt.result
                check_action = self._append_action(
                    connection,
                    current_attempt,
                    ActionKind.CHECK_RUN,
                    ActionPhase(current_artifact["phase"]),
                    {
                        "check_set_id": binding.check_set_id,
                        "passed": result.passed,
                        "failed": result.failed,
                        "errored": result.errored,
                        "skipped": result.skipped,
                        "result_digest": result_digest,
                    },
                    occurred=completed,
                    idempotency_key=None,
                    command_hash=_command_hash(
                        {
                            "operation": "record_artifact_run_check",
                            "claim_id": claim_id,
                            "result_digest": result_digest,
                        }
                    ),
                    projection_validated=True,
                )
            operational = OperationalArtifactRunReceipt(
                claim_id=claim_id,
                attempt_id=attempt_id,
                artifact_action_id=claim["artifact_action_id"],
                artifact_digest=claim["artifact_digest"],
                artifact_kind=claim["artifact_kind"],
                artifact_manifest_digest=claim[
                    "artifact_manifest_digest"
                ],
                check_set_id=claim["check_set_id"],
                check_set_manifest_digest=claim[
                    "check_set_manifest_digest"
                ],
                runner_id=claim["runner_id"],
                runner_version=claim["runner_version"],
                outcome=outcome,
                started_at=started_at,
                completed_at=completed_at,
                result_digest=result_digest,
                request_digest=claim["request_digest"],
                binding_digest=claim["binding_digest"],
            )
            self._validate_artifact_run_technical_event_key(
                connection,
                performance_artifact_run_receipt_event_key(
                    operational.digest
                ),
            )
            receipt_id = new_id("parr")
            observed_payload = performance_artifact_run_observed_payload(
                receipt_id=receipt_id,
                claim_id=claim_id,
                attempt_id=attempt_id,
                check_action_id=(
                    None if check_action is None else check_action.id
                ),
                outcome=outcome,
                result=process_terms,
                result_digest=result_digest,
                receipt=operational.terms(),
                receipt_digest=operational.digest,
                started_at=started_at,
                completed_at=completed_at,
            )
            observed_event = self.database.append_event(
                connection,
                stream_id=f"learner:{current_attempt['learner_id']}",
                event_type="PerformanceArtifactRunObserved",
                schema_version=PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION,
                payload=observed_payload,
                metadata={
                    "artifact_run_schema_version": (
                        PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION
                    ),
                    "observational_only": True,
                    "projection_applied": False,
                    "certification_applied": False,
                    "skill_authority": False,
                    "shadow_only": True,
                },
                learner_id=current_attempt["learner_id"],
                session_id=None,
                idempotency_key=performance_artifact_run_receipt_event_key(
                    operational.digest
                ),
                correlation_id=attempt_id,
                causation_id=claim["event_id"],
                occurred_at=completed,
            )
            connection.execute(
                """INSERT INTO performance_artifact_run_receipts(
                       id, event_id, claim_id, check_action_id, outcome,
                       result_json, result_digest, receipt_json,
                       receipt_digest, started_at, completed_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    receipt_id,
                    observed_event["event_id"],
                    claim_id,
                    None if check_action is None else check_action.id,
                    outcome,
                    (
                        None
                        if process_terms is None
                        else canonical_json(process_terms)
                    ),
                    result_digest,
                    canonical_json(operational.terms()),
                    operational.digest,
                    started_at,
                    completed_at,
                ),
            )
            return self._artifact_run_view(connection, claim)

    def list_scoring_claims(
        self,
        *,
        attempt_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List durable callback admissions without reconciling any claim."""

        if attempt_id is not None:
            _require_id(attempt_id, "attempt_id")
        if status not in {
            None,
            "unreconciled",
            "unknown",
            "definitely_absent",
            "completed",
            # Compatibility query for callers written before reconciliation.
            "unresolved",
        }:
            raise ValidationError(
                "Scoring claim status must be unreconciled, unknown, "
                "definitely_absent, completed, or unresolved."
            )
        clauses: list[str] = []
        parameters: list[Any] = []
        if attempt_id is not None:
            clauses.append("claim.attempt_id=?")
            parameters.append(attempt_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.database.read() as connection:
            validated_learners: set[str] = set()
            if attempt_id is not None:
                attempt = connection.execute(
                    "SELECT * FROM performance_attempts WHERE id=?",
                    (attempt_id,),
                ).fetchone()
                if attempt is None:
                    raise NotFoundError(
                        f"Performance attempt {attempt_id} does not exist."
                    )
                self._task_for_attempt(connection, attempt)
                validated_learners.add(attempt["learner_id"])
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
            orphan_reconciliation = connection.execute(
                """SELECT event.event_id
                   FROM events event
                   WHERE event.event_type='PerformanceScoringReconciled'
                     AND NOT EXISTS (
                         SELECT 1
                         FROM performance_scoring_reconciliations observation
                         WHERE observation.event_id=event.event_id
                     )
                   ORDER BY event.event_id LIMIT 1"""
            ).fetchone()
            if orphan_reconciliation is not None:
                raise ConflictError(
                    "Scoring reconciliation event "
                    f"{orphan_reconciliation['event_id']} is missing its "
                    "projection; run integrity verification and rebuild on "
                    "a database copy."
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
            for row_attempt_id in sorted(
                {row["attempt_id"] for row in rows}
            ):
                attempt = connection.execute(
                    "SELECT * FROM performance_attempts WHERE id=?",
                    (row_attempt_id,),
                ).fetchone()
                if attempt is None:
                    raise ValidationError(
                        f"Scoring claim has no performance attempt "
                        f"{row_attempt_id}."
                    )
                learner_id = attempt["learner_id"]
                if learner_id not in validated_learners:
                    require_performance_projection_consistency(
                        connection,
                        learner_id=learner_id,
                        trace_only=True,
                    )
                    validated_learners.add(learner_id)
                self._task_for_attempt(
                    connection,
                    attempt,
                    trace_validated=True,
                )
            result = [
                self._scoring_claim_view(connection, row) for row in rows
            ]
        if status is None:
            return result
        if status == "unresolved":
            return [
                item
                for item in result
                if item["status"] in {"unreconciled", "unknown"}
            ]
        return [item for item in result if item["status"] == status]

    @staticmethod
    def _scoring_claim_view(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        idempotent_replay: bool = False,
    ) -> dict[str, Any]:
        if row["event_type"] is None or row["metadata_json"] is None:
            raise ConflictError(
                f"Scoring claim {row['id']} is missing its admission event; "
                "run integrity verification."
            )
        metadata = _json_object(
            row["metadata_json"], "Stored scoring claim event metadata"
        )
        observations = connection.execute(
            """SELECT observation.*
               FROM performance_scoring_reconciliations observation
               WHERE observation.claim_id=?
               ORDER BY (
                   SELECT event.stream_version FROM events event
                   WHERE event.event_id=observation.event_id
               ), observation.id""",
            (row["id"],),
        ).fetchall()
        missing_event = next(
            (
                observation["event_id"]
                for observation in observations
                if connection.execute(
                    "SELECT 1 FROM events WHERE event_id=?",
                    (observation["event_id"],),
                ).fetchone()
                is None
            ),
            None,
        )
        if missing_event is not None:
            raise ConflictError(
                f"Scoring claim {row['id']} has reconciliation projection "
                f"without event {missing_event}; run integrity verification."
            )
        terminal = [
            item
            for item in observations
            if item["outcome"] in {
                ReconciliationOutcome.COMPLETED.value,
                ReconciliationOutcome.DEFINITELY_ABSENT.value,
            }
        ]
        if len(terminal) > 1:
            raise ConflictError(
                f"Scoring claim {row['id']} has multiple terminal "
                "reconciliation observations; run integrity verification."
            )
        completed_at = row["completed_at"]
        if (
            terminal
            and terminal[0]["outcome"]
            == ReconciliationOutcome.COMPLETED.value
        ):
            completion = terminal[0]
            if (
                completed_at is None
                or completion["evaluation_id"] != row["evaluation_id"]
                or completion["normalized_result_digest"] is None
            ):
                raise ConflictError(
                    f"Scoring claim {row['id']} has a completed "
                    "reconciliation without its exact recovered evaluation; "
                    "run integrity verification and rebuild on a database copy."
                )
            try:
                completion_receipt = ScoringReconciliationReceipt.from_json(
                    completion["receipt_json"]
                )
            except (TypeError, ValueError) as exc:
                raise ConflictError(
                    f"Scoring claim {row['id']} has an invalid completed "
                    "reconciliation receipt; run integrity verification."
                ) from exc
            evaluation_boundary = connection.execute(
                """SELECT evaluation.authority_json,
                          evaluation_event.causation_id,
                          evaluation_event.session_id,
                          evaluation_event.occurred_at AS evaluation_occurred_at,
                          evaluation_event.stream_id AS evaluation_stream_id,
                          evaluation_event.stream_version
                              AS evaluation_stream_version,
                          reconciliation_event.stream_id
                              AS reconciliation_stream_id,
                          reconciliation_event.stream_version
                              AS reconciliation_stream_version
                   FROM task_evaluations evaluation
                   JOIN events evaluation_event
                     ON evaluation_event.event_id=evaluation.event_id
                   JOIN events reconciliation_event
                     ON reconciliation_event.event_id=?
                   WHERE evaluation.id=?""",
                (completion["event_id"], row["evaluation_id"]),
            ).fetchone()
            try:
                authority = (
                    None
                    if evaluation_boundary is None
                    else _json_object(
                        evaluation_boundary["authority_json"],
                        "Recovered evaluation authority",
                    )
                )
            except (TypeError, ValueError, ValidationError) as exc:
                raise ConflictError(
                    f"Scoring claim {row['id']} has an invalid recovered "
                    "evaluation authority; run integrity verification."
                ) from exc
            if (
                evaluation_boundary is None
                or evaluation_boundary["causation_id"] != completion["event_id"]
                or evaluation_boundary["session_id"] is not None
                or evaluation_boundary["evaluation_occurred_at"]
                != completion_receipt.completed_at
                or evaluation_boundary["evaluation_stream_id"]
                != evaluation_boundary["reconciliation_stream_id"]
                or evaluation_boundary["evaluation_stream_version"]
                <= evaluation_boundary["reconciliation_stream_version"]
                or completion_receipt.result_digest is None
                or type(authority) is not dict
                or authority.get("normalized_result_digest")
                != completion["normalized_result_digest"]
            ):
                raise ConflictError(
                    f"Scoring claim {row['id']} has a completed "
                    "reconciliation that does not bind its recovered "
                    "evaluation; run integrity verification."
                )
        terminal_at: str | None = None
        if completed_at is not None:
            if (
                terminal
                and terminal[0]["outcome"]
                == ReconciliationOutcome.DEFINITELY_ABSENT.value
            ):
                raise ConflictError(
                    f"Scoring claim {row['id']} has both a completion and a "
                    "definitely-absent terminal; run integrity verification."
                )
            claim_status = "completed"
            source = (
                "reconciliation"
                if terminal
                and terminal[0]["outcome"]
                == ReconciliationOutcome.COMPLETED.value
                else "provider_callback"
            )
            is_terminal = True
            terminal_at = (
                terminal[0]["reconciled_at"]
                if terminal
                else completed_at
            )
        elif terminal:
            claim_status = terminal[0]["outcome"]
            source = "reconciliation"
            is_terminal = True
            terminal_at = terminal[0]["reconciled_at"]
        elif observations:
            claim_status = "unknown"
            source = "reconciliation"
            is_terminal = False
        else:
            claim_status = "unreconciled"
            source = (
                "legacy_claim"
                if row["claim_schema_version"] == 1
                else "callback_admission"
            )
            is_terminal = False
        latest = observations[-1] if observations else None
        return {
            "id": row["id"],
            "event_id": row["event_id"],
            "event_type": row["event_type"],
            "claim_schema_version": row["claim_schema_version"],
            "admission_mode": metadata.get("admission_mode"),
            "attempt_id": row["attempt_id"],
            "evaluation_id": row["evaluation_id"],
            "through_sequence": row["through_sequence"],
            "provider_id": row["provider_id"],
            "provider_version": row["provider_version"],
            "action_trace_digest": row["action_trace_digest"],
            "scoring_request_digest": row["scoring_request_digest"],
            "provider_binding_digest": row["provider_binding_digest"],
            "provider_operation_digest": row["provider_operation_digest"],
            "command_hash": row["command_hash"],
            "claimed_at": row["claimed_at"],
            "status": claim_status,
            "source": source,
            "status_source": source,
            "terminal": is_terminal,
            "completed_at": completed_at,
            "terminal_at": terminal_at,
            "reconciliation_count": len(observations),
            "latest_reconciliation_id": (
                None if latest is None else latest["id"]
            ),
            "latest_reconciliation_outcome": (
                None if latest is None else latest["outcome"]
            ),
            "caller_idempotency_key_present": (
                row["idempotency_key"] is not None
            ),
            "automatic_retry_allowed": False,
            "idempotent_replay": idempotent_replay,
        }

    @classmethod
    def _scoring_reconciliation_view(
        cls,
        connection: sqlite3.Connection,
        observation: sqlite3.Row,
        *,
        idempotent_replay: bool,
    ) -> dict[str, Any]:
        claim = connection.execute(
            """SELECT claim.*, claim_event.event_type,
                      claim_event.metadata_json,
                      evaluation.recorded_at AS completed_at
               FROM performance_scoring_claims claim
               LEFT JOIN events claim_event
                 ON claim_event.event_id=claim.event_id
               LEFT JOIN task_evaluations evaluation
                 ON evaluation.id=claim.evaluation_id
               WHERE claim.id=?""",
            (observation["claim_id"],),
        ).fetchone()
        if claim is None:
            raise ConflictError(
                "Scoring reconciliation is missing its claim projection."
            )
        try:
            receipt = ScoringReconciliationReceipt.from_json(
                observation["receipt_json"]
            )
        except (TypeError, ValueError) as exc:
            raise ConflictError(
                "Scoring reconciliation has an invalid receipt projection; "
                "run integrity verification."
            ) from exc
        return {
            **cls._scoring_claim_view(
                connection,
                claim,
                idempotent_replay=idempotent_replay,
            ),
            "reconciliation_id": observation["id"],
            "reconciliation_outcome": observation["outcome"],
            "reconciliation_receipt_digest": observation["receipt_digest"],
            "reconciliation_result_digest": receipt.result_digest,
            "reconciled_at": observation["reconciled_at"],
        }

    def inspect_scoring_claim(self, claim_id: str) -> dict[str, Any]:
        """Inspect one claim without invoking either scorer or reconciler."""

        _require_id(claim_id, "claim_id")
        with self.database.read() as connection:
            orphan_reconciliation = connection.execute(
                """SELECT event.event_id
                   FROM events event
                   WHERE event.event_type='PerformanceScoringReconciled'
                     AND json_extract(
                         event.payload_json, '$.claim_id'
                     )=?
                     AND NOT EXISTS (
                         SELECT 1
                         FROM performance_scoring_reconciliations observation
                         WHERE observation.event_id=event.event_id
                     )
                   ORDER BY event.event_id LIMIT 1""",
                (claim_id,),
            ).fetchone()
            if orphan_reconciliation is not None:
                raise ConflictError(
                    "Scoring reconciliation event "
                    f"{orphan_reconciliation['event_id']} is missing its "
                    "projection; run integrity verification and rebuild on "
                    "a database copy."
                )
            row = connection.execute(
                """SELECT claim.*, claim_event.event_type,
                          claim_event.metadata_json,
                          evaluation.recorded_at AS completed_at
                   FROM performance_scoring_claims claim
                   LEFT JOIN events claim_event
                     ON claim_event.event_id=claim.event_id
                   LEFT JOIN task_evaluations evaluation
                     ON evaluation.id=claim.evaluation_id
                   WHERE claim.id=?""",
                (claim_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(
                    f"Performance scoring claim {claim_id} does not exist."
                )
            attempt = connection.execute(
                "SELECT * FROM performance_attempts WHERE id=?",
                (row["attempt_id"],),
            ).fetchone()
            if attempt is None:
                raise ValidationError(
                    "Scoring claim has no performance attempt."
                )
            self._task_for_attempt(connection, attempt)
            return self._scoring_claim_view(connection, row)

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
        connection: sqlite3.Connection,
        attempt: sqlite3.Row,
        *,
        trace_validated: bool = False,
    ) -> LearningTask:
        if not trace_validated:
            require_performance_attempt_trace_consistency(
                connection,
                attempt_id=attempt["id"],
            )
        release, release_created_at = load_stored_task_release(
            connection,
            attempt["task_release_id"],
        )
        if (
            type(release.review) is not TaskReleaseReview
            or release.corpus_release_id != attempt["corpus_release_id"]
        ):
            raise ValidationError(
                "Performance attempt is not pinned to an exact "
                "human-reviewed release."
            )
        member = next(
            (
                (status, task)
                for status, task in release.tasks
                if task.id == attempt["task_id"]
                and task.version == attempt["task_version"]
            ),
            None,
        )
        if (
            member is None
            or member[0] not in SERVICEABLE_TASK_STATUSES
            or member[1].digest != attempt["task_digest"]
        ):
            raise ValidationError(
                "Performance attempt does not match an exact serviceable "
                "task-release member."
            )
        session = connection.execute(
            """SELECT learner_id, corpus_release_id FROM sessions WHERE id=?""",
            (attempt["session_id"],),
        ).fetchone()
        if (
            session is None
            or session["learner_id"] != attempt["learner_id"]
            or session["corpus_release_id"] != attempt["corpus_release_id"]
        ):
            raise ValidationError(
                "Performance attempt crosses its session ownership or corpus "
                "boundary."
            )
        started_at = _aware_timestamp(
            attempt["started_at"],
            f"Performance attempt {attempt['id']} started_at",
        )
        if started_at < release_created_at:
            raise ValidationError(
                "Performance attempt precedes its task release."
            )
        return member[1]

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

    @staticmethod
    def _validate_runner_source_authority(
        connection: sqlite3.Connection,
        task: LearningTask,
        result: NormalizedScoringResult,
        actions: tuple[LearningAction, ...],
    ) -> None:
        """Require process-backed provenance for deterministic artifact claims."""

        contracts = {contract.key: contract for contract in task.scorer_contracts}
        action_by_id = {action.id: action for action in actions}
        for criterion in result.evaluation.criteria:
            if criterion.scorer_kind is not ScorerKind.DETERMINISTIC:
                continue
            contract = contracts.get(
                (
                    criterion.scorer_kind,
                    criterion.scorer_id,
                    criterion.scorer_version,
                )
            )
            if contract is None:
                continue
            for source_id in criterion.source_action_ids:
                source = action_by_id.get(source_id)
                if source is None or source.kind not in {
                    ActionKind.ARTIFACT_CHECKPOINT,
                    ActionKind.CHECK_RUN,
                }:
                    continue
                if source.kind is ActionKind.ARTIFACT_CHECKPOINT:
                    clause = "claim.artifact_action_id=?"
                else:
                    clause = "receipt.check_action_id=?"
                rows = connection.execute(
                    """SELECT receipt.*,
                              claim.id AS bound_claim_id,
                              claim.artifact_action_id
                                  AS bound_artifact_action_id,
                              claim.artifact_digest
                                  AS bound_artifact_digest,
                              claim.artifact_kind AS bound_artifact_kind,
                              claim.artifact_manifest_digest
                                  AS bound_artifact_manifest_digest,
                              claim.check_set_id AS bound_check_set_id,
                              claim.check_set_manifest_digest
                                  AS bound_check_set_manifest_digest,
                              claim.request_json AS bound_request_json,
                              claim.request_digest AS bound_request_digest,
                              claim.binding_json AS bound_binding_json,
                              claim.binding_digest AS bound_binding_digest
                       FROM performance_artifact_run_receipts receipt
                       JOIN performance_artifact_run_claims claim
                         ON claim.id=receipt.claim_id
                       WHERE """
                    + clause
                    + """
                         AND receipt.outcome IN (
                             'completed', 'invalid_artifact'
                         )""",
                    (source_id,),
                ).fetchall()
                if len(rows) != 1:
                    raise ValidationError(
                        "Registered deterministic scoring cannot trust a "
                        "manually asserted artifact_checkpoint or check_run; "
                        "the source action requires one exact linked terminal "
                        "artifact-run receipt."
                    )
                row = rows[0]
                try:
                    operational = OperationalArtifactRunReceipt.from_terms(
                        _json_object(
                            row["receipt_json"],
                            "Stored operational artifact-run receipt",
                        )
                    )
                    process = ArtifactProcessReceipt.from_terms(
                        _json_object(
                            row["result_json"],
                            "Stored artifact process receipt",
                        )
                    )
                    request = ArtifactRunRequest.from_terms(
                        _json_object(
                            row["bound_request_json"],
                            "Stored artifact-run request",
                        )
                    )
                    binding = ArtifactRunnerBinding.from_terms(
                        _json_object(
                            row["bound_binding_json"],
                            "Stored artifact-run binding",
                        )
                    )
                except (
                    ArtifactRunnerProtocolError,
                    TypeError,
                    ValueError,
                    ValidationError,
                ) as exc:
                    raise ValidationError(
                        "Linked artifact-run evidence has an invalid receipt "
                        "or immutable runner binding."
                    ) from exc
                exact_contract = (
                    contract.evidence_action_kinds
                    == (
                        ActionKind.ARTIFACT_CHECKPOINT,
                        ActionKind.CHECK_RUN,
                    )
                    and contract.artifact_manifests
                    == (
                        (
                            row["bound_artifact_kind"],
                            row["bound_artifact_manifest_digest"],
                        ),
                    )
                    and contract.check_set_manifests
                    == (
                        (
                            row["bound_check_set_id"],
                            row["bound_check_set_manifest_digest"],
                        ),
                    )
                    and contract.requires_attestation is False
                )
                process_bound = (
                    request.digest == row["bound_request_digest"]
                    and binding.digest == row["bound_binding_digest"]
                    and process.request == request
                    and process.binding == binding
                    and process.digest == row["result_digest"]
                    and operational.digest == row["receipt_digest"]
                    and operational.claim_id == row["bound_claim_id"]
                    and operational.artifact_action_id
                    == row["bound_artifact_action_id"]
                    and operational.result_digest == process.digest
                    and row["outcome"] == operational.outcome
                    and operational.outcome
                    == process.result.outcome.value
                )
                check_action = connection.execute(
                    "SELECT * FROM performance_actions WHERE id=?",
                    (row["check_action_id"],),
                ).fetchone()
                check_bound = False
                if check_action is not None:
                    check_payload = _json_object(
                        check_action["payload_json"],
                        "Stored artifact-run check action",
                    )
                    _require_exact_fields(
                        check_payload,
                        frozenset(
                            {
                                "check_set_id",
                                "passed",
                                "failed",
                                "errored",
                                "skipped",
                                "result_digest",
                            }
                        ),
                        "Stored artifact-run check action",
                    )
                    check_bound = (
                        check_action["attempt_id"]
                        == result.evaluation.trace_id
                        and check_action["action_type"]
                        == ActionKind.CHECK_RUN.value
                        and check_payload.get("check_set_id")
                        == row["bound_check_set_id"]
                        and check_payload.get("result_digest")
                        == process.digest
                        and check_payload.get("passed")
                        == process.result.passed
                        and check_payload.get("failed")
                        == process.result.failed
                        and check_payload.get("errored")
                        == process.result.errored
                        and check_payload.get("skipped")
                        == process.result.skipped
                    )
                if source.kind is ActionKind.ARTIFACT_CHECKPOINT:
                    source_bound = (
                        source.id == row["bound_artifact_action_id"]
                        and source.payload["artifact_digest"]
                        == row["bound_artifact_digest"]
                        and source.payload["artifact_kind"]
                        == row["bound_artifact_kind"]
                    )
                else:
                    source_bound = (
                        source.id == row["check_action_id"]
                        and source.payload["check_set_id"]
                        == row["bound_check_set_id"]
                        and source.payload["result_digest"]
                        == process.digest
                    )
                if not (
                    exact_contract
                    and process_bound
                    and check_bound
                    and source_bound
                ):
                    raise ValidationError(
                        "Registered deterministic scoring source does not "
                        "match its exact released artifact/check manifests and "
                        "terminal runner receipt."
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
        reconciliation_event_id: str | None = None,
    ) -> dict[str, Any]:
        if claim_event_id is not None and reconciliation_event_id is not None:
            raise ValidationError(
                "A task evaluation cannot have both callback and "
                "reconciliation causation."
            )
        recovered = reconciliation_event_id is not None
        if not recovered:
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
        if recovered:
            if (
                result.normalization_mode is not NormalizationMode.DIRECT_IMPORT
                or not result.shadow_only
            ):
                raise ValidationError(
                    "A recovered scoring result must remain an authority-free "
                    "shadow-only direct import."
                )
        else:
            self._validate_authority_binding(task, result)
            self._validate_runner_source_authority(
                connection,
                task,
                result,
                actions,
            )
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
            session_id=None if recovered else attempt["session_id"],
            idempotency_key=None if recovered else idempotency_key,
            correlation_id=attempt["id"],
            causation_id=(
                reconciliation_event_id
                or claim_event_id
                or attempt["id"]
            ),
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
            session_id=None if recovered else attempt["session_id"],
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
        if idempotency_key is not None:
            _require_text(idempotency_key, "idempotency_key", 256)
            if idempotency_key.startswith(
                _PERFORMANCE_TECHNICAL_EVENT_KEY_PREFIXES
            ):
                raise ValidationError(
                    "Scoring idempotency keys cannot use a TSQ reserved "
                    "technical namespace."
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
            task = self._task_for_attempt(connection, attempt)
            through_sequence = self._submission_boundary(connection, attempt_id)
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
            scoring_request_digest = canonical_digest(request.terms())
            provider_binding_digest = provider.binding_digest
            claim_id = "psc_" + command_hash
            provider_operation_digest_value = (
                provider_scoring_operation_digest(
                    claim_id=claim_id,
                    evaluation_id=request.evaluation_id,
                    scoring_request_digest=scoring_request_digest,
                    provider_binding_digest=provider_binding_digest,
                )
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
            self._task_for_attempt(connection, current_attempt)
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
            claim_event = self.database.append_event(
                connection,
                stream_id=f"learner:{current_attempt['learner_id']}",
                event_type="PerformanceScoringClaimed",
                schema_version=2,
                payload=performance_scoring_claim_v2_payload(
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
                    scoring_request_digest=scoring_request_digest,
                    provider_binding_digest=provider_binding_digest,
                    provider_operation_digest=(
                        provider_operation_digest_value
                    ),
                    provider=provider.terms(),
                ),
                metadata={
                    "claim_schema_version": 2,
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
                       id, event_id, claim_schema_version, idempotency_key,
                       attempt_id, evaluation_id, through_sequence,
                       provider_id, provider_version, action_trace_digest,
                       scoring_request_digest, provider_binding_digest,
                       provider_operation_digest, command_hash, claimed_at
                   ) VALUES (?, ?, 2, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    scoring_request_digest,
                    provider_binding_digest,
                    provider_operation_digest_value,
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
            current_task = self._task_for_attempt(connection, current_attempt)
            current_through_sequence = self._submission_boundary(
                connection, attempt_id
            )
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
            recovered_evaluation = connection.execute(
                "SELECT id FROM task_evaluations WHERE id=?",
                (request.evaluation_id,),
            ).fetchone()
            if recovered_evaluation is not None:
                return self._evaluation_report(
                    connection,
                    recovered_evaluation["id"],
                    True,
                )
            terminal_reconciliation = connection.execute(
                """SELECT outcome
                   FROM performance_scoring_reconciliations
                   WHERE claim_id=?
                     AND outcome IN ('completed', 'definitely_absent')""",
                (claim_id,),
            ).fetchone()
            if terminal_reconciliation is not None:
                raise ConflictError(
                    "Provider completion arrived after the scoring claim had "
                    "already reached terminal reconciliation state "
                    f"{terminal_reconciliation['outcome']!r}; no second "
                    "completion was recorded."
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

    def _reconciliation_claim_context(
        self,
        connection: sqlite3.Connection,
        claim_id: str,
    ) -> tuple[
        sqlite3.Row,
        sqlite3.Row,
        sqlite3.Row,
        LearningTask,
        tuple[LearningAction, ...],
        RegisteredProvider,
        ScoringRequest,
        ScoringReconciliationRequest,
    ]:
        claim = connection.execute(
            "SELECT * FROM performance_scoring_claims WHERE id=?",
            (claim_id,),
        ).fetchone()
        if claim is None:
            raise NotFoundError(
                f"Performance scoring claim {claim_id} does not exist."
            )
        if claim["claim_schema_version"] != 2:
            raise ConflictError(
                f"Performance scoring claim {claim_id} uses legacy schema v1; "
                "its provider/request boundary was not recorded completely "
                "enough for safe reconciliation."
            )
        claim_event = connection.execute(
            "SELECT * FROM events WHERE event_id=?",
            (claim["event_id"],),
        ).fetchone()
        if claim_event is None:
            raise ConflictError(
                f"Performance scoring claim {claim_id} is missing its "
                "immutable admission event."
            )
        if (
            claim_event["event_type"] != "PerformanceScoringClaimed"
            or claim_event["schema_version"] != 2
        ):
            raise ConflictError(
                f"Performance scoring claim {claim_id} does not have a v2 "
                "admission event."
            )
        payload = _json_object(
            claim_event["payload_json"],
            f"Scoring claim {claim_id} admission payload",
        )
        metadata = _json_object(
            claim_event["metadata_json"],
            f"Scoring claim {claim_id} admission metadata",
        )
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
                    "scoring_request_digest",
                    "provider_binding_digest",
                    "provider_operation_digest",
                    "provider",
                }
            ),
            f"Scoring claim {claim_id} admission payload",
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
            f"Scoring claim {claim_id} admission metadata",
        )
        if (
            metadata["claim_schema_version"] != 2
            or metadata["admission_mode"] != "pre_callback"
            or metadata["source_schema_version"] is not None
            or metadata["shadow_only"] is not True
        ):
            raise ConflictError(
                f"Scoring claim {claim_id} has an invalid v2 admission "
                "metadata boundary."
            )
        attempt = connection.execute(
            "SELECT * FROM performance_attempts WHERE id=?",
            (claim["attempt_id"],),
        ).fetchone()
        if attempt is None:
            raise ConflictError(
                f"Scoring claim {claim_id} has no performance attempt."
            )
        try:
            provider = RegisteredProvider.from_terms(payload["provider"])
        except (TypeError, ValueError) as exc:
            raise ConflictError(
                f"Scoring claim {claim_id} has an invalid provider snapshot: "
                f"{exc}"
            ) from exc
        task = self._task_for_attempt(connection, attempt)
        actions = self._typed_actions(
            connection,
            claim["attempt_id"],
            through_sequence=claim["through_sequence"],
        )
        trace_digest = action_trace_digest(actions)
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
        scoring_request = ScoringRequest(
            evaluation_id=claim["evaluation_id"],
            trace_id=claim["attempt_id"],
            task_id=task.id,
            task_version=task.version,
            task_digest=task.digest,
            action_trace_digest=trace_digest,
            criterion_ids=(
                scorer_contract.criterion_ids
                if scorer_contract is not None
                else tuple(criterion.id for criterion in task.criteria)
            ),
            scorer_contract=scorer_contract,
        )
        expected_request_digest = canonical_digest(scoring_request.terms())
        expected_operation_digest = provider_scoring_operation_digest(
            claim_id=claim["id"],
            evaluation_id=claim["evaluation_id"],
            scoring_request_digest=expected_request_digest,
            provider_binding_digest=provider.binding_digest,
        )
        expected_payload = performance_scoring_claim_v2_payload(
            claim_id=claim["id"],
            caller_idempotency_key=claim["idempotency_key"],
            attempt_id=claim["attempt_id"],
            evaluation_id=claim["evaluation_id"],
            through_sequence=claim["through_sequence"],
            provider_id=claim["provider_id"],
            provider_version=claim["provider_version"],
            action_trace_digest_value=claim["action_trace_digest"],
            command_hash=claim["command_hash"],
            claimed_at=claim["claimed_at"],
            scoring_request_digest=claim["scoring_request_digest"],
            provider_binding_digest=claim["provider_binding_digest"],
            provider_operation_digest=claim["provider_operation_digest"],
            provider=provider.terms(),
        )
        if (
            canonical_json(payload) != canonical_json(expected_payload)
            or claim["provider_id"] != provider.provider_id
            or claim["provider_version"] != provider.provider_version
            or claim["action_trace_digest"] != trace_digest
            or claim["scoring_request_digest"] != expected_request_digest
            or claim["provider_binding_digest"] != provider.binding_digest
            or claim["provider_operation_digest"]
            != expected_operation_digest
        ):
            raise ConflictError(
                f"Scoring claim {claim_id} no longer matches its immutable "
                "request/provider/trace commitments."
            )
        request = ScoringReconciliationRequest(
            claim_id=claim["id"],
            attempt_id=claim["attempt_id"],
            evaluation_id=claim["evaluation_id"],
            through_sequence=claim["through_sequence"],
            provider_id=claim["provider_id"],
            provider_version=claim["provider_version"],
            action_trace_digest=claim["action_trace_digest"],
            command_hash=claim["command_hash"],
            scoring_request_digest=claim["scoring_request_digest"],
            provider_binding_digest=claim["provider_binding_digest"],
            provider_operation_digest=claim["provider_operation_digest"],
        )
        return (
            claim,
            claim_event,
            attempt,
            task,
            actions,
            provider,
            scoring_request,
            request,
        )

    def reconcile_scoring_claim(
        self,
        claim_id: str,
        registry: ScoringReconciliationRegistry,
        reconciler_id: str,
        reconciler_version: str,
        *,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Observe one admitted callback without retrying or rescoring it."""

        _require_id(claim_id, "claim_id")
        _require_id(reconciler_id, "reconciler_id")
        _require_id(reconciler_version, "reconciler_version")
        if not isinstance(registry, ScoringReconciliationRegistry):
            raise ValidationError(
                "registry must be a ScoringReconciliationRegistry."
            )
        if idempotency_key is not None:
            _require_text(idempotency_key, "idempotency_key", 256)
            if idempotency_key.startswith(
                _PERFORMANCE_TECHNICAL_EVENT_KEY_PREFIXES
            ):
                raise ValidationError(
                    "Reconciliation idempotency keys cannot use a TSQ "
                    "reserved technical namespace."
                )
        if self.database.read_only:
            raise ConflictError(
                "A read-only database cannot record reconciliation."
            )
        reconciled_at_value = _now(now)

        with self.database.read() as connection:
            learner = connection.execute(
                """SELECT attempt.*
                   FROM performance_scoring_claims claim
                   JOIN performance_attempts attempt
                     ON attempt.id=claim.attempt_id
                   WHERE claim.id=?""",
                (claim_id,),
            ).fetchone()
            if learner is None:
                raise NotFoundError(
                    f"Performance scoring claim {claim_id} does not exist."
                )
            # Reconciliation may append a recovered evaluation, so it obeys
            # the same learner-wide quarantine boundary as new scoring and
            # direct imports. Check before even invoking the observational
            # adapter; an exact-key replay is also withheld while evidence is
            # unsafe.
            self.database.require_learner_evidence_safe(
                learner["learner_id"],
                connection,
            )
            self._task_for_attempt(connection, learner)
            if idempotency_key is not None:
                prior_observation = connection.execute(
                    """SELECT * FROM performance_scoring_reconciliations
                       WHERE idempotency_key=?""",
                    (idempotency_key,),
                ).fetchone()
                if prior_observation is not None:
                    if (
                        prior_observation["claim_id"] != claim_id
                        or prior_observation["reconciler_id"]
                        != reconciler_id
                        or prior_observation["reconciler_version"]
                        != reconciler_version
                    ):
                        raise ConflictError(
                            "Idempotency key was already used for a different "
                            "reconciliation command."
                        )
                    return self._scoring_reconciliation_view(
                        connection,
                        prior_observation,
                        idempotent_replay=True,
                    )
                if connection.execute(
                    """SELECT 1 FROM performance_scoring_claims
                       WHERE idempotency_key=?""",
                    (idempotency_key,),
                ).fetchone() is not None or connection.execute(
                    "SELECT 1 FROM events WHERE idempotency_key=?",
                    (idempotency_key,),
                ).fetchone() is not None:
                    raise ConflictError(
                        "Idempotency key was already used for a different "
                        "command."
                    )
            (
                claim,
                _claim_event,
                _attempt,
                _task,
                _actions,
                _provider,
                _scoring_request,
                request,
            ) = self._reconciliation_claim_context(connection, claim_id)
            existing_evaluation = connection.execute(
                "SELECT 1 FROM task_evaluations WHERE id=?",
                (claim["evaluation_id"],),
            ).fetchone()
            terminal = connection.execute(
                """SELECT * FROM performance_scoring_reconciliations
                   WHERE claim_id=?
                     AND outcome IN ('completed', 'definitely_absent')""",
                (claim_id,),
            ).fetchone()
            if terminal is not None:
                raise ConflictError(
                    "Scoring claim already has terminal reconciliation "
                    f"{terminal['outcome']!r} under a different command; "
                    "inspect the claim instead of recording another "
                    "observation."
                )
            if existing_evaluation is not None:
                raise ConflictError(
                    "Scoring claim is already completed; inspect it instead "
                    "of recording a reconciliation observation."
                )
            try:
                registered_reconciler = registry.inspect(
                    claim["provider_id"],
                    claim["provider_version"],
                    reconciler_id,
                    reconciler_version,
                )
            except (LookupError, RuntimeError, ValueError) as exc:
                raise ValidationError(
                    f"Scoring reconciliation failed safely: {exc}"
                ) from exc

        # This is the sole external boundary.  The reconciliation registry
        # accepts lookup-only adapters and rejects anything exposing score().
        # It runs without SQLite's writer lock.
        try:
            reconciliation = registry.reconcile(
                reconciler_id,
                reconciler_version,
                request,
            )
        except (LookupError, RuntimeError, ValueError) as exc:
            raise ValidationError(
                f"Scoring reconciliation failed safely: {exc}"
            ) from exc
        if (
            reconciliation.reconciler.binding_digest
            != registered_reconciler.binding_digest
            or reconciliation.request.digest != request.digest
        ):
            raise ValidationError(
                "Scoring reconciliation changed its registered request or "
                "authority boundary."
            )
        if (
            reconciliation.outcome
            is ReconciliationOutcome.DEFINITELY_ABSENT
            and not reconciliation.reconciler.can_prove_absence
        ):
            raise ValidationError(
                "A reconciler without durable absence authority cannot close "
                "a claim as definitely absent."
            )

        receipt = reconciliation.observation.receipt
        try:
            claim_time = from_timestamp(claim["claimed_at"])
            observed_time = from_timestamp(receipt.observed_at)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Reconciliation temporal boundary is invalid: {exc}"
            ) from exc
        if observed_time < claim_time:
            raise ValidationError(
                "A reconciliation observation cannot predate callback "
                "admission."
            )
        if observed_time > reconciled_at_value:
            raise ValidationError(
                "A reconciliation observation cannot be recorded before its "
                "claimed observation time."
            )
        normalized_result: NormalizedScoringResult | None = None
        completion_time: datetime | None = None
        if reconciliation.outcome is ReconciliationOutcome.COMPLETED:
            imported = reconciliation.imported_evaluation
            if imported is None or receipt.completed_at is None:
                raise ValidationError(
                    "Completed reconciliation omitted its imported result or "
                    "completion time."
                )
            try:
                normalized_result = _normalize_recovered_imported_evaluation(
                    _scoring_request,
                    imported,
                    _provider,
                    request.provider_operation_digest,
                )
                completion_time = from_timestamp(receipt.completed_at)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Recovered scoring result failed safely: {exc}"
                ) from exc
            if completion_time < claim_time:
                raise ValidationError(
                    "Recovered scoring completion cannot predate callback "
                    "admission."
                )

        receipt_digest = receipt.digest
        command_hash = _command_hash(
            {
                "operation": "reconcile_scoring_claim",
                "claim_id": claim_id,
                "provider_operation_digest": request.provider_operation_digest,
                "reconciler_id": reconciler_id,
                "reconciler_version": reconciler_version,
                "reconciliation_binding_digest": (
                    reconciliation.reconciler.binding_digest
                ),
                "receipt_digest": receipt_digest,
                "caller_idempotency_key": idempotency_key,
            }
        )
        reconciliation_id = "psr_" + command_hash
        reconciled_at = to_timestamp(reconciled_at_value)

        with self.database.transaction() as connection:
            current_learner = connection.execute(
                """SELECT attempt.*
                   FROM performance_scoring_claims claim
                   JOIN performance_attempts attempt
                     ON attempt.id=claim.attempt_id
                   WHERE claim.id=?""",
                (claim_id,),
            ).fetchone()
            if current_learner is None:
                raise ConflictError(
                    "Performance scoring claim disappeared during "
                    "reconciliation."
                )
            # Recheck under the writer transaction so a quarantine committed
            # while the external observer ran wins the race and leaves no
            # reconciliation event or projection row.
            self.database.require_learner_evidence_safe(
                current_learner["learner_id"],
                connection,
            )
            self._task_for_attempt(connection, current_learner)
            if idempotency_key is not None:
                prior_observation = connection.execute(
                    """SELECT * FROM performance_scoring_reconciliations
                       WHERE idempotency_key=?""",
                    (idempotency_key,),
                ).fetchone()
                if prior_observation is not None:
                    if (
                        prior_observation["claim_id"] != claim_id
                        or prior_observation["reconciler_id"]
                        != reconciler_id
                        or prior_observation["reconciler_version"]
                        != reconciler_version
                    ):
                        raise ConflictError(
                            "Idempotency key was concurrently used for a "
                            "different reconciliation command."
                        )
                    return self._scoring_reconciliation_view(
                        connection,
                        prior_observation,
                        idempotent_replay=True,
                    )
            (
                current_claim,
                current_claim_event,
                current_attempt,
                current_task,
                current_actions,
                current_provider,
                current_scoring_request,
                current_request,
            ) = self._reconciliation_claim_context(connection, claim_id)
            if (
                current_request.digest != request.digest
                or current_provider.terms() != _provider.terms()
                or current_scoring_request.terms()
                != _scoring_request.terms()
            ):
                raise ConflictError(
                    "Scoring claim changed during reconciliation."
                )
            existing_evaluation = connection.execute(
                "SELECT id FROM task_evaluations WHERE id=?",
                (current_claim["evaluation_id"],),
            ).fetchone()
            if existing_evaluation is not None:
                raise ConflictError(
                    "Scoring completed while reconciliation lookup was in "
                    "flight; the observation was not recorded. Inspect the "
                    "claim before taking any further action."
                )
            prior_receipt = connection.execute(
                """SELECT * FROM performance_scoring_reconciliations
                   WHERE claim_id=? AND receipt_digest=?""",
                (claim_id, receipt_digest),
            ).fetchone()
            if prior_receipt is not None:
                raise ConflictError(
                    "The same reconciliation receipt was already recorded "
                    "under a different command; inspect that immutable "
                    "observation instead."
                )
            terminal = connection.execute(
                """SELECT *
                   FROM performance_scoring_reconciliations
                   WHERE claim_id=?
                     AND outcome IN ('completed', 'definitely_absent')""",
                (claim_id,),
            ).fetchone()
            if terminal is not None:
                raise ConflictError(
                    "Another reconciliation reached terminal state while "
                    "this lookup was in flight; this observation was not "
                    "recorded."
                )
            reserved_key = performance_scoring_reconciliation_event_key(
                command_hash
            )
            orphan_event = connection.execute(
                "SELECT event_id FROM events WHERE idempotency_key=?",
                (reserved_key,),
            ).fetchone()
            if orphan_event is not None:
                raise ConflictError(
                    "Reconciliation event history already contains this "
                    "observation but its projection is missing."
                )
            normalized_result_digest = (
                None
                if normalized_result is None
                else normalized_result.digest
            )
            reconciliation_event = self.database.append_event(
                connection,
                stream_id=f"learner:{current_attempt['learner_id']}",
                event_type="PerformanceScoringReconciled",
                schema_version=PERFORMANCE_EVENT_SCHEMA_VERSION,
                payload=performance_scoring_reconciliation_payload(
                    reconciliation_id=reconciliation_id,
                    caller_idempotency_key=idempotency_key,
                    claim_id=claim_id,
                    attempt_id=current_claim["attempt_id"],
                    evaluation_id=current_claim["evaluation_id"],
                    outcome=reconciliation.outcome.value,
                    scoring_request_digest=(
                        current_claim["scoring_request_digest"]
                    ),
                    provider_binding_digest=(
                        current_claim["provider_binding_digest"]
                    ),
                    provider_operation_digest=(
                        current_claim["provider_operation_digest"]
                    ),
                    reconciler_id=reconciler_id,
                    reconciler_version=reconciler_version,
                    reconciliation_binding_digest=(
                        reconciliation.reconciler.binding_digest
                    ),
                    receipt=receipt.terms(),
                    receipt_digest=receipt_digest,
                    normalized_result_digest=normalized_result_digest,
                    reconciled_at=reconciled_at,
                    command_hash=command_hash,
                    reconciler=reconciliation.reconciler.terms(),
                ),
                metadata={
                    "reconciliation_schema_version": 1,
                    "command_hash": command_hash,
                    "observational_only": True,
                    "automatic_retry_allowed": False,
                    "projection_applied": False,
                    "certification_applied": False,
                    "skill_authority": False,
                    "shadow_only": True,
                },
                learner_id=current_attempt["learner_id"],
                session_id=None,
                idempotency_key=reserved_key,
                correlation_id=current_attempt["id"],
                causation_id=current_claim_event["event_id"],
                occurred_at=reconciled_at_value,
            )
            connection.execute(
                """INSERT INTO performance_scoring_reconciliations(
                       id, event_id, idempotency_key, claim_id, attempt_id,
                       evaluation_id, outcome, scoring_request_digest,
                       provider_binding_digest, provider_operation_digest,
                       reconciler_id, reconciler_version,
                       reconciliation_binding_digest, receipt_json,
                       receipt_digest, normalized_result_digest,
                       reconciled_at, command_hash
                   ) VALUES (
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                   )""",
                (
                    reconciliation_id,
                    reconciliation_event["event_id"],
                    idempotency_key,
                    claim_id,
                    current_claim["attempt_id"],
                    current_claim["evaluation_id"],
                    reconciliation.outcome.value,
                    current_claim["scoring_request_digest"],
                    current_claim["provider_binding_digest"],
                    current_claim["provider_operation_digest"],
                    reconciler_id,
                    reconciler_version,
                    reconciliation.reconciler.binding_digest,
                    canonical_json(receipt.terms()),
                    receipt_digest,
                    normalized_result_digest,
                    reconciled_at,
                    command_hash,
                ),
            )
            if normalized_result is not None:
                if completion_time is None:
                    raise ValidationError(
                        "Completed reconciliation has no completion timestamp."
                    )
                report = self._record_result(
                    connection,
                    current_attempt,
                    current_task,
                    current_actions,
                    current_claim["through_sequence"],
                    normalized_result,
                    idempotency_key=None,
                    occurred=completion_time,
                    command_hash=current_claim["command_hash"],
                    reconciliation_event_id=(
                        reconciliation_event["event_id"]
                    ),
                )
                return {
                    **report,
                    "claim_id": claim_id,
                    "status": "completed",
                    "status_source": "reconciliation",
                    "terminal": True,
                    "reconciliation_id": reconciliation_id,
                    "reconciliation_outcome": reconciliation.outcome.value,
                    "reconciliation_receipt_digest": receipt_digest,
                    "reconciliation_result_digest": receipt.result_digest,
                    "reconciled_at": reconciled_at,
                    "reconciliation_count": connection.execute(
                        """SELECT COUNT(*) AS n
                           FROM performance_scoring_reconciliations
                           WHERE claim_id=?""",
                        (claim_id,),
                    ).fetchone()["n"],
                    "automatic_retry_allowed": False,
                }
            row = connection.execute(
                """SELECT claim.*, claim_event.event_type,
                          claim_event.metadata_json,
                          evaluation.recorded_at AS completed_at
                   FROM performance_scoring_claims claim
                   LEFT JOIN events claim_event
                     ON claim_event.event_id=claim.event_id
                   LEFT JOIN task_evaluations evaluation
                     ON evaluation.id=claim.evaluation_id
                   WHERE claim.id=?""",
                (claim_id,),
            ).fetchone()
            return {
                **self._scoring_claim_view(connection, row),
                "reconciliation_id": reconciliation_id,
                "reconciliation_outcome": reconciliation.outcome.value,
                "reconciliation_receipt_digest": receipt_digest,
                "reconciliation_result_digest": receipt.result_digest,
                "reconciled_at": reconciled_at,
            }

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
        if type(imported) is not ImportedEvaluation or any(
            type(item) is not ImportedCriterionResult
            for item in imported.criteria
        ):
            raise ValidationError(
                "imported must be an exact ImportedEvaluation containing exact "
                "ImportedCriterionResult values."
            )
        if type(declared_kind) is not ScorerKind:
            raise ValidationError("declared_kind must be an exact ScorerKind.")
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
            task = self._task_for_attempt(connection, attempt)
            through_sequence = self._submission_boundary(connection, attempt_id)
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
        connection: sqlite3.Connection,
        evaluation_id: str,
        replay: bool,
        *,
        evidence_validated: bool = False,
    ) -> dict[str, Any]:
        row = connection.execute(
            """SELECT evaluation.*, bundle.bundle_digest, bundle.bundle_json,
                      bundle.projection_applied, bundle.certification_applied,
                      attempt.learner_id
               FROM task_evaluations evaluation
               JOIN shadow_evidence_bundles bundle
                 ON bundle.evaluation_id=evaluation.id
               JOIN performance_attempts attempt
                 ON attempt.id=evaluation.attempt_id
               WHERE evaluation.id=?""",
            (evaluation_id,),
        ).fetchone()
        if row is None:
            raise ValidationError("Evaluation event lacks its shadow projection.")
        if not evidence_validated:
            require_performance_projection_consistency(
                connection,
                learner_id=row["learner_id"],
                comparison_names=("evaluations", "bundles"),
            )
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
        cls,
        connection: sqlite3.Connection,
        attempt_id: str,
        replay: bool,
        *,
        trace_validated: bool = False,
    ) -> dict[str, Any]:
        attempt = connection.execute(
            "SELECT * FROM performance_attempts WHERE id=?", (attempt_id,)
        ).fetchone()
        if attempt is None:
            raise NotFoundError(f"Performance attempt {attempt_id} does not exist.")
        task = cls._task_for_attempt(
            connection,
            attempt,
            trace_validated=trace_validated,
        )
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
            "task": task.terms(),
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
            require_performance_projection_consistency(
                connection,
                learner_id=owner["learner_id"],
                comparison_names=(
                    "attempts",
                    "actions",
                    "evaluations",
                    "bundles",
                ),
            )
            report = self._attempt_report(
                connection,
                attempt_id,
                False,
                trace_validated=True,
            )
            evaluations = connection.execute(
                """SELECT evaluation.id FROM task_evaluations evaluation
                   WHERE evaluation.attempt_id=?
                   ORDER BY evaluation.recorded_at, evaluation.id""",
                (attempt_id,),
            ).fetchall()
            report["evaluations"] = [
                self._evaluation_report(
                    connection,
                    row["id"],
                    False,
                    evidence_validated=True,
                )
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
    task_imported_at: dict[tuple[str, int], datetime] = {}
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
            imported_at = _aware_timestamp(
                row["imported_at"], f"{label} imported_at"
            )
        except ValidationError as exc:
            errors.append(str(exc))
        else:
            task_imported_at[(row["task_id"], row["task_version"])] = (
                imported_at
            )
        tasks[(row["task_id"], row["task_version"])] = task

    release_rows = connection.execute(
        "SELECT * FROM performance_task_releases ORDER BY id"
    ).fetchall()
    release_ids = {row["id"] for row in release_rows}
    for row in release_rows:
        label = f"performance task release {row['id']}"
        release_created_at: datetime | None = None
        try:
            release_created_at = _aware_timestamp(
                row["created_at"], f"{label} created_at"
            )
        except ValidationError as exc:
            errors.append(str(exc))
        review_terms = decode(row["review_json"], f"{label} review")
        if review_terms is None:
            continue
        try:
            review = _release_authority_from_terms(review_terms)
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
        all_misconception_objectives = release_misconception_objectives(
            connection,
            row["corpus_release_id"],
            accepted_only=False,
            exclude_revoked=False,
        )
        accepted_misconception_objectives = release_misconception_objectives(
            connection,
            row["corpus_release_id"],
            accepted_only=True,
            exclude_revoked=False,
        )
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
            imported_at = task_imported_at.get((task.id, task.version))
            if (
                release_created_at is not None
                and imported_at is not None
                and imported_at > release_created_at
            ):
                errors.append(
                    f"{label}: member {task.id}@{task.version} was imported "
                    "after publication"
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
            task_misconception_objectives = (
                accepted_misconception_objectives
                if member["status"] in SERVICEABLE_TASK_STATUSES
                else all_misconception_objectives
            )
            for criterion in task.criteria:
                criterion_concepts = set(criterion.concept_ids)
                for misconception_id in criterion.misconception_ids:
                    if (
                        not criterion.objective_ids
                        and misconception_concepts.get(misconception_id)
                        not in criterion_concepts
                    ):
                        errors.append(
                            f"{label}: task {task.id} criterion "
                            f"{criterion.id} misconception "
                            f"{misconception_id} is outside its concept mapping"
                        )
            for criterion_id, misconception_id in (
                missing_objective_misconception_bindings(
                    task,
                    task_misconception_objectives,
                )
            ):
                errors.append(
                    f"{label}: task {task.id} criterion "
                    f"{criterion_id} misconception "
                    f"{misconception_id} is not mapped to any of its "
                    "objectives"
                )
        try:
            reconstructed = PerformanceTaskRelease(
                title=row["title"],
                corpus_release_id=row["corpus_release_id"],
                review=review,
                tasks=tuple(definitions),
                schema_version=(
                    TASK_RELEASE_SCHEMA_VERSION
                    if type(review) is TaskReleaseReview
                    else SYNTHETIC_TASK_LAB_RELEASE_SCHEMA_VERSION
                ),
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
            authority_at = _aware_timestamp(
                (
                    review.reviewed_at
                    if type(review) is TaskReleaseReview
                    else review.declared_at
                ),
                (
                    f"{label} reviewed_at"
                    if type(review) is TaskReleaseReview
                    else f"{label} declared_at"
                ),
            )
            created_at = _aware_timestamp(
                row["created_at"], f"{label} created_at"
            )
            if authority_at > created_at:
                errors.append(
                    f"{label}: publication precedes review or declaration"
                )
        except ValidationError as exc:
            errors.append(str(exc))
        if row["sealed_at"] != row["created_at"]:
            errors.append(f"{label}: seal boundary differs from publication")

    performance_events = connection.execute(
        """SELECT * FROM events WHERE event_type IN (
               'PerformanceTaskStarted', 'PerformanceActionRecorded',
               'PerformanceScoringClaimed',
               'PerformanceScoringClaimMigrated',
               'PerformanceScoringReconciled',
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
            "PerformanceScoringReconciled",
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
        stored_release: PerformanceTaskRelease | None = None
        stored_release_created_at: datetime | None = None
        try:
            stored_release, stored_release_created_at = (
                load_stored_task_release(
                    connection,
                    row["task_release_id"],
                )
            )
        except (NotFoundError, ConflictError, ValidationError) as exc:
            errors.append(f"{label}: invalid task release ({exc})")
        stored_member = (
            next(
                (
                    (status, member_task)
                    for status, member_task in stored_release.tasks
                    if member_task.id == row["task_id"]
                    and member_task.version == row["task_version"]
                ),
                None,
            )
            if stored_release is not None
            else None
        )
        if (
            stored_release is None
            or type(stored_release.review) is not TaskReleaseReview
            or stored_release.corpus_release_id != row["corpus_release_id"]
            or stored_member is None
            or stored_member[0] not in SERVICEABLE_TASK_STATUSES
            or stored_member[1].digest != row["task_digest"]
            or row["task_release_id"] not in release_ids
        ):
            errors.append(f"{label}: invalid task-release boundary")
        elif event is not None and stored_release_created_at is not None:
            try:
                if _aware_timestamp(
                    event["occurred_at"], f"{label} event occurrence"
                ) < stored_release_created_at:
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
        reconciliation_causation = connection.execute(
            """SELECT reconciliation.event_id
               FROM performance_scoring_reconciliations reconciliation
               WHERE reconciliation.evaluation_id=?
                 AND reconciliation.outcome='completed'""",
            (row["id"],),
        ).fetchone()
        recovered = reconciliation_causation is not None
        expected_causation_id = (
            reconciliation_causation["event_id"]
            if recovered
            else (
                scoring_causation["event_id"]
                if scoring_causation is not None
                and scoring_causation["event_type"]
                == "PerformanceScoringClaimed"
                else attempt["id"]
            )
        )
        expected_session_id = None if recovered else attempt["session_id"]
        if (
            event["schema_version"] != PERFORMANCE_EVENT_SCHEMA_VERSION
            or event["stream_id"] != f"learner:{attempt['learner_id']}"
            or event["learner_id"] != attempt["learner_id"]
            or event["session_id"] != expected_session_id
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
    scoring_claim_rows_by_id = {
        row["id"]: row for row in scoring_claim_rows
    }
    claim_provider_snapshots: dict[str, RegisteredProvider] = {}
    claim_scoring_requests: dict[str, ScoringRequest] = {}
    projected_claim_events: set[str] = set()
    claim_rows_by_evaluation: dict[str, list[sqlite3.Row]] = {}
    for row in scoring_claim_rows:
        label = f"performance scoring claim {row['id']}"
        attempt = attempts.get(row["attempt_id"])
        claim_rows_by_evaluation.setdefault(row["evaluation_id"], []).append(row)
        expected_claim_event_key: str | None = None
        provider_snapshot: RegisteredProvider | None = None
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
            if row["claim_schema_version"] not in {1, 2}:
                raise ValidationError(
                    "claim_schema_version must be 1 or 2."
                )
            if row["claim_schema_version"] == 1:
                if any(
                    row[field] is not None
                    for field in (
                        "scoring_request_digest",
                        "provider_binding_digest",
                        "provider_operation_digest",
                    )
                ):
                    raise ValidationError(
                        "legacy claim carries prospective reconciliation "
                        "digests."
                    )
            else:
                for field in (
                    "scoring_request_digest",
                    "provider_binding_digest",
                    "provider_operation_digest",
                ):
                    _require_digest(row[field], field)
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
            claim_schema_version = row["claim_schema_version"]
            claim_payload_fields = {
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
            if claim_schema_version == 2:
                claim_payload_fields |= {
                    "scoring_request_digest",
                    "provider_binding_digest",
                    "provider_operation_digest",
                    "provider",
                }
            exact(
                claim_payload,
                claim_payload_fields,
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
            if claim_schema_version == 2 and claim_payload is not None:
                try:
                    provider_snapshot = RegisteredProvider.from_terms(
                        claim_payload.get("provider")
                    )
                except (TypeError, ValueError) as exc:
                    errors.append(
                        f"{label}: invalid provider snapshot ({exc})"
                    )
            expected_payload = (
                performance_scoring_claim_v2_payload(
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
                    scoring_request_digest=row["scoring_request_digest"],
                    provider_binding_digest=row[
                        "provider_binding_digest"
                    ],
                    provider_operation_digest=row[
                        "provider_operation_digest"
                    ],
                    provider=(
                        provider_snapshot.terms()
                        if provider_snapshot is not None
                        else {}
                    ),
                )
                if claim_schema_version == 2
                else performance_scoring_claim_payload(
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
                claim_metadata.get("claim_schema_version")
                != claim_schema_version
                or claim_metadata.get("admission_mode") != expected_mode
                or claim_metadata.get("source_schema_version")
                != expected_source_version
                or claim_metadata.get("shadow_only") is not True
            ):
                errors.append(f"{label}: admission event metadata mismatch")
            if (
                claim_event["schema_version"]
                != (2 if claim_schema_version == 2 else 1)
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
        if row["claim_schema_version"] == 2:
            task = attempt_tasks.get(row["attempt_id"])
            if task is None or provider_snapshot is None:
                errors.append(
                    f"{label}: cannot reconstruct v2 request/provider boundary"
                )
            else:
                scorer_contract = next(
                    (
                        contract
                        for contract in task.scorer_contracts
                        if contract.key
                        == (
                            provider_snapshot.declared_kind,
                            provider_snapshot.provider_id,
                            provider_snapshot.provider_version,
                        )
                    ),
                    None,
                )
                try:
                    scoring_request = ScoringRequest(
                        evaluation_id=row["evaluation_id"],
                        trace_id=row["attempt_id"],
                        task_id=task.id,
                        task_version=task.version,
                        task_digest=task.digest,
                        action_trace_digest=trace_digest,
                        criterion_ids=(
                            scorer_contract.criterion_ids
                            if scorer_contract is not None
                            else tuple(
                                criterion.id
                                for criterion in task.criteria
                            )
                        ),
                        scorer_contract=scorer_contract,
                    )
                    expected_scoring_request_digest = canonical_digest(
                        scoring_request.terms()
                    )
                    expected_provider_operation_digest = (
                        provider_scoring_operation_digest(
                            claim_id=row["id"],
                            evaluation_id=row["evaluation_id"],
                            scoring_request_digest=(
                                expected_scoring_request_digest
                            ),
                            provider_binding_digest=(
                                provider_snapshot.binding_digest
                            ),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    errors.append(
                        f"{label}: cannot reconstruct v2 scoring request "
                        f"({exc})"
                    )
                else:
                    if (
                        provider_snapshot.provider_id != row["provider_id"]
                        or provider_snapshot.provider_version
                        != row["provider_version"]
                        or provider_snapshot.binding_digest
                        != row["provider_binding_digest"]
                        or row["scoring_request_digest"]
                        != expected_scoring_request_digest
                        or row["provider_operation_digest"]
                        != expected_provider_operation_digest
                    ):
                        errors.append(
                            f"{label}: v2 request/provider commitment mismatch"
                        )
                    else:
                        claim_provider_snapshots[row["id"]] = (
                            provider_snapshot
                        )
                        claim_scoring_requests[row["id"]] = scoring_request
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
        completion_reconciliation = connection.execute(
            """SELECT reconciliation.event_id,
                      reconciliation.normalized_result_digest
               FROM performance_scoring_reconciliations reconciliation
               WHERE reconciliation.claim_id=?
                 AND reconciliation.outcome='completed'""",
            (row["id"],),
        ).fetchone()
        recovered_completion = completion_reconciliation is not None
        if (
            event["event_type"] != "TaskEvaluationRecorded"
            or event["idempotency_key"]
            != (None if recovered_completion else row["idempotency_key"])
            or metadata is None
            or metadata.get("command_hash") != row["command_hash"]
            or payload is None
            or payload.get("attempt_id") != row["attempt_id"]
            or payload.get("through_sequence") != row["through_sequence"]
            or type(evaluation_terms) is not dict
            or evaluation_terms.get("id") != row["evaluation_id"]
            or type(normalized_terms) is not dict
            or normalized_terms.get("normalization_mode")
            != (
                NormalizationMode.DIRECT_IMPORT.value
                if recovered_completion
                else NormalizationMode.REGISTERED_PROVIDER.value
            )
            or type(provider_terms) is not dict
            or (
                not recovered_completion
                and (
                    provider_terms.get("provider_id") != row["provider_id"]
                    or provider_terms.get("provider_version")
                    != row["provider_version"]
                )
            )
            or (
                recovered_completion
                and (
                    event["session_id"] is not None
                    or event["causation_id"]
                    != completion_reconciliation["event_id"]
                    or (
                        type(authority_terms) is not dict
                        or authority_terms.get("normalized_result_digest")
                        != completion_reconciliation[
                            "normalized_result_digest"
                        ]
                    )
                )
            )
            or (
                not recovered_completion
                and claim_event is not None
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

    reconciliation_rows = connection.execute(
        """SELECT observation.*
           FROM performance_scoring_reconciliations observation
           ORDER BY (
               SELECT event.stream_id FROM events event
               WHERE event.event_id=observation.event_id
           ), (
               SELECT event.stream_version FROM events event
               WHERE event.event_id=observation.event_id
           ), observation.id"""
    ).fetchall()
    projected_reconciliation_events: set[str] = set()
    terminal_claims: set[str] = set()
    for row in reconciliation_rows:
        label = f"performance scoring reconciliation {row['id']}"
        claim = scoring_claim_rows_by_id.get(row["claim_id"])
        attempt = attempts.get(row["attempt_id"])
        event = events_by_type["PerformanceScoringReconciled"].get(
            row["event_id"]
        )
        if event is None:
            errors.append(f"{label}: missing PerformanceScoringReconciled event")
            continue
        projected_reconciliation_events.add(event["event_id"])
        payload = decode(event["payload_json"], f"{label} event payload")
        metadata = decode(event["metadata_json"], f"{label} event metadata")
        exact(
            payload,
            {
                "reconciliation_id",
                "caller_idempotency_key",
                "claim_id",
                "attempt_id",
                "evaluation_id",
                "outcome",
                "scoring_request_digest",
                "provider_binding_digest",
                "provider_operation_digest",
                "reconciler_id",
                "reconciler_version",
                "reconciliation_binding_digest",
                "receipt",
                "receipt_digest",
                "normalized_result_digest",
                "reconciled_at",
                "command_hash",
                "reconciler",
            },
            f"{label} event payload",
        )
        exact(
            metadata,
            {
                "reconciliation_schema_version",
                "command_hash",
                "observational_only",
                "automatic_retry_allowed",
                "projection_applied",
                "certification_applied",
                "skill_authority",
                "shadow_only",
            },
            f"{label} event metadata",
        )
        receipt: ScoringReconciliationReceipt | None = None
        reconciler: RegisteredReconciler | None = None
        try:
            receipt_terms = _json_object(
                row["receipt_json"], f"{label} receipt projection"
            )
            receipt = ScoringReconciliationReceipt.from_terms(receipt_terms)
            if payload is not None:
                reconciler = RegisteredReconciler.from_terms(
                    payload.get("reconciler")
                )
            _require_id(row["id"], "reconciliation_id")
            if row["idempotency_key"] is not None:
                _require_text(
                    row["idempotency_key"], "idempotency_key", 256
                )
            _require_digest(
                row["reconciliation_binding_digest"],
                "reconciliation_binding_digest",
            )
            _require_digest(row["receipt_digest"], "receipt_digest")
            _require_digest(row["command_hash"], "command_hash")
            _aware_timestamp(row["reconciled_at"], f"{label} reconciled_at")
            if row["normalized_result_digest"] is not None:
                _require_digest(
                    row["normalized_result_digest"],
                    "normalized_result_digest",
                )
        except (TypeError, ValueError, ValidationError) as exc:
            errors.append(f"{label}: invalid reconciliation terms ({exc})")
        if claim is None or attempt is None:
            errors.append(f"{label}: missing claim or performance attempt")
            continue
        if claim["claim_schema_version"] != 2:
            errors.append(f"{label}: legacy claim cannot be reconciled")
        if row["claim_id"] in terminal_claims:
            errors.append(
                f"{label}: observation follows the claim's first terminal "
                "reconciliation"
            )
        if row["outcome"] in {
            ReconciliationOutcome.COMPLETED.value,
            ReconciliationOutcome.DEFINITELY_ABSENT.value,
        }:
            terminal_claims.add(row["claim_id"])
        try:
            outcome = ReconciliationOutcome(row["outcome"])
        except (TypeError, ValueError):
            errors.append(f"{label}: unknown reconciliation outcome")
            outcome = None
        expected_command_hash = (
            _command_hash(
                {
                    "operation": "reconcile_scoring_claim",
                    "claim_id": row["claim_id"],
                    "provider_operation_digest": row[
                        "provider_operation_digest"
                    ],
                    "reconciler_id": row["reconciler_id"],
                    "reconciler_version": row["reconciler_version"],
                    "reconciliation_binding_digest": row[
                        "reconciliation_binding_digest"
                    ],
                    "receipt_digest": row["receipt_digest"],
                    "caller_idempotency_key": row["idempotency_key"],
                }
            )
            if all(
                row[field] is not None
                for field in (
                    "provider_operation_digest",
                    "reconciler_id",
                    "reconciler_version",
                    "reconciliation_binding_digest",
                    "receipt_digest",
                )
            )
            else None
        )
        expected_payload = (
            performance_scoring_reconciliation_payload(
                reconciliation_id=row["id"],
                caller_idempotency_key=row["idempotency_key"],
                claim_id=row["claim_id"],
                attempt_id=row["attempt_id"],
                evaluation_id=row["evaluation_id"],
                outcome=row["outcome"],
                scoring_request_digest=row["scoring_request_digest"],
                provider_binding_digest=row["provider_binding_digest"],
                provider_operation_digest=row["provider_operation_digest"],
                reconciler_id=row["reconciler_id"],
                reconciler_version=row["reconciler_version"],
                reconciliation_binding_digest=row[
                    "reconciliation_binding_digest"
                ],
                receipt=receipt.terms() if receipt is not None else {},
                receipt_digest=row["receipt_digest"],
                normalized_result_digest=row["normalized_result_digest"],
                reconciled_at=row["reconciled_at"],
                command_hash=row["command_hash"],
                reconciler=(
                    reconciler.terms() if reconciler is not None else {}
                ),
            )
            if payload is not None
            else None
        )
        if payload is not None and expected_payload is not None and (
            canonical_json(payload) != canonical_json(expected_payload)
        ):
            errors.append(f"{label}: event payload mismatch")
        if metadata is not None and (
            metadata.get("reconciliation_schema_version") != 1
            or metadata.get("command_hash") != row["command_hash"]
            or metadata.get("observational_only") is not True
            or metadata.get("automatic_retry_allowed") is not False
            or metadata.get("projection_applied") is not False
            or metadata.get("certification_applied") is not False
            or metadata.get("skill_authority") is not False
            or metadata.get("shadow_only") is not True
        ):
            errors.append(f"{label}: unsafe or mismatched event metadata")
        claim_event = (
            events_by_type["PerformanceScoringClaimed"].get(
                claim["event_id"]
            )
            or events_by_type["PerformanceScoringClaimMigrated"].get(
                claim["event_id"]
            )
        )
        if (
            row["attempt_id"] != claim["attempt_id"]
            or row["evaluation_id"] != claim["evaluation_id"]
            or row["scoring_request_digest"]
            != claim["scoring_request_digest"]
            or row["provider_binding_digest"]
            != claim["provider_binding_digest"]
            or row["provider_operation_digest"]
            != claim["provider_operation_digest"]
            or expected_command_hash is None
            or row["command_hash"] != expected_command_hash
            or row["id"] != "psr_" + str(expected_command_hash)
            or event["schema_version"] != PERFORMANCE_EVENT_SCHEMA_VERSION
            or event["stream_id"] != f"learner:{attempt['learner_id']}"
            or event["learner_id"] != attempt["learner_id"]
            or event["session_id"] is not None
            or event["idempotency_key"]
            != (
                performance_scoring_reconciliation_event_key(
                    expected_command_hash
                )
                if expected_command_hash is not None
                else None
            )
            or event["correlation_id"] != attempt["id"]
            or claim_event is None
            or event["causation_id"] != claim["event_id"]
            or event["occurred_at"] != row["reconciled_at"]
            or (
                claim_event is not None
                and event["stream_version"] <= claim_event["stream_version"]
            )
        ):
            errors.append(f"{label}: claim/event boundary mismatch")
        if receipt is not None and (
            receipt.digest != row["receipt_digest"]
            or receipt.claim_id != claim["id"]
            or receipt.attempt_id != claim["attempt_id"]
            or receipt.evaluation_id != claim["evaluation_id"]
            or receipt.through_sequence != claim["through_sequence"]
            or receipt.provider_id != claim["provider_id"]
            or receipt.provider_version != claim["provider_version"]
            or receipt.action_trace_digest
            != claim["action_trace_digest"]
            or receipt.command_hash != claim["command_hash"]
            or receipt.scoring_request_digest
            != claim["scoring_request_digest"]
            or receipt.provider_binding_digest
            != claim["provider_binding_digest"]
            or receipt.provider_operation_digest
            != claim["provider_operation_digest"]
            or receipt.reconciler_id != row["reconciler_id"]
            or receipt.reconciler_version != row["reconciler_version"]
            or outcome is None
            or receipt.outcome is not outcome
        ):
            errors.append(f"{label}: receipt does not match its claim")
        if reconciler is not None and (
            reconciler.provider_id != claim["provider_id"]
            or reconciler.provider_version != claim["provider_version"]
            or reconciler.reconciler_id != row["reconciler_id"]
            or reconciler.reconciler_version != row["reconciler_version"]
            or reconciler.binding_digest
            != row["reconciliation_binding_digest"]
            or (
                outcome is ReconciliationOutcome.DEFINITELY_ABSENT
                and not reconciler.can_prove_absence
            )
        ):
            errors.append(f"{label}: reconciler authority mismatch")
        evaluation_row = evaluation_rows_by_id.get(row["evaluation_id"])
        if outcome is ReconciliationOutcome.COMPLETED:
            if (
                row["normalized_result_digest"] is None
                or evaluation_row is None
            ):
                errors.append(
                    f"{label}: completed outcome lacks its recovered evaluation"
                )
            else:
                evaluation_event = events_by_type[
                    "TaskEvaluationRecorded"
                ].get(evaluation_row["event_id"])
                if (
                    receipt is None
                    or evaluation_event is None
                    or evaluation_event["occurred_at"]
                    != receipt.completed_at
                ):
                    errors.append(
                        f"{label}: recovered evaluation occurrence does not "
                        "match its receipt completion"
                    )
                if (
                    evaluation_event is None
                    or evaluation_event["stream_id"] != event["stream_id"]
                    or evaluation_event["stream_version"]
                    <= event["stream_version"]
                ):
                    errors.append(
                        f"{label}: recovered evaluation does not follow its "
                        "reconciliation event"
                    )
                evaluation = evaluations.get(row["evaluation_id"])
                provider = claim_provider_snapshots.get(row["claim_id"])
                scoring_request = claim_scoring_requests.get(row["claim_id"])
                authority = decode(
                    evaluation_row["authority_json"],
                    f"{label} recovered evaluation authority",
                )
                normalized_terms = (
                    authority.get("normalized_result")
                    if authority is not None
                    else None
                )
                if (
                    receipt is None
                    or evaluation is None
                    or provider is None
                    or scoring_request is None
                    or type(normalized_terms) is not dict
                ):
                    errors.append(
                        f"{label}: recovered result boundary cannot be "
                        "reconstructed"
                    )
                else:
                    try:
                        imported = _authority_free_imported_evaluation(
                            evaluation
                        )
                        expected_result = (
                            _normalize_recovered_imported_evaluation(
                                scoring_request,
                                imported,
                                provider,
                                row["provider_operation_digest"],
                            )
                        )
                        stored_result = NormalizedScoringResult.from_terms(
                            normalized_terms
                        )
                    except (TypeError, ValueError) as exc:
                        errors.append(
                            f"{label}: recovered result is invalid ({exc})"
                        )
                    else:
                        if receipt.result_digest != imported.digest:
                            errors.append(
                                f"{label}: recovered imported result digest "
                                "does not match its receipt"
                            )
                        if (
                            row["normalized_result_digest"]
                            != expected_result.digest
                            or authority.get("normalized_result_digest")
                            != expected_result.digest
                            or canonical_json(stored_result.terms())
                            != canonical_json(expected_result.terms())
                        ):
                            errors.append(
                                f"{label}: recovered normalized result does "
                                "not match its claim-bound shadow "
                                "normalization"
                            )
        elif (
            row["normalized_result_digest"] is not None
            or (
                outcome is ReconciliationOutcome.DEFINITELY_ABSENT
                and evaluation_row is not None
            )
        ):
            errors.append(f"{label}: non-completed outcome has a result")
        if receipt is not None:
            try:
                claim_time = _aware_timestamp(
                    claim["claimed_at"], f"{label} claim admission"
                )
                observed_time = _aware_timestamp(
                    receipt.observed_at, f"{label} observed_at"
                )
                reconciliation_time = _aware_timestamp(
                    row["reconciled_at"], f"{label} reconciled_at"
                )
                if observed_time < claim_time:
                    errors.append(
                        f"{label}: observation predates callback admission"
                    )
                if observed_time > reconciliation_time:
                    errors.append(
                        f"{label}: observation is later than its ledger record"
                    )
                if receipt.completed_at is not None and _aware_timestamp(
                    receipt.completed_at, f"{label} completed_at"
                ) < claim_time:
                    errors.append(
                        f"{label}: completion predates callback admission"
                    )
            except ValidationError as exc:
                errors.append(str(exc))

    for event_id in events_by_type["PerformanceScoringReconciled"]:
        if event_id not in projected_reconciliation_events:
            errors.append(
                f"event {event_id}: PerformanceScoringReconciled has no "
                "reconciliation projection"
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
        elif claims:
            recovered_claims = [
                claim
                for claim in claims
                if connection.execute(
                    """SELECT 1
                       FROM performance_scoring_reconciliations
                       WHERE claim_id=? AND outcome='completed'
                         AND evaluation_id=?""",
                    (claim["id"], evaluation_id),
                ).fetchone()
                is not None
            ]
            if (
                len(claims) != 1
                or len(recovered_claims) != 1
                or exemption is not None
            ):
                errors.append(
                    f"task evaluation {evaluation_id}: direct import has an "
                    "unreconciled provider-callback claim or legacy exemption"
                )
        elif exemption is not None:
            errors.append(
                f"task evaluation {evaluation_id}: direct import cannot have "
                "a legacy provider exemption"
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
        evaluation_event = events_by_type["TaskEvaluationRecorded"].get(
            evaluation_row["event_id"]
        )
        if (
            event["schema_version"] != PERFORMANCE_EVENT_SCHEMA_VERSION
            or event["stream_id"] != f"learner:{attempt['learner_id']}"
            or event["learner_id"] != attempt["learner_id"]
            or evaluation_event is None
            or event["session_id"] != evaluation_event["session_id"]
            or event["correlation_id"] != attempt["id"]
            or event["causation_id"] != row["evaluation_id"]
            or event["recorded_at"] != row["recorded_at"]
        ):
            errors.append(f"{label}: event envelope mismatch")
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
    try:
        derived_projection, _ = derive_performance_projections(connection)
        stored_projection = performance_projection_snapshot(connection)
    except (TypeError, ValueError, ValidationError) as exc:
        errors.append(
            f"performance projection cannot be derived from events ({exc})"
        )
    else:
        for name in _PERFORMANCE_PROJECTION_TABLES:
            projection_name = {
                "performance_attempts": "attempts",
                "performance_actions": "actions",
                "performance_artifact_run_claims": (
                    "artifact_run_claims"
                ),
                "performance_artifact_run_receipts": (
                    "artifact_run_receipts"
                ),
                "performance_scoring_claims": "scoring_claims",
                "performance_scoring_reconciliations": (
                    "scoring_reconciliations"
                ),
                "task_evaluations": "evaluations",
                "shadow_evidence_bundles": "bundles",
            }[name]
            integrity_label = {
                "performance_attempts": "performance attempt projection",
                "performance_actions": "performance action projection",
                "performance_artifact_run_claims": (
                    "performance artifact-run claim projection"
                ),
                "performance_artifact_run_receipts": (
                    "performance artifact-run receipt projection"
                ),
                "performance_scoring_claims": (
                    "performance scoring claim projection"
                ),
                "performance_scoring_reconciliations": (
                    "performance scoring reconciliation projection"
                ),
                "task_evaluations": "task evaluation projection",
                "shadow_evidence_bundles": (
                    "shadow evidence bundle projection"
                ),
            }[name]
            if (
                canonical_json(stored_projection[projection_name])
                != canonical_json(derived_projection[projection_name])
            ):
                errors.append(
                    f"{integrity_label}: stored projection differs from "
                    "immutable events"
                )
    return errors


_PERFORMANCE_PROJECTION_TABLES = (
    "performance_attempts",
    "performance_actions",
    "performance_artifact_run_claims",
    "performance_artifact_run_receipts",
    "performance_scoring_claims",
    "performance_scoring_reconciliations",
    "task_evaluations",
    "shadow_evidence_bundles",
)


def performance_projection_snapshot(
    connection: sqlite3.Connection,
    *,
    learner_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return the exact mutable shadow projections in stable row order."""

    if learner_id is not None:
        _require_id(learner_id, "learner_id")
    if session_id is not None:
        _require_id(session_id, "session_id")
    if learner_id is not None and session_id is not None:
        raise ValidationError(
            "Performance projection snapshot accepts only one scope."
        )
    scope_column = (
        "learner_id"
        if learner_id is not None
        else ("session_id" if session_id is not None else None)
    )
    scope_value = learner_id if learner_id is not None else session_id
    attempt_scope = (
        "" if scope_column is None else f" WHERE {scope_column}=?"
    )
    dependent_scope = (
        ""
        if scope_column is None
        else (
            " WHERE attempt_id IN ("
            "SELECT id FROM performance_attempts "
            f"WHERE {scope_column}=?"
            ")"
        )
    )
    artifact_receipt_scope = (
        ""
        if scope_column is None
        else (
            " WHERE claim_id IN ("
            "SELECT claim.id FROM performance_artifact_run_claims claim "
            "JOIN performance_attempts attempt "
            "ON attempt.id=claim.attempt_id "
            f"WHERE attempt.{scope_column}=?"
            ")"
        )
    )
    scoring_reconciliation_scope = (
        ""
        if scope_column is None
        else (
            " WHERE observation.claim_id IN ("
            "SELECT claim.id FROM performance_scoring_claims claim "
            "JOIN performance_attempts attempt "
            "ON attempt.id=claim.attempt_id "
            f"WHERE attempt.{scope_column}=?"
            ")"
        )
    )
    queries = {
        "attempts": (
            "SELECT * FROM performance_attempts"
            + attempt_scope
            + " ORDER BY id"
        ),
        "actions": (
            "SELECT * FROM performance_actions"
            + dependent_scope
            + " "
            "ORDER BY attempt_id, sequence, id"
        ),
        "artifact_run_claims": (
            "SELECT * FROM performance_artifact_run_claims"
            + dependent_scope
            + " ORDER BY id"
        ),
        "artifact_run_receipts": (
            "SELECT * FROM performance_artifact_run_receipts"
            + artifact_receipt_scope
            + " ORDER BY id"
        ),
        "scoring_claims": (
            "SELECT * FROM performance_scoring_claims"
            + dependent_scope
            + " ORDER BY id"
        ),
        "scoring_reconciliations": (
            "SELECT observation.* "
            "FROM performance_scoring_reconciliations observation"
            + scoring_reconciliation_scope
            + " "
            "ORDER BY ("
            "SELECT event.stream_id FROM events event "
            "WHERE event.event_id=observation.event_id"
            "), ("
            "SELECT event.stream_version FROM events event "
            "WHERE event.event_id=observation.event_id"
            "), observation.id"
        ),
        "evaluations": (
            "SELECT * FROM task_evaluations"
            + dependent_scope
            + " "
            "ORDER BY attempt_id, recorded_at, id"
        ),
        "bundles": (
            "SELECT * FROM shadow_evidence_bundles"
            + dependent_scope
            + " "
            "ORDER BY attempt_id, evaluation_id, id"
        ),
    }
    snapshot: dict[str, list[dict[str, Any]]] = {}
    for name, query in queries.items():
        if name in {
            "artifact_run_claims",
            "artifact_run_receipts",
            "scoring_claims",
            "scoring_reconciliations",
        } and connection.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type='table' AND name=?""",
            (
                {
                    "artifact_run_claims": (
                        "performance_artifact_run_claims"
                    ),
                    "artifact_run_receipts": (
                        "performance_artifact_run_receipts"
                    ),
                    "scoring_claims": "performance_scoring_claims",
                    "scoring_reconciliations": (
                        "performance_scoring_reconciliations"
                    ),
                }[name],
            ),
        ).fetchone() is None:
            # Exact historical migration fixtures predate callback admission
            # and artifact-run receipts. An absent table and the newly
            # installed empty projection represent the same historical state;
            # no row is synthesized.
            snapshot[name] = []
            continue
        snapshot[name] = [
            _row_dict(row)
            for row in connection.execute(
                query,
                (() if scope_column is None else (scope_value,)),
            ).fetchall()
        ]
    return snapshot


def _attempt_scoped_performance_events(
    connection: sqlite3.Connection,
    *,
    attempt_id: str,
    session_id: str,
    start_event_id: str,
) -> list[sqlite3.Row]:
    """Return the fixed-point event graph for one attempt session.

    Membership follows only lifecycle identities that are unique to an
    attempt, action, artifact-run request, claim, process result, or receipt.
    Shared task, checker, artifact, check-set, runner, and binding identities
    deliberately do not join otherwise independent learner history.
    """

    events_by_id: dict[str, sqlite3.Row] = {}
    action_rows: dict[str, sqlite3.Row] = {}
    claim_rows: dict[str, sqlite3.Row] = {}
    receipt_rows: dict[str, sqlite3.Row] = {}
    attempt_ids = {attempt_id}
    loaded_projection_attempt_ids: set[str] = set()
    loaded_projection_event_ids: set[str] = set()
    queried_values: dict[str, set[str]] = {}

    def add_events(rows: Iterable[sqlite3.Row]) -> bool:
        changed = False
        for row in rows:
            event_id = row["event_id"]
            if event_id in events_by_id:
                continue
            events_by_id[event_id] = row
            changed = True
        return changed

    def json_object(raw: object) -> dict[str, Any]:
        if type(raw) is not str:
            return {}
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return {}
        return value if type(value) is dict else {}

    def nested_text(value: object, *path: str) -> str | None:
        current = value
        for field_name in path:
            if type(current) is not dict:
                return None
            current = current.get(field_name)
        return current if type(current) is str else None

    def scoped_values(values: set[str]) -> str:
        return canonical_json(sorted(values))

    def lookup_event_ids(event_ids: set[str]) -> list[sqlite3.Row]:
        pending = event_ids - loaded_projection_event_ids
        if not pending:
            return []
        loaded_projection_event_ids.update(pending)
        return connection.execute(
            "WITH scoped_values(value) AS MATERIALIZED ("
            "SELECT value FROM json_each(?) WHERE type='text'"
            ") SELECT event.* FROM scoped_values AS scope "
            "JOIN events AS event ON event.event_id=scope.value "
            "WHERE event.event_type IN ("
            + _PERFORMANCE_TRACE_EVENT_TYPES_SQL
            + ") ORDER BY event.stream_id, event.stream_version",
            (scoped_values(pending),),
        ).fetchall()

    def lookup_payload(
        *,
        key: str,
        index_name: str,
        event_predicate: str,
        path: str,
        values: set[str],
    ) -> list[sqlite3.Row]:
        prior = queried_values.setdefault(key, set())
        pending = values - prior
        if not pending:
            return []
        prior.update(pending)
        return connection.execute(
            "WITH scoped_values(value) AS MATERIALIZED ("
            "SELECT value FROM json_each(?) WHERE type='text'"
            ") SELECT event.* FROM scoped_values AS scope "
            "CROSS JOIN events AS event INDEXED BY "
            + index_name
            + " WHERE "
            + event_predicate
            + " AND json_extract(event.payload_json, '"
            + path
            + "')=scope.value "
            "ORDER BY event.stream_id, event.stream_version",
            (scoped_values(pending),),
        ).fetchall()

    def lookup_envelope(
        *,
        key: str,
        index_name: str,
        column: str,
        values: set[str],
    ) -> list[sqlite3.Row]:
        prior = queried_values.setdefault(key, set())
        pending = values - prior
        if not pending:
            return []
        prior.update(pending)
        return connection.execute(
            "WITH scoped_values(value) AS MATERIALIZED ("
            "SELECT value FROM json_each(?) WHERE type='text'"
            ") SELECT event.* FROM scoped_values AS scope "
            "CROSS JOIN events AS event INDEXED BY "
            + index_name
            + " WHERE event.event_type IN ("
            + _PERFORMANCE_TRACE_EVENT_TYPES_SQL
            + ") AND event."
            + column
            + "=scope.value "
            "ORDER BY event.stream_id, event.stream_version",
            (scoped_values(pending),),
        ).fetchall()

    add_events(
        connection.execute(
            "WITH scoped_session_ids(session_id) AS MATERIALIZED (SELECT ?) "
            "SELECT event.* FROM scoped_session_ids AS scope "
            "CROSS JOIN events AS event "
            "INDEXED BY idx_events_session_stream "
            "WHERE event.event_type IN ("
            + _PERFORMANCE_TRACE_EVENT_TYPES_SQL
            + ") AND event.session_id=scope.session_id "
            "UNION "
            "SELECT event.* FROM scoped_session_ids AS scope "
            "CROSS JOIN events AS event "
            "INDEXED BY idx_events_payload_session_stream "
            "WHERE event.event_type IN ("
            + _PERFORMANCE_TRACE_EVENT_TYPES_SQL
            + ") AND json_extract("
            "event.payload_json, '$.session_id')=scope.session_id "
            "UNION "
            "SELECT event.* FROM events AS event "
            "WHERE event.event_id=? "
            "AND event.event_type IN ("
            + _PERFORMANCE_TRACE_EVENT_TYPES_SQL
            + ") ORDER BY stream_id, stream_version",
            (session_id, start_event_id),
        ).fetchall()
    )

    while True:
        before = (
            len(events_by_id),
            len(attempt_ids),
            len(action_rows),
            len(claim_rows),
            len(receipt_rows),
        )

        for event in events_by_id.values():
            if event["event_type"] != "PerformanceTaskStarted":
                continue
            defined_attempt_id = nested_text(
                json_object(event["payload_json"]),
                "attempt_id",
            )
            if defined_attempt_id is not None:
                attempt_ids.add(defined_attempt_id)

        new_projection_attempt_ids = (
            attempt_ids - loaded_projection_attempt_ids
        )
        if new_projection_attempt_ids:
            loaded_projection_attempt_ids.update(new_projection_attempt_ids)
            values_json = scoped_values(new_projection_attempt_ids)
            for row in connection.execute(
                "WITH scoped_values(value) AS MATERIALIZED ("
                "SELECT value FROM json_each(?) WHERE type='text'"
                ") SELECT action.* FROM scoped_values AS scope "
                "CROSS JOIN performance_actions AS action "
                "INDEXED BY idx_performance_actions_attempt "
                "WHERE action.attempt_id=scope.value",
                (values_json,),
            ).fetchall():
                action_rows[row["id"]] = row
            for row in connection.execute(
                "WITH scoped_values(value) AS MATERIALIZED ("
                "SELECT value FROM json_each(?) WHERE type='text'"
                ") SELECT claim.* FROM scoped_values AS scope "
                "CROSS JOIN performance_artifact_run_claims AS claim "
                "INDEXED BY idx_performance_artifact_run_claims_attempt "
                "WHERE claim.attempt_id=scope.value",
                (values_json,),
            ).fetchall():
                claim_rows[row["id"]] = row
            for row in connection.execute(
                "WITH scoped_values(value) AS MATERIALIZED ("
                "SELECT value FROM json_each(?) WHERE type='text'"
                ") SELECT receipt.* FROM scoped_values AS scope "
                "CROSS JOIN performance_artifact_run_claims AS claim "
                "INDEXED BY idx_performance_artifact_run_claims_attempt "
                "JOIN performance_artifact_run_receipts AS receipt "
                "ON receipt.claim_id=claim.id "
                "WHERE claim.attempt_id=scope.value",
                (values_json,),
            ).fetchall():
                receipt_rows[row["id"]] = row

        projection_event_ids = {
            row["event_id"]
            for rows in (action_rows, claim_rows, receipt_rows)
            for row in rows.values()
        }
        add_events(lookup_event_ids(projection_event_ids))

        action_ids = set(action_rows)
        claim_ids = set(claim_rows)
        claim_caller_keys = {
            row["idempotency_key"]
            for row in claim_rows.values()
            if row["idempotency_key"] is not None
        }
        claim_command_hashes = {
            row["command_hash"] for row in claim_rows.values()
        }
        request_run_ids = {
            run_id
            for row in claim_rows.values()
            if (
                run_id := nested_text(
                    json_object(row["request_json"]),
                    "run_id",
                )
            )
            is not None
        }
        request_digests = {
            row["request_digest"] for row in claim_rows.values()
        }
        # A generic learner-authored check_run may legitimately repeat a
        # content result digest in another attempt. Only artifact-run process
        # receipts bind this digest to one exact request/observation identity.
        result_digests = {
            row["result_digest"]
            for row in receipt_rows.values()
            if row["result_digest"] is not None
        }
        receipt_ids = set(receipt_rows)
        receipt_digests = {
            row["receipt_digest"] for row in receipt_rows.values()
        }

        for event in events_by_id.values():
            payload = json_object(event["payload_json"])
            if event["event_type"] == "PerformanceActionRecorded":
                action_id_value = nested_text(payload, "action", "id")
                if action_id_value is not None:
                    action_ids.add(action_id_value)
            elif event["event_type"] == "PerformanceArtifactRunClaimed":
                claim_id_value = nested_text(payload, "claim_id")
                if claim_id_value is not None:
                    claim_ids.add(claim_id_value)
                caller_key = nested_text(
                    payload,
                    "caller_idempotency_key",
                )
                if caller_key is not None:
                    claim_caller_keys.add(caller_key)
                command_hash = nested_text(payload, "command_hash")
                if command_hash is not None:
                    claim_command_hashes.add(command_hash)
                run_id = nested_text(payload, "request", "run_id")
                if run_id is not None:
                    request_run_ids.add(run_id)
                request_digest = nested_text(payload, "request_digest")
                if request_digest is not None:
                    request_digests.add(request_digest)
            elif event["event_type"] == "PerformanceArtifactRunObserved":
                receipt_id_value = nested_text(payload, "receipt_id")
                if receipt_id_value is not None:
                    receipt_ids.add(receipt_id_value)
                receipt_digest = nested_text(payload, "receipt_digest")
                if receipt_digest is not None:
                    receipt_digests.add(receipt_digest)
                for digest_path in (
                    ("result_digest",),
                    ("receipt", "result_digest"),
                ):
                    result_digest = nested_text(payload, *digest_path)
                    if result_digest is not None:
                        result_digests.add(result_digest)

        event_ids = set(events_by_id)
        lookup_specs = (
            (
                "attempt_payload",
                "idx_events_performance_attempt_payload",
                "event.event_type IN ("
                + _PERFORMANCE_TRACE_EVENT_TYPES_SQL
                + ")",
                "$.attempt_id",
                attempt_ids,
            ),
            (
                "action_trace",
                "idx_events_action_trace_stream",
                "event.event_type='PerformanceActionRecorded'",
                "$.action.trace_id",
                attempt_ids,
            ),
            (
                "receipt_attempt",
                "idx_events_receipt_attempt_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.receipt.attempt_id",
                attempt_ids,
            ),
            (
                "action_id",
                "idx_events_action_id_stream",
                "event.event_type='PerformanceActionRecorded'",
                "$.action.id",
                action_ids,
            ),
            (
                "artifact_action",
                "idx_events_artifact_action_stream",
                "event.event_type='PerformanceArtifactRunClaimed'",
                "$.artifact_action_id",
                action_ids,
            ),
            (
                "check_action",
                "idx_events_check_action_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.check_action_id",
                action_ids,
            ),
            (
                "receipt_artifact_action",
                "idx_events_receipt_artifact_action_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.receipt.artifact_action_id",
                action_ids,
            ),
            (
                "claim_id",
                "idx_events_claim_payload_stream",
                "event.event_type IN ("
                "'PerformanceArtifactRunClaimed',"
                "'PerformanceArtifactRunObserved')",
                "$.claim_id",
                claim_ids,
            ),
            (
                "receipt_claim",
                "idx_events_receipt_claim_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.receipt.claim_id",
                claim_ids,
            ),
            (
                "claim_caller_key",
                "idx_events_claim_caller_key_stream",
                "event.event_type='PerformanceArtifactRunClaimed'",
                "$.caller_idempotency_key",
                claim_caller_keys,
            ),
            (
                "claim_command_hash",
                "idx_events_claim_command_hash_stream",
                "event.event_type='PerformanceArtifactRunClaimed'",
                "$.command_hash",
                claim_command_hashes,
            ),
            (
                "claim_request_run",
                "idx_events_claim_request_run_stream",
                "event.event_type='PerformanceArtifactRunClaimed'",
                "$.request.run_id",
                request_run_ids,
            ),
            (
                "observed_request_run",
                "idx_events_observed_request_run_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.result.request.run_id",
                request_run_ids,
            ),
            (
                "claim_request_digest",
                "idx_events_claim_request_digest_stream",
                "event.event_type='PerformanceArtifactRunClaimed'",
                "$.request_digest",
                request_digests,
            ),
            (
                "observed_request_digest",
                "idx_events_observed_request_digest_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.result.request_digest",
                request_digests,
            ),
            (
                "receipt_request_digest",
                "idx_events_receipt_request_digest_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.receipt.request_digest",
                request_digests,
            ),
            (
                "observed_result_digest",
                "idx_events_observed_result_digest_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.result_digest",
                result_digests,
            ),
            (
                "receipt_result_digest",
                "idx_events_receipt_result_digest_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.receipt.result_digest",
                result_digests,
            ),
            (
                "receipt_id",
                "idx_events_receipt_id_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.receipt_id",
                receipt_ids,
            ),
            (
                "receipt_digest",
                "idx_events_receipt_digest_stream",
                "event.event_type='PerformanceArtifactRunObserved'",
                "$.receipt_digest",
                receipt_digests,
            ),
        )
        for (
            key,
            index_name,
            event_predicate,
            path,
            values,
        ) in lookup_specs:
            add_events(
                lookup_payload(
                    key=key,
                    index_name=index_name,
                    event_predicate=event_predicate,
                    path=path,
                    values=values,
                )
            )
        add_events(
            lookup_envelope(
                key="attempt_correlation",
                index_name="idx_events_correlation_stream",
                column="correlation_id",
                values=attempt_ids,
            )
        )
        add_events(
            lookup_envelope(
                key="attempt_causation",
                index_name="idx_events_causation_stream",
                column="causation_id",
                values=attempt_ids,
            )
        )
        add_events(
            lookup_envelope(
                key="event_causation",
                index_name="idx_events_causation_stream",
                column="causation_id",
                values=event_ids,
            )
        )

        after = (
            len(events_by_id),
            len(attempt_ids),
            len(action_rows),
            len(claim_rows),
            len(receipt_rows),
        )
        if after == before:
            break

    return sorted(
        events_by_id.values(),
        key=lambda event: (event["stream_id"], event["stream_version"]),
    )


def derive_performance_projections(
    connection: sqlite3.Connection,
    *,
    learner_id: str | None = None,
    attempt_id: str | None = None,
    trace_only: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Derive shadow projection rows exclusively from immutable events."""

    if learner_id is not None:
        _require_id(learner_id, "learner_id")
    if attempt_id is not None:
        _require_id(attempt_id, "attempt_id")
    if learner_id is not None and attempt_id is not None:
        raise ValidationError(
            "Performance projection derivation accepts only one scope."
        )
    if type(trace_only) is not bool:
        raise ValidationError("trace_only must be a boolean.")
    if attempt_id is not None and not trace_only:
        raise ValidationError(
            "Attempt-scoped performance derivation is trace-only."
        )

    def reject_artifact_technical_key_collision(
        event_key: object,
        label: str,
    ) -> None:
        if type(event_key) is not str:
            raise ValidationError(
                f"{label} has no technical event idempotency key."
            )
        collision = connection.execute(
            """SELECT event_id FROM events
               WHERE event_type IN (
                   'PerformanceScoringClaimed',
                   'PerformanceScoringClaimMigrated',
                   'PerformanceScoringReconciled'
               )
                 AND json_extract(
                     payload_json, '$.caller_idempotency_key'
                 )=?
               ORDER BY stream_id, stream_version
               LIMIT 1""",
            (event_key,),
        ).fetchone()
        if collision is not None:
            raise ValidationError(
                f"{label} technical event key collides with scoring caller "
                f"event {collision['event_id']}."
            )

    attempts: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    artifact_run_claims: list[dict[str, Any]] = []
    artifact_run_receipts: list[dict[str, Any]] = []
    scoring_claims: list[dict[str, Any]] = []
    scoring_reconciliations: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    claim_provider_snapshots: dict[str, RegisteredProvider] = {}
    reconciliation_receipts: dict[
        str, ScoringReconciliationReceipt
    ] = {}
    reconciliation_events: dict[str, sqlite3.Row] = {}
    recovered_evaluations: dict[
        str, tuple[TaskEvaluation, Mapping[str, Any], sqlite3.Row]
    ] = {}
    evaluation_modes: dict[str, NormalizationMode] = {}
    legacy_scoring_exemptions: dict[
        str, tuple[dict[str, Any], sqlite3.Row]
    ] = {}
    attempts_by_id: dict[str, dict[str, Any]] = {}
    attempt_events_by_id: dict[str, sqlite3.Row] = {}
    attempt_tasks_by_id: dict[str, LearningTask] = {}
    action_contexts_by_id: dict[
        str, tuple[dict[str, Any], LearningAction, sqlite3.Row]
    ] = {}
    artifact_claim_contexts: dict[
        str,
        tuple[
            dict[str, Any],
            ArtifactRunRequest,
            ArtifactRunnerBinding,
            sqlite3.Row,
            LearningAction,
            sqlite3.Row,
        ],
    ] = {}
    artifact_receipt_claim_ids: set[str] = set()
    system_artifact_check_action_ids: set[str] = set()
    artifact_caller_keys: set[str] = set()
    artifact_command_hashes: set[str] = set()
    artifact_receipt_digests: set[str] = set()
    release_cache: dict[str, tuple[PerformanceTaskRelease, datetime]] = {}
    active_attempt_by_session: dict[str, str] = {}
    terminal_attempts: set[str] = set()
    next_action_sequence: dict[str, int] = {}
    prior_action_occurrence: dict[str, datetime] = {}

    def strict_release(
        release_id: str,
    ) -> tuple[PerformanceTaskRelease, datetime]:
        cached = release_cache.get(release_id)
        if cached is None:
            cached = load_stored_task_release(connection, release_id)
            release_cache[release_id] = cached
        return cached

    event_types = (
        _PERFORMANCE_TRACE_EVENT_TYPES
        if trace_only
        else (
            *_PERFORMANCE_TRACE_EVENT_TYPES,
            "PerformanceScoringClaimed",
            "PerformanceScoringClaimMigrated",
            "PerformanceScoringReconciled",
            "PerformanceScoringLegacyExempted",
            "TaskEvaluationRecorded",
            "ShadowEvidenceReduced",
        )
    )
    placeholders = ", ".join("?" for _item in event_types)
    if attempt_id is None:
        event_scope = (
            " AND stream_id=?"
            if learner_id is not None
            else ""
        )
        event_parameters: tuple[object, ...] = event_types + (
            (f"learner:{learner_id}",)
            if learner_id is not None
            else ()
        )
        events = connection.execute(
            "SELECT * FROM events WHERE event_type IN ("
            + placeholders
            + ")"
            + event_scope
            + " ORDER BY stream_id, stream_version",
            event_parameters,
        ).fetchall()
    else:
        stored_attempt = connection.execute(
            """SELECT event_id, session_id
               FROM performance_attempts WHERE id=?""",
            (attempt_id,),
        ).fetchone()
        if stored_attempt is None:
            raise ValidationError(
                f"Performance attempt {attempt_id} has no projection."
            )
        events = _attempt_scoped_performance_events(
            connection,
            attempt_id=attempt_id,
            session_id=stored_attempt["session_id"],
            start_event_id=stored_attempt["event_id"],
        )
    for event in events:
        event_type = event["event_type"]
        supported_schema = (
            event["schema_version"] in {1, 2}
            if event_type == "PerformanceScoringClaimed"
            else event["schema_version"] == PERFORMANCE_EVENT_SCHEMA_VERSION
        )
        if not supported_schema:
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
            _require_id(
                payload["attempt_id"],
                f"Performance event {event['event_id']} attempt_id",
            )
            _require_id(
                payload["session_id"],
                f"Performance event {event['event_id']} session_id",
            )
            _require_id(
                payload["learner_id"],
                f"Performance event {event['event_id']} learner_id",
            )
            _require_id(
                payload["task_release_id"],
                f"Performance event {event['event_id']} task_release_id",
            )
            _require_id(
                payload["corpus_release_id"],
                f"Performance event {event['event_id']} corpus_release_id",
            )
            _require_id(
                payload["task_id"],
                f"Performance event {event['event_id']} task_id",
            )
            _require_digest(
                payload["task_digest"],
                f"Performance event {event['event_id']} task_digest",
            )
            if (
                type(payload["task_version"]) is not int
                or payload["task_version"] < 1
                or type(payload["session_revision"]) is not int
                or payload["session_revision"] < 0
                or type(payload["learner_revision"]) is not int
                or payload["learner_revision"] < 0
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} has an invalid "
                    "task version or revision."
                )
            release, release_created_at = strict_release(
                payload["task_release_id"]
            )
            if (
                type(release.review) is not TaskReleaseReview
                or release.corpus_release_id != payload["corpus_release_id"]
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} is not pinned to "
                    "an exact human-reviewed task release."
                )
            release_member = next(
                (
                    (status, task)
                    for status, task in release.tasks
                    if task.id == payload["task_id"]
                    and task.version == payload["task_version"]
                ),
                None,
            )
            if (
                release_member is None
                or release_member[0] not in SERVICEABLE_TASK_STATUSES
                or release_member[1].digest != payload["task_digest"]
                or metadata["task_schema_version"]
                != release_member[1].schema_version
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} does not reference "
                    "an exact serviceable task-release member."
                )
            if (
                _aware_timestamp(
                    event["occurred_at"],
                    f"Performance event {event['event_id']} occurred_at",
                )
                < release_created_at
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} precedes its task "
                    "release."
                )
            session_row = connection.execute(
                """SELECT learner_id, corpus_release_id
                   FROM sessions WHERE id=?""",
                (payload["session_id"],),
            ).fetchone()
            if (
                session_row is None
                or session_row["learner_id"] != payload["learner_id"]
                or session_row["corpus_release_id"]
                != payload["corpus_release_id"]
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} crosses its "
                    "session ownership or corpus boundary."
                )
            if connection.execute(
                "SELECT 1 FROM learners WHERE id=?",
                (payload["learner_id"],),
            ).fetchone() is None:
                raise ValidationError(
                    f"Performance event {event['event_id']} references an "
                    "unknown learner."
                )
            expected_stream_id = f"learner:{payload['learner_id']}"
            if (
                event["stream_id"] != expected_stream_id
                or event["learner_id"] != payload["learner_id"]
                or event["session_id"] != payload["session_id"]
                or event["correlation_id"] != payload["attempt_id"]
                or event["causation_id"] != payload["session_id"]
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} has an invalid "
                    "stream or causal envelope."
                )
            session_boundaries = connection.execute(
                """SELECT event_type, stream_id, stream_version, learner_id,
                          session_id, occurred_at
                   FROM events
                   WHERE session_id=?
                     AND event_type IN ('SessionStarted', 'SessionEnded')
                   ORDER BY stream_version""",
                (payload["session_id"],),
            ).fetchall()
            session_starts = [
                boundary
                for boundary in session_boundaries
                if boundary["event_type"] == "SessionStarted"
            ]
            session_ends = [
                boundary
                for boundary in session_boundaries
                if boundary["event_type"] == "SessionEnded"
            ]
            occurred_at = _aware_timestamp(
                event["occurred_at"],
                f"Performance event {event['event_id']} occurred_at",
            )
            if (
                len(session_starts) != 1
                or session_starts[0]["stream_id"] != expected_stream_id
                or session_starts[0]["learner_id"] != payload["learner_id"]
                or session_starts[0]["session_id"] != payload["session_id"]
                or session_starts[0]["stream_version"]
                >= event["stream_version"]
                or _aware_timestamp(
                    session_starts[0]["occurred_at"],
                    f"Session {payload['session_id']} start occurrence",
                )
                > occurred_at
                or any(
                    boundary["stream_version"] < event["stream_version"]
                    or _aware_timestamp(
                        boundary["occurred_at"],
                        f"Session {payload['session_id']} end occurrence",
                    )
                    < occurred_at
                    for boundary in session_ends
                )
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} falls outside its "
                    "session-active interval."
                )
            expected_session_revision = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE stream_id=? AND session_id=?
                     AND stream_version < ?
                     AND event_type IN (
                         'QuestionSelected', 'ResponseSubmitted'
                     )""",
                (
                    expected_stream_id,
                    payload["session_id"],
                    event["stream_version"],
                ),
            ).fetchone()["n"]
            expected_learner_revision = connection.execute(
                """SELECT COUNT(*) AS n FROM events
                   WHERE stream_id=? AND learner_id=?
                     AND stream_version < ?
                     AND event_type='ResponseSubmitted'""",
                (
                    expected_stream_id,
                    payload["learner_id"],
                    event["stream_version"],
                ),
            ).fetchone()["n"]
            if (
                payload["session_revision"] != expected_session_revision
                or payload["learner_revision"] != expected_learner_revision
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} has a stale or "
                    "future revision snapshot."
                )
            open_decision_ids: set[str] = set()
            decision_events = connection.execute(
                """SELECT event_type, event_id, payload_json, causation_id
                   FROM events
                   WHERE stream_id=? AND session_id=?
                     AND stream_version < ?
                     AND event_type IN (
                         'QuestionSelected', 'ResponseSubmitted',
                         'DecisionInvalidated'
                     )
                   ORDER BY stream_version""",
                (
                    expected_stream_id,
                    payload["session_id"],
                    event["stream_version"],
                ),
            ).fetchall()
            for decision_event in decision_events:
                if decision_event["event_type"] == "QuestionSelected":
                    decision_payload = _json_object(
                        decision_event["payload_json"],
                        (
                            f"QuestionSelected event "
                            f"{decision_event['event_id']} payload"
                        ),
                    )
                    decision_id = decision_payload.get("decision_id")
                    _require_id(
                        decision_id,
                        (
                            f"QuestionSelected event "
                            f"{decision_event['event_id']} decision_id"
                        ),
                    )
                    if decision_id in open_decision_ids:
                        raise ValidationError(
                            f"QuestionSelected event "
                            f"{decision_event['event_id']} repeats decision "
                            f"{decision_id}."
                        )
                    open_decision_ids.add(decision_id)
                elif decision_event["event_type"] == "ResponseSubmitted":
                    open_decision_ids.discard(
                        decision_event["causation_id"]
                    )
                else:
                    invalidation_payload = _json_object(
                        decision_event["payload_json"],
                        (
                            f"DecisionInvalidated event "
                            f"{decision_event['event_id']} payload"
                        ),
                    )
                    open_decision_ids.discard(
                        invalidation_payload.get("decision_id")
                    )
            if open_decision_ids:
                raise ValidationError(
                    f"Performance event {event['event_id']} starts while "
                    "selected-response decision(s) remain pending."
                )
            _require_digest(
                metadata["command_hash"],
                f"Performance event {event['event_id']} command_hash",
            )
            possible_command_hashes = {
                _command_hash(
                    {
                        "operation": "start_attempt",
                        "session_id": payload["session_id"],
                        "task_id": payload["task_id"],
                        "task_version": version,
                        "task_release_id": release_id,
                    }
                )
                for version in (None, payload["task_version"])
                for release_id in (None, payload["task_release_id"])
            }
            if metadata["command_hash"] not in possible_command_hashes:
                raise ValidationError(
                    f"Performance event {event['event_id']} command commitment "
                    "does not match any valid start request."
                )
            attempt_row = {
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
            if attempt_row["id"] in attempts_by_id:
                raise ValidationError(
                    f"Performance event {event['event_id']} repeats attempt "
                    f"{attempt_row['id']}."
                )
            prior_active_attempt = active_attempt_by_session.get(
                payload["session_id"]
            )
            if prior_active_attempt is not None:
                raise ValidationError(
                    f"Performance event {event['event_id']} overlaps active "
                    f"attempt {prior_active_attempt} in session "
                    f"{payload['session_id']}."
                )
            attempts.append(attempt_row)
            attempts_by_id[attempt_row["id"]] = attempt_row
            attempt_events_by_id[attempt_row["id"]] = event
            attempt_tasks_by_id[attempt_row["id"]] = release_member[1]
            active_attempt_by_session[payload["session_id"]] = (
                attempt_row["id"]
            )
            next_action_sequence[attempt_row["id"]] = 0
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
            attempt = attempts_by_id.get(action.trace_id)
            if attempt is None:
                raise ValidationError(
                    f"Performance event {event['event_id']} action precedes "
                    f"attempt {action.trace_id}."
                )
            task = attempt_tasks_by_id[action.trace_id]
            if action.kind not in task.allowed_action_kinds:
                raise ValidationError(
                    f"Performance event {event['event_id']} action "
                    f"{action.kind.value} is not allowed by the exact task "
                    "release."
                )
            attempt_event = attempt_events_by_id[action.trace_id]
            if (
                event["stream_id"] != attempt_event["stream_id"]
                or event["stream_version"] <= attempt_event["stream_version"]
                or event["learner_id"] != attempt["learner_id"]
                or event["session_id"] != attempt["session_id"]
                or event["correlation_id"] != action.trace_id
                or event["causation_id"] != action.trace_id
                or metadata["task_digest"] != attempt["task_digest"]
                or metadata["task_release_id"]
                != attempt["task_release_id"]
                or metadata["corpus_release_id"]
                != attempt["corpus_release_id"]
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} action crosses "
                    "its attempt envelope."
                )
            expected_sequence = next_action_sequence.get(action.trace_id)
            if expected_sequence is None or action.sequence != expected_sequence:
                raise ValidationError(
                    f"Performance event {event['event_id']} action sequence is "
                    "not contiguous."
                )
            if (
                action.sequence == 0
                and action.kind is not ActionKind.STARTED
            ) or (
                action.sequence > 0
                and action.kind is ActionKind.STARTED
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} has an invalid "
                    "started-action boundary."
                )
            action_occurred_at = _aware_timestamp(
                event["occurred_at"],
                f"Performance event {event['event_id']} occurred_at",
            )
            prior_occurrence = prior_action_occurrence.get(action.trace_id)
            if (
                action_occurred_at
                < _aware_timestamp(
                    attempt["started_at"],
                    f"Attempt {action.trace_id} started_at",
                )
                or (
                    prior_occurrence is not None
                    and action_occurred_at < prior_occurrence
                )
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} action occurrence "
                    "is outside its monotonic attempt timeline."
                )
            expected_elapsed_ms = int(
                (
                    action_occurred_at
                    - _aware_timestamp(
                        attempt["started_at"],
                        f"Attempt {action.trace_id} started_at",
                    )
                ).total_seconds()
                * 1000
            )
            if action.elapsed_ms != expected_elapsed_ms:
                raise ValidationError(
                    f"Performance event {event['event_id']} action elapsed_ms "
                    "does not match its occurrence."
                )
            _require_digest(
                metadata["command_hash"],
                f"Performance event {event['event_id']} command_hash",
            )
            if action.kind is ActionKind.STARTED:
                allowed_action_command_hashes = {
                    _command_hash(
                        {
                            "operation": "automatic_start",
                            "attempt_id": action.trace_id,
                        }
                    )
                }
            elif action.kind is ActionKind.CHECK_RUN:
                system_action_command_hashes = {
                    _command_hash(
                        {
                            "operation": "record_artifact_run_check",
                            "claim_id": claim_id,
                            "result_digest": action.payload[
                                "result_digest"
                            ],
                        }
                    )
                    for claim_id, context in artifact_claim_contexts.items()
                    if (
                        context[0]["attempt_id"] == action.trace_id
                        and context[0]["through_sequence"]
                        == action.sequence - 1
                        and context[0]["check_set_id"]
                        == action.payload["check_set_id"]
                        and context[3]["stream_id"] == event["stream_id"]
                        and context[3]["stream_version"]
                        < event["stream_version"]
                    )
                }
                generic_action_command_hash = _command_hash(
                    {
                        "operation": "record_action",
                        "attempt_id": action.trace_id,
                        "action_type": action.kind.value,
                        "phase": action.phase.value,
                        "payload": action.terms()["payload"],
                    }
                )
                allowed_action_command_hashes = (
                    system_action_command_hashes
                    | {generic_action_command_hash}
                )
                if metadata["command_hash"] in system_action_command_hashes:
                    system_artifact_check_action_ids.add(action.id)
            else:
                allowed_action_command_hashes = {
                    _command_hash(
                        {
                            "operation": "record_action",
                            "attempt_id": action.trace_id,
                            "action_type": action.kind.value,
                            "phase": action.phase.value,
                            "payload": action.terms()["payload"],
                        }
                    )
                }
            if metadata["command_hash"] not in allowed_action_command_hashes:
                raise ValidationError(
                    f"Performance event {event['event_id']} action command "
                    "commitment mismatch."
                )
            if action.trace_id in terminal_attempts:
                if action.phase is not ActionPhase.POST_FEEDBACK:
                    raise ValidationError(
                        f"Performance event {event['event_id']} records a "
                        "non-feedback action after attempt termination."
                    )
            elif (
                active_attempt_by_session.get(attempt["session_id"])
                != action.trace_id
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} action is outside "
                    "the active attempt for its session."
                )
            if action.kind in {ActionKind.SUBMITTED, ActionKind.ABANDONED}:
                terminal_attempts.add(action.trace_id)
                active_attempt_by_session.pop(attempt["session_id"], None)
            next_action_sequence[action.trace_id] = action.sequence + 1
            prior_action_occurrence[action.trace_id] = action_occurred_at
            action_row = {
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
            if action.id in action_contexts_by_id:
                raise ValidationError(
                    f"Performance event {event['event_id']} repeats action "
                    f"{action.id}."
                )
            actions.append(action_row)
            action_contexts_by_id[action.id] = (action_row, action, event)
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event_type,
                    "attempt_id": action.trace_id,
                    "action_id": action.id,
                    "sequence": action.sequence,
                }
            )
        elif event_type == "PerformanceArtifactRunClaimed":
            label = f"Performance event {event['event_id']} artifact-run claim"
            if type(payload.get("claim_id")) is str:
                label += f" {payload['claim_id']}"
            reject_artifact_technical_key_collision(
                event["idempotency_key"],
                label,
            )
            _require_exact_fields(
                payload,
                frozenset(
                    {
                        "claim_id",
                        "caller_idempotency_key",
                        "attempt_id",
                        "session_id",
                        "session_revision",
                        "artifact_action_id",
                        "through_sequence",
                        "task_release_id",
                        "task_id",
                        "task_version",
                        "task_digest",
                        "artifact_digest",
                        "artifact_kind",
                        "artifact_manifest_digest",
                        "check_set_id",
                        "check_set_manifest_digest",
                        "runner_id",
                        "runner_version",
                        "request",
                        "request_digest",
                        "binding",
                        "binding_digest",
                        "command_hash",
                        "claimed_at",
                    }
                ),
                f"{label} payload",
            )
            _require_exact_fields(
                metadata,
                frozenset(
                    {
                        "artifact_run_schema_version",
                        "admission_mode",
                        "automatic_retry_allowed",
                        "shadow_only",
                    }
                ),
                f"{label} metadata",
            )
            try:
                request = ArtifactRunRequest.from_terms(payload["request"])
                binding = ArtifactRunnerBinding.from_terms(
                    payload["binding"]
                )
                expected_binding = bundled_synthetic_binding()
            except (ArtifactRunnerProtocolError, TypeError, ValueError) as exc:
                raise ValidationError(
                    f"{label} request or binding is invalid: {exc}"
                ) from exc
            for field_name in (
                "claim_id",
                "attempt_id",
                "session_id",
                "artifact_action_id",
                "task_release_id",
                "task_id",
                "artifact_kind",
                "check_set_id",
                "runner_id",
                "runner_version",
            ):
                _require_id(payload[field_name], f"{label} {field_name}")
            for field_name in (
                "task_digest",
                "artifact_digest",
                "artifact_manifest_digest",
                "check_set_manifest_digest",
                "request_digest",
                "binding_digest",
                "command_hash",
            ):
                _require_digest(payload[field_name], f"{label} {field_name}")
            if (
                type(payload["session_revision"]) is not int
                or payload["session_revision"] < 0
                or type(payload["through_sequence"]) is not int
                or payload["through_sequence"] < 0
                or type(payload["task_version"]) is not int
                or payload["task_version"] <= 0
            ):
                raise ValidationError(
                    f"{label} has an invalid revision, sequence, or task "
                    "version."
                )
            caller_key = payload["caller_idempotency_key"]
            if caller_key is not None:
                _require_text(caller_key, f"{label} caller key", 256)
                if caller_key.startswith(
                    (
                        PERFORMANCE_ARTIFACT_RUN_CLAIM_EVENT_KEY_PREFIX,
                        PERFORMANCE_ARTIFACT_RUN_RECEIPT_EVENT_KEY_PREFIX,
                        PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX,
                        PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX,
                    )
                ):
                    raise ValidationError(
                        f"{label} uses a reserved caller idempotency key."
                    )
                if caller_key in artifact_caller_keys:
                    raise ValidationError(
                        f"{label} repeats caller idempotency key {caller_key}."
                    )
                caller_collision = connection.execute(
                    """SELECT event_id FROM events
                       WHERE idempotency_key=?
                          OR (
                              event_type IN (
                                  'PerformanceScoringClaimed',
                                  'PerformanceScoringClaimMigrated',
                                  'PerformanceScoringReconciled'
                              )
                              AND json_extract(
                                  payload_json,
                                  '$.caller_idempotency_key'
                              )=?
                          )
                       ORDER BY stream_id, stream_version LIMIT 1""",
                    (caller_key, caller_key),
                ).fetchone()
                if caller_collision is not None:
                    raise ValidationError(
                        f"{label} caller idempotency key collides with event "
                        f"{caller_collision['event_id']}."
                    )
                artifact_caller_keys.add(caller_key)
            if payload["command_hash"] in artifact_command_hashes:
                raise ValidationError(
                    f"{label} repeats command hash {payload['command_hash']}."
                )
            artifact_command_hashes.add(payload["command_hash"])
            claimed = _aware_timestamp(
                payload["claimed_at"], f"{label} claimed_at"
            )
            if to_timestamp(claimed) != payload["claimed_at"]:
                raise ValidationError(
                    f"{label} claimed_at is not canonical UTC."
                )

            attempt = attempts_by_id.get(payload["attempt_id"])
            artifact_context = action_contexts_by_id.get(
                payload["artifact_action_id"]
            )
            if attempt is None or artifact_context is None:
                raise ValidationError(
                    f"{label} does not follow its attempt and artifact action."
                )
            artifact_row, artifact_action, artifact_event = artifact_context
            attempt_event = attempt_events_by_id[payload["attempt_id"]]
            attempt_action_contexts = [
                context
                for context in action_contexts_by_id.values()
                if context[0]["attempt_id"] == payload["attempt_id"]
            ]
            boundary_context = max(
                attempt_action_contexts,
                key=lambda context: context[0]["sequence"],
                default=None,
            )
            if (
                artifact_action.kind is not ActionKind.ARTIFACT_CHECKPOINT
                or artifact_row["attempt_id"] != payload["attempt_id"]
                or artifact_action.sequence > payload["through_sequence"]
                or boundary_context is None
                or boundary_context[0]["sequence"]
                != payload["through_sequence"]
                or artifact_action.payload["artifact_digest"]
                != payload["artifact_digest"]
                or artifact_action.payload["artifact_kind"]
                != payload["artifact_kind"]
            ):
                raise ValidationError(
                    f"{label} artifact or action-trace boundary mismatch."
                )
            boundary_event = boundary_context[2]
            revision_boundaries = connection.execute(
                """SELECT event_type FROM events
                   WHERE stream_id=?
                     AND stream_version > ?
                     AND stream_version < ?
                     AND session_id=?
                     AND event_type IN (
                         'QuestionSelected', 'ResponseSubmitted',
                         'SessionEnded'
                     )
                   ORDER BY stream_version""",
                (
                    event["stream_id"],
                    attempt_event["stream_version"],
                    event["stream_version"],
                    payload["session_id"],
                ),
            ).fetchall()
            expected_session_revision = attempt["session_revision"] + sum(
                boundary["event_type"]
                in {"QuestionSelected", "ResponseSubmitted"}
                for boundary in revision_boundaries
            )
            if (
                payload["session_revision"] != expected_session_revision
                or any(
                    boundary["event_type"] == "SessionEnded"
                    for boundary in revision_boundaries
                )
            ):
                raise ValidationError(
                    f"{label} session revision or active interval mismatch."
                )

            task_row = connection.execute(
                """SELECT task.definition_json,
                          task.task_digest AS definition_digest,
                          member.task_digest AS member_digest,
                          member.status,
                          release.corpus_release_id,
                          json_extract(
                              release.review_json, '$.reviewer_kind'
                          ) AS reviewer_kind
                   FROM performance_task_releases release
                   JOIN release_performance_tasks member
                     ON member.release_id=release.id
                   JOIN performance_tasks task
                     ON task.task_id=member.task_id
                    AND task.task_version=member.task_version
                   WHERE release.id=? AND member.task_id=?
                     AND member.task_version=?""",
                (
                    payload["task_release_id"],
                    payload["task_id"],
                    payload["task_version"],
                ),
            ).fetchone()
            if task_row is None:
                raise ValidationError(
                    f"{label} has no immutable released task."
                )
            try:
                task = LearningTask.from_terms(
                    _json_object(
                        task_row["definition_json"],
                        f"{label} task definition",
                    )
                )
                released_contract = PerformanceLedger._artifact_runner_contract(
                    task, binding
                )
            except (TypeError, ValueError, ValidationError) as exc:
                raise ValidationError(
                    f"{label} released runner contract is invalid: {exc}"
                ) from exc
            if (
                attempt["session_id"] != payload["session_id"]
                or attempt["task_release_id"] != payload["task_release_id"]
                or attempt["task_id"] != payload["task_id"]
                or attempt["task_version"] != payload["task_version"]
                or attempt["task_digest"] != payload["task_digest"]
                or attempt["corpus_release_id"]
                != task_row["corpus_release_id"]
                or task_row["status"] not in SERVICEABLE_TASK_STATUSES
                or task_row["reviewer_kind"] != "human"
                or task_row["definition_digest"] != payload["task_digest"]
                or task_row["member_digest"] != payload["task_digest"]
                or task.digest != payload["task_digest"]
                or binding != expected_binding
                or released_contract.artifact_manifests
                != (
                    (
                        payload["artifact_kind"],
                        payload["artifact_manifest_digest"],
                    ),
                )
                or released_contract.check_set_manifests
                != (
                    (
                        payload["check_set_id"],
                        payload["check_set_manifest_digest"],
                    ),
                )
            ):
                raise ValidationError(
                    f"{label} task, release, or manifest boundary mismatch."
                )
            expected_command_hash = _command_hash(
                {
                    "operation": "run_artifact_check",
                    "attempt_id": payload["attempt_id"],
                    "artifact_action_id": payload["artifact_action_id"],
                    "artifact_digest": payload["artifact_digest"],
                    "artifact_size_bytes": request.artifact_size_bytes,
                    "artifact_kind": payload["artifact_kind"],
                    "artifact_manifest_digest": payload[
                        "artifact_manifest_digest"
                    ],
                    "check_set_id": payload["check_set_id"],
                    "check_set_manifest_digest": payload[
                        "check_set_manifest_digest"
                    ],
                    "checker_id": binding.checker_id.value,
                    "checker_version": binding.checker_version,
                    "runner_id": payload["runner_id"],
                    "runner_version": payload["runner_version"],
                    "binding_digest": binding.digest,
                }
            )
            expected_payload = performance_artifact_run_claim_payload(
                claim_id=payload["claim_id"],
                caller_idempotency_key=caller_key,
                attempt_id=payload["attempt_id"],
                session_id=payload["session_id"],
                session_revision=payload["session_revision"],
                artifact_action_id=payload["artifact_action_id"],
                through_sequence=payload["through_sequence"],
                task_release_id=payload["task_release_id"],
                task_id=payload["task_id"],
                task_version=payload["task_version"],
                task_digest=payload["task_digest"],
                artifact_digest=payload["artifact_digest"],
                artifact_kind=payload["artifact_kind"],
                artifact_manifest_digest=payload[
                    "artifact_manifest_digest"
                ],
                check_set_id=payload["check_set_id"],
                check_set_manifest_digest=payload[
                    "check_set_manifest_digest"
                ],
                runner_id=payload["runner_id"],
                runner_version=payload["runner_version"],
                request=request.terms(),
                request_digest=request.digest,
                binding=binding.terms(),
                binding_digest=binding.digest,
                command_hash=expected_command_hash,
                claimed_at=payload["claimed_at"],
            )
            if (
                request.digest != payload["request_digest"]
                or binding.digest != payload["binding_digest"]
                or request.runner_binding_digest != binding.digest
                or request.run_id
                != "arun_" + expected_command_hash[:24]
                or request.artifact_sha256 != payload["artifact_digest"]
                or request.artifact_kind != payload["artifact_kind"]
                or request.artifact_manifest_digest
                != payload["artifact_manifest_digest"]
                or request.check_set_id != payload["check_set_id"]
                or request.check_set_manifest_digest
                != payload["check_set_manifest_digest"]
                or binding.artifact_kind != payload["artifact_kind"]
                or binding.artifact_manifest_digest
                != payload["artifact_manifest_digest"]
                or binding.check_set_id != payload["check_set_id"]
                or binding.check_set_manifest_digest
                != payload["check_set_manifest_digest"]
                or binding.runner_id != payload["runner_id"]
                or binding.runner_version != payload["runner_version"]
                or payload["command_hash"] != expected_command_hash
                or canonical_json(payload) != canonical_json(expected_payload)
            ):
                raise ValidationError(
                    f"{label} request, binding, or command digest mismatch."
                )
            if (
                event["schema_version"]
                != PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION
                or metadata["artifact_run_schema_version"]
                != PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION
                or metadata["admission_mode"] != "pre_runner"
                or metadata["automatic_retry_allowed"] is not False
                or metadata["shadow_only"] is not True
                or event["stream_id"]
                != f"learner:{attempt['learner_id']}"
                or event["learner_id"] != attempt["learner_id"]
                or event["session_id"] != payload["session_id"]
                or event["idempotency_key"]
                != performance_artifact_run_claim_event_key(
                    expected_command_hash
                )
                or event["correlation_id"] != payload["attempt_id"]
                or event["causation_id"] != artifact_event["event_id"]
                or event["occurred_at"] != payload["claimed_at"]
                or attempt_event["stream_id"] != event["stream_id"]
                or attempt_event["stream_version"]
                >= event["stream_version"]
                or artifact_event["stream_id"] != event["stream_id"]
                or artifact_event["stream_version"]
                >= event["stream_version"]
                or boundary_event["stream_id"] != event["stream_id"]
                or boundary_event["stream_version"]
                >= event["stream_version"]
                or claimed
                < _aware_timestamp(
                    artifact_event["occurred_at"],
                    f"{label} artifact occurrence",
                )
            ):
                raise ValidationError(
                    f"{label} event envelope or trace order mismatch."
                )
            claim_row = {
                "id": payload["claim_id"],
                "event_id": event["event_id"],
                "idempotency_key": caller_key,
                "attempt_id": payload["attempt_id"],
                "session_id": payload["session_id"],
                "session_revision": payload["session_revision"],
                "artifact_action_id": payload["artifact_action_id"],
                "through_sequence": payload["through_sequence"],
                "task_release_id": payload["task_release_id"],
                "task_id": payload["task_id"],
                "task_version": payload["task_version"],
                "task_digest": payload["task_digest"],
                "artifact_digest": payload["artifact_digest"],
                "artifact_kind": payload["artifact_kind"],
                "artifact_manifest_digest": payload[
                    "artifact_manifest_digest"
                ],
                "check_set_id": payload["check_set_id"],
                "check_set_manifest_digest": payload[
                    "check_set_manifest_digest"
                ],
                "runner_id": payload["runner_id"],
                "runner_version": payload["runner_version"],
                "request_json": canonical_json(request.terms()),
                "request_digest": request.digest,
                "binding_json": canonical_json(binding.terms()),
                "binding_digest": binding.digest,
                "command_hash": expected_command_hash,
                "claimed_at": payload["claimed_at"],
            }
            if payload["claim_id"] in artifact_claim_contexts:
                raise ValidationError(
                    f"{label} repeats claim {payload['claim_id']}."
                )
            artifact_run_claims.append(claim_row)
            artifact_claim_contexts[payload["claim_id"]] = (
                claim_row,
                request,
                binding,
                event,
                artifact_action,
                artifact_event,
            )
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event_type,
                    "attempt_id": payload["attempt_id"],
                    "claim_id": payload["claim_id"],
                    "artifact_action_id": payload["artifact_action_id"],
                    "through_sequence": payload["through_sequence"],
                }
            )
        elif event_type == "PerformanceArtifactRunObserved":
            label = (
                f"Performance event {event['event_id']} artifact-run "
                "observation"
            )
            if type(payload.get("receipt_id")) is str:
                label += f" {payload['receipt_id']}"
            reject_artifact_technical_key_collision(
                event["idempotency_key"],
                label,
            )
            _require_exact_fields(
                payload,
                frozenset(
                    {
                        "receipt_id",
                        "claim_id",
                        "attempt_id",
                        "check_action_id",
                        "outcome",
                        "result",
                        "result_digest",
                        "receipt",
                        "receipt_digest",
                        "started_at",
                        "completed_at",
                    }
                ),
                f"{label} payload",
            )
            _require_exact_fields(
                metadata,
                frozenset(
                    {
                        "artifact_run_schema_version",
                        "observational_only",
                        "projection_applied",
                        "certification_applied",
                        "skill_authority",
                        "shadow_only",
                    }
                ),
                f"{label} metadata",
            )
            for field_name in ("receipt_id", "claim_id", "attempt_id"):
                _require_id(payload[field_name], f"{label} {field_name}")
            _require_digest(
                payload["receipt_digest"], f"{label} receipt_digest"
            )
            claim_context = artifact_claim_contexts.get(payload["claim_id"])
            if claim_context is None:
                raise ValidationError(
                    f"{label} does not follow an artifact-run claim."
                )
            (
                claim,
                request,
                binding,
                claim_event,
                artifact_action,
                _artifact_event,
            ) = claim_context
            attempt = attempts_by_id[claim["attempt_id"]]
            session_boundaries = connection.execute(
                """SELECT event_id, event_type FROM events
                   WHERE stream_id=?
                     AND stream_version > ?
                     AND stream_version < ?
                     AND session_id=?
                     AND event_type IN (
                         'QuestionSelected', 'ResponseSubmitted',
                         'SessionEnded'
                     )
                   ORDER BY stream_version""",
                (
                    claim_event["stream_id"],
                    claim_event["stream_version"],
                    event["stream_version"],
                    claim["session_id"],
                ),
            ).fetchall()
            if session_boundaries:
                raise ValidationError(
                    f"{label} crosses a session revision or end boundary."
                )
            try:
                operational = OperationalArtifactRunReceipt.from_terms(
                    payload["receipt"]
                )
            except (TypeError, ValueError, ValidationError) as exc:
                raise ValidationError(
                    f"{label} operational receipt is invalid: {exc}"
                ) from exc
            process: ArtifactProcessReceipt | None
            if payload["result"] is None:
                process = None
            else:
                try:
                    process = ArtifactProcessReceipt.from_terms(
                        payload["result"]
                    )
                except (
                    ArtifactRunnerProtocolError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise ValidationError(
                        f"{label} process receipt is invalid: {exc}"
                    ) from exc
            started = _aware_timestamp(
                payload["started_at"], f"{label} started_at"
            )
            completed = _aware_timestamp(
                payload["completed_at"], f"{label} completed_at"
            )
            if (
                to_timestamp(started) != payload["started_at"]
                or to_timestamp(completed) != payload["completed_at"]
            ):
                raise ValidationError(
                    f"{label} timestamps are not canonical UTC."
                )
            outcome = payload["outcome"]
            successful_observation = outcome in {
                ArtifactRunOutcome.COMPLETED.value,
                ArtifactRunOutcome.INVALID_ARTIFACT.value,
            }
            if successful_observation:
                if process is None:
                    raise ValidationError(
                        f"{label} has no typed process receipt."
                    )
                expected_process_outcome = process.result.outcome.value
                if expected_process_outcome != outcome:
                    raise ValidationError(
                        f"{label} result outcome does not match its receipt."
                    )
                result_digest: str | None = process.digest
                result_terms: dict[str, Any] | None = process.terms()
            else:
                if outcome not in {
                    "runner_failed",
                    ArtifactRunOutcome.TIMED_OUT.value,
                }:
                    raise ValidationError(
                        f"{label} has an unknown terminal outcome."
                    )
                if process is not None:
                    raise ValidationError(
                        f"{label} failed observation carries a process result."
                    )
                result_digest = None
                result_terms = None
            expected_operational = OperationalArtifactRunReceipt(
                claim_id=claim["id"],
                attempt_id=claim["attempt_id"],
                artifact_action_id=claim["artifact_action_id"],
                artifact_digest=claim["artifact_digest"],
                artifact_kind=claim["artifact_kind"],
                artifact_manifest_digest=claim[
                    "artifact_manifest_digest"
                ],
                check_set_id=claim["check_set_id"],
                check_set_manifest_digest=claim[
                    "check_set_manifest_digest"
                ],
                runner_id=claim["runner_id"],
                runner_version=claim["runner_version"],
                outcome=outcome,
                started_at=payload["started_at"],
                completed_at=payload["completed_at"],
                result_digest=result_digest,
                request_digest=request.digest,
                binding_digest=binding.digest,
            )
            if (
                payload["claim_id"] in artifact_receipt_claim_ids
                or payload["receipt_digest"] in artifact_receipt_digests
            ):
                raise ValidationError(
                    f"{label} repeats a terminal claim or receipt digest."
                )
            artifact_receipt_claim_ids.add(payload["claim_id"])
            artifact_receipt_digests.add(payload["receipt_digest"])

            check_action_id = payload["check_action_id"]
            check_context = (
                None
                if check_action_id is None
                else action_contexts_by_id.get(check_action_id)
            )
            actions_between_claim_and_receipt = [
                context
                for context in action_contexts_by_id.values()
                if (
                    context[0]["attempt_id"] == claim["attempt_id"]
                    and context[2]["stream_id"] == event["stream_id"]
                    and claim_event["stream_version"]
                    < context[2]["stream_version"]
                    < event["stream_version"]
                )
            ]
            if successful_observation:
                if check_context is None or process is None:
                    raise ValidationError(
                        f"{label} is missing its generated check action."
                    )
                check_row, check_action, check_event = check_context
                result = process.result
                expected_check_command = _command_hash(
                    {
                        "operation": "record_artifact_run_check",
                        "claim_id": claim["id"],
                        "result_digest": process.digest,
                    }
                )
                if (
                    check_action.kind is not ActionKind.CHECK_RUN
                    or check_action.trace_id != claim["attempt_id"]
                    or check_action.sequence != claim["through_sequence"] + 1
                    or check_action.phase is not artifact_action.phase
                    or check_action.payload["check_set_id"]
                    != claim["check_set_id"]
                    or check_action.payload["passed"] != result.passed
                    or check_action.payload["failed"] != result.failed
                    or check_action.payload["errored"] != result.errored
                    or check_action.payload["skipped"] != result.skipped
                    or check_action.payload["result_digest"] != process.digest
                    or check_row["command_hash"] != expected_check_command
                    or check_row["occurred_at"] != payload["completed_at"]
                    or check_event["stream_id"] != event["stream_id"]
                    or check_event["stream_version"]
                    <= claim_event["stream_version"]
                    or check_event["stream_version"]
                    >= event["stream_version"]
                    or [context[0]["id"] for context in actions_between_claim_and_receipt]
                    != [check_action_id]
                ):
                    raise ValidationError(
                        f"{label} check action counters, outcome, or event "
                        "order mismatch."
                    )
            elif (
                check_action_id is not None
                or check_context is not None
                or actions_between_claim_and_receipt
            ):
                raise ValidationError(
                    f"{label} failed run has an intervening check action."
                )

            expected_payload = performance_artifact_run_observed_payload(
                receipt_id=payload["receipt_id"],
                claim_id=claim["id"],
                attempt_id=claim["attempt_id"],
                check_action_id=check_action_id,
                outcome=outcome,
                result=result_terms,
                result_digest=result_digest,
                receipt=expected_operational.terms(),
                receipt_digest=expected_operational.digest,
                started_at=payload["started_at"],
                completed_at=payload["completed_at"],
            )
            if (
                payload["attempt_id"] != claim["attempt_id"]
                or process is not None
                and (
                    process.request != request
                    or process.binding != binding
                    or process.digest != payload["result_digest"]
                )
                or payload["result_digest"] != result_digest
                or operational != expected_operational
                or operational.digest != payload["receipt_digest"]
                or canonical_json(payload) != canonical_json(expected_payload)
                or started
                < _aware_timestamp(
                    claim["claimed_at"], f"{label} claim occurrence"
                )
                or completed < started
            ):
                raise ValidationError(
                    f"{label} process or operational receipt mismatch."
                )
            if (
                event["schema_version"]
                != PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION
                or metadata["artifact_run_schema_version"]
                != PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION
                or metadata["observational_only"] is not True
                or metadata["projection_applied"] is not False
                or metadata["certification_applied"] is not False
                or metadata["skill_authority"] is not False
                or metadata["shadow_only"] is not True
                or event["stream_id"]
                != f"learner:{attempt['learner_id']}"
                or event["learner_id"] != attempt["learner_id"]
                or event["session_id"] is not None
                or event["idempotency_key"]
                != performance_artifact_run_receipt_event_key(
                    expected_operational.digest
                )
                or event["correlation_id"] != claim["attempt_id"]
                or event["causation_id"] != claim_event["event_id"]
                or event["occurred_at"] != payload["completed_at"]
                or claim_event["stream_id"] != event["stream_id"]
                or claim_event["stream_version"] >= event["stream_version"]
            ):
                raise ValidationError(
                    f"{label} event envelope or claim order mismatch."
                )
            receipt_row = {
                "id": payload["receipt_id"],
                "event_id": event["event_id"],
                "claim_id": claim["id"],
                "check_action_id": check_action_id,
                "outcome": outcome,
                "result_json": (
                    None
                    if result_terms is None
                    else canonical_json(result_terms)
                ),
                "result_digest": result_digest,
                "receipt_json": canonical_json(expected_operational.terms()),
                "receipt_digest": expected_operational.digest,
                "started_at": payload["started_at"],
                "completed_at": payload["completed_at"],
            }
            artifact_run_receipts.append(receipt_row)
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event_type,
                    "attempt_id": claim["attempt_id"],
                    "claim_id": claim["id"],
                    "receipt_id": payload["receipt_id"],
                    "outcome": outcome,
                    "check_action_id": check_action_id,
                }
            )
        elif event_type in {
            "PerformanceScoringClaimed",
            "PerformanceScoringClaimMigrated",
        }:
            claim_schema_version = metadata.get("claim_schema_version")
            base_payload_fields = {
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
            _require_exact_fields(
                payload,
                frozenset(
                    base_payload_fields
                    | (
                        {
                            "scoring_request_digest",
                            "provider_binding_digest",
                            "provider_operation_digest",
                            "provider",
                        }
                        if claim_schema_version == 2
                        else set()
                    )
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
            expected_event_schema = (
                2 if claim_schema_version == 2 else 1
            )
            if (
                claim_schema_version not in {1, 2}
                or (
                    claim_schema_version == 2
                    and event_type != "PerformanceScoringClaimed"
                )
                or event["schema_version"] != expected_event_schema
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
            if claim_schema_version == 2:
                try:
                    provider = RegisteredProvider.from_terms(
                        payload["provider"]
                    )
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        f"Performance event {event['event_id']} provider "
                        f"snapshot is invalid: {exc}"
                    ) from exc
                expected_operation_digest = (
                    provider_scoring_operation_digest(
                        claim_id=payload["claim_id"],
                        evaluation_id=payload["evaluation_id"],
                        scoring_request_digest=payload[
                            "scoring_request_digest"
                        ],
                        provider_binding_digest=payload[
                            "provider_binding_digest"
                        ],
                    )
                )
                if (
                    provider.provider_id != payload["provider_id"]
                    or provider.provider_version
                    != payload["provider_version"]
                    or provider.binding_digest
                    != payload["provider_binding_digest"]
                    or payload["provider_operation_digest"]
                    != expected_operation_digest
                ):
                    raise ValidationError(
                        f"Performance event {event['event_id']} provider "
                        "operation boundary mismatch."
                    )
                claim_provider_snapshots[payload["claim_id"]] = provider
            scoring_claims.append(
                {
                    "id": payload["claim_id"],
                    "event_id": event["event_id"],
                    "claim_schema_version": claim_schema_version,
                    "idempotency_key": payload["caller_idempotency_key"],
                    "attempt_id": payload["attempt_id"],
                    "evaluation_id": payload["evaluation_id"],
                    "through_sequence": payload["through_sequence"],
                    "provider_id": payload["provider_id"],
                    "provider_version": payload["provider_version"],
                    "action_trace_digest": payload["action_trace_digest"],
                    "scoring_request_digest": payload.get(
                        "scoring_request_digest"
                    ),
                    "provider_binding_digest": payload.get(
                        "provider_binding_digest"
                    ),
                    "provider_operation_digest": payload.get(
                        "provider_operation_digest"
                    ),
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
        elif event_type == "PerformanceScoringReconciled":
            _require_exact_fields(
                payload,
                frozenset(
                    {
                        "reconciliation_id",
                        "caller_idempotency_key",
                        "claim_id",
                        "attempt_id",
                        "evaluation_id",
                        "outcome",
                        "scoring_request_digest",
                        "provider_binding_digest",
                        "provider_operation_digest",
                        "reconciler_id",
                        "reconciler_version",
                        "reconciliation_binding_digest",
                        "receipt",
                        "receipt_digest",
                        "normalized_result_digest",
                        "reconciled_at",
                        "command_hash",
                        "reconciler",
                    }
                ),
                f"Performance event {event['event_id']} payload",
            )
            _require_exact_fields(
                metadata,
                frozenset(
                    {
                        "reconciliation_schema_version",
                        "command_hash",
                        "observational_only",
                        "automatic_retry_allowed",
                        "projection_applied",
                        "certification_applied",
                        "skill_authority",
                        "shadow_only",
                    }
                ),
                f"Performance event {event['event_id']} metadata",
            )
            try:
                receipt = ScoringReconciliationReceipt.from_terms(
                    payload["receipt"]
                )
                reconciler = RegisteredReconciler.from_terms(
                    payload["reconciler"]
                )
                outcome = ReconciliationOutcome(payload["outcome"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Performance event {event['event_id']} reconciliation "
                    f"terms are invalid: {exc}"
                ) from exc
            expected_command_hash = _command_hash(
                {
                    "operation": "reconcile_scoring_claim",
                    "claim_id": payload["claim_id"],
                    "provider_operation_digest": payload[
                        "provider_operation_digest"
                    ],
                    "reconciler_id": payload["reconciler_id"],
                    "reconciler_version": payload[
                        "reconciler_version"
                    ],
                    "reconciliation_binding_digest": payload[
                        "reconciliation_binding_digest"
                    ],
                    "receipt_digest": payload["receipt_digest"],
                    "caller_idempotency_key": payload[
                        "caller_idempotency_key"
                    ],
                }
            )
            if (
                receipt.digest != payload["receipt_digest"]
                or receipt.outcome is not outcome
                or receipt.claim_id != payload["claim_id"]
                or receipt.evaluation_id != payload["evaluation_id"]
                or receipt.provider_operation_digest
                != payload["provider_operation_digest"]
                or reconciler.reconciler_id != payload["reconciler_id"]
                or reconciler.reconciler_version
                != payload["reconciler_version"]
                or reconciler.binding_digest
                != payload["reconciliation_binding_digest"]
                or metadata["reconciliation_schema_version"] != 1
                or metadata["command_hash"] != expected_command_hash
                or metadata["observational_only"] is not True
                or metadata["automatic_retry_allowed"] is not False
                or metadata["projection_applied"] is not False
                or metadata["certification_applied"] is not False
                or metadata["skill_authority"] is not False
                or metadata["shadow_only"] is not True
                or payload["command_hash"] != expected_command_hash
                or event["idempotency_key"]
                != performance_scoring_reconciliation_event_key(
                    expected_command_hash
                )
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} reconciliation "
                    "boundary mismatch."
                )
            scoring_reconciliations.append(
                {
                    "id": payload["reconciliation_id"],
                    "event_id": event["event_id"],
                    "idempotency_key": payload[
                        "caller_idempotency_key"
                    ],
                    "claim_id": payload["claim_id"],
                    "attempt_id": payload["attempt_id"],
                    "evaluation_id": payload["evaluation_id"],
                    "outcome": outcome.value,
                    "scoring_request_digest": payload[
                        "scoring_request_digest"
                    ],
                    "provider_binding_digest": payload[
                        "provider_binding_digest"
                    ],
                    "provider_operation_digest": payload[
                        "provider_operation_digest"
                    ],
                    "reconciler_id": payload["reconciler_id"],
                    "reconciler_version": payload[
                        "reconciler_version"
                    ],
                    "reconciliation_binding_digest": payload[
                        "reconciliation_binding_digest"
                    ],
                    "receipt_json": canonical_json(receipt.terms()),
                    "receipt_digest": receipt.digest,
                    "normalized_result_digest": payload[
                        "normalized_result_digest"
                    ],
                    "reconciled_at": payload["reconciled_at"],
                    "command_hash": expected_command_hash,
                }
            )
            reconciliation_receipts[
                payload["reconciliation_id"]
            ] = receipt
            reconciliation_events[payload["reconciliation_id"]] = event
            checkpoints.append(
                {
                    "event_id": event["event_id"],
                    "event_type": event_type,
                    "attempt_id": payload["attempt_id"],
                    "claim_id": payload["claim_id"],
                    "evaluation_id": payload["evaluation_id"],
                    "reconciliation_id": payload["reconciliation_id"],
                    "outcome": outcome.value,
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
            _require_id(
                payload["evaluation_id"],
                f"Performance event {event['event_id']} evaluation_id",
            )
            _require_id(
                payload["attempt_id"],
                f"Performance event {event['event_id']} attempt_id",
            )
            _require_digest(
                payload["command_hash"],
                f"Performance event {event['event_id']} command_hash",
            )
            if payload["evaluation_id"] in legacy_scoring_exemptions:
                raise ValidationError(
                    f"Performance event {event['event_id']} duplicates a "
                    "legacy scoring exception."
                )
            legacy_scoring_exemptions[payload["evaluation_id"]] = (
                payload,
                event,
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
            authority = payload["authority"]
            if type(authority) is not dict:
                raise ValidationError(
                    f"Performance event {event['event_id']} authority must be "
                    "an object."
                )
            _require_exact_fields(
                authority,
                frozenset(
                    {"normalized_result", "normalized_result_digest"}
                ),
                f"Performance event {event['event_id']} authority",
            )
            try:
                evaluation = TaskEvaluation.from_terms(payload["evaluation"])
                normalized_result = NormalizedScoringResult.from_terms(
                    authority["normalized_result"]
                )
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Performance event {event['event_id']} evaluation or "
                    f"authority is invalid: {exc}"
                ) from exc
            if (
                evaluation.trace_id != payload["attempt_id"]
                or evaluation.digest != payload["evaluation_digest"]
                or normalized_result.evaluation.digest != evaluation.digest
                or authority["normalized_result_digest"]
                != normalized_result.digest
                or type(payload["through_sequence"]) is not int
                or payload["through_sequence"] < 0
                or metadata["shadow_only"] is not True
                or metadata["projection_applied"] is not False
                or metadata["certification_applied"] is not False
            ):
                raise ValidationError(
                    f"Performance event {event['event_id']} evaluation boundary mismatch."
                )
            _require_digest(
                metadata["command_hash"],
                f"Performance event {event['event_id']} command_hash",
            )
            matching_claims = [
                claim
                for claim in scoring_claims
                if claim["evaluation_id"] == evaluation.id
                and claim["attempt_id"] == evaluation.trace_id
                and claim["through_sequence"]
                == payload["through_sequence"]
            ]
            if matching_claims:
                if (
                    len(matching_claims) != 1
                    or metadata["command_hash"]
                    != matching_claims[0]["command_hash"]
                ):
                    raise ValidationError(
                        f"Performance event {event['event_id']} evaluation "
                        "does not match its exact scoring claim."
                    )
            elif (
                normalized_result.normalization_mode
                is NormalizationMode.DIRECT_IMPORT
            ):
                imported = _authority_free_imported_evaluation(evaluation)
                expected_import_command_hash = _command_hash(
                    {
                        "operation": "import_evaluation",
                        "attempt_id": evaluation.trace_id,
                        "through_sequence": payload["through_sequence"],
                        "provider_id": (
                            normalized_result.provider.provider_id
                        ),
                        "provider_version": (
                            normalized_result.provider.provider_version
                        ),
                        "declared_kind": (
                            normalized_result.provider.declared_kind.value
                        ),
                        "imported_digest": imported.digest,
                    }
                )
                if (
                    metadata["command_hash"]
                    != expected_import_command_hash
                ):
                    raise ValidationError(
                        f"Performance event {event['event_id']} direct-import "
                        "command commitment mismatch."
                    )
            evaluations.append(
                {
                    "id": evaluation.id,
                    "event_id": event["event_id"],
                    "attempt_id": evaluation.trace_id,
                    "through_sequence": payload["through_sequence"],
                    "evaluation_digest": evaluation.digest,
                    "evaluation_json": canonical_json(evaluation.terms()),
                    "authority_json": canonical_json(authority),
                    "recorded_at": event["recorded_at"],
                    "command_hash": metadata["command_hash"],
                }
            )
            recovered_evaluations[evaluation.id] = (
                evaluation,
                authority,
                event,
            )
            evaluation_modes[evaluation.id] = (
                normalized_result.normalization_mode
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
    for evaluation_id, (
        exemption_payload,
        exemption_event,
    ) in legacy_scoring_exemptions.items():
        recovered = recovered_evaluations.get(evaluation_id)
        if recovered is None:
            raise ValidationError(
                "Legacy scoring exception has no exact task evaluation."
            )
        evaluation, _authority, evaluation_event = recovered
        attempt = attempts_by_id.get(evaluation.trace_id)
        evaluation_metadata = _event_metadata(evaluation_event)
        claims = [
            claim
            for claim in scoring_claims
            if claim["evaluation_id"] == evaluation_id
        ]
        if (
            attempt is None
            or exemption_payload["attempt_id"] != evaluation.trace_id
            or exemption_payload["command_hash"]
            != evaluation_metadata["command_hash"]
            or evaluation_modes[evaluation_id]
            is not NormalizationMode.REGISTERED_PROVIDER
            or claims
            or exemption_event["schema_version"]
            != PERFORMANCE_EVENT_SCHEMA_VERSION
            or exemption_event["stream_id"]
            != f"learner:{attempt['learner_id']}"
            or exemption_event["learner_id"] != attempt["learner_id"]
            or exemption_event["session_id"]
            not in {None, attempt["session_id"]}
            or exemption_event["correlation_id"] != attempt["id"]
            or exemption_event["causation_id"] != evaluation_id
            or exemption_event["stream_version"]
            <= evaluation_event["stream_version"]
        ):
            raise ValidationError(
                "Legacy scoring exception does not match its exact "
                "registered-provider evaluation boundary."
            )

    for evaluation_id, mode in evaluation_modes.items():
        claims = [
            claim
            for claim in scoring_claims
            if claim["evaluation_id"] == evaluation_id
        ]
        exemption = legacy_scoring_exemptions.get(evaluation_id)
        if mode is NormalizationMode.REGISTERED_PROVIDER:
            matching_claims = [
                claim
                for claim in claims
                if claim["attempt_id"]
                == recovered_evaluations[evaluation_id][0].trace_id
                and claim["through_sequence"]
                == next(
                    item["through_sequence"]
                    for item in evaluations
                    if item["id"] == evaluation_id
                )
            ]
            if (
                len(matching_claims)
                + (1 if exemption is not None else 0)
                != 1
                or len(claims) != len(matching_claims)
            ):
                raise ValidationError(
                    "Registered-provider evaluation must have exactly one "
                    "matching scoring claim or schema-v14 exemption."
                )
        elif claims:
            recovered_claims = [
                claim
                for claim in claims
                if any(
                    observation["claim_id"] == claim["id"]
                    and observation["evaluation_id"] == evaluation_id
                    and observation["outcome"]
                    == ReconciliationOutcome.COMPLETED.value
                    for observation in scoring_reconciliations
                )
            ]
            if (
                len(claims) != 1
                or len(recovered_claims) != 1
                or exemption is not None
            ):
                raise ValidationError(
                    "Direct-import evaluation with a provider callback claim "
                    "requires exactly one completed reconciliation."
                )
        elif exemption is not None:
            raise ValidationError(
                "Direct-import evaluation cannot use a legacy provider "
                "exception."
            )

    receipt_bound_check_action_ids = {
        receipt["check_action_id"]
        for receipt in artifact_run_receipts
        if receipt["check_action_id"] is not None
    }
    if system_artifact_check_action_ids != receipt_bound_check_action_ids:
        raise ValidationError(
            "System artifact check actions must be bound one-for-one to "
            "successful terminal artifact-run observations."
        )

    for observation in scoring_reconciliations:
        if observation["outcome"] != ReconciliationOutcome.COMPLETED.value:
            continue
        receipt = reconciliation_receipts.get(observation["id"])
        reconciliation_event = reconciliation_events.get(observation["id"])
        provider = claim_provider_snapshots.get(observation["claim_id"])
        recovered = recovered_evaluations.get(observation["evaluation_id"])
        if (
            receipt is None
            or reconciliation_event is None
            or provider is None
            or recovered is None
        ):
            raise ValidationError(
                "Completed scoring reconciliation cannot reconstruct its "
                "claim-bound recovered result."
            )
        evaluation, authority, evaluation_event = recovered
        if (
            evaluation_event["occurred_at"] != receipt.completed_at
            or evaluation_event["stream_id"]
            != reconciliation_event["stream_id"]
            or evaluation_event["stream_version"]
            <= reconciliation_event["stream_version"]
        ):
            raise ValidationError(
                "Completed scoring reconciliation recovered evaluation has "
                "an invalid completion time or event order."
            )
        normalized_terms = authority.get("normalized_result")
        if type(normalized_terms) is not dict:
            raise ValidationError(
                "Completed scoring reconciliation has malformed recovered "
                "authority."
            )
        try:
            stored_result = NormalizedScoringResult.from_terms(
                normalized_terms
            )
            imported = _authority_free_imported_evaluation(evaluation)
            expected_result = _normalize_recovered_imported_evaluation(
                stored_result.request,
                imported,
                provider,
                observation["provider_operation_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Completed scoring reconciliation has an invalid recovered "
                f"result: {exc}"
            ) from exc
        if receipt.result_digest != imported.digest:
            raise ValidationError(
                "Completed scoring reconciliation receipt does not match "
                "its recovered imported result."
            )
        if (
            observation["normalized_result_digest"]
            != expected_result.digest
            or authority.get("normalized_result_digest")
            != expected_result.digest
            or canonical_json(stored_result.terms())
            != canonical_json(expected_result.terms())
        ):
            raise ValidationError(
                "Completed scoring reconciliation does not match its "
                "claim-bound shadow normalization."
            )

    snapshot = {
        "attempts": sorted(attempts, key=lambda item: item["id"]),
        "actions": sorted(
            actions,
            key=lambda item: (item["attempt_id"], item["sequence"], item["id"]),
        ),
        "artifact_run_claims": sorted(
            artifact_run_claims,
            key=lambda item: item["id"],
        ),
        "artifact_run_receipts": sorted(
            artifact_run_receipts,
            key=lambda item: item["id"],
        ),
        "scoring_claims": sorted(
            scoring_claims,
            key=lambda item: item["id"],
        ),
        # Events were read in canonical stream order.  Do not substitute wall
        # clock order: reconciled_at may move backward without changing the
        # append-only first-terminal boundary.
        "scoring_reconciliations": scoring_reconciliations,
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


def require_performance_attempt_trace_consistency(
    connection: sqlite3.Connection,
    *,
    attempt_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Require one session's productive trace to match immutable events.

    A target attempt cannot be validated in isolation from earlier attempts in
    the same session because their terminal actions establish whether its start
    was legal.  Unrelated sessions do not participate in that boundary and are
    intentionally excluded from this operational guard.  Aggregate reports,
    replay, and integrity continue to use the learner-wide derivation below.
    """

    _require_id(attempt_id, "attempt_id")
    attempt = connection.execute(
        """SELECT session_id FROM performance_attempts WHERE id=?""",
        (attempt_id,),
    ).fetchone()
    if attempt is None:
        raise ValidationError(
            f"Performance attempt {attempt_id} has no projection."
        )
    try:
        derived, _checkpoints = derive_performance_projections(
            connection,
            attempt_id=attempt_id,
            trace_only=True,
        )
        stored = performance_projection_snapshot(
            connection,
            session_id=attempt["session_id"],
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValidationError(
            "Performance attempt trace cannot be trusted against immutable "
            f"events: {exc}"
        ) from exc
    for name in (
        "attempts",
        "actions",
        "artifact_run_claims",
        "artifact_run_receipts",
    ):
        if canonical_json(stored[name]) != canonical_json(derived[name]):
            raise ValidationError(
                "Stored performance "
                f"{name.replace('_', ' ')} differ from immutable "
                "event derivation for the attempt session."
            )
    if not any(row["id"] == attempt_id for row in derived["attempts"]):
        raise ValidationError(
            f"Performance attempt {attempt_id} has no immutable start event."
        )
    return derived


def require_performance_projection_consistency(
    connection: sqlite3.Connection,
    *,
    learner_id: str,
    trace_only: bool = False,
    comparison_names: tuple[str, ...] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Require stored productive projections to match immutable events.

    Productive evidence is shadow-only, but it is still learner-facing and
    operational.  Read and append paths therefore cannot defer this boundary to
    an optional integrity command: malformed or manually inserted attempt,
    action, scoring, evaluation, or bundle rows must fail closed before they are
    reported or extended.  ``trace_only`` validates attempts, actions, and the
    complete artifact-run claim/receipt lifecycle (including system
    artifact-check authority) while leaving scoring-specific recovery paths to
    their dedicated projection guards.
    """

    _require_id(learner_id, "learner_id")
    if type(trace_only) is not bool:
        raise ValidationError("trace_only must be a boolean.")
    allowed_projection_names = frozenset(
        {
            "attempts",
            "actions",
            "artifact_run_claims",
            "artifact_run_receipts",
            "scoring_claims",
            "scoring_reconciliations",
            "evaluations",
            "bundles",
        }
    )
    if comparison_names is not None and (
        type(comparison_names) is not tuple
        or not comparison_names
        or len(set(comparison_names)) != len(comparison_names)
        or any(
            type(name) is not str or name not in allowed_projection_names
            for name in comparison_names
        )
    ):
        raise ValidationError(
            "comparison_names must be a non-empty tuple of unique performance "
            "projection names."
        )
    trace_projection_names = (
        "attempts",
        "actions",
        "artifact_run_claims",
        "artifact_run_receipts",
    )
    if trace_only and comparison_names is not None and any(
        name not in trace_projection_names for name in comparison_names
    ):
        raise ValidationError(
            "trace-only derivation can compare only attempts, actions, "
            "artifact-run claims, and artifact-run receipts."
        )
    try:
        derived, _checkpoints = derive_performance_projections(
            connection,
            learner_id=learner_id,
            trace_only=trace_only,
        )
        stored = performance_projection_snapshot(
            connection,
            learner_id=learner_id,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValidationError(
            "Performance projections cannot be trusted against immutable "
            f"events: {exc}"
        ) from exc
    projection_names = comparison_names or (
        trace_projection_names
        if trace_only
        else tuple(sorted(allowed_projection_names))
    )
    for name in projection_names:
        if canonical_json(stored[name]) != canonical_json(derived[name]):
            raise ValidationError(
                f"Stored performance {name.replace('_', ' ')} differ from "
                "immutable event derivation."
            )
    return derived


def rebuild_performance_projections(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    """Replace mutable shadow projections from events on a disposable copy."""

    snapshot, checkpoints = derive_performance_projections(connection)
    trigger_rows = connection.execute(
        """SELECT name, sql FROM sqlite_master
           WHERE type='trigger' AND tbl_name IN (
               'performance_attempts', 'performance_actions',
               'performance_artifact_run_claims',
               'performance_artifact_run_receipts',
               'performance_scoring_claims',
               'performance_scoring_reconciliations', 'task_evaluations',
               'shadow_evidence_bundles'
           ) ORDER BY name"""
    ).fetchall()
    for trigger in trigger_rows:
        escaped = trigger["name"].replace('"', '""')
        connection.execute(f'DROP TRIGGER "{escaped}"')
    connection.execute("DELETE FROM shadow_evidence_bundles")
    connection.execute("DELETE FROM task_evaluations")
    connection.execute("DELETE FROM performance_scoring_reconciliations")
    connection.execute("DELETE FROM performance_artifact_run_receipts")
    connection.execute("DELETE FROM performance_scoring_claims")
    connection.execute("DELETE FROM performance_artifact_run_claims")
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
        """INSERT INTO performance_artifact_run_claims(
               id, event_id, idempotency_key, attempt_id, session_id,
               session_revision, artifact_action_id, through_sequence,
               task_release_id, task_id, task_version, task_digest,
               artifact_digest, artifact_kind, artifact_manifest_digest,
               check_set_id, check_set_manifest_digest, runner_id,
               runner_version, request_json, request_digest, binding_json,
               binding_digest, command_hash, claimed_at
           ) VALUES (
               :id, :event_id, :idempotency_key, :attempt_id, :session_id,
               :session_revision, :artifact_action_id, :through_sequence,
               :task_release_id, :task_id, :task_version, :task_digest,
               :artifact_digest, :artifact_kind, :artifact_manifest_digest,
               :check_set_id, :check_set_manifest_digest, :runner_id,
               :runner_version, :request_json, :request_digest, :binding_json,
               :binding_digest, :command_hash, :claimed_at
           )""",
        snapshot["artifact_run_claims"],
    )
    connection.executemany(
        """INSERT INTO performance_artifact_run_receipts(
               id, event_id, claim_id, check_action_id, outcome,
               result_json, result_digest, receipt_json, receipt_digest,
               started_at, completed_at
           ) VALUES (
               :id, :event_id, :claim_id, :check_action_id, :outcome,
               :result_json, :result_digest, :receipt_json, :receipt_digest,
               :started_at, :completed_at
           )""",
        snapshot["artifact_run_receipts"],
    )
    connection.executemany(
        """INSERT INTO performance_scoring_claims(
               id, event_id, claim_schema_version, idempotency_key,
               attempt_id, evaluation_id, through_sequence, provider_id,
               provider_version, action_trace_digest,
               scoring_request_digest, provider_binding_digest,
               provider_operation_digest, command_hash, claimed_at
           ) VALUES (
               :id, :event_id, :claim_schema_version, :idempotency_key,
               :attempt_id, :evaluation_id, :through_sequence, :provider_id,
               :provider_version, :action_trace_digest,
               :scoring_request_digest, :provider_binding_digest,
               :provider_operation_digest, :command_hash, :claimed_at
           )""",
        snapshot["scoring_claims"],
    )
    connection.executemany(
        """INSERT INTO performance_scoring_reconciliations(
               id, event_id, idempotency_key, claim_id, attempt_id,
               evaluation_id, outcome, scoring_request_digest,
               provider_binding_digest, provider_operation_digest,
               reconciler_id, reconciler_version,
               reconciliation_binding_digest, receipt_json, receipt_digest,
               normalized_result_digest, reconciled_at, command_hash
           ) VALUES (
               :id, :event_id, :idempotency_key, :claim_id, :attempt_id,
               :evaluation_id, :outcome, :scoring_request_digest,
               :provider_binding_digest, :provider_operation_digest,
               :reconciler_id, :reconciler_version,
               :reconciliation_binding_digest, :receipt_json, :receipt_digest,
               :normalized_result_digest, :reconciled_at, :command_hash
           )""",
        snapshot["scoring_reconciliations"],
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
        "artifact_run_claim_count": len(snapshot["artifact_run_claims"]),
        "artifact_run_receipt_count": len(
            snapshot["artifact_run_receipts"]
        ),
        "scoring_claim_count": len(snapshot["scoring_claims"]),
        "scoring_reconciliation_count": len(
            snapshot["scoring_reconciliations"]
        ),
        "evaluation_count": len(snapshot["evaluations"]),
        "bundle_count": len(snapshot["bundles"]),
        "projection_hash": canonical_digest(snapshot),
    }


__all__ = [
    "MAX_TASK_RELEASE_BYTES",
    "PERFORMANCE_ARTIFACT_RUN_SCHEMA_VERSION",
    "PERFORMANCE_EVENT_SCHEMA_VERSION",
    "SERVICEABLE_TASK_STATUSES",
    "SYNTHETIC_TASK_LAB_RELEASE_SCHEMA_VERSION",
    "TASK_RELEASE_SCHEMA_VERSION",
    "TASK_STATUSES",
    "PerformanceLedger",
    "PerformanceTaskRelease",
    "OperationalArtifactRunReceipt",
    "SyntheticTaskLabDeclaration",
    "TaskReleaseReview",
    "derive_performance_projections",
    "performance_integrity_errors",
    "performance_projection_snapshot",
    "read_task_release",
    "rebuild_performance_projections",
    "require_performance_projection_consistency",
]
