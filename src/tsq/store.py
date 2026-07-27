# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Any, Iterator, Sequence

from .evidence import (
    ActionKind,
    ActionPhase,
    LearningAction,
    action_trace_digest,
    canonical_digest,
    canonical_json,
    summarize_actions,
)
from .errors import ConflictError, NotFoundError, ValidationError
from .event_contracts import (
    ATTEMPT_OUTCOME_COMPLETE_FIELDS,
    ATTEMPT_OUTCOME_OPTIONAL_FIELDS,
    PROJECTION_FIELDS_BY_SCHEMA,
    PROJECTION_METADATA_FIELDS,
    PROJECTION_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM,
    QUESTION_SELECTED_BASE_FIELDS,
    QUESTION_SELECTED_METADATA_FIELDS,
    QUESTION_SELECTED_OBJECTIVE_FIELDS,
    REMEDIATION_TRANSITION_FIELDS,
    REMEDIATION_TRANSITION_FIELDS_V1,
    REMEDIATION_TRANSITION_METADATA_FIELDS,
    REMEDIATION_TRANSITION_OBJECTIVE_FIELDS,
    RESPONSE_FIELDS,
    RESPONSE_METADATA_FIELDS,
    RESPONSE_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM,
    same_json_value,
)
from .graph import KnowledgeGraph
from .inference import (
    MISCONCEPTION_ALGORITHM_METADATA_KEY,
    MISCONCEPTION_ALGORITHM_VERSION,
    response_window,
)
from .models import (
    MAX_HINT_COUNT,
    MAX_REMEDIATION_DEPTH,
    MAX_RESPONSE_MS,
    CandidateScore,
    Concept,
    ConceptEdge,
    ConceptWeight,
    Domain,
    LearningObjective,
    Misconception,
    MisconceptionBelief,
    ObjectiveEdge,
    ObjectiveOperation,
    ObjectiveState,
    Option,
    Presentation,
    Question,
    QuestionKind,
    QuestionStatus,
    RelationType,
    SessionPhase,
    SkillState,
    Source,
    Topic,
)
from .objective_posterior import (
    OBJECTIVE_POSTERIOR_V1_IDENTITY,
    ObjectivePosterior,
    ObjectivePosteriorError,
    decode_objective_posterior,
    posterior_digest,
)
from .quality import audit_corpus
from .provenance import (
    generated_question_runtime_safe,
    legacy_question_identity_payload,
    legacy_unattested_member_compatible,
    question_provenance_issues,
)
from .versions import (
    AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS,
    BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION,
    COMPLETE_TRANSITION_OUTCOME_MODEL_VERSIONS,
    OBJECTIVE_GAUSSIAN_MODEL_VERSION,
    OBJECTIVE_GRID_MODEL_VERSIONS,
    OBJECTIVE_GRID_V6_MODEL_VERSION,
    OBJECTIVE_GRID_V7_MODEL_VERSION,
    OBJECTIVE_GRID_V8_MODEL_VERSION,
    PROJECTION_HASH_VERSION_BY_EVENT_SCHEMA,
    PROJECTION_MODEL_VERSIONS_BY_EVENT_SCHEMA,
    SUPPORTED_QUESTION_SELECTED_EVENT_SCHEMA_VERSIONS,
    SUPPORTED_MODEL_VERSIONS,
    question_selected_schema_for,
)


SCHEMA_VERSION = 19
PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX = (
    "performance-score-claim:v1:"
)
PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX = (
    "performance-score-reconcile:v1:"
)
LEGACY_UNREVIEWED_GENERATED_REVOCATION_REASON = (
    "Unreviewed generated question was active in a historical corpus release."
)
LEGACY_UNREVIEWED_GENERATED_REVOCATION_KEY_PREFIX = (
    "corpus-safety:legacy-unreviewed-generated:v1:"
)
HISTORICAL_GENERATED_EVIDENCE_POLICY = (
    "historical-generated-evidence-v1"
)
HISTORICAL_GENERATED_EVIDENCE_KEY_PREFIX = (
    "learner-safety:historical-generated-evidence:v1:"
)
CURRENT_SCHEMA_TABLES = frozenset(
    {
        "attempts",
        "concept_edges",
        "concepts",
        "corpus_releases",
        "decisions",
        "events",
        "generation_job_runs",
        "generation_jobs",
        "item_reviews",
        "item_stats",
        "learner_objective_families",
        "learner_skill_families",
        "learners",
        "learning_actions",
        "learning_artifacts",
        "learning_objectives",
        "meta",
        "misconception_beliefs",
        "misconceptions",
        "objective_grid_states",
        "objective_states",
        "options",
        "performance_actions",
        "performance_attempts",
        "performance_scoring_claims",
        "performance_scoring_reconciliations",
        "policy_shadow_evaluations",
        "performance_task_releases",
        "performance_tasks",
        "question_concepts",
        "question_revocations",
        "question_sources",
        "questions",
        "release_concepts",
        "release_domains",
        "release_edges",
        "release_learning_objectives",
        "release_misconceptions",
        "release_objective_edges",
        "release_objective_graphs",
        "release_option_objectives",
        "release_performance_tasks",
        "release_question_objectives",
        "release_question_topics",
        "release_questions",
        "release_sources",
        "release_topic_concepts",
        "release_topics",
        "sessions",
        "skill_states",
        "shadow_evidence_bundles",
        "sources",
        "stream_heads",
        "task_evaluations",
    }
)
# Compatibility name retained for code that previously imported store's
# singular current-grid identifier. Immutable dispatch uses explicit versions.
OBJECTIVE_GRID_MODEL_VERSION = OBJECTIVE_GRID_V8_MODEL_VERSION


def performance_scoring_claim_event_key(command_hash: str) -> str:
    """Return the reserved event idempotency key for a scoring admission."""

    if (
        type(command_hash) is not str
        or len(command_hash) != 64
        or any(character not in "0123456789abcdef" for character in command_hash)
    ):
        raise ValidationError(
            "Performance scoring claim command hash must be lowercase SHA-256."
        )
    return PERFORMANCE_SCORING_CLAIM_EVENT_KEY_PREFIX + command_hash


def performance_scoring_claim_payload(
    *,
    claim_id: str,
    caller_idempotency_key: str | None,
    attempt_id: str,
    evaluation_id: str,
    through_sequence: int,
    provider_id: str,
    provider_version: str,
    action_trace_digest_value: str,
    command_hash: str,
    claimed_at: str,
) -> dict[str, Any]:
    """Return the closed event payload for one provider-callback admission."""

    return {
        "claim_id": claim_id,
        "caller_idempotency_key": caller_idempotency_key,
        "attempt_id": attempt_id,
        "evaluation_id": evaluation_id,
        "through_sequence": through_sequence,
        "provider_id": provider_id,
        "provider_version": provider_version,
        "action_trace_digest": action_trace_digest_value,
        "command_hash": command_hash,
        "claimed_at": claimed_at,
    }


def performance_scoring_claim_v2_payload(
    *,
    claim_id: str,
    caller_idempotency_key: str | None,
    attempt_id: str,
    evaluation_id: str,
    through_sequence: int,
    provider_id: str,
    provider_version: str,
    action_trace_digest_value: str,
    command_hash: str,
    claimed_at: str,
    scoring_request_digest: str,
    provider_binding_digest: str,
    provider_operation_digest: str,
    provider: dict[str, Any],
) -> dict[str, Any]:
    """Return the closed v2 scoring-admission payload.

    The provider snapshot is committed only to immutable event history.  The
    projection stores the exact identity and the three digests needed to bind
    reconciliation without duplicating mutable registry terms.
    """

    return {
        **performance_scoring_claim_payload(
            claim_id=claim_id,
            caller_idempotency_key=caller_idempotency_key,
            attempt_id=attempt_id,
            evaluation_id=evaluation_id,
            through_sequence=through_sequence,
            provider_id=provider_id,
            provider_version=provider_version,
            action_trace_digest_value=action_trace_digest_value,
            command_hash=command_hash,
            claimed_at=claimed_at,
        ),
        "scoring_request_digest": scoring_request_digest,
        "provider_binding_digest": provider_binding_digest,
        "provider_operation_digest": provider_operation_digest,
        "provider": provider,
    }


def performance_scoring_reconciliation_event_key(command_hash: str) -> str:
    """Return the reserved key for one reconciliation observation command."""

    if (
        type(command_hash) is not str
        or len(command_hash) != 64
        or any(character not in "0123456789abcdef" for character in command_hash)
    ):
        raise ValidationError(
            "Performance scoring reconciliation command hash must be "
            "lowercase SHA-256."
        )
    return PERFORMANCE_SCORING_RECONCILIATION_EVENT_KEY_PREFIX + command_hash


def performance_scoring_reconciliation_payload(
    *,
    reconciliation_id: str,
    caller_idempotency_key: str | None,
    claim_id: str,
    attempt_id: str,
    evaluation_id: str,
    outcome: str,
    scoring_request_digest: str,
    provider_binding_digest: str,
    provider_operation_digest: str,
    reconciler_id: str,
    reconciler_version: str,
    reconciliation_binding_digest: str,
    receipt: dict[str, Any],
    receipt_digest: str,
    normalized_result_digest: str | None,
    reconciled_at: str,
    command_hash: str,
    reconciler: dict[str, Any],
) -> dict[str, Any]:
    """Return the closed event payload for one reconciliation observation."""

    return {
        "reconciliation_id": reconciliation_id,
        "caller_idempotency_key": caller_idempotency_key,
        "claim_id": claim_id,
        "attempt_id": attempt_id,
        "evaluation_id": evaluation_id,
        "outcome": outcome,
        "scoring_request_digest": scoring_request_digest,
        "provider_binding_digest": provider_binding_digest,
        "provider_operation_digest": provider_operation_digest,
        "reconciler_id": reconciler_id,
        "reconciler_version": reconciler_version,
        "reconciliation_binding_digest": reconciliation_binding_digest,
        "receipt": receipt,
        "receipt_digest": receipt_digest,
        "normalized_result_digest": normalized_result_digest,
        "reconciled_at": reconciled_at,
        "command_hash": command_hash,
        "reconciler": reconciler,
    }


def question_runtime_activation_safe(
    question: Question,
    *,
    status: str | None = None,
) -> bool:
    """Return whether a stored question may contribute new learner evidence."""

    return generated_question_runtime_safe(
        question.provenance,
        status=status or question.status.value,
    )

OBJECTIVE_STATE_WITH_GRID_SELECT = """SELECT
    state.learner_id,
    state.objective_id,
    state.mean,
    state.variance,
    state.stability_hours,
    state.exposures,
    state.last_seen_at,
    state.next_review_at,
    state.evidence_mass,
    state.as_of_event_id,
    state.model_version,
    grid.learner_id AS grid_learner_id,
    grid.objective_id AS grid_objective_id,
    grid.posterior_schema_version AS grid_posterior_schema_version,
    grid.algorithm AS grid_algorithm,
    grid.grid_id AS grid_grid_id,
    grid.codec AS grid_codec,
    grid.posterior_blob AS grid_posterior_blob,
    grid.posterior_sha256 AS grid_posterior_sha256,
    grid.mean AS grid_mean,
    grid.variance AS grid_variance,
    grid.mastery_probability AS grid_mastery_probability,
    grid.expected_competence AS grid_expected_competence,
    grid.edge_mass AS grid_edge_mass,
    grid.mastery_probability_error_bound
        AS grid_mastery_probability_error_bound,
    grid.evidence_mass AS grid_evidence_mass,
    grid.acquisition_mass AS grid_acquisition_mass,
    grid.as_of_event_id AS grid_as_of_event_id,
    grid.model_version AS grid_model_version
FROM objective_states state
LEFT JOIN objective_grid_states grid
  ON grid.learner_id = state.learner_id
 AND grid.objective_id = state.objective_id"""

# Candidate retrieval deliberately has a separate, compact SQL kernel.  Keeping
# it module-level lets the large-bank benchmark inspect the exact production
# query instead of maintaining a subtly different copy.
CANDIDATE_POOL_SQL = """SELECT q.id,
       q.family_id,
       CASE
           WHEN qc.concept_id = ? OR q.id IN (
               SELECT focused.question_id
               FROM options focused INDEXED BY idx_options_misconception_question
               WHERE focused.misconception_id = ?
           ) THEN 0 ELSE 1
       END AS focus_rank,
       COALESCE(personal.exposures, 0) AS personal_exposures,
       ABS(q.difficulty - ?) AS difficulty_distance
FROM requested_scope scope
CROSS JOIN question_concepts qc INDEXED BY idx_question_concepts_primary_scope
JOIN release_questions rq
  ON rq.question_id = qc.question_id AND rq.release_id = ?
JOIN questions q ON q.id = qc.question_id
LEFT JOIN (
    SELECT presented_question.family_id, COUNT(*) AS exposures
    FROM decisions presented INDEXED BY idx_decisions_learner_question
    JOIN questions presented_question ON presented_question.id = presented.question_id
    WHERE presented.learner_id = ?
    GROUP BY presented_question.family_id
) personal ON personal.family_id = q.family_id
WHERE qc.concept_id = scope.id
  AND qc.role = 'primary'
  AND rq.status IN (?, ?)
  AND NOT EXISTS (
      SELECT 1 FROM question_revocations revoked
      WHERE revoked.question_id = q.id
  )
ORDER BY focus_rank, personal_exposures, difficulty_distance,
         q.discrimination DESC, q.id
LIMIT ?"""


# Most production banks have one active item per family in a local difficulty
# neighborhood, so the compact query above is the steady-state fast path.  If
# its bounded result contains siblings, this fallback guarantees that prolific
# families cannot hide independent evidence paths beyond the cutoff.
FAMILY_DIVERSE_CANDIDATE_POOL_SQL = """WITH base_candidates AS (
    SELECT q.id,
           q.family_id,
           q.discrimination,
           CASE
               WHEN qc.concept_id = ? OR q.id IN (
                   SELECT focused.question_id
                   FROM options focused INDEXED BY idx_options_misconception_question
                   WHERE focused.misconception_id = ?
               ) THEN 0 ELSE 1
           END AS focus_rank,
           COALESCE(personal.exposures, 0) AS personal_exposures,
           ABS(q.difficulty - ?) AS difficulty_distance
    FROM requested_scope scope
    CROSS JOIN question_concepts qc INDEXED BY idx_question_concepts_primary_scope
    JOIN release_questions rq
      ON rq.question_id = qc.question_id AND rq.release_id = ?
    JOIN questions q ON q.id = qc.question_id
    LEFT JOIN (
        SELECT presented_question.family_id, COUNT(*) AS exposures
        FROM decisions presented INDEXED BY idx_decisions_learner_question
        JOIN questions presented_question
          ON presented_question.id = presented.question_id
        WHERE presented.learner_id = ?
        GROUP BY presented_question.family_id
    ) personal ON personal.family_id = q.family_id
    WHERE qc.concept_id = scope.id
      AND qc.role = 'primary'
      AND rq.status IN (?, ?)
      AND NOT EXISTS (
          SELECT 1 FROM question_revocations revoked
          WHERE revoked.question_id = q.id
      )
), ranked_candidates AS (
    SELECT base_candidates.*,
           ROW_NUMBER() OVER (
               PARTITION BY family_id
               ORDER BY focus_rank, personal_exposures, difficulty_distance,
                        discrimination DESC, id
           ) AS family_rank
    FROM base_candidates
)
SELECT id, family_id, focus_rank, personal_exposures, difficulty_distance
FROM ranked_candidates
ORDER BY family_rank, focus_rank, personal_exposures, difficulty_distance,
         discrimination DESC, id
LIMIT ?"""


# Focused objective retrieval bypasses the broad-concept cutoff entirely. An
# objective may be assessed by questions whose primary graph concept is one of
# its declared supporting concepts, and a large unrelated concept pool must
# never hide an exact repair or verification family.
OBJECTIVE_CANDIDATE_POOL_SQL = """WITH base_candidates AS (
    SELECT q.id,
           q.family_id,
           q.discrimination,
           CASE WHEN EXISTS (
               SELECT 1
               FROM options focused
               JOIN release_option_objectives diagnostic
                 ON diagnostic.release_id = mapping.release_id
                AND diagnostic.question_id = focused.question_id
                AND diagnostic.option_id = focused.option_id
               WHERE focused.question_id = q.id
                 AND focused.misconception_id = ?
                 AND diagnostic.objective_id = mapping.objective_id
           ) THEN 0 ELSE 1 END AS focus_rank,
           COALESCE(personal.exposures, 0) AS personal_exposures,
           ABS(q.difficulty - ?) AS difficulty_distance
    FROM release_question_objectives mapping
    JOIN release_questions release_question
      ON release_question.release_id = mapping.release_id
     AND release_question.question_id = mapping.question_id
    JOIN questions q ON q.id = mapping.question_id
    LEFT JOIN (
        SELECT presented_question.family_id, COUNT(*) AS exposures
        FROM decisions presented INDEXED BY idx_decisions_learner_question
        JOIN questions presented_question
          ON presented_question.id = presented.question_id
        WHERE presented.learner_id = ?
        GROUP BY presented_question.family_id
    ) personal ON personal.family_id = q.family_id
    WHERE mapping.release_id = ?
      AND mapping.objective_id = ?
      AND release_question.status IN (?, ?)
      AND NOT EXISTS (
          SELECT 1 FROM question_revocations revoked
          WHERE revoked.question_id = q.id
      )
), ranked_candidates AS (
    SELECT base_candidates.*,
           ROW_NUMBER() OVER (
               PARTITION BY family_id
               ORDER BY focus_rank, personal_exposures, difficulty_distance,
                        discrimination DESC, id
           ) AS family_rank
    FROM base_candidates
)
SELECT id, family_id, focus_rank, personal_exposures, difficulty_distance
FROM ranked_candidates
ORDER BY family_rank, focus_rank, personal_exposures, difficulty_distance,
         discrimination DESC, id
LIMIT ?"""


def new_id(prefix: str) -> str:
    """Return a time-sortable, locally generated identifier."""
    milliseconds = int(time.time() * 1000)
    return f"{prefix}_{milliseconds:013x}{secrets.token_hex(10)}"


def to_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("Timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat()


def from_timestamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def question_content_hash(question: Question) -> str:
    immutable = {
        "id": question.id,
        "version": question.version,
        "family_id": question.family_id,
        "stem": question.stem,
        "kind": question.kind.value,
        "difficulty": float(question.difficulty),
        "discrimination": float(question.discrimination),
        "guess_rate": float(question.guess_rate),
        "slip_rate": float(question.slip_rate),
        "concepts": [
            {
                "concept_id": item.concept_id,
                "weight": float(item.weight),
                "role": item.role,
            }
            for item in question.concepts
        ],
        "options": [
            {
                "id": option.id,
                "text": option.text,
                "correct": option.correct,
                "rationale": option.rationale,
                "misconception_id": option.misconception_id,
            }
            for option in question.options
        ],
        "source_ids": list(question.source_ids),
        "provenance": question.provenance,
        "tags": list(question.tags),
        "revision_of": question.revision_of,
    }
    encoded = json.dumps(
        immutable, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _legacy_question_identity(question: Question) -> dict[str, object]:
    return legacy_question_identity_payload(
        question_id=question.id,
        version=question.version,
        family_id=question.family_id,
        stem=question.stem,
        kind=question.kind.value,
        difficulty=question.difficulty,
        discrimination=question.discrimination,
        guess_rate=question.guess_rate,
        slip_rate=question.slip_rate,
        concepts=(
            (mapping.concept_id, mapping.weight, mapping.role.value)
            for mapping in question.concepts
        ),
        options=(
            (
                option.id,
                option.text,
                option.correct,
                option.rationale,
                option.misconception_id,
                option.diagnostic_objective_id,
            )
            for option in question.options
        ),
        source_ids=question.source_ids,
        provenance=question.provenance,
        tags=question.tags,
        revision_of=question.revision_of,
        learning_objective_id=question.objective_id,
    )


def _legacy_unreviewed_generated_revocation_key(question_id: str) -> str:
    identity = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    return LEGACY_UNREVIEWED_GENERATED_REVOCATION_KEY_PREFIX + identity


def _historical_generated_evidence_key(attempt_id: str) -> str:
    identity = hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()
    return HISTORICAL_GENERATED_EVIDENCE_KEY_PREFIX + identity


def _content_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def concept_content_hash(concept: Concept) -> str:
    return _content_hash(
        {
            "id": concept.id,
            "name": concept.name,
            "description": concept.description,
            "domain": concept.domain,
            "prior_mastery": float(concept.prior_mastery),
        }
    )


def objective_content_hash(objective: LearningObjective) -> str:
    return _content_hash(
        {
            "id": objective.id,
            "name": objective.name,
            "description": objective.description,
            "primary_concept_id": objective.primary_concept_id,
            "supporting_concept_ids": list(objective.supporting_concept_ids),
            "operation": objective.operation.value,
            "evidence_type": objective.evidence_type,
            "prior_mastery": float(objective.prior_mastery),
        }
    )


def domain_content_hash(domain: Domain) -> str:
    return _content_hash(
        {
            "id": domain.id,
            "name": domain.name,
            "description": domain.description,
            "sort_order": domain.sort_order,
        }
    )


def topic_content_hash(topic: Topic) -> str:
    return _content_hash(
        {
            "id": topic.id,
            "domain_id": topic.domain_id,
            "name": topic.name,
            "description": topic.description,
            "concept_ids": list(topic.concept_ids),
            "parent_id": topic.parent_id,
            "related_topic_ids": list(topic.related_topic_ids),
            "sort_order": topic.sort_order,
        }
    )


def misconception_content_hash(misconception: Misconception) -> str:
    return _content_hash(
        {
            "id": misconception.id,
            "concept_id": misconception.concept_id,
            "name": misconception.name,
            "description": misconception.description,
        }
    )


def source_content_hash(source: Source) -> str:
    return _content_hash(
        {
            "id": source.id,
            "title": source.title,
            "uri": source.uri,
            "license": source.license,
            "metadata": source.metadata,
        }
    )


DDL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    domain TEXT NOT NULL,
    prior_mastery REAL NOT NULL CHECK(prior_mastery > 0 AND prior_mastery < 1)
);

CREATE TABLE IF NOT EXISTS learning_objectives (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    primary_concept_id TEXT NOT NULL REFERENCES concepts(id),
    supporting_concept_ids_json TEXT NOT NULL CHECK(
        json_valid(supporting_concept_ids_json)
        AND json_type(supporting_concept_ids_json) = 'array'
    ),
    operation TEXT NOT NULL,
    evidence_type TEXT NOT NULL CHECK(evidence_type = 'selected_response'),
    prior_mastery REAL NOT NULL CHECK(prior_mastery > 0 AND prior_mastery < 1)
);

CREATE TABLE IF NOT EXISTS concept_edges (
    source_id TEXT NOT NULL REFERENCES concepts(id),
    target_id TEXT NOT NULL REFERENCES concepts(id),
    relation TEXT NOT NULL,
    weight REAL NOT NULL CHECK(weight > 0),
    PRIMARY KEY(source_id, target_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_edges_target ON concept_edges(target_id, relation);

CREATE TABLE IF NOT EXISTS misconceptions (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    concept_id TEXT NOT NULL REFERENCES concepts(id),
    name TEXT NOT NULL,
    description TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_misconceptions_concept ON misconceptions(concept_id);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    title TEXT NOT NULL,
    uri TEXT,
    license TEXT,
    metadata_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL CHECK(version > 0),
    content_hash TEXT NOT NULL,
    family_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stem TEXT NOT NULL,
    kind TEXT NOT NULL,
    difficulty REAL NOT NULL,
    discrimination REAL NOT NULL,
    guess_rate REAL NOT NULL,
    slip_rate REAL NOT NULL,
    provenance_json TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    revision_of TEXT REFERENCES questions(id),
    imported_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status);
CREATE INDEX IF NOT EXISTS idx_questions_family ON questions(family_id);

CREATE TABLE IF NOT EXISTS question_concepts (
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    concept_id TEXT NOT NULL REFERENCES concepts(id),
    weight REAL NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY(question_id, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_question_concepts_concept ON question_concepts(concept_id, question_id);
CREATE INDEX IF NOT EXISTS idx_question_concepts_primary_scope
ON question_concepts(concept_id, question_id) WHERE role = 'primary';

CREATE TABLE IF NOT EXISTS options (
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    option_id TEXT NOT NULL,
    text TEXT NOT NULL,
    is_correct INTEGER NOT NULL CHECK(is_correct IN (0, 1)),
    rationale TEXT NOT NULL,
    misconception_id TEXT REFERENCES misconceptions(id),
    PRIMARY KEY(question_id, option_id)
);

CREATE INDEX IF NOT EXISTS idx_options_misconception_question
ON options(misconception_id, question_id) WHERE misconception_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS question_sources (
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    source_id TEXT NOT NULL REFERENCES sources(id),
    PRIMARY KEY(question_id, source_id)
);

CREATE TABLE IF NOT EXISTS learners (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL REFERENCES learners(id),
    root_concept_id TEXT NOT NULL REFERENCES concepts(id),
    corpus_release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    mode TEXT NOT NULL,
    phase TEXT NOT NULL,
    focus_concept_id TEXT REFERENCES concepts(id),
    focus_misconception_id TEXT REFERENCES misconceptions(id),
    focus_objective_id TEXT REFERENCES learning_objectives(id),
    remediation_depth INTEGER NOT NULL DEFAULT 0,
    remediation_path_json TEXT NOT NULL DEFAULT '[]',
    revision INTEGER NOT NULL DEFAULT 0,
    rng_seed INTEGER NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    recent_families_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    topic_id TEXT,
    exploration_mode TEXT NOT NULL DEFAULT 'off'
);

CREATE INDEX IF NOT EXISTS idx_sessions_learner ON sessions(learner_id, created_at);

CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    stream_id TEXT NOT NULL,
    stream_version INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    learner_id TEXT,
    session_id TEXT,
    correlation_id TEXT NOT NULL,
    causation_id TEXT,
    idempotency_key TEXT UNIQUE,
    payload_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    previous_hash TEXT,
    payload_hash TEXT NOT NULL,
    UNIQUE(stream_id, stream_version)
);

CREATE INDEX IF NOT EXISTS idx_events_stream ON events(stream_id, stream_version);
CREATE INDEX IF NOT EXISTS idx_events_learner ON events(learner_id, recorded_at);

CREATE TABLE IF NOT EXISTS stream_heads (
    stream_id TEXT PRIMARY KEY,
    stream_version INTEGER NOT NULL,
    payload_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_releases (
    id TEXT PRIMARY KEY,
    bundle_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    sealed_at TEXT
);

CREATE TABLE IF NOT EXISTS release_edges (
    release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    source_id TEXT NOT NULL REFERENCES concepts(id),
    target_id TEXT NOT NULL REFERENCES concepts(id),
    relation TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY(release_id, source_id, target_id, relation)
);

CREATE TABLE IF NOT EXISTS release_concepts (
    release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    concept_id TEXT NOT NULL REFERENCES concepts(id),
    PRIMARY KEY(release_id, concept_id)
);

CREATE TABLE IF NOT EXISTS release_misconceptions (
    release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    misconception_id TEXT NOT NULL REFERENCES misconceptions(id),
    PRIMARY KEY(release_id, misconception_id)
);

CREATE TABLE IF NOT EXISTS release_sources (
    release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    source_id TEXT NOT NULL REFERENCES sources(id),
    PRIMARY KEY(release_id, source_id)
);

CREATE TABLE IF NOT EXISTS release_questions (
    release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    question_id TEXT NOT NULL REFERENCES questions(id),
    status TEXT NOT NULL,
    evidence_weight REAL NOT NULL,
    PRIMARY KEY(release_id, question_id)
);

CREATE INDEX IF NOT EXISTS idx_release_questions_question_release
ON release_questions(question_id, release_id);

CREATE TABLE IF NOT EXISTS release_learning_objectives (
    release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    objective_id TEXT NOT NULL REFERENCES learning_objectives(id),
    PRIMARY KEY(release_id, objective_id)
);

CREATE TABLE IF NOT EXISTS release_objective_graphs (
    release_id TEXT PRIMARY KEY REFERENCES corpus_releases(id),
    graph_version INTEGER NOT NULL CHECK(graph_version = 1)
);

CREATE TABLE IF NOT EXISTS release_objective_edges (
    release_id TEXT NOT NULL,
    edge_id TEXT NOT NULL,
    source_objective_id TEXT NOT NULL,
    target_objective_id TEXT NOT NULL,
    relation TEXT NOT NULL CHECK(relation IN ('prerequisite', 'requires')),
    weight REAL NOT NULL CHECK(weight > 0 AND weight <= 1),
    rationale TEXT NOT NULL CHECK(length(trim(rationale)) > 0),
    PRIMARY KEY(release_id, edge_id),
    UNIQUE(release_id, source_objective_id, target_objective_id),
    FOREIGN KEY(release_id, source_objective_id)
        REFERENCES release_learning_objectives(release_id, objective_id),
    FOREIGN KEY(release_id, target_objective_id)
        REFERENCES release_learning_objectives(release_id, objective_id),
    CHECK(source_objective_id != target_objective_id)
);

CREATE INDEX IF NOT EXISTS idx_release_objective_edges_target
ON release_objective_edges(release_id, target_objective_id, source_objective_id);

CREATE TABLE IF NOT EXISTS release_question_objectives (
    release_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    objective_id TEXT NOT NULL,
    PRIMARY KEY(release_id, question_id),
    FOREIGN KEY(release_id, question_id)
        REFERENCES release_questions(release_id, question_id),
    FOREIGN KEY(release_id, objective_id)
        REFERENCES release_learning_objectives(release_id, objective_id)
);

CREATE INDEX IF NOT EXISTS idx_release_question_objectives_objective
ON release_question_objectives(release_id, objective_id, question_id);

CREATE TABLE IF NOT EXISTS release_option_objectives (
    release_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    option_id TEXT NOT NULL,
    objective_id TEXT NOT NULL,
    PRIMARY KEY(release_id, question_id, option_id),
    FOREIGN KEY(release_id, question_id)
        REFERENCES release_questions(release_id, question_id),
    FOREIGN KEY(question_id, option_id)
        REFERENCES options(question_id, option_id),
    FOREIGN KEY(release_id, objective_id)
        REFERENCES release_learning_objectives(release_id, objective_id)
);

CREATE INDEX IF NOT EXISTS idx_release_option_objectives_objective
ON release_option_objectives(release_id, objective_id, question_id);

CREATE TABLE IF NOT EXISTS release_domains (
    release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    domain_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    sort_order INTEGER NOT NULL CHECK(sort_order >= 0),
    PRIMARY KEY(release_id, domain_id)
);

CREATE TABLE IF NOT EXISTS release_topics (
    release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    topic_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    domain_id TEXT NOT NULL,
    parent_topic_id TEXT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    related_topic_ids_json TEXT NOT NULL,
    sort_order INTEGER NOT NULL CHECK(sort_order >= 0),
    PRIMARY KEY(release_id, topic_id),
    FOREIGN KEY(release_id, domain_id)
        REFERENCES release_domains(release_id, domain_id),
    FOREIGN KEY(release_id, parent_topic_id)
        REFERENCES release_topics(release_id, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_release_topics_parent
ON release_topics(release_id, parent_topic_id, sort_order, topic_id);

CREATE TABLE IF NOT EXISTS release_topic_concepts (
    release_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK(position >= 0),
    PRIMARY KEY(release_id, topic_id, concept_id),
    UNIQUE(release_id, concept_id),
    FOREIGN KEY(release_id, topic_id)
        REFERENCES release_topics(release_id, topic_id),
    FOREIGN KEY(release_id, concept_id)
        REFERENCES release_concepts(release_id, concept_id)
);

CREATE INDEX IF NOT EXISTS idx_release_topic_concepts_concept
ON release_topic_concepts(release_id, concept_id, topic_id);

CREATE TABLE IF NOT EXISTS release_question_topics (
    release_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    relation TEXT NOT NULL CHECK(relation IN ('primary', 'cross')),
    PRIMARY KEY(release_id, question_id, topic_id),
    FOREIGN KEY(release_id, question_id)
        REFERENCES release_questions(release_id, question_id),
    FOREIGN KEY(release_id, topic_id)
        REFERENCES release_topics(release_id, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_release_question_topics_topic
ON release_question_topics(release_id, topic_id, relation, question_id);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    learner_id TEXT NOT NULL REFERENCES learners(id),
    question_id TEXT NOT NULL REFERENCES questions(id),
    question_objective_id TEXT REFERENCES learning_objectives(id),
    question_version INTEGER NOT NULL,
    question_content_hash TEXT NOT NULL,
    question_status TEXT NOT NULL,
    evidence_weight REAL NOT NULL,
    corpus_release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    session_revision INTEGER NOT NULL,
    learner_revision INTEGER NOT NULL,
    phase TEXT NOT NULL,
    focus_concept_id TEXT REFERENCES concepts(id),
    focus_misconception_id TEXT REFERENCES misconceptions(id),
    focus_objective_id TEXT REFERENCES learning_objectives(id),
    pedagogical_role TEXT NOT NULL,
    focus_valid INTEGER NOT NULL CHECK(focus_valid IN (0, 1)),
    policy_version TEXT NOT NULL,
    candidate_count INTEGER NOT NULL,
    candidate_digest TEXT NOT NULL,
    top_candidates_json TEXT NOT NULL,
    selected_score_json TEXT NOT NULL,
    propensity REAL NOT NULL,
    option_order_json TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL,
    consumed_at TEXT,
    invalidated_at TEXT,
    invalidation_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_decisions_pending ON decisions(session_id, consumed_at);
CREATE INDEX IF NOT EXISTS idx_decisions_learner_question
ON decisions(learner_id, question_id, created_at);

CREATE TABLE IF NOT EXISTS policy_shadow_evaluations (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    decision_id TEXT NOT NULL REFERENCES decisions(id),
    challenger_policy_version TEXT NOT NULL
        CHECK(length(trim(challenger_policy_version)) > 0),
    challenger_definition_digest TEXT NOT NULL CHECK(
        length(challenger_definition_digest) = 64
        AND challenger_definition_digest NOT GLOB '*[^0-9a-f]*'
    ),
    logging_policy_version TEXT NOT NULL
        CHECK(length(trim(logging_policy_version)) > 0),
    learner_model_version TEXT NOT NULL
        CHECK(length(trim(learner_model_version)) > 0),
    corpus_release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    candidate_count INTEGER NOT NULL CHECK(candidate_count > 0),
    candidate_digest TEXT NOT NULL CHECK(
        length(candidate_digest) = 64
        AND candidate_digest NOT GLOB '*[^0-9a-f]*'
    ),
    frontier_json TEXT NOT NULL CHECK(
        length(frontier_json) <= 1048576
        AND json_valid(frontier_json)
        AND json_type(frontier_json) = 'array'
        AND json_array_length(frontier_json) = min(5, candidate_count)
    ),
    frontier_digest TEXT NOT NULL CHECK(
        length(frontier_digest) = 64
        AND frontier_digest NOT GLOB '*[^0-9a-f]*'
    ),
    input_digest TEXT NOT NULL CHECK(
        length(input_digest) = 64
        AND input_digest NOT GLOB '*[^0-9a-f]*'
    ),
    output_digest TEXT NOT NULL CHECK(
        length(output_digest) = 64
        AND output_digest NOT GLOB '*[^0-9a-f]*'
    ),
    live_question_id TEXT NOT NULL REFERENCES questions(id),
    challenger_question_id TEXT NOT NULL REFERENCES questions(id),
    agreement INTEGER NOT NULL CHECK(
        agreement IN (0, 1)
        AND agreement = (live_question_id = challenger_question_id)
    ),
    evaluated_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    shadow_only INTEGER NOT NULL DEFAULT 1 CHECK(shadow_only = 1),
    selection_applied INTEGER NOT NULL DEFAULT 0 CHECK(selection_applied = 0),
    mastery_applied INTEGER NOT NULL DEFAULT 0 CHECK(mastery_applied = 0),
    UNIQUE(
        decision_id,
        challenger_policy_version,
        challenger_definition_digest
    )
);

CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    decision_id TEXT NOT NULL UNIQUE REFERENCES decisions(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    learner_id TEXT NOT NULL REFERENCES learners(id),
    question_id TEXT NOT NULL REFERENCES questions(id),
    question_version INTEGER NOT NULL,
    family_id TEXT NOT NULL,
    presented_order_json TEXT NOT NULL,
    selected_option_id TEXT,
    is_correct INTEGER NOT NULL CHECK(is_correct IN (0, 1)),
    confidence REAL,
    response_ms INTEGER,
    hint_count INTEGER NOT NULL DEFAULT 0,
    feedback_shown INTEGER NOT NULL CHECK(feedback_shown IN (0, 1)),
    answered_at TEXT NOT NULL,
    command_hash TEXT NOT NULL,
    outcome_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_attempts_learner ON attempts(learner_id, answered_at);
CREATE INDEX IF NOT EXISTS idx_attempts_learner_question
ON attempts(learner_id, question_id, family_id);
CREATE INDEX IF NOT EXISTS idx_attempts_learner_family
ON attempts(learner_id, family_id, answered_at);
CREATE INDEX IF NOT EXISTS idx_attempts_question ON attempts(question_id, answered_at);
CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_id, question_id, family_id);

CREATE TABLE IF NOT EXISTS learning_artifacts (
    id TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE
        CHECK(length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'),
    size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0 AND size_bytes <= 1073741824),
    media_type TEXT NOT NULL CHECK(length(media_type) BETWEEN 1 AND 127),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_actions (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    decision_id TEXT NOT NULL REFERENCES decisions(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    learner_id TEXT NOT NULL REFERENCES learners(id),
    sequence INTEGER NOT NULL CHECK(sequence > 0),
    stage TEXT NOT NULL CHECK(stage IN ('unassisted', 'assisted', 'post_feedback')),
    action_type TEXT NOT NULL CHECK(action_type IN (
        'started', 'hint_requested', 'answer_revised', 'artifact_checkpoint',
        'explanation_checkpoint', 'check_run', 'tool_used', 'submitted',
        'feedback_shown', 'abandoned'
    )),
    payload_json TEXT NOT NULL CHECK(
        length(payload_json) <= 16384
        AND json_valid(payload_json)
        AND json_type(payload_json) = 'object'
    ),
    artifact_id TEXT REFERENCES learning_artifacts(id),
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    command_hash TEXT NOT NULL
        CHECK(length(command_hash) = 64 AND command_hash NOT GLOB '*[^0-9a-f]*'),
    UNIQUE(decision_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_learning_actions_decision
ON learning_actions(decision_id, sequence);
CREATE INDEX IF NOT EXISTS idx_learning_actions_session
ON learning_actions(session_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_learning_actions_learner
ON learning_actions(learner_id, occurred_at);

-- Productive-skill tasks are versioned independently from selected-response
-- questions, but every task release is pinned to the exact curriculum release
-- whose concepts it names.  The operational ledger is shadow-only: none of
-- these tables is a learner projection or a certification projection.
CREATE TABLE IF NOT EXISTS performance_tasks (
    task_id TEXT NOT NULL,
    task_version INTEGER NOT NULL CHECK(task_version > 0),
    task_digest TEXT NOT NULL UNIQUE CHECK(
        length(task_digest) = 64
        AND task_digest NOT GLOB '*[^0-9a-f]*'
    ),
    definition_json TEXT NOT NULL CHECK(
        length(definition_json) <= 1048576
        AND json_valid(definition_json)
        AND json_type(definition_json) = 'object'
    ),
    imported_at TEXT NOT NULL,
    PRIMARY KEY(task_id, task_version)
);

CREATE TABLE IF NOT EXISTS performance_task_releases (
    id TEXT PRIMARY KEY,
    corpus_release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    bundle_hash TEXT NOT NULL UNIQUE CHECK(
        length(bundle_hash) = 64
        AND bundle_hash NOT GLOB '*[^0-9a-f]*'
    ),
    title TEXT NOT NULL CHECK(length(trim(title)) BETWEEN 1 AND 256),
    review_json TEXT NOT NULL CHECK(
        length(review_json) <= 16384
        AND json_valid(review_json)
        AND json_type(review_json) = 'object'
    ),
    created_at TEXT NOT NULL,
    sealed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS release_performance_tasks (
    release_id TEXT NOT NULL REFERENCES performance_task_releases(id),
    task_id TEXT NOT NULL,
    task_version INTEGER NOT NULL,
    task_digest TEXT NOT NULL CHECK(
        length(task_digest) = 64
        AND task_digest NOT GLOB '*[^0-9a-f]*'
    ),
    status TEXT NOT NULL CHECK(status IN ('quarantined', 'pilot', 'approved')),
    PRIMARY KEY(release_id, task_id, task_version),
    FOREIGN KEY(task_id, task_version)
        REFERENCES performance_tasks(task_id, task_version)
);

CREATE INDEX IF NOT EXISTS idx_release_performance_tasks_status
ON release_performance_tasks(release_id, status, task_id, task_version);

CREATE TABLE IF NOT EXISTS performance_attempts (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    task_release_id TEXT NOT NULL REFERENCES performance_task_releases(id),
    corpus_release_id TEXT NOT NULL REFERENCES corpus_releases(id),
    task_id TEXT NOT NULL,
    task_version INTEGER NOT NULL,
    task_digest TEXT NOT NULL CHECK(
        length(task_digest) = 64
        AND task_digest NOT GLOB '*[^0-9a-f]*'
    ),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    learner_id TEXT NOT NULL REFERENCES learners(id),
    session_revision INTEGER NOT NULL CHECK(session_revision >= 0),
    learner_revision INTEGER NOT NULL CHECK(learner_revision >= 0),
    started_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    command_hash TEXT NOT NULL CHECK(
        length(command_hash) = 64
        AND command_hash NOT GLOB '*[^0-9a-f]*'
    ),
    FOREIGN KEY(task_release_id, task_id, task_version)
        REFERENCES release_performance_tasks(release_id, task_id, task_version)
);

CREATE INDEX IF NOT EXISTS idx_performance_attempts_session
ON performance_attempts(session_id, started_at);
CREATE INDEX IF NOT EXISTS idx_performance_attempts_learner
ON performance_attempts(learner_id, started_at);

CREATE TABLE IF NOT EXISTS performance_actions (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    attempt_id TEXT NOT NULL REFERENCES performance_attempts(id),
    sequence INTEGER NOT NULL CHECK(sequence >= 0),
    phase TEXT NOT NULL CHECK(phase IN ('unassisted', 'assisted', 'post_feedback')),
    action_type TEXT NOT NULL CHECK(action_type IN (
        'started', 'hint_requested', 'answer_revised', 'artifact_checkpoint',
        'explanation_checkpoint', 'check_run', 'tool_used', 'submitted',
        'feedback_shown', 'abandoned'
    )),
    payload_json TEXT NOT NULL CHECK(
        length(payload_json) <= 16384
        AND json_valid(payload_json)
        AND json_type(payload_json) = 'object'
    ),
    elapsed_ms INTEGER CHECK(elapsed_ms IS NULL OR elapsed_ms >= 0),
    occurred_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    command_hash TEXT NOT NULL CHECK(
        length(command_hash) = 64
        AND command_hash NOT GLOB '*[^0-9a-f]*'
    ),
    UNIQUE(attempt_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_performance_actions_attempt
ON performance_actions(attempt_id, sequence);

-- A scoring claim is an immutable admission record for one logical provider
-- callback.  Its command hash, and the caller key when one is supplied, are
-- committed before the callback runs.  This provides cross-process,
-- at-most-once callback admission even for no-key callers.  A claim
-- deliberately has no automatic expiry: after an interrupted callback the
-- system fails closed instead of guessing whether an external scorer ran.
CREATE TABLE IF NOT EXISTS performance_scoring_claims (
    id TEXT PRIMARY KEY CHECK(length(id) BETWEEN 1 AND 128),
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    claim_schema_version INTEGER NOT NULL DEFAULT 1 CHECK(
        claim_schema_version IN (1, 2)
    ),
    idempotency_key TEXT UNIQUE CHECK(
        idempotency_key IS NULL
        OR (
            length(idempotency_key) BETWEEN 1 AND 256
            AND idempotency_key = trim(idempotency_key)
        )
    ),
    attempt_id TEXT NOT NULL,
    evaluation_id TEXT NOT NULL UNIQUE CHECK(
        length(evaluation_id) BETWEEN 1 AND 128
    ),
    through_sequence INTEGER NOT NULL CHECK(through_sequence >= 0),
    provider_id TEXT NOT NULL CHECK(length(trim(provider_id)) BETWEEN 1 AND 128),
    provider_version TEXT NOT NULL CHECK(
        length(trim(provider_version)) BETWEEN 1 AND 128
    ),
    action_trace_digest TEXT NOT NULL CHECK(
        length(action_trace_digest) = 64
        AND action_trace_digest NOT GLOB '*[^0-9a-f]*'
    ),
    scoring_request_digest TEXT CHECK(
        scoring_request_digest IS NULL
        OR (
            length(scoring_request_digest) = 64
            AND scoring_request_digest NOT GLOB '*[^0-9a-f]*'
        )
    ),
    provider_binding_digest TEXT CHECK(
        provider_binding_digest IS NULL
        OR (
            length(provider_binding_digest) = 64
            AND provider_binding_digest NOT GLOB '*[^0-9a-f]*'
        )
    ),
    provider_operation_digest TEXT CHECK(
        provider_operation_digest IS NULL
        OR (
            length(provider_operation_digest) = 64
            AND provider_operation_digest NOT GLOB '*[^0-9a-f]*'
        )
    ),
    command_hash TEXT NOT NULL UNIQUE CHECK(
        length(command_hash) = 64
        AND command_hash NOT GLOB '*[^0-9a-f]*'
    ),
    claimed_at TEXT NOT NULL,
    CHECK(
        (
            claim_schema_version = 1
            AND scoring_request_digest IS NULL
            AND provider_binding_digest IS NULL
            AND provider_operation_digest IS NULL
        )
        OR (
            claim_schema_version = 2
            AND scoring_request_digest IS NOT NULL
            AND provider_binding_digest IS NOT NULL
            AND provider_operation_digest IS NOT NULL
        )
    ),
    FOREIGN KEY(attempt_id) REFERENCES performance_attempts(id)
        DEFERRABLE INITIALLY DEFERRED
);

-- Reconciliation is observational and append-only.  Repeated unknown
-- receipts preserve uncertainty.  At most one terminal observation can close
-- a claim, and no observation may follow that terminal boundary.
CREATE TABLE IF NOT EXISTS performance_scoring_reconciliations (
    id TEXT PRIMARY KEY CHECK(length(id) BETWEEN 1 AND 128),
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    idempotency_key TEXT UNIQUE CHECK(
        idempotency_key IS NULL
        OR (
            length(idempotency_key) BETWEEN 1 AND 256
            AND idempotency_key = trim(idempotency_key)
        )
    ),
    claim_id TEXT NOT NULL REFERENCES performance_scoring_claims(id),
    attempt_id TEXT NOT NULL REFERENCES performance_attempts(id),
    evaluation_id TEXT NOT NULL CHECK(length(evaluation_id) BETWEEN 1 AND 128),
    outcome TEXT NOT NULL CHECK(
        outcome IN ('unknown', 'completed', 'definitely_absent')
    ),
    scoring_request_digest TEXT NOT NULL CHECK(
        length(scoring_request_digest) = 64
        AND scoring_request_digest NOT GLOB '*[^0-9a-f]*'
    ),
    provider_binding_digest TEXT NOT NULL CHECK(
        length(provider_binding_digest) = 64
        AND provider_binding_digest NOT GLOB '*[^0-9a-f]*'
    ),
    provider_operation_digest TEXT NOT NULL CHECK(
        length(provider_operation_digest) = 64
        AND provider_operation_digest NOT GLOB '*[^0-9a-f]*'
    ),
    reconciler_id TEXT NOT NULL CHECK(
        length(trim(reconciler_id)) BETWEEN 1 AND 128
    ),
    reconciler_version TEXT NOT NULL CHECK(
        length(trim(reconciler_version)) BETWEEN 1 AND 128
    ),
    reconciliation_binding_digest TEXT NOT NULL CHECK(
        length(reconciliation_binding_digest) = 64
        AND reconciliation_binding_digest NOT GLOB '*[^0-9a-f]*'
    ),
    receipt_json TEXT NOT NULL CHECK(
        length(receipt_json) <= 131072
        AND json_valid(receipt_json)
        AND json_type(receipt_json) = 'object'
    ),
    -- SQLite has no built-in SHA-256.  Admission checks lowercase digest
    -- shape and exact event/projection receipt equality; verify_integrity()
    -- parses the closed receipt and cryptographically recomputes this digest.
    receipt_digest TEXT NOT NULL CHECK(
        length(receipt_digest) = 64
        AND receipt_digest NOT GLOB '*[^0-9a-f]*'
    ),
    normalized_result_digest TEXT CHECK(
        normalized_result_digest IS NULL
        OR (
            length(normalized_result_digest) = 64
            AND normalized_result_digest NOT GLOB '*[^0-9a-f]*'
        )
    ),
    reconciled_at TEXT NOT NULL,
    command_hash TEXT NOT NULL UNIQUE CHECK(
        length(command_hash) = 64
        AND command_hash NOT GLOB '*[^0-9a-f]*'
    ),
    CHECK(
        (outcome = 'completed' AND normalized_result_digest IS NOT NULL)
        OR (
            outcome IN ('unknown', 'definitely_absent')
            AND normalized_result_digest IS NULL
        )
    ),
    UNIQUE(claim_id, receipt_digest)
);

CREATE UNIQUE INDEX IF NOT EXISTS
idx_performance_scoring_reconciliations_terminal
ON performance_scoring_reconciliations(claim_id)
WHERE outcome IN ('completed', 'definitely_absent');

CREATE TABLE IF NOT EXISTS task_evaluations (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    attempt_id TEXT NOT NULL REFERENCES performance_attempts(id),
    through_sequence INTEGER NOT NULL CHECK(through_sequence >= 0),
    evaluation_digest TEXT NOT NULL CHECK(
        length(evaluation_digest) = 64
        AND evaluation_digest NOT GLOB '*[^0-9a-f]*'
    ),
    evaluation_json TEXT NOT NULL CHECK(
        length(evaluation_json) <= 1048576
        AND json_valid(evaluation_json)
        AND json_type(evaluation_json) = 'object'
    ),
    authority_json TEXT NOT NULL CHECK(
        length(authority_json) <= 65536
        AND json_valid(authority_json)
        AND json_type(authority_json) = 'object'
    ),
    recorded_at TEXT NOT NULL,
    command_hash TEXT NOT NULL CHECK(
        length(command_hash) = 64
        AND command_hash NOT GLOB '*[^0-9a-f]*'
    ),
    UNIQUE(attempt_id, evaluation_digest)
);

CREATE INDEX IF NOT EXISTS idx_task_evaluations_attempt
ON task_evaluations(attempt_id, recorded_at);

CREATE TABLE IF NOT EXISTS shadow_evidence_bundles (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
    evaluation_id TEXT NOT NULL UNIQUE REFERENCES task_evaluations(id),
    attempt_id TEXT NOT NULL REFERENCES performance_attempts(id),
    bundle_digest TEXT NOT NULL CHECK(
        length(bundle_digest) = 64
        AND bundle_digest NOT GLOB '*[^0-9a-f]*'
    ),
    bundle_json TEXT NOT NULL CHECK(
        length(bundle_json) <= 1048576
        AND json_valid(bundle_json)
        AND json_type(bundle_json) = 'object'
    ),
    projection_applied INTEGER NOT NULL DEFAULT 0
        CHECK(projection_applied = 0),
    certification_applied INTEGER NOT NULL DEFAULT 0
        CHECK(certification_applied = 0),
    recorded_at TEXT NOT NULL,
    UNIQUE(attempt_id, bundle_digest)
);

CREATE TABLE IF NOT EXISTS skill_states (
    learner_id TEXT NOT NULL REFERENCES learners(id),
    concept_id TEXT NOT NULL REFERENCES concepts(id),
    mean REAL NOT NULL,
    variance REAL NOT NULL CHECK(variance > 0),
    stability_hours REAL NOT NULL CHECK(stability_hours > 0),
    exposures INTEGER NOT NULL,
    last_seen_at TEXT,
    next_review_at TEXT,
    evidence_mass REAL NOT NULL,
    as_of_event_id TEXT REFERENCES events(event_id),
    model_version TEXT NOT NULL,
    PRIMARY KEY(learner_id, concept_id)
);

CREATE TABLE IF NOT EXISTS objective_states (
    learner_id TEXT NOT NULL REFERENCES learners(id),
    objective_id TEXT NOT NULL REFERENCES learning_objectives(id),
    mean REAL NOT NULL,
    variance REAL NOT NULL CHECK(variance > 0),
    stability_hours REAL NOT NULL CHECK(stability_hours > 0),
    exposures INTEGER NOT NULL,
    last_seen_at TEXT,
    next_review_at TEXT,
    evidence_mass REAL NOT NULL,
    as_of_event_id TEXT REFERENCES events(event_id),
    model_version TEXT NOT NULL,
    PRIMARY KEY(learner_id, objective_id)
);

CREATE TABLE IF NOT EXISTS objective_grid_states (
    learner_id TEXT NOT NULL,
    objective_id TEXT NOT NULL,
    posterior_schema_version INTEGER NOT NULL CHECK(posterior_schema_version > 0),
    algorithm TEXT NOT NULL CHECK(length(algorithm) > 0),
    grid_id TEXT NOT NULL CHECK(length(grid_id) > 0),
    codec TEXT NOT NULL CHECK(length(codec) > 0),
    posterior_blob BLOB NOT NULL CHECK(typeof(posterior_blob) = 'blob'),
    posterior_sha256 TEXT NOT NULL CHECK(
        length(posterior_sha256) = 64
        AND posterior_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    mean REAL NOT NULL,
    variance REAL NOT NULL CHECK(variance > 0),
    mastery_probability REAL NOT NULL CHECK(
        mastery_probability >= 0 AND mastery_probability <= 1
    ),
    expected_competence REAL NOT NULL CHECK(
        expected_competence >= 0 AND expected_competence <= 1
    ),
    edge_mass REAL NOT NULL CHECK(edge_mass >= 0 AND edge_mass <= 1),
    mastery_probability_error_bound REAL NOT NULL CHECK(
        mastery_probability_error_bound >= 0
        AND mastery_probability_error_bound <= 1
    ),
    evidence_mass REAL NOT NULL CHECK(evidence_mass >= 0),
    acquisition_mass REAL NOT NULL CHECK(acquisition_mass >= 0),
    as_of_event_id TEXT NOT NULL REFERENCES events(event_id),
    model_version TEXT NOT NULL CHECK(length(model_version) > 0),
    PRIMARY KEY(learner_id, objective_id),
    FOREIGN KEY(learner_id, objective_id)
        REFERENCES objective_states(learner_id, objective_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS misconception_beliefs (
    learner_id TEXT NOT NULL REFERENCES learners(id),
    misconception_id TEXT NOT NULL REFERENCES misconceptions(id),
    log_odds REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    last_seen_at TEXT,
    as_of_event_id TEXT REFERENCES events(event_id),
    model_version TEXT NOT NULL,
    PRIMARY KEY(learner_id, misconception_id)
);

CREATE TABLE IF NOT EXISTS learner_skill_families (
    learner_id TEXT NOT NULL REFERENCES learners(id),
    concept_id TEXT NOT NULL REFERENCES concepts(id),
    family_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    first_unguided_correct_at TEXT NOT NULL,
    last_unguided_correct_at TEXT NOT NULL,
    delayed_unguided_correct_at TEXT,
    PRIMARY KEY(learner_id, concept_id, family_id)
);

CREATE TABLE IF NOT EXISTS learner_objective_families (
    learner_id TEXT NOT NULL REFERENCES learners(id),
    objective_id TEXT NOT NULL REFERENCES learning_objectives(id),
    family_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    first_unguided_correct_at TEXT NOT NULL,
    last_unguided_correct_at TEXT NOT NULL,
    delayed_unguided_correct_at TEXT,
    PRIMARY KEY(learner_id, objective_id, family_id)
);

CREATE TABLE IF NOT EXISTS item_stats (
    question_id TEXT PRIMARY KEY REFERENCES questions(id),
    exposures INTEGER NOT NULL DEFAULT 0,
    correct_count INTEGER NOT NULL DEFAULT 0,
    total_response_ms INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS question_revocations (
    question_id TEXT PRIMARY KEY REFERENCES questions(id),
    reason TEXT NOT NULL,
    revoked_at TEXT NOT NULL,
    event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id)
);

CREATE TABLE IF NOT EXISTS generation_jobs (
    id TEXT PRIMARY KEY,
    blueprint_json TEXT NOT NULL,
    status TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    prompt_version TEXT NOT NULL,
    raw_output_json TEXT,
    validation_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_generation_jobs_status ON generation_jobs(status, created_at);

CREATE TABLE IF NOT EXISTS generation_job_runs (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES generation_jobs(id),
    attempt INTEGER NOT NULL CHECK(attempt > 0),
    status TEXT NOT NULL CHECK(status IN ('running', 'reviewed', 'rejected', 'failed')),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    source_context_sha256 TEXT NOT NULL CHECK(length(source_context_sha256) = 64),
    raw_output_json TEXT,
    validation_json TEXT,
    error_json TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(job_id, attempt),
    CHECK(
        (status = 'running' AND raw_output_json IS NULL
         AND validation_json IS NULL AND error_json IS NULL
         AND completed_at IS NULL)
        OR
        (status IN ('reviewed', 'rejected') AND raw_output_json IS NOT NULL
         AND validation_json IS NOT NULL AND error_json IS NULL
         AND completed_at IS NOT NULL)
        OR
        (status = 'failed' AND error_json IS NOT NULL
         AND completed_at IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_generation_job_runs_job_attempt
ON generation_job_runs(job_id, attempt);

CREATE TABLE IF NOT EXISTS item_reviews (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL REFERENCES questions(id),
    reviewer_kind TEXT NOT NULL,
    verdict TEXT NOT NULL,
    issues_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS events_no_update
BEFORE UPDATE ON events BEGIN
    SELECT RAISE(ABORT, 'events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS events_no_delete
BEFORE DELETE ON events BEGIN
    SELECT RAISE(ABORT, 'events are append-only');
END;

CREATE TRIGGER IF NOT EXISTS policy_shadow_evaluations_validate_insert
BEFORE INSERT ON policy_shadow_evaluations BEGIN
    SELECT CASE WHEN EXISTS (
        SELECT 1
        FROM json_each(NEW.frontier_json) candidate
        WHERE json_type(candidate.value) != 'object'
           OR json_type(candidate.value, '$.question_id') != 'text'
    ) THEN RAISE(
        ABORT, 'policy shadow frontier must contain identified candidate objects'
    ) END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM json_each(NEW.frontier_json) candidate
        WHERE json_extract(candidate.value, '$.question_id') =
              NEW.live_question_id
    ) THEN RAISE(
        ABORT, 'policy shadow frontier omits the live question'
    ) END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM json_each(NEW.frontier_json) candidate
        WHERE json_extract(candidate.value, '$.question_id') =
              NEW.challenger_question_id
    ) THEN RAISE(
        ABORT, 'policy shadow frontier omits the challenger question'
    ) END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM decisions decision
        JOIN sessions session ON session.id = decision.session_id
        JOIN events shadow_event ON shadow_event.event_id = NEW.event_id
        JOIN events selection_event
          ON selection_event.event_id = shadow_event.causation_id
        WHERE decision.id = NEW.decision_id
          AND session.learner_id = decision.learner_id
          AND decision.policy_version = NEW.logging_policy_version
          AND decision.corpus_release_id = NEW.corpus_release_id
          AND decision.candidate_count = NEW.candidate_count
          AND decision.candidate_digest = NEW.candidate_digest
          AND decision.question_id = NEW.live_question_id
          AND decision.propensity = (
              SELECT json_extract(
                  candidate.value, '$.logging_probability'
              )
              FROM json_each(NEW.frontier_json) candidate
              WHERE json_extract(
                  candidate.value, '$.question_id'
              ) = NEW.live_question_id
          )
          AND json(decision.selected_score_json) = (
              SELECT json_remove(
                  candidate.value,
                  '$.question_id',
                  '$.logging_probability'
              )
              FROM json_each(NEW.frontier_json) candidate
              WHERE json_extract(
                  candidate.value, '$.question_id'
              ) = NEW.live_question_id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM json_each(NEW.frontier_json) shadow_candidate
              LEFT JOIN json_each(
                  decision.top_candidates_json
              ) logged_candidate
                ON logged_candidate.key = shadow_candidate.key
              WHERE json_extract(
                        shadow_candidate.value, '$.question_id'
                    ) IS NOT json_extract(
                        logged_candidate.value, '$.question_id'
                    )
          )
          AND shadow_event.event_type = 'PolicyShadowEvaluated'
          AND shadow_event.schema_version = 1
          AND shadow_event.stream_id =
              'learner:' || decision.learner_id
          AND shadow_event.learner_id = decision.learner_id
          AND shadow_event.session_id = decision.session_id
          AND shadow_event.correlation_id = NEW.decision_id
          AND shadow_event.idempotency_key =
              'policy-shadow:v1:' || NEW.decision_id || ':' ||
              NEW.challenger_definition_digest
          AND shadow_event.occurred_at = NEW.evaluated_at
          AND shadow_event.recorded_at = NEW.recorded_at
          AND selection_event.event_type = 'QuestionSelected'
          AND selection_event.stream_id = shadow_event.stream_id
          AND selection_event.learner_id = shadow_event.learner_id
          AND selection_event.session_id = shadow_event.session_id
          AND selection_event.stream_version + 1 =
              shadow_event.stream_version
          AND selection_event.occurred_at = shadow_event.occurred_at
          AND json_extract(
              selection_event.payload_json, '$.decision_id'
          ) = NEW.decision_id
          AND json_extract(
              selection_event.payload_json, '$.question_id'
          ) = NEW.live_question_id
          AND json_extract(
              selection_event.payload_json, '$.candidate_count'
          ) = NEW.candidate_count
          AND json_extract(
              selection_event.payload_json, '$.candidate_digest'
          ) = NEW.candidate_digest
          AND json_extract(
              selection_event.payload_json, '$.propensity'
          ) = decision.propensity
          AND json_extract(
              selection_event.payload_json, '$.score'
          ) = json(decision.selected_score_json)
          AND json_extract(
              selection_event.metadata_json, '$.policy_version'
          ) = NEW.logging_policy_version
          AND json_extract(
              selection_event.metadata_json, '$.learner_model_version'
          ) = NEW.learner_model_version
          AND json_extract(
              selection_event.metadata_json, '$.corpus_release_id'
          ) = NEW.corpus_release_id
          AND json_type(shadow_event.payload_json) = 'object'
          AND (
              SELECT COUNT(*) FROM json_each(shadow_event.payload_json)
          ) = 20
          AND json_extract(
              shadow_event.payload_json, '$.evaluation_id'
          ) = NEW.id
          AND json_extract(
              shadow_event.payload_json, '$.decision_id'
          ) = NEW.decision_id
          AND json_extract(
              shadow_event.payload_json, '$.challenger_policy_version'
          ) = NEW.challenger_policy_version
          AND json_extract(
              shadow_event.payload_json, '$.challenger_definition_digest'
          ) = NEW.challenger_definition_digest
          AND json_extract(
              shadow_event.payload_json, '$.logging_policy_version'
          ) = NEW.logging_policy_version
          AND json_extract(
              shadow_event.payload_json, '$.learner_model_version'
          ) = NEW.learner_model_version
          AND json_extract(
              shadow_event.payload_json, '$.corpus_release_id'
          ) = NEW.corpus_release_id
          AND json_type(
              shadow_event.payload_json, '$.candidate_count'
          ) = 'integer'
          AND json_extract(
              shadow_event.payload_json, '$.candidate_count'
          ) = NEW.candidate_count
          AND json_extract(
              shadow_event.payload_json, '$.candidate_digest'
          ) = NEW.candidate_digest
          AND json_type(
              shadow_event.payload_json, '$.frontier'
          ) = 'array'
          AND json_extract(
              shadow_event.payload_json, '$.frontier'
          ) = json(NEW.frontier_json)
          AND json_extract(
              shadow_event.payload_json, '$.frontier_digest'
          ) = NEW.frontier_digest
          AND json_extract(
              shadow_event.payload_json, '$.input_digest'
          ) = NEW.input_digest
          AND json_extract(
              shadow_event.payload_json, '$.output_digest'
          ) = NEW.output_digest
          AND json_extract(
              shadow_event.payload_json, '$.live_question_id'
          ) = NEW.live_question_id
          AND json_extract(
              shadow_event.payload_json, '$.challenger_question_id'
          ) = NEW.challenger_question_id
          AND json_type(
              shadow_event.payload_json, '$.agreement'
          ) = CASE NEW.agreement WHEN 1 THEN 'true' ELSE 'false' END
          AND json_extract(
              shadow_event.payload_json, '$.agreement'
          ) = NEW.agreement
          AND json_extract(
              shadow_event.payload_json, '$.evaluated_at'
          ) = NEW.evaluated_at
          AND json_type(
              shadow_event.payload_json, '$.shadow_only'
          ) = 'true'
          AND json_type(
              shadow_event.payload_json, '$.selection_applied'
          ) = 'false'
          AND json_type(
              shadow_event.payload_json, '$.mastery_applied'
          ) = 'false'
          AND json_type(shadow_event.metadata_json) = 'object'
          AND (
              SELECT COUNT(*) FROM json_each(shadow_event.metadata_json)
          ) = 4
          AND json_extract(
              shadow_event.metadata_json, '$.shadow_contract_version'
          ) = 'policy-shadow-v1'
          AND json_type(
              shadow_event.metadata_json, '$.shadow_only'
          ) = 'true'
          AND json_type(
              shadow_event.metadata_json, '$.selection_applied'
          ) = 'false'
          AND json_type(
              shadow_event.metadata_json, '$.mastery_applied'
          ) = 'false'
    ) THEN RAISE(
        ABORT, 'policy shadow evaluation does not match its decision/event'
    ) END;
END;

CREATE TRIGGER IF NOT EXISTS policy_shadow_evaluations_no_update
BEFORE UPDATE ON policy_shadow_evaluations BEGIN
    SELECT RAISE(ABORT, 'policy shadow evaluations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS policy_shadow_evaluations_no_delete
BEFORE DELETE ON policy_shadow_evaluations BEGIN
    SELECT RAISE(ABORT, 'policy shadow evaluations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS attempts_validate_insert
BEFORE INSERT ON attempts BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM decisions d JOIN sessions s ON s.id = d.session_id
        WHERE d.id = NEW.decision_id
          AND d.session_id = NEW.session_id
          AND d.learner_id = NEW.learner_id
          AND d.question_id = NEW.question_id
          AND s.learner_id = NEW.learner_id
    ) THEN RAISE(ABORT, 'attempt does not match decision/session') END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM events e
        WHERE e.event_id = NEW.event_id
          AND e.learner_id = NEW.learner_id
          AND e.session_id = NEW.session_id
          AND e.event_type = 'ResponseSubmitted'
    ) THEN RAISE(ABORT, 'attempt does not match response event') END;
    SELECT CASE WHEN NEW.selected_option_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM options o
        WHERE o.question_id = NEW.question_id AND o.option_id = NEW.selected_option_id
    ) THEN RAISE(ABORT, 'selected option is not part of question') END;
    SELECT CASE WHEN NEW.selected_option_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM json_each(NEW.presented_order_json)
        WHERE value = NEW.selected_option_id
    ) THEN RAISE(ABORT, 'selected option was not presented') END;
    SELECT CASE WHEN NEW.is_correct != COALESCE((
        SELECT o.is_correct FROM options o
        WHERE o.question_id = NEW.question_id AND o.option_id = NEW.selected_option_id
    ), 0) THEN RAISE(ABORT, 'attempt correctness does not match answer key') END;
END;

CREATE TRIGGER IF NOT EXISTS attempts_no_delete
BEFORE DELETE ON attempts BEGIN
    SELECT RAISE(ABORT, 'attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS question_revocations_validate_insert
BEFORE INSERT ON question_revocations BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM events event
        WHERE event.event_id = NEW.event_id
          AND event.event_type = 'QuestionEmergencyRevoked'
          AND json_extract(event.payload_json, '$.question_id') = NEW.question_id
          AND json_extract(event.payload_json, '$.reason') = NEW.reason
    ) THEN RAISE(ABORT, 'revocation does not match its safety event') END;
END;

CREATE TRIGGER IF NOT EXISTS question_revocations_no_update
BEFORE UPDATE ON question_revocations BEGIN
    SELECT RAISE(ABORT, 'question revocations are append-only');
END;

CREATE TRIGGER IF NOT EXISTS question_revocations_no_delete
BEFORE DELETE ON question_revocations BEGIN
    SELECT RAISE(ABORT, 'question revocations are append-only');
END;

CREATE TRIGGER IF NOT EXISTS performance_tasks_no_update
BEFORE UPDATE ON performance_tasks BEGIN
    SELECT RAISE(ABORT, 'performance tasks are immutable');
END;

CREATE TRIGGER IF NOT EXISTS performance_tasks_no_delete
BEFORE DELETE ON performance_tasks BEGIN
    SELECT RAISE(ABORT, 'performance tasks are immutable');
END;

CREATE TRIGGER IF NOT EXISTS performance_task_releases_no_update
BEFORE UPDATE ON performance_task_releases BEGIN
    SELECT RAISE(ABORT, 'performance task releases are immutable');
END;

CREATE TRIGGER IF NOT EXISTS performance_task_releases_no_delete
BEFORE DELETE ON performance_task_releases BEGIN
    SELECT RAISE(ABORT, 'performance task releases are immutable');
END;

CREATE TRIGGER IF NOT EXISTS release_performance_tasks_validate_insert
BEFORE INSERT ON release_performance_tasks BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM performance_tasks task
        WHERE task.task_id = NEW.task_id
          AND task.task_version = NEW.task_version
          AND task.task_digest = NEW.task_digest
    ) THEN RAISE(ABORT, 'performance task release digest mismatch') END;
END;

CREATE TRIGGER IF NOT EXISTS release_performance_tasks_no_update
BEFORE UPDATE ON release_performance_tasks BEGIN
    SELECT RAISE(ABORT, 'performance task membership is immutable');
END;

CREATE TRIGGER IF NOT EXISTS release_performance_tasks_no_delete
BEFORE DELETE ON release_performance_tasks BEGIN
    SELECT RAISE(ABORT, 'performance task membership is immutable');
END;

CREATE TRIGGER IF NOT EXISTS performance_attempts_validate_insert
BEFORE INSERT ON performance_attempts BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM sessions session
        JOIN learners learner ON learner.id = session.learner_id
        JOIN performance_task_releases task_release
          ON task_release.id = NEW.task_release_id
        JOIN release_performance_tasks member
          ON member.release_id = task_release.id
         AND member.task_id = NEW.task_id
         AND member.task_version = NEW.task_version
        WHERE session.id = NEW.session_id
          AND session.learner_id = NEW.learner_id
          AND session.corpus_release_id = NEW.corpus_release_id
          AND session.corpus_release_id = task_release.corpus_release_id
          AND session.status = 'active'
          AND session.revision = NEW.session_revision
          AND learner.revision = NEW.learner_revision
          AND member.task_digest = NEW.task_digest
          AND member.status IN ('pilot', 'approved')
    ) THEN RAISE(ABORT, 'performance attempt violates its release/session boundary') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM decisions decision
        WHERE decision.session_id = NEW.session_id
          AND decision.consumed_at IS NULL
          AND decision.invalidated_at IS NULL
    ) THEN RAISE(ABORT, 'performance attempt conflicts with a pending question') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM performance_attempts prior
        WHERE prior.session_id = NEW.session_id
          AND NOT EXISTS (
              SELECT 1 FROM performance_actions terminal
              WHERE terminal.attempt_id = prior.id
                AND terminal.action_type IN ('submitted', 'abandoned')
          )
    ) THEN RAISE(ABORT, 'session already has an active performance attempt') END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM events event
        WHERE event.event_id = NEW.event_id
          AND event.event_type = 'PerformanceTaskStarted'
          AND event.schema_version = 1
          AND event.stream_id = 'learner:' || NEW.learner_id
          AND event.learner_id = NEW.learner_id
          AND event.session_id = NEW.session_id
          AND event.occurred_at = NEW.started_at
          AND event.recorded_at = NEW.recorded_at
          AND json_extract(event.payload_json, '$.attempt_id') = NEW.id
          AND json_extract(event.payload_json, '$.task_release_id') = NEW.task_release_id
          AND json_extract(event.payload_json, '$.task_id') = NEW.task_id
          AND json_extract(event.payload_json, '$.task_version') = NEW.task_version
          AND json_extract(event.payload_json, '$.task_digest') = NEW.task_digest
    ) THEN RAISE(ABORT, 'performance attempt does not match its event') END;
END;

CREATE TRIGGER IF NOT EXISTS performance_attempts_no_update
BEFORE UPDATE ON performance_attempts BEGIN
    SELECT RAISE(ABORT, 'performance attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS performance_attempts_no_delete
BEFORE DELETE ON performance_attempts BEGIN
    SELECT RAISE(ABORT, 'performance attempts are immutable');
END;

CREATE TRIGGER IF NOT EXISTS performance_actions_validate_insert
BEFORE INSERT ON performance_actions BEGIN
    SELECT CASE WHEN NEW.sequence != COALESCE((
        SELECT MAX(action.sequence) + 1
        FROM performance_actions action
        WHERE action.attempt_id = NEW.attempt_id
    ), 0) THEN RAISE(ABORT, 'performance action sequence is not contiguous') END;
    SELECT CASE WHEN NEW.sequence = 0 AND NEW.action_type != 'started'
        THEN RAISE(ABORT, 'performance trace must start explicitly') END;
    SELECT CASE WHEN NEW.sequence > 0 AND NEW.action_type = 'started'
        THEN RAISE(ABORT, 'performance trace cannot restart') END;
    SELECT CASE WHEN NEW.action_type IN (
        'started', 'submitted', 'abandoned', 'feedback_shown'
    ) AND EXISTS (
        SELECT 1 FROM performance_actions singleton
        WHERE singleton.attempt_id = NEW.attempt_id
          AND singleton.action_type = NEW.action_type
    ) THEN RAISE(ABORT, 'performance lifecycle action cannot repeat') END;
    SELECT CASE WHEN NEW.phase = 'post_feedback' AND NOT EXISTS (
        SELECT 1 FROM performance_actions submission
        WHERE submission.attempt_id = NEW.attempt_id
          AND submission.action_type = 'submitted'
    ) THEN RAISE(ABORT, 'post-feedback action precedes submission') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM performance_actions terminal
        WHERE terminal.attempt_id = NEW.attempt_id
          AND terminal.action_type IN ('submitted', 'abandoned')
    ) AND NEW.phase != 'post_feedback'
        THEN RAISE(ABORT, 'pre-feedback action follows terminal checkpoint') END;
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM performance_actions terminal
        WHERE terminal.attempt_id = NEW.attempt_id
          AND terminal.action_type = 'abandoned'
    ) THEN RAISE(ABORT, 'performance action follows abandonment') END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM performance_attempts attempt
        JOIN sessions session ON session.id = attempt.session_id
        JOIN events start_event ON start_event.event_id = attempt.event_id
        JOIN events action_event ON action_event.event_id = NEW.event_id
        WHERE attempt.id = NEW.attempt_id
          AND session.status = 'active'
          AND action_event.event_type = 'PerformanceActionRecorded'
          AND action_event.schema_version = 1
          AND action_event.stream_id = 'learner:' || attempt.learner_id
          AND action_event.learner_id = attempt.learner_id
          AND action_event.session_id = attempt.session_id
          AND action_event.causation_id = attempt.id
          AND action_event.stream_version > start_event.stream_version
          AND action_event.occurred_at = NEW.occurred_at
          AND action_event.recorded_at = NEW.recorded_at
          AND json_extract(action_event.payload_json, '$.action.id') = NEW.id
          AND json_extract(action_event.payload_json, '$.action.trace_id') = NEW.attempt_id
          AND json_extract(action_event.payload_json, '$.action.sequence') = NEW.sequence
          AND json_extract(action_event.payload_json, '$.action.kind') = NEW.action_type
          AND json_extract(action_event.payload_json, '$.action.phase') = NEW.phase
    ) THEN RAISE(ABORT, 'performance action does not match its attempt/event') END;
END;

CREATE TRIGGER IF NOT EXISTS performance_actions_no_update
BEFORE UPDATE ON performance_actions BEGIN
    SELECT RAISE(ABORT, 'performance actions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS performance_actions_no_delete
BEFORE DELETE ON performance_actions BEGIN
    SELECT RAISE(ABORT, 'performance actions are immutable');
END;

CREATE TRIGGER IF NOT EXISTS performance_scoring_claims_validate_insert
BEFORE INSERT ON performance_scoring_claims BEGIN
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM events event
        WHERE event.idempotency_key = NEW.idempotency_key
          AND event.event_id != NEW.event_id
    ) THEN RAISE(ABORT, 'scoring claim idempotency key already has an event') END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM performance_attempts attempt
        JOIN performance_actions submission
          ON submission.attempt_id = attempt.id
         AND submission.sequence = NEW.through_sequence
         AND submission.action_type = 'submitted'
        WHERE attempt.id = NEW.attempt_id
    ) THEN RAISE(ABORT, 'scoring claim lacks its submitted trace boundary') END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM events claim_event
        JOIN performance_attempts attempt ON attempt.id = NEW.attempt_id
        WHERE claim_event.event_id = NEW.event_id
          AND claim_event.event_type IN (
              'PerformanceScoringClaimed',
              'PerformanceScoringClaimMigrated'
          )
          AND claim_event.schema_version = 1
          AND claim_event.stream_id = 'learner:' || attempt.learner_id
          AND claim_event.learner_id = attempt.learner_id
          AND (
              (
                  claim_event.event_type = 'PerformanceScoringClaimed'
                  AND claim_event.session_id = attempt.session_id
              )
              OR (
                  claim_event.event_type = 'PerformanceScoringClaimMigrated'
                  AND (
                      claim_event.session_id IS NULL
                      OR claim_event.session_id = attempt.session_id
                  )
              )
          )
          AND claim_event.idempotency_key =
              'performance-score-claim:v1:' || NEW.command_hash
          AND claim_event.correlation_id = NEW.attempt_id
          AND claim_event.causation_id = NEW.attempt_id
          AND json_extract(
              claim_event.payload_json, '$.claim_id'
          ) = NEW.id
          AND json_extract(
              claim_event.payload_json, '$.caller_idempotency_key'
          ) IS NEW.idempotency_key
          AND json_extract(
              claim_event.payload_json, '$.attempt_id'
          ) = NEW.attempt_id
          AND json_extract(
              claim_event.payload_json, '$.evaluation_id'
          ) = NEW.evaluation_id
          AND json_extract(
              claim_event.payload_json, '$.through_sequence'
          ) = NEW.through_sequence
          AND json_extract(
              claim_event.payload_json, '$.provider_id'
          ) = NEW.provider_id
          AND json_extract(
              claim_event.payload_json, '$.provider_version'
          ) = NEW.provider_version
          AND json_extract(
              claim_event.payload_json, '$.action_trace_digest'
          ) = NEW.action_trace_digest
          AND json_extract(
              claim_event.payload_json, '$.command_hash'
          ) = NEW.command_hash
          AND json_extract(
              claim_event.payload_json, '$.claimed_at'
          ) = NEW.claimed_at
    ) THEN RAISE(ABORT, 'scoring claim does not match its event') END;
END;

CREATE TRIGGER IF NOT EXISTS performance_scoring_claims_no_update
BEFORE UPDATE ON performance_scoring_claims BEGIN
    SELECT RAISE(ABORT, 'performance scoring claims are immutable');
END;

CREATE TRIGGER IF NOT EXISTS performance_scoring_claims_no_delete
BEFORE DELETE ON performance_scoring_claims BEGIN
    SELECT RAISE(ABORT, 'performance scoring claims are immutable');
END;

CREATE TRIGGER IF NOT EXISTS events_respect_performance_scoring_claim
BEFORE INSERT ON events
WHEN NEW.idempotency_key IS NOT NULL
 AND EXISTS (
     SELECT 1 FROM performance_scoring_claims claim
     WHERE claim.idempotency_key = NEW.idempotency_key
 )
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM performance_scoring_claims claim
        WHERE claim.idempotency_key = NEW.idempotency_key
          AND NEW.event_type = 'TaskEvaluationRecorded'
          AND json_extract(
              NEW.metadata_json, '$.command_hash'
          ) = claim.command_hash
          AND json_extract(
              NEW.payload_json, '$.attempt_id'
          ) = claim.attempt_id
          AND json_extract(
              NEW.payload_json, '$.through_sequence'
          ) = claim.through_sequence
          AND json_extract(
              NEW.payload_json, '$.evaluation.id'
          ) = claim.evaluation_id
    ) THEN RAISE(
        ABORT, 'event does not complete its performance scoring claim'
    ) END;
END;

CREATE TRIGGER IF NOT EXISTS task_evaluations_validate_scoring_claim
BEFORE INSERT ON task_evaluations BEGIN
    SELECT CASE WHEN EXISTS (
        SELECT 1
        FROM performance_scoring_claims claim
        JOIN events evaluation_event
          ON evaluation_event.event_id = NEW.event_id
        WHERE claim.command_hash = NEW.command_hash
          AND (
              claim.attempt_id != NEW.attempt_id
              OR claim.evaluation_id != NEW.id
              OR claim.through_sequence != NEW.through_sequence
              OR evaluation_event.idempotency_key
                 IS NOT claim.idempotency_key
          )
    ) THEN RAISE(
        ABORT, 'task evaluation does not complete its scoring claim'
    ) END;
    SELECT CASE WHEN json_extract(
        NEW.authority_json,
        '$.normalized_result.normalization_mode'
    ) = 'registered_provider' AND NOT EXISTS (
        SELECT 1 FROM performance_scoring_claims claim
        WHERE claim.command_hash = NEW.command_hash
          AND claim.attempt_id = NEW.attempt_id
          AND claim.evaluation_id = NEW.id
          AND claim.through_sequence = NEW.through_sequence
    ) AND NOT EXISTS (
        SELECT 1 FROM events exemption
        WHERE exemption.event_type = 'PerformanceScoringLegacyExempted'
          AND exemption.schema_version = 1
          AND json_extract(
              exemption.payload_json, '$.evaluation_id'
          ) = NEW.id
          AND json_extract(
              exemption.payload_json, '$.attempt_id'
          ) = NEW.attempt_id
          AND json_extract(
              exemption.payload_json, '$.command_hash'
          ) = NEW.command_hash
    ) THEN RAISE(
        ABORT, 'registered evaluation lacks its scoring claim'
    ) END;
END;

CREATE TRIGGER IF NOT EXISTS task_evaluations_validate_insert
BEFORE INSERT ON task_evaluations BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM performance_actions submission
        WHERE submission.attempt_id = NEW.attempt_id
          AND submission.action_type = 'submitted'
          AND submission.sequence = NEW.through_sequence
    ) THEN RAISE(ABORT, 'task evaluation lacks its submitted trace boundary') END;
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM performance_attempts attempt
        JOIN events evaluation_event ON evaluation_event.event_id = NEW.event_id
        WHERE attempt.id = NEW.attempt_id
          AND evaluation_event.event_type = 'TaskEvaluationRecorded'
          AND evaluation_event.schema_version = 1
          AND evaluation_event.stream_id = 'learner:' || attempt.learner_id
          AND evaluation_event.learner_id = attempt.learner_id
          AND evaluation_event.session_id = attempt.session_id
          AND (
              evaluation_event.causation_id = NEW.attempt_id
              OR EXISTS (
                  SELECT 1 FROM performance_scoring_claims claim
                  WHERE claim.event_id = evaluation_event.causation_id
                    AND claim.attempt_id = NEW.attempt_id
                    AND claim.evaluation_id = NEW.id
                    AND claim.command_hash = NEW.command_hash
              )
          )
          AND evaluation_event.recorded_at = NEW.recorded_at
          AND json_extract(evaluation_event.payload_json, '$.evaluation.id') = NEW.id
          AND json_extract(
              evaluation_event.payload_json, '$.evaluation_digest'
          ) = NEW.evaluation_digest
          AND json_extract(
              evaluation_event.payload_json, '$.through_sequence'
          ) = NEW.through_sequence
    ) THEN RAISE(ABORT, 'task evaluation does not match its event') END;
END;

CREATE TRIGGER IF NOT EXISTS task_evaluations_no_update
BEFORE UPDATE ON task_evaluations BEGIN
    SELECT RAISE(ABORT, 'task evaluations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS task_evaluations_no_delete
BEFORE DELETE ON task_evaluations BEGIN
    SELECT RAISE(ABORT, 'task evaluations are immutable');
END;

CREATE TRIGGER IF NOT EXISTS shadow_evidence_bundles_validate_insert
BEFORE INSERT ON shadow_evidence_bundles BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1
        FROM task_evaluations evaluation
        JOIN performance_attempts attempt ON attempt.id = evaluation.attempt_id
        JOIN events bundle_event ON bundle_event.event_id = NEW.event_id
        WHERE evaluation.id = NEW.evaluation_id
          AND evaluation.attempt_id = NEW.attempt_id
          AND bundle_event.event_type = 'ShadowEvidenceReduced'
          AND bundle_event.schema_version = 1
          AND bundle_event.stream_id = 'learner:' || attempt.learner_id
          AND bundle_event.learner_id = attempt.learner_id
          AND bundle_event.session_id = attempt.session_id
          AND bundle_event.causation_id = evaluation.id
          AND bundle_event.recorded_at = NEW.recorded_at
          AND json_extract(bundle_event.payload_json, '$.bundle_id') = NEW.id
          AND json_extract(
              bundle_event.payload_json, '$.bundle_digest'
          ) = NEW.bundle_digest
          AND json_extract(
              bundle_event.payload_json, '$.projection_applied'
          ) = 0
          AND json_extract(
              bundle_event.payload_json, '$.certification_applied'
          ) = 0
    ) THEN RAISE(ABORT, 'shadow evidence does not match its evaluation/event') END;
END;

CREATE TRIGGER IF NOT EXISTS shadow_evidence_bundles_no_update
BEFORE UPDATE ON shadow_evidence_bundles BEGIN
    SELECT RAISE(ABORT, 'shadow evidence bundles are immutable');
END;

CREATE TRIGGER IF NOT EXISTS shadow_evidence_bundles_no_delete
BEFORE DELETE ON shadow_evidence_bundles BEGIN
    SELECT RAISE(ABORT, 'shadow evidence bundles are immutable');
END;
"""


@dataclass(frozen=True, slots=True)
class _IndexSchemaContract:
    """Stable semantic shape of one SQLite index."""

    name: str | None
    unique: bool
    origin: str
    partial: bool
    columns: tuple[tuple[str | None, int, str | None, bool, bool], ...]
    sql: str | None


@dataclass(frozen=True, slots=True)
class _TableSchemaContract:
    """Read-only structural contract for one application table."""

    name: str
    definition: str
    columns: tuple[
        tuple[str, str, bool, str | None, int, int],
        ...,
    ]
    foreign_keys: tuple[
        tuple[
            str,
            str,
            str,
            str,
            tuple[tuple[int, str, str | None], ...],
        ],
        ...,
    ]
    indexes: tuple[_IndexSchemaContract, ...]
    checks: tuple[str, ...]
    without_rowid: bool
    strict: bool


@dataclass(frozen=True, slots=True)
class _CurrentSchemaContract:
    """Immutable, cache-safe description of the complete current schema."""

    tables: tuple[_TableSchemaContract, ...]
    triggers: tuple[tuple[str, str, str], ...]


def _quote_sqlite_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _execute_sql_script(
    connection: sqlite3.Connection,
    script: str,
) -> None:
    """Execute a SQL script without sqlite3.executescript's implicit commit.

    SQLite DDL is transactional, but ``Connection.executescript`` commits any
    pending transaction before it starts. Migration and guard installation
    must stay inside the caller's serialized transaction, so split only where
    SQLite itself recognizes a complete statement.
    """

    pending: list[str] = []
    for line in script.splitlines(keepends=True):
        pending.append(line)
        statement = "".join(pending).strip()
        if statement and sqlite3.complete_statement(statement):
            connection.execute(statement)
            pending.clear()
    remainder = "".join(pending).strip()
    if remainder:
        raise RuntimeError("SQL script ended with an incomplete statement.")


def _normalize_schema_sql(sql: str) -> str:
    """Normalize SQL outside quoted values while preserving their meaning."""

    normalized: list[str] = []
    pending_space = False
    quote: str | None = None
    index = 0
    while index < len(sql):
        character = sql[index]
        if quote is not None:
            normalized.append(character)
            if quote == "[":
                if character == "]":
                    if index + 1 < len(sql) and sql[index + 1] == "]":
                        normalized.append(sql[index + 1])
                        index += 1
                    else:
                        quote = None
            elif character == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    normalized.append(sql[index + 1])
                    index += 1
                else:
                    quote = None
            index += 1
            continue
        if character.isspace():
            pending_space = bool(normalized)
            index += 1
            continue
        if pending_space:
            normalized.append(" ")
            pending_space = False
        if character in {"'", '"', "`", "["}:
            quote = character
            normalized.append(character)
        else:
            normalized.append(character.casefold())
        index += 1
    result = "".join(normalized).strip()
    if result.endswith(";"):
        result = result[:-1].rstrip()
    return result


def _normalize_index_definition(sql: str) -> str:
    """Discard non-structural CREATE/IF NOT EXISTS/index-name spelling."""

    normalized = _normalize_schema_sql(sql)
    marker = " on "
    position = normalized.find(marker)
    return (
        normalized[position + len(marker) :]
        if position >= 0
        else normalized
    )


def _normalize_table_definition(sql: str) -> str:
    """Preserve table-level semantics omitted by SQLite's PRAGMA views."""

    normalized = _normalize_schema_sql(sql)
    opening = normalized.find("(")
    definition = normalized[opening:] if opening >= 0 else normalized
    compacted: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(definition):
        character = definition[index]
        if quote is not None:
            compacted.append(character)
            if quote == "[":
                if character == "]":
                    if (
                        index + 1 < len(definition)
                        and definition[index + 1] == "]"
                    ):
                        compacted.append(definition[index + 1])
                        index += 1
                    else:
                        quote = None
            elif character == quote:
                if (
                    index + 1 < len(definition)
                    and definition[index + 1] == quote
                ):
                    compacted.append(definition[index + 1])
                    index += 1
                else:
                    quote = None
            index += 1
            continue
        if character in {'"', "`", "["}:
            closing = "]" if character == "[" else character
            cursor = index + 1
            identifier: list[str] = []
            while cursor < len(definition):
                current = definition[cursor]
                if current == closing:
                    if (
                        cursor + 1 < len(definition)
                        and definition[cursor + 1] == closing
                    ):
                        identifier.append(closing)
                        cursor += 2
                        continue
                    break
                identifier.append(current)
                cursor += 1
            identifier_text = "".join(identifier)
            if (
                cursor < len(definition)
                and identifier_text
                and (
                    identifier_text[0].isalpha()
                    or identifier_text[0] == "_"
                )
                and all(
                    item.isalnum() or item == "_"
                    for item in identifier_text
                )
            ):
                compacted.extend(identifier_text.casefold())
                index = cursor + 1
                continue
            quote = character
            compacted.append(character)
        elif character == "'":
            quote = character
            compacted.append(character)
        elif character == " " and (
            (compacted and compacted[-1] in {"(", ","})
            or (
                index + 1 < len(definition)
                and definition[index + 1] in {")", ","}
            )
        ):
            pass
        else:
            compacted.append(character)
        index += 1
    return "".join(compacted)


def _normalize_trigger_definition(sql: str) -> str:
    """Normalize a trigger while ignoring idempotent installation syntax."""

    normalized = _normalize_schema_sql(sql)
    prefix = "create trigger if not exists "
    if normalized.startswith(prefix):
        return "create trigger " + normalized[len(prefix) :]
    return normalized


@lru_cache(maxsize=None)
def _canonical_table_sql_bundle(
    table_name: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return canonical table-tail, column order, and explicit indexes."""

    reference = sqlite3.connect(":memory:")
    reference.row_factory = sqlite3.Row
    try:
        reference.execute("PRAGMA foreign_keys = ON")
        reference.executescript(DDL)
        row = reference.execute(
            """SELECT sql FROM sqlite_master
               WHERE type='table' AND name=?""",
            (table_name,),
        ).fetchone()
        if row is None or not row["sql"]:
            raise RuntimeError(
                f"Current DDL has no table definition for {table_name}."
            )
        raw_sql = row["sql"]
        opening = raw_sql.find("(")
        if opening < 0:
            raise RuntimeError(
                f"Current DDL table {table_name} has no definition body."
            )
        columns = tuple(
            row["name"]
            for row in sorted(
                reference.execute(
                    f"PRAGMA table_xinfo({_quote_sqlite_identifier(table_name)})"
                ).fetchall(),
                key=lambda item: int(item["cid"]),
            )
            if int(row["hidden"]) == 0
        )
        indexes = tuple(
            row["sql"]
            for row in reference.execute(
                """SELECT sql FROM sqlite_master
                   WHERE type='index' AND tbl_name=? AND sql IS NOT NULL
                   ORDER BY name""",
                (table_name,),
            ).fetchall()
        )
        return raw_sql[opening:], columns, indexes
    finally:
        reference.close()


def _extract_check_constraints(sql: str) -> tuple[str, ...]:
    """Extract balanced CHECK bodies without confusing quoted text for SQL."""

    checks: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(sql):
        character = sql[index]
        if quote is not None:
            if quote == "[":
                if character == "]":
                    if index + 1 < len(sql) and sql[index + 1] == "]":
                        index += 2
                        continue
                    quote = None
            elif character == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if character in {"'", '"', "`", "["}:
            quote = character
            index += 1
            continue
        token = sql[index : index + 5]
        before = sql[index - 1] if index else " "
        after = sql[index + 5] if index + 5 < len(sql) else " "
        if (
            token.casefold() == "check"
            and not (before.isalnum() or before == "_")
            and not (after.isalnum() or after == "_")
        ):
            opening = index + 5
            while opening < len(sql) and sql[opening].isspace():
                opening += 1
            if opening < len(sql) and sql[opening] == "(":
                depth = 1
                cursor = opening + 1
                inner_quote: str | None = None
                while cursor < len(sql) and depth:
                    current = sql[cursor]
                    if inner_quote is not None:
                        if inner_quote == "[":
                            if current == "]":
                                if (
                                    cursor + 1 < len(sql)
                                    and sql[cursor + 1] == "]"
                                ):
                                    cursor += 2
                                    continue
                                inner_quote = None
                        elif current == inner_quote:
                            if (
                                cursor + 1 < len(sql)
                                and sql[cursor + 1] == inner_quote
                            ):
                                cursor += 2
                                continue
                            inner_quote = None
                    elif current in {"'", '"', "`", "["}:
                        inner_quote = current
                    elif current == "(":
                        depth += 1
                    elif current == ")":
                        depth -= 1
                    cursor += 1
                if depth == 0:
                    checks.append(
                        _normalize_schema_sql(sql[opening + 1 : cursor - 1])
                    )
                    index = cursor
                    continue
        index += 1
    return tuple(sorted(checks))


def _capture_current_schema_contract(
    connection: sqlite3.Connection,
) -> _CurrentSchemaContract:
    """Read SQLite metadata only; callers may keep ``query_only`` enabled."""

    table_rows = connection.execute(
        """SELECT name, sql FROM sqlite_master
           WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
           ORDER BY name"""
    ).fetchall()
    index_sql = {
        row["name"]: row["sql"]
        for row in connection.execute(
            """SELECT name, sql FROM sqlite_master
               WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"""
        ).fetchall()
    }
    tables: list[_TableSchemaContract] = []
    for table_row in table_rows:
        table = table_row["name"]
        quoted_table = _quote_sqlite_identifier(table)
        columns = tuple(
            sorted(
                (
                    row["name"],
                    " ".join((row["type"] or "").upper().split()),
                    bool(row["notnull"]),
                    (
                        _normalize_schema_sql(row["dflt_value"])
                        if row["dflt_value"] is not None
                        else None
                    ),
                    int(row["pk"]),
                    int(row["hidden"]),
                )
                for row in connection.execute(
                    f"PRAGMA table_xinfo({quoted_table})"
                ).fetchall()
            )
        )

        foreign_key_rows = connection.execute(
            f"PRAGMA foreign_key_list({quoted_table})"
        ).fetchall()
        foreign_keys_by_id: dict[int, list[sqlite3.Row]] = {}
        for row in foreign_key_rows:
            foreign_keys_by_id.setdefault(int(row["id"]), []).append(row)
        foreign_keys = tuple(
            sorted(
                [
                    (
                        rows[0]["table"],
                        rows[0]["on_update"],
                        rows[0]["on_delete"],
                        rows[0]["match"],
                        tuple(
                            (
                                int(row["seq"]),
                                row["from"],
                                row["to"],
                            )
                            for row in sorted(
                                rows, key=lambda item: int(item["seq"])
                            )
                        ),
                    )
                    for rows in foreign_keys_by_id.values()
                ],
                key=repr,
            )
        )

        indexes: list[_IndexSchemaContract] = []
        for index_row in connection.execute(
            f"PRAGMA index_list({quoted_table})"
        ).fetchall():
            index_name = index_row["name"]
            quoted_index = _quote_sqlite_identifier(index_name)
            index_columns = tuple(
                (
                    row["name"],
                    (
                        int(row["cid"])
                        if int(row["cid"]) in {-1, -2}
                        else 0
                    ),
                    row["coll"],
                    bool(row["desc"]),
                    bool(row["key"]),
                )
                for row in sorted(
                    connection.execute(
                        f"PRAGMA index_xinfo({quoted_index})"
                    ).fetchall(),
                    key=lambda item: int(item["seqno"]),
                )
            )
            origin = index_row["origin"]
            raw_index_sql = index_sql.get(index_name)
            indexes.append(
                _IndexSchemaContract(
                    name=index_name if origin == "c" else None,
                    unique=bool(index_row["unique"]),
                    origin=origin,
                    partial=bool(index_row["partial"]),
                    columns=index_columns,
                    sql=(
                        _normalize_index_definition(raw_index_sql)
                        if raw_index_sql is not None
                        else None
                    ),
                )
            )
        table_sql = table_row["sql"] or ""
        normalized_table_sql = _normalize_schema_sql(table_sql)
        table_tail = normalized_table_sql[
            normalized_table_sql.rfind(")") + 1 :
        ]
        tables.append(
            _TableSchemaContract(
                name=table,
                # PRAGMA foreign_key_list omits DEFERRABLE/INITIALLY terms.
                # Keeping the normalized definition tail also closes that and
                # any future table-level semantic blind spots while ignoring
                # CREATE IF NOT EXISTS and table-name spelling.
                definition=_normalize_table_definition(table_sql),
                columns=columns,
                foreign_keys=foreign_keys,
                indexes=tuple(sorted(indexes, key=repr)),
                checks=_extract_check_constraints(table_sql),
                without_rowid="without rowid" in table_tail,
                strict="strict" in table_tail.replace(",", " ").split(),
            )
        )

    triggers = tuple(
        (
            row["name"],
            row["tbl_name"],
            _normalize_trigger_definition(row["sql"] or ""),
        )
        for row in connection.execute(
            """SELECT name, tbl_name, sql FROM sqlite_master
               WHERE type = 'trigger' AND name NOT LIKE 'sqlite_%'
               ORDER BY name"""
        ).fetchall()
    )
    return _CurrentSchemaContract(
        tables=tuple(tables),
        triggers=triggers,
    )


@lru_cache(maxsize=1)
def _expected_current_schema_contract() -> _CurrentSchemaContract:
    """Build the authoritative contract once from a fresh in-memory schema."""

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    reference = Database(":memory:")
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(DDL)
        reference._migrate_v11_to_v12(connection)
        reference._migrate_v12_to_v13(connection)
        reference._migrate_v13_to_v14(connection)
        reference._migrate_v14_to_v15(connection)
        reference._install_current_performance_scoring_triggers(connection)
        reference._install_v5_indexes(connection)
        reference._install_v6_authoring_triggers(connection)
        reference._install_v4_attempt_triggers(connection)
        reference._install_v8_learning_action_triggers(connection)
        reference._install_release_snapshot_triggers(connection)
        reference._install_corpus_registry_triggers(connection)
        connection.commit()
        return _capture_current_schema_contract(connection)
    finally:
        connection.close()


@lru_cache(maxsize=1)
def _expected_v18_schema_contract() -> _CurrentSchemaContract:
    """Return the one exact v18 structure accepted as a migration source."""

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    reference = Database(":memory:")
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(DDL)
        reference._migrate_v11_to_v12(connection)
        reference._migrate_v12_to_v13(connection)
        reference._migrate_v13_to_v14(connection)
        reference._migrate_v14_to_v15(connection)
        reference._install_current_performance_scoring_triggers(connection)
        reference._install_v5_indexes(connection)
        reference._install_v6_authoring_triggers(connection)
        reference._install_v4_attempt_triggers(connection)
        reference._install_v8_learning_action_triggers(connection)
        reference._install_release_snapshot_triggers(connection)
        reference._install_corpus_registry_triggers(connection)
        reference._downgrade_v19_contract_to_v18(connection)
        connection.commit()
        return _capture_current_schema_contract(connection)
    finally:
        connection.close()


_POLICY_SHADOW_TABLE = "policy_shadow_evaluations"
_POLICY_SHADOW_TRIGGER_NAMES = frozenset(
    {
        "policy_shadow_evaluations_validate_insert",
        "policy_shadow_evaluations_no_update",
        "policy_shadow_evaluations_no_delete",
    }
)


@lru_cache(maxsize=1)
def _expected_v17_schema_contract() -> _CurrentSchemaContract:
    """Return the one exact v17 structure accepted as a migration source."""

    current = _expected_v18_schema_contract()
    return _CurrentSchemaContract(
        tables=tuple(
            table
            for table in current.tables
            if table.name != _POLICY_SHADOW_TABLE
        ),
        triggers=tuple(
            trigger
            for trigger in current.triggers
            if trigger[0] not in _POLICY_SHADOW_TRIGGER_NAMES
        ),
    )


_CURRENT_CLAIM_SESSION_TRIGGER_FRAGMENT = (
    "and ( ( claim_event.event_type = 'PerformanceScoringClaimed' "
    "and claim_event.session_id = attempt.session_id ) or ( "
    "claim_event.event_type = 'PerformanceScoringClaimMigrated' and ( "
    "claim_event.session_id is null or claim_event.session_id = "
    "attempt.session_id ) ) )"
)
_V16_CLAIM_SESSION_TRIGGER_FRAGMENT = (
    "and claim_event.session_id = attempt.session_id"
)


@lru_cache(maxsize=1)
def _expected_v16_schema_contract() -> _CurrentSchemaContract:
    """Return the one exact v16 structure accepted as a migration source."""

    current = _expected_v17_schema_contract()
    triggers: list[tuple[str, str, str]] = []
    replaced = False
    for name, table, sql in current.triggers:
        if name == "performance_scoring_claims_validate_insert":
            if sql.count(_CURRENT_CLAIM_SESSION_TRIGGER_FRAGMENT) != 1:
                raise RuntimeError(
                    "Current scoring-claim trigger cannot derive v16 contract."
                )
            sql = sql.replace(
                _CURRENT_CLAIM_SESSION_TRIGGER_FRAGMENT,
                _V16_CLAIM_SESSION_TRIGGER_FRAGMENT,
            )
            replaced = True
        triggers.append((name, table, sql))
    if not replaced:
        raise RuntimeError("Current schema lacks the scoring-claim trigger.")
    return _CurrentSchemaContract(
        tables=current.tables,
        triggers=tuple(triggers),
    )


_LEGACY_CANONICALIZATION_NULLABLE_COLUMNS = frozenset(
    {
        ("attempts", "command_hash"),
        ("concepts", "content_hash"),
        ("decisions", "corpus_release_id"),
        ("decisions", "evidence_weight"),
        ("decisions", "focus_valid"),
        ("decisions", "learner_revision"),
        ("decisions", "pedagogical_role"),
        ("decisions", "question_content_hash"),
        ("decisions", "question_status"),
        ("decisions", "question_version"),
        ("decisions", "session_revision"),
        ("learner_skill_families", "kind"),
        ("misconceptions", "content_hash"),
        ("questions", "content_hash"),
        ("sessions", "corpus_release_id"),
        ("sources", "content_hash"),
    }
)
_LEGACY_CANONICALIZATION_MISSING_FOREIGN_KEYS = {
    "decisions": frozenset(
        {
            "corpus_release_id",
            "focus_concept_id",
            "focus_misconception_id",
        }
    ),
    "sessions": frozenset({"corpus_release_id"}),
}
_LEGACY_CANONICALIZATION_MISSING_CHECKS = {
    "decisions": frozenset({"focus_valid in (0, 1)"}),
}
_LEGACY_CANONICALIZATION_MISSING_INDEXES = {
    "decisions": frozenset({"idx_one_pending_decision"}),
}
_TABLE_DEFINITION_SENSITIVE_TOKENS = frozenset(
    {
        "autoincrement",
        "collate",
        "conflict",
        "deferrable",
        "generated",
        "initially",
        "stored",
        "strict",
        "unique",
        "virtual",
        "without",
    }
)


def _validate_legacy_canonicalization_source(
    table: str,
    actual: _TableSchemaContract,
    expected: _TableSchemaContract,
) -> None:
    """Admit only semantic differences produced by known ALTER migrations."""

    actual_columns = {row[0]: row[1:] for row in actual.columns}
    expected_columns = {row[0]: row[1:] for row in expected.columns}
    if set(actual_columns) != set(expected_columns):
        raise ConflictError(
            f"Schema v13 cannot safely rebuild {table}: column set differs."
        )
    for name, expected_terms in expected_columns.items():
        actual_terms = actual_columns[name]
        if actual_terms == expected_terms:
            continue
        # Historical SQLite ADD COLUMN operations could not retrofit these
        # backfilled NOT NULL constraints. No type, default, key, or hidden
        # attribute is allowed to differ.
        relaxed = (
            expected_terms[0],
            False,
            expected_terms[2],
            expected_terms[3],
            expected_terms[4],
        )
        if (
            (table, name)
            not in _LEGACY_CANONICALIZATION_NULLABLE_COLUMNS
            or expected_terms[1] is not True
            or actual_terms != relaxed
        ):
            raise ConflictError(
                f"Schema v13 cannot safely rebuild {table}: "
                f"column {name} has an unknown legacy definition."
            )

    actual_foreign_keys = set(actual.foreign_keys)
    expected_foreign_keys = set(expected.foreign_keys)
    if not actual_foreign_keys <= expected_foreign_keys:
        raise ConflictError(
            f"Schema v13 cannot safely rebuild {table}: "
            "foreign-key definitions differ."
        )
    allowed_missing_fk_columns = (
        _LEGACY_CANONICALIZATION_MISSING_FOREIGN_KEYS.get(
            table,
            frozenset(),
        )
    )
    for foreign_key in expected_foreign_keys - actual_foreign_keys:
        source_columns = {
            source
            for _sequence, source, _target in foreign_key[4]
        }
        if not source_columns or not (
            source_columns <= allowed_missing_fk_columns
        ):
            raise ConflictError(
                f"Schema v13 cannot safely rebuild {table}: "
                "a non-legacy foreign key is missing."
            )

    actual_checks = set(actual.checks)
    expected_checks = set(expected.checks)
    allowed_missing_checks = (
        _LEGACY_CANONICALIZATION_MISSING_CHECKS.get(
            table,
            frozenset(),
        )
    )
    if (
        not actual_checks <= expected_checks
        or not (expected_checks - actual_checks) <= allowed_missing_checks
    ):
        raise ConflictError(
            f"Schema v13 cannot safely rebuild {table}: "
            "CHECK constraints differ."
        )

    actual_indexes = set(actual.indexes)
    expected_indexes = set(expected.indexes)
    if not actual_indexes <= expected_indexes:
        raise ConflictError(
            f"Schema v13 cannot safely rebuild {table}: "
            "an unknown or changed index is attached."
        )
    allowed_missing_indexes = (
        _LEGACY_CANONICALIZATION_MISSING_INDEXES.get(
            table,
            frozenset(),
        )
    )
    for index in expected_indexes - actual_indexes:
        if index.name not in allowed_missing_indexes:
            raise ConflictError(
                f"Schema v13 cannot safely rebuild {table}: "
                "a required index is missing."
            )

    if (
        actual.without_rowid != expected.without_rowid
        or actual.strict != expected.strict
    ):
        raise ConflictError(
            f"Schema v13 cannot safely rebuild {table}: "
            "storage constraints differ."
        )

    # PRAGMA metadata covers the admitted nullability, key, CHECK, index,
    # and foreign-key differences above. These terms cover remaining SQLite
    # table-definition semantics that PRAGMA views can omit.
    actual_tokens = re.findall(r"[a-z_]+", actual.definition)
    expected_tokens = re.findall(r"[a-z_]+", expected.definition)
    for token in _TABLE_DEFINITION_SENSITIVE_TOKENS:
        if actual_tokens.count(token) != expected_tokens.count(token):
            raise ConflictError(
                f"Schema v13 cannot safely rebuild {table}: "
                f"unexpected {token.upper()} semantics."
            )


def _require_no_foreign_key_violations(
    connection: sqlite3.Connection,
    *,
    context: str,
) -> None:
    """Reject an existing or newly produced referential-integrity breach."""

    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if not violations:
        return
    first = violations[0]
    raise ConflictError(
        f"{context} has a foreign-key violation in "
        f"{first['table']} at row {first['rowid']}."
    )


class Database:
    def __init__(self, path: str | Path, *, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only

    def connect(self) -> sqlite3.Connection:
        if self.read_only:
            if not self.path.exists() or not self.path.is_file():
                raise NotFoundError(f"Database does not exist: {self.path}")
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            connection: sqlite3.Connection | None = None
            try:
                connection = sqlite3.connect(
                    uri, uri=True, timeout=20.0
                )
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only = ON")
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA busy_timeout = 20000")
            except sqlite3.Error as exc:
                if connection is not None:
                    connection.close()
                raise ValidationError(
                    f"Could not open database read-only: {exc}"
                ) from exc
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=20.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 20000")
        return connection

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    def validate_current_schema(
        self,
        *,
        _allow_missing_triggers: bool = False,
    ) -> tuple[str, ...]:
        """Validate a current TSQ schema without installing or migrating it."""

        try:
            with self.read() as connection:
                present = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                if "meta" not in present:
                    raise ValidationError(
                        "Database has no TSQ schema metadata."
                    )
                row = connection.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
                if row is None:
                    raise ValidationError(
                        "Database has no TSQ schema version."
                    )
                try:
                    version = int(row["value"])
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        "Database schema version is not an integer."
                    ) from exc
                if version > SCHEMA_VERSION:
                    raise ConflictError(
                        f"Database schema is {version}; this engine supports "
                        f"at most {SCHEMA_VERSION}."
                    )
                if version < SCHEMA_VERSION:
                    raise ConflictError(
                        f"Database schema is {version}; inspection requires "
                        f"current schema {SCHEMA_VERSION}. Run an explicit "
                        "writable migration first."
                    )
                missing = CURRENT_SCHEMA_TABLES - present
                if missing:
                    raise ConflictError(
                        "Current database schema is incomplete; missing tables: "
                        + ", ".join(sorted(missing))
                    )
                expected = _expected_current_schema_contract()
                actual = _capture_current_schema_contract(connection)
                expected_tables = {
                    table.name: table for table in expected.tables
                }
                actual_tables = {
                    table.name: table for table in actual.tables
                }
                incompatibilities: list[str] = []
                unexpected_tables = sorted(
                    set(actual_tables) - set(expected_tables)
                )
                if unexpected_tables:
                    incompatibilities.append(
                        "unexpected tables: " + ", ".join(unexpected_tables)
                    )
                for table_name in sorted(
                    set(expected_tables) & set(actual_tables)
                ):
                    expected_table = expected_tables[table_name]
                    actual_table = actual_tables[table_name]
                    if expected_table.columns != actual_table.columns:
                        expected_columns = {
                            column[0]: column
                            for column in expected_table.columns
                        }
                        actual_columns = {
                            column[0]: column for column in actual_table.columns
                        }
                        details: list[str] = []
                        missing_columns = sorted(
                            set(expected_columns) - set(actual_columns)
                        )
                        unexpected_columns = sorted(
                            set(actual_columns) - set(expected_columns)
                        )
                        changed_columns = sorted(
                            name
                            for name in set(expected_columns)
                            & set(actual_columns)
                            if expected_columns[name] != actual_columns[name]
                        )
                        if missing_columns:
                            details.append(
                                "missing " + ", ".join(missing_columns)
                            )
                        if unexpected_columns:
                            details.append(
                                "unexpected " + ", ".join(unexpected_columns)
                            )
                        if changed_columns:
                            details.append(
                                "changed " + ", ".join(changed_columns)
                            )
                        incompatibilities.append(
                            f"table {table_name} column definitions differ"
                            + (f" ({'; '.join(details)})" if details else "")
                        )
                    if (
                        expected_table.definition
                        != actual_table.definition
                    ):
                        incompatibilities.append(
                            f"table {table_name} SQL definition differs"
                        )
                    if (
                        expected_table.foreign_keys
                        != actual_table.foreign_keys
                    ):
                        incompatibilities.append(
                            f"table {table_name} foreign keys differ"
                        )
                    if expected_table.indexes != actual_table.indexes:
                        incompatibilities.append(
                            f"table {table_name} indexes differ"
                        )
                    if expected_table.checks != actual_table.checks:
                        incompatibilities.append(
                            f"table {table_name} CHECK constraints differ"
                        )
                    if (
                        expected_table.without_rowid
                        != actual_table.without_rowid
                        or expected_table.strict != actual_table.strict
                    ):
                        incompatibilities.append(
                            f"table {table_name} storage constraints differ"
                        )

                expected_triggers = {
                    name: (table, sql)
                    for name, table, sql in expected.triggers
                }
                actual_triggers = {
                    name: (table, sql)
                    for name, table, sql in actual.triggers
                }
                missing_triggers = sorted(
                    set(expected_triggers) - set(actual_triggers)
                )
                unexpected_triggers = sorted(
                    set(actual_triggers) - set(expected_triggers)
                )
                changed_triggers = sorted(
                    name
                    for name in set(expected_triggers)
                    & set(actual_triggers)
                    if expected_triggers[name] != actual_triggers[name]
                )
                if missing_triggers and not _allow_missing_triggers:
                    incompatibilities.append(
                        "missing triggers: " + ", ".join(missing_triggers)
                    )
                if unexpected_triggers:
                    incompatibilities.append(
                        "unexpected triggers: "
                        + ", ".join(unexpected_triggers)
                    )
                if changed_triggers:
                    incompatibilities.append(
                        "changed triggers: " + ", ".join(changed_triggers)
                    )
                if incompatibilities:
                    raise ConflictError(
                        "Current database schema structure is incompatible: "
                        + "; ".join(incompatibilities)
                    )
                return tuple(missing_triggers)
        except (ConflictError, NotFoundError, ValidationError):
            raise
        except sqlite3.Error as exc:
            raise ValidationError(
                f"Could not validate TSQ database schema: {exc}"
            ) from exc

    def _restore_missing_schema_guards_for_replay_copy(self) -> tuple[str, ...]:
        """Restore only absent canonical triggers on an isolated replay copy.

        Replay must be able to diagnose a projection that was altered after an
        immutability trigger was removed.  Normal writable opens still reject
        that database before mutation.  This internal path first proves that
        tables, indexes, constraints, and every present trigger exactly match
        the current schema; it then recreates only missing canonical triggers
        on the disposable copy and validates the complete schema again.
        """

        if self.read_only:
            raise ConflictError(
                "A read-only database cannot prepare a replay copy."
            )
        missing = self.validate_current_schema(
            _allow_missing_triggers=True
        )
        if missing:
            expected = {
                name: sql
                for name, _table, sql in (
                    _expected_current_schema_contract().triggers
                )
            }
            with self.transaction() as connection:
                for trigger_name in missing:
                    trigger_sql = expected.get(trigger_name)
                    if trigger_sql is None:
                        raise ConflictError(
                            "Replay copy is missing an unknown schema guard: "
                            f"{trigger_name}"
                        )
                    connection.execute(trigger_sql)
        self.validate_current_schema()
        return missing

    def initialize(self) -> None:
        if self.read_only:
            raise ConflictError(
                "A read-only database cannot be initialized or migrated."
            )
        # A database already claiming the current version is not a migration
        # target.  Validate it through a genuinely read-only handle before a
        # writable SQLite connection can create a journal, reinstall a missing
        # trigger, or otherwise normalize tampered structure.  Older recognized
        # schemas continue through the explicit migration path below.
        if (
            self.path != Path(":memory:")
            and self.path.exists()
            and self.path.is_file()
            and self.path.stat().st_size > 0
        ):
            inspector = Database(self.path, read_only=True)
            try:
                with inspector.read() as existing_connection:
                    user_tables = {
                        row["name"]
                        for row in existing_connection.execute(
                            """SELECT name FROM sqlite_master
                               WHERE type='table'
                                 AND name NOT LIKE 'sqlite_%'"""
                        ).fetchall()
                    }
                    if "meta" in user_tables:
                        version_row = existing_connection.execute(
                            """SELECT value FROM meta
                               WHERE key='schema_version'"""
                        ).fetchone()
                        if version_row is None:
                            raise ValidationError(
                                "Database has no TSQ schema version."
                            )
                        try:
                            existing_version = int(version_row["value"])
                        except (TypeError, ValueError) as exc:
                            raise ValidationError(
                                "Database schema version is not an integer."
                            ) from exc
                        if existing_version > SCHEMA_VERSION:
                            raise ConflictError(
                                f"Database schema is {existing_version}; engine "
                                f"expects at most {SCHEMA_VERSION}."
                            )
                        if existing_version == SCHEMA_VERSION:
                            inspector.validate_current_schema()
                            _require_no_foreign_key_violations(
                                existing_connection,
                                context="Current database",
                            )
                            self._enforce_historical_generated_safety()
                            return
                        if existing_version == 18:
                            actual_v18 = _capture_current_schema_contract(
                                existing_connection
                            )
                            if actual_v18 != _expected_v18_schema_contract():
                                raise ConflictError(
                                    "Schema v18 structure is not the exact "
                                    "supported v19 migration source."
                                )
                        elif existing_version == 17:
                            actual_v17 = _capture_current_schema_contract(
                                existing_connection
                            )
                            if actual_v17 != _expected_v17_schema_contract():
                                raise ConflictError(
                                    "Schema v17 structure is not the exact "
                                    "supported v18 migration source."
                                )
                        elif existing_version == 16:
                            actual_v16 = _capture_current_schema_contract(
                                existing_connection
                            )
                            if actual_v16 != _expected_v16_schema_contract():
                                raise ConflictError(
                                    "Schema v16 structure is not the exact "
                                    "supported v17 migration source."
                                )
                            self._validate_v16_migration_lifecycle(
                                existing_connection
                            )
                    elif user_tables and "questions" not in user_tables:
                        raise ConflictError(
                            "Existing database is not a recognized TSQ schema; "
                            "refusing to modify it."
                        )
            except (ConflictError, NotFoundError, ValidationError):
                raise
            except sqlite3.Error as exc:
                raise ValidationError(
                    f"Could not inspect existing database before migration: {exc}"
                ) from exc
        connection = self.connect()
        legacy_foreign_keys_disabled = False
        try:
            preliminary_had_schema = bool(
                connection.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type='table' AND name='questions'"""
                ).fetchone()
            )
            preliminary_has_meta = bool(
                connection.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type='table' AND name='meta'"""
                ).fetchone()
            )
            preliminary_row = (
                connection.execute(
                    """SELECT value FROM meta WHERE key='schema_version'"""
                ).fetchone()
                if preliminary_has_meta
                else None
            )
            preliminary_version = (
                int(preliminary_row["value"])
                if preliminary_row is not None
                else (
                    1
                    if preliminary_had_schema
                    else SCHEMA_VERSION
                )
            )
            if preliminary_version < 16:
                # Canonicalizing legacy parent tables requires foreign-key
                # enforcement to be disabled before the transaction begins.
                # This PRAGMA is connection-local and non-durable; the source
                # version is rechecked after the writer lock is acquired.
                connection.execute("PRAGMA foreign_keys = OFF")
                legacy_foreign_keys_disabled = True
            # Every source read and every migration write below shares one
            # serialized boundary. The earlier read-only inspection gives
            # clean failures without creating journals, while this lock closes
            # the inspection/write race against another SQLite connection.
            connection.execute("BEGIN IMMEDIATE")
            had_schema = bool(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='questions'"
                ).fetchone()
            )
            had_release_concepts = bool(
                connection.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type='table' AND name='release_concepts'"""
                ).fetchone()
            )
            had_release_misconceptions = bool(
                connection.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type='table' AND name='release_misconceptions'"""
                ).fetchone()
            )
            had_release_sources = bool(
                connection.execute(
                    """SELECT 1 FROM sqlite_master
                       WHERE type='table' AND name='release_sources'"""
                ).fetchone()
            )
            has_meta = bool(
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
                ).fetchone()
            )
            existing = (
                connection.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
                if has_meta
                else None
            )
            current_version = (
                int(existing["value"])
                if existing
                else (1 if had_schema else SCHEMA_VERSION)
            )
            if current_version != preliminary_version:
                raise ConflictError(
                    "Database schema version changed while acquiring the "
                    "migration writer boundary."
                )
            starting_version = current_version
            if current_version > SCHEMA_VERSION:
                raise ConflictError(
                    f"Database schema is {current_version}; engine expects at most {SCHEMA_VERSION}."
                )
            if current_version == 18:
                if (
                    _capture_current_schema_contract(connection)
                    != _expected_v18_schema_contract()
                ):
                    raise ConflictError(
                        "Schema v18 structure is not the exact supported "
                        "v19 migration source."
                    )
            elif current_version == 17:
                if (
                    _capture_current_schema_contract(connection)
                    != _expected_v17_schema_contract()
                ):
                    raise ConflictError(
                        "Schema v17 structure is not the exact supported "
                        "v18 migration source."
                    )
            elif current_version == 16:
                if (
                    _capture_current_schema_contract(connection)
                    != _expected_v16_schema_contract()
                ):
                    raise ConflictError(
                        "Schema v16 structure is not the exact supported "
                        "v17 migration source."
                    )
                self._validate_v16_migration_lifecycle(connection)
            if current_version < SCHEMA_VERSION:
                # Defensive for databases that were manually downgraded or
                # produced by a prerelease build which installed protections
                # before recording the final schema version.  Migrations may
                # legitimately backfill immutable fields; protections return
                # after every migration succeeds.
                self._drop_corpus_registry_triggers(connection)
                self._drop_release_snapshot_triggers(connection)
                self._drop_v6_authoring_triggers(connection)
            _execute_sql_script(connection, DDL)
            if current_version < 2:
                self._migrate_v1_to_v2(connection)
                current_version = 2
            if current_version < 3:
                self._migrate_v2_to_v3(connection)
                current_version = 3
            if current_version < 4:
                self._migrate_v3_to_v4(connection)
                current_version = 4
            if current_version < 5:
                self._migrate_v4_to_v5(
                    connection,
                    had_release_concepts=had_release_concepts,
                    had_release_misconceptions=had_release_misconceptions,
                    had_release_sources=had_release_sources,
                    starting_version=starting_version,
                )
                current_version = 5
            if current_version < 6:
                self._migrate_v5_to_v6(connection)
                current_version = 6
            if current_version < 7:
                self._migrate_v6_to_v7(connection)
                current_version = 7
            if current_version < 8:
                self._migrate_v7_to_v8(connection)
                current_version = 8
            if current_version < 9:
                self._migrate_v8_to_v9(connection)
                current_version = 9
            if current_version < 10:
                self._migrate_v9_to_v10(connection)
                current_version = 10
            if current_version < 11:
                self._migrate_v10_to_v11(connection)
                current_version = 11
            if current_version < 12:
                current_version = 12
            # This migration is intentionally a schema validator: DDL installs
            # the table before version dispatch, and no historical posterior
            # rows are fabricated.  Run it for already-v12 databases too so a
            # partial prerelease schema cannot be trusted merely because its
            # metadata claims the current version.
            self._migrate_v11_to_v12(connection)
            if current_version < 13:
                current_version = 13
            # v1/v2 ALTER TABLE migrations necessarily installed these two
            # columns as nullable.  Rebuild the affected tables even for a
            # prerelease database already labelled v13, but return without
            # writes when their canonical constraints are already present.
            self._migrate_v12_to_v13(connection)
            if current_version < 14:
                current_version = 14
            # v14 installs an empty, shadow-only productive-skill ledger.
            # Historical events and learner projections are never fabricated.
            self._migrate_v13_to_v14(connection)
            if current_version < 15:
                current_version = 15
            # v15 adds immutable provider-callback admission claims.  Existing
            # v14 evaluations cannot be assigned synthetic historical claims.
            self._migrate_v14_to_v15(connection)
            if current_version < 16:
                self._migrate_v15_to_v16(
                    connection,
                    starting_version=starting_version,
                )
                current_version = 16
            # v16 commits every callback admission to the hash-chained event
            # history and makes the claim projection replayable.  Existing
            # v15 rows receive honest migration-observation events; a direct
            # v14 upgrade records explicit exceptions for older registered
            # evaluations instead of fabricating pre-callback admissions.
            if current_version < 17:
                self._migrate_v16_to_v17(connection)
                current_version = 17
            # v18 installs an empty event-backed policy-shadow projection.
            # Historical choices are not assigned synthetic challenger results.
            if current_version < 18:
                self._migrate_v17_to_v18(connection)
                current_version = 18
            # v19 binds new scoring claims to immutable request/provider/
            # operation commitments and adds a prospective, observational
            # reconciliation ledger.  Historical v18 claim bytes remain v1
            # evidence with NULL v2 commitments; no event is fabricated.
            if current_version < 19:
                self._migrate_v18_to_v19(connection)
                current_version = 19
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._install_v5_indexes(connection)
            self._install_v6_authoring_triggers(connection)
            self._install_v4_attempt_triggers(connection)
            self._install_v8_learning_action_triggers(connection)
            self._install_release_snapshot_triggers(connection)
            self._install_corpus_registry_triggers(connection)
            self._install_current_performance_scoring_triggers(connection)
            unsafe_generated_ids = (
                self._historically_active_unsafe_generated_ids(connection)
            )
            self._revoke_historically_active_unreviewed_generated_questions(
                connection,
                unsafe_generated_ids,
            )
            self._quarantine_historical_generated_evidence(connection)
            if (
                _capture_current_schema_contract(connection)
                != _expected_current_schema_contract()
            ):
                raise ConflictError(
                    "Migrated database does not match the exact current "
                    "schema contract."
                )
            _require_no_foreign_key_violations(
                connection,
                context="Migrated database",
            )
            connection.commit()
            if legacy_foreign_keys_disabled:
                connection.execute("PRAGMA foreign_keys = ON")
            # Journal mode is operational rather than semantic. Change it
            # only after the atomic schema/data migration commits so rejected
            # sources retain their prior durable contents.
            connection.execute("PRAGMA journal_mode = WAL")
        except Exception:
            if connection.in_transaction:
                connection.rollback()
            if legacy_foreign_keys_disabled:
                connection.execute("PRAGMA foreign_keys = ON")
            raise
        finally:
            connection.close()

    @staticmethod
    def _historically_active_unsafe_generated_ids(
        connection: sqlite3.Connection,
    ) -> tuple[str, ...]:
        """Find active historical members that fail today's generation gate."""

        rows = connection.execute(
            """SELECT DISTINCT question.id, question.provenance_json,
                              membership.status
               FROM release_questions membership
               JOIN questions question ON question.id=membership.question_id
               WHERE membership.status IN ('approved', 'calibrated')
               ORDER BY question.id"""
        ).fetchall()
        unsafe: set[str] = set()
        for row in rows:
            try:
                provenance = json.loads(row["provenance_json"])
            except (TypeError, json.JSONDecodeError):
                provenance = None
            if not generated_question_runtime_safe(
                provenance,
                status=row["status"],
            ):
                unsafe.add(row["id"])
        return tuple(sorted(unsafe))

    def _enforce_historical_generated_safety(self) -> int:
        """Append idempotent revocations after exact current-schema validation."""

        with self.transaction() as connection:
            version_row = connection.execute(
                """SELECT value FROM meta WHERE key='schema_version'"""
            ).fetchone()
            if (
                version_row is None
                or version_row["value"] != str(SCHEMA_VERSION)
                or _capture_current_schema_contract(connection)
                != _expected_current_schema_contract()
            ):
                raise ConflictError(
                    "Current database schema structure changed during "
                    "safety enforcement."
                )
            _require_no_foreign_key_violations(
                connection,
                context="Current database",
            )
            revoked = self._revoke_historically_active_unreviewed_generated_questions(
                connection,
                self._historically_active_unsafe_generated_ids(connection),
            )
            self._quarantine_historical_generated_evidence(connection)
            if (
                _capture_current_schema_contract(connection)
                != _expected_current_schema_contract()
            ):
                raise ConflictError(
                    "Current database schema structure changed during "
                    "safety enforcement."
                )
            _require_no_foreign_key_violations(
                connection,
                context="Current database",
            )
            return revoked

    def _install_v4_attempt_triggers(self, connection: sqlite3.Connection) -> None:
        """Make attempts immutable except for their atomic outcome finalization.

        This is deliberately installed after migrations: a v3 attempts table does
        not have ``command_hash`` or ``outcome_json`` when the main DDL first runs.
        """
        unchanged_columns = (
            "id",
            "event_id",
            "decision_id",
            "session_id",
            "learner_id",
            "question_id",
            "question_version",
            "family_id",
            "presented_order_json",
            "selected_option_id",
            "is_correct",
            "confidence",
            "response_ms",
            "hint_count",
            "feedback_shown",
            "answered_at",
            "command_hash",
        )
        unchanged = " AND ".join(
            f"NEW.{column} IS OLD.{column}" for column in unchanged_columns
        )
        connection.execute(
            f"""CREATE TRIGGER IF NOT EXISTS attempts_no_update
                BEFORE UPDATE ON attempts
                WHEN NOT (
                    OLD.outcome_json IS NULL
                    AND NEW.outcome_json IS NOT NULL
                    AND {unchanged}
                )
                BEGIN
                    SELECT RAISE(ABORT, 'attempts are immutable after outcome finalization');
                END"""
        )
        connection.execute("DROP TRIGGER IF EXISTS attempts_validate_bounds_insert")
        connection.execute(
            f"""CREATE TRIGGER attempts_validate_bounds_insert
                BEFORE INSERT ON attempts BEGIN
                    SELECT CASE WHEN NEW.response_ms IS NOT NULL AND (
                        typeof(NEW.response_ms) != 'integer'
                        OR NEW.response_ms < 0
                        OR NEW.response_ms > {MAX_RESPONSE_MS}
                    ) THEN RAISE(ABORT, 'attempt response_ms is out of bounds') END;
                    SELECT CASE WHEN typeof(NEW.hint_count) != 'integer'
                        OR NEW.hint_count < 0
                        OR NEW.hint_count > {MAX_HINT_COUNT}
                    THEN RAISE(ABORT, 'attempt hint_count is out of bounds') END;
                END"""
        )

    @staticmethod
    def _install_release_snapshot_triggers(connection: sqlite3.Connection) -> None:
        """Prevent an activated corpus snapshot from being rewritten in place."""
        connection.execute(
            """CREATE TRIGGER IF NOT EXISTS corpus_releases_no_update
               BEFORE UPDATE ON corpus_releases
               WHEN NOT (
                   OLD.sealed_at IS NULL
                   AND NEW.sealed_at IS NOT NULL
                   AND NEW.id IS OLD.id
                   AND NEW.bundle_hash IS OLD.bundle_hash
                   AND NEW.created_at IS OLD.created_at
               ) BEGIN
                   SELECT RAISE(ABORT, 'corpus release snapshots are immutable');
               END"""
        )
        connection.execute(
            """CREATE TRIGGER IF NOT EXISTS corpus_releases_no_delete
               BEFORE DELETE ON corpus_releases BEGIN
                   SELECT RAISE(ABORT, 'corpus release snapshots are immutable');
               END"""
        )
        for table in (
            "release_concepts",
            "release_edges",
            "release_misconceptions",
            "release_sources",
            "release_questions",
            "release_learning_objectives",
            "release_objective_graphs",
            "release_objective_edges",
            "release_question_objectives",
            "release_option_objectives",
            "release_domains",
            "release_topics",
            "release_topic_concepts",
            "release_question_topics",
        ):
            connection.execute(
                f"""CREATE TRIGGER IF NOT EXISTS {table}_no_update
                    BEFORE UPDATE ON {table} BEGIN
                        SELECT RAISE(ABORT, 'corpus release snapshots are immutable');
                    END"""
            )
            connection.execute(
                f"""CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table} BEGIN
                        SELECT RAISE(ABORT, 'corpus release snapshots are immutable');
                    END"""
            )
            connection.execute(
                f"""CREATE TRIGGER IF NOT EXISTS {table}_no_insert_after_seal
                    BEFORE INSERT ON {table}
                    WHEN EXISTS (
                        SELECT 1 FROM corpus_releases release
                        WHERE release.id = NEW.release_id
                          AND release.sealed_at IS NOT NULL
                    ) BEGIN
                        SELECT RAISE(ABORT, 'sealed corpus releases cannot gain members');
                    END"""
            )
        connection.execute(
            """CREATE TRIGGER IF NOT EXISTS active_release_insert_requires_seal
               BEFORE INSERT ON meta
               WHEN NEW.key = 'active_corpus_release'
                AND NOT EXISTS (
                    SELECT 1 FROM corpus_releases release
                    WHERE release.id = NEW.value
                      AND release.sealed_at IS NOT NULL
                ) BEGIN
                    SELECT RAISE(ABORT, 'active corpus release must be sealed');
               END"""
        )
        connection.execute(
            """CREATE TRIGGER IF NOT EXISTS active_release_update_requires_seal
               BEFORE UPDATE OF value ON meta
               WHEN NEW.key = 'active_corpus_release'
                AND NOT EXISTS (
                    SELECT 1 FROM corpus_releases release
                    WHERE release.id = NEW.value
                      AND release.sealed_at IS NOT NULL
                ) BEGIN
                    SELECT RAISE(ABORT, 'active corpus release must be sealed');
               END"""
        )

    @staticmethod
    def _drop_release_snapshot_triggers(connection: sqlite3.Connection) -> None:
        for table in (
            "corpus_releases",
            "release_concepts",
            "release_edges",
            "release_misconceptions",
            "release_sources",
            "release_questions",
            "release_learning_objectives",
            "release_objective_graphs",
            "release_objective_edges",
            "release_question_objectives",
            "release_option_objectives",
            "release_domains",
            "release_topics",
            "release_topic_concepts",
            "release_question_topics",
        ):
            for suffix in ("no_update", "no_delete"):
                connection.execute(
                    f"DROP TRIGGER IF EXISTS {table}_{suffix}"
                )
            connection.execute(
                f"DROP TRIGGER IF EXISTS {table}_no_insert_after_seal"
            )
        connection.execute("DROP TRIGGER IF EXISTS active_release_insert_requires_seal")
        connection.execute("DROP TRIGGER IF EXISTS active_release_update_requires_seal")

    @staticmethod
    def _install_corpus_registry_triggers(connection: sqlite3.Connection) -> None:
        """Protect immutable corpus identities outside the Python import path.

        Corpus releases store membership while the registry stores versioned
        content.  If registry rows could be edited with ad-hoc SQL, every
        historical release referring to those IDs would silently change.  A
        question's lifecycle status is the sole mutable registry field; content
        changes require a new question ID and ``revision_of`` link.
        """
        for table in (
            "concepts",
            "learning_objectives",
            "misconceptions",
            "sources",
        ):
            connection.execute(
                f"""CREATE TRIGGER IF NOT EXISTS {table}_no_update
                    BEFORE UPDATE ON {table} BEGIN
                        SELECT RAISE(ABORT, 'versioned corpus registry rows are immutable');
                    END"""
            )
            connection.execute(
                f"""CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table} BEGIN
                        SELECT RAISE(ABORT, 'versioned corpus registry rows are immutable');
                    END"""
            )
        immutable_question_columns = (
            "id",
            "version",
            "content_hash",
            "family_id",
            "stem",
            "kind",
            "difficulty",
            "discrimination",
            "guess_rate",
            "slip_rate",
            "provenance_json",
            "tags_json",
            "revision_of",
            "imported_at",
        )
        unchanged = " AND ".join(
            f"NEW.{column} IS OLD.{column}"
            for column in immutable_question_columns
        )
        connection.execute(
            f"""CREATE TRIGGER IF NOT EXISTS questions_immutable_content
                BEFORE UPDATE ON questions
                WHEN NOT ({unchanged}) BEGIN
                    SELECT RAISE(ABORT, 'published question content is immutable');
                END"""
        )
        allowed_statuses = ", ".join(
            f"'{status.value}'" for status in QuestionStatus
        )
        connection.execute(
            f"""CREATE TRIGGER IF NOT EXISTS questions_valid_status
                BEFORE UPDATE OF status ON questions
                WHEN NEW.status NOT IN ({allowed_statuses}) BEGIN
                    SELECT RAISE(ABORT, 'invalid question lifecycle status');
                END"""
        )
        connection.execute(
            """CREATE TRIGGER IF NOT EXISTS questions_no_delete
                BEFORE DELETE ON questions BEGIN
                    SELECT RAISE(ABORT, 'published questions are immutable');
                END"""
        )

        for table in ("question_concepts", "options", "question_sources"):
            connection.execute(
                f"""CREATE TRIGGER IF NOT EXISTS {table}_no_update
                    BEFORE UPDATE ON {table} BEGIN
                        SELECT RAISE(ABORT, 'published question components are immutable');
                    END"""
            )
            connection.execute(
                f"""CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table} BEGIN
                        SELECT RAISE(ABORT, 'published question components are immutable');
                    END"""
            )
            connection.execute(
                f"""CREATE TRIGGER IF NOT EXISTS {table}_no_insert_after_release
                    BEFORE INSERT ON {table}
                    WHEN EXISTS (
                        SELECT 1 FROM release_questions membership
                        JOIN corpus_releases release
                          ON release.id = membership.release_id
                        WHERE membership.question_id = NEW.question_id
                          AND release.sealed_at IS NOT NULL
                    ) BEGIN
                        SELECT RAISE(ABORT, 'released question components are sealed');
                    END"""
            )

    @staticmethod
    def _drop_corpus_registry_triggers(connection: sqlite3.Connection) -> None:
        names = [
            *(
                f"{table}_{suffix}"
                for table in (
                    "concepts",
                    "learning_objectives",
                    "misconceptions",
                    "sources",
                )
                for suffix in ("no_update", "no_delete")
            ),
            "questions_immutable_content",
            "questions_valid_status",
            "questions_no_delete",
            *(
                f"{table}_{suffix}"
                for table in ("question_concepts", "options", "question_sources")
                for suffix in ("no_update", "no_delete", "no_insert_after_release")
            ),
        ]
        for name in names:
            connection.execute(f"DROP TRIGGER IF EXISTS {name}")

    def _migrate_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        question_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(questions)").fetchall()
        }
        if "content_hash" not in question_columns:
            connection.execute("ALTER TABLE questions ADD COLUMN content_hash TEXT")
            rows = connection.execute("SELECT * FROM questions ORDER BY id").fetchall()
            for row in rows:
                question = self._question_from_row(connection, row)
                connection.execute(
                    "UPDATE questions SET content_hash = ? WHERE id = ?",
                    (question_content_hash(question), question.id),
                )
        family_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(learner_skill_families)").fetchall()
        }
        if "delayed_unguided_correct_at" not in family_columns:
            connection.execute(
                "ALTER TABLE learner_skill_families ADD COLUMN delayed_unguided_correct_at TEXT"
            )

    def _migrate_v2_to_v3(self, connection: sqlite3.Connection) -> None:
        family_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(learner_skill_families)").fetchall()
        }
        if "kind" not in family_columns:
            connection.execute("ALTER TABLE learner_skill_families ADD COLUMN kind TEXT")
            connection.execute(
                """UPDATE learner_skill_families AS evidence
                   SET kind = COALESCE((
                       SELECT q.kind FROM questions q
                       WHERE q.family_id = evidence.family_id ORDER BY q.id LIMIT 1
                   ), 'unknown')"""
            )

    def _migrate_v3_to_v4(self, connection: sqlite3.Connection) -> None:
        connection.execute("DROP TRIGGER IF EXISTS attempts_no_update")

        def add_column(table: str, name: str, declaration: str) -> None:
            columns = {
                row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if name not in columns:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

        add_column("concepts", "content_hash", "TEXT")
        add_column("misconceptions", "content_hash", "TEXT")
        add_column("sources", "content_hash", "TEXT")
        add_column("learners", "revision", "INTEGER NOT NULL DEFAULT 0")
        add_column("sessions", "revision", "INTEGER NOT NULL DEFAULT 0")
        add_column("sessions", "corpus_release_id", "TEXT")
        for row in connection.execute("SELECT * FROM concepts").fetchall():
            concept = Concept(
                row["id"], row["name"], row["description"], row["domain"], row["prior_mastery"]
            )
            connection.execute(
                "UPDATE concepts SET content_hash = ? WHERE id = ?",
                (concept_content_hash(concept), concept.id),
            )
        for row in connection.execute("SELECT * FROM misconceptions").fetchall():
            misconception = Misconception(
                row["id"], row["concept_id"], row["name"], row["description"]
            )
            connection.execute(
                "UPDATE misconceptions SET content_hash = ? WHERE id = ?",
                (misconception_content_hash(misconception), misconception.id),
            )
        for row in connection.execute("SELECT * FROM sources").fetchall():
            source = Source(
                row["id"],
                row["title"],
                row["uri"],
                row["license"],
                json.loads(row["metadata_json"]),
            )
            connection.execute(
                "UPDATE sources SET content_hash = ? WHERE id = ?",
                (source_content_hash(source), source.id),
            )

        release_payload = {
            "concepts": [
                tuple(row)
                for row in connection.execute(
                    "SELECT id, content_hash FROM concepts ORDER BY id"
                ).fetchall()
            ],
            "edges": [
                tuple(row)
                for row in connection.execute(
                    "SELECT source_id, target_id, relation, weight FROM concept_edges ORDER BY 1,2,3"
                ).fetchall()
            ],
            "misconceptions": [
                tuple(row)
                for row in connection.execute(
                    "SELECT id, content_hash FROM misconceptions ORDER BY id"
                ).fetchall()
            ],
            "sources": [
                tuple(row)
                for row in connection.execute(
                    "SELECT id, content_hash FROM sources ORDER BY id"
                ).fetchall()
            ],
            "questions": [
                tuple(row)
                for row in connection.execute(
                    "SELECT id, content_hash, status FROM questions ORDER BY id"
                ).fetchall()
            ],
        }
        bundle_hash = _content_hash(release_payload)
        release_id = f"rel_{bundle_hash[:24]}"
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            "INSERT OR IGNORE INTO corpus_releases(id, bundle_hash, created_at) VALUES (?, ?, ?)",
            (release_id, bundle_hash, now),
        )
        connection.execute("DELETE FROM release_edges WHERE release_id = ?", (release_id,))
        connection.execute(
            """INSERT OR IGNORE INTO release_edges(release_id, source_id, target_id, relation, weight)
               SELECT ?, source_id, target_id, relation, weight FROM concept_edges""",
            (release_id,),
        )
        connection.execute(
            "DELETE FROM release_concepts WHERE release_id = ?", (release_id,)
        )
        connection.execute(
            """INSERT OR IGNORE INTO release_concepts(release_id, concept_id)
               SELECT ?, id FROM concepts""",
            (release_id,),
        )
        connection.execute(
            """INSERT OR IGNORE INTO release_misconceptions(
                   release_id, misconception_id
               ) SELECT ?, id FROM misconceptions""",
            (release_id,),
        )
        connection.execute(
            """INSERT OR IGNORE INTO release_sources(release_id, source_id)
               SELECT ?, id FROM sources""",
            (release_id,),
        )
        connection.execute("DELETE FROM release_questions WHERE release_id = ?", (release_id,))
        for row in connection.execute("SELECT id, status FROM questions").fetchall():
            status = QuestionStatus(row["status"])
            connection.execute(
                """INSERT OR IGNORE INTO release_questions(
                       release_id, question_id, status, evidence_weight
                   ) VALUES (?, ?, ?, ?)""",
                (release_id, row["id"], status.value, status.evidence_weight),
            )
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('active_corpus_release', ?)",
            (release_id,),
        )
        connection.execute(
            "UPDATE sessions SET corpus_release_id = COALESCE(corpus_release_id, ?)",
            (release_id,),
        )

        decision_columns = {
            "question_version": "INTEGER",
            "question_content_hash": "TEXT",
            "question_status": "TEXT",
            "evidence_weight": "REAL",
            "corpus_release_id": "TEXT",
            "session_revision": "INTEGER",
            "learner_revision": "INTEGER",
            "focus_concept_id": "TEXT",
            "focus_misconception_id": "TEXT",
            "pedagogical_role": "TEXT",
            "focus_valid": "INTEGER",
        }
        for name, declaration in decision_columns.items():
            add_column("decisions", name, declaration)
        connection.execute(
            """UPDATE decisions SET
                   question_version = COALESCE(question_version, (SELECT version FROM questions q WHERE q.id=question_id)),
                   question_content_hash = COALESCE(question_content_hash, (SELECT content_hash FROM questions q WHERE q.id=question_id)),
                   question_status = COALESCE(question_status, (SELECT status FROM questions q WHERE q.id=question_id)),
                   evidence_weight = COALESCE(evidence_weight, 0.65),
                   corpus_release_id = COALESCE(corpus_release_id, ?),
                   session_revision = COALESCE(session_revision, 0),
                   learner_revision = COALESCE(learner_revision, 0),
                   focus_concept_id = COALESCE(focus_concept_id, (SELECT focus_concept_id FROM sessions s WHERE s.id=session_id)),
                   focus_misconception_id = COALESCE(focus_misconception_id, (SELECT focus_misconception_id FROM sessions s WHERE s.id=session_id)),
                   pedagogical_role = COALESCE(pedagogical_role, phase),
                   focus_valid = COALESCE(focus_valid, 1)""",
            (release_id,),
        )
        selection_field_map = {
            "question_version": "question_version",
            "question_content_hash": "question_content_hash",
            "question_status": "question_status",
            "evidence_weight": "evidence_weight",
            "corpus_release_id": "corpus_release_id",
            "session_revision": "session_revision",
            "learner_revision": "learner_revision",
            "focus_concept_id": "focus_concept_id",
            "focus_misconception_id": "focus_misconception_id",
            "pedagogical_role": "pedagogical_role",
            "focus_valid": "focus_valid",
        }
        for event in connection.execute(
            "SELECT payload_json FROM events WHERE event_type = 'QuestionSelected'"
        ).fetchall():
            try:
                payload = json.loads(event["payload_json"])
            except json.JSONDecodeError:
                continue
            decision_id = payload.get("decision_id") if isinstance(payload, dict) else None
            if not decision_id:
                continue
            updates = {
                column: (int(bool(payload[field])) if field == "focus_valid" else payload[field])
                for field, column in selection_field_map.items()
                if field in payload
            }
            if not updates:
                continue
            assignments = ", ".join(f"{column} = ?" for column in updates)
            connection.execute(
                f"UPDATE decisions SET {assignments} WHERE id = ?",
                (*updates.values(), decision_id),
            )
        add_column("attempts", "command_hash", "TEXT")
        add_column("attempts", "outcome_json", "TEXT")
        for row in connection.execute(
            """SELECT a.id, e.payload_json FROM attempts a
               JOIN events e ON e.event_id = a.event_id WHERE a.command_hash IS NULL"""
        ).fetchall():
            connection.execute(
                "UPDATE attempts SET command_hash = ? WHERE id = ?",
                (_content_hash(json.loads(row["payload_json"])), row["id"]),
            )
        connection.execute("DROP TRIGGER IF EXISTS events_no_update")
        connection.execute("DROP TRIGGER IF EXISTS events_no_delete")
        for event in connection.execute(
            """SELECT event_id, event_type, payload_json, metadata_json
               FROM events
               WHERE event_type IN ('QuestionSelected', 'ResponseSubmitted')"""
        ).fetchall():
            try:
                payload = json.loads(event["payload_json"])
                metadata = json.loads(event["metadata_json"])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or not isinstance(metadata, dict):
                continue
            decision_id = payload.get("decision_id")
            if not decision_id:
                continue
            decision = connection.execute(
                "SELECT * FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
            if not decision:
                continue
            if event["event_type"] == "QuestionSelected":
                selection_snapshot = {
                    "decision_id": decision["id"],
                    "question_id": decision["question_id"],
                    "phase": decision["phase"],
                    "candidate_count": decision["candidate_count"],
                    "candidate_digest": decision["candidate_digest"],
                    "propensity": decision["propensity"],
                    "score": json.loads(decision["selected_score_json"]),
                    "option_order": json.loads(decision["option_order_json"]),
                    "question_version": decision["question_version"],
                    "question_content_hash": decision["question_content_hash"],
                    "question_status": decision["question_status"],
                    "evidence_weight": decision["evidence_weight"],
                    "corpus_release_id": decision["corpus_release_id"],
                    "session_revision": decision["session_revision"],
                    "learner_revision": decision["learner_revision"],
                    "focus_concept_id": decision["focus_concept_id"],
                    "focus_misconception_id": decision["focus_misconception_id"],
                    "pedagogical_role": decision["pedagogical_role"],
                    "focus_valid": bool(decision["focus_valid"]),
                }
                for field, value in selection_snapshot.items():
                    payload.setdefault(field, value)
                metadata.setdefault("policy_version", decision["policy_version"])
                metadata.setdefault("corpus_release_id", decision["corpus_release_id"])
            else:
                response_snapshot = {
                    "policy_version": decision["policy_version"],
                    "corpus_release_id": decision["corpus_release_id"],
                    "question_content_hash": decision["question_content_hash"],
                    "question_status": decision["question_status"],
                    "evidence_weight": decision["evidence_weight"],
                    "selection_learner_revision": decision["learner_revision"],
                }
                for field, value in response_snapshot.items():
                    metadata.setdefault(field, value)
            connection.execute(
                "UPDATE events SET payload_json = ?, metadata_json = ? WHERE event_id = ?",
                (
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                    event["event_id"],
                ),
            )
        # Session and learner-projection events predate sealed corpus
        # releases.  The v3 database has exactly one reconstructable release,
        # so bind those historical envelopes to it before the append-only
        # triggers and stream heads are restored.
        for event in connection.execute(
            """SELECT event_id, event_type, payload_json, metadata_json
               FROM events
               WHERE event_type IN (
                   'SessionStarted', 'LearnerProjectionAdvanced',
                   'RemediationTransitioned'
               )"""
        ).fetchall():
            try:
                payload = json.loads(event["payload_json"])
                metadata = json.loads(event["metadata_json"])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict) or not isinstance(metadata, dict):
                continue
            if event["event_type"] in {
                "SessionStarted",
                "LearnerProjectionAdvanced",
            }:
                payload.setdefault("corpus_release_id", release_id)
            metadata.setdefault("corpus_release_id", release_id)
            connection.execute(
                """UPDATE events SET payload_json = ?, metadata_json = ?
                   WHERE event_id = ?""",
                (
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                    event["event_id"],
                ),
            )

        migrated_learner_revisions: dict[str, int] = {}
        for event in connection.execute(
            """SELECT learner_id, payload_json FROM events
               WHERE event_type = 'LearnerProjectionAdvanced'
                 AND learner_id IS NOT NULL
               ORDER BY stream_id, stream_version"""
        ).fetchall():
            try:
                payload = json.loads(event["payload_json"])
            except json.JSONDecodeError:
                continue
            revision = (
                payload.get("learner_revision")
                if isinstance(payload, dict)
                else None
            )
            if (
                type(revision) is int
                and revision >= 0
                and revision
                >= migrated_learner_revisions.get(event["learner_id"], 0)
            ):
                migrated_learner_revisions[event["learner_id"]] = revision
        connection.executemany(
            "UPDATE learners SET revision = ? WHERE id = ?",
            (
                (revision, learner_id)
                for learner_id, revision in sorted(
                    migrated_learner_revisions.items()
                )
            ),
        )

        def option_payload(option: Option | None) -> dict[str, Any] | None:
            if option is None:
                return None
            result: dict[str, Any] = {
                "id": option.id,
                "text": option.text,
                "correct": option.correct,
                "rationale": option.rationale,
                "misconception_id": option.misconception_id,
            }
            if option.diagnostic_objective_id is not None:
                result["diagnostic_objective_id"] = (
                    option.diagnostic_objective_id
                )
            return result

        # The old attempts projection had no finalized retry payload.  Its
        # immutable response and immediately caused learner projection contain
        # every observable outcome term, so reconstruct that payload instead
        # of leaving a successful migration with permanently non-idempotent
        # responses.
        for row in connection.execute(
            """SELECT attempt.*, projection.payload_json AS projection_payload
               FROM attempts attempt
               JOIN events response ON response.event_id = attempt.event_id
               LEFT JOIN events projection
                 ON projection.event_type = 'LearnerProjectionAdvanced'
                AND projection.causation_id = response.event_id
               WHERE attempt.outcome_json IS NULL"""
        ).fetchall():
            if row["projection_payload"] is None:
                continue
            try:
                projection_payload = json.loads(row["projection_payload"])
            except json.JSONDecodeError:
                continue
            if not isinstance(projection_payload, dict):
                continue
            question = self.get_question(row["question_id"], connection)
            selected_option = next(
                (
                    option
                    for option in question.options
                    if option.id == row["selected_option_id"]
                ),
                None,
            )
            outcome: dict[str, Any] = {
                "interaction_id": row["id"],
                "correct": bool(row["is_correct"]),
                "selected_option": option_payload(selected_option),
                "correct_option": option_payload(question.correct_option),
                "next_phase": projection_payload.get("phase"),
                "focus_concept_id": projection_payload.get(
                    "focus_concept_id"
                ),
                "focus_misconception_id": projection_payload.get(
                    "focus_misconception_id"
                ),
                "state_changes": projection_payload.get("state_changes", []),
                "transition_reason": projection_payload.get(
                    "transition_reason"
                ),
                "boundary_decision": projection_payload.get(
                    "boundary_decision"
                ),
            }
            for field in (
                "focus_objective_id",
                "remediation_depth",
                "remediation_path",
            ):
                if field in projection_payload:
                    outcome[field] = projection_payload[field]
            connection.execute(
                "UPDATE attempts SET outcome_json = ? WHERE id = ?",
                (
                    json.dumps(outcome, sort_keys=True, separators=(",", ":")),
                    row["id"],
                ),
            )
        streams = [
            row["stream_id"]
            for row in connection.execute("SELECT DISTINCT stream_id FROM events").fetchall()
        ]
        for stream_id in streams:
            previous_hash = None
            events = connection.execute(
                "SELECT * FROM events WHERE stream_id = ? ORDER BY stream_version",
                (stream_id,),
            ).fetchall()
            for event in events:
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
                    "payload": json.loads(event["payload_json"]),
                    "metadata": json.loads(event["metadata_json"]),
                    "previous_hash": previous_hash,
                }
                payload_hash = _content_hash(envelope)
                connection.execute(
                    "UPDATE events SET previous_hash = ?, payload_hash = ? WHERE event_id = ?",
                    (previous_hash, payload_hash, event["event_id"]),
                )
                previous_hash = payload_hash
        _execute_sql_script(
            connection,
            """CREATE TRIGGER events_no_update
               BEFORE UPDATE ON events BEGIN
                   SELECT RAISE(ABORT, 'events are append-only');
               END;
               CREATE TRIGGER events_no_delete
               BEFORE DELETE ON events BEGIN
                   SELECT RAISE(ABORT, 'events are append-only');
               END;""",
        )
        connection.execute("DELETE FROM stream_heads")
        connection.execute(
            """INSERT INTO stream_heads(stream_id, stream_version, payload_hash, updated_at)
               SELECT e.stream_id, e.stream_version, e.payload_hash, ?
               FROM events e JOIN (
                   SELECT stream_id, MAX(stream_version) AS version FROM events GROUP BY stream_id
               ) latest ON latest.stream_id=e.stream_id AND latest.version=e.stream_version""",
            (now,),
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_id, question_id, family_id)"
        )

    def _migrate_v4_to_v5(
        self,
        connection: sqlite3.Connection,
        *,
        had_release_concepts: bool,
        had_release_misconceptions: bool,
        had_release_sources: bool,
        starting_version: int,
    ) -> None:
        """Add sealed releases, resumable remediation state, and stale-decision state."""

        def add_column(table: str, name: str, declaration: str) -> None:
            columns = {
                row["name"]
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                )

        add_column(
            "sessions",
            "remediation_path_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )
        add_column("corpus_releases", "sealed_at", "TEXT")
        add_column("decisions", "invalidated_at", "TEXT")
        add_column("decisions", "invalidation_reason", "TEXT")

        # Some prerelease v4 builds omitted these manifest tables.  Exact
        # historical membership cannot be inferred from today's accumulated
        # registry, so fail closed instead of fabricating every release via a
        # cross join.  Databases migrated from v1-v3 are safe: the v3->v4 step
        # immediately above created the exact single-release manifests.
        if starting_version == 4 and not all(
            (
                had_release_concepts,
                had_release_misconceptions,
                had_release_sources,
            )
        ):
            release_count = connection.execute(
                "SELECT COUNT(*) AS n FROM corpus_releases"
            ).fetchone()["n"]
            if release_count:
                raise ConflictError(
                    "Legacy v4 corpus manifests are incomplete and cannot be "
                    "reconstructed safely; export learner events and re-import "
                    "the original corpus bundles."
                )
        connection.execute(
            "UPDATE corpus_releases SET sealed_at = COALESCE(sealed_at, created_at)"
        )
        connection.execute("DROP INDEX IF EXISTS idx_one_pending_decision")

    @staticmethod
    def _migrate_v5_to_v6(connection: sqlite3.Connection) -> None:
        """Backfill immutable execution history for legacy authoring jobs.

        Schema v5 retained only the latest mutable job summary.  A terminal
        summary can be preserved as attempt one, but an interrupted ``running``
        row cannot be proven complete and is therefore migrated to ``failed``.
        Planned jobs have no execution history to backfill.
        """

        def strict_object(raw: str | None) -> dict[str, Any] | None:
            if raw is None:
                return None

            def reject_constant(value: str) -> None:
                raise ValueError(f"non-finite JSON constant {value}")

            try:
                value = json.loads(raw, parse_constant=reject_constant)
            except (TypeError, ValueError):
                return None
            return value if type(value) is dict else None

        def canonical(value: dict[str, Any]) -> str:
            return json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )

        for row in connection.execute(
            "SELECT * FROM generation_jobs ORDER BY created_at, id"
        ).fetchall():
            if connection.execute(
                "SELECT 1 FROM generation_job_runs WHERE job_id = ? LIMIT 1",
                (row["id"],),
            ).fetchone():
                continue
            if row["status"] == "planned" and all(
                row[field] is None
                for field in ("provider", "model", "raw_output_json", "validation_json")
            ):
                continue

            provider = (
                row["provider"].strip()
                if type(row["provider"]) is str and row["provider"].strip()
                else "legacy-unknown-provider"
            )
            model = (
                row["model"].strip()
                if type(row["model"]) is str and row["model"].strip()
                else "legacy-unknown-model"
            )
            prompt_version = (
                row["prompt_version"].strip()
                if type(row["prompt_version"]) is str and row["prompt_version"].strip()
                else "legacy-unknown-prompt"
            )
            validation = strict_object(row["validation_json"])
            raw_output = strict_object(row["raw_output_json"])
            declared_context_hash = (
                validation.get("source_context_sha256")
                if validation is not None
                else None
            )
            if not (
                type(declared_context_hash) is str
                and len(declared_context_hash) == 64
                and all(character in "0123456789abcdef" for character in declared_context_hash)
            ):
                declared_context_hash = hashlib.sha256(
                    f"legacy source context unavailable:{row['id']}".encode("utf-8")
                ).hexdigest()

            legacy_status = row["status"]
            terminal_status = legacy_status
            error_record: dict[str, Any] | None = None
            if legacy_status not in {"reviewed", "rejected", "failed"}:
                terminal_status = "failed"
                error_record = {
                    "error_type": "LegacyGenerationJobState",
                    "error": (
                        f"Legacy job state {legacy_status!r} was not a complete, "
                        "verifiable execution and requires an explicit retry."
                    ),
                }
            elif legacy_status in {"reviewed", "rejected"} and (
                raw_output is None or validation is None
            ):
                terminal_status = "failed"
                error_record = {
                    "error_type": "LegacyGenerationJobState",
                    "error": (
                        "Legacy terminal authoring output was missing or invalid; "
                        "the job requires an explicit retry."
                    ),
                }
            elif legacy_status == "failed":
                error_record = {
                    "error_type": "LegacyGenerationJobFailure",
                    "error": (
                        str(validation.get("error"))
                        if validation is not None and validation.get("error") is not None
                        else "Legacy authoring attempt failed before schema v6."
                    ),
                }

            completed_at = row["updated_at"] or row["created_at"]
            run_id = "run_legacy_" + hashlib.sha256(
                row["id"].encode("utf-8")
            ).hexdigest()[:24]
            if terminal_status == "failed":
                assert error_record is not None
                validation_json = canonical(error_record)
                connection.execute(
                    """UPDATE generation_jobs
                       SET status='failed', provider=?, model=?, prompt_version=?,
                           raw_output_json=NULL, validation_json=?, updated_at=?
                       WHERE id=?""",
                    (
                        provider,
                        model,
                        prompt_version,
                        validation_json,
                        completed_at,
                        row["id"],
                    ),
                )
                raw_json = None
                run_validation_json = None
                error_json = validation_json
            else:
                raw_json = row["raw_output_json"]
                run_validation_json = row["validation_json"]
                error_json = None
                connection.execute(
                    """UPDATE generation_jobs SET provider=?, model=?, prompt_version=?
                       WHERE id=?""",
                    (provider, model, prompt_version, row["id"]),
                )
            connection.execute(
                """INSERT INTO generation_job_runs(
                       id, job_id, attempt, status, provider, model, prompt_version,
                       source_context_sha256, raw_output_json, validation_json,
                       error_json, started_at, completed_at
                   ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    row["id"],
                    terminal_status,
                    provider,
                    model,
                    prompt_version,
                    declared_context_hash,
                    raw_json,
                    run_validation_json,
                    error_json,
                    row["created_at"],
                    completed_at,
                ),
            )

    @staticmethod
    def _migrate_v6_to_v7(connection: sqlite3.Connection) -> None:
        """Add release-pinned curriculum catalogs and exploration state.

        Historical releases intentionally remain catalog-less: their bundle
        hashes and session meanings are left untouched.  A subsequent corpus
        import can publish a catalog as part of a new immutable release.
        """

        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(sessions)")
        }
        if "topic_id" not in columns:
            connection.execute("ALTER TABLE sessions ADD COLUMN topic_id TEXT")
        if "exploration_mode" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN exploration_mode TEXT NOT NULL DEFAULT 'off'"
            )

    @staticmethod
    def _migrate_v7_to_v8(connection: sqlite3.Connection) -> None:
        """Add an observational, append-only learner-action ledger.

        The v8 tables are created by the main DDL before this migration runs.
        Historical decisions intentionally receive no fabricated action rows:
        absence is neutral evidence, and old event streams remain byte-for-byte
        unchanged.
        """
        required = {"learning_artifacts", "learning_actions"}
        present = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = required - present
        if missing:
            raise ConflictError(
                "Schema v8 action tables were not installed: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _migrate_v8_to_v9(connection: sqlite3.Connection) -> None:
        """Add release-pinned learning objectives without fabricating history.

        Historical releases intentionally have no objective membership.  Their
        question hashes, manifests, event streams, and learner projections stay
        byte-for-byte meaningful.  The first corpus-v2 import publishes an
        objective catalog and exact question mappings in a new sealed release.
        """

        def add_column(table: str, name: str, declaration: str) -> None:
            columns = {
                row["name"]
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                )

        add_column(
            "sessions",
            "focus_objective_id",
            "TEXT REFERENCES learning_objectives(id)",
        )
        add_column(
            "decisions",
            "focus_objective_id",
            "TEXT REFERENCES learning_objectives(id)",
        )
        add_column(
            "decisions",
            "question_objective_id",
            "TEXT REFERENCES learning_objectives(id)",
        )
        required = {
            "learning_objectives",
            "release_learning_objectives",
            "release_question_objectives",
            "release_option_objectives",
            "objective_states",
            "learner_objective_families",
        }
        present = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = required - present
        if missing:
            raise ConflictError(
                "Schema v9 objective tables were not installed: "
                + ", ".join(sorted(missing))
            )

    def _migrate_v9_to_v10(self, connection: sqlite3.Connection) -> None:
        """Close legacy pending decisions whose answer promise is no longer live.

        Older session shutdowns and cross-session learner updates could leave a
        decision marked pending after its pinned session or learner projection
        had advanced.  Preserve the original selection and append an explicit
        invalidation boundary; never fabricate an answer or delete history.
        """

        pending = connection.execute(
            """SELECT decision.*,
                      session.status AS session_status,
                      session.phase AS current_phase,
                      session.focus_concept_id AS current_focus_concept_id,
                      session.focus_misconception_id
                          AS current_focus_misconception_id,
                      session.focus_objective_id
                          AS current_focus_objective_id,
                      session.corpus_release_id AS current_release_id,
                      session.revision AS current_session_revision,
                      learner.revision AS current_learner_revision,
                      revocation.question_id AS revoked_question_id
               FROM decisions decision
               JOIN sessions session ON session.id = decision.session_id
               JOIN learners learner ON learner.id = decision.learner_id
               LEFT JOIN question_revocations revocation
                 ON revocation.question_id = decision.question_id
               WHERE decision.consumed_at IS NULL
                 AND decision.invalidated_at IS NULL
               ORDER BY decision.created_at, decision.id"""
        ).fetchall()
        repaired_at = datetime.now(timezone.utc)
        repaired_timestamp = to_timestamp(repaired_at)
        for decision in pending:
            reason: str | None = None
            if decision["revoked_question_id"] is not None:
                reason = "question_emergency_revoked"
            elif decision["session_status"] != "active":
                reason = "legacy_session_inactive"
            elif any(
                (
                    decision["current_phase"] != decision["phase"],
                    decision["current_focus_concept_id"]
                    != decision["focus_concept_id"],
                    decision["current_focus_misconception_id"]
                    != decision["focus_misconception_id"],
                    decision["current_focus_objective_id"]
                    != decision["focus_objective_id"],
                    decision["current_release_id"]
                    != decision["corpus_release_id"],
                    decision["current_session_revision"]
                    != decision["session_revision"] + 1,
                )
            ):
                reason = "legacy_session_state_changed"
            elif (
                decision["current_learner_revision"]
                != decision["learner_revision"]
            ):
                reason = "learner_projection_advanced"
            if reason is None:
                continue
            if (
                decision["current_learner_revision"]
                < decision["learner_revision"]
            ):
                raise ConflictError(
                    f"Pending decision {decision['id']} is ahead of its learner "
                    "projection; migration cannot repair it safely."
                )
            selection_events = connection.execute(
                """SELECT metadata_json FROM events
                   WHERE event_type = 'QuestionSelected'
                     AND session_id = ?
                     AND json_extract(payload_json, '$.decision_id') = ?
                   ORDER BY stream_version""",
                (decision["session_id"], decision["id"]),
            ).fetchall()
            if len(selection_events) != 1:
                raise ConflictError(
                    f"Pending decision {decision['id']} lacks a unique selection "
                    "boundary; migration cannot repair it safely."
                )
            try:
                metadata = json.loads(selection_events[0]["metadata_json"])
            except (TypeError, ValueError) as exc:
                raise ConflictError(
                    f"Pending decision {decision['id']} has invalid selection "
                    "metadata; migration cannot repair it safely."
                ) from exc
            if (
                type(metadata) is not dict
                or metadata.get("policy_version") != decision["policy_version"]
                or metadata.get("corpus_release_id")
                != decision["corpus_release_id"]
                or not isinstance(metadata.get("learner_model_version"), str)
                or not metadata["learner_model_version"]
            ):
                raise ConflictError(
                    f"Pending decision {decision['id']} selection metadata is "
                    "inconsistent; migration cannot repair it safely."
                )
            updated = connection.execute(
                """UPDATE decisions
                   SET invalidated_at = ?, invalidation_reason = ?
                   WHERE id = ? AND consumed_at IS NULL
                     AND invalidated_at IS NULL""",
                (repaired_timestamp, reason, decision["id"]),
            )
            if updated.rowcount != 1:
                raise ConflictError(
                    f"Pending decision {decision['id']} changed during migration."
                )
            self.append_event(
                connection,
                stream_id=f"learner:{decision['learner_id']}",
                event_type="DecisionInvalidated",
                payload={
                    "decision_id": decision["id"],
                    "reason": reason,
                    "selection_learner_revision": decision[
                        "learner_revision"
                    ],
                    "current_learner_revision": decision[
                        "current_learner_revision"
                    ],
                },
                metadata={
                    "policy_version": decision["policy_version"],
                    "learner_model_version": metadata[
                        "learner_model_version"
                    ],
                    "corpus_release_id": decision["corpus_release_id"],
                },
                learner_id=decision["learner_id"],
                session_id=decision["session_id"],
                causation_id=decision["id"],
                occurred_at=repaired_at,
            )

    @staticmethod
    def _migrate_v10_to_v11(connection: sqlite3.Connection) -> None:
        """Install release-pinned objective prerequisite graph manifests.

        Historical releases deliberately receive no capability row.  This is
        how the engine distinguishes an old release whose objective graph was
        never declared from a new release declaring an intentionally empty
        graph.
        """

        required = {"release_objective_graphs", "release_objective_edges"}
        present = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = required - present
        if missing:
            raise ConflictError(
                "Schema v11 objective graph tables were not installed: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _migrate_v11_to_v12(connection: sqlite3.Connection) -> None:
        """Install exact objective posteriors without inventing old evidence.

        Existing v5 objective rows deliberately remain parent-only.  A later
        v6 response lifts a v5 Gaussian once and writes the exact child in the
        same transaction as the new parent state.  Fabricating child rows here
        would erase that causal model-version boundary from replay.
        """

        present = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "objective_grid_states" not in present:
            raise ConflictError(
                "Schema v12 objective-grid state table was not installed."
            )
        expected_columns = {
            "learner_id",
            "objective_id",
            "posterior_schema_version",
            "algorithm",
            "grid_id",
            "codec",
            "posterior_blob",
            "posterior_sha256",
            "mean",
            "variance",
            "mastery_probability",
            "expected_competence",
            "edge_mass",
            "mastery_probability_error_bound",
            "evidence_mass",
            "acquisition_mass",
            "as_of_event_id",
            "model_version",
        }
        table_info = connection.execute(
            "PRAGMA table_info(objective_grid_states)"
        ).fetchall()
        actual_columns = {row["name"] for row in table_info}
        if actual_columns != expected_columns:
            missing = sorted(expected_columns - actual_columns)
            unexpected = sorted(actual_columns - expected_columns)
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unexpected:
                details.append("unexpected " + ", ".join(unexpected))
            raise ConflictError(
                "Schema v12 objective-grid columns are incompatible ("
                + "; ".join(details)
                + ")."
            )
        primary_key = {
            row["name"]: row["pk"] for row in table_info if row["pk"]
        }
        if primary_key != {"learner_id": 1, "objective_id": 2}:
            raise ConflictError(
                "Schema v12 objective-grid primary key is incompatible."
            )
        foreign_keys = [
            dict(row)
            for row in connection.execute(
                "PRAGMA foreign_key_list(objective_grid_states)"
            )
        ]
        objective_links = {
            (row["from"], row["to"], row["on_delete"])
            for row in foreign_keys
            if row["table"] == "objective_states"
        }
        event_links = {
            (row["from"], row["to"], row["on_delete"])
            for row in foreign_keys
            if row["table"] == "events"
        }
        if objective_links != {
            ("learner_id", "learner_id", "CASCADE"),
            ("objective_id", "objective_id", "CASCADE"),
        } or event_links != {
            ("as_of_event_id", "event_id", "NO ACTION")
        }:
            raise ConflictError(
                "Schema v12 objective-grid foreign keys are incompatible."
            )

    def _migrate_v12_to_v13(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        """Canonicalize every table shaped by legacy ALTER migrations.

        The legacy upgrade path added required fields as nullable trailing
        columns. Values were backfilled correctly, but the resulting table
        definitions still differed in nullability, order, and foreign-key
        declarations. Rebuild only definitions that differ from current DDL,
        with explicit column sets and a caller-owned transaction. ``initialize``
        disables foreign-key enforcement before its serialized transaction for
        pre-v13 sources and performs a complete check before committing.
        """

        canonicalizable = (
            "attempts",
            "concepts",
            "decisions",
            "learner_skill_families",
            "learners",
            "misconceptions",
            "questions",
            "sessions",
            "sources",
        )
        plans: list[
            tuple[str, str, tuple[str, ...], tuple[str, ...]]
        ] = []
        for table in canonicalizable:
            canonical_tail, canonical_columns, canonical_indexes = (
                _canonical_table_sql_bundle(table)
            )
            quoted = _quote_sqlite_identifier(table)
            table_row = connection.execute(
                """SELECT sql FROM sqlite_master
                   WHERE type='table' AND name=?""",
                (table,),
            ).fetchone()
            if table_row is None or not table_row["sql"]:
                raise ConflictError(
                    f"Schema v13 cannot find required table {table}."
                )
            actual_columns = tuple(
                row["name"]
                for row in sorted(
                    connection.execute(
                        f"PRAGMA table_xinfo({quoted})"
                    ).fetchall(),
                    key=lambda item: int(item["cid"]),
                )
                if int(row["hidden"]) == 0
            )
            if set(actual_columns) != set(canonical_columns):
                missing = sorted(
                    set(canonical_columns) - set(actual_columns)
                )
                unexpected = sorted(
                    set(actual_columns) - set(canonical_columns)
                )
                details: list[str] = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if unexpected:
                    details.append(
                        "unexpected " + ", ".join(unexpected)
                    )
                raise ConflictError(
                    f"Schema v13 cannot safely rebuild {table} ("
                    + "; ".join(details)
                    + ")."
                )
            if _normalize_table_definition(
                table_row["sql"]
            ) != _normalize_table_definition(canonical_tail):
                plans.append(
                    (
                        table,
                        canonical_tail,
                        canonical_columns,
                        canonical_indexes,
                    )
                )

        if not plans:
            return
        if connection.execute("PRAGMA foreign_keys").fetchone()[0]:
            raise ConflictError(
                "Schema v13 canonical rebuild requires the serialized legacy "
                "migration boundary."
            )
        expected_contract = _expected_current_schema_contract()
        expected_tables = {
            table.name: table for table in expected_contract.tables
        }
        actual_contract = _capture_current_schema_contract(connection)
        actual_tables = {
            table.name: table for table in actual_contract.tables
        }
        planned_tables = {table for table, *_rest in plans}
        for table in sorted(planned_tables):
            _validate_legacy_canonicalization_source(
                table,
                actual_tables[table],
                expected_tables[table],
            )

        expected_triggers = {
            name: (table, definition)
            for name, table, definition in expected_contract.triggers
        }
        for name, table, definition in actual_contract.triggers:
            if table not in planned_tables:
                continue
            if expected_triggers.get(name) != (table, definition):
                raise ConflictError(
                    f"Schema v13 cannot safely rebuild {table}: "
                    f"unknown or changed trigger {name} is attached."
                )

        # These backfills precede the NOT NULL copy and therefore fail closed
        # if a legacy row cannot be reconstructed exactly.
        if any(table == "questions" for table, *_ in plans):
            for row in connection.execute(
                """SELECT * FROM questions
                   WHERE content_hash IS NULL ORDER BY id"""
            ).fetchall():
                question = self._question_from_row(connection, row)
                connection.execute(
                    """UPDATE questions SET content_hash=? WHERE id=?""",
                    (question_content_hash(question), question.id),
                )
        if any(
            table == "learner_skill_families"
            for table, *_ in plans
        ):
            connection.execute(
                """UPDATE learner_skill_families AS evidence
                   SET kind=COALESCE((
                       SELECT question.kind FROM questions question
                       WHERE question.family_id=evidence.family_id
                       ORDER BY question.id LIMIT 1
                   ), 'unknown')
                   WHERE kind IS NULL"""
            )

        self._drop_corpus_registry_triggers(connection)
        # DDL has already installed current guards for every newly introduced
        # table. Some of those guards join legacy parents (for example a
        # performance-attempt guard joins decisions), and SQLite recompiles
        # them while a parent is renamed. Remove only triggers whose complete
        # definition is exactly the current trusted definition. Unknown or
        # modified triggers remain in place, causing the rebuild or final
        # exact-contract check to fail and the caller-owned transaction to
        # roll back without discarding local schema objects.
        for row in connection.execute(
            """SELECT name, tbl_name, sql FROM sqlite_master
               WHERE type='trigger' AND name NOT LIKE 'sqlite_%'
               ORDER BY name"""
        ).fetchall():
            expected = expected_triggers.get(row["name"])
            actual = (
                row["tbl_name"],
                _normalize_trigger_definition(row["sql"] or ""),
            )
            if expected == actual:
                connection.execute(
                    "DROP TRIGGER " + _quote_sqlite_identifier(row["name"])
                )
        for table, canonical_tail, columns, indexes in plans:
            temporary = f"_tsq_v13_{table}_new"
            quoted_table = _quote_sqlite_identifier(table)
            quoted_temporary = _quote_sqlite_identifier(temporary)
            if connection.execute(
                """SELECT 1 FROM sqlite_master
                   WHERE type='table' AND name=?""",
                (temporary,),
            ).fetchone():
                raise ConflictError(
                    f"Schema v13 temporary table already exists: {temporary}."
                )
            connection.execute(
                f"CREATE TABLE {quoted_temporary} {canonical_tail}"
            )
            column_sql = ", ".join(
                _quote_sqlite_identifier(column) for column in columns
            )
            connection.execute(
                f"""INSERT INTO {quoted_temporary} ({column_sql})
                    SELECT {column_sql} FROM {quoted_table}"""
            )
            connection.execute(f"DROP TABLE {quoted_table}")
            connection.execute(
                f"ALTER TABLE {quoted_temporary} RENAME TO {quoted_table}"
            )
            for index_sql in indexes:
                connection.execute(index_sql)
        # Restore the static guards that were removed above. Dynamic guards
        # are reinstalled by initialize after every migration step.
        _execute_sql_script(connection, DDL)

    @staticmethod
    def _migrate_v13_to_v14(connection: sqlite3.Connection) -> None:
        """Validate the empty shadow-performance ledger installed by DDL.

        Productive-skill observations did not exist in earlier schemas.  An
        upgrade therefore creates no task, action, evaluation, or evidence
        rows and leaves every learner/session revision and event stream intact.
        """

        required = {
            "performance_tasks",
            "performance_task_releases",
            "release_performance_tasks",
            "performance_attempts",
            "performance_actions",
            "task_evaluations",
            "shadow_evidence_bundles",
        }
        present = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = required - present
        if missing:
            raise ConflictError(
                "Schema v14 performance ledger was not installed: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _migrate_v14_to_v15(connection: sqlite3.Connection) -> None:
        """Validate immutable scoring claims installed by the current DDL.

        Schema v14 already contains the complete shadow-performance event and
        projection ledger.  Callback admission claims did not yet exist, so an
        upgrade installs an empty table and never fabricates claims for prior
        evaluations.
        """

        table = connection.execute(
            """SELECT 1 FROM sqlite_master
               WHERE type='table' AND name='performance_scoring_claims'"""
        ).fetchone()
        required_triggers = {
            "performance_scoring_claims_validate_insert",
            "performance_scoring_claims_no_update",
            "performance_scoring_claims_no_delete",
            "events_respect_performance_scoring_claim",
            "task_evaluations_validate_scoring_claim",
        }
        triggers = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
        missing_triggers = required_triggers - triggers
        if table is None or missing_triggers:
            details: list[str] = []
            if table is None:
                details.append("performance_scoring_claims")
            details.extend(sorted(missing_triggers))
            raise ConflictError(
                "Schema v15 scoring admission was not installed: "
                + ", ".join(details)
            )

    @staticmethod
    def _create_v16_scoring_claim_table(
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute(
            """
            CREATE TABLE performance_scoring_claims (
                id TEXT PRIMARY KEY CHECK(length(id) BETWEEN 1 AND 128),
                event_id TEXT NOT NULL UNIQUE REFERENCES events(event_id),
                idempotency_key TEXT UNIQUE CHECK(
                    idempotency_key IS NULL
                    OR (
                        length(idempotency_key) BETWEEN 1 AND 256
                        AND idempotency_key = trim(idempotency_key)
                    )
                ),
                attempt_id TEXT NOT NULL,
                evaluation_id TEXT NOT NULL UNIQUE CHECK(
                    length(evaluation_id) BETWEEN 1 AND 128
                ),
                through_sequence INTEGER NOT NULL CHECK(through_sequence >= 0),
                provider_id TEXT NOT NULL CHECK(
                    length(trim(provider_id)) BETWEEN 1 AND 128
                ),
                provider_version TEXT NOT NULL CHECK(
                    length(trim(provider_version)) BETWEEN 1 AND 128
                ),
                action_trace_digest TEXT NOT NULL CHECK(
                    length(action_trace_digest) = 64
                    AND action_trace_digest NOT GLOB '*[^0-9a-f]*'
                ),
                command_hash TEXT NOT NULL UNIQUE CHECK(
                    length(command_hash) = 64
                    AND command_hash NOT GLOB '*[^0-9a-f]*'
                ),
                claimed_at TEXT NOT NULL,
                FOREIGN KEY(attempt_id) REFERENCES performance_attempts(id)
                    DEFERRABLE INITIALLY DEFERRED
            )
            """
        )

    @staticmethod
    def _install_v18_shadow_evidence_bundle_trigger(
        connection: sqlite3.Connection,
    ) -> None:
        """Install the exact session-bound bundle guard used by schema v18."""

        _execute_sql_script(
            connection,
            """
            DROP TRIGGER IF EXISTS shadow_evidence_bundles_validate_insert;
            CREATE TRIGGER shadow_evidence_bundles_validate_insert
            BEFORE INSERT ON shadow_evidence_bundles BEGIN
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM task_evaluations evaluation
                    JOIN performance_attempts attempt
                      ON attempt.id = evaluation.attempt_id
                    JOIN events bundle_event
                      ON bundle_event.event_id = NEW.event_id
                    WHERE evaluation.id = NEW.evaluation_id
                      AND evaluation.attempt_id = NEW.attempt_id
                      AND bundle_event.event_type = 'ShadowEvidenceReduced'
                      AND bundle_event.schema_version = 1
                      AND bundle_event.stream_id =
                          'learner:' || attempt.learner_id
                      AND bundle_event.learner_id = attempt.learner_id
                      AND bundle_event.session_id = attempt.session_id
                      AND bundle_event.causation_id = evaluation.id
                      AND bundle_event.recorded_at = NEW.recorded_at
                      AND json_extract(
                          bundle_event.payload_json, '$.bundle_id'
                      ) = NEW.id
                      AND json_extract(
                          bundle_event.payload_json, '$.bundle_digest'
                      ) = NEW.bundle_digest
                      AND json_extract(
                          bundle_event.payload_json, '$.projection_applied'
                      ) = 0
                      AND json_extract(
                          bundle_event.payload_json, '$.certification_applied'
                      ) = 0
                ) THEN RAISE(
                    ABORT, 'shadow evidence does not match its evaluation/event'
                ) END;
            END;
            """,
        )

    @classmethod
    def _downgrade_v19_contract_to_v18(
        cls,
        connection: sqlite3.Connection,
    ) -> None:
        """Derive the exact empty v18 contract for migration preflight.

        This helper is used only against a fresh in-memory reference schema.
        It never rewrites a user database.
        """

        for trigger in (
            "performance_scoring_claims_validate_insert",
            "performance_scoring_claims_no_update",
            "performance_scoring_claims_no_delete",
            "events_respect_performance_scoring_claim",
            "performance_scoring_reconciliations_validate_insert",
            "performance_scoring_reconciliations_no_update",
            "performance_scoring_reconciliations_no_delete",
            "events_respect_performance_scoring_reconciliation",
            "task_evaluations_validate_scoring_claim",
            "task_evaluations_validate_insert",
            "shadow_evidence_bundles_validate_insert",
        ):
            connection.execute(
                "DROP TRIGGER IF EXISTS "
                + _quote_sqlite_identifier(trigger)
            )
        if connection.execute(
            "SELECT 1 FROM performance_scoring_claims LIMIT 1"
        ).fetchone() is not None:
            raise RuntimeError(
                "The v18 contract can only be derived from an empty reference."
            )
        if connection.execute(
            "SELECT 1 FROM performance_scoring_reconciliations LIMIT 1"
        ).fetchone() is not None:
            raise RuntimeError(
                "The v18 contract can only be derived from an empty reference."
            )
        connection.execute("DROP TABLE performance_scoring_reconciliations")
        connection.execute("DROP TABLE performance_scoring_claims")
        cls._create_v16_scoring_claim_table(connection)
        cls._install_v18_performance_scoring_triggers(connection)
        cls._install_v18_shadow_evidence_bundle_trigger(connection)

    @staticmethod
    def _legacy_scoring_claim_trace(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> tuple[tuple[LearningAction, ...], sqlite3.Row]:
        attempt = connection.execute(
            "SELECT * FROM performance_attempts WHERE id=?",
            (row["attempt_id"],),
        ).fetchone()
        if attempt is None:
            raise ConflictError(
                f"Schema v15 scoring claim {row['id']} has no attempt."
            )
        action_rows = connection.execute(
            """SELECT * FROM performance_actions
               WHERE attempt_id=? AND sequence<=?
               ORDER BY sequence, id""",
            (row["attempt_id"], row["through_sequence"]),
        ).fetchall()
        actions: list[LearningAction] = []
        try:
            for action_row in action_rows:
                actions.append(
                    LearningAction.from_terms(
                        {
                            "id": action_row["id"],
                            "trace_id": action_row["attempt_id"],
                            "sequence": action_row["sequence"],
                            "kind": action_row["action_type"],
                            "phase": action_row["phase"],
                            "payload": json.loads(action_row["payload_json"]),
                            "elapsed_ms": action_row["elapsed_ms"],
                            "schema_version": 1,
                        }
                    )
                )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConflictError(
                f"Schema v15 scoring claim {row['id']} has an invalid trace: {exc}"
            ) from exc
        trace = tuple(actions)
        submissions = [
            action
            for action in trace
            if action.kind is ActionKind.SUBMITTED
            and action.sequence == row["through_sequence"]
        ]
        try:
            trace_digest = action_trace_digest(trace)
        except (TypeError, ValueError) as exc:
            raise ConflictError(
                f"Schema v15 scoring claim {row['id']} trace cannot be committed: {exc}"
            ) from exc
        expected_command_hash = canonical_digest(
            {
                "type": "tsq.performance_command",
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
            raise ConflictError(
                f"Schema v15 scoring claim {row['id']} fails its trace commitment."
            )
        evaluation = connection.execute(
            "SELECT * FROM task_evaluations WHERE id=?",
            (row["evaluation_id"],),
        ).fetchone()
        if evaluation is not None:
            event = connection.execute(
                "SELECT * FROM events WHERE event_id=?",
                (evaluation["event_id"],),
            ).fetchone()
            if (
                evaluation["attempt_id"] != row["attempt_id"]
                or evaluation["through_sequence"] != row["through_sequence"]
                or evaluation["command_hash"] != row["command_hash"]
                or event is None
                or event["idempotency_key"] != row["idempotency_key"]
            ):
                raise ConflictError(
                    f"Schema v15 scoring claim {row['id']} does not match its completion."
                )
        return trace, attempt

    def _migrate_v15_to_v16(
        self,
        connection: sqlite3.Connection,
        *,
        starting_version: int,
    ) -> None:
        """Commit callback admissions to events without inventing history."""

        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(performance_scoring_claims)"
            ).fetchall()
        }
        if not columns:
            raise ConflictError(
                "Schema v16 scoring claim projection was not installed."
            )
        old_rows: list[sqlite3.Row] = []
        rebuilding = "event_id" not in columns
        if rebuilding:
            old_rows = connection.execute(
                "SELECT * FROM performance_scoring_claims ORDER BY id"
            ).fetchall()
            for row in old_rows:
                self._legacy_scoring_claim_trace(connection, row)
            for trigger in (
                "performance_scoring_claims_validate_insert",
                "performance_scoring_claims_no_update",
                "performance_scoring_claims_no_delete",
                "events_respect_performance_scoring_claim",
                "task_evaluations_validate_scoring_claim",
                "task_evaluations_validate_insert",
            ):
                connection.execute(f'DROP TRIGGER IF EXISTS "{trigger}"')
            connection.execute(
                """ALTER TABLE performance_scoring_claims
                   RENAME TO _tsq_v16_scoring_claims_old"""
            )
            self._create_v16_scoring_claim_table(connection)

            for row in old_rows:
                attempt = connection.execute(
                    "SELECT * FROM performance_attempts WHERE id=?",
                    (row["attempt_id"],),
                ).fetchone()
                if attempt is None:
                    raise ConflictError(
                        f"Schema v15 scoring claim {row['id']} lost its attempt."
                    )
                payload = performance_scoring_claim_payload(
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
                event = self.append_event(
                    connection,
                    stream_id=f"learner:{attempt['learner_id']}",
                    event_type="PerformanceScoringClaimMigrated",
                    schema_version=1,
                    payload=payload,
                    metadata={
                        "claim_schema_version": 1,
                        "admission_mode": "legacy_projection_migration",
                        "source_schema_version": 15,
                        "shadow_only": True,
                    },
                    learner_id=attempt["learner_id"],
                    # This is a migration observation appended after the
                    # historical stream tail, not an event that occurred in
                    # the learner's possibly already-ended session.
                    session_id=None,
                    idempotency_key=performance_scoring_claim_event_key(
                        row["command_hash"]
                    ),
                    correlation_id=row["attempt_id"],
                    causation_id=row["attempt_id"],
                    occurred_at=from_timestamp(row["claimed_at"]),
                )
                connection.execute(
                    """INSERT INTO performance_scoring_claims(
                           id, event_id, idempotency_key, attempt_id,
                           evaluation_id, through_sequence, provider_id,
                           provider_version, action_trace_digest, command_hash,
                           claimed_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row["id"],
                        event["event_id"],
                        row["idempotency_key"],
                        row["attempt_id"],
                        row["evaluation_id"],
                        row["through_sequence"],
                        row["provider_id"],
                        row["provider_version"],
                        row["action_trace_digest"],
                        row["command_hash"],
                        row["claimed_at"],
                    ),
                )
            connection.execute("DROP TABLE _tsq_v16_scoring_claims_old")

        unclaimed_registered = connection.execute(
            """SELECT evaluation.id AS evaluation_id,
                      evaluation.attempt_id,
                      evaluation.command_hash,
                      attempt.learner_id, attempt.session_id
               FROM task_evaluations evaluation
               JOIN performance_attempts attempt
                 ON attempt.id=evaluation.attempt_id
               WHERE json_extract(
                   evaluation.authority_json,
                   '$.normalized_result.normalization_mode'
               )='registered_provider'
                 AND NOT EXISTS (
                     SELECT 1 FROM performance_scoring_claims claim
                     WHERE claim.evaluation_id=evaluation.id
                       AND claim.attempt_id=evaluation.attempt_id
                       AND claim.command_hash=evaluation.command_hash
                 )
                 AND NOT EXISTS (
                     SELECT 1 FROM events exemption
                     WHERE exemption.event_type =
                           'PerformanceScoringLegacyExempted'
                       AND exemption.schema_version = 1
                       AND json_extract(
                           exemption.payload_json, '$.evaluation_id'
                       ) = evaluation.id
                       AND json_extract(
                           exemption.payload_json, '$.attempt_id'
                       ) = evaluation.attempt_id
                       AND json_extract(
                           exemption.payload_json, '$.command_hash'
                       ) = evaluation.command_hash
                       AND json_extract(
                           exemption.payload_json, '$.reason'
                       ) = 'schema_v14_predates_callback_claims'
                 )
               ORDER BY evaluation.id"""
        ).fetchall()
        if unclaimed_registered and starting_version >= 15:
            raise ConflictError(
                "Schema v15 contains a registered-provider evaluation without "
                "its required scoring claim; migration fails closed."
            )
        for row in unclaimed_registered:
            self.append_event(
                connection,
                stream_id=f"learner:{row['learner_id']}",
                event_type="PerformanceScoringLegacyExempted",
                schema_version=1,
                payload={
                    "evaluation_id": row["evaluation_id"],
                    "attempt_id": row["attempt_id"],
                    "command_hash": row["command_hash"],
                    "reason": "schema_v14_predates_callback_claims",
                },
                metadata={
                    "migration_from_schema_version": 14,
                    "shadow_only": True,
                },
                learner_id=row["learner_id"],
                # The exception records a schema migration fact. Binding it
                # to an ended historical session would put a new event after
                # SessionEnded and invalidate the immutable lifecycle.
                session_id=None,
                idempotency_key=(
                    "performance-score-legacy:v1:" + row["evaluation_id"]
                ),
                correlation_id=row["attempt_id"],
                causation_id=row["evaluation_id"],
            )

    @staticmethod
    def _validate_v16_migration_lifecycle(
        connection: sqlite3.Connection,
    ) -> None:
        """Reject v16 migration observations already appended after session end."""

        invalid = connection.execute(
            """SELECT observation.event_id
               FROM events observation
               JOIN events ended
                 ON ended.session_id=observation.session_id
                AND ended.event_type='SessionEnded'
               WHERE observation.event_type IN (
                   'PerformanceScoringClaimMigrated',
                   'PerformanceScoringLegacyExempted'
               )
                 AND observation.session_id IS NOT NULL
                 AND observation.stream_id=ended.stream_id
                 AND observation.stream_version > ended.stream_version
               ORDER BY observation.event_id LIMIT 1"""
        ).fetchone()
        if invalid is not None:
            raise ConflictError(
                "Schema v16 contains a migration observation after "
                "SessionEnded; immutable event history requires explicit "
                f"repair before v17 ({invalid['event_id']})."
            )

    @classmethod
    def _migrate_v16_to_v17(
        cls,
        connection: sqlite3.Connection,
    ) -> None:
        """Move migration-only scoring observations outside session envelopes."""

        # Exact v16 structure was checked read-only before any migration write.
        # Existing session-bound migration observations remain valid only when
        # they precede a later SessionEnded (or the session is still active).
        # New v15/v14 migrations already emit unbound observations.
        cls._validate_v16_migration_lifecycle(connection)

    @staticmethod
    def _migrate_v17_to_v18(
        connection: sqlite3.Connection,
    ) -> None:
        """Install an empty prospective policy-shadow projection."""

        # DDL creates the guarded projection before version dispatch. Historical
        # decisions do not identify counterfactual challenger actions, so this
        # migration must never backfill them or invent matching history.
        projected = connection.execute(
            "SELECT 1 FROM policy_shadow_evaluations LIMIT 1"
        ).fetchone()
        if projected is not None:
            raise ConflictError(
                "Schema v18 policy-shadow migration requires an empty "
                "projection."
            )
        historical_event = connection.execute(
            """SELECT event_id FROM events
               WHERE event_type='PolicyShadowEvaluated'
               ORDER BY event_id LIMIT 1"""
        ).fetchone()
        if historical_event is not None:
            raise ConflictError(
                "Schema v17 contains a PolicyShadowEvaluated event without an "
                "event-backed projection; explicit repair is required before "
                f"v18 ({historical_event['event_id']})."
            )
        shadow_required_decision = connection.execute(
            """SELECT id FROM decisions
               WHERE policy_version='recursive-evidence-graph-v18'
               ORDER BY id LIMIT 1"""
        ).fetchone()
        if shadow_required_decision is not None:
            raise ConflictError(
                "Schema v17 contains a decision labeled with the v18 policy "
                "without its required prospective shadow evidence; explicit "
                "repair is required before v18 "
                f"({shadow_required_decision['id']})."
            )

    @staticmethod
    def _migrate_v18_to_v19(
        connection: sqlite3.Connection,
    ) -> None:
        """Bind prospective claims and install empty reconciliation history.

        Exact v18 preflight proves the source projection shape.  Existing
        claim/event bytes predate request and provider-operation commitments,
        so they are copied verbatim as claim-schema v1 with NULL v2 fields.
        No historical reconciliation or upgraded claim event is invented.
        """

        historical_event = connection.execute(
            """SELECT event_id FROM events
               WHERE event_type='PerformanceScoringReconciled'
               ORDER BY event_id LIMIT 1"""
        ).fetchone()
        if historical_event is not None:
            raise ConflictError(
                "Schema v18 contains a PerformanceScoringReconciled event "
                "without a v19 event-backed projection; explicit repair is "
                f"required before v19 ({historical_event['event_id']})."
            )
        projected = connection.execute(
            """SELECT id FROM performance_scoring_reconciliations
               ORDER BY id LIMIT 1"""
        ).fetchone()
        if projected is not None:
            raise ConflictError(
                "Schema v19 reconciliation migration requires an empty "
                "prospective projection."
            )

        old_rows = connection.execute(
            "SELECT * FROM performance_scoring_claims ORDER BY id"
        ).fetchall()
        for trigger in (
            "performance_scoring_claims_validate_insert",
            "performance_scoring_claims_no_update",
            "performance_scoring_claims_no_delete",
            "events_respect_performance_scoring_claim",
            "performance_scoring_reconciliations_validate_insert",
            "performance_scoring_reconciliations_no_update",
            "performance_scoring_reconciliations_no_delete",
            "events_respect_performance_scoring_reconciliation",
            "task_evaluations_validate_scoring_claim",
            "task_evaluations_validate_insert",
        ):
            connection.execute(
                "DROP TRIGGER IF EXISTS "
                + _quote_sqlite_identifier(trigger)
            )
        connection.execute(
            "DROP TABLE performance_scoring_reconciliations"
        )
        connection.execute(
            """ALTER TABLE performance_scoring_claims
               RENAME TO _tsq_v19_scoring_claims_old"""
        )
        # Re-executing idempotent DDL after the two intentional removals
        # creates the exact current tables and static guards.  All other exact
        # v18 objects already exist and are therefore left byte-for-byte alone.
        _execute_sql_script(connection, DDL)
        # A completed v18 claim legitimately has a later evaluation event
        # using its caller idempotency key.  The prospective insert guard is
        # correct at admission time but cannot be reapplied while copying that
        # already-established immutable history.
        connection.execute(
            "DROP TRIGGER performance_scoring_claims_validate_insert"
        )
        for row in old_rows:
            connection.execute(
                """INSERT INTO performance_scoring_claims(
                       id, event_id, claim_schema_version, idempotency_key,
                       attempt_id, evaluation_id, through_sequence,
                       provider_id, provider_version, action_trace_digest,
                       scoring_request_digest, provider_binding_digest,
                       provider_operation_digest, command_hash, claimed_at
                   ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL,
                             ?, ?)""",
                (
                    row["id"],
                    row["event_id"],
                    row["idempotency_key"],
                    row["attempt_id"],
                    row["evaluation_id"],
                    row["through_sequence"],
                    row["provider_id"],
                    row["provider_version"],
                    row["action_trace_digest"],
                    row["command_hash"],
                    row["claimed_at"],
                ),
            )
        connection.execute("DROP TABLE _tsq_v19_scoring_claims_old")

    @staticmethod
    def _drop_policy_shadow_triggers(
        connection: sqlite3.Connection,
    ) -> None:
        """Remove policy-shadow projection guards on an isolated rebuild copy."""

        for trigger_name in sorted(_POLICY_SHADOW_TRIGGER_NAMES):
            connection.execute(
                "DROP TRIGGER IF EXISTS "
                + _quote_sqlite_identifier(trigger_name)
            )

    @classmethod
    def _install_policy_shadow_triggers(
        cls,
        connection: sqlite3.Connection,
    ) -> None:
        """Restore the exact policy-shadow event/projection guards."""

        cls._drop_policy_shadow_triggers(connection)
        expected = {
            name: sql
            for name, _table, sql in (
                _expected_current_schema_contract().triggers
            )
            if name in _POLICY_SHADOW_TRIGGER_NAMES
        }
        if set(expected) != set(_POLICY_SHADOW_TRIGGER_NAMES):
            raise RuntimeError(
                "Current schema lacks a policy-shadow projection guard."
            )
        for trigger_name in sorted(expected):
            connection.execute(expected[trigger_name])

    @staticmethod
    def _install_v18_performance_scoring_triggers(
        connection: sqlite3.Connection,
    ) -> None:
        """Install the exact callback-admission guards used by schema v18."""

        _execute_sql_script(
            connection,
            """
            DROP TRIGGER IF EXISTS performance_scoring_claims_validate_insert;
            DROP TRIGGER IF EXISTS performance_scoring_claims_no_update;
            DROP TRIGGER IF EXISTS performance_scoring_claims_no_delete;
            DROP TRIGGER IF EXISTS events_respect_performance_scoring_claim;
            DROP TRIGGER IF EXISTS task_evaluations_validate_scoring_claim;
            DROP TRIGGER IF EXISTS task_evaluations_validate_insert;

            CREATE TRIGGER performance_scoring_claims_validate_insert
            BEFORE INSERT ON performance_scoring_claims BEGIN
                SELECT CASE WHEN EXISTS (
                    SELECT 1 FROM events event
                    WHERE event.idempotency_key = NEW.idempotency_key
                      AND event.event_id != NEW.event_id
                ) THEN RAISE(
                    ABORT, 'scoring claim idempotency key already has an event'
                ) END;
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM performance_attempts attempt
                    JOIN performance_actions submission
                      ON submission.attempt_id = attempt.id
                     AND submission.sequence = NEW.through_sequence
                     AND submission.action_type = 'submitted'
                    WHERE attempt.id = NEW.attempt_id
                ) THEN RAISE(
                    ABORT, 'scoring claim lacks its submitted trace boundary'
                ) END;
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM events claim_event
                    JOIN performance_attempts attempt
                      ON attempt.id = NEW.attempt_id
                    WHERE claim_event.event_id = NEW.event_id
                      AND claim_event.event_type IN (
                          'PerformanceScoringClaimed',
                          'PerformanceScoringClaimMigrated'
                      )
                      AND claim_event.schema_version = 1
                      AND claim_event.stream_id =
                          'learner:' || attempt.learner_id
                      AND claim_event.learner_id = attempt.learner_id
                      AND (
                          (
                              claim_event.event_type =
                                  'PerformanceScoringClaimed'
                              AND claim_event.session_id = attempt.session_id
                          )
                          OR (
                              claim_event.event_type =
                                  'PerformanceScoringClaimMigrated'
                              AND (
                                  claim_event.session_id IS NULL
                                  OR claim_event.session_id =
                                      attempt.session_id
                              )
                          )
                      )
                      AND claim_event.idempotency_key =
                          'performance-score-claim:v1:' || NEW.command_hash
                      AND claim_event.correlation_id = NEW.attempt_id
                      AND claim_event.causation_id = NEW.attempt_id
                      AND json_extract(
                          claim_event.payload_json, '$.claim_id'
                      ) = NEW.id
                      AND json_extract(
                          claim_event.payload_json,
                          '$.caller_idempotency_key'
                      ) IS NEW.idempotency_key
                      AND json_extract(
                          claim_event.payload_json, '$.attempt_id'
                      ) = NEW.attempt_id
                      AND json_extract(
                          claim_event.payload_json, '$.evaluation_id'
                      ) = NEW.evaluation_id
                      AND json_extract(
                          claim_event.payload_json, '$.through_sequence'
                      ) = NEW.through_sequence
                      AND json_extract(
                          claim_event.payload_json, '$.provider_id'
                      ) = NEW.provider_id
                      AND json_extract(
                          claim_event.payload_json, '$.provider_version'
                      ) = NEW.provider_version
                      AND json_extract(
                          claim_event.payload_json, '$.action_trace_digest'
                      ) = NEW.action_trace_digest
                      AND json_extract(
                          claim_event.payload_json, '$.command_hash'
                      ) = NEW.command_hash
                      AND json_extract(
                          claim_event.payload_json, '$.claimed_at'
                      ) = NEW.claimed_at
                ) THEN RAISE(
                    ABORT, 'scoring claim does not match its event'
                ) END;
            END;

            CREATE TRIGGER performance_scoring_claims_no_update
            BEFORE UPDATE ON performance_scoring_claims BEGIN
                SELECT RAISE(ABORT, 'performance scoring claims are immutable');
            END;

            CREATE TRIGGER performance_scoring_claims_no_delete
            BEFORE DELETE ON performance_scoring_claims BEGIN
                SELECT RAISE(ABORT, 'performance scoring claims are immutable');
            END;

            CREATE TRIGGER events_respect_performance_scoring_claim
            BEFORE INSERT ON events
            WHEN NEW.idempotency_key IS NOT NULL
             AND EXISTS (
                 SELECT 1 FROM performance_scoring_claims claim
                 WHERE claim.idempotency_key = NEW.idempotency_key
             )
            BEGIN
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM performance_scoring_claims claim
                    WHERE claim.idempotency_key = NEW.idempotency_key
                      AND NEW.event_type = 'TaskEvaluationRecorded'
                      AND json_extract(
                          NEW.metadata_json, '$.command_hash'
                      ) = claim.command_hash
                      AND json_extract(
                          NEW.payload_json, '$.attempt_id'
                      ) = claim.attempt_id
                      AND json_extract(
                          NEW.payload_json, '$.through_sequence'
                      ) = claim.through_sequence
                      AND json_extract(
                          NEW.payload_json, '$.evaluation.id'
                      ) = claim.evaluation_id
                ) THEN RAISE(
                    ABORT, 'event does not complete its performance scoring claim'
                ) END;
            END;

            CREATE TRIGGER task_evaluations_validate_scoring_claim
            BEFORE INSERT ON task_evaluations BEGIN
                SELECT CASE WHEN EXISTS (
                    SELECT 1
                    FROM performance_scoring_claims claim
                    JOIN events evaluation_event
                      ON evaluation_event.event_id = NEW.event_id
                    WHERE claim.command_hash = NEW.command_hash
                      AND (
                          claim.attempt_id != NEW.attempt_id
                          OR claim.evaluation_id != NEW.id
                          OR claim.through_sequence != NEW.through_sequence
                          OR evaluation_event.idempotency_key
                             IS NOT claim.idempotency_key
                      )
                ) THEN RAISE(
                    ABORT, 'task evaluation does not complete its scoring claim'
                ) END;
                SELECT CASE WHEN json_extract(
                    NEW.authority_json,
                    '$.normalized_result.normalization_mode'
                ) = 'registered_provider' AND NOT EXISTS (
                    SELECT 1 FROM performance_scoring_claims claim
                    WHERE claim.command_hash = NEW.command_hash
                      AND claim.attempt_id = NEW.attempt_id
                      AND claim.evaluation_id = NEW.id
                      AND claim.through_sequence = NEW.through_sequence
                ) AND NOT EXISTS (
                    SELECT 1 FROM events exemption
                    WHERE exemption.event_type =
                          'PerformanceScoringLegacyExempted'
                      AND exemption.schema_version = 1
                      AND json_extract(
                          exemption.payload_json, '$.evaluation_id'
                      ) = NEW.id
                      AND json_extract(
                          exemption.payload_json, '$.attempt_id'
                      ) = NEW.attempt_id
                      AND json_extract(
                          exemption.payload_json, '$.command_hash'
                      ) = NEW.command_hash
                ) THEN RAISE(
                    ABORT, 'registered evaluation lacks its scoring claim'
                ) END;
            END;

            CREATE TRIGGER task_evaluations_validate_insert
            BEFORE INSERT ON task_evaluations BEGIN
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1 FROM performance_actions submission
                    WHERE submission.attempt_id = NEW.attempt_id
                      AND submission.action_type = 'submitted'
                      AND submission.sequence = NEW.through_sequence
                ) THEN RAISE(
                    ABORT, 'task evaluation lacks its submitted trace boundary'
                ) END;
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM performance_attempts attempt
                    JOIN events evaluation_event
                      ON evaluation_event.event_id = NEW.event_id
                    WHERE attempt.id = NEW.attempt_id
                      AND evaluation_event.event_type = 'TaskEvaluationRecorded'
                      AND evaluation_event.schema_version = 1
                      AND evaluation_event.stream_id =
                          'learner:' || attempt.learner_id
                      AND evaluation_event.learner_id = attempt.learner_id
                      AND evaluation_event.session_id = attempt.session_id
                      AND (
                          evaluation_event.causation_id = NEW.attempt_id
                          OR EXISTS (
                              SELECT 1
                              FROM performance_scoring_claims claim
                              WHERE claim.event_id =
                                    evaluation_event.causation_id
                                AND claim.attempt_id = NEW.attempt_id
                                AND claim.evaluation_id = NEW.id
                                AND claim.command_hash = NEW.command_hash
                          )
                      )
                      AND evaluation_event.recorded_at = NEW.recorded_at
                      AND json_extract(
                          evaluation_event.payload_json, '$.evaluation.id'
                      ) = NEW.id
                      AND json_extract(
                          evaluation_event.payload_json, '$.evaluation_digest'
                      ) = NEW.evaluation_digest
                      AND json_extract(
                          evaluation_event.payload_json, '$.through_sequence'
                      ) = NEW.through_sequence
                ) THEN RAISE(
                    ABORT, 'task evaluation does not match its event'
                ) END;
            END;
            """,
        )

    @staticmethod
    def _install_current_performance_scoring_triggers(
        connection: sqlite3.Connection,
    ) -> None:
        """Install the v19 scoring, reconciliation, and recovery guards.

        SQLite can enforce closed JSON shape, scalar types, fixed flags,
        projection equality, and canonical UTC temporal ordering.  It cannot
        compute SHA-256: request, trace, command, provider-binding, operation,
        and receipt digest recomputation deliberately remains an integrity and
        deterministic-replay responsibility.
        """

        _execute_sql_script(
            connection,
            """
            DROP TRIGGER IF EXISTS performance_scoring_claims_validate_insert;
            DROP TRIGGER IF EXISTS performance_scoring_claims_no_update;
            DROP TRIGGER IF EXISTS performance_scoring_claims_no_delete;
            DROP TRIGGER IF EXISTS events_respect_performance_scoring_claim;
            DROP TRIGGER IF EXISTS
                performance_scoring_reconciliations_validate_insert;
            DROP TRIGGER IF EXISTS
                performance_scoring_reconciliations_no_update;
            DROP TRIGGER IF EXISTS
                performance_scoring_reconciliations_no_delete;
            DROP TRIGGER IF EXISTS
                events_respect_performance_scoring_reconciliation;
            DROP TRIGGER IF EXISTS task_evaluations_validate_scoring_claim;
            DROP TRIGGER IF EXISTS task_evaluations_validate_insert;
            DROP TRIGGER IF EXISTS shadow_evidence_bundles_validate_insert;

            CREATE TRIGGER performance_scoring_claims_validate_insert
            BEFORE INSERT ON performance_scoring_claims BEGIN
                SELECT CASE WHEN NEW.idempotency_key IS NOT NULL
                  AND EXISTS (
                      SELECT 1
                      FROM performance_scoring_reconciliations reconciliation
                      WHERE reconciliation.idempotency_key =
                          NEW.idempotency_key
                  )
                THEN RAISE(
                    ABORT,
                    'scoring claim idempotency key belongs to a reconciliation'
                ) END;
                SELECT CASE WHEN EXISTS (
                    SELECT 1 FROM events event
                    WHERE event.idempotency_key = NEW.idempotency_key
                      AND event.event_id != NEW.event_id
                ) THEN RAISE(
                    ABORT, 'scoring claim idempotency key already has an event'
                ) END;
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM performance_attempts attempt
                    JOIN performance_actions submission
                      ON submission.attempt_id = attempt.id
                     AND submission.sequence = NEW.through_sequence
                     AND submission.action_type = 'submitted'
                    WHERE attempt.id = NEW.attempt_id
                ) THEN RAISE(
                    ABORT, 'scoring claim lacks its submitted trace boundary'
                ) END;
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM events claim_event
                    JOIN performance_attempts attempt
                      ON attempt.id = NEW.attempt_id
                    JOIN performance_actions submission
                      ON submission.attempt_id = attempt.id
                     AND submission.sequence = NEW.through_sequence
                     AND submission.action_type = 'submitted'
                    JOIN events submission_event
                      ON submission_event.event_id = submission.event_id
                    WHERE claim_event.event_id = NEW.event_id
                      AND claim_event.stream_id =
                          'learner:' || attempt.learner_id
                      AND claim_event.learner_id = attempt.learner_id
                      AND submission_event.stream_id = claim_event.stream_id
                      AND claim_event.stream_version >
                          submission_event.stream_version
                      AND julianday(NEW.claimed_at) >=
                          julianday(submission_event.occurred_at)
                      AND claim_event.idempotency_key =
                          'performance-score-claim:v1:' || NEW.command_hash
                      AND claim_event.correlation_id = NEW.attempt_id
                      AND claim_event.causation_id = NEW.attempt_id
                      AND json_extract(
                          claim_event.payload_json, '$.claim_id'
                      ) = NEW.id
                      AND json_extract(
                          claim_event.payload_json,
                          '$.caller_idempotency_key'
                      ) IS NEW.idempotency_key
                      AND json_extract(
                          claim_event.payload_json, '$.attempt_id'
                      ) = NEW.attempt_id
                      AND json_extract(
                          claim_event.payload_json, '$.evaluation_id'
                      ) = NEW.evaluation_id
                      AND json_extract(
                          claim_event.payload_json, '$.through_sequence'
                      ) = NEW.through_sequence
                      AND json_extract(
                          claim_event.payload_json, '$.provider_id'
                      ) = NEW.provider_id
                      AND json_extract(
                          claim_event.payload_json, '$.provider_version'
                      ) = NEW.provider_version
                      AND json_extract(
                          claim_event.payload_json, '$.action_trace_digest'
                      ) = NEW.action_trace_digest
                      AND json_extract(
                          claim_event.payload_json, '$.command_hash'
                      ) = NEW.command_hash
                      AND json_extract(
                          claim_event.payload_json, '$.claimed_at'
                      ) = NEW.claimed_at
                      -- Runtime admission is v2-only.  Historical v1 rows
                      -- are copied/replayed while this trigger is explicitly
                      -- absent, then protected immutably after installation.
                      AND NEW.claim_schema_version = 2
                      AND (
                          (
                              NEW.claim_schema_version = 1
                              AND claim_event.event_type IN (
                                  'PerformanceScoringClaimed',
                                  'PerformanceScoringClaimMigrated'
                              )
                              AND claim_event.schema_version = 1
                              AND json_extract(
                                  claim_event.metadata_json,
                                  '$.claim_schema_version'
                              ) = 1
                              AND (
                                  SELECT count(*)
                                  FROM json_each(claim_event.payload_json)
                              ) = 10
                              AND (
                                  (
                                      claim_event.event_type =
                                          'PerformanceScoringClaimed'
                                      AND claim_event.session_id =
                                          attempt.session_id
                                  )
                                  OR (
                                      claim_event.event_type =
                                          'PerformanceScoringClaimMigrated'
                                      AND (
                                          claim_event.session_id IS NULL
                                          OR claim_event.session_id =
                                              attempt.session_id
                                      )
                                  )
                              )
                          )
                          OR (
                              NEW.claim_schema_version = 2
                              AND claim_event.event_type =
                                  'PerformanceScoringClaimed'
                              AND claim_event.schema_version = 2
                              AND claim_event.session_id = attempt.session_id
                              AND claim_event.occurred_at = NEW.claimed_at
                              AND json_extract(
                                  claim_event.metadata_json,
                                  '$.claim_schema_version'
                              ) = 2
                              AND json_type(
                                  claim_event.metadata_json,
                                  '$.claim_schema_version'
                              ) = 'integer'
                              AND json_extract(
                                  claim_event.metadata_json,
                                  '$.admission_mode'
                              ) = 'pre_callback'
                              AND json_type(
                                  claim_event.metadata_json,
                                  '$.admission_mode'
                              ) = 'text'
                              AND json_type(
                                  claim_event.metadata_json,
                                  '$.source_schema_version'
                              ) = 'null'
                              AND json_type(
                                  claim_event.metadata_json,
                                  '$.shadow_only'
                              ) = 'true'
                              AND (
                                  SELECT count(*)
                                  FROM json_each(claim_event.metadata_json)
                              ) = 4
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM json_each(claim_event.metadata_json)
                                  WHERE key NOT IN (
                                      'claim_schema_version',
                                      'admission_mode',
                                      'source_schema_version',
                                      'shadow_only'
                                  )
                              )
                              AND (
                                  SELECT count(*)
                                  FROM json_each(claim_event.payload_json)
                              ) = 14
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM json_each(claim_event.payload_json)
                                  WHERE key NOT IN (
                                      'claim_id',
                                      'caller_idempotency_key',
                                      'attempt_id',
                                      'evaluation_id',
                                      'through_sequence',
                                      'provider_id',
                                      'provider_version',
                                      'action_trace_digest',
                                      'command_hash',
                                      'claimed_at',
                                      'scoring_request_digest',
                                      'provider_binding_digest',
                                      'provider_operation_digest',
                                      'provider'
                                  )
                              )
                              AND json_type(
                                  claim_event.payload_json, '$.claim_id'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.caller_idempotency_key'
                              ) IN ('null', 'text')
                              AND json_type(
                                  claim_event.payload_json, '$.attempt_id'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json, '$.evaluation_id'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.through_sequence'
                              ) = 'integer'
                              AND json_type(
                                  claim_event.payload_json, '$.provider_id'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider_version'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.action_trace_digest'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json, '$.command_hash'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json, '$.claimed_at'
                              ) = 'text'
                              AND (
                                  (
                                      length(NEW.claimed_at) = 25
                                      AND NEW.claimed_at GLOB
                                          '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00'
                                  )
                                  OR (
                                      length(NEW.claimed_at) = 32
                                      AND NEW.claimed_at GLOB
                                          '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00'
                                      AND substr(
                                          NEW.claimed_at, 21, 6
                                      ) != '000000'
                                  )
                              )
                              AND substr(NEW.claimed_at, 1, 4) != '0000'
                              AND substr(NEW.claimed_at, 5, 1) = '-'
                              AND substr(NEW.claimed_at, 8, 1) = '-'
                              AND substr(NEW.claimed_at, 11, 1) = 'T'
                              AND substr(NEW.claimed_at, 14, 1) = ':'
                              AND substr(NEW.claimed_at, 17, 1) = ':'
                              AND substr(NEW.claimed_at, -6) = '+00:00'
                              AND substr(NEW.claimed_at, 12, 2)
                                  BETWEEN '00' AND '23'
                              AND substr(NEW.claimed_at, 15, 2)
                                  BETWEEN '00' AND '59'
                              AND substr(NEW.claimed_at, 18, 2)
                                  BETWEEN '00' AND '59'
                              AND julianday(NEW.claimed_at) IS NOT NULL
                              AND strftime(
                                  '%Y-%m-%d', NEW.claimed_at
                              ) = substr(NEW.claimed_at, 1, 10)
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.scoring_request_digest'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider_binding_digest'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider_operation_digest'
                              ) = 'text'
                              AND json_extract(
                                  claim_event.payload_json,
                                  '$.scoring_request_digest'
                              ) = NEW.scoring_request_digest
                              AND json_extract(
                                  claim_event.payload_json,
                                  '$.provider_binding_digest'
                              ) = NEW.provider_binding_digest
                              AND json_extract(
                                  claim_event.payload_json,
                                  '$.provider_operation_digest'
                              ) = NEW.provider_operation_digest
                              AND json_type(
                                  claim_event.payload_json, '$.provider'
                              ) = 'object'
                              AND (
                                  SELECT count(*)
                                  FROM json_each(
                                      claim_event.payload_json, '$.provider'
                                  )
                              ) = 11
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM json_each(
                                      claim_event.payload_json, '$.provider'
                                  )
                                  WHERE key NOT IN (
                                      'provider_id',
                                      'provider_version',
                                      'declared_kind',
                                      'authority_id',
                                      'authority_manifest_digest',
                                      'binding_digest',
                                      'check_set_manifests',
                                      'artifact_manifests',
                                      'verified',
                                      'synthetic',
                                      'shadow_only'
                                  )
                              )
                              AND json_extract(
                                  claim_event.payload_json,
                                  '$.provider.provider_id'
                              ) = NEW.provider_id
                              AND json_extract(
                                  claim_event.payload_json,
                                  '$.provider.provider_version'
                              ) = NEW.provider_version
                              AND json_extract(
                                  claim_event.payload_json,
                                  '$.provider.binding_digest'
                              ) = NEW.provider_binding_digest
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.provider_id'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.provider_version'
                              ) = 'text'
                              AND substr(NEW.provider_id, 1, 1)
                                  GLOB '[A-Za-z0-9]'
                              AND NEW.provider_id NOT GLOB
                                  '*[^A-Za-z0-9._:-]*'
                              AND substr(NEW.provider_version, 1, 1)
                                  GLOB '[A-Za-z0-9]'
                              AND NEW.provider_version NOT GLOB
                                  '*[^A-Za-z0-9._:-]*'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.declared_kind'
                              ) = 'text'
                              AND json_extract(
                                  claim_event.payload_json,
                                  '$.provider.declared_kind'
                              ) IN (
                                  'deterministic',
                                  'human',
                                  'model',
                                  'imported'
                              )
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.authority_id'
                              ) = 'text'
                              AND length(json_extract(
                                  claim_event.payload_json,
                                  '$.provider.authority_id'
                              )) BETWEEN 1 AND 128
                              AND substr(json_extract(
                                  claim_event.payload_json,
                                  '$.provider.authority_id'
                              ), 1, 1) GLOB '[A-Za-z0-9]'
                              AND json_extract(
                                  claim_event.payload_json,
                                  '$.provider.authority_id'
                              ) NOT GLOB '*[^A-Za-z0-9._:-]*'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.authority_manifest_digest'
                              ) = 'text'
                              AND length(json_extract(
                                  claim_event.payload_json,
                                  '$.provider.authority_manifest_digest'
                              )) = 64
                              AND json_extract(
                                  claim_event.payload_json,
                                  '$.provider.authority_manifest_digest'
                              ) NOT GLOB '*[^0-9a-f]*'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.binding_digest'
                              ) = 'text'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.check_set_manifests'
                              ) = 'array'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.artifact_manifests'
                              ) = 'array'
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.verified'
                              ) IN ('true', 'false')
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.synthetic'
                              ) IN ('true', 'false')
                              AND json_type(
                                  claim_event.payload_json,
                                  '$.provider.shadow_only'
                              ) IN ('true', 'false')
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM json_each(
                                      claim_event.payload_json,
                                      '$.provider.check_set_manifests'
                                  ) manifest
                                  WHERE json_type(manifest.value) != 'object'
                                     OR (
                                         SELECT count(*)
                                         FROM json_each(manifest.value)
                                     ) != 2
                                     OR EXISTS (
                                         SELECT 1
                                         FROM json_each(manifest.value)
                                         WHERE key NOT IN (
                                             'check_set_id',
                                             'manifest_digest'
                                         )
                                     )
                                     OR (
                                         SELECT count(*)
                                         FROM json_each(manifest.value)
                                         WHERE key = 'check_set_id'
                                     ) != 1
                                     OR (
                                         SELECT count(*)
                                         FROM json_each(manifest.value)
                                         WHERE key = 'manifest_digest'
                                     ) != 1
                                     OR json_type(
                                         manifest.value, '$.check_set_id'
                                     ) != 'text'
                                     OR length(json_extract(
                                         manifest.value, '$.check_set_id'
                                     )) NOT BETWEEN 1 AND 128
                                     OR substr(json_extract(
                                         manifest.value, '$.check_set_id'
                                     ), 1, 1) NOT GLOB '[A-Za-z0-9]'
                                     OR json_extract(
                                         manifest.value, '$.check_set_id'
                                     ) GLOB '*[^A-Za-z0-9._:-]*'
                                     OR json_type(
                                         manifest.value, '$.manifest_digest'
                                     ) != 'text'
                                     OR length(json_extract(
                                         manifest.value, '$.manifest_digest'
                                     )) != 64
                                     OR json_extract(
                                         manifest.value, '$.manifest_digest'
                                     ) GLOB '*[^0-9a-f]*'
                              )
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM json_each(
                                      claim_event.payload_json,
                                      '$.provider.artifact_manifests'
                                  ) manifest
                                  WHERE json_type(manifest.value) != 'object'
                                     OR (
                                         SELECT count(*)
                                         FROM json_each(manifest.value)
                                     ) != 2
                                     OR EXISTS (
                                         SELECT 1
                                         FROM json_each(manifest.value)
                                         WHERE key NOT IN (
                                             'artifact_kind',
                                             'manifest_digest'
                                         )
                                     )
                                     OR (
                                         SELECT count(*)
                                         FROM json_each(manifest.value)
                                         WHERE key = 'artifact_kind'
                                     ) != 1
                                     OR (
                                         SELECT count(*)
                                         FROM json_each(manifest.value)
                                         WHERE key = 'manifest_digest'
                                     ) != 1
                                     OR json_type(
                                         manifest.value, '$.artifact_kind'
                                     ) != 'text'
                                     OR length(json_extract(
                                         manifest.value, '$.artifact_kind'
                                     )) NOT BETWEEN 1 AND 128
                                     OR substr(json_extract(
                                         manifest.value, '$.artifact_kind'
                                     ), 1, 1) NOT GLOB '[A-Za-z0-9]'
                                     OR json_extract(
                                         manifest.value, '$.artifact_kind'
                                     ) GLOB '*[^A-Za-z0-9._:-]*'
                                     OR json_type(
                                         manifest.value, '$.manifest_digest'
                                     ) != 'text'
                                     OR length(json_extract(
                                         manifest.value, '$.manifest_digest'
                                     )) != 64
                                     OR json_extract(
                                         manifest.value, '$.manifest_digest'
                                     ) GLOB '*[^0-9a-f]*'
                              )
                              AND (
                                  SELECT count(*)
                                  FROM json_each(
                                      claim_event.payload_json,
                                      '$.provider.check_set_manifests'
                                  )
                              ) = (
                                  SELECT count(DISTINCT json_extract(
                                      value, '$.check_set_id'
                                  ))
                                  FROM json_each(
                                      claim_event.payload_json,
                                      '$.provider.check_set_manifests'
                                  )
                              )
                              AND (
                                  SELECT count(*)
                                  FROM json_each(
                                      claim_event.payload_json,
                                      '$.provider.artifact_manifests'
                                  )
                              ) = (
                                  SELECT count(DISTINCT json_extract(
                                      value, '$.artifact_kind'
                                  ))
                                  FROM json_each(
                                      claim_event.payload_json,
                                      '$.provider.artifact_manifests'
                                  )
                              )
                              AND (
                                  (
                                      json_extract(
                                          claim_event.payload_json,
                                          '$.provider.synthetic'
                                      ) = 1
                                      AND NEW.provider_id LIKE 'synthetic.%'
                                      AND json_extract(
                                          claim_event.payload_json,
                                          '$.provider.verified'
                                      ) = 0
                                  )
                                  OR (
                                      json_extract(
                                          claim_event.payload_json,
                                          '$.provider.synthetic'
                                      ) = 0
                                      AND NEW.provider_id NOT LIKE
                                          'synthetic.%'
                                  )
                              )
                              AND json_extract(
                                  claim_event.payload_json,
                                  '$.provider.shadow_only'
                              ) = CASE
                                  WHEN json_extract(
                                      claim_event.payload_json,
                                      '$.provider.synthetic'
                                  ) = 1
                                    OR json_extract(
                                      claim_event.payload_json,
                                      '$.provider.verified'
                                  ) = 0
                                    OR json_extract(
                                      claim_event.payload_json,
                                      '$.provider.declared_kind'
                                  ) IN ('model', 'imported')
                                  THEN 1 ELSE 0 END
                              AND NOT (
                                  json_extract(
                                      claim_event.payload_json,
                                      '$.provider.verified'
                                  ) = 1
                                  AND json_extract(
                                      claim_event.payload_json,
                                      '$.provider.declared_kind'
                                  ) IN ('model', 'imported')
                              )
                              AND (
                                  json_extract(
                                      claim_event.payload_json,
                                      '$.provider.verified'
                                  ) = 0
                                  OR json_extract(
                                      claim_event.payload_json,
                                      '$.provider.declared_kind'
                                  ) != 'deterministic'
                                  OR json_array_length(
                                      claim_event.payload_json,
                                      '$.provider.check_set_manifests'
                                  ) > 0
                                  OR json_array_length(
                                      claim_event.payload_json,
                                      '$.provider.artifact_manifests'
                                  ) > 0
                              )
                          )
                      )
                ) THEN RAISE(
                    ABORT, 'scoring claim does not match its event'
                ) END;
            END;

            CREATE TRIGGER performance_scoring_claims_no_update
            BEFORE UPDATE ON performance_scoring_claims BEGIN
                SELECT RAISE(ABORT, 'performance scoring claims are immutable');
            END;

            CREATE TRIGGER performance_scoring_claims_no_delete
            BEFORE DELETE ON performance_scoring_claims BEGIN
                SELECT RAISE(ABORT, 'performance scoring claims are immutable');
            END;

            CREATE TRIGGER events_respect_performance_scoring_claim
            BEFORE INSERT ON events
            WHEN NEW.idempotency_key IS NOT NULL
             AND EXISTS (
                 SELECT 1 FROM performance_scoring_claims claim
                 WHERE claim.idempotency_key = NEW.idempotency_key
             )
            BEGIN
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM performance_scoring_claims claim
                    WHERE claim.idempotency_key = NEW.idempotency_key
                      AND NEW.event_type = 'TaskEvaluationRecorded'
                      AND json_extract(
                          NEW.metadata_json, '$.command_hash'
                      ) = claim.command_hash
                      AND json_extract(
                          NEW.payload_json, '$.attempt_id'
                      ) = claim.attempt_id
                      AND json_extract(
                          NEW.payload_json, '$.through_sequence'
                      ) = claim.through_sequence
                      AND json_extract(
                          NEW.payload_json, '$.evaluation.id'
                      ) = claim.evaluation_id
                ) THEN RAISE(
                    ABORT, 'event does not complete its performance scoring claim'
                ) END;
            END;

            CREATE TRIGGER
            performance_scoring_reconciliations_validate_insert
            BEFORE INSERT ON performance_scoring_reconciliations BEGIN
                SELECT CASE WHEN NEW.idempotency_key IS NOT NULL
                  AND EXISTS (
                      SELECT 1
                      FROM performance_scoring_claims claim
                      WHERE claim.idempotency_key = NEW.idempotency_key
                  )
                THEN RAISE(
                    ABORT,
                    'reconciliation idempotency key belongs to a scoring claim'
                ) END;
                SELECT CASE WHEN EXISTS (
                    SELECT 1 FROM events event
                    WHERE event.idempotency_key = NEW.idempotency_key
                      AND event.event_id != NEW.event_id
                ) THEN RAISE(
                    ABORT,
                    'reconciliation idempotency key already has an event'
                ) END;
                SELECT CASE WHEN EXISTS (
                    SELECT 1
                    FROM performance_scoring_reconciliations prior
                    WHERE prior.claim_id = NEW.claim_id
                      AND prior.outcome IN (
                          'completed', 'definitely_absent'
                      )
                ) THEN RAISE(
                    ABORT, 'performance scoring claim is already reconciled'
                ) END;
                SELECT CASE WHEN EXISTS (
                    SELECT 1
                    FROM task_evaluations evaluation
                    WHERE evaluation.id = NEW.evaluation_id
                      AND evaluation.attempt_id = NEW.attempt_id
                ) THEN RAISE(
                    ABORT,
                    'completed performance scoring claim cannot be reconciled'
                ) END;
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM performance_scoring_claims claim
                    JOIN performance_attempts attempt
                      ON attempt.id = claim.attempt_id
                    JOIN events claim_event
                      ON claim_event.event_id = claim.event_id
                    JOIN events reconciliation_event
                      ON reconciliation_event.event_id = NEW.event_id
                    WHERE claim.id = NEW.claim_id
                      AND claim.claim_schema_version = 2
                      AND claim.attempt_id = NEW.attempt_id
                      AND claim.evaluation_id = NEW.evaluation_id
                      AND claim.scoring_request_digest =
                          NEW.scoring_request_digest
                      AND claim.provider_binding_digest =
                          NEW.provider_binding_digest
                      AND claim.provider_operation_digest =
                          NEW.provider_operation_digest
                      AND reconciliation_event.event_type =
                          'PerformanceScoringReconciled'
                      AND reconciliation_event.schema_version = 1
                      AND reconciliation_event.stream_id =
                          'learner:' || attempt.learner_id
                      AND reconciliation_event.learner_id =
                          attempt.learner_id
                      AND reconciliation_event.session_id IS NULL
                      AND reconciliation_event.idempotency_key =
                          'performance-score-reconcile:v1:' ||
                          NEW.command_hash
                      AND reconciliation_event.correlation_id =
                          NEW.attempt_id
                      AND reconciliation_event.causation_id =
                          claim.event_id
                      AND reconciliation_event.occurred_at =
                          NEW.reconciled_at
                      AND reconciliation_event.stream_version >
                          claim_event.stream_version
                      AND NOT EXISTS (
                          SELECT 1
                          FROM performance_scoring_reconciliations prior
                          JOIN events prior_event
                            ON prior_event.event_id = prior.event_id
                          WHERE prior.claim_id = NEW.claim_id
                            AND (
                                prior_event.stream_id !=
                                    reconciliation_event.stream_id
                                OR prior_event.stream_version >=
                                    reconciliation_event.stream_version
                            )
                      )
                      AND json_extract(
                          reconciliation_event.metadata_json,
                          '$.reconciliation_schema_version'
                      ) = 1
                      AND json_extract(
                          reconciliation_event.metadata_json,
                          '$.command_hash'
                      ) = NEW.command_hash
                      AND json_extract(
                          reconciliation_event.metadata_json,
                          '$.shadow_only'
                      ) = 1
                      AND json_extract(
                          reconciliation_event.metadata_json,
                          '$.observational_only'
                      ) = 1
                      AND json_extract(
                          reconciliation_event.metadata_json,
                          '$.automatic_retry_allowed'
                      ) = 0
                      AND json_extract(
                          reconciliation_event.metadata_json,
                          '$.projection_applied'
                      ) = 0
                      AND json_extract(
                          reconciliation_event.metadata_json,
                          '$.certification_applied'
                      ) = 0
                      AND json_extract(
                          reconciliation_event.metadata_json,
                          '$.skill_authority'
                      ) = 0
                      AND json_type(
                          reconciliation_event.metadata_json,
                          '$.reconciliation_schema_version'
                      ) = 'integer'
                      AND json_type(
                          reconciliation_event.metadata_json,
                          '$.command_hash'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.metadata_json,
                          '$.shadow_only'
                      ) = 'true'
                      AND json_type(
                          reconciliation_event.metadata_json,
                          '$.observational_only'
                      ) = 'true'
                      AND json_type(
                          reconciliation_event.metadata_json,
                          '$.automatic_retry_allowed'
                      ) = 'false'
                      AND json_type(
                          reconciliation_event.metadata_json,
                          '$.projection_applied'
                      ) = 'false'
                      AND json_type(
                          reconciliation_event.metadata_json,
                          '$.certification_applied'
                      ) = 'false'
                      AND json_type(
                          reconciliation_event.metadata_json,
                          '$.skill_authority'
                      ) = 'false'
                      AND (
                          SELECT count(*)
                          FROM json_each(
                              reconciliation_event.metadata_json
                          )
                      ) = 8
                      AND NOT EXISTS (
                          SELECT 1
                          FROM json_each(
                              reconciliation_event.metadata_json
                          )
                          WHERE key NOT IN (
                              'reconciliation_schema_version',
                              'command_hash',
                              'observational_only',
                              'automatic_retry_allowed',
                              'projection_applied',
                              'certification_applied',
                              'skill_authority',
                              'shadow_only'
                          )
                      )
                      AND (
                          SELECT count(*)
                          FROM json_each(
                              reconciliation_event.payload_json
                          )
                      ) = 18
                      AND NOT EXISTS (
                          SELECT 1
                          FROM json_each(
                              reconciliation_event.payload_json
                          )
                          WHERE key NOT IN (
                              'reconciliation_id',
                              'caller_idempotency_key',
                              'claim_id',
                              'attempt_id',
                              'evaluation_id',
                              'outcome',
                              'scoring_request_digest',
                              'provider_binding_digest',
                              'provider_operation_digest',
                              'reconciler_id',
                              'reconciler_version',
                              'reconciliation_binding_digest',
                              'receipt',
                              'receipt_digest',
                              'normalized_result_digest',
                              'reconciled_at',
                              'command_hash',
                              'reconciler'
                          )
                      )
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciliation_id'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.caller_idempotency_key'
                      ) IN ('null', 'text')
                      AND json_type(
                          reconciliation_event.payload_json, '$.claim_id'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json, '$.attempt_id'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json, '$.evaluation_id'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json, '$.outcome'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.scoring_request_digest'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.provider_binding_digest'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.provider_operation_digest'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler_id'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler_version'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciliation_binding_digest'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json, '$.receipt'
                      ) = 'object'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.receipt_digest'
                      ) = 'text'
                      AND (
                          (
                              NEW.outcome = 'completed'
                              AND json_type(
                                  reconciliation_event.payload_json,
                                  '$.normalized_result_digest'
                              ) = 'text'
                          )
                          OR (
                              NEW.outcome IN (
                                  'unknown', 'definitely_absent'
                              )
                              AND json_type(
                                  reconciliation_event.payload_json,
                                  '$.normalized_result_digest'
                              ) = 'null'
                          )
                      )
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciled_at'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json, '$.command_hash'
                      ) = 'text'
                      AND (
                          (
                              length(NEW.reconciled_at) = 25
                              AND NEW.reconciled_at GLOB
                                  '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00'
                          )
                          OR (
                              length(NEW.reconciled_at) = 32
                              AND NEW.reconciled_at GLOB
                                  '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00'
                              AND substr(
                                  NEW.reconciled_at, 21, 6
                              ) != '000000'
                          )
                      )
                      AND substr(NEW.reconciled_at, 1, 4) != '0000'
                      AND substr(NEW.reconciled_at, 5, 1) = '-'
                      AND substr(NEW.reconciled_at, 8, 1) = '-'
                      AND substr(NEW.reconciled_at, 11, 1) = 'T'
                      AND substr(NEW.reconciled_at, 14, 1) = ':'
                      AND substr(NEW.reconciled_at, 17, 1) = ':'
                      AND substr(NEW.reconciled_at, -6) = '+00:00'
                      AND substr(NEW.reconciled_at, 12, 2)
                          BETWEEN '00' AND '23'
                      AND substr(NEW.reconciled_at, 15, 2)
                          BETWEEN '00' AND '59'
                      AND substr(NEW.reconciled_at, 18, 2)
                          BETWEEN '00' AND '59'
                      AND julianday(NEW.reconciled_at) IS NOT NULL
                      AND strftime(
                          '%Y-%m-%d', NEW.reconciled_at
                      ) = substr(NEW.reconciled_at, 1, 10)
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciliation_id'
                      ) = NEW.id
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.caller_idempotency_key'
                      ) IS NEW.idempotency_key
                      AND json_extract(
                          reconciliation_event.payload_json, '$.claim_id'
                      ) = NEW.claim_id
                      AND json_extract(
                          reconciliation_event.payload_json, '$.attempt_id'
                      ) = NEW.attempt_id
                      AND json_extract(
                          reconciliation_event.payload_json, '$.evaluation_id'
                      ) = NEW.evaluation_id
                      AND json_extract(
                          reconciliation_event.payload_json, '$.outcome'
                      ) = NEW.outcome
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.scoring_request_digest'
                      ) = NEW.scoring_request_digest
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.provider_binding_digest'
                      ) = NEW.provider_binding_digest
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.provider_operation_digest'
                      ) = NEW.provider_operation_digest
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler_id'
                      ) = NEW.reconciler_id
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler_version'
                      ) = NEW.reconciler_version
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciliation_binding_digest'
                      ) = NEW.reconciliation_binding_digest
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.receipt_digest'
                      ) = NEW.receipt_digest
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.normalized_result_digest'
                      ) IS NEW.normalized_result_digest
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciled_at'
                      ) = NEW.reconciled_at
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.command_hash'
                      ) = NEW.command_hash
                      AND json_extract(
                          reconciliation_event.payload_json, '$.receipt'
                      ) = json(NEW.receipt_json)
                      AND json_type(
                          reconciliation_event.payload_json, '$.reconciler'
                      ) = 'object'
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler.provider_id'
                      ) = claim.provider_id
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler.provider_version'
                      ) = claim.provider_version
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler.reconciler_id'
                      ) = NEW.reconciler_id
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler.reconciler_version'
                      ) = NEW.reconciler_version
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler.binding_digest'
                      ) = NEW.reconciliation_binding_digest
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler.observational_only'
                      ) = 1
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler.skill_authority'
                      ) = 0
                      AND (
                          NEW.outcome != 'definitely_absent'
                          OR json_extract(
                              reconciliation_event.payload_json,
                              '$.reconciler.can_prove_absence'
                          ) = 1
                      )
                      AND (
                          SELECT count(*)
                          FROM json_each(
                              reconciliation_event.payload_json,
                              '$.reconciler'
                          )
                      ) = 10
                      AND NOT EXISTS (
                          SELECT 1
                          FROM json_each(
                              reconciliation_event.payload_json,
                              '$.reconciler'
                          )
                          WHERE key NOT IN (
                              'provider_id',
                              'provider_version',
                              'reconciler_id',
                              'reconciler_version',
                              'manifest_digest',
                              'binding_digest',
                              'synthetic',
                              'can_prove_absence',
                              'observational_only',
                              'skill_authority'
                          )
                      )
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler.provider_id'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler.provider_version'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler.reconciler_id'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler.reconciler_version'
                      ) = 'text'
                      AND substr(NEW.reconciler_id, 1, 1)
                          GLOB '[A-Za-z0-9]'
                      AND NEW.reconciler_id NOT GLOB
                          '*[^A-Za-z0-9._:-]*'
                      AND substr(NEW.reconciler_version, 1, 1)
                          GLOB '[A-Za-z0-9]'
                      AND NEW.reconciler_version NOT GLOB
                          '*[^A-Za-z0-9._:-]*'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler.manifest_digest'
                      ) = 'text'
                      AND length(json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler.manifest_digest'
                      )) = 64
                      AND json_extract(
                          reconciliation_event.payload_json,
                          '$.reconciler.manifest_digest'
                      ) NOT GLOB '*[^0-9a-f]*'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler.binding_digest'
                      ) = 'text'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler.synthetic'
                      ) IN ('true', 'false')
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler.can_prove_absence'
                      ) IN ('true', 'false')
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler.observational_only'
                      ) = 'true'
                      AND json_type(
                          reconciliation_event.payload_json,
                          '$.reconciler.skill_authority'
                      ) = 'false'
                      AND (
                          (
                              json_extract(
                                  reconciliation_event.payload_json,
                                  '$.reconciler.synthetic'
                              ) = 1
                              AND NEW.reconciler_id LIKE 'synthetic.%'
                          )
                          OR (
                              json_extract(
                                  reconciliation_event.payload_json,
                                  '$.reconciler.synthetic'
                              ) = 0
                              AND NEW.reconciler_id NOT LIKE 'synthetic.%'
                          )
                      )
                      AND (
                          SELECT count(*) FROM json_each(NEW.receipt_json)
                      ) = 21
                      AND NOT EXISTS (
                          SELECT 1 FROM json_each(NEW.receipt_json)
                          WHERE key NOT IN (
                              'claim_id',
                              'attempt_id',
                              'evaluation_id',
                              'through_sequence',
                              'provider_id',
                              'provider_version',
                              'reconciler_id',
                              'reconciler_version',
                              'action_trace_digest',
                              'command_hash',
                              'scoring_request_digest',
                              'provider_binding_digest',
                              'outcome',
                              'observed_at',
                              'completed_at',
                              'result_digest',
                              'reason_code',
                              'provider_operation_digest',
                              'provider_receipt_digest',
                              'attestation_digest',
                              'schema_version'
                          )
                      )
                      AND json_extract(
                          NEW.receipt_json, '$.schema_version'
                      ) = 1
                      AND json_type(
                          NEW.receipt_json, '$.schema_version'
                      ) = 'integer'
                      AND json_type(
                          NEW.receipt_json, '$.claim_id'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.attempt_id'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.evaluation_id'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.through_sequence'
                      ) = 'integer'
                      AND json_type(
                          NEW.receipt_json, '$.provider_id'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.provider_version'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.reconciler_id'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.reconciler_version'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.action_trace_digest'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.command_hash'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.scoring_request_digest'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.provider_binding_digest'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.outcome'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.observed_at'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.reason_code'
                      ) = 'text'
                      AND length(json_extract(
                          NEW.receipt_json, '$.reason_code'
                      )) BETWEEN 1 AND 128
                      AND substr(json_extract(
                          NEW.receipt_json, '$.reason_code'
                      ), 1, 1) GLOB '[A-Za-z0-9]'
                      AND json_extract(
                          NEW.receipt_json, '$.reason_code'
                      ) NOT GLOB '*[^A-Za-z0-9._:-]*'
                      AND json_type(
                          NEW.receipt_json, '$.provider_operation_digest'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.provider_receipt_digest'
                      ) = 'text'
                      AND json_type(
                          NEW.receipt_json, '$.attestation_digest'
                      ) = 'text'
                      AND (
                          (
                              length(json_extract(
                                  NEW.receipt_json, '$.observed_at'
                              )) = 25
                              AND json_extract(
                                  NEW.receipt_json, '$.observed_at'
                              ) GLOB
                                  '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00'
                          )
                          OR (
                              length(json_extract(
                                  NEW.receipt_json, '$.observed_at'
                              )) = 32
                              AND json_extract(
                                  NEW.receipt_json, '$.observed_at'
                              ) GLOB
                                  '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00'
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.observed_at'
                              ), 21, 6) != '000000'
                          )
                      )
                      AND substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), 1, 4) != '0000'
                      AND substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), 5, 1) = '-'
                      AND substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), 8, 1) = '-'
                      AND substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), 11, 1) = 'T'
                      AND substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), 14, 1) = ':'
                      AND substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), 17, 1) = ':'
                      AND substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), -6) = '+00:00'
                      AND substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), 12, 2) BETWEEN '00' AND '23'
                      AND substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), 15, 2) BETWEEN '00' AND '59'
                      AND substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), 18, 2) BETWEEN '00' AND '59'
                      AND julianday(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      )) IS NOT NULL
                      AND strftime(
                          '%Y-%m-%d',
                          json_extract(
                              NEW.receipt_json, '$.observed_at'
                          )
                      ) = substr(json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ), 1, 10)
                      AND json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ) >= claim.claimed_at
                      AND json_extract(
                          NEW.receipt_json, '$.observed_at'
                      ) <= NEW.reconciled_at
                      AND json_extract(
                          NEW.receipt_json, '$.claim_id'
                      ) = claim.id
                      AND json_extract(
                          NEW.receipt_json, '$.attempt_id'
                      ) = claim.attempt_id
                      AND json_extract(
                          NEW.receipt_json, '$.evaluation_id'
                      ) = claim.evaluation_id
                      AND json_extract(
                          NEW.receipt_json, '$.through_sequence'
                      ) = claim.through_sequence
                      AND json_extract(
                          NEW.receipt_json, '$.provider_id'
                      ) = claim.provider_id
                      AND json_extract(
                          NEW.receipt_json, '$.provider_version'
                      ) = claim.provider_version
                      AND json_extract(
                          NEW.receipt_json, '$.reconciler_id'
                      ) = NEW.reconciler_id
                      AND json_extract(
                          NEW.receipt_json, '$.reconciler_version'
                      ) = NEW.reconciler_version
                      AND json_extract(
                          NEW.receipt_json, '$.action_trace_digest'
                      ) = claim.action_trace_digest
                      AND json_extract(
                          NEW.receipt_json, '$.command_hash'
                      ) = claim.command_hash
                      AND json_extract(
                          NEW.receipt_json, '$.scoring_request_digest'
                      ) = claim.scoring_request_digest
                      AND json_extract(
                          NEW.receipt_json, '$.provider_binding_digest'
                      ) = claim.provider_binding_digest
                      AND json_extract(
                          NEW.receipt_json, '$.provider_operation_digest'
                      ) = claim.provider_operation_digest
                      AND json_extract(
                          NEW.receipt_json, '$.outcome'
                      ) = NEW.outcome
                      AND length(json_extract(
                          NEW.receipt_json, '$.provider_receipt_digest'
                      )) = 64
                      AND json_extract(
                          NEW.receipt_json, '$.provider_receipt_digest'
                      ) NOT GLOB '*[^0-9a-f]*'
                      AND length(json_extract(
                          NEW.receipt_json, '$.attestation_digest'
                      )) = 64
                      AND json_extract(
                          NEW.receipt_json, '$.attestation_digest'
                      ) NOT GLOB '*[^0-9a-f]*'
                      AND (
                          (
                              NEW.outcome = 'completed'
                              AND json_type(
                                  NEW.receipt_json, '$.completed_at'
                              ) = 'text'
                              AND (
                                  (
                                      length(json_extract(
                                          NEW.receipt_json, '$.completed_at'
                                      )) = 25
                                      AND json_extract(
                                          NEW.receipt_json, '$.completed_at'
                                      ) GLOB
                                          '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]+00:00'
                                  )
                                  OR (
                                      length(json_extract(
                                          NEW.receipt_json, '$.completed_at'
                                      )) = 32
                                      AND json_extract(
                                          NEW.receipt_json, '$.completed_at'
                                      ) GLOB
                                          '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]+00:00'
                                      AND substr(json_extract(
                                          NEW.receipt_json, '$.completed_at'
                                      ), 21, 6) != '000000'
                                  )
                              )
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), 1, 4) != '0000'
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), 5, 1) = '-'
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), 8, 1) = '-'
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), 11, 1) = 'T'
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), 14, 1) = ':'
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), 17, 1) = ':'
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), -6) = '+00:00'
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), 12, 2) BETWEEN '00' AND '23'
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), 15, 2) BETWEEN '00' AND '59'
                              AND substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), 18, 2) BETWEEN '00' AND '59'
                              AND julianday(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              )) IS NOT NULL
                              AND strftime(
                                  '%Y-%m-%d',
                                  json_extract(
                                      NEW.receipt_json, '$.completed_at'
                                  )
                              ) = substr(json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ), 1, 10)
                              AND json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ) >= claim.claimed_at
                              AND json_extract(
                                  NEW.receipt_json, '$.completed_at'
                              ) <= json_extract(
                                  NEW.receipt_json, '$.observed_at'
                              )
                              AND json_type(
                                  NEW.receipt_json, '$.result_digest'
                              ) = 'text'
                              AND length(json_extract(
                                  NEW.receipt_json, '$.result_digest'
                              )) = 64
                              AND json_extract(
                                  NEW.receipt_json, '$.result_digest'
                              ) NOT GLOB '*[^0-9a-f]*'
                          )
                          OR (
                              NEW.outcome IN (
                                  'unknown', 'definitely_absent'
                              )
                              AND json_type(
                                  NEW.receipt_json, '$.completed_at'
                              ) = 'null'
                              AND json_type(
                                  NEW.receipt_json, '$.result_digest'
                              ) = 'null'
                          )
                      )
                ) THEN RAISE(
                    ABORT,
                    'performance scoring reconciliation does not match its claim/event/receipt'
                ) END;
            END;

            CREATE TRIGGER performance_scoring_reconciliations_no_update
            BEFORE UPDATE ON performance_scoring_reconciliations BEGIN
                SELECT RAISE(
                    ABORT, 'performance scoring reconciliations are immutable'
                );
            END;

            CREATE TRIGGER performance_scoring_reconciliations_no_delete
            BEFORE DELETE ON performance_scoring_reconciliations BEGIN
                SELECT RAISE(
                    ABORT, 'performance scoring reconciliations are immutable'
                );
            END;

            CREATE TRIGGER
            events_respect_performance_scoring_reconciliation
            BEFORE INSERT ON events
            WHEN NEW.idempotency_key IS NOT NULL
             AND EXISTS (
                 SELECT 1
                 FROM performance_scoring_reconciliations reconciliation
                 WHERE reconciliation.idempotency_key = NEW.idempotency_key
             )
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'event idempotency key is reserved by a reconciliation'
                );
            END;

            CREATE TRIGGER task_evaluations_validate_scoring_claim
            BEFORE INSERT ON task_evaluations BEGIN
                SELECT CASE WHEN EXISTS (
                    SELECT 1
                    FROM performance_scoring_claims claim
                    JOIN events evaluation_event
                      ON evaluation_event.event_id = NEW.event_id
                    WHERE claim.command_hash = NEW.command_hash
                      AND (
                          claim.attempt_id != NEW.attempt_id
                          OR claim.evaluation_id != NEW.id
                          OR claim.through_sequence != NEW.through_sequence
                          OR (
                              evaluation_event.idempotency_key
                                  IS NOT claim.idempotency_key
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM performance_scoring_reconciliations
                                       reconciliation
                                  WHERE reconciliation.claim_id = claim.id
                                    AND reconciliation.event_id =
                                        evaluation_event.causation_id
                                    AND reconciliation.outcome = 'completed'
                                    AND evaluation_event.idempotency_key IS NULL
                              )
                          )
                      )
                ) THEN RAISE(
                    ABORT, 'task evaluation does not complete its scoring claim'
                ) END;
                SELECT CASE WHEN EXISTS (
                    SELECT 1
                    FROM performance_scoring_claims claim
                    JOIN performance_scoring_reconciliations reconciliation
                      ON reconciliation.claim_id = claim.id
                    WHERE claim.command_hash = NEW.command_hash
                      AND claim.attempt_id = NEW.attempt_id
                      AND claim.evaluation_id = NEW.id
                      AND reconciliation.outcome = 'definitely_absent'
                ) THEN RAISE(
                    ABORT,
                    'definitely absent scoring operation cannot be evaluated'
                ) END;
                SELECT CASE WHEN json_extract(
                    NEW.authority_json,
                    '$.normalized_result.normalization_mode'
                ) = 'registered_provider' AND NOT EXISTS (
                    SELECT 1 FROM performance_scoring_claims claim
                    WHERE claim.command_hash = NEW.command_hash
                      AND claim.attempt_id = NEW.attempt_id
                      AND claim.evaluation_id = NEW.id
                      AND claim.through_sequence = NEW.through_sequence
                ) AND NOT EXISTS (
                    SELECT 1 FROM events exemption
                    WHERE exemption.event_type =
                          'PerformanceScoringLegacyExempted'
                      AND exemption.schema_version = 1
                      AND json_extract(
                          exemption.payload_json, '$.evaluation_id'
                      ) = NEW.id
                      AND json_extract(
                          exemption.payload_json, '$.attempt_id'
                      ) = NEW.attempt_id
                      AND json_extract(
                          exemption.payload_json, '$.command_hash'
                      ) = NEW.command_hash
                ) THEN RAISE(
                    ABORT, 'registered evaluation lacks its scoring claim'
                ) END;
            END;

            CREATE TRIGGER task_evaluations_validate_insert
            BEFORE INSERT ON task_evaluations BEGIN
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1 FROM performance_actions submission
                    WHERE submission.attempt_id = NEW.attempt_id
                      AND submission.action_type = 'submitted'
                      AND submission.sequence = NEW.through_sequence
                ) THEN RAISE(
                    ABORT, 'task evaluation lacks its submitted trace boundary'
                ) END;
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM performance_attempts attempt
                    JOIN events evaluation_event
                      ON evaluation_event.event_id = NEW.event_id
                    WHERE attempt.id = NEW.attempt_id
                      AND evaluation_event.event_type = 'TaskEvaluationRecorded'
                      AND evaluation_event.schema_version = 1
                      AND evaluation_event.stream_id =
                          'learner:' || attempt.learner_id
                      AND evaluation_event.learner_id = attempt.learner_id
                      AND (
                          (
                              evaluation_event.session_id =
                                  attempt.session_id
                              AND (
                                  evaluation_event.causation_id =
                                      NEW.attempt_id
                                  OR EXISTS (
                                      SELECT 1
                                      FROM performance_scoring_claims claim
                                      WHERE claim.event_id =
                                            evaluation_event.causation_id
                                        AND claim.attempt_id = NEW.attempt_id
                                        AND claim.evaluation_id = NEW.id
                                        AND claim.command_hash =
                                            NEW.command_hash
                                  )
                              )
                              AND NOT EXISTS (
                                  SELECT 1
                                  FROM performance_scoring_claims claim
                                  JOIN performance_scoring_reconciliations
                                       reconciliation
                                    ON reconciliation.claim_id = claim.id
                                  WHERE claim.attempt_id = NEW.attempt_id
                                    AND claim.evaluation_id = NEW.id
                                    AND claim.command_hash =
                                        NEW.command_hash
                                    AND reconciliation.outcome IN (
                                        'completed', 'definitely_absent'
                                    )
                              )
                          )
                          OR (
                              evaluation_event.session_id IS NULL
                              AND EXISTS (
                                  SELECT 1
                                  FROM performance_scoring_reconciliations
                                       reconciliation
                                  JOIN performance_scoring_claims claim
                                    ON claim.id =
                                       reconciliation.claim_id
                                  JOIN events reconciliation_event
                                    ON reconciliation_event.event_id =
                                       reconciliation.event_id
                                  WHERE reconciliation.event_id =
                                        evaluation_event.causation_id
                                    AND reconciliation.outcome = 'completed'
                                    AND reconciliation.attempt_id =
                                        NEW.attempt_id
                                    AND reconciliation.evaluation_id = NEW.id
                                    AND claim.command_hash =
                                        NEW.command_hash
                                    AND json_extract(
                                        NEW.authority_json,
                                        '$.normalized_result_digest'
                                    ) =
                                        reconciliation.normalized_result_digest
                                    AND json_extract(
                                        evaluation_event.payload_json,
                                        '$.authority.normalized_result_digest'
                                    ) =
                                        reconciliation.normalized_result_digest
                                    AND evaluation_event.stream_version >
                                        reconciliation_event.stream_version
                              )
                          )
                      )
                      AND evaluation_event.recorded_at = NEW.recorded_at
                      AND json_extract(
                          evaluation_event.payload_json, '$.evaluation.id'
                      ) = NEW.id
                      AND json_extract(
                          evaluation_event.payload_json, '$.evaluation_digest'
                      ) = NEW.evaluation_digest
                      AND json_extract(
                          evaluation_event.payload_json, '$.authority'
                      ) = json(NEW.authority_json)
                      AND json_extract(
                          evaluation_event.payload_json, '$.through_sequence'
                      ) = NEW.through_sequence
                ) THEN RAISE(
                    ABORT, 'task evaluation does not match its event'
                ) END;
            END;

            CREATE TRIGGER shadow_evidence_bundles_validate_insert
            BEFORE INSERT ON shadow_evidence_bundles BEGIN
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM task_evaluations evaluation
                    JOIN performance_attempts attempt
                      ON attempt.id = evaluation.attempt_id
                    JOIN events evaluation_event
                      ON evaluation_event.event_id = evaluation.event_id
                    JOIN events bundle_event
                      ON bundle_event.event_id = NEW.event_id
                    WHERE evaluation.id = NEW.evaluation_id
                      AND evaluation.attempt_id = NEW.attempt_id
                      AND bundle_event.event_type = 'ShadowEvidenceReduced'
                      AND bundle_event.schema_version = 1
                      AND bundle_event.stream_id =
                          'learner:' || attempt.learner_id
                      AND bundle_event.learner_id = attempt.learner_id
                      AND bundle_event.session_id IS
                          evaluation_event.session_id
                      AND bundle_event.causation_id = evaluation.id
                      AND bundle_event.stream_version >
                          evaluation_event.stream_version
                      AND bundle_event.recorded_at = NEW.recorded_at
                      AND json_extract(
                          bundle_event.payload_json, '$.bundle_id'
                      ) = NEW.id
                      AND json_extract(
                          bundle_event.payload_json, '$.bundle_digest'
                      ) = NEW.bundle_digest
                      AND json_extract(
                          bundle_event.payload_json, '$.projection_applied'
                      ) = 0
                      AND json_extract(
                          bundle_event.payload_json, '$.certification_applied'
                      ) = 0
                ) THEN RAISE(
                    ABORT, 'shadow evidence does not match its evaluation/event'
                ) END;
            END;
            """,
        )

    @staticmethod
    def _install_v8_learning_action_triggers(
        connection: sqlite3.Connection,
    ) -> None:
        """Bind immutable action projections to their semantic events."""
        _execute_sql_script(
            connection,
            """
            DROP TRIGGER IF EXISTS learning_artifacts_no_update;
            DROP TRIGGER IF EXISTS learning_artifacts_no_delete;
            DROP TRIGGER IF EXISTS learning_actions_validate_insert;
            DROP TRIGGER IF EXISTS learning_actions_no_update;
            DROP TRIGGER IF EXISTS learning_actions_no_delete;

            CREATE TRIGGER learning_artifacts_no_update
            BEFORE UPDATE ON learning_artifacts BEGIN
                SELECT RAISE(ABORT, 'learning artifacts are immutable');
            END;

            CREATE TRIGGER learning_artifacts_no_delete
            BEFORE DELETE ON learning_artifacts BEGIN
                SELECT RAISE(ABORT, 'learning artifacts are immutable');
            END;

            CREATE TRIGGER learning_actions_validate_insert
            BEFORE INSERT ON learning_actions BEGIN
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1
                    FROM decisions decision
                    JOIN sessions session ON session.id = decision.session_id
                    WHERE decision.id = NEW.decision_id
                      AND decision.session_id = NEW.session_id
                      AND decision.learner_id = NEW.learner_id
                      AND session.learner_id = NEW.learner_id
                ) THEN RAISE(ABORT, 'learning action does not match decision/session') END;
                SELECT CASE WHEN NEW.sequence != COALESCE((
                    SELECT MAX(action.sequence) + 1
                    FROM learning_actions action
                    WHERE action.decision_id = NEW.decision_id
                ), 1) THEN RAISE(ABORT, 'learning action sequence is not contiguous') END;
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1 FROM events event
                    WHERE event.event_id = NEW.event_id
                      AND event.event_type = 'LearnerActionRecorded'
                      AND event.schema_version = 1
                      AND event.stream_id = 'learner:' || NEW.learner_id
                      AND event.learner_id = NEW.learner_id
                      AND event.session_id = NEW.session_id
                      AND event.causation_id = NEW.decision_id
                      AND event.occurred_at = NEW.occurred_at
                      AND event.recorded_at = NEW.recorded_at
                      AND json_extract(event.payload_json, '$.action_id') = NEW.id
                      AND json_extract(event.payload_json, '$.decision_id') = NEW.decision_id
                      AND json_extract(event.payload_json, '$.sequence') = NEW.sequence
                      AND json_extract(event.payload_json, '$.stage') = NEW.stage
                      AND json_extract(event.payload_json, '$.action_type') = NEW.action_type
                      AND json_extract(event.payload_json, '$.payload') = json(NEW.payload_json)
                      AND (
                          (NEW.artifact_id IS NULL
                           AND json_type(event.payload_json, '$.artifact') = 'null')
                          OR EXISTS (
                              SELECT 1 FROM learning_artifacts artifact
                              WHERE artifact.id = NEW.artifact_id
                                AND json_extract(
                                    event.payload_json, '$.artifact.sha256'
                                ) = artifact.sha256
                                AND json_extract(
                                    event.payload_json, '$.artifact.size_bytes'
                                ) = artifact.size_bytes
                                AND json_extract(
                                    event.payload_json, '$.artifact.media_type'
                                ) = artifact.media_type
                          )
                      )
                ) THEN RAISE(ABORT, 'learning action does not match its event') END;
                SELECT CASE WHEN (
                    SELECT COUNT(*)
                    FROM events selection_event
                    JOIN events action_event ON action_event.event_id = NEW.event_id
                    WHERE selection_event.event_type = 'QuestionSelected'
                      AND selection_event.schema_version IN (1, 2, 3)
                      AND selection_event.stream_id = action_event.stream_id
                      AND selection_event.learner_id = NEW.learner_id
                      AND selection_event.session_id = NEW.session_id
                      AND json_extract(
                          selection_event.payload_json, '$.decision_id'
                      ) = NEW.decision_id
                      AND selection_event.stream_version < action_event.stream_version
                ) != 1 THEN RAISE(
                    ABORT, 'learning action lacks a unique prior selection event'
                ) END;
                SELECT CASE WHEN (
                    SELECT COUNT(*)
                    FROM events started_event
                    JOIN events action_event ON action_event.event_id = NEW.event_id
                    WHERE started_event.event_type = 'SessionStarted'
                      AND started_event.schema_version = 1
                      AND started_event.stream_id = action_event.stream_id
                      AND started_event.learner_id = NEW.learner_id
                      AND started_event.session_id = NEW.session_id
                      AND json_extract(
                          started_event.payload_json, '$.session_id'
                      ) = NEW.session_id
                      AND started_event.stream_version < action_event.stream_version
                ) != 1 THEN RAISE(
                    ABORT, 'learning action falls outside session start boundary'
                ) END;
                SELECT CASE WHEN EXISTS (
                    SELECT 1
                    FROM events ended_event
                    JOIN events action_event ON action_event.event_id = NEW.event_id
                    WHERE ended_event.event_type = 'SessionEnded'
                      AND ended_event.stream_id = action_event.stream_id
                      AND ended_event.learner_id = NEW.learner_id
                      AND ended_event.session_id = NEW.session_id
                      AND ended_event.stream_version <= action_event.stream_version
                ) THEN RAISE(
                    ABORT, 'learning action follows session end boundary'
                ) END;
                SELECT CASE WHEN EXISTS (
                    SELECT 1
                    FROM events invalidation_event
                    JOIN events action_event ON action_event.event_id = NEW.event_id
                    WHERE invalidation_event.event_type = 'DecisionInvalidated'
                      AND invalidation_event.stream_id = action_event.stream_id
                      AND invalidation_event.learner_id = NEW.learner_id
                      AND invalidation_event.session_id = NEW.session_id
                      AND invalidation_event.causation_id = NEW.decision_id
                      AND invalidation_event.stream_version <= action_event.stream_version
                ) THEN RAISE(
                    ABORT, 'learning action follows decision invalidation boundary'
                ) END;
                SELECT CASE WHEN EXISTS (
                    SELECT 1
                    FROM decisions decision
                    JOIN question_revocations revocation
                      ON revocation.question_id = decision.question_id
                    JOIN events revocation_event
                      ON revocation_event.event_id = revocation.event_id
                    JOIN events action_event ON action_event.event_id = NEW.event_id
                    WHERE decision.id = NEW.decision_id
                      AND action_event.recorded_at >= revocation_event.recorded_at
                ) THEN RAISE(
                    ABORT, 'learning action was recorded after emergency revocation'
                ) END;
                SELECT CASE WHEN NEW.stage IN ('unassisted', 'assisted') AND EXISTS (
                    SELECT 1
                    FROM events selection_event
                    JOIN events action_event ON action_event.event_id = NEW.event_id
                    JOIN events projection_event
                      ON projection_event.stream_id = action_event.stream_id
                     AND projection_event.event_type = 'LearnerProjectionAdvanced'
                     AND projection_event.stream_version
                         > selection_event.stream_version
                     AND projection_event.stream_version
                         < action_event.stream_version
                    WHERE selection_event.event_type = 'QuestionSelected'
                      AND selection_event.stream_id = action_event.stream_id
                      AND selection_event.session_id = NEW.session_id
                      AND json_extract(
                          selection_event.payload_json, '$.decision_id'
                      ) = NEW.decision_id
                ) THEN RAISE(
                    ABORT, 'pre-response learning action follows learner projection advance'
                ) END;
                SELECT CASE WHEN NEW.artifact_id IS NOT NULL AND NOT EXISTS (
                    SELECT 1
                    FROM learning_artifacts artifact
                    WHERE artifact.id = NEW.artifact_id
                      AND artifact.sha256 = CASE NEW.action_type
                          WHEN 'answer_revised' THEN
                              json_extract(NEW.payload_json, '$.answer_digest')
                          WHEN 'artifact_checkpoint' THEN
                              json_extract(NEW.payload_json, '$.artifact_digest')
                          WHEN 'explanation_checkpoint' THEN
                              json_extract(NEW.payload_json, '$.explanation_digest')
                          WHEN 'check_run' THEN
                              json_extract(NEW.payload_json, '$.result_digest')
                          WHEN 'submitted' THEN
                              json_extract(NEW.payload_json, '$.submission_digest')
                          WHEN 'feedback_shown' THEN
                              json_extract(NEW.payload_json, '$.feedback_digest')
                          ELSE NULL
                      END
                ) THEN RAISE(ABORT, 'learning action artifact does not match payload digest') END;
                SELECT CASE WHEN NEW.action_type = 'feedback_shown'
                    AND NEW.stage != 'post_feedback'
                    THEN RAISE(ABORT, 'feedback_shown requires post_feedback stage') END;
                SELECT CASE WHEN NEW.action_type IN (
                        'started', 'submitted', 'abandoned', 'feedback_shown'
                    ) AND EXISTS (
                        SELECT 1 FROM learning_actions action
                        WHERE action.decision_id = NEW.decision_id
                          AND action.action_type = NEW.action_type
                    ) THEN RAISE(ABORT, 'learning action repeats lifecycle singleton') END;
                SELECT CASE WHEN NEW.action_type = 'hint_requested' AND (
                    SELECT COUNT(*) FROM learning_actions action
                    WHERE action.decision_id = NEW.decision_id
                      AND action.action_type = 'hint_requested'
                ) >= 10000 THEN RAISE(
                    ABORT, 'learning action exceeds hint request bound'
                ) END;
                SELECT CASE WHEN NEW.action_type = 'started' AND EXISTS (
                        SELECT 1 FROM learning_actions action
                        WHERE action.decision_id = NEW.decision_id
                    ) THEN RAISE(ABORT, 'started must be the first learning action') END;
                SELECT CASE WHEN EXISTS (
                        SELECT 1 FROM learning_actions action
                        WHERE action.decision_id = NEW.decision_id
                          AND action.action_type = 'abandoned'
                    ) THEN RAISE(ABORT, 'learning action follows abandoned trace') END;
                SELECT CASE WHEN NEW.action_type = 'abandoned' AND EXISTS (
                        SELECT 1 FROM learning_actions action
                        WHERE action.decision_id = NEW.decision_id
                          AND action.action_type = 'submitted'
                    ) THEN RAISE(ABORT, 'learning trace cannot be submitted and abandoned') END;
                SELECT CASE WHEN NEW.stage IN ('unassisted', 'assisted') AND EXISTS (
                        SELECT 1 FROM learning_actions action
                        WHERE action.decision_id = NEW.decision_id
                          AND action.action_type = 'submitted'
                    ) THEN RAISE(ABORT, 'pre-response action follows submitted checkpoint') END;
                SELECT CASE WHEN NEW.stage IN ('unassisted', 'assisted')
                    AND NOT EXISTS (
                        SELECT 1 FROM decisions decision
                        JOIN sessions session ON session.id = decision.session_id
                        JOIN learners learner ON learner.id = decision.learner_id
                        WHERE decision.id = NEW.decision_id
                          AND decision.consumed_at IS NULL
                          AND decision.invalidated_at IS NULL
                          AND session.status = 'active'
                          AND session.phase = decision.phase
                          AND session.focus_concept_id IS decision.focus_concept_id
                          AND session.focus_misconception_id
                              IS decision.focus_misconception_id
                          AND session.focus_objective_id
                              IS decision.focus_objective_id
                          AND session.corpus_release_id = decision.corpus_release_id
                          AND session.revision = decision.session_revision + 1
                          AND learner.revision = decision.learner_revision
                    ) THEN RAISE(ABORT, 'pre-response action requires a pending decision') END;
                SELECT CASE WHEN NEW.stage = 'post_feedback'
                    AND NOT EXISTS (
                        SELECT 1 FROM decisions decision
                        JOIN sessions session ON session.id = decision.session_id
                        JOIN attempts attempt ON attempt.decision_id = decision.id
                        JOIN events response_event
                          ON response_event.event_id = attempt.event_id
                        JOIN events action_event
                          ON action_event.event_id = NEW.event_id
                        WHERE decision.id = NEW.decision_id
                          AND decision.invalidated_at IS NULL
                          AND decision.consumed_at IS NOT NULL
                          AND session.status = 'active'
                          AND attempt.answered_at <= NEW.occurred_at
                          AND action_event.stream_id = response_event.stream_id
                          AND action_event.stream_version > response_event.stream_version
                    ) THEN RAISE(ABORT, 'post-feedback action requires an answered decision') END;
            END;

            CREATE TRIGGER learning_actions_no_update
            BEFORE UPDATE ON learning_actions BEGIN
                SELECT RAISE(ABORT, 'learning actions are immutable');
            END;

            CREATE TRIGGER learning_actions_no_delete
            BEFORE DELETE ON learning_actions BEGIN
                SELECT RAISE(ABORT, 'learning actions are immutable');
            END;
            """,
        )

    @staticmethod
    def _install_v5_indexes(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_one_pending_decision
               ON decisions(session_id)
               WHERE consumed_at IS NULL AND invalidated_at IS NULL"""
        )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS idx_decisions_learner_question
               ON decisions(learner_id, question_id, created_at)"""
        )

    @staticmethod
    def _install_v6_authoring_triggers(connection: sqlite3.Connection) -> None:
        """Constrain job state transitions and make completed runs immutable."""

        _execute_sql_script(
            connection,
            """
            DROP TRIGGER IF EXISTS generation_jobs_validate_insert;
            DROP TRIGGER IF EXISTS generation_jobs_validate_update;
            DROP TRIGGER IF EXISTS generation_jobs_no_delete;
            DROP TRIGGER IF EXISTS generation_job_runs_validate_insert;
            DROP TRIGGER IF EXISTS generation_job_runs_validate_update;
            DROP TRIGGER IF EXISTS generation_job_runs_no_delete;

            CREATE TRIGGER generation_jobs_validate_insert
            BEFORE INSERT ON generation_jobs BEGIN
                SELECT CASE WHEN NEW.status != 'planned'
                    THEN RAISE(ABORT, 'new generation jobs must be planned') END;
                SELECT CASE WHEN NEW.provider IS NOT NULL OR NEW.model IS NOT NULL
                                  OR NEW.raw_output_json IS NOT NULL
                                  OR NEW.validation_json IS NOT NULL
                    THEN RAISE(ABORT, 'planned generation job has execution data') END;
            END;

            CREATE TRIGGER generation_jobs_validate_update
            BEFORE UPDATE ON generation_jobs BEGIN
                SELECT CASE WHEN NEW.id IS NOT OLD.id
                                  OR NEW.blueprint_json IS NOT OLD.blueprint_json
                                  OR NEW.prompt_version IS NOT OLD.prompt_version
                                  OR NEW.created_at IS NOT OLD.created_at
                    THEN RAISE(ABORT, 'generation job identity is immutable') END;
                SELECT CASE WHEN NOT (
                    (OLD.status = 'planned' AND NEW.status = 'running')
                    OR (OLD.status = 'running'
                        AND NEW.status IN ('reviewed', 'rejected', 'failed'))
                    OR (OLD.status IN ('rejected', 'failed')
                        AND NEW.status = 'planned')
                ) THEN RAISE(ABORT, 'invalid generation job transition') END;
                SELECT CASE WHEN NEW.status = 'planned' AND (
                                  NEW.provider IS NOT NULL OR NEW.model IS NOT NULL
                                  OR NEW.raw_output_json IS NOT NULL
                                  OR NEW.validation_json IS NOT NULL)
                    THEN RAISE(ABORT, 'planned generation job has execution data') END;
                SELECT CASE WHEN NEW.status = 'running' AND (
                                  NEW.provider IS NULL OR trim(NEW.provider) = ''
                                  OR NEW.model IS NULL OR trim(NEW.model) = ''
                                  OR NEW.raw_output_json IS NOT NULL
                                  OR NEW.validation_json IS NOT NULL)
                    THEN RAISE(ABORT, 'running generation job has invalid execution data') END;
                SELECT CASE WHEN NEW.status IN ('reviewed', 'rejected') AND (
                                  NEW.provider IS NULL OR trim(NEW.provider) = ''
                                  OR NEW.model IS NULL OR trim(NEW.model) = ''
                                  OR NEW.raw_output_json IS NULL
                                  OR NEW.validation_json IS NULL)
                    THEN RAISE(ABORT, 'terminal generation job lacks reviewed output') END;
                SELECT CASE WHEN NEW.status = 'failed' AND (
                                  NEW.provider IS NULL OR trim(NEW.provider) = ''
                                  OR NEW.model IS NULL OR trim(NEW.model) = ''
                                  OR NEW.validation_json IS NULL)
                    THEN RAISE(ABORT, 'failed generation job lacks error data') END;
            END;

            CREATE TRIGGER generation_jobs_no_delete
            BEFORE DELETE ON generation_jobs BEGIN
                SELECT RAISE(ABORT, 'generation jobs are immutable records');
            END;

            CREATE TRIGGER generation_job_runs_validate_insert
            BEFORE INSERT ON generation_job_runs BEGIN
                SELECT CASE WHEN NOT EXISTS (
                    SELECT 1 FROM generation_jobs job
                    WHERE job.id = NEW.job_id
                      AND job.status = 'running'
                      AND job.provider = NEW.provider
                      AND job.model = NEW.model
                      AND job.prompt_version = NEW.prompt_version
                ) THEN RAISE(ABORT, 'generation run does not match running job') END;
                SELECT CASE WHEN NEW.attempt != COALESCE((
                    SELECT MAX(run.attempt) + 1 FROM generation_job_runs run
                    WHERE run.job_id = NEW.job_id
                ), 1) THEN RAISE(ABORT, 'generation run attempt is not sequential') END;
                SELECT CASE WHEN EXISTS (
                    SELECT 1 FROM generation_job_runs run
                    WHERE run.job_id = NEW.job_id AND run.status = 'running'
                ) THEN RAISE(ABORT, 'generation job already has a running attempt') END;
            END;

            CREATE TRIGGER generation_job_runs_validate_update
            BEFORE UPDATE ON generation_job_runs BEGIN
                SELECT CASE WHEN OLD.status != 'running'
                                  OR NEW.status NOT IN ('reviewed', 'rejected', 'failed')
                    THEN RAISE(ABORT, 'completed generation runs are immutable') END;
                SELECT CASE WHEN NEW.id IS NOT OLD.id
                                  OR NEW.job_id IS NOT OLD.job_id
                                  OR NEW.attempt IS NOT OLD.attempt
                                  OR NEW.provider IS NOT OLD.provider
                                  OR NEW.model IS NOT OLD.model
                                  OR NEW.prompt_version IS NOT OLD.prompt_version
                                  OR NEW.source_context_sha256 IS NOT OLD.source_context_sha256
                                  OR NEW.started_at IS NOT OLD.started_at
                    THEN RAISE(ABORT, 'generation run identity is immutable') END;
            END;

            CREATE TRIGGER generation_job_runs_no_delete
            BEFORE DELETE ON generation_job_runs BEGIN
                SELECT RAISE(ABORT, 'generation runs are immutable');
            END;
            """,
        )

    @staticmethod
    def _drop_v6_authoring_triggers(connection: sqlite3.Connection) -> None:
        _execute_sql_script(
            connection,
            """
            DROP TRIGGER IF EXISTS generation_jobs_validate_insert;
            DROP TRIGGER IF EXISTS generation_jobs_validate_update;
            DROP TRIGGER IF EXISTS generation_jobs_no_delete;
            DROP TRIGGER IF EXISTS generation_job_runs_validate_insert;
            DROP TRIGGER IF EXISTS generation_job_runs_validate_update;
            DROP TRIGGER IF EXISTS generation_job_runs_no_delete;
            """,
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise ConflictError(
                "A read-only database cannot start a write transaction."
            )
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def import_corpus(
        self,
        concepts: Sequence[Concept],
        edges: Sequence[ConceptEdge],
        misconceptions: Sequence[Misconception],
        sources: Sequence[Source],
        questions: Sequence[Question],
        domains: Sequence[Domain] = (),
        topics: Sequence[Topic] = (),
    ) -> dict[str, int | str]:
        self._validate_corpus_activation(
            concepts, edges, misconceptions, sources, questions, domains, topics
        )
        objectives: dict[str, LearningObjective] = {}
        for question in questions:
            if question.objective is None:
                continue
            prior = objectives.get(question.objective.id)
            if prior is not None and prior != question.objective:
                raise ValidationError(
                    f"Learning objective {question.objective.id} has conflicting definitions."
                )
            objectives[question.objective.id] = question.objective
        graph_versions = {
            objective.objective_graph_version
            for objective in objectives.values()
        }
        objective_graph_version = (
            1 if graph_versions == {1} else None
        )
        objective_edges = sorted(
            {
                edge.id: edge
                for objective in objectives.values()
                for edge in objective.prerequisites
            }.values(),
            key=lambda edge: edge.id,
        )
        # Canonicalization is CPU work and must not extend SQLite's single-writer
        # critical section. Reuse these exact hashes for registry inserts and the
        # immutable release manifest.
        concept_hashes = {
            concept.id: concept_content_hash(concept) for concept in concepts
        }
        objective_hashes = {
            objective.id: objective_content_hash(objective)
            for objective in objectives.values()
        }
        misconception_hashes = {
            item.id: misconception_content_hash(item)
            for item in misconceptions
        }
        source_hashes = {
            source.id: source_content_hash(source) for source in sources
        }
        question_hashes = {
            question.id: question_content_hash(question) for question in questions
        }
        domain_hashes = {
            domain.id: domain_content_hash(domain) for domain in domains
        }
        topic_hashes = {
            topic.id: topic_content_hash(topic) for topic in topics
        }
        topic_by_concept = {
            concept_id: topic.id
            for topic in topics
            for concept_id in topic.concept_ids
        }
        legacy_unreviewed_generated_ids = tuple(
            sorted(
                question.id
                for question in questions
                if question.provenance.get("generated") is True
                and question.provenance.get("human_review") is not True
                and not question.status.eligible_for_adaptation
            )
        )
        question_topic_rows: list[tuple[str, str, str]] = []
        for question in questions:
            relations: dict[str, str] = {}
            for mapping in question.concepts:
                topic_id = topic_by_concept.get(mapping.concept_id)
                if topic_id is None:
                    continue
                relation = (
                    "primary"
                    if mapping.role.value == "primary"
                    else "cross"
                )
                if relation == "primary" or topic_id not in relations:
                    relations[topic_id] = relation
            question_topic_rows.extend(
                (question.id, topic_id, relation)
                for topic_id, relation in sorted(relations.items())
            )
        ordered_topics: list[Topic] = []
        remaining_topics = {topic.id: topic for topic in topics}
        while remaining_topics:
            ready = sorted(
                (
                    topic
                    for topic in remaining_topics.values()
                    if topic.parent_id is None
                    or topic.parent_id not in remaining_topics
                ),
                key=lambda topic: (topic.sort_order, topic.id),
            )
            if not ready:
                raise ValidationError("Curriculum topic hierarchy contains a cycle.")
            ordered_topics.extend(ready)
            for topic in ready:
                del remaining_topics[topic.id]
        release_payload = {
            "concepts": sorted(concept_hashes.items()),
            "edges": sorted(
                (
                    edge.source_id,
                    edge.target_id,
                    edge.relation.value,
                    float(edge.weight),
                )
                for edge in edges
            ),
            "misconceptions": sorted(misconception_hashes.items()),
            "sources": sorted(source_hashes.items()),
            "questions": sorted(
                (
                    question.id,
                    question_hashes[question.id],
                    question.status.value,
                )
                for question in questions
            ),
        }
        if objectives:
            release_payload.update(
                {
                    "learning_objectives": sorted(objective_hashes.items()),
                    "question_objectives": sorted(
                        (question.id, question.objective_id)
                        for question in questions
                        if question.objective_id is not None
                    ),
                    "option_objectives": sorted(
                        (
                            question.id,
                            option.id,
                            option.diagnostic_objective_id,
                        )
                        for question in questions
                        for option in question.options
                        if option.diagnostic_objective_id is not None
                    ),
                }
            )
        if objective_graph_version is not None:
            release_payload.update(
                {
                    "objective_graph_version": objective_graph_version,
                    "objective_edges": [
                        (
                            edge.id,
                            edge.source_id,
                            edge.target_id,
                            edge.relation.value,
                            float(edge.weight),
                            edge.rationale,
                        )
                        for edge in objective_edges
                    ],
                }
            )
        if domains or topics:
            release_payload.update(
                {
                    "domains": sorted(domain_hashes.items()),
                    "topics": sorted(topic_hashes.items()),
                    "question_topics": sorted(question_topic_rows),
                }
            )
        bundle_hash = _content_hash(release_payload)
        release_id = f"rel_{bundle_hash[:24]}"
        imported_at = datetime.now(timezone.utc).isoformat()
        with self.transaction() as connection:
            connection.execute(
                "CREATE TEMP TABLE incoming_question_ids(id TEXT PRIMARY KEY)"
            )
            connection.executemany(
                "INSERT INTO incoming_question_ids(id) VALUES (?)",
                ((question.id,) for question in questions),
            )
            existing_questions = {
                row["id"]: row
                for row in connection.execute(
                    """SELECT question.id, question.version,
                              question.content_hash, question.status
                       FROM questions question
                       JOIN incoming_question_ids incoming
                         ON incoming.id = question.id"""
                ).fetchall()
            }
            for concept in concepts:
                content_hash = concept_hashes[concept.id]
                existing = connection.execute(
                    "SELECT content_hash FROM concepts WHERE id = ?", (concept.id,)
                ).fetchone()
                if existing and existing["content_hash"] != content_hash:
                    raise ConflictError(
                        f"Concept {concept.id} is immutable; publish a new concept ID."
                    )
                connection.execute(
                    """INSERT OR IGNORE INTO concepts(
                           id, content_hash, name, description, domain, prior_mastery
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        concept.id,
                        content_hash,
                        concept.name,
                        concept.description,
                        concept.domain,
                        concept.prior_mastery,
                    ),
                )
            for objective in objectives.values():
                content_hash = objective_hashes[objective.id]
                existing = connection.execute(
                    "SELECT content_hash FROM learning_objectives WHERE id = ?",
                    (objective.id,),
                ).fetchone()
                if existing and existing["content_hash"] != content_hash:
                    raise ConflictError(
                        f"Learning objective {objective.id} is immutable; publish a new objective ID."
                    )
                connection.execute(
                    """INSERT OR IGNORE INTO learning_objectives(
                           id, content_hash, name, description, primary_concept_id,
                           supporting_concept_ids_json,
                           operation, evidence_type, prior_mastery
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        objective.id,
                        content_hash,
                        objective.name,
                        objective.description,
                        objective.primary_concept_id,
                        json.dumps(list(objective.supporting_concept_ids)),
                        objective.operation.value,
                        objective.evidence_type,
                        objective.prior_mastery,
                    ),
                )
            for source in sources:
                content_hash = source_hashes[source.id]
                existing = connection.execute(
                    "SELECT content_hash FROM sources WHERE id = ?", (source.id,)
                ).fetchone()
                if existing and existing["content_hash"] != content_hash:
                    raise ConflictError(f"Source {source.id} is immutable; publish a new source ID.")
                connection.execute(
                    """INSERT OR IGNORE INTO sources(
                           id, content_hash, title, uri, license, metadata_json
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        source.id,
                        content_hash,
                        source.title,
                        source.uri,
                        source.license,
                        json.dumps(source.metadata, sort_keys=True),
                    ),
                )
            connection.execute("DELETE FROM concept_edges")
            for edge in edges:
                connection.execute(
                    "INSERT INTO concept_edges(source_id, target_id, relation, weight) VALUES (?, ?, ?, ?)",
                    (edge.source_id, edge.target_id, edge.relation.value, edge.weight),
                )
            for misconception in misconceptions:
                content_hash = misconception_hashes[misconception.id]
                existing = connection.execute(
                    "SELECT content_hash FROM misconceptions WHERE id = ?", (misconception.id,)
                ).fetchone()
                if existing and existing["content_hash"] != content_hash:
                    raise ConflictError(
                        f"Misconception {misconception.id} is immutable; publish a new ID."
                    )
                connection.execute(
                    """INSERT OR IGNORE INTO misconceptions(
                           id, content_hash, concept_id, name, description
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        misconception.id,
                        content_hash,
                        misconception.concept_id,
                        misconception.name,
                        misconception.description,
                    ),
                )
            for question in questions:
                content_hash = question_hashes[question.id]
                existing = existing_questions.get(question.id)
                if existing and (
                    existing["version"] != question.version
                    or existing["content_hash"] != content_hash
                ):
                    raise ConflictError(
                        f"Question {question.id} is immutable; publish a new ID with revision_of instead."
                    )
                if existing:
                    if existing["status"] != question.status.value:
                        connection.execute(
                            "UPDATE questions SET status = ? WHERE id = ?",
                            (question.status.value, question.id),
                        )
                    continue
                connection.execute(
                    """INSERT INTO questions(
                           id, version, content_hash, family_id, status, stem, kind, difficulty,
                           discrimination, guess_rate, slip_rate, provenance_json,
                           tags_json, revision_of, imported_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        question.id,
                        question.version,
                        content_hash,
                        question.family_id,
                        question.status.value,
                        question.stem,
                        question.kind.value,
                        question.difficulty,
                        question.discrimination,
                        question.guess_rate,
                        question.slip_rate,
                        json.dumps(question.provenance, sort_keys=True),
                        json.dumps(question.tags),
                        question.revision_of,
                        imported_at,
                    ),
                )
                for mapping in question.concepts:
                    connection.execute(
                        "INSERT INTO question_concepts(question_id, concept_id, weight, role) VALUES (?, ?, ?, ?)",
                        (question.id, mapping.concept_id, mapping.weight, mapping.role),
                    )
                for option in question.options:
                    connection.execute(
                        """INSERT INTO options(question_id, option_id, text, is_correct, rationale, misconception_id)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            question.id,
                            option.id,
                            option.text,
                            int(option.correct),
                            option.rationale,
                            option.misconception_id,
                        ),
                    )
                for source_id in question.source_ids:
                    connection.execute(
                        "INSERT INTO question_sources(question_id, source_id) VALUES (?, ?)",
                        (question.id, source_id),
                    )
                connection.execute(
                    "INSERT OR IGNORE INTO item_stats(question_id) VALUES (?)", (question.id,)
                )
            existing_release = connection.execute(
                "SELECT bundle_hash, sealed_at FROM corpus_releases WHERE id = ?",
                (release_id,),
            ).fetchone()
            if existing_release and existing_release["bundle_hash"] != bundle_hash:
                raise ConflictError(
                    "Corpus release ID collision; refusing to mutate an existing snapshot."
                )
            is_new_release = existing_release is None
            if existing_release and existing_release["sealed_at"] is None:
                raise ConflictError(
                    f"Corpus release {release_id} exists but is not sealed; run integrity repair."
                )
            if is_new_release:
                connection.execute(
                    """INSERT INTO corpus_releases(
                           id, bundle_hash, created_at, sealed_at
                       ) VALUES (?, ?, ?, NULL)""",
                    (release_id, bundle_hash, imported_at),
                )
                connection.execute(
                    """INSERT INTO release_edges(
                           release_id, source_id, target_id, relation, weight
                       )
                       SELECT ?, source_id, target_id, relation, weight
                       FROM concept_edges""",
                    (release_id,),
                )
                connection.executemany(
                    """INSERT INTO release_concepts(release_id, concept_id)
                       VALUES (?, ?)""",
                    ((release_id, concept.id) for concept in concepts),
                )
                connection.executemany(
                    """INSERT INTO release_misconceptions(
                           release_id, misconception_id
                       ) VALUES (?, ?)""",
                    ((release_id, item.id) for item in misconceptions),
                )
                connection.executemany(
                    """INSERT INTO release_sources(release_id, source_id)
                       VALUES (?, ?)""",
                    ((release_id, source.id) for source in sources),
                )
                connection.executemany(
                    """INSERT INTO release_questions(
                           release_id, question_id, status, evidence_weight
                       ) VALUES (?, ?, ?, ?)""",
                    (
                        (
                            release_id,
                            question.id,
                            question.status.value,
                            question.status.evidence_weight,
                        )
                        for question in questions
                    ),
                )
                connection.executemany(
                    """INSERT INTO release_learning_objectives(
                           release_id, objective_id
                       ) VALUES (?, ?)""",
                    (
                        (release_id, objective_id)
                        for objective_id in sorted(objectives)
                    ),
                )
                if objective_graph_version is not None:
                    connection.execute(
                        """INSERT INTO release_objective_graphs(
                               release_id, graph_version
                           ) VALUES (?, ?)""",
                        (release_id, objective_graph_version),
                    )
                    connection.executemany(
                        """INSERT INTO release_objective_edges(
                               release_id, edge_id, source_objective_id,
                               target_objective_id, relation, weight, rationale
                           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            (
                                release_id,
                                edge.id,
                                edge.source_id,
                                edge.target_id,
                                edge.relation.value,
                                edge.weight,
                                edge.rationale,
                            )
                            for edge in objective_edges
                        ),
                    )
                connection.executemany(
                    """INSERT INTO release_question_objectives(
                           release_id, question_id, objective_id
                       ) VALUES (?, ?, ?)""",
                    (
                        (release_id, question.id, question.objective_id)
                        for question in questions
                        if question.objective_id is not None
                    ),
                )
                connection.executemany(
                    """INSERT INTO release_option_objectives(
                           release_id, question_id, option_id, objective_id
                       ) VALUES (?, ?, ?, ?)""",
                    (
                        (
                            release_id,
                            question.id,
                            option.id,
                            option.diagnostic_objective_id,
                        )
                        for question in questions
                        for option in question.options
                        if option.diagnostic_objective_id is not None
                    ),
                )
                connection.executemany(
                    """INSERT INTO release_domains(
                           release_id, domain_id, content_hash, name,
                           description, sort_order
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        (
                            release_id,
                            domain.id,
                            domain_hashes[domain.id],
                            domain.name,
                            domain.description,
                            domain.sort_order,
                        )
                        for domain in sorted(
                            domains, key=lambda item: (item.sort_order, item.id)
                        )
                    ),
                )
                for topic in ordered_topics:
                    connection.execute(
                        """INSERT INTO release_topics(
                               release_id, topic_id, content_hash, domain_id,
                               parent_topic_id, name, description,
                               related_topic_ids_json, sort_order
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            release_id,
                            topic.id,
                            topic_hashes[topic.id],
                            topic.domain_id,
                            topic.parent_id,
                            topic.name,
                            topic.description,
                            json.dumps(list(topic.related_topic_ids)),
                            topic.sort_order,
                        ),
                    )
                    connection.executemany(
                        """INSERT INTO release_topic_concepts(
                               release_id, topic_id, concept_id, position
                           ) VALUES (?, ?, ?, ?)""",
                        (
                            (release_id, topic.id, concept_id, position)
                            for position, concept_id in enumerate(topic.concept_ids)
                        ),
                    )
                connection.executemany(
                    """INSERT INTO release_question_topics(
                           release_id, question_id, topic_id, relation
                       ) VALUES (?, ?, ?, ?)""",
                    (
                        (release_id, question_id, topic_id, relation)
                        for question_id, topic_id, relation in question_topic_rows
                    ),
                )
            stored_edges = [
                tuple(row)
                for row in connection.execute(
                    """SELECT source_id, target_id, relation, weight
                       FROM release_edges WHERE release_id = ? ORDER BY 1, 2, 3""",
                    (release_id,),
                ).fetchall()
            ]
            stored_concepts = [
                row["concept_id"]
                for row in connection.execute(
                    """SELECT concept_id FROM release_concepts
                       WHERE release_id = ? ORDER BY concept_id""",
                    (release_id,),
                ).fetchall()
            ]
            stored_questions = [
                tuple(row)
                for row in connection.execute(
                    """SELECT question_id, status, evidence_weight
                       FROM release_questions WHERE release_id = ? ORDER BY question_id""",
                    (release_id,),
                ).fetchall()
            ]
            stored_objectives = [
                row["objective_id"]
                for row in connection.execute(
                    """SELECT objective_id FROM release_learning_objectives
                       WHERE release_id = ? ORDER BY objective_id""",
                    (release_id,),
                ).fetchall()
            ]
            stored_objective_graph = connection.execute(
                """SELECT graph_version FROM release_objective_graphs
                   WHERE release_id = ?""",
                (release_id,),
            ).fetchone()
            stored_objective_edges = [
                tuple(row)
                for row in connection.execute(
                    """SELECT edge_id, source_objective_id,
                              target_objective_id, relation, weight, rationale
                       FROM release_objective_edges WHERE release_id = ?
                       ORDER BY edge_id""",
                    (release_id,),
                ).fetchall()
            ]
            stored_question_objectives = [
                tuple(row)
                for row in connection.execute(
                    """SELECT question_id, objective_id
                       FROM release_question_objectives WHERE release_id = ?
                       ORDER BY question_id""",
                    (release_id,),
                ).fetchall()
            ]
            stored_option_objectives = [
                tuple(row)
                for row in connection.execute(
                    """SELECT question_id, option_id, objective_id
                       FROM release_option_objectives WHERE release_id = ?
                       ORDER BY question_id, option_id""",
                    (release_id,),
                ).fetchall()
            ]
            stored_misconceptions = [
                row["misconception_id"]
                for row in connection.execute(
                    """SELECT misconception_id FROM release_misconceptions
                       WHERE release_id = ? ORDER BY misconception_id""",
                    (release_id,),
                ).fetchall()
            ]
            stored_sources = [
                row["source_id"]
                for row in connection.execute(
                    """SELECT source_id FROM release_sources
                       WHERE release_id = ? ORDER BY source_id""",
                    (release_id,),
                ).fetchall()
            ]
            stored_domains = [
                (row["domain_id"], row["content_hash"])
                for row in connection.execute(
                    """SELECT domain_id, content_hash FROM release_domains
                       WHERE release_id = ? ORDER BY domain_id""",
                    (release_id,),
                ).fetchall()
            ]
            stored_topics = [
                (row["topic_id"], row["content_hash"])
                for row in connection.execute(
                    """SELECT topic_id, content_hash FROM release_topics
                       WHERE release_id = ? ORDER BY topic_id""",
                    (release_id,),
                ).fetchall()
            ]
            stored_question_topics = [
                tuple(row)
                for row in connection.execute(
                    """SELECT question_id, topic_id, relation
                       FROM release_question_topics WHERE release_id = ?
                       ORDER BY question_id, topic_id, relation""",
                    (release_id,),
                ).fetchall()
            ]
            expected_questions = sorted(
                (
                    question.id,
                    question.status.value,
                    question.status.evidence_weight,
                )
                for question in questions
            )
            if (
                stored_edges != release_payload["edges"]
                or stored_concepts != sorted(concept.id for concept in concepts)
                or stored_misconceptions
                != sorted(item.id for item in misconceptions)
                or stored_sources != sorted(source.id for source in sources)
                or stored_questions != expected_questions
                or stored_objectives != sorted(objectives)
                or (
                    stored_objective_graph["graph_version"]
                    if stored_objective_graph is not None
                    else None
                )
                != objective_graph_version
                or stored_objective_edges
                != [
                    (
                        edge.id,
                        edge.source_id,
                        edge.target_id,
                        edge.relation.value,
                        edge.weight,
                        edge.rationale,
                    )
                    for edge in objective_edges
                ]
                or stored_question_objectives
                != sorted(
                    (question.id, question.objective_id)
                    for question in questions
                    if question.objective_id is not None
                )
                or stored_option_objectives
                != sorted(
                    (
                        question.id,
                        option.id,
                        option.diagnostic_objective_id,
                    )
                    for question in questions
                    for option in question.options
                    if option.diagnostic_objective_id is not None
                )
                or stored_domains != sorted(domain_hashes.items())
                or stored_topics != sorted(topic_hashes.items())
                or stored_question_topics != sorted(question_topic_rows)
            ):
                raise ConflictError(
                    "Stored corpus release snapshot differs from its immutable manifest."
                )
            if is_new_release:
                sealed = connection.execute(
                    """UPDATE corpus_releases SET sealed_at = ?
                       WHERE id = ? AND sealed_at IS NULL""",
                    (imported_at, release_id),
                )
                if sealed.rowcount != 1:
                    raise ConflictError("Corpus release could not be sealed atomically.")
            legacy_generated_revocations = (
                self._revoke_historically_active_unreviewed_generated_questions(
                    connection,
                    legacy_unreviewed_generated_ids,
                )
            )
            legacy_generated_evidence_quarantines = (
                self._quarantine_historical_generated_evidence(connection)
            )
            active_release = connection.execute(
                "SELECT value FROM meta WHERE key='active_corpus_release'"
            ).fetchone()
            activation_changed = (
                active_release is None or active_release["value"] != release_id
            )
            if activation_changed:
                connection.execute(
                    """INSERT INTO meta(key, value)
                       VALUES('active_corpus_release', ?)
                       ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                    (release_id,),
                )
                self.append_event(
                    connection,
                    stream_id="corpus:main",
                    event_type="CorpusImported",
                    payload={
                        "concept_count": len(concepts),
                        "misconception_count": len(misconceptions),
                        "question_count": len(questions),
                        "domain_count": len(domains),
                        "topic_count": len(topics),
                        "learning_objective_count": len(objectives),
                        "objective_edge_count": len(objective_edges),
                        "objective_graph_version": objective_graph_version,
                        "corpus_release_id": release_id,
                        "bundle_hash": bundle_hash,
                    },
                    metadata={
                        "schema_version": SCHEMA_VERSION,
                        "corpus_release_id": release_id,
                    },
                )
        return {
            "concepts": len(concepts),
            "edges": len(edges),
            "misconceptions": len(misconceptions),
            "sources": len(sources),
            "questions": len(questions),
            "domains": len(domains),
            "topics": len(topics),
            "learning_objectives": len(objectives),
            "objective_edges": len(objective_edges),
            "release_id": release_id,
            "legacy_generated_revocations": legacy_generated_revocations,
            "legacy_generated_evidence_quarantines": (
                legacy_generated_evidence_quarantines
            ),
        }

    def _revoke_historically_active_unreviewed_generated_questions(
        self,
        connection: sqlite3.Connection,
        question_ids: Sequence[str],
    ) -> int:
        """Globally revoke quarantined generated IDs that an old release served."""

        created = 0
        active = connection.execute(
            "SELECT value FROM meta WHERE key = 'active_corpus_release'"
        ).fetchone()
        for question_id in question_ids:
            historical_active = connection.execute(
                """SELECT 1
                   FROM release_questions
                   WHERE question_id = ?
                     AND status IN ('approved', 'calibrated')
                   LIMIT 1""",
                (question_id,),
            ).fetchone()
            if historical_active is None:
                continue
            existing = connection.execute(
                "SELECT event_id FROM question_revocations WHERE question_id = ?",
                (question_id,),
            ).fetchone()
            if existing is not None:
                continue

            reason = LEGACY_UNREVIEWED_GENERATED_REVOCATION_REASON
            payload = {"question_id": question_id, "reason": reason}
            idempotency_key = _legacy_unreviewed_generated_revocation_key(
                question_id
            )
            prior_event = connection.execute(
                "SELECT * FROM events WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if prior_event is not None:
                if (
                    prior_event["event_type"] != "QuestionEmergencyRevoked"
                    or json.loads(prior_event["payload_json"]) != payload
                ):
                    raise ConflictError(
                        "Legacy generated-question revocation key was reused "
                        "for a different safety event."
                    )
                raise ConflictError(
                    "Legacy generated-question revocation event has no matching "
                    "projection; database needs repair."
                )

            revoked_at = datetime.now(timezone.utc)
            event = self.append_event(
                connection,
                stream_id="corpus:safety",
                event_type="QuestionEmergencyRevoked",
                payload=payload,
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "active_corpus_release_id": (
                        active["value"] if active is not None else None
                    ),
                    "migration": "legacy-unreviewed-generated-v1",
                },
                idempotency_key=idempotency_key,
                occurred_at=revoked_at,
            )
            connection.execute(
                """INSERT INTO question_revocations(
                       question_id, reason, revoked_at, event_id
                   ) VALUES (?, ?, ?, ?)""",
                (
                    question_id,
                    reason,
                    to_timestamp(revoked_at),
                    event["event_id"],
                ),
            )
            created += 1
        return created

    def _historically_contaminated_generated_attempts(
        self,
        connection: sqlite3.Connection,
        learner_id: str | None = None,
    ) -> tuple[sqlite3.Row, ...]:
        """Return responses to independently classified unsafe generation."""

        unsafe_question_ids = (
            self._historically_active_unsafe_generated_ids(connection)
        )
        if not unsafe_question_ids:
            return ()
        placeholders = ", ".join("?" for _item in unsafe_question_ids)
        parameters: list[object] = list(unsafe_question_ids)
        learner_clause = ""
        if learner_id is not None:
            learner_clause = " AND attempt.learner_id = ?"
            parameters.append(learner_id)
        return tuple(
            connection.execute(
                f"""SELECT attempt.id AS attempt_id,
                           attempt.event_id AS response_event_id,
                           attempt.learner_id,
                           attempt.question_id
                    FROM attempts attempt
                    WHERE attempt.question_id IN ({placeholders})
                    {learner_clause}
                    ORDER BY attempt.learner_id, attempt.id""",
                tuple(parameters),
            ).fetchall()
        )

    def _quarantine_historical_generated_evidence(
        self,
        connection: sqlite3.Connection,
    ) -> int:
        """Mark contaminated responses without rewriting immutable history."""

        created = 0
        reason = LEGACY_UNREVIEWED_GENERATED_REVOCATION_REASON
        metadata = {
            "safety_policy": HISTORICAL_GENERATED_EVIDENCE_POLICY,
            "requires_explicit_rebuild": True,
        }
        for row in self._historically_contaminated_generated_attempts(
            connection
        ):
            payload = {
                "attempt_id": row["attempt_id"],
                "response_event_id": row["response_event_id"],
                "learner_id": row["learner_id"],
                "question_id": row["question_id"],
                "reason": reason,
                "projection_applied": False,
            }
            idempotency_key = _historical_generated_evidence_key(
                row["attempt_id"]
            )
            prior = connection.execute(
                "SELECT * FROM events WHERE idempotency_key=?",
                (idempotency_key,),
            ).fetchone()
            if prior is not None:
                if (
                    prior["event_type"] != "ResponseEvidenceQuarantined"
                    or prior["schema_version"] != 1
                    or prior["stream_id"]
                    != f"learner:{row['learner_id']}"
                    or prior["learner_id"] != row["learner_id"]
                    or prior["session_id"] is not None
                    or prior["correlation_id"] != row["attempt_id"]
                    or prior["causation_id"] != row["response_event_id"]
                    or json.loads(prior["payload_json"]) != payload
                    or json.loads(prior["metadata_json"]) != metadata
                ):
                    raise ConflictError(
                        "Historical generated-evidence quarantine key was "
                        "reused for a different safety event."
                    )
                continue
            self.append_event(
                connection,
                stream_id=f"learner:{row['learner_id']}",
                event_type="ResponseEvidenceQuarantined",
                schema_version=1,
                payload=payload,
                metadata=metadata,
                learner_id=row["learner_id"],
                session_id=None,
                idempotency_key=idempotency_key,
                correlation_id=row["attempt_id"],
                causation_id=row["response_event_id"],
                occurred_at=datetime.now(timezone.utc),
            )
            created += 1
        return created

    def require_learner_evidence_safe(
        self,
        learner_id: str,
        connection: sqlite3.Connection,
    ) -> None:
        """Fail closed while revoked generated evidence remains projected."""

        contaminated = self._historically_contaminated_generated_attempts(
            connection,
            learner_id,
        )
        if contaminated:
            raise ConflictError(
                f"Learner {learner_id} has {len(contaminated)} response(s) "
                "with quarantined generated-question evidence; adaptive use "
                "and reporting require an explicit audited projection rebuild."
            )

    @staticmethod
    def _validate_corpus_activation(
        concepts: Sequence[Concept],
        edges: Sequence[ConceptEdge],
        misconceptions: Sequence[Misconception],
        sources: Sequence[Source],
        questions: Sequence[Question],
        domains: Sequence[Domain] = (),
        topics: Sequence[Topic] = (),
    ) -> None:
        """Revalidate typed corpus objects at the storage trust boundary."""

        def duplicates(values: Sequence[str]) -> list[str]:
            seen: set[str] = set()
            repeated: set[str] = set()
            for value in values:
                if value in seen:
                    repeated.add(value)
                seen.add(value)
            return sorted(repeated)

        objective_definitions: dict[str, LearningObjective] = {}
        objective_conflicts: set[str] = set()
        for question in questions:
            if question.objective is None:
                continue
            existing = objective_definitions.get(question.objective.id)
            if existing is not None and existing != question.objective:
                objective_conflicts.add(question.objective.id)
            objective_definitions[question.objective.id] = question.objective

        identity_groups = {
            "concept": [item.id for item in concepts],
            "misconception": [item.id for item in misconceptions],
            "source": [item.id for item in sources],
            "question": [item.id for item in questions],
            "domain": [item.id for item in domains],
            "topic": [item.id for item in topics],
        }
        violations: list[str] = []
        if objective_conflicts:
            violations.append(
                "conflicting learning-objective definitions: "
                + ", ".join(sorted(objective_conflicts))
            )
        for label, identifiers in identity_groups.items():
            repeated = duplicates(identifiers)
            if repeated:
                violations.append(
                    f"duplicate {label} IDs: {', '.join(repeated)}"
                )

        edge_keys = [
            f"{edge.source_id}|{edge.target_id}|{edge.relation.value}"
            for edge in edges
        ]
        repeated_edges = duplicates(edge_keys)
        if repeated_edges:
            violations.append("duplicate typed concept edges")

        concept_ids = set(identity_groups["concept"])
        misconception_by_id = {item.id: item for item in misconceptions}
        source_ids = set(identity_groups["source"])
        question_by_id = {item.id: item for item in questions}
        domain_ids = set(identity_groups["domain"])
        topic_by_id = {item.id: item for item in topics}

        objective_enabled_concepts = {
            concept_id
            for objective in objective_definitions.values()
            for concept_id in objective.concept_ids
        }
        for objective in objective_definitions.values():
            unknown = set(objective.concept_ids) - concept_ids
            if unknown:
                violations.append(
                    f"learning objective {objective.id} references unknown concepts: "
                    + ", ".join(sorted(unknown))
                )
        objective_graph_versions = {
            objective.objective_graph_version
            for objective in objective_definitions.values()
        }
        if len(objective_graph_versions) > 1:
            violations.append(
                "learning objectives mix declared and legacy objective graphs"
            )
        objective_edges = [
            edge
            for objective in objective_definitions.values()
            for edge in objective.prerequisites
        ]
        repeated_objective_edge_ids = duplicates(
            [edge.id for edge in objective_edges]
        )
        if repeated_objective_edge_ids:
            violations.append(
                "duplicate objective-edge IDs: "
                + ", ".join(repeated_objective_edge_ids)
            )
        repeated_objective_edges = duplicates(
            [
                f"{edge.source_id}|{edge.target_id}"
                for edge in objective_edges
            ]
        )
        if repeated_objective_edges:
            violations.append("duplicate typed objective edges")
        objective_ids = set(objective_definitions)
        objective_adjacency = {
            objective_id: [] for objective_id in objective_ids
        }
        objective_indegree = {objective_id: 0 for objective_id in objective_ids}
        for edge in objective_edges:
            unknown = {edge.source_id, edge.target_id} - objective_ids
            if unknown:
                violations.append(
                    f"objective edge {edge.id} references unknown objectives: "
                    + ", ".join(sorted(unknown))
                )
                continue
            objective_adjacency[edge.source_id].append(edge.target_id)
            objective_indegree[edge.target_id] += 1
        queue = deque(
            objective_id
            for objective_id, degree in objective_indegree.items()
            if degree == 0
        )
        visited = 0
        while queue:
            objective_id = queue.popleft()
            visited += 1
            for dependent_id in objective_adjacency[objective_id]:
                objective_indegree[dependent_id] -= 1
                if objective_indegree[dependent_id] == 0:
                    queue.append(dependent_id)
        if visited != len(objective_ids):
            violations.append("objective prerequisite graph contains a cycle")

        if bool(domains) != bool(topics):
            violations.append(
                "curriculum activation requires both domains and topics"
            )
        for domain in domains:
            if (
                not domain.id.strip()
                or not domain.name.strip()
                or not domain.description.strip()
                or isinstance(domain.sort_order, bool)
                or not isinstance(domain.sort_order, int)
                or domain.sort_order < 0
            ):
                violations.append(f"domain {domain.id!r} has invalid fields")

        concept_owners: dict[str, list[str]] = {}
        for topic in topics:
            if (
                not topic.id.strip()
                or not topic.name.strip()
                or not topic.description.strip()
                or isinstance(topic.sort_order, bool)
                or not isinstance(topic.sort_order, int)
                or topic.sort_order < 0
            ):
                violations.append(f"topic {topic.id!r} has invalid fields")
            if topic.domain_id not in domain_ids:
                violations.append(
                    f"topic {topic.id} references unknown domain {topic.domain_id}"
                )
            if topic.parent_id:
                parent = topic_by_id.get(topic.parent_id)
                if parent is None:
                    violations.append(
                        f"topic {topic.id} references unknown parent {topic.parent_id}"
                    )
                elif parent.domain_id != topic.domain_id:
                    violations.append(
                        f"topic {topic.id} and its parent must share a domain"
                    )
            if len(set(topic.concept_ids)) != len(topic.concept_ids):
                violations.append(f"topic {topic.id} repeats a concept")
            for concept_id in topic.concept_ids:
                if concept_id not in concept_ids:
                    violations.append(
                        f"topic {topic.id} references unknown concept {concept_id}"
                    )
                concept_owners.setdefault(concept_id, []).append(topic.id)
            if len(set(topic.related_topic_ids)) != len(topic.related_topic_ids):
                violations.append(f"topic {topic.id} repeats a related topic")
            for related_id in topic.related_topic_ids:
                related = topic_by_id.get(related_id)
                if related_id == topic.id:
                    violations.append(f"topic {topic.id} relates to itself")
                elif related is None:
                    violations.append(
                        f"topic {topic.id} references unknown related topic {related_id}"
                    )
                elif topic.id not in related.related_topic_ids:
                    violations.append(
                        f"topic relation {topic.id}<->{related_id} is asymmetric"
                    )

        for start in topic_by_id:
            trail: set[str] = set()
            current: str | None = start
            while current is not None and current in topic_by_id:
                if current in trail:
                    violations.append(
                        f"topic parent hierarchy contains a cycle at {current}"
                    )
                    break
                trail.add(current)
                current = topic_by_id[current].parent_id
        if topics:
            for concept_id in concept_ids:
                owners = concept_owners.get(concept_id, [])
                if len(owners) != 1:
                    violations.append(
                        f"concept {concept_id} must have exactly one topic owner"
                    )
            child_parents = {topic.parent_id for topic in topics if topic.parent_id}
            for topic in topics:
                if not topic.concept_ids and topic.id not in child_parents:
                    violations.append(
                        f"topic {topic.id} owns no concepts and has no children"
                    )
            for domain_id in domain_ids:
                if not any(
                    topic.domain_id == domain_id and topic.parent_id is None
                    for topic in topics
                ):
                    violations.append(
                        f"domain {domain_id} has no top-level topic"
                    )
        for edge in edges:
            unknown = {edge.source_id, edge.target_id} - concept_ids
            if unknown:
                violations.append(
                    f"edge {edge.source_id}->{edge.target_id} references unknown "
                    f"concepts: {', '.join(sorted(unknown))}"
                )
        for misconception in misconceptions:
            if misconception.concept_id not in concept_ids:
                violations.append(
                    f"misconception {misconception.id} has unknown owner "
                    f"{misconception.concept_id}"
                )
        for question in questions:
            legacy_unattested_compatible = bool(
                type(question.provenance) is dict
                and "generated" not in question.provenance
                and legacy_unattested_member_compatible(
                    _legacy_question_identity(question)
                )
            )
            for issue in question_provenance_issues(
                question.provenance,
                status=question.status.value,
                legacy_unattested_compatible=(
                    legacy_unattested_compatible
                ),
            ):
                violations.append(
                    f"question {question.id} provenance [{issue.code}]: "
                    f"{issue.message}"
                )
            mapped = {mapping.concept_id for mapping in question.concepts}
            unknown_concepts = mapped - concept_ids
            if unknown_concepts:
                violations.append(
                    f"question {question.id} maps unknown concepts: "
                    f"{', '.join(sorted(unknown_concepts))}"
                )
            unknown_sources = set(question.source_ids) - source_ids
            if unknown_sources:
                violations.append(
                    f"question {question.id} cites unknown sources: "
                    f"{', '.join(sorted(unknown_sources))}"
                )
            if question.objective is not None:
                if question.primary_concept_id not in question.objective.concept_ids:
                    violations.append(
                        f"question {question.id} objective {question.objective.id} "
                        "does not include its primary concept"
                    )
            elif (
                question.status.eligible_for_adaptation
                and question.primary_concept_id in objective_enabled_concepts
            ):
                violations.append(
                    f"question {question.id} omits an objective for objective-enabled "
                    f"concept {question.primary_concept_id}"
                )
            for misconception_id in question.misconception_ids:
                definition = misconception_by_id.get(misconception_id)
                if definition is None:
                    violations.append(
                        f"question {question.id} uses unknown misconception "
                        f"{misconception_id}"
                    )
                elif definition.concept_id not in mapped:
                    violations.append(
                        f"question {question.id} does not map misconception owner "
                        f"{definition.concept_id}"
                    )
            for option in question.options:
                diagnostic_id = option.diagnostic_objective_id
                if option.correct and diagnostic_id is not None:
                    violations.append(
                        f"question {question.id} correct option {option.id} "
                        "declares a diagnostic objective"
                    )
                    continue
                if diagnostic_id is None:
                    continue
                diagnostic = objective_definitions.get(diagnostic_id)
                if diagnostic is None:
                    violations.append(
                        f"question {question.id} option {option.id} references "
                        f"unknown diagnostic objective {diagnostic_id}"
                    )
                    continue
                misconception = misconception_by_id.get(
                    option.misconception_id
                )
                if (
                    misconception is not None
                    and misconception.concept_id not in diagnostic.concept_ids
                ):
                    violations.append(
                        f"question {question.id} option {option.id} diagnostic "
                        f"objective {diagnostic_id} excludes misconception owner "
                        f"{misconception.concept_id}"
                    )
            if question.revision_of:
                parent = question_by_id.get(question.revision_of)
                if parent is None:
                    violations.append(
                        f"question {question.id} has unknown revision parent "
                        f"{question.revision_of}"
                    )
                elif (
                    question.version <= parent.version
                    or question.family_id != parent.family_id
                ):
                    violations.append(
                        f"question {question.id} violates revision version/family invariants"
                    )

        if violations:
            raise ValidationError(
                "Corpus activation rejected: " + "; ".join(violations[:20])
            )
        # Construction validates both strict-readiness and part-of DAGs.
        knowledge_graph = KnowledgeGraph(concepts, edges)
        quality_errors = [
            issue
            for issue in audit_corpus(
                questions,
                expected_primary_concept_ids={
                    mapping.concept_id
                    for question in questions
                    for mapping in question.concepts
                },
                knowledge_graph=knowledge_graph,
                misconceptions=misconceptions,
            )
            if issue.severity == "error"
        ]
        if quality_errors:
            rendered = "; ".join(
                f"{issue.question_id or issue.path or 'corpus'}: {issue.message}"
                for issue in quality_errors[:20]
            )
            raise ValidationError(
                f"Corpus activation failed {len(quality_errors)} deterministic "
                f"quality checks: {rendered}",
                issues=quality_errors,
            )

    def append_event(
        self,
        connection: sqlite3.Connection,
        *,
        stream_id: str,
        event_type: str,
        payload: dict[str, Any],
        schema_version: int = 1,
        metadata: dict[str, Any] | None = None,
        learner_id: str | None = None,
        session_id: str | None = None,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> sqlite3.Row:
        if type(schema_version) is not int or schema_version < 1:
            raise ValidationError("Event schema_version must be a positive integer.")
        if idempotency_key:
            prior = connection.execute(
                "SELECT * FROM events WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if prior:
                if prior["event_type"] != event_type or json.loads(prior["payload_json"]) != payload:
                    raise ConflictError("Idempotency key was already used for a different command.")
                return prior
            scoring_claim = connection.execute(
                """SELECT * FROM performance_scoring_claims
                   WHERE idempotency_key=?""",
                (idempotency_key,),
            ).fetchone()
            if scoring_claim is not None:
                evaluation = payload.get("evaluation")
                if not (
                    event_type == "TaskEvaluationRecorded"
                    and (metadata or {}).get("command_hash")
                    == scoring_claim["command_hash"]
                    and payload.get("attempt_id")
                    == scoring_claim["attempt_id"]
                    and payload.get("through_sequence")
                    == scoring_claim["through_sequence"]
                    and type(evaluation) is dict
                    and evaluation.get("id")
                    == scoring_claim["evaluation_id"]
                ):
                    raise ConflictError(
                        "Idempotency key is reserved by an unfinished "
                        "performance scoring claim."
                    )
        head = connection.execute(
            "SELECT stream_version, payload_hash FROM stream_heads WHERE stream_id = ?",
            (stream_id,),
        ).fetchone()
        tail = connection.execute(
            "SELECT stream_version, payload_hash FROM events WHERE stream_id = ? ORDER BY stream_version DESC LIMIT 1",
            (stream_id,),
        ).fetchone()
        if (head is None) != (tail is None) or (
            head is not None
            and tail is not None
            and (
                head["stream_version"] != tail["stream_version"]
                or head["payload_hash"] != tail["payload_hash"]
            )
        ):
            raise ConflictError(
                f"Stream {stream_id} head does not match its event tail; run integrity repair."
            )
        stream_version = (head["stream_version"] + 1) if head else 1
        previous_hash = head["payload_hash"] if head else None
        event_id = new_id("evt")
        now = datetime.now(timezone.utc)
        occurred_timestamp = to_timestamp(occurred_at or now)
        recorded_timestamp = to_timestamp(now)
        resolved_correlation_id = correlation_id or event_id
        try:
            payload_json = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            metadata_json = json.dumps(
                metadata or {}, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Event payload and metadata must be finite JSON: {exc}") from exc
        envelope = {
            "event_id": event_id,
            "stream_id": stream_id,
            "stream_version": stream_version,
            "event_type": event_type,
            "schema_version": schema_version,
            "occurred_at": occurred_timestamp,
            "recorded_at": recorded_timestamp,
            "learner_id": learner_id,
            "session_id": session_id,
            "correlation_id": resolved_correlation_id,
            "causation_id": causation_id,
            "idempotency_key": idempotency_key,
            "payload": payload,
            "metadata": metadata or {},
            "previous_hash": previous_hash,
        }
        payload_hash = _content_hash(envelope)
        connection.execute(
            """INSERT INTO events(
                   event_id, stream_id, stream_version, event_type, schema_version,
                   occurred_at, recorded_at, learner_id, session_id, correlation_id,
                   causation_id, idempotency_key, payload_json, metadata_json,
                   previous_hash, payload_hash
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                stream_id,
                stream_version,
                event_type,
                schema_version,
                occurred_timestamp,
                recorded_timestamp,
                learner_id,
                session_id,
                resolved_correlation_id,
                causation_id,
                idempotency_key,
                payload_json,
                metadata_json,
                previous_hash,
                payload_hash,
            ),
        )
        connection.execute(
            """INSERT INTO stream_heads(stream_id, stream_version, payload_hash, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(stream_id) DO UPDATE SET
                   stream_version=excluded.stream_version,
                   payload_hash=excluded.payload_hash,
                   updated_at=excluded.updated_at""",
            (stream_id, stream_version, payload_hash, recorded_timestamp),
        )
        return connection.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()

    def revoke_question(
        self,
        question_id: str,
        reason: str,
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Globally quarantine a question across every pinned corpus release."""
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValidationError("question_id must be a non-blank string.")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason.strip()) > 500
        ):
            raise ValidationError(
                "reason must be a non-blank string of at most 500 characters."
            )
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key) > 200
        ):
            raise ValidationError(
                "idempotency_key must be a non-blank string of at most 200 characters."
            )
        normalized_reason = reason.strip()
        with self.transaction() as connection:
            if not connection.execute(
                "SELECT 1 FROM questions WHERE id = ?", (question_id,)
            ).fetchone():
                raise NotFoundError(f"Unknown question: {question_id}")
            payload = {"question_id": question_id, "reason": normalized_reason}
            if idempotency_key:
                prior_event = connection.execute(
                    "SELECT * FROM events WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if prior_event:
                    if (
                        prior_event["event_type"] != "QuestionEmergencyRevoked"
                        or json.loads(prior_event["payload_json"]) != payload
                    ):
                        raise ConflictError(
                            "Idempotency key was reused for a different revocation."
                        )
                    prior = connection.execute(
                        "SELECT * FROM question_revocations WHERE question_id = ?",
                        (question_id,),
                    ).fetchone()
                    if not prior or prior["event_id"] != prior_event["event_id"]:
                        raise ConflictError(
                            "Revocation event has no matching safety projection."
                        )
                    return {**dict(prior), "idempotent": True}
            existing = connection.execute(
                "SELECT * FROM question_revocations WHERE question_id = ?",
                (question_id,),
            ).fetchone()
            if existing:
                if existing["reason"] != normalized_reason:
                    raise ConflictError(
                        f"Question {question_id} is already revoked for a different reason."
                    )
                return {**dict(existing), "idempotent": True}
            revoked_at = datetime.now(timezone.utc)
            active = connection.execute(
                "SELECT value FROM meta WHERE key = 'active_corpus_release'"
            ).fetchone()
            event = self.append_event(
                connection,
                stream_id="corpus:safety",
                event_type="QuestionEmergencyRevoked",
                payload=payload,
                metadata={
                    "schema_version": SCHEMA_VERSION,
                    "active_corpus_release_id": active["value"] if active else None,
                },
                idempotency_key=idempotency_key,
                occurred_at=revoked_at,
            )
            connection.execute(
                """INSERT INTO question_revocations(
                       question_id, reason, revoked_at, event_id
                   ) VALUES (?, ?, ?, ?)""",
                (
                    question_id,
                    normalized_reason,
                    revoked_at.isoformat(),
                    event["event_id"],
                ),
            )
            row = connection.execute(
                "SELECT * FROM question_revocations WHERE question_id = ?",
                (question_id,),
            ).fetchone()
            return {**dict(row), "idempotent": False}

    def ensure_learner(self, learner_id: str, display_name: str | None = None) -> dict[str, Any]:
        if not isinstance(learner_id, str) or not learner_id.strip() or len(learner_id) > 128:
            raise ValidationError("learner_id must be a non-blank string of at most 128 characters.")
        if display_name is not None and (
            not isinstance(display_name, str)
            or not display_name.strip()
            or len(display_name) > 200
        ):
            raise ValidationError(
                "display_name must be a non-blank string of at most 200 characters."
            )
        now = datetime.now(timezone.utc).isoformat()
        with self.transaction() as connection:
            row = connection.execute("SELECT * FROM learners WHERE id = ?", (learner_id,)).fetchone()
            if not row:
                connection.execute(
                    "INSERT INTO learners(id, display_name, created_at) VALUES (?, ?, ?)",
                    (learner_id, display_name or learner_id, now),
                )
                self.append_event(
                    connection,
                    stream_id=f"learner:{learner_id}",
                    event_type="LearnerCreated",
                    payload={"learner_id": learner_id, "display_name": display_name or learner_id},
                    learner_id=learner_id,
                )
                row = connection.execute("SELECT * FROM learners WHERE id = ?", (learner_id,)).fetchone()
            elif display_name is not None and row["display_name"] != display_name:
                raise ConflictError(
                    f"Learner {learner_id} already exists with a different display name."
                )
        return dict(row)

    def get_concepts(self) -> list[Concept]:
        with self.read() as connection:
            rows = connection.execute("SELECT * FROM concepts ORDER BY domain, name").fetchall()
        return [Concept(row["id"], row["name"], row["description"], row["domain"], row["prior_mastery"]) for row in rows]

    def get_graph(self, release_id: str | None = None) -> KnowledgeGraph:
        """Load the current graph or the immutable edge snapshot for a release."""
        with self.read() as connection:
            resolved_release = release_id
            if resolved_release is None:
                active = connection.execute(
                    "SELECT value FROM meta WHERE key = 'active_corpus_release'"
                ).fetchone()
                resolved_release = active["value"] if active else None
            if resolved_release is None:
                concept_rows = connection.execute(
                    "SELECT * FROM concepts ORDER BY domain, name"
                ).fetchall()
                edge_rows = connection.execute("SELECT * FROM concept_edges").fetchall()
            else:
                release = connection.execute(
                    "SELECT 1 FROM corpus_releases WHERE id = ?", (resolved_release,)
                ).fetchone()
                if not release:
                    raise NotFoundError(f"Unknown corpus release: {resolved_release}")
                concept_rows = connection.execute(
                    """SELECT concept.* FROM concepts concept
                       JOIN release_concepts membership
                         ON membership.concept_id = concept.id
                       WHERE membership.release_id = ?
                       ORDER BY concept.domain, concept.name""",
                    (resolved_release,),
                ).fetchall()
                edge_rows = connection.execute(
                    "SELECT * FROM release_edges WHERE release_id = ?",
                    (resolved_release,),
                ).fetchall()
        concepts = [
            Concept(
                row["id"],
                row["name"],
                row["description"],
                row["domain"],
                row["prior_mastery"],
            )
            for row in concept_rows
        ]
        edges = [
            ConceptEdge(row["source_id"], row["target_id"], RelationType(row["relation"]), row["weight"])
            for row in edge_rows
        ]
        return KnowledgeGraph(concepts, edges)

    def get_catalog(self, release_id: str | None = None) -> dict[str, Any]:
        """Return the immutable learner-facing curriculum catalog for a release."""
        with self.read() as connection:
            resolved_release = release_id or self.get_active_release_id(connection)
            if not connection.execute(
                "SELECT 1 FROM corpus_releases WHERE id = ?", (resolved_release,)
            ).fetchone():
                raise NotFoundError(f"Unknown corpus release: {resolved_release}")
            domain_rows = connection.execute(
                """SELECT domain_id, name, description, sort_order
                   FROM release_domains WHERE release_id = ?
                   ORDER BY sort_order, name, domain_id""",
                (resolved_release,),
            ).fetchall()
            topic_rows = connection.execute(
                """SELECT topic_id, domain_id, parent_topic_id, name,
                          description, related_topic_ids_json, sort_order
                   FROM release_topics WHERE release_id = ?
                   ORDER BY sort_order, name, topic_id""",
                (resolved_release,),
            ).fetchall()
            concept_rows = connection.execute(
                """SELECT membership.topic_id, membership.concept_id,
                          membership.position, concept.name
                   FROM release_topic_concepts membership
                   JOIN concepts concept ON concept.id = membership.concept_id
                   WHERE membership.release_id = ?
                   ORDER BY membership.topic_id, membership.position""",
                (resolved_release,),
            ).fetchall()
            coverage_rows = connection.execute(
                """SELECT mapping.topic_id,
                          COUNT(DISTINCT CASE WHEN mapping.relation = 'primary'
                                             THEN mapping.question_id END) AS primary_items,
                          COUNT(DISTINCT CASE WHEN mapping.relation = 'cross'
                                             THEN mapping.question_id END) AS cross_items
                   FROM release_question_topics mapping
                   JOIN release_questions question
                     ON question.release_id = mapping.release_id
                    AND question.question_id = mapping.question_id
                   WHERE mapping.release_id = ?
                     AND question.status IN ('approved', 'calibrated')
                     AND NOT EXISTS (
                         SELECT 1 FROM question_revocations revoked
                         WHERE revoked.question_id = mapping.question_id
                     )
                   GROUP BY mapping.topic_id""",
                (resolved_release,),
            ).fetchall()
        concepts_by_topic: dict[str, list[dict[str, Any]]] = {}
        for row in concept_rows:
            concepts_by_topic.setdefault(row["topic_id"], []).append(
                {"id": row["concept_id"], "name": row["name"]}
            )
        coverage = {row["topic_id"]: row for row in coverage_rows}
        return {
            "release_id": resolved_release,
            "domains": [
                {
                    "id": row["domain_id"],
                    "name": row["name"],
                    "description": row["description"],
                    "sort_order": row["sort_order"],
                }
                for row in domain_rows
            ],
            "topics": [
                {
                    "id": row["topic_id"],
                    "domain_id": row["domain_id"],
                    "parent_id": row["parent_topic_id"],
                    "name": row["name"],
                    "description": row["description"],
                    "related_topic_ids": json.loads(
                        row["related_topic_ids_json"]
                    ),
                    "sort_order": row["sort_order"],
                    "concepts": concepts_by_topic.get(row["topic_id"], []),
                    "direct_primary_questions": int(
                        coverage.get(row["topic_id"], {"primary_items": 0})[
                            "primary_items"
                        ]
                    ),
                    "cross_topic_questions": int(
                        coverage.get(row["topic_id"], {"cross_items": 0})[
                            "cross_items"
                        ]
                    ),
                }
                for row in topic_rows
            ],
        }

    def resolve_topic(
        self, reference: str, release_id: str | None = None
    ) -> dict[str, Any]:
        if not isinstance(reference, str) or not reference.strip():
            raise ValidationError("Topic reference must be a non-blank string.")
        catalog = self.get_catalog(release_id)
        normalized = " ".join(reference.split()).casefold()
        exact = [topic for topic in catalog["topics"] if topic["id"] == reference]
        named = [
            topic
            for topic in catalog["topics"]
            if " ".join(topic["name"].split()).casefold() == normalized
        ]
        matches = exact or named
        if not matches:
            raise NotFoundError(f"Unknown curriculum topic: {reference}")
        if len(matches) > 1:
            raise ValidationError(
                f"Topic name {reference!r} is ambiguous; use a stable topic ID."
            )
        return {**matches[0], "release_id": catalog["release_id"]}

    def topic_scope(self, topic_id: str, release_id: str | None = None) -> set[str]:
        """Return owned objectives, descendants, and their prerequisites."""
        catalog = self.get_catalog(release_id)
        owned = self.topic_owned_concepts(
            topic_id, catalog["release_id"], include_descendants=True
        )
        graph = self.get_graph(catalog["release_id"])
        scope: set[str] = set()
        for concept_id in owned:
            scope.update(graph.learning_scope(concept_id))
        return scope

    def topic_owned_concepts(
        self,
        topic_id: str,
        release_id: str | None = None,
        *,
        include_descendants: bool = True,
    ) -> set[str]:
        catalog = self.get_catalog(release_id)
        topic_by_id = {topic["id"]: topic for topic in catalog["topics"]}
        if topic_id not in topic_by_id:
            raise NotFoundError(
                f"Topic {topic_id} is not in corpus release {catalog['release_id']}."
            )
        selected_topics = {topic_id}
        if include_descendants:
            changed = True
            while changed:
                changed = False
                for topic in catalog["topics"]:
                    if (
                        topic["parent_id"] in selected_topics
                        and topic["id"] not in selected_topics
                    ):
                        selected_topics.add(topic["id"])
                        changed = True
        return {
            concept["id"]
            for candidate_id in selected_topics
            for concept in topic_by_id[candidate_id]["concepts"]
        }

    def topic_for_concept(
        self, concept_id: str, release_id: str | None = None
    ) -> dict[str, Any] | None:
        catalog = self.get_catalog(release_id)
        for topic in catalog["topics"]:
            if any(concept["id"] == concept_id for concept in topic["concepts"]):
                return {**topic, "release_id": catalog["release_id"]}
        return None

    def question_topics(
        self, question_id: str, release_id: str | None = None
    ) -> list[dict[str, Any]]:
        catalog = self.get_catalog(release_id)
        topics = {topic["id"]: topic for topic in catalog["topics"]}
        with self.read() as connection:
            rows = connection.execute(
                """SELECT topic_id, relation FROM release_question_topics
                   WHERE release_id = ? AND question_id = ?
                   ORDER BY CASE relation WHEN 'primary' THEN 0 ELSE 1 END, topic_id""",
                (catalog["release_id"], question_id),
            ).fetchall()
        return [
            {
                "id": row["topic_id"],
                "name": topics[row["topic_id"]]["name"],
                "relation": row["relation"],
            }
            for row in rows
            if row["topic_id"] in topics
        ]

    def get_misconceptions(
        self,
        ids: set[str] | None = None,
        *,
        release_id: str | None = None,
    ) -> list[Misconception]:
        if ids is not None and not ids:
            return []
        with self.read() as connection:
            parameters: list[Any] = []
            joins: list[str] = []
            where: list[str] = []
            if release_id is not None:
                joins.append(
                    " JOIN release_misconceptions membership"
                    " ON membership.misconception_id = misconception.id"
                )
                where.append("membership.release_id = ?")
                parameters.append(release_id)
            if ids is not None:
                where.append(
                    "misconception.id IN (SELECT value FROM json_each(?))"
                )
                parameters.append(
                    json.dumps(
                        sorted(ids), ensure_ascii=False, separators=(",", ":")
                    )
                )
            clause = " WHERE " + " AND ".join(where) if where else ""
            rows = connection.execute(
                f"SELECT misconception.* FROM misconceptions misconception"
                f"{''.join(joins)}{clause}",
                tuple(parameters),
            ).fetchall()
        return [Misconception(row["id"], row["concept_id"], row["name"], row["description"]) for row in rows]

    @staticmethod
    def _objective_from_row(row: sqlite3.Row) -> LearningObjective:
        return LearningObjective(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            primary_concept_id=row["primary_concept_id"],
            supporting_concept_ids=tuple(
                json.loads(row["supporting_concept_ids_json"])
            ),
            operation=ObjectiveOperation(row["operation"]),
            evidence_type=row["evidence_type"],
            prior_mastery=row["prior_mastery"],
        )

    def get_learning_objectives(
        self,
        release_id: str | None = None,
        *,
        primary_concept_ids: set[str] | None = None,
    ) -> list[LearningObjective]:
        if primary_concept_ids is not None and not primary_concept_ids:
            return []
        with self.read() as connection:
            resolved_release = release_id or self.get_active_release_id(connection)
            parameters: list[Any] = [resolved_release]
            where = ["membership.release_id = ?"]
            if primary_concept_ids is not None:
                placeholders = ",".join("?" for _ in primary_concept_ids)
                where.append(f"objective.primary_concept_id IN ({placeholders})")
                parameters.extend(sorted(primary_concept_ids))
            rows = connection.execute(
                """SELECT objective.* FROM learning_objectives objective
                   JOIN release_learning_objectives membership
                     ON membership.objective_id = objective.id
                   WHERE """
                + " AND ".join(where)
                + " ORDER BY objective.primary_concept_id, objective.name, objective.id",
                tuple(parameters),
            ).fetchall()
        graph_version, graph_edges = self.get_objective_graph(resolved_release)
        edges_by_target: dict[str, list[ObjectiveEdge]] = {}
        for edge in graph_edges:
            edges_by_target.setdefault(edge.target_id, []).append(edge)
        return [
            replace(
                self._objective_from_row(row),
                prerequisites=tuple(edges_by_target.get(row["id"], [])),
                objective_graph_version=graph_version,
            )
            for row in rows
        ]

    def get_objective_graph(
        self, release_id: str | None = None
    ) -> tuple[int | None, tuple[ObjectiveEdge, ...]]:
        """Return the pinned objective graph capability and its immutable edges.

        ``None`` means the release predates objective-graph declarations.  It
        is intentionally different from ``(1, ())``, which is an explicitly
        empty graph and must not trigger broad-concept fallback routing.
        """

        with self.read() as connection:
            resolved_release = release_id or self.get_active_release_id(connection)
            graph = connection.execute(
                """SELECT graph_version FROM release_objective_graphs
                   WHERE release_id = ?""",
                (resolved_release,),
            ).fetchone()
            rows = connection.execute(
                """SELECT edge_id, source_objective_id,
                          target_objective_id, relation, weight, rationale
                   FROM release_objective_edges
                   WHERE release_id = ?
                   ORDER BY edge_id""",
                (resolved_release,),
            ).fetchall()
            objective_ids = {
                row["objective_id"]
                for row in connection.execute(
                    """SELECT objective_id FROM release_learning_objectives
                       WHERE release_id = ?""",
                    (resolved_release,),
                ).fetchall()
            }
        if graph is None:
            if rows:
                raise ValidationError(
                    "Objective edges exist without a declared pinned graph."
                )
            return None, ()
        if graph["graph_version"] != 1:
            raise ValidationError(
                f"Unsupported objective graph version {graph['graph_version']}."
            )
        edges = tuple(
            ObjectiveEdge(
                id=row["edge_id"],
                source_id=row["source_objective_id"],
                target_id=row["target_objective_id"],
                relation=RelationType(row["relation"]),
                weight=row["weight"],
                rationale=row["rationale"],
            )
            for row in rows
        )
        if not objective_ids:
            raise ValidationError(
                "A declared objective graph requires release learning objectives."
            )
        for edge in edges:
            outside = {edge.source_id, edge.target_id} - objective_ids
            if outside:
                raise ValidationError(
                    f"Objective edge {edge.id} references objectives outside "
                    "the pinned release."
                )
        endpoint_pairs = [(edge.source_id, edge.target_id) for edge in edges]
        if len(endpoint_pairs) != len(set(endpoint_pairs)):
            raise ValidationError(
                "Objective graph has ambiguous duplicate prerequisite routes."
            )
        adjacency = {objective_id: [] for objective_id in objective_ids}
        indegree = {objective_id: 0 for objective_id in objective_ids}
        for edge in edges:
            adjacency[edge.source_id].append(edge.target_id)
            indegree[edge.target_id] += 1
        queue = deque(
            objective_id
            for objective_id, degree in indegree.items()
            if degree == 0
        )
        visited = 0
        while queue:
            objective_id = queue.popleft()
            visited += 1
            for dependent_id in adjacency[objective_id]:
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    queue.append(dependent_id)
        if visited != len(objective_ids):
            raise ValidationError(
                "Objective prerequisite graph contains a cycle."
            )
        return 1, edges

    def _question_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        release_id: str | None = None,
    ) -> Question:
        concept_rows = connection.execute(
            """SELECT * FROM question_concepts
               WHERE question_id = ? ORDER BY rowid""",
            (row["id"],),
        ).fetchall()
        option_rows = connection.execute(
            "SELECT * FROM options WHERE question_id = ? ORDER BY rowid", (row["id"],)
        ).fetchall()
        diagnostic_by_option: dict[str, str] = {}
        if release_id is not None:
            diagnostic_by_option = {
                diagnostic_row["option_id"]: diagnostic_row["objective_id"]
                for diagnostic_row in connection.execute(
                    """SELECT option_id, objective_id
                       FROM release_option_objectives
                       WHERE release_id = ? AND question_id = ?""",
                    (release_id, row["id"]),
                ).fetchall()
            }
        source_rows = connection.execute(
            """SELECT source_id FROM question_sources
               WHERE question_id = ? ORDER BY rowid""",
            (row["id"],),
        ).fetchall()
        objective = None
        if release_id is not None:
            objective_row = connection.execute(
                """SELECT objective.* FROM learning_objectives objective
                   JOIN release_question_objectives mapping
                     ON mapping.objective_id = objective.id
                   WHERE mapping.release_id = ? AND mapping.question_id = ?""",
                (release_id, row["id"]),
            ).fetchone()
            if objective_row is not None:
                objective = self._objective_from_row(objective_row)
        return Question(
            id=row["id"],
            version=row["version"],
            family_id=row["family_id"],
            status=QuestionStatus(row["status"]),
            stem=row["stem"],
            kind=QuestionKind(row["kind"]),
            difficulty=row["difficulty"],
            discrimination=row["discrimination"],
            guess_rate=row["guess_rate"],
            slip_rate=row["slip_rate"],
            concepts=tuple(
                ConceptWeight(r["concept_id"], r["weight"], r["role"]) for r in concept_rows
            ),
            options=tuple(
                Option(
                    id=r["option_id"],
                    text=r["text"],
                    correct=bool(r["is_correct"]),
                    rationale=r["rationale"],
                    misconception_id=r["misconception_id"],
                    diagnostic_objective_id=diagnostic_by_option.get(
                        r["option_id"]
                    ),
                )
                for r in option_rows
            ),
            source_ids=tuple(r["source_id"] for r in source_rows),
            provenance=json.loads(row["provenance_json"]),
            tags=tuple(json.loads(row["tags_json"])),
            revision_of=row["revision_of"],
            objective=objective,
        )

    def get_question(
        self,
        question_id: str,
        connection: sqlite3.Connection | None = None,
        *,
        release_id: str | None = None,
    ) -> Question:
        owns_connection = connection is None
        conn = connection or self.connect()
        try:
            row = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
            if not row:
                raise NotFoundError(f"Unknown question: {question_id}")
            return self._question_from_row(conn, row, release_id=release_id)
        finally:
            if owns_connection:
                conn.close()

    def get_active_release_id(
        self, connection: sqlite3.Connection | None = None
    ) -> str:
        owns_connection = connection is None
        conn = connection or self.connect()
        try:
            row = conn.execute(
                """SELECT meta.value FROM meta
                   JOIN corpus_releases release ON release.id = meta.value
                   WHERE meta.key = 'active_corpus_release'
                     AND release.sealed_at IS NOT NULL"""
            ).fetchone()
            if not row:
                raise NotFoundError(
                    "No sealed active corpus release. Import a corpus first."
                )
            return row["value"]
        finally:
            if owns_connection:
                conn.close()

    def questions_for_scope(
        self,
        concept_ids: set[str],
        *,
        learner_id: str | None = None,
        focus_concept_id: str | None = None,
        focus_misconception_id: str | None = None,
        focus_objective_id: str | None = None,
        release_id: str | None = None,
        target_difficulty: float = 0.0,
        limit: int = 600,
    ) -> list[Question]:
        """Retrieve a bounded, indexed candidate pool instead of scanning the bank.

        The temporary scope table avoids SQLite's parameter limit for large concept
        closures. Retrieval favors a live remediation focus, low personal exposure,
        and proximity to the learner's current latent ability. Family-level exposure
        pushes every sibling of a previously presented item behind unseen families.
        When the fast bounded result itself contains siblings, a family-ranked query
        takes the first variant from every available family before second variants,
        so a prolific family cannot hide independent repair or verification paths.
        The adaptive policy performs richer scoring only over this bounded pool.
        """
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not (1 <= limit <= 600)
        ):
            raise ValidationError("Candidate retrieval limit must be between 1 and 600.")
        if focus_objective_id is not None and (
            not isinstance(focus_objective_id, str)
            or not focus_objective_id.strip()
        ):
            raise ValidationError(
                "Focused objective ID must be a non-blank string or null."
            )
        if not concept_ids and focus_objective_id is None:
            return []
        with self.read() as connection:
            release_id = release_id or self.get_active_release_id(connection)
            if focus_objective_id is not None:
                id_rows = connection.execute(
                    OBJECTIVE_CANDIDATE_POOL_SQL,
                    (
                        focus_misconception_id,
                        target_difficulty,
                        learner_id or "",
                        release_id,
                        focus_objective_id,
                        QuestionStatus.APPROVED.value,
                        QuestionStatus.CALIBRATED.value,
                        max(1, limit),
                    ),
                ).fetchall()
                question_ids = [row["id"] for row in id_rows]
                questions = self._questions_by_ids(
                    connection, question_ids, release_id=release_id
                )
                return [
                    question
                    for question in questions
                    if question_runtime_activation_safe(question)
                ]
            connection.execute("CREATE TEMP TABLE requested_scope(id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO requested_scope(id) VALUES (?)",
                ((concept_id,) for concept_id in sorted(concept_ids)),
            )
            parameters = (
                focus_concept_id,
                focus_misconception_id,
                target_difficulty,
                release_id,
                learner_id or "",
                QuestionStatus.APPROVED.value,
                QuestionStatus.CALIBRATED.value,
                max(1, limit),
            )
            id_rows = connection.execute(
                CANDIDATE_POOL_SQL, parameters
            ).fetchall()
            if len({row["family_id"] for row in id_rows}) != len(id_rows):
                id_rows = connection.execute(
                    FAMILY_DIVERSE_CANDIDATE_POOL_SQL, parameters
                ).fetchall()
            question_ids = [row["id"] for row in id_rows]
            questions = self._questions_by_ids(
                connection, question_ids, release_id=release_id
            )
            return [
                question
                for question in questions
                if question_runtime_activation_safe(question)
            ]

    def _questions_by_ids(
        self,
        connection: sqlite3.Connection,
        question_ids: Sequence[str],
        *,
        release_id: str | None = None,
    ) -> list[Question]:
        if not question_ids:
            return []
        placeholders = ",".join("?" for _ in question_ids)
        if release_id is None:
            question_rows = connection.execute(
                f"SELECT q.*, q.status AS resolved_status "
                f"FROM questions q WHERE q.id IN ({placeholders})",
                tuple(question_ids),
            ).fetchall()
        else:
            question_rows = connection.execute(
                f"""SELECT q.*, rq.status AS resolved_status
                    FROM questions q
                    JOIN release_questions rq
                      ON rq.question_id = q.id AND rq.release_id = ?
                    WHERE q.id IN ({placeholders})""",
                (release_id, *question_ids),
            ).fetchall()
        concept_rows = connection.execute(
            f"""SELECT * FROM question_concepts
                WHERE question_id IN ({placeholders})
                ORDER BY question_id, rowid""",
            tuple(question_ids),
        ).fetchall()
        option_rows = connection.execute(
            f"""SELECT * FROM options WHERE question_id IN ({placeholders})
                ORDER BY question_id, rowid""",
            tuple(question_ids),
        ).fetchall()
        diagnostic_by_question_option: dict[tuple[str, str], str] = {}
        if release_id is not None:
            diagnostic_rows = connection.execute(
                f"""SELECT question_id, option_id, objective_id
                    FROM release_option_objectives
                    WHERE release_id = ?
                      AND question_id IN ({placeholders})""",
                (release_id, *question_ids),
            ).fetchall()
            diagnostic_by_question_option = {
                (diagnostic_row["question_id"], diagnostic_row["option_id"]):
                    diagnostic_row["objective_id"]
                for diagnostic_row in diagnostic_rows
            }
        source_rows = connection.execute(
            f"""SELECT * FROM question_sources
                WHERE question_id IN ({placeholders})
                ORDER BY question_id, rowid""",
            tuple(question_ids),
        ).fetchall()
        objective_by_question: dict[str, LearningObjective] = {}
        if release_id is not None:
            objective_rows = connection.execute(
                f"""SELECT mapping.question_id, objective.*
                    FROM release_question_objectives mapping
                    JOIN learning_objectives objective
                      ON objective.id = mapping.objective_id
                    WHERE mapping.release_id = ?
                      AND mapping.question_id IN ({placeholders})""",
                (release_id, *question_ids),
            ).fetchall()
            objective_by_question = {
                objective_row["question_id"]: self._objective_from_row(
                    objective_row
                )
                for objective_row in objective_rows
            }
        concepts_by_question: dict[str, list[ConceptWeight]] = {}
        options_by_question: dict[str, list[Option]] = {}
        sources_by_question: dict[str, list[str]] = {}
        for row in concept_rows:
            concepts_by_question.setdefault(row["question_id"], []).append(
                ConceptWeight(row["concept_id"], row["weight"], row["role"])
            )
        for row in option_rows:
            options_by_question.setdefault(row["question_id"], []).append(
                Option(
                    id=row["option_id"],
                    text=row["text"],
                    correct=bool(row["is_correct"]),
                    rationale=row["rationale"],
                    misconception_id=row["misconception_id"],
                    diagnostic_objective_id=diagnostic_by_question_option.get(
                        (row["question_id"], row["option_id"])
                    ),
                )
            )
        for row in source_rows:
            sources_by_question.setdefault(row["question_id"], []).append(row["source_id"])
        by_id = {
            row["id"]: Question(
                id=row["id"],
                version=row["version"],
                family_id=row["family_id"],
                status=QuestionStatus(row["resolved_status"]),
                stem=row["stem"],
                kind=QuestionKind(row["kind"]),
                difficulty=row["difficulty"],
                discrimination=row["discrimination"],
                guess_rate=row["guess_rate"],
                slip_rate=row["slip_rate"],
                concepts=tuple(
                    concepts_by_question.get(row["id"], [])
                ),
                options=tuple(options_by_question.get(row["id"], [])),
                source_ids=tuple(sources_by_question.get(row["id"], [])),
                provenance=json.loads(row["provenance_json"]),
                tags=tuple(json.loads(row["tags_json"])),
                revision_of=row["revision_of"],
                objective=objective_by_question.get(row["id"]),
            )
            for row in question_rows
        }
        return [by_id[question_id] for question_id in question_ids if question_id in by_id]

    def get_skill_states(
        self,
        learner_id: str,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, SkillState]:
        owns_connection = connection is None
        conn = connection or self.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM skill_states WHERE learner_id = ?", (learner_id,)
            ).fetchall()
        finally:
            if owns_connection:
                conn.close()
        return {
            row["concept_id"]: SkillState(
                learner_id=row["learner_id"],
                concept_id=row["concept_id"],
                mean=row["mean"],
                variance=row["variance"],
                stability_hours=row["stability_hours"],
                exposures=row["exposures"],
                last_seen_at=from_timestamp(row["last_seen_at"]),
                next_review_at=from_timestamp(row["next_review_at"]),
                evidence_mass=row["evidence_mass"],
            )
            for row in rows
        }

    def get_objective_states(
        self,
        learner_id: str,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, ObjectiveState]:
        owns_connection = connection is None
        conn = connection or self.connect()
        try:
            rows = conn.execute(
                OBJECTIVE_STATE_WITH_GRID_SELECT
                + " WHERE state.learner_id = ? ORDER BY state.objective_id",
                (learner_id,),
            ).fetchall()
            result = {
                row["objective_id"]: self._objective_state_from_joined_row(row)
                for row in rows
            }
        finally:
            if owns_connection:
                conn.close()
        return result

    @staticmethod
    def load_objective_state(
        connection: sqlite3.Connection,
        learner_id: str,
        objective_id: str,
    ) -> ObjectiveState | None:
        """Load one objective projection through the exact-state trust boundary.

        The Gaussian columns remain useful for reporting and backwards replay,
        but they are only redundant summaries for exact-grid v6/v7 states.
        Never reconstruct a missing or damaged exact posterior from those
        summaries.
        """

        row = connection.execute(
            OBJECTIVE_STATE_WITH_GRID_SELECT
            + " WHERE state.learner_id = ? AND state.objective_id = ?",
            (learner_id, objective_id),
        ).fetchone()
        if row is None:
            return None
        return Database._objective_state_from_joined_row(row)

    @staticmethod
    def _objective_state_from_joined_row(row: sqlite3.Row) -> ObjectiveState:
        learner_id = row["learner_id"]
        objective_id = row["objective_id"]
        label = f"objective state {learner_id}/{objective_id}"
        model_version = row["model_version"]
        has_grid = row["grid_learner_id"] is not None

        if model_version in OBJECTIVE_GRID_MODEL_VERSIONS:
            if not has_grid:
                raise ValidationError(
                    f"{label}: exact-grid state is missing its exact posterior."
                )
        elif model_version == OBJECTIVE_GAUSSIAN_MODEL_VERSION:
            if has_grid:
                raise ValidationError(
                    f"{label}: legacy v5 state must not have an exact-posterior child."
                )
        else:
            raise ValidationError(
                f"{label}: unsupported objective model version {model_version!r}."
            )

        posterior: ObjectivePosterior | None = None
        if has_grid:
            if row["grid_objective_id"] != objective_id or row[
                "grid_learner_id"
            ] != learner_id:
                raise ValidationError(f"{label}: exact-posterior identity mismatch.")
            identity = OBJECTIVE_POSTERIOR_V1_IDENTITY
            expected_metadata = {
                "grid_posterior_schema_version": (
                    identity.schema_version
                ),
                "grid_algorithm": identity.algorithm,
                "grid_grid_id": identity.grid_id,
                "grid_codec": identity.codec,
                "grid_model_version": model_version,
                "grid_as_of_event_id": row["as_of_event_id"],
            }
            for field, expected in expected_metadata.items():
                if row[field] != expected:
                    raise ValidationError(
                        f"{label}: exact-posterior {field.removeprefix('grid_')} "
                        "mismatch."
                    )
            blob = row["grid_posterior_blob"]
            digest = row["grid_posterior_sha256"]
            try:
                posterior = decode_objective_posterior(
                    blob, expected_digest=digest
                )
            except ObjectivePosteriorError as exc:
                raise ValidationError(
                    f"{label}: exact posterior cannot be decoded ({exc})"
                ) from exc
            if posterior_digest(blob) != digest:
                # The decoder already checks this.  Keep the explicit semantic
                # assertion here so this boundary remains correct if a future
                # decoder gains another verification mode.
                raise ValidationError(f"{label}: exact-posterior digest mismatch.")
            metrics = posterior.metrics()
            redundant_metrics = {
                "grid_mean": metrics.mean,
                "grid_variance": metrics.variance,
                "grid_mastery_probability": metrics.mastery_probability,
                "grid_expected_competence": metrics.expected_competence,
                "grid_edge_mass": metrics.edge_mass,
                "grid_mastery_probability_error_bound": (
                    metrics.mastery_probability_error_bound
                ),
                "grid_evidence_mass": metrics.evidence_mass,
                "grid_acquisition_mass": metrics.acquisition_mass,
                "mean": metrics.mean,
                "variance": metrics.variance,
                "evidence_mass": metrics.evidence_mass,
            }
            for field, derived in redundant_metrics.items():
                stored = row[field]
                if (
                    type(stored) not in {int, float}
                    or not isfinite(stored)
                    or float(stored) != derived
                ):
                    name = field.removeprefix("grid_").replace("_", " ")
                    raise ValidationError(
                        f"{label}: stored {name} does not match the exact posterior."
                    )

        try:
            return ObjectiveState(
                learner_id=learner_id,
                objective_id=objective_id,
                mean=row["mean"],
                variance=row["variance"],
                stability_hours=row["stability_hours"],
                exposures=row["exposures"],
                last_seen_at=from_timestamp(row["last_seen_at"]),
                next_review_at=from_timestamp(row["next_review_at"]),
                evidence_mass=row["evidence_mass"],
                posterior=posterior,
                model_version=model_version,
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{label}: invalid projection ({exc})") from exc

    @staticmethod
    def upsert_objective_grid_state(
        connection: sqlite3.Connection,
        *,
        learner_id: str,
        objective_id: str,
        posterior: ObjectivePosterior,
        as_of_event_id: str,
        model_version: str,
    ) -> None:
        """Persist an exact posterior and all auditable summaries atomically."""

        if model_version not in OBJECTIVE_GRID_MODEL_VERSIONS:
            raise ValidationError(
                "Exact objective posteriors require learner model "
                f"in {sorted(OBJECTIVE_GRID_MODEL_VERSIONS)!r}."
            )
        if not isinstance(posterior, ObjectivePosterior):
            raise ValidationError(
                "Exact objective posterior must be an ObjectivePosterior."
            )
        parent = connection.execute(
            """SELECT mean, variance, evidence_mass, as_of_event_id, model_version
               FROM objective_states
               WHERE learner_id = ? AND objective_id = ?""",
            (learner_id, objective_id),
        ).fetchone()
        if parent is None:
            raise ValidationError(
                "An exact objective posterior requires its parent objective state."
            )
        metrics = posterior.metrics()
        for field, expected in (
            ("mean", metrics.mean),
            ("variance", metrics.variance),
            ("evidence_mass", metrics.evidence_mass),
            ("as_of_event_id", as_of_event_id),
            ("model_version", model_version),
        ):
            if parent[field] != expected:
                raise ValidationError(
                    f"Objective-state parent {field.replace('_', ' ')} does not "
                    "match its exact posterior write."
                )
        encoded = posterior.encode()
        digest = posterior_digest(encoded)
        identity = OBJECTIVE_POSTERIOR_V1_IDENTITY
        connection.execute(
            """INSERT INTO objective_grid_states(
                   learner_id, objective_id, posterior_schema_version,
                   algorithm, grid_id, codec, posterior_blob,
                   posterior_sha256, mean, variance, mastery_probability,
                   expected_competence, edge_mass,
                   mastery_probability_error_bound, evidence_mass,
                   acquisition_mass, as_of_event_id, model_version
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(learner_id, objective_id) DO UPDATE SET
                   posterior_schema_version=excluded.posterior_schema_version,
                   algorithm=excluded.algorithm,
                   grid_id=excluded.grid_id,
                   codec=excluded.codec,
                   posterior_blob=excluded.posterior_blob,
                   posterior_sha256=excluded.posterior_sha256,
                   mean=excluded.mean,
                   variance=excluded.variance,
                   mastery_probability=excluded.mastery_probability,
                   expected_competence=excluded.expected_competence,
                   edge_mass=excluded.edge_mass,
                   mastery_probability_error_bound=
                       excluded.mastery_probability_error_bound,
                   evidence_mass=excluded.evidence_mass,
                   acquisition_mass=excluded.acquisition_mass,
                   as_of_event_id=excluded.as_of_event_id,
                   model_version=excluded.model_version""",
            (
                learner_id,
                objective_id,
                identity.schema_version,
                identity.algorithm,
                identity.grid_id,
                identity.codec,
                encoded,
                digest,
                metrics.mean,
                metrics.variance,
                metrics.mastery_probability,
                metrics.expected_competence,
                metrics.edge_mass,
                metrics.mastery_probability_error_bound,
                metrics.evidence_mass,
                metrics.acquisition_mass,
                as_of_event_id,
                model_version,
            ),
        )

    def get_misconception_beliefs(self, learner_id: str) -> dict[str, MisconceptionBelief]:
        with self.read() as connection:
            rows = connection.execute(
                "SELECT * FROM misconception_beliefs WHERE learner_id = ?", (learner_id,)
            ).fetchall()
        return {
            row["misconception_id"]: MisconceptionBelief(
                learner_id=row["learner_id"],
                misconception_id=row["misconception_id"],
                log_odds=row["log_odds"],
                evidence_count=row["evidence_count"],
                last_seen_at=from_timestamp(row["last_seen_at"]),
            )
            for row in rows
        }

    def learner_projection_hash(
        self,
        learner_id: str,
        connection: sqlite3.Connection | None = None,
        *,
        hash_version: int | None = None,
    ) -> str:
        """Hash every mutable learner-model projection in canonical order."""
        owns_connection = connection is None
        conn = connection or self.connect()
        try:
            learner = conn.execute(
                "SELECT id, revision FROM learners WHERE id = ?", (learner_id,)
            ).fetchone()
            if not learner:
                raise NotFoundError(f"Unknown learner: {learner_id}")
            skill_states = [
                dict(row)
                for row in conn.execute(
                    """SELECT learner_id, concept_id, mean, variance,
                              stability_hours, exposures, last_seen_at,
                              next_review_at, evidence_mass, as_of_event_id,
                              model_version
                       FROM skill_states WHERE learner_id = ?
                       ORDER BY concept_id""",
                    (learner_id,),
                )
            ]
            misconception_beliefs = [
                dict(row)
                for row in conn.execute(
                    """SELECT learner_id, misconception_id, log_odds,
                              evidence_count, last_seen_at, as_of_event_id,
                              model_version
                       FROM misconception_beliefs WHERE learner_id = ?
                       ORDER BY misconception_id""",
                    (learner_id,),
                )
            ]
            skill_families = [
                dict(row)
                for row in conn.execute(
                    """SELECT learner_id, concept_id, family_id, kind,
                              first_unguided_correct_at,
                              last_unguided_correct_at,
                              delayed_unguided_correct_at
                       FROM learner_skill_families WHERE learner_id = ?
                       ORDER BY concept_id, family_id""",
                    (learner_id,),
                )
            ]
            objective_states = [
                dict(row)
                for row in conn.execute(
                    """SELECT learner_id, objective_id, mean, variance,
                              stability_hours, exposures, last_seen_at,
                              next_review_at, evidence_mass, as_of_event_id,
                              model_version
                       FROM objective_states WHERE learner_id = ?
                       ORDER BY objective_id""",
                    (learner_id,),
                )
            ]
            objective_families = [
                dict(row)
                for row in conn.execute(
                    """SELECT learner_id, objective_id, family_id, kind,
                              first_unguided_correct_at,
                              last_unguided_correct_at,
                              delayed_unguided_correct_at
                       FROM learner_objective_families WHERE learner_id = ?
                       ORDER BY objective_id, family_id""",
                    (learner_id,),
                )
            ]
            objective_grid_states = [
                dict(row)
                for row in conn.execute(
                    """SELECT learner_id, objective_id,
                              posterior_schema_version, algorithm, grid_id,
                              codec, posterior_sha256, mean, variance,
                              mastery_probability, expected_competence,
                              edge_mass, mastery_probability_error_bound,
                              evidence_mass, acquisition_mass,
                              as_of_event_id, model_version
                       FROM objective_grid_states WHERE learner_id = ?
                       ORDER BY objective_id""",
                    (learner_id,),
                )
            ]
            # Hashing is a trust boundary too: validate the blob and every
            # redundant summary before committing to its compact semantic
            # digest.  Version 3 intentionally does not copy the ~23 KiB blob
            # into each projection event.
            joined_objective_rows = conn.execute(
                OBJECTIVE_STATE_WITH_GRID_SELECT
                + " WHERE state.learner_id = ? ORDER BY state.objective_id",
                (learner_id,),
            ).fetchall()
            for row in joined_objective_rows:
                self._objective_state_from_joined_row(row)
            resolved_hash_version = hash_version
            if resolved_hash_version is None:
                resolved_hash_version = (
                    3
                    if objective_grid_states
                    else (2 if objective_states or objective_families else 1)
                )
            if resolved_hash_version not in {1, 2, 3}:
                raise ValidationError(
                    "Learner projection hash version must be 1, 2, or 3."
                )
            if resolved_hash_version == 1 and (
                objective_states
                or objective_families
                or objective_grid_states
            ):
                raise ValidationError(
                    "Projection hash version 1 cannot omit objective projections."
                )
            if resolved_hash_version == 2 and objective_grid_states:
                raise ValidationError(
                    "Projection hash version 2 cannot omit exact objective posteriors."
                )
            payload: dict[str, Any] = {
                "learner_id": learner["id"],
                "learner_revision": learner["revision"],
                "skill_states": skill_states,
                "misconception_beliefs": misconception_beliefs,
                "skill_families": skill_families,
            }
            if resolved_hash_version >= 2:
                payload.update(
                    {
                        "projection_hash_version": resolved_hash_version,
                        "objective_states": objective_states,
                        "objective_families": objective_families,
                    }
                )
            if resolved_hash_version == 3:
                payload["objective_grid_states"] = objective_grid_states
            return _content_hash(payload)
        finally:
            if owns_connection:
                conn.close()

    def get_exposure_summary(
        self,
        learner_id: str,
        *,
        question_ids: set[str] | None = None,
        family_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        with self.read() as connection:
            if question_ids is None:
                question_rows = connection.execute(
                    """SELECT question_id, COUNT(*) AS n FROM decisions
                       WHERE learner_id = ? GROUP BY question_id""",
                    (learner_id,),
                ).fetchall()
            elif question_ids:
                placeholders = ",".join("?" for _ in question_ids)
                question_rows = connection.execute(
                    f"""SELECT question_id, COUNT(*) AS n FROM decisions
                        WHERE learner_id = ? AND question_id IN ({placeholders})
                        GROUP BY question_id""",
                    (learner_id, *sorted(question_ids)),
                ).fetchall()
            else:
                question_rows = []
            if family_ids is None:
                family_rows = connection.execute(
                    """SELECT question.family_id, COUNT(*) AS n,
                              MAX(decision.created_at) AS last_at
                       FROM decisions decision
                       JOIN questions question ON question.id = decision.question_id
                       WHERE decision.learner_id = ? GROUP BY question.family_id""",
                    (learner_id,),
                ).fetchall()
            elif family_ids:
                placeholders = ",".join("?" for _ in family_ids)
                family_rows = connection.execute(
                    f"""SELECT question.family_id, COUNT(*) AS n,
                               MAX(decision.created_at) AS last_at
                        FROM decisions decision
                        JOIN questions question ON question.id = decision.question_id
                        WHERE decision.learner_id = ?
                          AND question.family_id IN ({placeholders})
                        GROUP BY question.family_id""",
                    (learner_id, *sorted(family_ids)),
                ).fetchall()
            else:
                family_rows = []
        return {
            "questions": {row["question_id"]: row["n"] for row in question_rows},
            "families": {row["family_id"]: {"count": row["n"], "last_at": row["last_at"]} for row in family_rows},
        }

    def session_exposure_summary(self, session_id: str) -> dict[str, set[str]]:
        with self.read() as connection:
            rows = connection.execute(
                """SELECT decision.question_id, question.family_id
                   FROM decisions decision
                   JOIN questions question ON question.id = decision.question_id
                   WHERE decision.session_id = ?""",
                (session_id,),
            ).fetchall()
        return {
            "questions": {row["question_id"] for row in rows},
            "families": {row["family_id"] for row in rows},
        }

    def session_recent_performance(
        self, session_id: str, *, limit: int = 3
    ) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValidationError("Performance history limit must be a positive integer.")
        with self.read() as connection:
            rows = connection.execute(
                """SELECT attempt.is_correct, attempt.selected_option_id,
                          attempt.confidence,
                          attempt.response_ms, attempt.hint_count,
                          decision.pedagogical_role, question.id AS question_id,
                          primary_mapping.concept_id AS primary_concept_id,
                          json_extract(
                              response_event.metadata_json,
                              '$.learner_model_version'
                          ) AS learner_model_version
                   FROM attempts attempt
                   JOIN decisions decision ON decision.id = attempt.decision_id
                   JOIN events response_event
                     ON response_event.event_id = attempt.event_id
                   JOIN questions question ON question.id = attempt.question_id
                   JOIN question_concepts primary_mapping
                     ON primary_mapping.question_id = question.id
                    AND primary_mapping.role = 'primary'
                   WHERE attempt.session_id = ?
                   ORDER BY attempt.answered_at DESC, attempt.id DESC
                   LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [
            {
                "correct": bool(row["is_correct"]),
                "selected_option_id": row["selected_option_id"],
                "confidence": row["confidence"],
                "response_ms": row["response_ms"],
                "hint_count": row["hint_count"],
                "pedagogical_role": row["pedagogical_role"],
                "question_id": row["question_id"],
                "primary_concept_id": row["primary_concept_id"],
                "learner_model_version": row["learner_model_version"],
            }
            for row in rows
        ]

    def independent_evidence_summary(
        self,
        learner_id: str,
        concept_id: str,
        *,
        release_id: str | None = None,
    ) -> dict[str, int]:
        return self.independent_evidence_summaries(
            learner_id,
            {concept_id},
            release_id=release_id,
        )[concept_id]

    def independent_evidence_summaries(
        self,
        learner_id: str,
        concept_ids: set[str],
        *,
        release_id: str | None = None,
    ) -> dict[str, dict[str, int]]:
        """Summarize certificate evidence accepted by one corpus release.

        The family ledgers are lifetime projections and remain immutable
        historical evidence.  A current certificate count is narrower: the
        release must still contain an approved or calibrated, non-revoked
        primary item from that family for the same concept.
        """

        summaries = {
            concept_id: {"families": 0, "delayed": 0, "operation_kinds": 0}
            for concept_id in concept_ids
        }
        if not concept_ids:
            return summaries
        with self.read() as connection:
            resolved_release = release_id or self.get_active_release_id(
                connection
            )
            rows = connection.execute(
                """SELECT evidence.concept_id, COUNT(*) AS families,
                          COUNT(DISTINCT CASE WHEN evidence.kind != 'unknown'
                                              THEN evidence.kind END) AS operation_kinds,
                          SUM(CASE WHEN evidence.delayed_unguided_correct_at IS NOT NULL
                                   THEN 1 ELSE 0 END) AS delayed
                   FROM learner_skill_families evidence
                   JOIN json_each(?) scope
                     ON scope.value = evidence.concept_id
                   WHERE evidence.learner_id = ?
                     AND EXISTS (
                         SELECT 1
                         FROM release_questions released
                         JOIN questions question
                           ON question.id = released.question_id
                         JOIN question_concepts mapping
                           ON mapping.question_id = question.id
                          AND mapping.role = 'primary'
                         WHERE released.release_id = ?
                           AND released.status IN (?, ?)
                           AND mapping.concept_id = evidence.concept_id
                           AND question.family_id = evidence.family_id
                           AND NOT EXISTS (
                               SELECT 1
                               FROM question_revocations revoked
                               WHERE revoked.question_id = question.id
                           )
                     )
                   GROUP BY evidence.concept_id""",
                (
                    json.dumps(
                        sorted(concept_ids),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    learner_id,
                    resolved_release,
                    QuestionStatus.APPROVED.value,
                    QuestionStatus.CALIBRATED.value,
                ),
            ).fetchall()
        for row in rows:
            summaries[row["concept_id"]] = {
                "families": int(row["families"] or 0),
                "delayed": int(row["delayed"] or 0),
                "operation_kinds": int(row["operation_kinds"] or 0),
            }
        return summaries

    def create_session(
        self,
        learner_id: str,
        root_concept_id: str | None,
        *,
        topic_id: str | None = None,
        exploration_mode: str = "off",
        mode: str = "learn",
        seed: int | None = None,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if not isinstance(learner_id, str) or not learner_id.strip():
            raise ValidationError("learner_id must be a non-blank string.")
        if root_concept_id is not None and (
            not isinstance(root_concept_id, str) or not root_concept_id.strip()
        ):
            raise ValidationError("root_concept_id must be null or a non-blank string.")
        if topic_id is not None and (
            not isinstance(topic_id, str) or not topic_id.strip()
        ):
            raise ValidationError("topic_id must be null or a non-blank string.")
        if root_concept_id is None and topic_id is None:
            raise ValidationError("A concept or curriculum topic is required.")
        if exploration_mode not in {"off", "adaptive"}:
            raise ValidationError("exploration_mode must be off or adaptive.")
        if topic_id is None and exploration_mode != "off":
            raise ValidationError(
                "Related-topic exploration requires a curriculum topic session."
            )
        if mode not in {"learn", "diagnose", "review"}:
            raise ValidationError("Mode must be learn, diagnose, or review.")
        if seed is not None and (
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or not (-(2**63) <= seed < 2**63)
        ):
            raise ValidationError("seed must be a signed 64-bit integer.")
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key) > 200
        ):
            raise ValidationError(
                "idempotency_key must be a non-blank string of at most 200 characters."
            )
        if now is not None and not isinstance(now, datetime):
            raise ValidationError("now must be a timezone-aware datetime.")
        session_time = now or datetime.now(timezone.utc)
        session_timestamp = to_timestamp(session_time)
        assert session_timestamp is not None
        phase = {
            "diagnose": SessionPhase.DIAGNOSE,
            "review": SessionPhase.REVIEW,
        }.get(mode, SessionPhase.LEARN)
        session_id = new_id("ses")
        resolved_seed = seed if seed is not None else secrets.randbelow(2**31)
        with self.transaction() as connection:
            if not connection.execute(
                "SELECT 1 FROM learners WHERE id = ?", (learner_id,)
            ).fetchone():
                raise NotFoundError(f"Unknown learner: {learner_id}")
            self.require_learner_evidence_safe(learner_id, connection)
            release_id = self.get_active_release_id(connection)
            requested_root_concept_id = root_concept_id
            if topic_id is not None:
                if not connection.execute(
                    """SELECT 1 FROM release_topics
                       WHERE release_id = ? AND topic_id = ?""",
                    (release_id, topic_id),
                ).fetchone():
                    raise NotFoundError(
                        f"Topic {topic_id} is not in active release {release_id}."
                    )
                owned_rows = connection.execute(
                    """WITH RECURSIVE descendants(topic_id) AS (
                           SELECT ?
                           UNION ALL
                           SELECT child.topic_id
                           FROM release_topics child
                           JOIN descendants parent
                             ON child.parent_topic_id = parent.topic_id
                           WHERE child.release_id = ?
                       )
                       SELECT membership.concept_id,
                              CASE WHEN membership.topic_id = ? THEN 0 ELSE 1 END AS rank,
                              membership.position
                       FROM descendants
                       JOIN release_topic_concepts membership
                         ON membership.release_id = ?
                        AND membership.topic_id = descendants.topic_id
                       ORDER BY rank, membership.position, membership.concept_id""",
                    (topic_id, release_id, topic_id, release_id),
                ).fetchall()
                if not owned_rows:
                    raise ConflictError(
                        f"Topic {topic_id} has no assessable concept scope."
                    )
                owned_concepts = {row["concept_id"] for row in owned_rows}
                if root_concept_id is None:
                    root_concept_id = owned_rows[0]["concept_id"]
                elif root_concept_id not in owned_concepts:
                    raise ValidationError(
                        f"Root concept {root_concept_id} is not owned by topic {topic_id}."
                    )
            assert root_concept_id is not None
            if not connection.execute(
                """SELECT 1 FROM release_concepts
                   WHERE release_id = ? AND concept_id = ?""",
                (release_id, root_concept_id),
            ).fetchone():
                raise NotFoundError(
                    f"Concept {root_concept_id} is not in active release {release_id}."
                )
            if idempotency_key:
                prior = connection.execute(
                    "SELECT * FROM events WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if prior:
                    if prior["event_type"] != "SessionStarted":
                        raise ConflictError("Idempotency key belongs to a different command.")
                    payload = json.loads(prior["payload_json"])
                    if (
                        payload.get(
                            "requested_root_concept_id",
                            payload.get("root_concept_id"),
                        ) != requested_root_concept_id
                        or payload.get("topic_id") != topic_id
                        or payload.get("exploration_mode", "off")
                        != exploration_mode
                        or payload.get("mode") != mode
                        or payload.get("requested_seed") != seed
                        or prior["learner_id"] != learner_id
                    ):
                        raise ConflictError("Idempotency key was reused with different session inputs.")
                    return self.get_session(payload["session_id"], connection)
            self.append_event(
                connection,
                stream_id=f"learner:{learner_id}",
                event_type="SessionStarted",
                payload={
                    "session_id": session_id,
                    "root_concept_id": root_concept_id,
                    "requested_root_concept_id": requested_root_concept_id,
                    "topic_id": topic_id,
                    "exploration_mode": exploration_mode,
                    "mode": mode,
                    "initial_phase": phase.value,
                    "requested_seed": seed,
                    "rng_seed": resolved_seed,
                    "corpus_release_id": release_id,
                },
                metadata={"corpus_release_id": release_id},
                learner_id=learner_id,
                session_id=session_id,
                idempotency_key=idempotency_key,
                occurred_at=session_time,
            )
            connection.execute(
                """INSERT INTO sessions(
                       id, learner_id, root_concept_id, corpus_release_id,
                       mode, phase, rng_seed, created_at, updated_at,
                       topic_id, exploration_mode
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    learner_id,
                    root_concept_id,
                    release_id,
                    mode,
                    phase.value,
                    resolved_seed,
                    session_timestamp,
                    session_timestamp,
                    topic_id,
                    exploration_mode,
                ),
            )
            return self.get_session(session_id, connection)

    def end_session(
        self,
        session_id: str,
        *,
        status: str = "completed",
        reason: str | None = None,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if status not in {"completed", "abandoned"}:
            raise ValidationError("Session status must be completed or abandoned.")
        if reason is not None and (
            not isinstance(reason, str) or not reason.strip() or len(reason) > 500
        ):
            raise ValidationError(
                "Session end reason must be a non-blank string of at most 500 characters."
            )
        if idempotency_key is not None and (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key) > 200
        ):
            raise ValidationError(
                "idempotency_key must be a non-blank string of at most 200 characters."
            )
        if now is not None and not isinstance(now, datetime):
            raise ValidationError("now must be a timezone-aware datetime.")
        close_time = now or datetime.now(timezone.utc)
        if close_time.tzinfo is None or close_time.utcoffset() is None:
            raise ValidationError("now must be a timezone-aware datetime.")
        close_time = close_time.astimezone(timezone.utc)
        with self.transaction() as connection:
            session = self.get_session(session_id, connection)
            payload = {
                "session_id": session_id,
                "status": status,
                "reason": reason,
            }
            if idempotency_key:
                prior = connection.execute(
                    "SELECT * FROM events WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if prior:
                    if (
                        prior["event_type"] != "SessionEnded"
                        or prior["session_id"] != session_id
                        or json.loads(prior["payload_json"]) != payload
                    ):
                        raise ConflictError(
                            "Idempotency key was reused with different session-end inputs."
                        )
                    return session
            if session["status"] == status:
                return session
            if session["status"] != "active":
                raise ConflictError(
                    f"Session is already {session['status']} and cannot become {status}."
                )
            open_performance_attempt = connection.execute(
                """SELECT attempt.id
                   FROM performance_attempts attempt
                   WHERE attempt.session_id = ?
                     AND NOT EXISTS (
                         SELECT 1 FROM performance_actions terminal
                         WHERE terminal.attempt_id = attempt.id
                           AND terminal.action_type IN ('submitted', 'abandoned')
                     )
                   ORDER BY attempt.started_at, attempt.id LIMIT 1""",
                (session_id,),
            ).fetchone()
            if open_performance_attempt is not None:
                raise ConflictError(
                    "Session has an active performance task "
                    f"{open_performance_attempt['id']}; submit or abandon it "
                    "before ending the session."
                )
            try:
                session_started_at = datetime.fromisoformat(
                    session["created_at"]
                )
                if (
                    session_started_at.tzinfo is None
                    or session_started_at.utcoffset() is None
                ):
                    raise ValueError("timezone-naive timestamp")
                session_started_at = session_started_at.astimezone(
                    timezone.utc
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise ConflictError(
                    "Session has an invalid start timestamp; run integrity "
                    "verification before ending it."
                ) from exc
            if close_time < session_started_at:
                raise ValidationError(
                    "A session cannot end before it started."
                )
            for prior_event in connection.execute(
                """SELECT event_id, event_type, occurred_at
                   FROM events WHERE session_id = ?
                   ORDER BY stream_version, event_id""",
                (session_id,),
            ).fetchall():
                try:
                    prior_time = datetime.fromisoformat(
                        prior_event["occurred_at"]
                    )
                    if (
                        prior_time.tzinfo is None
                        or prior_time.utcoffset() is None
                    ):
                        raise ValueError("timezone-naive timestamp")
                    prior_time = prior_time.astimezone(timezone.utc)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ConflictError(
                        f"Session event {prior_event['event_id']} has an "
                        "invalid occurrence timestamp; run integrity "
                        "verification before ending the session."
                    ) from exc
                if close_time < prior_time:
                    raise ValidationError(
                        "A session cannot end before its latest recorded "
                        f"event ({prior_event['event_type']})."
                    )
            invalidation_reason = f"session_{status}"
            pending_decisions = connection.execute(
                """SELECT * FROM decisions
                   WHERE session_id = ? AND consumed_at IS NULL
                     AND invalidated_at IS NULL
                   ORDER BY created_at, id""",
                (session_id,),
            ).fetchall()
            learner = connection.execute(
                "SELECT revision FROM learners WHERE id = ?",
                (session["learner_id"],),
            ).fetchone()
            if learner is None:
                raise ConflictError("Session learner no longer exists.")

            # A selected question is an auditable, answerable promise. Ending a
            # session closes every such promise explicitly; otherwise the row
            # remains pending even though no answer may legally be submitted.
            # All projections and boundaries are written in this transaction,
            # before the session leaves its active revision.
            for decision in pending_decisions:
                if learner["revision"] < decision["learner_revision"]:
                    raise ConflictError(
                        "Pending decision is ahead of the learner projection."
                    )
                selection_events = connection.execute(
                    """SELECT metadata_json FROM events
                       WHERE event_type = 'QuestionSelected'
                         AND session_id = ?
                         AND json_extract(payload_json, '$.decision_id') = ?
                       ORDER BY stream_version""",
                    (session_id, decision["id"]),
                ).fetchall()
                if len(selection_events) != 1:
                    raise ConflictError(
                        "Pending decision lacks a unique QuestionSelected boundary."
                    )
                try:
                    selection_metadata = json.loads(
                        selection_events[0]["metadata_json"]
                    )
                except (TypeError, ValueError) as exc:
                    raise ConflictError(
                        "Pending decision has invalid selection metadata."
                    ) from exc
                if (
                    type(selection_metadata) is not dict
                    or selection_metadata.get("policy_version")
                    != decision["policy_version"]
                    or selection_metadata.get("corpus_release_id")
                    != decision["corpus_release_id"]
                    or not isinstance(
                        selection_metadata.get("learner_model_version"), str
                    )
                    or not selection_metadata["learner_model_version"]
                ):
                    raise ConflictError(
                        "Pending decision selection metadata is inconsistent."
                    )
                invalidated = connection.execute(
                    """UPDATE decisions
                       SET invalidated_at = ?, invalidation_reason = ?
                       WHERE id = ? AND session_id = ?
                         AND consumed_at IS NULL AND invalidated_at IS NULL""",
                    (
                        to_timestamp(close_time),
                        invalidation_reason,
                        decision["id"],
                        session_id,
                    ),
                )
                if invalidated.rowcount != 1:
                    raise ConflictError(
                        "Pending decision changed while the session was ending."
                    )
                self.append_event(
                    connection,
                    stream_id=f"learner:{session['learner_id']}",
                    event_type="DecisionInvalidated",
                    payload={
                        "decision_id": decision["id"],
                        "reason": invalidation_reason,
                        "selection_learner_revision": decision[
                            "learner_revision"
                        ],
                        "current_learner_revision": learner["revision"],
                    },
                    metadata={
                        "policy_version": decision["policy_version"],
                        "learner_model_version": selection_metadata[
                            "learner_model_version"
                        ],
                        "corpus_release_id": decision["corpus_release_id"],
                    },
                    learner_id=session["learner_id"],
                    session_id=session_id,
                    causation_id=decision["id"],
                    occurred_at=close_time,
                )

            still_pending = connection.execute(
                """SELECT 1 FROM decisions
                   WHERE session_id = ? AND consumed_at IS NULL
                     AND invalidated_at IS NULL LIMIT 1""",
                (session_id,),
            ).fetchone()
            if still_pending is not None:
                raise ConflictError(
                    "Session still has pending decisions after invalidation."
                )

            self.append_event(
                connection,
                stream_id=f"learner:{session['learner_id']}",
                event_type="SessionEnded",
                payload=payload,
                metadata={"corpus_release_id": session["corpus_release_id"]},
                learner_id=session["learner_id"],
                session_id=session_id,
                idempotency_key=idempotency_key,
                occurred_at=close_time,
            )
            updated = connection.execute(
                """UPDATE sessions SET status = ?, revision = revision + 1, updated_at = ?
                   WHERE id = ? AND status = 'active' AND revision = ?""",
                (
                    status,
                    to_timestamp(close_time),
                    session_id,
                    session["revision"],
                ),
            )
            if updated.rowcount != 1:
                raise ConflictError("Session changed while it was being ended.")
            return self.get_session(session_id, connection)

    def get_session(
        self, session_id: str, connection: sqlite3.Connection | None = None
    ) -> dict[str, Any]:
        owns_connection = connection is None
        conn = connection or self.connect()
        try:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                raise NotFoundError(f"Unknown session: {session_id}")
            result = dict(row)
            result["recent_families"] = json.loads(result.pop("recent_families_json"))
            result["remediation_path"] = json.loads(
                result.pop("remediation_path_json")
            )
            return result
        finally:
            if owns_connection:
                conn.close()

    def validate_release_focus_tuple(
        self,
        release_id: str,
        concept_id: str | None,
        misconception_id: str | None,
        objective_id: str | None,
        *,
        connection: sqlite3.Connection | None = None,
        label: str = "adaptive focus",
    ) -> None:
        """Fail closed when adaptive focus escapes its immutable release.

        Fine-grained objective focus always uses the objective's canonical
        owner concept. A named misconception may be attached only when the
        pinned release maps that exact misconception to that objective. Legacy
        focus remains valid only when the misconception belongs to the focused
        concept.
        """

        owns_connection = connection is None
        conn = connection or self.connect()
        try:
            if concept_id is None:
                if misconception_id is not None or objective_id is not None:
                    raise ValidationError(
                        f"{label} has a misconception or objective without a concept."
                    )
                return
            if not conn.execute(
                """SELECT 1 FROM release_concepts
                   WHERE release_id = ? AND concept_id = ?""",
                (release_id, concept_id),
            ).fetchone():
                raise ValidationError(
                    f"{label} concept {concept_id} is not in release {release_id}."
                )

            misconception_owner: str | None = None
            if misconception_id is not None:
                misconception = conn.execute(
                    """SELECT misconception.concept_id
                       FROM release_misconceptions membership
                       JOIN misconceptions misconception
                         ON misconception.id = membership.misconception_id
                       WHERE membership.release_id = ?
                         AND membership.misconception_id = ?""",
                    (release_id, misconception_id),
                ).fetchone()
                if misconception is None:
                    raise ValidationError(
                        f"{label} misconception {misconception_id} is not in "
                        f"release {release_id}."
                    )
                misconception_owner = misconception["concept_id"]

            if objective_id is None:
                if (
                    misconception_owner is not None
                    and misconception_owner != concept_id
                ):
                    raise ValidationError(
                        f"{label} misconception owner does not match its concept."
                    )
                return

            objective = conn.execute(
                """SELECT objective.primary_concept_id,
                          objective.supporting_concept_ids_json
                   FROM release_learning_objectives membership
                   JOIN learning_objectives objective
                     ON objective.id = membership.objective_id
                   WHERE membership.release_id = ?
                     AND membership.objective_id = ?""",
                (release_id, objective_id),
            ).fetchone()
            if objective is None:
                raise ValidationError(
                    f"{label} objective {objective_id} is not in release {release_id}."
                )
            if concept_id != objective["primary_concept_id"]:
                raise ValidationError(
                    f"{label} concept is not the objective's canonical owner."
                )
            try:
                supporting = json.loads(
                    objective["supporting_concept_ids_json"]
                )
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"{label} objective has invalid supporting concepts."
                ) from exc
            if not isinstance(supporting, list) or not all(
                isinstance(value, str) for value in supporting
            ):
                raise ValidationError(
                    f"{label} objective has invalid supporting concepts."
                )
            if (
                misconception_owner is not None
                and misconception_owner
                not in {objective["primary_concept_id"], *supporting}
            ):
                raise ValidationError(
                    f"{label} misconception is outside the objective's concept scope."
                )
            if misconception_id is not None and not conn.execute(
                """SELECT 1
                   FROM release_option_objectives mapping
                   JOIN options option
                     ON option.question_id = mapping.question_id
                    AND option.option_id = mapping.option_id
                   WHERE mapping.release_id = ?
                     AND mapping.objective_id = ?
                     AND option.misconception_id = ?
                   LIMIT 1""",
                (release_id, objective_id, misconception_id),
            ).fetchone():
                raise ValidationError(
                    f"{label} misconception is not mapped to its objective in the "
                    "pinned release."
                )
        finally:
            if owns_connection:
                conn.close()

    def validate_session_focus(
        self,
        session: dict[str, Any] | sqlite3.Row,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        """Validate current and suspended focus frames against one release."""

        self.validate_release_focus_tuple(
            session["corpus_release_id"],
            session["focus_concept_id"],
            session["focus_misconception_id"],
            session["focus_objective_id"],
            connection=connection,
            label=f"session {session['id']} focus",
        )
        path = (
            session["remediation_path"]
            if isinstance(session, dict)
            else json.loads(session["remediation_path_json"])
        )
        if not isinstance(path, list):
            raise ValidationError(
                f"session {session['id']} remediation path must be a list."
            )
        for index, frame in enumerate(path):
            if not isinstance(frame, dict):
                raise ValidationError(
                    f"session {session['id']} remediation path frame {index} "
                    "must be an object."
                )
            expected = {"concept_id", "misconception_id"}
            if "objective_id" in frame:
                expected.add("objective_id")
            if set(frame) != expected:
                raise ValidationError(
                    f"session {session['id']} remediation path frame {index} "
                    "has invalid fields."
                )
            self.validate_release_focus_tuple(
                session["corpus_release_id"],
                frame.get("concept_id"),
                frame.get("misconception_id"),
                frame.get("objective_id"),
                connection=connection,
                label=(
                    f"session {session['id']} remediation path frame {index}"
                ),
            )

    def pending_presentation(self, session_id: str) -> Presentation | None:
        with self.read() as connection:
            row = connection.execute(
                """SELECT * FROM decisions WHERE session_id = ?
                   AND consumed_at IS NULL AND invalidated_at IS NULL
                   AND NOT EXISTS (
                       SELECT 1 FROM question_revocations revoked
                       WHERE revoked.question_id = decisions.question_id
                   )
                   ORDER BY created_at DESC LIMIT 1""",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            self.validate_release_focus_tuple(
                row["corpus_release_id"],
                row["focus_concept_id"],
                row["focus_misconception_id"],
                row["focus_objective_id"],
                connection=connection,
                label=f"decision {row['id']} focus",
            )
            questions = self._questions_by_ids(
                connection,
                [row["question_id"]],
                release_id=row["corpus_release_id"],
            )
            if not questions:
                raise NotFoundError(
                    f"Decision {row['id']} references a question outside its corpus release."
                )
            question = questions[0]
            if not question_runtime_activation_safe(question):
                return None
            terms = json.loads(row["selected_score_json"])
            score = CandidateScore(question_id=question.id, **terms)
            return Presentation(
                decision_id=row["id"],
                session_id=session_id,
                question=question,
                option_order=tuple(json.loads(row["option_order_json"])),
                phase=SessionPhase(row["phase"]),
                score=score,
                propensity=row["propensity"],
                rationale=row["rationale"],
                pedagogical_role=row["pedagogical_role"],
            )

    def recent_decisions(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.read() as connection:
            rows = connection.execute(
                "SELECT * FROM decisions WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["selected_score"] = json.loads(item.pop("selected_score_json"))
            item["top_candidates"] = json.loads(item.pop("top_candidates_json"))
            item["option_order"] = json.loads(item.pop("option_order_json"))
            result.append(item)
        return result

    def verify_integrity(self, stream_id: str | None = None) -> dict[str, Any]:
        """Verify storage, hash-chain, stream-head, and projection invariants."""
        errors: list[str] = []
        payload_cache: dict[str, dict[str, Any] | None] = {}
        metadata_cache: dict[str, dict[str, Any] | None] = {}

        def reject_non_finite_json_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON constant {value}")

        def finite_json_float(value: str) -> float:
            parsed = float(value)
            if not isfinite(parsed):
                raise ValueError(f"non-finite JSON number {value}")
            return parsed

        def reject_duplicate_json_object(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError(f"duplicate JSON object key {key!r}")
                value[key] = item
            return value

        def event_object(
            event: sqlite3.Row, column: str, cache: dict[str, dict[str, Any] | None]
        ) -> dict[str, Any] | None:
            event_id = event["event_id"]
            if event_id in cache:
                return cache[event_id]
            label = "payload" if column == "payload_json" else "metadata"
            try:
                value = json.loads(
                    event[column],
                    parse_constant=reject_non_finite_json_constant,
                    parse_float=finite_json_float,
                    object_pairs_hook=reject_duplicate_json_object,
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"event {event_id}: invalid {label} JSON ({exc})")
                cache[event_id] = None
                return None
            if not isinstance(value, dict):
                errors.append(f"event {event_id}: {label} JSON is not an object")
                cache[event_id] = None
                return None
            cache[event_id] = value
            return value

        def json_value(
            raw: str | None,
            label: str,
            expected_type: type,
        ) -> Any | None:
            if raw is None:
                return None
            try:
                value = json.loads(
                    raw,
                    parse_constant=reject_non_finite_json_constant,
                    parse_float=finite_json_float,
                    object_pairs_hook=reject_duplicate_json_object,
                )
            except (TypeError, ValueError) as exc:
                errors.append(f"{label}: invalid JSON ({exc})")
                return None
            if not isinstance(value, expected_type):
                errors.append(f"{label}: expected {expected_type.__name__} JSON")
                return None
            return value

        def compare_payload(
            actual: dict[str, Any],
            expected: dict[str, Any],
            label: str,
            *,
            exact: bool = False,
        ) -> None:
            if exact:
                unexpected = sorted(set(actual) - set(expected))
                if unexpected:
                    errors.append(
                        f"{label}: unexpected fields {unexpected!r}"
                    )
            for field, expected_value in expected.items():
                if field not in actual:
                    errors.append(f"{label}: missing {field}")
                elif not same_json_value(actual[field], expected_value):
                    errors.append(f"{label}: {field} mismatch")

        def require_exact_fields(
            actual: dict[str, Any], expected: frozenset[str], label: str
        ) -> None:
            missing = sorted(expected - set(actual))
            unexpected = sorted(set(actual) - expected)
            if missing:
                errors.append(f"{label}: missing fields {missing!r}")
            if unexpected:
                errors.append(f"{label}: unexpected fields {unexpected!r}")

        def aware_timestamp(raw: Any, label: str) -> datetime | None:
            """Parse one integrity timestamp without allowing naive arithmetic."""

            try:
                parsed = datetime.fromisoformat(raw)
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    errors.append(f"{label}: timestamp is timezone-naive")
                    return None
                return parsed.astimezone(timezone.utc)
            except (TypeError, ValueError, OverflowError) as exc:
                errors.append(f"{label}: invalid timestamp ({exc})")
                return None

        with self.read() as connection:
            quick_check = [row[0] for row in connection.execute("PRAGMA quick_check")]
            if quick_check != ["ok"]:
                errors.extend(f"SQLite quick_check: {message}" for message in quick_check)

            foreign_key_failures = [
                dict(row) for row in connection.execute("PRAGMA foreign_key_check")
            ]
            if foreign_key_failures:
                errors.append(f"{len(foreign_key_failures)} foreign-key violations")

            # A v6 objective parent is only a cache of the exact posterior.
            # Decode every child and recompute every duplicated metric instead
            # of allowing internally inconsistent rows to pass a hash check.
            objective_projection_rows = connection.execute(
                OBJECTIVE_STATE_WITH_GRID_SELECT
                + " ORDER BY state.learner_id, state.objective_id"
            ).fetchall()
            for objective_row in objective_projection_rows:
                try:
                    self._objective_state_from_joined_row(objective_row)
                except (ValidationError, TypeError, ValueError) as exc:
                    errors.append(str(exc))

            orphan_grid_count = connection.execute(
                """SELECT COUNT(*) AS n
                   FROM objective_grid_states grid
                   LEFT JOIN objective_states state
                     ON state.learner_id = grid.learner_id
                    AND state.objective_id = grid.objective_id
                   WHERE state.learner_id IS NULL"""
            ).fetchone()["n"]
            if orphan_grid_count:
                errors.append(
                    f"{orphan_grid_count} exact objective posterior rows lack "
                    "parent objective states"
                )

            # Action telemetry is observational, but it is still part of the
            # learner's immutable semantic history.  Verify both halves of the
            # projection instead of trusting either the row or event alone.
            artifact_rows = connection.execute(
                "SELECT * FROM learning_artifacts ORDER BY id"
            ).fetchall()
            artifacts = {row["id"]: row for row in artifact_rows}
            referenced_artifact_ids: set[str] = set()
            for artifact in artifact_rows:
                prefix = f"learning artifact {artifact['id']}"
                digest = artifact["sha256"]
                if not (
                    type(digest) is str
                    and len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest)
                ):
                    errors.append(f"{prefix}: invalid SHA-256 digest")
                if artifact["id"] != f"art_{digest}":
                    errors.append(f"{prefix}: ID is not content addressed")
                if (
                    type(artifact["size_bytes"]) is not int
                    or not 0 <= artifact["size_bytes"] <= 1_073_741_824
                ):
                    errors.append(f"{prefix}: invalid size")
                media_type = artifact["media_type"]
                if not (
                    type(media_type) is str
                    and media_type.strip() == media_type
                    and 1 <= len(media_type) <= 127
                    and "/" in media_type
                    and not any(character.isspace() for character in media_type)
                ):
                    errors.append(f"{prefix}: invalid media type")
                aware_timestamp(artifact["created_at"], f"{prefix} creation time")

            action_rows = connection.execute(
                """SELECT action.*,
                          decision.created_at AS decision_created_at,
                          decision.corpus_release_id AS decision_release_id,
                          decision.question_id AS decision_question_id,
                          decision.invalidated_at AS decision_invalidated_at,
                          attempt.answered_at,
                          response_event.event_id AS response_event_id,
                          response_event.stream_id AS response_stream_id,
                          response_event.stream_version AS response_stream_version,
                          artifact.sha256 AS artifact_sha256,
                          artifact.size_bytes AS artifact_size_bytes,
                          artifact.media_type AS artifact_media_type
                   FROM learning_actions action
                   JOIN decisions decision ON decision.id = action.decision_id
                   LEFT JOIN attempts attempt ON attempt.decision_id = action.decision_id
                   LEFT JOIN events response_event
                     ON response_event.event_id = attempt.event_id
                   LEFT JOIN learning_artifacts artifact
                     ON artifact.id = action.artifact_id
                   ORDER BY action.decision_id, action.sequence"""
            ).fetchall()
            action_events = {
                row["event_id"]: row
                for row in connection.execute(
                    """SELECT * FROM events
                       WHERE event_type='LearnerActionRecorded'"""
                ).fetchall()
            }
            selection_events_by_decision: dict[str, list[sqlite3.Row]] = {}
            invalidation_events_for_actions: dict[str, list[sqlite3.Row]] = {}
            session_boundaries: dict[str, dict[str, list[sqlite3.Row]]] = {}
            projection_versions_by_stream: dict[str, list[int]] = {}
            for boundary in connection.execute(
                """SELECT * FROM events
                   WHERE event_type IN (
                       'QuestionSelected', 'DecisionInvalidated',
                       'SessionStarted', 'SessionEnded',
                       'LearnerProjectionAdvanced'
                   )
                   ORDER BY stream_id, stream_version"""
            ).fetchall():
                if boundary["event_type"] == "QuestionSelected":
                    boundary_payload = event_object(
                        boundary, "payload_json", payload_cache
                    )
                    decision_id = (
                        boundary_payload.get("decision_id")
                        if boundary_payload is not None
                        else None
                    )
                    if isinstance(decision_id, str) and decision_id:
                        selection_events_by_decision.setdefault(
                            decision_id, []
                        ).append(boundary)
                elif boundary["event_type"] == "DecisionInvalidated":
                    boundary_payload = event_object(
                        boundary, "payload_json", payload_cache
                    )
                    decision_id = (
                        boundary_payload.get("decision_id")
                        if boundary_payload is not None
                        else None
                    )
                    if isinstance(decision_id, str) and decision_id:
                        invalidation_events_for_actions.setdefault(
                            decision_id, []
                        ).append(boundary)
                elif boundary["event_type"] in {"SessionStarted", "SessionEnded"}:
                    boundary_session_id = boundary["session_id"]
                    if isinstance(boundary_session_id, str) and boundary_session_id:
                        session_boundaries.setdefault(
                            boundary_session_id, {"started": [], "ended": []}
                        )[
                            "started"
                            if boundary["event_type"] == "SessionStarted"
                            else "ended"
                        ].append(boundary)
                else:
                    projection_versions_by_stream.setdefault(
                        boundary["stream_id"], []
                    ).append(boundary["stream_version"])

            # A session row is a projection of two immutable lifecycle
            # boundaries.  Validate that projection directly even when the
            # session has no questions or productive-skill actions: relying on
            # action checks alone would let an impossible empty session pass.
            session_rows = {
                row["id"]: row
                for row in connection.execute(
                    """SELECT id, learner_id, root_concept_id,
                              corpus_release_id, mode, status, revision,
                              created_at, updated_at, topic_id,
                              exploration_mode
                       FROM sessions ORDER BY id"""
                ).fetchall()
            }
            session_events: dict[str, list[sqlite3.Row]] = {}
            for event in connection.execute(
                """SELECT * FROM events WHERE session_id IS NOT NULL
                   ORDER BY stream_id, stream_version"""
            ).fetchall():
                session_events.setdefault(event["session_id"], []).append(
                    event
                )

            for session_id, session in session_rows.items():
                prefix = f"session {session_id}"
                status = session["status"]
                if status not in {"active", "completed", "abandoned"}:
                    errors.append(f"{prefix}: invalid status {status!r}")
                created_at = aware_timestamp(
                    session["created_at"], f"{prefix} creation time"
                )
                updated_at = aware_timestamp(
                    session["updated_at"], f"{prefix} update time"
                )
                if (
                    created_at is not None
                    and updated_at is not None
                    and updated_at < created_at
                ):
                    errors.append(f"{prefix}: updated_at precedes created_at")

                events_for_session = session_events.get(session_id, [])
                expected_stream = f"learner:{session['learner_id']}"
                for event in events_for_session:
                    if (
                        event["stream_id"] != expected_stream
                        or event["learner_id"] != session["learner_id"]
                        or event["session_id"] != session_id
                    ):
                        errors.append(
                            f"{prefix}: event {event['event_id']} envelope "
                            "does not match session"
                        )

                bounds = session_boundaries.get(
                    session_id, {"started": [], "ended": []}
                )
                started_events = bounds["started"]
                ended_events = bounds["ended"]
                started_event = (
                    started_events[0] if len(started_events) == 1 else None
                )
                ended_event = (
                    ended_events[0] if len(ended_events) == 1 else None
                )
                if len(started_events) != 1:
                    errors.append(
                        f"{prefix}: expected one SessionStarted boundary, "
                        f"found {len(started_events)}"
                    )
                if len(ended_events) > 1:
                    errors.append(
                        f"{prefix}: expected at most one SessionEnded "
                        f"boundary, found {len(ended_events)}"
                    )
                if status == "active" and ended_events:
                    errors.append(
                        f"{prefix}: active status has a SessionEnded boundary"
                    )
                elif status in {"completed", "abandoned"} and len(
                    ended_events
                ) != 1:
                    errors.append(
                        f"{prefix}: {status} status requires one "
                        "SessionEnded boundary"
                    )

                started_at: datetime | None = None
                if started_event is not None:
                    started_payload = event_object(
                        started_event, "payload_json", payload_cache
                    )
                    started_metadata = event_object(
                        started_event, "metadata_json", metadata_cache
                    )
                    started_at = aware_timestamp(
                        started_event["occurred_at"],
                        f"{prefix} SessionStarted occurrence time",
                    )
                    aware_timestamp(
                        started_event["recorded_at"],
                        f"{prefix} SessionStarted recording time",
                    )
                    if (
                        started_event["schema_version"] != 1
                        or started_event["stream_id"] != expected_stream
                        or started_event["learner_id"]
                        != session["learner_id"]
                        or started_event["session_id"] != session_id
                        or started_event["causation_id"] is not None
                    ):
                        errors.append(
                            f"{prefix}: SessionStarted event envelope mismatch"
                        )
                    if started_payload is not None:
                        compare_payload(
                            started_payload,
                            {
                                "session_id": session_id,
                                "root_concept_id": session[
                                    "root_concept_id"
                                ],
                                "corpus_release_id": session[
                                    "corpus_release_id"
                                ],
                                "mode": session["mode"],
                            },
                            f"{prefix} SessionStarted payload",
                        )
                    if (
                        started_metadata is not None
                        and started_metadata.get("corpus_release_id")
                        != session["corpus_release_id"]
                    ):
                        errors.append(
                            f"{prefix}: SessionStarted metadata corpus "
                            "release mismatch"
                        )
                    if (
                        created_at is not None
                        and started_at is not None
                        and session["created_at"]
                        != started_event["occurred_at"]
                    ):
                        errors.append(
                            f"{prefix}: created_at does not match "
                            "SessionStarted occurrence"
                        )
                    if events_for_session and (
                        started_event["stream_version"]
                        != min(
                            event["stream_version"]
                            for event in events_for_session
                        )
                    ):
                        errors.append(
                            f"{prefix}: SessionStarted is not the first "
                            "session event"
                        )

                ended_at: datetime | None = None
                if ended_event is not None:
                    ended_payload = event_object(
                        ended_event, "payload_json", payload_cache
                    )
                    ended_metadata = event_object(
                        ended_event, "metadata_json", metadata_cache
                    )
                    ended_at = aware_timestamp(
                        ended_event["occurred_at"],
                        f"{prefix} SessionEnded occurrence time",
                    )
                    aware_timestamp(
                        ended_event["recorded_at"],
                        f"{prefix} SessionEnded recording time",
                    )
                    if (
                        ended_event["schema_version"] != 1
                        or ended_event["stream_id"] != expected_stream
                        or ended_event["learner_id"] != session["learner_id"]
                        or ended_event["session_id"] != session_id
                        or ended_event["causation_id"] is not None
                    ):
                        errors.append(
                            f"{prefix}: SessionEnded event envelope mismatch"
                        )
                    if ended_payload is not None:
                        require_exact_fields(
                            ended_payload,
                            frozenset({"session_id", "status", "reason"}),
                            f"{prefix} SessionEnded payload",
                        )
                        compare_payload(
                            ended_payload,
                            {
                                "session_id": session_id,
                                "status": status,
                            },
                            f"{prefix} SessionEnded payload",
                        )
                        end_reason = ended_payload.get("reason")
                        if end_reason is not None and (
                            not isinstance(end_reason, str)
                            or not end_reason.strip()
                            or len(end_reason) > 500
                        ):
                            errors.append(
                                f"{prefix}: SessionEnded reason is invalid"
                            )
                    if ended_metadata is not None:
                        require_exact_fields(
                            ended_metadata,
                            frozenset({"corpus_release_id"}),
                            f"{prefix} SessionEnded metadata",
                        )
                        if (
                            ended_metadata.get("corpus_release_id")
                            != session["corpus_release_id"]
                        ):
                            errors.append(
                                f"{prefix}: SessionEnded metadata corpus "
                                "release mismatch"
                            )
                    if (
                        updated_at is not None
                        and ended_at is not None
                        and session["updated_at"]
                        != ended_event["occurred_at"]
                    ):
                        errors.append(
                            f"{prefix}: updated_at does not match "
                            "SessionEnded occurrence"
                        )
                    if (
                        started_at is not None
                        and ended_at is not None
                        and ended_at < started_at
                    ):
                        errors.append(
                            f"{prefix}: SessionEnded occurred before "
                            "SessionStarted"
                        )
                    if events_for_session and (
                        ended_event["stream_version"]
                        != max(
                            event["stream_version"]
                            for event in events_for_session
                        )
                    ):
                        errors.append(
                            f"{prefix}: SessionEnded is not the final "
                            "session event"
                        )

                for event in events_for_session:
                    event_at = aware_timestamp(
                        event["occurred_at"],
                        f"{prefix} event {event['event_id']} occurrence time",
                    )
                    if (
                        started_at is not None
                        and event_at is not None
                        and event_at < started_at
                    ):
                        errors.append(
                            f"{prefix}: event {event['event_id']} occurred "
                            "before SessionStarted"
                        )
                    if (
                        ended_at is not None
                        and event_at is not None
                        and event_at > ended_at
                    ):
                        errors.append(
                            f"{prefix}: event {event['event_id']} occurred "
                            "after SessionEnded"
                        )

            for unknown_session_id in sorted(
                set(session_events) - set(session_rows)
            ):
                errors.append(
                    f"session {unknown_session_id}: events have no session "
                    "projection"
                )
            revocation_boundaries = {
                row["question_id"]: row
                for row in connection.execute(
                    """SELECT revocation.question_id,
                              revocation.revoked_at,
                              event.event_id AS revocation_event_id,
                              event.occurred_at AS revocation_occurred_at,
                              event.recorded_at AS revocation_recorded_at
                       FROM question_revocations revocation
                       JOIN events event ON event.event_id = revocation.event_id"""
                ).fetchall()
            }
            projected_action_events: set[str] = set()
            expected_action_sequence: dict[str, int] = {}
            latest_action_time: dict[str, datetime] = {}
            hint_actions_by_decision: dict[str, int] = {}
            typed_actions_by_decision: dict[str, list[LearningAction]] = {}
            artifact_digest_fields = {
                ActionKind.ANSWER_REVISED.value: "answer_digest",
                ActionKind.ARTIFACT_CHECKPOINT.value: "artifact_digest",
                ActionKind.EXPLANATION_CHECKPOINT.value: "explanation_digest",
                ActionKind.CHECK_RUN.value: "result_digest",
                ActionKind.SUBMITTED.value: "submission_digest",
                ActionKind.FEEDBACK_SHOWN.value: "feedback_digest",
            }
            for action in action_rows:
                prefix = f"learning action {action['id']}"
                expected_sequence = expected_action_sequence.get(
                    action["decision_id"], 1
                )
                if action["sequence"] != expected_sequence:
                    errors.append(
                        f"{prefix}: expected sequence {expected_sequence}, "
                        f"found {action['sequence']}"
                    )
                expected_action_sequence[action["decision_id"]] = (
                    action["sequence"] + 1
                    if type(action["sequence"]) is int
                    else expected_sequence + 1
                )
                if action["artifact_id"] is not None:
                    referenced_artifact_ids.add(action["artifact_id"])
                    if action["artifact_id"] not in artifacts:
                        errors.append(f"{prefix}: missing artifact projection")
                artifact_payload = (
                    {
                        "sha256": action["artifact_sha256"],
                        "size_bytes": action["artifact_size_bytes"],
                        "media_type": action["artifact_media_type"],
                    }
                    if action["artifact_sha256"] is not None
                    else None
                )
                payload = json_value(
                    action["payload_json"], f"{prefix} payload", dict
                )
                if action["artifact_id"] is not None and payload is not None:
                    digest_field = artifact_digest_fields.get(action["action_type"])
                    if (
                        digest_field is None
                        or payload.get(digest_field) != action["artifact_sha256"]
                    ):
                        errors.append(f"{prefix}: artifact digest does not match payload")
                if (
                    action["action_type"] == ActionKind.FEEDBACK_SHOWN.value
                    and action["stage"] != ActionPhase.POST_FEEDBACK.value
                ):
                    errors.append(
                        f"{prefix}: feedback_shown action is not post_feedback"
                    )
                selected_at = aware_timestamp(
                    action["decision_created_at"], f"{prefix} selection time"
                )
                occurred_at = aware_timestamp(
                    action["occurred_at"], f"{prefix} occurrence time"
                )
                recorded_at = aware_timestamp(
                    action["recorded_at"], f"{prefix} recording time"
                )
                answered_at = (
                    aware_timestamp(
                        action["answered_at"], f"{prefix} linked answer time"
                    )
                    if action["answered_at"] is not None
                    else None
                )
                if selected_at is not None and occurred_at is not None:
                    prior_time = latest_action_time.get(action["decision_id"])
                    if prior_time is not None and occurred_at < prior_time:
                        errors.append(f"{prefix}: occurrence time is not monotonic")
                    latest_action_time[action["decision_id"]] = occurred_at
                    if occurred_at < selected_at:
                        errors.append(f"{prefix}: occurred before selection")
                    if action["stage"] in {"unassisted", "assisted"}:
                        if answered_at is not None and occurred_at > answered_at:
                            errors.append(f"{prefix}: pre-response action follows answer")
                    elif action["stage"] == "post_feedback":
                        if action["answered_at"] is None:
                            errors.append(
                                f"{prefix}: post-feedback action has no answer"
                            )
                        elif answered_at is not None and occurred_at < answered_at:
                            errors.append(
                                f"{prefix}: post-feedback action precedes answer"
                            )
                if payload is not None:
                    try:
                        kind = ActionKind(action["action_type"])
                        phase = ActionPhase(action["stage"])
                        typed_action = LearningAction(
                            id=action["id"],
                            trace_id=action["decision_id"],
                            sequence=action["sequence"],
                            kind=kind,
                            phase=phase,
                            payload=payload,
                            elapsed_ms=(
                                max(
                                    0,
                                    int(
                                        (occurred_at - selected_at).total_seconds()
                                        * 1000
                                    ),
                                )
                                if occurred_at is not None and selected_at is not None
                                else None
                            ),
                        )
                        typed_actions_by_decision.setdefault(
                            action["decision_id"], []
                        ).append(typed_action)
                        if action["payload_json"] != canonical_json(payload):
                            errors.append(f"{prefix}: payload is not canonical JSON")
                    except (TypeError, ValueError) as exc:
                        errors.append(f"{prefix}: invalid typed payload ({exc})")
                if action["action_type"] == ActionKind.HINT_REQUESTED.value:
                    hint_actions_by_decision[action["decision_id"]] = (
                        hint_actions_by_decision.get(action["decision_id"], 0) + 1
                    )

                event = action_events.get(action["event_id"])
                if event is None:
                    errors.append(f"{prefix}: missing LearnerActionRecorded event")
                    continue
                projected_action_events.add(event["event_id"])
                response_stream_version = action["response_stream_version"]
                if response_stream_version is not None:
                    if event["stream_id"] != action["response_stream_id"]:
                        errors.append(f"{prefix}: action and response use different streams")
                    elif action["stage"] in {"unassisted", "assisted"} and (
                        event["stream_version"] >= response_stream_version
                    ):
                        errors.append(
                            f"{prefix}: pre-response event does not precede response event"
                        )
                    elif action["stage"] == "post_feedback" and (
                        event["stream_version"] <= response_stream_version
                    ):
                        errors.append(
                            f"{prefix}: post-feedback event does not follow response event"
                        )
                event_payload = event_object(event, "payload_json", payload_cache)
                event_metadata = event_object(
                    event, "metadata_json", metadata_cache
                )
                expected_event_payload = {
                    "action_id": action["id"],
                    "decision_id": action["decision_id"],
                    "sequence": action["sequence"],
                    "stage": action["stage"],
                    "action_type": action["action_type"],
                    "payload": payload,
                    "artifact": artifact_payload,
                }
                if event_payload is not None and event_payload != expected_event_payload:
                    errors.append(f"{prefix}: event payload mismatch")
                if (
                    event["schema_version"] != 1
                    or event["stream_id"] != f"learner:{action['learner_id']}"
                    or event["learner_id"] != action["learner_id"]
                    or event["session_id"] != action["session_id"]
                    or event["causation_id"] != action["decision_id"]
                    or event["occurred_at"] != action["occurred_at"]
                    or event["recorded_at"] != action["recorded_at"]
                ):
                    errors.append(f"{prefix}: event envelope mismatch")

                selection_events = selection_events_by_decision.get(
                    action["decision_id"], []
                )
                selection_event = (
                    selection_events[0] if len(selection_events) == 1 else None
                )
                if selection_event is None:
                    errors.append(
                        f"{prefix}: expected one QuestionSelected event boundary, "
                        f"found {len(selection_events)}"
                    )
                elif (
                    selection_event["schema_version"]
                    not in SUPPORTED_QUESTION_SELECTED_EVENT_SCHEMA_VERSIONS
                    or selection_event["stream_id"] != event["stream_id"]
                    or selection_event["learner_id"] != action["learner_id"]
                    or selection_event["session_id"] != action["session_id"]
                    or selection_event["stream_version"] >= event["stream_version"]
                ):
                    errors.append(
                        f"{prefix}: action does not follow its selection boundary"
                    )

                bounds = session_boundaries.get(action["session_id"])
                started_events = bounds["started"] if bounds is not None else []
                ended_events = bounds["ended"] if bounds is not None else []
                if len(started_events) != 1:
                    errors.append(
                        f"{prefix}: expected one SessionStarted boundary, "
                        f"found {len(started_events)}"
                    )
                else:
                    started_event = started_events[0]
                    started_payload = event_object(
                        started_event, "payload_json", payload_cache
                    )
                    if (
                        started_payload is None
                        or started_payload.get("session_id") != action["session_id"]
                        or started_event["stream_id"] != event["stream_id"]
                        or started_event["learner_id"] != action["learner_id"]
                        or started_event["stream_version"] >= event["stream_version"]
                    ):
                        errors.append(
                            f"{prefix}: action falls outside its session-active interval"
                        )
                if len(ended_events) > 1:
                    errors.append(
                        f"{prefix}: multiple SessionEnded boundaries exist"
                    )
                elif ended_events:
                    ended_event = ended_events[0]
                    ended_payload = event_object(
                        ended_event, "payload_json", payload_cache
                    )
                    if (
                        ended_payload is None
                        or ended_payload.get("session_id") != action["session_id"]
                        or ended_event["stream_id"] != event["stream_id"]
                        or ended_event["learner_id"] != action["learner_id"]
                        or ended_event["stream_version"] <= event["stream_version"]
                    ):
                        errors.append(
                            f"{prefix}: action falls outside its session-active interval"
                        )

                invalidation_events = invalidation_events_for_actions.get(
                    action["decision_id"], []
                )
                if action["decision_invalidated_at"] is not None:
                    if len(invalidation_events) != 1:
                        errors.append(
                            f"{prefix}: expected one DecisionInvalidated boundary, "
                            f"found {len(invalidation_events)}"
                        )
                    elif invalidation_events[0]["stream_version"] <= event[
                        "stream_version"
                    ]:
                        errors.append(
                            f"{prefix}: action was appended at or after decision invalidation"
                        )
                elif invalidation_events:
                    errors.append(
                        f"{prefix}: unprojected DecisionInvalidated boundary exists"
                    )

                revocation = revocation_boundaries.get(
                    action["decision_question_id"]
                )
                if revocation is not None:
                    revoked_recorded_at = aware_timestamp(
                        revocation["revocation_recorded_at"],
                        f"revocation event {revocation['revocation_event_id']} recording time",
                    )
                    if (
                        recorded_at is not None
                        and revoked_recorded_at is not None
                        and recorded_at >= revoked_recorded_at
                    ):
                        errors.append(
                            f"{prefix}: action was recorded at or after emergency revocation"
                        )

                if (
                    action["stage"] in {"unassisted", "assisted"}
                    and selection_event is not None
                    and any(
                        selection_event["stream_version"]
                        < version
                        < event["stream_version"]
                        for version in projection_versions_by_stream.get(
                            event["stream_id"], []
                        )
                    )
                ):
                    errors.append(
                        f"{prefix}: pre-response action follows a learner projection advance"
                    )
                if event_metadata is not None:
                    compare_payload(
                        event_metadata,
                        {
                            "action_schema_version": 1,
                            "observational_only": True,
                            "corpus_release_id": action["decision_release_id"],
                        },
                        f"{prefix} metadata",
                    )
                expected_command_hash = _content_hash(
                    {
                        "decision_id": action["decision_id"],
                        "stage": action["stage"],
                        "action_type": action["action_type"],
                        "payload": payload,
                        "artifact": artifact_payload,
                    }
                )
                if action["command_hash"] != expected_command_hash:
                    errors.append(f"{prefix}: command hash mismatch")

            for decision_id, typed_actions in typed_actions_by_decision.items():
                try:
                    summarize_actions(typed_actions)
                except (TypeError, ValueError) as exc:
                    errors.append(
                        f"decision {decision_id}: invalid learning-action lifecycle ({exc})"
                    )

            for decision_id, traced_hints in hint_actions_by_decision.items():
                if traced_hints > MAX_HINT_COUNT:
                    errors.append(
                        f"decision {decision_id}: traced hint count is out of bounds"
                    )

            for event_id in sorted(set(action_events) - projected_action_events):
                errors.append(
                    f"event {event_id}: LearnerActionRecorded has no action projection"
                )
            for artifact_id in sorted(set(artifacts) - referenced_artifact_ids):
                errors.append(
                    f"learning artifact {artifact_id}: no action references this artifact"
                )
            for attempt in connection.execute(
                "SELECT decision_id, hint_count FROM attempts"
            ):
                traced_hints = hint_actions_by_decision.get(
                    attempt["decision_id"], 0
                )
                if attempt["hint_count"] < traced_hints:
                    errors.append(
                        f"decision {attempt['decision_id']}: attempt omits "
                        f"{traced_hints - attempt['hint_count']} traced hints"
                    )

            # Offline generation is intentionally outside the live question
            # registry, but its operational history still needs ledger-like
            # guarantees. Validate every mutable job summary against its
            # immutable attempts and their self-contained attestations.
            job_rows = connection.execute(
                "SELECT * FROM generation_jobs ORDER BY created_at, id"
            ).fetchall()
            run_rows = connection.execute(
                "SELECT * FROM generation_job_runs ORDER BY job_id, attempt"
            ).fetchall()
            runs_by_job: dict[str, list[sqlite3.Row]] = {}
            for run in run_rows:
                runs_by_job.setdefault(run["job_id"], []).append(run)
                prefix = f"generation run {run['id']}"
                if run["status"] not in {"running", "reviewed", "rejected", "failed"}:
                    errors.append(f"{prefix}: invalid status {run['status']!r}")
                if not run["provider"] or not run["model"] or not run["prompt_version"]:
                    errors.append(f"{prefix}: incomplete provider provenance")
                context_hash = run["source_context_sha256"]
                if not (
                    type(context_hash) is str
                    and len(context_hash) == 64
                    and all(character in "0123456789abcdef" for character in context_hash)
                ):
                    errors.append(f"{prefix}: invalid source-context hash")
                raw_output = (
                    json_value(run["raw_output_json"], f"{prefix} raw output", dict)
                    if run["raw_output_json"] is not None
                    else None
                )
                validation = (
                    json_value(run["validation_json"], f"{prefix} validation", dict)
                    if run["validation_json"] is not None
                    else None
                )
                error_record = (
                    json_value(run["error_json"], f"{prefix} error", dict)
                    if run["error_json"] is not None
                    else None
                )
                if run["status"] == "running":
                    if any(
                        value is not None
                        for value in (raw_output, validation, error_record, run["completed_at"])
                    ):
                        errors.append(f"{prefix}: running attempt has terminal output")
                    continue
                if run["completed_at"] is None:
                    errors.append(f"{prefix}: terminal attempt has no completion time")
                if run["status"] in {"reviewed", "rejected"}:
                    if raw_output is None or validation is None or error_record is not None:
                        errors.append(f"{prefix}: reviewed attempt has invalid output shape")
                        continue
                    if raw_output.get("status") != "quarantined":
                        errors.append(f"{prefix}: generated artifact is not quarantined")
                    if validation.get("source_context_sha256") != context_hash:
                        errors.append(f"{prefix}: source-context attestation mismatch")

                    def attestation_hash(value: Any) -> str | None:
                        try:
                            encoded = json.dumps(
                                value,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                                allow_nan=False,
                            ).encode("utf-8")
                        except (TypeError, ValueError):
                            return None
                        return hashlib.sha256(encoded).hexdigest()

                    generator_provenance = validation.get("generator_provenance")
                    if type(generator_provenance) is not dict or validation.get(
                        "generator_provenance_sha256"
                    ) != attestation_hash(generator_provenance):
                        errors.append(f"{prefix}: generator provenance hash mismatch")
                    reviews = validation.get("reviews")
                    if type(reviews) is not list:
                        errors.append(f"{prefix}: validation has no review array")
                    else:
                        if validation.get("reviews_sha256") != attestation_hash(reviews):
                            errors.append(f"{prefix}: review-set hash mismatch")
                        reviewer_names: set[str] = set()
                        reviewer_models: set[tuple[str, str]] = set()
                        all_accept = bool(reviews)
                        for index, review in enumerate(reviews):
                            review_prefix = f"{prefix} review {index + 1}"
                            if type(review) is not dict:
                                errors.append(f"{review_prefix}: record is not an object")
                                all_accept = False
                                continue
                            provenance = review.get("reviewer")
                            output = review.get("output")
                            if type(provenance) is not dict or review.get(
                                "reviewer_provenance_sha256"
                            ) != attestation_hash(provenance):
                                errors.append(
                                    f"{review_prefix}: reviewer provenance hash mismatch"
                                )
                            if type(output) is not dict or review.get(
                                "reviewer_output_sha256"
                            ) != attestation_hash(output):
                                errors.append(f"{review_prefix}: output hash mismatch")
                            reviewer_name = (
                                provenance.get("reviewer_name")
                                if type(provenance) is dict
                                else None
                            )
                            normalized_name = (
                                " ".join(reviewer_name.split()).casefold()
                                if type(reviewer_name) is str
                                else ""
                            )
                            if not normalized_name or normalized_name in reviewer_names:
                                errors.append(
                                    f"{review_prefix}: missing or duplicate reviewer identity"
                                )
                            reviewer_names.add(normalized_name)
                            if type(provenance) is dict:
                                provider = provenance.get("provider_name")
                                model = provenance.get("model_name")
                                if type(provider) is str and type(model) is str:
                                    model_identity = (
                                        " ".join(provider.split()).casefold(),
                                        " ".join(model.split()).casefold(),
                                    )
                                    if model_identity in reviewer_models:
                                        errors.append(
                                            f"{review_prefix}: duplicate reviewer model identity"
                                        )
                                    reviewer_models.add(model_identity)
                                    if model_identity == (
                                        " ".join(run["provider"].split()).casefold(),
                                        " ".join(run["model"].split()).casefold(),
                                    ):
                                        errors.append(
                                            f"{review_prefix}: reviewer shares generator model"
                                        )
                            if not (
                                review.get("valid") is True
                                and type(output) is dict
                                and output.get("verdict") == "accept"
                            ):
                                all_accept = False
                        deterministic_issues = validation.get("deterministic_issues")
                        deterministic_errors = (
                            [
                                issue
                                for issue in deterministic_issues
                                if type(issue) is dict and issue.get("severity") == "error"
                            ]
                            if type(deterministic_issues) is list
                            else ["malformed"]
                        )
                        should_be_reviewed = all_accept and not deterministic_errors
                        if (run["status"] == "reviewed") != should_be_reviewed:
                            errors.append(f"{prefix}: terminal verdict mismatch")
                elif error_record is None:
                    errors.append(f"{prefix}: failed attempt has no error record")

            valid_job_statuses = {"planned", "running", "reviewed", "rejected", "failed"}
            for job in job_rows:
                prefix = f"generation job {job['id']}"
                status = job["status"]
                if status not in valid_job_statuses:
                    errors.append(f"{prefix}: invalid status {status!r}")
                blueprint = json_value(job["blueprint_json"], f"{prefix} blueprint", dict)
                if blueprint is not None and not blueprint:
                    errors.append(f"{prefix}: blueprint is empty")
                runs = runs_by_job.get(job["id"], [])
                attempts = [run["attempt"] for run in runs]
                if attempts != list(range(1, len(runs) + 1)):
                    errors.append(f"{prefix}: run attempts are not contiguous from one")
                running_runs = [run for run in runs if run["status"] == "running"]
                latest = runs[-1] if runs else None
                if status == "planned":
                    if running_runs:
                        errors.append(f"{prefix}: planned job has a running attempt")
                    if any(
                        job[field] is not None
                        for field in (
                            "provider",
                            "model",
                            "raw_output_json",
                            "validation_json",
                        )
                    ):
                        errors.append(f"{prefix}: planned job retains execution summary")
                elif latest is None:
                    errors.append(f"{prefix}: non-planned job has no run history")
                else:
                    if latest["status"] != status:
                        errors.append(f"{prefix}: summary status differs from latest run")
                    if job["provider"] != latest["provider"] or job["model"] != latest["model"]:
                        errors.append(f"{prefix}: provider summary differs from latest run")
                    if status == "running" and (
                        len(running_runs) != 1 or running_runs[0]["id"] != latest["id"]
                    ):
                        errors.append(f"{prefix}: running summary has no unique latest owner")
                    if status != "running" and running_runs:
                        errors.append(f"{prefix}: terminal summary retains a running attempt")
                    if status in {"reviewed", "rejected"} and (
                        job["raw_output_json"] != latest["raw_output_json"]
                        or job["validation_json"] != latest["validation_json"]
                    ):
                        errors.append(f"{prefix}: artifact summary differs from latest run")
                    if status == "failed" and job["validation_json"] != latest["error_json"]:
                        errors.append(f"{prefix}: failure summary differs from latest run")

            # A release is only immutable if both its membership rows and the
            # versioned registry content addressed by those rows are intact.
            # Recompute hashes from canonical domain objects rather than trusting
            # the stored hash columns that an attacker could change alongside data.
            for row in connection.execute("SELECT * FROM concepts ORDER BY id"):
                try:
                    concept = Concept(
                        row["id"],
                        row["name"],
                        row["description"],
                        row["domain"],
                        row["prior_mastery"],
                    )
                    expected_hash = concept_content_hash(concept)
                except (TypeError, ValueError) as exc:
                    errors.append(f"concept {row['id']}: invalid registry row ({exc})")
                    continue
                if row["content_hash"] != expected_hash:
                    errors.append(f"concept {row['id']}: content hash mismatch")

            for row in connection.execute(
                "SELECT * FROM learning_objectives ORDER BY id"
            ):
                try:
                    supporting_concept_ids = json.loads(
                        row["supporting_concept_ids_json"]
                    )
                    if not isinstance(supporting_concept_ids, list):
                        raise ValueError(
                            "supporting_concept_ids_json must contain an array"
                        )
                    objective = LearningObjective(
                        id=row["id"],
                        name=row["name"],
                        description=row["description"],
                        primary_concept_id=row["primary_concept_id"],
                        supporting_concept_ids=tuple(supporting_concept_ids),
                        operation=ObjectiveOperation(row["operation"]),
                        evidence_type=row["evidence_type"],
                        prior_mastery=row["prior_mastery"],
                    )
                    expected_hash = objective_content_hash(objective)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(
                        f"learning objective {row['id']}: invalid registry row ({exc})"
                    )
                    continue
                if row["content_hash"] != expected_hash:
                    errors.append(
                        f"learning objective {row['id']}: content hash mismatch"
                    )

            for row in connection.execute("SELECT * FROM misconceptions ORDER BY id"):
                try:
                    misconception = Misconception(
                        row["id"],
                        row["concept_id"],
                        row["name"],
                        row["description"],
                    )
                    expected_hash = misconception_content_hash(misconception)
                except (TypeError, ValueError) as exc:
                    errors.append(
                        f"misconception {row['id']}: invalid registry row ({exc})"
                    )
                    continue
                if row["content_hash"] != expected_hash:
                    errors.append(f"misconception {row['id']}: content hash mismatch")

            for row in connection.execute("SELECT * FROM sources ORDER BY id"):
                try:
                    metadata = json.loads(row["metadata_json"])
                    if not isinstance(metadata, dict):
                        raise ValueError("metadata_json must contain an object")
                    source = Source(
                        row["id"],
                        row["title"],
                        row["uri"],
                        row["license"],
                        metadata,
                    )
                    expected_hash = source_content_hash(source)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"source {row['id']}: invalid registry row ({exc})")
                    continue
                if row["content_hash"] != expected_hash:
                    errors.append(f"source {row['id']}: content hash mismatch")

            question_batch_size = 400
            question_registry_cursor = connection.execute(
                "SELECT * FROM questions ORDER BY id"
            )
            while batch_rows := question_registry_cursor.fetchmany(
                question_batch_size
            ):
                batch_ids = [row["id"] for row in batch_rows]
                try:
                    hydrated = self._questions_by_ids(connection, batch_ids)
                    hydrated_by_id = {question.id: question for question in hydrated}
                except (
                    TypeError,
                    ValueError,
                    KeyError,
                    json.JSONDecodeError,
                    ValidationError,
                ):
                    # Preserve useful per-item diagnostics on a corrupted batch;
                    # the healthy 100k path remains four queries per 400 items.
                    hydrated_by_id = {}
                    for row in batch_rows:
                        try:
                            hydrated_by_id[row["id"]] = self._question_from_row(
                                connection, row
                            )
                        except (
                            TypeError,
                            ValueError,
                            KeyError,
                            json.JSONDecodeError,
                            ValidationError,
                        ) as exc:
                            errors.append(
                                f"question {row['id']}: invalid registry row ({exc})"
                            )
                for row in batch_rows:
                    question = hydrated_by_id.get(row["id"])
                    if question is None:
                        if not any(
                            error.startswith(
                                f"question {row['id']}: invalid registry row"
                            )
                            for error in errors
                        ):
                            errors.append(
                                f"question {row['id']}: missing registry components"
                            )
                        continue
                    expected_hash = question_content_hash(question)
                    if row["content_hash"] != expected_hash:
                        errors.append(
                            f"question {row['id']}: content hash mismatch"
                        )

            release_rows = connection.execute(
                "SELECT * FROM corpus_releases ORDER BY id"
            ).fetchall()
            for release in release_rows:
                release_id = release["id"]
                prefix = f"release {release_id}"
                if release["sealed_at"] is None:
                    errors.append(f"{prefix}: release is not sealed")
                concept_rows = connection.execute(
                    """SELECT membership.concept_id, concept.content_hash
                       FROM release_concepts membership
                       JOIN concepts concept ON concept.id = membership.concept_id
                       WHERE membership.release_id = ?
                       ORDER BY membership.concept_id""",
                    (release_id,),
                ).fetchall()
                domain_rows = connection.execute(
                    """SELECT domain_id, content_hash, name, description, sort_order
                       FROM release_domains WHERE release_id = ?
                       ORDER BY domain_id""",
                    (release_id,),
                ).fetchall()
                topic_rows = connection.execute(
                    """SELECT topic_id, content_hash, domain_id, parent_topic_id,
                              name, description, related_topic_ids_json, sort_order
                       FROM release_topics WHERE release_id = ?
                       ORDER BY topic_id""",
                    (release_id,),
                ).fetchall()
                topic_concept_rows = connection.execute(
                    """SELECT topic_id, concept_id, position
                       FROM release_topic_concepts WHERE release_id = ?
                       ORDER BY topic_id, position, concept_id""",
                    (release_id,),
                ).fetchall()
                question_topic_rows = connection.execute(
                    """SELECT question_id, topic_id, relation
                       FROM release_question_topics WHERE release_id = ?
                       ORDER BY question_id, topic_id, relation""",
                    (release_id,),
                ).fetchall()
                concept_ids_in_release = {row["concept_id"] for row in concept_rows}
                edge_rows = connection.execute(
                    """SELECT source_id, target_id, relation, weight
                       FROM release_edges WHERE release_id = ?
                       ORDER BY source_id, target_id, relation""",
                    (release_id,),
                ).fetchall()
                misconception_rows = connection.execute(
                    """SELECT membership.misconception_id, misconception.content_hash,
                              misconception.concept_id
                       FROM release_misconceptions membership
                       JOIN misconceptions misconception
                         ON misconception.id = membership.misconception_id
                       WHERE membership.release_id = ?
                       ORDER BY membership.misconception_id""",
                    (release_id,),
                ).fetchall()
                misconception_ids_in_release = {
                    row["misconception_id"] for row in misconception_rows
                }
                source_rows = connection.execute(
                    """SELECT membership.source_id, source.content_hash
                       FROM release_sources membership
                       JOIN sources source ON source.id = membership.source_id
                       WHERE membership.release_id = ?
                       ORDER BY membership.source_id""",
                    (release_id,),
                ).fetchall()
                source_ids_in_release = {row["source_id"] for row in source_rows}
                question_rows = connection.execute(
                    """SELECT membership.question_id, question.content_hash,
                              question.provenance_json,
                              membership.status, membership.evidence_weight
                       FROM release_questions membership
                       JOIN questions question ON question.id = membership.question_id
                       WHERE membership.release_id = ?
                       ORDER BY membership.question_id""",
                    (release_id,),
                ).fetchall()
                for question_row in question_rows:
                    if question_row["status"] not in {
                        QuestionStatus.APPROVED.value,
                        QuestionStatus.CALIBRATED.value,
                    }:
                        continue
                    try:
                        provenance = json.loads(
                            question_row["provenance_json"]
                        )
                    except (TypeError, json.JSONDecodeError):
                        provenance = None
                    if generated_question_runtime_safe(
                        provenance,
                        status=question_row["status"],
                    ):
                        continue
                    revoked = connection.execute(
                        """SELECT 1 FROM question_revocations
                           WHERE question_id=?""",
                        (question_row["question_id"],),
                    ).fetchone()
                    if revoked is None:
                        errors.append(
                            f"{prefix}: active question "
                            f"{question_row['question_id']} fails the generated-content "
                            "activation gate and is not emergency-revoked"
                        )
                objective_rows = connection.execute(
                    """SELECT membership.objective_id, objective.content_hash,
                              objective.primary_concept_id,
                              objective.supporting_concept_ids_json
                       FROM release_learning_objectives membership
                       JOIN learning_objectives objective
                         ON objective.id = membership.objective_id
                       WHERE membership.release_id = ?
                       ORDER BY membership.objective_id""",
                    (release_id,),
                ).fetchall()
                objective_graph_row = connection.execute(
                    """SELECT graph_version FROM release_objective_graphs
                       WHERE release_id = ?""",
                    (release_id,),
                ).fetchone()
                objective_edge_rows = connection.execute(
                    """SELECT edge_id, source_objective_id,
                              target_objective_id, relation, weight, rationale
                       FROM release_objective_edges
                       WHERE release_id = ? ORDER BY edge_id""",
                    (release_id,),
                ).fetchall()
                question_objective_rows = connection.execute(
                    """SELECT question_id, objective_id
                       FROM release_question_objectives
                       WHERE release_id = ?
                       ORDER BY question_id, objective_id""",
                    (release_id,),
                ).fetchall()
                option_objective_rows = connection.execute(
                    """SELECT mapping.question_id, mapping.option_id,
                              mapping.objective_id, option.is_correct,
                              option.misconception_id,
                              misconception.concept_id AS misconception_concept_id
                       FROM release_option_objectives mapping
                       JOIN options option
                         ON option.question_id = mapping.question_id
                        AND option.option_id = mapping.option_id
                       LEFT JOIN misconceptions misconception
                         ON misconception.id = option.misconception_id
                       WHERE mapping.release_id = ?
                       ORDER BY mapping.question_id, mapping.option_id,
                                mapping.objective_id""",
                    (release_id,),
                ).fetchall()
                objective_ids_in_release = {
                    row["objective_id"] for row in objective_rows
                }
                if objective_graph_row is None and objective_edge_rows:
                    errors.append(
                        f"{prefix}: objective edges exist without a declared graph"
                    )
                if objective_graph_row is not None:
                    if objective_graph_row["graph_version"] != 1:
                        errors.append(
                            f"{prefix}: unsupported objective graph version"
                        )
                    if not objective_rows:
                        errors.append(
                            f"{prefix}: objective graph has no learning objectives"
                        )
                objective_adjacency = {
                    objective_id: [] for objective_id in objective_ids_in_release
                }
                objective_indegree = {
                    objective_id: 0 for objective_id in objective_ids_in_release
                }
                for edge in objective_edge_rows:
                    endpoints = {
                        edge["source_objective_id"],
                        edge["target_objective_id"],
                    }
                    outside = endpoints - objective_ids_in_release
                    if outside:
                        errors.append(
                            f"{prefix}: objective edge {edge['edge_id']} "
                            "references objectives outside the release: "
                            + ", ".join(sorted(outside))
                        )
                        continue
                    try:
                        ObjectiveEdge(
                            id=edge["edge_id"],
                            source_id=edge["source_objective_id"],
                            target_id=edge["target_objective_id"],
                            relation=RelationType(edge["relation"]),
                            weight=edge["weight"],
                            rationale=edge["rationale"],
                        )
                    except (TypeError, ValueError) as exc:
                        errors.append(
                            f"{prefix}: objective edge {edge['edge_id']} is "
                            f"invalid ({exc})"
                        )
                        continue
                    objective_adjacency[edge["source_objective_id"]].append(
                        edge["target_objective_id"]
                    )
                    objective_indegree[edge["target_objective_id"]] += 1
                objective_queue = deque(
                    objective_id
                    for objective_id, degree in objective_indegree.items()
                    if degree == 0
                )
                objective_visited = 0
                while objective_queue:
                    objective_id = objective_queue.popleft()
                    objective_visited += 1
                    for dependent_id in objective_adjacency[objective_id]:
                        objective_indegree[dependent_id] -= 1
                        if objective_indegree[dependent_id] == 0:
                            objective_queue.append(dependent_id)
                if objective_visited != len(objective_ids_in_release):
                    errors.append(
                        f"{prefix}: objective prerequisite graph contains a cycle"
                    )
                objective_concepts: dict[str, set[str]] = {}
                for objective in objective_rows:
                    try:
                        supporting_ids = json.loads(
                            objective["supporting_concept_ids_json"]
                        )
                    except (TypeError, json.JSONDecodeError) as exc:
                        errors.append(
                            f"{prefix}: learning objective "
                            f"{objective['objective_id']} has invalid supporting concepts "
                            f"({exc})"
                        )
                        continue
                    if not isinstance(supporting_ids, list):
                        errors.append(
                            f"{prefix}: learning objective "
                            f"{objective['objective_id']} supporting concepts are not an array"
                        )
                        continue
                    concept_ids = {
                        objective["primary_concept_id"], *supporting_ids
                    }
                    objective_concepts[objective["objective_id"]] = concept_ids
                    outside = concept_ids - concept_ids_in_release
                    if outside:
                        errors.append(
                            f"{prefix}: learning objective "
                            f"{objective['objective_id']} references concepts outside "
                            f"the release: {', '.join(sorted(outside))}"
                        )
                for mapping in question_objective_rows:
                    if mapping["objective_id"] not in objective_ids_in_release:
                        errors.append(
                            f"{prefix}: question {mapping['question_id']} maps to "
                            f"objective {mapping['objective_id']} outside the release"
                        )
                for mapping in option_objective_rows:
                    objective_id = mapping["objective_id"]
                    if objective_id not in objective_ids_in_release:
                        errors.append(
                            f"{prefix}: option {mapping['question_id']}/"
                            f"{mapping['option_id']} maps to objective {objective_id} "
                            "outside the release"
                        )
                    if mapping["is_correct"]:
                        errors.append(
                            f"{prefix}: correct option {mapping['question_id']}/"
                            f"{mapping['option_id']} has a diagnostic objective"
                        )
                    if mapping["misconception_id"] is None:
                        errors.append(
                            f"{prefix}: diagnostic option {mapping['question_id']}/"
                            f"{mapping['option_id']} has no named misconception"
                        )
                    elif mapping["misconception_concept_id"] not in (
                        objective_concepts.get(objective_id) or set()
                    ):
                        errors.append(
                            f"{prefix}: option {mapping['question_id']}/"
                            f"{mapping['option_id']} misconception is outside diagnostic "
                            f"objective {objective_id}"
                        )
                if objective_rows:
                    primary_objective_rows = connection.execute(
                        """SELECT rq.question_id, rq.status, qc.concept_id,
                                  rqo.objective_id
                           FROM release_questions rq
                           JOIN question_concepts qc
                             ON qc.question_id = rq.question_id
                            AND qc.role = 'primary'
                           LEFT JOIN release_question_objectives rqo
                             ON rqo.release_id = rq.release_id
                            AND rqo.question_id = rq.question_id
                           WHERE rq.release_id = ?
                           ORDER BY rq.question_id""",
                        (release_id,),
                    ).fetchall()
                    covered_concepts = set().union(
                        *objective_concepts.values()
                    ) if objective_concepts else set()
                    used_objective_ids: set[str] = set()
                    for mapping in primary_objective_rows:
                        objective_id = mapping["objective_id"]
                        primary_concept_id = mapping["concept_id"]
                        if objective_id is not None:
                            used_objective_ids.add(objective_id)
                            if primary_concept_id not in (
                                objective_concepts.get(objective_id) or set()
                            ):
                                errors.append(
                                    f"{prefix}: question {mapping['question_id']} "
                                    f"primary concept {primary_concept_id} is outside "
                                    f"objective {objective_id}"
                                )
                        elif (
                            mapping["status"] in {"approved", "calibrated"}
                            and primary_concept_id in covered_concepts
                        ):
                            errors.append(
                                f"{prefix}: eligible question "
                                f"{mapping['question_id']} lacks an objective mapping"
                            )
                    unused_objective_ids = (
                        objective_ids_in_release - used_objective_ids
                    )
                    if unused_objective_ids:
                        errors.append(
                            f"{prefix}: learning objectives have no direct questions: "
                            + ", ".join(sorted(unused_objective_ids))
                        )
                    missing_option_mappings = connection.execute(
                        """SELECT option.question_id, option.option_id
                           FROM release_question_objectives direct
                           JOIN options option
                             ON option.question_id = direct.question_id
                            AND option.is_correct = 0
                           LEFT JOIN release_option_objectives diagnostic
                             ON diagnostic.release_id = direct.release_id
                            AND diagnostic.question_id = option.question_id
                            AND diagnostic.option_id = option.option_id
                           WHERE direct.release_id = ?
                             AND diagnostic.objective_id IS NULL
                           ORDER BY option.question_id, option.option_id""",
                        (release_id,),
                    ).fetchall()
                    for missing in missing_option_mappings:
                        errors.append(
                            f"{prefix}: distractor {missing['question_id']}/"
                            f"{missing['option_id']} lacks a diagnostic objective"
                        )
                for edge in edge_rows:
                    absent = {
                        edge["source_id"], edge["target_id"]
                    } - concept_ids_in_release
                    if absent:
                        errors.append(
                            f"{prefix}: edge references concepts outside the release: "
                            f"{', '.join(sorted(absent))}"
                        )
                for misconception in misconception_rows:
                    if misconception["concept_id"] not in concept_ids_in_release:
                        errors.append(
                            f"{prefix}: misconception {misconception['misconception_id']} "
                            "belongs to a concept outside the release"
                        )

                for question in question_rows:
                    try:
                        status = QuestionStatus(question["status"])
                    except ValueError:
                        errors.append(
                            f"{prefix}: question {question['question_id']} has invalid status"
                        )
                    else:
                        if question["evidence_weight"] != status.evidence_weight:
                            errors.append(
                                f"{prefix}: question {question['question_id']} has "
                                "an evidence weight inconsistent with its status"
                            )

                catalog_present = bool(
                    domain_rows
                    or topic_rows
                    or topic_concept_rows
                    or question_topic_rows
                )
                domain_ids_in_release = {row["domain_id"] for row in domain_rows}
                topic_by_id = {row["topic_id"]: row for row in topic_rows}
                topic_concepts: dict[str, list[str]] = {}
                owner_by_concept: dict[str, str] = {}
                expected_position: dict[str, int] = {}
                for membership in topic_concept_rows:
                    topic_id = membership["topic_id"]
                    position = membership["position"]
                    if position != expected_position.get(topic_id, 0):
                        errors.append(
                            f"{prefix}: topic {topic_id} concept positions are not contiguous"
                        )
                    expected_position[topic_id] = position + 1
                    topic_concepts.setdefault(topic_id, []).append(
                        membership["concept_id"]
                    )
                    prior_owner = owner_by_concept.get(membership["concept_id"])
                    if prior_owner is not None:
                        errors.append(
                            f"{prefix}: concept {membership['concept_id']} has multiple "
                            "topic owners"
                        )
                    owner_by_concept[membership["concept_id"]] = topic_id

                related_by_topic: dict[str, tuple[str, ...]] = {}
                for domain in domain_rows:
                    try:
                        hydrated_domain = Domain(
                            domain["domain_id"],
                            domain["name"],
                            domain["description"],
                            domain["sort_order"],
                        )
                        expected_hash = domain_content_hash(hydrated_domain)
                    except (TypeError, ValueError) as exc:
                        errors.append(
                            f"{prefix}: domain {domain['domain_id']} is invalid ({exc})"
                        )
                    else:
                        if domain["content_hash"] != expected_hash:
                            errors.append(
                                f"{prefix}: domain {domain['domain_id']} content hash mismatch"
                            )
                for topic in topic_rows:
                    related_value = json_value(
                        topic["related_topic_ids_json"],
                        f"{prefix} topic {topic['topic_id']} related topics",
                        list,
                    )
                    related_ids = tuple(related_value or [])
                    related_by_topic[topic["topic_id"]] = related_ids
                    try:
                        hydrated_topic = Topic(
                            id=topic["topic_id"],
                            domain_id=topic["domain_id"],
                            name=topic["name"],
                            description=topic["description"],
                            concept_ids=tuple(
                                topic_concepts.get(topic["topic_id"], [])
                            ),
                            parent_id=topic["parent_topic_id"],
                            related_topic_ids=related_ids,
                            sort_order=topic["sort_order"],
                        )
                        expected_hash = topic_content_hash(hydrated_topic)
                    except (TypeError, ValueError) as exc:
                        errors.append(
                            f"{prefix}: topic {topic['topic_id']} is invalid ({exc})"
                        )
                    else:
                        if topic["content_hash"] != expected_hash:
                            errors.append(
                                f"{prefix}: topic {topic['topic_id']} content hash mismatch"
                            )
                    if topic["domain_id"] not in domain_ids_in_release:
                        errors.append(
                            f"{prefix}: topic {topic['topic_id']} has no release domain"
                        )
                    parent_id = topic["parent_topic_id"]
                    if parent_id:
                        parent = topic_by_id.get(parent_id)
                        if parent is None:
                            errors.append(
                                f"{prefix}: topic {topic['topic_id']} has no release parent"
                            )
                        elif parent["domain_id"] != topic["domain_id"]:
                            errors.append(
                                f"{prefix}: topic {topic['topic_id']} crosses domain hierarchy"
                            )
                    for related_id in related_ids:
                        if related_id == topic["topic_id"]:
                            errors.append(
                                f"{prefix}: topic {topic['topic_id']} relates to itself"
                            )
                        elif related_id not in topic_by_id:
                            errors.append(
                                f"{prefix}: topic {topic['topic_id']} has unknown relation "
                                f"{related_id}"
                            )

                for topic_id, related_ids in related_by_topic.items():
                    for related_id in related_ids:
                        if (
                            related_id in related_by_topic
                            and topic_id not in related_by_topic[related_id]
                        ):
                            errors.append(
                                f"{prefix}: topic relation {topic_id}<->{related_id} "
                                "is asymmetric"
                            )
                for start in topic_by_id:
                    trail: set[str] = set()
                    current: str | None = start
                    while current is not None and current in topic_by_id:
                        if current in trail:
                            errors.append(
                                f"{prefix}: topic hierarchy contains a cycle at {current}"
                            )
                            break
                        trail.add(current)
                        current = topic_by_id[current]["parent_topic_id"]

                if catalog_present:
                    missing_owners = concept_ids_in_release - set(owner_by_concept)
                    outside_owners = set(owner_by_concept) - concept_ids_in_release
                    if missing_owners:
                        errors.append(
                            f"{prefix}: concepts have no topic owner: "
                            + ", ".join(sorted(missing_owners))
                        )
                    if outside_owners:
                        errors.append(
                            f"{prefix}: topic ownership includes outside concepts: "
                            + ", ".join(sorted(outside_owners))
                        )
                    if not domain_rows or not topic_rows:
                        errors.append(f"{prefix}: curriculum catalog is incomplete")

                    expected_question_topics: list[tuple[str, str, str]] = []
                    mapped_rows = connection.execute(
                        """SELECT qc.question_id, qc.concept_id, qc.role
                           FROM release_questions rq
                           JOIN question_concepts qc
                             ON qc.question_id = rq.question_id
                           WHERE rq.release_id = ?
                           ORDER BY qc.question_id, qc.concept_id, qc.role""",
                        (release_id,),
                    ).fetchall()
                    relations_by_question: dict[str, dict[str, str]] = {}
                    for mapping in mapped_rows:
                        topic_id = owner_by_concept.get(mapping["concept_id"])
                        if topic_id is None:
                            continue
                        relation = (
                            "primary" if mapping["role"] == "primary" else "cross"
                        )
                        relations = relations_by_question.setdefault(
                            mapping["question_id"], {}
                        )
                        if relation == "primary" or topic_id not in relations:
                            relations[topic_id] = relation
                    for question_id, relations in relations_by_question.items():
                        expected_question_topics.extend(
                            (question_id, topic_id, relation)
                            for topic_id, relation in relations.items()
                        )
                    actual_question_topics = [
                        tuple(row) for row in question_topic_rows
                    ]
                    if actual_question_topics != sorted(expected_question_topics):
                        errors.append(
                            f"{prefix}: question-topic projection does not match concept mappings"
                        )

                outside_concepts = connection.execute(
                    """SELECT DISTINCT qc.question_id, qc.concept_id
                       FROM release_questions rq
                       JOIN question_concepts qc
                         ON qc.question_id = rq.question_id
                       WHERE rq.release_id = ?
                         AND NOT EXISTS (
                             SELECT 1 FROM release_concepts membership
                             WHERE membership.release_id = ?
                               AND membership.concept_id = qc.concept_id
                         )
                       ORDER BY qc.question_id, qc.concept_id""",
                    (release_id, release_id),
                ).fetchall()
                outside_misconceptions = connection.execute(
                    """SELECT DISTINCT option.question_id,
                                      option.misconception_id
                       FROM release_questions rq
                       JOIN options option ON option.question_id = rq.question_id
                       WHERE rq.release_id = ?
                         AND option.misconception_id IS NOT NULL
                         AND NOT EXISTS (
                             SELECT 1 FROM release_misconceptions membership
                             WHERE membership.release_id = ?
                               AND membership.misconception_id =
                                   option.misconception_id
                         )
                       ORDER BY option.question_id, option.misconception_id""",
                    (release_id, release_id),
                ).fetchall()
                outside_sources = connection.execute(
                    """SELECT DISTINCT source.question_id, source.source_id
                       FROM release_questions rq
                       JOIN question_sources source
                         ON source.question_id = rq.question_id
                       WHERE rq.release_id = ?
                         AND NOT EXISTS (
                             SELECT 1 FROM release_sources membership
                             WHERE membership.release_id = ?
                               AND membership.source_id = source.source_id
                         )
                       ORDER BY source.question_id, source.source_id""",
                    (release_id, release_id),
                ).fetchall()
                for row in outside_concepts:
                    errors.append(
                        f"{prefix}: question {row['question_id']} maps concept "
                        f"{row['concept_id']} outside the release"
                    )
                for row in outside_misconceptions:
                    errors.append(
                        f"{prefix}: question {row['question_id']} uses misconception "
                        f"{row['misconception_id']} outside the release"
                    )
                for row in outside_sources:
                    errors.append(
                        f"{prefix}: question {row['question_id']} cites source "
                        f"{row['source_id']} outside the release"
                    )

                release_payload = {
                    "concepts": [
                        (row["concept_id"], row["content_hash"])
                        for row in concept_rows
                    ],
                    "edges": [tuple(row) for row in edge_rows],
                    "misconceptions": [
                        (row["misconception_id"], row["content_hash"])
                        for row in misconception_rows
                    ],
                    "sources": [
                        (row["source_id"], row["content_hash"])
                        for row in source_rows
                    ],
                    "questions": [
                        (row["question_id"], row["content_hash"], row["status"])
                        for row in question_rows
                    ],
                }
                if objective_rows:
                    release_payload.update(
                        {
                            "learning_objectives": [
                                (row["objective_id"], row["content_hash"])
                                for row in objective_rows
                            ],
                            "question_objectives": [
                                tuple(row) for row in question_objective_rows
                            ],
                            "option_objectives": [
                                (
                                    row["question_id"],
                                    row["option_id"],
                                    row["objective_id"],
                                )
                                for row in option_objective_rows
                            ],
                        }
                    )
                if objective_graph_row is not None:
                    release_payload.update(
                        {
                            "objective_graph_version": objective_graph_row[
                                "graph_version"
                            ],
                            "objective_edges": [
                                tuple(row) for row in objective_edge_rows
                            ],
                        }
                    )
                if catalog_present:
                    release_payload.update(
                        {
                            "domains": [
                                (row["domain_id"], row["content_hash"])
                                for row in domain_rows
                            ],
                            "topics": [
                                (row["topic_id"], row["content_hash"])
                                for row in topic_rows
                            ],
                            "question_topics": [
                                tuple(row) for row in question_topic_rows
                            ],
                        }
                    )
                expected_bundle_hash = _content_hash(release_payload)
                if release["bundle_hash"] != expected_bundle_hash:
                    errors.append(f"{prefix}: bundle hash mismatch")

            active_release = connection.execute(
                "SELECT value FROM meta WHERE key = 'active_corpus_release'"
            ).fetchone()
            if active_release:
                active = connection.execute(
                    "SELECT sealed_at FROM corpus_releases WHERE id = ?",
                    (active_release["value"],),
                ).fetchone()
                if active is None:
                    errors.append("active corpus release does not exist")
                elif active["sealed_at"] is None:
                    errors.append("active corpus release is not sealed")

            invalid_sessions = connection.execute(
                """SELECT session.id, session.topic_id, session.exploration_mode,
                          session.root_concept_id, session.corpus_release_id,
                          session.focus_concept_id,
                          session.focus_misconception_id,
                          session.focus_objective_id,
                          topic.topic_id AS matched_topic,
                          concept.concept_id AS matched_concept,
                          focus_concept.concept_id AS matched_focus_concept,
                          focus_misconception.misconception_id
                              AS matched_focus_misconception,
                          objective.objective_id AS matched_focus_objective,
                          objective_definition.primary_concept_id
                              AS objective_primary_concept_id,
                          objective_definition.supporting_concept_ids_json,
                          misconception.concept_id
                              AS misconception_owner_concept_id,
                          EXISTS (
                              SELECT 1
                              FROM release_option_objectives mapping
                              JOIN options option
                                ON option.question_id = mapping.question_id
                               AND option.option_id = mapping.option_id
                              WHERE mapping.release_id = session.corpus_release_id
                                AND mapping.objective_id = session.focus_objective_id
                                AND option.misconception_id
                                    = session.focus_misconception_id
                          ) AS matched_focus_pair
                   FROM sessions session
                   LEFT JOIN release_topics topic
                     ON topic.release_id = session.corpus_release_id
                    AND topic.topic_id = session.topic_id
                   LEFT JOIN release_concepts concept
                     ON concept.release_id = session.corpus_release_id
                    AND concept.concept_id = session.root_concept_id
                   LEFT JOIN release_concepts focus_concept
                     ON focus_concept.release_id = session.corpus_release_id
                    AND focus_concept.concept_id = session.focus_concept_id
                   LEFT JOIN release_misconceptions focus_misconception
                     ON focus_misconception.release_id = session.corpus_release_id
                    AND focus_misconception.misconception_id
                        = session.focus_misconception_id
                   LEFT JOIN release_learning_objectives objective
                     ON objective.release_id = session.corpus_release_id
                    AND objective.objective_id = session.focus_objective_id
                   LEFT JOIN learning_objectives objective_definition
                     ON objective_definition.id = objective.objective_id
                   LEFT JOIN misconceptions misconception
                     ON misconception.id = session.focus_misconception_id
                   WHERE session.exploration_mode NOT IN ('off', 'adaptive')
                      OR (session.topic_id IS NULL
                          AND session.exploration_mode != 'off')
                      OR (session.topic_id IS NOT NULL
                          AND topic.topic_id IS NULL)
                      OR concept.concept_id IS NULL
                      OR (session.focus_concept_id IS NOT NULL
                          AND focus_concept.concept_id IS NULL)
                      OR (session.focus_misconception_id IS NOT NULL
                          AND focus_misconception.misconception_id IS NULL)
                      OR (session.focus_objective_id IS NOT NULL
                          AND objective.objective_id IS NULL)
                      OR (session.focus_objective_id IS NOT NULL
                          AND session.focus_concept_id
                              IS NOT objective_definition.primary_concept_id)
                      OR (session.focus_misconception_id IS NOT NULL
                          AND session.focus_objective_id IS NULL
                          AND session.focus_concept_id
                              IS NOT misconception.concept_id)
                      OR (session.focus_misconception_id IS NOT NULL
                          AND session.focus_objective_id IS NOT NULL
                          AND misconception.concept_id
                              IS NOT objective_definition.primary_concept_id
                          AND NOT EXISTS (
                              SELECT 1
                              FROM json_each(
                                  objective_definition.supporting_concept_ids_json
                              ) supporting
                              WHERE supporting.value = misconception.concept_id
                          ))
                      OR (session.focus_misconception_id IS NOT NULL
                          AND session.focus_objective_id IS NOT NULL
                          AND NOT EXISTS (
                              SELECT 1
                              FROM release_option_objectives mapping
                              JOIN options option
                                ON option.question_id = mapping.question_id
                               AND option.option_id = mapping.option_id
                              WHERE mapping.release_id = session.corpus_release_id
                                AND mapping.objective_id
                                    = session.focus_objective_id
                                AND option.misconception_id
                                    = session.focus_misconception_id
                          ))
                   ORDER BY session.id"""
            ).fetchall()
            for session_row in invalid_sessions:
                prefix = f"session {session_row['id']}"
                if session_row["exploration_mode"] not in {"off", "adaptive"}:
                    errors.append(f"{prefix}: invalid exploration mode")
                if (
                    session_row["topic_id"] is None
                    and session_row["exploration_mode"] != "off"
                ):
                    errors.append(f"{prefix}: exploration has no topic")
                if (
                    session_row["topic_id"] is not None
                    and session_row["matched_topic"] is None
                ):
                    errors.append(f"{prefix}: topic is outside its corpus release")
                if session_row["matched_concept"] is None:
                    errors.append(f"{prefix}: root concept is outside its corpus release")
                if (
                    session_row["focus_concept_id"] is not None
                    and session_row["matched_focus_concept"] is None
                ):
                    errors.append(
                        f"{prefix}: focus concept is outside its corpus release"
                    )
                if (
                    session_row["focus_misconception_id"] is not None
                    and session_row["matched_focus_misconception"] is None
                ):
                    errors.append(
                        f"{prefix}: focus misconception is outside its corpus release"
                    )
                if (
                    session_row["focus_objective_id"] is not None
                    and session_row["matched_focus_objective"] is None
                ):
                    errors.append(
                        f"{prefix}: focus objective is outside its corpus release"
                    )
                if (
                    session_row["focus_objective_id"] is not None
                    and session_row["focus_concept_id"]
                    != session_row["objective_primary_concept_id"]
                ):
                    errors.append(
                        f"{prefix}: focus concept is not the objective's canonical owner"
                    )
                if session_row["focus_misconception_id"] is not None:
                    owner = session_row["misconception_owner_concept_id"]
                    if session_row["focus_objective_id"] is None:
                        if session_row["focus_concept_id"] != owner:
                            errors.append(
                                f"{prefix}: focus misconception owner does not match focus concept"
                            )
                    else:
                        try:
                            supporting = set(
                                json.loads(
                                    session_row[
                                        "supporting_concept_ids_json"
                                    ]
                                )
                            )
                        except (TypeError, ValueError):
                            supporting = set()
                        if owner not in {
                            session_row["objective_primary_concept_id"],
                            *supporting,
                        }:
                            errors.append(
                                f"{prefix}: focus misconception is outside the focus objective"
                            )
                        if not session_row["matched_focus_pair"]:
                            errors.append(
                                f"{prefix}: focus misconception is not mapped to its "
                                "objective in the pinned release"
                            )
            topic_sessions = connection.execute(
                """SELECT id, topic_id, root_concept_id, corpus_release_id
                   FROM sessions WHERE topic_id IS NOT NULL ORDER BY id"""
            ).fetchall()
            for session_row in topic_sessions:
                owned = connection.execute(
                    """WITH RECURSIVE descendants(topic_id) AS (
                           SELECT ?
                           UNION ALL
                           SELECT child.topic_id
                           FROM release_topics child
                           JOIN descendants parent
                             ON child.parent_topic_id = parent.topic_id
                           WHERE child.release_id = ?
                       )
                       SELECT 1 FROM descendants
                       JOIN release_topic_concepts membership
                         ON membership.release_id = ?
                        AND membership.topic_id = descendants.topic_id
                       WHERE membership.concept_id = ? LIMIT 1""",
                    (
                        session_row["topic_id"],
                        session_row["corpus_release_id"],
                        session_row["corpus_release_id"],
                        session_row["root_concept_id"],
                    ),
                ).fetchone()
                if owned is None:
                    errors.append(
                        f"session {session_row['id']}: root concept is not owned "
                        "by its topic hierarchy"
                    )

            if stream_id:
                events = connection.execute(
                    "SELECT * FROM events WHERE stream_id = ? ORDER BY stream_id, stream_version",
                    (stream_id,),
                ).fetchall()
            else:
                events = connection.execute(
                    "SELECT * FROM events ORDER BY stream_id, stream_version"
                ).fetchall()
            expected_version: dict[str, int] = {}
            expected_previous: dict[str, str | None] = {}
            stream_tails: dict[str, sqlite3.Row] = {}
            explicit_misconception_algorithm_seen: set[str] = set()
            response_misconception_algorithms: dict[
                str, tuple[str, str | None]
            ] = {}
            response_learner_models: dict[str, tuple[str, object]] = {}
            for event in events:
                stream = event["stream_id"]
                version = expected_version.get(stream, 1)
                previous_hash = expected_previous.get(stream)
                if event["stream_version"] != version:
                    errors.append(
                        f"{stream}: expected version {version}, found {event['stream_version']}"
                    )
                if event["previous_hash"] != previous_hash:
                    errors.append(f"{stream}@{event['stream_version']}: previous hash mismatch")
                payload = event_object(event, "payload_json", payload_cache)
                metadata = event_object(event, "metadata_json", metadata_cache)
                if metadata is not None and event["event_type"] in {
                    "ResponseSubmitted",
                    "LearnerProjectionAdvanced",
                }:
                    learner_model_version = metadata.get(
                        "learner_model_version"
                    )
                    if (
                        type(learner_model_version) is not str
                        or not learner_model_version.strip()
                    ):
                        errors.append(
                            f"event {event['event_id']}: missing or invalid "
                            "learner model version"
                        )
                    elif learner_model_version not in SUPPORTED_MODEL_VERSIONS:
                        errors.append(
                            f"event {event['event_id']}: unsupported learner "
                            f"model {learner_model_version!r}"
                        )
                    marker = metadata.get(
                        MISCONCEPTION_ALGORITHM_METADATA_KEY
                    )
                    if event["event_type"] == "ResponseSubmitted":
                        if payload is not None:
                            require_exact_fields(
                                payload,
                                RESPONSE_FIELDS,
                                f"event {event['event_id']} response payload",
                            )
                        if event["schema_version"] == 1 and marker is not None:
                            errors.append(
                                f"event {event['event_id']}: legacy response "
                                "schema has a misconception algorithm marker"
                            )
                        elif (
                            event["schema_version"] == 2
                            and marker
                            != MISCONCEPTION_ALGORITHM_VERSION
                        ):
                            errors.append(
                                f"event {event['event_id']}: current response "
                                "schema lacks the supported misconception "
                                "algorithm marker"
                            )
                        elif event["schema_version"] not in {1, 2}:
                            errors.append(
                                f"event {event['event_id']}: unsupported "
                                "ResponseSubmitted schema"
                            )
                        require_exact_fields(
                            metadata,
                            (
                                RESPONSE_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM
                                if event["schema_version"] == 2
                                else RESPONSE_METADATA_FIELDS
                            ),
                            f"event {event['event_id']} response metadata",
                        )
                        if (
                            marker is None
                            and stream
                            in explicit_misconception_algorithm_seen
                        ):
                            errors.append(
                                f"event {event['event_id']}: misconception "
                                "algorithm marker regressed to legacy"
                            )
                        elif marker is not None:
                            explicit_misconception_algorithm_seen.add(stream)
                        response_misconception_algorithms[event["event_id"]] = (
                            stream,
                            marker,
                        )
                        response_learner_models[event["event_id"]] = (
                            stream,
                            learner_model_version,
                        )
                    else:
                        projection_fields = PROJECTION_FIELDS_BY_SCHEMA.get(
                            event["schema_version"]
                        )
                        if projection_fields is None:
                            errors.append(
                                f"event {event['event_id']}: unsupported "
                                "LearnerProjectionAdvanced schema"
                            )
                        elif payload is not None:
                            require_exact_fields(
                                payload,
                                projection_fields,
                                f"event {event['event_id']} projection payload",
                            )
                        require_exact_fields(
                            metadata,
                            (
                                PROJECTION_METADATA_FIELDS_WITH_MISCONCEPTION_ALGORITHM
                                if marker is not None
                                else PROJECTION_METADATA_FIELDS
                            ),
                            f"event {event['event_id']} projection metadata",
                        )
                        if marker is not None and marker != (
                            MISCONCEPTION_ALGORITHM_VERSION
                        ):
                            errors.append(
                                f"event {event['event_id']}: unsupported "
                                "misconception algorithm marker"
                            )
                        response_marker = (
                            response_misconception_algorithms.get(
                                event["causation_id"]
                            )
                        )
                        if response_marker is not None and (
                            response_marker[0] != stream
                            or response_marker[1] != marker
                        ):
                            errors.append(
                                f"event {event['event_id']}: projection "
                                "misconception algorithm does not match its "
                                "response"
                            )
                        response_model = response_learner_models.get(
                            event["causation_id"]
                        )
                        if response_model is not None and (
                            response_model[0] != stream
                            or response_model[1] != learner_model_version
                        ):
                            errors.append(
                                f"event {event['event_id']}: projection "
                                "learner model does not match its response"
                            )
                        required_models = (
                            PROJECTION_MODEL_VERSIONS_BY_EVENT_SCHEMA.get(
                                event["schema_version"]
                            )
                        )
                        if (
                            required_models is not None
                            and learner_model_version not in required_models
                        ):
                            errors.append(
                                f"event {event['event_id']}: projection schema "
                                f"{event['schema_version']} requires learner "
                                f"model in {sorted(required_models)!r}"
                            )
                if payload is not None and metadata is not None:
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
                        "payload": payload,
                        "metadata": metadata,
                        "previous_hash": event["previous_hash"],
                    }
                    actual_hash = _content_hash(envelope)
                    if actual_hash != event["payload_hash"]:
                        errors.append(
                            f"{stream}@{event['stream_version']}: payload hash mismatch"
                        )
                expected_version[stream] = event["stream_version"] + 1
                expected_previous[stream] = event["payload_hash"]
                stream_tails[stream] = event

            if stream_id:
                head_rows = connection.execute(
                    "SELECT * FROM stream_heads WHERE stream_id = ?", (stream_id,)
                ).fetchall()
            else:
                head_rows = connection.execute("SELECT * FROM stream_heads").fetchall()
            heads = {row["stream_id"]: row for row in head_rows}
            for stream, tail in stream_tails.items():
                head = heads.get(stream)
                if head is None:
                    errors.append(f"{stream}: missing stream head")
                    continue
                if head["stream_version"] != tail["stream_version"]:
                    errors.append(
                        f"{stream}: stream head version mismatch "
                        f"({head['stream_version']} != {tail['stream_version']})"
                    )
                if head["payload_hash"] != tail["payload_hash"]:
                    errors.append(f"{stream}: stream head hash mismatch")
            for stream in heads.keys() - stream_tails.keys():
                errors.append(f"{stream}: stream head has no matching event tail")

            revocation_events: dict[str, list[sqlite3.Row]] = {}
            for event in connection.execute(
                """SELECT rowid AS storage_order, * FROM events
                   WHERE event_type = 'QuestionEmergencyRevoked'
                   ORDER BY recorded_at, storage_order"""
            ).fetchall():
                payload = event_object(event, "payload_json", payload_cache)
                event_object(event, "metadata_json", metadata_cache)
                if payload is None:
                    continue
                question_id = payload.get("question_id")
                if not isinstance(question_id, str) or not question_id:
                    errors.append(
                        f"event {event['event_id']}: missing revoked question_id"
                    )
                    continue
                revocation_events.setdefault(question_id, []).append(event)

            # Domain timestamps on decisions and attempts describe when the
            # interaction occurred and are intentionally supplied by callers
            # (for deterministic simulation and offline ingestion).  They
            # therefore cannot establish whether an operation was accepted
            # before or after a safety revocation.  Use the immutable semantic
            # event's system-recorded timestamp instead, with SQLite insertion
            # order only as a tie-breaker for equal clock readings.
            decision_questions = {
                row["id"]: row["question_id"]
                for row in connection.execute(
                    "SELECT id, question_id FROM decisions"
                )
            }
            attempt_questions = {
                row["event_id"]: row["question_id"]
                for row in connection.execute(
                    "SELECT event_id, question_id FROM attempts"
                )
            }
            recorded_selections: dict[str, list[sqlite3.Row]] = {}
            recorded_attempts: dict[str, list[sqlite3.Row]] = {}
            for event in connection.execute(
                """SELECT rowid AS storage_order, * FROM events
                   WHERE event_type IN ('QuestionSelected', 'ResponseSubmitted')
                   ORDER BY recorded_at, storage_order"""
            ).fetchall():
                payload = event_object(event, "payload_json", payload_cache)
                if payload is None:
                    continue
                question_id = payload.get("question_id")
                if not isinstance(question_id, str) or not question_id:
                    continue
                if event["event_type"] == "QuestionSelected":
                    decision_id = payload.get("decision_id")
                    if decision_questions.get(decision_id) == question_id:
                        recorded_selections.setdefault(question_id, []).append(event)
                elif attempt_questions.get(event["event_id"]) == question_id:
                    recorded_attempts.setdefault(question_id, []).append(event)

            def recorded_after(candidate: sqlite3.Row, anchor: sqlite3.Row) -> bool:
                return (
                    candidate["recorded_at"],
                    candidate["storage_order"],
                ) > (anchor["recorded_at"], anchor["storage_order"])

            revocation_rows = connection.execute(
                "SELECT * FROM question_revocations ORDER BY question_id"
            ).fetchall()
            revoked_question_ids = {row["question_id"] for row in revocation_rows}
            for revocation in revocation_rows:
                question_id = revocation["question_id"]
                prefix = f"question revocation {question_id}"
                matching = revocation_events.get(question_id, [])
                if len(matching) != 1:
                    errors.append(
                        f"{prefix}: expected one safety event, found {len(matching)}"
                    )
                    continue
                event = matching[0]
                payload = event_object(event, "payload_json", payload_cache)
                if event["event_id"] != revocation["event_id"]:
                    errors.append(f"{prefix}: event ID mismatch")
                if event["occurred_at"] != revocation["revoked_at"]:
                    errors.append(f"{prefix}: revocation time mismatch")
                if payload is not None:
                    compare_payload(
                        payload,
                        {
                            "question_id": question_id,
                            "reason": revocation["reason"],
                        },
                        prefix,
                    )
                late_decisions = sum(
                    recorded_after(candidate, event)
                    for candidate in recorded_selections.get(question_id, [])
                )
                late_attempts = sum(
                    recorded_after(candidate, event)
                    for candidate in recorded_attempts.get(question_id, [])
                )
                if late_decisions:
                    errors.append(
                        f"{prefix}: {late_decisions} decisions were selected after revocation"
                    )
                if late_attempts:
                    errors.append(
                        f"{prefix}: {late_attempts} attempts were accepted after revocation"
                    )
            for question_id, matching in revocation_events.items():
                if question_id not in revoked_question_ids:
                    for event in matching:
                        errors.append(
                            f"event {event['event_id']}: safety event has no revocation projection"
                        )

            quarantine_events_by_attempt: dict[
                str, list[sqlite3.Row]
            ] = {}
            for event in connection.execute(
                """SELECT * FROM events
                   WHERE event_type='ResponseEvidenceQuarantined'
                   ORDER BY stream_id, stream_version"""
            ).fetchall():
                label = (
                    "historical generated-evidence quarantine "
                    f"{event['event_id']}"
                )
                payload = event_object(
                    event,
                    "payload_json",
                    payload_cache,
                )
                metadata = event_object(
                    event,
                    "metadata_json",
                    metadata_cache,
                )
                if payload is None or metadata is None:
                    continue
                require_exact_fields(
                    payload,
                    frozenset(
                        {
                            "attempt_id",
                            "response_event_id",
                            "learner_id",
                            "question_id",
                            "reason",
                            "projection_applied",
                        }
                    ),
                    f"{label} payload",
                )
                require_exact_fields(
                    metadata,
                    frozenset(
                        {
                            "safety_policy",
                            "requires_explicit_rebuild",
                        }
                    ),
                    f"{label} metadata",
                )
                attempt_id = payload.get("attempt_id")
                if isinstance(attempt_id, str) and attempt_id:
                    quarantine_events_by_attempt.setdefault(
                        attempt_id,
                        [],
                    ).append(event)
                attempt = (
                    connection.execute(
                        """SELECT * FROM attempts WHERE id=?""",
                        (attempt_id,),
                    ).fetchone()
                    if isinstance(attempt_id, str)
                    else None
                )
                if (
                    attempt is None
                    or event["schema_version"] != 1
                    or event["stream_id"]
                    != f"learner:{attempt['learner_id']}"
                    or event["learner_id"] != attempt["learner_id"]
                    or event["session_id"] is not None
                    or event["correlation_id"] != attempt["id"]
                    or event["causation_id"] != attempt["event_id"]
                    or payload.get("response_event_id")
                    != attempt["event_id"]
                    or payload.get("learner_id")
                    != attempt["learner_id"]
                    or payload.get("question_id")
                    != attempt["question_id"]
                    or payload.get("reason")
                    != LEGACY_UNREVIEWED_GENERATED_REVOCATION_REASON
                    or payload.get("projection_applied") is not False
                    or metadata.get("safety_policy")
                    != HISTORICAL_GENERATED_EVIDENCE_POLICY
                    or metadata.get("requires_explicit_rebuild")
                    is not True
                    or event["idempotency_key"]
                    != _historical_generated_evidence_key(
                        attempt["id"] if attempt is not None else ""
                    )
                ):
                    errors.append(f"{label}: invalid safety boundary")

            contaminated_attempts = (
                self._historically_contaminated_generated_attempts(
                    connection
                )
            )
            contaminated_ids = {
                row["attempt_id"] for row in contaminated_attempts
            }
            for row in contaminated_attempts:
                matching = quarantine_events_by_attempt.get(
                    row["attempt_id"],
                    [],
                )
                if len(matching) != 1:
                    errors.append(
                        "historical generated-evidence attempt "
                        f"{row['attempt_id']}: expected one quarantine event, "
                        f"found {len(matching)}"
                    )
                errors.append(
                    f"learner {row['learner_id']}: projection contains "
                    "quarantined generated-question evidence from attempt "
                    f"{row['attempt_id']}; explicit audited rebuild required"
                )
            for attempt_id, matching in (
                quarantine_events_by_attempt.items()
            ):
                if attempt_id not in contaminated_ids:
                    for event in matching:
                        errors.append(
                            f"event {event['event_id']}: generated-evidence "
                            "quarantine has no contaminated attempt"
                        )

            # The latest projection event commits to all mutable learner-model
            # tables.  This catches out-of-band edits that a valid event chain
            # alone cannot reveal.
            projection_learner_ids = {
                row["learner_id"]
                for row in connection.execute(
                    """SELECT learner_id FROM skill_states
                       UNION SELECT learner_id FROM misconception_beliefs
                       UNION SELECT learner_id FROM learner_skill_families
                       UNION SELECT learner_id FROM objective_states
                       UNION SELECT learner_id FROM objective_grid_states
                       UNION SELECT learner_id FROM learner_objective_families"""
                )
            }
            learners_with_projection_events = {
                row["learner_id"]
                for row in connection.execute(
                    """SELECT DISTINCT learner_id FROM events
                       WHERE event_type = 'LearnerProjectionAdvanced'
                         AND learner_id IS NOT NULL"""
                )
            }
            for learner_id in sorted(
                projection_learner_ids - learners_with_projection_events
            ):
                errors.append(
                    f"learner {learner_id}: mutable projection rows exist without "
                    "a LearnerProjectionAdvanced event"
                )
            latest_projection_events: dict[str, sqlite3.Row] = {}
            for event in events:
                if event["event_type"] == "LearnerProjectionAdvanced" and event[
                    "learner_id"
                ]:
                    latest_projection_events[event["learner_id"]] = event
            for learner_id, projection_event in latest_projection_events.items():
                payload = event_object(
                    projection_event, "payload_json", payload_cache
                )
                if payload is None or "projection_hash" not in payload:
                    # Pre-v5 events did not carry a projection commitment.
                    continue
                committed_hash = payload.get("projection_hash")
                if not isinstance(committed_hash, str):
                    errors.append(
                        f"learner {learner_id}: projection hash is not a string"
                    )
                    continue
                declared_hash_version = payload.get("projection_hash_version", 1)
                if (
                    type(declared_hash_version) is not int
                    or declared_hash_version not in {1, 2, 3}
                ):
                    errors.append(
                        f"learner {learner_id}: invalid projection hash version"
                    )
                    continue
                expected_hash_version = (
                    PROJECTION_HASH_VERSION_BY_EVENT_SCHEMA.get(
                        projection_event["schema_version"]
                    )
                )
                if expected_hash_version is None:
                    errors.append(
                        f"learner {learner_id}: unsupported projection event "
                        f"schema {projection_event['schema_version']}"
                    )
                    continue
                if declared_hash_version != expected_hash_version:
                    errors.append(
                        f"learner {learner_id}: projection event schema/hash "
                        "version mismatch"
                    )
                    continue
                if declared_hash_version == 1:
                    unbound_objective_projection = connection.execute(
                        """SELECT 1 FROM objective_states WHERE learner_id = ?
                           UNION ALL
                           SELECT 1 FROM objective_grid_states WHERE learner_id = ?
                           UNION ALL
                           SELECT 1 FROM learner_objective_families
                           WHERE learner_id = ?
                           LIMIT 1""",
                        (learner_id, learner_id, learner_id),
                    ).fetchone()
                    if unbound_objective_projection is not None:
                        errors.append(
                            f"learner {learner_id}: objective projection rows are "
                            "not bound by projection hash version 1"
                        )
                elif declared_hash_version == 2:
                    unbound_exact_projection = connection.execute(
                        """SELECT 1 FROM objective_grid_states
                           WHERE learner_id = ? LIMIT 1""",
                        (learner_id,),
                    ).fetchone()
                    if unbound_exact_projection is not None:
                        errors.append(
                            f"learner {learner_id}: exact objective posterior rows "
                            "are not bound by projection hash version 2"
                        )
                try:
                    actual_projection_hash = self.learner_projection_hash(
                        learner_id,
                        connection,
                        hash_version=declared_hash_version,
                    )
                except (
                    NotFoundError,
                    ValidationError,
                    sqlite3.DatabaseError,
                    TypeError,
                    ValueError,
                ) as exc:
                    errors.append(
                        f"learner {learner_id}: projection cannot be hashed ({exc})"
                    )
                    continue
                if actual_projection_hash != committed_hash:
                    errors.append(f"learner {learner_id}: projection hash mismatch")

            semantic_events = connection.execute(
                """SELECT * FROM events
                   WHERE event_type IN ('QuestionSelected', 'ResponseSubmitted')
                   ORDER BY recorded_at, event_id"""
            ).fetchall()
            question_selected: dict[str, list[sqlite3.Row]] = {}
            response_events_by_id: dict[str, sqlite3.Row] = {}
            response_events_by_decision: dict[str, list[sqlite3.Row]] = {}
            for event in semantic_events:
                payload = event_object(event, "payload_json", payload_cache)
                event_object(event, "metadata_json", metadata_cache)
                if event["event_type"] == "ResponseSubmitted":
                    response_events_by_id[event["event_id"]] = event
                if payload is None:
                    continue
                decision_id = payload.get("decision_id")
                if not isinstance(decision_id, str) or not decision_id:
                    errors.append(f"event {event['event_id']}: missing decision_id")
                    continue
                if event["event_type"] == "QuestionSelected":
                    question_selected.setdefault(decision_id, []).append(event)
                else:
                    response_events_by_decision.setdefault(decision_id, []).append(event)

            projection_events_by_response: dict[
                str, list[sqlite3.Row]
            ] = {}
            for event in connection.execute(
                """SELECT * FROM events
                   WHERE event_type='LearnerProjectionAdvanced'
                   ORDER BY stream_id, stream_version"""
            ).fetchall():
                payload = event_object(event, "payload_json", payload_cache)
                event_object(event, "metadata_json", metadata_cache)
                if payload is None:
                    continue
                response_event_id = payload.get("response_event_id")
                if type(response_event_id) is not str or not response_event_id:
                    errors.append(
                        f"event {event['event_id']}: missing response_event_id"
                    )
                    continue
                projection_events_by_response.setdefault(
                    response_event_id, []
                ).append(event)

            transition_events_by_projection: dict[
                str, list[sqlite3.Row]
            ] = {}
            for event in connection.execute(
                """SELECT * FROM events
                   WHERE event_type='RemediationTransitioned'
                   ORDER BY stream_id, stream_version"""
            ).fetchall():
                event_object(event, "payload_json", payload_cache)
                event_object(event, "metadata_json", metadata_cache)
                projection_event_id = event["causation_id"]
                if (
                    type(projection_event_id) is not str
                    or not projection_event_id
                ):
                    errors.append(
                        f"event {event['event_id']}: remediation transition "
                        "has no projection cause"
                    )
                    continue
                transition_events_by_projection.setdefault(
                    projection_event_id, []
                ).append(event)

            invalidation_events_by_decision: dict[str, list[sqlite3.Row]] = {}
            for event in connection.execute(
                """SELECT * FROM events WHERE event_type = 'DecisionInvalidated'
                   ORDER BY recorded_at, event_id"""
            ).fetchall():
                payload = event_object(event, "payload_json", payload_cache)
                event_object(event, "metadata_json", metadata_cache)
                if payload is None:
                    continue
                decision_id = payload.get("decision_id")
                if not isinstance(decision_id, str) or not decision_id:
                    errors.append(
                        f"event {event['event_id']}: missing decision_id"
                    )
                    continue
                invalidation_events_by_decision.setdefault(decision_id, []).append(
                    event
                )

            decision_rows = connection.execute(
                """SELECT d.*,
                          s.learner_id AS session_learner_id,
                          s.corpus_release_id AS session_release_id,
                          s.status AS session_status,
                          s.phase AS session_phase,
                          s.focus_concept_id AS session_focus_concept_id,
                          s.focus_misconception_id
                              AS session_focus_misconception_id,
                          s.focus_objective_id AS session_focus_objective_id,
                          s.revision AS current_session_revision,
                          learner.revision AS current_learner_revision,
                          q.version AS current_question_version,
                          q.content_hash AS current_question_content_hash,
                          q.family_id AS current_family_id,
                          rq.status AS release_question_status,
                          rq.evidence_weight AS release_evidence_weight,
                          rqo.objective_id AS release_question_objective_id,
                          rlo.objective_id AS release_focus_objective_id,
                          rfc.concept_id AS release_focus_concept_id,
                          rfm.misconception_id
                              AS release_focus_misconception_id,
                          focus_objective.primary_concept_id
                              AS focus_objective_primary_concept_id,
                          focus_objective.supporting_concept_ids_json
                              AS focus_objective_supporting_concept_ids_json,
                          focus_misconception.concept_id
                              AS focus_misconception_owner_concept_id
                   FROM decisions d
                   LEFT JOIN sessions s ON s.id = d.session_id
                   LEFT JOIN learners learner ON learner.id = d.learner_id
                   LEFT JOIN questions q ON q.id = d.question_id
                   LEFT JOIN release_questions rq
                     ON rq.release_id = d.corpus_release_id
                    AND rq.question_id = d.question_id
                   LEFT JOIN release_question_objectives rqo
                     ON rqo.release_id = d.corpus_release_id
                    AND rqo.question_id = d.question_id
                   LEFT JOIN release_learning_objectives rlo
                     ON rlo.release_id = d.corpus_release_id
                    AND rlo.objective_id = d.focus_objective_id
                   LEFT JOIN release_concepts rfc
                     ON rfc.release_id = d.corpus_release_id
                    AND rfc.concept_id = d.focus_concept_id
                   LEFT JOIN release_misconceptions rfm
                     ON rfm.release_id = d.corpus_release_id
                    AND rfm.misconception_id = d.focus_misconception_id
                   LEFT JOIN learning_objectives focus_objective
                     ON focus_objective.id = rlo.objective_id
                   LEFT JOIN misconceptions focus_misconception
                     ON focus_misconception.id = d.focus_misconception_id
                   ORDER BY d.created_at, d.id"""
            ).fetchall()
            decisions = {row["id"]: row for row in decision_rows}

            option_rows = connection.execute(
                """SELECT DISTINCT d.corpus_release_id, o.question_id,
                          o.option_id, o.text, o.is_correct, o.rationale,
                          o.misconception_id,
                          diagnostic.objective_id AS diagnostic_objective_id
                   FROM decisions d
                   JOIN options o ON o.question_id = d.question_id
                   LEFT JOIN release_option_objectives diagnostic
                     ON diagnostic.release_id = d.corpus_release_id
                    AND diagnostic.question_id = o.question_id
                    AND diagnostic.option_id = o.option_id"""
            ).fetchall()
            option_ids: dict[str, set[str]] = {}
            answer_keys: dict[tuple[str, str], bool] = {}
            option_snapshots: dict[
                tuple[str, str, str], dict[str, Any]
            ] = {}
            correct_option_snapshots: dict[
                tuple[str, str], list[dict[str, Any]]
            ] = {}
            for option in option_rows:
                option_ids.setdefault(option["question_id"], set()).add(option["option_id"])
                answer_keys[(option["question_id"], option["option_id"])] = bool(
                    option["is_correct"]
                )
                option_snapshot = {
                    "id": option["option_id"],
                    "text": option["text"],
                    "correct": bool(option["is_correct"]),
                    "rationale": option["rationale"],
                    "misconception_id": option["misconception_id"],
                }
                if option["diagnostic_objective_id"] is not None:
                    option_snapshot["diagnostic_objective_id"] = option[
                        "diagnostic_objective_id"
                    ]
                option_key = (
                    option["corpus_release_id"],
                    option["question_id"],
                    option["option_id"],
                )
                option_snapshots[option_key] = option_snapshot
                if option["is_correct"]:
                    correct_option_snapshots.setdefault(
                        (
                            option["corpus_release_id"],
                            option["question_id"],
                        ),
                        [],
                    ).append(
                        option_snapshot
                    )
            concept_ids = {
                row["id"] for row in connection.execute("SELECT id FROM concepts")
            }
            misconception_ids = {
                row["id"] for row in connection.execute("SELECT id FROM misconceptions")
            }
            objective_ids = {
                row["id"]
                for row in connection.execute(
                    "SELECT id FROM learning_objectives"
                )
            }

            def validate_focus_tuple(
                concept_id: Any,
                misconception_id: Any,
                objective_id: Any,
                label: str,
                release_id: str,
            ) -> None:
                try:
                    self.validate_release_focus_tuple(
                        release_id,
                        concept_id,
                        misconception_id,
                        objective_id,
                        connection=connection,
                        label=label,
                    )
                except ValidationError as exc:
                    errors.append(str(exc))

            for session_path_row in connection.execute(
                """SELECT id, corpus_release_id, remediation_path_json
                   FROM sessions ORDER BY id"""
            ).fetchall():
                label = f"session {session_path_row['id']} remediation path"
                path_value = json_value(
                    session_path_row["remediation_path_json"], label, list
                )
                if path_value is None:
                    continue
                for frame_index, frame in enumerate(path_value):
                    frame_label = f"{label} frame {frame_index}"
                    if not isinstance(frame, dict):
                        errors.append(f"{frame_label}: must be an object")
                        continue
                    expected_fields = {"concept_id", "misconception_id"}
                    if "objective_id" in frame:
                        expected_fields.add("objective_id")
                    if set(frame) != expected_fields:
                        errors.append(
                            f"{frame_label}: invalid fields"
                        )
                        continue
                    concept_id = frame.get("concept_id")
                    misconception_id = frame.get("misconception_id")
                    objective_id = frame.get("objective_id")
                    if concept_id not in concept_ids:
                        errors.append(
                            f"{frame_label}: invalid concept_id"
                        )
                    if (
                        misconception_id is not None
                        and misconception_id not in misconception_ids
                    ):
                        errors.append(
                            f"{frame_label}: invalid misconception_id"
                        )
                    if (
                        objective_id is not None
                        and objective_id not in objective_ids
                    ):
                        errors.append(
                            f"{frame_label}: invalid objective_id"
                        )
                    validate_focus_tuple(
                        concept_id,
                        misconception_id,
                        objective_id,
                        frame_label,
                        session_path_row["corpus_release_id"],
                    )
            option_question_keys = {
                (release_id, question_id)
                for release_id, question_id, _ in option_snapshots
            }
            for option_release_id, question_id in option_question_keys:
                correct_count = len(
                    correct_option_snapshots.get(
                        (option_release_id, question_id), []
                    )
                )
                if correct_count != 1:
                    errors.append(
                        f"release {option_release_id} question {question_id}: "
                        f"expected one correct option, found {correct_count}"
                    )

            decision_orders: dict[str, list[Any] | None] = {}
            selection_models: dict[str, str] = {}
            selection_times: dict[str, datetime] = {}
            for decision in decision_rows:
                decision_id = decision["id"]
                prefix = f"decision {decision_id}"
                if decision["session_learner_id"] is None:
                    errors.append(f"{prefix}: missing session")
                elif decision["learner_id"] != decision["session_learner_id"]:
                    errors.append(f"{prefix}: learner/session mismatch")
                if decision["session_release_id"] is None:
                    errors.append(f"{prefix}: session has no corpus release")
                elif decision["corpus_release_id"] != decision["session_release_id"]:
                    errors.append(f"{prefix}: corpus release/session mismatch")
                if decision["current_question_version"] is None:
                    errors.append(f"{prefix}: missing question")
                else:
                    if decision["question_version"] != decision["current_question_version"]:
                        errors.append(f"{prefix}: question version mismatch")
                    if (
                        decision["question_content_hash"]
                        != decision["current_question_content_hash"]
                    ):
                        errors.append(f"{prefix}: question content hash mismatch")
                if decision["release_question_status"] is None:
                    errors.append(f"{prefix}: question absent from pinned release")
                else:
                    if decision["question_status"] != decision["release_question_status"]:
                        errors.append(f"{prefix}: pinned question status mismatch")
                    if decision["evidence_weight"] != decision["release_evidence_weight"]:
                        errors.append(f"{prefix}: pinned evidence weight mismatch")
                if (
                    decision["question_objective_id"]
                    != decision["release_question_objective_id"]
                ):
                    errors.append(f"{prefix}: pinned question objective mismatch")
                if (
                    decision["focus_objective_id"] is not None
                    and decision["release_focus_objective_id"] is None
                ):
                    errors.append(f"{prefix}: focus objective is outside pinned release")
                if (
                    decision["focus_concept_id"] is not None
                    and decision["release_focus_concept_id"] is None
                ):
                    errors.append(f"{prefix}: focus concept is outside pinned release")
                if (
                    decision["focus_misconception_id"] is not None
                    and decision["release_focus_misconception_id"] is None
                ):
                    errors.append(
                        f"{prefix}: focus misconception is outside pinned release"
                    )
                if (
                    decision["focus_objective_id"] is not None
                    and decision["focus_concept_id"]
                    != decision["focus_objective_primary_concept_id"]
                ):
                    errors.append(
                        f"{prefix}: focus concept is not the objective's canonical owner"
                    )
                if decision["focus_misconception_id"] is not None:
                    misconception_owner = decision[
                        "focus_misconception_owner_concept_id"
                    ]
                    if decision["focus_objective_id"] is None:
                        if decision["focus_concept_id"] != misconception_owner:
                            errors.append(
                                f"{prefix}: focus misconception owner does not match focus concept"
                            )
                    else:
                        try:
                            supporting = set(
                                json.loads(
                                    decision[
                                        "focus_objective_supporting_concept_ids_json"
                                    ]
                                )
                            )
                        except (TypeError, ValueError):
                            supporting = set()
                        if misconception_owner not in {
                            decision["focus_objective_primary_concept_id"],
                            *supporting,
                        }:
                            errors.append(
                                f"{prefix}: focus misconception is outside the focus objective"
                            )
                if (
                    decision["consumed_at"] is None
                    and decision["invalidated_at"] is None
                    and decision["session_learner_id"] is not None
                ):
                    if decision["session_status"] != "active":
                        errors.append(f"{prefix}: pending decision has inactive session")
                    if decision["session_phase"] != decision["phase"]:
                        errors.append(f"{prefix}: pending decision/session phase mismatch")
                    if (
                        decision["session_focus_concept_id"]
                        != decision["focus_concept_id"]
                    ):
                        errors.append(
                            f"{prefix}: pending decision/session focus concept mismatch"
                        )
                    if (
                        decision["session_focus_misconception_id"]
                        != decision["focus_misconception_id"]
                    ):
                        errors.append(
                            f"{prefix}: pending decision/session focus misconception mismatch"
                        )
                    if (
                        decision["session_focus_objective_id"]
                        != decision["focus_objective_id"]
                    ):
                        errors.append(
                            f"{prefix}: pending decision/session focus objective mismatch"
                        )
                    if (
                        decision["current_session_revision"]
                        != decision["session_revision"] + 1
                    ):
                        errors.append(
                            f"{prefix}: pending decision/session revision mismatch"
                        )
                    if (
                        decision["current_learner_revision"]
                        != decision["learner_revision"]
                    ):
                        errors.append(
                            f"{prefix}: pending decision/learner revision mismatch"
                        )

                order = json_value(
                    decision["option_order_json"], f"{prefix} option order", list
                )
                decision_orders[decision_id] = order
                if order is not None:
                    expected_options = option_ids.get(decision["question_id"], set())
                    if not all(isinstance(option_id, str) for option_id in order):
                        errors.append(f"{prefix}: option order contains a non-string ID")
                    else:
                        ordered_options = set(order)
                        if len(order) != len(ordered_options):
                            errors.append(f"{prefix}: option order contains duplicates")
                        if ordered_options != expected_options:
                            errors.append(
                                f"{prefix}: option order does not match question options"
                            )

                selection_events = question_selected.get(decision_id, [])
                if len(selection_events) != 1:
                    errors.append(
                        f"{prefix}: expected one QuestionSelected event, found "
                        f"{len(selection_events)}"
                    )
                else:
                    selection = selection_events[0]
                    selection_payload = event_object(
                        selection, "payload_json", payload_cache
                    )
                    selection_metadata = event_object(
                        selection, "metadata_json", metadata_cache
                    )
                    selected_score = json_value(
                        decision["selected_score_json"],
                        f"{prefix} selected score",
                        dict,
                    )
                    if (
                        selection["stream_id"]
                        != f"learner:{decision['learner_id']}"
                        or selection["learner_id"] != decision["learner_id"]
                        or selection["session_id"] != decision["session_id"]
                    ):
                        errors.append(f"{prefix}: selection event envelope mismatch")
                    objective_aware = bool(
                        decision["question_objective_id"] is not None
                        or decision["focus_objective_id"] is not None
                    )
                    selection_model: str | None = None
                    expected_selection_schema: int | None = None
                    if selection_metadata is not None:
                        require_exact_fields(
                            selection_metadata,
                            QUESTION_SELECTED_METADATA_FIELDS,
                            f"{prefix} selection metadata",
                        )
                        candidate_model = selection_metadata.get(
                            "learner_model_version"
                        )
                        if (
                            type(candidate_model) is not str
                            or candidate_model not in SUPPORTED_MODEL_VERSIONS
                        ):
                            errors.append(
                                f"{prefix}: selection event has an unsupported "
                                "learner model"
                            )
                        else:
                            selection_model = candidate_model
                            selection_models[decision_id] = candidate_model
                            try:
                                expected_selection_schema = (
                                    question_selected_schema_for(
                                        candidate_model,
                                        objective_aware=objective_aware,
                                    )
                                )
                            except ValueError as exc:
                                errors.append(f"{prefix}: {exc}")
                        if (
                            selection_metadata.get("policy_version")
                            != decision["policy_version"]
                            or selection_metadata.get("corpus_release_id")
                            != decision["corpus_release_id"]
                        ):
                            errors.append(
                                f"{prefix}: selection metadata does not match decision"
                            )
                    if (
                        selection["schema_version"]
                        not in SUPPORTED_QUESTION_SELECTED_EVENT_SCHEMA_VERSIONS
                    ):
                        errors.append(
                            f"{prefix}: unsupported selection event schema"
                        )
                    elif (
                        expected_selection_schema is not None
                        and selection["schema_version"]
                        != expected_selection_schema
                    ):
                        errors.append(f"{prefix}: selection event schema mismatch")
                    selected_at = aware_timestamp(
                        selection["occurred_at"],
                        f"{prefix} selection event time",
                    )
                    if selected_at is not None:
                        selection_times[decision_id] = selected_at
                    if (
                        expected_selection_schema
                        == BOUND_QUESTION_SELECTED_EVENT_SCHEMA_VERSION
                        and selection["occurred_at"] != decision["created_at"]
                    ):
                        errors.append(
                            f"{prefix}: selection event time is not bound to decision"
                        )
                    if (
                        selection_payload is not None
                        and order is not None
                        and selected_score is not None
                    ):
                        require_exact_fields(
                            selection_payload,
                            (
                                QUESTION_SELECTED_OBJECTIVE_FIELDS
                                if objective_aware
                                else QUESTION_SELECTED_BASE_FIELDS
                            ),
                            f"{prefix} selection event",
                        )
                        expected_selection_payload = {
                                "decision_id": decision_id,
                                "question_id": decision["question_id"],
                                "phase": decision["phase"],
                                "candidate_count": decision["candidate_count"],
                                "candidate_digest": decision["candidate_digest"],
                                "propensity": decision["propensity"],
                                "score": selected_score,
                                "option_order": order,
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
                            expected_selection_payload.update(
                                {
                                    "question_objective_id": decision[
                                        "question_objective_id"
                                    ],
                                    "focus_objective_id": decision[
                                        "focus_objective_id"
                                    ],
                                }
                            )
                        compare_payload(
                            selection_payload,
                            expected_selection_payload,
                            f"{prefix} selection event",
                        )

            for decision_id, selection_events in question_selected.items():
                if decision_id not in decisions:
                    for event in selection_events:
                        errors.append(
                            f"event {event['event_id']}: QuestionSelected has no decision"
                        )

            session_rows_for_transitions = {
                row["id"]: row
                for row in connection.execute(
                    """SELECT id, learner_id, root_concept_id,
                              corpus_release_id, phase, focus_concept_id,
                              focus_misconception_id, focus_objective_id,
                              remediation_depth, remediation_path_json
                       FROM sessions ORDER BY id"""
                ).fetchall()
            }
            session_transition_states: dict[str, dict[str, Any]] = {}
            for session_id, session_row in (
                session_rows_for_transitions.items()
            ):
                started_events = connection.execute(
                    """SELECT * FROM events
                       WHERE event_type='SessionStarted' AND session_id=?
                       ORDER BY stream_version""",
                    (session_id,),
                ).fetchall()
                if len(started_events) != 1:
                    errors.append(
                        f"session {session_id}: expected one SessionStarted "
                        f"event, found {len(started_events)}"
                    )
                    continue
                started_event = started_events[0]
                started_payload = event_object(
                    started_event, "payload_json", payload_cache
                )
                started_metadata = event_object(
                    started_event, "metadata_json", metadata_cache
                )
                if started_payload is None:
                    continue
                if (
                    started_event["schema_version"] != 1
                    or started_event["stream_id"]
                    != f"learner:{session_row['learner_id']}"
                    or started_event["learner_id"]
                    != session_row["learner_id"]
                    or started_event["session_id"] != session_id
                ):
                    errors.append(
                        f"session {session_id}: SessionStarted event envelope mismatch"
                    )
                for payload_field, session_field in (
                    ("session_id", "id"),
                    ("root_concept_id", "root_concept_id"),
                    ("corpus_release_id", "corpus_release_id"),
                ):
                    if not same_json_value(
                        started_payload.get(payload_field),
                        session_row[session_field],
                    ):
                        errors.append(
                            f"session {session_id}: SessionStarted {payload_field} "
                            "does not match session"
                        )
                if (
                    started_metadata is not None
                    and not same_json_value(
                        started_metadata.get("corpus_release_id"),
                        session_row["corpus_release_id"],
                    )
                ):
                    errors.append(
                        f"session {session_id}: SessionStarted metadata corpus "
                        "release does not match session"
                    )
                initial_phase = started_payload.get("initial_phase")
                try:
                    SessionPhase(initial_phase)
                except (TypeError, ValueError):
                    errors.append(
                        f"session {session_id}: invalid initial phase"
                    )
                    continue
                session_transition_states[session_id] = {
                    "phase": initial_phase,
                    "focus_concept_id": None,
                    "focus_misconception_id": None,
                    "focus_objective_id": None,
                    "remediation_depth": 0,
                    "remediation_path": [],
                }

            attempt_rows = connection.execute(
                """SELECT a.*, q.version AS current_question_version,
                          q.family_id AS current_family_id
                   FROM attempts a
                   LEFT JOIN questions q ON q.id = a.question_id
                   LEFT JOIN events response_event
                     ON response_event.event_id = a.event_id
                   ORDER BY response_event.stream_id,
                            response_event.stream_version, a.id"""
            ).fetchall()
            attempts_by_decision: dict[str, list[sqlite3.Row]] = {}
            attempts_by_event: dict[str, sqlite3.Row] = {}
            for attempt in attempt_rows:
                attempt_id = attempt["id"]
                prefix = f"attempt {attempt_id}"
                if attempt["response_ms"] is not None and (
                    type(attempt["response_ms"]) is not int
                    or not 0 <= attempt["response_ms"] <= MAX_RESPONSE_MS
                ):
                    errors.append(f"{prefix}: response_ms is out of bounds")
                if attempt["confidence"] is not None and (
                    isinstance(attempt["confidence"], bool)
                    or not isinstance(attempt["confidence"], (int, float))
                    or not isfinite(float(attempt["confidence"]))
                    or not 0.0 <= float(attempt["confidence"]) <= 1.0
                ):
                    errors.append(f"{prefix}: confidence is out of bounds")
                if (
                    type(attempt["hint_count"]) is not int
                    or not 0 <= attempt["hint_count"] <= MAX_HINT_COUNT
                ):
                    errors.append(f"{prefix}: hint_count is out of bounds")
                attempts_by_decision.setdefault(attempt["decision_id"], []).append(attempt)
                attempts_by_event[attempt["event_id"]] = attempt
                decision = decisions.get(attempt["decision_id"])
                if decision is None:
                    errors.append(f"{prefix}: missing decision")
                    continue
                prior_transition_state = session_transition_states.get(
                    attempt["session_id"]
                )
                if prior_transition_state is not None:
                    for decision_field, state_field in (
                        ("phase", "phase"),
                        ("focus_concept_id", "focus_concept_id"),
                        (
                            "focus_misconception_id",
                            "focus_misconception_id",
                        ),
                        ("focus_objective_id", "focus_objective_id"),
                    ):
                        if not same_json_value(
                            decision[decision_field],
                            prior_transition_state[state_field],
                        ):
                            errors.append(
                                f"{prefix}: decision {decision_field} does not "
                                "match prior session transition"
                            )

                field_pairs = (
                    ("session_id", "session_id"),
                    ("learner_id", "learner_id"),
                    ("question_id", "question_id"),
                    ("question_version", "question_version"),
                )
                for attempt_field, decision_field in field_pairs:
                    if attempt[attempt_field] != decision[decision_field]:
                        errors.append(
                            f"{prefix}: {attempt_field} does not match decision"
                        )
                if attempt["question_version"] != attempt["current_question_version"]:
                    errors.append(f"{prefix}: current question version mismatch")
                if attempt["family_id"] != attempt["current_family_id"]:
                    errors.append(f"{prefix}: question family mismatch")

                presented_order = json_value(
                    attempt["presented_order_json"], f"{prefix} presented order", list
                )
                if presented_order != decision_orders.get(attempt["decision_id"]):
                    errors.append(f"{prefix}: presented order does not match decision")
                selected_option_id = attempt["selected_option_id"]
                if selected_option_id is not None and (
                    presented_order is None or selected_option_id not in presented_order
                ):
                    errors.append(f"{prefix}: selected option was not presented")
                expected_correct = (
                    answer_keys.get((attempt["question_id"], selected_option_id), False)
                    if selected_option_id is not None
                    else False
                )
                if bool(attempt["is_correct"]) != expected_correct:
                    errors.append(f"{prefix}: correctness does not match answer key")

                command_payload = {
                    "decision_id": attempt["decision_id"],
                    "question_id": attempt["question_id"],
                    "question_version": attempt["question_version"],
                    "selected_option_id": selected_option_id,
                    "is_correct": bool(attempt["is_correct"]),
                    "confidence": attempt["confidence"],
                    "response_ms": attempt["response_ms"],
                    "hint_count": attempt["hint_count"],
                    "feedback_shown": bool(attempt["feedback_shown"]),
                    "presented_order": presented_order,
                }
                if presented_order is not None:
                    try:
                        expected_attempt_command_hash = _content_hash(
                            command_payload
                        )
                    except (TypeError, ValueError) as exc:
                        errors.append(
                            f"{prefix}: command payload cannot be hashed ({exc})"
                        )
                    else:
                        if (
                            attempt["command_hash"]
                            != expected_attempt_command_hash
                        ):
                            errors.append(f"{prefix}: command hash mismatch")

                outcome: dict[str, Any] | None = None
                if attempt["outcome_json"] is not None:
                    outcome = json_value(
                        attempt["outcome_json"], f"{prefix} outcome", dict
                    )
                    if outcome is not None:
                        correct_options = correct_option_snapshots.get(
                            (
                                decision["corpus_release_id"],
                                attempt["question_id"],
                            ),
                            [],
                        )
                        expected_selected = (
                            option_snapshots.get(
                                (
                                    decision["corpus_release_id"],
                                    attempt["question_id"],
                                    selected_option_id,
                                )
                            )
                            if selected_option_id is not None
                            else None
                        )
                        expected_correct_option = (
                            correct_options[0] if len(correct_options) == 1 else None
                        )
                        # Outcomes written before diagnostic-objective payloads
                        # were introduced remain valid. When the field is
                        # present, however, it must match the decision's pinned
                        # release mapping exactly.
                        actual_selected = outcome.get("selected_option")
                        if (
                            isinstance(actual_selected, dict)
                            and "diagnostic_objective_id"
                            not in actual_selected
                            and isinstance(expected_selected, dict)
                        ):
                            expected_selected = {
                                key: value
                                for key, value in expected_selected.items()
                                if key != "diagnostic_objective_id"
                            }
                        compare_payload(
                            outcome,
                            {
                                "interaction_id": attempt_id,
                                "correct": bool(attempt["is_correct"]),
                                "selected_option": expected_selected,
                                "correct_option": expected_correct_option,
                            },
                            f"{prefix} outcome",
                        )
                        next_phase = outcome.get("next_phase")
                        try:
                            SessionPhase(next_phase)
                        except (TypeError, ValueError):
                            errors.append(f"{prefix} outcome: invalid next_phase")
                        for field, known_ids in (
                            ("focus_concept_id", concept_ids),
                            ("focus_misconception_id", misconception_ids),
                        ):
                            if field not in outcome:
                                errors.append(f"{prefix} outcome: missing {field}")
                                continue
                            value = outcome.get(field)
                            if value is not None and value not in known_ids:
                                errors.append(f"{prefix} outcome: invalid {field}")
                        if "focus_objective_id" in outcome:
                            focus_objective_id = outcome.get("focus_objective_id")
                            if (
                                focus_objective_id is not None
                                and focus_objective_id not in objective_ids
                            ):
                                errors.append(
                                    f"{prefix} outcome: invalid focus_objective_id"
                                )
                        else:
                            focus_objective_id = None
                        validate_focus_tuple(
                            outcome.get("focus_concept_id"),
                            outcome.get("focus_misconception_id"),
                            focus_objective_id,
                            f"{prefix} outcome",
                            decision["corpus_release_id"],
                        )
                        remediation_path = outcome.get("remediation_path", [])
                        if not isinstance(remediation_path, list):
                            errors.append(
                                f"{prefix} outcome: remediation_path must be a list"
                            )
                        else:
                            for frame_index, frame in enumerate(
                                remediation_path
                            ):
                                frame_label = (
                                    f"{prefix} outcome remediation_path "
                                    f"frame {frame_index}"
                                )
                                if not isinstance(frame, dict):
                                    errors.append(
                                        f"{frame_label}: must be an object"
                                    )
                                    continue
                                if set(frame) - {
                                    "concept_id",
                                    "misconception_id",
                                    "objective_id",
                                }:
                                    errors.append(
                                        f"{frame_label}: contains unexpected fields"
                                    )
                                if "concept_id" not in frame:
                                    errors.append(
                                        f"{frame_label}: missing concept_id"
                                    )
                                    continue
                                validate_focus_tuple(
                                    frame.get("concept_id"),
                                    frame.get("misconception_id"),
                                    frame.get("objective_id"),
                                    frame_label,
                                    decision["corpus_release_id"],
                                )
                        state_changes = outcome.get("state_changes")
                        if not isinstance(state_changes, list) or not all(
                            isinstance(change, dict) for change in state_changes
                        ):
                            errors.append(
                                f"{prefix} outcome: state_changes must be a list of objects"
                            )

                response = response_events_by_id.get(attempt["event_id"])
                if response is None:
                    errors.append(f"{prefix}: missing ResponseSubmitted event")
                    continue
                response_payload = event_object(response, "payload_json", payload_cache)
                response_metadata = event_object(response, "metadata_json", metadata_cache)
                if response["learner_id"] != attempt["learner_id"]:
                    errors.append(f"{prefix}: response event learner mismatch")
                if response["session_id"] != attempt["session_id"]:
                    errors.append(f"{prefix}: response event session mismatch")
                if response["causation_id"] != attempt["decision_id"]:
                    errors.append(f"{prefix}: response event causation mismatch")
                if response["occurred_at"] != attempt["answered_at"]:
                    errors.append(f"{prefix}: response event time mismatch")
                selection_events = question_selected.get(
                    attempt["decision_id"], []
                )
                selection = (
                    selection_events[0]
                    if len(selection_events) == 1
                    else None
                )
                if selection is not None and (
                    selection["stream_id"] != response["stream_id"]
                    or selection["stream_version"] >= response["stream_version"]
                ):
                    errors.append(
                        f"{prefix}: response does not follow its selection boundary"
                    )
                response_model = (
                    response_metadata.get("learner_model_version")
                    if response_metadata is not None
                    else None
                )
                selection_model = selection_models.get(attempt["decision_id"])
                if (
                    selection_model is not None
                    and response_model != selection_model
                ):
                    errors.append(
                        f"{prefix}: response learner model does not match selection"
                    )
                if (
                    response_metadata is not None
                    and response_model
                    in AUTHORITATIVE_RESPONSE_WINDOW_MODEL_VERSIONS
                ):
                    selected_at = selection_times.get(attempt["decision_id"])
                    if selected_at is None:
                        errors.append(
                            f"{prefix}: authoritative selection time is unavailable"
                        )
                    try:
                        authoritative_window = response_window(
                            selected_at=selected_at,
                            answered_at=aware_timestamp(
                                attempt["answered_at"],
                                f"{prefix} answer time",
                            ),
                            response_ms=attempt["response_ms"],
                        )
                    except (TypeError, ValueError, OverflowError) as exc:
                        errors.append(
                            f"{prefix}: invalid authoritative response window "
                            f"({exc})"
                        )
                    else:
                        if not authoritative_window.consistent:
                            errors.append(
                                f"{prefix}: response_ms exceeds the "
                                "authoritative selection-to-answer window"
                            )
                if response_payload is not None and presented_order is not None:
                    compare_payload(
                        response_payload,
                        command_payload,
                        f"{prefix} response event",
                        exact=True,
                    )
                if response_metadata is not None:
                    expected_response_metadata = {
                        "policy_version": decision["policy_version"],
                        "learner_model_version": selection_model,
                        "corpus_release_id": decision["corpus_release_id"],
                        "question_content_hash": decision["question_content_hash"],
                        "question_status": decision["question_status"],
                        "evidence_weight": decision["evidence_weight"],
                        "selection_learner_revision": decision["learner_revision"],
                        "application_learner_revision": decision["learner_revision"],
                    }
                    if response["schema_version"] == 2:
                        expected_response_metadata[
                            MISCONCEPTION_ALGORITHM_METADATA_KEY
                        ] = MISCONCEPTION_ALGORITHM_VERSION
                    compare_payload(
                        response_metadata,
                        expected_response_metadata,
                        f"{prefix} response metadata",
                        exact=True,
                    )
                if (
                    outcome is not None
                    and response_model
                    in COMPLETE_TRANSITION_OUTCOME_MODEL_VERSIONS
                ):
                    outcome_fields = set(outcome)
                    missing_outcome_fields = (
                        ATTEMPT_OUTCOME_COMPLETE_FIELDS - outcome_fields
                    )
                    unexpected_outcome_fields = outcome_fields - (
                        ATTEMPT_OUTCOME_COMPLETE_FIELDS
                        | ATTEMPT_OUTCOME_OPTIONAL_FIELDS
                    )
                    if missing_outcome_fields:
                        errors.append(
                            f"{prefix} outcome: missing fields "
                            f"{sorted(missing_outcome_fields)!r}"
                        )
                    if unexpected_outcome_fields:
                        errors.append(
                            f"{prefix} outcome: unexpected fields "
                            f"{sorted(unexpected_outcome_fields)!r}"
                        )

                projection_events = projection_events_by_response.get(
                    response["event_id"], []
                )
                if len(projection_events) != 1:
                    errors.append(
                        f"{prefix}: expected one LearnerProjectionAdvanced "
                        f"event, found {len(projection_events)}"
                    )
                else:
                    projection = projection_events[0]
                    projection_payload = event_object(
                        projection, "payload_json", payload_cache
                    )
                    projection_metadata = event_object(
                        projection, "metadata_json", metadata_cache
                    )
                    if (
                        projection["stream_id"] != response["stream_id"]
                        or projection["learner_id"] != attempt["learner_id"]
                        or projection["session_id"] != attempt["session_id"]
                        or projection["causation_id"] != response["event_id"]
                        or projection["occurred_at"] != response["occurred_at"]
                        or projection["stream_version"]
                        != response["stream_version"] + 1
                    ):
                        errors.append(
                            f"{prefix}: projection event envelope mismatch"
                        )
                    if projection_payload is not None:
                        remediation_depth = projection_payload.get(
                            "remediation_depth"
                        )
                        if (
                            type(remediation_depth) is not int
                            or not 0
                            <= remediation_depth
                            <= MAX_REMEDIATION_DEPTH
                        ):
                            errors.append(
                                f"{prefix}: projection remediation depth is "
                                "out of bounds"
                            )
                        projection_path = projection_payload.get(
                            "remediation_path"
                        )
                        if (
                            type(projection_path) is not list
                            or len(projection_path) > MAX_REMEDIATION_DEPTH
                        ):
                            errors.append(
                                f"{prefix}: projection remediation path is invalid"
                            )
                        else:
                            for frame_index, frame in enumerate(projection_path):
                                frame_label = (
                                    f"{prefix} projection remediation path "
                                    f"frame {frame_index}"
                                )
                                if type(frame) is not dict or set(frame) - {
                                    "concept_id",
                                    "misconception_id",
                                    "objective_id",
                                } or "concept_id" not in frame:
                                    errors.append(
                                        f"{frame_label}: invalid fields"
                                    )
                                    continue
                                validate_focus_tuple(
                                    frame.get("concept_id"),
                                    frame.get("misconception_id"),
                                    frame.get("objective_id"),
                                    frame_label,
                                    decision["corpus_release_id"],
                                )
                        validate_focus_tuple(
                            projection_payload.get("focus_concept_id"),
                            projection_payload.get("focus_misconception_id"),
                            projection_payload.get("focus_objective_id"),
                            f"{prefix} projection focus",
                            decision["corpus_release_id"],
                        )
                        expected_projection = {
                            "response_event_id": response["event_id"],
                            "corpus_release_id": decision[
                                "corpus_release_id"
                            ],
                            "learner_revision": decision[
                                "learner_revision"
                            ]
                            + 1,
                        }
                        if outcome is not None:
                            expected_projection.update(
                                {
                                    "state_changes": outcome.get(
                                        "state_changes"
                                    ),
                                    "phase": outcome.get("next_phase"),
                                    "focus_concept_id": outcome.get(
                                        "focus_concept_id"
                                    ),
                                    "focus_misconception_id": outcome.get(
                                        "focus_misconception_id"
                                    ),
                                }
                            )
                            if projection["schema_version"] >= 2:
                                expected_projection.update(
                                    {
                                        "transition_reason": outcome.get(
                                            "transition_reason"
                                        ),
                                        "boundary_decision": outcome.get(
                                            "boundary_decision"
                                        ),
                                    }
                                )
                            if projection["schema_version"] >= 3:
                                expected_projection.update(
                                    {
                                        "question_objective_id": decision[
                                            "question_objective_id"
                                        ],
                                        "focus_objective_id": outcome.get(
                                            "focus_objective_id"
                                        ),
                                    }
                                )
                            if (
                                response_model
                                in COMPLETE_TRANSITION_OUTCOME_MODEL_VERSIONS
                            ):
                                if not {
                                    "remediation_depth",
                                    "remediation_path",
                                }.issubset(outcome):
                                    errors.append(
                                        f"{prefix}: complete transition outcome "
                                        "is missing remediation state"
                                    )
                                else:
                                    expected_projection.update(
                                        {
                                            "remediation_depth": outcome[
                                                "remediation_depth"
                                            ],
                                            "remediation_path": outcome[
                                                "remediation_path"
                                            ],
                                        }
                                    )
                            compare_payload(
                                projection_payload,
                                expected_projection,
                                f"{prefix} projection event",
                            )
                        elif (
                            response_model
                            in COMPLETE_TRANSITION_OUTCOME_MODEL_VERSIONS
                        ):
                            errors.append(
                                f"{prefix}: complete transition outcome is missing"
                            )
                        next_transition_state = {
                            "phase": projection_payload.get("phase"),
                            "focus_concept_id": projection_payload.get(
                                "focus_concept_id"
                            ),
                            "focus_misconception_id": projection_payload.get(
                                "focus_misconception_id"
                            ),
                            "focus_objective_id": projection_payload.get(
                                "focus_objective_id"
                            ),
                            "remediation_depth": projection_payload.get(
                                "remediation_depth"
                            ),
                            "remediation_path": projection_payload.get(
                                "remediation_path"
                            ),
                        }
                        if prior_transition_state is not None:
                            transition_changed = any(
                                not same_json_value(
                                    next_transition_state[field],
                                    prior_transition_state[field],
                                )
                                for field in next_transition_state
                            )
                            transition_events = (
                                transition_events_by_projection.get(
                                    projection["event_id"], []
                                )
                            )
                            expected_transition_count = int(transition_changed)
                            if len(transition_events) != expected_transition_count:
                                errors.append(
                                    f"{prefix}: expected "
                                    f"{expected_transition_count} "
                                    "RemediationTransitioned event, found "
                                    f"{len(transition_events)}"
                                )
                            elif transition_events:
                                transition_event = transition_events[0]
                                transition_payload = event_object(
                                    transition_event,
                                    "payload_json",
                                    payload_cache,
                                )
                                transition_metadata = event_object(
                                    transition_event,
                                    "metadata_json",
                                    metadata_cache,
                                )
                                expected_transition_schema = {
                                    1: 1,
                                    2: 2,
                                    3: 3,
                                    4: 3,
                                }.get(projection["schema_version"])
                                if expected_transition_schema is None:
                                    errors.append(
                                        f"{prefix}: cannot bind remediation "
                                        "transition to unsupported projection schema"
                                    )
                                elif (
                                    transition_event["schema_version"]
                                    != expected_transition_schema
                                    or transition_event["stream_id"]
                                    != projection["stream_id"]
                                    or transition_event["learner_id"]
                                    != attempt["learner_id"]
                                    or transition_event["session_id"]
                                    != attempt["session_id"]
                                    or transition_event["occurred_at"]
                                    != response["occurred_at"]
                                    or transition_event["stream_version"]
                                    != projection["stream_version"] + 1
                                ):
                                    errors.append(
                                        f"{prefix}: remediation transition "
                                        "event envelope mismatch"
                                    )
                                if transition_payload is not None:
                                    expected_transition_payload = {
                                        "from_phase": prior_transition_state[
                                            "phase"
                                        ],
                                        "to_phase": next_transition_state[
                                            "phase"
                                        ],
                                        "focus_concept_id": (
                                            next_transition_state[
                                                "focus_concept_id"
                                            ]
                                        ),
                                        "focus_misconception_id": (
                                            next_transition_state[
                                                "focus_misconception_id"
                                            ]
                                        ),
                                        "remediation_depth": (
                                            next_transition_state[
                                                "remediation_depth"
                                            ]
                                        ),
                                        "remediation_path": (
                                            next_transition_state[
                                                "remediation_path"
                                            ]
                                        ),
                                        "pedagogical_role": decision[
                                            "pedagogical_role"
                                        ],
                                        "focus_valid": bool(
                                            decision["focus_valid"]
                                        ),
                                        "unguided": (
                                            attempt["hint_count"] == 0
                                        ),
                                    }
                                    expected_transition_fields = (
                                        REMEDIATION_TRANSITION_FIELDS_V1
                                    )
                                    if expected_transition_schema in {2, 3}:
                                        expected_transition_payload.update(
                                            {
                                                "transition_reason": (
                                                    projection_payload.get(
                                                        "transition_reason"
                                                    )
                                                ),
                                                "boundary_decision": (
                                                    projection_payload.get(
                                                        "boundary_decision"
                                                    )
                                                ),
                                            }
                                        )
                                        expected_transition_fields = (
                                            REMEDIATION_TRANSITION_FIELDS
                                        )
                                    if expected_transition_schema == 3:
                                        expected_transition_payload[
                                            "focus_objective_id"
                                        ] = next_transition_state[
                                            "focus_objective_id"
                                        ]
                                        expected_transition_fields = (
                                            REMEDIATION_TRANSITION_OBJECTIVE_FIELDS
                                        )
                                    require_exact_fields(
                                        transition_payload,
                                        expected_transition_fields,
                                        f"{prefix} remediation transition",
                                    )
                                    compare_payload(
                                        transition_payload,
                                        expected_transition_payload,
                                        f"{prefix} remediation transition",
                                    )
                                if transition_metadata is not None:
                                    require_exact_fields(
                                        transition_metadata,
                                        REMEDIATION_TRANSITION_METADATA_FIELDS,
                                        f"{prefix} remediation transition metadata",
                                    )
                                    compare_payload(
                                        transition_metadata,
                                        {
                                            "policy_version": decision[
                                                "policy_version"
                                            ],
                                            "corpus_release_id": decision[
                                                "corpus_release_id"
                                            ],
                                        },
                                        f"{prefix} remediation transition metadata",
                                    )
                            session_transition_states[
                                attempt["session_id"]
                            ] = next_transition_state
                    if projection_metadata is not None:
                        expected_projection_metadata = {
                            "learner_model_version": response_model,
                            "corpus_release_id": decision[
                                "corpus_release_id"
                            ],
                            "evidence_weight": decision["evidence_weight"],
                        }
                        if response["schema_version"] == 2:
                            expected_projection_metadata[
                                MISCONCEPTION_ALGORITHM_METADATA_KEY
                            ] = MISCONCEPTION_ALGORITHM_VERSION
                        compare_payload(
                            projection_metadata,
                            expected_projection_metadata,
                            f"{prefix} projection metadata",
                            exact=True,
                        )

            for decision in decision_rows:
                projected = attempts_by_decision.get(decision["id"], [])
                if len(projected) > 1:
                    errors.append(
                        f"decision {decision['id']}: multiple attempt projections"
                    )
                if projected:
                    if decision["consumed_at"] is None:
                        errors.append(
                            f"decision {decision['id']}: attempt exists but decision is pending"
                        )
                    elif decision["consumed_at"] != projected[0]["answered_at"]:
                        errors.append(
                            f"decision {decision['id']}: consumed time mismatch"
                        )
                elif decision["consumed_at"] is not None:
                    errors.append(
                        f"decision {decision['id']}: consumed without an attempt"
                    )

                invalidations = invalidation_events_by_decision.get(
                    decision["id"], []
                )
                if decision["invalidated_at"] is not None:
                    if decision["consumed_at"] is not None or projected:
                        errors.append(
                            f"decision {decision['id']}: invalidated decision was consumed"
                        )
                    if not decision["invalidation_reason"]:
                        errors.append(
                            f"decision {decision['id']}: missing invalidation reason"
                        )
                    if len(invalidations) != 1:
                        errors.append(
                            f"decision {decision['id']}: expected one invalidation event, "
                            f"found {len(invalidations)}"
                        )
                    else:
                        invalidation = invalidations[0]
                        payload = event_object(
                            invalidation, "payload_json", payload_cache
                        )
                        if invalidation["learner_id"] != decision["learner_id"]:
                            errors.append(
                                f"decision {decision['id']}: invalidation learner mismatch"
                            )
                        if invalidation["session_id"] != decision["session_id"]:
                            errors.append(
                                f"decision {decision['id']}: invalidation session mismatch"
                            )
                        if invalidation["causation_id"] != decision["id"]:
                            errors.append(
                                f"decision {decision['id']}: invalidation causation mismatch"
                            )
                        if invalidation["occurred_at"] != decision["invalidated_at"]:
                            errors.append(
                                f"decision {decision['id']}: invalidation time mismatch"
                            )
                        if payload is not None:
                            compare_payload(
                                payload,
                                {
                                    "decision_id": decision["id"],
                                    "reason": decision["invalidation_reason"],
                                    "selection_learner_revision": decision[
                                        "learner_revision"
                                    ],
                                },
                                f"decision {decision['id']} invalidation event",
                            )
                            current_revision = payload.get(
                                "current_learner_revision"
                            )
                            valid_revision = (
                                isinstance(current_revision, int)
                                and not isinstance(current_revision, bool)
                                and (
                                    current_revision > decision["learner_revision"]
                                    if decision["invalidation_reason"]
                                    == "learner_projection_advanced"
                                    else current_revision
                                    >= decision["learner_revision"]
                                )
                            )
                            if not valid_revision:
                                errors.append(
                                    f"decision {decision['id']}: invalidation revision did not advance"
                                )
                elif invalidations:
                    errors.append(
                        f"decision {decision['id']}: invalidation event without invalidated state"
                    )
                elif decision["invalidation_reason"] is not None:
                    errors.append(
                        f"decision {decision['id']}: reason without invalidation time"
                    )

                response_count = len(response_events_by_decision.get(decision["id"], []))
                if response_count != len(projected):
                    errors.append(
                        f"decision {decision['id']}: {response_count} response events for "
                        f"{len(projected)} attempts"
                    )

            for session_id, expected_state in (
                session_transition_states.items()
            ):
                current_session = session_rows_for_transitions[session_id]
                current_path = json_value(
                    current_session["remediation_path_json"],
                    f"session {session_id} current remediation path",
                    list,
                )
                current_state = {
                    "phase": current_session["phase"],
                    "focus_concept_id": current_session[
                        "focus_concept_id"
                    ],
                    "focus_misconception_id": current_session[
                        "focus_misconception_id"
                    ],
                    "focus_objective_id": current_session[
                        "focus_objective_id"
                    ],
                    "remediation_depth": current_session[
                        "remediation_depth"
                    ],
                    "remediation_path": current_path,
                }
                if not same_json_value(current_state, expected_state):
                    errors.append(
                        f"session {session_id}: current remediation state does "
                        "not match event history"
                    )

            for event_id, event in response_events_by_id.items():
                if event_id not in attempts_by_event:
                    errors.append(
                        f"event {event_id}: ResponseSubmitted has no attempt projection"
                    )
            for response_event_id, projection_events in (
                projection_events_by_response.items()
            ):
                if response_event_id not in response_events_by_id:
                    for projection_event in projection_events:
                        errors.append(
                            f"event {projection_event['event_id']}: "
                            "LearnerProjectionAdvanced has no response event"
                        )
            known_projection_event_ids = {
                projection_event["event_id"]
                for projection_events in projection_events_by_response.values()
                for projection_event in projection_events
            }
            for projection_event_id, transition_events in (
                transition_events_by_projection.items()
            ):
                if projection_event_id not in known_projection_event_ids:
                    for transition_event in transition_events:
                        errors.append(
                            f"event {transition_event['event_id']}: "
                            "RemediationTransitioned has no projection event"
                        )
            for decision_id, invalidations in invalidation_events_by_decision.items():
                if decision_id not in decisions:
                    for event in invalidations:
                        errors.append(
                            f"event {event['event_id']}: DecisionInvalidated has no decision"
                        )
            # Productive-skill observations live in a separate shadow-only
            # projection, but they share this event stream and integrity
            # boundary.  Import locally to avoid a store/service import cycle.
            from .performance_ledger import performance_integrity_errors
            from .policy_shadow import policy_shadow_integrity_errors

            errors.extend(performance_integrity_errors(connection))
            errors.extend(policy_shadow_integrity_errors(connection))
            performance_attempt_count = connection.execute(
                "SELECT COUNT(*) AS n FROM performance_attempts"
            ).fetchone()["n"]
            performance_action_count = connection.execute(
                "SELECT COUNT(*) AS n FROM performance_actions"
            ).fetchone()["n"]
            task_evaluation_count = connection.execute(
                "SELECT COUNT(*) AS n FROM task_evaluations"
            ).fetchone()["n"]
            shadow_evidence_bundle_count = connection.execute(
                "SELECT COUNT(*) AS n FROM shadow_evidence_bundles"
            ).fetchone()["n"]
            policy_shadow_evaluation_count = connection.execute(
                "SELECT COUNT(*) AS n FROM policy_shadow_evaluations"
            ).fetchone()["n"]
        return {
            "ok": not errors,
            "event_count": len(events),
            "stream_count": len(expected_version),
            "learning_action_count": len(action_rows),
            "learning_artifact_count": len(artifact_rows),
            "performance_attempt_count": performance_attempt_count,
            "performance_action_count": performance_action_count,
            "task_evaluation_count": task_evaluation_count,
            "shadow_evidence_bundle_count": shadow_evidence_bundle_count,
            "policy_shadow_evaluation_count": policy_shadow_evaluation_count,
            "errors": errors,
            "foreign_key_failures": foreign_key_failures,
            "quick_check": quick_check,
        }
