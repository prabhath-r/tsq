# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from .evidence import (
    ActionKind,
    ActionPhase,
    LearningAction,
    canonical_json,
    summarize_actions,
)
from .errors import ConflictError, NotFoundError, ValidationError
from .graph import KnowledgeGraph
from .models import (
    MAX_HINT_COUNT,
    MAX_RESPONSE_MS,
    CandidateScore,
    Concept,
    ConceptEdge,
    ConceptWeight,
    Domain,
    Misconception,
    MisconceptionBelief,
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
from .quality import audit_corpus


SCHEMA_VERSION = 8

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
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
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

    def initialize(self) -> None:
        connection = self.connect()
        try:
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
            starting_version = current_version
            if current_version > SCHEMA_VERSION:
                raise ConflictError(
                    f"Database schema is {current_version}; engine expects at most {SCHEMA_VERSION}."
                )
            if current_version < SCHEMA_VERSION:
                # Defensive for databases that were manually downgraded or
                # produced by a prerelease build which installed protections
                # before recording the final schema version.  Migrations may
                # legitimately backfill immutable fields; protections return
                # after every migration succeeds.
                self._drop_corpus_registry_triggers(connection)
                self._drop_release_snapshot_triggers(connection)
                self._drop_v6_authoring_triggers(connection)
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(DDL)
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
            connection.commit()
        finally:
            connection.close()

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
        for table in ("concepts", "misconceptions", "sources"):
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
                for table in ("concepts", "misconceptions", "sources")
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
        connection.executescript(
            """CREATE TRIGGER events_no_update
               BEFORE UPDATE ON events BEGIN
                   SELECT RAISE(ABORT, 'events are append-only');
               END;
               CREATE TRIGGER events_no_delete
               BEFORE DELETE ON events BEGIN
                   SELECT RAISE(ABORT, 'events are append-only');
               END;"""
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
    def _install_v8_learning_action_triggers(
        connection: sqlite3.Connection,
    ) -> None:
        """Bind immutable action projections to their semantic events."""
        connection.executescript(
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
                      AND selection_event.schema_version = 1
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
            """
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

        connection.executescript(
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
            """
        )

    @staticmethod
    def _drop_v6_authoring_triggers(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            DROP TRIGGER IF EXISTS generation_jobs_validate_insert;
            DROP TRIGGER IF EXISTS generation_jobs_validate_update;
            DROP TRIGGER IF EXISTS generation_jobs_no_delete;
            DROP TRIGGER IF EXISTS generation_job_runs_validate_insert;
            DROP TRIGGER IF EXISTS generation_job_runs_validate_update;
            DROP TRIGGER IF EXISTS generation_job_runs_no_delete;
            """
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
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
        # Canonicalization is CPU work and must not extend SQLite's single-writer
        # critical section. Reuse these exact hashes for registry inserts and the
        # immutable release manifest.
        concept_hashes = {
            concept.id: concept_content_hash(concept) for concept in concepts
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
                    "corpus_release_id": release_id,
                    "bundle_hash": bundle_hash,
                },
                metadata={"schema_version": SCHEMA_VERSION, "corpus_release_id": release_id},
            )
        return {
            "concepts": len(concepts),
            "edges": len(edges),
            "misconceptions": len(misconceptions),
            "sources": len(sources),
            "questions": len(questions),
            "domains": len(domains),
            "topics": len(topics),
            "release_id": release_id,
        }

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

        identity_groups = {
            "concept": [item.id for item in concepts],
            "misconception": [item.id for item in misconceptions],
            "source": [item.id for item in sources],
            "question": [item.id for item in questions],
            "domain": [item.id for item in domains],
            "topic": [item.id for item in topics],
        }
        violations: list[str] = []
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
                connection.execute(
                    "CREATE TEMP TABLE requested_misconceptions(id TEXT PRIMARY KEY)"
                )
                connection.executemany(
                    "INSERT INTO requested_misconceptions(id) VALUES (?)",
                    ((misconception_id,) for misconception_id in sorted(ids)),
                )
                joins.append(
                    " JOIN requested_misconceptions requested"
                    " ON requested.id = misconception.id"
                )
            clause = " WHERE " + " AND ".join(where) if where else ""
            rows = connection.execute(
                f"SELECT misconception.* FROM misconceptions misconception"
                f"{''.join(joins)}{clause}",
                tuple(parameters),
            ).fetchall()
        return [Misconception(row["id"], row["concept_id"], row["name"], row["description"]) for row in rows]

    def _question_from_row(self, connection: sqlite3.Connection, row: sqlite3.Row) -> Question:
        concept_rows = connection.execute(
            """SELECT * FROM question_concepts
               WHERE question_id = ? ORDER BY rowid""",
            (row["id"],),
        ).fetchall()
        option_rows = connection.execute(
            "SELECT * FROM options WHERE question_id = ? ORDER BY rowid", (row["id"],)
        ).fetchall()
        source_rows = connection.execute(
            """SELECT source_id FROM question_sources
               WHERE question_id = ? ORDER BY rowid""",
            (row["id"],),
        ).fetchall()
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
                )
                for r in option_rows
            ),
            source_ids=tuple(r["source_id"] for r in source_rows),
            provenance=json.loads(row["provenance_json"]),
            tags=tuple(json.loads(row["tags_json"])),
            revision_of=row["revision_of"],
        )

    def get_question(self, question_id: str, connection: sqlite3.Connection | None = None) -> Question:
        owns_connection = connection is None
        conn = connection or self.connect()
        try:
            row = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
            if not row:
                raise NotFoundError(f"Unknown question: {question_id}")
            return self._question_from_row(conn, row)
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
        if not concept_ids:
            return []
        with self.read() as connection:
            release_id = release_id or self.get_active_release_id(connection)
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
            return self._questions_by_ids(
                connection, question_ids, release_id=release_id
            )

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
        source_rows = connection.execute(
            f"""SELECT * FROM question_sources
                WHERE question_id IN ({placeholders})
                ORDER BY question_id, rowid""",
            tuple(question_ids),
        ).fetchall()
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
        self, learner_id: str, connection: sqlite3.Connection | None = None
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
            return _content_hash(
                {
                    "learner_id": learner["id"],
                    "learner_revision": learner["revision"],
                    "skill_states": skill_states,
                    "misconception_beliefs": misconception_beliefs,
                    "skill_families": skill_families,
                }
            )
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
                """SELECT attempt.is_correct, attempt.confidence,
                          attempt.response_ms, attempt.hint_count,
                          decision.pedagogical_role, question.id AS question_id,
                          primary_mapping.concept_id AS primary_concept_id
                   FROM attempts attempt
                   JOIN decisions decision ON decision.id = attempt.decision_id
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
                "confidence": row["confidence"],
                "response_ms": row["response_ms"],
                "hint_count": row["hint_count"],
                "pedagogical_role": row["pedagogical_role"],
                "question_id": row["question_id"],
                "primary_concept_id": row["primary_concept_id"],
            }
            for row in rows
        ]

    def independent_evidence_summary(self, learner_id: str, concept_id: str) -> dict[str, int]:
        return self.independent_evidence_summaries(learner_id, {concept_id})[concept_id]

    def independent_evidence_summaries(
        self, learner_id: str, concept_ids: set[str]
    ) -> dict[str, dict[str, int]]:
        summaries = {
            concept_id: {"families": 0, "delayed": 0, "operation_kinds": 0}
            for concept_id in concept_ids
        }
        if not concept_ids:
            return summaries
        with self.read() as connection:
            connection.execute("CREATE TEMP TABLE evidence_scope(id TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT INTO evidence_scope(id) VALUES (?)",
                ((concept_id,) for concept_id in sorted(concept_ids)),
            )
            rows = connection.execute(
                """SELECT evidence.concept_id, COUNT(*) AS families,
                          COUNT(DISTINCT CASE WHEN evidence.kind != 'unknown'
                                              THEN evidence.kind END) AS operation_kinds,
                          SUM(CASE WHEN evidence.delayed_unguided_correct_at IS NOT NULL
                                   THEN 1 ELSE 0 END) AS delayed
                   FROM learner_skill_families evidence
                   JOIN evidence_scope scope ON scope.id = evidence.concept_id
                   WHERE evidence.learner_id = ? GROUP BY evidence.concept_id""",
                (learner_id,),
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
        phase = {
            "diagnose": SessionPhase.DIAGNOSE,
            "review": SessionPhase.REVIEW,
        }.get(mode, SessionPhase.LEARN)
        session_id = new_id("ses")
        now = datetime.now(timezone.utc).isoformat()
        resolved_seed = seed if seed is not None else secrets.randbelow(2**31)
        with self.transaction() as connection:
            if not connection.execute(
                "SELECT 1 FROM learners WHERE id = ?", (learner_id,)
            ).fetchone():
                raise NotFoundError(f"Unknown learner: {learner_id}")
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
                    now,
                    now,
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
            now = datetime.now(timezone.utc)
            event = self.append_event(
                connection,
                stream_id=f"learner:{session['learner_id']}",
                event_type="SessionEnded",
                payload=payload,
                metadata={"corpus_release_id": session["corpus_release_id"]},
                learner_id=session["learner_id"],
                session_id=session_id,
                idempotency_key=idempotency_key,
                occurred_at=now,
            )
            updated = connection.execute(
                """UPDATE sessions SET status = ?, revision = revision + 1, updated_at = ?
                   WHERE id = ? AND status = 'active' AND revision = ?""",
                (status, to_timestamp(now), session_id, session["revision"]),
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

        def event_object(
            event: sqlite3.Row, column: str, cache: dict[str, dict[str, Any] | None]
        ) -> dict[str, Any] | None:
            event_id = event["event_id"]
            if event_id in cache:
                return cache[event_id]
            label = "payload" if column == "payload_json" else "metadata"
            try:
                value = json.loads(
                    event[column], parse_constant=reject_non_finite_json_constant
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
                value = json.loads(raw, parse_constant=reject_non_finite_json_constant)
            except (TypeError, ValueError) as exc:
                errors.append(f"{label}: invalid JSON ({exc})")
                return None
            if not isinstance(value, expected_type):
                errors.append(f"{label}: expected {expected_type.__name__} JSON")
                return None
            return value

        def compare_payload(
            actual: dict[str, Any], expected: dict[str, Any], label: str
        ) -> None:
            for field, expected_value in expected.items():
                if field not in actual:
                    errors.append(f"{label}: missing {field}")
                elif actual[field] != expected_value:
                    errors.append(f"{label}: {field} mismatch")

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
                    selection_event["schema_version"] != 1
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
                              membership.status, membership.evidence_weight
                       FROM release_questions membership
                       JOIN questions question ON question.id = membership.question_id
                       WHERE membership.release_id = ?
                       ORDER BY membership.question_id""",
                    (release_id,),
                ).fetchall()
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
                          topic.topic_id AS matched_topic,
                          concept.concept_id AS matched_concept
                   FROM sessions session
                   LEFT JOIN release_topics topic
                     ON topic.release_id = session.corpus_release_id
                    AND topic.topic_id = session.topic_id
                   LEFT JOIN release_concepts concept
                     ON concept.release_id = session.corpus_release_id
                    AND concept.concept_id = session.root_concept_id
                   WHERE session.exploration_mode NOT IN ('off', 'adaptive')
                      OR (session.topic_id IS NULL
                          AND session.exploration_mode != 'off')
                      OR (session.topic_id IS NOT NULL
                          AND topic.topic_id IS NULL)
                      OR concept.concept_id IS NULL
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

            # The latest projection event commits to all mutable learner-model
            # tables.  This catches out-of-band edits that a valid event chain
            # alone cannot reveal.
            projection_learner_ids = {
                row["learner_id"]
                for row in connection.execute(
                    """SELECT learner_id FROM skill_states
                       UNION SELECT learner_id FROM misconception_beliefs
                       UNION SELECT learner_id FROM learner_skill_families"""
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
                try:
                    actual_projection_hash = self.learner_projection_hash(
                        learner_id, connection
                    )
                except (NotFoundError, TypeError, ValueError) as exc:
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
                          q.version AS current_question_version,
                          q.content_hash AS current_question_content_hash,
                          q.family_id AS current_family_id,
                          rq.status AS release_question_status,
                          rq.evidence_weight AS release_evidence_weight
                   FROM decisions d
                   LEFT JOIN sessions s ON s.id = d.session_id
                   LEFT JOIN questions q ON q.id = d.question_id
                   LEFT JOIN release_questions rq
                     ON rq.release_id = d.corpus_release_id
                    AND rq.question_id = d.question_id
                   ORDER BY d.created_at, d.id"""
            ).fetchall()
            decisions = {row["id"]: row for row in decision_rows}

            option_rows = connection.execute(
                """SELECT DISTINCT o.question_id, o.option_id, o.text,
                          o.is_correct, o.rationale, o.misconception_id
                   FROM options o
                   JOIN decisions d ON d.question_id = o.question_id"""
            ).fetchall()
            option_ids: dict[str, set[str]] = {}
            answer_keys: dict[tuple[str, str], bool] = {}
            option_snapshots: dict[tuple[str, str], dict[str, Any]] = {}
            correct_option_snapshots: dict[str, list[dict[str, Any]]] = {}
            for option in option_rows:
                option_ids.setdefault(option["question_id"], set()).add(option["option_id"])
                answer_keys[(option["question_id"], option["option_id"])] = bool(
                    option["is_correct"]
                )
                option_snapshots[(option["question_id"], option["option_id"])] = {
                    "id": option["option_id"],
                    "text": option["text"],
                    "correct": bool(option["is_correct"]),
                    "rationale": option["rationale"],
                    "misconception_id": option["misconception_id"],
                }
                if option["is_correct"]:
                    correct_option_snapshots.setdefault(
                        option["question_id"], []
                    ).append(
                        option_snapshots[(option["question_id"], option["option_id"])]
                    )
            concept_ids = {
                row["id"] for row in connection.execute("SELECT id FROM concepts")
            }
            misconception_ids = {
                row["id"] for row in connection.execute("SELECT id FROM misconceptions")
            }
            for question_id in option_ids:
                correct_count = len(correct_option_snapshots.get(question_id, []))
                if correct_count != 1:
                    errors.append(
                        f"question {question_id}: expected one correct option, found "
                        f"{correct_count}"
                    )

            decision_orders: dict[str, list[Any] | None] = {}
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
                    selected_score = json_value(
                        decision["selected_score_json"],
                        f"{prefix} selected score",
                        dict,
                    )
                    if selection["learner_id"] != decision["learner_id"]:
                        errors.append(f"{prefix}: selection event learner mismatch")
                    if selection["session_id"] != decision["session_id"]:
                        errors.append(f"{prefix}: selection event session mismatch")
                    if (
                        selection_payload is not None
                        and order is not None
                        and selected_score is not None
                    ):
                        compare_payload(
                            selection_payload,
                            {
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
                            },
                            f"{prefix} selection event",
                        )

            for decision_id, selection_events in question_selected.items():
                if decision_id not in decisions:
                    for event in selection_events:
                        errors.append(
                            f"event {event['event_id']}: QuestionSelected has no decision"
                        )

            attempt_rows = connection.execute(
                """SELECT a.*, q.version AS current_question_version,
                          q.family_id AS current_family_id
                   FROM attempts a
                   LEFT JOIN questions q ON q.id = a.question_id
                   ORDER BY a.answered_at, a.id"""
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
                if presented_order is not None and attempt["command_hash"] != _content_hash(
                    command_payload
                ):
                    errors.append(f"{prefix}: command hash mismatch")

                if attempt["outcome_json"] is not None:
                    outcome = json_value(
                        attempt["outcome_json"], f"{prefix} outcome", dict
                    )
                    if outcome is not None:
                        correct_options = correct_option_snapshots.get(
                            attempt["question_id"], []
                        )
                        expected_selected = (
                            option_snapshots.get(
                                (attempt["question_id"], selected_option_id)
                            )
                            if selected_option_id is not None
                            else None
                        )
                        expected_correct_option = (
                            correct_options[0] if len(correct_options) == 1 else None
                        )
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
                if response_payload is not None and presented_order is not None:
                    compare_payload(
                        response_payload,
                        command_payload,
                        f"{prefix} response event",
                    )
                if response_metadata is not None:
                    compare_payload(
                        response_metadata,
                        {
                            "policy_version": decision["policy_version"],
                            "corpus_release_id": decision["corpus_release_id"],
                            "question_content_hash": decision["question_content_hash"],
                            "question_status": decision["question_status"],
                            "evidence_weight": decision["evidence_weight"],
                            "selection_learner_revision": decision["learner_revision"],
                        },
                        f"{prefix} response metadata",
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

            for event_id, event in response_events_by_id.items():
                if event_id not in attempts_by_event:
                    errors.append(
                        f"event {event_id}: ResponseSubmitted has no attempt projection"
                    )
            for decision_id, invalidations in invalidation_events_by_decision.items():
                if decision_id not in decisions:
                    for event in invalidations:
                        errors.append(
                            f"event {event['event_id']}: DecisionInvalidated has no decision"
                        )
        return {
            "ok": not errors,
            "event_count": len(events),
            "stream_count": len(expected_version),
            "learning_action_count": len(action_rows),
            "learning_artifact_count": len(artifact_rows),
            "errors": errors,
            "foreign_key_failures": foreign_key_failures,
            "quick_check": quick_check,
        }
