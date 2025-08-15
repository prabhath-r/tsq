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

from .errors import ConflictError, NotFoundError, ValidationError
from .graph import KnowledgeGraph
from .models import (
    CandidateScore,
    Concept,
    ConceptEdge,
    ConceptWeight,
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
)
from .quality import audit_corpus


SCHEMA_VERSION = 5

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
    updated_at TEXT NOT NULL
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
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._install_v5_indexes(connection)
            self._install_v4_attempt_triggers(connection)
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
    ) -> dict[str, int | str]:
        self._validate_corpus_activation(
            concepts, edges, misconceptions, sources, questions
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
            "release_id": release_id,
        }

    @staticmethod
    def _validate_corpus_activation(
        concepts: Sequence[Concept],
        edges: Sequence[ConceptEdge],
        misconceptions: Sequence[Misconception],
        sources: Sequence[Source],
        questions: Sequence[Question],
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
        KnowledgeGraph(concepts, edges)
        quality_errors = [
            issue
            for issue in audit_corpus(
                questions,
                expected_primary_concept_ids={
                    mapping.concept_id
                    for question in questions
                    for mapping in question.concepts
                },
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
