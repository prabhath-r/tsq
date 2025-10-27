# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Protocol

from .models import (
    ConceptRole,
    ConceptWeight,
    Option,
    Question,
    QuestionKind,
    QuestionStatus,
)
from .quality import validate_question
from .store import Database, new_id


PROMPT_VERSION = "item-blueprint-v1"


def _canonical_json(value: Any) -> str:
    """Canonical encoding used for persisted authoring attestations."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strict_json_issues(
    value: Any,
    path: str = "$.",
    *,
    _active_containers: set[int] | None = None,
    _depth: int = 0,
) -> list[str]:
    """Reject values outside the JSON data model instead of stringifying them.

    Exact container checks are intentional. An LLM adapter must return ordinary
    JSON objects and arrays, not Python objects with surprising iteration or
    serialization behavior.
    """

    if _depth > 64:
        return [f"{path} exceeds the maximum JSON nesting depth."]
    if value is None or type(value) in {str, bool, int}:
        return []
    if type(value) is float:
        return [] if isfinite(value) else [f"{path} must be finite."]
    if type(value) is list:
        active = set() if _active_containers is None else _active_containers
        if id(value) in active:
            return [f"{path} contains a cyclic JSON array."]
        active.add(id(value))
        issues: list[str] = []
        for index, entry in enumerate(value):
            issues.extend(
                _strict_json_issues(
                    entry,
                    f"{path}[{index}]",
                    _active_containers=active,
                    _depth=_depth + 1,
                )
            )
        active.remove(id(value))
        return issues
    if type(value) is dict:
        active = set() if _active_containers is None else _active_containers
        if id(value) in active:
            return [f"{path} contains a cyclic JSON object."]
        active.add(id(value))
        issues = []
        for key, entry in value.items():
            if type(key) is not str:
                issues.append(f"{path} contains a non-string object key.")
                continue
            issues.extend(
                _strict_json_issues(
                    entry,
                    f"{path}{key}.",
                    _active_containers=active,
                    _depth=_depth + 1,
                )
            )
        active.remove(id(value))
        return issues
    return [f"{path} has unsupported type {type(value).__name__}."]


def _normalized_identity(value: str) -> str:
    return " ".join(value.split()).casefold()


def _blind_for_review(item: dict[str, Any]) -> dict[str, Any]:
    """Construct an allow-listed solver copy with no authored answer metadata."""

    blinded = {
        field: copy.deepcopy(item[field])
        for field in (
            "id",
            "version",
            "family_id",
            "stem",
            "kind",
            "difficulty",
            "concepts",
            "source_ids",
        )
        if field in item
    }
    options = item.get("options")
    if type(options) is list:
        blinded["options"] = [
            {
                field: copy.deepcopy(option[field])
                for field in ("id", "text")
                if type(option) is dict and field in option
            }
            if type(option) is dict
            else copy.deepcopy(option)
            for option in options
        ]
    return blinded


class StructuredItemGenerator(Protocol):
    """Port implemented by an LLM adapter in an offline worker."""

    provider_name: str
    model_name: str

    def generate(self, blueprint: "GenerationBlueprint", source_context: str) -> dict[str, Any]: ...


class IndependentItemReviewer(Protocol):
    """A separate solver/critic; it must not share hidden generator state."""

    reviewer_name: str

    def review(self, item: dict[str, Any], source_context: str) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class GenerationBlueprint:
    concept_id: str
    concept_name: str
    kind: str
    target_difficulty: float
    misconception_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    family_constraint: str
    quality_contract: tuple[str, ...] = (
        "Exactly one defensible best answer under the stated assumptions.",
        "Every distractor instantiates a named misconception, not random noise.",
        "Options are parallel in grammar, specificity, and approximate length.",
        "The stem requires reasoning; no answer-position or wording clue is usable.",
        "Every option has a local rationale and every factual claim is source-grounded.",
    )


@dataclass(frozen=True, slots=True)
class CoverageGap:
    priority: float
    blueprint: GenerationBlueprint
    current_count: int
    target_count: int


class CoveragePlanner:
    """Turn corpus coverage debt into explicit authoring blueprints.

    Aggregate learner uncertainty can later be added as a priority term. It must
    never bypass the review lifecycle; this planner creates work, not live items.
    """

    KIND_TARGETS = {
        "diagnostic": 2,
        "conceptual": 2,
        "application": 3,
        "debugging": 2,
        "counterfactual": 1,
        "transfer": 2,
    }
    DIFFICULTY_SEQUENCE = (-1.0, -0.35, 0.25, 0.85, 1.45)

    def __init__(self, database: Database):
        self.database = database

    def gaps(self, *, limit: int = 100, source_ids: tuple[str, ...] = ()) -> list[CoverageGap]:
        graph = self.database.get_graph()
        with self.database.read() as connection:
            release_id = self.database.get_active_release_id(connection)
            counts = {
                (row["concept_id"], row["kind"]): row["n"]
                for row in connection.execute(
                    """SELECT qc.concept_id, q.kind, COUNT(DISTINCT q.family_id) AS n
                       FROM question_concepts qc
                       JOIN questions q ON q.id = qc.question_id
                       JOIN release_questions rq ON rq.question_id = q.id
                       WHERE rq.release_id = ?
                         AND rq.status IN ('approved', 'calibrated')
                         AND qc.role = 'primary'
                         AND NOT EXISTS (
                             SELECT 1 FROM question_revocations revoked
                             WHERE revoked.question_id = q.id
                         )
                       GROUP BY qc.concept_id, q.kind""",
                    (release_id,),
                ).fetchall()
            }
            misconception_rows = connection.execute(
                """SELECT m.id, m.concept_id,
                          COUNT(DISTINCT CASE WHEN rq.status IN ('approved', 'calibrated')
                                              AND NOT EXISTS (
                                                  SELECT 1
                                                  FROM question_revocations revoked
                                                  WHERE revoked.question_id = q.id
                                              )
                                              THEN q.family_id END) AS uses
                   FROM misconceptions m
                   LEFT JOIN options o ON o.misconception_id = m.id
                   LEFT JOIN questions q ON q.id = o.question_id
                   LEFT JOIN release_questions rq
                     ON rq.question_id = q.id AND rq.release_id = ?
                   JOIN release_misconceptions rm
                     ON rm.misconception_id = m.id AND rm.release_id = ?
                   GROUP BY m.id, m.concept_id ORDER BY uses, m.id""",
                (release_id, release_id),
            ).fetchall()
            available_sources = tuple(
                row["id"]
                for row in connection.execute(
                    """SELECT source.id FROM sources source
                       JOIN release_sources membership
                         ON membership.source_id = source.id
                       WHERE membership.release_id = ? ORDER BY source.id""",
                    (release_id,),
                ).fetchall()
            )
            source_rows = connection.execute(
                """SELECT DISTINCT qc.concept_id, qs.source_id
                   FROM question_concepts qc
                   JOIN question_sources qs ON qs.question_id = qc.question_id
                   JOIN release_questions rq ON rq.question_id = qc.question_id
                   WHERE qc.role = 'primary' AND rq.release_id = ?
                     AND NOT EXISTS (
                         SELECT 1 FROM question_revocations revoked
                         WHERE revoked.question_id = qc.question_id
                     )
                   ORDER BY qc.concept_id, qs.source_id""",
                (release_id,),
            ).fetchall()
        misconceptions_by_concept: dict[str, list[str]] = {}
        for row in misconception_rows:
            misconceptions_by_concept.setdefault(row["concept_id"], []).append(row["id"])
        sources_by_concept: dict[str, list[str]] = {}
        for row in source_rows:
            sources_by_concept.setdefault(row["concept_id"], []).append(row["source_id"])

        # Containers have PART_OF children and are navigation nodes, not mastery variables.
        containers = {
            edge.target_id for edge in graph.edges if edge.relation.value == "part_of"
        }
        gaps: list[CoverageGap] = []
        for concept_id, concept in graph.concepts.items():
            if concept_id in containers:
                continue
            misconception_ids = tuple(misconceptions_by_concept.get(concept_id, [])[:3])
            chosen_sources = source_ids or tuple(sources_by_concept.get(concept_id, ()))
            if not chosen_sources:
                chosen_sources = available_sources
            concept_total = sum(
                count for (cid, _), count in counts.items() if cid == concept_id
            )
            for kind, target in self.KIND_TARGETS.items():
                current = int(counts.get((concept_id, kind), 0))
                missing = target - current
                for offset in range(max(0, missing)):
                    target_difficulty = self.DIFFICULTY_SEQUENCE[
                        (current + offset) % len(self.DIFFICULTY_SEQUENCE)
                    ]
                    blueprint = GenerationBlueprint(
                        concept_id=concept_id,
                        concept_name=concept.name,
                        kind=kind,
                        target_difficulty=target_difficulty,
                        misconception_ids=misconception_ids,
                        source_ids=chosen_sources,
                        family_constraint=(
                            "Create a new solution path and surface context; do not paraphrase an existing family."
                        ),
                    )
                    # Diagnostics and concepts with no active item receive first priority.
                    priority = 2.0 + (1.0 if concept_total == 0 else 0.0)
                    priority += {"diagnostic": 0.45, "transfer": 0.30, "application": 0.20}.get(kind, 0.0)
                    priority += 0.05 * (target - current)
                    gaps.append(CoverageGap(priority, blueprint, current, target))
        gaps.sort(
            key=lambda gap: (
                -gap.priority,
                gap.blueprint.concept_id,
                gap.blueprint.kind,
                gap.blueprint.target_difficulty,
            )
        )
        return gaps[: max(0, limit)]

    def enqueue(self, gaps: list[CoverageGap]) -> list[str]:
        job_ids: list[str] = []
        with self.database.transaction() as connection:
            existing = {
                json.dumps(
                    json.loads(row["blueprint_json"]),
                    sort_keys=True,
                    separators=(",", ":"),
                ): row["id"]
                for row in connection.execute(
                    """SELECT id, blueprint_json FROM generation_jobs
                       WHERE status IN ('planned', 'running')
                       ORDER BY created_at"""
                ).fetchall()
            }
            for gap in gaps:
                blueprint_json = json.dumps(
                    asdict(gap.blueprint), sort_keys=True, separators=(",", ":")
                )
                if blueprint_json in existing:
                    job_ids.append(existing[blueprint_json])
                    continue
                job_id = new_id("gen")
                now = datetime.now(timezone.utc).isoformat()
                connection.execute(
                    """INSERT INTO generation_jobs(
                       id, blueprint_json, status, prompt_version, created_at, updated_at
                       ) VALUES (?, ?, 'planned', ?, ?, ?)""",
                    (job_id, blueprint_json, PROMPT_VERSION, now, now),
                )
                job_ids.append(job_id)
                existing[blueprint_json] = job_id
        return job_ids
