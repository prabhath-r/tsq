# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Mapping, Protocol

from .models import (
    Concept,
    ConceptRole,
    ConceptWeight,
    LearningObjective,
    Misconception,
    ObjectiveOperation,
    Option,
    Question,
    QuestionKind,
    QuestionStatus,
    Source,
)
from .errors import ConflictError, NotFoundError, ValidationError
from .provenance import question_provenance_issues
from .quality import validate_question
from .store import (
    Database,
    concept_content_hash,
    misconception_content_hash,
    new_id,
    objective_content_hash,
    question_content_hash,
    source_content_hash,
)


PROMPT_VERSION = "item-blueprint-v2"
SUPPORTED_PROMPT_VERSIONS = frozenset({"item-blueprint-v1", PROMPT_VERSION})
QUARANTINE_REVIEW_PACKET_SCHEMA = "tsq-quarantine-review-packet-v2"
GENERATOR_AUTHORITY_PROVENANCE_FIELDS = frozenset(
    {
        "activation",
        "activation_review",
        "human_review",
        "human_review_status",
        "independent_review",
        "review_chain_sha256",
        "review_payload_sha256",
        "review_sha256",
        "review_status",
        "reviewed_on",
    }
)
SOURCE_CONTEXT_REDACTION = "[REDACTED_EXACT_SOURCE_CONTEXT]"


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


def _redact_exact_source_context(
    value: Any,
    source_context: str,
) -> tuple[Any, int]:
    """Remove exact source-context copies from JSON-safe persisted artifacts."""

    if not source_context:
        return copy.deepcopy(value), 0
    if type(value) is str:
        if len(source_context) < 32:
            count = int(value == source_context)
            redacted_value = (
                SOURCE_CONTEXT_REDACTION if count else value
            )
        else:
            count = value.count(source_context)
            redacted_value = value.replace(
                source_context,
                SOURCE_CONTEXT_REDACTION,
            )
        return (
            redacted_value,
            count,
        )
    if type(value) is list:
        redacted: list[Any] = []
        count = 0
        for entry in value:
            sanitized, entry_count = _redact_exact_source_context(
                entry, source_context
            )
            redacted.append(sanitized)
            count += entry_count
        return redacted, count
    if type(value) is dict:
        redacted_dict: dict[str, Any] = {}
        count = 0
        for key, entry in value.items():
            sanitized_key, key_count = _redact_exact_source_context(
                key, source_context
            )
            sanitized_entry, entry_count = _redact_exact_source_context(
                entry, source_context
            )
            if sanitized_key in redacted_dict:
                sanitized_key = (
                    f"{sanitized_key}:"
                    f"{_sha256_text(key)[:16]}"
                )
            redacted_dict[sanitized_key] = sanitized_entry
            count += key_count + entry_count
        return redacted_dict, count
    return copy.deepcopy(value), 0


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


def _blind_for_review(
    item: dict[str, Any], blueprint: "GenerationBlueprint | None" = None
) -> dict[str, Any]:
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
            "learning_objective_id",
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
    if blueprint is not None and blueprint.learning_objective_id is not None:
        blinded["learning_objective"] = {
            "id": blueprint.learning_objective_id,
            "name": blueprint.learning_objective_name,
            "description": blueprint.learning_objective_description,
            "operation": blueprint.learning_objective_operation,
            "evidence_type": blueprint.learning_objective_evidence_type,
        }
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
    corpus_release_id: str | None = None
    learning_objective_id: str | None = None
    learning_objective_name: str | None = None
    learning_objective_description: str | None = None
    learning_objective_operation: str | None = None
    learning_objective_evidence_type: str | None = None
    target_misconception_id: str | None = None
    coverage_goal: str = "concept_kind"
    quality_contract: tuple[str, ...] = (
        "Exactly one defensible best answer under the stated assumptions.",
        "Every distractor instantiates a named misconception, not random noise.",
        "Options are parallel in grammar, specificity, and approximate length.",
        "The stem requires reasoning; no answer-position or wording clue is usable.",
        "Every option has a local rationale and every factual claim is source-grounded.",
    )


def _decode_json(
    raw: str | None,
    *,
    label: str,
    expected_type: type[Any],
    nullable: bool = False,
) -> Any:
    """Decode persisted authoring JSON without accepting NaN or scalar coercion."""

    if raw is None:
        if nullable:
            return None
        raise ValidationError(f"{label} is missing.")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(raw, parse_constant=reject_constant)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} contains invalid JSON: {exc}") from exc
    if type(value) is not expected_type:
        raise ValidationError(
            f"{label} must contain a JSON {expected_type.__name__}, "
            f"not {type(value).__name__}."
        )
    return value


def _parse_blueprint(raw: str, *, label: str = "Generation blueprint") -> GenerationBlueprint:
    payload = _decode_json(raw, label=label, expected_type=dict)
    required = {
        "concept_id",
        "concept_name",
        "kind",
        "target_difficulty",
        "misconception_ids",
        "source_ids",
        "family_constraint",
    }
    optional = {
        "quality_contract",
        "corpus_release_id",
        "learning_objective_id",
        "learning_objective_name",
        "learning_objective_description",
        "learning_objective_operation",
        "learning_objective_evidence_type",
        "target_misconception_id",
        "coverage_goal",
    }
    allowed = required | optional
    missing = required - set(payload)
    unknown = set(payload) - allowed
    if missing:
        raise ValidationError(f"{label} is missing fields: {', '.join(sorted(missing))}.")
    if unknown:
        raise ValidationError(f"{label} has unknown fields: {', '.join(sorted(unknown))}.")
    for field in ("concept_id", "concept_name", "kind", "family_constraint"):
        if type(payload[field]) is not str or not payload[field].strip():
            raise ValidationError(f"{label} field {field!r} must be a non-empty string.")
    if payload["kind"] not in {kind.value for kind in QuestionKind}:
        raise ValidationError(f"{label} has unsupported question kind {payload['kind']!r}.")
    difficulty = payload["target_difficulty"]
    if type(difficulty) not in {int, float} or not isfinite(difficulty):
        raise ValidationError(f"{label} target_difficulty must be a finite JSON number.")
    if not -3.0 <= float(difficulty) <= 3.0:
        raise ValidationError(f"{label} target_difficulty must be between -3 and 3.")

    arrays: dict[str, tuple[str, ...]] = {}
    for field in ("misconception_ids", "source_ids"):
        value = payload[field]
        if type(value) is not list or any(
            type(entry) is not str or not entry.strip() for entry in value
        ):
            raise ValidationError(f"{label} field {field!r} must be an array of strings.")
        if len(value) != len(set(value)):
            raise ValidationError(f"{label} field {field!r} contains duplicate IDs.")
        arrays[field] = tuple(value)
    if not arrays["source_ids"]:
        raise ValidationError(f"{label} must cite at least one approved source ID.")

    corpus_release_id = payload.get("corpus_release_id")
    if corpus_release_id is not None and (
        type(corpus_release_id) is not str or not corpus_release_id.strip()
    ):
        raise ValidationError(
            f"{label} field 'corpus_release_id' must be a non-empty string or null."
        )
    objective_fields = (
        "learning_objective_id",
        "learning_objective_name",
        "learning_objective_description",
        "learning_objective_operation",
        "learning_objective_evidence_type",
    )
    objective_values = {field: payload.get(field) for field in objective_fields}
    if objective_values["learning_objective_id"] is None:
        supplied = sorted(
            field for field, value in objective_values.items() if value is not None
        )
        if supplied:
            raise ValidationError(
                f"{label} supplies objective metadata without learning_objective_id: "
                + ", ".join(supplied)
                + "."
            )
    else:
        invalid = [
            field
            for field, value in objective_values.items()
            if type(value) is not str or not value.strip()
        ]
        if invalid:
            raise ValidationError(
                f"{label} objective fields must all be non-empty strings: "
                + ", ".join(sorted(invalid))
                + "."
            )
        try:
            ObjectiveOperation(objective_values["learning_objective_operation"])
        except ValueError as exc:
            raise ValidationError(
                f"{label} has unsupported learning objective operation "
                f"{objective_values['learning_objective_operation']!r}."
            ) from exc
        if objective_values["learning_objective_evidence_type"] != "selected_response":
            raise ValidationError(
                f"{label} supports only selected_response objective evidence."
            )

    target_misconception_id = payload.get("target_misconception_id")
    if target_misconception_id is not None and (
        type(target_misconception_id) is not str
        or not target_misconception_id.strip()
    ):
        raise ValidationError(
            f"{label} field 'target_misconception_id' must be a non-empty string or null."
        )
    if target_misconception_id is not None and target_misconception_id not in arrays[
        "misconception_ids"
    ]:
        raise ValidationError(
            f"{label} target_misconception_id must occur in misconception_ids."
        )
    if target_misconception_id is not None and objective_values[
        "learning_objective_id"
    ] is None:
        raise ValidationError(
            f"{label} cannot target an exact misconception without a learning objective."
        )
    coverage_goal = payload.get("coverage_goal", "concept_kind")
    if type(coverage_goal) is not str or coverage_goal not in {
        "concept_kind",
        "objective_serviceability",
        "objective_misconception_serviceability",
        "live_corpus_gap",
    }:
        raise ValidationError(f"{label} has unsupported coverage_goal {coverage_goal!r}.")

    quality_contract = payload.get("quality_contract")
    if quality_contract is not None and (
        type(quality_contract) is not list
        or not quality_contract
        or any(type(entry) is not str or not entry.strip() for entry in quality_contract)
    ):
        raise ValidationError(
            f"{label} field 'quality_contract' must be a non-empty array of strings."
        )
    kwargs: dict[str, Any] = {
        "concept_id": payload["concept_id"],
        "concept_name": payload["concept_name"],
        "kind": payload["kind"],
        "target_difficulty": float(difficulty),
        "misconception_ids": arrays["misconception_ids"],
        "source_ids": arrays["source_ids"],
        "family_constraint": payload["family_constraint"],
        "corpus_release_id": corpus_release_id,
        "learning_objective_id": objective_values["learning_objective_id"],
        "learning_objective_name": objective_values["learning_objective_name"],
        "learning_objective_description": objective_values[
            "learning_objective_description"
        ],
        "learning_objective_operation": objective_values[
            "learning_objective_operation"
        ],
        "learning_objective_evidence_type": objective_values[
            "learning_objective_evidence_type"
        ],
        "target_misconception_id": target_misconception_id,
        "coverage_goal": coverage_goal,
    }
    if quality_contract is not None:
        kwargs["quality_contract"] = tuple(quality_contract)
    return GenerationBlueprint(**kwargs)


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
    SERVICEABILITY_TARGET = 3
    _VERIFICATION_KINDS = (
        "application",
        "calculation",
        "comparison",
        "counterfactual",
        "debugging",
        "transfer",
    )

    def __init__(self, database: Database):
        self.database = database

    def gaps(
        self,
        *,
        limit: int = 100,
        source_ids: tuple[str, ...] = (),
        topic_filter: str | None = None,
        concept_filter: str | None = None,
        objective_filter: str | None = None,
        misconception_filter: str | None = None,
        kind_filter: str | None = None,
        goal_filter: str | None = None,
        maximum_difficulty: float | None = None,
    ) -> list[CoverageGap]:
        for label, value in (
            ("topic", topic_filter),
            ("concept_id", concept_filter),
            ("learning_objective_id", objective_filter),
            ("target_misconception_id", misconception_filter),
            ("kind", kind_filter),
            ("coverage_goal", goal_filter),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValidationError(
                    f"Coverage filter {label} must be a non-blank string."
                )
        if kind_filter is not None and kind_filter not in self.KIND_TARGETS:
            raise ValidationError(
                f"Unknown coverage question kind: {kind_filter!r}."
            )
        if goal_filter is not None and goal_filter not in {
            "concept_kind",
            "objective_misconception_serviceability",
            "objective_serviceability",
        }:
            raise ValidationError(
                f"Unknown coverage goal: {goal_filter!r}."
            )
        if maximum_difficulty is not None and (
            isinstance(maximum_difficulty, bool)
            or not isinstance(maximum_difficulty, (int, float))
            or not isfinite(float(maximum_difficulty))
        ):
            raise ValidationError(
                "Coverage maximum difficulty must be a finite number."
            )
        with self.database.read() as connection:
            release_id = self.database.get_active_release_id(connection)
        topic_concepts: set[str] | None = None
        if topic_filter is not None:
            resolved_topic = self.database.resolve_topic(
                topic_filter, release_id
            )
            topic_concepts = self.database.topic_owned_concepts(
                resolved_topic["id"],
                release_id,
                include_descendants=True,
            )
        graph = self.database.get_graph(release_id)
        with self.database.read() as connection:
            verification_placeholders = ",".join(
                "?" for _ in self._VERIFICATION_KINDS
            )
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
                     AND rq.status IN ('approved', 'calibrated')
                     AND NOT EXISTS (
                         SELECT 1 FROM question_revocations revoked
                         WHERE revoked.question_id = qc.question_id
                     )
                   ORDER BY qc.concept_id, qs.source_id""",
                (release_id,),
            ).fetchall()
            objective_count_rows = connection.execute(
                """SELECT direct.objective_id, q.kind,
                          COUNT(DISTINCT q.family_id) AS n
                   FROM release_question_objectives direct
                   JOIN release_questions rq
                     ON rq.release_id = direct.release_id
                    AND rq.question_id = direct.question_id
                   JOIN questions q ON q.id = direct.question_id
                   WHERE direct.release_id = ?
                     AND rq.status IN ('approved', 'calibrated')
                     AND NOT EXISTS (
                         SELECT 1 FROM question_revocations revoked
                         WHERE revoked.question_id = q.id
                     )
                   GROUP BY direct.objective_id, q.kind""",
                (release_id,),
            ).fetchall()
            objective_family_rows = connection.execute(
                f"""SELECT direct.objective_id,
                          COUNT(DISTINCT q.family_id) AS n,
                          COUNT(DISTINCT CASE
                              WHEN q.kind IN ({verification_placeholders})
                              THEN q.family_id END) AS verification_n
                   FROM release_question_objectives direct
                   JOIN release_questions rq
                     ON rq.release_id = direct.release_id
                    AND rq.question_id = direct.question_id
                   JOIN questions q ON q.id = direct.question_id
                   WHERE direct.release_id = ?
                     AND rq.status IN ('approved', 'calibrated')
                     AND NOT EXISTS (
                         SELECT 1 FROM question_revocations revoked
                         WHERE revoked.question_id = q.id
                     )
                   GROUP BY direct.objective_id""",
                (*self._VERIFICATION_KINDS, release_id),
            ).fetchall()
            objective_source_rows = connection.execute(
                """SELECT DISTINCT direct.objective_id, qs.source_id
                   FROM release_question_objectives direct
                   JOIN question_sources qs ON qs.question_id = direct.question_id
                   JOIN release_questions rq
                     ON rq.release_id = direct.release_id
                    AND rq.question_id = direct.question_id
                   WHERE direct.release_id = ?
                     AND rq.status IN ('approved', 'calibrated')
                     AND NOT EXISTS (
                         SELECT 1 FROM question_revocations revoked
                         WHERE revoked.question_id = direct.question_id
                     )
                   ORDER BY direct.objective_id, qs.source_id""",
                (release_id,),
            ).fetchall()
            objective_misconception_rows = connection.execute(
                f"""SELECT diagnostic.objective_id, option.misconception_id,
                           misconception.concept_id,
                           COUNT(DISTINCT CASE
                               WHEN rq.status IN ('approved', 'calibrated')
                                AND direct.objective_id = diagnostic.objective_id
                                AND NOT EXISTS (
                                    SELECT 1 FROM question_revocations revoked
                                    WHERE revoked.question_id = q.id
                                ) THEN q.family_id END) AS uses,
                           COUNT(DISTINCT CASE
                               WHEN rq.status IN ('approved', 'calibrated')
                                AND direct.objective_id = diagnostic.objective_id
                                AND q.kind IN ({verification_placeholders})
                                AND NOT EXISTS (
                                    SELECT 1 FROM question_revocations revoked
                                    WHERE revoked.question_id = q.id
                                ) THEN q.family_id END) AS verification_uses
                    FROM release_option_objectives diagnostic
                    JOIN release_questions rq
                      ON rq.release_id = diagnostic.release_id
                     AND rq.question_id = diagnostic.question_id
                    JOIN questions q ON q.id = diagnostic.question_id
                    LEFT JOIN release_question_objectives direct
                      ON direct.release_id = diagnostic.release_id
                     AND direct.question_id = diagnostic.question_id
                    JOIN options option
                      ON option.question_id = diagnostic.question_id
                     AND option.option_id = diagnostic.option_id
                    JOIN misconceptions misconception
                      ON misconception.id = option.misconception_id
                    WHERE diagnostic.release_id = ? AND option.is_correct = 0
                    GROUP BY diagnostic.objective_id, option.misconception_id,
                             misconception.concept_id
                    ORDER BY uses, diagnostic.objective_id,
                             option.misconception_id""",
                (*self._VERIFICATION_KINDS, release_id),
            ).fetchall()
        objectives = self.database.get_learning_objectives(release_id)
        misconceptions_by_concept: dict[str, list[str]] = {}
        for row in misconception_rows:
            misconceptions_by_concept.setdefault(row["concept_id"], []).append(row["id"])
        sources_by_concept: dict[str, list[str]] = {}
        for row in source_rows:
            sources_by_concept.setdefault(row["concept_id"], []).append(row["source_id"])
        objective_counts = {
            (row["objective_id"], row["kind"]): int(row["n"])
            for row in objective_count_rows
        }
        objective_family_capacity = {
            row["objective_id"]: (
                int(row["n"]), int(row["verification_n"])
            )
            for row in objective_family_rows
        }
        objective_total_counts: dict[str, int] = {}
        for (objective_id, _kind), count in objective_counts.items():
            objective_total_counts[objective_id] = (
                objective_total_counts.get(objective_id, 0) + count
            )
        sources_by_objective: dict[str, list[str]] = {}
        for row in objective_source_rows:
            sources_by_objective.setdefault(row["objective_id"], []).append(
                row["source_id"]
            )
        objectives_by_concept: dict[str, list[LearningObjective]] = {}
        for objective in objectives:
            for concept_id in objective.concept_ids:
                objectives_by_concept.setdefault(concept_id, []).append(objective)
        misconception_rows_by_objective: dict[str, list[Any]] = {}
        for row in objective_misconception_rows:
            misconception_rows_by_objective.setdefault(
                row["objective_id"], []
            ).append(row)

        # Containers have PART_OF children and are navigation nodes, not mastery variables.
        containers = {
            edge.target_id for edge in graph.edges if edge.relation.value == "part_of"
        }
        gaps: list[CoverageGap] = []

        # Exact objective/misconception serviceability is a separate authoring
        # debt from broad concept-kind coverage.  Three families, including at
        # least two verification-capable families, are sufficient for any one
        # trigger to retain a distinct repair and a distinct verification.
        objective_by_id = {objective.id: objective for objective in objectives}
        planned_direct_jobs: dict[str, int] = {
            objective.id: 0 for objective in objectives
        }
        planned_verification_jobs: dict[str, int] = {
            objective.id: 0 for objective in objectives
        }
        for objective_id, rows in sorted(misconception_rows_by_objective.items()):
            objective = objective_by_id[objective_id]
            effective_capacity_by_misconception = {
                row["misconception_id"]: min(
                    int(row["uses"]), int(row["verification_uses"]) + 1
                )
                for row in rows
            }
            remaining = {
                misconception_id: max(
                    0, self.SERVICEABILITY_TARGET - effective_capacity
                )
                for misconception_id, effective_capacity in (
                    effective_capacity_by_misconception.items()
                )
            }
            chosen_sources = source_ids or tuple(
                sources_by_objective.get(objective.id, ())
            )
            if not chosen_sources:
                chosen_sources = tuple(
                    sources_by_concept.get(objective.primary_concept_id, ())
                ) or available_sources
            while any(deficit > 0 for deficit in remaining.values()):
                misconception_ids = tuple(
                    misconception_id
                    for misconception_id, _deficit in sorted(
                        (
                            (misconception_id, deficit)
                            for misconception_id, deficit in remaining.items()
                            if deficit > 0
                        ),
                        key=lambda pair: (-pair[1], pair[0]),
                    )[:3]
                )
                planned_target_misconception_id = misconception_ids[0]
                current_capacity = (
                    self.SERVICEABILITY_TARGET
                    - remaining[planned_target_misconception_id]
                )
                blueprint = self._blueprint(
                    release_id=release_id,
                    concept_id=objective.primary_concept_id,
                    concept_name=graph.concepts[objective.primary_concept_id].name,
                    kind="transfer",
                    target_difficulty=self.DIFFICULTY_SEQUENCE[
                        current_capacity % len(self.DIFFICULTY_SEQUENCE)
                    ],
                    misconception_ids=misconception_ids,
                    source_ids=chosen_sources,
                    objective=objective,
                    target_misconception_id=(
                        planned_target_misconception_id
                    ),
                    coverage_goal="objective_misconception_serviceability",
                    family_constraint=(
                        "Create an independent transfer family that directly assesses "
                        f"{objective.id} and exposes every listed named misconception; "
                        "do not reuse a scenario, derivation, or solution path from "
                        "the release."
                    ),
                )
                priority = 5.0 + 0.2 * max(remaining.values()) - 0.01 * current_capacity
                gaps.append(
                    CoverageGap(
                        priority,
                        blueprint,
                        current_capacity,
                        self.SERVICEABILITY_TARGET,
                    )
                )
                planned_direct_jobs[objective.id] += 1
                planned_verification_jobs[objective.id] += 1
                for misconception_id in misconception_ids:
                    remaining[misconception_id] -= 1

        planned_objective_kind_counts = dict(objective_counts)
        for concept_id, concept in graph.concepts.items():
            if concept_id in containers:
                continue
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
                    candidates = objectives_by_concept.get(concept_id, ())
                    objective = min(
                        candidates,
                        key=lambda candidate: (
                            planned_objective_kind_counts.get(
                                (candidate.id, kind), 0
                            ),
                            objective_total_counts.get(candidate.id, 0),
                            candidate.id,
                        ),
                        default=None,
                    )
                    if objective is not None:
                        objective_misconceptions = tuple(
                            row["misconception_id"]
                            for row in misconception_rows_by_objective.get(
                                objective.id, ()
                            )[:3]
                        )
                        chosen_sources = source_ids or tuple(
                            dict.fromkeys(
                                [
                                    *sources_by_objective.get(objective.id, ()),
                                    *sources_by_concept.get(concept_id, ()),
                                ]
                            )
                        )
                        planned_objective_kind_counts[(objective.id, kind)] = (
                            planned_objective_kind_counts.get(
                                (objective.id, kind), 0
                            )
                            + 1
                        )
                        planned_direct_jobs[objective.id] += 1
                        if kind in self._VERIFICATION_KINDS:
                            planned_verification_jobs[objective.id] += 1
                    else:
                        objective_misconceptions = tuple(
                            misconceptions_by_concept.get(concept_id, [])[:3]
                        )
                        chosen_sources = source_ids or tuple(
                            sources_by_concept.get(concept_id, ())
                        )
                    if not chosen_sources:
                        chosen_sources = available_sources
                    blueprint = self._blueprint(
                        release_id=release_id,
                        concept_id=concept_id,
                        concept_name=concept.name,
                        kind=kind,
                        target_difficulty=target_difficulty,
                        misconception_ids=objective_misconceptions,
                        source_ids=chosen_sources,
                        objective=objective,
                        target_misconception_id=None,
                        coverage_goal="concept_kind",
                        family_constraint=(
                            "Create a new solution path and surface context; do not "
                            "paraphrase an existing family."
                        ),
                    )
                    # Diagnostics and concepts with no active item receive first priority.
                    priority = 2.0 + (1.0 if concept_total == 0 else 0.0)
                    priority += {"diagnostic": 0.45, "transfer": 0.30, "application": 0.20}.get(kind, 0.0)
                    priority += 0.05 * (target - current)
                    gaps.append(CoverageGap(priority, blueprint, current, target))

        # A sparse objective can remain unsafe even when its broad concept-kind
        # quotas are already satisfied by sibling objectives.  Account for the
        # exact-pair and concept-kind jobs already planned above, then request
        # only the additional independent direct families still needed.
        for objective in objectives:
            current_total, current_verification = objective_family_capacity.get(
                objective.id, (0, 0)
            )
            projected_total = current_total + planned_direct_jobs.get(
                objective.id, 0
            )
            projected_verification = current_verification + (
                planned_verification_jobs.get(objective.id, 0)
            )
            current = min(projected_total, projected_verification + 1)
            missing = max(0, self.SERVICEABILITY_TARGET - current)
            if missing <= 0:
                continue
            objective_misconceptions = tuple(
                row["misconception_id"]
                for row in misconception_rows_by_objective.get(objective.id, ())[:3]
            )
            chosen_sources = source_ids or tuple(
                sources_by_objective.get(objective.id, ())
            )
            if not chosen_sources:
                chosen_sources = tuple(
                    sources_by_concept.get(objective.primary_concept_id, ())
                ) or available_sources
            for offset in range(missing):
                projected_current = current + offset
                blueprint = self._blueprint(
                    release_id=release_id,
                    concept_id=objective.primary_concept_id,
                    concept_name=graph.concepts[objective.primary_concept_id].name,
                    kind="transfer",
                    target_difficulty=self.DIFFICULTY_SEQUENCE[
                        projected_current % len(self.DIFFICULTY_SEQUENCE)
                    ],
                    misconception_ids=objective_misconceptions,
                    source_ids=chosen_sources,
                    objective=objective,
                    target_misconception_id=None,
                    coverage_goal="objective_serviceability",
                    family_constraint=(
                        "Create a new independent direct family for learning objective "
                        f"{objective.id}; preserve a distinct verification route and do "
                        "not paraphrase an existing family."
                    ),
                )
                gaps.append(
                    CoverageGap(
                        4.5 + 0.1 * missing,
                        blueprint,
                        projected_current,
                        self.SERVICEABILITY_TARGET,
                    )
                )
        gaps.sort(
            key=lambda gap: (
                -gap.priority,
                gap.blueprint.concept_id,
                gap.blueprint.learning_objective_id or "",
                gap.blueprint.target_misconception_id or "",
                gap.blueprint.kind,
                gap.blueprint.target_difficulty,
            )
        )
        if (
            concept_filter is not None
            and concept_filter not in graph.concepts
        ):
            raise NotFoundError(
                f"Unknown concept in active corpus release: {concept_filter}"
            )
        if (
            objective_filter is not None
            and objective_filter not in objective_by_id
        ):
            raise NotFoundError(
                "Unknown learning objective in active corpus release: "
                f"{objective_filter}"
            )
        known_misconception_ids = {
            row["id"] for row in misconception_rows
        }
        if (
            misconception_filter is not None
            and misconception_filter not in known_misconception_ids
        ):
            raise NotFoundError(
                "Unknown misconception in active corpus release: "
                f"{misconception_filter}"
            )
        gaps = [
            gap
            for gap in gaps
            if (
                topic_concepts is None
                or gap.blueprint.concept_id in topic_concepts
            )
            and (
                concept_filter is None
                or gap.blueprint.concept_id == concept_filter
            )
            and (
                objective_filter is None
                or gap.blueprint.learning_objective_id
                == objective_filter
            )
            and (
                misconception_filter is None
                or misconception_filter
                in gap.blueprint.misconception_ids
            )
            and (
                kind_filter is None
                or gap.blueprint.kind == kind_filter
            )
            and (
                goal_filter is None
                or gap.blueprint.coverage_goal == goal_filter
            )
            and (
                maximum_difficulty is None
                or gap.blueprint.target_difficulty
                <= float(maximum_difficulty)
            )
        ]
        return gaps[: max(0, limit)]

    @staticmethod
    def _blueprint(
        *,
        release_id: str,
        concept_id: str,
        concept_name: str,
        kind: str,
        target_difficulty: float,
        misconception_ids: tuple[str, ...],
        source_ids: tuple[str, ...],
        objective: LearningObjective | None,
        target_misconception_id: str | None,
        coverage_goal: str,
        family_constraint: str,
    ) -> GenerationBlueprint:
        return GenerationBlueprint(
            concept_id=concept_id,
            concept_name=concept_name,
            kind=kind,
            target_difficulty=target_difficulty,
            misconception_ids=misconception_ids,
            source_ids=source_ids,
            family_constraint=family_constraint,
            corpus_release_id=release_id,
            learning_objective_id=(objective.id if objective is not None else None),
            learning_objective_name=(objective.name if objective is not None else None),
            learning_objective_description=(
                objective.description if objective is not None else None
            ),
            learning_objective_operation=(
                objective.operation.value if objective is not None else None
            ),
            learning_objective_evidence_type=(
                objective.evidence_type if objective is not None else None
            ),
            target_misconception_id=target_misconception_id,
            coverage_goal=coverage_goal,
        )

    def enqueue(self, gaps: list[CoverageGap]) -> list[str]:
        job_ids: list[str] = []
        with self.database.transaction() as connection:
            existing = {
                json.dumps(
                    asdict(
                        _parse_blueprint(
                            row["blueprint_json"],
                            label=f"Generation job {row['id']} blueprint",
                        )
                    ),
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


class QuarantineReviewQueue:
    """Inspect release-pinned candidates without creating an activation path."""

    def __init__(self, database: Database):
        self.database = database

    @staticmethod
    def _question_payload(question: Question) -> dict[str, Any]:
        return {
            "id": question.id,
            "version": question.version,
            "family_id": question.family_id,
            "status": question.status.value,
            "stem": question.stem,
            "kind": question.kind.value,
            "difficulty": float(question.difficulty),
            "discrimination": float(question.discrimination),
            "guess_rate": float(question.guess_rate),
            "slip_rate": float(question.slip_rate),
            "learning_objective_id": question.objective_id,
            "concepts": [
                {
                    "concept_id": mapping.concept_id,
                    "weight": float(mapping.weight),
                    "role": mapping.role.value,
                }
                for mapping in question.concepts
            ],
            "options": [
                {
                    "id": option.id,
                    "text": option.text,
                    "correct": option.correct,
                    "rationale": option.rationale,
                    "misconception_id": option.misconception_id,
                    "diagnostic_objective_id": (
                        option.diagnostic_objective_id
                    ),
                }
                for option in question.options
            ],
            "source_ids": list(question.source_ids),
            "provenance": copy.deepcopy(question.provenance),
            "tags": list(question.tags),
            "revision_of": question.revision_of,
        }

    @staticmethod
    def _provenance_summary(
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "generated": provenance.get("generated"),
            "provider": provenance.get("provider"),
            "batch_id": provenance.get("batch_id"),
            "review_status": provenance.get("review_status"),
            "independent_review": provenance.get("independent_review"),
            "human_review": provenance.get("human_review"),
            "human_review_status": provenance.get(
                "human_review_status"
            ),
            "activation": provenance.get("activation"),
            "source_gap_id": provenance.get("source_gap_id"),
        }

    @staticmethod
    def _objective_payload(
        objective: LearningObjective,
        content_hash: str,
    ) -> dict[str, Any]:
        return {
            "id": objective.id,
            "content_sha256": content_hash,
            "name": objective.name,
            "description": objective.description,
            "primary_concept_id": objective.primary_concept_id,
            "supporting_concept_ids": list(
                objective.supporting_concept_ids
            ),
            "operation": objective.operation.value,
            "evidence_type": objective.evidence_type,
            "prior_mastery": float(objective.prior_mastery),
        }

    @staticmethod
    def _concept_payload(
        concept: Concept,
        content_hash: str,
    ) -> dict[str, Any]:
        return {
            "id": concept.id,
            "content_sha256": content_hash,
            "name": concept.name,
            "description": concept.description,
            "domain": concept.domain,
            "prior_mastery": float(concept.prior_mastery),
        }

    @staticmethod
    def _misconception_payload(
        misconception: Misconception,
        content_hash: str,
    ) -> dict[str, Any]:
        return {
            "id": misconception.id,
            "content_sha256": content_hash,
            "concept_id": misconception.concept_id,
            "name": misconception.name,
            "description": misconception.description,
        }

    def list(
        self,
        *,
        topic: str | None = None,
        concept_id: str | None = None,
        learning_objective_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValidationError(
                "Quarantine list limit must be an integer from 1 to 500."
            )
        for label, value in (
            ("topic", topic),
            ("concept", concept_id),
            ("objective", learning_objective_id),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValidationError(
                    f"Quarantine {label} filter must be a non-blank string."
                )
        with self.database.read() as connection:
            release_id = self.database.get_active_release_id(connection)
            if concept_id is not None and connection.execute(
                """SELECT 1 FROM release_concepts
                   WHERE release_id = ? AND concept_id = ?""",
                (release_id, concept_id),
            ).fetchone() is None:
                raise NotFoundError(
                    f"Unknown concept in active corpus release: {concept_id}"
                )
            if learning_objective_id is not None and connection.execute(
                """SELECT 1 FROM release_learning_objectives
                   WHERE release_id = ? AND objective_id = ?""",
                (release_id, learning_objective_id),
            ).fetchone() is None:
                raise NotFoundError(
                    "Unknown learning objective in active corpus release: "
                    f"{learning_objective_id}"
                )
        topic_ids: set[str] | None = None
        if topic is not None:
            resolved = self.database.resolve_topic(topic, release_id)
            catalog = self.database.get_catalog(release_id)
            topic_ids = {resolved["id"]}
            changed = True
            while changed:
                changed = False
                for candidate in catalog["topics"]:
                    if (
                        candidate["parent_id"] in topic_ids
                        and candidate["id"] not in topic_ids
                    ):
                        topic_ids.add(candidate["id"])
                        changed = True

        where = [
            "membership.release_id = ?",
            "membership.status = 'quarantined'",
            "question.status = 'quarantined'",
        ]
        parameters: list[Any] = [release_id]
        if topic_ids is not None:
            where.append(
                "EXISTS (SELECT 1 FROM release_question_topics "
                "topic_mapping WHERE topic_mapping.release_id = "
                "membership.release_id AND topic_mapping.question_id = "
                "question.id AND topic_mapping.topic_id IN "
                "(SELECT value FROM json_each(?)))"
            )
            parameters.append(_canonical_json(sorted(topic_ids)))
        if concept_id is not None:
            where.append(
                "EXISTS (SELECT 1 FROM question_concepts "
                "concept_filter_mapping WHERE "
                "concept_filter_mapping.question_id = question.id "
                "AND concept_filter_mapping.concept_id = ?)"
            )
            parameters.append(concept_id)
        if learning_objective_id is not None:
            where.append("objective_mapping.objective_id = ?")
            parameters.append(learning_objective_id)
        parameters.append(limit)
        with self.database.read() as connection:
            rows = connection.execute(
                f"""SELECT question.id AS question_id,
                           question.version,
                           question.content_hash,
                           question.family_id,
                           question.kind,
                           question.difficulty,
                           question.provenance_json,
                           primary_mapping.concept_id
                               AS primary_concept_id,
                           objective_mapping.objective_id,
                           COUNT(DISTINCT source.source_id)
                               AS source_count,
                           COUNT(DISTINCT review.id) AS review_count,
                           revocation.question_id IS NOT NULL AS revoked,
                           revocation.reason AS revocation_reason,
                           revocation.revoked_at
                    FROM release_questions membership
                    JOIN questions question
                      ON question.id = membership.question_id
                    JOIN question_concepts primary_mapping
                      ON primary_mapping.question_id = question.id
                     AND primary_mapping.role = 'primary'
                    LEFT JOIN release_question_objectives objective_mapping
                      ON objective_mapping.release_id = membership.release_id
                     AND objective_mapping.question_id = question.id
                    LEFT JOIN question_sources source
                      ON source.question_id = question.id
                    LEFT JOIN item_reviews review
                      ON review.question_id = question.id
                    LEFT JOIN question_revocations revocation
                      ON revocation.question_id = question.id
                    WHERE {' AND '.join(where)}
                    GROUP BY question.id, primary_mapping.concept_id,
                             objective_mapping.objective_id
                    ORDER BY question.difficulty, question.id
                    LIMIT ?""",
                tuple(parameters),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            provenance = _decode_json(
                row["provenance_json"],
                label=(
                    f"Quarantined question {row['question_id']} provenance"
                ),
                expected_type=dict,
            )
            result.append(
                {
                    "question_id": row["question_id"],
                    "version": row["version"],
                    "question_content_sha256": row["content_hash"],
                    "family_id": row["family_id"],
                    "kind": row["kind"],
                    "difficulty": row["difficulty"],
                    "primary_concept_id": row["primary_concept_id"],
                    "learning_objective_id": row["objective_id"],
                    "source_count": int(row["source_count"]),
                    "recorded_item_review_count": int(
                        row["review_count"]
                    ),
                    "revoked": bool(row["revoked"]),
                    "revocation_reason": row["revocation_reason"],
                    "revoked_at": row["revoked_at"],
                    "runtime_eligible": False,
                    "activation_ceiling": (
                        "revoked" if row["revoked"] else "quarantined"
                    ),
                    "provenance_claims": self._provenance_summary(
                        provenance
                    ),
                }
            )
        return result

    def show(self, question_id: str) -> dict[str, Any]:
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValidationError(
                "Quarantined question ID must be a non-blank string."
            )
        with self.database.read() as connection:
            release_id = self.database.get_active_release_id(connection)
            release = connection.execute(
                """SELECT id, bundle_hash, created_at, sealed_at
                   FROM corpus_releases WHERE id = ?""",
                (release_id,),
            ).fetchone()
            membership = connection.execute(
                """SELECT question.content_hash, membership.status,
                          membership.evidence_weight
                   FROM release_questions membership
                   JOIN questions question
                     ON question.id = membership.question_id
                   WHERE membership.release_id = ?
                     AND membership.question_id = ?""",
                (release_id, question_id),
            ).fetchone()
            if (
                membership is None
                or membership["status"] != "quarantined"
            ):
                raise NotFoundError(
                    f"Question {question_id} is not quarantined in the "
                    f"active release {release_id}."
                )
            source_rows = connection.execute(
                """SELECT source.id, source.content_hash, source.title,
                          source.uri, source.license, source.metadata_json
                   FROM question_sources mapping
                   JOIN sources source ON source.id = mapping.source_id
                   JOIN release_sources release_source
                     ON release_source.release_id = ?
                    AND release_source.source_id = source.id
                   WHERE mapping.question_id = ?
                   ORDER BY source.id""",
                (release_id, question_id),
            ).fetchall()
            review_rows = connection.execute(
                """SELECT * FROM item_reviews
                   WHERE question_id = ?
                   ORDER BY created_at, id""",
                (question_id,),
            ).fetchall()
            revocation = connection.execute(
                """SELECT reason, revoked_at, event_id
                   FROM question_revocations WHERE question_id = ?""",
                (question_id,),
            ).fetchone()
            question = self.database.get_question(
                question_id,
                connection,
                release_id=release_id,
            )
            referenced_objective_ids = sorted(
                {
                    objective_id
                    for objective_id in (
                        question.objective_id,
                        *(
                            option.diagnostic_objective_id
                            for option in question.options
                        ),
                    )
                    if objective_id is not None
                }
            )
            concept_rows = connection.execute(
                """SELECT concept.* FROM concepts concept
                   JOIN release_concepts membership
                     ON membership.concept_id = concept.id
                   WHERE membership.release_id = ?
                     AND concept.id IN (SELECT value FROM json_each(?))
                   ORDER BY concept.id""",
                (
                    release_id,
                    _canonical_json(
                        sorted(
                            mapping.concept_id
                            for mapping in question.concepts
                        )
                    ),
                ),
            ).fetchall()
            misconception_rows = connection.execute(
                """SELECT misconception.* FROM misconceptions misconception
                   JOIN release_misconceptions membership
                     ON membership.misconception_id = misconception.id
                   WHERE membership.release_id = ?
                     AND misconception.id IN (
                         SELECT value FROM json_each(?)
                     )
                   ORDER BY misconception.id""",
                (
                    release_id,
                    _canonical_json(
                        sorted(question.misconception_ids)
                    ),
                ),
            ).fetchall()
            objective_rows = connection.execute(
                """SELECT objective.*
                   FROM learning_objectives objective
                   JOIN release_learning_objectives membership
                     ON membership.objective_id = objective.id
                   WHERE membership.release_id = ?
                     AND objective.id IN (
                         SELECT value FROM json_each(?)
                     )
                   ORDER BY objective.id""",
                (
                    release_id,
                    _canonical_json(referenced_objective_ids),
                ),
            ).fetchall()
            comparison_rows = connection.execute(
                """SELECT candidate.id, candidate.content_hash,
                          candidate.family_id, candidate.kind,
                          candidate.difficulty, candidate.stem,
                          candidate.provenance_json, candidate.revision_of,
                          candidate_membership.status,
                          (
                              SELECT mapping.objective_id
                              FROM release_question_objectives mapping
                              WHERE mapping.release_id = ?
                                AND mapping.question_id = candidate.id
                          ) AS objective_id,
                          (
                              SELECT mapping.concept_id
                              FROM question_concepts mapping
                              WHERE mapping.question_id = candidate.id
                                AND mapping.role = 'primary'
                          ) AS primary_concept_id
                   FROM release_questions candidate_membership
                   JOIN questions candidate
                     ON candidate.id = candidate_membership.question_id
                   WHERE candidate_membership.release_id = ?
                     AND candidate.id != ?
                     AND (
                         EXISTS (
                             SELECT 1
                             FROM release_question_objectives mapping
                             WHERE mapping.release_id = ?
                               AND mapping.question_id = candidate.id
                               AND mapping.objective_id = ?
                         )
                         OR EXISTS (
                             SELECT 1 FROM question_concepts mapping
                             WHERE mapping.question_id = candidate.id
                               AND mapping.role = 'primary'
                               AND mapping.concept_id = ?
                         )
                     )
                   ORDER BY candidate_membership.status,
                            candidate.difficulty, candidate.id
                   LIMIT 101""",
                (
                    release_id,
                    release_id,
                    question_id,
                    release_id,
                    question.objective_id,
                    question.primary_concept_id,
                ),
            ).fetchall()
        if (
            question.status is not QuestionStatus.QUARANTINED
            or float(membership["evidence_weight"]) != 0.0
        ):
            raise ValidationError(
                f"Question {question_id} has an inconsistent quarantine "
                "status or nonzero release evidence weight."
            )
        computed_hash = question_content_hash(question)
        if computed_hash != membership["content_hash"]:
            raise ValidationError(
                f"Quarantined question {question_id} content hash does not "
                "match its stored immutable identity."
            )
        if (
            len(source_rows) != len(question.source_ids)
            or {row["id"] for row in source_rows}
            != set(question.source_ids)
        ):
            raise ValidationError(
                f"Quarantined question {question_id} source registry is "
                "incomplete for its release."
            )
        expected_concept_ids = sorted(
            mapping.concept_id for mapping in question.concepts
        )
        if (
            len(concept_rows) != len(expected_concept_ids)
            or [row["id"] for row in concept_rows] != expected_concept_ids
        ):
            raise ValidationError(
                f"Quarantined question {question_id} concept registry is "
                "incomplete for its release."
            )
        concepts: list[dict[str, Any]] = []
        for row in concept_rows:
            concept = Concept(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                domain=row["domain"],
                prior_mastery=row["prior_mastery"],
            )
            if concept_content_hash(concept) != row["content_hash"]:
                raise ValidationError(
                    f"Concept {row['id']} content hash does not match its "
                    "stored immutable identity."
                )
            concepts.append(
                self._concept_payload(concept, row["content_hash"])
            )
        if (
            len(objective_rows) != len(referenced_objective_ids)
            or [row["id"] for row in objective_rows]
            != referenced_objective_ids
        ):
            raise ValidationError(
                f"Quarantined question {question_id} objective registry is "
                "incomplete for its release."
            )
        objective_payloads: dict[str, dict[str, Any]] = {}
        for objective_row in objective_rows:
            objective = LearningObjective(
                id=objective_row["id"],
                name=objective_row["name"],
                description=objective_row["description"],
                primary_concept_id=objective_row["primary_concept_id"],
                supporting_concept_ids=tuple(
                    _decode_json(
                        objective_row["supporting_concept_ids_json"],
                        label=(
                            f"Learning objective {objective_row['id']} "
                            "supporting concepts"
                        ),
                        expected_type=list,
                    )
                ),
                operation=ObjectiveOperation(objective_row["operation"]),
                evidence_type=objective_row["evidence_type"],
                prior_mastery=objective_row["prior_mastery"],
            )
            if (
                objective_content_hash(objective)
                != objective_row["content_hash"]
            ):
                raise ValidationError(
                    f"Learning objective {objective.id} content hash does not "
                    "match its stored immutable identity."
                )
            objective_payloads[objective.id] = self._objective_payload(
                objective, objective_row["content_hash"]
            )
        learning_objective = (
            objective_payloads[question.objective_id]
            if question.objective_id is not None
            else None
        )
        diagnostic_objectives = [
            objective_payloads[objective_id]
            for objective_id in sorted(
                {
                    option.diagnostic_objective_id
                    for option in question.options
                    if option.diagnostic_objective_id is not None
                }
            )
        ]
        expected_misconception_ids = sorted(question.misconception_ids)
        if (
            len(misconception_rows) != len(expected_misconception_ids)
            or [row["id"] for row in misconception_rows]
            != expected_misconception_ids
        ):
            raise ValidationError(
                f"Quarantined question {question_id} misconception registry "
                "is incomplete for its release."
            )
        misconceptions: list[dict[str, Any]] = []
        for row in misconception_rows:
            misconception = Misconception(
                id=row["id"],
                concept_id=row["concept_id"],
                name=row["name"],
                description=row["description"],
            )
            if (
                misconception_content_hash(misconception)
                != row["content_hash"]
            ):
                raise ValidationError(
                    f"Misconception {row['id']} content hash does not match "
                    "its stored immutable identity."
                )
            misconceptions.append(
                self._misconception_payload(
                    misconception, row["content_hash"]
                )
            )
        sources: list[dict[str, Any]] = []
        for row in source_rows:
            metadata = _decode_json(
                row["metadata_json"],
                label=f"Source {row['id']} metadata",
                expected_type=dict,
            )
            source = Source(
                id=row["id"],
                title=row["title"],
                uri=row["uri"],
                license=row["license"],
                metadata=metadata,
            )
            if source_content_hash(source) != row["content_hash"]:
                raise ValidationError(
                    f"Source {row['id']} content hash does not match its "
                    "stored immutable identity."
                )
            sources.append(
                {
                    "id": row["id"],
                    "content_sha256": row["content_hash"],
                    "title": row["title"],
                    "uri": row["uri"],
                    "license": row["license"],
                    "metadata": metadata,
                }
            )
        reviews = [
            {
                "review_id": row["id"],
                "reviewer_kind": row["reviewer_kind"],
                "verdict": row["verdict"],
                "issues": _decode_json(
                    row["issues_json"],
                    label=f"Item review {row['id']} issues",
                    expected_type=list,
                ),
                "metadata": _decode_json(
                    row["metadata_json"],
                    label=f"Item review {row['id']} metadata",
                    expected_type=dict,
                ),
                "created_at": row["created_at"],
            }
            for row in review_rows
        ]
        local_comparison_truncated = len(comparison_rows) > 100
        local_family_comparison: list[dict[str, Any]] = []
        for row in comparison_rows[:100]:
            candidate = self.database.get_question(
                row["id"], release_id=release_id
            )
            if question_content_hash(candidate) != row["content_hash"]:
                raise ValidationError(
                    f"Comparison question {row['id']} content hash does not "
                    "match its stored immutable identity."
                )
            local_family_comparison.append(
                {
                    "question_content_sha256": row["content_hash"],
                    "release_status": row["status"],
                    "same_learning_objective": (
                        question.objective_id is not None
                        and row["objective_id"] == question.objective_id
                    ),
                    "same_primary_concept": (
                        row["primary_concept_id"]
                        == question.primary_concept_id
                    ),
                    "independence_note": candidate.provenance.get(
                        "independence_note"
                    ),
                    "question": self._question_payload(candidate),
                }
            )
        return {
            "release": {
                "release_id": release["id"],
                "bundle_sha256": release["bundle_hash"],
                "created_at": release["created_at"],
                "sealed_at": release["sealed_at"],
            },
            "question_content_sha256": membership["content_hash"],
            "release_status": membership["status"],
            "release_evidence_weight": membership["evidence_weight"],
            "runtime_eligible": False,
            "activation_ceiling": (
                "revoked" if revocation is not None else "quarantined"
            ),
            "question": self._question_payload(question),
            "topics": self.database.question_topics(
                question_id, release_id
            ),
            "concepts": concepts,
            "learning_objective": learning_objective,
            "diagnostic_objectives": diagnostic_objectives,
            "sources": sources,
            "misconceptions": misconceptions,
            "local_family_comparison": local_family_comparison,
            "local_comparison_scope": {
                "included_if_same_direct_learning_objective": True,
                "included_if_same_primary_concept": True,
                "secondary_concept_only_matches_included": False,
                "cross_topic_only_matches_included": False,
                "unrelated_objective_matches_included": False,
                "maximum_rows": 100,
            },
            "local_comparison_scope_complete": (
                not local_comparison_truncated
            ),
            "local_comparison_total_lower_bound": len(comparison_rows),
            "recorded_item_reviews": reviews,
            "revocation": (
                {
                    "reason": revocation["reason"],
                    "revoked_at": revocation["revoked_at"],
                    "event_id": revocation["event_id"],
                }
                if revocation is not None
                else None
            ),
            "provenance_claims": self._provenance_summary(
                question.provenance
            ),
            "review_boundary": (
                "Stored provenance and AI-review labels are claims to inspect, "
                "not human activation authority."
            ),
        }

    def packet(
        self,
        question_id: str,
        *,
        stage: str = "combined",
    ) -> dict[str, Any]:
        if stage not in {"combined", "blind", "critic"}:
            raise ValidationError(
                f"Unknown quarantine review packet stage: {stage!r}."
            )
        detail = self.show(question_id)
        blind_solver_material = _blind_for_review(detail["question"])
        if detail["learning_objective"] is not None:
            blind_solver_material["learning_objective"] = copy.deepcopy(
                detail["learning_objective"]
            )
        critic_material = {
            "question": detail["question"],
            "topics": detail["topics"],
            "concepts": detail["concepts"],
            "learning_objective": detail["learning_objective"],
            "diagnostic_objectives": detail["diagnostic_objectives"],
            "source_registry": detail["sources"],
            "misconceptions": detail["misconceptions"],
            "recorded_item_reviews": detail[
                "recorded_item_reviews"
            ],
            "local_family_comparison": detail[
                "local_family_comparison"
            ],
            "local_comparison_scope": detail[
                "local_comparison_scope"
            ],
            "local_comparison_scope_complete": detail[
                "local_comparison_scope_complete"
            ],
            "local_comparison_total_lower_bound": detail[
                "local_comparison_total_lower_bound"
            ],
        }
        release_identity = {
            "release_id": detail["release"]["release_id"],
            "bundle_sha256": detail["release"]["bundle_sha256"],
        }
        packet_core = {
            "schema": QUARANTINE_REVIEW_PACKET_SCHEMA,
            "packet_audience": "review_coordinator",
            "question_id": detail["question"]["id"],
            "activation_ceiling": detail["activation_ceiling"],
            "runtime_eligible": False,
            "release": release_identity,
            "question_content_sha256": detail[
                "question_content_sha256"
            ],
            "review_sequence": (
                "Complete the blind solver stage using only "
                "blind_solver_material before revealing critic_material."
            ),
            "blind_solver_material": blind_solver_material,
            "blind_solver_material_sha256": _sha256_json(
                blind_solver_material
            ),
            "critic_material": critic_material,
            "critic_material_sha256": _sha256_json(critic_material),
            "revocation": detail["revocation"],
            "review_contract": {
                "source_claim_review_required": True,
                "blind_solver_review_required": True,
                "blind_solver_must_not_receive_critic_material": True,
                "combined_packet_itself_enforces_stage_isolation": False,
                "keyed_critic_follows_blind_solver": True,
                "one_best_answer_review_required": True,
                "named_misconception_review_required": True,
                "family_independence_review_required": True,
                "local_comparison_scope_complete": detail[
                    "local_comparison_scope_complete"
                ],
                "family_independence_claim_resolved": False,
                "human_activation_review_required": (
                    detail["revocation"] is None
                ),
                "new_immutable_revision_required_if_reconsidered": (
                    detail["revocation"] is not None
                ),
                "human_review_required_for_new_revision": (
                    detail["revocation"] is not None
                ),
                "automatic_activation_permitted": False,
                "revoked_identity_must_not_be_promoted": (
                    detail["revocation"] is not None
                ),
            },
            "evidence_boundaries": [
                (
                    "Source registry hashes bind this packet to registered "
                    "metadata, not to unpersisted source excerpts."
                ),
                (
                    "The packet is an inspection artifact and carries no "
                    "signature, attestation, promotion, or activation effect."
                ),
                (
                    "The stored independence note and comparison set are "
                    "review inputs, not proof of family independence."
                ),
                (
                    "The combined packet is coordinator-only. Use a blind "
                    "stage export when handing material to a blind solver."
                ),
                (
                    "Authored difficulty is an uncalibrated prior until "
                    "reviewed pilot evidence exists."
                ),
            ],
        }
        combined = {
            **packet_core,
            "packet_sha256": _sha256_json(packet_core),
        }
        if stage == "combined":
            return combined
        material_key = (
            "blind_solver_material"
            if stage == "blind"
            else "critic_material"
        )
        material = combined[material_key]
        stage_core = {
            "schema": "tsq-quarantine-review-stage-v1",
            "stage": stage,
            "question_id": detail["question"]["id"],
            "activation_ceiling": detail["activation_ceiling"],
            "runtime_eligible": False,
            "release": release_identity,
            "question_content_sha256": detail[
                "question_content_sha256"
            ],
            "material": material,
            "material_sha256": _sha256_json(material),
            "coordinator_packet_sha256": combined["packet_sha256"],
            "stage_boundary": (
                "This export contains no keyed critic material."
                if stage == "blind"
                else (
                    "Reveal this keyed material only after the blind solver "
                    "stage is complete."
                )
            ),
        }
        return {
            **stage_core,
            "packet_sha256": _sha256_json(stage_core),
        }


class AuthoringJobs:
    """Read and transition quarantined authoring work without activating items."""

    STATUSES = frozenset({"planned", "running", "reviewed", "rejected", "failed"})

    def __init__(self, database: Database):
        self.database = database

    @staticmethod
    def _run_record(row: Any) -> dict[str, Any]:
        record = dict(row)
        record["raw_output"] = _decode_json(
            record.pop("raw_output_json"),
            label=f"Generation run {record['id']} raw output",
            expected_type=dict,
            nullable=True,
        )
        record["validation"] = _decode_json(
            record.pop("validation_json"),
            label=f"Generation run {record['id']} validation",
            expected_type=dict,
            nullable=True,
        )
        record["error"] = _decode_json(
            record.pop("error_json"),
            label=f"Generation run {record['id']} error",
            expected_type=dict,
            nullable=True,
        )
        return record

    @staticmethod
    def _job_record(row: Any, *, include_artifact: bool) -> dict[str, Any]:
        record = dict(row)
        record["blueprint"] = asdict(
            _parse_blueprint(
                record.pop("blueprint_json"),
                label=f"Generation job {record['id']} blueprint",
            )
        )
        raw_json = record.pop("raw_output_json")
        validation_json = record.pop("validation_json")
        if include_artifact:
            record["raw_output"] = _decode_json(
                raw_json,
                label=f"Generation job {record['id']} raw output",
                expected_type=dict,
                nullable=True,
            )
            record["validation"] = _decode_json(
                validation_json,
                label=f"Generation job {record['id']} validation",
                expected_type=dict,
                nullable=True,
            )
        else:
            record["has_artifact"] = raw_json is not None
            record["has_validation"] = validation_json is not None
        return record

    def list(self, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if status is not None and status not in self.STATUSES:
            raise ValidationError(f"Unknown generation job status: {status!r}.")
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValidationError("Generation job list limit must be an integer from 1 to 500.")
        where = "WHERE job.status = ?" if status is not None else ""
        parameters: tuple[Any, ...] = (status, limit) if status is not None else (limit,)
        with self.database.read() as connection:
            rows = connection.execute(
                f"""SELECT job.*, COUNT(run.id) AS run_count
                    FROM generation_jobs job
                    LEFT JOIN generation_job_runs run ON run.job_id = job.id
                    {where}
                    GROUP BY job.id
                    ORDER BY job.created_at DESC, job.id DESC
                    LIMIT ?""",
                parameters,
            ).fetchall()
        return [self._job_record(row, include_artifact=False) for row in rows]

    def show(self, job_id: str) -> dict[str, Any]:
        if type(job_id) is not str or not job_id.strip():
            raise ValidationError("Generation job ID must be a non-empty string.")
        with self.database.read() as connection:
            row = connection.execute(
                """SELECT job.*, COUNT(run.id) AS run_count
                   FROM generation_jobs job
                   LEFT JOIN generation_job_runs run ON run.job_id = job.id
                   WHERE job.id = ? GROUP BY job.id""",
                (job_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Unknown generation job: {job_id}")
            run_rows = connection.execute(
                """SELECT * FROM generation_job_runs
                   WHERE job_id = ? ORDER BY attempt""",
                (job_id,),
            ).fetchall()
        record = self._job_record(row, include_artifact=True)
        record["runs"] = [self._run_record(run) for run in run_rows]
        return record

    def reviews(self, job_id: str) -> dict[str, Any]:
        job = self.show(job_id)
        review_attempts: list[dict[str, Any]] = []
        for run in job["runs"]:
            validation = run["validation"]
            if validation is None:
                continue
            reviews = validation.get("reviews")
            if type(reviews) is not list:
                raise ValidationError(
                    f"Generation run {run['id']} validation has no JSON review array."
                )
            if any(type(review) is not dict for review in reviews):
                raise ValidationError(
                    f"Generation run {run['id']} contains a malformed review record."
                )
            review_attempts.append(
                {
                    "attempt": run["attempt"],
                    "run_id": run["id"],
                    "status": run["status"],
                    "reviews": reviews,
                }
            )
        return {
            "job_id": job_id,
            "job_status": job["status"],
            "review_attempts": review_attempts,
        }

    def retry(self, job_id: str, *, recover_running: bool = False) -> dict[str, Any]:
        if type(recover_running) is not bool:
            raise TypeError("recover_running must be a boolean.")
        now = datetime.now(timezone.utc).isoformat()
        with self.database.transaction() as connection:
            job = connection.execute(
                "SELECT * FROM generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if job is None:
                raise NotFoundError(f"Unknown generation job: {job_id}")
            status = job["status"]
            if status == "running":
                if not recover_running:
                    raise ConflictError(
                        f"Generation job {job_id} is running; use explicit running "
                        "recovery only after confirming no worker still owns it."
                    )
                running_runs = connection.execute(
                    """SELECT * FROM generation_job_runs
                       WHERE job_id = ? AND status = 'running' ORDER BY attempt""",
                    (job_id,),
                ).fetchall()
                if len(running_runs) != 1:
                    raise ValidationError(
                        f"Generation job {job_id} does not have exactly one running attempt."
                    )
                error = {
                    "error_type": "InterruptedGenerationRun",
                    "error": "Running attempt was explicitly recovered by an operator.",
                }
                error_json = _canonical_json(error)
                connection.execute(
                    """UPDATE generation_job_runs
                       SET status='failed', error_json=?, completed_at=? WHERE id=?""",
                    (error_json, now, running_runs[0]["id"]),
                )
                connection.execute(
                    """UPDATE generation_jobs
                       SET status='failed', validation_json=?, updated_at=? WHERE id=?""",
                    (error_json, now, job_id),
                )
                status = "failed"
            if status not in {"failed", "rejected"}:
                raise ConflictError(
                    f"Generation job {job_id} is {status}; only failed or rejected "
                    "jobs can be retried."
                )
            latest = connection.execute(
                """SELECT status FROM generation_job_runs
                   WHERE job_id = ? ORDER BY attempt DESC LIMIT 1""",
                (job_id,),
            ).fetchone()
            if latest is None or latest["status"] != status:
                raise ValidationError(
                    f"Generation job {job_id} summary does not match immutable run history."
                )
            updated = connection.execute(
                """UPDATE generation_jobs
                   SET status='planned', provider=NULL, model=NULL,
                       raw_output_json=NULL, validation_json=NULL, updated_at=?
                   WHERE id=? AND status=?""",
                (now, job_id, status),
            )
            if updated.rowcount != 1:
                raise ConflictError(f"Generation job {job_id} changed while retrying.")
        return self.show(job_id)


class DeterministicTestGenerator:
    """Explicit test-only provider for exercising authoring operations offline."""

    provider_name = "deterministic-test-generator"
    model_name = "blueprint-fixture-v2"

    def __init__(self, misconceptions: dict[str, tuple[str, str, str]]):
        self._misconceptions = copy.deepcopy(misconceptions)

    def generate(
        self, blueprint: GenerationBlueprint, source_context: str
    ) -> dict[str, Any]:
        material = {
            "blueprint": asdict(blueprint),
            "source_context_sha256": _sha256_text(source_context),
        }
        digest = _sha256_json(material)
        misconception_ids = blueprint.misconception_ids
        if not misconception_ids:
            # This deliberately yields an invalid artifact rather than inventing
            # an unnamed distractor. Deterministic validation will reject it.
            misconception_ids = ("missing_named_misconception",)
        distractor_ids = tuple(
            misconception_ids[index % len(misconception_ids)] for index in range(3)
        )
        distractor_options: list[dict[str, Any]] = []
        mapped_owner_ids: list[str] = []
        option_ids = ("a", "c", "d")
        for index, (option_id, misconception_id) in enumerate(
            zip(option_ids, distractor_ids, strict=True), start=1
        ):
            name, description, owner_id = self._misconceptions.get(
                misconception_id,
                (
                    misconception_id,
                    "No registered misconception description is available.",
                    blueprint.concept_id,
                ),
            )
            if owner_id != blueprint.concept_id and owner_id not in mapped_owner_ids:
                mapped_owner_ids.append(owner_id)
            distractor_options.append(
                {
                    "id": option_id,
                    "text": (
                        f"Adopt misconception {index}, {name}: {description}"
                    ),
                    "correct": False,
                    "misconception_id": misconception_id,
                    "diagnostic_objective_id": blueprint.learning_objective_id,
                    "rationale": (
                        f"This option directly states the named misconception {name}; "
                        "it is retained only as a quarantined fixture distractor."
                    ),
                }
            )
        if mapped_owner_ids:
            primary_weight = 0.70
            supporting_weight = (1.0 - primary_weight) / len(mapped_owner_ids)
        else:
            primary_weight = 1.0
            supporting_weight = 0.0
        concepts = [
            {
                "concept_id": blueprint.concept_id,
                "weight": primary_weight,
                "role": "primary",
            }
        ]
        concepts.extend(
            {
                "concept_id": owner_id,
                "weight": supporting_weight,
                "role": "supporting",
            }
            for owner_id in mapped_owner_ids
        )
        item = {
            "id": f"q_generated_fixture_{digest[:20]}",
            "version": 1,
            "family_id": f"f_generated_fixture_{digest[20:40]}",
            "status": "quarantined",
            "stem": (
                f"A practitioner is evaluating a claim about {blueprint.concept_name}. "
                "Which response best separates supported reasoning from a named "
                "misconception under the supplied source context?"
            ),
            "kind": blueprint.kind,
            "difficulty": blueprint.target_difficulty,
            "discrimination": 1.0,
            "guess_rate": 0.25,
            "slip_rate": 0.05,
            "concepts": concepts,
            "source_ids": list(blueprint.source_ids),
            "options": [
                distractor_options[0],
                {
                    "id": "b",
                    "text": (
                        "State the governing assumptions, compare the claim with the "
                        "approved source evidence, and preserve unresolved uncertainty."
                    ),
                    "correct": True,
                    "misconception_id": None,
                    "diagnostic_objective_id": None,
                    "rationale": (
                        "This response tests assumptions against evidence without "
                        "turning an unsupported shortcut into a conclusion."
                    ),
                },
                distractor_options[1],
                distractor_options[2],
            ],
            "provenance": {"deterministic_test_fixture": True},
            "tags": ["generated-fixture", "test-provider"],
            "revision_of": None,
        }
        if blueprint.learning_objective_id is not None:
            item["learning_objective_id"] = blueprint.learning_objective_id
        return item


class DeterministicTestReviewer:
    """Separate deterministic shape reviewer for the test provider."""

    reviewer_name = "deterministic-independent-fixture-reviewer"
    provider_name = "deterministic-test-reviewer"
    model_name = "blind-shape-review-v1"

    def review(self, item: dict[str, Any], source_context: str) -> dict[str, Any]:
        options = item.get("options")
        option_ids = (
            [option.get("id") for option in options if type(option) is dict]
            if type(options) is list
            else []
        )
        checks = {
            "nonempty_source_context": bool(source_context.strip()),
            "reasoning_stem": type(item.get("stem")) is str and len(item["stem"]) >= 40,
            "four_distinct_options": (
                len(option_ids) == 4 and len(set(option_ids)) == 4
            ),
            "source_ids_present": (
                type(item.get("source_ids")) is list and bool(item["source_ids"])
            ),
        }
        return {
            "verdict": "accept" if all(checks.values()) else "reject",
            "independent": True,
            "checks": checks,
            "source_context_sha256": _sha256_text(source_context),
        }


def deterministic_test_pipeline(database: Database) -> "OfflineAuthoringPipeline":
    """Build the only bundled provider pair; it is explicit and test-only."""

    with database.read() as connection:
        misconceptions = {
            row["id"]: (row["name"], row["description"], row["concept_id"])
            for row in connection.execute(
                "SELECT id, name, description, concept_id FROM misconceptions ORDER BY id"
            )
        }
    return OfflineAuthoringPipeline(
        database,
        DeterministicTestGenerator(misconceptions),
        (DeterministicTestReviewer(),),
    )


class OfflineAuthoringPipeline:
    """Offline generator/solver/critic orchestration with mandatory quarantine.

    The pipeline deliberately stops at a reviewed artifact. Activation remains an
    explicit corpus-release operation after deterministic checks and, eventually,
    empirical pilot calibration.
    """

    def __init__(
        self,
        database: Database,
        generator: StructuredItemGenerator,
        reviewers: tuple[IndependentItemReviewer, ...],
    ):
        if not reviewers:
            raise ValueError("At least one independent reviewer is required.")
        provider_name = getattr(generator, "provider_name", None)
        model_name = getattr(generator, "model_name", None)
        if type(provider_name) is not str or not provider_name.strip():
            raise ValueError("Generator provider_name must be a non-empty string.")
        if type(model_name) is not str or not model_name.strip():
            raise ValueError("Generator model_name must be a non-empty string.")

        generator_aliases = {
            _normalized_identity(provider_name),
            _normalized_identity(model_name),
            _normalized_identity(f"{provider_name}:{model_name}"),
            _normalized_identity(f"{provider_name}/{model_name}"),
            _normalized_identity(f"{provider_name} {model_name}"),
        }
        reviewer_identities: set[str] = set()
        reviewer_model_identities: set[tuple[str, str]] = set()
        reviewer_provenances: list[dict[str, str]] = []
        for reviewer in reviewers:
            if reviewer is generator:
                raise ValueError("A generator instance cannot review its own item.")
            reviewer_name = getattr(reviewer, "reviewer_name", None)
            if type(reviewer_name) is not str or not reviewer_name.strip():
                raise ValueError("Every reviewer must have a non-empty reviewer_name.")
            normalized = _normalized_identity(reviewer_name)
            if normalized in reviewer_identities:
                raise ValueError(f"Duplicate reviewer identity: {reviewer_name}.")
            if normalized in generator_aliases:
                raise ValueError(
                    f"Reviewer identity {reviewer_name} collides with the generator identity."
                )
            reviewer_provider = getattr(reviewer, "provider_name", None)
            reviewer_model = getattr(reviewer, "model_name", None)
            if reviewer_provider is not None and (
                type(reviewer_provider) is not str or not reviewer_provider.strip()
            ):
                raise ValueError("Reviewer provider_name must be a non-empty string when set.")
            if reviewer_model is not None and (
                type(reviewer_model) is not str or not reviewer_model.strip()
            ):
                raise ValueError("Reviewer model_name must be a non-empty string when set.")
            if (
                type(reviewer_provider) is str
                and type(reviewer_model) is str
            ):
                model_identity = (
                    _normalized_identity(reviewer_provider),
                    _normalized_identity(reviewer_model),
                )
                if model_identity == (
                    _normalized_identity(provider_name),
                    _normalized_identity(model_name),
                ):
                    raise ValueError(
                        f"Reviewer {reviewer_name} declares the same provider/model identity "
                        "as the generator."
                    )
                if model_identity in reviewer_model_identities:
                    raise ValueError(
                        "Reviewers with different labels cannot declare the same "
                        "provider/model identity."
                    )
                reviewer_model_identities.add(model_identity)
            reviewer_identities.add(normalized)
            provenance = {
                "reviewer_name": reviewer_name,
                "implementation": (
                    f"{type(reviewer).__module__}.{type(reviewer).__qualname__}"
                ),
            }
            if type(reviewer_provider) is str:
                provenance["provider_name"] = reviewer_provider
            if type(reviewer_model) is str:
                provenance["model_name"] = reviewer_model
            reviewer_provenances.append(provenance)
        self.database = database
        self.generator = generator
        self.reviewers = reviewers
        # Freeze identity claims at construction so a mutable adapter cannot
        # change its recorded provenance between validation and execution.
        self.generator_provider_name = provider_name
        self.generator_model_name = model_name
        self._reviewer_provenances = tuple(reviewer_provenances)

    @staticmethod
    def _item_shape_issues(item: Any) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []

        def add(message: str) -> None:
            issues.append(
                {
                    "code": "invalid_item_shape",
                    "severity": "error",
                    "message": message,
                }
            )

        if type(item) is not dict:
            add("Generated item must be a JSON object.")
            return issues
        for message in _strict_json_issues(item):
            add(message)
        if issues:
            return issues

        required_scalar_types: dict[str, type[Any]] = {
            "id": str,
            "family_id": str,
            "stem": str,
        }
        for field, expected_type in required_scalar_types.items():
            if field not in item:
                add(f"Generated item is missing required field {field!r}.")
            elif type(item[field]) is not expected_type:
                add(f"Field {field!r} must be {expected_type.__name__}.")
            elif not item[field].strip():
                add(f"Field {field!r} must not be empty.")

        if "version" in item and type(item["version"]) is not int:
            add("Field 'version' must be an integer, not a coerced value.")
        for field in ("status", "kind"):
            if field in item and type(item[field]) is not str:
                add(f"Field {field!r} must be a string.")
        learning_objective_id = item.get("learning_objective_id")
        if learning_objective_id is not None and (
            type(learning_objective_id) is not str
            or not learning_objective_id.strip()
        ):
            add("Field 'learning_objective_id' must be a non-empty string or null.")
        for field in ("difficulty", "discrimination", "guess_rate", "slip_rate"):
            if field in item and (
                type(item[field]) not in {int, float}
                or not isfinite(item[field])
            ):
                add(f"Field {field!r} must be a finite JSON number.")

        concepts = item.get("concepts")
        if type(concepts) is not list:
            add("Field 'concepts' must be a JSON array.")
        else:
            for index, mapping in enumerate(concepts):
                prefix = f"concepts[{index}]"
                if type(mapping) is not dict:
                    add(f"{prefix} must be a JSON object.")
                    continue
                if type(mapping.get("concept_id")) is not str:
                    add(f"{prefix}.concept_id must be a string.")
                weight = mapping.get("weight")
                if type(weight) not in {int, float} or not isfinite(weight):
                    add(f"{prefix}.weight must be a finite JSON number.")
                if "role" in mapping and type(mapping["role"]) is not str:
                    add(f"{prefix}.role must be a string.")

        options = item.get("options")
        if type(options) is not list:
            add("Field 'options' must be a JSON array.")
        else:
            for index, option in enumerate(options):
                prefix = f"options[{index}]"
                if type(option) is not dict:
                    add(f"{prefix} must be a JSON object.")
                    continue
                for field in ("id", "text", "rationale"):
                    if type(option.get(field)) is not str:
                        add(f"{prefix}.{field} must be a string.")
                    elif not option[field].strip():
                        add(f"{prefix}.{field} must not be blank.")
                if type(option.get("correct")) is not bool:
                    add(f"{prefix}.correct must be a JSON boolean.")
                misconception_id = option.get("misconception_id")
                if misconception_id is not None and type(misconception_id) is not str:
                    add(f"{prefix}.misconception_id must be a string or null.")
                diagnostic_objective_id = option.get("diagnostic_objective_id")
                if diagnostic_objective_id is not None and (
                    type(diagnostic_objective_id) is not str
                    or not diagnostic_objective_id.strip()
                ):
                    add(
                        f"{prefix}.diagnostic_objective_id must be a non-empty "
                        "string or null."
                    )

        source_ids = item.get("source_ids")
        if type(source_ids) is not list:
            add("Field 'source_ids' must be a JSON array.")
        elif any(type(source_id) is not str for source_id in source_ids):
            add("Every source_ids entry must be a string.")
        elif any(not source_id.strip() for source_id in source_ids):
            add("Every source_ids entry must be non-blank.")
        elif len(source_ids) != len(set(source_ids)):
            add("Field 'source_ids' must not contain duplicates.")

        provenance = item.get("provenance", {})
        if type(provenance) is not dict:
            add("Field 'provenance' must be a JSON object.")
        tags = item.get("tags", [])
        if type(tags) is not list or any(
            type(tag) is not str or not tag.strip() for tag in tags
        ):
            add(
                "Field 'tags' must be a JSON array of non-blank strings."
            )
        revision_of = item.get("revision_of")
        if revision_of is not None and (
            type(revision_of) is not str or not revision_of.strip()
        ):
            add("Field 'revision_of' must be a non-blank string or null.")
        elif revision_of is not None:
            add(
                "Coverage-generation jobs require revision_of to be null; "
                "immutable revisions need a separate reviewed workflow."
            )
        return issues

    def _collect_review(
        self,
        reviewer: IndependentItemReviewer,
        reviewer_provenance: dict[str, str],
        blinded_item: dict[str, Any],
        source_context: str,
        source_context_sha256: str,
    ) -> dict[str, Any]:
        provenance = copy.deepcopy(reviewer_provenance)
        blinded_copy = copy.deepcopy(blinded_item)
        blinded_hash = _sha256_json(blinded_copy)
        output = reviewer.review(blinded_copy, source_context)
        output_issues = _strict_json_issues(output, "$.review.")
        output_issues, issue_redaction_count = (
            _redact_exact_source_context(
                output_issues,
                source_context,
            )
        )
        if type(output) is not dict:
            output_issues.insert(0, "Reviewer output must be a JSON object.")

        if output_issues:
            persisted_output: dict[str, Any] = {
                "rejected_output_type": type(output).__name__,
                "validation_errors": output_issues,
            }
            valid = False
            output_redaction_count = issue_redaction_count
        else:
            persisted_output, output_redaction_count = (
                _redact_exact_source_context(
                    output,
                    source_context,
                )
            )
            verdict = persisted_output.get("verdict")
            valid = type(verdict) is str and verdict in {"accept", "reject", "revise"}
            if not valid:
                output_issues.append(
                    "Reviewer verdict must be exactly 'accept', 'reject', or 'revise'."
                )

        return {
            "reviewer": provenance,
            "reviewer_provenance_sha256": _sha256_json(provenance),
            "output": persisted_output,
            "reviewer_output_sha256": _sha256_json(persisted_output),
            "blinded_item_sha256": blinded_hash,
            "source_context_sha256": source_context_sha256,
            "exact_source_context_redaction_count": (
                output_redaction_count
            ),
            "valid": valid,
            "validation_errors": output_issues,
        }

    @staticmethod
    def _prior_artifact_collisions(
        connection: Any,
        item: dict[str, Any],
        *,
        exclude_job_id: str | None,
    ) -> tuple[bool, bool, tuple[str, ...]]:
        """Find identity reuse across immutable quarantined run artifacts."""

        question_collision = False
        family_collision = False
        invalid_run_ids: list[str] = []
        rows = connection.execute(
            """SELECT id, raw_output_json FROM generation_job_runs
               WHERE raw_output_json IS NOT NULL
                 AND (? IS NULL OR job_id != ?)
               ORDER BY id""",
            (exclude_job_id, exclude_job_id),
        ).fetchall()
        for row in rows:
            try:
                prior_item = _decode_json(
                    row["raw_output_json"],
                    label=f"Generation run {row['id']} raw output",
                    expected_type=dict,
                )
            except ValidationError:
                invalid_run_ids.append(row["id"])
                continue
            question_collision = question_collision or (
                prior_item.get("id") == item.get("id")
            )
            family_collision = family_collision or (
                prior_item.get("family_id") == item.get("family_id")
            )
        return question_collision, family_collision, tuple(invalid_run_ids)

    def _deterministic_validation(
        self,
        item: Any,
        blueprint: GenerationBlueprint,
        *,
        job_id: str | None = None,
    ) -> list[dict[str, str]]:
        issues = self._item_shape_issues(item)
        if issues:
            return issues

        def add(code: str, message: str) -> None:
            issues.append({"code": code, "severity": "error", "message": message})

        with self.database.read() as connection:
            release_id = blueprint.corpus_release_id
            if release_id is None:
                release_id = self.database.get_active_release_id(connection)
            release = connection.execute(
                "SELECT sealed_at FROM corpus_releases WHERE id = ?", (release_id,)
            ).fetchone()
            if release is None or release["sealed_at"] is None:
                add(
                    "unknown_blueprint_release",
                    f"Blueprint corpus release {release_id!r} is unavailable or unsealed.",
                )
                return issues
            objective_by_id = {
                row["id"]: self.database._objective_from_row(row)
                for row in connection.execute(
                    """SELECT objective.* FROM learning_objectives objective
                       JOIN release_learning_objectives membership
                         ON membership.objective_id = objective.id
                       WHERE membership.release_id = ?""",
                    (release_id,),
                )
            }
            objective_covered_concepts = {
                concept_id
                for objective in objective_by_id.values()
                for concept_id in objective.concept_ids
            }
            release_source_ids = {
                row["source_id"]
                for row in connection.execute(
                    "SELECT source_id FROM release_sources WHERE release_id = ?",
                    (release_id,),
                )
            }
            release_concept_ids = {
                row["concept_id"]
                for row in connection.execute(
                    "SELECT concept_id FROM release_concepts WHERE release_id = ?",
                    (release_id,),
                )
            }
            misconception_owners = {
                row["id"]: row["concept_id"]
                for row in connection.execute(
                    """SELECT misconception.id, misconception.concept_id
                       FROM misconceptions misconception
                       JOIN release_misconceptions membership
                         ON membership.misconception_id = misconception.id
                       WHERE membership.release_id = ?""",
                    (release_id,),
                )
            }
            family_collision = connection.execute(
                "SELECT 1 FROM questions WHERE family_id = ? LIMIT 1",
                (item["family_id"],),
            ).fetchone()
            question_id_collision = connection.execute(
                "SELECT 1 FROM questions WHERE id = ? LIMIT 1",
                (item["id"],),
            ).fetchone()
            (
                prior_question_id_collision,
                prior_family_collision,
                invalid_prior_run_ids,
            ) = self._prior_artifact_collisions(
                connection, item, exclude_job_id=job_id
            )
        for invalid_run_id in invalid_prior_run_ids:
            add(
                "invalid_prior_artifact",
                f"Generation run {invalid_run_id} has an unreadable immutable artifact.",
            )

        blueprint_objective = None
        if blueprint.learning_objective_id is not None:
            blueprint_objective = objective_by_id.get(
                blueprint.learning_objective_id
            )
            if blueprint_objective is None:
                add(
                    "unknown_blueprint_objective",
                    f"Learning objective {blueprint.learning_objective_id} does not "
                    f"belong to pinned release {release_id}.",
                )
            else:
                declared_definition = (
                    blueprint.learning_objective_name,
                    blueprint.learning_objective_description,
                    blueprint.learning_objective_operation,
                    blueprint.learning_objective_evidence_type,
                )
                release_definition = (
                    blueprint_objective.name,
                    blueprint_objective.description,
                    blueprint_objective.operation.value,
                    blueprint_objective.evidence_type,
                )
                if declared_definition != release_definition:
                    add(
                        "blueprint_objective_definition_mismatch",
                        "Blueprint learning-objective metadata does not match its "
                        "immutable pinned-release definition.",
                    )
                if blueprint.concept_id not in blueprint_objective.concept_ids:
                    add(
                        "blueprint_objective_concept_mismatch",
                        f"Concept {blueprint.concept_id} is outside learning objective "
                        f"{blueprint_objective.id}.",
                    )
        elif blueprint.concept_id in objective_covered_concepts:
            add(
                "missing_blueprint_objective",
                f"Concept {blueprint.concept_id} is objective-enabled in pinned "
                f"release {release_id}; a concept-only item cannot be promoted into "
                "that schema-v2 release.",
            )

        item_objective_id = item.get("learning_objective_id")
        item_objective = (
            objective_by_id.get(item_objective_id)
            if item_objective_id is not None
            else None
        )
        if item_objective_id != blueprint.learning_objective_id:
            add(
                "blueprint_objective_mismatch",
                "Generated item learning_objective_id must exactly match its blueprint.",
            )
        if item_objective_id is not None and item_objective is None:
            add(
                "unknown_learning_objective",
                f"Generated item objective {item_objective_id} does not belong to "
                f"pinned release {release_id}.",
            )

        try:
            question = Question(
                id=item["id"],
                version=item.get("version", 1),
                family_id=item["family_id"],
                status=QuestionStatus.QUARANTINED,
                stem=item["stem"],
                kind=QuestionKind(item.get("kind", blueprint.kind)),
                difficulty=item.get("difficulty", blueprint.target_difficulty),
                discrimination=item.get("discrimination", 1.0),
                guess_rate=item.get("guess_rate", 0.25),
                slip_rate=item.get("slip_rate", 0.05),
                concepts=tuple(
                    ConceptWeight(
                        concept_id=mapping["concept_id"],
                        weight=mapping["weight"],
                        role=ConceptRole(mapping.get("role", "secondary")),
                    )
                    for mapping in item["concepts"]
                ),
                options=tuple(
                    Option(
                        id=option["id"],
                        text=option["text"],
                        correct=option["correct"],
                        rationale=option["rationale"],
                        misconception_id=option.get("misconception_id"),
                        diagnostic_objective_id=option.get(
                            "diagnostic_objective_id"
                        ),
                    )
                    for option in item["options"]
                ),
                source_ids=tuple(item["source_ids"]),
                provenance=copy.deepcopy(item.get("provenance", {})),
                tags=tuple(item.get("tags", ())),
                revision_of=item.get("revision_of"),
                objective=item_objective,
            )
        except (KeyError, TypeError, ValueError) as exc:
            add("invalid_item_shape", f"Generated item cannot be parsed: {exc}")
            return issues

        for issue in validate_question(question):
            issues.append(
                {
                    "code": issue.code,
                    "severity": issue.severity,
                    "message": issue.message,
                }
            )
        if question.concepts and question.primary_concept_id != blueprint.concept_id:
            add(
                "blueprint_concept_mismatch",
                f"Primary concept must be {blueprint.concept_id}, not {question.primary_concept_id}.",
            )
        if question.kind.value != blueprint.kind:
            add(
                "blueprint_kind_mismatch",
                f"Question kind must be {blueprint.kind}, not {question.kind.value}.",
            )
        if set(question.source_ids) - set(blueprint.source_ids):
            add("unapproved_source", "Generated item cites a source outside its blueprint.")
        if family_collision:
            add("family_collision", "Generated item reuses an existing item family.")
        if question_id_collision:
            add("question_id_collision", "Generated item reuses an existing question ID.")
        if prior_family_collision:
            add(
                "quarantine_family_collision",
                "Generated item reuses a family already present in another immutable "
                "generation-job artifact.",
            )
        if prior_question_id_collision:
            add(
                "quarantine_question_id_collision",
                "Generated item reuses a question ID already present in another "
                "immutable generation-job artifact.",
            )
        unknown_sources = set(question.source_ids) - release_source_ids
        if unknown_sources:
            add(
                "unknown_source",
                "Generated item cites source IDs outside its pinned release: "
                + ", ".join(sorted(unknown_sources))
                + ".",
            )
        unknown_concepts = {
            mapping.concept_id for mapping in question.concepts
        } - release_concept_ids
        if unknown_concepts:
            add(
                "unknown_concept",
                "Generated item maps concepts outside its pinned release: "
                + ", ".join(sorted(unknown_concepts))
                + ".",
            )

        misconception_ids = question.misconception_ids
        unknown_misconceptions = misconception_ids - set(misconception_owners)
        if unknown_misconceptions:
            add(
                "unknown_misconception",
                "Unknown pinned-release misconception IDs: "
                + ", ".join(sorted(unknown_misconceptions))
                + ".",
            )
        unexpected_misconceptions = misconception_ids - set(
            blueprint.misconception_ids
        )
        if unexpected_misconceptions:
            add(
                "unplanned_misconception",
                "Generated distractors use misconceptions outside the blueprint: "
                + ", ".join(sorted(unexpected_misconceptions))
                + ".",
            )
        mapped_concepts = {mapping.concept_id for mapping in question.concepts}
        absent_owners = {
            owner
            for misconception_id, owner in misconception_owners.items()
            if misconception_id in misconception_ids and owner not in mapped_concepts
        }
        if absent_owners:
            add(
                "unmapped_misconception_owner",
                "Every distractor misconception's concept must be mapped by the item.",
            )

        exact_diagnostic_misconceptions: set[str] = set()
        for option in question.options:
            diagnostic_id = option.diagnostic_objective_id
            if option.correct:
                if diagnostic_id is not None:
                    add(
                        "correct_option_diagnostic_objective",
                        f"Correct option {option.id} cannot declare a diagnostic objective.",
                    )
                continue
            if blueprint_objective is not None and diagnostic_id is None:
                add(
                    "missing_diagnostic_objective",
                    f"Distractor {option.id} must explicitly preserve diagnostic_objective_id.",
                )
                continue
            diagnostic_objective = (
                objective_by_id.get(diagnostic_id)
                if diagnostic_id is not None
                else None
            )
            if diagnostic_id is not None and diagnostic_objective is None:
                add(
                    "unknown_diagnostic_objective",
                    f"Distractor {option.id} diagnostic objective {diagnostic_id} "
                    f"does not belong to pinned release {release_id}.",
                )
                continue
            if blueprint_objective is None and diagnostic_id is not None:
                add(
                    "unexpected_diagnostic_objective",
                    f"Distractor {option.id} declares a diagnostic objective that "
                    "was not authorized by its legacy blueprint.",
                )
            if (
                blueprint_objective is not None
                and diagnostic_id != blueprint_objective.id
            ):
                add(
                    "blueprint_diagnostic_objective_mismatch",
                    f"Distractor {option.id} must diagnose blueprint objective "
                    f"{blueprint_objective.id}.",
                )
            owner_id = misconception_owners.get(option.misconception_id)
            if (
                owner_id is not None
                and diagnostic_objective is not None
                and owner_id not in diagnostic_objective.concept_ids
            ):
                add(
                    "diagnostic_objective_owner_mismatch",
                    f"Distractor {option.id} misconception owner {owner_id} is outside "
                    f"diagnostic objective {diagnostic_objective.id}.",
                )
            if (
                option.misconception_id is not None
                and diagnostic_id == blueprint.learning_objective_id
            ):
                exact_diagnostic_misconceptions.add(option.misconception_id)
        if (
            blueprint.target_misconception_id is not None
            and blueprint.target_misconception_id
            not in exact_diagnostic_misconceptions
        ):
            add(
                "missing_exact_diagnostic_target",
                "Generated item does not expose the blueprint's exact "
                "objective/misconception target.",
            )
        if blueprint.coverage_goal == "objective_misconception_serviceability":
            missing_targets = set(blueprint.misconception_ids) - (
                exact_diagnostic_misconceptions
            )
            if missing_targets:
                add(
                    "missing_exact_diagnostic_targets",
                    "Generated item omits planned exact objective/misconception "
                    "targets: "
                    + ", ".join(sorted(missing_targets))
                    + ".",
                )
        return issues

    def run_job(self, job_id: str, source_context: str) -> dict[str, Any]:
        if type(job_id) is not str or not job_id.strip():
            raise ValidationError("Generation job ID must be a non-empty string.")
        if type(source_context) is not str:
            raise TypeError("source_context must be a string; implicit coercion is forbidden.")
        if not source_context.strip():
            raise ValidationError("Source context must contain approved source material.")
        source_context_sha256 = _sha256_text(source_context)
        started_at = datetime.now(timezone.utc).isoformat()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if not row:
                raise NotFoundError(f"Unknown generation job: {job_id}")
            if row["status"] != "planned":
                if row["status"] in {"failed", "rejected"}:
                    guidance = (
                        "use `jobs retry` before another execution"
                    )
                elif row["status"] == "running":
                    guidance = (
                        "a worker may still own it; use `jobs retry "
                        "--recover-running` only after confirming that worker "
                        "has stopped"
                    )
                else:
                    guidance = (
                        "reviewed jobs are terminal; author a new coverage job "
                        "for any revision"
                    )
                raise ConflictError(
                    f"Generation job {job_id} is {row['status']}; {guidance}."
                )
            if row["prompt_version"] not in SUPPORTED_PROMPT_VERSIONS:
                raise ConflictError(
                    f"Generation job {job_id} uses unsupported prompt version "
                    f"{row['prompt_version']!r}."
                )
            blueprint = _parse_blueprint(
                row["blueprint_json"], label=f"Generation job {job_id} blueprint"
            )
            if (
                row["prompt_version"] == PROMPT_VERSION
                and blueprint.corpus_release_id is None
            ):
                raise ValidationError(
                    f"Generation job {job_id} uses {PROMPT_VERSION} but is not "
                    "pinned to an immutable corpus release."
                )
            next_attempt = connection.execute(
                """SELECT COALESCE(MAX(attempt), 0) + 1 AS attempt
                   FROM generation_job_runs WHERE job_id = ?""",
                (job_id,),
            ).fetchone()["attempt"]
            run_id = new_id("run")
            claimed = connection.execute(
                """UPDATE generation_jobs
                   SET status='running', provider=?, model=?, raw_output_json=NULL,
                       validation_json=NULL, updated_at=?
                   WHERE id=? AND status='planned'""",
                (
                    self.generator_provider_name,
                    self.generator_model_name,
                    started_at,
                    job_id,
                ),
            )
            if claimed.rowcount != 1:
                raise ConflictError(f"Generation job {job_id} was claimed by another worker.")
            connection.execute(
                """INSERT INTO generation_job_runs(
                       id, job_id, attempt, status, provider, model, prompt_version,
                       source_context_sha256, started_at
                   ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    job_id,
                    next_attempt,
                    self.generator_provider_name,
                    self.generator_model_name,
                    row["prompt_version"],
                    source_context_sha256,
                    started_at,
                ),
            )
        try:
            raw_output = self.generator.generate(blueprint, source_context)
            raw_json_issues = _strict_json_issues(raw_output)
            if not raw_json_issues:
                generator_output_sha256 = _sha256_json(raw_output)
            else:
                generator_output_sha256 = _sha256_json(
                    {
                        "rejected_output_type": type(raw_output).__name__,
                        "validation_errors": raw_json_issues,
                    }
                )

            # The raw object is never parsed by truthiness or scalar coercion.
            # A malformed output is represented only by an inert quarantine
            # record, while its deterministic errors make acceptance impossible.
            if type(raw_output) is dict and not raw_json_issues:
                item, generator_output_redaction_count = (
                    _redact_exact_source_context(
                        raw_output,
                        source_context,
                    )
                )
            else:
                item = {"generator_output_rejected": True}
                generator_output_redaction_count = 0
            item["status"] = "quarantined"
            declared_provenance = item.get("provenance")
            if type(declared_provenance) is not dict:
                declared_provenance = {}
            raw_declared_provenance = (
                raw_output.get("provenance", {})
                if type(raw_output) is dict
                and not raw_json_issues
                and type(raw_output.get("provenance", {})) is dict
                else {}
            )
            stripped_authority_fields = sorted(
                set(raw_declared_provenance)
                & GENERATOR_AUTHORITY_PROVENANCE_FIELDS
            )
            declared_provenance = {
                key: copy.deepcopy(value)
                for key, value in declared_provenance.items()
                if key not in GENERATOR_AUTHORITY_PROVENANCE_FIELDS
            }
            generator_provenance = {
                "provider_name": self.generator_provider_name,
                "model_name": self.generator_model_name,
                "prompt_version": row["prompt_version"],
                "generation_job_id": job_id,
                "generation_run_id": run_id,
                "attempt": next_attempt,
            }
            generator_provenance_sha256 = _sha256_json(generator_provenance)
            item["provenance"] = copy.deepcopy(declared_provenance)
            item["provenance"].update(
                {
                    "generated": True,
                    "human_review": False,
                    "human_review_status": "required_before_activation",
                    "activation": (
                        "manual_only_after_human_review_and_new_immutable_release"
                    ),
                    "provider": self.generator_provider_name,
                    "model": self.generator_model_name,
                    "prompt_version": row["prompt_version"],
                    "generation_job_id": job_id,
                    "generation_run_id": run_id,
                    "generation_attempt": next_attempt,
                    "source_context_sha256": source_context_sha256,
                    "generator_output_sha256": generator_output_sha256,
                    "generator_provenance_sha256": generator_provenance_sha256,
                    "generator_declared_provenance_sha256": (
                        _sha256_json(raw_declared_provenance)
                    ),
                    "stripped_generator_authority_fields": (
                        stripped_authority_fields
                    ),
                }
            )
            if type(raw_output) is not dict or raw_json_issues:
                deterministic_issues = self._item_shape_issues(
                    raw_output
                )
            else:
                deterministic_issues = self._deterministic_validation(
                    item,
                    blueprint,
                    job_id=job_id,
                )
            if generator_output_redaction_count:
                deterministic_issues.append(
                    {
                        "code": "source_context_echo_in_generator_output",
                        "severity": "error",
                        "message": (
                            "Generator output contained an exact source-context "
                            "copy; the persisted artifact was redacted and must "
                            "not enter reviewed quarantine."
                        ),
                    }
                )
            deterministic_issues, deterministic_redaction_count = (
                _redact_exact_source_context(
                    deterministic_issues,
                    source_context,
                )
            )
            for provenance_issue in question_provenance_issues(
                item["provenance"],
                status="quarantined",
            ):
                deterministic_issues.append(
                    {
                        "code": (
                            "finalized_provenance_"
                            f"{provenance_issue.code}"
                        ),
                        "severity": "error",
                        "message": provenance_issue.message,
                    }
                )

            blinded_item = _blind_for_review(item, blueprint)
            reviews = [
                self._collect_review(
                    reviewer,
                    reviewer_provenance,
                    blinded_item,
                    source_context,
                    source_context_sha256,
                )
                for reviewer, reviewer_provenance in zip(
                    self.reviewers, self._reviewer_provenances, strict=True
                )
            ]
            accepted_by_critics = all(
                review["valid"] and review["output"].get("verdict") == "accept"
                for review in reviews
            )
            item["provenance"]["authoring_review_state"] = (
                "independent_model_reviews_accepted"
                if accepted_by_critics
                else "independent_model_reviews_not_accepted"
            )
            item["provenance"]["independent_model_review_count"] = len(
                reviews
            )
            deterministic_errors = [
                issue for issue in deterministic_issues if issue["severity"] == "error"
            ]
            accepted = accepted_by_critics and not deterministic_errors
            reviews_sha256 = _sha256_json(reviews)
            result = {
                "job_id": job_id,
                "run_id": run_id,
                "attempt": next_attempt,
                "item": item,
                "deterministic_issues": deterministic_issues,
                "reviews": reviews,
                "source_context_sha256": source_context_sha256,
                "generator_output_sha256": generator_output_sha256,
                "generator_provenance": generator_provenance,
                "generator_provenance_sha256": generator_provenance_sha256,
                "reviews_sha256": reviews_sha256,
                "exact_source_context_redactions": {
                    "generator_output": (
                        generator_output_redaction_count
                    ),
                    "deterministic_validation": (
                        deterministic_redaction_count
                    ),
                    "reviewer_outputs": sum(
                        review[
                            "exact_source_context_redaction_count"
                        ]
                        for review in reviews
                    ),
                },
                "accepted_by_critics": accepted_by_critics,
                "accepted_for_reviewed_quarantine": accepted,
            }
            status = "reviewed" if accepted else "rejected"
            result["status"] = status
            validation_record = {
                "deterministic_issues": deterministic_issues,
                "source_context_sha256": source_context_sha256,
                "generator_output_sha256": generator_output_sha256,
                "generator_provenance": generator_provenance,
                "generator_provenance_sha256": generator_provenance_sha256,
                "reviews": reviews,
                "reviews_sha256": reviews_sha256,
                "exact_source_context_redactions": result[
                    "exact_source_context_redactions"
                ],
            }
            with self.database.transaction() as connection:
                # Generation and review run outside the write lock. Recheck
                # identities here, under BEGIN IMMEDIATE, so two workers cannot
                # concurrently certify the same question or family as independent.
                final_question_collision = connection.execute(
                    "SELECT 1 FROM questions WHERE id = ? LIMIT 1",
                    (item.get("id"),),
                ).fetchone()
                final_family_collision = connection.execute(
                    "SELECT 1 FROM questions WHERE family_id = ? LIMIT 1",
                    (item.get("family_id"),),
                ).fetchone()
                (
                    final_prior_question_collision,
                    final_prior_family_collision,
                    final_invalid_prior_runs,
                ) = self._prior_artifact_collisions(
                    connection, item, exclude_job_id=job_id
                )
                existing_issue_codes = {
                    issue.get("code")
                    for issue in deterministic_issues
                    if type(issue) is dict
                }

                def final_issue(code: str, message: str) -> None:
                    if code not in existing_issue_codes:
                        deterministic_issues.append(
                            {"code": code, "severity": "error", "message": message}
                        )
                        existing_issue_codes.add(code)

                if final_question_collision:
                    final_issue(
                        "question_id_collision",
                        "Generated item reuses an existing question ID.",
                    )
                if final_family_collision:
                    final_issue(
                        "family_collision",
                        "Generated item reuses an existing item family.",
                    )
                if final_prior_question_collision:
                    final_issue(
                        "quarantine_question_id_collision",
                        "Generated item reuses a question ID already present in "
                        "another immutable generation-job artifact.",
                    )
                if final_prior_family_collision:
                    final_issue(
                        "quarantine_family_collision",
                        "Generated item reuses a family already present in another "
                        "immutable generation-job artifact.",
                    )
                if final_invalid_prior_runs:
                    final_issue(
                        "invalid_prior_artifact",
                        "An immutable prior generation artifact is unreadable; "
                        "identity uniqueness cannot be certified.",
                    )
                accepted = accepted_by_critics and not any(
                    issue.get("severity") == "error"
                    for issue in deterministic_issues
                    if type(issue) is dict
                )
                status = "reviewed" if accepted else "rejected"
                result["accepted_for_reviewed_quarantine"] = accepted
                result["status"] = status
                completed_at = datetime.now(timezone.utc).isoformat()
                finalized_run = connection.execute(
                    """UPDATE generation_job_runs
                       SET status=?, raw_output_json=?, validation_json=?, completed_at=?
                       WHERE id=? AND job_id=? AND status='running'""",
                    (
                        status,
                        json.dumps(item, sort_keys=True, allow_nan=False),
                        json.dumps(validation_record, sort_keys=True, allow_nan=False),
                        completed_at,
                        run_id,
                        job_id,
                    ),
                )
                if finalized_run.rowcount != 1:
                    raise ConflictError(
                        f"Generation run {run_id} changed before finalization."
                    )
                finalized_job = connection.execute(
                    """UPDATE generation_jobs SET status=?, raw_output_json=?,
                           validation_json=?, updated_at=?
                       WHERE id=? AND status='running'""",
                    (
                        status,
                        json.dumps(item, sort_keys=True, allow_nan=False),
                        json.dumps(validation_record, sort_keys=True, allow_nan=False),
                        completed_at,
                        job_id,
                    ),
                )
                if finalized_job.rowcount != 1:
                    raise ConflictError(
                        f"Generation job {job_id} changed before finalization."
                    )
            return result
        except BaseException as exc:
            completed_at = datetime.now(timezone.utc).isoformat()
            error_record = {
                "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "error": (
                    "Generation provider or reviewer failed; raw exception "
                    "text and source context were not persisted."
                ),
            }
            error_json = _canonical_json(error_record)
            try:
                with self.database.transaction() as connection:
                    run = connection.execute(
                        "SELECT status FROM generation_job_runs WHERE id = ? AND job_id = ?",
                        (run_id, job_id),
                    ).fetchone()
                    if run is not None and run["status"] == "running":
                        connection.execute(
                            """UPDATE generation_job_runs
                               SET status='failed', error_json=?, completed_at=?
                               WHERE id=? AND status='running'""",
                            (error_json, completed_at, run_id),
                        )
                        failed_job = connection.execute(
                            """UPDATE generation_jobs
                               SET status='failed', raw_output_json=NULL,
                                   validation_json=?, updated_at=?
                               WHERE id=? AND status='running'""",
                            (error_json, completed_at, job_id),
                        )
                        if failed_job.rowcount != 1:
                            raise ConflictError(
                                f"Generation job {job_id} changed while recording failure."
                            )
            except Exception as persistence_error:
                exc.add_note(
                    "TSQ could not persist the generation-run failure: "
                    f"{persistence_error}"
                )
            raise
