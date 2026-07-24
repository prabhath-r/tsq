# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
from collections import Counter, deque
from math import isfinite
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .graph import KnowledgeGraph
from .models import (
    Concept,
    ConceptEdge,
    ConceptRole,
    ConceptWeight,
    Domain,
    LearningObjective,
    Misconception,
    ObjectiveEdge,
    ObjectiveOperation,
    Option,
    Question,
    QuestionKind,
    QuestionStatus,
    QualityIssue,
    RelationType,
    Source,
    Topic,
)
from .quality import audit_corpus
from .provenance import (
    legacy_question_identity_payload,
    legacy_unattested_member_compatible,
    question_provenance_issues,
)


def _relation(value: str) -> RelationType:
    aliases = {"requires": "requires", "prerequisite": "prerequisite", "contrasts": "contrasts_with"}
    return RelationType(aliases.get(value, value))


_LIST_FIELDS = ("concepts", "edges", "misconceptions", "sources", "questions")
_CATALOG_LIST_FIELDS = ("domains", "topics")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_bundle(bundle: object) -> list[QualityIssue]:
    """Validate raw JSON types and scalar domains without coercing input values.

    The parser runs this complete pass before constructing domain objects. This is
    intentionally separate from semantic/reference checks so a producer receives
    every inexpensive schema defect in one response.
    """

    issues: list[QualityIssue] = []

    def add(code: str, message: str, path: str, question_id: str | None = None) -> None:
        issues.append(QualityIssue(code, "error", message, question_id, path))

    if not isinstance(bundle, dict):
        add("bundle_type", "Corpus root must be a JSON object.", "$")
        return issues

    schema_version = bundle.get("schema_version")
    if "schema_version" not in bundle:
        add("missing_field", "Required field 'schema_version' is missing.", "schema_version")
    elif not isinstance(schema_version, int) or isinstance(schema_version, bool):
        add(
            "schema_version_type",
            "schema_version must be the integer 1, 2, or 3.",
            "schema_version",
        )
    elif schema_version not in {1, 2, 3}:
        add(
            "schema_version",
            "Only corpus schema_version 1, 2, and 3 are supported.",
            "schema_version",
        )
    if "title" not in bundle:
        add("missing_field", "Required field 'title' is missing.", "title")
    elif not isinstance(bundle["title"], str):
        add("field_type", "title must be a string.", "title")
    elif not bundle["title"].strip():
        add("blank_field", "title cannot be blank.", "title")

    for field in _LIST_FIELDS:
        if field not in bundle:
            add("missing_field", f"Required field '{field}' is missing.", field)
        elif not isinstance(bundle[field], list):
            add("field_type", f"Field '{field}' must be a list.", field)

    if schema_version in {2, 3}:
        if "learning_objectives" not in bundle:
            add(
                "missing_field",
                f"Corpus schema_version {schema_version} requires 'learning_objectives'.",
                "learning_objectives",
            )
        elif not isinstance(bundle["learning_objectives"], list):
            add(
                "field_type",
                "Field 'learning_objectives' must be a list.",
                "learning_objectives",
            )
        elif schema_version == 3 and not bundle["learning_objectives"]:
            add(
                "empty_objective_graph",
                "Corpus schema_version 3 requires at least one learning objective.",
                "learning_objectives",
            )
    elif "learning_objectives" in bundle:
        add(
            "schema_version",
            "learning_objectives require corpus schema_version 2 or 3.",
            "learning_objectives",
        )

    if schema_version == 3:
        if "objective_edges" not in bundle:
            add(
                "missing_field",
                "Corpus schema_version 3 requires 'objective_edges'.",
                "objective_edges",
            )
        elif not isinstance(bundle["objective_edges"], list):
            add(
                "field_type",
                "Field 'objective_edges' must be a list.",
                "objective_edges",
            )
    elif "objective_edges" in bundle:
        add(
            "schema_version",
            "objective_edges require corpus schema_version 3.",
            "objective_edges",
        )

    catalog_present = any(field in bundle for field in _CATALOG_LIST_FIELDS)
    if catalog_present:
        for field in _CATALOG_LIST_FIELDS:
            if field not in bundle:
                add(
                    "missing_field",
                    f"Catalog field '{field}' is required when a curriculum catalog is present.",
                    field,
                )
            elif not isinstance(bundle[field], list):
                add("field_type", f"Field '{field}' must be a list.", field)

    def rows(field: str):
        value = bundle.get(field)
        return value if isinstance(value, list) else []

    def row_object(value: object, path: str) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            add("row_type", "List entry must be an object.", path)
            return None
        return value

    def raw_legacy_identity(row: dict[str, Any]) -> dict[str, object]:
        objective_id = row.get("learning_objective_id")
        return legacy_question_identity_payload(
            question_id=row["id"],
            version=row.get("version", 1),
            family_id=row["family_id"],
            stem=row["stem"],
            kind=row["kind"],
            difficulty=row["difficulty"],
            discrimination=row["discrimination"],
            guess_rate=row.get("guess_rate", 0.25),
            slip_rate=row.get("slip_rate", 0.05),
            concepts=(
                (
                    mapping["concept_id"],
                    mapping["weight"],
                    mapping.get("role", ConceptRole.SECONDARY.value),
                )
                for mapping in row["concepts"]
            ),
            options=(
                (
                    option["id"],
                    option["text"],
                    option["correct"],
                    option["rationale"],
                    option.get("misconception_id"),
                    option.get("diagnostic_objective_id")
                    or (
                        objective_id
                        if option.get("correct") is False
                        else None
                    ),
                )
                for option in row["options"]
            ),
            source_ids=row["source_ids"],
            provenance=row.get("provenance", {}),
            tags=row.get("tags", []),
            revision_of=row.get("revision_of"),
            learning_objective_id=objective_id,
        )

    legacy_unattested_compatible_ids: set[str] = set()
    for legacy_row in rows("questions"):
        if (
            not isinstance(legacy_row, dict)
            or not isinstance(legacy_row.get("provenance", {}), dict)
            or "generated" in legacy_row.get("provenance", {})
        ):
            continue
        try:
            compatible = legacy_unattested_member_compatible(
                raw_legacy_identity(legacy_row)
            )
        except (KeyError, TypeError, ValueError):
            compatible = False
        if compatible and isinstance(legacy_row.get("id"), str):
            legacy_unattested_compatible_ids.add(legacy_row["id"])

    def string_field(
        row: dict[str, Any],
        name: str,
        path: str,
        *,
        question_id: str | None = None,
        required: bool = True,
        nullable: bool = False,
    ) -> str | None:
        field_path = f"{path}.{name}"
        if name not in row:
            if required:
                add("missing_field", f"Required field '{name}' is missing.", field_path, question_id)
            return None
        value = row[name]
        if value is None and nullable:
            return None
        if not isinstance(value, str):
            add("field_type", f"Field '{name}' must be a string.", field_path, question_id)
            return None
        if not value.strip():
            add("blank_field", f"Field '{name}' cannot be blank.", field_path, question_id)
        return value

    def list_field(
        row: dict[str, Any], name: str, path: str, *, question_id: str | None = None
    ) -> list[Any]:
        field_path = f"{path}.{name}"
        if name not in row:
            add("missing_field", f"Required field '{name}' is missing.", field_path, question_id)
            return []
        value = row[name]
        if not isinstance(value, list):
            add("field_type", f"Field '{name}' must be a list.", field_path, question_id)
            return []
        return value

    def number_field(
        row: dict[str, Any],
        name: str,
        path: str,
        *,
        question_id: str | None = None,
        required: bool = True,
        default: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        exclusive_minimum: bool = False,
        exclusive_maximum: bool = False,
    ) -> float | None:
        field_path = f"{path}.{name}"
        if name not in row:
            if required:
                add("missing_field", f"Required field '{name}' is missing.", field_path, question_id)
                return None
            return default
        value = row[name]
        if not _is_number(value):
            add("field_type", f"Field '{name}' must be a number.", field_path, question_id)
            return None
        try:
            numeric = float(value)
        except (OverflowError, ValueError):
            add("non_finite", f"Field '{name}' must be finite.", field_path, question_id)
            return None
        if not isfinite(numeric):
            add("non_finite", f"Field '{name}' must be finite.", field_path, question_id)
            return None
        below = minimum is not None and (
            numeric <= minimum if exclusive_minimum else numeric < minimum
        )
        above = maximum is not None and (
            numeric >= maximum if exclusive_maximum else numeric > maximum
        )
        if below or above:
            if minimum is None:
                comparator = "less than" if exclusive_maximum else "at most"
                constraint = f"{comparator} {maximum}"
            elif maximum is None:
                comparator = "greater than" if exclusive_minimum else "at least"
                constraint = f"{comparator} {minimum}"
            else:
                left = "(" if exclusive_minimum else "["
                right = ")" if exclusive_maximum else "]"
                constraint = f"in {left}{minimum}, {maximum}{right}"
            add(
                "out_of_range",
                f"Field '{name}' must be {constraint}.",
                field_path,
                question_id,
            )
        return numeric

    for index, value in enumerate(rows("concepts")):
        path = f"concepts[{index}]"
        row = row_object(value, path)
        if row is None:
            continue
        string_field(row, "id", path)
        string_field(row, "name", path)
        string_field(row, "description", path)
        string_field(row, "domain", path, required=False)
        number_field(
            row,
            "prior_mastery",
            path,
            required=False,
            default=0.20,
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
            exclusive_maximum=True,
        )

    for index, value in enumerate(rows("learning_objectives")):
        path = f"learning_objectives[{index}]"
        row = row_object(value, path)
        if row is None:
            continue
        string_field(row, "id", path)
        string_field(row, "name", path)
        string_field(row, "description", path)
        operation = string_field(row, "operation", path)
        if operation is not None:
            try:
                ObjectiveOperation(operation)
            except ValueError:
                allowed = ", ".join(item.value for item in ObjectiveOperation)
                add(
                    "invalid_enum",
                    f"Unknown objective operation '{operation}'; allowed values are {allowed}.",
                    f"{path}.operation",
                )
        evidence_type = row.get("evidence_type", "selected_response")
        if not isinstance(evidence_type, str):
            add(
                "field_type",
                "Field 'evidence_type' must be a string.",
                f"{path}.evidence_type",
            )
        elif evidence_type != "selected_response":
            add(
                "unsupported_evidence_type",
                "Corpus schema_version 2 and 3 support only selected_response objectives.",
                f"{path}.evidence_type",
            )
        string_field(row, "primary_concept_id", path)
        concept_ids = row.get("supporting_concept_ids", [])
        if not isinstance(concept_ids, list):
            add(
                "field_type",
                "Field 'supporting_concept_ids' must be a list.",
                f"{path}.supporting_concept_ids",
            )
            concept_ids = []
        for concept_index, concept_id in enumerate(concept_ids):
            if not isinstance(concept_id, str) or not concept_id.strip():
                add(
                    "field_type",
                    "Every supporting_concept_ids entry must be a non-blank string.",
                    f"{path}.supporting_concept_ids[{concept_index}]",
                )
        number_field(
            row,
            "prior_mastery",
            path,
            required=False,
            default=0.20,
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
            exclusive_maximum=True,
        )

    for index, value in enumerate(rows("objective_edges")):
        path = f"objective_edges[{index}]"
        row = row_object(value, path)
        if row is None:
            continue
        string_field(row, "id", path)
        source = string_field(row, "source", path)
        target = string_field(row, "target", path)
        relation = string_field(row, "relation", path)
        string_field(row, "rationale", path)
        if source is not None and target is not None and source == target:
            add(
                "self_objective_edge",
                "A learning objective cannot require itself.",
                path,
            )
        if relation is not None:
            try:
                parsed_relation = _relation(relation)
                if not parsed_relation.is_strict_prerequisite:
                    raise ValueError
            except ValueError:
                add(
                    "invalid_enum",
                    "Objective edges support only prerequisite or requires relations.",
                    f"{path}.relation",
                )
        number_field(
            row,
            "weight",
            path,
            minimum=0.0,
            maximum=1.0,
            exclusive_minimum=True,
        )

    for index, value in enumerate(rows("domains")):
        path = f"domains[{index}]"
        row = row_object(value, path)
        if row is None:
            continue
        string_field(row, "id", path)
        string_field(row, "name", path)
        string_field(row, "description", path)
        sort_order = row.get("sort_order", 0)
        if not isinstance(sort_order, int) or isinstance(sort_order, bool):
            add("field_type", "Field 'sort_order' must be an integer.", f"{path}.sort_order")
        elif sort_order < 0:
            add("out_of_range", "Field 'sort_order' must be non-negative.", f"{path}.sort_order")

    for index, value in enumerate(rows("topics")):
        path = f"topics[{index}]"
        row = row_object(value, path)
        if row is None:
            continue
        string_field(row, "id", path)
        string_field(row, "domain_id", path)
        string_field(row, "name", path)
        string_field(row, "description", path)
        string_field(row, "parent_id", path, required=False, nullable=True)
        concept_ids = list_field(row, "concept_ids", path)
        for concept_index, concept_id in enumerate(concept_ids):
            if not isinstance(concept_id, str) or not concept_id.strip():
                add(
                    "field_type",
                    "Every concept_ids entry must be a non-blank string.",
                    f"{path}.concept_ids[{concept_index}]",
                )
        related = row.get("related_topic_ids", [])
        if not isinstance(related, list):
            add("field_type", "Field 'related_topic_ids' must be a list.", f"{path}.related_topic_ids")
        else:
            for related_index, topic_id in enumerate(related):
                if not isinstance(topic_id, str) or not topic_id.strip():
                    add(
                        "field_type",
                        "Every related_topic_ids entry must be a non-blank string.",
                        f"{path}.related_topic_ids[{related_index}]",
                    )
        sort_order = row.get("sort_order", 0)
        if not isinstance(sort_order, int) or isinstance(sort_order, bool):
            add("field_type", "Field 'sort_order' must be an integer.", f"{path}.sort_order")
        elif sort_order < 0:
            add("out_of_range", "Field 'sort_order' must be non-negative.", f"{path}.sort_order")

    for index, value in enumerate(rows("edges")):
        path = f"edges[{index}]"
        row = row_object(value, path)
        if row is None:
            continue
        string_field(row, "source", path)
        string_field(row, "target", path)
        relation = string_field(row, "relation", path)
        if relation is not None:
            try:
                _relation(relation)
            except ValueError:
                add("invalid_enum", f"Unknown concept relation '{relation}'.", f"{path}.relation")
        number_field(
            row,
            "weight",
            path,
            required=False,
            default=1.0,
            minimum=0.0,
            exclusive_minimum=True,
        )

    for index, value in enumerate(rows("misconceptions")):
        path = f"misconceptions[{index}]"
        row = row_object(value, path)
        if row is None:
            continue
        string_field(row, "id", path)
        string_field(row, "concept_id", path)
        string_field(row, "name", path)
        string_field(row, "description", path)

    for index, value in enumerate(rows("sources")):
        path = f"sources[{index}]"
        row = row_object(value, path)
        if row is None:
            continue
        string_field(row, "id", path)
        string_field(row, "title", path)
        string_field(row, "uri", path, required=False, nullable=True)
        string_field(row, "license", path, required=False, nullable=True)

    for question_index, value in enumerate(rows("questions")):
        path = f"questions[{question_index}]"
        row = row_object(value, path)
        if row is None:
            continue
        raw_id = row.get("id")
        question_id = raw_id if isinstance(raw_id, str) and raw_id.strip() else None
        string_field(row, "id", path, question_id=question_id)
        string_field(row, "family_id", path, question_id=question_id)
        string_field(row, "stem", path, question_id=question_id)
        string_field(
            row,
            "learning_objective_id",
            path,
            question_id=question_id,
            required=False,
            nullable=True,
        )
        status = string_field(row, "status", path, question_id=question_id)
        if status is not None:
            try:
                QuestionStatus(status)
            except ValueError:
                add("invalid_enum", f"Unknown question status '{status}'.", f"{path}.status", question_id)
        kind = string_field(row, "kind", path, question_id=question_id)
        if kind is not None:
            try:
                QuestionKind(kind)
            except ValueError:
                add("invalid_enum", f"Unknown question kind '{kind}'.", f"{path}.kind", question_id)
        version = row.get("version", 1)
        if not isinstance(version, int) or isinstance(version, bool):
            add("field_type", "Field 'version' must be an integer.", f"{path}.version", question_id)
        elif version < 1:
            add("out_of_range", "Field 'version' must be at least 1.", f"{path}.version", question_id)
        number_field(
            row, "difficulty", path, question_id=question_id, minimum=-4.0, maximum=4.0
        )
        number_field(
            row,
            "discrimination",
            path,
            question_id=question_id,
            minimum=0.25,
            maximum=3.0,
        )
        guess_rate = number_field(
            row,
            "guess_rate",
            path,
            question_id=question_id,
            required=False,
            default=0.25,
            minimum=0.0,
            maximum=0.35,
        )
        number_field(
            row,
            "slip_rate",
            path,
            question_id=question_id,
            required=False,
            default=0.05,
            minimum=0.0,
            maximum=0.25,
        )

        mappings = list_field(row, "concepts", path, question_id=question_id)
        for mapping_index, mapping_value in enumerate(mappings):
            mapping_path = f"{path}.concepts[{mapping_index}]"
            mapping = row_object(mapping_value, mapping_path)
            if mapping is None:
                continue
            string_field(mapping, "concept_id", mapping_path, question_id=question_id)
            number_field(
                mapping,
                "weight",
                mapping_path,
                question_id=question_id,
                minimum=0.0,
                exclusive_minimum=True,
            )
            role = mapping.get("role", ConceptRole.SECONDARY.value)
            if not isinstance(role, str):
                add(
                    "field_type",
                    "Field 'role' must be a string.",
                    f"{mapping_path}.role",
                    question_id,
                )
            else:
                try:
                    ConceptRole(role)
                except ValueError:
                    allowed = ", ".join(role.value for role in ConceptRole)
                    add(
                        "invalid_concept_role",
                        f"Unknown concept role '{role}'; allowed roles are {allowed}.",
                        f"{mapping_path}.role",
                        question_id,
                    )

        options = list_field(row, "options", path, question_id=question_id)
        if guess_rate is not None and options:
            chance_rate = 1.0 / len(options)
            if guess_rate + 1e-9 < chance_rate:
                add(
                    "guess_below_forced_choice_chance",
                    f"Field 'guess_rate' ({guess_rate:.3f}) is below the forced-choice "
                    f"chance floor ({chance_rate:.3f}) for {len(options)} options; use a "
                    "nominal option model for systematic below-chance misconceptions.",
                    f"{path}.guess_rate",
                    question_id,
                )
        for option_index, option_value in enumerate(options):
            option_path = f"{path}.options[{option_index}]"
            option = row_object(option_value, option_path)
            if option is None:
                continue
            string_field(option, "id", option_path, question_id=question_id)
            string_field(option, "text", option_path, question_id=question_id)
            string_field(option, "rationale", option_path, question_id=question_id)
            string_field(
                option,
                "misconception_id",
                option_path,
                question_id=question_id,
                required=False,
                nullable=True,
            )
            string_field(
                option,
                "diagnostic_objective_id",
                option_path,
                question_id=question_id,
                required=False,
                nullable=True,
            )
            if "correct" not in option:
                add(
                    "missing_field",
                    "Required field 'correct' is missing.",
                    f"{option_path}.correct",
                    question_id,
                )
            elif not isinstance(option["correct"], bool):
                add(
                    "field_type",
                    "Field 'correct' must be a boolean.",
                    f"{option_path}.correct",
                    question_id,
                )

        source_ids = list_field(row, "source_ids", path, question_id=question_id)
        for source_index, source_id in enumerate(source_ids):
            if not isinstance(source_id, str) or not source_id.strip():
                add(
                    "field_type",
                    "Every source_ids entry must be a non-blank string.",
                    f"{path}.source_ids[{source_index}]",
                    question_id,
                )
        provenance = row.get("provenance", {})
        if not isinstance(provenance, dict):
            add("field_type", "Field 'provenance' must be an object.", f"{path}.provenance", question_id)
        else:
            for issue in question_provenance_issues(
                provenance,
                status=row.get("status"),
                legacy_unattested_compatible=(
                    row.get("id") in legacy_unattested_compatible_ids
                    and "generated" not in provenance
                ),
            ):
                field_path = (
                    f"{path}.provenance.{issue.field}"
                    if issue.field
                    else f"{path}.provenance"
                )
                add(issue.code, issue.message, field_path, question_id)
        tags = row.get("tags", [])
        if not isinstance(tags, list):
            add("field_type", "Field 'tags' must be a list.", f"{path}.tags", question_id)
        else:
            for tag_index, tag in enumerate(tags):
                if not isinstance(tag, str) or not tag.strip():
                    add(
                        "field_type",
                        "Every tag must be a non-blank string.",
                        f"{path}.tags[{tag_index}]",
                        question_id,
                    )
        string_field(
            row,
            "revision_of",
            path,
            question_id=question_id,
            required=False,
            nullable=True,
        )
    return issues


def _raise_issues(prefix: str, issues: list[QualityIssue]) -> None:
    errors = [issue for issue in issues if issue.severity == "error"]
    if not errors:
        return
    rendered = "; ".join(
        f"{issue.path or issue.question_id or 'corpus'}: {issue.message}" for issue in errors[:20]
    )
    suffix = f" (+{len(errors) - 20} more)" if len(errors) > 20 else ""
    raise ValidationError(f"{prefix} failed {len(errors)} checks: {rendered}{suffix}", issues=errors)


def load_bundle(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    source_path = Path(path)
    try:
        bundle = json.loads(
            source_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Could not read corpus bundle {source_path}: {exc}") from exc
    if validate:
        _raise_issues("Corpus structure", validate_bundle(bundle))
    return bundle


def parse_catalog(
    bundle: dict[str, Any],
    concepts: list[Concept],
    questions: list[Question] | None = None,
) -> tuple[list[Domain], list[Topic]]:
    """Parse and validate the optional learner-facing curriculum catalog.

    Concept ownership is intentionally canonical: every concept belongs directly
    to exactly one topic.  Cross-topic questions are expressed by mapping the
    question to concepts owned by different topics, rather than assigning one
    concept ambiguously to several buckets.
    """

    if "domains" not in bundle and "topics" not in bundle:
        return [], []
    _raise_issues("Corpus structure", validate_bundle(bundle))
    try:
        domains = [
            Domain(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                sort_order=row.get("sort_order", 0),
            )
            for row in bundle["domains"]
        ]
        topics = [
            Topic(
                id=row["id"],
                domain_id=row["domain_id"],
                name=row["name"],
                description=row["description"],
                concept_ids=tuple(row["concept_ids"]),
                parent_id=row.get("parent_id"),
                related_topic_ids=tuple(row.get("related_topic_ids", [])),
                sort_order=row.get("sort_order", 0),
            )
            for row in bundle["topics"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid curriculum catalog object: {exc}") from exc

    issues: list[QualityIssue] = []

    def add(code: str, message: str, path: str) -> None:
        issues.append(QualityIssue(code, "error", message, path=path))

    domain_counts = Counter(domain.id for domain in domains)
    topic_counts = Counter(topic.id for topic in topics)
    duplicate_domains = sorted(key for key, count in domain_counts.items() if count > 1)
    duplicate_topics = sorted(key for key, count in topic_counts.items() if count > 1)
    if duplicate_domains:
        add(
            "duplicate_domain_id",
            "Domain IDs must be unique: " + ", ".join(duplicate_domains) + ".",
            "domains",
        )
    if duplicate_topics:
        add(
            "duplicate_topic_id",
            "Topic IDs must be unique: " + ", ".join(duplicate_topics) + ".",
            "topics",
        )

    domain_ids = set(domain_counts)
    topic_by_id = {
        topic.id: topic for topic in topics if topic_counts[topic.id] == 1
    }
    concept_ids = {concept.id for concept in concepts}
    children: Counter[str] = Counter()
    concept_owners: dict[str, list[str]] = {}
    for topic in topics:
        if topic.domain_id not in domain_ids:
            add(
                "unknown_domain_reference",
                f"Topic {topic.id} references unknown domain {topic.domain_id}.",
                "topics[].domain_id",
            )
        if topic.parent_id:
            parent = topic_by_id.get(topic.parent_id)
            if parent is None:
                add(
                    "unknown_parent_topic",
                    f"Topic {topic.id} references unknown parent {topic.parent_id}.",
                    "topics[].parent_id",
                )
            elif parent.domain_id != topic.domain_id:
                add(
                    "cross_domain_parent",
                    f"Topic {topic.id} and parent {parent.id} must share a domain.",
                    "topics[].parent_id",
                )
            children[topic.parent_id] += 1
        duplicate_concepts = sorted(
            key for key, count in Counter(topic.concept_ids).items() if count > 1
        )
        if duplicate_concepts:
            add(
                "duplicate_topic_concept",
                f"Topic {topic.id} repeats concepts: {', '.join(duplicate_concepts)}.",
                "topics[].concept_ids",
            )
        for concept_id in topic.concept_ids:
            if concept_id not in concept_ids:
                add(
                    "unknown_topic_concept",
                    f"Topic {topic.id} references unknown concept {concept_id}.",
                    "topics[].concept_ids",
                )
            concept_owners.setdefault(concept_id, []).append(topic.id)
        duplicate_related = sorted(
            key for key, count in Counter(topic.related_topic_ids).items() if count > 1
        )
        if duplicate_related:
            add(
                "duplicate_related_topic",
                f"Topic {topic.id} repeats related topics: {', '.join(duplicate_related)}.",
                "topics[].related_topic_ids",
            )
        for related_id in topic.related_topic_ids:
            if related_id == topic.id:
                add(
                    "self_related_topic",
                    f"Topic {topic.id} cannot relate to itself.",
                    "topics[].related_topic_ids",
                )
            elif related_id not in topic_by_id:
                add(
                    "unknown_related_topic",
                    f"Topic {topic.id} references unknown related topic {related_id}.",
                    "topics[].related_topic_ids",
                )

    for topic in topics:
        for related_id in topic.related_topic_ids:
            related = topic_by_id.get(related_id)
            if related and topic.id not in related.related_topic_ids:
                add(
                    "asymmetric_related_topic",
                    f"Topic relation {topic.id} <-> {related_id} must be declared symmetrically.",
                    "topics[].related_topic_ids",
                )

    for topic in topics:
        if not topic.concept_ids and children[topic.id] == 0:
            add(
                "empty_topic",
                f"Topic {topic.id} owns no concepts and has no child topics.",
                "topics[].concept_ids",
            )

    for start in topic_by_id:
        trail: set[str] = set()
        current = start
        while current in topic_by_id:
            if current in trail:
                add(
                    "topic_cycle",
                    f"Topic parent hierarchy contains a cycle involving {current}.",
                    "topics[].parent_id",
                )
                break
            trail.add(current)
            parent = topic_by_id[current].parent_id
            if not parent:
                break
            current = parent

    unowned = sorted(concept_ids - set(concept_owners))
    multiply_owned = sorted(
        (concept_id, owners)
        for concept_id, owners in concept_owners.items()
        if len(owners) != 1
    )
    if unowned:
        add(
            "unowned_concept",
            f"Every concept must have one canonical topic owner; unowned: {', '.join(unowned)}.",
            "topics[].concept_ids",
        )
    if multiply_owned:
        rendered = "; ".join(
            f"{concept_id}={','.join(sorted(owners))}"
            for concept_id, owners in multiply_owned
        )
        add(
            "multiple_topic_owners",
            "Concepts must have one canonical topic owner: " + rendered + ".",
            "topics[].concept_ids",
        )

    if questions is not None:
        for question in questions:
            mapped = {mapping.concept_id for mapping in question.concepts}
            missing = sorted(mapped - set(concept_owners))
            if missing:
                add(
                    "unbucketed_question_concept",
                    f"Question {question.id} maps concepts without topic ownership: "
                    + ", ".join(missing)
                    + ".",
                    "questions[].concepts",
                )

    for domain_id in domain_ids:
        if not any(
            topic.domain_id == domain_id and topic.parent_id is None
            for topic in topics
        ):
            add(
                "domain_without_root_topic",
                f"Domain {domain_id} has no top-level topic.",
                "topics[].domain_id",
            )

    _raise_issues("Curriculum catalog", issues)
    return domains, topics


def parse_bundle(bundle: dict[str, Any]) -> tuple[
    list[Concept], list[ConceptEdge], list[Misconception], list[Source], list[Question]
]:
    _raise_issues("Corpus structure", validate_bundle(bundle))
    try:
        concepts = [
            Concept(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                domain=row.get("domain", "ai"),
                prior_mastery=float(row.get("prior_mastery", 0.20)),
            )
            for row in bundle["concepts"]
        ]
        objective_edges = [
            ObjectiveEdge(
                id=row["id"],
                source_id=row["source"],
                target_id=row["target"],
                relation=_relation(row["relation"]),
                weight=float(row["weight"]),
                rationale=row["rationale"],
            )
            for row in bundle.get("objective_edges", [])
        ]
        prerequisites_by_target: dict[str, list[ObjectiveEdge]] = {}
        for edge in objective_edges:
            prerequisites_by_target.setdefault(edge.target_id, []).append(edge)
        objective_graph_version = (
            1 if bundle.get("schema_version") == 3 else None
        )
        objectives = [
            LearningObjective(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                primary_concept_id=row["primary_concept_id"],
                supporting_concept_ids=tuple(
                    row.get("supporting_concept_ids", [])
                ),
                operation=ObjectiveOperation(row["operation"]),
                evidence_type=row.get("evidence_type", "selected_response"),
                prior_mastery=float(row.get("prior_mastery", 0.20)),
                prerequisites=tuple(
                    sorted(
                        prerequisites_by_target.get(row["id"], []),
                        key=lambda edge: edge.id,
                    )
                ),
                objective_graph_version=objective_graph_version,
            )
            for row in bundle.get("learning_objectives", [])
        ]
        objective_by_id = {objective.id: objective for objective in objectives}
        edges = [
            ConceptEdge(
                source_id=row["source"],
                target_id=row["target"],
                relation=_relation(row["relation"]),
                weight=float(row.get("weight", 1.0)),
            )
            for row in bundle["edges"]
        ]
        misconceptions = [
            Misconception(
                id=row["id"],
                concept_id=row["concept_id"],
                name=row["name"],
                description=row["description"],
            )
            for row in bundle["misconceptions"]
        ]
        sources = [
            Source(
                id=row["id"],
                title=row["title"],
                uri=row.get("uri"),
                license=row.get("license"),
                metadata={k: v for k, v in row.items() if k not in {"id", "title", "uri", "license"}},
            )
            for row in bundle["sources"]
        ]
        questions = []
        for row in bundle["questions"]:
            questions.append(
                Question(
                    id=row["id"],
                    version=int(row.get("version", 1)),
                    family_id=row["family_id"],
                    status=QuestionStatus(row["status"]),
                    stem=row["stem"],
                    kind=QuestionKind(row["kind"]),
                    difficulty=float(row["difficulty"]),
                    discrimination=float(row["discrimination"]),
                    guess_rate=float(row.get("guess_rate", 0.25)),
                    slip_rate=float(row.get("slip_rate", 0.05)),
                    concepts=tuple(
                        ConceptWeight(
                            concept_id=mapping["concept_id"],
                            weight=float(mapping["weight"]),
                            role=ConceptRole(mapping.get("role", ConceptRole.SECONDARY.value)),
                        )
                        for mapping in row["concepts"]
                    ),
                    options=tuple(
                        Option(
                            id=option["id"],
                            text=option["text"],
                            correct=bool(option["correct"]),
                            misconception_id=option.get("misconception_id"),
                            rationale=option["rationale"],
                            diagnostic_objective_id=(
                                option.get("diagnostic_objective_id")
                                or (
                                    row.get("learning_objective_id")
                                    if not bool(option["correct"])
                                    else None
                                )
                            ),
                        )
                        for option in row["options"]
                    ),
                    source_ids=tuple(row["source_ids"]),
                    provenance=dict(row.get("provenance", {})),
                    tags=tuple(row.get("tags", [])),
                    revision_of=row.get("revision_of"),
                    objective=objective_by_id.get(
                        row.get("learning_objective_id")
                    ),
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid corpus object: {exc}") from exc

    issues: list[QualityIssue] = []

    def add(
        code: str,
        message: str,
        *,
        path: str | None = None,
        question_id: str | None = None,
        severity: str = "error",
    ) -> None:
        issues.append(QualityIssue(code, severity, message, question_id, path))

    concept_ids = {concept.id for concept in concepts}
    objective_id_counts = Counter(objective.id for objective in objectives)
    objective_by_id = {
        objective.id: objective
        for objective in objectives
        if objective_id_counts[objective.id] == 1
    }
    misconception_ids = {misconception.id for misconception in misconceptions}
    source_ids = {source.id for source in sources}
    question_ids = {question.id for question in questions}
    misconception_id_counts = Counter(misconception.id for misconception in misconceptions)
    question_id_counts = Counter(question.id for question in questions)
    misconception_owners = {
        misconception.id: misconception.concept_id
        for misconception in misconceptions
        if misconception_id_counts[misconception.id] == 1
    }
    for field, values in (
        ("concepts", [concept.id for concept in concepts]),
        ("learning_objectives", [objective.id for objective in objectives]),
        ("misconceptions", [misconception.id for misconception in misconceptions]),
        ("sources", [source.id for source in sources]),
        ("questions", [question.id for question in questions]),
    ):
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            add(
                "duplicate_id",
                f"{field} IDs must be unique; duplicates: {', '.join(duplicates)}.",
                path=field,
            )

    declared_objective_references = {
        row.get("learning_objective_id")
        for row in bundle.get("questions", [])
        if isinstance(row, dict) and row.get("learning_objective_id") is not None
    }
    unknown_objective_references = sorted(
        value
        for value in declared_objective_references
        if isinstance(value, str) and value not in objective_by_id
    )
    if unknown_objective_references:
        add(
            "unknown_learning_objective",
            "Questions reference unknown learning objectives: "
            + ", ".join(unknown_objective_references)
            + ".",
            path="questions[].learning_objective_id",
        )
    for objective in objectives:
        unknown = sorted(set(objective.concept_ids) - concept_ids)
        if unknown:
            add(
                "unknown_objective_concept",
                f"Learning objective {objective.id} references unknown concepts: "
                + ", ".join(unknown)
                + ".",
                path="learning_objectives[].concept_ids",
            )
    edge_id_counts = Counter(edge.id for edge in objective_edges)
    duplicate_objective_edge_ids = sorted(
        edge_id for edge_id, count in edge_id_counts.items() if count > 1
    )
    if duplicate_objective_edge_ids:
        add(
            "duplicate_objective_edge_id",
            "Objective-edge IDs must be unique; duplicates: "
            + ", ".join(duplicate_objective_edge_ids)
            + ".",
            path="objective_edges",
        )
    objective_edge_keys = [
        (edge.source_id, edge.target_id)
        for edge in objective_edges
    ]
    duplicate_objective_edges = sorted(
        key for key, count in Counter(objective_edge_keys).items() if count > 1
    )
    if duplicate_objective_edges:
        add(
            "duplicate_objective_edge",
            "Objective prerequisite relations must be unique; duplicates: "
            + repr(duplicate_objective_edges)
            + ".",
            path="objective_edges",
        )
    for edge in objective_edges:
        unknown = sorted(
            {edge.source_id, edge.target_id} - set(objective_by_id)
        )
        if unknown:
            add(
                "unknown_objective_edge_reference",
                f"Objective edge {edge.id} references unknown objectives: "
                + ", ".join(unknown)
                + ".",
                path="objective_edges",
            )
    objective_adjacency: dict[str, list[str]] = {
        objective_id: [] for objective_id in objective_by_id
    }
    objective_indegree = {objective_id: 0 for objective_id in objective_by_id}
    for edge in objective_edges:
        if (
            edge.source_id not in objective_by_id
            or edge.target_id not in objective_by_id
        ):
            continue
        objective_adjacency[edge.source_id].append(edge.target_id)
        objective_indegree[edge.target_id] += 1
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
    if objective_visited != len(objective_by_id):
        cyclic = sorted(
            objective_id
            for objective_id, degree in objective_indegree.items()
            if degree > 0
        )
        add(
            "objective_prerequisite_cycle",
            "Objective prerequisite edges contain a cycle: "
            + ", ".join(cyclic)
            + ".",
            path="objective_edges",
        )
    used_objective_ids = {
        question.objective_id for question in questions if question.objective_id
    }
    unused_objective_ids = sorted(set(objective_by_id) - used_objective_ids)
    if unused_objective_ids:
        add(
            "unused_learning_objective",
            "Every learning objective must map at least one question; unused: "
            + ", ".join(unused_objective_ids)
            + ".",
            path="learning_objectives",
        )
    covered_concept_ids = {
        concept_id
        for objective in objectives
        for concept_id in objective.concept_ids
    }
    for question in questions:
        if question.objective is not None:
            if question.primary_concept_id not in question.objective.concept_ids:
                add(
                    "objective_primary_concept_mismatch",
                    f"Question {question.id} maps objective {question.objective.id}, "
                    f"which does not declare question primary concept "
                    f"{question.primary_concept_id} as owner or supporting context.",
                    question_id=question.id,
                    path="questions[].learning_objective_id",
                )
        elif (
            question.status.eligible_for_adaptation
            and question.primary_concept_id in covered_concept_ids
        ):
            add(
                "missing_learning_objective",
                f"Eligible question {question.id} has primary concept "
                f"{question.primary_concept_id}, covered by the objective catalog, "
                "but does not declare a learning objective.",
                question_id=question.id,
                path="questions[].learning_objective_id",
            )
        for option in question.options:
            diagnostic_id = option.diagnostic_objective_id
            if option.correct and diagnostic_id is not None:
                add(
                    "correct_option_diagnostic_objective",
                    f"Correct option {option.id} on {question.id} cannot declare "
                    "a diagnostic objective.",
                    question_id=question.id,
                    path="questions[].options[].diagnostic_objective_id",
                )
                continue
            if diagnostic_id is None:
                continue
            diagnostic = objective_by_id.get(diagnostic_id)
            if diagnostic is None:
                add(
                    "unknown_diagnostic_objective",
                    f"Option {option.id} on {question.id} references unknown "
                    f"diagnostic objective {diagnostic_id}.",
                    question_id=question.id,
                    path="questions[].options[].diagnostic_objective_id",
                )
                continue
            owner = misconception_owners.get(option.misconception_id)
            if owner is not None and owner not in diagnostic.concept_ids:
                add(
                    "diagnostic_objective_owner_mismatch",
                    f"Option {option.id} on {question.id} maps misconception "
                    f"{option.misconception_id}, owned by {owner}, but diagnostic "
                    f"objective {diagnostic_id} does not include that concept.",
                    question_id=question.id,
                    path="questions[].options[].diagnostic_objective_id",
                )

    edge_keys = [
        (edge.source_id, edge.target_id, edge.relation.value) for edge in edges
    ]
    duplicate_edges = sorted(key for key, count in Counter(edge_keys).items() if count > 1)
    if duplicate_edges:
        add(
            "duplicate_edge",
            f"Concept edges must be unique; duplicates: {duplicate_edges}.",
            path="edges",
        )

    for index, edge in enumerate(edges):
        unknown = sorted({edge.source_id, edge.target_id} - concept_ids)
        if unknown:
            add(
                "unknown_concept_reference",
                f"Edge references unknown concepts: {', '.join(unknown)}.",
                path=f"edges[{index}]",
            )
    for misconception in misconceptions:
        if misconception.concept_id not in concept_ids:
            add(
                "unknown_concept_reference",
                f"Misconception {misconception.id} references unknown concept "
                f"{misconception.concept_id}.",
                path="misconceptions",
            )
    for question in questions:
        mapped_concept_ids = {mapping.concept_id for mapping in question.concepts}
        distractor_misconception_ids = {
            option.misconception_id
            for option in question.options
            if not option.correct and option.misconception_id
        }
        for mapping in question.concepts:
            if mapping.concept_id not in concept_ids:
                add(
                    "unknown_concept_reference",
                    f"Question references unknown concept {mapping.concept_id}.",
                    path="questions[].concepts",
                    question_id=question.id,
                )
        for misconception_id in question.misconception_ids:
            if misconception_id not in misconception_ids:
                add(
                    "unknown_misconception_reference",
                    f"Question references unknown misconception {misconception_id}.",
                    path="questions[].options",
                    question_id=question.id,
                )
                continue
            owner_concept_id = misconception_owners.get(misconception_id)
            if (
                misconception_id in distractor_misconception_ids
                and owner_concept_id in concept_ids
                and owner_concept_id not in mapped_concept_ids
            ):
                add(
                    "unmapped_misconception_owner",
                    f"Distractor misconception {misconception_id} belongs to concept "
                    f"{owner_concept_id}, which is absent from the question's concept mappings.",
                    path="questions[].options",
                    question_id=question.id,
                )
        unknown_sources = set(question.source_ids) - source_ids
        if unknown_sources:
            add(
                "unknown_source_reference",
                f"Question references unknown sources {sorted(unknown_sources)}.",
                path="questions[].source_ids",
                question_id=question.id,
            )
        duplicate_sources = sorted(
            source_id
            for source_id, count in Counter(question.source_ids).items()
            if count > 1
        )
        if duplicate_sources:
            add(
                "duplicate_source_reference",
                f"Question repeats source IDs {duplicate_sources}.",
                path="questions[].source_ids",
                question_id=question.id,
            )
        if question.revision_of and question.revision_of not in question_ids:
            add(
                "unknown_revision_reference",
                f"Question revision_of references unknown question {question.revision_of}.",
                path="questions[].revision_of",
                question_id=question.id,
            )

    questions_by_id = {
        question.id: question
        for question in questions
        if question_id_counts[question.id] == 1
    }
    for question in questions_by_id.values():
        parent_id = question.revision_of
        if not parent_id or parent_id not in questions_by_id:
            continue
        if parent_id == question.id:
            add(
                "revision_self_reference",
                "Question revision_of cannot reference the question itself.",
                path="questions[].revision_of",
                question_id=question.id,
            )
            continue
        parent = questions_by_id[parent_id]
        if question.version <= parent.version:
            add(
                "revision_version_order",
                f"Revision version {question.version} must be greater than parent "
                f"{parent.id} version {parent.version}.",
                path="questions[].version",
                question_id=question.id,
            )
        if question.family_id != parent.family_id:
            add(
                "revision_family_mismatch",
                f"Revision must preserve parent {parent.id} family_id "
                f"{parent.family_id}; received {question.family_id}.",
                path="questions[].family_id",
                question_id=question.id,
            )

    completed_revision_nodes: set[str] = set()
    for start_id in questions_by_id:
        if start_id in completed_revision_nodes:
            continue
        trail: list[str] = []
        position: dict[str, int] = {}
        current_id = start_id
        while current_id in questions_by_id and current_id not in completed_revision_nodes:
            if current_id in position:
                cycle = trail[position[current_id] :]
                if len(cycle) > 1:
                    add(
                        "revision_cycle",
                        "revision_of chain contains a cycle: "
                        + " -> ".join([*cycle, cycle[0]]),
                        path="questions[].revision_of",
                        question_id=cycle[0],
                    )
                break
            position[current_id] = len(trail)
            trail.append(current_id)
            parent_id = questions_by_id[current_id].revision_of
            if not parent_id or parent_id == current_id:
                break
            current_id = parent_id
        completed_revision_nodes.update(trail)

    knowledge_graph = None
    if not any(issue.code == "unknown_concept_reference" for issue in issues):
        try:
            knowledge_graph = KnowledgeGraph(concepts, edges)
        except ValidationError as exc:
            add("graph_integrity", str(exc), path="edges")
    if not any(issue.severity == "error" for issue in issues):
        parse_catalog(bundle, concepts, questions)
    issues.extend(
        audit_corpus(
            questions,
            expected_primary_concept_ids={
                mapping.concept_id for question in questions for mapping in question.concepts
            },
            knowledge_graph=knowledge_graph,
            misconceptions=misconceptions,
        )
    )
    _raise_issues("Corpus", issues)
    return concepts, edges, misconceptions, sources, questions


def read_and_parse(path: str | Path, *, include_catalog: bool = False):
    bundle = load_bundle(path)
    parsed = parse_bundle(bundle)
    if not include_catalog:
        return parsed
    domains, topics = parse_catalog(bundle, parsed[0], parsed[4])
    return (*parsed, domains, topics)
