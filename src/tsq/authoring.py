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
from .errors import ConflictError, NotFoundError, ValidationError
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
    allowed = required | {"quality_contract"}
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
    model_name = "blueprint-fixture-v1"

    def __init__(self, misconceptions: dict[str, tuple[str, str]]):
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
        option_ids = ("a", "c", "d")
        for index, (option_id, misconception_id) in enumerate(
            zip(option_ids, distractor_ids, strict=True), start=1
        ):
            name, description = self._misconceptions.get(
                misconception_id,
                (misconception_id, "No registered misconception description is available."),
            )
            distractor_options.append(
                {
                    "id": option_id,
                    "text": (
                        f"Adopt misconception {index}, {name}: {description}"
                    ),
                    "correct": False,
                    "misconception_id": misconception_id,
                    "rationale": (
                        f"This option directly states the named misconception {name}; "
                        "it is retained only as a quarantined fixture distractor."
                    ),
                }
            )
        return {
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
            "concepts": [
                {
                    "concept_id": blueprint.concept_id,
                    "weight": 1.0,
                    "role": "primary",
                }
            ],
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
            row["id"]: (row["name"], row["description"])
            for row in connection.execute(
                "SELECT id, name, description FROM misconceptions ORDER BY id"
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
                if type(option.get("correct")) is not bool:
                    add(f"{prefix}.correct must be a JSON boolean.")
                misconception_id = option.get("misconception_id")
                if misconception_id is not None and type(misconception_id) is not str:
                    add(f"{prefix}.misconception_id must be a string or null.")

        source_ids = item.get("source_ids")
        if type(source_ids) is not list:
            add("Field 'source_ids' must be a JSON array.")
        elif any(type(source_id) is not str for source_id in source_ids):
            add("Every source_ids entry must be a string.")

        provenance = item.get("provenance", {})
        if type(provenance) is not dict:
            add("Field 'provenance' must be a JSON object.")
        tags = item.get("tags", [])
        if type(tags) is not list or any(type(tag) is not str for tag in tags):
            add("Field 'tags' must be a JSON array of strings.")
        revision_of = item.get("revision_of")
        if revision_of is not None and type(revision_of) is not str:
            add("Field 'revision_of' must be a string or null.")
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
        if type(output) is not dict:
            output_issues.insert(0, "Reviewer output must be a JSON object.")

        if output_issues:
            persisted_output: dict[str, Any] = {
                "rejected_output_type": type(output).__name__,
                "validation_errors": output_issues,
            }
            valid = False
        else:
            persisted_output = copy.deepcopy(output)
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
            "valid": valid,
            "validation_errors": output_issues,
        }

    def _deterministic_validation(
        self, item: Any, blueprint: GenerationBlueprint
    ) -> list[dict[str, str]]:
        issues = self._item_shape_issues(item)
        if issues:
            return issues

        def add(code: str, message: str) -> None:
            issues.append({"code": code, "severity": "error", "message": message})

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
                    )
                    for option in item["options"]
                ),
                source_ids=tuple(item["source_ids"]),
                provenance=copy.deepcopy(item.get("provenance", {})),
                tags=tuple(item.get("tags", ())),
                revision_of=item.get("revision_of"),
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

        with self.database.read() as connection:
            if connection.execute(
                "SELECT 1 FROM questions WHERE family_id = ? LIMIT 1",
                (question.family_id,),
            ).fetchone():
                add("family_collision", "Generated item reuses an existing item family.")
            known_sources = {
                row["id"] for row in connection.execute("SELECT id FROM sources")
            }
            if set(question.source_ids) - known_sources:
                add("unknown_source", "Generated item cites an unknown source ID.")
            misconception_ids = question.misconception_ids
            if misconception_ids:
                placeholders = ",".join("?" for _ in misconception_ids)
                owners = {
                    row["id"]: row["concept_id"]
                    for row in connection.execute(
                        f"SELECT id, concept_id FROM misconceptions WHERE id IN ({placeholders})",
                        tuple(sorted(misconception_ids)),
                    )
                }
                unknown = misconception_ids - set(owners)
                if unknown:
                    add(
                        "unknown_misconception",
                        f"Unknown misconception IDs: {', '.join(sorted(unknown))}.",
                    )
                mapped_concepts = {mapping.concept_id for mapping in question.concepts}
                absent_owners = {
                    owner for owner in owners.values() if owner not in mapped_concepts
                }
                if absent_owners:
                    add(
                        "unmapped_misconception_owner",
                        "Every distractor misconception's concept must be mapped by the item.",
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
                raise ConflictError(
                    f"Generation job {job_id} is {row['status']}; retry it explicitly "
                    "before another execution."
                )
            if row["prompt_version"] != PROMPT_VERSION:
                raise ConflictError(
                    f"Generation job {job_id} uses unsupported prompt version "
                    f"{row['prompt_version']!r}."
                )
            blueprint = _parse_blueprint(
                row["blueprint_json"], label=f"Generation job {job_id} blueprint"
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
            deterministic_issues = self._deterministic_validation(raw_output, blueprint)
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
                item = copy.deepcopy(raw_output)
            else:
                item = {"generator_output_rejected": True}
            item["status"] = "quarantined"
            declared_provenance = item.get("provenance")
            if type(declared_provenance) is not dict:
                declared_provenance = {}
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
                    "provider": self.generator_provider_name,
                    "model": self.generator_model_name,
                    "prompt_version": row["prompt_version"],
                    "generation_job_id": job_id,
                    "generation_run_id": run_id,
                    "generation_attempt": next_attempt,
                    "source_context_sha256": source_context_sha256,
                    "generator_output_sha256": generator_output_sha256,
                    "generator_provenance_sha256": generator_provenance_sha256,
                }
            )

            blinded_item = _blind_for_review(item)
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
            }
            with self.database.transaction() as connection:
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
                "error": str(exc)[:4000],
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
